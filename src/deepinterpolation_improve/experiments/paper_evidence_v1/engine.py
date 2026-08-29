from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_recording_datasets
from .metrics import MetricAccumulator
from .models import create_variant, variant_spec
from .profiling import benchmark_latency, conv_profile, inference_copy, parameter_count
from .study import ExperimentTask, sha256_file


def _seed_worker(base_seed: int, worker_id: int) -> None:
    seed = int(base_seed) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _new_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    worker_seed: int,
    pin_memory: bool,
    generator: torch.Generator,
) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": bool(pin_memory),
        "generator": generator,
    }
    if workers > 0:
        options["worker_init_fn"] = partial(_seed_worker, int(worker_seed))
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def _prepare(tensor: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def _objective_terms(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    teacher_predictions: torch.Tensor | None = None,
    distillation_weight: float = 0.0,
    objective_name: str = "target_l1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    target_mae = torch.nn.functional.l1_loss(predictions, targets)
    if teacher_predictions is None:
        if objective_name == "target_l1":
            objective = target_mae
        elif objective_name == "target_mse":
            objective = torch.nn.functional.mse_loss(predictions, targets)
        else:
            raise ValueError(f"Unsupported training objective without a teacher: {objective_name}")
        return objective, target_mae, None
    if objective_name not in {"target_l1", "target_l1_plus_naf_teacher"}:
        raise ValueError(f"Teacher predictions are incompatible with objective {objective_name}")
    if not 0.0 < float(distillation_weight) < 1.0:
        raise ValueError("distillation_weight must be between zero and one")
    teacher_mae = torch.nn.functional.l1_loss(predictions, teacher_predictions)
    objective = (1.0 - distillation_weight) * target_mae + distillation_weight * teacher_mae
    return objective, target_mae, teacher_mae


def _run_epoch(
    model,
    loader,
    device: torch.device,
    optimiser,
    scaler,
    use_amp: bool,
    channels_last: bool,
    teacher_model=None,
    distillation_weight: float = 0.0,
    objective_name: str = "target_l1",
) -> dict[str, float | None]:
    training = optimiser is not None
    model.train(training)
    total_objective = 0.0
    total_target_mae = 0.0
    total_target_mse = 0.0
    total_teacher_mae = 0.0
    total_samples = 0
    for batch in loader:
        inputs = _prepare(batch["input"], device, channels_last)
        targets = _prepare(batch["target"], device, channels_last)
        if training:
            optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            predictions = model(inputs)
            teacher_predictions = None
            if training and teacher_model is not None:
                with torch.no_grad():
                    teacher_predictions = teacher_model(inputs)
            loss, target_mae, teacher_mae = _objective_terms(
                predictions,
                targets,
                teacher_predictions,
                distillation_weight,
                objective_name,
            )
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        batch_size = int(inputs.shape[0])
        total_objective += float(loss.detach().cpu()) * batch_size
        total_target_mae += float(target_mae.detach().cpu()) * batch_size
        total_target_mse += float(
            torch.nn.functional.mse_loss(predictions.detach().float(), targets.float()).cpu()
        ) * batch_size
        if teacher_mae is not None:
            total_teacher_mae += float(teacher_mae.detach().cpu()) * batch_size
        total_samples += batch_size
    denominator = max(total_samples, 1)
    return {
        "objective": total_objective / denominator,
        "target_mae": total_target_mae / denominator,
        "target_mse": total_target_mse / denominator,
        "teacher_mae": total_teacher_mae / denominator if teacher_model is not None else None,
    }


def _selection_value(validation_stats: dict[str, float | None], metric: str) -> float:
    key_by_metric = {"val_mae": "target_mae", "val_mse": "target_mse"}
    if metric not in key_by_metric:
        raise ValueError(f"Unsupported checkpoint selection metric: {metric}")
    value = validation_stats[key_by_metric[metric]]
    if value is None:
        raise ValueError(f"Checkpoint selection metric is unavailable: {metric}")
    return float(value)


def _evaluate(model, loader, device: torch.device, use_amp: bool, channels_last: bool):
    model.eval()
    accumulator = MetricAccumulator()
    temporal_weight_sum: torch.Tensor | None = None
    temporal_entropy_sum = 0.0
    temporal_max_sum = 0.0
    temporal_pixel_count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = _prepare(batch["input"], device, channels_last)
            targets = _prepare(batch["target"], device, channels_last)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                predictions = model(inputs)
            accumulator.update(predictions.float(), targets.float(), batch["frame_index"])
            if hasattr(model, "temporal_weights"):
                weights = model.temporal_weights(inputs).float()
                per_frame = weights.sum(dim=(0, 2, 3)).detach().cpu()
                temporal_weight_sum = per_frame if temporal_weight_sum is None else temporal_weight_sum + per_frame
                entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
                temporal_entropy_sum += float(entropy.sum().detach().cpu())
                temporal_max_sum += float(weights.max(dim=1).values.sum().detach().cpu())
                temporal_pixel_count += int(entropy.numel())
    diagnostics = None
    if temporal_weight_sum is not None and temporal_pixel_count > 0:
        frame_count = int(temporal_weight_sum.numel())
        normaliser = float(temporal_weight_sum.sum())
        diagnostics = {
            "mean_frame_weights": [float(value / normaliser) for value in temporal_weight_sum],
            "mean_normalised_entropy": temporal_entropy_sum / temporal_pixel_count / np.log(frame_count),
            "mean_max_frame_weight": temporal_max_sum / temporal_pixel_count,
        }
        if hasattr(model, "attention_anchor") and hasattr(model.attention_anchor, "residual_gate"):
            diagnostics["attention_residual_gate"] = float(
                torch.tanh(model.attention_anchor.residual_gate).detach().cpu()
            )
    return accumulator.rows, accumulator.summary(), diagnostics


def _checkpoint_payload(
    model,
    optimiser,
    scheduler,
    scaler,
    epoch: int,
    selection_metric: str,
    best_val_metric: float,
    best_val_mae: float,
    best_val_mse: float,
    history,
    loader_generators: dict[str, torch.Generator],
):
    return {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "selection_metric": selection_metric,
        "best_val_metric": float(best_val_metric),
        "best_val_mae": float(best_val_mae),
        "best_val_mse": float(best_val_mse),
        "history": history,
        "rng_state": {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "loader_rng_state": {name: generator.get_state() for name, generator in loader_generators.items()},
    }


def _restore_rng_state(checkpoint: dict[str, Any], loader_generators: dict[str, torch.Generator]) -> None:
    rng_state = checkpoint.get("rng_state")
    loader_state = checkpoint.get("loader_rng_state")
    if rng_state is None or loader_state is None:
        raise ValueError("Checkpoint lacks deterministic RNG state and cannot be resumed exactly")
    torch.random.set_rng_state(rng_state["torch"].detach().cpu().to(dtype=torch.uint8))
    if rng_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.detach().cpu().to(dtype=torch.uint8) for state in rng_state["cuda"]]
        )
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])
    for name, generator in loader_generators.items():
        if name not in loader_state:
            raise ValueError(f"Checkpoint lacks {name} DataLoader RNG state")
        generator.set_state(loader_state[name].detach().cpu().to(dtype=torch.uint8))


def _run_task_impl(
    task: ExperimentTask,
    *,
    device_override: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one task. Callers must provide the explicit execution gate."""

    training = task.training
    device = torch.device(device_override or str(training.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(task.output_dir)
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    if summary_path.exists():
        raise FileExistsError(f"Completed output already exists: {summary_path}")
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"Checkpoint exists; pass --resume to continue: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(task.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    datasets, normalisation = build_recording_datasets(task.dataset_root, task.split_manifest)
    batch_size = int(training.get("batch_size", 4))
    workers = int(training.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    loader_generators = {
        "train": _new_generator(task.seed),
        "val": _new_generator(task.seed + 1_000_003),
        "test": _new_generator(task.seed + 2_000_003),
    }
    loaders = {
        "train": _loader(
            datasets["train"], batch_size, workers, True, task.seed, pin_memory, loader_generators["train"]
        ),
        "val": _loader(
            datasets["val"], batch_size, workers, False, task.seed + 1_000_003, pin_memory, loader_generators["val"]
        ),
        "test": _loader(
            datasets["test"], batch_size, workers, False, task.seed + 2_000_003, pin_memory, loader_generators["test"]
        ),
    }

    spec = variant_spec(task.variant)
    model = create_variant(task.variant, task.input_channels)
    training_parameters = parameter_count(model)
    deployed = inference_copy(model)
    profile = conv_profile(deployed, (task.input_channels, *task.frame_shape), device=device)
    del deployed
    model = model.to(device)
    channels_last = bool(training.get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    teacher_model = None
    distillation_weight = 0.0
    objective_protocol: dict[str, Any] = {
        "name": spec.training_objective,
        "target_weight": 1.0,
    }
    if spec.training_objective == "target_l1_plus_naf_teacher":
        distillation = dict(training.get("distillation", {}))
        teacher_variant = str(distillation.get("teacher_variant", ""))
        teacher_path = Path(str(distillation.get("teacher_checkpoint", "")))
        distillation_weight = float(distillation.get("weight", 0.0))
        if not teacher_path.is_file():
            raise FileNotFoundError(f"Distillation teacher checkpoint is missing: {teacher_path}")
        teacher_summary_path = teacher_path.with_name("summary.json")
        if not teacher_summary_path.is_file():
            raise FileNotFoundError(f"Distillation teacher summary is missing: {teacher_summary_path}")
        teacher_summary = json.loads(teacher_summary_path.read_text(encoding="utf-8"))
        expected_teacher = (task.recording, task.window, task.seed, teacher_variant)
        observed_teacher = (
            str(teacher_summary.get("recording")),
            str(teacher_summary.get("window")),
            int(teacher_summary.get("seed", -1)),
            str(teacher_summary.get("variant")),
        )
        if observed_teacher != expected_teacher:
            raise ValueError(
                f"Distillation teacher condition mismatch: expected {expected_teacher}, found {observed_teacher}"
            )
        teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher_model = create_variant(teacher_variant, task.input_channels)
        teacher_model.load_state_dict(teacher_checkpoint["model_state"], strict=True)
        teacher_model = teacher_model.to(device).eval()
        if channels_last:
            teacher_model = teacher_model.to(memory_format=torch.channels_last)
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        objective_protocol = {
            "name": spec.training_objective,
            "target_weight": 1.0 - distillation_weight,
            "distillation_weight": distillation_weight,
            "teacher_variant": teacher_variant,
            "teacher_checkpoint_sha256": sha256_file(teacher_path),
            "teacher_task_id": teacher_summary.get("task_id"),
        }
        del teacher_checkpoint
    requested_training_precision = str(training.get("precision", "amp_fp16"))
    if requested_training_precision != "amp_fp16":
        raise ValueError("This study requires training.precision=amp_fp16")
    metric_precision = str(training.get("metric_precision", "fp32"))
    if metric_precision != "fp32":
        raise ValueError("This study requires training.metric_precision=fp32")
    use_amp = device.type == "cuda"
    executed_training_precision = "amp_fp16" if use_amp else "fp32"
    precision_protocol = {
        "training_requested": requested_training_precision,
        "training_executed": executed_training_precision,
        "model_parameters": "fp32",
        "optimiser_state": "fp32",
        "gradient_scaling": bool(use_amp),
        "test_forward": executed_training_precision,
        "metric_computation": metric_precision,
        "latency_benchmark": (
            str(task.benchmark.get("precision"))
            if bool(task.benchmark.get("enabled", False))
            else None
        ),
    }
    optimiser = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 1e-3)))
    scheduler_step = training.get("scheduler_step_size")
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimiser,
            step_size=int(scheduler_step),
            gamma=float(training.get("scheduler_gamma", 0.5)),
        )
        if scheduler_step is not None
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    selection_metric = spec.selection_metric
    best_val = float("inf")
    best_val_mae = float("inf")
    best_val_mse = float("inf")
    history: list[dict[str, Any]] = []
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimiser.load_state_dict(checkpoint["optimiser_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        checkpoint_metric = str(checkpoint.get("selection_metric", "val_mae"))
        if checkpoint_metric != selection_metric:
            raise ValueError(
                f"Checkpoint selection metric mismatch: expected {selection_metric}, "
                f"found {checkpoint_metric}"
            )
        best_val = float(checkpoint.get("best_val_metric", checkpoint["best_val_mae"]))
        best_val_mae = float(checkpoint["best_val_mae"])
        if "best_val_mse" in checkpoint:
            best_val_mse = float(checkpoint["best_val_mse"])
        elif history:
            selected = min(history, key=lambda row: float(row["val_mae"]))
            best_val_mse = float(selected.get("val_mse", float("inf")))
        _restore_rng_state(checkpoint, loader_generators)
        del checkpoint

    spec_payload = asdict(spec)
    spec_payload.pop("builder")
    run_manifest = {
        "schema_version": 1,
        "task": task.to_dict(),
        "variant": spec_payload,
        "normalisation": normalisation.to_dict(),
        "split_manifest_sha256": sha256_file(task.split_manifest),
        "training_parameters": training_parameters,
        "deployed_profile": profile,
        "precision_protocol": precision_protocol,
        "training_objective": objective_protocol,
        "checkpoint_selection_metric": selection_metric,
        "evaluation_split": str(training.get("evaluation_split", "test")),
        "claim_scope": "noisy-centre-frame reconstruction fidelity only",
        "centre_frame_excluded": True,
        "process_id": os.getpid(),
    }
    _write_json(run_manifest, output_dir / "run_manifest.json")

    epochs = int(training.get("epochs", 1))
    checkpoint_every = int(training.get("checkpoint_every", 1))
    best_path = output_dir / "best.pt"
    payload: dict[str, Any] | None = None
    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = _run_epoch(
            model,
            loaders["train"],
            device,
            optimiser,
            scaler,
            use_amp,
            channels_last,
            teacher_model=teacher_model,
            distillation_weight=distillation_weight,
            objective_name=spec.training_objective,
        )
        validation_objective = (
            "target_l1"
            if spec.training_objective == "target_l1_plus_naf_teacher"
            else spec.training_objective
        )
        val_stats = _run_epoch(
            model,
            loaders["val"],
            device,
            None,
            scaler,
            use_amp,
            channels_last,
            objective_name=validation_objective,
        )
        train_mae = float(train_stats["target_mae"])
        val_mae = float(val_stats["target_mae"])
        train_mse = float(train_stats["target_mse"])
        val_mse = float(val_stats["target_mse"])
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_objective": train_stats["objective"],
            "train_teacher_mae": train_stats["teacher_mae"],
            "learning_rate": optimiser.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        current_selection_value = _selection_value(val_stats, selection_metric)
        improved = current_selection_value < best_val
        if improved:
            best_val = current_selection_value
            best_val_mae = val_mae
            best_val_mse = val_mse
        payload = _checkpoint_payload(
            model,
            optimiser,
            scheduler,
            scaler,
            epoch,
            selection_metric,
            best_val,
            best_val_mae,
            best_val_mse,
            history,
            loader_generators,
        )
        if improved:
            _atomic_torch_save(payload, best_path)
        if epoch == epochs or epoch % checkpoint_every == 0:
            _atomic_torch_save(payload, checkpoint_path)
        _write_rows(history, output_dir / "loss_history.csv")
    if payload is not None:
        del payload
    if teacher_model is not None:
        teacher_model = teacher_model.cpu()
        del teacher_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    evaluation_split = str(training.get("evaluation_split", "test"))
    metric_rows, metric_summary, temporal_diagnostics = _evaluate(
        model, loaders[evaluation_split], device, use_amp, channels_last
    )
    structural_diagnostics = (
        model.structural_diagnostics() if hasattr(model, "structural_diagnostics") else None
    )
    _write_rows(metric_rows, output_dir / f"{evaluation_split}_metrics.csv")
    best_epoch = int(best_checkpoint["epoch"])
    best_val_metric = float(
        best_checkpoint.get("best_val_metric", best_checkpoint["best_val_mae"])
    )
    best_val_mae = float(best_checkpoint["best_val_mae"])
    best_val_mse = float(best_checkpoint.get("best_val_mse", float("nan")))

    latency = None
    if bool(task.benchmark.get("enabled", False)):
        model = model.cpu()
        del best_checkpoint, optimiser, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        latency = benchmark_latency(
            model,
            (task.input_channels, *task.frame_shape),
            str(device),
            int(task.benchmark.get("batch_size", 1)),
            int(task.benchmark.get("warmup", 50)),
            int(task.benchmark.get("repeats", 200)),
            str(task.benchmark.get("precision", "fp32")),
            bool(task.benchmark.get("channels_last", False)),
        )
    summary = {
        "schema_version": 1,
        "task_id": task.task_id,
        "stage": task.stage,
        "recording": task.recording,
        "window": task.window,
        "variant": task.variant,
        "seed": task.seed,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_val_metric": best_val_metric,
        "best_val_mae": best_val_mae,
        "best_val_mse": best_val_mse,
        "evaluation_split": evaluation_split,
        "metrics": metric_summary,
        "temporal_weight_diagnostics": temporal_diagnostics,
        "structural_diagnostics": structural_diagnostics,
        "training_parameters": training_parameters,
        "deployed_profile": profile,
        "latency": latency,
        "precision_protocol": precision_protocol,
        "training_objective": objective_protocol,
        "claim_scope": "noisy-centre-frame reconstruction fidelity only",
    }
    _write_json(summary, summary_path)
    return summary


def run_task(
    task: ExperimentTask,
    *,
    device_override: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one task with an auditable running/completed/failed status file."""

    output_dir = Path(task.output_dir)
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    if summary_path.exists():
        raise FileExistsError(f"Completed output already exists: {summary_path}")
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"Checkpoint exists; pass --resume to continue: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    started_at = datetime.now(timezone.utc)
    _write_json(
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "state": "running",
            "started_at": started_at.isoformat(),
            "process_id": os.getpid(),
            "resume": bool(resume),
        },
        status_path,
    )
    try:
        summary = _run_task_impl(task, device_override=device_override, resume=resume)
    except BaseException as exc:
        _write_json(
            {
                "schema_version": 1,
                "task_id": task.task_id,
                "state": "failed",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            status_path,
        )
        raise
    _write_json(
        {
            "schema_version": 1,
            "task_id": task.task_id,
            "state": "completed",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "summary": str(summary_path),
        },
        status_path,
    )
    return summary
