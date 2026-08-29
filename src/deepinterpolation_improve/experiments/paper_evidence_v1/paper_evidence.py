from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .data import build_recording_datasets
from .engine import _evaluate, _loader, _new_generator, _write_json, _write_rows
from .models import create_variant
from .study import ExperimentTask, expand_tasks, sha256_file
from .validation import validate_run_directory


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "paper_evidence_v1"
CORE_CONFIG = PROJECT_ROOT / "configs" / "paper_evidence_v1" / "core.json"
MECHANISM_CONFIG = PROJECT_ROOT / "configs" / "paper_evidence_v1" / "mechanism.json"
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "paper_evidence_v1" / "protocol.json"
FREEZE_PATH = EVIDENCE_ROOT / "evaluation_freeze.json"
DISCLOSURE = (
    "The matched evaluation segments were inspected in prior exploratory work and are not "
    "claimed as an independent post-selection test set."
)
CORE_VARIANTS = (
    "paper_fullres_b8d3_mae",
    "paper_fullres_b8d3_mse",
    "paper_subpixel_b8d3_mae",
    "paper_subpixel_b8d3_mse",
    "paper_fullres_b8d4_mse",
)
MECHANISM_VARIANTS = (
    "paper_subpixel_dense_b8d3_mse",
    "paper_subpixel_naf_lite_b8d3_mse",
    "paper_ctpf_subpixel_b8d3_mse",
)
PAIR_DEFINITIONS = {
    "structure_mae": ("paper_subpixel_b8d3_mae", "paper_fullres_b8d3_mae"),
    "structure_mse": ("paper_subpixel_b8d3_mse", "paper_fullres_b8d3_mse"),
    "objective_fullres": ("paper_fullres_b8d3_mse", "paper_fullres_b8d3_mae"),
    "objective_subpixel": ("paper_subpixel_b8d3_mse", "paper_subpixel_b8d3_mae"),
    "capacity_reference": ("paper_fullres_b8d4_mse", "paper_subpixel_b8d3_mse"),
    "mechanism_dense": ("paper_subpixel_dense_b8d3_mse", "paper_subpixel_b8d3_mse"),
    "mechanism_naf_lite": ("paper_subpixel_naf_lite_b8d3_mse", "paper_subpixel_b8d3_mse"),
    "mechanism_ctpf": ("paper_ctpf_subpixel_b8d3_mse", "paper_subpixel_b8d3_mse"),
}
METRICS = ("psnr_db", "normalised_mse", "normalised_mae", "gssim")


def paper_tasks(scope: str = "all", *, validate_paths: bool = True) -> list[ExperimentTask]:
    if scope not in {"core", "mechanism", "all"}:
        raise ValueError("scope must be core, mechanism or all")
    core = expand_tasks(CORE_CONFIG, validate_paths=validate_paths) if scope != "mechanism" else []
    mechanism = (
        expand_tasks(MECHANISM_CONFIG, validate_paths=validate_paths) if scope != "core" else []
    )
    if scope in {"core", "all"} and len(core) != 225:
        raise ValueError(f"paper-core must expand to 225 tasks, found {len(core)}")
    if scope in {"mechanism", "all"} and len(mechanism) != 27:
        raise ValueError(f"paper-mechanism must expand to 27 tasks, found {len(mechanism)}")
    return core + mechanism


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_entry(task: ExperimentTask) -> dict[str, Any]:
    output = Path(task.output_dir)
    issues = validate_run_directory(task)
    if issues:
        raise ValueError(f"Invalid training artifacts for {task.task_id}: {'; '.join(issues)}")
    if str(task.training.get("evaluation_split")) != "val":
        raise ValueError(f"Training evaluation split is not val: {task.task_id}")
    forbidden = [name for name in ("test_metrics.csv", "evaluation_summary.json") if (output / name).exists()]
    if forbidden:
        raise ValueError(f"Pre-freeze test artifacts exist for {task.task_id}: {', '.join(forbidden)}")
    return {
        "task_id": task.task_id,
        "stage": task.stage,
        "recording": task.recording,
        "window": task.window,
        "variant": task.variant,
        "seed": task.seed,
        "source_config_sha256": sha256_file(task.source_config),
        "split_manifest_sha256": sha256_file(task.split_manifest),
        "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
        "training_summary_sha256": sha256_file(output / "summary.json"),
        "loss_history_sha256": sha256_file(output / "loss_history.csv"),
        "final_checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
    }


def freeze_paper_evidence() -> dict[str, Any]:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Evaluation freeze already exists: {FREEZE_PATH}")
    tasks = paper_tasks("all", validate_paths=True)
    keys = [(task.recording, task.window, task.variant, task.seed) for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate paper-evidence conditions were detected")
    entries = [_freeze_entry(task) for task in tasks]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_name": "paper_evidence_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "core_task_count": 225,
        "mechanism_task_count": 27,
        "evaluation_split_unlocked": "test",
        "matched_evaluation_segments_disclosure": DISCLOSURE,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "core_config_sha256": sha256_file(CORE_CONFIG),
        "mechanism_config_sha256": sha256_file(MECHANISM_CONFIG),
        "tasks": entries,
    }
    payload["freeze_hash"] = _canonical_hash(payload)
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(payload, FREEZE_PATH)
    return payload


def load_verified_freeze(tasks: Iterable[ExperimentTask] | None = None) -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError(f"Evaluation is locked until paper-freeze succeeds: {FREEZE_PATH}")
    payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    observed_hash = str(payload.get("freeze_hash", ""))
    unsigned = dict(payload)
    unsigned.pop("freeze_hash", None)
    if observed_hash != _canonical_hash(unsigned):
        raise ValueError("Evaluation freeze hash is invalid")
    expected_static = {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "core_config_sha256": sha256_file(CORE_CONFIG),
        "mechanism_config_sha256": sha256_file(MECHANISM_CONFIG),
    }
    for key, value in expected_static.items():
        if payload.get(key) != value:
            raise ValueError(f"Evaluation freeze no longer matches {key}")
    by_id = {entry["task_id"]: entry for entry in payload.get("tasks", [])}
    selected = list(tasks) if tasks is not None else paper_tasks("all", validate_paths=True)
    for task in selected:
        entry = by_id.get(task.task_id)
        if entry is None:
            raise ValueError(f"Task is absent from the evaluation freeze: {task.task_id}")
        output = Path(task.output_dir)
        current = {
            "source_config_sha256": sha256_file(task.source_config),
            "split_manifest_sha256": sha256_file(task.split_manifest),
            "run_manifest_sha256": sha256_file(output / "run_manifest.json"),
            "training_summary_sha256": sha256_file(output / "summary.json"),
            "loss_history_sha256": sha256_file(output / "loss_history.csv"),
            "final_checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
            "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        }
        for key, value in current.items():
            if entry.get(key) != value:
                raise ValueError(f"Frozen artifact changed for {task.task_id}: {key}")
    return payload


def _evaluation_paths(task: ExperimentTask) -> tuple[Path, Path]:
    output = Path(task.output_dir)
    return output / "test_metrics.csv", output / "evaluation_summary.json"


def validate_frozen_evaluation(task: ExperimentTask, freeze_hash: str) -> list[str]:
    metrics_path, summary_path = _evaluation_paths(task)
    issues: list[str] = []
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        issues.append("missing or empty test_metrics.csv")
    if not summary_path.is_file() or summary_path.stat().st_size == 0:
        issues.append("missing or empty evaluation_summary.json")
    if issues:
        return issues
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        return [f"cannot parse evaluation artifacts: {exc}"]
    if summary.get("task_id") != task.task_id:
        issues.append("evaluation task_id mismatch")
    if summary.get("freeze_hash") != freeze_hash:
        issues.append("evaluation freeze hash mismatch")
    if summary.get("evaluation_split") != "test":
        issues.append("evaluation split is not test")
    if summary.get("best_checkpoint_sha256") != sha256_file(Path(task.output_dir) / "best.pt"):
        issues.append("evaluation checkpoint hash mismatch")
    if summary.get("test_metrics_sha256") != sha256_file(metrics_path):
        issues.append("test metric hash mismatch")
    if int(summary.get("metrics", {}).get("frame_count", -1)) != len(rows):
        issues.append("test metric row count mismatch")
    if len({int(row["frame_index"]) for row in rows}) != len(rows):
        issues.append("duplicate test frame indices")
    return issues


def evaluate_frozen_task(task: ExperimentTask, freeze: dict[str, Any], device_name: str) -> dict[str, Any]:
    freeze_hash = str(freeze["freeze_hash"])
    metrics_path, summary_path = _evaluation_paths(task)
    if summary_path.exists():
        issues = validate_frozen_evaluation(task, freeze_hash)
        if issues:
            raise ValueError(f"Invalid existing evaluation for {task.task_id}: {'; '.join(issues)}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if metrics_path.exists():
        raise ValueError(f"Orphan test_metrics.csv exists for {task.task_id}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    datasets, normalisation = build_recording_datasets(task.dataset_root, task.split_manifest)
    workers = int(task.training.get("num_workers", 0))
    generator = _new_generator(task.seed + 2_000_003)
    loader = _loader(
        datasets["test"],
        int(task.training.get("batch_size", 4)),
        workers,
        False,
        task.seed + 2_000_003,
        device.type == "cuda",
        generator,
    )
    model = create_variant(task.variant, task.input_channels)
    checkpoint_path = Path(task.output_dir) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    channels_last = bool(task.training.get("channels_last", True)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    rows, metric_summary, temporal_diagnostics = _evaluate(
        model,
        loader,
        device,
        device.type == "cuda",
        channels_last,
    )
    structural_diagnostics = model.structural_diagnostics() if hasattr(model, "structural_diagnostics") else None
    _write_rows(rows, metrics_path)
    summary = {
        "schema_version": 1,
        "task_id": task.task_id,
        "stage": task.stage,
        "recording": task.recording,
        "window": task.window,
        "variant": task.variant,
        "seed": task.seed,
        "evaluation_split": "test",
        "metrics": metric_summary,
        "temporal_weight_diagnostics": temporal_diagnostics,
        "structural_diagnostics": structural_diagnostics,
        "normalisation": normalisation.to_dict(),
        "best_checkpoint_sha256": sha256_file(checkpoint_path),
        "test_metrics_sha256": sha256_file(metrics_path),
        "freeze_hash": freeze_hash,
        "evaluation_precision": "amp_fp16_forward_fp32_metrics" if device.type == "cuda" else "fp32",
        "matched_evaluation_segments_disclosure": DISCLOSURE,
        "claim_scope": "noisy-centre-frame reconstruction fidelity only",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(summary, summary_path)
    issues = validate_frozen_evaluation(task, freeze_hash)
    if issues:
        raise ValueError(f"Evaluation contract failed for {task.task_id}: {'; '.join(issues)}")
    return summary


def evaluate_paper_tasks(tasks: list[ExperimentTask], device_name: str) -> dict[str, Any]:
    freeze = load_verified_freeze(tasks)
    completed_now = 0
    skipped = 0
    for position, task in enumerate(tasks, start=1):
        _, summary_path = _evaluation_paths(task)
        if summary_path.exists():
            print(f"[{position}/{len(tasks)}] skip evaluated: {task.task_id}", flush=True)
            evaluate_frozen_task(task, freeze, device_name)
            skipped += 1
            continue
        print(f"[{position}/{len(tasks)}] evaluate: {task.task_id}", flush=True)
        evaluate_frozen_task(task, freeze, device_name)
        completed_now += 1
    return {"selected": len(tasks), "evaluated_now": completed_now, "skipped_valid": skipped}


class _TemporalMean(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=1, keepdim=True)


def evaluate_temporal_mean(device_name: str) -> list[dict[str, Any]]:
    freeze = load_verified_freeze()
    representatives = {
        (task.recording, task.window): task
        for task in paper_tasks("core")
        if task.seed == 0 and task.variant == CORE_VARIANTS[0]
    }
    output = []
    for (recording, window), task in sorted(representatives.items()):
        target_dir = EVIDENCE_ROOT / "temporal_mean" / recording / window
        metrics_path = target_dir / "test_metrics.csv"
        summary_path = target_dir / "summary.json"
        if summary_path.is_file():
            output.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        device = torch.device(device_name)
        datasets, normalisation = build_recording_datasets(task.dataset_root, task.split_manifest)
        loader = _loader(
            datasets["test"],
            int(task.training.get("batch_size", 4)),
            int(task.training.get("num_workers", 0)),
            False,
            2_000_003,
            device.type == "cuda",
            _new_generator(2_000_003),
        )
        rows, metrics, _ = _evaluate(
            _TemporalMean().to(device), loader, device, False, False
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(rows, metrics_path)
        summary = {
            "schema_version": 1,
            "recording": recording,
            "window": window,
            "variant": "temporal_mean",
            "seed": None,
            "evaluation_split": "test",
            "metrics": metrics,
            "normalisation": normalisation.to_dict(),
            "split_manifest_sha256": sha256_file(task.split_manifest),
            "freeze_hash": freeze["freeze_hash"],
            "matched_evaluation_segments_disclosure": DISCLOSURE,
        }
        _write_json(summary, summary_path)
        output.append(summary)
    return output


def _read_metric_rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[int, dict[str, float]] = {}
    for row in rows:
        frame_index = int(row["frame_index"])
        if frame_index in indexed:
            raise ValueError(f"Duplicate frame index in {path}: {frame_index}")
        indexed[frame_index] = {metric: float(row[metric]) for metric in METRICS}
    return indexed


def _delta_rows(summaries: dict[tuple[str, str, int, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for effect, (candidate, reference) in PAIR_DEFINITIONS.items():
        candidate_keys = [key for key in summaries if key[3] == candidate]
        for recording, window, seed, _ in sorted(candidate_keys):
            candidate_summary = summaries[(recording, window, seed, candidate)]
            reference_summary = summaries.get((recording, window, seed, reference))
            if reference_summary is None:
                raise ValueError(f"Missing paired reference for {effect}: {recording}/{window}/seed{seed}")
            row: dict[str, Any] = {
                "effect": effect,
                "candidate": candidate,
                "reference": reference,
                "recording": recording,
                "window": window,
                "seed": seed,
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = float(candidate_summary["metrics"][f"mean_{metric}"]) - float(
                    reference_summary["metrics"][f"mean_{metric}"]
                )
            run_rows.append(row)
            candidate_frames = _read_metric_rows(Path(candidate_summary["summary_path"]).with_name("test_metrics.csv"))
            reference_frames = _read_metric_rows(Path(reference_summary["summary_path"]).with_name("test_metrics.csv"))
            if set(candidate_frames) != set(reference_frames):
                raise ValueError(f"Frame-index mismatch for {effect}: {recording}/{window}/seed{seed}")
            for frame_index in sorted(candidate_frames):
                frame_row: dict[str, Any] = {
                    "effect": effect,
                    "recording": recording,
                    "window": window,
                    "seed": seed,
                    "frame_index": frame_index,
                }
                for metric in METRICS:
                    frame_row[f"delta_{metric}"] = candidate_frames[frame_index][metric] - reference_frames[frame_index][metric]
                frame_rows.append(frame_row)

    factorial = {effect: [row for row in run_rows if row["effect"] == effect] for effect in ("structure_mae", "structure_mse")}
    by_key = {
        effect: {(row["recording"], row["window"], row["seed"]): row for row in rows}
        for effect, rows in factorial.items()
    }
    if set(by_key["structure_mae"]) != set(by_key["structure_mse"]):
        raise ValueError("Factorial interaction keys do not match")
    for key in sorted(by_key["structure_mae"]):
        mae_row = by_key["structure_mae"][key]
        mse_row = by_key["structure_mse"][key]
        row = {
            "effect": "structure_objective_interaction",
            "candidate": "difference_of_structure_effects",
            "reference": "MSE minus MAE",
            "recording": key[0],
            "window": key[1],
            "seed": key[2],
        }
        for metric in METRICS:
            row[f"delta_{metric}"] = mse_row[f"delta_{metric}"] - mae_row[f"delta_{metric}"]
        run_rows.append(row)
    frame_factorial = {
        effect: {
            (row["recording"], row["window"], row["seed"], row["frame_index"]): row
            for row in frame_rows
            if row["effect"] == effect
        }
        for effect in ("structure_mae", "structure_mse")
    }
    if set(frame_factorial["structure_mae"]) != set(frame_factorial["structure_mse"]):
        raise ValueError("Frame-level factorial interaction keys do not match")
    for key in sorted(frame_factorial["structure_mae"]):
        mae_row = frame_factorial["structure_mae"][key]
        mse_row = frame_factorial["structure_mse"][key]
        frame_row = {
            "effect": "structure_objective_interaction",
            "recording": key[0],
            "window": key[1],
            "seed": key[2],
            "frame_index": key[3],
        }
        for metric in METRICS:
            frame_row[f"delta_{metric}"] = mse_row[f"delta_{metric}"] - mae_row[f"delta_{metric}"]
        frame_rows.append(frame_row)
    return run_rows, frame_rows


def _percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _hierarchical_bootstrap(rows: list[dict[str, Any]], replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    recordings = sorted({str(row["recording"]) for row in rows})
    lookup = {(str(row["recording"]), str(row["window"]), int(row["seed"])): float(row["delta_psnr_db"]) for row in rows}
    windows = {recording: sorted({key[1] for key in lookup if key[0] == recording}) for recording in recordings}
    seeds = {(recording, window): sorted(key[2] for key in lookup if key[:2] == (recording, window)) for recording in recordings for window in windows[recording]}
    samples = []
    for _ in range(replicates):
        values = []
        for recording_index in rng.integers(0, len(recordings), size=len(recordings)):
            recording = recordings[int(recording_index)]
            available_windows = windows[recording]
            for window_index in rng.integers(0, len(available_windows), size=len(available_windows)):
                window = available_windows[int(window_index)]
                available_seeds = seeds[(recording, window)]
                for seed_index in rng.integers(0, len(available_seeds), size=len(available_seeds)):
                    selected_seed = available_seeds[int(seed_index)]
                    values.append(lookup[(recording, window, selected_seed)])
        samples.append(statistics.fmean(values))
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _aggregate_effects(run_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates = []
    conditions = []
    recordings = []
    bootstrap = protocol["bootstrap"]
    for effect in sorted({str(row["effect"]) for row in run_rows}):
        selected = [row for row in run_rows if row["effect"] == effect]
        deltas = [float(row["delta_psnr_db"]) for row in selected]
        lower, upper = _hierarchical_bootstrap(selected, int(bootstrap["replicates"]), int(bootstrap["seed"]))
        aggregates.append({
            "effect": effect,
            "paired_runs": len(selected),
            "mean_delta_psnr_db": statistics.fmean(deltas),
            "sd_delta_psnr_db": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "median_delta_psnr_db": statistics.median(deltas),
            "min_delta_psnr_db": min(deltas),
            "max_delta_psnr_db": max(deltas),
            "positive_delta_count": sum(value > 0 for value in deltas),
            "bootstrap_sensitivity_low_db": lower,
            "bootstrap_sensitivity_high_db": upper,
        })
        for recording in sorted({str(row["recording"]) for row in selected}):
            recording_rows = [row for row in selected if row["recording"] == recording]
            recordings.append({
                "effect": effect,
                "recording": recording,
                "run_count": len(recording_rows),
                "mean_delta_psnr_db": statistics.fmean(float(row["delta_psnr_db"]) for row in recording_rows),
            })
            for window in sorted({str(row["window"]) for row in recording_rows}):
                condition_rows = [row for row in recording_rows if row["window"] == window]
                conditions.append({
                    "effect": effect,
                    "recording": recording,
                    "window": window,
                    "seed_count": len(condition_rows),
                    "mean_delta_psnr_db": statistics.fmean(float(row["delta_psnr_db"]) for row in condition_rows),
                })
    return aggregates, conditions, recordings


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        _write_rows(rows, path)


def build_paper_report() -> dict[str, Any]:
    tasks = paper_tasks("all", validate_paths=True)
    freeze = load_verified_freeze(tasks)
    summaries: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    run_index = []
    ctpf_diagnostics = []
    for task in tasks:
        issues = validate_frozen_evaluation(task, str(freeze["freeze_hash"]))
        if issues:
            raise ValueError(f"Incomplete frozen evaluation for {task.task_id}: {'; '.join(issues)}")
        summary_path = Path(task.output_dir) / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["summary_path"] = str(summary_path)
        key = (task.recording, task.window, task.seed, task.variant)
        if key in summaries:
            raise ValueError(f"Duplicate evaluated condition: {key}")
        summaries[key] = summary
        training_summary = json.loads((Path(task.output_dir) / "summary.json").read_text(encoding="utf-8"))
        run_index.append({
            "task_id": task.task_id,
            "stage": task.stage,
            "recording": task.recording,
            "window": task.window,
            "variant": task.variant,
            "seed": task.seed,
            "training_objective": training_summary["training_objective"]["name"],
            "selection_metric": training_summary["selection_metric"],
            "best_epoch": training_summary["best_epoch"],
            "parameters": training_summary["deployed_profile"]["parameters"],
            "conv_gflops": training_summary["deployed_profile"]["conv_gflops"],
            "mean_psnr_db": summary["metrics"]["mean_psnr_db"],
            "mean_normalised_mse": summary["metrics"]["mean_normalised_mse"],
            "mean_normalised_mae": summary["metrics"]["mean_normalised_mae"],
            "mean_gssim": summary["metrics"]["mean_gssim"],
            "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
            "test_metrics_sha256": summary["test_metrics_sha256"],
        })
        temporal = summary.get("temporal_weight_diagnostics")
        if temporal is not None:
            ctpf_diagnostics.append({
                "task_id": task.task_id,
                "recording": task.recording,
                "window": task.window,
                "seed": task.seed,
                "variant": task.variant,
                "mean_normalised_entropy": temporal.get("mean_normalised_entropy"),
                "mean_max_frame_weight": temporal.get("mean_max_frame_weight"),
                "mean_frame_weights_json": json.dumps(temporal.get("mean_frame_weights", [])),
            })
    temporal_mean = evaluate_temporal_mean("cuda" if torch.cuda.is_available() else "cpu")
    run_rows, frame_rows = _delta_rows(summaries)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    aggregates, conditions, recording_rows = _aggregate_effects(run_rows, protocol)
    tables = EVIDENCE_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_table(run_index, tables / "run_index.csv")
    _write_table(run_rows, tables / "paired_effects_run_level.csv")
    _write_table(aggregates, tables / "paired_effects_aggregate.csv")
    _write_table(conditions, tables / "paired_effects_condition_level.csv")
    _write_table(recording_rows, tables / "paired_effects_recording_level.csv")
    _write_table(frame_rows, tables / "paired_effects_frame_level.csv")
    _write_table(ctpf_diagnostics, tables / "ctpf_temporal_weight_diagnostics.csv")
    _write_table(
        [
            {
                "recording": row["recording"],
                "window": row["window"],
                **row["metrics"],
            }
            for row in temporal_mean
        ],
        tables / "temporal_mean_results.csv",
    )
    source_rows = []
    for task in paper_tasks("core"):
        if task.seed != 0 or task.variant != CORE_VARIANTS[0]:
            continue
        source_rows.append({
            "recording": task.recording,
            "window": task.window,
            "dataset_root": task.dataset_root,
            "split_manifest": task.split_manifest,
            "split_manifest_sha256": sha256_file(task.split_manifest),
            "official_source": "https://github.com/codeneuro/neurofinder",
            "redistribution_boundary": "No explicit dataset licence identified; pixels and recoverable arrays are not redistributed.",
        })
    _write_table(source_rows, tables / "source_mapping.csv")
    structure_conditions = [row for row in conditions if row["effect"] == "structure_mse"]
    structure_average = statistics.fmean(float(row["mean_delta_psnr_db"]) for row in structure_conditions)
    thresholds = protocol["engineering_thresholds"]
    gates = {
        "quality_retention": {
            "passed": structure_average >= float(thresholds["quality_retention_mean_delta_psnr_db_min"])
            and min(float(row["mean_delta_psnr_db"]) for row in structure_conditions)
            >= float(thresholds["quality_retention_min_condition_delta_psnr_db"]),
            "equal_condition_mean_delta_psnr_db": structure_average,
            "minimum_condition_delta_psnr_db": min(float(row["mean_delta_psnr_db"]) for row in structure_conditions),
        },
        "quality_improvement": {
            "passed": structure_average >= float(thresholds["quality_improvement_mean_delta_psnr_db_min"]),
            "equal_condition_mean_delta_psnr_db": structure_average,
        },
        "speed": {"passed": None, "reason": "requires completed paper-latency audit"},
    }
    latency_summary_path = EVIDENCE_ROOT / "latency" / "latency_summary.json"
    if latency_summary_path.is_file():
        latency_summary = json.loads(latency_summary_path.read_text(encoding="utf-8"))
        gates["speed"] = dict(latency_summary["speed_gate"])
    claim_rows = [
        {
            "claim_id": "C1",
            "claim": "Training-objective effect at fixed full-resolution and subpixel inference graphs",
            "primary_evidence": "tables/paired_effects_run_level.csv; tables/paired_effects_aggregate.csv",
            "decision_rule": "Report paired estimates and sensitivity intervals without significance or equivalence claims.",
            "status": "measured",
        },
        {
            "claim_id": "C2",
            "claim": "Subpixel quality retention relative to the matched full-resolution MSE arm",
            "primary_evidence": "tables/paired_effects_condition_level.csv",
            "decision_rule": "Mean delta >= -0.01 dB and minimum condition delta >= -0.03 dB.",
            "status": "supported" if gates["quality_retention"]["passed"] else "not_supported",
        },
        {
            "claim_id": "C3",
            "claim": "Quality return from the B8-D4 capacity reference relative to G4",
            "primary_evidence": "tables/paired_effects_run_level.csv",
            "decision_rule": "Descriptive paired reconstruction evidence only.",
            "status": "measured",
        },
        {
            "claim_id": "C4",
            "claim": "Device-specific batch-1 inference speed gain for G4 relative to G2",
            "primary_evidence": "latency/subpixel_vs_fullres_batch1.csv",
            "decision_rule": "p50 ratio <= 0.75 in at least eight of nine conditions.",
            "status": (
                "supported" if gates["speed"].get("passed") is True else
                "not_supported" if gates["speed"].get("passed") is False else
                "pending_latency"
            ),
        },
    ]
    _write_table(claim_rows, tables / "claim_evidence_matrix.csv")
    report = {
        "schema_version": 1,
        "protocol_name": "paper_evidence_v1",
        "freeze_hash": freeze["freeze_hash"],
        "training_tasks": len(tasks),
        "evaluated_tasks": len(summaries),
        "temporal_mean_conditions": len(temporal_mean),
        "matched_evaluation_segments_disclosure": DISCLOSURE,
        "effect_aggregates": aggregates,
        "engineering_gates": gates,
        "claim_scope": protocol["claim_scope"],
    }
    _write_json(report, EVIDENCE_ROOT / "paper_report.json")
    evidence_files = sorted(path for path in tables.glob("*.csv")) + [
        FREEZE_PATH,
        EVIDENCE_ROOT / "paper_report.json",
        PROTOCOL_PATH,
        PROJECT_ROOT / "scripts" / "plot_paper_evidence_figure6.R",
    ]
    if latency_summary_path.is_file():
        evidence_files.extend(sorted((EVIDENCE_ROOT / "latency").glob("*.csv")))
        evidence_files.append(latency_summary_path)
    manifest = {
        "schema_version": 1,
        "protocol_name": "paper_evidence_v1",
        "freeze_hash": freeze["freeze_hash"],
        "files": [
            {"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in evidence_files
        ],
        "excluded_content": ["Neurofinder pixels", "prediction frames", "recoverable image arrays"],
    }
    _write_json(manifest, EVIDENCE_ROOT / "evidence_manifest.json")
    return report
