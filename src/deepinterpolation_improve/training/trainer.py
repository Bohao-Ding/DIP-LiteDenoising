from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepinterpolation_improve.config import TrainingConfig
from deepinterpolation_improve.manifest import build_manifest
from .losses import get_loss


def _seed_worker(worker_seed: int, worker_id: int) -> None:
    seed = int(worker_seed) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _cpu_byte_rng_state(state: torch.Tensor) -> torch.Tensor:
    return state.detach().to(device="cpu", dtype=torch.uint8)


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config: TrainingConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.channels_last = bool(config.channels_last and len(getattr(train_dataset, "input_shape", ())) == 3)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32)
            torch.backends.cudnn.allow_tf32 = bool(config.tf32)
            torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high" if config.tf32 else "highest")
        if config.compile_model:
            self.model = torch.compile(self.model)
        self.use_amp = bool(config.mixed_precision and config.precision != "fp32" and self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if config.precision == "amp_bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(self.use_amp and config.precision == "amp_fp16"))
        self.loss_fn = get_loss(config.loss)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma,
            )
            if config.scheduler_step_size
            else None
        )
        self.start_epoch = 0
        self.best_loss: float | None = None
        self.history: list[dict[str, float | int | str]] = []
        self._set_seed(config.seed)
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(int(config.seed))
        self.val_generator = torch.Generator()
        self.val_generator.manual_seed(int(config.seed) + 1_000_003)
        self.loader_rng_restored = False
        object.__setattr__(self.config, "loader_rng_restored", False)
        if config.resume_from is not None:
            self._load_checkpoint(config.resume_from)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        generator = self.train_generator if shuffle else self.val_generator
        worker_seed = int(self.config.worker_seed if self.config.worker_seed is not None else self.config.seed)

        kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "pin_memory": bool(self.config.pin_memory and self.device.type == "cuda"),
            "persistent_workers": bool(self.config.persistent_workers and self.config.num_workers > 0),
            "generator": generator,
        }
        if self.config.num_workers > 0:
            kwargs["worker_init_fn"] = partial(_seed_worker, worker_seed)
        if self.config.num_workers > 0 and self.config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(
            dataset,
            **kwargs,
        )

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, dtype=torch.float32, non_blocking=self.config.non_blocking)
        if self.channels_last and tensor.ndim == 4:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _write_progress(
        self,
        stage: str,
        *,
        epoch: int | None = None,
        current_loss: float | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        error: str | None = None,
    ) -> None:
        if self.config.progress_path is None:
            return
        path = Path(self.config.progress_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.config.progress_metadata or {})
        payload = {
            "schema_version": 1,
            "run_uid": self.config.run_uid,
            "model_name": self.config.model_name,
            "stage": stage,
            "epoch": int(epoch if epoch is not None else self.start_epoch),
            "epochs": int(self.config.epochs),
            "current_loss": None if current_loss is None else float(current_loss),
            "train_loss": None if train_loss is None else float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pid": int(os.getpid()),
            "output_dir": str(self.output_dir),
        }
        if error is not None:
            payload["error"] = error
        payload.update(metadata)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int | None = None) -> dict[str, float | int | str]:
        self.model.train(train)
        total_loss = 0.0
        total_samples = 0
        batch_count = 0
        frame_indices: list[int] = []
        data_seconds = 0.0
        h2d_seconds = 0.0
        compute_seconds = 0.0
        epoch_start = time.perf_counter()
        last_progress_write = 0.0
        with torch.set_grad_enabled(train):
            iterator = iter(loader)
            while True:
                load_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                data_seconds += time.perf_counter() - load_start
                transfer_start = time.perf_counter()
                inputs = self._prepare_tensor(batch["input"])
                targets = self._prepare_tensor(batch["target"])
                h2d_seconds += time.perf_counter() - transfer_start
                if "frame_index" in batch:
                    frame_indices.extend([int(value) for value in batch["frame_index"].detach().cpu().numpy().tolist()])
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                compute_start = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    predictions = self.model(inputs)
                    loss = self.loss_fn(predictions, targets)
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = int(inputs.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_samples += batch_size
                batch_count += 1
                compute_seconds += time.perf_counter() - compute_start
                if train and self.config.progress_path is not None:
                    now = time.perf_counter()
                    if batch_count == 1 or now - last_progress_write >= 5.0:
                        running_loss = total_loss / max(total_samples, 1)
                        self._write_progress(
                            "training",
                            epoch=epoch,
                            current_loss=running_loss,
                            train_loss=running_loss,
                        )
                        last_progress_write = now
        order_hash = hashlib.sha256(",".join(str(value) for value in frame_indices).encode("utf-8")).hexdigest()
        seconds = time.perf_counter() - epoch_start
        return {
            "loss": total_loss / max(total_samples, 1),
            "frame_index_hash": order_hash,
            "batch_count": batch_count,
            "sample_count": total_samples,
            "seconds": seconds,
            "data_seconds": data_seconds,
            "h2d_seconds": h2d_seconds,
            "compute_seconds": compute_seconds,
            "samples_per_second": float(total_samples) / max(seconds, 1e-9),
        }

    def fit(self) -> dict[str, float | int]:
        train_loader = self._loader(self.train_dataset, shuffle=bool(self.config.shuffle_train))
        val_loader = self._loader(self.val_dataset, shuffle=False)
        try:
            for epoch in range(self.start_epoch + 1, self.config.epochs + 1):
                self._write_progress("training", epoch=epoch)
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                epoch_start = time.perf_counter()
                train_result = self._run_epoch(train_loader, train=True, epoch=epoch)
                train_loss = float(train_result["loss"])
                self._write_progress("validation", epoch=epoch, current_loss=train_loss, train_loss=train_loss)
                val_result = self._run_epoch(val_loader, train=False, epoch=epoch)
                val_loss = float(val_result["loss"])
                if self.scheduler is not None:
                    self.scheduler.step()
                peak_memory = (
                    float(torch.cuda.max_memory_allocated(self.device) / (1024**2))
                    if self.device.type == "cuda"
                    else 0.0
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "cuda_peak_memory_mb": peak_memory,
                    "train_frame_index_hash": str(train_result["frame_index_hash"]),
                    "train_batches": int(train_result["batch_count"]),
                    "val_batches": int(val_result["batch_count"]),
                    "train_samples": int(train_result["sample_count"]),
                    "val_samples": int(val_result["sample_count"]),
                    "samples_per_second": float(train_result["samples_per_second"]),
                    "train_samples_per_second": float(train_result["samples_per_second"]),
                    "val_samples_per_second": float(val_result["samples_per_second"]),
                    "train_data_seconds": float(train_result["data_seconds"]),
                    "train_h2d_seconds": float(train_result["h2d_seconds"]),
                    "train_compute_seconds": float(train_result["compute_seconds"]),
                    "val_data_seconds": float(val_result["data_seconds"]),
                    "val_h2d_seconds": float(val_result["h2d_seconds"]),
                    "val_compute_seconds": float(val_result["compute_seconds"]),
                }
                self.history.append(row)
                is_best = self.best_loss is None or val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                self._write_progress(
                    "checkpointing",
                    epoch=epoch,
                    current_loss=val_loss,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                self._save_checkpoint(epoch, is_best=is_best, is_final=epoch == self.config.epochs)
                self._write_history()
                self._write_manifest(epoch)
        except Exception as exc:
            self._write_progress("failed", epoch=self.start_epoch, error=repr(exc))
            raise
        if not self.history:
            return {"epoch": self.start_epoch, "train_loss": 0.0, "val_loss": 0.0}
        last = self.history[-1]
        self._write_progress(
            "complete",
            epoch=int(last["epoch"]),
            current_loss=float(last["val_loss"]),
            train_loss=float(last["train_loss"]),
            val_loss=float(last["val_loss"]),
        )
        return self.history[-1]

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "history": self.history,
            "rng_state": {
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "loader_rng_state": {
                "train": self.train_generator.get_state(),
                "val": self.val_generator.get_state(),
            },
        }

    def _atomic_save(self, payload: dict, path: Path) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def _save_checkpoint(self, epoch: int, is_best: bool, is_final: bool = False) -> None:
        payload = self._checkpoint_payload(epoch)
        should_save_latest = is_final or epoch % int(self.config.checkpoint_every) == 0
        if should_save_latest:
            self._atomic_save(payload, self.output_dir / "checkpoint.pt")
        if is_best:
            self._atomic_save(payload, self.output_dir / "best.pt")

    def _load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and checkpoint.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = int(checkpoint["epoch"])
        self.best_loss = checkpoint.get("best_loss")
        self.history = list(checkpoint.get("history", []))
        loader_rng_state = checkpoint.get("loader_rng_state")
        if loader_rng_state:
            self.train_generator.set_state(_cpu_byte_rng_state(loader_rng_state["train"]))
            self.val_generator.set_state(_cpu_byte_rng_state(loader_rng_state["val"]))
            self.loader_rng_restored = True
            object.__setattr__(self.config, "loader_rng_restored", True)
        rng_state = checkpoint.get("rng_state", {})
        if rng_state:
            torch.random.set_rng_state(_cpu_byte_rng_state(rng_state["torch"]))
            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([_cpu_byte_rng_state(state) for state in rng_state["cuda"]])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])

    def _write_history(self) -> None:
        path = self.output_dir / "loss_history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "epoch_seconds",
                    "cuda_peak_memory_mb",
                    "train_frame_index_hash",
                    "train_batches",
                    "val_batches",
                    "train_samples",
                    "val_samples",
                    "samples_per_second",
                    "train_samples_per_second",
                    "val_samples_per_second",
                    "train_data_seconds",
                    "train_h2d_seconds",
                    "train_compute_seconds",
                    "val_data_seconds",
                    "val_h2d_seconds",
                    "val_compute_seconds",
                ],
            )
            writer.writeheader()
            writer.writerows(self.history)

    def _write_manifest(self, epoch: int) -> None:
        manifest = build_manifest(
            self.config.model_name,
            self.train_dataset,
            self.config,
            self.best_loss,
            epoch,
        )
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
