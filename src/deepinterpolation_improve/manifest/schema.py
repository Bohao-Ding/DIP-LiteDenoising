from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class ManifestValidationError(ValueError):
    pass


def build_manifest(model_name: str, dataset: Any, config: Any, best_loss: float | None, epoch: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {
            "name": model_name,
            "input_shape": list(dataset.input_shape),
            "target_shape": list(dataset.target_shape),
        },
        "window": dataset.window.to_dict(),
        "normalization": dataset.normalization_dict(),
        "training": {
            "run_uid": config.run_uid,
            "loss": config.loss,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "completed_epoch": epoch,
            "best_loss": best_loss,
            "mixed_precision": getattr(config, "mixed_precision", False),
            "precision": getattr(config, "precision", "fp32"),
            "pin_memory": getattr(config, "pin_memory", False),
            "non_blocking": getattr(config, "non_blocking", False),
            "channels_last": getattr(config, "channels_last", False),
            "tf32": getattr(config, "tf32", False),
            "compile_model": getattr(config, "compile_model", False),
            "cudnn_benchmark": getattr(config, "cudnn_benchmark", False),
            "num_workers": getattr(config, "num_workers", 0),
            "prefetch_factor": getattr(config, "prefetch_factor", None),
            "shuffle_train": getattr(config, "shuffle_train", False),
            "checkpoint_every": getattr(config, "checkpoint_every", 1),
            "worker_seed": getattr(config, "worker_seed", None),
            "loader_rng_restored": getattr(config, "loader_rng_restored", False),
        },
        "runtime": {"torch_version": torch.__version__, "device": config.device},
        "checkpoints": {"latest": "checkpoint.pt", "best": "best.pt"},
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest_for_dataset(manifest: dict[str, Any], dataset: Any) -> None:
    if int(manifest.get("schema_version", 0)) != 1:
        raise ManifestValidationError("Unsupported or missing schema_version")
    model = manifest.get("model", {})
    expected_input = list(dataset.input_shape)
    if model.get("input_shape") != expected_input:
        raise ManifestValidationError(
            f"input_shape mismatch: manifest={model.get('input_shape')} dataset={expected_input}"
        )
    window = manifest.get("window", {})
    current_window = dataset.window.to_dict()
    if window != current_window:
        raise ManifestValidationError(f"window mismatch: manifest={window} dataset={current_window}")
