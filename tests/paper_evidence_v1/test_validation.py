from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from deepinterpolation_improve.experiments.paper_evidence_v1.study import ExperimentTask, sha256_file
from deepinterpolation_improve.experiments.paper_evidence_v1.validation import CLAIM_SCOPE, validate_run_directory


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_completed_run_contract_detects_missing_and_inconsistent_artifacts() -> None:
    with TemporaryDirectory(prefix="dip_validation_test_") as temporary:
        root = Path(temporary)
        output = root / "run"
        output.mkdir()
        split = root / "split.json"
        split.write_text("{}", encoding="utf-8")
        task = ExperimentTask(
            task_id="contract-task",
            stage="quick_screen",
            recording="01.00",
            window="w2",
            variant="baseline_b8d3",
            seed=0,
            dataset_root="data",
            split_manifest=str(split),
            output_dir=str(output),
            input_channels=4,
            frame_shape=(8, 8),
            training={"epochs": 1, "precision": "amp_fp16", "metric_precision": "fp32"},
            benchmark={"enabled": False},
            source_config="config.json",
        )
        (output / "status.json").write_text(
            json.dumps({"task_id": task.task_id, "state": "completed"}), encoding="utf-8"
        )
        (output / "run_manifest.json").write_text(
            json.dumps(
                {
                    "task": task.to_dict(),
                    "centre_frame_excluded": True,
                    "claim_scope": CLAIM_SCOPE,
                    "split_manifest_sha256": sha256_file(split),
                    "precision_protocol": {
                        "training_requested": "amp_fp16",
                        "training_executed": "fp32",
                        "model_parameters": "fp32",
                        "optimiser_state": "fp32",
                        "gradient_scaling": False,
                        "test_forward": "fp32",
                        "metric_computation": "fp32",
                        "latency_benchmark": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        _write_csv(output / "loss_history.csv", [{"epoch": 1, "train_mae": 1.0, "val_mae": 1.0}])
        _write_csv(
            output / "test_metrics.csv",
            [
                {"frame_index": 1, "psnr_db": 18.0},
                {"frame_index": 2, "psnr_db": 18.1},
            ],
        )
        (output / "checkpoint.pt").write_bytes(b"checkpoint")
        (output / "best.pt").write_bytes(b"best")
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "best_epoch": 1,
                    "metrics": {"frame_count": 2, "mean_psnr_db": 18.05},
                    "temporal_weight_diagnostics": None,
                    "latency": None,
                    "precision_protocol": {
                        "training_requested": "amp_fp16",
                        "training_executed": "fp32",
                        "model_parameters": "fp32",
                        "optimiser_state": "fp32",
                        "gradient_scaling": False,
                        "test_forward": "fp32",
                        "metric_computation": "fp32",
                        "latency_benchmark": None,
                    },
                    "claim_scope": CLAIM_SCOPE,
                }
            ),
            encoding="utf-8",
        )
        assert validate_run_directory(task) == []
        (output / "best.pt").unlink()
        assert "missing or empty best.pt" in validate_run_directory(task)
