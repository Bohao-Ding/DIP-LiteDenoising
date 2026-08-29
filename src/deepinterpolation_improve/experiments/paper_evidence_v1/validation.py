from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .models import variant_spec
from .study import ExperimentTask, sha256_file


REQUIRED_RUN_FILES = (
    "status.json",
    "run_manifest.json",
    "loss_history.csv",
    "checkpoint.pt",
    "best.pt",
    "summary.json",
)
CLAIM_SCOPE = "noisy-centre-frame reconstruction fidelity only"


def _validate_precision_protocol(payload: dict[str, Any], task: ExperimentTask, issues: list[str], source: str) -> None:
    protocol = payload.get("precision_protocol")
    if not isinstance(protocol, dict):
        issues.append(f"{source} precision protocol is missing")
        return
    expected = {
        "training_requested": task.training.get("precision"),
        "model_parameters": "fp32",
        "optimiser_state": "fp32",
        "metric_computation": task.training.get("metric_precision"),
        "latency_benchmark": (
            task.benchmark.get("precision") if bool(task.benchmark.get("enabled", False)) else None
        ),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            issues.append(f"{source} precision protocol mismatch: {key}")
    executed = protocol.get("training_executed")
    if executed not in {"amp_fp16", "fp32"}:
        issues.append(f"{source} training execution precision is invalid")
    if protocol.get("test_forward") != executed:
        issues.append(f"{source} test-forward precision does not match training execution")
    if protocol.get("gradient_scaling") is not (executed == "amp_fp16"):
        issues.append(f"{source} gradient-scaling state does not match execution precision")


def _read_json(path: Path, issues: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot parse {path.name}: {exc}")
        return None


def _validate_training_objective(
    payload: dict[str, Any], task: ExperimentTask, issues: list[str], source: str
) -> None:
    objective = payload.get("training_objective")
    if not isinstance(objective, dict):
        if task.stage == "innovation_screen":
            issues.append(f"{source} training objective is missing")
        return
    expected_name = variant_spec(task.variant).training_objective
    if objective.get("name") != expected_name:
        issues.append(f"{source} training objective mismatch")
    if expected_name in {"target_l1", "target_mse"}:
        if float(objective.get("target_weight", -1.0)) != 1.0:
            issues.append(f"{source} target weight mismatch")
        return
    distillation = task.training.get("distillation", {})
    expected_weight = float(distillation.get("weight", -1.0))
    if not math.isclose(float(objective.get("distillation_weight", -1.0)), expected_weight):
        issues.append(f"{source} distillation weight mismatch")
    if not math.isclose(float(objective.get("target_weight", -1.0)), 1.0 - expected_weight):
        issues.append(f"{source} target weight mismatch")
    if objective.get("teacher_variant") != distillation.get("teacher_variant"):
        issues.append(f"{source} teacher variant mismatch")
    teacher_checkpoint = Path(str(distillation.get("teacher_checkpoint", "")))
    if teacher_checkpoint.is_file() and objective.get("teacher_checkpoint_sha256") != sha256_file(
        teacher_checkpoint
    ):
        issues.append(f"{source} teacher checkpoint SHA-256 mismatch")


def _csv_rows(path: Path, issues: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        issues.append(f"cannot parse {path.name}: {exc}")
        return []


def validate_run_directory(task: ExperimentTask) -> list[str]:
    output = Path(task.output_dir)
    issues: list[str] = []
    for filename in REQUIRED_RUN_FILES:
        path = output / filename
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing or empty {filename}")
    evaluation_split = str(task.training.get("evaluation_split", "test"))
    metrics_filename = f"{evaluation_split}_metrics.csv"
    metrics_path = output / metrics_filename
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        issues.append(f"missing or empty {metrics_filename}")
    if issues:
        return issues

    status = _read_json(output / "status.json", issues)
    manifest = _read_json(output / "run_manifest.json", issues)
    summary = _read_json(output / "summary.json", issues)
    history = _csv_rows(output / "loss_history.csv", issues)
    metrics = _csv_rows(metrics_path, issues)
    if status and (status.get("task_id") != task.task_id or status.get("state") != "completed"):
        issues.append("status.json does not identify a completed matching task")
    if manifest:
        if manifest.get("task", {}).get("task_id") != task.task_id:
            issues.append("run_manifest task_id mismatch")
        if manifest.get("centre_frame_excluded") is not True:
            issues.append("run_manifest does not confirm centre-frame exclusion")
        if manifest.get("claim_scope") != CLAIM_SCOPE:
            issues.append("run_manifest claim scope mismatch")
        _validate_precision_protocol(manifest, task, issues, "run_manifest")
        _validate_training_objective(manifest, task, issues, "run_manifest")
        if Path(task.split_manifest).is_file() and manifest.get("split_manifest_sha256") != sha256_file(
            task.split_manifest
        ):
            issues.append("split manifest SHA-256 mismatch")
    if summary:
        if summary.get("task_id") != task.task_id:
            issues.append("summary task_id mismatch")
        if summary.get("claim_scope") != CLAIM_SCOPE:
            issues.append("summary claim scope mismatch")
        if summary.get("evaluation_split", "test") != evaluation_split:
            issues.append("summary evaluation split mismatch")
        _validate_precision_protocol(summary, task, issues, "summary")
        _validate_training_objective(summary, task, issues, "summary")
        expected_epochs = int(task.training.get("epochs", 0))
        best_epoch = int(summary.get("best_epoch", 0))
        if not (1 <= best_epoch <= expected_epochs):
            issues.append("best_epoch is outside the configured training range")
        metric_summary = summary.get("metrics", {})
        if int(metric_summary.get("frame_count", -1)) != len(metrics):
            issues.append(f"{evaluation_split} metric row count does not match summary")
        for key, value in metric_summary.items():
            if key != "frame_count" and not math.isfinite(float(value)):
                issues.append(f"non-finite summary metric: {key}")
        if bool(task.benchmark.get("enabled", False)) and summary.get("latency") is None:
            issues.append("configured latency benchmark is missing")
        if bool(task.benchmark.get("enabled", False)) and summary.get("latency") is not None:
            latency = summary["latency"]
            for key in ("batch_size", "warmup", "repeats", "precision", "channels_last"):
                expected = task.benchmark.get(key)
                if latency.get(key) != expected:
                    issues.append(f"latency protocol mismatch: {key}")
        expects_temporal_weights = variant_spec(task.variant).reports_temporal_weights
        if expects_temporal_weights != (summary.get("temporal_weight_diagnostics") is not None):
            issues.append("temporal-weight diagnostics do not match the model contract")
    if len(history) != int(task.training.get("epochs", 0)):
        issues.append("loss history row count does not match configured epochs")
    return issues


def audit_stage_artifacts(tasks: list[ExperimentTask]) -> dict[str, Any]:
    checked = 0
    valid = 0
    failures = []
    for task in tasks:
        if not (Path(task.output_dir) / "summary.json").is_file():
            continue
        checked += 1
        issues = validate_run_directory(task)
        if issues:
            failures.append({"task_id": task.task_id, "issues": issues})
        else:
            valid += 1
    return {"checked_completed_runs": checked, "valid_runs": valid, "failures": failures}
