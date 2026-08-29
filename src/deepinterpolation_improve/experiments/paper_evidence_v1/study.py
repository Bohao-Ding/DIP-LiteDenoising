from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import registered_variants
from .paths import load_path_overrides, resolve_override


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ExperimentTask:
    task_id: str
    stage: str
    recording: str
    window: str
    variant: str
    seed: int
    dataset_root: str
    split_manifest: str
    output_dir: str
    input_channels: int
    frame_shape: tuple[int, int]
    training: dict[str, Any]
    benchmark: dict[str, Any]
    source_config: str
    path_override: str | None = None
    path_override_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported stage schema in {config_path}")
    validate_stage_config(config, config_path)
    return config


def _require_unique_nonempty(config: dict[str, Any], key: str) -> list[Any]:
    values = list(config.get(key, []))
    if not values:
        raise ValueError(f"{key} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{key} contains duplicate values")
    return values


def validate_stage_config(config: dict[str, Any], source: str | Path = "<config>") -> None:
    """Validate the experimental contract before any task is created."""

    stage = str(config.get("stage", ""))
    supported = {
        "quick_screen",
        "targeted_screen",
        "innovation_screen",
        "single_seed_confirmation",
        "multi_seed_confirmation",
        "next_generation_screen",
        "next_generation_single_seed_confirmation",
        "next_generation_multi_seed_confirmation",
        "objective_controlled_structure_screen",
        "paper_evidence_core",
        "paper_evidence_mechanism",
    }
    if stage not in supported:
        raise ValueError(f"Unsupported stage {stage!r} in {source}")
    recordings = _require_unique_nonempty(config, "recordings")
    windows = _require_unique_nonempty(config, "windows")
    variants = _require_unique_nonempty(config, "variants")
    seeds = [int(value) for value in _require_unique_nonempty(config, "seeds")]
    reference_variant = str(config.get("reference_variant", "baseline_b8d3"))
    if reference_variant not in variants:
        raise ValueError(
            f"reference_variant {reference_variant!r} must be included in variants"
        )
    if any(not re.fullmatch(r"w[1-9][0-9]*", str(window)) for window in windows):
        raise ValueError(f"Invalid window labels in {source}")
    data = config.get("data", {})
    roots = data.get("recording_roots", {})
    missing_roots = [recording for recording in recordings if recording not in roots]
    if missing_roots:
        raise ValueError(f"Missing recording roots: {', '.join(missing_roots)}")
    if "{recording}" not in str(data.get("split_pattern", "")) or "{window}" not in str(
        data.get("split_pattern", "")
    ):
        raise ValueError("split_pattern must contain {recording} and {window}")
    if stage in {
        "quick_screen",
        "targeted_screen",
        "innovation_screen",
        "single_seed_confirmation",
        "next_generation_screen",
        "next_generation_single_seed_confirmation",
        "objective_controlled_structure_screen",
        "paper_evidence_mechanism",
    } and seeds != [0]:
        raise ValueError(f"{stage} is restricted to seed 0")
    if stage in {
        "multi_seed_confirmation",
        "next_generation_multi_seed_confirmation",
        "paper_evidence_core",
    } and len(seeds) < 2:
        raise ValueError(f"{stage} requires at least two seeds")

    training = config.get("training", {})
    for key in ("epochs", "batch_size", "checkpoint_every"):
        if int(training.get(key, 0)) < 1:
            raise ValueError(f"training.{key} must be >= 1")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if int(training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers must be >= 0")
    scheduler_step = training.get("scheduler_step_size")
    if scheduler_step is not None and int(scheduler_step) < 1:
        raise ValueError("training.scheduler_step_size must be >= 1 when provided")
    if float(training.get("scheduler_gamma", 0.0)) <= 0:
        raise ValueError("training.scheduler_gamma must be > 0")
    if "mixed_precision" in training:
        raise ValueError("training.mixed_precision is obsolete; use training.precision")
    if str(training.get("precision", "")) != "amp_fp16":
        raise ValueError("training.precision must be amp_fp16 for this study")
    if str(training.get("metric_precision", "")) != "fp32":
        raise ValueError("training.metric_precision must be fp32 for this study")
    if str(training.get("evaluation_split", "test")) not in {"val", "test"}:
        raise ValueError("training.evaluation_split must be val or test")
    if stage in {"paper_evidence_core", "paper_evidence_mechanism"} and str(
        training.get("evaluation_split", "test")
    ) != "val":
        raise ValueError(f"{stage} requires training.evaluation_split=val before freezing")
    if "subpixel_distilled_b8d3" in variants:
        distillation = training.get("distillation")
        if not isinstance(distillation, dict):
            raise ValueError("subpixel_distilled_b8d3 requires training.distillation")
        teacher_variant = str(distillation.get("teacher_variant", ""))
        if not teacher_variant or teacher_variant == "subpixel_distilled_b8d3":
            raise ValueError("training.distillation.teacher_variant must identify a separate teacher")
        if teacher_variant not in set(registered_variants()):
            raise ValueError(
                f"Unknown training.distillation.teacher_variant: {teacher_variant}"
            )
        if not str(distillation.get("teacher_checkpoint", "")).strip():
            raise ValueError("training.distillation.teacher_checkpoint must not be empty")
        weight = float(distillation.get("weight", 0.0))
        if not 0.0 < weight < 1.0:
            raise ValueError("training.distillation.weight must be between zero and one")
    benchmark = config.get("benchmark", {"enabled": False})
    if bool(benchmark.get("enabled", False)):
        if int(benchmark.get("batch_size", 0)) < 1:
            raise ValueError("benchmark.batch_size must be >= 1")
        if int(benchmark.get("warmup", 0)) < 1 or int(benchmark.get("repeats", 0)) < 2:
            raise ValueError("benchmark requires warmup >= 1 and repeats >= 2")
        if str(benchmark.get("precision", "")) not in {"fp32", "fp16", "bf16"}:
            raise ValueError("benchmark.precision must be fp32, fp16 or bf16")
    if not str(config.get("output_root", "")).strip():
        raise ValueError("output_root must not be empty")


def _project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def _frame_info(dataset_root: Path) -> tuple[tuple[int, int], int]:
    import tifffile

    image_paths = sorted((dataset_root / "images").glob("*.tif*"))
    if not image_paths:
        image_paths = sorted(dataset_root.rglob("*.tif*"))
    if not image_paths:
        raise FileNotFoundError(f"No TIFF frames found under {dataset_root}")
    frame = tifffile.imread(image_paths[0])
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frames under {dataset_root}, received {frame.shape}")
    return (int(frame.shape[0]), int(frame.shape[1])), len(image_paths)


def _validate_split(path: Path, expected_window: str, available_frames: int) -> tuple[int, list[int]]:
    with path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    offsets = [int(value) for value in split["input_offsets"]]
    if 0 in offsets:
        raise ValueError(f"Centre frame is present in {path}")
    if len(offsets) != len(set(offsets)):
        raise ValueError(f"Duplicate temporal offsets in {path}")
    window_value = int(expected_window.removeprefix("w"))
    expected_offsets = list(range(-window_value, 0)) + list(range(1, window_value + 1))
    if offsets != expected_offsets:
        raise ValueError(f"{expected_window} requires offsets {expected_offsets}, found {offsets}")
    declared_frames = int(split.get("total_frames", available_frames))
    if declared_frames != available_frames:
        raise ValueError(f"Frame-count mismatch in {path}: split={declared_frames}, files={available_frames}")
    ranges: dict[str, tuple[int, int]] = {}
    for name in ("train", "val", "test"):
        start, end = (int(value) for value in split["purged_splits"][name])
        if start > end or start + min(offsets) < 0 or end + max(offsets) >= available_frames:
            raise ValueError(f"Invalid {name} range [{start}, {end}] in {path}")
        ranges[name] = (start, end)
        if "raw_splits" in split:
            raw_start, raw_end = (int(value) for value in split["raw_splits"][name])
            if not (raw_start <= start <= end <= raw_end):
                raise ValueError(f"Purged {name} range is outside its raw range in {path}")
    if not (ranges["train"][1] < ranges["val"][0] and ranges["val"][1] < ranges["test"][0]):
        raise ValueError(f"Purged splits overlap or are out of order in {path}")
    return len(offsets), offsets


def expand_tasks(config_path: str | Path, validate_paths: bool = True) -> list[ExperimentTask]:
    source = Path(config_path).resolve()
    config = load_stage_config(source)
    stage = str(config["stage"])
    data = config["data"]
    available = set(registered_variants())
    variants = [str(value) for value in config["variants"]]
    unknown = sorted(set(variants) - available)
    if unknown:
        raise ValueError(f"Unknown variants in {source}: {', '.join(unknown)}")

    overrides, override_source, override_sha256 = load_path_overrides()
    output_root = (
        resolve_override(override_source, str(overrides["output_root"]))
        if override_source is not None and "output_root" in overrides
        else _project_path(source, str(config["output_root"]))
    )
    project_root = REPOSITORY_ROOT
    try:
        output_root.relative_to(project_root / "artifacts")
    except ValueError as exc:
        raise ValueError(f"output_root must remain under {project_root / 'artifacts'}") from exc
    split_pattern = str(overrides.get("split_pattern", data["split_pattern"]))
    training_config = deepcopy(config["training"])
    if "distillation" in training_config:
        distillation = dict(training_config["distillation"])
        teacher_checkpoint = (
            resolve_override(override_source, str(overrides["teacher_checkpoint"]))
            if override_source is not None and "teacher_checkpoint" in overrides
            else _project_path(source, str(distillation["teacher_checkpoint"]))
        )
        if validate_paths and not teacher_checkpoint.is_file():
            raise FileNotFoundError(teacher_checkpoint)
        distillation["teacher_checkpoint"] = str(teacher_checkpoint)
        training_config["distillation"] = distillation
    tasks: list[ExperimentTask] = []
    frame_cache: dict[Path, tuple[tuple[int, int], int]] = {}
    for recording in config["recordings"]:
        recording_roots = overrides.get("recording_roots", {})
        dataset_root = (
            resolve_override(override_source, str(recording_roots[recording]))
            if override_source is not None and recording in recording_roots
            else _project_path(source, str(data["recording_roots"][recording]))
        )
        if validate_paths and not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)
        if dataset_root not in frame_cache:
            frame_cache[dataset_root] = _frame_info(dataset_root) if validate_paths else ((0, 0), 0)
        for window in config["windows"]:
            split_value = split_pattern.format(recording=recording, window=window)
            split_path = (
                resolve_override(override_source, split_value)
                if override_source is not None and "split_pattern" in overrides
                else _project_path(source, split_value)
            )
            if validate_paths and not split_path.is_file():
                raise FileNotFoundError(split_path)
            input_channels, _ = (
                _validate_split(split_path, window, frame_cache[dataset_root][1])
                if validate_paths
                else (2 * int(window[1:]), [])
            )
            for variant in variants:
                for seed in config["seeds"]:
                    task_training = deepcopy(training_config)
                    if variant != "subpixel_distilled_b8d3":
                        task_training.pop("distillation", None)
                    task_id = f"{stage}__{recording}__{window}__{variant}__seed{int(seed)}"
                    tasks.append(
                        ExperimentTask(
                            task_id=task_id,
                            stage=stage,
                            recording=str(recording),
                            window=str(window),
                            variant=variant,
                            seed=int(seed),
                            dataset_root=str(dataset_root),
                            split_manifest=str(split_path),
                            output_dir=str(output_root / recording / window / variant / f"seed{int(seed)}"),
                            input_channels=input_channels,
                            frame_shape=frame_cache[dataset_root][0],
                            training=task_training,
                            benchmark=dict(config.get("benchmark", {"enabled": False})),
                            source_config=str(source),
                            path_override=str(override_source) if override_source is not None else None,
                            path_override_sha256=override_sha256,
                        )
                    )
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Task IDs are not unique in {source}")
    return tasks


def write_plan(tasks: list[ExperimentTask], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "task_count": len(tasks),
        "tasks": [task.to_dict() for task in tasks],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def read_plan(path: str | Path) -> list[ExperimentTask]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [ExperimentTask(**task) for task in payload["tasks"]]
