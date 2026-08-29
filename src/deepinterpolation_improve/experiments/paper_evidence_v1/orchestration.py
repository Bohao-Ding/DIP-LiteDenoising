from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .engine import run_task
from .study import ExperimentTask, expand_tasks, load_stage_config


PROJECT_ROOT = Path(__file__).resolve().parents[4]
STAGE_CONFIGS = {
    "quick": PROJECT_ROOT / "configs" / "exploratory" / "quick_screen.json",
    "targeted": PROJECT_ROOT / "configs" / "exploratory" / "targeted_screen.json",
    "innovate": PROJECT_ROOT / "configs" / "exploratory" / "innovation_screen.json",
    "confirm": PROJECT_ROOT / "configs" / "exploratory" / "single_seed_confirmation.json",
    "multi": PROJECT_ROOT / "configs" / "exploratory" / "multi_seed_confirmation.template.json",
    "next": PROJECT_ROOT / "configs" / "exploratory" / "next_generation_screen.json",
    "next-confirm": PROJECT_ROOT / "configs" / "exploratory" / "next_generation_confirmation.json",
    "next-multi": PROJECT_ROOT / "configs" / "exploratory" / "next_generation_multi_seed.template.json",
    "objective": PROJECT_ROOT / "configs" / "exploratory" / "objective_controlled_structure_screen.json",
    "paper-core": PROJECT_ROOT / "configs" / "paper_evidence_v1" / "core.json",
    "paper-mechanism": PROJECT_ROOT / "configs" / "paper_evidence_v1" / "mechanism.json",
}

EXPLICIT_MODEL_STAGES = {"confirm", "multi", "next-confirm", "next-multi"}


def parse_task_indices(value: str, task_count: int) -> list[int]:
    if task_count < 1:
        raise ValueError("No tasks are available")
    if value == "all":
        return list(range(task_count))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Empty task-index element")
        if "-" in part:
            start, end = (int(item) for item in part.split("-", maxsplit=1))
            if start > end:
                raise ValueError(f"Descending task range is not allowed: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    result = sorted(selected)
    if not result or result[0] < 0 or result[-1] >= task_count:
        raise ValueError(f"Task indices must be within [0, {task_count - 1}]")
    return result


def select_stage_tasks(
    stage: str,
    models: list[str] | None = None,
    task_indices: str = "all",
    *,
    validate_paths: bool = True,
) -> list[ExperimentTask]:
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage: {stage}")
    tasks = expand_tasks(STAGE_CONFIGS[stage], validate_paths=validate_paths)
    available_models = {task.variant for task in tasks}
    if stage in EXPLICIT_MODEL_STAGES and not models:
        raise ValueError(f"{stage} requires an explicit --models list")
    if models:
        requested = set(models)
        unknown = sorted(requested - available_models)
        if unknown:
            raise ValueError(f"Unknown models for {stage}: {', '.join(unknown)}")
        config = load_stage_config(STAGE_CONFIGS[stage])
        reference_variant = str(config.get("reference_variant", "baseline_b8d3"))
        if reference_variant not in requested:
            raise ValueError(f"Selected models must include {reference_variant}")
        tasks = [task for task in tasks if task.variant in requested]
    selected = parse_task_indices(task_indices, len(tasks))
    return [tasks[index] for index in selected]


def task_state(task: ExperimentTask) -> str:
    output = Path(task.output_dir)
    if (output / "summary.json").is_file():
        return "completed"
    if (output / "checkpoint.pt").is_file():
        return "resumable"
    if output.exists() and any(output.iterdir()):
        return "partial"
    return "pending"


def stage_snapshot(tasks: list[ExperimentTask]) -> dict[str, Any]:
    states = Counter(task_state(task) for task in tasks)
    return {
        "task_count": len(tasks),
        "models": sorted({task.variant for task in tasks}),
        "seeds": sorted({task.seed for task in tasks}),
        "recordings": sorted({task.recording for task in tasks}),
        "windows": sorted({task.window for task in tasks}),
        "states": dict(sorted(states.items())),
        "first_task": tasks[0].task_id if tasks else None,
        "last_task": tasks[-1].task_id if tasks else None,
    }


def execute_stage(
    tasks: list[ExperimentTask],
    *,
    device: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    completed = 0
    skipped = 0
    for position, task in enumerate(tasks, start=1):
        state = task_state(task)
        if state == "completed":
            print(f"[{position}/{len(tasks)}] skip completed: {task.task_id}", flush=True)
            skipped += 1
            continue
        if state == "resumable" and not resume:
            raise RuntimeError(f"Task is resumable; add --resume: {task.task_id}")
        if state == "partial":
            raise RuntimeError(f"Task has partial output without a resumable checkpoint: {task.task_id}")
        print(f"[{position}/{len(tasks)}] run: {task.task_id}", flush=True)
        run_task(task, device_override=device, resume=resume and state == "resumable")
        completed += 1
    return {"selected": len(tasks), "completed_now": completed, "skipped_completed": skipped}
