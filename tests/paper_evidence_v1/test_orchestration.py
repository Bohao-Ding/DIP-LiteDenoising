from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deepinterpolation_improve.experiments.paper_evidence_v1 import engine
from deepinterpolation_improve.experiments.paper_evidence_v1.orchestration import (
    parse_task_indices,
    select_stage_tasks,
    stage_snapshot,
    task_state,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.study import ExperimentTask


def _task(output_dir: str) -> ExperimentTask:
    return ExperimentTask(
        task_id="test-task",
        stage="quick_screen",
        recording="01.00",
        window="w2",
        variant="baseline_b8d3",
        seed=0,
        dataset_root="data",
        split_manifest="split.json",
        output_dir=output_dir,
        input_channels=4,
        frame_shape=(8, 8),
        training={},
        benchmark={},
        source_config="config.json",
    )


def test_stage_selection_keeps_single_seed_separate_from_multi_seed() -> None:
    quick = select_stage_tasks("quick", validate_paths=False)
    assert len(quick) == 8
    assert {task.seed for task in quick} == {0}
    targeted = select_stage_tasks("targeted", validate_paths=False)
    assert len(targeted) == 5
    assert {task.seed for task in targeted} == {0}
    innovate = select_stage_tasks("innovate", validate_paths=False)
    assert len(innovate) == 9
    assert {task.seed for task in innovate} == {0}
    objective = select_stage_tasks("objective", validate_paths=False)
    assert len(objective) == 15
    assert {task.seed for task in objective} == {0}
    assert {task.training["evaluation_split"] for task in objective} == {"val"}
    paper_core = select_stage_tasks("paper-core", validate_paths=False)
    paper_mechanism = select_stage_tasks("paper-mechanism", validate_paths=False)
    assert len(paper_core) == 225
    assert len(paper_mechanism) == 27
    assert {task.seed for task in paper_core} == {0, 1, 2, 3, 4}
    assert {task.seed for task in paper_mechanism} == {0}
    with pytest.raises(ValueError, match="explicit --models"):
        select_stage_tasks("confirm", validate_paths=False)
    multi = select_stage_tasks(
        "multi",
        ["baseline_b8d3", "consensus_attention_b8d3"],
        validate_paths=False,
    )
    assert len(multi) == 90
    assert {task.seed for task in multi} == {0, 1, 2, 3, 4}
    assert stage_snapshot(multi)["task_count"] == 90


def test_custom_reference_is_required_when_filtering_objective_models() -> None:
    with pytest.raises(ValueError, match="subpixel_psnr_aligned_b8d3"):
        select_stage_tasks(
            "objective",
            ["baseline_psnr_aligned_b8d3"],
            validate_paths=False,
        )


def test_task_index_parser_rejects_ambiguous_or_invalid_ranges() -> None:
    assert parse_task_indices("0,2-4", 6) == [0, 2, 3, 4]
    with pytest.raises(ValueError, match="Descending"):
        parse_task_indices("4-2", 6)
    with pytest.raises(ValueError, match="within"):
        parse_task_indices("6", 6)


def test_task_state_distinguishes_pending_resumable_partial_and_completed() -> None:
    with TemporaryDirectory(prefix="dip_orchestration_test_") as temporary:
        root = Path(temporary)
        task = _task(str(root / "run"))
        assert task_state(task) == "pending"
        Path(task.output_dir).mkdir()
        (Path(task.output_dir) / "status.json").write_text("{}", encoding="utf-8")
        assert task_state(task) == "partial"
        (Path(task.output_dir) / "checkpoint.pt").write_bytes(b"checkpoint")
        assert task_state(task) == "resumable"
        (Path(task.output_dir) / "summary.json").write_text("{}", encoding="utf-8")
        assert task_state(task) == "completed"


def test_run_task_records_completed_and_failed_status_without_training(monkeypatch) -> None:
    with TemporaryDirectory(prefix="dip_status_test_") as temporary:
        completed_task = _task(str(Path(temporary) / "completed"))

        def fake_complete(task, **_kwargs):
            summary = {"task_id": task.task_id}
            Path(task.output_dir, "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return summary

        monkeypatch.setattr(engine, "_run_task_impl", fake_complete)
        engine.run_task(completed_task)
        status = json.loads(Path(completed_task.output_dir, "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "completed"

        failed_task = _task(str(Path(temporary) / "failed"))

        def fake_failure(_task, **_kwargs):
            raise RuntimeError("expected failure")

        monkeypatch.setattr(engine, "_run_task_impl", fake_failure)
        with pytest.raises(RuntimeError, match="expected failure"):
            engine.run_task(failed_task)
        status = json.loads(Path(failed_task.output_dir, "status.json").read_text(encoding="utf-8"))
        assert status["state"] == "failed"
        assert status["error_type"] == "RuntimeError"
