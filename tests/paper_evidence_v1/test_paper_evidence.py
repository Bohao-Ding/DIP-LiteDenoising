from __future__ import annotations

from deepinterpolation_improve.experiments.paper_evidence_v1 import (
    build_paper_report,
    evaluate_paper_tasks,
    freeze_paper_evidence,
    load_verified_freeze,
    paper_tasks,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.paper_evidence import (
    _canonical_hash,
    _hierarchical_bootstrap,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.paper_latency import _condition_tasks


def test_paper_workflow_is_available_from_the_package_public_api() -> None:
    exported = (
        paper_tasks,
        freeze_paper_evidence,
        load_verified_freeze,
        evaluate_paper_tasks,
        build_paper_report,
    )
    assert all(callable(function) for function in exported)
    assert all(function.__module__.endswith(".paper_evidence") for function in exported)


def test_paper_task_contract_has_unique_conditions_and_locked_counts() -> None:
    tasks = paper_tasks("all")
    keys = {(task.recording, task.window, task.variant, task.seed) for task in tasks}
    assert len(tasks) == len(keys) == 252
    assert sum(task.stage == "paper_evidence_core" for task in tasks) == 225
    assert sum(task.stage == "paper_evidence_mechanism" for task in tasks) == 27
    assert all(task.training["evaluation_split"] == "val" for task in tasks)


def test_freeze_hash_is_order_stable_for_mapping_keys() -> None:
    assert _canonical_hash({"a": 1, "b": 2}) == _canonical_hash({"b": 2, "a": 1})


def test_hierarchical_bootstrap_preserves_a_constant_effect() -> None:
    rows = [
        {
            "recording": recording,
            "window": window,
            "seed": seed,
            "delta_psnr_db": 0.125,
        }
        for recording in ("01.00", "01.01", "03.00")
        for window in ("w1", "w2", "w4")
        for seed in range(5)
    ]
    low, high = _hierarchical_bootstrap(rows, 100, 17)
    assert low == high == 0.125


def test_paper_latency_uses_five_seed_zero_checkpoints_per_condition() -> None:
    grouped = _condition_tasks()
    assert len(grouped) == 9
    assert all(len(tasks) == 5 for tasks in grouped.values())
    assert all(task.seed == 0 for tasks in grouped.values() for task in tasks)
