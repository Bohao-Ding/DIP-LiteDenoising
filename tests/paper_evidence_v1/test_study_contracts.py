from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deepinterpolation_improve.experiments.paper_evidence_v1.study import (
    expand_tasks,
    load_stage_config,
    validate_stage_config,
)


def test_all_stage_matrices_have_explicit_seed_boundaries() -> None:
    root = Path(__file__).resolve().parents[2] / "configs"
    exploratory = root / "exploratory"
    formal = root / "paper_evidence_v1"
    quick = expand_tasks(exploratory / "quick_screen.json", validate_paths=False)
    targeted = expand_tasks(exploratory / "targeted_screen.json", validate_paths=False)
    innovate = expand_tasks(exploratory / "innovation_screen.json", validate_paths=False)
    confirm = expand_tasks(
        exploratory / "single_seed_confirmation.json", validate_paths=False
    )
    multi = expand_tasks(
        exploratory / "multi_seed_confirmation.template.json", validate_paths=False
    )
    next_screen = expand_tasks(
        exploratory / "next_generation_screen.json", validate_paths=False
    )
    next_confirm = expand_tasks(
        exploratory / "next_generation_confirmation.json", validate_paths=False
    )
    next_multi = expand_tasks(
        exploratory / "next_generation_multi_seed.template.json",
        validate_paths=False,
    )
    objective = expand_tasks(
        exploratory / "objective_controlled_structure_screen.json",
        validate_paths=False,
    )
    paper_core = expand_tasks(formal / "core.json", validate_paths=False)
    paper_mechanism = expand_tasks(formal / "mechanism.json", validate_paths=False)
    assert (len(quick), {task.seed for task in quick}) == (8, {0})
    assert (len(targeted), {task.seed for task in targeted}) == (5, {0})
    assert (len(innovate), {task.seed for task in innovate}) == (9, {0})
    assert (len(confirm), {task.seed for task in confirm}) == (144, {0})
    assert (len(multi), {task.seed for task in multi}) == (720, {0, 1, 2, 3, 4})
    assert (len(next_screen), {task.seed for task in next_screen}) == (21, {0})
    assert (len(next_confirm), {task.seed for task in next_confirm}) == (63, {0})
    assert (len(next_multi), {task.seed for task in next_multi}) == (
        315,
        {0, 1, 2, 3, 4},
    )
    assert (len(objective), {task.seed for task in objective}) == (15, {0})
    assert (len(paper_core), {task.seed for task in paper_core}) == (225, {0, 1, 2, 3, 4})
    assert (len(paper_mechanism), {task.seed for task in paper_mechanism}) == (27, {0})
    tasks = (
        quick
        + targeted
        + innovate
        + confirm
        + multi
        + next_screen
        + next_confirm
        + next_multi
        + objective
        + paper_core
        + paper_mechanism
    )
    assert all(task.training["precision"] == "amp_fp16" for task in tasks)
    assert all(task.training["metric_precision"] == "fp32" for task in tasks)
    distilled = next(task for task in innovate if task.variant == "subpixel_distilled_b8d3")
    assert Path(distilled.training["distillation"]["teacher_checkpoint"]).name == "best.pt"
    assert all("distillation" not in task.training for task in innovate if task is not distilled)
    psnr_aligned = next(task for task in next_screen if task.variant == "subpixel_psnr_aligned_b8d3")
    assert psnr_aligned.training["precision"] == "amp_fp16"
    assert {task.training["evaluation_split"] for task in next_screen} == {"val"}
    assert {task.training["evaluation_split"] for task in next_confirm + next_multi} == {"test"}
    assert {task.training["evaluation_split"] for task in objective} == {"val"}
    assert {task.training["evaluation_split"] for task in paper_core + paper_mechanism} == {"val"}


def test_stage_reference_must_be_present_in_the_variant_matrix() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "exploratory"
        / "objective_controlled_structure_screen.json"
    )
    config = load_stage_config(source)
    invalid = deepcopy(config)
    invalid["variants"].remove(invalid["reference_variant"])
    with pytest.raises(ValueError, match="reference_variant"):
        validate_stage_config(invalid)


def test_single_seed_stage_rejects_multiple_seeds() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "exploratory"
        / "quick_screen.json"
    )
    config = load_stage_config(source)
    invalid = deepcopy(config)
    invalid["seeds"] = [0, 1]
    with pytest.raises(ValueError, match="restricted to seed 0"):
        validate_stage_config(invalid)


def test_stage_rejects_ambiguous_or_wrong_precision_contracts() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "exploratory"
        / "quick_screen.json"
    )
    config = load_stage_config(source)

    obsolete = deepcopy(config)
    obsolete["training"]["mixed_precision"] = True
    with pytest.raises(ValueError, match="mixed_precision is obsolete"):
        validate_stage_config(obsolete)

    wrong_metrics = deepcopy(config)
    wrong_metrics["training"]["metric_precision"] = "fp16"
    with pytest.raises(ValueError, match="metric_precision must be fp32"):
        validate_stage_config(wrong_metrics)

    wrong_split = deepcopy(config)
    wrong_split["training"]["evaluation_split"] = "train"
    with pytest.raises(ValueError, match="evaluation_split must be val or test"):
        validate_stage_config(wrong_split)


def test_output_root_cannot_escape_workspace_artifacts() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "exploratory"
        / "quick_screen.json"
    )
    config = load_stage_config(source)
    invalid = deepcopy(config)
    invalid["output_root"] = "../outside"
    with TemporaryDirectory(prefix="dip_config_test_") as temporary:
        config_dir = Path(temporary) / "configs"
        config_dir.mkdir()
        path = config_dir / "invalid.json"
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="output_root must remain"):
            expand_tasks(path, validate_paths=False)
