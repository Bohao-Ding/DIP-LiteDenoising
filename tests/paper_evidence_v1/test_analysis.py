from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from deepinterpolation_improve.experiments.paper_evidence_v1.analysis import (
    aggregate_comparisons,
    aggregate_frame_comparisons,
    audit_result_coverage,
    compare_with_baseline,
    compare_with_reference,
    paired_frame_comparisons,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.study import ExperimentTask


def _summary(variant: str, seed: int, psnr: float, task_id: str | None = None) -> dict:
    return {
        "task_id": task_id or f"task-{variant}-{seed}",
        "recording": "01.00",
        "window": "w2",
        "seed": seed,
        "variant": variant,
        "metrics": {"mean_psnr_db": psnr},
        "deployed_profile": {"conv_flops": 100.0 if variant == "baseline_b8d3" else 70.0, "parameters": 100},
        "latency": {"latency_ms_p50": 1.0 if variant == "baseline_b8d3" else 0.8},
    }


def _task(task_id: str) -> ExperimentTask:
    return ExperimentTask(
        task_id=task_id,
        stage="quick_screen",
        recording="01.00",
        window="w2",
        variant="baseline_b8d3",
        seed=0,
        dataset_root="data",
        split_manifest="split.json",
        output_dir="output",
        input_channels=4,
        frame_shape=(8, 8),
        training={},
        benchmark={},
        source_config="config.json",
    )


def test_paired_and_aggregate_results_preserve_seed_conditions() -> None:
    summaries = []
    for seed, delta in ((0, 0.02), (1, -0.01)):
        summaries.extend(
            [
                _summary("baseline_b8d3", seed, 18.0),
                _summary("candidate", seed, 18.0 + delta),
            ]
        )
    comparisons = compare_with_baseline(summaries)
    aggregate = aggregate_comparisons(comparisons)[0]
    assert len(comparisons) == 2
    assert aggregate["paired_conditions"] == 2
    assert aggregate["positive_delta_count"] == 1
    assert aggregate["mean_delta_psnr_db"] == pytest.approx(0.005)
    assert aggregate["mean_latency_p50_ratio"] == pytest.approx(0.8)


def test_missing_baseline_and_duplicate_condition_are_not_silent() -> None:
    candidate = _summary("candidate", 0, 18.1)
    with pytest.raises(ValueError, match="lack paired baselines"):
        compare_with_baseline([candidate])
    assert compare_with_baseline([candidate], strict=False) == []
    baseline = _summary("baseline_b8d3", 0, 18.0)
    with pytest.raises(ValueError, match="Duplicate result conditions"):
        compare_with_baseline([baseline, dict(baseline)])


def test_arbitrary_reference_and_paired_frame_directions() -> None:
    with TemporaryDirectory(prefix="dip_frame_analysis_") as temporary:
        root = Path(temporary)
        summaries = []
        for variant, psnr, rows in (
            (
                "reference",
                18.0,
                [
                    {"frame_index": 1, "normalised_mse": 2.0, "normalised_mae": 1.0, "psnr_db": 18.0, "gssim": 0.2},
                    {"frame_index": 2, "normalised_mse": 2.0, "normalised_mae": 1.0, "psnr_db": 18.0, "gssim": 0.2},
                ],
            ),
            (
                "candidate",
                18.1,
                [
                    {"frame_index": 1, "normalised_mse": 1.8, "normalised_mae": 0.9, "psnr_db": 18.2, "gssim": 0.3},
                    {"frame_index": 2, "normalised_mse": 2.2, "normalised_mae": 1.1, "psnr_db": 18.0, "gssim": 0.1},
                ],
            ),
        ):
            run = root / variant
            run.mkdir()
            summary_path = run / "summary.json"
            summary_path.write_text(json.dumps({}), encoding="utf-8")
            with (run / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = _summary(variant, 0, psnr)
            summary["summary_path"] = str(summary_path)
            summaries.append(summary)

        assert compare_with_reference(summaries, "reference")[0]["delta_psnr_db"] == pytest.approx(0.1)
        paired = paired_frame_comparisons(summaries, "reference")[0]
        assert paired["paired_frames"] == 2
        assert paired["metrics"]["psnr_db"]["improved_frames"] == 1
        assert paired["metrics"]["normalised_mae"]["improved_frames"] == 1


def test_result_coverage_reports_missing_unexpected_and_duplicate_tasks() -> None:
    tasks = [_task("a"), _task("b")]
    summaries = [
        _summary("baseline_b8d3", 0, 18.0, "a"),
        _summary("baseline_b8d3", 1, 18.0, "a"),
        _summary("candidate", 2, 18.0, "extra"),
    ]
    coverage = audit_result_coverage(tasks, summaries)
    assert coverage["completed_tasks"] == 1
    assert coverage["missing_task_ids"] == ["b"]
    assert coverage["unexpected_task_ids"] == ["extra"]
    assert coverage["duplicate_task_ids"] == ["a"]
    assert coverage["complete"] is False


def test_paired_frame_analysis_reads_the_declared_validation_split() -> None:
    with TemporaryDirectory(prefix="dip_val_analysis_") as temporary:
        root = Path(temporary)
        summaries = []
        for variant, psnr in (("reference", 18.0), ("candidate", 18.1)):
            run = root / variant
            run.mkdir()
            summary_path = run / "summary.json"
            summary_path.write_text(json.dumps({}), encoding="utf-8")
            rows = [
                {
                    "frame_index": 1,
                    "normalised_mse": 1.0,
                    "normalised_mae": 1.0,
                    "psnr_db": psnr,
                    "gssim": 0.2,
                }
            ]
            with (run / "val_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = _summary(variant, 0, psnr)
            summary["summary_path"] = str(summary_path)
            summary["evaluation_split"] = "val"
            summaries.append(summary)

        paired = paired_frame_comparisons(summaries, "reference")[0]
        assert paired["metrics"]["psnr_db"]["mean_delta"] == pytest.approx(0.1)


def test_frame_aggregation_weights_conditions_by_frame_count() -> None:
    def row(frames: int, mean: float, sd: float, improved: int) -> dict:
        metric = {
            "mean_delta": mean,
            "sd_delta": sd,
            "improved_frames": improved,
            "degraded_frames": frames - improved,
            "unchanged_frames": 0,
        }
        return {
            "variant": "candidate",
            "paired_frames": frames,
            "metrics": {name: dict(metric) for name in (
                "psnr_db", "gssim", "normalised_mae", "normalised_mse"
            )},
        }

    aggregate = aggregate_frame_comparisons([row(2, 1.0, 0.0, 2), row(1, -1.0, 0.0, 0)])[0]
    assert aggregate["paired_frames"] == 3
    assert aggregate["metrics"]["psnr_db"]["mean_delta"] == pytest.approx(1.0 / 3.0)
    assert aggregate["metrics"]["psnr_db"]["sd_delta"] == pytest.approx(2.0 / 3.0**0.5)
    assert aggregate["metrics"]["psnr_db"]["improved_frames"] == 2
