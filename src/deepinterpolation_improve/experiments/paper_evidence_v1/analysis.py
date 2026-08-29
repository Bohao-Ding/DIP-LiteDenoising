from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_summaries(root: str | Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(Path(root).rglob("summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["summary_path"] = str(path.resolve())
        summaries.append(summary)
    return summaries


def compare_with_reference(
    summaries: list[dict[str, Any]],
    reference: str,
    strict: bool = True,
) -> list[dict[str, Any]]:
    by_condition: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    duplicates = []
    for row in summaries:
        key = (row["recording"], row["window"], int(row["seed"]), row["variant"])
        if key in by_condition:
            duplicates.append(key)
        by_condition[key] = row
    if duplicates:
        raise ValueError(f"Duplicate result conditions: {duplicates}")
    comparisons: list[dict[str, Any]] = []
    missing_references = []
    for row in summaries:
        if row["variant"] == reference:
            continue
        key = (row["recording"], row["window"], int(row["seed"]), reference)
        control = by_condition.get(key)
        if control is None:
            missing_references.append(key)
            continue
        control_psnr = float(control["metrics"]["mean_psnr_db"])
        candidate_psnr = float(row["metrics"]["mean_psnr_db"])
        control_flops = float(control["deployed_profile"]["conv_flops"])
        candidate_flops = float(row["deployed_profile"]["conv_flops"])
        numeric_values = (control_psnr, candidate_psnr, control_flops, candidate_flops)
        if not all(math.isfinite(value) for value in numeric_values) or control_flops <= 0:
            raise ValueError(f"Non-finite or invalid result values for {key}")
        control_latency = control.get("latency")
        candidate_latency = row.get("latency")
        comparison = {
            "recording": row["recording"],
            "window": row["window"],
            "seed": int(row["seed"]),
            "variant": row["variant"],
            "delta_psnr_db": candidate_psnr - control_psnr,
            "conv_flops_ratio": candidate_flops / control_flops,
            "parameter_ratio": float(row["deployed_profile"]["parameters"])
            / float(control["deployed_profile"]["parameters"]),
            "latency_p50_ratio": None,
        }
        if control_latency and candidate_latency:
            comparison["latency_p50_ratio"] = float(candidate_latency["latency_ms_p50"]) / float(
                control_latency["latency_ms_p50"]
            )
        comparisons.append(comparison)
    if strict and missing_references:
        noun = "baselines" if reference == "baseline_b8d3" else "references"
        raise ValueError(f"Candidate results lack paired {noun}: {missing_references}")
    return comparisons


def compare_with_baseline(
    summaries: list[dict[str, Any]],
    baseline: str = "baseline_b8d3",
    strict: bool = True,
) -> list[dict[str, Any]]:
    return compare_with_reference(summaries, baseline, strict)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _indexed_metric_rows(summary: dict[str, Any]) -> dict[int, dict[str, float]]:
    summary_path = summary.get("summary_path")
    if not summary_path:
        raise ValueError(f"Summary path is missing for {summary.get('task_id', '<unknown>')}")
    evaluation_split = str(summary.get("evaluation_split", "test"))
    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split in {summary_path}: {evaluation_split}")
    metrics_path = Path(summary_path).with_name(f"{evaluation_split}_metrics.csv")
    rows: dict[int, dict[str, float]] = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            frame_index = int(raw["frame_index"])
            if frame_index in rows:
                raise ValueError(f"Duplicate frame index {frame_index} in {metrics_path}")
            rows[frame_index] = {
                key: float(raw[key])
                for key in ("psnr_db", "normalised_mae", "normalised_mse", "gssim")
            }
    if not rows:
        raise ValueError(f"No {evaluation_split} metrics in {metrics_path}")
    return rows


def paired_frame_comparisons(
    summaries: list[dict[str, Any]],
    reference: str,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Describe paired per-frame metric changes relative to one model."""

    by_condition = {}
    duplicates = []
    for row in summaries:
        key = (row["recording"], row["window"], int(row["seed"]), row["variant"])
        if key in by_condition:
            duplicates.append(key)
        by_condition[key] = row
    if duplicates:
        raise ValueError(f"Duplicate result conditions: {duplicates}")
    missing_references = []
    output = []
    directions = {"psnr_db": 1, "gssim": 1, "normalised_mae": -1, "normalised_mse": -1}
    for candidate in summaries:
        if candidate["variant"] == reference:
            continue
        key = (candidate["recording"], candidate["window"], int(candidate["seed"]), reference)
        control = by_condition.get(key)
        if control is None:
            missing_references.append(key)
            continue
        control_rows = _indexed_metric_rows(control)
        candidate_rows = _indexed_metric_rows(candidate)
        if set(control_rows) != set(candidate_rows):
            raise ValueError(f"Frame-index mismatch for {candidate['variant']} relative to {reference}")
        metric_results = {}
        for metric, direction in directions.items():
            deltas = [candidate_rows[index][metric] - control_rows[index][metric] for index in control_rows]
            if not all(math.isfinite(value) for value in deltas):
                raise ValueError(f"Non-finite paired {metric} delta for {candidate['variant']}")
            signed = [direction * value for value in deltas]
            metric_results[metric] = {
                "mean_delta": statistics.fmean(deltas),
                "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "median_delta": statistics.median(deltas),
                "p05_delta": _percentile(deltas, 0.05),
                "p95_delta": _percentile(deltas, 0.95),
                "improved_frames": sum(value > 0 for value in signed),
                "degraded_frames": sum(value < 0 for value in signed),
                "unchanged_frames": sum(value == 0 for value in signed),
            }
        output.append(
            {
                "recording": candidate["recording"],
                "window": candidate["window"],
                "seed": int(candidate["seed"]),
                "reference_variant": reference,
                "variant": candidate["variant"],
                "paired_frames": len(control_rows),
                "metrics": metric_results,
            }
        )
    if strict and missing_references:
        raise ValueError(f"Candidate results lack paired references: {missing_references}")
    return output


def aggregate_comparisons(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate paired effects across recording, window and seed conditions."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[str(row["variant"])].append(row)
    output = []
    for variant, rows in sorted(grouped.items()):
        deltas = [float(row["delta_psnr_db"]) for row in rows]
        flops = [float(row["conv_flops_ratio"]) for row in rows]
        parameters = [float(row["parameter_ratio"]) for row in rows]
        latencies = [float(row["latency_p50_ratio"]) for row in rows if row.get("latency_p50_ratio") is not None]
        output.append(
            {
                "variant": variant,
                "paired_conditions": len(rows),
                "mean_delta_psnr_db": statistics.fmean(deltas),
                "sd_delta_psnr_db": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "min_delta_psnr_db": min(deltas),
                "max_delta_psnr_db": max(deltas),
                "positive_delta_count": sum(value > 0 for value in deltas),
                "mean_conv_flops_ratio": statistics.fmean(flops),
                "mean_parameter_ratio": statistics.fmean(parameters),
                "latency_pair_count": len(latencies),
                "mean_latency_p50_ratio": statistics.fmean(latencies) if latencies else None,
            }
        )
    return output


def aggregate_frame_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool paired-frame summaries without treating recordings as equal-sized."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    directions = {"psnr_db": 1, "gssim": 1, "normalised_mae": -1, "normalised_mse": -1}
    output = []
    for variant, variant_rows in sorted(grouped.items()):
        total_frames = sum(int(row["paired_frames"]) for row in variant_rows)
        metrics = {}
        for metric, direction in directions.items():
            weighted_sum = sum(
                int(row["paired_frames"]) * float(row["metrics"][metric]["mean_delta"])
                for row in variant_rows
            )
            pooled_mean = weighted_sum / total_frames
            sum_squares = sum(
                (int(row["paired_frames"]) - 1)
                * float(row["metrics"][metric]["sd_delta"]) ** 2
                + int(row["paired_frames"])
                * float(row["metrics"][metric]["mean_delta"]) ** 2
                for row in variant_rows
            )
            pooled_variance = (
                max(0.0, (sum_squares - total_frames * pooled_mean**2) / (total_frames - 1))
                if total_frames > 1
                else 0.0
            )
            metrics[metric] = {
                "mean_delta": pooled_mean,
                "sd_delta": math.sqrt(pooled_variance),
                "improved_frames": sum(
                    int(row["metrics"][metric]["improved_frames"]) for row in variant_rows
                ),
                "degraded_frames": sum(
                    int(row["metrics"][metric]["degraded_frames"]) for row in variant_rows
                ),
                "unchanged_frames": sum(
                    int(row["metrics"][metric]["unchanged_frames"]) for row in variant_rows
                ),
                "improved_condition_count": sum(
                    direction * float(row["metrics"][metric]["mean_delta"]) > 0
                    for row in variant_rows
                ),
            }
        output.append(
            {
                "variant": variant,
                "paired_conditions": len(variant_rows),
                "paired_frames": total_frames,
                "metrics": metrics,
            }
        )
    return output


def audit_result_coverage(tasks, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [task.task_id for task in tasks]
    observed = [str(row.get("task_id", "")) for row in summaries]
    counts = Counter(observed)
    duplicate_task_ids = sorted(task_id for task_id, count in counts.items() if task_id and count > 1)
    expected_set = set(expected)
    observed_set = {task_id for task_id in observed if task_id}
    return {
        "expected_tasks": len(expected),
        "completed_tasks": len(observed_set & expected_set),
        "missing_task_ids": sorted(expected_set - observed_set),
        "unexpected_task_ids": sorted(observed_set - expected_set),
        "duplicate_task_ids": duplicate_task_ids,
        "complete": observed_set == expected_set and not duplicate_task_ids,
    }


def promotion_assessment(
    comparisons: list[dict[str, Any]],
    quality_delta_db: float = 0.01,
    speed_delta_floor_db: float = -0.02,
    speed_flops_ratio: float = 0.75,
    speed_latency_ratio: float = 0.90,
) -> list[dict[str, Any]]:
    output = []
    for row in comparisons:
        quality_route = float(row["delta_psnr_db"]) >= float(quality_delta_db)
        theoretical_speed_route = (
            float(row["delta_psnr_db"]) >= float(speed_delta_floor_db)
            and float(row["conv_flops_ratio"]) <= float(speed_flops_ratio)
        )
        latency_ratio = row.get("latency_p50_ratio")
        measured_speed_route = (
            float(row["delta_psnr_db"]) >= float(speed_delta_floor_db)
            and latency_ratio is not None
            and float(latency_ratio) <= float(speed_latency_ratio)
        )
        assessed = dict(row)
        assessed.update(
            {
                "quality_route": quality_route,
                "theoretical_speed_route": theoretical_speed_route,
                "measured_speed_route": measured_speed_route,
                "promote_to_full_single_seed": quality_route or measured_speed_route,
                "note": (
                    "FLOPs-only speed support requires a standardised latency result."
                    if theoretical_speed_route and not measured_speed_route
                    else ""
                ),
            }
        )
        output.append(assessed)
    return output


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    if not rows:
        raise ValueError("No rows to write")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output
