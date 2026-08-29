from __future__ import annotations

import pytest

from deepinterpolation_improve.experiments.paper_evidence_v1.latency_audit import aggregate_latency_records, balanced_orders


def test_balanced_orders_rotate_every_model_through_every_position() -> None:
    labels = ["a", "b", "c"]
    orders = balanced_orders(labels, 3)
    assert len(orders) == 3
    for position in range(3):
        assert {order[position] for order in orders} == set(labels)


def test_balanced_orders_reject_an_incomplete_rotation_cycle() -> None:
    with pytest.raises(ValueError, match="multiple of the number of models"):
        balanced_orders(["a", "b", "c"], 2)


def test_latency_aggregation_uses_round_medians_and_baseline_ratio() -> None:
    records = [
        {"variant": "baseline", "latency_ms_p50": 2.0, "latency_ms_p95": 2.4},
        {"variant": "candidate", "latency_ms_p50": 1.0, "latency_ms_p95": 1.2},
        {"variant": "baseline", "latency_ms_p50": 2.2, "latency_ms_p95": 2.6},
        {"variant": "candidate", "latency_ms_p50": 1.1, "latency_ms_p95": 1.3},
    ]
    aggregates = aggregate_latency_records(records, "baseline")
    candidate = next(row for row in aggregates if row["variant"] == "candidate")
    assert candidate["median_of_round_p50_ms"] == pytest.approx(1.05)
    assert candidate["p50_ratio_to_baseline"] == pytest.approx(0.5)
