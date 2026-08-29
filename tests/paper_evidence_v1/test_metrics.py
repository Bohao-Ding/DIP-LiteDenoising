from __future__ import annotations

import torch

from deepinterpolation_improve.experiments.paper_evidence_v1.metrics import MetricAccumulator, frame_metrics


def test_metrics_are_per_frame_and_exact_prediction_is_finite() -> None:
    target = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    prediction = target.clone()
    metrics = frame_metrics(prediction, target)
    assert metrics["psnr_db"].shape == (2,)
    assert torch.isfinite(metrics["psnr_db"]).all()
    assert torch.equal(metrics["normalised_mae"], torch.zeros(2))

    accumulator = MetricAccumulator()
    accumulator.update(prediction, target, torch.tensor([10, 11]))
    summary = accumulator.summary()
    assert summary["frame_count"] == 2
    assert summary["mean_normalised_mae"] == 0.0


def test_metric_arithmetic_is_promoted_to_fp32() -> None:
    target = torch.arange(16, dtype=torch.float16).reshape(1, 1, 4, 4)
    metrics = frame_metrics(target.clone(), target)
    assert all(value.dtype == torch.float32 for value in metrics.values())
