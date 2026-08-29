from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def frame_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return FP32 per-frame reconstruction metrics on normalised arrays."""

    prediction = prediction.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    dimensions = tuple(range(1, target.ndim))
    error = prediction - target
    mse = error.square().mean(dim=dimensions)
    mae = error.abs().mean(dim=dimensions)
    data_range = target.amax(dim=dimensions) - target.amin(dim=dimensions)
    safe_range = data_range.clamp_min(torch.finfo(target.dtype).eps)
    psnr = 10.0 * torch.log10(safe_range.square() / mse.clamp_min(torch.finfo(target.dtype).eps))

    pred_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    pred_mean = pred_flat.mean(dim=1)
    target_mean = target_flat.mean(dim=1)
    pred_var = pred_flat.var(dim=1, unbiased=False)
    target_var = target_flat.var(dim=1, unbiased=False)
    covariance = ((pred_flat - pred_mean[:, None]) * (target_flat - target_mean[:, None])).mean(dim=1)
    c1 = (0.01 * safe_range).square()
    c2 = (0.03 * safe_range).square()
    gssim = ((2 * pred_mean * target_mean + c1) * (2 * covariance + c2)) / (
        (pred_mean.square() + target_mean.square() + c1) * (pred_var + target_var + c2)
    ).clamp_min(torch.finfo(target.dtype).eps)
    return {"normalised_mse": mse, "normalised_mae": mae, "psnr_db": psnr, "gssim": gssim}


@dataclass
class MetricAccumulator:
    rows: list[dict[str, float | int]] = field(default_factory=list)

    def update(self, prediction: torch.Tensor, target: torch.Tensor, frame_indices: torch.Tensor) -> None:
        values = frame_metrics(prediction.detach(), target.detach())
        for position, frame_index in enumerate(frame_indices.detach().cpu().tolist()):
            row: dict[str, float | int] = {"frame_index": int(frame_index)}
            for name, tensor in values.items():
                row[name] = float(tensor[position].detach().cpu())
            self.rows.append(row)

    def summary(self) -> dict[str, float | int]:
        if not self.rows:
            raise ValueError("No metric rows were accumulated")
        output: dict[str, float | int] = {"frame_count": len(self.rows)}
        for name in ("normalised_mse", "normalised_mae", "psnr_db", "gssim"):
            values = [float(row[name]) for row in self.rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
            output[f"mean_{name}"] = mean
            output[f"sd_{name}"] = math.sqrt(variance)
        return output
