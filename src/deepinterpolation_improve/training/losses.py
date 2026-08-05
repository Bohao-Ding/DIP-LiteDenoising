from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_squareroot_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sqrt(torch.abs(y_pred - y_true) + 1e-8))


def annealed_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.pow(torch.abs(y_pred - y_true) + 1e-8, 4))


def get_loss(name: str):
    if name in {"mean_absolute_error", "mae", "l1"}:
        return F.l1_loss
    if name in {"mean_squared_error", "mse"}:
        return F.mse_loss
    if name == "mean_squareroot_error":
        return mean_squareroot_error
    if name == "annealed_loss":
        return annealed_loss
    raise ValueError(f"Unknown loss: {name}")
