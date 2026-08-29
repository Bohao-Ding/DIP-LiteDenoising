from __future__ import annotations

import pytest
import torch

from deepinterpolation_improve.experiments.paper_evidence_v1.engine import _objective_terms, _selection_value


def test_teacher_objective_preserves_target_mae_and_applies_declared_weight() -> None:
    predictions = torch.tensor([0.0, 1.0])
    targets = torch.tensor([1.0, 1.0])
    teacher = torch.tensor([0.0, 0.0])
    objective, target_mae, teacher_mae = _objective_terms(
        predictions, targets, teacher, distillation_weight=0.25
    )
    assert float(target_mae) == pytest.approx(0.5)
    assert float(teacher_mae) == pytest.approx(0.5)
    assert float(objective) == pytest.approx(0.5)


def test_teacher_objective_rejects_unbounded_weight() -> None:
    values = torch.zeros(2)
    with pytest.raises(ValueError, match="between zero and one"):
        _objective_terms(values, values, values, distillation_weight=1.0)


def test_psnr_aligned_objective_uses_mse_and_selects_validation_mse() -> None:
    predictions = torch.tensor([0.0, 2.0])
    targets = torch.tensor([0.0, 0.0])
    objective, target_mae, teacher_mae = _objective_terms(
        predictions,
        targets,
        objective_name="target_mse",
    )
    assert float(objective) == pytest.approx(2.0)
    assert float(target_mae) == pytest.approx(1.0)
    assert teacher_mae is None
    stats = {"target_mae": 1.0, "target_mse": 2.0}
    assert _selection_value(stats, "val_mse") == pytest.approx(2.0)


def test_selection_rejects_undeclared_metrics() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint selection metric"):
        _selection_value({"target_mae": 1.0}, "val_psnr")
