import pytest
import torch

from deepinterpolation_improve.models import (
    create_model_variant,
    registered_variants,
)


CANONICAL_VARIANTS = (
    "bf4_d2_standard",
    "bf4_d3_standard",
    "bf8_d2_standard",
    "bf8_d3_standard_full_reference",
    "bf8_d4_standard",
    "bf16_d3_standard",
    "bf8_d3_depthwise",
    "bf8_d3_inverted_residual",
    "bf8_d3_ghost",
)


@pytest.mark.parametrize("label", CANONICAL_VARIANTS)
def test_registered_study_variant_preserves_spatial_shape(label):
    model = create_model_variant(label, input_channels=4)

    output = model(torch.zeros(1, 4, 16, 16))

    assert output.shape == (1, 1, 16, 16)


def test_compact_di_alias_matches_historical_variant():
    alias = create_model_variant("compact_di", input_channels=4)
    historical = create_model_variant(
        "bf8_d3_standard_full_reference",
        input_channels=4,
    )

    assert sum(parameter.numel() for parameter in alias.parameters()) == sum(
        parameter.numel() for parameter in historical.parameters()
    )
    assert "compact_di" in registered_variants()


def test_unknown_variant_lists_available_labels():
    with pytest.raises(ValueError, match="Unknown model variant"):
        create_model_variant("not_a_model", input_channels=4)
