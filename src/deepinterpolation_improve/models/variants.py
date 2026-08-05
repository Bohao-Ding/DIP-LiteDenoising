from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .registry import create_model


@dataclass(frozen=True)
class ModelVariantSpec:
    """Architecture settings for one model variant used in the study."""

    label: str
    registry_name: str
    base_features: int
    depth: int
    block_type: str = "standard"


_VARIANTS = (
    ModelVariantSpec("bf4_d2_standard", "unet_single_1024", 4, 2),
    ModelVariantSpec("bf4_d3_standard", "unet_single_1024", 4, 3),
    ModelVariantSpec("bf8_d2_standard", "unet_single_1024", 8, 2),
    ModelVariantSpec("bf8_d3_standard_full_reference", "unet_single_1024", 8, 3),
    ModelVariantSpec("bf8_d4_standard", "unet_single_1024", 8, 4),
    ModelVariantSpec("bf16_d3_standard", "unet_single_1024", 16, 3),
    ModelVariantSpec("bf8_d3_depthwise", "depthwise_unet", 8, 3, "depthwise"),
    ModelVariantSpec(
        "bf8_d3_inverted_residual",
        "inverted_residual_unet",
        8,
        3,
        "inverted_residual",
    ),
    ModelVariantSpec("bf8_d3_ghost", "ghost_unet", 8, 3, "ghost"),
)
_VARIANTS_BY_LABEL = {variant.label: variant for variant in _VARIANTS}
_ALIASES = {"compact_di": "bf8_d3_standard_full_reference"}


def registered_variants() -> tuple[str, ...]:
    """Return the nine study labels followed by supported readable aliases."""

    return tuple(_VARIANTS_BY_LABEL) + tuple(_ALIASES)


def create_model_variant(
    label: str,
    input_channels: int,
    output_channels: int = 1,
) -> nn.Module:
    """Build a registered study variant for the requested temporal context."""

    canonical_label = _ALIASES.get(label, label)
    try:
        variant = _VARIANTS_BY_LABEL[canonical_label]
    except KeyError as exc:
        available = ", ".join(registered_variants())
        raise ValueError(f"Unknown model variant: {label}. Available variants: {available}") from exc
    return create_model(
        variant.registry_name,
        {
            "input_channels": int(input_channels),
            "output_channels": int(output_channels),
            "base_features": variant.base_features,
            "depth": variant.depth,
            "block_type": variant.block_type,
        },
    )
