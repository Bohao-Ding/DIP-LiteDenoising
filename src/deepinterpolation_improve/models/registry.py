from __future__ import annotations

from typing import Any

from .unet import UNet2D, UNet3D


def _unet2d(config: dict[str, Any]) -> UNet2D:
    return UNet2D(
        input_channels=int(config.get("input_channels", 60)),
        output_channels=int(config.get("output_channels", 1)),
        base_features=int(config.get("base_features", 16)),
        depth=int(config.get("depth", 4)),
        block_type=str(config.get("block_type", "standard")),
    )


def _unet2d_with_block(block_type: str):
    def factory(config: dict[str, Any]) -> UNet2D:
        merged = dict(config)
        merged.setdefault("block_type", block_type)
        return _unet2d(merged)

    return factory


def _unet3d(config: dict[str, Any]) -> UNet3D:
    return UNet3D(
        input_channels=int(config.get("input_channels", 1)),
        output_channels=int(config.get("output_channels", 1)),
        base_features=int(config.get("base_features", 8)),
    )


_REGISTRY = {
    "unet_single_1024": _unet2d,
    "unet_2d": _unet2d,
    "lightweight_unet": _unet2d,
    "depthwise_unet": _unet2d_with_block("depthwise"),
    "inverted_residual_unet": _unet2d_with_block("inverted_residual"),
    "ghost_unet": _unet2d_with_block("ghost"),
    "unet_3d": _unet3d,
}


def create_model(name: str, config: dict[str, Any]) -> nn.Module:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model name: {name}") from exc
    return factory(config)


def registered_models() -> list[str]:
    return sorted(_REGISTRY)
