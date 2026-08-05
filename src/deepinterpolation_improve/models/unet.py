from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class InvertedResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 2):
        super().__init__()
        hidden_channels = max(in_channels * int(expand_ratio), out_channels)
        self.use_residual = in_channels == out_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        if self.use_residual:
            output = output + x
        return output


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            InvertedResidualUnit(in_channels, out_channels),
            InvertedResidualUnit(out_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class GhostConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: int = 2):
        super().__init__()
        primary_channels = max(1, (out_channels + ratio - 1) // ratio)
        cheap_channels = primary_channels if out_channels > primary_channels else 0
        self.out_channels = out_channels
        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, primary_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.cheap = (
            nn.Sequential(
                nn.Conv2d(
                    primary_channels,
                    cheap_channels,
                    kernel_size=3,
                    padding=1,
                    groups=primary_channels,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )
            if cheap_channels > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        if self.cheap is None:
            return primary[:, : self.out_channels]
        return torch.cat([primary, self.cheap(primary)], dim=1)[:, : self.out_channels]


class GhostBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            GhostConv(in_channels, out_channels),
            GhostConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _block_factory(block_type: str):
    normalized = block_type.lower().replace("-", "_")
    if normalized in {"standard", "conv", "conv_block"}:
        return ConvBlock
    if normalized in {"depthwise", "depthwise_separable"}:
        return DepthwiseSeparableBlock
    if normalized in {"inverted_residual", "mobilenetv2"}:
        return InvertedResidualBlock
    if normalized == "ghost":
        return GhostBlock
    raise ValueError(f"Unknown U-Net block_type: {block_type}")


class UNet2D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        base_features: int = 16,
        depth: int = 4,
        block_type: str = "standard",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        block_cls = _block_factory(block_type)
        self.block_type = block_type
        features = [base_features * (2**level) for level in range(depth)]
        self.encoder = nn.ModuleList()
        in_channels = input_channels
        for feature_count in features:
            self.encoder.append(block_cls(in_channels, feature_count))
            in_channels = feature_count
        self.bottleneck = block_cls(features[-1], features[-1] * 2)
        self.decoder = nn.ModuleList()
        decoder_in = features[-1] * 2
        for feature_count in reversed(features):
            self.decoder.append(block_cls(decoder_in + feature_count, feature_count))
            decoder_in = feature_count
        self.output = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        current = x
        for block in self.encoder:
            current = block(current)
            skips.append(current)
            current = F.max_pool2d(current, kernel_size=2, stride=2)
        current = self.bottleneck(current)
        for block, skip in zip(self.decoder, reversed(skips)):
            current = F.interpolate(current, size=skip.shape[-2:], mode="nearest")
            current = torch.cat([current, skip], dim=1)
            current = block(current)
        return self.output(current)


class UNet3D(nn.Module):
    def __init__(self, input_channels: int = 1, output_channels: int = 1, base_features: int = 8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_channels, base_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features, base_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
