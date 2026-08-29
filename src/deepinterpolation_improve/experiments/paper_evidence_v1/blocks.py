from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3 x 3 convolutions matching the current PyTorch baseline."""

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


class ProjectedAddDecoderBlock(nn.Module):
    """Project an upsampled decoder tensor and add the same-width skip tensor."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1)
        self.refine = DoubleConv(out_channels, out_channels)

    def forward(self, current: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.refine(self.project(current) + skip)


class RepConv2d(nn.Module):
    """No-BatchNorm structural re-parameterisation for one 3 x 3 layer.

    Training uses 3 x 3, 1 x 1 and optional identity branches.  All branches
    are linear before ReLU, so they can be fused exactly into one 3 x 3
    convolution for inference.
    """

    def __init__(self, in_channels: int, out_channels: int, deploy: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.deploy = bool(deploy)
        self.use_identity = self.in_channels == self.out_channels
        if self.deploy:
            self.fused = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        else:
            self.branch_3x3 = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
            self.branch_1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.fused(x)
        output = self.branch_3x3(x) + self.branch_1x1(x)
        if self.use_identity:
            output = output + x
        return output

    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.fused.weight, self.fused.bias
        kernel = self.branch_3x3.weight.clone()
        bias = self.branch_3x3.bias.clone()
        kernel[:, :, 1:2, 1:2] += self.branch_1x1.weight
        bias += self.branch_1x1.bias
        if self.use_identity:
            indices = torch.arange(self.in_channels, device=kernel.device)
            kernel[indices, indices, 1, 1] += 1.0
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.equivalent_kernel_bias()
        fused = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        fused = fused.to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            fused.weight.copy_(kernel)
            fused.bias.copy_(bias)
        self.fused = fused
        del self.branch_3x3
        del self.branch_1x1
        self.deploy = True


class RepDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deploy: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            RepConv2d(in_channels, out_channels, deploy=deploy),
            nn.ReLU(inplace=True),
            RepConv2d(out_channels, out_channels, deploy=deploy),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Compact NAF-style residual block used only at the bottleneck."""

    def __init__(self, channels: int):
        super().__init__()
        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.expand1 = nn.Conv2d(channels, expanded, 1)
        self.depthwise = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.gate1 = SimpleGate()
        self.channel_scale = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project1 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.expand2 = nn.Conv2d(channels, expanded, 1)
        self.gate2 = SimpleGate()
        self.project2 = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.expand1(self.norm1(x))
        first = self.gate1(self.depthwise(first))
        first = self.project1(first * self.channel_scale(first))
        y = x + first * self.beta
        second = self.project2(self.gate2(self.expand2(self.norm2(y))))
        return y + second * self.gamma


class DenseBottleneckRefiner(nn.Module):
    """Two regular 3 x 3 kernels for a low-resolution, GPU-friendly residual update."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.gate) * self.body(x)


class PartialNAFBlock(nn.Module):
    """Apply one NAF block to a fixed channel subset and bypass the rest."""

    def __init__(self, channels: int, active_fraction: float):
        super().__init__()
        if not 0.0 < float(active_fraction) < 1.0:
            raise ValueError("active_fraction must be between zero and one")
        self.channels = int(channels)
        self.active_channels = max(1, round(self.channels * float(active_fraction)))
        self.block = NAFBlock(self.active_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.block(x[:, : self.active_channels])
        return torch.cat((active, x[:, self.active_channels :]), dim=1)


def _shift_without_wrap(x: torch.Tensor, vertical: int, horizontal: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    padded = F.pad(
        x,
        (
            max(horizontal, 0),
            max(-horizontal, 0),
            max(vertical, 0),
            max(-vertical, 0),
        ),
    )
    top = max(-vertical, 0)
    left = max(-horizontal, 0)
    return padded[..., top : top + height, left : left + width]


class GroupedSpatiotemporalShift(nn.Module):
    """Zero-gated spatial-phase shifts and symmetric temporal exchange."""

    def __init__(self, input_frames: int):
        super().__init__()
        if input_frames < 2:
            raise ValueError("grouped shift requires at least two neighbouring frames")
        self.input_frames = int(input_frames)
        # Keep trainable tensors four-dimensional so ``Module.to`` can apply
        # channels-last format to the full model.  The batch axis is restored
        # only for the broadcast in ``forward``.
        gate_shape = (self.input_frames, 4, 1, 1)
        self.spatial_gate = nn.Parameter(torch.zeros(gate_shape))
        self.temporal_gate = nn.Parameter(torch.zeros(gate_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        expected_channels = self.input_frames * 4
        if channels != expected_channels:
            raise ValueError(f"expected {expected_channels} unshuffled channels, received {channels}")
        frames = x.reshape(batch, self.input_frames, 4, height, width)
        phase_groups = frames.unbind(dim=2)
        shifted = torch.stack(
            (
                _shift_without_wrap(phase_groups[0], -1, 0),
                _shift_without_wrap(phase_groups[1], 1, 0),
                _shift_without_wrap(phase_groups[2], 0, -1),
                _shift_without_wrap(phase_groups[3], 0, 1),
            ),
            dim=2,
        )
        symmetric_frames = torch.flip(frames, dims=(1,))
        spatial_gate = torch.tanh(self.spatial_gate).unsqueeze(0)
        temporal_gate = torch.tanh(self.temporal_gate).unsqueeze(0)
        mixed = (
            frames
            + spatial_gate * (shifted - frames)
            + temporal_gate * (symmetric_frames - frames)
        )
        return mixed.reshape(batch, channels, height, width)


class LocalTemporalConsensusAttention(nn.Module):
    """Efficient attention across neighbouring frames, never across all pixels.

    A shared spatial stem describes each input frame at half resolution.  A
    consensus query scores one key per neighbouring frame and spatial location.
    The resulting weights mix the original full-resolution frames.  The query
    starts at uniform temporal weights.  A small non-zero residual
    gate keeps the initial anchor equal to the temporal mean while preserving a
    gradient path into the attention parameters.
    """

    def __init__(self, input_frames: int, embedding_channels: int = 8):
        super().__init__()
        if input_frames < 2:
            raise ValueError("temporal attention requires at least two neighbouring frames")
        self.input_frames = int(input_frames)
        self.embedding_channels = int(embedding_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, self.embedding_channels, 3, padding=1),
            nn.GELU(),
        )
        self.query = nn.Conv2d(self.embedding_channels, self.embedding_channels, 1, bias=False)
        self.key = nn.Conv2d(self.embedding_channels, self.embedding_channels, 1, bias=False)
        nn.init.zeros_(self.query.weight)
        self.temporal_bias = nn.Parameter(torch.zeros(1, self.input_frames, 1, 1))
        self.residual_gate = nn.Parameter(torch.tensor(math.atanh(0.1)))

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, height, width = x.shape
        if frames != self.input_frames:
            raise ValueError(f"expected {self.input_frames} frames, received {frames}")
        flattened = x.reshape(batch * frames, 1, height, width)
        if height >= 2 and width >= 2:
            flattened = F.avg_pool2d(flattened, 2, 2)
        features = self.stem(flattened)
        low_height, low_width = features.shape[-2:]
        features = features.reshape(batch, frames, self.embedding_channels, low_height, low_width)
        consensus = features.mean(dim=1)
        query = self.query(consensus)
        keys = self.key(features.reshape(batch * frames, self.embedding_channels, low_height, low_width))
        keys = keys.reshape(batch, frames, self.embedding_channels, low_height, low_width)
        logits = (query[:, None] * keys).sum(dim=2) / math.sqrt(self.embedding_channels)
        low_weights = torch.softmax(logits + self.temporal_bias, dim=1)
        weights = F.interpolate(
            low_weights.reshape(batch * frames, 1, low_height, low_width),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, height, width)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(x)
        attended = (weights * x).sum(dim=1, keepdim=True)
        temporal_mean = x.mean(dim=1, keepdim=True)
        return temporal_mean + torch.tanh(self.residual_gate) * (attended - temporal_mean)


def fuse_reparameterised_convolutions(module: nn.Module) -> nn.Module:
    """Fuse every :class:`RepConv2d` in-place and return the model."""

    for child in list(module.modules()):
        if isinstance(child, RepConv2d):
            child.switch_to_deploy()
    return module


def pad_to_even(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_bottom = height % 2
    pad_right = width % 2
    if pad_bottom or pad_right:
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode=mode)
    return x, (height, width)
