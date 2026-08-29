from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    DenseBottleneckRefiner,
    DoubleConv,
    GroupedSpatiotemporalShift,
    LocalTemporalConsensusAttention,
    NAFBlock,
    PartialNAFBlock,
    ProjectedAddDecoderBlock,
    RepDoubleConv,
    fuse_reparameterised_convolutions,
    pad_to_even,
)


class ScheduledUNet(nn.Module):
    """U-Net with an explicit width schedule and nearest-neighbour decoding."""

    def __init__(
        self,
        input_channels: int,
        features: tuple[int, ...],
        bottleneck_channels: int,
        output_channels: int = 1,
        block_kind: str = "standard",
        naf_bottleneck_blocks: int = 0,
        partial_naf_fraction: float = 0.0,
        dense_bottleneck_blocks: int = 0,
        decoder_fusion: str = "concat",
    ):
        super().__init__()
        if not features or any(value < 1 for value in features):
            raise ValueError("features must contain positive channel counts")
        block: Callable[[int, int], nn.Module]
        if block_kind == "standard":
            block = DoubleConv
        elif block_kind == "repconv":
            block = RepDoubleConv
        else:
            raise ValueError(f"Unsupported block_kind: {block_kind}")

        self.encoder = nn.ModuleList()
        current_channels = int(input_channels)
        for channels in features:
            self.encoder.append(block(current_channels, channels))
            current_channels = channels
        self.bottleneck = block(features[-1], int(bottleneck_channels))
        refiners: list[nn.Module] = [
            NAFBlock(int(bottleneck_channels)) for _ in range(int(naf_bottleneck_blocks))
        ]
        if float(partial_naf_fraction) > 0.0:
            refiners.append(PartialNAFBlock(int(bottleneck_channels), partial_naf_fraction))
        self.naf_bottleneck = nn.Sequential(*refiners)
        self.dense_bottleneck = nn.Sequential(
            *(DenseBottleneckRefiner(int(bottleneck_channels)) for _ in range(int(dense_bottleneck_blocks)))
        )
        if decoder_fusion not in {"concat", "projected_add"}:
            raise ValueError(f"Unsupported decoder_fusion: {decoder_fusion}")
        self.decoder_fusion = decoder_fusion
        self.decoder = nn.ModuleList()
        current_channels = int(bottleneck_channels)
        for channels in reversed(features):
            if decoder_fusion == "concat":
                self.decoder.append(block(current_channels + channels, channels))
            else:
                if block_kind != "standard":
                    raise ValueError("projected_add currently requires standard convolution blocks")
                self.decoder.append(ProjectedAddDecoderBlock(current_channels, channels))
            current_channels = channels
        self.output = nn.Conv2d(features[0], int(output_channels), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        current = x
        for block in self.encoder:
            current = block(current)
            skips.append(current)
            current = F.max_pool2d(current, 2, 2)
        current = self.dense_bottleneck(self.naf_bottleneck(self.bottleneck(current)))
        for block, skip in zip(self.decoder, reversed(skips)):
            current = F.interpolate(current, size=skip.shape[-2:], mode="nearest")
            if self.decoder_fusion == "concat":
                current = block(torch.cat((current, skip), dim=1))
            else:
                current = block(current, skip)
        return self.output(current)


class MeanResidualModel(nn.Module):
    def __init__(self, residual_model: nn.Module):
        super().__init__()
        self.residual_model = residual_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True) + self.residual_model(x)


class AdaptiveAnchorModel(nn.Module):
    """Per-pixel temporal softmax anchor plus a learned residual."""

    def __init__(self, input_channels: int, residual_model: nn.Module):
        super().__init__()
        self.anchor_logits = nn.Conv2d(input_channels, input_channels, 1)
        nn.init.zeros_(self.anchor_logits.weight)
        nn.init.zeros_(self.anchor_logits.bias)
        self.residual_model = residual_model

    def temporal_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.anchor_logits(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.temporal_weights(x)
        anchor = (weights * x).sum(dim=1, keepdim=True)
        return anchor + self.residual_model(x)


class ConsensusAttentionResidualModel(nn.Module):
    """Local temporal consensus attention plus a centre-frame residual path."""

    def __init__(self, input_channels: int, residual_model: nn.Module):
        super().__init__()
        self.attention_anchor = LocalTemporalConsensusAttention(input_channels)
        self.residual_model = residual_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_anchor(x) + self.residual_model(x)

    def temporal_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention_anchor.attention_weights(x)


class SubpixelResidualUNet(nn.Module):
    """Run the learned residual path on 2 x pixel-unshuffled subimages."""

    def __init__(
        self,
        input_channels: int,
        *,
        naf_bottleneck_blocks: int = 0,
        partial_naf_fraction: float = 0.0,
        detail_correction: bool = False,
        highpass_correction: bool = False,
        grouped_shift: bool = False,
        dense_bottleneck_blocks: int = 0,
        decoder_fusion: str = "concat",
    ):
        super().__init__()
        self.residual_model = ScheduledUNet(
            input_channels=input_channels * 4,
            features=(8, 16, 32),
            bottleneck_channels=64,
            output_channels=4,
            naf_bottleneck_blocks=naf_bottleneck_blocks,
            partial_naf_fraction=partial_naf_fraction,
            dense_bottleneck_blocks=dense_bottleneck_blocks,
            decoder_fusion=decoder_fusion,
        )
        _zero_output(self.residual_model)
        self.low_resolution_mixer = (
            GroupedSpatiotemporalShift(input_channels) if grouped_shift else nn.Identity()
        )
        self.detail_correction = (
            nn.Conv2d(input_channels + 1, 1, 3, padding=1) if detail_correction else None
        )
        if self.detail_correction is not None:
            nn.init.zeros_(self.detail_correction.weight)
            nn.init.zeros_(self.detail_correction.bias)
        self.highpass_projection = nn.Conv2d(input_channels, 1, 1) if highpass_correction else None
        self.highpass_gate_logit = (
            nn.Parameter(torch.tensor(-2.1972246)) if highpass_correction else None
        )
        if self.highpass_projection is not None:
            nn.init.zeros_(self.highpass_projection.weight)
            nn.init.zeros_(self.highpass_projection.bias)

    def _anchor(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded, original_shape = pad_to_even(x)
        low_resolution = self.low_resolution_mixer(F.pixel_unshuffle(padded, 2))
        residual = F.pixel_shuffle(self.residual_model(low_resolution), 2)
        temporal_mean = padded.mean(dim=1, keepdim=True)
        prediction = self._anchor(padded) + residual
        if self.detail_correction is not None:
            detail_features = torch.cat((padded - temporal_mean, temporal_mean), dim=1)
            prediction = prediction + self.detail_correction(detail_features)
        prediction = prediction[..., : original_shape[0], : original_shape[1]]
        if self.highpass_projection is not None and self.highpass_gate_logit is not None:
            highpass = x - F.avg_pool2d(x, 3, stride=1, padding=1)
            detail = self.highpass_projection(highpass)
            detail = detail - detail.mean(dim=(-2, -1), keepdim=True)
            prediction = prediction + torch.sigmoid(self.highpass_gate_logit) * detail
        return prediction

    def structural_diagnostics(self) -> dict[str, float | int] | None:
        diagnostics: dict[str, float | int] = {}
        for refiner in self.residual_model.naf_bottleneck:
            block = refiner.block if isinstance(refiner, PartialNAFBlock) else refiner
            if isinstance(refiner, PartialNAFBlock):
                diagnostics["partial_naf_active_channels"] = refiner.active_channels
                diagnostics["partial_naf_total_channels"] = refiner.channels
            if isinstance(block, NAFBlock):
                diagnostics["naf_beta_mean_abs"] = float(block.beta.detach().abs().mean().cpu())
                diagnostics["naf_gamma_mean_abs"] = float(block.gamma.detach().abs().mean().cpu())
        if self.highpass_gate_logit is not None:
            diagnostics["highpass_gate"] = float(
                torch.sigmoid(self.highpass_gate_logit.detach()).cpu()
            )
        if isinstance(self.low_resolution_mixer, GroupedSpatiotemporalShift):
            diagnostics["spatial_shift_gate_mean_abs"] = float(
                torch.tanh(self.low_resolution_mixer.spatial_gate.detach()).abs().mean().cpu()
            )
            diagnostics["temporal_shift_gate_mean_abs"] = float(
                torch.tanh(self.low_resolution_mixer.temporal_gate.detach()).abs().mean().cpu()
            )
        return diagnostics or None


class SubpixelAdaptiveAnchorUNet(SubpixelResidualUNet):
    """Subpixel residual path with a zero-initialised per-pixel temporal anchor."""

    def __init__(self, input_channels: int):
        super().__init__(input_channels)
        self.anchor_logits = nn.Conv2d(input_channels, input_channels, 1)
        nn.init.zeros_(self.anchor_logits.weight)
        nn.init.zeros_(self.anchor_logits.bias)

    def temporal_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.anchor_logits(x), dim=1)

    def _anchor(self, x: torch.Tensor) -> torch.Tensor:
        return (self.temporal_weights(x) * x).sum(dim=1, keepdim=True)


class SubpixelSymmetricAnchorUNet(SubpixelResidualUNet):
    """Subpixel residual path with weights shared by equidistant frame pairs."""

    def __init__(self, input_channels: int):
        if input_channels % 2 != 0:
            raise ValueError("symmetric temporal anchoring requires an even number of frames")
        super().__init__(input_channels)
        self.pair_count = input_channels // 2
        self.pair_logits = nn.Conv2d(self.pair_count, self.pair_count, 1)
        nn.init.zeros_(self.pair_logits.weight)
        nn.init.zeros_(self.pair_logits.bias)

    def temporal_weights(self, x: torch.Tensor) -> torch.Tensor:
        past = x[:, : self.pair_count]
        future = torch.flip(x[:, self.pair_count :], dims=(1,))
        pair_features = 0.5 * (past + future)
        pair_logits = self.pair_logits(pair_features)
        frame_logits = torch.cat((pair_logits, torch.flip(pair_logits, dims=(1,))), dim=1)
        return torch.softmax(frame_logits, dim=1)

    def _anchor(self, x: torch.Tensor) -> torch.Tensor:
        return (self.temporal_weights(x) * x).sum(dim=1, keepdim=True)


class CentreExcludedPairFusion(nn.Module):
    """Fuse invertible symmetric and signed-difference features for equidistant frame pairs."""

    def __init__(self, input_frames: int, embedding_channels: int = 8):
        super().__init__()
        if input_frames < 2 or input_frames % 2 != 0:
            raise ValueError("pair fusion requires an even number of neighbouring frames")
        self.input_frames = int(input_frames)
        self.pair_count = self.input_frames // 2
        self.embedding_channels = int(embedding_channels)
        self.pair_stem = nn.Sequential(
            nn.Conv2d(8, self.embedding_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embedding_channels, self.embedding_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(self.embedding_channels, 1, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.distance_bias = nn.Parameter(torch.zeros(1, self.pair_count, 1, 1))

    def pair_basis(self, low_resolution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = low_resolution.shape
        if channels != self.input_frames * 4:
            raise ValueError(
                f"expected {self.input_frames * 4} unshuffled channels, received {channels}"
            )
        frames = low_resolution.reshape(batch, self.input_frames, 4, height, width)
        past = torch.flip(frames[:, : self.pair_count], dims=(1,))
        future = frames[:, self.pair_count :]
        symmetric = 0.5 * (past + future)
        signed_difference = 0.5 * (future - past)
        return symmetric, signed_difference

    def forward(
        self, low_resolution: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        symmetric, signed_difference = self.pair_basis(low_resolution)
        batch, pairs, _, height, width = symmetric.shape
        pair_input = torch.cat((symmetric, signed_difference), dim=2)
        embeddings = self.pair_stem(pair_input.reshape(batch * pairs, 8, height, width))
        embeddings = embeddings.reshape(
            batch, pairs, self.embedding_channels, height, width
        )
        logits = self.score(
            embeddings.reshape(batch * pairs, self.embedding_channels, height, width)
        ).reshape(batch, pairs, height, width)
        weights = torch.softmax(logits + self.distance_bias, dim=1)
        weighted_symmetric = (weights[:, :, None] * symmetric).sum(dim=1)
        weighted_difference = (weights[:, :, None] * signed_difference).sum(dim=1)
        weighted_embedding = (weights[:, :, None] * embeddings).sum(dim=1)
        fused = torch.cat((weighted_symmetric, weighted_difference, weighted_embedding), dim=1)
        return fused, weighted_symmetric, weights


class PairFusionSubpixelUNet(nn.Module):
    """Centre-excluded temporal pair fusion followed by a low-resolution residual U-Net."""

    def __init__(self, input_channels: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.temporal_fusion = CentreExcludedPairFusion(self.input_channels)
        self.residual_model = ScheduledUNet(
            input_channels=16,
            features=(8, 16, 32),
            bottleneck_channels=64,
            output_channels=4,
        )
        _zero_output(self.residual_model)

    def _fused_state(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        padded, original_shape = pad_to_even(x)
        low_resolution = F.pixel_unshuffle(padded, 2)
        fused, anchor_low, pair_weights = self.temporal_fusion(low_resolution)
        return fused, anchor_low, pair_weights, original_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused, anchor_low, _, original_shape = self._fused_state(x)
        prediction = F.pixel_shuffle(anchor_low + self.residual_model(fused), 2)
        return prediction[..., : original_shape[0], : original_shape[1]]

    def temporal_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, _, pair_weights, original_shape = self._fused_state(x)
        frame_pair_weights = 0.5 * F.interpolate(
            pair_weights,
            size=(x.shape[-2] + x.shape[-2] % 2, x.shape[-1] + x.shape[-1] % 2),
            mode="nearest",
        )
        past_weights = torch.flip(frame_pair_weights, dims=(1,))
        frame_weights = torch.cat((past_weights, frame_pair_weights), dim=1)
        return frame_weights[..., : original_shape[0], : original_shape[1]]


def _baseline(input_channels: int) -> nn.Module:
    return ScheduledUNet(input_channels, (8, 16, 32), 64)


def _zero_output(model: ScheduledUNet) -> None:
    nn.init.zeros_(model.output.weight)
    nn.init.zeros_(model.output.bias)


def _mean_residual(input_channels: int) -> nn.Module:
    model = _baseline(input_channels)
    _zero_output(model)
    return MeanResidualModel(model)


def _mean_residual_b8d4(input_channels: int) -> nn.Module:
    model = ScheduledUNet(input_channels, (8, 16, 32, 64), 128)
    _zero_output(model)
    return MeanResidualModel(model)


def _adaptive_anchor(input_channels: int) -> nn.Module:
    model = _baseline(input_channels)
    _zero_output(model)
    return AdaptiveAnchorModel(input_channels, model)


def _lowres_heavy(input_channels: int) -> nn.Module:
    model = ScheduledUNet(input_channels, (6, 12, 24), 80)
    _zero_output(model)
    return MeanResidualModel(model)


def _rep_residual(input_channels: int) -> nn.Module:
    model = ScheduledUNet(input_channels, (8, 16, 32), 64, block_kind="repconv")
    _zero_output(model)
    return MeanResidualModel(model)


def _consensus_attention(input_channels: int) -> nn.Module:
    model = _baseline(input_channels)
    _zero_output(model)
    return ConsensusAttentionResidualModel(input_channels, model)


def _naf_bottleneck(input_channels: int) -> nn.Module:
    model = ScheduledUNet(input_channels, (8, 16, 32), 64, naf_bottleneck_blocks=2)
    _zero_output(model)
    return MeanResidualModel(model)


def _subpixel_naf_lite(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, naf_bottleneck_blocks=1)


def _subpixel_detail_fusion(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, detail_correction=True)


def _subpixel_partial_naf_quarter(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, partial_naf_fraction=0.25)


def _subpixel_partial_naf_half(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, partial_naf_fraction=0.5)


def _subpixel_highpass_gate(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, highpass_correction=True)


def _subpixel_group_shift(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, grouped_shift=True)


def _subpixel_dense_refiner(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, dense_bottleneck_blocks=1)


def _subpixel_projected_decoder(input_channels: int) -> nn.Module:
    return SubpixelResidualUNet(input_channels, decoder_fusion="projected_add")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    risk: str
    objective: str
    hypothesis: str
    inference_form: str
    builder: Callable[[int], nn.Module]
    training_objective: str = "target_l1"
    selection_metric: str = "val_mae"
    reports_temporal_weights: bool = False


_SPECS = (
    VariantSpec(
        "baseline_b8d3",
        "control",
        "reference",
        "Exact structural control for the current 122K B8-D3 standard U-Net.",
        "standard full-resolution U-Net",
        _baseline,
    ),
    VariantSpec(
        "mean_residual_b8d3",
        "low",
        "PSNR and convergence",
        "Starting from the temporal mean may reserve network capacity for centre-frame corrections.",
        "baseline U-Net plus parameter-free temporal anchor",
        _mean_residual,
    ),
    VariantSpec(
        "adaptive_anchor_b8d3",
        "medium",
        "PSNR",
        "Per-pixel temporal weights may suppress neighbours affected by local motion or transients.",
        "1 x 1 temporal gate plus baseline residual U-Net",
        _adaptive_anchor,
        reports_temporal_weights=True,
    ),
    VariantSpec(
        "lowres_heavy_residual",
        "low",
        "runtime at retained fidelity",
        "Moving channels from full-resolution stages to the bottleneck may reduce FLOPs at similar capacity.",
        "6-12-24 encoder, 80-channel bottleneck, temporal-mean residual",
        _lowres_heavy,
    ),
    VariantSpec(
        "rep_residual_b8d3",
        "medium",
        "PSNR without deployment overhead",
        "Training-time linear branches may improve optimisation while fusing exactly to plain 3 x 3 convolutions.",
        "multi-branch training; fused standard convolution inference",
        _rep_residual,
    ),
    VariantSpec(
        "consensus_attention_b8d3",
        "aggressive",
        "PSNR and temporal selectivity",
        "A low-cost consensus query may down-weight locally inconsistent neighbours without full spatial attention.",
        "half-resolution temporal attention anchor plus baseline residual U-Net",
        _consensus_attention,
        reports_temporal_weights=True,
    ),
    VariantSpec(
        "naf_bottleneck_residual",
        "aggressive",
        "PSNR",
        "NAF-style multiplicative restoration blocks may add useful low-resolution capacity at limited spatial cost.",
        "standard U-Net with two low-resolution NAF blocks",
        _naf_bottleneck,
    ),
    VariantSpec(
        "subpixel_residual_b8d3",
        "aggressive",
        "runtime",
        "Pixel unshuffle may shift most computation to one-quarter spatial area while preserving all samples.",
        "2 x pixel unshuffle, residual U-Net, pixel shuffle",
        SubpixelResidualUNet,
    ),
    VariantSpec(
        "subpixel_adaptive_anchor_b8d3",
        "medium",
        "PSNR at retained runtime",
        "A learned temporal anchor may retain the adaptive-anchor gain while the residual path remains low resolution.",
        "2 x pixel unshuffle with a 1 x 1 temporal anchor",
        SubpixelAdaptiveAnchorUNet,
        reports_temporal_weights=True,
    ),
    VariantSpec(
        "subpixel_naf_lite_b8d3",
        "medium",
        "PSNR at retained runtime",
        "One NAF-style block at the deepest subpixel scale may retain restoration capacity without the full-resolution latency penalty.",
        "2 x pixel unshuffle with one deepest-scale NAF block",
        _subpixel_naf_lite,
    ),
    VariantSpec(
        "subpixel_detail_fusion_b8d3",
        "medium",
        "MAE and PSNR at retained runtime",
        "A zero-initialised full-resolution correction may recover detail lost by the low-resolution residual path.",
        "2 x pixel unshuffle plus one shallow full-resolution correction",
        _subpixel_detail_fusion,
    ),
    VariantSpec(
        "subpixel_partial_naf_quarter_b8d3",
        "medium",
        "PSNR at reduced NAF latency",
        "Applying NAF restoration to one quarter of bottleneck channels may retain quality with fewer small operators.",
        "2 x pixel unshuffle with quarter-channel deepest-scale NAF",
        _subpixel_partial_naf_quarter,
    ),
    VariantSpec(
        "subpixel_partial_naf_half_b8d3",
        "medium",
        "PSNR at reduced NAF latency",
        "Applying NAF restoration to half of bottleneck channels may balance restoration capacity and device cost.",
        "2 x pixel unshuffle with half-channel deepest-scale NAF",
        _subpixel_partial_naf_half,
    ),
    VariantSpec(
        "subpixel_symmetric_anchor_b8d3",
        "medium",
        "PSNR with constrained temporal weighting",
        "Sharing weights between equidistant past and future frames may reduce unnecessary anchor freedom.",
        "2 x pixel unshuffle with distance-symmetric temporal anchor",
        SubpixelSymmetricAnchorUNet,
        reports_temporal_weights=True,
    ),
    VariantSpec(
        "subpixel_highpass_gate_b8d3",
        "medium",
        "MAE and PSNR",
        "A zero-mean gated high-frequency correction may improve detail without shifting low-frequency reconstruction.",
        "2 x pixel unshuffle plus gated zero-mean high-pass correction",
        _subpixel_highpass_gate,
    ),
    VariantSpec(
        "subpixel_group_shift_b8d3",
        "aggressive",
        "PSNR at retained convolution cost",
        "Zero-gated spatial-phase shifts and symmetric temporal exchange may expand low-resolution aggregation context.",
        "2 x pixel unshuffle with grouped low-resolution spatiotemporal shift",
        _subpixel_group_shift,
    ),
    VariantSpec(
        "subpixel_distilled_b8d3",
        "medium",
        "PSNR with unchanged inference graph",
        "A NAF-lite teacher may transfer reconstruction preferences into the unchanged Subpixel student.",
        "same inference graph as subpixel_residual_b8d3",
        SubpixelResidualUNet,
        "target_l1_plus_naf_teacher",
    ),
    VariantSpec(
        "ctpf_subpixel_b8d3",
        "aggressive",
        "temporally explicit reconstruction",
        "An invertible symmetric/difference basis may preserve directional temporal evidence before feature fusion.",
        "shared equidistant-pair stem, low-resolution distance gate and subpixel residual U-Net",
        PairFusionSubpixelUNet,
        reports_temporal_weights=True,
    ),
    VariantSpec(
        "subpixel_dense_refiner_b8d3",
        "medium",
        "PSNR with regular GPU kernels",
        "Two regular low-resolution convolutions may retain the NAF-lite quality signal with fewer small operators.",
        "subpixel residual U-Net with one dense bottleneck residual block",
        _subpixel_dense_refiner,
    ),
    VariantSpec(
        "subpixel_projected_decoder_b8d3",
        "medium",
        "runtime at retained fidelity",
        "Projected additive skip fusion may reduce decoder convolution and concatenation cost.",
        "subpixel residual U-Net with projected-add decoder fusion",
        _subpixel_projected_decoder,
    ),
    VariantSpec(
        "subpixel_psnr_aligned_b8d3",
        "low",
        "PSNR without inference overhead",
        "MSE training and validation selection may align optimisation with the reported PSNR endpoint.",
        "same inference graph as subpixel_residual_b8d3",
        SubpixelResidualUNet,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "baseline_psnr_aligned_b8d3",
        "control",
        "PSNR-aligned full-resolution reference",
        "MSE training isolates the effect of the subpixel structure from objective alignment.",
        "standard full-resolution U-Net",
        _baseline,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "ctpf_psnr_aligned_b8d3",
        "aggressive",
        "temporally explicit PSNR",
        "Pairwise temporal fusion may add reconstruction value after every model is aligned to MSE.",
        "same inference graph as ctpf_subpixel_b8d3",
        PairFusionSubpixelUNet,
        "target_mse",
        "val_mse",
        reports_temporal_weights=True,
    ),
    VariantSpec(
        "subpixel_dense_refiner_psnr_aligned_b8d3",
        "medium",
        "PSNR with regular GPU kernels",
        "A dense bottleneck refiner may add quality after controlling the training objective.",
        "same inference graph as subpixel_dense_refiner_b8d3",
        _subpixel_dense_refiner,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "subpixel_naf_lite_psnr_aligned_b8d3",
        "aggressive",
        "quality-heavy PSNR",
        "NAF-lite may stack with MSE alignment strongly enough to justify its measured latency.",
        "same inference graph as subpixel_naf_lite_b8d3",
        _subpixel_naf_lite,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "paper_fullres_b8d3_mae",
        "control",
        "factorial full-resolution MAE arm",
        "A temporal-mean residual anchor controls the computation-location comparison.",
        "full-resolution B8-D3 temporal-mean residual U-Net",
        _mean_residual,
    ),
    VariantSpec(
        "paper_fullres_b8d3_mse",
        "control",
        "factorial full-resolution MSE arm",
        "MSE training isolates objective alignment at fixed full-resolution structure.",
        "same inference graph as paper_fullres_b8d3_mae",
        _mean_residual,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "paper_subpixel_b8d3_mae",
        "controlled",
        "factorial subpixel MAE arm",
        "PixelUnshuffle relocates learned computation while preserving the temporal anchor.",
        "2 x pixel unshuffle, B8-D3 residual U-Net and pixel shuffle",
        SubpixelResidualUNet,
    ),
    VariantSpec(
        "paper_subpixel_b8d3_mse",
        "controlled",
        "factorial subpixel MSE arm",
        "MSE aligns the subpixel model with the reported PSNR endpoint.",
        "same inference graph as paper_subpixel_b8d3_mae",
        SubpixelResidualUNet,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "paper_fullres_b8d4_mse",
        "reference",
        "higher-capacity quality reference",
        "A B8-D4 temporal-mean residual U-Net measures the local return from added capacity.",
        "full-resolution B8-D4 temporal-mean residual U-Net",
        _mean_residual_b8d4,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "paper_subpixel_dense_b8d3_mse",
        "mechanism control",
        "regular low-resolution capacity",
        "A dense bottleneck tests whether ordinary low-resolution capacity adds value beyond G4.",
        "same inference graph as subpixel_dense_refiner_b8d3",
        _subpixel_dense_refiner,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "paper_subpixel_naf_lite_b8d3_mse",
        "mechanism control",
        "small-operator restoration capacity",
        "A NAF-lite bottleneck tests whether theoretical low-FLOP operators add value beyond G4.",
        "same inference graph as subpixel_naf_lite_b8d3",
        _subpixel_naf_lite,
        "target_mse",
        "val_mse",
    ),
    VariantSpec(
        "paper_ctpf_subpixel_b8d3_mse",
        "mechanism control",
        "explicit temporal pair fusion",
        "Centre-excluded pair fusion tests whether learned temporal weighting adds value beyond G4.",
        "same inference graph as ctpf_subpixel_b8d3",
        PairFusionSubpixelUNet,
        "target_mse",
        "val_mse",
        reports_temporal_weights=True,
    ),
)
_BY_NAME = {spec.name: spec for spec in _SPECS}


def registered_variants() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def variant_spec(name: str) -> VariantSpec:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant {name!r}; choose from {', '.join(registered_variants())}") from exc


def create_variant(name: str, input_channels: int, deploy: bool = False) -> nn.Module:
    model = variant_spec(name).builder(int(input_channels))
    if deploy:
        fuse_reparameterised_convolutions(model)
    return model
