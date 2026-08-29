from __future__ import annotations

import torch

from deepinterpolation_improve.experiments.paper_evidence_v1.blocks import GroupedSpatiotemporalShift
from deepinterpolation_improve.experiments.paper_evidence_v1.models import (
    CentreExcludedPairFusion,
    create_variant,
    registered_variants,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.profiling import benchmark_latency, conv_profile, inference_copy, parameter_count


torch.set_num_threads(1)


def test_all_variants_preserve_odd_spatial_shape_for_every_window() -> None:
    for channels in (2, 4, 8):
        inputs = torch.randn(1, channels, 33, 35)
        for name in registered_variants():
            with torch.inference_mode():
                output = create_variant(name, channels)(inputs)
            assert output.shape == (1, 1, 33, 35), name


def test_residual_candidates_start_from_temporal_mean() -> None:
    inputs = torch.randn(1, 4, 33, 35)
    expected = inputs.mean(dim=1, keepdim=True)
    residual_candidates = set(registered_variants()) - {
        "baseline_b8d3",
        "baseline_psnr_aligned_b8d3",
    }
    for name in residual_candidates:
        with torch.inference_mode():
            output = create_variant(name, 4)(inputs)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6, msg=name)


def test_repconv_deploy_conversion_is_numerically_equivalent() -> None:
    torch.manual_seed(3)
    model = create_variant("rep_residual_b8d3", 4).eval()
    inputs = torch.randn(2, 4, 33, 35)
    with torch.inference_mode():
        before = model(inputs)
        deployed = inference_copy(model)
        after = deployed(inputs)
    torch.testing.assert_close(after, before, atol=2e-6, rtol=2e-6)


def test_reparameterised_inference_matches_baseline_parameter_count() -> None:
    baseline = create_variant("baseline_b8d3", 4)
    rep_training = create_variant("rep_residual_b8d3", 4)
    rep_deployed = inference_copy(rep_training)
    assert parameter_count(rep_training) > parameter_count(baseline)
    assert parameter_count(rep_deployed) == parameter_count(baseline)


def test_speed_candidates_reduce_conv_flops_on_small_input() -> None:
    shape = (4, 64, 64)
    baseline = conv_profile(inference_copy(create_variant("baseline_b8d3", 4)), shape)
    lowres = conv_profile(inference_copy(create_variant("lowres_heavy_residual", 4)), shape)
    subpixel = conv_profile(inference_copy(create_variant("subpixel_residual_b8d3", 4)), shape)
    targeted = [
        conv_profile(inference_copy(create_variant(name, 4)), shape)
        for name in (
            "subpixel_adaptive_anchor_b8d3",
            "subpixel_naf_lite_b8d3",
            "subpixel_detail_fusion_b8d3",
            "subpixel_partial_naf_quarter_b8d3",
            "subpixel_partial_naf_half_b8d3",
            "subpixel_symmetric_anchor_b8d3",
            "subpixel_highpass_gate_b8d3",
            "subpixel_group_shift_b8d3",
            "subpixel_distilled_b8d3",
        )
    ]
    assert lowres["conv_flops"] < baseline["conv_flops"]
    assert subpixel["conv_flops"] < baseline["conv_flops"]
    assert all(profile["conv_flops"] < baseline["conv_flops"] for profile in targeted)


def test_consensus_attention_has_small_parameter_overhead() -> None:
    baseline = parameter_count(create_variant("baseline_b8d3", 4))
    attention = parameter_count(create_variant("consensus_attention_b8d3", 4))
    assert attention > baseline
    assert attention / baseline < 1.01


def test_temporal_attention_weights_are_normalised_and_initially_uniform() -> None:
    inputs = torch.randn(2, 4, 31, 29)
    for name in (
        "adaptive_anchor_b8d3",
        "consensus_attention_b8d3",
        "subpixel_adaptive_anchor_b8d3",
        "subpixel_symmetric_anchor_b8d3",
    ):
        model = create_variant(name, 4)
        with torch.inference_mode():
            weights = model.temporal_weights(inputs)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
        torch.testing.assert_close(weights, torch.full_like(weights, 0.25), atol=1e-6, rtol=1e-6)


def test_consensus_attention_has_a_live_gradient_path_at_initialisation() -> None:
    torch.manual_seed(11)
    model = create_variant("consensus_attention_b8d3", 4)
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-2)
    inputs = torch.randn(2, 4, 17, 19)
    target = torch.randn(2, 1, 17, 19)

    optimiser.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(inputs), target).backward()
    first_query_gradient = model.attention_anchor.query.weight.grad
    assert first_query_gradient is not None
    assert float(first_query_gradient.abs().sum()) > 0.0
    optimiser.step()

    optimiser.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(inputs), target).backward()
    second_key_gradient = model.attention_anchor.key.weight.grad
    assert second_key_gradient is not None
    assert float(second_key_gradient.abs().sum()) > 0.0


def test_cpu_latency_contract_records_precision_and_memory_format() -> None:
    result = benchmark_latency(
        create_variant("baseline_b8d3", 4),
        (4, 16, 16),
        device="cpu",
        batch_size=1,
        warmup=1,
        repeats=3,
        precision="fp32",
        channels_last=False,
    )
    assert result["precision"] == "fp32"
    assert result["channels_last"] is False
    assert result["latency_ms_p50"] > 0
    assert result["resident_memory_mb"] == 0.0
    assert result["incremental_peak_memory_mb"] == 0.0
    assert result["peak_allocated_memory_mb"] == 0.0


def test_targeted_anchor_and_detail_paths_receive_initial_gradients() -> None:
    torch.manual_seed(17)
    inputs = torch.randn(2, 4, 17, 19)
    target = torch.randn(2, 1, 17, 19)

    anchor = create_variant("subpixel_adaptive_anchor_b8d3", 4)
    torch.nn.functional.l1_loss(anchor(inputs), target).backward()
    assert anchor.anchor_logits.weight.grad is not None
    assert float(anchor.anchor_logits.weight.grad.abs().sum()) > 0.0

    detail = create_variant("subpixel_detail_fusion_b8d3", 4)
    torch.nn.functional.l1_loss(detail(inputs), target).backward()
    assert detail.detail_correction.weight.grad is not None
    assert float(detail.detail_correction.weight.grad.abs().sum()) > 0.0


def test_subpixel_naf_lite_is_smaller_than_the_first_round_naf_candidate() -> None:
    lite = create_variant("subpixel_naf_lite_b8d3", 4)
    first_round = create_variant("naf_bottleneck_residual", 4)
    assert parameter_count(lite) < parameter_count(first_round)


def test_partial_naf_capacity_is_ordered_and_below_full_naf_lite() -> None:
    shape = (4, 64, 64)
    quarter = create_variant("subpixel_partial_naf_quarter_b8d3", 4)
    half = create_variant("subpixel_partial_naf_half_b8d3", 4)
    full = create_variant("subpixel_naf_lite_b8d3", 4)
    assert parameter_count(quarter) < parameter_count(half) < parameter_count(full)
    profiles = [conv_profile(inference_copy(model), shape)["conv_flops"] for model in (quarter, half, full)]
    assert profiles[0] < profiles[1] < profiles[2]


def test_symmetric_anchor_ties_equidistant_frame_weights() -> None:
    torch.manual_seed(23)
    model = create_variant("subpixel_symmetric_anchor_b8d3", 4)
    with torch.no_grad():
        model.pair_logits.weight.normal_()
        model.pair_logits.bias.normal_()
        weights = model.temporal_weights(torch.randn(2, 4, 19, 21))
    torch.testing.assert_close(weights[:, 0], weights[:, 3])
    torch.testing.assert_close(weights[:, 1], weights[:, 2])


def test_highpass_correction_is_zero_mean_and_has_a_live_initial_gradient() -> None:
    torch.manual_seed(29)
    model = create_variant("subpixel_highpass_gate_b8d3", 4)
    inputs = torch.randn(2, 4, 17, 19)
    target = torch.randn(2, 1, 17, 19)
    torch.nn.functional.l1_loss(model(inputs), target).backward()
    assert model.highpass_projection.weight.grad is not None
    assert float(model.highpass_projection.weight.grad.abs().sum()) > 0.0

    with torch.no_grad():
        model.highpass_projection.weight.normal_()
        correction = model(inputs) - inputs.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        correction.mean(dim=(-2, -1)),
        torch.zeros_like(correction.mean(dim=(-2, -1))),
        atol=2e-6,
        rtol=0.0,
    )


def test_grouped_shift_starts_as_identity_and_its_gates_are_trainable() -> None:
    torch.manual_seed(31)
    mixer = GroupedSpatiotemporalShift(4)
    inputs = torch.randn(2, 16, 17, 19)
    torch.testing.assert_close(mixer(inputs), inputs)
    mixer(inputs).square().mean().backward()
    assert mixer.spatial_gate.grad is not None
    assert mixer.temporal_gate.grad is not None
    assert float(mixer.spatial_gate.grad.abs().sum()) > 0.0
    assert float(mixer.temporal_gate.grad.abs().sum()) > 0.0


def test_grouped_shift_model_supports_channels_last_training_contract() -> None:
    model = create_variant("subpixel_group_shift_b8d3", 4).to(
        memory_format=torch.channels_last
    )
    inputs = torch.randn(2, 4, 33, 35).contiguous(memory_format=torch.channels_last)
    output = model(inputs)
    assert output.shape == (2, 1, 33, 35)
    output.square().mean().backward()


def test_distilled_student_keeps_the_exact_subpixel_inference_contract() -> None:
    reference = create_variant("subpixel_residual_b8d3", 4)
    student = create_variant("subpixel_distilled_b8d3", 4)
    assert tuple(reference.state_dict()) == tuple(student.state_dict())
    assert parameter_count(reference) == parameter_count(student)
    reference_profile = conv_profile(inference_copy(reference), (4, 64, 64))
    student_profile = conv_profile(inference_copy(student), (4, 64, 64))
    assert reference_profile == student_profile


def test_pair_basis_is_invertible_for_every_supported_window() -> None:
    torch.manual_seed(37)
    for frames in (2, 4, 8):
        module = CentreExcludedPairFusion(frames)
        low_resolution = torch.randn(2, frames * 4, 9, 11)
        symmetric, signed_difference = module.pair_basis(low_resolution)
        recovered_past = symmetric - signed_difference
        recovered_future = symmetric + signed_difference
        original = low_resolution.reshape(2, frames, 4, 9, 11)
        expected_past = torch.flip(original[:, : frames // 2], dims=(1,))
        expected_future = original[:, frames // 2 :]
        torch.testing.assert_close(recovered_past, expected_past)
        torch.testing.assert_close(recovered_future, expected_future)


def test_pair_fusion_initialises_to_the_temporal_mean_and_has_live_weights() -> None:
    torch.manual_seed(41)
    model = create_variant("ctpf_subpixel_b8d3", 4)
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-2)
    inputs = torch.randn(2, 4, 33, 35)
    target = torch.randn(2, 1, 33, 35)
    torch.testing.assert_close(model(inputs), inputs.mean(dim=1, keepdim=True))
    weights = model.temporal_weights(inputs)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    optimiser.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(inputs), target).backward()
    assert model.temporal_fusion.score.weight.grad is not None
    assert float(model.temporal_fusion.score.weight.grad.abs().sum()) > 0.0
    optimiser.step()
    optimiser.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(inputs), target).backward()
    stem_gradient = model.temporal_fusion.pair_stem[0].weight.grad
    assert stem_gradient is not None
    assert float(stem_gradient.abs().sum()) > 0.0


def test_projected_decoder_reduces_subpixel_convolution_cost() -> None:
    shape = (4, 64, 64)
    reference = conv_profile(create_variant("subpixel_residual_b8d3", 4), shape)
    projected = conv_profile(create_variant("subpixel_projected_decoder_b8d3", 4), shape)
    assert projected["conv_flops"] < reference["conv_flops"]


def test_psnr_aligned_variant_keeps_the_subpixel_inference_contract() -> None:
    reference = create_variant("subpixel_residual_b8d3", 4)
    aligned = create_variant("subpixel_psnr_aligned_b8d3", 4)
    assert tuple(reference.state_dict()) == tuple(aligned.state_dict())
    assert parameter_count(reference) == parameter_count(aligned)


def test_psnr_aligned_structure_aliases_keep_their_parent_inference_graphs() -> None:
    pairs = (
        ("baseline_b8d3", "baseline_psnr_aligned_b8d3"),
        ("ctpf_subpixel_b8d3", "ctpf_psnr_aligned_b8d3"),
        ("subpixel_dense_refiner_b8d3", "subpixel_dense_refiner_psnr_aligned_b8d3"),
        ("subpixel_naf_lite_b8d3", "subpixel_naf_lite_psnr_aligned_b8d3"),
    )
    for parent_name, aligned_name in pairs:
        parent = create_variant(parent_name, 4)
        aligned = create_variant(aligned_name, 4)
        assert tuple(parent.state_dict()) == tuple(aligned.state_dict())
        assert parameter_count(parent) == parameter_count(aligned)


def test_temporal_diagnostic_contract_follows_structure_aliases() -> None:
    from deepinterpolation_improve.experiments.paper_evidence_v1.models import variant_spec

    assert variant_spec("ctpf_subpixel_b8d3").reports_temporal_weights is True
    assert variant_spec("ctpf_psnr_aligned_b8d3").reports_temporal_weights is True
    assert variant_spec("subpixel_residual_b8d3").reports_temporal_weights is False
    assert variant_spec("subpixel_psnr_aligned_b8d3").reports_temporal_weights is False


def test_paper_factorial_arms_change_only_structure_or_objective() -> None:
    full_mae = create_variant("paper_fullres_b8d3_mae", 4)
    full_mse = create_variant("paper_fullres_b8d3_mse", 4)
    subpixel_mae = create_variant("paper_subpixel_b8d3_mae", 4)
    subpixel_mse = create_variant("paper_subpixel_b8d3_mse", 4)
    assert tuple(full_mae.state_dict()) == tuple(full_mse.state_dict())
    assert tuple(subpixel_mae.state_dict()) == tuple(subpixel_mse.state_dict())
    assert parameter_count(full_mae) == parameter_count(full_mse)
    assert parameter_count(subpixel_mae) == parameter_count(subpixel_mse)


def test_paper_b8d4_reference_has_the_expected_capacity_order() -> None:
    b8d3 = create_variant("paper_fullres_b8d3_mse", 4)
    b8d4 = create_variant("paper_fullres_b8d4_mse", 4)
    assert parameter_count(b8d3) == 122185
    assert parameter_count(b8d4) == 491209
    inputs = torch.randn(1, 4, 33, 35)
    assert b8d4(inputs).shape == (1, 1, 33, 35)


def test_paper_mechanism_aliases_keep_expected_objectives_and_diagnostics() -> None:
    from deepinterpolation_improve.experiments.paper_evidence_v1.models import variant_spec

    for name in (
        "paper_subpixel_dense_b8d3_mse",
        "paper_subpixel_naf_lite_b8d3_mse",
        "paper_ctpf_subpixel_b8d3_mse",
    ):
        assert variant_spec(name).training_objective == "target_mse"
        assert variant_spec(name).selection_metric == "val_mse"
    assert variant_spec("paper_ctpf_subpixel_b8d3_mse").reports_temporal_weights is True
