import numpy as np
import pytest

from deepinterpolation_improve.config import WindowSpec
from deepinterpolation_improve.data.windowing import build_sample_indices, input_offsets, purged_center_range


def test_input_offsets_exclude_target_and_omission_neighbors():
    spec = WindowSpec(pre_frame=2, post_frame=3, pre_post_omission=1)

    assert input_offsets(spec) == [-3, -2, 2, 3, 4]


def test_build_sample_indices_respects_edges_and_negative_end_frame():
    spec = WindowSpec(pre_frame=2, post_frame=2, pre_post_omission=1)

    indices = build_sample_indices(
        total_frames=13,
        start_frame=0,
        end_frame=-1,
        window=spec,
        randomize=False,
    )

    assert indices.tolist() == list(range(3, 10))


def test_build_sample_indices_randomization_is_seeded():
    spec = WindowSpec(pre_frame=1, post_frame=1, pre_post_omission=0)

    left = build_sample_indices(12, 0, -1, spec, randomize=True, seed=11)
    right = build_sample_indices(12, 0, -1, spec, randomize=True, seed=11)

    assert left.tolist() == right.tolist()
    assert left.tolist() != list(range(1, 11))


def test_build_sample_indices_rejects_empty_valid_range():
    spec = WindowSpec(pre_frame=5, post_frame=5, pre_post_omission=1)

    with pytest.raises(ValueError, match="No valid samples"):
        build_sample_indices(8, 0, -1, spec, randomize=False)


def test_purged_center_range_keeps_all_context_frames_inside_split():
    spec = WindowSpec(pre_frame=2, post_frame=1, pre_post_omission=1)
    split_start, split_end = 10, 30

    purged_start, purged_end = purged_center_range(
        total_frames=40,
        start_frame=split_start,
        end_frame=split_end,
        window=spec,
    )

    assert (purged_start, purged_end) == (13, 28)
    for center in [purged_start, purged_end]:
        used_frames = [center + offset for offset in input_offsets(spec)]
        assert all(split_start <= frame <= split_end for frame in used_frames)
