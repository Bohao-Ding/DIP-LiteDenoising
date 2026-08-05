from __future__ import annotations

import numpy as np

from deepinterpolation_improve.config import WindowSpec


def input_offsets(window: WindowSpec) -> list[int]:
    start = -window.pre_frame - window.pre_post_omission
    stop = window.post_frame + window.pre_post_omission
    omitted = set(range(-window.pre_post_omission, window.pre_post_omission + 1))
    return [offset for offset in range(start, stop + 1) if offset not in omitted]


def resolve_end_frame(total_frames: int, end_frame: int) -> int:
    if end_frame < 0:
        return total_frames + end_frame
    return min(end_frame, total_frames - 1)


def purged_center_range(total_frames: int, start_frame: int, end_frame: int, window: WindowSpec) -> tuple[int, int]:
    resolved_end = resolve_end_frame(total_frames, end_frame)
    offsets = input_offsets(window)
    left_context = max(0, -min(offsets, default=0))
    right_context = max(0, max(offsets, default=0))
    purged_start = int(start_frame) + left_context
    purged_end = resolved_end - right_context
    if purged_end < purged_start:
        raise ValueError(
            "No valid purged samples after applying temporal split gap: "
            f"start={purged_start}, end={purged_end}, total_frames={total_frames}"
        )
    return int(purged_start), int(purged_end)


def build_sample_indices(
    total_frames: int,
    start_frame: int,
    end_frame: int,
    window: WindowSpec,
    randomize: bool,
    seed: int | None = None,
) -> np.ndarray:
    resolved_end = resolve_end_frame(total_frames, end_frame)
    start_sample = max(start_frame, window.pre_frame + window.pre_post_omission)
    end_sample = min(
        resolved_end,
        total_frames - 1 - window.post_frame - window.pre_post_omission,
    )
    if end_sample < start_sample:
        raise ValueError(
            "No valid samples after applying temporal window: "
            f"start={start_sample}, end={end_sample}, total_frames={total_frames}"
        )

    indices = np.arange(start_sample, end_sample + 1, dtype=np.int64)
    if randomize:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    return indices
