from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

from deepinterpolation_improve.experiments.paper_evidence_v1.data import (
    NeurofinderCentreFrameDataset,
    Normalisation,
    find_frames,
)
from deepinterpolation_improve.experiments.paper_evidence_v1.study import expand_tasks


def test_quick_screen_is_eight_single_seed_tasks_with_locked_splits() -> None:
    root = Path(__file__).resolve().parents[2]
    tasks = expand_tasks(
        root / "configs" / "exploratory" / "quick_screen.json",
        validate_paths=False,
    )
    assert len(tasks) == 8
    assert {task.seed for task in tasks} == {0}
    assert {task.recording for task in tasks} == {"01.00"}
    assert {task.window for task in tasks} == {"w2"}
    assert {task.input_channels for task in tasks} == {4}


def test_dataset_uses_neighbours_but_not_centre_frame() -> None:
    with TemporaryDirectory(prefix="dip_structural_test_") as temporary:
        root = Path(temporary)
        images = root / "images"
        images.mkdir()
        for index in range(12):
            tifffile.imwrite(images / f"image{index:05d}.tiff", np.full((8, 8), index, dtype=np.float32))
        split = {
            "input_offsets": [-2, -1, 1, 2],
            "purged_splits": {"train": [2, 4], "val": [5, 7], "test": [8, 9]},
        }
        split_path = root / "split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        dataset = NeurofinderCentreFrameDataset(root, split_path, "train", Normalisation(0.0, 1.0, 12))
        sample = dataset[0]
        assert sample["frame_index"] == 2
        assert sample["target"].unique().item() == 2.0
        assert sample["input"][:, 0, 0].tolist() == [0.0, 1.0, 3.0, 4.0]
        assert len(find_frames(root)) == 12
