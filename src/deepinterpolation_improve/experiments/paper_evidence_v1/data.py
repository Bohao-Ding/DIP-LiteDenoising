from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _numeric_key(path: Path) -> tuple[str, int, str]:
    digits = re.findall(r"\d+", path.stem)
    return path.parent.as_posix(), int(digits[-1]) if digits else -1, path.name


def find_frames(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    frames = list((root / "images").glob("*.tif*"))
    if not frames:
        frames = list(root.rglob("*.tif*"))
    frames = sorted(frames, key=_numeric_key)
    if not frames:
        raise FileNotFoundError(f"No TIFF frames found under {root}")
    return frames


@dataclass(frozen=True)
class Normalisation:
    mean: float
    std: float
    source_frame_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "std": self.std, "source_frame_count": self.source_frame_count}


def recording_normalisation(frame_paths: list[Path], source_frame_count: int = 100) -> Normalisation:
    import tifffile

    count = min(len(frame_paths), int(source_frame_count))
    sample = np.stack([tifffile.imread(path).astype(np.float32) for path in frame_paths[:count]])
    mean = float(sample.mean())
    std = float(sample.std())
    return Normalisation(mean, std if std > 0 else 1.0, count)


def read_split_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    if 0 in [int(value) for value in split["input_offsets"]]:
        raise ValueError("The centre frame must not be included in model inputs")
    return split


class NeurofinderCentreFrameDataset(Dataset):
    """Read-only adapter for locked Neurofinder recording splits."""

    def __init__(
        self,
        dataset_root: str | Path,
        split_manifest: str | Path,
        split_name: str,
        normalisation: Normalisation,
    ):
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split_name}")
        self.frame_paths = find_frames(dataset_root)
        self.manifest = read_split_manifest(split_manifest)
        self.offsets = tuple(int(value) for value in self.manifest["input_offsets"])
        start, end = (int(value) for value in self.manifest["purged_splits"][split_name])
        if start < 0 or end >= len(self.frame_paths) or start > end:
            raise ValueError(f"Invalid {split_name} range [{start}, {end}]")
        self.centre_indices = tuple(range(start, end + 1))
        self.normalisation = normalisation
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.centre_indices)

    def _read(self, index: int) -> np.ndarray:
        import tifffile

        return tifffile.imread(self.frame_paths[index]).astype(np.float32)

    def _normalise(self, value: np.ndarray) -> np.ndarray:
        return (value - self.normalisation.mean) / self.normalisation.std

    def __getitem__(self, item: int) -> dict[str, Any]:
        centre = self.centre_indices[item]
        neighbours = np.stack([self._read(centre + offset) for offset in self.offsets])
        target = self._read(centre)[None]
        return {
            "input": torch.from_numpy(self._normalise(neighbours)),
            "target": torch.from_numpy(self._normalise(target)),
            "frame_index": centre,
        }

    @property
    def input_shape(self) -> tuple[int, int, int]:
        import tifffile

        shape = tifffile.imread(self.frame_paths[0]).shape
        return len(self.offsets), int(shape[0]), int(shape[1])


def build_recording_datasets(dataset_root: str | Path, split_manifest: str | Path):
    frames = find_frames(dataset_root)
    normalisation = recording_normalisation(frames)
    datasets = {
        name: NeurofinderCentreFrameDataset(
            dataset_root=dataset_root,
            split_manifest=split_manifest,
            split_name=name,
            normalisation=normalisation,
        )
        for name in ("train", "val", "test")
    }
    return datasets, normalisation
