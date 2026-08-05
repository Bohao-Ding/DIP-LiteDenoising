from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from deepinterpolation_improve.config import WindowSpec
from .windowing import build_sample_indices, input_offsets


class ArrayTemporalDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        window: WindowSpec,
        start_frame: int = 0,
        end_frame: int = -1,
        randomize: bool = False,
        seed: int | None = None,
        normalize: bool = True,
        mean: float | None = None,
        std: float | None = None,
    ):
        if data.ndim != 3:
            raise ValueError("ArrayTemporalDataset expects data shaped [time, height, width].")
        self.data = np.asarray(data)
        self.window = window
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.randomize = bool(randomize)
        self.seed = seed
        self.normalize = normalize
        self.sample_indices = build_sample_indices(
            total_frames=self.data.shape[0],
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            window=window,
            randomize=self.randomize,
            seed=seed,
        )
        local_data = self.data[: min(self.data.shape[0], 1000)].astype(np.float32)
        self.local_mean = float(np.mean(local_data) if mean is None else mean)
        local_std = float(np.std(local_data) if std is None else std)
        self.local_std = local_std if local_std > 0 else 1.0

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (len(input_offsets(self.window)), self.data.shape[1], self.data.shape[2])

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (1, self.data.shape[1], self.data.shape[2])

    def normalization_dict(self) -> dict[str, float]:
        return {"mean": self.local_mean, "std": self.local_std}

    def ordered_copy(self) -> "ArrayTemporalDataset":
        copied = self.__class__.__new__(self.__class__)
        copied.__dict__ = dict(self.__dict__)
        copied.randomize = False
        copied.sample_indices = np.array(sorted(self.sample_indices.tolist()), dtype=np.int64)
        return copied

    def _read_frame(self, frame_index: int) -> np.ndarray:
        return self.data[int(frame_index)]

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        if not self.normalize:
            return array
        return (array - self.local_mean) / self.local_std

    def __getitem__(self, item: int) -> dict[str, Any]:
        frame_index = int(self.sample_indices[item])
        frames = np.stack([self._read_frame(frame_index + offset) for offset in input_offsets(self.window)])
        target = self._read_frame(frame_index)[None, :, :]
        return {
            "input": torch.from_numpy(self._normalize(frames)),
            "target": torch.from_numpy(self._normalize(target)),
            "frame_index": frame_index,
        }


class SingleTifDataset(ArrayTemporalDataset):
    def __init__(self, path: str | Path, *args: Any, **kwargs: Any):
        import tifffile

        self.path = Path(path)
        data = tifffile.imread(self.path)
        super().__init__(data, *args, **kwargs)


class OphysH5Dataset(ArrayTemporalDataset):
    def __init__(self, path: str | Path, dataset_key: str = "data", *args: Any, **kwargs: Any):
        self.path = Path(path)
        self.dataset_key = dataset_key
        with h5py.File(self.path, "r") as handle:
            data = handle[dataset_key][:]
        super().__init__(data, *args, **kwargs)


class MultiTifDataset(ArrayTemporalDataset):
    def __init__(self, folder: str | Path, *args: Any, **kwargs: Any):
        import tifffile

        self.folder = Path(folder)
        paths = sorted(self.folder.glob("*.tif"))
        if not paths:
            raise FileNotFoundError(f"No .tif files found in {self.folder}")
        data = np.concatenate([tifffile.imread(path) for path in paths], axis=0)
        super().__init__(data, *args, **kwargs)


class EphysDataset(ArrayTemporalDataset):
    nb_probes = 384

    def __init__(self, path: str | Path, *args: Any, **kwargs: Any):
        self.path = Path(path)
        raw = np.memmap(self.path, dtype="int16")
        total_frames = int(raw.size / self.nb_probes)
        self.raw_data = np.memmap(self.path, dtype="int16", shape=(total_frames, self.nb_probes // 2, 2))
        mapped = np.zeros((total_frames, self.nb_probes, 2), dtype=np.float32)
        mapped[:, 0::2, 0] = self.raw_data[:, :, 0]
        mapped[:, 1::2, 1] = self.raw_data[:, :, 1]
        super().__init__(mapped, *args, **kwargs)


class FmriVolumeDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        window: WindowSpec,
        start_frame: int = 0,
        end_frame: int = -1,
        randomize: bool = False,
        seed: int | None = None,
        normalize: bool = True,
    ):
        import nibabel as nib

        self.path = Path(path)
        self.data = np.asarray(nib.load(self.path).get_fdata(), dtype=np.float32)
        if self.data.ndim != 4:
            raise ValueError("FmriVolumeDataset expects data shaped [x, y, z, time].")
        self.window = window
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.randomize = randomize
        self.normalize = normalize
        self.sample_indices = build_sample_indices(
            self.data.shape[3], start_frame, end_frame, window, randomize, seed
        )
        self.local_mean = float(np.mean(self.data))
        std = float(np.std(self.data))
        self.local_std = std if std > 0 else 1.0

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (len(input_offsets(self.window)), *self.data.shape[:3])

    @property
    def target_shape(self) -> tuple[int, int, int, int]:
        return (1, *self.data.shape[:3])

    def ordered_copy(self) -> "FmriVolumeDataset":
        copied = self.__class__.__new__(self.__class__)
        copied.__dict__ = dict(self.__dict__)
        copied.randomize = False
        copied.sample_indices = np.array(sorted(self.sample_indices.tolist()), dtype=np.int64)
        return copied

    def normalization_dict(self) -> dict[str, float]:
        return {"mean": self.local_mean, "std": self.local_std}

    def __getitem__(self, item: int) -> dict[str, Any]:
        frame_index = int(self.sample_indices[item])
        frames = np.stack(
            [self.data[:, :, :, frame_index + offset] for offset in input_offsets(self.window)],
            axis=0,
        )
        target = self.data[:, :, :, frame_index][None, :, :, :]
        if self.normalize:
            frames = (frames - self.local_mean) / self.local_std
            target = (target - self.local_mean) / self.local_std
        return {
            "input": torch.from_numpy(frames.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "frame_index": frame_index,
        }


class MovieJSONDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        window: WindowSpec,
        candidate_filenames: Iterable[str],
        dataset_key: str = "data",
        batch_size: int = 1,
        randomize: bool = False,
        seed: int | None = None,
    ):
        self.json_path = Path(json_path)
        self.window = window
        self.dataset_key = dataset_key
        self.batch_size = batch_size
        self.randomize = randomize
        with self.json_path.open("r", encoding="utf-8") as handle:
            self.frame_data_location = json.load(handle)
        self.records: list[dict[str, Any]] = []
        self._frame_shape: tuple[int, int] | None = None
        for lims_id, record in self.frame_data_location.items():
            movie_path = self._resolve_movie_path(Path(record["path"]), candidate_filenames)
            if self._frame_shape is None:
                with h5py.File(movie_path, "r") as handle:
                    self._frame_shape = tuple(handle[self.dataset_key].shape[1:3])
            for frame in record["frames"]:
                self.records.append(
                    {
                        "lims_id": lims_id,
                        "movie_path": movie_path,
                        "frame": int(frame),
                        "mean": float(record["mean"]),
                        "std": float(record["std"]) or 1.0,
                    }
                )
        if randomize:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.records)
        if self._frame_shape is None:
            raise ValueError("MovieJSONDataset has no records.")

    def _resolve_movie_path(self, base_path: Path, candidate_filenames: Iterable[str]) -> Path:
        for filename in candidate_filenames:
            path = base_path / filename
            if path.exists() and not path.is_symlink():
                return path
        raise FileNotFoundError(f"No candidate movie found under {base_path}")

    def __len__(self) -> int:
        return len(self.records)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (len(input_offsets(self.window)), *self._frame_shape)

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (1, *self._frame_shape)

    def normalization_dict(self) -> dict[str, float]:
        first = self.records[0]
        return {"mean": float(first["mean"]), "std": float(first["std"])}

    def ordered_copy(self) -> "MovieJSONDataset":
        copied = self.__class__.__new__(self.__class__)
        copied.__dict__ = dict(self.__dict__)
        copied.randomize = False
        copied.records = sorted(
            self.records,
            key=lambda item: (str(item["movie_path"]), int(item["frame"])),
        )
        return copied

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        with h5py.File(record["movie_path"], "r") as handle:
            data = handle[self.dataset_key]
            frames = np.stack([data[record["frame"] + offset] for offset in input_offsets(self.window)])
            target = data[record["frame"]][None, :, :]
        frames = (frames.astype(np.float32) - record["mean"]) / record["std"]
        target = (target.astype(np.float32) - record["mean"]) / record["std"]
        return {
            "input": torch.from_numpy(frames),
            "target": torch.from_numpy(target),
            "frame_index": record["frame"],
        }
