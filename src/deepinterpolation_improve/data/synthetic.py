from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from deepinterpolation_improve.config import WindowSpec
from .windowing import build_sample_indices, input_offsets


def _normalize_unit_interval(movie: np.ndarray) -> np.ndarray:
    movie = np.asarray(movie, dtype=np.float32)
    min_value = float(np.min(movie))
    max_value = float(np.max(movie))
    if max_value <= min_value:
        return np.zeros_like(movie, dtype=np.float32)
    return ((movie - min_value) / (max_value - min_value)).astype(np.float32)


def add_poisson_gaussian_noise(
    clean: np.ndarray,
    seed: int = 0,
    poisson_peak: float = 32.0,
    gaussian_std: float = 0.03,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean = _normalize_unit_interval(clean)
    rng = np.random.default_rng(seed)
    if poisson_peak > 0:
        noisy = rng.poisson(np.clip(clean, 0.0, None) * float(poisson_peak)).astype(np.float32) / float(poisson_peak)
    else:
        noisy = np.array(clean, dtype=np.float32)
    if gaussian_std > 0:
        noisy = noisy + rng.normal(0.0, float(gaussian_std), size=clean.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    metadata = {
        "target_kind": "synthetic_clean_reference",
        "noise_model": "poisson_gaussian",
        "seed": int(seed),
        "poisson_peak": float(poisson_peak),
        "gaussian_std": float(gaussian_std),
    }
    return noisy, metadata


def make_synthetic_clean_reference_from_movie(
    clean_movie: np.ndarray,
    seed: int = 0,
    poisson_peak: float = 32.0,
    gaussian_std: float = 0.03,
) -> dict[str, Any]:
    clean = _normalize_unit_interval(clean_movie)
    noisy, metadata = add_poisson_gaussian_noise(
        clean,
        seed=seed,
        poisson_peak=poisson_peak,
        gaussian_std=gaussian_std,
    )
    metadata["source"] = "provided_clean_movie"
    return {"clean": clean, "noisy": noisy, "metadata": metadata}


def generate_synthetic_calcium_movie(
    frames: int = 64,
    height: int = 64,
    width: int = 64,
    cells: int = 8,
    seed: int = 0,
    poisson_peak: float = 32.0,
    gaussian_std: float = 0.03,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    clean = np.zeros((frames, height, width), dtype=np.float32)
    for _ in range(int(cells)):
        center_y = rng.uniform(0.15 * height, 0.85 * height)
        center_x = rng.uniform(0.15 * width, 0.85 * width)
        radius = rng.uniform(max(1.0, min(height, width) * 0.04), max(1.5, min(height, width) * 0.12))
        footprint = np.exp(-(((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2) / (2.0 * radius**2)))
        spikes = rng.random(frames) < rng.uniform(0.04, 0.12)
        trace = np.zeros(frames, dtype=np.float32)
        decay = rng.uniform(0.75, 0.92)
        for index in range(frames):
            trace[index] = (trace[index - 1] * decay if index else 0.0) + spikes[index] * rng.uniform(0.5, 1.5)
        clean += trace[:, None, None] * footprint[None, :, :].astype(np.float32)
    clean = _normalize_unit_interval(clean)
    noisy, metadata = add_poisson_gaussian_noise(
        clean,
        seed=seed,
        poisson_peak=poisson_peak,
        gaussian_std=gaussian_std,
    )
    metadata.update(
        {
            "source": "simulated_calcium_movie",
            "frames": int(frames),
            "height": int(height),
            "width": int(width),
            "cells": int(cells),
        }
    )
    return {"clean": clean, "noisy": noisy, "metadata": metadata}


class SyntheticCleanReferenceDataset(Dataset):
    target_kind = "synthetic_clean_reference"

    def __init__(
        self,
        clean_data: np.ndarray,
        noisy_data: np.ndarray,
        window: WindowSpec,
        start_frame: int = 0,
        end_frame: int = -1,
        randomize: bool = False,
        seed: int | None = None,
        normalize: bool = True,
        noise_metadata: dict[str, Any] | None = None,
        source_path: str | Path | None = None,
    ):
        clean = np.asarray(clean_data, dtype=np.float32)
        noisy = np.asarray(noisy_data, dtype=np.float32)
        if clean.shape != noisy.shape or clean.ndim != 3:
            raise ValueError("SyntheticCleanReferenceDataset expects matching [time, height, width] clean/noisy arrays.")
        self.clean_data = clean
        self.noisy_data = noisy
        self.data = noisy
        self.window = window
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.randomize = bool(randomize)
        self.seed = seed
        self.normalize = bool(normalize)
        self.path = Path(source_path) if source_path is not None else Path("synthetic_clean_reference")
        self.noise_metadata = dict(noise_metadata or {})
        self.noise_metadata.setdefault("target_kind", self.target_kind)
        self.noise_metadata.setdefault("seed", seed)
        self.sample_indices = build_sample_indices(
            total_frames=self.noisy_data.shape[0],
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            window=window,
            randomize=self.randomize,
            seed=seed,
        )
        local_data = self.noisy_data[: min(self.noisy_data.shape[0], 1000)]
        self.local_mean = float(np.mean(local_data))
        local_std = float(np.std(local_data))
        self.local_std = local_std if local_std > 0 else 1.0

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (len(input_offsets(self.window)), self.noisy_data.shape[1], self.noisy_data.shape[2])

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (1, self.clean_data.shape[1], self.clean_data.shape[2])

    def normalization_dict(self) -> dict[str, float]:
        return {"mean": self.local_mean, "std": self.local_std}

    def ordered_copy(self) -> "SyntheticCleanReferenceDataset":
        copied = self.__class__.__new__(self.__class__)
        copied.__dict__ = dict(self.__dict__)
        copied.randomize = False
        copied.sample_indices = np.array(sorted(self.sample_indices.tolist()), dtype=np.int64)
        return copied

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        if not self.normalize:
            return array
        return (array - self.local_mean) / self.local_std

    def __getitem__(self, item: int) -> dict[str, Any]:
        frame_index = int(self.sample_indices[item])
        frames = np.stack([self.noisy_data[frame_index + offset] for offset in input_offsets(self.window)])
        target = self.clean_data[frame_index][None, :, :]
        return {
            "input": torch.from_numpy(self._normalize(frames)),
            "target": torch.from_numpy(self._normalize(target)),
            "frame_index": frame_index,
        }
