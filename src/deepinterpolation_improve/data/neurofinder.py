from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from deepinterpolation_improve.config import WindowSpec

from .windowing import build_sample_indices, input_offsets


NEUROFINDER_BASE_URL = "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder"
NEUROFINDER_SOURCE_REFERENCE = "https://github.com/codeneuro/neurofinder/blob/master/README.md#datasets"


class UnsafeArchiveError(RuntimeError):
    pass


def neurofinder_url(dataset_id: str) -> str:
    return f"{NEUROFINDER_BASE_URL}/neurofinder.{dataset_id}.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_download(url_or_path: str | Path, destination: Path, retries: int = 3) -> dict[str, Any]:
    source = Path(url_or_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, destination)
        return {
            "url": str(source),
            "http_status": 200,
            "content_length": destination.stat().st_size,
            "content_type": "application/zip",
            "reused_existing": False,
            "resumed_from": 0,
        }
    if destination.exists() and zipfile.is_zipfile(destination):
        return {
            "url": str(url_or_path),
            "http_status": 200,
            "content_length": destination.stat().st_size,
            "content_type": "application/zip",
            "reused_existing": True,
            "resumed_from": 0,
        }

    url = str(url_or_path)
    last_error: BaseException | None = None
    resumed_from = 0
    for attempt in range(1, int(retries) + 1):
        existing_size = destination.stat().st_size if destination.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            resumed_from = existing_size
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = int(getattr(response, "status", 200))
                append = existing_size > 0 and status == 206
                with destination.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle)
                return {
                    "url": url,
                    "http_status": status,
                    "content_length": destination.stat().st_size,
                    "content_type": response.headers.get("Content-Type"),
                    "reused_existing": False,
                    "resumed_from": resumed_from if append else 0,
                }
        except BaseException as exc:
            last_error = exc
            if attempt >= int(retries):
                break
            time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise UnsafeArchiveError(f"Unsafe archive member path: {member.filename}")
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination_resolved)):
                raise UnsafeArchiveError(f"Unsafe archive target path: {member.filename}")
        archive.extractall(destination)


def download_neurofinder_datasets(
    dataset_ids: Iterable[str],
    raw_dir: str | Path,
    processed_dir: str | Path,
    url_map: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    url_map = url_map or {}

    entries: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        archive_path = raw_dir / f"neurofinder.{dataset_id}.zip"
        source = url_map.get(dataset_id, neurofinder_url(dataset_id))
        download_info = _copy_or_download(source, archive_path)
        _safe_extract(archive_path, processed_dir)
        extracted_root = processed_dir / f"neurofinder.{dataset_id}"
        entry = {
            "dataset_id": dataset_id,
            "url": download_info["url"],
            "http_status": download_info["http_status"],
            "content_length": download_info["content_length"],
            "content_type": download_info.get("content_type"),
            "sha256": _sha256(archive_path),
            "archive_path": str(archive_path),
            "extract_path": str(extracted_root),
            "source_reference": NEUROFINDER_SOURCE_REFERENCE,
        }
        entries.append(entry)

    manifest = {"schema_version": 1, "source": "neurofinder", "datasets": entries}
    (processed_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def find_neurofinder_image_paths(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    candidates = sorted((root / "images").glob("*.tif*"))
    if not candidates:
        candidates = sorted(root.rglob("*.tif*"))
    if not candidates:
        raise FileNotFoundError(f"No TIFF frames found under {root}")
    return sorted(candidates, key=_numeric_sort_key)


def _numeric_sort_key(path: Path) -> tuple[str, int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (path.parent.as_posix(), int(digits) if digits else -1, path.name)


def find_neurofinder_regions_path(dataset_root: str | Path) -> Path | None:
    root = Path(dataset_root)
    for candidate in [root / "regions" / "regions.json", root / "regions.json"]:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob("regions.json"))
    return matches[0] if matches else None


def load_neurofinder_regions(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = find_neurofinder_regions_path(dataset_root)
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    if isinstance(content, dict) and "regions" in content:
        content = content["regions"]
    if not isinstance(content, list):
        raise ValueError(f"Unsupported Neurofinder regions format in {path}")
    return content


def regions_to_masks(regions: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    masks = np.zeros((len(regions), shape[0], shape[1]), dtype=bool)
    for roi_index, region in enumerate(regions):
        coordinates = region.get("coordinates", [])
        for coordinate in coordinates:
            if len(coordinate) != 2:
                continue
            row, col = int(coordinate[0]), int(coordinate[1])
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                masks[roi_index, row, col] = True
    return masks


class NeurofinderDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        window: WindowSpec,
        start_frame: int = 0,
        end_frame: int = -1,
        randomize: bool = False,
        seed: int | None = None,
        normalize: bool = True,
        mean: float | None = None,
        std: float | None = None,
    ):
        import tifffile

        self.dataset_root = Path(dataset_root)
        self.window = window
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.randomize = bool(randomize)
        self.seed = seed
        self.normalize = normalize
        self.frame_paths = find_neurofinder_image_paths(self.dataset_root)
        first_frame = tifffile.imread(self.frame_paths[0])
        if first_frame.ndim != 2:
            raise ValueError("NeurofinderDataset expects 2D TIFF frames.")
        self.frame_shape = tuple(int(value) for value in first_frame.shape)
        self.sample_indices = build_sample_indices(
            total_frames=len(self.frame_paths),
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            window=self.window,
            randomize=self.randomize,
            seed=self.seed,
        )
        self.regions = load_neurofinder_regions(self.dataset_root)
        self._roi_masks: np.ndarray | None = None

        sample_count = min(len(self.frame_paths), 100)
        sample = np.stack([tifffile.imread(path).astype(np.float32) for path in self.frame_paths[:sample_count]])
        self.local_mean = float(np.mean(sample) if mean is None else mean)
        local_std = float(np.std(sample) if std is None else std)
        self.local_std = local_std if local_std > 0 else 1.0

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (len(input_offsets(self.window)), *self.frame_shape)

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (1, *self.frame_shape)

    @property
    def roi_count(self) -> int:
        return len(self.regions)

    @property
    def roi_masks(self) -> np.ndarray:
        if self._roi_masks is None:
            self._roi_masks = regions_to_masks(self.regions, self.frame_shape)
        return self._roi_masks

    def normalization_dict(self) -> dict[str, float]:
        return {"mean": self.local_mean, "std": self.local_std}

    def ordered_copy(self) -> "NeurofinderDataset":
        copied = self.__class__.__new__(self.__class__)
        copied.__dict__ = dict(self.__dict__)
        copied.randomize = False
        copied.sample_indices = np.array(sorted(self.sample_indices.tolist()), dtype=np.int64)
        return copied

    def _read_frame(self, frame_index: int) -> np.ndarray:
        import tifffile

        return tifffile.imread(self.frame_paths[int(frame_index)])

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
