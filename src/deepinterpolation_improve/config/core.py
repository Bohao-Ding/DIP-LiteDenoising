from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WindowSpec:
    pre_frame: int
    post_frame: int
    pre_post_omission: int = 0

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> "WindowSpec":
        if "pre_post_frame" in params:
            pre_frame = int(params["pre_post_frame"])
            post_frame = int(params["pre_post_frame"])
        else:
            pre_frame = int(params["pre_frame"])
            post_frame = int(params["post_frame"])
        return cls(
            pre_frame=pre_frame,
            post_frame=post_frame,
            pre_post_omission=int(params.get("pre_post_omission", 0)),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetConfig:
    kind: str
    path: Path
    window: WindowSpec
    start_frame: int = 0
    end_frame: int = -1
    randomize: bool = False
    seed: int | None = None
    batch_size: int = 1
    normalize: bool = True
    dataset_key: str = "data"
    candidate_filenames: list[str] | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str
    input_channels: int
    base_features: int = 16
    depth: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    run_uid: str
    model_name: str
    loss: str = "mean_absolute_error"
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 1e-4
    device: str = "cpu"
    seed: int = 0
    num_workers: int = 0
    scheduler_step_size: int | None = None
    scheduler_gamma: float = 0.5
    resume_from: Path | None = None
    mixed_precision: bool = True
    precision: str = "amp_fp16"
    pin_memory: bool = True
    non_blocking: bool = True
    channels_last: bool = False
    tf32: bool = False
    compile_model: bool = False
    cudnn_benchmark: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    shuffle_train: bool = True
    checkpoint_every: int = 1
    worker_seed: int | None = None
    progress_path: Path | None = None
    progress_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))
        if self.progress_path is not None:
            object.__setattr__(self, "progress_path", Path(self.progress_path))
        if self.precision not in {"amp_fp16", "amp_bf16", "fp32"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")


@dataclass(frozen=True)
class InferenceConfig:
    output_file: Path
    batch_size: int = 8
    device: str = "cpu"
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_file", Path(self.output_file))
        if self.manifest_path is not None:
            object.__setattr__(self, "manifest_path", Path(self.manifest_path))
