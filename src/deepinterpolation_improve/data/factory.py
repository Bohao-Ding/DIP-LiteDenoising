from __future__ import annotations

from deepinterpolation_improve.config import DatasetConfig
from .datasets import EphysDataset, FmriVolumeDataset, MovieJSONDataset, MultiTifDataset, OphysH5Dataset, SingleTifDataset
from .neurofinder import NeurofinderDataset


def create_dataset(config: DatasetConfig):
    kwargs = {
        "window": config.window,
        "start_frame": config.start_frame,
        "end_frame": config.end_frame,
        "randomize": config.randomize,
        "seed": config.seed,
        "normalize": config.normalize,
    }
    if config.kind == "ephys":
        return EphysDataset(config.path, **kwargs)
    if config.kind == "single_tif":
        return SingleTifDataset(config.path, **kwargs)
    if config.kind == "multi_tif":
        return MultiTifDataset(config.path, **kwargs)
    if config.kind == "neurofinder":
        return NeurofinderDataset(config.path, **kwargs)
    if config.kind == "ophys_h5":
        return OphysH5Dataset(config.path, dataset_key=config.dataset_key, **kwargs)
    if config.kind == "fmri_volume":
        return FmriVolumeDataset(config.path, **kwargs)
    if config.kind == "movie_json":
        if not config.candidate_filenames:
            raise ValueError("movie_json datasets require candidate_filenames")
        return MovieJSONDataset(
            config.path,
            window=config.window,
            candidate_filenames=config.candidate_filenames,
            dataset_key=config.dataset_key,
            batch_size=config.batch_size,
            randomize=config.randomize,
            seed=config.seed,
        )
    raise ValueError(f"Unknown dataset kind: {config.kind}")
