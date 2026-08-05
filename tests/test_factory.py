import h5py
import json
import nibabel as nib
import numpy as np
import tifffile

from deepinterpolation_improve.config import DatasetConfig, WindowSpec
from deepinterpolation_improve.data import (
    FmriVolumeDataset,
    MovieJSONDataset,
    MultiTifDataset,
    OphysH5Dataset,
    SingleTifDataset,
    create_dataset,
)


def test_create_dataset_supports_ophys_h5(tmp_path):
    h5_path = tmp_path / "movie.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("data", data=np.arange(10 * 4 * 5, dtype=np.float32).reshape(10, 4, 5))

    dataset = create_dataset(
        DatasetConfig(
            kind="ophys_h5",
            path=h5_path,
            window=WindowSpec(pre_frame=1, post_frame=1),
            batch_size=4,
        )
    )

    assert isinstance(dataset, OphysH5Dataset)
    assert len(dataset) == 8


def test_create_dataset_supports_single_and_multi_tif(tmp_path):
    single_path = tmp_path / "movie.tif"
    tifffile.imwrite(single_path, np.ones((10, 4, 5), dtype=np.float32))
    multi_dir = tmp_path / "multi"
    multi_dir.mkdir()
    tifffile.imwrite(multi_dir / "a.tif", np.ones((5, 4, 5), dtype=np.float32))
    tifffile.imwrite(multi_dir / "b.tif", np.ones((5, 4, 5), dtype=np.float32))

    single = create_dataset(
        DatasetConfig(kind="single_tif", path=single_path, window=WindowSpec(1, 1))
    )
    multi = create_dataset(
        DatasetConfig(kind="multi_tif", path=multi_dir, window=WindowSpec(1, 1))
    )

    assert isinstance(single, SingleTifDataset)
    assert isinstance(multi, MultiTifDataset)


def test_create_dataset_supports_fmri_volume(tmp_path):
    nii_path = tmp_path / "volume.nii.gz"
    data = np.arange(3 * 4 * 5 * 8, dtype=np.float32).reshape(3, 4, 5, 8)
    nib.Nifti1Image(data, affine=np.eye(4)).to_filename(nii_path)

    dataset = create_dataset(
        DatasetConfig(kind="fmri_volume", path=nii_path, window=WindowSpec(1, 1))
    )
    sample = dataset[0]

    assert isinstance(dataset, FmriVolumeDataset)
    assert len(dataset) == 6
    assert dataset.input_shape == (2, 3, 4, 5)
    assert sample["input"].shape == (2, 3, 4, 5)
    assert sample["target"].shape == (1, 3, 4, 5)


def test_create_dataset_supports_movie_json_with_explicit_candidates(tmp_path):
    movie_dir = tmp_path / "movie"
    movie_dir.mkdir()
    h5_path = movie_dir / "custom_movie.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("frames", data=np.arange(10 * 4 * 5, dtype=np.float32).reshape(10, 4, 5))
    json_path = tmp_path / "movies.json"
    json_path.write_text(
        json.dumps({"a": {"path": str(movie_dir), "frames": [3, 4], "mean": 0.0, "std": 1.0}}),
        encoding="utf-8",
    )

    dataset = create_dataset(
        DatasetConfig(
            kind="movie_json",
            path=json_path,
            window=WindowSpec(1, 1),
            dataset_key="frames",
            candidate_filenames=["custom_movie.h5"],
        )
    )

    assert isinstance(dataset, MovieJSONDataset)
    assert len(dataset) == 2
    assert dataset.input_shape == (2, 4, 5)
    assert dataset.target_shape == (1, 4, 5)
    assert dataset.normalization_dict() == {"mean": 0.0, "std": 1.0}
    assert dataset.ordered_copy().records[0]["frame"] == 3
