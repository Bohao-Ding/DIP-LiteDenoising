import h5py
import json
import numpy as np

from deepinterpolation_improve.config import WindowSpec
from deepinterpolation_improve.data import ArrayTemporalDataset, MovieJSONDataset
from deepinterpolation_improve.inference import InferenceRunner
from deepinterpolation_improve.models import create_model


def test_inference_writes_sorted_frame_indices_even_when_dataset_is_randomized(tmp_path, make_movie):
    dataset = ArrayTemporalDataset(
        make_movie(frames=10, height=8, width=7),
        window=WindowSpec(pre_frame=1, post_frame=1),
        start_frame=0,
        end_frame=-1,
        randomize=True,
        seed=13,
        normalize=True,
    )
    model = create_model("unet_single_1024", {"input_channels": 2, "base_features": 2, "depth": 2})
    output_file = tmp_path / "ordered.h5"

    InferenceRunner(model, dataset, output_file, batch_size=3, device="cpu").run()

    with h5py.File(output_file, "r") as handle:
        assert handle["frame_index"][:].tolist() == sorted(dataset.sample_indices.tolist())


def test_movie_json_inference_also_forces_order(tmp_path):
    movie_dir = tmp_path / "movie"
    movie_dir.mkdir()
    with h5py.File(movie_dir / "custom.h5", "w") as handle:
        handle.create_dataset("data", data=np.arange(10 * 4 * 5, dtype=np.float32).reshape(10, 4, 5))
    json_path = tmp_path / "movies.json"
    json_path.write_text(
        json.dumps({"a": {"path": str(movie_dir), "frames": [3, 4, 5, 6], "mean": 0.0, "std": 1.0}}),
        encoding="utf-8",
    )
    dataset = MovieJSONDataset(
        json_path,
        window=WindowSpec(pre_frame=1, post_frame=1),
        candidate_filenames=["custom.h5"],
        randomize=True,
        seed=7,
    )
    model = create_model("unet_single_1024", {"input_channels": 2, "base_features": 2, "depth": 2})
    output_file = tmp_path / "movie_json.h5"

    InferenceRunner(model, dataset, output_file, batch_size=2, device="cpu").run()

    with h5py.File(output_file, "r") as handle:
        assert handle["frame_index"][:].tolist() == [3, 4, 5, 6]
