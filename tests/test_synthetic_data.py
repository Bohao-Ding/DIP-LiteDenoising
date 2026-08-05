import numpy as np

from deepinterpolation_improve.config import WindowSpec
from deepinterpolation_improve.data import SyntheticCleanReferenceDataset, generate_synthetic_calcium_movie
from deepinterpolation_improve.data.windowing import input_offsets


def test_synthetic_movie_generation_is_seed_reproducible_and_records_noise_metadata():
    first = generate_synthetic_calcium_movie(frames=8, height=6, width=5, seed=123)
    second = generate_synthetic_calcium_movie(frames=8, height=6, width=5, seed=123)

    assert np.array_equal(first["clean"], second["clean"])
    assert np.array_equal(first["noisy"], second["noisy"])
    assert first["metadata"]["target_kind"] == "synthetic_clean_reference"
    assert first["metadata"]["noise_model"] == "poisson_gaussian"
    assert not np.array_equal(first["clean"], first["noisy"])


def test_synthetic_clean_reference_dataset_uses_noisy_inputs_and_clean_targets():
    payload = generate_synthetic_calcium_movie(frames=7, height=4, width=3, seed=4, gaussian_std=0.0)
    window = WindowSpec(pre_frame=1, post_frame=1, pre_post_omission=0)
    dataset = SyntheticCleanReferenceDataset(
        payload["clean"],
        payload["noisy"],
        window=window,
        normalize=False,
        noise_metadata=payload["metadata"],
    )

    sample = dataset[0]
    frame_index = int(sample["frame_index"])

    assert 0 not in input_offsets(window)
    assert dataset.target_kind == "synthetic_clean_reference"
    assert dataset.noise_metadata["seed"] == 4
    assert np.array_equal(sample["input"].numpy()[0], payload["noisy"][frame_index - 1])
    assert np.array_equal(sample["target"].numpy()[0], payload["clean"][frame_index])


def test_synthetic_dataset_ordered_copy_preserves_clean_reference_target():
    payload = generate_synthetic_calcium_movie(frames=10, height=4, width=4, seed=5)
    dataset = SyntheticCleanReferenceDataset(
        payload["clean"],
        payload["noisy"],
        window=WindowSpec(pre_frame=1, post_frame=1, pre_post_omission=0),
        randomize=True,
        seed=9,
    )

    ordered = dataset.ordered_copy()

    assert ordered.target_kind == "synthetic_clean_reference"
    assert ordered.noise_metadata == dataset.noise_metadata
    assert ordered.sample_indices.tolist() == sorted(dataset.sample_indices.tolist())
