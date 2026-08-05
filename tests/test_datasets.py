import numpy as np
import torch
from torch.utils.data import DataLoader

from deepinterpolation_improve.config import WindowSpec
from deepinterpolation_improve.data import ArrayTemporalDataset


def test_array_dataset_excludes_target_window_and_keeps_frame_index(make_movie):
    movie = make_movie(frames=13, height=4, width=5)
    dataset = ArrayTemporalDataset(
        movie,
        window=WindowSpec(pre_frame=2, post_frame=2, pre_post_omission=1),
        start_frame=0,
        end_frame=-1,
        randomize=False,
        normalize=False,
    )

    sample = dataset[0]

    assert sample["frame_index"] == 3
    assert sample["input"].shape == (4, 4, 5)
    assert sample["target"].shape == (1, 4, 5)
    assert torch.equal(sample["input"][0], torch.as_tensor(movie[0]))
    assert torch.equal(sample["input"][1], torch.as_tensor(movie[1]))
    assert torch.equal(sample["input"][2], torch.as_tensor(movie[5]))
    assert torch.equal(sample["input"][3], torch.as_tensor(movie[6]))
    assert torch.equal(sample["target"][0], torch.as_tensor(movie[3]))


def test_dataloader_last_batch_is_not_zero_padded(make_movie):
    movie = make_movie(frames=13, height=4, width=5)
    dataset = ArrayTemporalDataset(
        movie,
        window=WindowSpec(pre_frame=2, post_frame=2, pre_post_omission=1),
        start_frame=0,
        end_frame=-1,
        randomize=False,
        normalize=False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batches = list(loader)

    assert [batch["input"].shape[0] for batch in batches] == [4, 3]
    assert batches[-1]["frame_index"].tolist() == [7, 8, 9]


def test_ordered_copy_restores_inference_order(make_movie):
    movie = make_movie(frames=13, height=4, width=5)
    randomized = ArrayTemporalDataset(
        movie,
        window=WindowSpec(pre_frame=1, post_frame=1, pre_post_omission=0),
        start_frame=0,
        end_frame=-1,
        randomize=True,
        seed=7,
        normalize=False,
    )

    ordered = randomized.ordered_copy()

    assert randomized.sample_indices.tolist() != ordered.sample_indices.tolist()
    assert ordered.sample_indices.tolist() == sorted(randomized.sample_indices.tolist())
