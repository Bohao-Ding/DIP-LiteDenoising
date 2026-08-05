import h5py
import torch

from deepinterpolation_improve.config import TrainingConfig, WindowSpec
from deepinterpolation_improve.data import (
    SyntheticCleanReferenceDataset,
    generate_synthetic_calcium_movie,
)
from deepinterpolation_improve.inference import InferenceRunner
from deepinterpolation_improve.models import create_model_variant
from deepinterpolation_improve.training import Trainer


def test_synthetic_training_and_ordered_inference(tmp_path):
    payload = generate_synthetic_calcium_movie(
        frames=20,
        height=8,
        width=8,
        cells=2,
        seed=17,
    )
    window = WindowSpec(pre_frame=1, post_frame=1)
    common = {
        "clean_data": payload["clean"],
        "noisy_data": payload["noisy"],
        "window": window,
        "normalize": True,
        "noise_metadata": payload["metadata"],
    }
    train_dataset = SyntheticCleanReferenceDataset(
        **common,
        start_frame=0,
        end_frame=11,
        randomize=True,
        seed=17,
    )
    validation_dataset = SyntheticCleanReferenceDataset(
        **common,
        start_frame=13,
        end_frame=-1,
        randomize=False,
        seed=18,
    )
    run_dir = tmp_path / "run"
    model = create_model_variant("bf4_d2_standard", input_channels=2)
    config = TrainingConfig(
        output_dir=run_dir,
        run_uid="core_test",
        model_name="bf4_d2_standard",
        loss="mean_absolute_error",
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        device="cpu",
        seed=17,
        num_workers=0,
        mixed_precision=False,
        precision="fp32",
        pin_memory=False,
        non_blocking=False,
    )

    Trainer(model, train_dataset, validation_dataset, config).fit()

    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "loss_history.csv").exists()

    inference_model = create_model_variant("bf4_d2_standard", input_channels=2)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    inference_model.load_state_dict(checkpoint["model_state"])
    output_path = tmp_path / "predictions.h5"
    InferenceRunner(
        inference_model,
        validation_dataset,
        output_path,
        batch_size=2,
        device="cpu",
        manifest_path=run_dir / "manifest.json",
    ).run()

    with h5py.File(output_path, "r") as handle:
        assert handle["data"].shape[0] == len(validation_dataset)
        assert handle["frame_index"][:].tolist() == sorted(
            validation_dataset.sample_indices.tolist()
        )
