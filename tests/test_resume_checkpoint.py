from deepinterpolation_improve.config import TrainingConfig, WindowSpec
from deepinterpolation_improve.data import ArrayTemporalDataset
from deepinterpolation_improve.models import create_model
from deepinterpolation_improve.training import Trainer


def _dataset(movie, randomize):
    return ArrayTemporalDataset(
        movie,
        window=WindowSpec(pre_frame=1, post_frame=1),
        start_frame=0,
        end_frame=-1,
        randomize=randomize,
        seed=21,
        normalize=True,
    )


def test_resume_checkpoint_continues_from_previous_epoch(tmp_path, make_movie):
    train_dataset = _dataset(make_movie(frames=9, height=8, width=7), randomize=True)
    val_dataset = _dataset(make_movie(frames=9, height=8, width=7), randomize=False)
    first_model = create_model("unet_single_1024", {"input_channels": 2, "base_features": 2, "depth": 2})
    first_config = TrainingConfig(
        output_dir=tmp_path,
        run_uid="resume",
        model_name="unet_single_1024",
        loss="mean_squared_error",
        batch_size=2,
        epochs=1,
        learning_rate=0.001,
        device="cpu",
        seed=21,
    )
    Trainer(first_model, train_dataset, val_dataset, first_config).fit()

    second_model = create_model("unet_single_1024", {"input_channels": 2, "base_features": 2, "depth": 2})
    second_config = TrainingConfig(
        output_dir=tmp_path,
        run_uid="resume",
        model_name="unet_single_1024",
        loss="mean_squared_error",
        batch_size=2,
        epochs=2,
        learning_rate=0.001,
        device="cpu",
        seed=21,
        resume_from=tmp_path / "checkpoint.pt",
    )

    result = Trainer(second_model, train_dataset, val_dataset, second_config).fit()

    assert result["epoch"] == 2
