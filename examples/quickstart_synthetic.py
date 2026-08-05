from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deepinterpolation_improve.config import TrainingConfig, WindowSpec
from deepinterpolation_improve.data import (
    SyntheticCleanReferenceDataset,
    generate_synthetic_calcium_movie,
)
from deepinterpolation_improve.inference import InferenceRunner
from deepinterpolation_improve.models import (
    create_model_variant,
    registered_variants,
)
from deepinterpolation_improve.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small synthetic training and inference check without downloaded data."
    )
    parser.add_argument("--variant", default="compact_di", choices=registered_variants())
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quickstart"))
    parser.add_argument("--list-variants", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def main() -> None:
    args = parse_args()
    if args.list_variants:
        print("\n".join(registered_variants()))
        return
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    device = resolve_device(args.device)
    payload = generate_synthetic_calcium_movie(
        frames=48,
        height=32,
        width=32,
        cells=6,
        seed=7,
    )
    window = WindowSpec(pre_frame=2, post_frame=2, pre_post_omission=0)
    dataset_kwargs = {
        "clean_data": payload["clean"],
        "noisy_data": payload["noisy"],
        "window": window,
        "normalize": True,
        "noise_metadata": payload["metadata"],
    }
    train_dataset = SyntheticCleanReferenceDataset(
        **dataset_kwargs,
        start_frame=0,
        end_frame=29,
        randomize=True,
        seed=7,
    )
    validation_dataset = SyntheticCleanReferenceDataset(
        **dataset_kwargs,
        start_frame=34,
        end_frame=-1,
        randomize=False,
        seed=8,
    )

    model = create_model_variant(args.variant, train_dataset.input_shape[0])
    training_dir = args.output_dir / "training"
    config = TrainingConfig(
        output_dir=training_dir,
        run_uid="synthetic_quickstart",
        model_name=args.variant,
        loss="mean_absolute_error",
        batch_size=4,
        epochs=args.epochs,
        learning_rate=1e-3,
        device=device,
        seed=7,
        num_workers=0,
        mixed_precision=device == "cuda",
        precision="amp_fp16" if device == "cuda" else "fp32",
        pin_memory=device == "cuda",
        non_blocking=device == "cuda",
    )
    history = Trainer(model, train_dataset, validation_dataset, config).fit()

    inference_model = create_model_variant(args.variant, validation_dataset.input_shape[0])
    checkpoint = torch.load(training_dir / "best.pt", map_location=device, weights_only=False)
    inference_model.load_state_dict(checkpoint["model_state"])
    prediction_path = InferenceRunner(
        inference_model,
        validation_dataset,
        args.output_dir / "predictions.h5",
        batch_size=4,
        device=device,
        manifest_path=training_dir / "manifest.json",
    ).run()

    parameter_count = sum(parameter.numel() for parameter in inference_model.parameters())
    print(f"variant={args.variant}")
    print(f"device={device}")
    print(f"parameters={parameter_count}")
    print(f"completed_epoch={history['epoch']}")
    print(f"checkpoint={training_dir / 'best.pt'}")
    print(f"predictions={prediction_path}")


if __name__ == "__main__":
    main()
