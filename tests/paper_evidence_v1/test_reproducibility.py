from __future__ import annotations

import csv
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from deepinterpolation_improve.experiments.paper_evidence_v1.engine import _new_generator, _restore_rng_state
from deepinterpolation_improve.experiments.paper_evidence_v1 import engine
from deepinterpolation_improve.experiments.paper_evidence_v1.study import ExperimentTask


def test_checkpoint_rng_restore_reproduces_global_and_loader_sequences() -> None:
    original_torch = torch.random.get_rng_state()
    original_numpy = np.random.get_state()
    original_python = random.getstate()
    try:
        torch.manual_seed(17)
        np.random.seed(17)
        random.seed(17)
        generators = {"train": _new_generator(17), "val": _new_generator(18), "test": _new_generator(19)}
        checkpoint = {
            "rng_state": {
                "torch": torch.random.get_rng_state(),
                "cuda": None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "loader_rng_state": {name: generator.get_state() for name, generator in generators.items()},
        }
        expected = {
            "torch": torch.rand(4),
            "numpy": np.random.random(4),
            "python": [random.random() for _ in range(4)],
            "train": torch.randperm(10, generator=generators["train"]),
            "val": torch.randperm(10, generator=generators["val"]),
            "test": torch.randperm(10, generator=generators["test"]),
        }
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        for generator in generators.values():
            generator.manual_seed(999)
        _restore_rng_state(checkpoint, generators)
        torch.testing.assert_close(torch.rand(4), expected["torch"])
        np.testing.assert_allclose(np.random.random(4), expected["numpy"])
        assert [random.random() for _ in range(4)] == expected["python"]
        for name in ("train", "val", "test"):
            assert torch.equal(torch.randperm(10, generator=generators[name]), expected[name])
    finally:
        torch.random.set_rng_state(original_torch)
        np.random.set_state(original_numpy)
        random.setstate(original_python)


def test_interrupted_training_resumes_from_a_complete_epoch(monkeypatch) -> None:
    import json
    import tifffile

    with TemporaryDirectory(prefix="dip_resume_contract_") as temporary:
        root = Path(temporary)
        images = root / "data" / "images"
        images.mkdir(parents=True)
        for index in range(20):
            tifffile.imwrite(images / f"frame_{index:03d}.tif", np.full((16, 16), index, dtype=np.float32))
        split = root / "split.json"
        split.write_text(
            json.dumps(
                {
                    "input_offsets": [-1, 1],
                    "total_frames": 20,
                    "purged_splits": {"train": [2, 5], "val": [7, 9], "test": [11, 13]},
                }
            ),
            encoding="utf-8",
        )
        task = ExperimentTask(
            task_id="resume-contract",
            stage="paper_evidence_core",
            recording="synthetic",
            window="w1",
            variant="paper_fullres_b8d3_mse",
            seed=0,
            dataset_root=str(root / "data"),
            split_manifest=str(split),
            output_dir=str(root / "run"),
            input_channels=2,
            frame_shape=(16, 16),
            training={
                "epochs": 2,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "scheduler_step_size": 1,
                "scheduler_gamma": 0.5,
                "checkpoint_every": 1,
                "num_workers": 0,
                "precision": "amp_fp16",
                "metric_precision": "fp32",
                "evaluation_split": "val",
                "channels_last": False,
                "device": "cpu",
            },
            benchmark={"enabled": False},
            source_config=str(split),
        )
        original = engine._run_epoch
        calls = 0

        def interrupt_on_second_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated interruption")
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, "_run_epoch", interrupt_on_second_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            engine.run_task(task)
        checkpoint = torch.load(Path(task.output_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1
        monkeypatch.setattr(engine, "_run_epoch", original)
        summary = engine.run_task(task, device_override="cpu", resume=True)
        assert summary["evaluation_split"] == "val"
        with (Path(task.output_dir) / "loss_history.csv").open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 2
