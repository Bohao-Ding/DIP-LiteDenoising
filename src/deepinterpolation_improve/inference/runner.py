from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepinterpolation_improve.manifest import load_manifest, validate_manifest_for_dataset


class InferenceRunner:
    def __init__(
        self,
        model,
        dataset,
        output_file: str | Path,
        batch_size: int = 8,
        device: str = "cpu",
        manifest_path: str | Path | None = None,
        enforce_ordered: bool = True,
    ):
        self.model = model
        self.dataset = dataset.ordered_copy() if enforce_ordered and hasattr(dataset, "ordered_copy") else dataset
        self.output_file = Path(output_file)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.model.to(self.device)

    def run(self) -> Path:
        if self.manifest_path is not None:
            validate_manifest_for_dataset(load_manifest(self.manifest_path), self.dataset)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        output_shape = (len(self.dataset), *self.dataset.target_shape[1:])
        written = 0
        self.model.eval()
        with h5py.File(self.output_file, "w") as handle:
            data_out = handle.create_dataset("data", shape=output_shape, dtype="float32")
            frame_out = handle.create_dataset("frame_index", shape=(len(self.dataset),), dtype="int64")
            with torch.no_grad():
                for batch in tqdm(loader, leave=False):
                    inputs = batch["input"].to(self.device, dtype=torch.float32)
                    predictions = self.model(inputs).detach().cpu().numpy()
                    predictions = np.squeeze(predictions, axis=1)
                    local_size = predictions.shape[0]
                    data_out[written : written + local_size] = predictions
                    frame_out[written : written + local_size] = batch["frame_index"].numpy()
                    written += local_size
        return self.output_file
