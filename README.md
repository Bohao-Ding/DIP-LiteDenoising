# DIP-LiteDenoising

[English](README.md) | [中文](README.zh-CN.md)

DIP-LiteDenoising is a compact, source-only PyTorch implementation of the
U-Net model family studied for DeepInterpolation-style self-supervised temporal
prediction. It predicts an observed centre frame from neighbouring frames while
excluding the centre frame from the input. The repository is intended for
architecture inspection, model development, training and inference on
scientific image sequences.

The repository contains no downloaded datasets, pretrained weights, paper
evaluation code, archived metrics or training outputs. Its built-in quickstart
checks that the model, training and inference paths work; it does not reproduce
the performance claims of a paper.

## Why this project

High-throughput scientific imaging produces long recordings whose analysis can
be limited by model capacity, memory and device behaviour. This project exposes
the tested width, depth and convolution-block choices as a small, explicit
PyTorch model family. It supports controlled selection of model capacity
without embedding evaluation results into the source repository.

The code focuses on three reusable components:

- centre-excluded temporal-window construction;
- configurable 2D U-Nets with standard, depthwise, inverted-residual and Ghost
  blocks;
- a minimal training, checkpoint, manifest and ordered-inference path.

## Model family

`bf` denotes the base feature width and `d` denotes encoder depth.

| Variant | Base width | Depth | Block |
|---|---:|---:|---|
| `bf4_d2_standard` | 4 | 2 | standard double convolution |
| `bf4_d3_standard` | 4 | 3 | standard double convolution |
| `bf8_d2_standard` | 8 | 2 | standard double convolution |
| `bf8_d3_standard_full_reference` | 8 | 3 | standard double convolution |
| `bf8_d4_standard` | 8 | 4 | standard double convolution |
| `bf16_d3_standard` | 16 | 3 | standard double convolution |
| `bf8_d3_depthwise` | 8 | 3 | depthwise separable |
| `bf8_d3_inverted_residual` | 8 | 3 | inverted residual |
| `bf8_d3_ghost` | 8 | 3 | Ghost |

`compact_di` is a readable alias for
`bf8_d3_standard_full_reference`. The historical phrase `full_reference` is a
registry label and does not mean that clean reference frames are used.

The number of input channels is determined by the temporal window. For example,
`WindowSpec(pre_frame=2, post_frame=2)` produces four neighbouring-frame input
channels. The centre frame is always excluded.

## Repository structure

```text
DIP-LiteDenoising/
├── examples/
│   └── quickstart_synthetic.py
├── src/deepinterpolation_improve/
│   ├── config/       # Window, dataset, training and inference configuration
│   ├── data/         # Temporal datasets, windowing and synthetic data
│   ├── inference/    # Ordered HDF5 inference
│   ├── manifest/     # Run metadata and input compatibility checks
│   ├── models/       # U-Net blocks, registry and the study model family
│   └── training/     # Losses, training, checkpointing and resume support
├── tests/            # Core model, data, training and inference tests
├── LICENSE
├── NOTICE.md
├── THIRD_PARTY_NOTICES.md
├── README.md
├── README.zh-CN.md
└── pyproject.toml
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.8 or newer
- NumPy, h5py, tifffile, nibabel and tqdm

For CUDA use, install a PyTorch build that matches the local CUDA environment
before installing this project. Follow the official
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for the
appropriate command.

## Installation

Windows PowerShell:

```powershell
git clone https://github.com/Bohao-Ding/DIP-LiteDenoising.git
cd DIP-LiteDenoising
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Linux or macOS:

```bash
git clone https://github.com/Bohao-Ding/DIP-LiteDenoising.git
cd DIP-LiteDenoising
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Quick functional reproduction

List all model labels:

```powershell
python examples/quickstart_synthetic.py --list-variants
```

Run a one-epoch CPU training and inference check:

```powershell
python examples/quickstart_synthetic.py `
  --variant compact_di `
  --device cpu `
  --epochs 1 `
  --output-dir outputs/quickstart
```

On Linux or macOS, replace the PowerShell backticks with backslashes or enter
the command on one line.

The script generates a small calcium-like movie in memory, adds
Poisson–Gaussian noise, trains the requested variant, reloads `best.pt`, and
writes ordered predictions to HDF5. It does not download data or calculate
PSNR, SSIM, latency or paper tables.

Expected runtime outputs are:

```text
outputs/quickstart/
├── predictions.h5
└── training/
    ├── best.pt
    ├── checkpoint.pt
    ├── loss_history.csv
    └── manifest.json
```

These files are ignored by Git.

## Creating a model

Use one of the registered study variants:

```python
from deepinterpolation_improve.models import create_model_variant

model = create_model_variant(
    "bf8_d3_ghost",
    input_channels=4,
    output_channels=1,
)
```

Use `compact_di` for the compact standard-block configuration:

```python
model = create_model_variant("compact_di", input_channels=4)
```

The lower-level `create_model` API remains available when a custom width, depth
or block configuration is required:

```python
from deepinterpolation_improve.models import create_model

model = create_model(
    "unet_single_1024",
    {
        "input_channels": 4,
        "output_channels": 1,
        "base_features": 12,
        "depth": 3,
        "block_type": "standard",
    },
)
```

## Training on a TIFF movie

The input TIFF is expected to contain a `[time, height, width]` sequence.

```python
from pathlib import Path

from deepinterpolation_improve.config import TrainingConfig, WindowSpec
from deepinterpolation_improve.data import SingleTifDataset
from deepinterpolation_improve.models import create_model_variant
from deepinterpolation_improve.training import Trainer

movie_path = Path("path/to/movie.tif")
window = WindowSpec(pre_frame=2, post_frame=2, pre_post_omission=0)

train_data = SingleTifDataset(
    movie_path,
    window=window,
    start_frame=0,
    end_frame=799,
    randomize=True,
    seed=7,
)
validation_data = SingleTifDataset(
    movie_path,
    window=window,
    start_frame=804,
    end_frame=999,
    randomize=False,
)

model = create_model_variant("compact_di", train_data.input_shape[0])
config = TrainingConfig(
    output_dir=Path("outputs/my_run"),
    run_uid="my_run",
    model_name="compact_di",
    loss="mean_absolute_error",
    batch_size=4,
    epochs=75,
    learning_rate=1e-3,
    scheduler_step_size=25,
    scheduler_gamma=0.5,
    device="cuda",
    seed=7,
)

Trainer(model, train_data, validation_data, config).fit()
```

Choose non-overlapping train and validation intervals with enough temporal
clearance for the selected window. The repository does not impose a universal
split because recording length and acquisition conditions vary.

## Resuming training

Create the same model and dataset configuration, then point `resume_from` to a
checkpoint:

```python
config = TrainingConfig(
    output_dir=Path("outputs/my_run"),
    run_uid="my_run",
    model_name="compact_di",
    epochs=100,
    resume_from=Path("outputs/my_run/checkpoint.pt"),
)
```

The checkpoint stores the model, optimizer, scheduler, AMP scaler and random
states needed by the trainer.

## Inference

```python
from pathlib import Path

import torch

from deepinterpolation_improve.inference import InferenceRunner
from deepinterpolation_improve.models import create_model_variant

model = create_model_variant("compact_di", validation_data.input_shape[0])
checkpoint = torch.load(
    "outputs/my_run/best.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state"])

InferenceRunner(
    model,
    validation_data,
    output_file=Path("outputs/my_run/predictions.h5"),
    batch_size=4,
    device="cpu",
    manifest_path=Path("outputs/my_run/manifest.json"),
).run()
```

Inference uses an ordered copy of the dataset by default and stores both
predictions and source frame indices. Manifest validation rejects incompatible
window or input-shape settings.

## Supported data adapters

The source package includes adapters for:

- in-memory NumPy arrays;
- single and multiple TIFF movies;
- optical-physiology HDF5 movies;
- electrophysiology binary data;
- Neurofinder directory layouts;
- NIfTI fMRI volumes;
- JSON-indexed movie collections;
- synthetic calcium-like sequences.

No data files are distributed in this repository. Users are responsible for
checking the license and citation requirements of external datasets.

## Third-party data and licence boundary

The source package includes readers and download metadata for externally
hosted data, including Neurofinder. These interfaces do not redistribute or
relicense the data. As of 6 August 2026, the public CodeNeuro Neurofinder and
datasets repositories did not state an explicit dataset licence or dataset DOI.
Public download access should therefore not be interpreted as permission to
republish the recordings. Users must obtain third-party data from the source
provider and determine whether their intended use complies with the applicable
conditions.

The Allen Institute DeepInterpolation project is governed by its own licence,
which includes restrictions not present in the MIT License. The MIT License in
this repository covers only the independently written PyTorch source and its
documentation. It grants no rights to Allen source code, Neurofinder data or
other third-party materials. See [NOTICE.md](NOTICE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Tests

```powershell
python -m pytest -q
```

The tests create their data and outputs in temporary directories. They verify
the nine model variants, the `compact_di` alias, centre-frame exclusion,
dataset ordering, checkpoint/resume behaviour and the minimal training and
inference path.

## Scope and limitations

- The observed centre frame is a self-supervised target, not a clean ground
  truth image.
- Successful centre-frame prediction does not by itself prove noise removal,
  biological-signal preservation or improved downstream analysis.
- This source-only repository does not include the paper evaluation pipeline,
  public-dataset downloads, trained checkpoints or archived metrics.
- The quickstart demonstrates software execution only. It does not reproduce
  paper PSNR, structural-similarity or hardware-timing results.
- Performance and memory use depend on the temporal window, image size, batch
  size, PyTorch build and hardware.

## Relationship to DeepInterpolation

This project is inspired by the self-supervised temporal-prediction principle
introduced in:

Lecoq, Jérôme, et al. “Removing Independent Noise in Systems Neuroscience Data
Using DeepInterpolation.” *Nature Methods*, vol. 18, 2021, pp. 1401–1408.
[https://doi.org/10.1038/s41592-021-01285-2](https://doi.org/10.1038/s41592-021-01285-2).

The original Allen Institute project is available at
[AllenInstitute/deepinterpolation](https://github.com/AllenInstitute/deepinterpolation).
No legacy TensorFlow source code is included here.

## License

The original PyTorch source code in this repository is released under the
[MIT License](LICENSE). The licence does not cover third-party software or
datasets. See [NOTICE.md](NOTICE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and reuse
boundaries.

No Zenodo archive or repository DOI has been created.
