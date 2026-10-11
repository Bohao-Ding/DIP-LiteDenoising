# DIP-LiteDenoising

[English](README.md) | [中文](README.zh-CN.md)

## Final experimental results

**[Browse every completed experiment](results/README.md)** · [Experiment catalog](results/experiments.csv) · [Full final metrics](results/results.csv) · [Architecture comparison](results/figure/architecture_comparison.png)

The principal result is **temporal-mean-anchored compact sub-pixel residual reconstruction**. At matched w30 (60 context frames), G4 has **139,204 parameters** versus **15,704,769** for original-architecture DI: **112.8x fewer parameters**, **80.4–83.9x lower convolution arithmetic**, and **3.7–4.2x faster isolated batch-1 forward execution** on RTX 5070 Ti. Seed-0 validation reconstruction PSNR differs from MSE-DI by less than **0.009 dB** on each of three recordings.

The result collection contains **124 experiment entries and 16,617 scalar endpoints**: capacity, G1–G5 attribution, original DI MAE/MSE, structural/window and widening controls, simulation, standard local SSIM, efficiency, ROI/Suite2p, causal inputs and additional-recording validation. Completed individual seeds and their descriptive summaries are distinguished. See the [technical explanation](docs/technical_report.md) for the contribution and measurement definitions. Model-forward and whole-process timings are reported separately.

![Original DI and compact G4 information flow](results/figure/architecture_comparison.png)

This explanatory schematic was generated and revised with OpenAI imagegen on 12 October 2026; the tool interface did not expose its model version. The information flow and dimensions were checked against the implementation. Numeric plots and results use saved measurements.

DIP-LiteDenoising is a compact PyTorch implementation of the
U-Net model family studied for DeepInterpolation-style self-supervised temporal
prediction. It predicts an observed centre frame from neighbouring frames while
excluding the centre frame from the input. The repository is intended for
architecture inspection, model development, training and inference on
scientific image sequences.

The same Python distribution also contains the locked `paper_evidence_v1`
experiment workflow and lightweight public evidence for the 405-run capacity
study and the later 252-run objective-by-computation study. It does not contain
downloaded recordings, pretrained weights, checkpoints, predicted frames or
recoverable image arrays.

## Why this project

High-throughput scientific imaging produces long recordings whose analysis can
be limited by model capacity, memory and device behaviour. This project exposes
the tested width, depth and convolution-block choices as a small, explicit
PyTorch model family. It supports controlled selection of model capacity and
keeps the frozen experiment protocol, executable workflow and public audit
evidence in one repository.

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
├── configs/
│   ├── paper_evidence_v1/  # Frozen core, mechanism and protocol contracts
│   ├── exploratory/        # Non-confirmatory screen configurations
│   └── splits/             # Published split manifests (no image pixels)
├── docs/                   # Technical report, reproduction and data boundaries
├── evidence/
│   ├── capacity_study_v1/  # Lightweight evidence for 405 trained runs
│   └── paper_evidence_v1/  # Lightweight evidence for 252 trained runs
├── scripts/                # Pipeline, plotting and manifest verification
├── src/deepinterpolation_improve/
│   ├── config/       # Window, dataset, training and inference configuration
│   ├── data/         # Temporal datasets, windowing and synthetic data
│   ├── inference/    # Ordered HDF5 inference
│   ├── manifest/     # Run metadata and input compatibility checks
│   ├── models/       # U-Net blocks, registry and the study model family
│   ├── training/     # Losses, training, checkpointing and resume support
│   └── experiments/paper_evidence_v1/ # Integrated experiment subpackage
├── tests/            # Core and paper-evidence contract tests
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

## Paper-evidence workflow

The installed `dip-evidence` command uses the same package as the reusable core.
Formal configurations are isolated in `configs/paper_evidence_v1/`; exploratory
screens are kept in `configs/exploratory/`.

Audit the code/configuration contracts and enumerate the locked core matrix
without local recordings:

```powershell
dip-evidence check
dip-evidence paper-core --no-path-check
```

For a full local rerun, create an ignored machine-local path map, edit its three
Neurofinder roots, then validate both the recordings and frozen splits:

```powershell
Copy-Item configs/paths.example.json configs/paths.local.json
dip-evidence data-check --paths configs/paths.local.json
dip-evidence paper-core --paths configs/paths.local.json
dip-evidence paper-mechanism --paths configs/paths.local.json
```

The last two commands are dry plans. Add `--execute` only after checking their
task snapshots; add `--resume` to continue existing local checkpoints. The
complete locked sequence is also available as:

```powershell
python scripts/run_paper_evidence_pipeline.py --paths configs/paths.local.json
```

Verify the lightweight public evidence without training or data access:

```powershell
python scripts/build_distribution_manifest.py --check
```

The public path derivative is
`evidence/paper_evidence_v1/tables/source_mapping.public.csv`. Its machine-local
paths were replaced by repository-relative locations; the redaction and the
original private-manifest hash are recorded in
`evidence/distribution_manifest.json`. See [the reproduction guide](docs/reproduction.md)
and [full technical report](docs/technical_report.md).

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

No raw recording is distributed in this repository. The `evidence/` directory
contains only repository-authored aggregate, run-level or frame-level metrics,
metadata and rendered figures. Users remain responsible for checking the
licence and citation requirements of external datasets.

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
the reusable model family, centre-frame exclusion, ordering, checkpoint/resume,
the integrated experiment variants, strict state-dict loading, frozen task
matrices, path overrides and evidence-analysis contracts.

## Scope and limitations

- The observed centre frame is a self-supervised target, not a clean ground
  truth image.
- Successful centre-frame prediction does not by itself prove noise removal,
  biological-signal preservation or improved downstream analysis.
- Public evidence permits audit of the two frozen local studies, but trained
  checkpoints, raw recordings and predicted frames remain excluded.
- The synthetic quickstart demonstrates software execution only. Reproducing
  reported measurements requires the external recordings and the frozen study
  protocol.
- Performance and memory use depend on the temporal window, image size, batch
  size, PyTorch build and hardware.
- The matched evaluation segments in `paper_evidence_v1` had been inspected in
  earlier exploration and are not an independent post-selection test set.

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

## Local research and release responsibilities

This checkout remains the independent public release repository. The complete local research workspace uses its own Git history and the consolidated research package as its development source. Do not install both checkouts into the same Python environment. This maintenance preserved the 51-file evidence manifest and did not publish new scientific results or push commits. See [maintenance context](MEMORY.md).
