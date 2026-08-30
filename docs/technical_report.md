# Technical report: capacity placement and objective-aligned subpixel inference

Status: repository-facing technical record; revised after the full-package manuscript review
Scientific manuscript authority: maintained separately; this report is not submission material
Frozen extension authority: `evidence/paper_evidence_v1/evaluation_freeze.json` and `evidence/paper_evidence_v1/paper_report.json`
Language authority: this English report; `technical_report.zh-CN.md` is the synchronized Chinese translation.

## 1. Purpose and claim boundary

This report preserves the implementation, provenance and negative-result details that are necessary for local reproduction but would obscure the manuscript's central argument. It is not intended to accompany the journal submission, and the manuscript and supplementary material do not depend on it to define the methods or interpret the primary results.

The supported endpoint is **noisy-centre-frame reconstruction fidelity within three source recordings, plus model-only inference measurements on one audited device**. The experiments do not establish clean-reference denoising, removal of independent noise, biological-signal preservation, ROI or trace recovery, event detection, unseen-recording generalisation, cross-hardware speed, acquisition throughput or clinical utility.

The project contains 405 training runs in the capacity study and 252 in the frozen extension, for 657 algorithmic runs. These are nested within three Neurofinder acquisition recordings and are not 657 independent biological replicates. This report preserves the real 405-to-252 execution order and freeze lineage. The submission main text does not organise its argument around the total run count or execution chronology; it follows the compact model, computation placement, capacity return, training objective and device performance.

## 2. Authority map and chronology

| Stage | Purpose | Runs | Current authority |
|---|---|---:|---|
| Capacity study | Compare width, depth, block family and temporal context | 405 | `evidence/capacity_study_v1/` |
| Capacity audit | Identify where parameters produce convolutional arithmetic | deterministic | `evidence/capacity_study_v1/gflops_stage_attribution.csv` |
| Anomaly audit | Test sensitivity of the lower-capacity chain | post hoc diagnostic | `evidence/capacity_study_v1/anomaly_*.csv` |
| Core extension | Cross full-resolution/subpixel computation with MAE/MSE | 225 | frozen `evidence/paper_evidence_v1/` artifacts |
| Mechanism controls | Dense refiner, NAF-lite and temporal pair fusion at seed 0 | 27 | frozen `evidence/paper_evidence_v1/` artifacts |
| Formal device audit | Repeated RTX 4060 measurements | 18 raw condition records plus summaries | `evidence/paper_evidence_v1/latency/` |

### 2.1 Manuscript--technical-report crosswalk

| Element | Submission main-text organization | Technical-report interpretation |
|---|---|---|
| Design factors | Width, depth, block family, inference precision, computation placement and objective | The engineering chronology still separates the 405-run capacity study from the later 252-run frozen extension |
| Architecture-and-capacity evidence | Explains compact B8-D3 selection and computation placement | Executed first; includes architecture screening, capacity placement, anomaly audit and model-selection evidence |
| Controlled-attribution evidence | Explains the G4 computation placement and MSE choice | Executed later; preserves 225 core runs and 27 mechanism controls with frozen provenance |

This crosswalk changes document organisation, not provenance. The submission follows the scientific questions needed to understand the compact model, whereas this report retains the chronological 405-to-252 evidence lineage and does not retroactively portray the second block as having been fixed before the first block was inspected.

The discovery study preceded the subpixel extension. Its most useful result was a structural diagnostic rather than an architecture ranking: B8-D4 and B16-D3 had similar parameter counts (491,257 and 487,537) but required 6.950 and 20.983 convolutional GFLOPs. This showed that parameter count did not specify computation placement. The subpixel residual design was then tested in an isolated namespace.

The extension training stage produced validation artifacts only. Formal evaluation was blocked until all 252 training directories passed the artifact contract. The protocol SHA-256 is:

```text
82efd3d6aa3170d5f9acb1c15eab4acbb0ad27723e768fe517bf558eacf1aa85
```

The evaluation-freeze identifier is:

```text
a723648a6ed1e510d65701b44a647b8fa6018dc0ad36474c3ebcec0fd9a71e74
```

The matched held-out evaluation segments were temporally separated from training and validation but inspected during earlier exploratory work. They are not an independent post-selection confirmation set.

## 3. Data, splits and licensing

The study uses Neurofinder recordings 01.00, 01.01 and 03.00. Recordings 01.00 and 01.01 contain GCaMP6s measurements from primary visual cortex; recording 03.00 contains GCaMP6f measurements from dorsal hippocampal CA1. All recordings are sampled at 7.5 Hz.

Frames were partitioned sequentially into 70% training, 15% validation and 15% evaluation segments. Window-specific purges exclude centres whose temporal context crosses a split boundary. Half-windows `w1`, `w2` and `w4` correspond to 2, 4 and 8 centre-excluded input frames. Normalisation uses frames 0--99 of the corresponding recording; these lie inside the training segment.

No explicit dataset licence or dataset DOI was identified in the downloaded Neurofinder metadata as checked on 6 August 2026. Consequently, the public evidence package must not include TIFF archives, complete target or prediction frames, image-equivalent arrays, or other content from which the source recordings could be reconstructed. Users should obtain the recordings from the data provider and comply with provider terms.

## 4. Model registry and architecture derivation

### 4.1 Discovery models

The 405-run study uses a sparse set of nine U-Nets. It is not a full width-by-depth-by-block factorial design.

| Artifact | Base width | Depth | Block | Mean parameters | Conv. GFLOPs |
|---|---:|---:|---|---:|---:|
| `bf4_d2_standard` | 4 | 2 | standard | 7,597 | 0.973 |
| `bf4_d3_standard` | 4 | 3 | standard | 30,733 | 1.378 |
| `bf8_d2_standard` | 8 | 2 | standard | 29,881 | 3.715 |
| `bf8_d3_standard_full_reference` | 8 | 3 | standard | 122,233 | 5.334 |
| `bf8_d4_standard` | 8 | 4 | standard | 491,257 | 6.950 |
| `bf16_d3_standard` | 16 | 3 | standard | 487,537 | 20.983 |
| `bf8_d3_depthwise` | 8 | 3 | depthwise | 17,949 | 0.961 |
| `bf8_d3_inverted_residual` | 8 | 3 | inverted residual | 75,257 | 4.236 |
| `bf8_d3_ghost` | 8 | 3 | Ghost | 8,364 | 0.434 |

The historical artifact suffix `full_reference` does not refer to a clean target. It identifies the B8-D3 comparison anchor in the source registry.

### 4.2 Original DeepInterpolation context

A source-level arithmetic recalculation of the original `unet_single_1024` architecture gives 15,704,769 parameters for its 60-channel input and single-convolution hierarchy. This is approximately 128.5 times the 122K anchor and 32.0 times the 491K reference. The calculation is unmatched architecture context only. Input context, blocks, data, optimisation and software differ, so it is not a performance comparison.

### 4.3 Extension arms

All extension arms use a temporal-mean anchor and a zero-initialised residual output.

| Arm | Computation | Capacity | Objective and selection | Interpretation |
|---|---|---|---|---|
| G1 | full resolution | B8-D3 | MAE | full-resolution control |
| G2 | full resolution | B8-D3 | MSE | objective-aligned control |
| G3 | 2x subpixel | B8-D3 | MAE | structure contrast under MAE |
| G4 | 2x subpixel | B8-D3 | MSE | principal subpixel arm |
| G5 | full resolution | B8-D4 | MSE | cross-design reference |

For $X\in\mathbb{R}^{C\times H\times W}$, omitting the batch dimension, factor-two PixelUnshuffle maps

$$
[P_2(X)]_{4c+2a+b,i,j}=X_{c,2i+a,2j+b},\qquad a,b\in\{0,1\}.
$$

Hence $P_2(X)\in\mathbb{R}^{4C\times H/2\times W/2}$, and $P_2^{-1}$ restores each phase to its original spatial location. The B8-D3 residual network uses encoder widths 8, 16 and 32, a 64-channel bottleneck, two $3\times3$ convolution--ReLU operations per level, max pooling, nearest-neighbour upsampling and skip concatenation. Its final $1\times1$ convolution produces four phase-residual channels. The prediction is

$$
\widehat Y=A(X)+P_2^{-1}[R_\theta(P_2(X))],\qquad A(X)=C^{-1}\sum_c X_c.
$$

The final convolution weights and bias are initialised to zero, so the initial prediction equals the temporal-mean anchor. Odd dimensions are padded before rearrangement and cropped after reconstruction. PixelUnshuffle is invertible and does not discard input samples. Its use is not novel: related low-resolution restoration principles appear in FFDNet and MWCNN. The contribution is the controlled combination with centre exclusion, the full-resolution temporal-mean anchor, zero-initialised residual learning, compact capacity, matched attribution and repeated device audit.

## 5. Training and frozen evaluation

The discovery and extension runs use Adam, learning rate `1e-3`, StepLR every 25 epochs with multiplier 0.5, 75 epochs, batch size 4, AMP FP16 and TF32. MAE arms train and select checkpoints with MAE; MSE arms use MSE. No arm receives a model-specific schedule.

The core extension has 225 runs: three recordings x three windows x five seeds x five arms. Three seed-0 mechanisms across nine conditions add 27 runs. The evaluation freeze records hashes for protocol and stage configurations, split and run manifests, training summaries, loss histories, final checkpoints and best checkpoints. A missing or changed file invalidates the freeze.

PSNR uses the observed target's per-frame maximum-minus-minimum range. `gSSIM` is a whole-frame luminance-contrast-covariance statistic and is not standard local-window SSIM. A 10,000-replicate hierarchy resamples recordings, then windows and seeds. Its bounds are sensitivity intervals, not population confidence intervals or equivalence tests.

The protocol-defined engineering rules are:

- quality retention: equal-condition G4-G2 mean at least -0.01 dB and minimum condition at least -0.03 dB;
- quality improvement: mean gain at least +0.02 dB;
- speed: G4/G2 batch-1 p50 latency ratio at most 0.75 in at least eight of nine conditions.

These rules were fixed before formal evaluation. They are not preregistered statistical hypotheses or biologically calibrated margins.

## 6. Complete results

### 6.1 Factorial attribution

| Effect | Mean PSNR difference (dB) | Positive | Sensitivity interval (dB) |
|---|---:|---:|---:|
| G3-G1 structure under MAE | -0.002008 | 22/45 | -0.008498 to 0.003682 |
| G4-G2 structure under MSE | -0.003552 | 18/45 | -0.011883 to 0.004090 |
| G2-G1 objective at full resolution | +0.245022 | 45/45 | 0.097184 to 0.445117 |
| G4-G3 objective under subpixel computation | +0.243477 | 45/45 | 0.088306 to 0.445907 |
| Interaction | -0.001545 | 21/45 | -0.009126 to 0.005124 |

MSE provides the dominant PSNR change in both graphs. It lowers normalized MSE but raises normalized MAE by about 0.020. The gain is therefore metric-specific.

### 6.2 Quality retention and repeated RTX 4060 audit

G2 uses 122,233 parameters and 5.334 convolutional GFLOPs. G4 uses 123,268 parameters and 1.464 GFLOPs. The G4-G2 mean is -0.003552 dB and the minimum condition mean is -0.019086 dB; both retention limits pass.

The formal device audit uses actual seed-0 checkpoints on an NVIDIA GeForce RTX 4060 Laptop GPU, FP16, channels-last tensors, 100 warm-up iterations, 500 timed repetitions and five cyclic rounds. Batch-1 p50 ratios range from 0.6378 to 0.7479 and pass in all nine conditions. Mean p50 latency is 1.3125 ms for G2 and 0.9181 ms for G4. Mean peak allocated model memory is 37.29 MB and 13.35 MB.

The audit supports model-only inference on this device and protocol. It excludes data loading, preprocessing, transfer, writing and acquisition integration.

### 6.3 Cross-design and mechanism references

G5-G4 averages +0.016852 dB and is positive in 44/45 runs. It does not reach the +0.02 dB improvement rule and is confounded by structure, depth, parameters and arithmetic.

At seed 0, dense refinement averages +0.006521 dB, NAF-lite +0.006112 dB and temporal pair fusion +0.002272 dB. None reaches +0.02 dB. The temporal-fusion normalized weight entropy ranges from 0.973 to 1.000, so the learned weights remain broadly distributed.

## 7. Discovery-study anomalies and aggregation sensitivity

Three discovery runs were identified post hoc because they departed visibly from the other seeds in the same condition. Two affect the first step of the standard-block capacity chain:

- `01.00/w2/bf4_d2_standard/seed4`;
- `01.00/w2/bf4_d3_standard/seed4`.

Both selected epoch 25, had the same minimum validation loss, shared 61 of 75 validation-loss values exactly and had matching training-frame-index hashes for all 75 epochs. Their checkpoint and loss-history files have different hashes. The available evidence proves that separate files exist but cannot establish independent optimisation histories. The pair is classified as an unexplained process-level anomaly; it is not assigned a mechanism.

The first capacity-chain increment is 0.0506 dB under complete-run means, 0.0105 dB after excluding the two disclosed chain runs with equal condition weighting, and 0.0108 dB under within-condition seed medians. The 122K-to-491K increment remains about 0.012 dB. The lower-capacity curve therefore cannot support a smooth scaling law, failure-rate estimate or unique elbow.

## 8. Two device records with different evidential strength

The RTX 4060 audit is repeated, warmed and cyclically balanced. It supports the formal G4-versus-G2 device claim.

The earlier RTX 5070 Ti block-screening record used batch size 4, zero warm-up and one repeat. Its stored mean, median and p95 are consequently identical within a run and do not describe a latency distribution. It orders some efficient blocks differently from parameter count and convolutional GFLOPs, but it cannot establish a stable device ranking. It is retained for historical traceability and must not be pooled with the RTX 4060 audit.

## 9. Repository runbook

The integrated repository exposes the experiment under `deepinterpolation_improve.experiments.paper_evidence_v1`. The public API retains the model registry and frozen-report functions, while the `dip-evidence` command provides inspection, dry-run planning and explicitly authorized execution.

Commit 83566c93cd1f29c276d39eb88e2ff6bdafd72e6e is the integrated manuscript snapshot referenced by the paper. This technical report may be updated in later documentation commits; such commits are not releases, tags or DOI-bearing versions unless explicitly recorded as such.

Recommended sequence:

```powershell
python -m pip install -e ".[test]"
dip-evidence check
Copy-Item configs\paths.example.json configs\paths.local.json
dip-evidence data-check --paths configs\paths.local.json
```

Run planning must default to dry-run. Training or formal evaluation requires `--execute`; continuation requires `--resume`. Machine-local paths belong in the ignored `configs/paths.local.json`, not in frozen scientific configurations. A run manifest records the override-file hash, resolved paths and split hash.

The public evidence distribution excludes `.pt` checkpoints, raw TIFFs, complete target/prediction frames, recoverable arrays, private absolute paths and any file over 100 MiB. `evidence/distribution_manifest.json` records public file hashes, redactions and exclusions. The original freeze and public distribution manifests serve different purposes and must not be conflated.

## 10. Data dictionary

| Artifact | Meaning |
|---|---|
| `run_index.csv` | one row per frozen extension run with metrics, complexity and hashes |
| `paired_effects_run_level.csv` | matched run-level deltas for all registered effects |
| `paired_effects_condition_level.csv` | equal-seed condition summaries |
| `paired_effects_recording_level.csv` | recording-stratified summaries |
| `paired_effects_frame_level.csv` | frame-level matched values; no image arrays |
| `temporal_mean_results.csv` | deterministic anchor outcomes |
| `latency_condition_aggregates.csv` | repeated condition/device summaries |
| `latency_summary.json` | aggregate audit endpoints and protocol |
| `paper_report.json` | frozen numerical report used by the manuscript |
| `evaluation_freeze.json` | training-artifact evaluation barrier |
| `source_mapping.public.csv` | portable recording/split map with repository-relative paths |
| `distribution_manifest.json` | public hashes, sizes, freeze lineage, redactions and exclusions |

## 11. Prohibited claims

Do not describe the evidence as any of the following:

- clean-reference denoising accuracy;
- verified independent-noise removal;
- biological-signal or event preservation;
- cross-recording or cross-laboratory generalisation;
- 657 biological replicates;
- statistical equivalence or a clinical non-inferiority result;
- a pure capacity effect for G5-G4;
- a stable RTX 5070 ranking;
- cross-hardware, end-to-end or real-time deployment performance;
- proof that PixelUnshuffle is novel.

## 12. Verification record

The unified repository integration was verified on 30 August 2026. The complete local suite passed **95 tests**. The integrated `models.py` and `blocks.py` matched their source authority byte-for-byte, with SHA-256 values `ac2e018aae62313a3eff06720be13a2ef10aececa0d738e119454034cf34e224` and `8db91054fe2db214916ecb1142731894b4126cb3f6d31ff5755547f327d6d1bd`. Strict state-dict loading, odd-shape output preservation, path overrides, task counts, split contracts, freeze analysis and latency-analysis contracts are covered by tests.

`python scripts/build_distribution_manifest.py --check` verified 51 public payload files totalling 13,365,803 bytes. The distribution manifest records the source protocol, freeze hash, original private-manifest SHA-256, per-file hashes and two explicit redactions. The portable source map's nine split hashes match `configs/splits/neurofinder/`. No checkpoint, raw-pixel type, private machine path or file over 100 MiB is distributed.

Tests ran in system temporary directories with pytest cache generation disabled. The temporary directories were deleted; generated bytecode caches were enumerated, boundary-checked and removed. The repository was then checked for residual `__pycache__`, `.pytest_cache` and smoke-test output. The technical report may receive new verification receipts, but frozen scientific artifacts and identifiers must remain unchanged.

## 13. Bilingual section and numerical crosswalk

| English section | Chinese mirror | Canonical anchor |
|---|---|---|
| 1--3: scope, chronology, data | 1--3 | the manuscript is organised by design factors; the real 405-to-252 execution order is retained here |
| 4--5: models, training, freeze | 4--5 | protocol SHA-256 `82efd3d6...`; freeze `a723648a...` |
| 6: complete results | 6 | G4-G2 −0.003552 dB; G2→G4 latency ratio 0.6378--0.7479 |
| 7--8: anomalies and devices | 7--8 | two disclosed chain anomalies; formal RTX 4060 versus exploratory RTX 5070 Ti |
| 9--10: runbook and dictionary | 9--10 | `dip-evidence`; 51 payload files / 13,365,803 bytes |
| 11--12: prohibited claims and verification | 11--12 | 95 tests; no raw pixels, weights or private absolute paths |

The section numbers, model labels, thresholds, run counts and numerical results in `technical_report.zh-CN.md` mirror this document. When wording diverges, the frozen JSON/CSV evidence and hashes take precedence over either prose version.
