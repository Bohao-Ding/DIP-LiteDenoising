# TMASR-Net: efficient self-supervised sequence reconstruction

Updated 12 October 2026. [All completed experiments](../results/README.md), [experiment catalog](../results/experiments.csv) and [final metrics](../results/results.csv) provide the numerical record. This report explains the principal result; the English manuscript is maintained locally.

## Design and contribution

The Temporal-Mean-Anchored Sub-pixel Residual Network (TMASR-Net; G4) predicts an excluded centre frame from its temporal neighbours. It adapts DeepInterpolation's sequence-learning principle by combining native-resolution temporal averaging with joint neighbour processing in a compact spatial representation and learned residual correction. The entire prediction is trained on noisy targets without clean reference images. Their full-resolution mean is the output anchor. PixelUnshuffle(2) rearranges the input into four spatial phases; a compact U-Net learns a residual on this half-resolution grid. Encoder widths are 8/16/32, with a 64-channel bottleneck, concatenated skips and decoder widths 32/16/8. A zero-initialized four-phase head starts from the mean, and PixelShuffle restores the correction before addition to the anchor.

The contribution is the integrated reconstruction framework and its measured quality–parameter–runtime performance. Individual averaging, rearrangement and residual mechanisms are credited design foundations. At matched w30, both G4 and the original-architecture DI reproduction consume 60 centre-excluded frames. The original hierarchy uses 64/128/256/512/1024 channels and direct prediction.

| Quantity | Original-architecture DI | G4 |
|---|---:|---:|
| Parameters, w30 | 15,704,769 | 139,204 |
| Conv. GFLOPs, 512×512 | 288.7359 | 3.5903 |
| Conv. GFLOPs, 498×490 | 279.7129 | 3.3333 |
| Validation PSNR, 01.00, MSE/seed 0 | 15.8617 | 15.8586 |
| Validation PSNR, 01.01, MSE/seed 0 | 28.7237 | 28.7149 |
| Validation PSNR, 03.00, MSE/seed 0 | 13.3705 | 13.3642 |

Parameters fall by 112.8x and convolution arithmetic by 80.4–83.9x. The three PSNR differences are within 0.009 dB. The 75-epoch reproduction comparison uses batch 4, Adam, StepLR and objective-specific validation selection; initial learning rates are 1e-4 for DI and 1e-3 for G4. Three-seed MAE-DI/G4 and the additional completed MSE-DI seed remain separately available. Deployment w2 has four neighbours and 123,076 G4 parameters.

## What the controls explain

The 225-run G1–G5 study crosses objective, placement and depth over three recordings, three windows and five seeds. MSE versus MAE adds 0.2450 dB on the full grid and 0.2435 dB on the sub-pixel grid. At a fixed objective, placement changes PSNR by -0.0020/-0.0036 dB while reducing averaged convolution cost from 5.334 to 1.464 GFLOPs. The objective explains the major reconstruction change; computation location explains the arithmetic reduction.

The 405 trained capacity runs compare width, depth and block choices, alongside 135 deterministic baseline evaluations. Structural controls, window experiments and the two-seed 4x widening study explain which capacity changes recover quality and their cost. Completed controls are shared even when they do not improve G4.

Clean-reference simulation reports G4/G2 PSNR of 42.22/44.08 dB and relative amplitude error of 0.390/0.491. Standard local SSIM is separate from whole-frame gSSIM: five-seed G4/G2 means are 0.7012/0.7005, 0.9711/0.9710 and 0.6008/0.6013. Suite2p ROI counts change from 245/434/412 to 437/816/1176, accompanied by lower median relative trace noise. Past-only offsets (-2,-1) retain validation PSNR within 0.025 dB of matched symmetric runs. The fixed design, separately retrained on 02.00/04.00, gives 26.6323/34.6562 dB; this is additional-recording validation.

## Corrected event ROC

The [corrected diagnostic](event_roc_correction.md) uses original-frame lookup for both peak and baseline windows. G4 AUC is 1.000000 at peaks 16/32; the four arms span 0.997646–1.000000. All reach 100% sensitivity at the negative-score 95th percentile, with empirical FPR 4.802–5.085%. Descriptive bootstrap intervals and thresholds are shared in the scalar table. The ceiling result does not distinguish transient-amplitude fidelity. Existing weights were evaluated without training.

## Measurement definitions

Real-data PSNR uses the observed centre frame's intensity range; clean PSNR refers to synthetic ground truth. Historical evaluation segments were previously inspected. Seed counts and split labels accompany every endpoint. ROI counts and trace variability describe the detected sets and are not measures of detection accuracy.

The 3.7–4.2x DI/G4 isolated-forward speedup uses RTX 5070 Ti, batch 1, FP16, channels-last, 20 warmup and 100 timed calls, with ratios of three-seed mean median times. Historical G1–G5 timing remains labelled RTX 4060 Laptop. Corrected w2 sequence processing includes TIFF reads, preprocessing, transfers, forward execution and CPU output consumption: G4 takes 8.26–9.81 ms/frame across two passes. This diagnostic comparison shows no consistent whole-process G4 advantage over G2. Future-frame acquisition waiting is a separate cost.

Conv3d arithmetic includes the temporal kernel. The adapted DeepCAD-RT arm uses spatial-only pooling and temporal-mean output aggregation, with 1,459,585 parameters and 254.6110/242.8005 GFLOPs for the two native shapes. Missing measurements remain blank, and zero placeholders are replaced by actual arithmetic counts.

## Figure and sharing

The architecture schematic is available as [editable SVG](../results/figure/architecture_comparison.svg) and a [PNG preview](../results/figure/architecture_comparison.png). It was redrawn with R/grid and svglite on 13 October 2026 using Codex-assisted code, adapting the layout of an earlier imagegen draft whose model version was not reported. Labels, dimensions, skips and centre-exclusion paths were checked against the implementation. It explains the model and contains no experimental image evidence.

The new sharing package contains final endpoint tables and explanations. Training histories, interrupted attempts, new configurations/scripts, weights and checksum manifests are retained locally rather than added to this release. Existing source and frozen public evidence remain available in their original repository locations.
