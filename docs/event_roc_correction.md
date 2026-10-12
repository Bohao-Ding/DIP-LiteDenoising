# Event ROC correction and current results

Updated 12 October 2026. This correction replaces the event ROC endpoints in release `4788fc2`. Existing seed-0 simulation checkpoints were re-evaluated in FP32; no training was started. The experiment catalog remains at 124 entries, with 16,677 scalar rows after adding the threshold, empirical false-positive rate and descriptive bootstrap bounds.

## What changed and why

Prediction row 0 represents original frame 2. The old evaluator used the original event frame as a row number, shifting both peak and baseline windows by two frames. The corrected calculation looks up each original frame explicitly: peak `[t_e-2,t_e+2]`, baseline median `[t_e-8,t_e-4]`. Positive and negative windows use the same mapping.

The saved ROC curve also used to sort points by false-positive rate alone and omitted the final endpoint. It now follows descending score thresholds, including `(0,0)` and `(1,1)`; curve integration agrees with the independently computed rank AUC. This curve repair does not change the rank-based scalar AUC.

## Corrected final endpoints

Each noise condition uses signal seed 300, Poisson peak 16 or 32, Gaussian SD 0.03, 30 events from cells 3–7, and 708 overlapping no-event windows from cells 0–2. The score is projected peak fluorescence minus the preceding median. Sensitivity uses the negative-score 95th percentile. This is a nominal 5% operating point; the empirical FPR is measured, not bounded by assumption.

| Model | Peak | AUC | Descriptive bootstrap 95% interval | Sensitivity (%) | Empirical FPR (%) |
|---|---:|---:|---|---:|---:|
| g4 | 16 | 1.000000 | 1.000000–1.000000 | 100.0 | 5.085 |
| g4 | 32 | 1.000000 | 1.000000–1.000000 | 100.0 | 5.085 |
| g2 | 16 | 0.999812 | 0.999247–1.000000 | 100.0 | 5.085 |
| g2 | 32 | 1.000000 | 1.000000–1.000000 | 100.0 | 5.085 |
| g8 | 16 | 0.997646 | 0.994114–0.999953 | 100.0 | 4.802 |
| g8 | 32 | 0.999341 | 0.997974–1.000000 | 100.0 | 5.085 |
| temporal_mean | 16 | 0.998258 | 0.995009–0.999953 | 100.0 | 5.085 |
| temporal_mean | 32 | 0.999153 | 0.997363–1.000000 | 100.0 | 4.944 |

The full-precision values, thresholds, counts and bootstrap bounds are in [results.csv](../results/results.csv). The 1,000-draw, seed-0 bootstrap resamples diagnostic event/window scores. Its intervals are descriptive: negative windows overlap, the two movies share a signal seed, and event-bearing and non-event cell types differ. They do not measure independent-film or biological replication.

G4 has AUC 1.000000 at both noise levels, while every arm reaches 100% sensitivity at this threshold. The ROC diagnostic is therefore close to its ceiling even for temporal averaging. It supports event separability in this simulator; amplitude and peak-time errors answer different questions and remain the dynamic-response endpoints. This correction does not establish a new denoising-quality or biological-fidelity advantage.

## Verification and figure

Original-frame window tests cover several prediction start offsets and non-contiguous rows. Corrected event scores equal the independently implemented amplitude-diagnostic values. sklearn rank AUC agrees with saved AUC and curve integration for all eight conditions. Checkpoint and input-configuration identities are unchanged, and all 16,613 unrelated scalar rows match the preceding release.

The [architecture figure](../results/figure/architecture_comparison.png) keeps horizontal model chains compared above/below, removes the four-item footer and redundant prediction cards, uses quieter colors, and separates both prediction-to-loss paths. It is an explanatory image revised with OpenAI imagegen on 12 October 2026; the interface did not expose the image model version. The local manuscript places it on a portrait page.
