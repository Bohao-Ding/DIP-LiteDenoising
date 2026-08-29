# Public evidence package

This directory contains lightweight, repository-authored evidence for two local
studies. It contains no model weights, Neurofinder pixels, predicted frames or
recoverable image arrays.

- `capacity_study_v1/`: 405 trained runs across three recordings, three windows,
  nine trained models and five seeds; 135 deterministic temporal-mean records
  are labelled separately and are not independent trained runs.
- `paper_evidence_v1/`: 225 locked core runs plus 27 mechanism runs, paired
  effects, device-specific latency records and Figure 6.
- `distribution_manifest.json`: SHA-256, byte size, source freeze, public
  redactions and exclusion rules for every distributed evidence file.

Run `python scripts/build_distribution_manifest.py --check` from the repository
root to verify the package. The path table distributed here is explicitly named
`source_mapping.public.csv`; the private mapping with machine-local absolute
paths is not included.

See `docs/reproduction.md` and `docs/data_and_licensing.md` before reuse.
