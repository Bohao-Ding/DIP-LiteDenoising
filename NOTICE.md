# Project notice

DIP-LiteDenoising is an independent PyTorch implementation for studying compact
U-Net model families in self-supervised temporal prediction. Its integrated
`paper_evidence_v1` experiment package is also independently written. This
repository does not contain or redistribute the legacy TensorFlow source code,
pretrained models, datasets or evaluation outputs from the Allen Institute
DeepInterpolation project.

The method is inspired by:

> Lecoq, Jérôme, et al. “Removing Independent Noise in Systems Neuroscience
> Data Using DeepInterpolation.” *Nature Methods*, vol. 18, 2021,
> pp. 1401–1408. https://doi.org/10.1038/s41592-021-01285-2.

The original DeepInterpolation project is available at
https://github.com/AllenInstitute/deepinterpolation and is governed by its own
licence, which includes restrictions that are not part of the MIT License. No
rights to the Allen Institute source code are granted by this repository.

The MIT License in `LICENSE` applies to independently authored materials in
this repository unless a file states otherwise. It does not grant rights to
any third-party software, methods, publications or datasets. In particular,
Neurofinder recordings are not distributed here. Public evidence contains only
metrics, metadata and rendered figures; checkpoints, pixels and recoverable
image arrays are excluded. Users must obtain third-party data from the provider
and comply with the provider's terms. See `THIRD_PARTY_NOTICES.md` and
`docs/data_and_licensing.md` for method, architecture and dataset provenance.
