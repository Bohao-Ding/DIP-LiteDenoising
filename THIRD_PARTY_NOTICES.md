# Third-party notices

This file records the intellectual provenance of methods, architecture ideas
and optional external data interfaces referenced by DIP-LiteDenoising. These
references do not indicate that third-party source code has been copied into
this repository.

## DeepInterpolation

The centre-excluded temporal-prediction task is based on:

> Lecoq, Jérôme, et al. “Removing Independent Noise in Systems Neuroscience
> Data Using DeepInterpolation.” *Nature Methods*, vol. 18, 2021,
> pp. 1401–1408. https://doi.org/10.1038/s41592-021-01285-2.

The Allen Institute reference implementation is available at
https://github.com/AllenInstitute/deepinterpolation. It is governed by the
licence distributed with that project. This repository contains an independent
PyTorch implementation and does not include the Allen TensorFlow source.

## Neural-network architectures

The implemented model family uses architectural concepts described in the
following publications:

- Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. “U-Net: Convolutional
  Networks for Biomedical Image Segmentation.” *Medical Image Computing and
  Computer-Assisted Intervention*, 2015, pp. 234–241.
  https://doi.org/10.1007/978-3-319-24574-4_28.
- Howard, Andrew G., et al. “MobileNets: Efficient Convolutional Neural Networks
  for Mobile Vision Applications.” arXiv:1704.04861, 2017.
  https://doi.org/10.48550/arXiv.1704.04861.
- Sandler, Mark, et al. “MobileNetV2: Inverted Residuals and Linear
  Bottlenecks.” *Proceedings of the IEEE Conference on Computer Vision and
  Pattern Recognition*, 2018, pp. 4510–4520.
  https://doi.org/10.1109/CVPR.2018.00474.
- Han, Kai, et al. “GhostNet: More Features from Cheap Operations.”
  *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
  Recognition*, 2020, pp. 1580–1589.
  https://doi.org/10.1109/CVPR42600.2020.00165.

The experiment subpackage also uses independently implemented ideas attributed
to the following works:

- Zhang et al., “Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for
  Image Denoising” (DnCNN), arXiv:1608.03981.
- Ding et al., “RepVGG: Making VGG-Style ConvNets Great Again,” CVPR 2021.
- Chu et al., “NAFNet: Simple Baselines for Image Restoration,” ECCV 2022.
- Chen et al., “Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
  Networks” (FasterNet), CVPR 2023.
- Zhang et al., “FFDNet: Toward a Fast and Flexible Solution for CNN-Based
  Image Denoising,” IEEE TIP 2018.
- Zamir et al., “Restormer: Efficient Transformer for High-Resolution Image
  Restoration,” CVPR 2022.
- Tassano et al., “FastDVDnet: Towards Real-Time Deep Video Denoising Without
  Flow Estimation,” CVPR 2020.
- Li et al., “A Simple Baseline for Video Restoration With Grouped
  Spatial-Temporal Shift,” CVPR 2023.

No source code, weights or experiment configurations from these projects are
included. `RepConv2d`, NAF-style blocks, grouped shifts and PixelUnshuffle-based
models were written independently and are covered by numerical or contract
tests in this repository. PyTorch's public `PixelUnshuffle` operator is used.

These publications are cited for method and architecture attribution. The
PyTorch implementation in this repository is released under `LICENSE`.

## Neurofinder data

The optional Neurofinder adapter and source metadata refer users to the public
CodeNeuro distribution channels:

- https://github.com/codeneuro/neurofinder
- https://github.com/codeneuro/datasets

As of 6 August 2026, the authors did not identify an explicit dataset licence
or dataset DOI in those repositories. Public download access does not by itself
grant permission to redistribute the data. No Neurofinder TIFF files, complete
frames or equivalent pixel arrays are included here. Users must obtain the
recordings from the source provider and determine whether their intended use
and redistribution comply with the provider's conditions.

The repository's public evidence includes only non-pixel metrics, metadata and
rendered figures. The portable `source_mapping.public.csv` is a documented
derivative with repository-relative paths; the original machine-local mapping
is excluded. See `docs/data_and_licensing.md` and
`evidence/distribution_manifest.json` for hashes and redaction rules.

## Licence boundary

The MIT License applies only to the original source and documentation in this
repository. It does not supersede or relicense any third-party software,
publication or dataset named above.
