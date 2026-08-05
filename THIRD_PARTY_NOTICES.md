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

## Licence boundary

The MIT License applies only to the original source and documentation in this
repository. It does not supersede or relicense any third-party software,
publication or dataset named above.
