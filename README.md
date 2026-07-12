# Medical Imaging

Compressed sensing and model-based reconstruction for accelerated MRI and CT.

**Status:** Under development ⸺ expected Q1 2027.

## Planned

- Compressed sensing MRI reconstruction with total variation and wavelet
  sparsity priors
- CT reconstruction from sparse-view projections (filtered back-projection,
  iterative SIRT/SART)
- Learned proximal operators for plug-and-play reconstruction
- Comparison with classical regularisation (Tikhonov, TV)

## Motivation

Accelerated acquisition in MRI and CT reduces scan time and radiation dose, but
the resulting under-sampled data lead to an ill-posed inverse problem.
Compressed sensing theory guarantees exact recovery under sparsity assumptions;
practical reconstruction requires careful choice of sparsifying transform and
optimisation algorithm. This project provides reference implementations of
classical and learning-based methods for image reconstruction from incomplete
measurements.

## References

- Lustig, M., Donoho, D. & Pauly, J.M. (2007). Sparse MRI: the application of
  compressed sensing for rapid MR imaging. *Magnetic Resonance in Medicine*,
  58(6), 1182--1195.
- Sidky, E.Y. & Pan, X. (2008). Image reconstruction in circular cone-beam
  computed tomography by constrained, total-variation minimization. *Physics in
  Medicine and Biology*, 53(17), 4777--4807.
