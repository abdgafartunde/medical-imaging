# Figures

Sample results, committed so the README and the tech note can show them.
Regenerate them from the repository root with

```bash
python examples/01_mri_cs.py
python examples/02_ct_sparse_view.py
```

Each figure is saved as PNG (for the README) and PDF (for papers).

- `01_mri_cs_comparison`: 256 x 256 Shepp-Logan phantom, 25% variable-density
  Cartesian sampling (4x acceleration), complex Gaussian noise at a 30 dB data
  SNR. PSNR: zero-filled 21.2 dB, Tikhonov 22.1 dB, TV 44.0 dB,
  CS (Haar wavelets) 34.7 dB.
- `02_ct_sparse_view_comparison`: 256 x 256 Shepp-Logan phantom, 45
  parallel-beam views over 180 degrees, 1% Gaussian noise (40 dB SNR). PSNR:
  FBP 22.9 dB, Tikhonov 25.3 dB, TV 31.9 dB, CS 27.7 dB.

Regularisation parameters were chosen by a coarse search for the best PSNR on
this phantom, so the numbers are best cases for it.
