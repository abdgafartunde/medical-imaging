"""MRI compressed sensing reconstruction demo.

Demonstrates variable-density k-space under-sampling and reconstruction
with Tikhonov, total variation (Chambolle-Pock), and wavelet-domain L1
minimisation (FISTA).

Usage
-----
    python examples/01_mri_cs.py

The script saves a side-by-side comparison figure to ``figures/``.
Edit the CONFIG dictionary below to change the acceleration factor,
noise level, or regularisation parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from medimg.phantoms import shepp_logan
from medimg.forward import generate_cartesian_mask, mri_forward, mri_adjoint
from medimg.reconstruct import tikhonov_cg, tv_chambolle_pock, cs_fista
from medimg.transforms import wavelet_forward, wavelet_adjoint
from medimg.metrics import psnr, ssim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "image_size": 256,
    "sampling_rate": 0.25,       # 4× acceleration
    "calib_lines": 24,           # fully-sampled central lines
    "noise_std": 0.001,          # complex Gaussian noise in k-space
    "seed": 42,
    # Regularisation parameters (tuned for 4× acceleration at 256²)
    "tikhonov_lambda": 5e-4,
    "tv_alpha": 5e-4,
    "cs_lambda": 1e-4,
    "cs_wavelet": "haar",
    "cs_level": 4,
    # Algorithm settings
    "tikhonov_iters": 200,
    "tv_iters": 2000,
    "cs_iters": 1000,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = CONFIG
    rng = np.random.default_rng(cfg["seed"])
    shape = (cfg["image_size"], cfg["image_size"])

    # --- 1. Ground-truth phantom ---
    print("Generating phantom ...")
    phantom = shepp_logan(shape)

    # --- 2. Variable-density under-sampling mask ---
    print(f"Generating {cfg['sampling_rate']:.0%} sampling mask "
          f"({1/cfg['sampling_rate']:.1f}× acceleration) ...")
    mask = generate_cartesian_mask(
        shape,
        sampling_rate=cfg["sampling_rate"],
        calib_lines=cfg["calib_lines"],
        rng=int(rng.integers(0, 2**31)),
    )
    print(f"  Fraction sampled: {mask.mean():.2%}")

    # --- 3. Forward operator ---
    print("Applying forward operator ...")
    kspace = mri_forward(phantom, mask, noise_std=cfg["noise_std"],
                         rng=int(rng.integers(0, 2**31)))

    # Linear operators
    A  = lambda x: mri_forward(x, mask)
    At = lambda y: mri_adjoint(y, mask)

    # Zero-filled reconstruction (adjoint only)
    x_zf = At(kspace)

    # --- 4. Tikhonov (CG) ---
    print(f"Tikhonov CG (lambda={cfg['tikhonov_lambda']:.1e}) ...")
    x_tik = tikhonov_cg(A, At, kspace, lambd=cfg["tikhonov_lambda"],
                         shape=shape, max_iter=cfg["tikhonov_iters"])

    # --- 5. TV (Chambolle-Pock) ---
    print(f"TV Chambolle-Pock (alpha={cfg['tv_alpha']:.1e}) ...")
    x_tv = tv_chambolle_pock(A, At, kspace, alpha=cfg["tv_alpha"],
                              shape=shape, max_iter=cfg["tv_iters"],
                              rng=int(rng.integers(0, 2**31)))

    # --- 6. CS (FISTA with wavelet sparsity) ---
    print(f"CS FISTA (lambda={cfg['cs_lambda']:.1e}, "
          f"wavelet={cfg['cs_wavelet']}, level={cfg['cs_level']}) ...")
    def W(x):
        return wavelet_forward(x, wavelet=cfg["cs_wavelet"],
                               level=cfg["cs_level"])
    def Wt(c):
        return wavelet_adjoint(c, wavelet=cfg["cs_wavelet"])
    x_cs = cs_fista(A, At, kspace, W, Wt, lambd=cfg["cs_lambda"],
                     shape=shape, max_iter=cfg["cs_iters"],
                     rng=int(rng.integers(0, 2**31)))

    # --- 7. Metrics ---
    print("\nReconstruction quality:")
    for name, x in [("Zero-filled", x_zf), ("Tikhonov", x_tik),
                    ("TV", x_tv), ("CS (wavelet)", x_cs)]:
        print(f"  {name:>14s}: PSNR {psnr(x, phantom):6.2f} dB  "
              f"SSIM {ssim(x, phantom):.4f}")

    # --- 8. Plot ---
    images = [phantom, x_zf, x_tik, x_tv, x_cs]
    titles = [
        "Ground truth",
        f"Zero-filled\n{psnr(x_zf, phantom):.1f} dB",
        f"Tikhonov\n{psnr(x_tik, phantom):.1f} dB",
        f"TV\n{psnr(x_tv, phantom):.1f} dB",
        f"CS (wavelet)\n{psnr(x_cs, phantom):.1f} dB",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    vmin, vmax = phantom.min(), phantom.max()
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"CS-MRI: {cfg['sampling_rate']:.0%} sampling "
        f"({1/cfg['sampling_rate']:.1f}× acceleration), "
        f"σ = {cfg['noise_std']:.0e}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    # Save
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/01_mri_cs_comparison.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
