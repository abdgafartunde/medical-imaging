"""CT sparse-view reconstruction demo.

Demonstrates reconstruction from sparse parallel-beam projections
using Tikhonov, total variation (Chambolle-Pock), and wavelet-domain
L1 minimisation (FISTA).  Compares against filtered back-projection
(FBP) from the same sparse views.

Usage
-----
    python examples/02_ct_sparse_view.py

The script saves a side-by-side comparison figure to ``figures/``.
Edit the CONFIG dictionary below to change the number of views or
regularisation parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from medimg.phantoms import shepp_logan
from medimg.forward import ct_forward, ct_adjoint
from medimg.reconstruct import tikhonov_cg, tv_chambolle_pock, cs_fista
from medimg.transforms import wavelet_forward, wavelet_adjoint
from medimg.metrics import psnr, ssim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "image_size": 256,
    "n_angles": 45,              # sparse views (Nyquist ~ 400 for 256²)
    "angle_range": np.pi,        # [0, π) for parallel-beam
    "noise_std": 0.0,            # no noise (CT noise is Poisson; omitted here)
    "seed": 42,
    # Regularisation parameters (tuned for normalised operator, ‖A‖ ≈ 1)
    "tikhonov_lambda": 1e-4,
    "tv_alpha": 5e-4,
    "cs_lambda": 5e-5,
    "cs_wavelet": "haar",
    "cs_level": 4,
    # Algorithm settings
    "tikhonov_iters": 200,
    "tv_iters": 2000,
    "cs_iters": 1500,
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

    # --- 2. Sparse projection angles ---
    angles = np.linspace(0, cfg["angle_range"], cfg["n_angles"],
                         endpoint=False)
    print(f"Using {cfg['n_angles']} projection angles "
          f"(Nyquist ~ {int(np.pi/2 * cfg['image_size'])} for {cfg['image_size']}^2)")

    # --- 3. Forward operator ---
    print("Computing sinogram ...")
    # Raw sinogram for FBP comparison (no normalisation)
    sinogram_raw = ct_forward(phantom, angles, normalize=False)
    # Normalised sinogram for iterative methods (‖A‖ ≈ 1)
    sinogram = ct_forward(phantom, angles, normalize=True)
    print(f"  Sinogram shape: {sinogram.shape}")

    # Linear operators (normalised)
    A  = lambda x: ct_forward(x, angles, normalize=True)
    At = lambda y: ct_adjoint(y, angles, output_size=cfg["image_size"],
                              normalize=True)

    # --- 4. Filtered back-projection (baseline, on raw sinogram) ---
    from skimage.transform import iradon
    # iradon expects (n_detectors, n_angles); raw sinogram is (n_angles, n_detectors)
    x_fbp = iradon(sinogram_raw.T, theta=np.rad2deg(angles),
                   output_size=cfg["image_size"], circle=True,
                   filter_name="ramp")

    # --- 5. Tikhonov (CG) ---
    print(f"Tikhonov CG (lambda={cfg['tikhonov_lambda']:.1e}) ...")
    x_tik = tikhonov_cg(A, At, sinogram, lambd=cfg["tikhonov_lambda"],
                         shape=shape, max_iter=cfg["tikhonov_iters"])

    # --- 6. TV (Chambolle-Pock) ---
    print(f"TV Chambolle-Pock (alpha={cfg['tv_alpha']:.1e}) ...")
    x_tv = tv_chambolle_pock(A, At, sinogram, alpha=cfg["tv_alpha"],
                              shape=shape, max_iter=cfg["tv_iters"],
                              rng=int(rng.integers(0, 2**31)))

    # --- 7. CS (FISTA with wavelet sparsity) ---
    print(f"CS FISTA (lambda={cfg['cs_lambda']:.1e}, "
          f"wavelet={cfg['cs_wavelet']}, level={cfg['cs_level']}) ...")
    def W(x):
        return wavelet_forward(x, wavelet=cfg["cs_wavelet"],
                               level=cfg["cs_level"])
    def Wt(c):
        return wavelet_adjoint(c, wavelet=cfg["cs_wavelet"])
    x_cs = cs_fista(A, At, sinogram, W, Wt, lambd=cfg["cs_lambda"],
                     shape=shape, max_iter=cfg["cs_iters"],
                     rng=int(rng.integers(0, 2**31)))

    # --- 8. Metrics ---
    print("\nReconstruction quality:")
    for name, x in [("FBP", x_fbp), ("Tikhonov", x_tik),
                    ("TV", x_tv), ("CS (wavelet)", x_cs)]:
        print(f"  {name:>14s}: PSNR {psnr(x, phantom):6.2f} dB  "
              f"SSIM {ssim(x, phantom):.4f}")

    # --- 9. Plot ---
    images = [phantom, x_fbp, x_tik, x_tv, x_cs]
    titles = [
        "Ground truth",
        f"FBP\n{psnr(x_fbp, phantom):.1f} dB",
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
        f"Sparse-view CT: {cfg['n_angles']} projection angles "
        f"(parallel-beam, 180° coverage)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    # Save
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/02_ct_sparse_view_comparison.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
