"""MRI compressed sensing reconstruction demo.

Demonstrates variable-density k-space under-sampling and reconstruction
with Tikhonov, total variation (Chambolle-Pock), and wavelet-domain L1
minimisation (FISTA).

Usage
-----
    python examples/01_mri_cs.py            # 256x256, a few seconds
    python examples/01_mri_cs.py --quick    # 128x128

The script runs from any directory and saves a side-by-side comparison
figure to ``figures/`` in the repository (or ``--outdir DIR``).  Noise is
added at the sampled k-space locations so that the data SNR is
``snr_db`` (30 dB by default).  Regularisation parameters were chosen by
a coarse search for the best PSNR on this phantom.
Edit the CONFIG dictionary below to change the acceleration factor,
noise level, or regularisation parameters.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
    "snr_db": 30.0,              # SNR of the sampled k-space data
    "seed": 42,
    # Regularisation parameters (unitary FFT, ‖A‖ = 1; 4× acceleration, 30 dB)
    "tikhonov_lambda": 1e-3,
    "tv_alpha": 3e-3,
    "cs_lambda": 3e-3,
    "cs_wavelet": "haar",
    "cs_level": 4,
    # Algorithm settings
    "tikhonov_iters": 200,
    "tv_iters": 2000,
    "cs_iters": 1000,
}

QUICK = {"image_size": 128, "calib_lines": 12, "tv_iters": 500, "cs_iters": 500}

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="128x128 image and fewer iterations")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "figures",
                        help="directory for the figure (default: figures/ in the repository)")
    parser.add_argument("--show", action="store_true",
                        help="also open the figure in a window")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = {**CONFIG, **(QUICK if args.quick else {})}
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
    clean = mri_forward(phantom, mask)
    signal_rms = np.sqrt(np.mean(np.abs(clean[mask]) ** 2))
    noise_std = signal_rms / (np.sqrt(2.0) * 10 ** (cfg["snr_db"] / 20))
    kspace = mri_forward(phantom, mask, noise_std=noise_std,
                         rng=int(rng.integers(0, 2**31)))
    snr = 20 * np.log10(np.linalg.norm(clean[mask])
                        / np.linalg.norm((kspace - clean)[mask]))
    print(f"  Noise std {noise_std:.2e} per real/imag part; data SNR {snr:.1f} dB")

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
        f"data SNR {snr:.0f} dB",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    # Save
    args.outdir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        out_path = args.outdir / ("01_mri_cs_comparison" + suffix)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
