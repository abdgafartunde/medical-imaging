"""Golden-output tests for the quick configurations of both examples.

The pipelines below match ``examples/01_mri_cs.py --quick`` and
``examples/02_ct_sparse_view.py --quick`` (phantom, sampling, noise, and
regularisation parameters).  Expected PSNR and SSIM values are stored in
``tests/golden/quick_examples.json``; if a change is meant to alter them,
regenerate the file with

    python tests/test_golden.py

and explain the change in the commit message.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from medimg.forward import (
    ct_adjoint,
    ct_forward,
    generate_cartesian_mask,
    mri_adjoint,
    mri_forward,
)
from medimg.metrics import psnr, ssim
from medimg.phantoms import shepp_logan
from medimg.reconstruct import cs_fista, tikhonov_cg, tv_chambolle_pock
from medimg.transforms import wavelet_adjoint, wavelet_forward

GOLDEN = Path(__file__).parent / "golden" / "quick_examples.json"
N = 128


def _reconstruct(A, At, y, shape, lambdas, rng):
    def W(x):
        return wavelet_forward(x, wavelet="haar", level=4)

    def Wt(c):
        return wavelet_adjoint(c, wavelet="haar")

    return {
        "tikhonov": tikhonov_cg(A, At, y, lambd=lambdas[0], shape=shape,
                                max_iter=200),
        "tv": tv_chambolle_pock(A, At, y, alpha=lambdas[1], shape=shape,
                                max_iter=500, rng=int(rng.integers(0, 2**31))),
        "cs": cs_fista(A, At, y, W, Wt, lambd=lambdas[2], shape=shape,
                       max_iter=500, rng=int(rng.integers(0, 2**31))),
    }


def run_mri():
    rng = np.random.default_rng(42)
    phantom = shepp_logan((N, N))
    mask = generate_cartesian_mask((N, N), sampling_rate=0.25, calib_lines=12,
                                   rng=int(rng.integers(0, 2**31)))
    clean = mri_forward(phantom, mask)
    noise_std = (np.sqrt(np.mean(np.abs(clean[mask]) ** 2))
                 / (np.sqrt(2.0) * 10 ** (30.0 / 20)))
    y = mri_forward(phantom, mask, noise_std=noise_std,
                    rng=int(rng.integers(0, 2**31)))
    recons = _reconstruct(lambda x: mri_forward(x, mask),
                          lambda k: mri_adjoint(k, mask), y, (N, N),
                          (1e-3, 3e-3, 3e-3), rng)
    recons["zero_filled"] = mri_adjoint(y, mask)
    return {name: [psnr(x, phantom), ssim(x, phantom)]
            for name, x in recons.items()}


def run_ct():
    rng = np.random.default_rng(42)
    phantom = shepp_logan((N, N))
    angles = np.linspace(0, np.pi, 30, endpoint=False)
    clean = ct_forward(phantom, angles)
    y = clean + 0.01 * np.sqrt(np.mean(clean ** 2)) * rng.standard_normal(clean.shape)
    recons = _reconstruct(lambda x: ct_forward(x, angles),
                          lambda s: ct_adjoint(s, angles), y, (N, N),
                          (3e-4, 1e-3, 3e-4), rng)
    return {name: [psnr(x, phantom), ssim(x, phantom)]
            for name, x in recons.items()}


@pytest.fixture(scope="module")
def expected():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("modality, run", [("mri", run_mri), ("ct", run_ct)])
def test_quick_example_metrics(expected, modality, run):
    results = run()
    assert results.keys() == expected[modality].keys()
    for name, (p, s) in results.items():
        p_ref, s_ref = expected[modality][name]
        assert p == pytest.approx(p_ref, abs=1e-3), f"{modality} {name} PSNR"
        assert s == pytest.approx(s_ref, abs=1e-4), f"{modality} {name} SSIM"


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps({"mri": run_mri(), "ct": run_ct()}, indent=2) + "\n")
    print(f"Wrote {GOLDEN}")
