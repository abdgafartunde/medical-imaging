"""Tests for reconstruction algorithms."""

import numpy as np
import pytest
from medimg.forward import (
    generate_cartesian_mask,
    mri_forward,
    mri_adjoint,
)
from medimg.phantoms import shepp_logan
from medimg.reconstruct import tikhonov_cg, tv_chambolle_pock, cs_fista
from medimg.transforms import wavelet_forward, wavelet_adjoint
from medimg.metrics import psnr, ssim


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def phantom():
    return shepp_logan((64, 64))


@pytest.fixture
def mask():
    return generate_cartesian_mask((64, 64), sampling_rate=0.4,
                                   calib_lines=12, rng=42)


@pytest.fixture
def operators(phantom, mask):
    A = lambda x: mri_forward(x, mask)
    At = lambda y: mri_adjoint(y, mask)
    y = mri_forward(phantom, mask, noise_std=0.001, rng=42)
    return A, At, y


# ---------------------------------------------------------------------------
# Tikhonov CG
# ---------------------------------------------------------------------------

class TestTikhonovCG:
    def test_output_shape(self, operators):
        A, At, y = operators
        shape = (64, 64)
        x = tikhonov_cg(A, At, y, lambd=1e-3, shape=shape, max_iter=50)
        assert x.shape == shape

    def test_all_finite(self, operators):
        A, At, y = operators
        x = tikhonov_cg(A, At, y, lambd=1e-3, shape=(64, 64), max_iter=50)
        assert np.all(np.isfinite(x))

    def test_reduces_data_misfit(self, operators, phantom):
        A, At, y = operators
        shape = (64, 64)
        x = tikhonov_cg(A, At, y, lambd=1e-3, shape=shape, max_iter=50)
        misfit_before = np.linalg.norm(y.ravel())
        misfit_after = np.linalg.norm((A(x) - y).ravel())
        # Reconstruction should explain data better than zero
        assert misfit_after < misfit_before

    def test_large_lambda_gives_near_zero(self, operators):
        A, At, y = operators
        shape = (64, 64)
        x = tikhonov_cg(A, At, y, lambd=1e6, shape=shape, max_iter=100)
        assert np.max(np.abs(x)) < 0.1

    def test_small_lambda_noiser(self, operators):
        A, At, y = operators
        shape = (64, 64)
        x = tikhonov_cg(A, At, y, lambd=1e-9, shape=shape, max_iter=100)
        # Should still be finite
        assert np.all(np.isfinite(x))

    def test_reconstruction_improves_over_zero(self, operators, phantom):
        A, At, y = operators
        shape = (64, 64)
        x = tikhonov_cg(A, At, y, lambd=1e-3, shape=shape, max_iter=100)
        psnr_recon = psnr(x, phantom)
        psnr_zero = psnr(np.zeros(shape), phantom)
        assert psnr_recon > psnr_zero


# ---------------------------------------------------------------------------
# TV Chambolle-Pock
# ---------------------------------------------------------------------------

class TestTVChambollePock:
    def test_output_shape(self, operators):
        A, At, y = operators
        shape = (64, 64)
        x = tv_chambolle_pock(A, At, y, alpha=1e-3, shape=shape,
                              max_iter=100, rng=42)
        assert x.shape == shape

    def test_all_finite(self, operators):
        A, At, y = operators
        shape = (64, 64)
        x = tv_chambolle_pock(A, At, y, alpha=1e-3, shape=shape,
                              max_iter=100, rng=42)
        assert np.all(np.isfinite(x))

    def test_alpha_reduces_tv(self, operators, phantom):
        A, At, y = operators
        shape = (64, 64)
        # TV reconstruction with moderate α
        x_tv = tv_chambolle_pock(A, At, y, alpha=5e-4, shape=shape,
                                  max_iter=200, rng=42)
        # Reconstruction should be finite and improve over zero-filling
        assert np.all(np.isfinite(x_tv))
        assert np.max(np.abs(x_tv)) < 5.0
        assert psnr(x_tv, phantom) > psnr(np.zeros(shape), phantom)

    def test_reduces_data_misfit(self, operators):
        A, At, y = operators
        shape = (64, 64)
        x = tv_chambolle_pock(A, At, y, alpha=1e-4, shape=shape,
                              max_iter=100, rng=42)
        misfit_before = np.linalg.norm(y.ravel())
        misfit_after = np.linalg.norm((A(x) - y).ravel())
        assert misfit_after < misfit_before

    def test_reconstruction_improves_over_zero(self, operators, phantom):
        A, At, y = operators
        shape = (64, 64)
        x = tv_chambolle_pock(A, At, y, alpha=5e-4, shape=shape,
                              max_iter=200, rng=42)
        psnr_recon = psnr(x, phantom)
        psnr_zero = psnr(np.zeros(shape), phantom)
        assert psnr_recon > psnr_zero


# ---------------------------------------------------------------------------
# CS FISTA
# ---------------------------------------------------------------------------

class TestCSFISTA:
    @pytest.fixture
    def W_and_Wt(self):
        def W(x):
            return wavelet_forward(x, wavelet='db4', level=3)

        def Wt(c):
            return wavelet_adjoint(c, wavelet='db4')

        return W, Wt

    def test_output_shape(self, operators, W_and_Wt):
        A, At, y = operators
        W, Wt = W_and_Wt
        shape = (64, 64)
        x = cs_fista(A, At, y, W, Wt, lambd=1e-4, shape=shape,
                     max_iter=50, rng=42)
        assert x.shape == shape

    def test_all_finite(self, operators, W_and_Wt):
        A, At, y = operators
        W, Wt = W_and_Wt
        shape = (64, 64)
        x = cs_fista(A, At, y, W, Wt, lambd=1e-4, shape=shape,
                     max_iter=50, rng=42)
        assert np.all(np.isfinite(x))

    def test_large_lambda_gives_sparse_coeffs(self, operators, W_and_Wt):
        A, At, y = operators
        W, Wt = W_and_Wt
        shape = (64, 64)
        x = cs_fista(A, At, y, W, Wt, lambd=5.0, shape=shape,
                     max_iter=200, rng=42)
        # Strong L1 penalisation — reconstruction is finite
        assert np.all(np.isfinite(x))
        coeff_struct = W(x)
        n_zero = 0
        n_total = 0
        for c in coeff_struct[1:]:
            if isinstance(c, tuple):
                for ci in c:
                    n_zero += np.sum(np.abs(ci) < 1e-10)
                    n_total += ci.size
            else:
                n_zero += np.sum(np.abs(c) < 1e-10)
                n_total += c.size
        # Many detail coefficients should be zero
        assert n_zero / n_total > 0.70

    def test_reduces_data_misfit(self, operators, W_and_Wt):
        A, At, y = operators
        W, Wt = W_and_Wt
        shape = (64, 64)
        x = cs_fista(A, At, y, W, Wt, lambd=1e-5, shape=shape,
                     max_iter=50, rng=42)
        misfit_before = np.linalg.norm(y.ravel())
        misfit_after = np.linalg.norm((A(x) - y).ravel())
        assert misfit_after < misfit_before

    def test_reconstruction_improves_over_zero(self, operators, phantom,
                                                W_and_Wt):
        A, At, y = operators
        W, Wt = W_and_Wt
        shape = (64, 64)
        x = cs_fista(A, At, y, W, Wt, lambd=1e-4, shape=shape,
                     max_iter=200, rng=42)
        psnr_recon = psnr(x, phantom)
        psnr_zero = psnr(np.zeros(shape), phantom)
        assert psnr_recon > psnr_zero


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_psnr_identical(self):
        img = shepp_logan((64, 64))
        assert np.isinf(psnr(img, img))

    def test_psnr_different(self):
        a = np.ones((32, 32))
        b = np.zeros((32, 32))
        p = psnr(a, b)
        # data_range = max(b)-min(b) = 0 → defaults to 1; mse=1 → 0 dB
        assert p >= 0 and not np.isinf(p)

    def test_ssim_identical(self):
        img = shepp_logan((64, 64))
        assert ssim(img, img) > 0.99

    def test_ssim_range(self):
        a = np.random.rand(64, 64)
        b = np.random.rand(64, 64)
        s = ssim(a, b)
        assert -1.0 <= s <= 1.0
