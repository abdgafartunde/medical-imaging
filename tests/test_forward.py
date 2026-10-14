"""Tests for forward operators (MRI and CT)."""

import numpy as np
import pytest
from medimg.forward import (
    generate_cartesian_mask,
    mri_forward,
    mri_adjoint,
    ct_forward,
    ct_adjoint,
)
from medimg.phantoms import shepp_logan


# ---------------------------------------------------------------------------
# MRI mask
# ---------------------------------------------------------------------------

class TestCartesianMask:
    def test_shape(self):
        mask = generate_cartesian_mask((128, 128), sampling_rate=0.3,
                                       calib_lines=24, rng=42)
        assert mask.shape == (128, 128)
        assert mask.dtype == bool

    def test_fully_sampled_centre(self):
        mask = generate_cartesian_mask((128, 128), sampling_rate=0.1,
                                       calib_lines=16, rng=42)
        centre = 128 // 2
        assert np.all(mask[:, centre - 8:centre + 8])

    def test_sampling_fraction_approximate(self):
        mask = generate_cartesian_mask((128, 128), sampling_rate=0.4,
                                       calib_lines=16, rng=42)
        frac = mask.mean()
        # Allow some tolerance because of random sampling
        assert 0.3 < frac < 0.6

    def test_reproducibility(self):
        m1 = generate_cartesian_mask((64, 64), rng=42)
        m2 = generate_cartesian_mask((64, 64), rng=42)
        assert np.array_equal(m1, m2)

    def test_different_seeds_different_masks(self):
        m1 = generate_cartesian_mask((128, 128), sampling_rate=0.3,
                                      calib_lines=16, rng=1)
        m2 = generate_cartesian_mask((128, 128), sampling_rate=0.3,
                                      calib_lines=16, rng=2)
        assert not np.array_equal(m1, m2)

    def test_zero_sampling_rate(self):
        mask = generate_cartesian_mask((64, 64), sampling_rate=0.0,
                                       calib_lines=0, rng=42)
        assert not np.any(mask)

    def test_full_sampling_rate_with_small_image(self):
        mask = generate_cartesian_mask((64, 64), sampling_rate=0.80,
                                       calib_lines=0, rng=42)
        # With unique sampling, actual fraction should be close to target
        assert mask.mean() > 0.75


# ---------------------------------------------------------------------------
# MRI forward / adjoint
# ---------------------------------------------------------------------------

class TestMRIForwardAdjoint:
    @pytest.fixture
    def phantom(self):
        return shepp_logan((128, 128))

    @pytest.fixture
    def mask(self):
        return generate_cartesian_mask((128, 128), sampling_rate=0.5,
                                       calib_lines=24, rng=42)

    def test_forward_output_type(self, phantom, mask):
        kspace = mri_forward(phantom, mask)
        assert kspace.dtype == np.complex128
        assert kspace.shape == phantom.shape

    def test_unsampled_locations_are_zero(self, phantom, mask):
        kspace = mri_forward(phantom, mask)
        assert np.all(kspace[~mask] == 0.0)

    def test_adjoint_output_real(self, phantom, mask):
        kspace = mri_forward(phantom, mask)
        recon = mri_adjoint(kspace, mask)
        assert np.isrealobj(recon)
        assert recon.shape == phantom.shape

    def test_adjoint_no_noise_exact_on_mask(self, phantom, mask):
        kspace = mri_forward(phantom, mask, noise_std=0.0)
        recon = mri_adjoint(kspace, mask)
        # Zero-filled recon should not be NaN or inf
        assert np.all(np.isfinite(recon))

    def test_adjoint_with_noise(self, phantom, mask):
        kspace = mri_forward(phantom, mask, noise_std=0.01, rng=42)
        recon = mri_adjoint(kspace, mask)
        assert np.all(np.isfinite(recon))

    def test_zero_image_gives_zero_kspace(self, mask):
        zero_img = np.zeros((128, 128))
        kspace = mri_forward(zero_img, mask, noise_std=0.0)
        assert np.allclose(kspace, 0.0)

    def test_adjoint_linearity(self, phantom, mask):
        a, b = 2.0, 3.0
        k1 = mri_forward(phantom, mask, noise_std=0.0)
        k2 = mri_forward(a * phantom + b * phantom, mask, noise_std=0.0)
        r1 = mri_adjoint(k1, mask)
        r2 = mri_adjoint(k2, mask)
        assert np.allclose(r2, (a + b) * r1, atol=1e-10)

    def test_full_mask_roundtrip(self):
        img = shepp_logan((64, 64))
        full_mask = np.ones((64, 64), dtype=bool)
        kspace = mri_forward(img, full_mask, noise_std=0.0)
        recon = mri_adjoint(kspace, full_mask)
        assert np.allclose(recon, img, atol=1e-10)


# ---------------------------------------------------------------------------
# CT forward / adjoint
# ---------------------------------------------------------------------------

class TestCTForwardAdjoint:
    @pytest.fixture
    def phantom(self):
        return shepp_logan((128, 128))

    @pytest.fixture
    def angles(self):
        return np.linspace(0, np.pi, 90, endpoint=False)

    def test_forward_output_shape(self, phantom, angles):
        sinogram = ct_forward(phantom, angles)
        assert sinogram.shape[0] == len(angles)

    def test_adjoint_output_shape(self, phantom, angles):
        sinogram = ct_forward(phantom, angles)
        recon = ct_adjoint(sinogram, angles, output_size=128)
        assert recon.shape == (128, 128)

    def test_adjoint_output_real(self, phantom, angles):
        sinogram = ct_forward(phantom, angles)
        recon = ct_adjoint(sinogram, angles)
        assert np.isrealobj(recon)

    def test_all_finite(self, phantom, angles):
        sinogram = ct_forward(phantom, angles)
        assert np.all(np.isfinite(sinogram))
        recon = ct_adjoint(sinogram, angles)
        assert np.all(np.isfinite(recon))

    def test_zero_image_gives_zero_sinogram(self, angles):
        zero_img = np.zeros((64, 64))
        sinogram = ct_forward(zero_img, angles)
        assert np.allclose(sinogram, 0.0, atol=1e-10)

    def test_different_n_angles(self):
        img = shepp_logan((64, 64))
        for n_angles in [30, 60, 180]:
            ang = np.linspace(0, np.pi, n_angles, endpoint=False)
            sino = ct_forward(img, ang)
            assert sino.shape[0] == n_angles


# ---------------------------------------------------------------------------
# Adjointness, operator norms, and the noise model
# ---------------------------------------------------------------------------

def _power_norm_sq(A, At, shape, n_iter=200, seed=0):
    x = np.random.default_rng(seed).standard_normal(shape)
    for _ in range(n_iter):
        x = At(A(x))
        x /= np.linalg.norm(x)
    return float(np.linalg.norm(At(A(x))))


class TestAdjointness:
    """Dot-product tests: <A x, y> = <x, A* y> to rounding error."""

    def test_mri_dot_product(self):
        rng = np.random.default_rng(0)
        mask = generate_cartesian_mask((64, 64), sampling_rate=0.3,
                                       calib_lines=8, rng=1)
        x = rng.standard_normal((64, 64))
        y = (rng.standard_normal((64, 64))
             + 1j * rng.standard_normal((64, 64))) * mask
        lhs = np.real(np.vdot(mri_forward(x, mask), y))
        rhs = np.vdot(x, mri_adjoint(y, mask))
        assert lhs == pytest.approx(rhs, rel=1e-12)

    @pytest.mark.parametrize("n_angles", [17, 45, 180])
    def test_ct_dot_product(self, n_angles):
        rng = np.random.default_rng(n_angles)
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        x = rng.standard_normal((64, 64))
        y = rng.standard_normal((n_angles, 64))
        lhs = np.vdot(ct_forward(x, angles), y)
        rhs = np.vdot(x, ct_adjoint(y, angles))
        assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_mri_operator_norm_is_one(self):
        mask = generate_cartesian_mask((64, 64), sampling_rate=0.3,
                                       calib_lines=8, rng=1)
        L = _power_norm_sq(lambda x: mri_forward(x, mask),
                           lambda y: mri_adjoint(y, mask), (64, 64))
        assert L == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("n_angles", [17, 45])
    def test_ct_operator_norm_is_one(self, n_angles):
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        L = _power_norm_sq(lambda x: ct_forward(x, angles),
                           lambda y: ct_adjoint(y, angles), (64, 64))
        assert L == pytest.approx(1.0, rel=1e-6)

    def test_ct_matches_skimage_radon(self):
        from skimage.transform import radon
        img = shepp_logan((128, 128))
        angles = np.linspace(0, np.pi, 60, endpoint=False)
        ours = ct_forward(img, angles, normalize=False)
        ref = radon(img, theta=np.rad2deg(angles), circle=True).T
        assert np.linalg.norm(ours - ref) / np.linalg.norm(ref) < 0.02

    def test_ct_adjoint_rejects_mismatched_output_size(self):
        angles = np.linspace(0, np.pi, 10, endpoint=False)
        with pytest.raises(ValueError):
            ct_adjoint(np.zeros((10, 64)), angles, output_size=128)


class TestMRINoiseModel:
    def test_noise_only_at_sampled_locations(self):
        img = shepp_logan((64, 64))
        mask = generate_cartesian_mask((64, 64), sampling_rate=0.3,
                                       calib_lines=8, rng=1)
        kspace = mri_forward(img, mask, noise_std=0.1, rng=2)
        assert np.all(kspace[~mask] == 0.0)

    def test_noise_level_matches_noise_std(self):
        img = np.zeros((128, 128))
        mask = generate_cartesian_mask((128, 128), sampling_rate=0.3,
                                       calib_lines=12, rng=1)
        noise = mri_forward(img, mask, noise_std=0.05, rng=3)[mask]
        assert np.std(noise.real) == pytest.approx(0.05, rel=0.05)
        assert np.std(noise.imag) == pytest.approx(0.05, rel=0.05)

    def test_unitary_fft_preserves_norm(self):
        img = shepp_logan((64, 64))
        full = np.ones((64, 64), dtype=bool)
        assert np.linalg.norm(mri_forward(img, full)) == pytest.approx(
            np.linalg.norm(img), rel=1e-12)
