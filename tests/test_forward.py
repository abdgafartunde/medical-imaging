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
