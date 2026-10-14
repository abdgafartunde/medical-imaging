"""Tests for the wavelet transforms used as sparsity priors."""

import numpy as np
import pytest
import pywt

from medimg.transforms import soft_threshold, wavelet_adjoint, wavelet_forward


@pytest.mark.parametrize("wavelet, level", [("haar", 4), ("db4", 3), ("sym8", 2)])
def test_wavelet_transform_is_orthogonal(wavelet, level):
    """W is orthogonal, so wavelet_adjoint is its exact adjoint and inverse.

    With PyWavelets' default 'symmetric' extension, db4 and sym8 produce
    redundant coefficients and ||W x|| != ||x||; cs_fista's thresholding step
    is then not the exact proximal map.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64))
    arr, slices = pywt.coeffs_to_array(wavelet_forward(x, wavelet, level))
    assert arr.size == x.size
    assert np.linalg.norm(arr) == pytest.approx(np.linalg.norm(x), rel=1e-12)

    c = rng.standard_normal(arr.shape)
    coeffs = pywt.array_to_coeffs(c, slices, output_format="wavedec2")
    assert np.vdot(arr, c) == pytest.approx(
        np.vdot(x, wavelet_adjoint(coeffs, wavelet)), rel=1e-12)
    np.testing.assert_allclose(
        wavelet_adjoint(wavelet_forward(x, wavelet, level), wavelet), x, atol=1e-12)


def test_soft_threshold():
    x = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(soft_threshold(x, 1.0), [-2.0, 0.0, 0.0, 0.0, 2.0])
