"""Sparsifying transforms for compressed sensing reconstruction.

Provides wavelet and soft-thresholding operators used as sparsity
priors in CS-MRI and CS-CT.

References
----------
- Daubechies, Defrise & De Mol (2004). An iterative thresholding algorithm
  for linear inverse problems with a sparsity constraint. *CPAM*, 57(11),
  1413–1457.
"""

import numpy as np
import pywt


def wavelet_forward(x, wavelet='db4', level=4):
    """Forward 2-D discrete wavelet transform (analysis operator).

    Uses periodic extension (``mode="periodization"``), so for an
    orthogonal wavelet (``haar``, ``dbN``, ``symN``, ``coifN``) and image
    sides divisible by ``2**level`` the transform is orthogonal:
    ``||W x|| = ||x||`` and :func:`wavelet_adjoint` is both its inverse and
    its adjoint.  This is what makes the soft-thresholding step in
    :func:`medimg.reconstruct.cs_fista` the exact proximal map.

    Returns the PyWavelets coefficient structure (a list of numpy
    arrays) which can be passed directly to ``wavelet_adjoint`` or
    thresholded with ``soft_threshold_coeffs``.

    Parameters
    ----------
    x : ndarray of shape (n_rows, n_cols)
        Input image.
    wavelet : str
        Wavelet name (PyWavelets convention).  Default ``'db4'``
        (Daubechies-4), standard in CS-MRI.
    level : int
        Decomposition levels.

    Returns
    -------
    coeff_struct : list
        PyWavelets coefficient structure.  ``coeff_struct[0]`` is the
        approximation coefficients; ``coeff_struct[1:]`` are the detail
        coefficients at each level (each a 3-tuple of (LH, HL, HH)
        sub-bands).
    """
    n_rows, n_cols = np.shape(x)
    if n_rows % 2**level or n_cols % 2**level:
        raise ValueError(
            f"image shape {np.shape(x)} is not divisible by 2**level = {2**level}; "
            "the transform would not be orthogonal (use a smaller level or pad the image)"
        )
    return pywt.wavedec2(x, wavelet=wavelet, level=level, mode="periodization")


def wavelet_adjoint(coeff_struct, wavelet='db4'):
    """Adjoint (inverse) 2-D discrete wavelet transform (synthesis).

    Reconstructs an image from a PyWavelets coefficient structure, with
    the same periodic extension as :func:`wavelet_forward`, so it is the
    exact adjoint of that transform for orthogonal wavelets.

    Parameters
    ----------
    coeff_struct : list
        Coefficient structure as returned by ``wavelet_forward`` or
        ``soft_threshold_coeffs``.
    wavelet : str
        Wavelet name matching the forward transform.

    Returns
    -------
    x : ndarray
        Reconstructed image.
    """
    return pywt.waverec2(coeff_struct, wavelet=wavelet, mode="periodization")


def soft_threshold(x, lambd):
    r"""Element-wise soft-thresholding (proximal operator of the L1 norm).

    .. math::

        S_\lambda(x) = \operatorname{sign}(x) \cdot \max(|x| - \lambda, 0)

    Parameters
    ----------
    x : ndarray
        Input array.
    lambd : float
        Threshold.

    Returns
    -------
    ndarray
        Thresholded array.
    """
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0.0)


def soft_threshold_coeffs(coeff_struct, lambd):
    """Apply soft-thresholding to wavelet detail coefficients only.

    The approximation coefficients (``coeff_struct[0]``) are kept
    unchanged.  Each detail sub-band is thresholded element-wise.

    Parameters
    ----------
    coeff_struct : list
        PyWavelets coefficient structure from ``wavelet_forward``.
    lambd : float
        Threshold.

    Returns
    -------
    list
        Thresholded coefficient structure (can be passed to
        ``wavelet_adjoint``).
    """
    result = [coeff_struct[0]]  # keep approximation unchanged
    for c in coeff_struct[1:]:
        if isinstance(c, tuple):
            result.append(tuple(soft_threshold(ci, lambd) for ci in c))
        else:
            result.append(soft_threshold(c, lambd))
    return result
