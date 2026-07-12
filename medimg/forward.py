"""Forward operators for accelerated MRI and sparse-view CT.

Provides the sensing models and their adjoints:

- MRI:   Fourier encoding with variable-density Cartesian under-sampling.
- CT:    Parallel-beam Radon transform with sparse angular sampling.

References
----------
- Lustig, Donoho & Pauly (2007). Sparse MRI. *MRM*, 58(6), 1182–1195.
- Sidky & Pan (2008). Image reconstruction in circular cone-beam CT by
  constrained, total-variation minimization. *PMB*, 53(17), 4777–4807.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


# ---------------------------------------------------------------------------
# MRI
# ---------------------------------------------------------------------------

def generate_cartesian_mask(shape, sampling_rate=0.30, calib_lines=24,
                            rng=None):
    """Generate a variable-density Cartesian k-space under-sampling mask.

    Fully samples a central calibration band (low spatial frequencies)
    and randomly sub-samples higher frequencies with probability decaying
    as a polynomial in distance from the k-space centre.  This follows
    the variable-density scheme of Lustig et al. (2007).

    Parameters
    ----------
    shape : tuple of int
        (n_rows, n_cols) of k-space.
    sampling_rate : float
        Approximate fraction of k-space samples to retain (excluding the
        fully-sampled calibration region).
    calib_lines : int
        Number of contiguous central phase-encode lines to sample fully.
    rng : int or numpy.random.Generator, optional
        Random seed or generator instance.

    Returns
    -------
    mask : ndarray of bool, shape *shape*
        ``True`` where k-space is sampled.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    n_rows, n_cols = shape
    mask = np.zeros(shape, dtype=bool)

    centre = n_cols // 2
    half_calib = calib_lines // 2
    lo = centre - half_calib
    hi = centre + half_calib
    mask[:, lo:hi] = True

    n_calib = calib_lines * n_rows
    n_total = int(sampling_rate * n_rows * n_cols)
    n_random = max(0, n_total - n_calib)

    if n_random > 0:
        # Build probability for each (row, col) — depends only on col
        cols = np.arange(n_cols)
        dist = np.abs(cols - centre) / max(centre, 1)
        prob_col = (1.0 - dist ** 2) ** 2
        prob_col[lo:hi] = 0.0
        prob_col /= prob_col.sum()

        # Tile to 2-D
        prob_2d = np.tile(prob_col, (n_rows, 1)).ravel()

        # Sample pixel indices without replacement
        all_indices = np.arange(n_rows * n_cols)
        calib_indices = (all_indices.reshape(n_rows, n_cols)[:, lo:hi]
                         .ravel())
        eligible = np.setdiff1d(all_indices, calib_indices)

        n_random = min(n_random, len(eligible))
        chosen = rng.choice(eligible, size=n_random, replace=False,
                            p=prob_2d[eligible] / prob_2d[eligible].sum())
        mask.ravel()[chosen] = True

    return mask


def mri_forward(image, mask, noise_std=0.0, rng=None):
    """MRI forward operator: 2-D FFT + k-space under-sampling + noise.

    Parameters
    ----------
    image : ndarray of shape (n_rows, n_cols)
        Real-valued ground-truth image.
    mask : ndarray of bool, shape (n_rows, n_cols)
        k-space sampling mask.
    noise_std : float
        Standard deviation of complex white Gaussian noise added in k-space.
    rng : int or numpy.random.Generator, optional

    Returns
    -------
    kspace : ndarray of complex, shape (n_rows, n_cols)
        Under-sampled (and possibly noisy) k-space data.  Unsampled
        locations are zero.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    k_full = fftshift(fft2(ifftshift(image)))
    kspace = k_full * mask

    if noise_std > 0:
        noise = (rng.normal(0, noise_std, kspace.shape)
                 + 1j * rng.normal(0, noise_std, kspace.shape))
        kspace += noise

    return kspace


def mri_adjoint(kspace, mask):
    """Adjoint MRI operator: masked inverse FFT (zero-filled reconstruction).

    Parameters
    ----------
    kspace : ndarray of complex, shape (n_rows, n_cols)
    mask : ndarray of bool, shape (n_rows, n_cols)

    Returns
    -------
    image : ndarray, shape (n_rows, n_cols)
        Real part of the zero-filled inverse Fourier transform.
    """
    return np.real(fftshift(ifft2(ifftshift(kspace * mask))))


# ---------------------------------------------------------------------------
# CT
# ---------------------------------------------------------------------------

# Theoretical spectral norm of the continuous Radon transform for
# n_angles uniformly spaced over [0, π) is √(π · n_angles).
# We normalise by this factor so that ‖A‖ ≈ 1, matching the MRI
# scaling and making regularisation parameters transferable across
# modalities.
def _ct_norm_factor(n_angles):
    """Normalisation constant for the Radon operator pair."""
    return np.sqrt(np.pi * max(n_angles, 1))


def ct_forward(image, angles, normalize=True):
    """CT forward operator: parallel-beam Radon transform.

    Uses ``skimage.transform.radon``.

    Parameters
    ----------
    image : ndarray of shape (n_rows, n_cols)
    angles : ndarray
        Projection angles in radians.
    normalize : bool
        If True (default), normalises by :math:`1/\\sqrt{\\pi n_\\theta}`
        so that :math:`\\|A\\| \\approx 1`.

    Returns
    -------
    sinogram : ndarray of shape (n_angles, n_detectors)
    """
    from skimage.transform import radon
    # skimage radon returns (n_detectors, n_angles); we standardise to
    # (n_angles, n_detectors) for consistency.
    sino = radon(image, theta=np.rad2deg(angles), circle=True).T
    if normalize:
        sino = sino / _ct_norm_factor(len(angles))
    return sino


def ct_adjoint(sinogram, angles, output_size=None, normalize=True):
    """Adjoint CT operator: unfiltered back-projection.

    Uses ``skimage.transform.iradon`` with no ramp filter, which gives
    the true adjoint of the Radon transform (not the FBP reconstruction).

    Parameters
    ----------
    sinogram : ndarray of shape (n_angles, n_detectors)
        Radon sinogram (angles × detectors).
    angles : ndarray
        Projection angles in radians.
    output_size : int, optional
        Side length of the reconstructed square image.
    normalize : bool
        Must match the setting used in ``ct_forward``.

    Returns
    -------
    image : ndarray of shape (output_size, output_size)
    """
    from skimage.transform import iradon
    # iradon expects (n_detectors, n_angles)
    reco = iradon(sinogram.T, theta=np.rad2deg(angles),
                  output_size=output_size, circle=True, filter_name=None)
    if normalize:
        reco = reco / _ct_norm_factor(len(angles))
    return reco
