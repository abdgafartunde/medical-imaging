"""Forward operators for accelerated MRI and sparse-view CT.

Provides the sensing models and their adjoints:

- MRI:   Fourier encoding with variable-density Cartesian under-sampling.
- CT:    Parallel-beam Radon transform with sparse angular sampling.

Each adjoint is the exact adjoint of its forward operator, so the
dot-product test <A x, y> = <x, A* y> holds to rounding error (see
``tests/test_forward.py``).  With the default normalisation both
operators have spectral norm 1.

References
----------
- Lustig, Donoho & Pauly (2007). Sparse MRI. *MRM*, 58(6), 1182–1195.
- Sidky & Pan (2008). Image reconstruction in circular cone-beam CT by
  constrained, total-variation minimization. *PMB*, 53(17), 4777–4807.
"""

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds


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

    The FFT is unitary (``norm="ortho"``), so ``||F x|| = ||x||`` and
    :func:`mri_adjoint` is the exact adjoint.  Noise is added only at
    sampled k-space locations.

    Parameters
    ----------
    image : ndarray of shape (n_rows, n_cols)
        Real-valued ground-truth image.
    mask : ndarray of bool, shape (n_rows, n_cols)
        k-space sampling mask.
    noise_std : float
        Standard deviation of the real and of the imaginary part of the
        complex white Gaussian noise added at each sampled k-space
        location.  Because the FFT is unitary, it is on the same scale as
        the image intensities.
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

    k_full = fftshift(fft2(ifftshift(image), norm="ortho"))
    kspace = k_full * mask

    if noise_std > 0:
        noise = (rng.normal(0, noise_std, kspace.shape)
                 + 1j * rng.normal(0, noise_std, kspace.shape))
        kspace += noise * mask

    return kspace


def mri_adjoint(kspace, mask):
    """Adjoint MRI operator: masked inverse FFT (zero-filled reconstruction).

    This is the exact adjoint of :func:`mri_forward` (without noise) for
    real-valued images, with the real inner product on k-space.

    Parameters
    ----------
    kspace : ndarray of complex, shape (n_rows, n_cols)
    mask : ndarray of bool, shape (n_rows, n_cols)

    Returns
    -------
    image : ndarray, shape (n_rows, n_cols)
        Real part of the zero-filled inverse Fourier transform.
    """
    return np.real(fftshift(ifft2(ifftshift(kspace * mask), norm="ortho")))


# ---------------------------------------------------------------------------
# CT
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _radon_operator(n, angles_key):
    """Sparse parallel-beam projection matrix and its spectral norm.

    Pixel-driven projector with linear interpolation onto ``n`` unit-width
    detector bins, using the geometry of ``skimage.transform.radon`` with
    ``circle=True``: only pixels inside the inscribed circle contribute,
    and detector bin ``n // 2`` passes through the image centre.  The
    back-projection is the matrix transpose, so the pair is exactly
    adjoint.
    """
    angles = np.frombuffer(angles_key, dtype=np.float64)
    c = n // 2
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    inside = ((cols - (n - 1) / 2) ** 2 + (rows - (n - 1) / 2) ** 2
              <= (n / 2) ** 2).ravel()
    pixels = np.flatnonzero(inside)
    x = (cols.ravel() - c)[pixels].astype(np.float64)
    y = (c - rows.ravel())[pixels].astype(np.float64)

    r_idx, c_idx, vals = [], [], []
    for a, theta in enumerate(angles):
        s = x * np.cos(theta) + y * np.sin(theta) + c
        k0 = np.floor(s).astype(np.int64)
        frac = s - k0
        for k, w in ((k0, 1.0 - frac), (k0 + 1, frac)):
            keep = (k >= 0) & (k < n) & (w > 0)
            r_idx.append(a * n + k[keep])
            c_idx.append(pixels[keep])
            vals.append(w[keep])
    R = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
        shape=(len(angles) * n, n * n),
    )
    sigma_max = svds(R, k=1, return_singular_vectors=False,
                     v0=np.ones(min(R.shape)), tol=1e-8)[0]
    return R, float(sigma_max)


def _ct_operator(n, angles):
    angles = np.ascontiguousarray(angles, dtype=np.float64)
    return _radon_operator(int(n), angles.tobytes())


def ct_forward(image, angles, normalize=True):
    """CT forward operator: parallel-beam Radon transform.

    Uses a sparse pixel-driven projector whose geometry matches
    ``skimage.transform.radon(image, circle=True)``; pixels outside the
    inscribed circle are ignored.

    Parameters
    ----------
    image : ndarray of shape (n, n)
    angles : ndarray
        Projection angles in radians.
    normalize : bool
        If True (default), divides by the spectral norm of the projector so
        that :math:`\\|A\\| = 1`.  If False, returns line integrals in pixel
        units (suitable for filtered back-projection).

    Returns
    -------
    sinogram : ndarray of shape (n_angles, n)
    """
    image = np.asarray(image, dtype=np.float64)
    n = image.shape[0]
    if image.shape != (n, n):
        raise ValueError(f"image must be square, got shape {image.shape}")
    R, sigma_max = _ct_operator(n, angles)
    sino = (R @ image.ravel()).reshape(len(angles), n)
    if normalize:
        sino = sino / sigma_max
    return sino


def ct_adjoint(sinogram, angles, output_size=None, normalize=True):
    """Adjoint CT operator: unfiltered back-projection.

    The exact adjoint (matrix transpose) of :func:`ct_forward`; it is not
    the filtered back-projection (FBP) reconstruction.

    Parameters
    ----------
    sinogram : ndarray of shape (n_angles, n)
        Radon sinogram (angles × detectors).
    angles : ndarray
        Projection angles in radians.
    output_size : int, optional
        Side length of the image; must equal the number of detectors
        ``n`` if given.
    normalize : bool
        Must match the setting used in ``ct_forward``.

    Returns
    -------
    image : ndarray of shape (n, n)
    """
    sinogram = np.asarray(sinogram, dtype=np.float64)
    n = sinogram.shape[1]
    if output_size is not None and int(output_size) != n:
        raise ValueError(
            f"output_size={output_size} does not match the {n} detector bins"
        )
    R, sigma_max = _ct_operator(n, angles)
    image = (R.T @ sinogram.ravel()).reshape(n, n)
    if normalize:
        image = image / sigma_max
    return image
