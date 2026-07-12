"""Image quality metrics for reconstruction evaluation."""

import numpy as np
from scipy.ndimage import uniform_filter


def psnr(x, x_ref):
    """Peak Signal-to-Noise Ratio (dB).

    Parameters
    ----------
    x : ndarray
        Reconstructed image.
    x_ref : ndarray
        Reference (ground-truth) image.

    Returns
    -------
    float
        PSNR in dB.  Returns ``inf`` if the images are identical.
    """
    x = np.asarray(x, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    mse = np.mean((x - x_ref) ** 2)
    if mse == 0:
        return np.inf
    data_range = x_ref.max() - x_ref.min()
    if data_range == 0:
        data_range = 1.0
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim(x, x_ref, data_range=None, K1=0.01, K2=0.03):
    """Structural Similarity Index (SSIM).

    Implements the original SSIM of Wang et al. (2004) with default
    stabilisation constants.

    Parameters
    ----------
    x : ndarray
        Reconstructed image.
    x_ref : ndarray
        Reference image.
    data_range : float, optional
        Dynamic range of the data.  If ``None``, computed as
        ``x_ref.max() - x_ref.min()``.
    K1, K2 : float
        Stabilisation constants (default 0.01, 0.03).

    Returns
    -------
    float
        SSIM value in [-1, 1] (1 = identical).

    References
    ----------
    Wang, Bovik, Sheikh & Simoncelli (2004). Image quality assessment:
    from error visibility to structural similarity. *IEEE TIP*, 13(4),
    600–612.
    """
    x = np.asarray(x, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)

    if data_range is None:
        data_range = x_ref.max() - x_ref.min()
    if data_range == 0:
        data_range = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_x = uniform_filter(x, size=11)
    mu_ref = uniform_filter(x_ref, size=11)
    mu_x_sq = mu_x ** 2
    mu_ref_sq = mu_ref ** 2
    mu_xx_ref = mu_x * mu_ref

    sigma_x_sq = uniform_filter(x ** 2, size=11) - mu_x_sq
    sigma_ref_sq = uniform_filter(x_ref ** 2, size=11) - mu_ref_sq
    sigma_xx_ref = uniform_filter(x * x_ref, size=11) - mu_xx_ref

    ssim_map = ((2.0 * mu_xx_ref + C1) * (2.0 * sigma_xx_ref + C2)) / \
               ((mu_x_sq + mu_ref_sq + C1) * (sigma_x_sq + sigma_ref_sq + C2))

    return float(np.mean(ssim_map))
