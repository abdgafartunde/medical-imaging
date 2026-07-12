"""Phantoms for testing and demonstrating MRI and CT reconstruction."""

import numpy as np
from skimage.data import shepp_logan_phantom as _sk_shepp_logan
from skimage.transform import resize


def shepp_logan(shape=(256, 256)):
    """Return the Shepp-Logan phantom at the requested resolution.

    The Shepp-Logan phantom is the standard test image for tomographic
    reconstruction, comprising ten ellipses with varying attenuations placed
    inside a brain-shaped outer ellipse.

    Parameters
    ----------
    shape : tuple of int
        (n_rows, n_cols) of the output image.

    Returns
    -------
    phantom : ndarray
        2-D array with values in [0.0, 1.0].
    """
    phantom = _sk_shepp_logan()
    if phantom.shape != shape:
        phantom = resize(phantom, shape, mode='reflect', anti_aliasing=True)
    return phantom


def circle_phantom(shape=(256, 256), circles=(), background=0.0):
    """Build a phantom from circles on a uniform background.

    Parameters
    ----------
    shape : tuple of int
        (n_rows, n_cols).
    circles : sequence of dict
        Each dict defines a circle with keys:
        - ``cx``, ``cy`` : float (centre, in normalised coordinates [-1, 1])
        - ``r`` : float (radius)
        - ``value`` : float (intensity)
    background : float
        Uniform background intensity.

    Returns
    -------
    phantom : ndarray
        2-D array.
    """
    n_rows, n_cols = shape
    ys = np.linspace(-1, 1, n_rows)
    xs = np.linspace(-1, 1, n_cols)
    X, Y = np.meshgrid(xs, ys)

    phantom = np.full(shape, background, dtype=np.float64)
    for circ in circles:
        cx, cy, r, val = circ["cx"], circ["cy"], circ["r"], circ["value"]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        phantom[mask] = val

    return phantom
