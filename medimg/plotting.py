"""Visualisation utilities for reconstruction results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_reconstructions(images, titles=None, suptitle=None, figsize=None,
                         cmap='gray', show_metrics=True, ref_idx=0):
    """Side-by-side comparison of reconstruction results.

    Parameters
    ----------
    images : list of ndarray
        List of 2-D images to display (first is usually the ground truth).
    titles : list of str, optional
        Sub-plot titles.
    suptitle : str, optional
        Overall figure title.
    figsize : tuple of float, optional
        (width, height) per sub-plot.
    cmap : str
        Matplotlib colormap.
    show_metrics : bool
        If True, annotates each reconstruction (after the first) with
        PSNR and SSIM relative to ``images[ref_idx]``.
    ref_idx : int
        Index of the reference image for metrics (default 0 = ground truth).

    Returns
    -------
    fig, axes
    """
    from medimg.metrics import psnr, ssim

    n = len(images)
    if titles is None:
        titles = [f'Image {i}' for i in range(n)]
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    x_ref = images[ref_idx]
    vmin, vmax = x_ref.min(), x_ref.max()

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(titles[i])
        ax.axis('off')

        if show_metrics and i != ref_idx:
            p = psnr(img, x_ref)
            s = ssim(img, x_ref)
            ax.set_xlabel(f'PSNR {p:.2f} dB | SSIM {s:.4f}', fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)

    fig.tight_layout()
    return fig, axes


def plot_error_map(x, x_ref, cmap='hot', figsize=(5, 5)):
    """Display the absolute error between a reconstruction and the reference.

    Parameters
    ----------
    x : ndarray
        Reconstructed image.
    x_ref : ndarray
        Reference image.
    cmap : str
        Colormap for the error.
    figsize : tuple of float

    Returns
    -------
    fig, ax
    """
    error = np.abs(np.asarray(x, dtype=np.float64)
                   - np.asarray(x_ref, dtype=np.float64))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(error, cmap=cmap, interpolation='nearest')
    ax.set_title('Absolute error')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax
