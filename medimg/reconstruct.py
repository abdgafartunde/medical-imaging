"""Reconstruction algorithms for compressed sensing MRI and CT.

Implements three classical variational methods for linear inverse problems:

- Tikhonov regularisation (conjugate gradient)
- Total variation regularisation (Chambolle-Pock primal-dual)
- Wavelet-domain L1 minimisation (FISTA)

All methods operate on abstract forward/adjoint operator pairs, making them
applicable to both MRI (Fourier + mask) and CT (Radon) forward models.

References
----------
- Lustig, Donoho & Pauly (2007). Sparse MRI. *MRM*, 58(6), 1182–1195.
- Chambolle & Pock (2011). A first-order primal-dual algorithm for convex
  problems with applications to imaging. *JMIV*, 40(1), 120–145.
- Beck & Teboulle (2009). A fast iterative shrinkage-thresholding algorithm
  for linear inverse problems. *SIAM J. Imaging Sci.*, 2(1), 183–202.
"""

import numpy as np
from medimg.transforms import soft_threshold_coeffs


# ---------------------------------------------------------------------------
# Utility: power method for Lipschitz constant estimation
# ---------------------------------------------------------------------------

def _lipschitz_estimate(A, At, shape, n_iter=30, rng=None):
    """Estimate the spectral norm ||A|| via power iteration.

    Parameters
    ----------
    A, At : callable
        Forward and adjoint operators.
    shape : tuple of int
        Shape of the image space.
    n_iter : int
        Number of power iterations.

    Returns
    -------
    float
        Estimate of ||A||^2 (the Lipschitz constant of ∇f for the
        least-squares data-fidelity term).
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    x = rng.normal(size=shape).astype(np.float64)
    for _ in range(n_iter):
        x = At(A(x))
        norm = np.linalg.norm(x.ravel())
        x /= norm
    return float(np.linalg.norm(At(A(x)).ravel()) / np.linalg.norm(x.ravel()))


# ---------------------------------------------------------------------------
# Tikhonov — conjugate gradient
# ---------------------------------------------------------------------------

def tikhonov_cg(A, At, y, lambd, shape, x0=None, max_iter=100, tol=1e-6):
    r"""Tikhonov-regularised least squares via conjugate gradient.

    Solves

    .. math::

        \min_x \; \tfrac12 \|A x - y\|_2^2
                 + \tfrac{\lambda}{2} \|x\|_2^2,

    which has the normal equation
    :math:`(A^* A + \lambda I) x = A^* y`.

    Parameters
    ----------
    A : callable
        Forward operator.
    At : callable
        Adjoint operator.
    y : ndarray
        Data vector or array (k-space or sinogram).
    lambd : float
        Regularisation parameter (:math:`\lambda > 0`).
    shape : tuple of int
        Shape of the image to reconstruct.
    x0 : ndarray, optional
        Initial guess (zero by default).
    max_iter : int
        Maximum CG iterations.
    tol : float
        Relative residual tolerance.

    Returns
    -------
    x : ndarray of shape *shape*
        Reconstructed image.
    """
    if x0 is None:
        x = np.zeros(shape, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    At_y = At(y).astype(np.float64)

    # Operator: H(x) = A^* A x + λ x
    def H(v):
        return At(A(v)) + lambd * v

    r = At_y - H(x)
    p = r.copy()
    rs_old = np.dot(r.ravel(), r.ravel())

    for _ in range(max_iter):
        Hp = H(p)
        alpha = rs_old / np.dot(p.ravel(), Hp.ravel())
        x += alpha * p
        r -= alpha * Hp
        rs_new = np.dot(r.ravel(), r.ravel())
        if np.sqrt(rs_new) < tol * np.sqrt(rs_old):
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


# ---------------------------------------------------------------------------
# Total variation — Chambolle-Pock primal-dual
# ---------------------------------------------------------------------------

def tv_chambolle_pock(A, At, y, alpha, shape, L=None, max_iter=500,
                      theta=1.0, rng=None):
    r"""Total variation regularisation via the Chambolle-Pock algorithm.

    Solves

    .. math::

        \min_x \; \tfrac12 \|A x - y\|_2^2 + \alpha \operatorname{TV}(x),

    where :math:`\operatorname{TV}(x) = \|\nabla x\|_{2,1}` is the
    discrete isotropic total variation.  The algorithm solves the
    full saddle-point problem with the proximal operator of the
    Fenchel conjugate of the objective, following Chambolle & Pock
    (2011) exactly.

    Parameters
    ----------
    A : callable
        Forward operator.
    At : callable
        Adjoint operator.
    y : ndarray
        Data.
    alpha : float
        TV regularisation parameter (:math:`\alpha > 0`).
    shape : tuple of int
        Image shape (n_rows, n_cols).
    L : float, optional
        Lipschitz constant :math:`\|A\|^2`.  Estimated via power
        iteration if ``None``.
    max_iter : int
        Maximum primal-dual iterations.
    theta : float
        Over-relaxation parameter (1.0 for standard PDHG).
    rng : int or numpy.random.Generator, optional

    Returns
    -------
    x : ndarray of shape *shape*
        Reconstructed image.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    n_rows, n_cols = shape

    # Estimate Lipschitz constant of A
    if L is None:
        L = _lipschitz_estimate(A, At, shape, rng=rng)

    # Discrete gradient and its adjoint (divergence)
    def grad(v):
        g1 = np.roll(v, -1, axis=0) - v
        g2 = np.roll(v, -1, axis=1) - v
        return np.stack([g1, g2], axis=0)

    def div(p):
        d1 = p[0] - np.roll(p[0], 1, axis=0)
        d2 = p[1] - np.roll(p[1], 1, axis=1)
        return d1 + d2

    # Step sizes: τ σ ‖K‖² < 1 with ‖K‖² ≤ L + ‖∇‖² ≤ L + 8.
    # Use τ = σ = 0.99 / √(L + 8) for guaranteed convergence.
    grad_norm_sq = 8.0
    K_norm_sq = L + grad_norm_sq
    tau = 0.99 / np.sqrt(K_norm_sq)
    sigma = 0.99 / np.sqrt(K_norm_sq)

    # Initialise
    x = np.zeros(shape, dtype=np.float64)
    x_bar = x.copy()
    # Dual variables: p (data, complex), q (gradient, 2-channel real)
    p_data = np.zeros_like(y, dtype=np.complex128)
    q_grad = np.zeros((2, n_rows, n_cols), dtype=np.float64)

    # Pre-compute ‖y‖² term for F^* proximal
    At_y = At(y).astype(np.float64)

    for _ in range(max_iter):
        # --- Dual update ---
        # p̃ = prox_{σ (½‖· - y‖²)^*} (p + σ A x̄)
        #    = (p + σ A x̄ - σ y) / (1 + σ)
        p_data = (p_data + sigma * A(x_bar) - sigma * y) / (1.0 + sigma)

        # q̃ = Π_α (q + σ ∇ x̄)  — pointwise L2 projection
        q_grad = q_grad + sigma * grad(x_bar)
        q_norm = np.sqrt(q_grad[0] ** 2 + q_grad[1] ** 2)
        q_norm_safe = np.maximum(q_norm, 1e-12)
        scale = np.minimum(alpha, q_norm_safe) / q_norm_safe
        q_grad[0] *= scale
        q_grad[1] *= scale

        # --- Primal update ---
        # K^* = [A^*, ∇^*] = [At, -div] since div = -∇^*
        x_old = x.copy()
        x = x - tau * (At(p_data) - div(q_grad))

        # --- Over-relaxation ---
        x_bar = x + theta * (x - x_old)

    return x


# ---------------------------------------------------------------------------
# Compressed sensing — FISTA with wavelet sparsity
# ---------------------------------------------------------------------------

def cs_fista(A, At, y, W, Wt, lambd, shape, L=None, max_iter=500, tol=1e-6,
             rng=None):
    r"""Compressed sensing reconstruction via FISTA with wavelet L1 penalty.

    Solves

    .. math::

        \min_x \; \tfrac12 \|A x - y\|_2^2 + \lambda \|W x\|_1,

    where :math:`W` is a sparsifying wavelet transform.  FISTA achieves
    the optimal :math:`O(1/k^2)` convergence rate for this class of
    composite convex problems.

    Parameters
    ----------
    A : callable
        Forward operator.
    At : callable
        Adjoint operator.
    y : ndarray
        Data.
    W : callable
        Sparsifying transform (returns coefficient structure).
    Wt : callable
        Adjoint (synthesis) transform (takes coefficient structure).
    lambd : float
        L1 regularisation parameter (:math:`\lambda > 0`).
    shape : tuple of int
        Image shape.
    L : float, optional
        Lipschitz constant :math:`\|A\|^2`.  Estimated if ``None``.
    max_iter : int
        Maximum FISTA iterations.
    tol : float
        Relative change tolerance for early stopping.
    rng : int or numpy.random.Generator, optional

    Returns
    -------
    x : ndarray of shape *shape*
        Reconstructed image.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    if L is None:
        L = _lipschitz_estimate(A, At, shape, rng=rng)

    step = 1.0 / max(L, 1e-12)

    x = np.zeros(shape, dtype=np.float64)
    z = x.copy()
    t = 1.0

    for _ in range(max_iter):
        x_old = x.copy()

        # Gradient step on the data-fidelity term
        gradient = At(A(z) - y)
        x_new = z - step * gradient

        # Proximal step: soft-threshold wavelet coefficients
        coeff_struct = W(x_new)
        coeffs_thresh = soft_threshold_coeffs(coeff_struct, lambd * step)
        x = Wt(coeffs_thresh)

        # FISTA momentum
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t ** 2))
        z = x + ((t - 1.0) / t_new) * (x - x_old)
        t = t_new

        # Check convergence
        change = np.linalg.norm((x - x_old).ravel()) / max(
            np.linalg.norm(x.ravel()), 1e-12
        )
        if change < tol:
            break

    return x
