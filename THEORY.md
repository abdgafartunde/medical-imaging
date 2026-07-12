# Theory — Compressed Sensing Reconstruction for MRI and CT

## Notation

- $x \in \mathbb{R}^N$ — vectorised image ($N = n_{\text{rows}} \times n_{\text{cols}}$)
- $y \in \mathbb{C}^M$ — measured data (k-space samples or sinogram)
- $A: \mathbb{R}^N \to \mathbb{C}^M$ — forward operator (linear)
- $A^*: \mathbb{C}^M \to \mathbb{R}^N$ — adjoint operator

## 1. Forward Models

### 1.1 MRI

The MRI measurement process in the ideal single-coil setting is the
2-D Fourier transform sampled at discrete k-space locations.  Let
$\mathcal{F}$ denote the unitary 2-D DFT.  With an under-sampling
mask $M \in \{0, 1\}^{n_{\text{rows}} \times n_{\text{cols}}}$ and
additive complex Gaussian noise $\eta \sim \mathcal{CN}(0, \sigma^2 I)$,
the forward model is

$$A_{\text{MRI}}(x) = M \odot \mathcal{F}(x), \qquad
y = A_{\text{MRI}}(x) + \eta.$$

The adjoint is the masked inverse Fourier transform:

$$A_{\text{MRI}}^*(y) = \operatorname{Re}\left(\mathcal{F}^{-1}(M \odot y)\right).$$

When $M$ has fewer non-zero entries than the number of image pixels,
the system is under-determined and regularisation is required.

**Variable-density under-sampling.**  Following Lustig et al. (2007),
the mask fully samples a central calibration band of $n_c$ low-frequency
phase-encode lines and randomly sub-samples higher frequencies with
probability decaying polynomially with distance from the k-space centre:

$$P(k_y) \propto \left(1 - \left(\frac{|k_y|}{k_y^{\max}}\right)^2\right)^2.$$

This exploits the fact that most of the image energy is concentrated
at low spatial frequencies.

### 1.2 CT

For parallel-beam geometry, the forward operator is the 2-D Radon
transform $\mathcal{R}$.  Let $\theta_j \in [0, \pi)$ be the projection
angles and $s$ the detector coordinate.  Then

$$A_{\text{CT}}(x)(\theta_j, s) = \mathcal{R}x(\theta_j, s)
= \int_{-\infty}^{\infty} x(s \cos\theta_j - t \sin\theta_j,\;
                          s \sin\theta_j + t \cos\theta_j)\, dt.$$

The adjoint $A_{\text{CT}}^* = \mathcal{R}^*$ is the unfiltered
back-projection (not the FBP reconstruction, which applies a ramp
filter).  For $N_\theta$ projection angles and $N_s$ detector bins,
the sinogram has dimensions $N_\theta \times N_s$.  Sparse-view CT
uses $N_\theta$ much smaller than the Nyquist criterion would demand,
making the reconstruction problem under-determined.

## 2. Compressed Sensing Principles

Compressed sensing (Candès, Romberg & Tao 2006; Donoho 2006) guarantees
exact or near-exact recovery of a sparse signal from under-sampled
linear measurements under two conditions:

1. **Sparsity.**  The image $x$ admits a sparse representation in some
   transform domain $\Psi$: $x = \Psi^*c$, where $c$ has few non-zero
   entries.
2. **Incoherence.**  The sensing basis (Fourier for MRI; pixel basis
   for CT) and the sparsity basis (wavelet, finite differences) are
   sufficiently incoherent.

Under these conditions, the L1-regularised problem

$$\min_x \frac12 \|A x - y\|_2^2 + \lambda \|\Psi x\|_1$$

recovers the true image with high probability when $M \gtrsim
s \log(N/s)$ measurements are acquired, where $s$ is the sparsity
level.

## 3. Variational Reconstruction Methods

### 3.1 Tikhonov Regularisation (L2)

The simplest convex regulariser penalises the $\ell_2$ energy of
the image:

$$\min_x \frac12 \|A x - y\|_2^2 + \frac{\lambda}{2} \|x\|_2^2.$$

The solution satisfies the normal equation

$$(A^* A + \lambda I) x = A^* y,$$

which is a symmetric positive-definite linear system solved by the
conjugate gradient method.  Tikhonov regularisation produces smooth
reconstructions; it does not promote sparsity or preserve edges.

### 3.2 Total Variation (TV)

The isotropic discrete total variation is

$$\operatorname{TV}(x) = \|\nabla x\|_{2,1}
= \sum_{i,j} \sqrt{(\nabla_1 x)_{i,j}^2 + (\nabla_2 x)_{i,j}^2},$$

where $\nabla_1, \nabla_2$ are forward finite-difference operators
in the row and column directions.  TV penalises the $\ell_1$ norm
of the gradient magnitude, promoting piecewise-constant
reconstructions with sharp edges.

The reconstruction problem

$$\min_x \frac12 \|A x - y\|_2^2 + \alpha \operatorname{TV}(x)$$

is solved via the **Chambolle-Pock primal-dual algorithm**
(Chambolle & Pock 2011).  The saddle-point formulation is

$$\min_x \max_{p:\,\|p\|_{2,\infty} \le \alpha}
\frac12 \|A x - y\|_2^2 + \langle \nabla x, p \rangle,$$

where $p \in \mathbb{R}^{2 \times N}$ is the dual variable and the
constraint $\|p_{i,j}\|_2 \le \alpha$ is enforced pointwise.  The
algorithm alternates:

1. **Dual ascent:** $p^{k+1} = \Pi_\alpha(p^k + \sigma \nabla \bar{x}^k)$,
   where $\Pi_\alpha$ is pointwise L2-ball projection.
2. **Primal descent:** $x^{k+1} = x^k - \tau\big(A^*(A x^k - y) + \operatorname{div} p^{k+1}\big)$.
3. **Over-relaxation:** $\bar{x}^{k+1} = x^{k+1} + \theta(x^{k+1} - x^k)$.

Convergence is guaranteed when $\tau\sigma\|\nabla\|^2 < 1$ and
$\tau\sigma\|A\|^2 < 1$.  With $\|\nabla\|^2 \le 8$, the step sizes
$\tau = \sigma = 1/\sqrt{\|A\|^2 + 8}$ satisfy the condition.

### 3.3 Wavelet-Domain L1 Minimisation (CS-FISTA)

Let $W: \mathbb{R}^N \to \mathbb{R}^K$ be a sparsifying wavelet
transform (e.g., Daubechies-4 with 4 decomposition levels) and
$W^*$ its adjoint (synthesis).  The CS objective is

$$\min_x \frac12 \|A x - y\|_2^2 + \lambda \|W x\|_1.$$

This is a composite convex problem $f(x) + g(x)$ with
$f(x) = \frac12\|A x - y\|_2^2$ (smooth, gradient-Lipschitz with
constant $L = \|A\|^2$) and $g(x) = \lambda\|W x\|_1$ (non-smooth,
prox-friendly).  **FISTA** (Beck & Teboulle 2009) achieves the
optimal $O(1/k^2)$ convergence rate:

1. **Gradient step:** $\tilde{x}^k = z^k - \frac{1}{L} A^*(A z^k - y)$.
2. **Proximal step:** $x^{k+1} = W^*\big(S_{\lambda/L}(W \tilde{x}^k)\big)$,
   where $S_\tau(c) = \operatorname{sign}(c)\max(|c| - \tau, 0)$ is
   element-wise soft-thresholding.
3. **Momentum:** $t_{k+1} = \frac12\left(1 + \sqrt{1 + 4t_k^2}\right)$,
   $z^{k+1} = x^{k+1} + \frac{t_k - 1}{t_{k+1}}(x^{k+1} - x^k)$.

The Lipschitz constant $L$ is estimated by power iteration when not
known analytically.  For MRI, $L = 1$ exactly (the FFT is unitary
and the mask only zeros entries).  For CT, $L$ depends on the
number and distribution of projection angles.

## 4. Image Quality Metrics

**PSNR** (Peak Signal-to-Noise Ratio, in dB):

$$\operatorname{PSNR}(x, x_{\text{ref}}) =
20\log_{10}(\operatorname{range}) -
10\log_{10}\!\left(\frac{1}{N}\|x - x_{\text{ref}}\|_2^2\right).$$

**SSIM** (Structural Similarity, Wang et al. 2004):

$$\operatorname{SSIM}(x, x_{\text{ref}}) =
\frac{(2\mu_x\mu_{\text{ref}} + C_1)(2\sigma_{x,\text{ref}} + C_2)}
{(\mu_x^2 + \mu_{\text{ref}}^2 + C_1)(\sigma_x^2 + \sigma_{\text{ref}}^2 + C_2)},$$

where $\mu, \sigma$ are local means and standard deviations computed
over $11\times11$ Gaussian windows, and $C_1, C_2$ are stabilisation
constants.

## 5. Key References

- Candès, E.J., Romberg, J. & Tao, T. (2006). Robust uncertainty
  principles: exact signal reconstruction from highly incomplete
  frequency information. *IEEE TIT*, 52(2), 489–509.
- Donoho, D.L. (2006). Compressed sensing. *IEEE TIT*, 52(4),
  1289–1306.
- Lustig, M., Donoho, D. & Pauly, J.M. (2007). Sparse MRI.
  *MRM*, 58(6), 1182–1195.
- Sidky, E.Y. & Pan, X. (2008). Image reconstruction in circular
  cone-beam CT by constrained, TV minimization. *PMB*, 53(17),
  4777–4807.
- Beck, A. & Teboulle, M. (2009). A fast iterative
  shrinkage-thresholding algorithm for linear inverse problems.
  *SIAM J. Imaging Sci.*, 2(1), 183–202.
- Chambolle, A. & Pock, T. (2011). A first-order primal-dual
  algorithm for convex problems with applications to imaging.
  *JMIV*, 40(1), 120–145.
