"""Invertible transforms of the real line: the layers of a one-dimensional normalizing flow.

Every transform maps a data value ``x`` to ``z = f(x)`` and returns the log-derivative that the change of variables
needs::

    z, logdet = transform(x)          # logdet = log |dz/dx|, same shape as x
    x = transform.inverse(z)

A flow stacks transforms, ``z = f_L(... f_2(f_1(x)))``, and its density is (see :class:`~prob_reco_sbi.flows1d.Flow`)::

    log q(x) = log p_base(z) + sum over the layers of log |dz/dx|.

==========  ==============================================  ==========  ====================================  =========
transform   z = f(x)                                        parameters  strictly increasing if                inverse
==========  ==============================================  ==========  ====================================  =========
Affine      exp(s) x + t                                    2           always                                exact
Planar      x + u_hat tanh(w x + b)                         3           u_hat w > -1                          bisection
Radial      x + beta_hat (x - x0) / (alpha_hat + |x - x0|)  3           alpha_hat > 0, beta_hat > -alpha_hat  bisection
RQSpline    rational-quadratic on K bins in [-B, B]         3 K - 1     by construction                       exact
Sigmoid     1 / (1 + exp(-x)), maps R onto (0, 1)           0           always                                exact
==========  ==============================================  ==========  ====================================  =========

The parameters of every transform are unconstrained real numbers, and the transform maps them into the region where
``f`` is strictly increasing, so in exact arithmetic no gradient step can break the invertibility. Planar and radial
transforms accept ``constrained=False``, which skips this mapping: it only exists to show what goes wrong without it.

References: D. J. Rezende and S. Mohamed, "Variational inference with normalizing flows", ICML 2015 (planar, radial);
C. Durkan, A. Bekasov, I. Murray and G. Papamakarios, "Neural spline flows", NeurIPS 2019 (rational-quadratic spline).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["Affine", "Planar", "RQSpline", "Radial", "Sigmoid", "Transform", "invert_monotone"]


def _inverse_softplus(y: float) -> float:
    """Return the raw value ``a`` with ``softplus(a) = y`` (``y > 0``)."""
    return math.log(math.expm1(y))


def invert_monotone(
    f: Callable[[Tensor], Tensor], z: Tensor, lo: Tensor | float, hi: Tensor | float, iters: int = 64
) -> Tensor:
    """Solve ``f(x) = z`` elementwise for a strictly increasing ``f`` by bisection.

    Parameters
    ----------
    f
        Strictly increasing elementwise function.
    z
        Target values.
    lo, hi
        A bracket of the solution, ``f(lo) <= z <= f(hi)`` (scalars or tensors broadcastable to ``z``).
    iters
        Number of halvings: the result is within ``(hi - lo) / 2**(iters + 1)`` of the solution (64 halvings reach
        the float64 resolution for brackets up to 10^3 wide).

    Returns
    -------
    Tensor
        ``x`` with ``f(x) = z``, without gradient (bisection is not differentiable).
    """
    with torch.no_grad():
        lo_t = torch.as_tensor(lo, dtype=z.dtype, device=z.device).expand_as(z).clone()
        hi_t = torch.as_tensor(hi, dtype=z.dtype, device=z.device).expand_as(z).clone()
        for _ in range(iters):
            mid = 0.5 * (lo_t + hi_t)
            below = f(mid) < z
            lo_t = torch.where(below, mid, lo_t)
            hi_t = torch.where(below, hi_t, mid)
        return 0.5 * (lo_t + hi_t)


class Transform(nn.Module):
    """Base class of the transforms: ``forward(x) -> (z, log|dz/dx|)`` and ``inverse(z) -> x``, elementwise."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data values to ``z = f(x)`` and return ``(z, log|dz/dx|)``."""
        raise NotImplementedError

    def inverse(self, z: Tensor) -> Tensor:
        """Map ``z`` back to ``x = f^-1(z)``."""
        raise NotImplementedError(
            f"{type(self).__name__} has no inverse: define inverse(z) to sample from a flow with this transform"
        )

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Affine(Transform):
    """Affine transform ``z = exp(s) x + t``: a shift and a positive scale.

    ``log|dz/dx| = s`` everywhere, so an affine flow with a Gaussian base is a Gaussian with mean ``-t exp(-s)`` and
    standard deviation ``exp(-s)``: it can match the mean and width of a target, nothing more. Parameters ``s``
    (``log_scale``) and ``t`` (``shift``), initialised to the identity (0, 0).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.shift = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data values to ``z = f(x)`` and return ``(z, log|dz/dx|)``."""
        return torch.exp(self.log_scale) * x + self.shift, self.log_scale.expand_as(x)

    def inverse(self, z: Tensor) -> Tensor:
        """Map ``z`` back to ``x = (z - t) exp(-s)``."""
        return (z - self.shift) * torch.exp(-self.log_scale)


class Planar(Transform):
    """Planar transform (Rezende and Mohamed 2015) in one dimension.

    ::

        z     = x + u_hat tanh(w x + b)
        dz/dx = 1 + u_hat w (1 - tanh^2(w x + b))

    It bends the identity around ``x = -b / w`` over a width of about ``1 / |w|``; far from there it is a shift by
    ``+-u_hat``. As ``1 - tanh^2`` takes every value in (0, 1], the derivative is positive everywhere (the map is
    invertible with a finite log-derivative) if and only if ``u_hat w > -1``. The raw parameter ``u`` is mapped to
    (appendix A of the paper, in one dimension)::

        u_hat = u + (m(w u) - w u) w / w^2,    m(a) = -1 + softplus(a),

    so that ``u_hat w = m(w u) > -1`` for every ``u`` and ``w``. This holds in exact arithmetic: in float32,
    ``1 + u_hat w`` rounds to 0 for ``w u`` below about -17 (the log-derivative at the bend centre is then -inf), and
    the clamp of ``w^2`` below changes the bound for ``|w| < 1e-6``; training does not come near either. Three
    parameters: ``u``, ``w``, ``b``. The inverse has no closed form; since ``|z - x| < |u_hat|``, it is found by
    bisection in ``[z - |u_hat|, z + |u_hat|]``.

    Parameters
    ----------
    constrained
        Map ``u`` to ``u_hat`` as above (default). With ``False``, ``u_hat = u`` and the map may fold back on itself:
        only for demonstrating that failure.

    Notes
    -----
    Initialisation (seeded by :func:`torch.manual_seed`): ``w, b ~ N(0, 1)`` and ``u ~ N(0, 0.1^2)``. A small raw
    ``u`` does not give a small ``u_hat``: for ``u -> 0``, ``u_hat w -> m(0) = log 2 - 1 = -0.31``, so ``u_hat`` is
    about ``-0.31 / w`` and a layer whose ``w`` starts near 0 starts as a large shift. A start at ``u = w = b = 0``
    is a stationary point (``u_hat = 0`` and every gradient vanishes).
    """

    def __init__(self, constrained: bool = True) -> None:
        super().__init__()
        self.constrained = constrained
        self.u = nn.Parameter(0.1 * torch.randn(()))
        self.w = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))

    @property
    def u_hat(self) -> Tensor:
        """The constrained ``u`` used in the map (``u_hat w > -1``)."""
        if not self.constrained:
            return self.u
        wu = self.w * self.u
        m = -1 + F.softplus(wu)
        return self.u + (m - wu) * self.w / (self.w * self.w).clamp_min(1e-12)  # clamp: no 0 / 0 at w = 0

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data values to ``z = f(x)`` and return ``(z, log|dz/dx|)``."""
        u_hat = self.u_hat
        h = torch.tanh(self.w * x + self.b)
        return x + u_hat * h, torch.log(torch.abs(1 + u_hat * self.w * (1 - h * h)))

    def inverse(self, z: Tensor) -> Tensor:
        """Map ``z`` back to ``x`` by bisection (requires the constraint)."""
        reach = self.u_hat.detach().abs()
        return invert_monotone(lambda x: self.forward(x)[0], z, z - reach, z + reach)


class Radial(Transform):
    """Radial transform (Rezende and Mohamed 2015) in one dimension.

    ::

        z     = x + beta_hat (x - x0) / (alpha_hat + |x - x0|)
        dz/dx = 1 + alpha_hat beta_hat / (alpha_hat + |x - x0|)^2

    It pushes points away from ``x0`` (``beta_hat > 0``) or pulls them towards it (``beta_hat < 0``), over a distance
    of about ``alpha_hat``; far away it is a shift by ``+-beta_hat``. At ``x = x0`` the derivative is
    ``1 + beta_hat / alpha_hat``, its smallest value for ``beta_hat < 0`` (its largest for ``beta_hat > 0``, where it
    is above 1 everywhere): the derivative is positive everywhere if and only if ``beta_hat > -alpha_hat`` (with
    ``alpha_hat > 0``). The raw parameters are mapped as in the paper::

        alpha_hat = softplus(alpha),    beta_hat = -alpha_hat + softplus(beta).

    Three parameters: ``x0``, ``alpha``, ``beta``. The inverse has no closed form; since ``|z - x| < |beta_hat|``, it
    is found by bisection in ``[z - |beta_hat|, z + |beta_hat|]``.

    Parameters
    ----------
    constrained
        Map ``beta`` to ``beta_hat`` as above (default). With ``False``, ``beta_hat = beta`` and the map may fold back
        on itself: only for demonstrating that failure (``alpha_hat = softplus(alpha)`` in both cases).

    Notes
    -----
    Initialisation (seeded by :func:`torch.manual_seed`): ``x0, alpha, beta ~ N(0, 1)``.
    """

    def __init__(self, constrained: bool = True) -> None:
        super().__init__()
        self.constrained = constrained
        self.x0 = nn.Parameter(torch.randn(()))
        self.alpha = nn.Parameter(torch.randn(()))
        self.beta = nn.Parameter(torch.randn(()))

    @property
    def alpha_hat(self) -> Tensor:
        """The positive width ``alpha_hat = softplus(alpha)``."""
        return F.softplus(self.alpha)

    @property
    def beta_hat(self) -> Tensor:
        """The constrained strength used in the map (``beta_hat > -alpha_hat``)."""
        if not self.constrained:
            return self.beta
        return -self.alpha_hat + F.softplus(self.beta)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data values to ``z = f(x)`` and return ``(z, log|dz/dx|)``."""
        alpha, beta = self.alpha_hat, self.beta_hat
        r = torch.abs(x - self.x0)
        z = x + beta * (x - self.x0) / (alpha + r)
        return z, torch.log(torch.abs(1 + alpha * beta / (alpha + r) ** 2))

    def inverse(self, z: Tensor) -> Tensor:
        """Map ``z`` back to ``x`` by bisection (requires the constraint)."""
        reach = self.beta_hat.detach().abs()
        return invert_monotone(lambda x: self.forward(x)[0], z, z - reach, z + reach)


class RQSpline(Transform):
    """Monotone rational-quadratic spline (Durkan et al. 2019) on ``[-B, B]``, the identity outside.

    The interval is cut into ``K`` bins by the knots ``(x_k, y_k)``, ``k = 0 ... K``, with ``x_0 = y_0 = -B`` and
    ``x_K = y_K = B``; the derivatives at the knots ``d_k`` are positive, and ``d_0 = d_K = 1`` so that the spline
    joins the identity tails smoothly. In bin ``k``, with the bin width ``w_k = x_{k+1} - x_k``, its height
    ``h_k = y_{k+1} - y_k``, its slope ``s_k = h_k / w_k`` and the position ``xi = (x - x_k) / w_k`` in [0, 1]::

        z     = y_k + h_k (s_k xi^2 + d_k xi (1 - xi)) / D
        dz/dx = s_k^2 (d_{k+1} xi^2 + 2 s_k xi (1 - xi) + d_k (1 - xi)^2) / D^2
        D     = s_k + (d_k + d_{k+1} - 2 s_k) xi (1 - xi)

    Every term is positive when all ``w_k``, ``h_k`` and ``d_k`` are, so the map is strictly increasing by
    construction. The raw parameters: ``K`` widths and ``K`` heights (softmax over the bins, then a minimum share,
    times ``2 B``) and ``K - 1`` interior derivatives (softplus plus a minimum): ``3 K - 1`` in all. The inverse
    solves a quadratic equation in ``xi`` (exact). Initialised to the identity (equal bins, unit derivatives).

    Parameters
    ----------
    bins
        Number of bins ``K``.
    bound
        Half-width ``B`` of the interval; outside it the map is the identity.
    min_width, min_height, min_derivative
        Lower limits of the bin widths and heights (as fractions of ``2 B``) and of the derivatives.
    """

    def __init__(
        self,
        bins: int = 16,
        bound: float = 10.0,
        min_width: float = 1e-3,
        min_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ) -> None:
        super().__init__()
        if bins < 1:
            raise ValueError(f"bins must be at least 1, got {bins}")
        if bound <= 0:
            raise ValueError(f"bound must be positive, got {bound}")
        if not (min_width * bins < 1 and min_height * bins < 1 and 0 < min_derivative < 1):
            raise ValueError("the minimum widths and heights must leave room for the bins, and 0 < min_derivative < 1")
        self.bins, self.bound = bins, float(bound)
        self.min_width, self.min_height, self.min_derivative = min_width, min_height, min_derivative
        self.widths = nn.Parameter(torch.zeros(bins))
        self.heights = nn.Parameter(torch.zeros(bins))
        self.derivatives = nn.Parameter(torch.full((bins - 1,), _inverse_softplus(1 - min_derivative)))

    def _knot_positions(self, raw: Tensor, minimum: float) -> Tensor:
        """Knot positions from -B to B (``K + 1`` values) for raw bin sizes."""
        share = minimum + (1 - minimum * self.bins) * torch.softmax(raw, dim=0)
        knots = 2 * self.bound * F.pad(torch.cumsum(share, dim=0), (1, 0)) - self.bound
        ends = knots.new_tensor([-self.bound, self.bound])
        return torch.cat([ends[:1], knots[1:-1], ends[1:]])  # exact end points despite rounding

    def knots(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return the knots ``x_k``, ``y_k`` and the derivatives ``d_k`` (``K + 1`` values each)."""
        x_knots = self._knot_positions(self.widths, self.min_width)
        y_knots = self._knot_positions(self.heights, self.min_height)
        one = self.derivatives.new_ones(1)
        d = torch.cat([one, self.min_derivative + F.softplus(self.derivatives), one])
        return x_knots, y_knots, d

    def _bin(self, knots: Tensor, v: Tensor) -> Tensor:
        """Index of the bin that contains each (clamped) value."""
        return (torch.searchsorted(knots, v.contiguous(), right=True) - 1).clamp(0, self.bins - 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data values to ``z = f(x)`` and return ``(z, log|dz/dx|)``."""
        cx, cy, d = self.knots()
        inside = (x >= -self.bound) & (x <= self.bound)
        xc = x.clamp(-self.bound, self.bound)
        k = self._bin(cx, xc)
        xk, wk = cx[k], cx[k + 1] - cx[k]
        yk, hk = cy[k], cy[k + 1] - cy[k]
        dk, dk1 = d[k], d[k + 1]
        s = hk / wk
        xi = (xc - xk) / wk
        t = xi * (1 - xi)
        denom = s + (dk + dk1 - 2 * s) * t
        z = yk + hk * (s * xi * xi + dk * t) / denom
        dz = s * s * (dk1 * xi * xi + 2 * s * t + dk * (1 - xi) ** 2) / denom**2
        return torch.where(inside, z, x), torch.where(inside, torch.log(dz), torch.zeros_like(x))

    def inverse(self, z: Tensor) -> Tensor:
        """Map ``z`` back to ``x`` (exact: the root in [0, 1] of a quadratic equation in ``xi``)."""
        cx, cy, d = self.knots()
        inside = (z >= -self.bound) & (z <= self.bound)
        zc = z.clamp(-self.bound, self.bound)
        k = self._bin(cy, zc)
        xk, wk = cx[k], cx[k + 1] - cx[k]
        yk, hk = cy[k], cy[k + 1] - cy[k]
        dk, dk1 = d[k], d[k + 1]
        s = hk / wk
        dy = zc - yk
        a = hk * (s - dk) + dy * (dk1 + dk - 2 * s)
        b = hk * dk - dy * (dk1 + dk - 2 * s)
        c = -s * dy
        xi = 2 * c / (-b - torch.sqrt((b * b - 4 * a * c).clamp_min(0)))
        return torch.where(inside, xk + xi * wk, z)


class Sigmoid(Transform):
    """Logistic squashing ``z = 1 / (1 + exp(-x))``: maps the real line onto (0, 1), for the uniform base.

    ``log|dz/dx| = log z + log(1 - z) = -softplus(-x) - softplus(x)``; the inverse is the logit
    ``x = log z - log(1 - z)``, with ``z`` clipped to ``[eps, 1 - eps]`` (``eps`` of the dtype) so that the samples stay
    finite. No parameters.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data values to ``z = f(x)`` and return ``(z, log|dz/dx|)``."""
        return torch.sigmoid(x), -F.softplus(-x) - F.softplus(x)

    def inverse(self, z: Tensor) -> Tensor:
        """Map ``z`` in (0, 1) back to ``x = logit(z)``."""
        eps = torch.finfo(z.dtype).eps
        return torch.logit(z, eps=eps)
