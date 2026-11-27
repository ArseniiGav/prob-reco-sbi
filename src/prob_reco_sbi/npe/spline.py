"""Rational-quadratic splines with one set of knots per value: the transform of every factor of a conditional flow.

The formula is that of :class:`prob_reco_sbi.flows1d.RQSpline` (Durkan et al. 2019). The difference is where the
parameters come from: ``RQSpline`` owns its ``3 K - 1`` raw parameters, the same for every value; here they are an
input with one row per value, so that a network can compute them from the conditioning variables::

    z, log_det = rq_spline(theta, params, bound=5.0)    # theta [...], params [..., 3 K - 1]: one spline per value
    theta = rq_spline_inverse(z, params, bound=5.0)

The raw parameters of one spline, in this order: ``K`` widths and ``K`` heights (softmax over the bins, a minimum
share, times ``2 B``) and ``K - 1`` interior derivatives (softplus plus a minimum). The derivatives are shifted so that
**all-zero parameters give the identity** (equal bins, unit derivatives): a network whose last layer starts at zero
starts as the identity map. Relative to ``RQSpline`` the shift is ``c = log(expm1(1 - min_derivative))``::

    params = torch.cat([spline.widths, spline.heights, spline.derivatives - c])   # the same map as RQSpline

Inside ``[-B, B]`` the map is the rational-quadratic spline, outside it the identity (log-derivative 0). The inverse
is exact: the root in ``[0, 1]`` of a quadratic equation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["n_spline_params", "rq_knots", "rq_spline", "rq_spline_inverse"]

#: Default lower limits of the bin widths and heights (as fractions of ``2 B``) and of the derivatives, as in RQSpline.
MIN_WIDTH = 1e-3
MIN_HEIGHT = 1e-3
MIN_DERIVATIVE = 1e-3


def n_spline_params(bins: int) -> int:
    """Return the number of raw parameters of a spline with ``bins`` bins, ``3 K - 1``."""
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    return 3 * bins - 1


def _bins(params: Tensor) -> int:
    """Return the number of bins ``K`` from the size of the last axis of ``params`` (``3 K - 1``)."""
    n = params.shape[-1]
    if n < 2 or (n + 1) % 3 != 0:
        raise ValueError(f"the last axis of params must have 3 K - 1 entries (2, 5, 8, ...), got {n}")
    return (n + 1) // 3


def _positions(raw: Tensor, minimum: float, bound: float) -> Tensor:
    """Knot positions from -B to B (``K + 1`` values on the last axis) for raw bin sizes ``[..., K]``."""
    k = raw.shape[-1]
    share = minimum + (1 - minimum * k) * torch.softmax(raw, dim=-1)
    inner = 2 * bound * torch.cumsum(share, dim=-1)[..., :-1] - bound
    ends = inner.new_full((*inner.shape[:-1], 1), bound)
    return torch.cat([-ends, inner, ends], dim=-1)  # exact end points despite rounding


def rq_knots(
    params: Tensor,
    bound: float = 5.0,
    min_width: float = MIN_WIDTH,
    min_height: float = MIN_HEIGHT,
    min_derivative: float = MIN_DERIVATIVE,
) -> tuple[Tensor, Tensor, Tensor]:
    """Knots of the splines given by raw parameters.

    Parameters
    ----------
    params
        Raw parameters, shape ``[..., 3 K - 1]``.
    bound
        Half-width ``B`` of the interval of the spline.
    min_width, min_height, min_derivative
        Lower limits of the bin widths and heights (fractions of ``2 B``) and of the derivatives.

    Returns
    -------
    tuple of Tensor
        ``(x_k, y_k, d_k)``, each of shape ``[..., K + 1]``: the knot positions on the input and the output axis and
        the derivatives at the knots (``d_0 = d_K = 1``).
    """
    k = _bins(params)
    if bound <= 0:
        raise ValueError(f"bound must be positive, got {bound}")
    if not (min_width * k < 1 and min_height * k < 1 and 0 < min_derivative < 1):
        raise ValueError("the minimum widths and heights must leave room for the bins, and 0 < min_derivative < 1")
    x_knots = _positions(params[..., :k], min_width, bound)
    y_knots = _positions(params[..., k : 2 * k], min_height, bound)
    shift = math.log(math.expm1(1 - min_derivative))  # zero raw parameters give a unit derivative
    inner = min_derivative + F.softplus(params[..., 2 * k :] + shift)
    one = inner.new_ones((*inner.shape[:-1], 1))
    return x_knots, y_knots, torch.cat([one, inner, one], dim=-1)


def _gather(values: Tensor, index: Tensor) -> Tensor:
    """``values[..., index]`` for an index of shape ``[..., 1]``, returned without the last axis."""
    return values.gather(-1, index).squeeze(-1)


def _prepare(v: Tensor, params: Tensor, bound: float, **limits: float) -> tuple[Tensor, ...]:
    """Broadcast the values and the parameters, and return the values with the knots of their splines."""
    knots = rq_knots(params, bound, **limits)
    shape = torch.broadcast_shapes(v.shape, params.shape[:-1])
    v = v.expand(shape)
    cx, cy, d = (k.expand(*shape, k.shape[-1]).contiguous() for k in knots)
    return v, cx, cy, d


def rq_spline(
    theta: Tensor,
    params: Tensor,
    bound: float = 5.0,
    *,
    min_width: float = MIN_WIDTH,
    min_height: float = MIN_HEIGHT,
    min_derivative: float = MIN_DERIVATIVE,
) -> tuple[Tensor, Tensor]:
    """Map values through their splines, in the normalizing direction: ``z = f(theta)`` and ``log|dz/dtheta|``.

    In bin ``k`` of a spline, with the bin width ``w_k``, its height ``h_k``, its slope ``s_k = h_k / w_k`` and the
    position ``xi = (theta - x_k) / w_k`` in [0, 1]::

        z        = y_k + h_k (s_k xi^2 + d_k xi (1 - xi)) / D
        dz/dtheta = s_k^2 (d_{k+1} xi^2 + 2 s_k xi (1 - xi) + d_k (1 - xi)^2) / D^2
        D        = s_k + (d_k + d_{k+1} - 2 s_k) xi (1 - xi)

    Parameters
    ----------
    theta
        Values, shape ``[...]``.
    params
        Raw parameters of one spline per value, shape ``[..., 3 K - 1]`` (broadcast against ``theta``; a single row
        ``[3 K - 1]`` is one spline for every value).
    bound
        Half-width ``B`` of the interval of the spline; outside it the map is the identity.
    min_width, min_height, min_derivative
        Lower limits of the bin widths and heights (fractions of ``2 B``) and of the derivatives.

    Returns
    -------
    tuple of Tensor
        ``(z, log_det)`` with ``log_det = log|dz/dtheta|``, both of the broadcast shape.
    """
    x, cx, cy, d = _prepare(
        theta, params, bound, min_width=min_width, min_height=min_height, min_derivative=min_derivative
    )
    inside = (x >= -bound) & (x <= bound)
    xc = x.clamp(-bound, bound)
    k = (torch.searchsorted(cx, xc.unsqueeze(-1), right=True) - 1).clamp(0, cx.shape[-1] - 2)
    xk, wk = _gather(cx, k), _gather(cx, k + 1) - _gather(cx, k)
    yk, hk = _gather(cy, k), _gather(cy, k + 1) - _gather(cy, k)
    dk, dk1 = _gather(d, k), _gather(d, k + 1)
    s = hk / wk
    xi = (xc - xk) / wk
    t = xi * (1 - xi)
    denom = s + (dk + dk1 - 2 * s) * t
    z = yk + hk * (s * xi * xi + dk * t) / denom
    dz = s * s * (dk1 * xi * xi + 2 * s * t + dk * (1 - xi) ** 2) / denom**2
    return torch.where(inside, z, x), torch.where(inside, torch.log(dz), torch.zeros_like(x))


def rq_spline_inverse(
    z: Tensor,
    params: Tensor,
    bound: float = 5.0,
    *,
    min_width: float = MIN_WIDTH,
    min_height: float = MIN_HEIGHT,
    min_derivative: float = MIN_DERIVATIVE,
) -> Tensor:
    """Map values back through their splines, in the generative direction: ``theta = f^-1(z)`` (exact).

    Parameters
    ----------
    z
        Values on the base side, shape ``[...]``.
    params, bound, min_width, min_height, min_derivative
        As for :func:`rq_spline`.

    Returns
    -------
    Tensor
        ``theta`` of the broadcast shape, with ``rq_spline(theta, params)[0] = z``.
    """
    y, cx, cy, d = _prepare(z, params, bound, min_width=min_width, min_height=min_height, min_derivative=min_derivative)
    inside = (y >= -bound) & (y <= bound)
    yc = y.clamp(-bound, bound)
    k = (torch.searchsorted(cy, yc.unsqueeze(-1), right=True) - 1).clamp(0, cy.shape[-1] - 2)
    xk, wk = _gather(cx, k), _gather(cx, k + 1) - _gather(cx, k)
    yk, hk = _gather(cy, k), _gather(cy, k + 1) - _gather(cy, k)
    dk, dk1 = _gather(d, k), _gather(d, k + 1)
    s = hk / wk
    dy = yc - yk
    a = hk * (s - dk) + dy * (dk1 + dk - 2 * s)
    b = hk * dk - dy * (dk1 + dk - 2 * s)
    c = -s * dy
    xi = 2 * c / (-b - torch.sqrt((b * b - 4 * a * c).clamp_min(0)))
    return torch.where(inside, xk + xi * wk, y)
