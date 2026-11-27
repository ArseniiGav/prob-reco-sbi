"""Fixed transforms of the event parameters to unbounded coordinates, with their inverses and log-Jacobians.

The prior of the event parameters has edges: the vertex lies in the disc ``r0 <= R`` and the energy in
``[E_min, E_max]``. A flow with a normal base puts some probability everywhere, so before the flow the parameters are
stretched until the edges lie at infinity::

    (x0, y0)  ->  artanh(r0 / R) / r0 * (x0, y0)                   the wall r0 = R moves to infinity
    E         ->  logit((E - E_min) / (E_max - E_min))              the energy limits move to -+ infinity

The flow then works on the unbounded coordinates ``u``, and the change of variables adds the log-Jacobian of this
fixed map, ``log|det du/dtheta|``. Every sample sent back with the inverse map lies inside the prior range.

The log-Jacobians::

    vertex:  log|det| = -2 log R - log(1 - q^2) + log(artanh(q) / q),    q = r0 / R
    energy:  log|det| = -log(E_max - E_min) - log p - log(1 - p),         p = (E - E_min) / (E_max - E_min)

(the radial map ``p -> g(r) p / r`` of the plane has the determinant ``g'(r) g(r) / r``). These are the inverses of
squashing layers: the relative of the sigmoid layer that a flow with a uniform base needs after it (2a, F5).

Parameter layouts: ``theta[..., 3] = (x0, y0, E)``, ``theta[..., 2] = (x0, y0)`` or ``theta[..., 1] = (E,)``; the
functions :func:`to_unbounded` and :func:`from_unbounded` pick the map by the size of the last axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "ENERGY_RANGE",
    "PARAMETER_NAMES",
    "R_MAX",
    "energy_from_unbounded",
    "energy_to_unbounded",
    "from_unbounded",
    "inside_prior",
    "to_unbounded",
    "vertex_from_unbounded",
    "vertex_to_unbounded",
]

#: The prior of stage 1: the vertex in the disc of radius 1 m, the energy in [0.5, 5] MeV.
R_MAX = 1.0
ENERGY_RANGE = (0.5, 5.0)

#: Names of the parameters for the three layouts, by the size of the last axis of ``theta``.
PARAMETER_NAMES = {1: ("E",), 2: ("x0", "y0"), 3: ("x0", "y0", "E")}

#: Distance from the edge (as a fraction of the range) at which values are clamped, so that the map stays finite.
EDGE = 1e-6

TensorLike = Tensor | np.ndarray | Sequence[float] | float


def _tensor(v: TensorLike) -> Tensor:
    """Return a floating-point tensor: tensors keep their type, everything else becomes float64."""
    if isinstance(v, Tensor) and v.is_floating_point():
        return v
    return torch.as_tensor(v, dtype=torch.float64)


def _artanh_ratio(q2: Tensor) -> Tensor:
    """``artanh(q) / q`` as a function of ``q^2`` (finite value and gradient at ``q = 0``)."""
    small = q2 < 1e-8
    q = torch.sqrt(torch.where(small, torch.ones_like(q2), q2))
    return torch.where(small, 1 + q2 / 3, torch.atanh(q) / q)


def _tanh_ratio(s2: Tensor) -> Tensor:
    """``tanh(s) / s`` as a function of ``s^2`` (finite value and gradient at ``s = 0``)."""
    small = s2 < 1e-8
    s = torch.sqrt(torch.where(small, torch.ones_like(s2), s2))
    return torch.where(small, 1 - s2 / 3, torch.tanh(s) / s)


def vertex_to_unbounded(vertex: TensorLike, r_max: float = R_MAX) -> tuple[Tensor, Tensor]:
    """Map vertices in the disc ``r0 <= R`` to the plane: ``u = artanh(r0 / R) (x0, y0) / r0``.

    Parameters
    ----------
    vertex
        Vertices ``(x0, y0)`` [m], shape ``[..., 2]``. Values at or beyond the wall are clamped to
        ``r0 = (1 - 1e-6) R``.
    r_max
        Radius ``R`` of the disc [m].

    Returns
    -------
    tuple of Tensor
        ``(u, log_det)``: the unbounded coordinates (dimensionless, shape ``[..., 2]``) and
        ``log|det du/d(x0, y0)|`` (shape ``[...]``).
    """
    v = _tensor(vertex)
    q2 = ((v * v).sum(-1) / r_max**2).clamp(max=(1 - EDGE) ** 2)
    ratio = _artanh_ratio(q2)
    u = v / r_max * ratio.unsqueeze(-1)
    log_det = -2 * math.log(r_max) - torch.log1p(-q2) + torch.log(ratio)
    return u, log_det


def vertex_from_unbounded(u: TensorLike, r_max: float = R_MAX) -> Tensor:
    """Inverse of :func:`vertex_to_unbounded`: ``(x0, y0) = R tanh(|u|) u / |u|``, always inside the disc."""
    u_t = _tensor(u)
    return u_t * r_max * _tanh_ratio((u_t * u_t).sum(-1)).unsqueeze(-1)


def energy_to_unbounded(energy: TensorLike, energy_range: tuple[float, float] = ENERGY_RANGE) -> tuple[Tensor, Tensor]:
    """Map energies in ``[E_min, E_max]`` to the real line: ``u = logit((E - E_min) / (E_max - E_min))``.

    Parameters
    ----------
    energy
        Energies [MeV], any shape (one value per energy, no parameter axis). Values at or beyond the limits are
        clamped to ``1e-6`` of the range inside them.
    energy_range
        The limits ``(E_min, E_max)`` [MeV].

    Returns
    -------
    tuple of Tensor
        ``(u, log_det)``: the unbounded coordinate and ``log|du/dE|``, both of the shape of ``energy``.
    """
    lo, hi = energy_range
    p = ((_tensor(energy) - lo) / (hi - lo)).clamp(EDGE, 1 - EDGE)
    log_p, log_q = torch.log(p), torch.log1p(-p)
    return log_p - log_q, -math.log(hi - lo) - log_p - log_q


def energy_from_unbounded(u: TensorLike, energy_range: tuple[float, float] = ENERGY_RANGE) -> Tensor:
    """Inverse of :func:`energy_to_unbounded`: ``E = E_min + (E_max - E_min) sigmoid(u)``, always inside the range."""
    lo, hi = energy_range
    return lo + (hi - lo) * torch.sigmoid(_tensor(u))


def _layout(theta: Tensor) -> int:
    n = theta.shape[-1] if theta.dim() else 0
    if n not in PARAMETER_NAMES:
        raise ValueError(
            f"the last axis of theta must hold (x0, y0, E), (x0, y0) or (E,): 3, 2 or 1 values, got shape "
            f"{tuple(theta.shape)}"
        )
    return n


def to_unbounded(
    theta: TensorLike, r_max: float = R_MAX, energy_range: tuple[float, float] = ENERGY_RANGE
) -> tuple[Tensor, Tensor]:
    """Map event parameters to unbounded coordinates (artanh for the vertex, logit for the energy).

    Parameters
    ----------
    theta
        Event parameters, shape ``[..., 3]`` for ``(x0, y0, E)``, ``[..., 2]`` for ``(x0, y0)`` or ``[..., 1]`` for
        ``(E,)``; vertices in m, energies in MeV.
    r_max
        Radius ``R`` of the prior disc [m].
    energy_range
        The limits ``(E_min, E_max)`` of the energy prior [MeV].

    Returns
    -------
    tuple of Tensor
        ``(u, log_det)``: the unbounded coordinates (shape of ``theta``) and ``log|det du/dtheta|`` (shape
        ``theta.shape[:-1]``).
    """
    t = _tensor(theta)
    n = _layout(t)
    if n == 1:
        u, log_det = energy_to_unbounded(t[..., 0], energy_range)
        return u.unsqueeze(-1), log_det
    u_v, log_det = vertex_to_unbounded(t[..., :2], r_max)
    if n == 2:
        return u_v, log_det
    u_e, log_det_e = energy_to_unbounded(t[..., 2], energy_range)
    return torch.cat([u_v, u_e.unsqueeze(-1)], dim=-1), log_det + log_det_e


def from_unbounded(u: TensorLike, r_max: float = R_MAX, energy_range: tuple[float, float] = ENERGY_RANGE) -> Tensor:
    """Inverse of :func:`to_unbounded`: unbounded coordinates ``[..., 3]``, ``[..., 2]`` or ``[..., 1]`` to theta."""
    u_t = _tensor(u)
    n = _layout(u_t)
    if n == 1:
        return energy_from_unbounded(u_t, energy_range)
    vertex = vertex_from_unbounded(u_t[..., :2], r_max)
    if n == 2:
        return vertex
    return torch.cat([vertex, energy_from_unbounded(u_t[..., 2:], energy_range)], dim=-1)


def inside_prior(
    theta: TensorLike,
    r_max: float = R_MAX,
    energy_range: tuple[float, float] = ENERGY_RANGE,
    tolerance: float = EDGE,
) -> Tensor:
    """Whether event parameters lie inside the prior range (boolean, shape ``theta.shape[:-1]``).

    ``tolerance`` is a relative margin (of ``R`` and of the energy range) that counts values just outside as inside,
    so that rounding errors of float32 data at the wall do not exclude events.
    """
    t = _tensor(theta)
    n = _layout(t)
    ok = torch.ones(t.shape[:-1], dtype=torch.bool, device=t.device)
    if n >= 2:
        ok = ok & ((t[..., :2] ** 2).sum(-1) <= (r_max * (1 + tolerance)) ** 2)
    if n != 2:
        lo, hi = energy_range
        e = t[..., -1]
        margin = tolerance * (hi - lo)
        ok = ok & (e >= lo - margin) & (e <= hi + margin)
    return ok
