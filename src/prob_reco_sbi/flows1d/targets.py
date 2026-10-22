"""Target distributions with an exact density, CDF and quantile function, and the oracle map.

The reference target of the course is "the camel with a shoulder", a mixture of three Gaussians::

    p(x) = 0.30 N(-2.2, 0.25^2) + 0.50 N(0.5, 0.9^2) + 0.20 N(3.5, 1.4^2)

a narrow spike, a broad hump and a wide shoulder on the right. Its parameters are the dictionary :data:`CAMEL`; any
other Gaussian mixture is one line away::

    from prob_reco_sbi.flows1d import CAMEL, Mixture1D

    camel = Mixture1D(**CAMEL)
    twins = Mixture1D(weights=[0.5, 0.5], means=[-1.0, 1.0], stds=[0.3, 0.3])

In one dimension the best possible flow is known in closed form: with the CDF ``F`` of the target, ``z =
Phi^-1(F(x))`` is exactly N(0, 1) distributed (Gaussian base) and ``u = F(x)`` exactly uniform (uniform base).
:func:`oracle_map` returns this map as a transform, so ``Flow([oracle_map(target)])`` has ``log q = log p``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from ._utils import ArrayLike, Seed, as_tensor, make_generator
from .flow import Base, StandardNormal, Uniform
from .transforms import Transform, invert_monotone

__all__ = ["CAMEL", "Mixture1D", "OracleMap", "camel", "oracle_map"]

#: Parameters of the camel with a shoulder: a narrow spike, a broad hump and a wide shoulder on the right.
CAMEL: dict[str, tuple[float, ...]] = {
    "weights": (0.30, 0.50, 0.20),
    "means": (-2.2, 0.5, 3.5),
    "stds": (0.25, 0.90, 1.40),
}

_SQRT2 = math.sqrt(2.0)


class Mixture1D:
    """Mixture of Gaussians on the real line, ``p(x) = sum_k w_k N(x; mu_k, sigma_k^2)``, computed in float64.

    Parameters
    ----------
    weights
        Positive component weights (normalised to sum to 1).
    means
        Component means ``mu_k``.
    stds
        Component standard deviations ``sigma_k > 0``.

    Notes
    -----
    Every method accepts tensors, arrays, lists or numbers and returns a float64 tensor of the same shape.
    """

    def __init__(self, weights: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> None:
        w = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        mu = torch.as_tensor(means, dtype=torch.float64).reshape(-1)
        sd = torch.as_tensor(stds, dtype=torch.float64).reshape(-1)
        if not (len(w) == len(mu) == len(sd) >= 1):
            raise ValueError(f"weights, means and stds need the same length >= 1, got {len(w)}, {len(mu)}, {len(sd)}")
        if not (torch.all(w > 0) and torch.all(sd > 0)):
            raise ValueError("weights and stds must be positive")
        if not (torch.isfinite(w).all() and torch.isfinite(mu).all() and torch.isfinite(sd).all()):
            raise ValueError("weights, means and stds must be finite")
        self.weights, self.means, self.stds = w / w.sum(), mu, sd

    @property
    def params(self) -> dict[str, tuple[float, ...]]:
        """The parameters as a dictionary (``Mixture1D(**target.params)`` is a copy)."""
        return {name: tuple(getattr(self, name).tolist()) for name in ("weights", "means", "stds")}

    def __repr__(self) -> str:
        return "Mixture1D(" + ", ".join(f"{k}={list(v)}" for k, v in self.params.items()) + ")"

    @property
    def mean(self) -> float:
        """Mean of the mixture."""
        return float((self.weights * self.means).sum())

    @property
    def std(self) -> float:
        """Standard deviation of the mixture."""
        second = (self.weights * (self.stds**2 + self.means**2)).sum()
        return math.sqrt(float(second) - self.mean**2)

    def interval(self, n_sigma: float = 10.0) -> tuple[float, float]:
        """Return an interval that holds practically all the mass: every component's ``mu +- n_sigma sigma``."""
        return float((self.means - n_sigma * self.stds).min()), float((self.means + n_sigma * self.stds).max())

    def _standardised(self, x: ArrayLike) -> Tensor:
        """``(x - mu_k) / sigma_k`` with a trailing component axis."""
        return (as_tensor(x)[..., None] - self.means) / self.stds

    def log_pdf(self, x: ArrayLike) -> Tensor:
        """Log-density ``log p(x)`` (log-sum-exp over the components, finite far in the tails)."""
        t = self._standardised(x)
        log_k = -0.5 * t * t - torch.log(self.stds) - 0.5 * math.log(2 * math.pi)
        return torch.logsumexp(log_k + torch.log(self.weights), dim=-1)

    def pdf(self, x: ArrayLike) -> Tensor:
        """Density ``p(x)``."""
        return torch.exp(self.log_pdf(x))

    def cdf(self, x: ArrayLike) -> Tensor:
        """``F(x) = sum_k w_k (1 + erf((x - mu_k) / (sqrt(2) sigma_k))) / 2``.

        Computed as ``(1 + erf(t)) / 2 = erfc(-t) / 2``, which stays accurate in the left tail, where ``1 + erf(t)``
        loses every digit (``F(x)`` of 10^-24 instead of 0 at ten standard deviations).
        """
        return (self.weights * 0.5 * torch.erfc(-self._standardised(x) / _SQRT2)).sum(-1)

    def sf(self, x: ArrayLike) -> Tensor:
        """Survival function ``1 - F(x)``, accurate in the right tail (``erfc``)."""
        return (self.weights * 0.5 * torch.erfc(self._standardised(x) / _SQRT2)).sum(-1)

    def quantile(self, p: ArrayLike) -> Tensor:
        """Quantile function ``F^-1(p)`` by bisection (``-inf`` for ``p <= 0``, ``+inf`` for ``p >= 1``)."""
        p_t = as_tensor(p)
        lo, hi = self.interval(40.0)
        x = invert_monotone(self.cdf, p_t, lo, hi)
        x = torch.where(p_t <= 0, -math.inf, x)
        return torch.where(p_t >= 1, math.inf, x)

    def sample(self, n: int, seed: Seed = None) -> Tensor:
        """Draw ``n`` samples: a component ``k`` with probability ``w_k``, then ``x ~ N(mu_k, sigma_k^2)``.

        ``seed``: an integer, a :class:`torch.Generator` or ``None`` (the global random state).
        """
        generator = make_generator(seed)
        k = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        return self.means[k] + self.stds[k] * torch.randn(n, generator=generator, dtype=torch.float64)

    def entropy(self, n_points: int = 20_001) -> float:
        """Differential entropy ``H(p) = -int p log p dx`` [nats] by the trapezoidal rule.

        The training loss of a flow (mean negative log-likelihood) tends to ``H(p) + KL(p || q)`` for many samples.
        """
        x = torch.linspace(*self.interval(), n_points, dtype=torch.float64)
        log_p = self.log_pdf(x)
        return -float(torch.trapezoid(torch.exp(log_p) * log_p, x))


def _result_dtype(x: ArrayLike) -> torch.dtype:
    """Return the dtype of a floating-point input tensor, float64 for everything else."""
    return x.dtype if isinstance(x, Tensor) and x.is_floating_point() else torch.float64


def camel() -> Mixture1D:
    """Return the reference target of the course, ``Mixture1D(**CAMEL)``."""
    return Mixture1D(**CAMEL)


class OracleMap(Transform):
    """The exact flow of a target: ``z = Phi^-1(F(x))`` (Gaussian base) or ``u = F(x)`` (uniform base).

    ``log|dz/dx| = log p(x) - log phi(z)`` (Gaussian) or ``log p(x)`` (uniform), so a flow made of this single layer
    has ``log q(x) = log p(x)``. In the right tail ``z = -Phi^-1(1 - F(x))`` is computed from the survival function, so
    the map stays accurate where ``F(x)`` rounds to 1. Computed in float64, returned in the dtype of the input; no
    parameters. The inverse is found by bisection.

    Parameters
    ----------
    target
        The target distribution.
    base
        ``"normal"`` or ``"uniform"`` (or a :class:`StandardNormal` / :class:`Uniform` instance).
    """

    def __init__(self, target: Mixture1D, base: Literal["normal", "uniform"] | Base = "normal") -> None:
        super().__init__()
        if isinstance(base, Uniform) or base == "uniform":
            self.kind = "uniform"
        elif isinstance(base, StandardNormal) or base == "normal":
            self.kind = "normal"
        else:
            raise ValueError(f"base must be 'normal', 'uniform', StandardNormal() or Uniform(), got {base!r}")
        self.target = target

    def _map(self, x: Tensor) -> Tensor:
        """Compute the map in float64."""
        if self.kind == "uniform":
            return self.target.cdf(x)
        f, s = self.target.cdf(x), self.target.sf(x)
        return torch.where(f < 0.5, torch.special.ndtri(f), -torch.special.ndtri(s))

    def forward(self, x: ArrayLike) -> tuple[Tensor, Tensor]:
        """Map data values to ``z`` and return ``(z, log|dz/dx|)``."""
        dtype = _result_dtype(x)
        x64 = as_tensor(x)
        z = self._map(x64)
        logdet = self.target.log_pdf(x64)
        if self.kind == "normal":
            logdet = logdet + 0.5 * z * z + 0.5 * math.log(2 * math.pi)  # minus log phi(z)
        return z.to(dtype), logdet.to(dtype)

    def inverse(self, z: ArrayLike) -> Tensor:
        """Map ``z`` back to ``x`` by bisection (``F^-1(Phi(z))`` or ``F^-1(u)``)."""
        lo, hi = self.target.interval(40.0)
        return invert_monotone(self._map, as_tensor(z), lo, hi).to(_result_dtype(z))


def oracle_map(target: Mixture1D, base: Literal["normal", "uniform"] | Base = "normal") -> OracleMap:
    """Return the exact map of ``target`` to the base as a transform: ``Flow([oracle_map(target)])`` has ``q = p``.

    Parameters
    ----------
    target
        The target distribution.
    base
        ``"normal"`` (``z = Phi^-1(F(x))``) or ``"uniform"`` (``u = F(x)``), or a base instance.
    """
    return OracleMap(target, base)
