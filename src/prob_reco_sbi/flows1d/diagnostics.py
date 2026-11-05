"""How close is a flow to the truth: the exact KL divergence, QQ data and the Kolmogorov-Smirnov test.

In one dimension the KL divergence from the target ``p`` to the model ``q`` is an ordinary integral::

    KL(p || q) = int p(x) (log p(x) - log q(x)) dx,

which :func:`kl_to_truth` evaluates by the trapezoidal rule on a fine grid: the "truth metric" of every comparison in
the course (in nats; 0 only for ``q = p``). It is the same divergence that maximum-likelihood training minimises, but
exact instead of estimated from samples.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import NamedTuple, Protocol, TypeVar, cast

import numpy as np
import torch

from ._utils import ArrayLike, as_tensor
from .flow import Base, Flow, StandardNormal
from .targets import Mixture1D

__all__ = ["KSResult", "kl_to_truth", "ks_statistic", "qq_points"]


class _HasCDF(Protocol):
    def cdf(self, x: ArrayLike) -> torch.Tensor: ...


class _HasQuantile(Protocol):
    def quantile(self, p: ArrayLike) -> torch.Tensor: ...


_Reference = TypeVar("_Reference")


def _samples(samples: ArrayLike) -> torch.Tensor:
    """Return the samples as a flat float64 tensor without gradient (``flow(x)[0]`` may carry a graph)."""
    return as_tensor(samples).detach().reshape(-1)


def _in_float64(reference: _Reference) -> _Reference:
    """Return a float64 copy of a base (CDF and quantiles at full precision); any other reference unchanged."""
    return cast(_Reference, copy.deepcopy(reference).double()) if isinstance(reference, Base) else reference


class KSResult(NamedTuple):
    """Result of :func:`ks_statistic`."""

    statistic: float  #: the Kolmogorov-Smirnov distance D = max |F_n(x) - F(x)|
    pvalue: float  #: asymptotic p-value of D under the hypothesis that the samples follow F


def kl_to_truth(flow: Flow, target: Mixture1D, n_points: int = 20_001, n_sigma: float = 10.0) -> float:
    """Exact ``KL(p || q)`` [nats] of the flow ``q`` to the target ``p``, by quadrature.

    Parameters
    ----------
    flow
        The model (evaluated on a float64 copy; the flow itself is not changed).
    target
        The target distribution.
    n_points
        Number of grid points of the trapezoidal rule.
    n_sigma
        The grid spans every component's ``mu +- n_sigma sigma`` (:meth:`Mixture1D.interval`).

    Returns
    -------
    float
        The divergence; ``inf`` if the model puts zero density where the target has mass.
    """
    x = torch.linspace(*target.interval(n_sigma), n_points, dtype=torch.float64)
    model = copy.deepcopy(flow).double()
    with torch.no_grad():
        log_q = model.log_prob(x)
    log_p = target.log_pdf(x)
    p = torch.exp(log_p)
    integrand = torch.where(p > 0, p * (log_p - log_q), torch.zeros_like(p))
    return float(torch.trapezoid(integrand, x))


def qq_points(
    samples: ArrayLike, reference: _HasQuantile | None = None, n_points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Return the data of a quantile-quantile plot: reference and sample quantiles at the same probabilities.

    Parameters
    ----------
    samples
        The samples (e.g. ``z = f(x)`` of a flow, or samples drawn from it); a gradient graph is ignored.
    reference
        Anything with a ``quantile`` method: a base (:class:`StandardNormal`, the default; :class:`Uniform`), which
        is evaluated in float64, or a target (:class:`Mixture1D`).
    n_points
        Number of probability levels ``(i + 0.5) / n_points``, ``i = 0 ... n_points - 1``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(theoretical, empirical)``; on the diagonal for samples that follow the reference.
    """
    reference = _in_float64(reference) if reference is not None else StandardNormal().double()
    probs = (np.arange(n_points) + 0.5) / n_points
    empirical = np.quantile(_samples(samples).numpy(), probs)
    theoretical = reference.quantile(torch.as_tensor(probs, dtype=torch.float64)).detach().double().numpy()
    return theoretical, empirical


def _kolmogorov_sf(lam: float) -> float:
    """Survival function of the Kolmogorov distribution, ``P(sqrt(n) D > lam)`` for large ``n``."""
    if lam <= 0:
        return 1.0
    if lam < 1.18:  # theta-function form, fast for small lam
        terms = [math.exp(-((2 * k - 1) ** 2) * math.pi**2 / (8 * lam**2)) for k in range(1, 40)]
        return min(1.0, max(0.0, 1.0 - math.sqrt(2 * math.pi) / lam * sum(terms)))
    terms = [(-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 100)]
    return min(1.0, max(0.0, 2 * sum(terms)))


def ks_statistic(
    samples: ArrayLike, reference: _HasCDF | Callable[[torch.Tensor], torch.Tensor] | None = None
) -> KSResult:
    """One-sample Kolmogorov-Smirnov test of samples against a continuous distribution.

    Parameters
    ----------
    samples
        The samples; a gradient graph is ignored, so ``ks_statistic(flow(x)[0])`` works.
    reference
        Anything with a ``cdf`` method (a base or a target), or a CDF function of a float64 tensor;
        :class:`StandardNormal` by default. A base is evaluated in float64, whatever the type of the flow.

    Returns
    -------
    KSResult
        The distance ``D = max_x |F_n(x) - F(x)|`` and its asymptotic p-value (Kolmogorov distribution with Stephens'
        correction ``(sqrt(n) + 0.12 + 0.11 / sqrt(n)) D``, accurate for n of a few tens and more).
    """
    x = torch.sort(_samples(samples)).values
    n = len(x)
    reference = _in_float64(reference) if reference is not None else StandardNormal().double()
    cdf = reference.cdf if hasattr(reference, "cdf") else reference
    f = torch.as_tensor(cdf(x), dtype=torch.float64).detach().numpy()
    ranks = np.arange(1, n + 1)
    d = float(max(np.max(ranks / n - f), np.max(f - (ranks - 1) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    return KSResult(d, _kolmogorov_sf(lam))
