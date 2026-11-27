"""Exact posteriors of the toy where it has a likelihood formula: the references that a learned posterior is checked on.

Without scattering and reflection (presets ``straight`` and ``absorb``) every photon flies straight, and it is
detected by sensor ``j`` with a probability ``kappa_j(x0, y0)`` that depends on the vertex only: the angle under which
the sensor arc is seen, weighted with the survival probability ``exp(-s / lambda_a)`` along the path ``s`` to the wall::

    kappa_j = (1 / 2 pi) int over the directions alpha that hit arc j of exp(-s(alpha) / lambda_a) d alpha

With ``N ~ Poisson(Y E)`` photons, the charges are independent, ``n_j ~ Poisson(mu_j)`` with ``mu_j = Y E kappa_j``,
and the likelihood of an event is the product of the Poisson probabilities. Its total charge
``Q = sum_j n_j ~ Poisson(Y E kappa)``, ``kappa = sum_j kappa_j``, which holds at a fixed vertex in every preset.

* :func:`energy_posterior`: one parameter (N1). With a flat prior in ``[E_min, E_max]``, ``q(E | Q)`` is a Gamma
  distribution of shape ``Q + 1`` and rate ``Y kappa``, cut to the prior range.
* :func:`energy_evidence`: the likelihood of ``Q`` integrated over the energy prior,
  ``int Poisson(Q; Y kappa E) pi(E) dE``, the step that removes the energy from the posterior of the vertex (N2.4).
* :func:`grid_posterior`: the exact posterior of one event of ``straight`` or ``absorb`` on a grid of vertices; the
  product of the Poisson factors splits into a multinomial of the pattern and a Poisson of ``Q``::

      q(x0, y0 | n) ~ prod_j (kappa_j / kappa)^{n_j} * int Poisson(Q; Y kappa E) pi(E) dE      (flat prior in the disc)

* :func:`credible_levels`: the density levels of the highest-density regions, e.g. for 68 % and 95 % contours.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import ArrayLike

from ..flash import Prior, SimConfig, preset, sensor_edges

__all__ = [
    "EnergyPosterior",
    "GridPosterior",
    "credible_levels",
    "detection_probability",
    "energy_evidence",
    "energy_posterior",
    "grid_posterior",
]

#: Gauss-Legendre nodes per sensor arc for the absorption integral.
N_NODES = 16
#: Grid points evaluated at once (memory control: points x sensors x nodes values).
POINT_CHUNK = 2048


def _config(config: SimConfig | str) -> SimConfig:
    return preset(config) if isinstance(config, str) else config


def _straight_light(config: SimConfig, what: str) -> None:
    if config.optics.scattering or config.optics.reflection:
        raise ValueError(
            f"{what} needs straight light (no scattering, no reflection): the presets 'straight' and 'absorb'; the "
            "other presets have no closed-form likelihood"
        )


def detection_probability(
    x0: ArrayLike, y0: ArrayLike, config: SimConfig | str = "absorb", *, n_nodes: int = N_NODES
) -> np.ndarray:
    """Probability ``kappa_j(x0, y0)`` that a photon emitted at the vertex is detected by sensor ``j``.

    For ``straight`` light it is exact: the angle ``alpha_j`` under which arc ``j`` is seen from the vertex, over
    ``2 pi``. With absorption the survival probability ``exp(-s / lambda_a)`` is integrated over that angle by
    Gauss-Legendre quadrature (the integrand is smooth in the direction, so the quadrature converges fast also near the
    wall). Both are multiplied by the detection efficiency. At the center ``kappa_j = a exp(-R / lambda_a) / (2 pi R)``.

    Parameters
    ----------
    x0, y0
        Vertex coordinates [m] (broadcast against each other), inside the detector.
    config
        A configuration without scattering and reflection, or the name of a preset (``"straight"``, ``"absorb"``).
    n_nodes
        Quadrature nodes per sensor.

    Returns
    -------
    numpy.ndarray
        Shape ``[..., M]``: one probability per vertex and sensor; ``sum_j kappa_j = kappa``, the light collection.
    """
    config = _config(config)
    _straight_light(config, "detection_probability")
    g, o = config.geometry, config.optics
    x_np, y_np = np.broadcast_arrays(np.asarray(x0, dtype=np.float64), np.asarray(y0, dtype=np.float64))
    if np.any(x_np * x_np + y_np * y_np >= g.radius**2):
        raise ValueError(f"every vertex must lie inside the detector (r0 < {g.radius} m)")
    x, y = (
        torch.tensor(x_np)[..., None],
        torch.tensor(y_np)[..., None],
    )
    edges = torch.from_numpy(sensor_edges(g))  # [M, 2]
    wx, wy = g.radius * torch.cos(edges), g.radius * torch.sin(edges)
    ax, ay = wx[:, 0] - x, wy[:, 0] - y  # vertex -> first end of every arc, [..., M]
    bx, by = wx[:, 1] - x, wy[:, 1] - y  # vertex -> second end
    # angle under which every arc is seen, counterclockwise from its first end: above pi for a vertex between an
    # arc and its chord, hence the modulo
    alpha = torch.remainder(torch.atan2(ax * by - ay * bx, ax * bx + ay * by), 2 * math.pi)
    if not o.absorption:
        kappa = alpha / (2 * math.pi)
    else:
        nodes_np, weights_np = np.polynomial.legendre.leggauss(n_nodes)
        nodes, weights = torch.from_numpy(nodes_np), torch.from_numpy(weights_np)
        angle = torch.atan2(ay, ax)[..., None] + alpha[..., None] * (nodes + 1) / 2  # [..., M, n_nodes]
        p_u = x[..., None] * torch.cos(angle) + y[..., None] * torch.sin(angle)
        r2 = (x * x + y * y)[..., None]
        path = torch.sqrt(p_u * p_u + g.radius**2 - r2) - p_u  # distance to the wall along every direction
        survival = (weights / 2 * torch.exp(-path / o.absorption_length)).sum(-1)
        kappa = alpha / (2 * math.pi) * survival
    return (kappa * config.readout.detection_efficiency).numpy()


# ---------------------------------------------------------------------------------------------- truncated Gamma
def _log_gamma_window(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """``log(P(a, hi) - P(a, lo))`` with the regularized lower incomplete Gamma function ``P`` (float64).

    The difference is taken in the tail where it does not cancel; values below 1e-280 are integrated numerically in
    log space instead.
    """
    a, lo, hi = torch.broadcast_tensors(a, lo, hi)
    lower = torch.special.gammainc(a, hi) - torch.special.gammainc(a, lo)
    upper = torch.special.gammaincc(a, lo) - torch.special.gammaincc(a, hi)
    window = torch.where(lo > a - 1, upper, lower)
    out = torch.log(window.clamp_min(1e-300))
    bad = window < 1e-280
    if bad.any():
        t = torch.linspace(0, 1, 20_001, dtype=torch.float64)
        ab, lb, hb = a[bad][:, None], lo[bad][:, None], hi[bad][:, None]
        tt = lb + (hb - lb) * t
        log_f = (ab - 1) * torch.log(tt) - tt - torch.lgamma(ab)
        log_w = torch.log(hb - lb) - math.log(20_000) + torch.where((t == 0) | (t == 1), math.log(0.5), 0.0)
        out[bad] = torch.logsumexp(log_f + log_w, dim=-1)
    return out


def _t(v: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(np.asarray(v, dtype=np.float64))


def _energy_setup(prior: Prior | SimConfig | str | tuple[float, float] | None, light_yield: float | None) -> tuple:
    if isinstance(prior, str):
        prior = preset(prior)
    if isinstance(prior, SimConfig):
        y = prior.optics.light_yield if light_yield is None else light_yield
        return prior.prior.energy_min, prior.prior.energy_max, y
    if isinstance(prior, Prior):
        return prior.energy_min, prior.energy_max, 1000.0 if light_yield is None else light_yield
    lo, hi = (0.5, 5.0) if prior is None else prior
    return float(lo), float(hi), 1000.0 if light_yield is None else light_yield


@dataclass
class EnergyPosterior:
    """The exact posterior ``q(E | Q)`` at a known vertex: a Gamma distribution cut to ``[low, high]``.

    ``q(E | Q) ~ (Y kappa E)^Q exp(-Y kappa E)`` on ``[low, high]``: shape ``Q + 1``, rate ``Y kappa``. Every array
    broadcasts (a batch of ``Q`` against a grid of energies, for example). Made by :func:`energy_posterior`.
    """

    shape: np.ndarray  #: Q + 1
    rate: np.ndarray  #: Y kappa [1/MeV]
    low: float  #: lower end of the energy prior [MeV]
    high: float  #: upper end of the energy prior [MeV]
    _log_norm: torch.Tensor = field(repr=False)

    def log_pdf(self, energy: ArrayLike) -> np.ndarray:
        """Log-density at ``energy`` [log(1/MeV)]; ``-inf`` outside the prior range."""
        e = _t(energy)
        a, rate = _t(self.shape), _t(self.rate)
        inside = (e >= self.low) & (e <= self.high)
        e_safe = e.clamp(self.low, self.high)
        lp = (a - 1) * torch.log(rate * e_safe) - rate * e_safe - torch.lgamma(a) + torch.log(rate) - self._log_norm
        return torch.where(inside, lp, -math.inf).numpy()

    def pdf(self, energy: ArrayLike) -> np.ndarray:
        """Density at ``energy`` [1/MeV]."""
        return np.exp(self.log_pdf(energy))

    def cdf(self, energy: ArrayLike) -> np.ndarray:
        """Cumulative distribution function at ``energy``."""
        e = _t(energy).clamp(self.low, self.high)
        a, rate = _t(self.shape), _t(self.rate)
        part = _log_gamma_window(a, rate * self.low, rate * e)
        return torch.where(e <= self.low, 0.0, torch.exp(part - self._log_norm)).numpy()

    def _moment_ratio(self, k: int) -> torch.Tensor:
        a, rate = _t(self.shape), _t(self.rate)
        window = _log_gamma_window(a + k, rate * self.low, rate * self.high)
        return torch.exp(torch.lgamma(a + k) - torch.lgamma(a) - k * torch.log(rate) + window - self._log_norm)

    @property
    def mean(self) -> np.ndarray:
        """Posterior mean [MeV]."""
        return self._moment_ratio(1).numpy()

    @property
    def std(self) -> np.ndarray:
        """Posterior standard deviation [MeV]."""
        m1, m2 = self._moment_ratio(1), self._moment_ratio(2)
        return torch.sqrt((m2 - m1 * m1).clamp_min(0)).numpy()


def energy_posterior(
    q: ArrayLike,
    kappa: ArrayLike,
    prior: Prior | SimConfig | str | tuple[float, float] | None = None,
    light_yield: float | None = None,
) -> EnergyPosterior:
    """Return the exact posterior of the energy at a known vertex, given the total charge ``Q`` (N1).

    ``Q ~ Poisson(Y kappa E)`` and a flat prior in ``[E_min, E_max]`` give
    ``q(E | Q) = Gamma(E; Q + 1, Y kappa)`` cut to the prior range and normalized there.

    Parameters
    ----------
    q
        Total charge(s) ``Q``.
    kappa
        Light collection ``kappa = <Q> / (Y E)`` at the vertex.
    prior
        The energy prior: a :class:`~prob_reco_sbi.flash.Prior`, a :class:`~prob_reco_sbi.flash.SimConfig` or the name
        of a preset (then also its light yield), a pair ``(E_min, E_max)`` [MeV] or ``None`` for the stage-1 prior
        (0.5, 5) MeV.
    light_yield
        ``Y`` [photons/MeV]; default 1000 (or the configuration's).

    Returns
    -------
    EnergyPosterior
        With ``pdf``, ``log_pdf``, ``cdf``, ``mean`` and ``std``.
    """
    lo, hi, y = _energy_setup(prior, light_yield)
    a = np.asarray(q, dtype=np.float64) + 1
    rate = y * np.asarray(kappa, dtype=np.float64)
    log_norm = _log_gamma_window(_t(a), _t(rate) * lo, _t(rate) * hi)
    return EnergyPosterior(shape=a, rate=rate, low=lo, high=hi, _log_norm=log_norm)


def energy_evidence(
    q: ArrayLike,
    kappa: ArrayLike,
    prior: Prior | SimConfig | str | tuple[float, float] | None = None,
    light_yield: float | None = None,
    log: bool = True,
) -> np.ndarray:
    """Return the likelihood of the total charge integrated over the energy prior (N2.4).

    ::

        int Poisson(Q; Y kappa E) pi(E) dE
            = [P(Q + 1, Y kappa E_max) - P(Q + 1, Y kappa E_min)] / (Y kappa (E_max - E_min))

    with the flat prior ``pi(E) = 1 / (E_max - E_min)`` and the regularized lower incomplete Gamma function ``P``.

    Parameters
    ----------
    q
        Total charge(s) ``Q``.
    kappa
        Light collection(s) ``kappa = sum_j kappa_j`` (e.g. one per grid point; broadcast against ``q``).
    prior, light_yield
        As for :func:`energy_posterior`.
    log
        Return the natural logarithm (default), which stays finite where the value itself underflows.

    Returns
    -------
    numpy.ndarray
        The (log) evidence, of the broadcast shape of ``q`` and ``kappa``.
    """
    lo, hi, y = _energy_setup(prior, light_yield)
    a = _t(q) + 1
    rate = y * _t(kappa)
    value = _log_gamma_window(a, rate * lo, rate * hi) - torch.log(rate) - math.log(hi - lo)
    return value.numpy() if log else torch.exp(value).numpy()


# ---------------------------------------------------------------------------------------------- grid posterior
@dataclass
class GridPosterior:
    """The exact posterior of one event on a grid of vertices, with the energy integrated analytically.

    Made by :func:`grid_posterior`. The grid is square in a frame turned to the direction of the vertex (radial and
    tangential axes), so ``x`` and ``y`` are 2D arrays of the coordinates of every grid point, as ``matplotlib``'s
    ``contour(x, y, density)`` takes them. All cells have the same area ``cell_area``.
    """

    x: np.ndarray  #: x0 of every grid point [m], shape [n_grid, n_grid]
    y: np.ndarray  #: y0 of every grid point [m], shape [n_grid, n_grid]
    density: np.ndarray  #: posterior density of (x0, y0) [1/m^2], shape [n_grid, n_grid] (0 outside the detector)
    cell_area: float  #: area of one grid cell [m^2]
    energy: np.ndarray  #: grid of E [MeV]
    energy_density: np.ndarray  #: marginal posterior density of E [1/MeV]
    mean: np.ndarray  #: posterior mean of (x0, y0, E)
    covariance: np.ndarray  #: posterior covariance of (x0, y0, E), shape [3, 3]
    edge_mass: float  #: posterior mass in the outer 10 % of the grid on each side (small if the grid holds it)

    @property
    def std(self) -> np.ndarray:
        """Posterior standard deviations of ``(x0, y0, E)``."""
        return np.sqrt(np.diag(self.covariance))

    @property
    def vertex_width(self) -> float:
        """Width of the vertex posterior, ``sqrt(var x0 + var y0)`` [m]."""
        return float(np.sqrt(self.covariance[0, 0] + self.covariance[1, 1]))

    @property
    def energy_width(self) -> float:
        """Standard deviation of the energy posterior [MeV]."""
        return float(np.sqrt(self.covariance[2, 2]))


def _vertex_log_likelihood(
    n: np.ndarray, x: np.ndarray, y: np.ndarray, config: SimConfig, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Log-likelihood of the vertex (energy integrated, up to a constant) and ``kappa`` at points ``x``, ``y``."""
    q = n.sum()
    log_like, kappa = np.empty(len(x)), np.empty(len(x))
    for start in range(0, len(x), POINT_CHUNK):
        k = detection_probability(
            x[start : start + POINT_CHUNK], y[start : start + POINT_CHUNK], config, n_nodes=n_nodes
        )
        total = k.sum(-1)
        pattern = (n * np.log(k / total[:, None])).sum(-1)
        log_like[start : start + POINT_CHUNK] = pattern + energy_evidence(q, total, config)
        kappa[start : start + POINT_CHUNK] = total
    return log_like, kappa


@dataclass
class _Pass:
    """One evaluation of the posterior on a turned square grid."""

    x: np.ndarray  # coordinates of all grid points, [n * n]
    y: np.ndarray
    inside: np.ndarray  # the points inside the detector
    weights: np.ndarray  # normalized posterior weights of the points inside
    kappa: np.ndarray  # light collection at the points inside
    center: np.ndarray  # new center: posterior mean of the vertex
    extent: np.ndarray  # distance from the new center that holds all but 1e-7 of the mass on each side, per axis
    edge_mass: float  # mass in the outer 10 % of the grid


def _grid_pass(
    n: np.ndarray, config: SimConfig, center: np.ndarray, frame: np.ndarray, half: np.ndarray, size: int, n_nodes: int
) -> _Pass:
    a = np.linspace(-half[0], half[0], size)
    b = np.linspace(-half[1], half[1], size)
    aa, bb = (v.ravel() for v in np.meshgrid(a, b, indexing="ij"))
    x = center[0] + aa * frame[0, 0] + bb * frame[1, 0]
    y = center[1] + aa * frame[0, 1] + bb * frame[1, 1]
    inside = x * x + y * y < config.prior.r_max**2
    log_like, kappa = _vertex_log_likelihood(n, x[inside], y[inside], config, n_nodes)
    w = _weights(log_like)
    ai, bi = aa[inside], bb[inside]
    ma, mb = w @ ai, w @ bi
    extent = np.array([_tail_extent(ai - ma, w), _tail_extent(bi - mb, w)])
    band = (np.abs(ai) > 0.9 * half[0]) | (np.abs(bi) > 0.9 * half[1])
    new_center = center + ma * frame[0] + mb * frame[1]
    return _Pass(x, y, inside, w, kappa, new_center, extent, float(w[band].sum()))


def _tail_extent(offset: np.ndarray, w: np.ndarray, tail: float = 1e-7) -> float:
    """Largest distance from 0 of the interval that leaves at most ``tail`` of the mass on each side."""
    order = np.argsort(offset)
    cumulative = np.cumsum(w[order])
    last = len(w) - 1
    low = offset[order][min(int(np.searchsorted(cumulative, tail)), last)]
    high = offset[order][min(int(np.searchsorted(cumulative, 1 - tail)), last)]
    return float(max(-low, high, 0.0))


def _weights(log_like: np.ndarray) -> np.ndarray:
    w = np.exp(log_like - log_like.max())
    return w / w.sum()


def grid_posterior(
    charge: ArrayLike,
    config: SimConfig | str = "absorb",
    *,
    n_grid: int = 161,
    coarse_step: float = 0.02,
    n_energy: int = 201,
    n_nodes: int = N_NODES,
) -> GridPosterior:
    """Return the exact posterior of one event of ``straight`` or ``absorb``: vertex on a grid, energy analytic.

    A coarse pass over the whole disc (step ``coarse_step``) finds the posterior. Quick passes on a 61 x 61 grid,
    turned to the radial and tangential directions of the vertex, then fit its window: widened until it holds the
    posterior, then shrunk to the range that leaves 1e-7 of the mass outside on each side. The final grid has
    ``n_grid x n_grid`` points in that window. The prior is the one of ``config``: the vertex uniform in the disc,
    the energy uniform in its range. The energy is integrated analytically at every grid point
    (:func:`energy_evidence`); its moments and its marginal density follow from the truncated Gamma distributions.

    Parameters
    ----------
    charge
        Sensor charges ``n_j`` of the event, shape ``[M]``.
    config
        The configuration of the simulation (no scattering, no reflection, Poisson photon numbers, no dark noise),
        or a preset name.
    n_grid
        Points per axis of the final grid.
    coarse_step
        Step of the coarse grid [m].
    n_energy
        Points of the energy grid of the marginal density.
    n_nodes
        Quadrature nodes per sensor (see :func:`detection_probability`).

    Returns
    -------
    GridPosterior
        The density on the grid, the energy marginal, the means and the covariance of ``(x0, y0, E)``.
    """
    config = _config(config)
    _straight_light(config, "grid_posterior")
    if config.optics.photon_statistics != "poisson" or config.readout.dark_rate > 0:
        raise ValueError("grid_posterior needs Poisson photon numbers and no dark noise (independent Poisson charges)")
    n = np.asarray(charge, dtype=np.float64).reshape(-1)
    if n.shape != (config.geometry.n_sensors,):
        raise ValueError(f"charge must hold one value per sensor ({config.geometry.n_sensors}), got {n.shape}")
    r_max = config.prior.r_max
    lo, hi, light_yield = _energy_setup(config, None)

    # coarse pass over the disc
    axis = np.arange(-r_max, r_max + coarse_step / 2, coarse_step)
    gx, gy = (v.ravel() for v in np.meshgrid(axis, axis, indexing="ij"))
    keep = gx**2 + gy**2 < r_max**2
    w = _weights(_vertex_log_likelihood(n, gx[keep], gy[keep], config, n_nodes)[0])
    center = np.array([w @ gx[keep], w @ gy[keep]])
    radius = float(np.hypot(*center))
    e_r = center / radius if radius > 1e-3 else np.array([1.0, 0.0])
    frame = np.array([e_r, [-e_r[1], e_r[0]]])  # radial and tangential unit vectors
    coarse = np.array([[w @ ((gx[keep] - center[0]) * f[0] + (gy[keep] - center[1]) * f[1]) ** 2] for f in frame])
    half = np.maximum(5 * np.sqrt(coarse.ravel()), 2.5 * coarse_step)

    # quick passes: widen the window until it holds the posterior, then shrink it to the tails, then the final grid
    for _ in range(12):
        quick = _grid_pass(n, config, center, frame, half, 61, n_nodes)
        center = quick.center
        if quick.edge_mass > 1e-6 and half.max() < 2 * r_max:
            half = 2 * half
            continue
        target = quick.extent + 2 * half / 60  # two cells of margin
        if np.all(target > 0.6 * half):
            break
        half = np.minimum(target, half)
    final = _grid_pass(n, config, center, frame, half, n_grid, n_nodes)
    w, kappa = final.weights, final.kappa
    px, py = final.x[final.inside], final.y[final.inside]
    cell = (2 * half[0] / (n_grid - 1)) * (2 * half[1] / (n_grid - 1))
    density = np.zeros(n_grid * n_grid)
    density[final.inside] = w / cell

    # the energy at every grid point: a Gamma distribution of shape Q + 1 and rate Y kappa, cut to [lo, hi]
    post = energy_posterior(n.sum(), kappa, (lo, hi), light_yield)
    m1, sd = post.mean, post.std
    m2 = sd**2 + m1**2
    mean = np.array([w @ px, w @ py, w @ m1])
    second = np.empty((3, 3))
    columns = (px, py)
    for i in range(2):
        for k in range(2):
            second[i, k] = w @ (columns[i] * columns[k])
        second[i, 2] = second[2, i] = w @ (columns[i] * m1)
    second[2, 2] = w @ m2
    covariance = second - np.outer(mean, mean)

    e_sd = math.sqrt(max(covariance[2, 2], 0.0))
    energy = np.linspace(max(lo, mean[2] - 7 * e_sd), min(hi, mean[2] + 7 * e_sd), n_energy)
    use = w > 1e-12 * w.max()
    part = energy_posterior(n.sum(), kappa[use][:, None], (lo, hi), light_yield)
    energy_density = w[use] @ part.pdf(energy[None, :]) / w[use].sum()
    return GridPosterior(
        x=final.x.reshape(n_grid, n_grid),
        y=final.y.reshape(n_grid, n_grid),
        density=density.reshape(n_grid, n_grid),
        cell_area=float(cell),
        energy=energy,
        energy_density=energy_density,
        mean=mean,
        covariance=covariance,
        edge_mass=final.edge_mass,
    )


def credible_levels(density: ArrayLike, probs: Sequence[float] | float = (0.68, 0.95)) -> np.ndarray:
    """Density levels of the highest-density credible regions, e.g. for the 68 % and 95 % contours.

    The region where the density is at least ``level`` holds the probability ``p``. The density is given on a grid of
    equal cells (any shape; non-finite values count as zero), e.g. ``GridPosterior.density`` or a histogram of
    posterior samples.

    Parameters
    ----------
    density
        Density values on the grid.
    probs
        Probabilities of the regions.

    Returns
    -------
    numpy.ndarray
        One level per probability, in the order of ``probs`` (higher probabilities have lower levels; matplotlib's
        ``contour`` wants them increasing: ``levels=np.sort(levels)``).
    """
    d = np.asarray(density, dtype=np.float64).ravel()
    d = np.sort(np.where(np.isfinite(d), d, 0.0))[::-1]
    total = d.sum()
    if total <= 0:
        raise ValueError("the density must have positive values")
    cumulative = np.cumsum(d) / total
    p = np.atleast_1d(np.asarray(probs, dtype=np.float64))
    if np.any((p <= 0) | (p > 1)):
        raise ValueError(f"probabilities must be in (0, 1], got {p}")
    index = np.minimum(np.searchsorted(cumulative, p), len(d) - 1)
    return d[index]
