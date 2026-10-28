"""Shared pieces of the task-3 reference solutions: "switch the physics on step by step".

Six configurations are compared; each has the same colour and line style in every T3 figure::

    key             configuration                                  colour      line
    straight        preset "straight"                              darkgreen   solid
    absorb          preset "absorb" (= straight + absorption)      royalblue   solid
    absorb-scatter  preset "absorb-scatter"                        firebrick   dash-dot
    full            preset "full"                                  indigo      solid
    scatter-only    preset "straight" with scattering=True         peru        dashed
    reflect-only    preset "straight" with reflection=True         teal        dashed

The cumulative physics ladder is drawn with solid lines, except absorb-scatter, which is dash-dotted so that it differs
from absorb by more than its hue; a single process on top of straight light is dashed (straight + absorption is the
"absorb" preset itself). Analytic expectations are drawn as wide light-grey bands beneath the simulated curves
(:data:`ANALYTIC`). The colours are the paper's named colours, shared with T0.2 and T4.4 through
``common.PRESET_COLOURS`` (the four presets) and ``common.PROCESS_COLOURS`` (the single processes).

Data sets (generated once on the CPU through ``common.dataset`` and cached in ``solutions/stage1_flash/data``)::

    main     100 000 events per configuration from the prior, seed 30 for every configuration: all six see the same
             vertices, energies, event times and photon numbers, only the photon transport differs
    calib    50 000 events per configuration, seed 32 (calibration of the naive estimators of T3.4)
    hits     5 000 events per configuration with the hit list, seed 31 (photon arrival delays)
    vertex   5 fixed vertices x 6 configurations at E = 2 MeV (T3.2): 50 000 events each for straight and absorb,
             20 000 for the others; seed 40 + vertex number

The analytic helpers (subtended angle, expected charge with absorption, azimuth-averaged transmission) are the ones
derived in ``answers/T3.2.md`` and ``answers/T3.3.md``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from common import PRESET_COLOURS, PROCESS_COLOURS, dataset
from numpy.typing import ArrayLike

from prob_reco_sbi.flash import Geometry, SimConfig, SimResult, preset, sensor_edges, sensor_positions

# The thread warm-up that an earlier version of this module did before the first simulation (the first multi-threaded
# CPU transport of a process was not reproducible) now lives in the package: transport.warm_up_cpu_kernels().

__all__ = [
    "ANALYTIC",
    "CALIB_SEED",
    "CATEGORIES",
    "CONFIGS",
    "HITS_SEED",
    "LADDER",
    "MAIN_SEED",
    "N_CALIB",
    "N_HITS",
    "N_MAIN",
    "R_EDGES",
    "SINGLE",
    "VERTEX_ENERGY",
    "VERTICES",
    "binned_mean",
    "calib",
    "chi2_sf",
    "config",
    "expected_charge",
    "hits",
    "main",
    "mean_transmission",
    "photon_delays",
    "quantile_summary",
    "sensor_distance",
    "subtended_angle",
    "vertex_run",
]

MAIN_SEED, HITS_SEED, CALIB_SEED, VERTEX_SEED = 30, 31, 32, 40
N_MAIN, N_CALIB, N_HITS = 100_000, 50_000, 5_000
N_VERTEX = {"straight": 50_000, "absorb": 50_000}  # the configurations with an analytic prediction
N_VERTEX_OTHER = 20_000
VERTEX_ENERGY = 2.0  # MeV

#: Vertex-radius bins of all T3 figures [m]: 0.05 m wide up to 0.9 m, 0.01 m wide in the last 10 cm, where the
#: response changes fastest. Index 0 is the centre bin [0, 0.05), index -1 the wall bin [0.99, 1.0].
R_EDGES = np.concatenate([np.arange(0.0, 0.9 - 1e-9, 0.05), np.arange(0.9, 1.0 + 1e-9, 0.01)])

#: Configurations: preset, overrides and plotting style (one colour and line style per configuration in all figures).
CONFIGS: dict[str, dict[str, Any]] = {
    "straight": {
        "preset": "straight",
        "overrides": {},
        "label": "straight",
        "color": PRESET_COLOURS["straight"],
        "ls": "-",
    },
    "absorb": {
        "preset": "absorb",
        "overrides": {},
        "label": "absorb (= straight + absorption)",
        "short": "absorb",
        "color": PRESET_COLOURS["absorb"],
        "ls": "-",
    },
    "absorb-scatter": {
        "preset": "absorb-scatter",
        "overrides": {},
        "label": "absorb-scatter",
        "color": PRESET_COLOURS["absorb-scatter"],
        "ls": "-.",
    },
    "full": {"preset": "full", "overrides": {}, "label": "full", "color": PRESET_COLOURS["full"], "ls": "-"},
    "scatter-only": {
        "preset": "straight",
        "overrides": {"scattering": True},
        "label": "straight + scattering",
        "color": PROCESS_COLOURS["scatter-only"],
        "ls": "--",
    },
    "reflect-only": {
        "preset": "straight",
        "overrides": {"reflection": True},
        "label": "straight + reflection",
        "color": PROCESS_COLOURS["reflect-only"],
        "ls": "--",
    },
}
for _style in CONFIGS.values():
    _style.setdefault("ls", "-")
    _style.setdefault("short", _style["label"])

#: Line style of analytic expectations: a wide light-grey band drawn beneath the simulated curves.
ANALYTIC: dict[str, Any] = {"color": "#c9c8c3", "lw": 6.0, "solid_capstyle": "butt", "zorder": 1}

LADDER = ["straight", "absorb", "absorb-scatter", "full"]  #: the cumulative physics ladder
SINGLE = ["straight", "absorb", "scatter-only", "reflect-only"]  #: straight light plus one process at a time

#: Photon categories of the detected light (category rule of the package: reflected if reflected at least once, else
#: scattered if scattered at least once, else direct).
CATEGORIES = ("direct", "scattered", "reflected")

#: Test vertices of T3.2 (x0, y0) [m]; sensor 16 sits at phi = pi, the gap between sensors 16 and 17 at pi + pi/32,
#: and the edge of sensor 16 at pi + a / (2 R) = pi + 0.05 rad.
VERTICES: dict[str, dict[str, Any]] = {
    "centre": {"r": 0.0, "phi": 0.0, "label": "centre"},
    "r0.5": {"r": 0.5, "phi": 2.5, "label": "r = 0.5 m"},
    "r0.9-sensor": {"r": 0.9, "phi": math.pi, "label": "r = 0.9 m, facing a sensor"},
    "r0.9-gap": {"r": 0.9, "phi": math.pi + math.pi / 32, "label": "r = 0.9 m, facing a gap"},
    "r0.98-edge": {"r": 0.98, "phi": math.pi + 0.05, "label": "r = 0.98 m, facing a sensor edge"},
}
for _v in VERTICES.values():
    _v["x"] = round(_v["r"] * math.cos(_v["phi"]), 6)
    _v["y"] = round(_v["r"] * math.sin(_v["phi"]), 6)


# ----------------------------------------------------------------------------------------------------------------------
# Configurations and data sets


def config(key: str, **extra: Any) -> SimConfig:
    """Return the configuration of ``key`` (see :data:`CONFIGS`), with optional further overrides."""
    spec = CONFIGS[key]
    return preset(spec["preset"], **spec["overrides"], **extra)


def _dataset(name: str, n_events: int, key: str, seed: int, **kwargs: Any) -> SimResult:
    spec = CONFIGS[key]
    save_hits = bool(kwargs.pop("save_hits", False))
    return dataset(name, n_events, spec["preset"], seed, save_hits=save_hits, **spec["overrides"], **kwargs)


def main(key: str) -> SimResult:
    """Return the main data set of a configuration: 100 000 prior events, the same seed for all configurations."""
    return _dataset(f"t3_main_{key}", N_MAIN, key, MAIN_SEED)


def calib(key: str) -> SimResult:
    """Independent calibration sample of a configuration (50 000 events from the prior)."""
    return _dataset(f"t3_calib_{key}", N_CALIB, key, CALIB_SEED)


def hits(key: str) -> SimResult:
    """Events with the full hit list (5 000 events from the prior), for photon delays."""
    return _dataset(f"t3_hits_{key}", N_HITS, key, HITS_SEED, save_hits=True)


def vertex_run(vertex: str, key: str) -> SimResult:
    """Repeated events at one fixed vertex of :data:`VERTICES` with E = 2 MeV."""
    v = VERTICES[vertex]
    number = list(VERTICES).index(vertex)
    n_events = N_VERTEX.get(key, N_VERTEX_OTHER)
    return _dataset(
        f"t3_vertex_{vertex}_{key}",
        n_events,
        key,
        VERTEX_SEED + number,
        fixed_vertex=(v["x"], v["y"]),
        fixed_energy=VERTEX_ENERGY,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Geometry and analytic expectations


def sensor_distance(x0: np.ndarray, y0: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Distance d_j from each vertex to the centre of each sensor [m], shape ``[..., M]``."""
    sx, sy = sensor_positions(geometry)
    x0 = np.asarray(x0, dtype=np.float64)[..., None]
    y0 = np.asarray(y0, dtype=np.float64)[..., None]
    return np.hypot(sx - x0, sy - y0)


def subtended_angle(x0: np.ndarray, y0: np.ndarray, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Angle Delta_j under which sensor j is seen from the vertex, and the direction to the start of its arc.

    The arc of sensor j runs counter-clockwise from ``A = R (cos phi_1, sin phi_1)`` to ``B = R (cos phi_2, sin
    phi_2)``. Seen from a point inside the disc, the direction to the wall point turns monotonically counter-clockwise
    while the point moves from A to B, so the photons that hit the arc are those emitted between the directions
    ``u = A - p`` and ``v = B - p``: ``Delta_j`` is the counter-clockwise angle from u to v,
    ``atan2(u x v, u . v)`` taken in ``[0, 2 pi)`` (it exceeds pi for a vertex between the arc and its chord).

    Returns
    -------
    delta : numpy.ndarray
        Subtended angles [rad], shape ``[..., M]``.
    alpha_start : numpy.ndarray
        Direction angle of ``u`` [rad], shape ``[..., M]``.
    """
    edges = sensor_edges(geometry)
    radius = geometry.radius
    x0 = np.asarray(x0, dtype=np.float64)[..., None]
    y0 = np.asarray(y0, dtype=np.float64)[..., None]
    ux, uy = radius * np.cos(edges[:, 0]) - x0, radius * np.sin(edges[:, 0]) - y0
    vx, vy = radius * np.cos(edges[:, 1]) - x0, radius * np.sin(edges[:, 1]) - y0
    delta = np.mod(np.arctan2(ux * vy - uy * vx, ux * vx + uy * vy), 2 * np.pi)
    return delta, np.arctan2(uy, ux)


def _wall_distance(x0: float, y0: float, alpha: np.ndarray, radius: float) -> np.ndarray:
    """Distance from (x0, y0) to the wall along the directions alpha: s = -b + sqrt(b^2 - c)."""
    b = x0 * np.cos(alpha) + y0 * np.sin(alpha)
    c = x0 * x0 + y0 * y0 - radius * radius
    return -b + np.sqrt(np.maximum(b * b - c, 0.0))


def expected_charge(x0: float, y0: float, energy: float, cfg: SimConfig, n_nodes: int = 256) -> np.ndarray:
    """Mean charge of every sensor for straight light, with or without bulk absorption.

    ``E[n_j] = epsilon Y E / (2 pi) * integral over the directions alpha that hit sensor j of exp(-s(alpha) /
    lambda_a) d alpha`` (the exponential is 1 without absorption, which gives ``epsilon Y E Delta_j / (2 pi)``).
    The integral is done with Gauss-Legendre quadrature over the direction angle.

    Parameters
    ----------
    x0, y0
        Vertex [m].
    energy
        Energy [MeV].
    cfg
        Configuration without scattering and reflection (the formula ignores both).
    n_nodes
        Quadrature nodes per sensor.

    Returns
    -------
    numpy.ndarray
        Expected charge of each sensor, shape ``[M]``.
    """
    if cfg.optics.scattering or cfg.optics.reflection:
        raise ValueError("the expected-charge formula holds for straight light with or without absorption only")
    delta, alpha_start = subtended_angle(np.float64(x0), np.float64(y0), cfg.geometry)
    scale = cfg.readout.detection_efficiency * cfg.optics.light_yield * energy / (2 * np.pi)
    if not cfg.optics.absorption:
        return scale * delta
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    alpha = alpha_start[:, None] + delta[:, None] * (nodes + 1) / 2
    s = _wall_distance(float(x0), float(y0), alpha, cfg.geometry.radius)
    integral = delta / 2 * (weights * np.exp(-s / cfg.optics.absorption_length)).sum(axis=1)
    return scale * integral


def mean_transmission(r: ArrayLike, length: float, radius: float = 1.0, n_angles: int = 4096) -> np.ndarray:
    """Direction-averaged probability to reach the wall without a bulk interaction, ``<exp(-s / length)>``.

    For a vertex at radius r the average over the emission directions alpha is ``T(r) = 1/(2 pi) * integral of
    exp(-s(alpha) / length) d alpha``; averaged over the vertex azimuth, the direct light collected by the sensors is
    ``c * T(r)`` with the photocoverage c. ``length = inf`` gives 1.
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    if math.isinf(length):
        return np.ones_like(r)
    alpha = (np.arange(n_angles) + 0.5) * 2 * np.pi / n_angles  # direction relative to the radial direction
    b = r[:, None] * np.cos(alpha)  # vertex on the x axis: b = p . u, c = r^2 - R^2
    s = -b + np.sqrt(np.maximum(b * b - (r[:, None] ** 2 - radius**2), 0.0))
    return np.exp(-s / length).mean(axis=1)


# ----------------------------------------------------------------------------------------------------------------------
# Hits and statistics


def photon_delays(result: SimResult) -> dict[str, np.ndarray]:
    """Delay ``t - t0 - d_j / c_n`` [ns] and category (0 direct, 1 scattered, 2 reflected) of every detected photon.

    ``d_j`` is the distance from the vertex to the centre of the hit sensor (the hit point on the arc is not stored),
    so even direct photons scatter by a fraction of a nanosecond around 0 besides the time jitter.
    """
    h = result.hits
    if h is None:
        raise ValueError("the data set has no hit list (generate it with save_hits=True)")
    signal = ~h.is_noise
    event = h.event_id[signal] - result.truth.event_id[0]
    sensor = h.sensor[signal]
    sx, sy = sensor_positions(result.config.geometry)
    x0 = result.truth.x.astype(np.float64)[event]
    y0 = result.truth.y.astype(np.float64)[event]
    d = np.hypot(sx[sensor] - x0, sy[sensor] - y0)
    delay = h.time[signal] - result.truth.t0.astype(np.float64)[event] - d / result.config.optics.speed_of_light
    category = np.where(h.n_reflect[signal] > 0, 2, np.where(h.n_scatter[signal] > 0, 1, 0))
    return {"delay": delay, "category": category, "distance": d}


def binned_mean(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of y, its standard error and the number of entries in bins of x (NaN for empty bins)."""
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    idx, y = idx[ok], np.asarray(y, dtype=np.float64)[ok]
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)[ok]
    nb = len(edges) - 1
    sw = np.bincount(idx, w, nb)
    swy = np.bincount(idx, w * y, nb)
    swy2 = np.bincount(idx, w * y * y, nb)
    count = np.bincount(idx, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = swy / sw
        var = swy2 / sw - mean**2
        sem = np.sqrt(np.maximum(var, 0) / np.maximum(count - 1, 1))
    return mean, sem, count


def quantile_summary(values: np.ndarray, threshold: float = 3.0) -> dict[str, float]:
    """Median, central 68 % half-width, RMS, 90 % and 99 % quantiles and the fraction above ``threshold``."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    q16, q50, q84, q90, q99 = np.quantile(v, [0.16, 0.5, 0.84, 0.9, 0.99])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(q50),
        "half_width_68": float((q84 - q16) / 2),
        "rms": float(np.sqrt(np.mean((v - v.mean()) ** 2))),
        "q90": float(q90),
        "q99": float(q99),
        f"fraction_above_{threshold:g}ns": float(np.mean(v > threshold)),
    }


def chi2_sf(chi2: float, ndf: int) -> float:
    """Survival function P(X > chi2) of the chi-square distribution with ``ndf`` degrees of freedom.

    Regularised upper incomplete gamma function Q(ndf/2, chi2/2): series for x < a + 1, continued fraction otherwise
    (no SciPy needed).
    """
    a, x = ndf / 2.0, chi2 / 2.0
    if x <= 0:
        return 1.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(10_000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefactor))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(log_prefactor) * h


if __name__ == "__main__":
    for name, spec in CONFIGS.items():
        print(f"{name:15s} {spec['label']:36s} {spec['color']}  {spec['ls']}")
