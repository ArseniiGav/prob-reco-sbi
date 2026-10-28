"""T2.5 Own proposals: ten looks at the detector response that prepare the inference stage.

Run from the repository root::

    python solutions/stage1_flash/task2_proposals.py

The proposals of the task sheet:

 1. charge versus vertex-sensor distance d_j                       t2_5_charge_vs_distance
 2. first-hit time versus d_j (time of flight, slope 1/c_n)        t2_5_first_hit_time
 3. charge-weighted barycentre versus the true vertex              t2_5_barycentre
 4. naive energy from Q with one calibration constant              t2_5_naive_energy
 5. sensor-sensor charge correlations                              t2_5_correlations
 6. photon-fate fractions versus r                                 t2_5_photon_fates
 7. direct / scattered / reflected composition versus r            t2_5_composition

and three more that point at what a learned posterior has to do:

 8. azimuthal modulation near the wall (vertex facing a sensor or a gap)   t2_5_azimuthal_modulation
 9. the energy-radius degeneracy of Q at fixed Q, and how the pattern breaks it   t2_5_energy_radius_degeneracy
10. principal components of the log charges: the input "knows" E and the vertex   t2_5_pca_network_input

Data (CPU, cached in solutions/stage1_flash/data): 200 000 events from the default prior (seed 22, shared with T1 and
T2), 100 000 events at a fixed energy of 2 MeV with vertices from the prior (seed 24, shared with T2.2), and 100 000
events each with the same theta = (0, 0) m, 2 MeV (seed 25) and (0.7, 0) m, 2 MeV (seed 26).
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common import C, dataset, figure_legend, legend_inside, save_results, savefig, style
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

from prob_reco_sbi.flash import SimResult

Y = 1000.0  # light yield [photons / MeV]
M = 32  # sensors
C_N = 0.2  # speed of light [m/ns]
SENSOR_WIDTH = 0.1  # [m]
LAMBDA_A, LAMBDA_S = 1.5, 2.0  # absorption and scattering lengths [m]
SIGMA_T = 1.0  # time jitter [ns]
#: Sequential map of the 2-D histograms, white -> royalblue -> indigo (black stays free for the markers drawn on top).
DENSITY_CMAP = LinearSegmentedColormap.from_list("density", ["#f3f7fc", "#a9c8ea", C["blue"], C["violet"]])
#: Diverging map of the correlation matrices: royalblue (-1) -> white (0) -> firebrick (+1).
CORR_CMAP = LinearSegmentedColormap.from_list("corr", [C["blue"], "#ffffff", C["red"]])
R_EDGES = np.linspace(0, 1, 21)


# ----------------------------------------------------------------------------------------------------------------------
# helpers


def centres(edges: np.ndarray) -> np.ndarray:
    """Bin centres of the bin edges ``edges``."""
    return 0.5 * (edges[1:] + edges[:-1])


def binned_mean(x: np.ndarray, v: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, standard error of the mean and number of entries of ``v`` in bins of ``x``."""
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    k = len(edges) - 1
    count = np.bincount(idx[ok], minlength=k).astype(float)
    total = np.bincount(idx[ok], weights=v[ok], minlength=k)
    squares = np.bincount(idx[ok], weights=v[ok] ** 2, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        var = np.maximum(squares / count - mean**2, 0) * count / np.maximum(count - 1, 1)
        return mean, np.sqrt(var / count), count


def binned_quantiles(x: np.ndarray, v: np.ndarray, edges: np.ndarray, q: tuple[float, ...]) -> np.ndarray:
    """Quantiles ``q`` (percent) of ``v`` in bins of ``x``, shape [len(q), n_bins] (NaN for empty bins)."""
    idx = np.digitize(x, edges) - 1
    order = np.argsort(idx, kind="stable")
    idx_sorted, v_sorted = idx[order], v[order]
    bounds = np.searchsorted(idx_sorted, np.arange(len(edges)))
    out = np.full((len(q), len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        part = v_sorted[bounds[b] : bounds[b + 1]]
        if len(part):
            out[:, b] = np.percentile(part, q)
    return out


def hist2d(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, bins: list[np.ndarray], label: str, per_area: bool = False
) -> None:
    """2-D histogram with a logarithmic colour scale and a colour bar (``per_area``: counts / bin area)."""
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    if per_area:
        h = h / (np.diff(xe)[:, None] * np.diff(ye)[None, :])
    vmin = np.min(h[h > 0])
    h[h == 0] = np.nan
    mesh = ax.pcolormesh(xe, ye, h.T, cmap=DENSITY_CMAP, norm=LogNorm(vmin=vmin, vmax=np.nanmax(h)), rasterized=True)
    plt.colorbar(mesh, ax=ax, pad=0.02).set_label(label)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Cumulative distribution of the standard normal distribution."""
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def median_of_minimum(n: np.ndarray, sigma: float) -> np.ndarray:
    """Median of the smallest of ``n`` independent N(0, sigma^2) values: Phi(m / sigma) = 1 - 2^(-1/n)."""
    grid = np.linspace(-8, 8, 16001)
    cdf = normal_cdf(grid)
    return sigma * np.interp(1 - 2.0 ** (-1.0 / np.asarray(n, float)), cdf, grid)


def r6(v: Any) -> float:
    """Round to six decimals (for the results files)."""
    return round(float(v), 6)


def rlist(values: np.ndarray, digits: int = 5) -> list[float | None]:
    """Round a sequence for the results files, with None for NaN."""
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]


def legend_above(ax: plt.Axes, **kwargs: object) -> None:
    """Unframed legend just above the data area of ``ax``, for panels without an empty corner."""
    kwargs.setdefault("frameon", False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), **kwargs)


def shade(colour: str, fraction: float) -> tuple[float, float, float]:
    """``colour`` mixed with white: 1 = the colour itself, smaller = lighter (sequential shades, as in the paper)."""
    rgb = np.array(to_rgb(colour))
    r, g, b = 1 - fraction * (1 - rgb)
    return (float(r), float(g), float(b))


def vertex_sensor_distance(x: np.ndarray, y: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Distances d_j from every vertex to every sensor centre, shape [n_events, M]."""
    return np.hypot(x[:, None] - sx[None, :], y[:, None] - sy[None, :])


# ----------------------------------------------------------------------------------------------------------------------
# the proposals


def charge_vs_distance(fixed: SimResult, sx: np.ndarray, sy: np.ndarray) -> dict[str, Any]:
    """1. Mean sensor charge versus the vertex-sensor distance, E = 2 MeV."""
    x, y = fixed.truth.x.astype(float), fixed.truth.y.astype(float)
    n = fixed.response.charge.astype(float)
    assert fixed.config.prior.fixed_energy is not None
    energy = float(fixed.config.prior.fixed_energy)
    d = vertex_sensor_distance(x, y, sx, sy).ravel()
    nj = n.ravel()
    edges = np.logspace(np.log10(0.005), np.log10(2.0), 41)
    mean, _, count = binned_mean(d, nj, edges)
    q = binned_quantiles(d, nj, edges, (16, 50, 84))
    grid = np.logspace(np.log10(0.005), np.log10(2.0), 200)
    unscattered = Y * energy * SENSOR_WIDTH / (2 * np.pi * grid) * np.exp(-grid / LAMBDA_A - grid / LAMBDA_S)
    ok = count >= 20

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(
        centres(edges)[ok],
        np.maximum(q[0][ok], 0.5),
        q[2][ok],
        color=C["blue"],
        alpha=0.18,
        lw=0,
        label="16-84 % of the sensors",
    )
    ax.plot(centres(edges)[ok], mean[ok], "o-", color=C["blue"], ms=4, lw=1.3, label="mean charge")
    ax.plot(
        grid,
        unscattered,
        color=C["blue"],
        ls="--",
        lw=1.25,
        label="unscattered light at normal incidence,\n$Y E\\, a\\, e^{-d/\\lambda_a - d/\\lambda_s} / (2\\pi d)$",
    )
    ax.axhline(Y * energy / 2, color=C["ink2"], ls=":", lw=1.3, label="half of all photons, $Y E / 2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3000)
    ax.set_xlabel("Distance from the vertex to the sensor centre $d_j$ [m]")
    ax.set_ylabel("Sensor charge $n_j$ at $E$ = 2 MeV [p.e.]")
    legend_inside(ax, loc="lower left")
    savefig(fig, "t2_5_charge_vs_distance")

    at = {
        f"{v:g}m": float(np.interp(np.log(v), np.log(centres(edges)[ok]), mean[ok])) for v in (0.02, 0.1, 0.3, 1.0, 1.9)
    }
    ref = {
        f"{v:g}m": float(Y * energy * SENSOR_WIDTH / (2 * np.pi * v) * np.exp(-v / LAMBDA_A - v / LAMBDA_S))
        for v in (0.1, 0.3, 1.0, 1.9)
    }
    return {
        "energy_MeV": energy,
        "n_pairs": int(d.size),
        "d_bin_edges_m": rlist(edges, 4),
        "mean_charge_by_d": rlist(mean, 3),
        "median_charge_by_d": rlist(q[1], 1),
        "pairs_per_bin": count.astype(int).tolist(),
        "mean_charge_at": {k: round(v, 2) for k, v in at.items()},
        "unscattered_normal_incidence_at": {k: round(v, 2) for k, v in ref.items()},
        "max_sensor_charge": int(nj.max()),
    }


def first_hit_time(fixed: SimResult, sx: np.ndarray, sy: np.ndarray) -> dict[str, Any]:
    """2. First-hit time minus t0 versus d_j, and its offset from the straight time of flight versus the charge."""
    x, y, t0 = fixed.truth.x.astype(float), fixed.truth.y.astype(float), fixed.truth.t0.astype(float)
    n = fixed.response.charge.astype(float)
    hit = n > 0
    d = vertex_sensor_distance(x, y, sx, sy)
    t_first = fixed.response.time_first.astype(float) - t0[:, None]
    t_mean = fixed.response.time_mean.astype(float) - t0[:, None]
    dh, tf, tm, nh = d[hit], t_first[hit], t_mean[hit], n[hit]
    edges = np.linspace(0, 2.0, 41)
    q_first = binned_quantiles(dh, tf, edges, (50,))[0]
    q_mean = binned_quantiles(dh, tm, edges, (50,))[0]
    fit_range = (centres(edges) > 0.3) & (centres(edges) < 1.8)
    slope, intercept = np.polyfit(centres(edges)[fit_range], q_first[fit_range], 1)
    slope_mean, intercept_mean = np.polyfit(centres(edges)[fit_range], q_mean[fit_range], 1)
    residual_first = tf - dh / C_N
    residual_mean = tm - dh / C_N
    n_edges = np.unique(np.round(np.logspace(0, np.log10(nh.max() + 1), 40))) - 0.5
    res_first = binned_quantiles(nh, residual_first, n_edges, (50,))[0]
    res_mean = binned_quantiles(nh, residual_mean, n_edges, (50,))[0]
    _, _, n_count = binned_mean(nh, residual_first, n_edges)
    ok = n_count >= 30
    n_grid = np.unique(np.round(np.logspace(0, np.log10(nh.max()), 60)))
    jitter_only = median_of_minimum(n_grid, SIGMA_T)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    hist2d(ax, dh, tf, [np.linspace(0, 2.0, 81), np.arange(-6, 30.01, 0.25)], "Sensors per bin")
    ax.plot(centres(edges), q_first, "o", color=C["ink"], ms=3.5, label="median first-hit time")
    ax.plot(centres(edges), q_mean, "s", color=C["red"], ms=3.5, label="median mean time $\\tau_j - t_0$")
    ax.plot(
        [0, 2],
        [0, 2 / C_N],
        color=C["ink"],
        lw=1.5,
        ls="--",
        label=f"straight flight $d_j / c_n$ (slope 1/$c_n$ = {1 / C_N:g} ns/m)",
    )
    ax.set_xlabel("Distance from the vertex to the sensor centre $d_j$ [m]")
    ax.set_ylabel("First-hit time $-\\ t_0$ at $E$ = 2 MeV [ns]")
    legend_inside(ax, loc="upper left")
    ax = axes[1]
    ax.plot(centres(n_edges)[ok], res_first[ok], "o-", color=C["ink"], ms=4, lw=1.2, label="first-hit time")
    ax.plot(centres(n_edges)[ok], res_mean[ok], "s-", color=C["red"], ms=4, lw=1.2, label="mean time")
    ax.plot(
        n_grid,
        jitter_only,
        color=C["ink"],
        ls="--",
        lw=1.25,
        label="first hit expected from the jitter alone:\n"
        f"earliest of $n_j$ photons, $\\sigma_t$ = {SIGMA_T:g} ns (median)",
    )
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-5.2, 1.6)  # room for the legend below the first-hit curve
    ax.set_xlabel("Sensor charge $n_j$ [p.e.]")
    ax.set_ylabel("Median of (time $-\\ t_0 - d_j / c_n$) [ns]")
    legend_inside(ax, loc="lower left")
    savefig(fig, "t2_5_first_hit_time")

    pick = {f"n={v}": int(np.argmin(np.abs(centres(n_edges) - v))) for v in (1, 3, 10, 30, 100, 300)}
    return {
        "fit_range_m": [0.3, 1.8],
        "median_first_hit_slope_ns_per_m": round(float(slope), 3),
        "median_first_hit_intercept_ns": round(float(intercept), 3),
        "median_mean_time_slope_ns_per_m": round(float(slope_mean), 3),
        "median_mean_time_intercept_ns": round(float(intercept_mean), 3),
        "expected_slope_ns_per_m": 1 / C_N,
        "median_residual_first_by_charge": {k: round(float(res_first[i]), 3) for k, i in pick.items()},
        "median_residual_mean_by_charge": {k: round(float(res_mean[i]), 3) for k, i in pick.items()},
        "jitter_only_median_minimum": {
            f"n={v}": round(float(median_of_minimum(np.array([v]), SIGMA_T)[0]), 3) for v in (1, 3, 10, 30, 100, 300)
        },
        "fraction_first_hits_before_t0_plus_d_over_c": r6(np.mean(residual_first < 0)),
    }


def barycentre(data: SimResult, sx: np.ndarray, sy: np.ndarray) -> dict[str, Any]:
    """3. Charge-weighted barycentre of the sensor positions as a naive vertex estimate."""
    x, y = data.truth.x.astype(float), data.truth.y.astype(float)
    n = data.response.charge.astype(float)
    q = n.sum(axis=1)
    bx, by = n @ sx / q, n @ sy / q
    r, br = np.hypot(x, y), np.hypot(bx, by)
    error = np.hypot(bx - x, by - y)
    radial = br - r
    dphi = np.angle(np.exp(1j * (np.arctan2(by, bx) - np.arctan2(y, x))))
    q_br = binned_quantiles(r, br, R_EDGES, (16, 50, 84))
    q_err = binned_quantiles(r, error, R_EDGES, (16, 50, 84))
    q_rad = binned_quantiles(r, radial, R_EDGES, (16, 50, 84))
    ratio = binned_quantiles(r, br / np.maximum(r, 1e-9), R_EDGES, (50,))[0]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    hist2d(ax, r, br, [np.linspace(0, 1, 81), np.linspace(0, 1, 81)], "Events per bin")
    ax.plot([0, 1], [0, 1], color=C["ink2"], ls=":", lw=1.3, label="barycentre = true radius")
    ax.plot(centres(R_EDGES), q_br[1], "o-", color=C["orange"], ms=4, lw=1.3, label="median barycentre radius")
    ax.set_xlabel("True vertex radius $r_0$ [m]")
    ax.set_ylabel("Barycentre radius $|\\mathbf{b}|$ [m]")
    ax.set_aspect("equal")
    legend_inside(ax, loc="upper left")
    ax = axes[1]
    ax.fill_between(centres(R_EDGES), 100 * q_err[0], 100 * q_err[2], color=C["blue"], alpha=0.18, lw=0)
    ax.plot(
        centres(R_EDGES),
        100 * q_err[1],
        "o-",
        color=C["blue"],
        ms=4,
        lw=1.3,
        label="distance $|\\mathbf{b} - \\mathbf{v}|$: median and 16-84 %",
    )
    ax.fill_between(centres(R_EDGES), 100 * q_rad[0], 100 * q_rad[2], color=C["orange"], alpha=0.18, lw=0)
    ax.plot(
        centres(R_EDGES),
        100 * q_rad[1],
        "s-",
        color=C["orange"],
        ms=4,
        lw=1.3,
        label="radial shift $|\\mathbf{b}| - r_0$: median and 16-84 %",
    )
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xlabel("True vertex radius $r_0$ [m]")
    ax.set_ylabel("Barycentre error [cm]")
    legend_inside(ax, loc="upper left")
    savefig(fig, "t2_5_barycentre")

    outer = r > 0.3
    return {
        "r_bin_edges": rlist(R_EDGES, 3),
        "median_ratio_barycentre_to_true_radius": rlist(ratio, 4),
        "median_error_cm": rlist(100 * q_err[1], 2),
        "median_radial_shift_cm": rlist(100 * q_rad[1], 2),
        "median_error_cm_centre_r_below_0.1": round(float(100 * np.median(error[r < 0.1])), 2),
        "median_error_cm_wall_r_above_0.95": round(float(100 * np.median(error[r > 0.95])), 2),
        "median_error_cm_all": round(float(100 * np.median(error)), 2),
        "median_abs_azimuth_error_deg_r_above_0.3": round(float(np.degrees(np.median(np.abs(dphi[outer])))), 3),
        "max_barycentre_radius": round(float(br.max()), 4),
    }


def naive_energy(data: SimResult, fixed: SimResult) -> dict[str, Any]:
    """4. Energy from the total charge with one calibration constant (the disc average): bias and resolution vs r.

    The calibration constant and the collection curve f(r) come from the prior sample; the bias is shown for the
    prior sample (all energies), the resolution for events at a fixed energy of 2 MeV, where the photon-statistics
    limit is simply 1 / sqrt(mean Q).
    """
    x, y = data.truth.x.astype(float), data.truth.y.astype(float)
    energy = data.truth.energy.astype(float)
    q = data.response.charge.sum(axis=1).astype(float)
    r = np.hypot(x, y)
    f = q / (Y * energy)
    f_bar = float(f.mean())
    fine = np.linspace(0, 1, 41)
    f_r, _, _ = binned_mean(r, f, fine)

    def estimates(q_: np.ndarray, r_: np.ndarray, e_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """E_est / E for one constant and for the collection at the true radius."""
        return q_ / (Y * f_bar) / e_, q_ / (Y * np.interp(r_, centres(fine), f_r)) / e_

    naive, oracle = estimates(q, r, energy)
    qn = binned_quantiles(r, naive - 1, R_EDGES, (16, 50, 84))
    qo = binned_quantiles(r, oracle - 1, R_EDGES, (16, 50, 84))

    assert fixed.config.prior.fixed_energy is not None
    e2 = float(fixed.config.prior.fixed_energy)
    q2 = fixed.response.charge.sum(axis=1).astype(float)
    r2 = np.hypot(fixed.truth.x.astype(float), fixed.truth.y.astype(float))
    naive2, oracle2 = estimates(q2, r2, np.full_like(q2, e2))
    qn2 = binned_quantiles(r2, naive2, R_EDGES, (16, 50, 84))
    qo2 = binned_quantiles(r2, oracle2, R_EDGES, (16, 50, 84))
    res_naive = (qn2[2] - qn2[0]) / 2 / qn2[1]  # relative to the estimate's own scale
    res_oracle = (qo2[2] - qo2[0]) / 2 / qo2[1]
    q2_mean, _, _ = binned_mean(r2, q2, R_EDGES)
    poisson = 1 / np.sqrt(q2_mean)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    ax.fill_between(centres(R_EDGES), 100 * qn[0], 100 * qn[2], color=C["red"], alpha=0.15, lw=0)
    ax.plot(
        centres(R_EDGES),
        100 * qn[1],
        "s-",
        color=C["red"],
        ms=4,
        lw=1.3,
        label=f"one constant (disc average $\\bar f$ = {f_bar:.3f}): median, 16-84 %",
    )
    ax.fill_between(centres(R_EDGES), 100 * qo[0], 100 * qo[2], color=C["blue"], alpha=0.15, lw=0)
    ax.plot(
        centres(R_EDGES),
        100 * qo[1],
        "o-",
        color=C["blue"],
        ms=4,
        lw=1.3,
        label="collection $f(r_0)$ at the true radius (needs the vertex)",
    )
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xlabel("True vertex radius $r_0$ [m]")
    ax.set_ylabel("Energy error $E_\\mathrm{est} / E - 1$ [%]")
    legend_inside(ax, loc="upper left", title="Bias, all energies of the prior")
    ax = axes[1]
    ax.plot(centres(R_EDGES), 100 * res_naive, "s-", color=C["red"], ms=4, lw=1.3, label="one constant")
    ax.plot(
        centres(R_EDGES), 100 * res_oracle, "o-", color=C["blue"], ms=4, lw=1.3, label="$f(r_0)$ at the true radius"
    )
    ax.plot(
        centres(R_EDGES),
        100 * poisson,
        color=C["ink"],
        ls="--",
        lw=1.25,
        label="photon statistics alone, $1/\\sqrt{\\bar Q}$",
    )
    ax.set_ylim(0, None)  # linear: the values span one decade, where 10^n labels would be too sparse
    ax.set_xlabel("True vertex radius $r_0$ [m]")
    ax.set_ylabel("Relative energy resolution [%]")
    legend_inside(ax, loc="upper left", title=f"Resolution at {e2:g} MeV,\nhalf of the 16-84 % range / median")
    savefig(fig, "t2_5_naive_energy")

    q16, q50, q84 = np.percentile(naive, [16, 50, 84])
    o16, o50, o84 = np.percentile(oracle, [16, 50, 84])
    return {
        "calibration_constant_disc_average": r6(f_bar),
        "r_bin_edges": rlist(R_EDGES, 3),
        "naive_median_bias_pct": rlist(100 * qn[1], 2),
        "naive_q16_q84_pct": [rlist(100 * qn[0], 2), rlist(100 * qn[2], 2)],
        "true_radius_median_bias_pct": rlist(100 * qo[1], 2),
        "naive_bias_pct_centre_bin": round(float(100 * qn[1][0]), 2),
        "naive_bias_pct_wall_bin": round(float(100 * qn[1][-1]), 2),
        "naive_bias_zero_crossing_r_m": round(float(np.interp(0, qn[1], centres(R_EDGES))), 3),
        "E2MeV_naive_resolution_pct": rlist(100 * res_naive, 2),
        "E2MeV_true_radius_resolution_pct": rlist(100 * res_oracle, 2),
        "E2MeV_poisson_limit_pct": rlist(100 * poisson, 2),
        "all_events_naive_half_width_pct": round(float(100 * (q84 - q16) / 2 / q50), 2),
        "all_events_true_radius_half_width_pct": round(float(100 * (o84 - o16) / 2 / o50), 2),
        "all_events_poisson_median_pct": round(float(100 * np.median(1 / np.sqrt(q))), 2),
    }


def independence_test(charge: np.ndarray, n_permutations: int = 50, seed: int = 5) -> dict[str, Any]:
    """Are the sensor charges independent? z = r sqrt(N) of all pairs, against a permutation null, and var/mean.

    Shuffling every column on its own keeps the distribution of each sensor charge and destroys any dependence, so the
    largest |z| of the shuffled samples shows what pure noise gives for these 496 pairs.
    """
    n = len(charge)
    upper = np.triu_indices(M, 1)
    z = np.corrcoef(charge.T)[upper] * math.sqrt(n)
    rng = np.random.default_rng(seed)
    null_max = []
    for _ in range(n_permutations):
        shuffled = np.stack([rng.permutation(column) for column in charge.T], axis=1)
        null_max.append(np.abs(np.corrcoef(shuffled.T)[upper]).max() * math.sqrt(n))
    var_over_mean = charge.var(axis=0, ddof=1) / charge.mean(axis=0)
    q = charge.sum(axis=1)
    return {
        "z_std_over_pairs": round(float(z.std()), 4),
        "z_max_abs": round(float(np.abs(z).max()), 3),
        "permutation_null_z_max_abs_median": round(float(np.median(null_max)), 3),
        "permutation_p_value_of_z_max": round(float(np.mean(np.array(null_max) >= np.abs(z).max())), 3),
        "n_permutations": n_permutations,
        "sensor_var_over_mean_min_max": [round(float(var_over_mean.min()), 4), round(float(var_over_mean.max()), 4)],
        "total_charge_var_over_mean": round(float(q.var(ddof=1) / q.mean()), 4),
    }


def correlations(data: SimResult, fixed: SimResult, centre: SimResult, offcentre: SimResult) -> dict[str, Any]:
    """5. Sensor-sensor charge correlation matrices for several event selections."""
    r1 = np.hypot(data.truth.x.astype(float), data.truth.y.astype(float))
    r2 = np.hypot(fixed.truth.x.astype(float), fixed.truth.y.astype(float))
    sets = [
        ("same $\\theta$: (0, 0) m, 2 MeV", centre.response.charge, C["ink"], "o"),
        ("same $\\theta$: (0.7, 0) m, 2 MeV", offcentre.response.charge, C["ink2"], "D"),
        ("2 MeV, vertices $r_0 <$ 0.2 m", fixed.response.charge[r2 < 0.2], C["blue"], "s"),
        ("2 MeV, vertices 0.6 $\\leq r_0 <$ 0.8 m", fixed.response.charge[(r2 >= 0.6) & (r2 < 0.8)], C["aqua"], "^"),
        ("prior energies, vertices $r_0 <$ 0.2 m", data.response.charge[r1 < 0.2], C["orange"], "v"),
    ]
    out: dict[str, Any] = {}
    fig = plt.figure(figsize=(11.2, 6.4))
    grid = fig.add_gridspec(2, 3)
    norm = Normalize(-1, 1)
    sep = np.arange(M // 2 + 1)
    line_ax = fig.add_subplot(grid[1, 2])
    matrix_axes = []
    for k, (label, charge, col, mk) in enumerate(sets):
        corr = np.corrcoef(charge.astype(float).T)
        by_sep = np.array([np.mean([corr[j, (j + s) % M] for j in range(M)]) for s in sep])
        off = corr[~np.eye(M, dtype=bool)]
        ax = fig.add_subplot(grid[k // 3, k % 3])
        matrix_axes.append(ax)
        image = ax.imshow(corr, cmap=CORR_CMAP, norm=norm, origin="lower", interpolation="nearest")
        ax.set_title(f"{label}\n{len(charge):,} events", fontsize=9.5)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])
        ax.set_xlabel("Sensor $j$")
        ax.set_ylabel("Sensor $k$")
        line_ax.plot(sep, by_sep, marker=mk, ms=4, lw=1.2, color=col, label=label)
        out[f"set{k + 1}"] = {
            "selection": label.replace("$\\theta$", "theta")
            .replace("$r_0 <$", "r <")
            .replace("$\\leq r_0 <$", "<= r <"),
            "n_events": len(charge),
            "mean_corr_by_separation": rlist(by_sep, 4),
            "mean_off_diagonal": round(float(off.mean()), 4),
            "max_abs_off_diagonal": round(float(np.abs(off).max()), 4),
            "statistical_noise_1_over_sqrt_n": round(1 / math.sqrt(len(charge)), 4),
        }
        if k < 2:  # the same theta every time: test that the sensors are independent
            out[f"set{k + 1}"] |= independence_test(charge.astype(float))
    fig.colorbar(image, ax=matrix_axes, shrink=0.8, pad=0.02, label="Correlation of the sensor charges")
    line_ax.axhline(0, color=C["ink2"], lw=0.8)
    line_ax.set_xlabel("Sensor separation $|j - k|$ (around the ring)")
    line_ax.set_ylabel("Mean correlation")
    line_ax.set_xticks([0, 4, 8, 12, 16])
    line_ax.set_ylim(-0.5, 1.02)
    line_ax.set_title("Correlation versus separation", fontsize=9.5)
    figure_legend(fig, *line_ax.get_legend_handles_labels(), where="top", ncol=3)
    savefig(fig, "t2_5_correlations")
    return out


def photon_fates(data: SimResult) -> dict[str, Any]:
    """6. Where the photons end, versus the vertex radius."""
    r = np.hypot(data.truth.x.astype(float), data.truth.y.astype(float))
    ph = data.photons
    emitted = ph.n_emitted.astype(float)
    kinds = [
        ("detected by a sensor", "n_detected", C["blue"], "o"),
        ("absorbed in the bulk", "n_absorbed_bulk", C["orange"], "s"),
        ("absorbed by the wall between the sensors", "n_absorbed_wall", C["violet"], "D"),
        ("absorbed by a sensor (efficiency < 1)", "n_absorbed_sensor", C["aqua"], "^"),
        ("lost (still in flight after 80 steps)", "n_lost", C["red"], "v"),
    ]
    idx = np.digitize(r, R_EDGES) - 1
    tot = np.bincount(idx, weights=emitted, minlength=len(R_EDGES) - 1)[: len(R_EDGES) - 1]
    out: dict[str, Any] = {"r_bin_edges": rlist(R_EDGES, 3)}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, key, col, mk in kinds:
        counts = getattr(ph, key).astype(float)
        frac = np.bincount(idx, weights=counts, minlength=len(R_EDGES) - 1)[: len(R_EDGES) - 1] / tot
        total = counts.sum() / emitted.sum()
        ax.plot(centres(R_EDGES), frac, marker=mk, ms=4, lw=1.3, color=col, label=f"{label} (all: {100 * total:.1f} %)")
        out[key] = {"by_r": rlist(frac, 5), "disc_average": r6(total)}
    ax.set_xlabel("Vertex radius $r_0$ [m]")
    ax.set_ylabel("Fraction of the emitted photons")
    ax.set_ylim(-0.02, 1.0)  # room for the legend above the curves
    legend_inside(ax, loc="upper left")
    savefig(fig, "t2_5_photon_fates")
    return out


def composition(data: SimResult, centre: SimResult) -> dict[str, Any]:
    """7. Direct, scattered and reflected shares of the detected light versus r."""
    r = np.hypot(data.truth.x.astype(float), data.truth.y.astype(float))
    ph = data.photons
    keys = [
        ("direct", "n_detected_direct", C["blue"], "o"),
        ("scattered", "n_detected_scattered", C["violet"], "s"),
        ("reflected", "n_detected_reflected", C["aqua"], "D"),
    ]
    idx = np.digitize(r, R_EDGES) - 1
    nb = len(R_EDGES) - 1
    det = np.bincount(idx, weights=ph.n_detected.astype(float), minlength=nb)[:nb]
    emi = np.bincount(idx, weights=ph.n_emitted.astype(float), minlength=nb)[:nb]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    bottom = np.zeros(nb)
    out: dict[str, Any] = {"r_bin_edges": rlist(R_EDGES, 3)}
    for label, key, col, mk in keys:
        counts = np.bincount(idx, weights=getattr(ph, key).astype(float), minlength=nb)[:nb]
        share = counts / det
        per_emitted = counts / emi
        axes[0].plot(centres(R_EDGES), 100 * share, marker=mk, ms=4, lw=1.3, color=col, label=label)
        axes[1].fill_between(centres(R_EDGES), bottom, bottom + per_emitted, color=col, alpha=0.55, lw=0, label=label)
        bottom = bottom + per_emitted
        out[label] = {
            "share_of_detected_by_r": rlist(share, 4),
            "per_emitted_by_r": rlist(per_emitted, 5),
            "disc_average_share": r6(getattr(ph, key).sum() / ph.n_detected.sum()),
            "exact_centre_share": r6(getattr(centre.photons, key).sum() / centre.photons.n_detected.sum()),
        }
    axes[1].plot(centres(R_EDGES), bottom, color=C["ink"], lw=1.2, label="all detected light")
    handles: list[Any] = [
        (plt.Line2D([], [], color=col, marker=mk, lw=1.3), Patch(color=col, alpha=0.55)) for _, _, col, mk in keys
    ]
    handles.append(plt.Line2D([], [], color=C["ink"], lw=1.2))
    labels = [label for label, *_ in keys] + ["all detected light"]
    axes[0].set_xlabel("Vertex radius $r_0$ [m]")
    axes[0].set_ylabel("Share of the detected light [%]")
    axes[0].set_ylim(0, 100)
    axes[1].set_xlabel("Vertex radius $r_0$ [m]")
    axes[1].set_ylabel("Detected photons per emitted photon")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 0.45)
    figure_legend(
        fig,
        handles,
        labels,
        where="top",
        ncol=4,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        title="Detected light: direct (no scattering, no reflection), scattered (no reflection), "
        "reflected at least once",
    )
    savefig(fig, "t2_5_composition")
    out["n_events_exact_centre"] = len(centre)
    return out


def azimuthal_modulation(data: SimResult) -> dict[str, Any]:
    """8. Near the wall the response depends on the azimuth: vertex in front of a sensor or in front of a gap."""
    x, y = data.truth.x.astype(float), data.truth.y.astype(float)
    energy = data.truth.energy.astype(float)
    n = data.response.charge.astype(float)
    q = n.sum(axis=1)
    r = np.hypot(x, y)
    pitch = 2 * np.pi / M
    offset = np.arctan2(y, x) / pitch
    offset_deg = np.abs(offset - np.round(offset)) * 360 / M  # 0 = facing a sensor centre, 5.625 = facing a gap centre
    half_width_deg = np.degrees(SENSOR_WIDTH / 2)  # arc half-width a / (2 R) for R = 1 m
    edges = np.linspace(0, 180 / M, 13)
    slices = [(0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0001)]
    colours = [shade(C["blue"], f) for f in (0.45, 0.63, 0.81, 1.0)]  # lighter = closer to the centre
    markers = ["o", "s", "D", "^"]
    out: dict[str, Any] = {"offset_bin_edges_deg": rlist(edges, 4), "sensor_half_width_deg": round(half_width_deg, 4)}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for (lo, hi), col, mk in zip(slices, colours, markers, strict=True):
        sel = (r >= lo) & (r < hi)
        f_mean, f_err, _ = binned_mean(offset_deg[sel], q[sel] / (Y * energy[sel]), edges)
        m_mean, m_err, _ = binned_mean(offset_deg[sel], n[sel].max(axis=1) / q[sel], edges)
        label = f"{lo:g} $\\leq r_0 <$ {min(hi, 1):g} m" if hi < 1 else f"{lo:g} $\\leq r_0 \\leq$ 1 m"
        axes[0].errorbar(
            centres(edges), f_mean, yerr=f_err, fmt=mk + "-", ms=4, lw=1.3, color=col, capsize=0, label=label
        )
        axes[1].errorbar(
            centres(edges), 100 * m_mean, yerr=100 * m_err, fmt=mk + "-", ms=4, lw=1.3, color=col, capsize=0
        )
        out[f"r {lo}-{min(hi, 1)}"] = {
            "n_events": int(sel.sum()),
            "collection_by_offset": rlist(f_mean, 4),
            "brightest_sensor_share_by_offset": rlist(m_mean, 4),
            "collection_facing_sensor_centre": round(float(f_mean[0]), 4),
            "collection_facing_gap_centre": round(float(f_mean[-1]), 4),
        }
    for ax in axes:
        ax.axvspan(0, half_width_deg, color=C["grid"], alpha=0.6, lw=0, zorder=0)
        ax.set_xlim(0, 180 / M)
        ax.set_xlabel("Azimuth of the vertex from the nearest sensor centre [deg]")
    axes[0].set_ylabel("Light collection $Q / (Y E)$")
    axes[1].set_ylabel("Share of the light in the brightest sensor [%]")
    handles, _ = axes[0].get_legend_handles_labels()
    handles.append(Patch(color=C["grid"], alpha=0.6, label="the vertex faces the sensor arc"))
    figure_legend(fig, handles, where="top", ncol=5)
    savefig(fig, "t2_5_azimuthal_modulation")
    return out


def energy_radius_degeneracy(data: SimResult, sx: np.ndarray, sy: np.ndarray) -> dict[str, Any]:
    """9. Events with the same total charge: their (r, E) lie on a ridge; the charge pattern selects a part of it."""
    x, y = data.truth.x.astype(float), data.truth.y.astype(float)
    energy = data.truth.energy.astype(float)
    n = data.response.charge.astype(float)
    q = n.sum(axis=1)
    r = np.hypot(x, y)
    br = np.hypot(n @ sx / q, n @ sy / q)
    fine = np.linspace(0, 1, 41)
    f_r, _, _ = binned_mean(r, q / (Y * energy), fine)
    r_grid = np.linspace(0, 1, 200)
    f_grid = np.interp(r_grid, centres(fine), f_r)
    q_values = [400, 800, 1200]
    colours = [C["blue"], C["orange"], C["violet"]]
    out: dict[str, Any] = {"window_relative": 0.015}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    ax = axes[0]
    for qv, col in zip(q_values, colours, strict=True):
        sel = np.abs(q - qv) < 0.015 * qv
        ax.plot(r[sel], energy[sel], ".", ms=2.2, color=col, alpha=0.5, rasterized=True)
        ax.plot(
            r_grid, qv / (Y * f_grid), color=col, lw=1.6, label=f"$Q$ = {qv} p.e. $\\pm$ 1.5 % ({sel.sum()} events)"
        )
        out[f"Q{qv}"] = {
            "n_events": int(sel.sum()),
            "energy_range_from_mean_collection_MeV": [
                round(qv / (Y * f_grid.max()), 3),
                round(qv / (Y * f_grid.min()), 3),
            ],
            "energy_q16_q50_q84": rlist(np.percentile(energy[sel], [16, 50, 84]), 3),
            "energy_std": round(float(energy[sel].std()), 4),
        }
    ax.set_xlabel("True vertex radius $r_0$ [m]")
    ax.set_ylabel("True energy $E$ [MeV]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.8, 5)
    legend_above(ax, ncol=1, title="lines: $E = Q / (Y f(r_0))$ with the mean collection $f(r_0)$")
    ax = axes[1]
    qv = 800
    sel = np.abs(q - qv) < 0.015 * qv
    bins = np.linspace(1.6, 3.4, 46)
    selections = [
        ("all events with this $Q$", "all", sel, C["ink"]),
        ("and barycentre radius $|\\mathbf{b}| <$ 0.2 m", "barycentre_below_0.2m", sel & (br < 0.2), C["blue"]),
        (
            "and 0.6 $\\leq |\\mathbf{b}| <$ 0.7 m",
            "barycentre_0.6_to_0.7m",
            sel & (br >= 0.6) & (br < 0.7),
            C["orange"],
        ),
    ]
    for label, key, s, col in selections:
        ax.hist(
            energy[s],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            alpha=0.6,
            color=col,
            label=f"{label}: $\\sigma_E$ = {energy[s].std():.2f} MeV ({s.sum()} events)",
        )
        out[f"Q{qv}_{key}"] = {
            "n_events": int(s.sum()),
            "energy_mean": round(float(energy[s].mean()), 4),
            "energy_std": round(float(energy[s].std()), 4),
        }
    ax.set_xlabel("True energy $E$ [MeV]")
    ax.set_ylabel("Density [per MeV]")
    legend_above(ax, ncol=1, title=f"$Q$ = {qv} p.e. $\\pm$ 1.5 %")
    savefig(fig, "t2_5_energy_radius_degeneracy")
    return out


def pca_network_input(data: SimResult) -> dict[str, Any]:
    """10. Principal components of the charge half of the network input."""
    x, y = data.truth.x.astype(float), data.truth.y.astype(float)
    energy = data.truth.energy.astype(float)
    r = np.hypot(x, y)
    log_n = np.log1p(data.response.charge.astype(float))
    centred = log_n - log_n.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    explained = s**2 / np.sum(s**2)
    pcs = centred @ vt[:5].T
    # the sign of a principal component is arbitrary: orient PC1 so that it grows with the energy
    if np.corrcoef(pcs[:, 0], np.log(energy))[0, 1] < 0:
        pcs[:, 0] *= -1
        vt[0] *= -1
    phi_pc = np.arctan2(pcs[:, 2], pcs[:, 1])
    phi_true = np.arctan2(y, x)
    same = np.abs(np.mean(np.exp(1j * (phi_pc - phi_true))))
    mirrored = np.abs(np.mean(np.exp(1j * (phi_pc + phi_true))))
    outer = r > 0.3
    same_outer = np.abs(np.mean(np.exp(1j * (phi_pc[outer] - phi_true[outer]))))
    mirrored_outer = np.abs(np.mean(np.exp(1j * (phi_pc[outer] + phi_true[outer]))))
    rho_pc = np.hypot(pcs[:, 1], pcs[:, 2])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    ax = axes[0, 0]
    k = np.arange(1, 11)
    ax.bar(k, 100 * explained[:10], color=C["blue"])
    ax.set_xticks(k)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Share of the variance [%]")
    ax.set_title(f"Variance per component (first three: {100 * explained[:3].sum():.0f} %)", fontsize=10)
    ax = axes[0, 1]
    j = np.arange(M)
    for i, (col, mk) in enumerate(
        zip([C["ink"], C["blue"], C["orange"], C["aqua"], C["violet"]], ["o", "s", "D", "^", "v"], strict=True)
    ):
        ax.plot(j, vt[i], marker=mk, ms=3, lw=1.1, color=col, label=f"PC{i + 1}")
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.set_xlabel("Sensor $j$")
    ax.set_ylabel("Weight of sensor $j$ in the component")
    legend_above(
        ax,
        ncol=5,
        title="flat (PC1), then cos / sin of $\\phi_j$ (PC2, PC3) and of $2\\phi_j$ (PC4, PC5)",
    )
    ax = axes[1, 0]
    show = slice(0, 6000)
    sc = ax.scatter(np.log(energy[show]), pcs[show, 0], c=r[show], s=2, cmap="viridis", vmin=0, vmax=1, rasterized=True)
    ax.set_xlabel("True energy: $\\log(E / \\mathrm{MeV})$")
    ax.set_ylabel("PC1")
    ax.set_title(f"PC1 tracks the energy (correlation {np.corrcoef(pcs[:, 0], np.log(energy))[0, 1]:.3f})", fontsize=10)
    ax = axes[1, 1]
    ax.scatter(pcs[show, 1], pcs[show, 2], c=r[show], s=2, cmap="viridis", vmin=0, vmax=1, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_title("PC2 and PC3 map the vertex position", fontsize=10)
    fig.colorbar(sc, ax=axes[1, :], pad=0.02, shrink=0.9, label="True vertex radius $r_0$ [m]")
    savefig(fig, "t2_5_pca_network_input")
    return {
        "explained_variance_first10": rlist(explained[:10], 5),
        "explained_first3": round(float(explained[:3].sum()), 4),
        "explained_first5": round(float(explained[:5].sum()), 4),
        "corr_PC1_logE": round(float(np.corrcoef(pcs[:, 0], np.log(energy))[0, 1]), 4),
        "corr_PC1_r": round(float(np.corrcoef(pcs[:, 0], r)[0, 1]), 4),
        "azimuth_agreement_PC2_PC3_all": round(float(max(same, mirrored)), 4),
        "azimuth_agreement_PC2_PC3_r_above_0.3": round(float(max(same_outer, mirrored_outer)), 4),
        "azimuth_mirrored": bool(mirrored > same),
        "corr_PC23_radius_vs_true_r": round(float(np.corrcoef(rho_pc, r)[0, 1]), 4),
        "note": "azimuth agreement = |mean exp(i (phi_PC +- phi_true))|, 1 = identical up to a rotation",
    }


def main() -> None:
    """Run the ten proposals and store their numbers in results/T2.json under T2.5."""
    style()
    data = dataset("t1t2_full_prior_200k", 200_000, "full", seed=22)
    fixed = dataset("t2_full_E2_100k", 100_000, "full", seed=24, fixed_energy=2.0)
    centre = dataset("t2_5_full_centre_E2_100k", 100_000, "full", seed=25, fixed_vertex=(0.0, 0.0), fixed_energy=2.0)
    offcentre = dataset("t2_5_full_x0.7_E2_100k", 100_000, "full", seed=26, fixed_vertex=(0.7, 0.0), fixed_energy=2.0)
    det = data.detector
    sx, sy = det.sensor_x, det.sensor_y
    results = {
        "charge_vs_distance": charge_vs_distance(fixed, sx, sy),
        "first_hit_time": first_hit_time(fixed, sx, sy),
        "barycentre": barycentre(data, sx, sy),
        "naive_energy": naive_energy(data, fixed),
        "correlations": correlations(data, fixed, centre, offcentre),
        "photon_fates": photon_fates(data),
        "composition": composition(data, centre),
        "azimuthal_modulation": azimuthal_modulation(data),
        "energy_radius_degeneracy": energy_radius_degeneracy(data, sx, sy),
        "pca_network_input": pca_network_input(data),
    }
    path = save_results("T2", {"T2.5": results})

    b, e, fh = results["barycentre"], results["naive_energy"], results["first_hit_time"]
    comp, az, pca = results["composition"], results["azimuthal_modulation"], results["pca_network_input"]
    print(f"1. charge at d = 0.1 / 1.0 m: {results['charge_vs_distance']['mean_charge_at']}")
    print(
        f"2. first-hit slope {fh['median_first_hit_slope_ns_per_m']} ns/m (expected {1 / C_N:g}), "
        f"intercept {fh['median_first_hit_intercept_ns']} ns"
    )
    print(
        f"3. barycentre error {b['median_error_cm_centre_r_below_0.1']} cm (centre) -> "
        f"{b['median_error_cm_wall_r_above_0.95']} cm (wall)"
    )
    print(
        f"4. naive energy bias {e['naive_bias_pct_centre_bin']} % (centre) -> {e['naive_bias_pct_wall_bin']} % (wall); "
        f"spread {e['all_events_naive_half_width_pct']} % (all), "
        f"{e['all_events_true_radius_half_width_pct']} % with f(r)"
    )
    print(
        "5. correlations: "
        + ", ".join(f"{v['selection']}: nn {v['mean_corr_by_separation'][1]}" for v in results["correlations"].values())
    )
    print(
        f"6. fates (disc): detected {results['photon_fates']['n_detected']['disc_average']:.4f}, bulk "
        f"{results['photon_fates']['n_absorbed_bulk']['disc_average']:.4f}, "
        f"wall {results['photon_fates']['n_absorbed_wall']['disc_average']:.4f}"
    )
    print(
        f"7. composition centre {[comp[k]['exact_centre_share'] for k in ('direct', 'scattered', 'reflected')]}, "
        f"disc {[comp[k]['disc_average_share'] for k in ('direct', 'scattered', 'reflected')]}"
    )
    wall = az["r 0.95-1"]
    print(
        f"8. r >= 0.95 m: collection {wall['collection_facing_sensor_centre']} facing a sensor, "
        f"{wall['collection_facing_gap_centre']} facing a gap"
    )
    print(f"9. Q = 800: E std {results['energy_radius_degeneracy']['Q800']['energy_std']} MeV")
    print(f"10. PCA: first 3 components {pca['explained_first3']:.3f}, corr(PC1, log E) {pca['corr_PC1_logE']}")
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
