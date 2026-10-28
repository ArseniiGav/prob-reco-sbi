"""T2.1-T2.4 Detector response = what the network will see.

Run from the repository root::

    python solutions/stage1_flash/task2_response.py

T2.1  Per-sensor charges pooled over all events and sensors (log axes), number of hit sensors per event, fraction of
      empty sensors versus energy and vertex radius (with the Poisson expectation for central events).
T2.2  Total charge Q: distribution, Q versus E, light collection Q / (Y E) versus r with errors, and Q versus r at a
      fixed energy of 2 MeV.
T2.3  Mean and first hit times (with and without the unknown t0), relative times, and a demonstration that the
      relative times do not change when t0 is shifted.
T2.4  The network input ``network_input(charge, time_mean)``: both halves, their joint distribution, and what an
      empty sensor looks like.

Data: 200 000 events of the full model from the default prior (seed 22, shared with T1) and 100 000 events at a fixed
energy of 2 MeV with vertices from the prior (seed 24, shared with T2.5).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common import C, dataset, figure_legend, legend_inside, save_results, savefig, style
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from prob_reco_sbi.flash import network_input, simulate

Y = 1000.0  # light yield [photons / MeV]
M = 32  # sensors
C_N = 0.2  # speed of light in the medium [m/ns]
#: 2-D histograms: white (empty) -> royalblue -> indigo -> black (densest), the paper's blues.
DENSITY_CMAP = LinearSegmentedColormap.from_list("density", ["#f3f7fc", "#b3c6f2", C["blue"], C["violet"], C["ink"]])
R_BANDS = [(0.0, 0.2), (0.2, 0.6), (0.6, 0.85), (0.85, 0.95), (0.95, 1.0)]
E_BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 5.0)]
BAND_COLOURS = [C["green"], C["blue"], C["red"], C["violet"], C["orange"], C["aqua"]]  # the paper's order
BAND_MARKERS = ["o", "s", "^", "D", "v", "P"]  # one marker per band, so that the bands differ by more than their hue


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
        std = np.sqrt(np.maximum(squares / count - mean**2, 0) * count / np.maximum(count - 1, 1))
    return mean, std / np.sqrt(count), count


def binned_quantiles(x: np.ndarray, v: np.ndarray, edges: np.ndarray, q: tuple[float, ...]) -> np.ndarray:
    """Quantiles ``q`` (in percent) of ``v`` in bins of ``x``, shape [len(q), n_bins]."""
    idx = np.digitize(x, edges) - 1
    out = np.full((len(q), len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        sel = v[idx == b]
        if len(sel):
            out[:, b] = np.percentile(sel, q)
    return out


def centres(edges: np.ndarray) -> np.ndarray:
    """Bin centres of the bin edges ``edges``."""
    return 0.5 * (edges[1:] + edges[:-1])


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
    bar = plt.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label(label)


def r6(v: Any) -> float:
    """Round to six decimals (for the results files)."""
    return round(float(v), 6)


def main() -> None:
    """Make the T2.1-T2.4 figures and store the numbers in results/T2.json."""
    style()
    data = dataset("t1t2_full_prior_200k", 200_000, "full", seed=22)
    fixed = dataset("t2_full_E2_100k", 100_000, "full", seed=24, fixed_energy=2.0)
    x, y = data.truth.x.astype(np.float64), data.truth.y.astype(np.float64)
    energy, t0 = data.truth.energy.astype(np.float64), data.truth.t0.astype(np.float64)
    r = np.hypot(x, y)
    n = data.response.charge.astype(np.float64)
    n_events = len(r)
    results: dict[str, dict[str, Any]] = {}

    # ================================================================================================== T2.1
    empty = n == 0
    r_sensor = np.repeat(r, M)
    e_sensor = np.repeat(energy, M)
    edges = np.unique(np.round(np.logspace(0, np.log10(3000), 60))) - 0.5
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups = [
        ("all events", "all", np.ones(n_events, bool), C["ink"]),
        ("$0 \\leq r_0 <$ 0.5 m", "r 0-0.5", r < 0.5, C["green"]),
        ("$0.5 \\leq r_0 <$ 0.9 m", "r 0.5-0.9", (r >= 0.5) & (r < 0.9), C["blue"]),
        ("$0.9 \\leq r_0 \\leq$ 1 m", "r 0.9-1", r >= 0.9, C["red"]),
    ]
    charge_stats = {}
    for label, name, sel, col in groups:
        values = n[sel].ravel()
        counts, _ = np.histogram(values, bins=edges)
        density = counts / (values.size * np.diff(edges))
        ok = counts > 0
        zero = np.mean(values == 0)
        ax.step(
            centres(edges)[ok],
            density[ok],
            where="mid",
            color=col,
            lw=1.8 if name == "all" else 1.5,
            alpha=0.6,
            label=f"{label} (empty: {100 * zero:.2f} %)",
        )
        charge_stats[name] = {
            "n_sensors": int(values.size),
            "empty_fraction": r6(zero),
            "mean": round(float(values.mean()), 2),
            "median": float(np.median(values)),
            "q99": float(np.percentile(values, 99)),
            "max": int(values.max()),
        }
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1)
    ax.set_xlabel("Sensor charge $n_j$ [p.e.]")
    ax.set_ylabel("Fraction of sensors per p.e.")
    ax.set_title(f"Charges of all {M} sensors pooled over {n_events:,} events")
    legend_inside(ax, loc="lower left", title="Vertex radius")
    savefig(fig, "t2_1_sensor_charges")

    n_hit = (n > 0).sum(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6), gridspec_kw={"width_ratios": [0.85, 1, 1]})
    ax = axes[0]
    bins = np.arange(-0.5, M + 1.5)
    low_wall = (energy < 1.0) & (r > 0.9)
    for label, sel, col in [
        ("all events", np.ones(n_events, bool), C["ink"]),
        ("$E <$ 1 MeV", energy < 1.0, C["blue"]),
        ("$E <$ 1 MeV, $r_0 >$ 0.9 m", low_wall, C["red"]),
    ]:
        counts, _ = np.histogram(n_hit[sel], bins=bins)
        frac = counts / sel.sum()
        ok = frac > 0
        ax.step(np.arange(M + 1)[ok], frac[ok], where="mid", color=col, alpha=0.6, label=label)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 2)
    ax.set_xlim(14.5, M + 0.5)
    ax.set_xlabel("Number of hit sensors per event")
    ax.set_ylabel("Fraction of events")
    legend_inside(ax, loc="lower right")  # the steps rise to the right: the lower-right corner is empty

    e_edges = np.arange(0.5, 5.0001, 0.25)
    ax = axes[1]
    empty_vs_e = {}
    for (lo, hi), col, mk in zip(R_BANDS, BAND_COLOURS, BAND_MARKERS, strict=False):
        sel = (r_sensor >= lo) & (r_sensor < hi + (1e-9 if hi == 1.0 else 0))
        frac, err, _ = binned_mean(e_sensor[sel], empty.ravel()[sel].astype(float), e_edges)
        ok = frac > 0
        ax.errorbar(
            centres(e_edges)[ok],
            frac[ok],
            yerr=err[ok],
            fmt=f"{mk}-",
            ms=3.5,
            lw=1.2,
            color=col,
            capsize=0,
            label=f"{lo:g}-{hi:g} m",
        )
        empty_vs_e[f"r {lo}-{hi}"] = [r6(v) for v in frac]
    # n_j ~ Poisson(Y E f_c / M) for a vertex exactly at the centre: P(n_j = 0) = exp(-Y E f_c / M), averaged over
    # the energies of the central events in each bin (the expectation for the innermost band, dashed in its colour)
    centre_f = np.mean(n[r < 0.1].sum(axis=1) / (Y * energy[r < 0.1]))
    central = r < R_BANDS[0][1]
    p0_centre, _, _ = binned_mean(energy[central], np.exp(-Y * energy[central] * centre_f / M), e_edges)
    ok = p0_centre > 1e-6
    ax.plot(
        centres(e_edges)[ok],
        p0_centre[ok],
        color=BAND_COLOURS[0],
        ls="--",
        lw=1.25,
        label="Poisson, centre",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("Energy $E$ [MeV]")
    ax.set_ylabel("Fraction of empty sensors")
    legend_inside(ax, loc="upper right", title="Vertex radius $r_0$")

    r_edges = np.linspace(0, 1, 21)
    ax = axes[2]
    empty_vs_r = {}
    for (lo, hi), col, mk in zip(E_BANDS, BAND_COLOURS, BAND_MARKERS, strict=False):
        sel = (e_sensor >= lo) & (e_sensor < hi + (1e-9 if hi == 5.0 else 0))
        frac, err, _ = binned_mean(r_sensor[sel], empty.ravel()[sel].astype(float), r_edges)
        ok = frac > 0
        ax.errorbar(
            centres(r_edges)[ok],
            frac[ok],
            yerr=err[ok],
            fmt=f"{mk}-",
            ms=3.5,
            lw=1.2,
            color=col,
            capsize=0,
            label=f"{lo:g}-{hi:g} MeV",
        )
        empty_vs_r[f"E {lo}-{hi}"] = [r6(v) for v in frac]
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("Vertex radius $r_0$ [m]")
    ax.set_ylabel("Fraction of empty sensors")
    legend_inside(ax, loc="upper left", title="Energy $E$")
    savefig(fig, "t2_1_hit_and_empty_sensors")

    results["T2.1"] = {
        "n_events": n_events,
        "pooled_charge": charge_stats,
        "mean_hit_sensors": round(float(n_hit.mean()), 3),
        "fraction_events_all_sensors_hit": r6(np.mean(n_hit == M)),
        "fraction_events_all_hit_E_below_1_r_above_0.9": r6(np.mean(n_hit[low_wall] == M)),
        "min_hit_sensors": int(n_hit.min()),
        "empty_fraction_overall": r6(empty.mean()),
        "empty_fraction_vs_E": {"E_bin_edges": e_edges.round(3).tolist(), **empty_vs_e},
        "empty_fraction_vs_r": {"r_bin_edges": r_edges.round(3).tolist(), **empty_vs_r},
        "centre_collection_used_for_poisson_line": r6(centre_f),
        "poisson_empty_centre_by_E_bin": [r6(v) for v in p0_centre],
        "poisson_empty_centre_E0.5": r6(np.exp(-Y * 0.5 * centre_f / M)),
        "poisson_empty_centre_E1": r6(np.exp(-Y * 1.0 * centre_f / M)),
        "empty_fraction_E_below_1_r_below_0.5": r6(empty[(energy < 1) & (r < 0.5)].mean()),
        "empty_fraction_E_below_1_r_above_0.95": r6(empty[(energy < 1) & (r > 0.95)].mean()),
        "empty_fraction_E_above_3.5_r_above_0.95": r6(empty[(energy > 3.5) & (r > 0.95)].mean()),
    }

    # ================================================================================================== T2.2
    q = n.sum(axis=1)
    f = q / (Y * energy)
    f_mean, f_err, f_cnt = binned_mean(r, f, r_edges)
    f_q = binned_quantiles(r, f, r_edges, (16, 84))
    f_centre, f_wall = float(np.mean(f[r < 0.1])), float(np.mean(f[r > 0.95]))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [0.8, 1.1]})
    ax = axes[0]
    q_top = 100 * np.ceil(q.max() / 100)
    ax.hist(q, bins=np.arange(0, q_top + 1, 25), histtype="step", color=C["blue"], alpha=0.6, lw=1.5)
    ax.set_xlabel("Total charge $Q = \\sum_j n_j$ [p.e.]")
    ax.set_ylabel("Events per 25 p.e.")
    ax.set_title(f"{n_events:,} events from the prior")
    ax = axes[1]
    hist2d(ax, energy, q, [np.arange(0.5, 5.0001, 0.05), np.arange(0, q_top + 1, 20)], "Events per bin")
    e_line = np.array([0.5, 5.0])
    ax.plot(
        e_line,
        Y * f_wall * e_line,
        color=C["red"],
        lw=1.5,
        ls="--",
        label=f"$Y E f$, $f$ = {f_wall:.3f} (wall, $r_0 >$ 0.95 m)",
    )
    ax.plot(
        e_line,
        Y * f_centre * e_line,
        color=C["green"],
        lw=1.5,
        ls="--",
        label=f"$Y E f$, $f$ = {f_centre:.3f} (centre, $r_0 <$ 0.1 m)",
    )
    e_mid = np.arange(0.5, 5.0001, 0.25)
    q_mean, _, _ = binned_mean(energy, q, e_mid)
    ax.plot(centres(e_mid), q_mean, "o", color=C["ink"], ms=3.5, label="mean $Q$ per energy bin")
    ax.set_xlabel("Energy $E$ [MeV]")
    ax.set_ylabel("Total charge $Q$ [p.e.]")
    ax.set_ylim(0, 1.2 * q_top)  # the empty upper-left corner holds the legend
    legend_inside(ax, loc="upper left")
    savefig(fig, "t2_2_total_charge")

    q2 = fixed.response.charge.sum(axis=1).astype(np.float64)
    r2 = np.hypot(fixed.truth.x.astype(np.float64), fixed.truth.y.astype(np.float64))
    q2_mean, q2_err, _ = binned_mean(r2, q2, r_edges)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    ax = axes[0]
    ax.fill_between(
        centres(r_edges), f_q[0], f_q[1], color=C["blue"], alpha=0.2, lw=0, label="16-84 % of single events"
    )
    ax.errorbar(
        centres(r_edges),
        f_mean,
        yerr=f_err,
        fmt="o-",
        color=C["blue"],
        ms=4,
        capsize=2,
        lw=1.3,
        label="mean $\\pm$ standard error (within the markers)",
    )
    ax.axhline(f.mean(), color=C["blue"], ls="--", lw=1.25, label=f"average over the disc: {f.mean():.3f}")
    ax.set_xlabel("Vertex radius $r_0$ [m]")
    ax.set_ylabel("Light collection $Q / (Y E)$")
    ax.set_xlim(0, 1)
    legend_inside(ax, loc="upper left")
    ax = axes[1]
    hist2d(
        ax,
        r2,
        q2,
        [np.linspace(0, 1, 51), np.arange(100 * np.floor(q2.min() / 100), 100 * np.ceil(q2.max() / 100) + 1, 10)],
        "Events per bin",
    )
    ax.errorbar(centres(r_edges), q2_mean, yerr=q2_err, fmt="o", color=C["ink"], ms=3.5, capsize=0, label="mean $Q$")
    ax.set_xlabel("Vertex radius $r_0$ [m]")
    ax.set_ylabel("Total charge $Q$ at $E$ = 2 MeV [p.e.]")
    legend_inside(ax, loc="upper left")
    savefig(fig, "t2_2_light_collection")

    e2 = (energy > 1.95) & (energy < 2.05)
    results["T2.2"] = {
        "Q_mean": round(float(q.mean()), 1),
        "Q_quantiles_1_50_99": np.percentile(q, [1, 50, 99]).round(1).tolist(),
        "Q_min_max": [int(q.min()), int(q.max())],
        "collection_disc_average": r6(f.mean()),
        "collection_r_bin_edges": r_edges.round(3).tolist(),
        "collection_mean_by_r": [r6(v) for v in f_mean],
        "collection_sem_by_r": [r6(v) for v in f_err],
        "collection_q16_by_r": [r6(v) for v in f_q[0]],
        "collection_q84_by_r": [r6(v) for v in f_q[1]],
        "events_per_r_bin": f_cnt.astype(int).tolist(),
        "collection_centre_r_below_0.1": r6(f_centre),
        "collection_wall_r_above_0.95": r6(f_wall),
        "wall_over_centre": round(f_wall / f_centre, 4),
        "Q_at_2MeV_by_r_mean": [round(float(v), 1) for v in q2_mean],
        "Q_at_2MeV_min_max": [int(q2.min()), int(q2.max())],
        "Q_at_2MeV_centre_r_below_0.1": round(float(q2[r2 < 0.1].mean()), 1),
        "Q_at_2MeV_wall_r_above_0.95": round(float(q2[r2 > 0.95].mean()), 1),
        "Q_spread_E_1.95_to_2.05": {
            "mean": round(float(q[e2].mean()), 1),
            "std": round(float(q[e2].std()), 1),
            "poisson_std": round(float(np.sqrt(q[e2].mean())), 1),
        },
        "Q_spread_E2_fixed_r_below_0.1": {
            "mean": round(float(q2[r2 < 0.1].mean()), 1),
            "std": round(float(q2[r2 < 0.1].std()), 1),
            "poisson_std": round(float(np.sqrt(q2[r2 < 0.1].mean())), 1),
        },
    }

    # ================================================================================================== T2.3
    tm = data.response.time_mean.astype(np.float64)
    tf = data.response.time_first.astype(np.float64)
    hit = n > 0
    since_mean = (tm - t0[:, None])[hit]
    since_first = (tf - t0[:, None])[hit]
    raw_mean = tm[hit]
    features = network_input(data.response.charge, data.response.time_mean)
    rel = features[:, M:].astype(np.float64)
    rel_hit = rel[hit]
    # one colour per kind of time in the whole figure: royalblue tau_j - t0, firebrick first hit, indigo as recorded,
    # teal relative time (as in t2_4_network_input); one legend for the three panels
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    tbins = np.arange(-5, 45.01, 0.5)
    ax = axes[0]
    ax.hist(since_mean, bins=tbins, histtype="step", color=C["blue"], lw=1.5, alpha=0.6, density=True)
    ax.hist(since_first, bins=tbins, histtype="step", color=C["red"], lw=1.5, alpha=0.6, density=True)
    ax.axvspan(0, 2 / C_N, color=C["grid"], alpha=0.8, lw=0)
    ax.set_xlabel("Time since emission [ns]")
    ax = axes[1]
    ax.hist(raw_mean, bins=tbins, histtype="step", color=C["violet"], lw=1.5, alpha=0.6, density=True)
    ax.hist(since_mean, bins=tbins, histtype="step", color=C["blue"], lw=1.5, alpha=0.6, density=True)
    ax.set_xlabel("Mean hit time [ns]")
    ax = axes[2]
    ax.hist(
        rel_hit, bins=np.arange(-12, 25.01, 0.25), histtype="step", color=C["aqua"], lw=1.5, alpha=0.6, density=True
    )
    ax.set_xlabel("Relative mean time [ns]")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1)  # the same floor for the three compared panels
        ax.set_ylabel("Fraction of hit sensors per ns")
    handles = [
        plt.Line2D(
            [], [], color=C["blue"], lw=1.5, alpha=0.6, label="mean time $\\tau_j - t_0$ (needs the true $t_0$)"
        ),
        plt.Line2D([], [], color=C["red"], lw=1.5, alpha=0.6, label="first hit time $-\\ t_0$"),
        plt.Rectangle((0, 0), 1, 1, color=C["grid"], alpha=0.8, lw=0, label="straight flight $d_j / c_n$ (0-10 ns)"),
        plt.Line2D([], [], color=C["violet"], lw=1.5, alpha=0.6, label="as recorded: $\\tau_j$ (contains $t_0$)"),
        plt.Line2D(
            [], [], color=C["aqua"], lw=1.5, alpha=0.6, label="relative time $\\tau_j - \\bar\\tau$ (hit sensors)"
        ),
    ]
    figure_legend(fig, handles, ncol=3)
    savefig(fig, "t2_3_times")

    # one event, three emission times, the same seed: identical photons, times shifted by t0
    vertex, e_demo, seed_demo = (0.6, 0.3), 2.0, 7
    demo = {}
    for t_emit in (0.0, 4.0, 9.0):
        res = simulate(vertex[0], vertex[1], e_demo, t_emit, config=data.config, seed=seed_demo, device="cpu")
        demo[t_emit] = (
            res.response.charge[0],
            res.response.time_mean[0],
            network_input(res.response.charge, res.response.time_mean)[0, M:],
        )
    demo_diff = max(float(np.max(np.abs(demo[t][2] - demo[0.0][2]))) for t in demo)
    charge_same = all(np.array_equal(demo[t][0], demo[0.0][0]) for t in demo)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
    j = np.arange(M)
    for (t_emit, (_, tau, rel_demo)), col, mk in zip(
        demo.items(), [C["green"], C["blue"], C["red"]], ["o", "s", "D"], strict=True
    ):
        axes[0].plot(j, tau, marker=mk, ms=4, lw=1, color=col, label=f"$t_0$ = {t_emit:g} ns")
        axes[1].plot(
            j,
            rel_demo,
            marker=mk,
            ms=7 - 1.5 * list(demo).index(t_emit),
            lw=1,
            color=col,
            mfc="none" if t_emit else col,
        )
    axes[0].set_xlabel("Sensor $j$")
    axes[1].set_xlabel("Sensor $j$")
    axes[0].set_ylabel("Mean hit time $\\tau_j$ [ns]")
    axes[1].set_ylabel("Relative time $\\tau_j - \\bar\\tau$ [ns]")
    axes[0].set_xticks([0, 8, 16, 24, 31])
    axes[0].set_title("Recorded times: shifted by $t_0$", fontsize=10)
    mantissa, exponent = f"{demo_diff:.0e}".split("e")
    shown = f"$10^{{{int(exponent)}}}$" if mantissa == "1" else f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    axes[1].set_title(f"Relative times: identical (largest difference {shown} ns)", fontsize=10)
    figure_legend(
        fig,
        *axes[0].get_legend_handles_labels(),
        ncol=3,
        title=f"One event at ({vertex[0]:g}, {vertex[1]:g}) m, E = {e_demo:g} MeV, "
        "simulated three times with the same seed",
    )
    savefig(fig, "t2_3_relative_times")

    # the same check on 2000 events of the prior sample: shift every t0 by 7.5 ns
    k = 2000
    args = (data.truth.x[:k], data.truth.y[:k], data.truth.energy[:k])
    a = simulate(*args, data.truth.t0[:k], config=data.config, seed=31, device="cpu")
    b = simulate(*args, data.truth.t0[:k] + 7.5, config=data.config, seed=31, device="cpu")
    xa = network_input(a.response.charge, a.response.time_mean)
    xb = network_input(b.response.charge, b.response.time_mean)
    abs_shift = np.nanmax(np.abs(b.response.time_mean - a.response.time_mean - 7.5))

    q_since = np.percentile(since_mean, [1, 50, 99])
    results["T2.3"] = {
        "mean_time_since_emission_ns": {
            "mean": round(float(since_mean.mean()), 3),
            "q01_q50_q99": q_since.round(3).tolist(),
        },
        "first_time_since_emission_ns": {
            "mean": round(float(since_first.mean()), 3),
            "q01_q50_q99": np.percentile(since_first, [1, 50, 99]).round(3).tolist(),
        },
        "raw_mean_time_ns": {"mean": round(float(raw_mean.mean()), 3), "std": round(float(raw_mean.std()), 3)},
        "since_emission_mean_time_std_ns": round(float(since_mean.std()), 3),
        "relative_time_hit_sensors_ns": {
            "mean": round(float(rel_hit.mean()), 4),
            "std": round(float(rel_hit.std()), 3),
            "q01_q50_q99": np.percentile(rel_hit, [1, 50, 99]).round(3).tolist(),
        },
        "fraction_mean_time_later_than_10ns_after_t0": r6(np.mean(since_mean > 10)),
        "fraction_first_time_before_t0": r6(np.mean(since_first < 0)),
        "t0_prior_ns": [data.config.prior.t0_min, data.config.prior.t0_max],
        "demo_event": {
            "vertex": list(vertex),
            "energy": e_demo,
            "t0": list(demo),
            "seed": seed_demo,
            "max_abs_diff_relative_time_ns": demo_diff,
            "charges_identical": charge_same,
        },
        "shift_test_2000_events": {
            "shift_ns": 7.5,
            "charges_identical": bool(np.array_equal(a.response.charge, b.response.charge)),
            "max_abs_diff_network_input_charge_half": float(np.max(np.abs(xa[:, :M] - xb[:, :M]))),
            "max_abs_diff_network_input_time_half_ns": float(np.max(np.abs(xa[:, M:] - xb[:, M:]))),
            "max_abs_deviation_of_absolute_time_shift_from_7.5_ns": float(abs_shift),
        },
    }

    # ================================================================================================== T2.4
    log_n = features[:, :M].astype(np.float64)
    zero_charge = log_n == 0
    zero_time = rel == 0
    # bins aligned with the integer charges: one bin per value up to 40 p.e., logarithmic above
    n_edges = np.unique(
        np.concatenate(
            [np.arange(0.5, 40.6, 1.0), np.round(np.logspace(np.log10(40.5), np.log10(n.max() + 1), 45)) + 0.5]
        )
    )
    l_edges = np.log1p(n_edges)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    # royalblue charge half, teal time half (the colour of the relative times in t2_3_times), firebrick = empty sensors
    ax = axes[0]
    counts, _ = np.histogram(n[n > 0], bins=n_edges)
    ax.stairs(counts / n.size / np.diff(l_edges), l_edges, color=C["blue"], lw=1.5, alpha=0.6)
    ax.axvline(0, color=C["red"], lw=2.4)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-0.2, l_edges[-1])
    ax.set_xlabel("Charge half: $\\log(1 + n_j)$")
    ax.set_ylabel("Fraction of entries per unit")
    ax = axes[1]
    rbins = np.arange(-12, 25.01, 0.25)
    counts, _ = np.histogram(rel[~zero_time], bins=rbins)
    ax.stairs(counts / rel.size / 0.25, rbins, color=C["aqua"], lw=1.5, alpha=0.6)
    ax.axvline(0, color=C["red"], lw=2.4)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1)  # as the relative-time panel of t2_3_times
    ax.set_xlabel("Time half: $\\tau_j - \\bar\\tau$ [ns]")
    ax.set_ylabel("Fraction of entries per ns")
    ax = axes[2]
    hist2d(
        ax,
        log_n[~zero_charge],
        rel[~zero_charge],
        [l_edges, rbins],
        "Sensors per unit of $\\log(1+n_j)$ per ns",
        per_area=True,
    )
    ax.plot([0], [0], "D", color=C["red"], ms=6, clip_on=False)
    ax.set_xlim(-0.15, l_edges[-1])
    ax.set_xlabel("Charge half: $\\log(1 + n_j)$")
    ax.set_ylabel("Time half: $\\tau_j - \\bar\\tau$ [ns]")
    handles = [
        plt.Line2D([], [], color=C["blue"], lw=1.5, alpha=0.6, label="hit sensors, charge half"),
        plt.Line2D([], [], color=C["aqua"], lw=1.5, alpha=0.6, label="hit sensors, time half"),
        plt.Line2D(
            [],
            [],
            color=C["red"],
            lw=2.4,
            label=f"empty sensors: exactly 0 in both halves ({100 * zero_charge.mean():.2f} % of the entries)",
        ),
        plt.Line2D([], [], color=C["red"], marker="D", ls="none", ms=6, label="every empty sensor sits at (0, 0)"),
    ]
    figure_legend(fig, handles, ncol=2)
    savefig(fig, "t2_4_network_input")

    single = (n > 0).sum(axis=1) == 1
    results["T2.4"] = {
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "charge_half": {
            "mean": round(float(log_n.mean()), 4),
            "std": round(float(log_n.std()), 4),
            "min_max": [float(log_n.min()), round(float(log_n.max()), 3)],
            "fraction_exact_zero": r6(zero_charge.mean()),
        },
        "time_half": {
            "mean": round(float(rel.mean()), 4),
            "std": round(float(rel.std()), 4),
            "q01_q99_ns": np.percentile(rel, [1, 99]).round(3).tolist(),
            "fraction_exact_zero": r6(zero_time.mean()),
            "fraction_exact_zero_on_hit_sensors": r6(np.mean(rel[~zero_charge] == 0)),
        },
        "empty_sensor_value": [0.0, 0.0],
        "events_with_a_single_hit_sensor": int(single.sum()),
        "pearson_logcharge_vs_reltime_hit_sensors": round(
            float(np.corrcoef(log_n[~zero_charge], rel[~zero_charge])[0, 1]), 4
        ),
        "charge_weighted_mean_of_relative_times_max_abs_ns": float(
            np.max(np.abs((n * rel).sum(axis=1) / np.maximum(n.sum(axis=1), 1)))
        ),
    }

    path = save_results("T2", results)
    t21, t22, t23, t24 = (results[k] for k in ("T2.1", "T2.2", "T2.3", "T2.4"))
    print(
        f"T2.1: empty sensors {100 * t21['empty_fraction_overall']:.3f} %; mean hit sensors {t21['mean_hit_sensors']}; "
        f"all 32 hit in {100 * t21['fraction_events_all_sensors_hit']:.1f} % of the events"
    )
    print(
        f"T2.2: collection {t22['collection_centre_r_below_0.1']:.4f} (centre) -> "
        f"{t22['collection_wall_r_above_0.95']:.4f} (wall), disc average {t22['collection_disc_average']:.4f}; "
        f"Q at 2 MeV {t22['Q_at_2MeV_centre_r_below_0.1']} -> {t22['Q_at_2MeV_wall_r_above_0.95']} p.e."
    )
    print(
        f"T2.3: mean time since t0 {t23['mean_time_since_emission_ns']['mean']} ns; relative-time spread "
        f"{t23['relative_time_hit_sensors_ns']['std']} ns; t0-shift test max diff "
        f"{t23['shift_test_2000_events']['max_abs_diff_network_input_time_half_ns']:.2e} ns"
    )
    print(
        f"T2.4: input {t24['shape']}, charge half {t24['charge_half']['mean']} +- {t24['charge_half']['std']}, "
        f"time half {t24['time_half']['mean']} +- {t24['time_half']['std']} ns"
    )
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
