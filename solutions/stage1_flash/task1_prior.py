"""T1 What did we simulate? The prior of the event parameters.

Run from the repository root::

    python solutions/stage1_flash/task1_prior.py

T1.1  Distributions of x0, y0, E and t0 with the expected densities; the vertices in the plane; r and r^2. The vertex
      is uniform in the disc, so p(r) = 2 r / R^2 and r^2 is uniform. A naive sampler with r uniform is shown for
      contrast. Kolmogorov-Smirnov tests check every marginal against its expected distribution.
T1.2  Why the prior matters: the posterior is prior x likelihood. With the simulated events, p(E | Q) given only the
      total charge Q is obtained by keeping the events whose Q falls in a narrow window (rejection ABC); the result
      changes with the vertex prior, and the prior edge E >= 0.5 MeV cuts it.

Data: 200 000 events of the full model from the default prior (seed 22), shared with T2, and 100 000 events with
an energy prior extended below the lower edge, to [0.3, 0.8] MeV, for the low-energy edge (seed 23).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common import C, dataset, figure_legend, legend_inside, save_results, savefig, style
from matplotlib.patches import Circle

R = 1.0  # detector radius and r_max [m]
Y = 1000.0  # light yield [photons / MeV]


def ks_test(cdf_values: np.ndarray) -> dict[str, float]:
    """Kolmogorov-Smirnov test of F(sample) ~ U(0, 1): statistic D and asymptotic p-value (Kolmogorov series)."""
    u = np.sort(np.asarray(cdf_values, dtype=np.float64))
    n = len(u)
    upper = np.arange(1, n + 1) / n
    d = float(max(np.max(upper - u), np.max(u - (upper - 1 / n))))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1.0) ** (k - 1) * np.exp(-2 * (k * lam) ** 2)), 0.0, 1.0))
    return {"D": round(d, 5), "p_value": round(p, 3), "n": n}


def disc_x_pdf(x: np.ndarray) -> np.ndarray:
    """Marginal density of one coordinate of a point uniform in the disc of radius R: 2 sqrt(R^2 - x^2) / (pi R^2)."""
    return 2 * np.sqrt(np.clip(R**2 - x**2, 0, None)) / (np.pi * R**2)


def disc_x_cdf(x: np.ndarray) -> np.ndarray:
    """Cumulative distribution of one coordinate of a point uniform in the disc."""
    s = np.clip(x / R, -1, 1)
    return 0.5 + (s * np.sqrt(1 - s**2) + np.arcsin(s)) / np.pi


def quantiles(values: np.ndarray) -> dict[str, float]:
    """Size, mean, standard deviation and the 16, 50, 84 % quantiles of a sample."""
    q16, q50, q84 = np.percentile(values, [16, 50, 84])
    return {
        "n": len(values),
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "q16": round(float(q16), 4),
        "median": round(float(q50), 4),
        "q84": round(float(q84), 4),
    }


def draw_detector(ax: plt.Axes, sensor_x: np.ndarray, sensor_y: np.ndarray) -> None:
    """Draw the wall and the sensor positions and set equal axes."""
    ax.add_patch(Circle((0, 0), R, fc="none", ec=C["ink2"], lw=1.2))
    ax.plot(1.04 * sensor_x, 1.04 * sensor_y, "s", ms=3.2, color=C["ink"], ls="none")
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_xlabel("$x_0$ [m]")
    ax.set_ylabel("$y_0$ [m]")


def main() -> None:
    """Make the T1 figures and store the numbers in results/T1.json."""
    style()
    data = dataset("t1t2_full_prior_200k", 200_000, "full", seed=22)
    prior = data.config.prior
    x, y = data.truth.x.astype(np.float64), data.truth.y.astype(np.float64)
    energy, t0 = data.truth.energy.astype(np.float64), data.truth.t0.astype(np.float64)
    r = np.hypot(x, y)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    n = len(r)
    e_lo, e_hi, t_lo, t_hi = prior.energy_min, prior.energy_max, prior.t0_min, prior.t0_max

    # ------------------------------------------------------------------------------------------------ T1.1 numbers
    ks = {
        "x0_disc_marginal": ks_test(disc_x_cdf(x)),
        "y0_disc_marginal": ks_test(disc_x_cdf(y)),
        "r2_uniform": ks_test(r**2 / R**2),
        "r_uniform_wrong_hypothesis": ks_test(r / R),
        "phi_uniform": ks_test(phi / (2 * np.pi)),
        "energy_uniform": ks_test((energy - e_lo) / (e_hi - e_lo)),
        "t0_uniform": ks_test((t0 - t_lo) / (t_hi - t_lo)),
    }
    rng = np.random.default_rng(22)
    naive_r = R * rng.random(n)  # the classic mistake: r uniform
    t11: dict[str, Any] = {
        "n_events": n,
        "prior": data.config.to_dict()["prior"],
        "mean_r": {"simulated": round(float(r.mean()), 4), "expected": round(2 * R / 3, 4)},
        "median_r": {"simulated": round(float(np.median(r)), 4), "expected": round(R / np.sqrt(2), 4)},
        "fraction_r_below_0.5": {"simulated": round(float(np.mean(r < 0.5)), 4), "expected": 0.25},
        "fraction_r_above_0.9": {"simulated": round(float(np.mean(r > 0.9)), 4), "expected": 0.19},
        "fraction_r_above_0.95": {"simulated": round(float(np.mean(r > 0.95)), 4), "expected": 0.0975},
        "fraction_r_below_0.5_if_r_uniform": round(float(np.mean(naive_r < 0.5)), 4),
        "x0_mean_std": [round(float(x.mean()), 4), round(float(x.std()), 4)],
        "x0_std_expected": R / 2,
        "y0_mean_std": [round(float(y.mean()), 4), round(float(y.std()), 4)],
        "energy_mean_std": [round(float(energy.mean()), 4), round(float(energy.std()), 4)],
        "energy_mean_std_expected": [(e_lo + e_hi) / 2, round((e_hi - e_lo) / np.sqrt(12), 4)],
        "t0_mean_std": [round(float(t0.mean()), 4), round(float(t0.std()), 4)],
        "t0_mean_std_expected": [(t_lo + t_hi) / 2, round((t_hi - t_lo) / np.sqrt(12), 4)],
        "ks_tests": ks,
    }

    # ------------------------------------------------------------------------------------------------ T1.1 figures
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
    panels = [
        (x, np.linspace(-R, R, 41), "Vertex $x_0$ [m]", disc_x_pdf, "per m"),
        (y, np.linspace(-R, R, 41), "Vertex $y_0$ [m]", disc_x_pdf, "per m"),
        (
            energy,
            np.linspace(e_lo, e_hi, 37),
            "Energy $E$ [MeV]",
            lambda v: np.full_like(v, 1 / (e_hi - e_lo)),
            "per MeV",
        ),
        (
            t0,
            np.linspace(t_lo, t_hi, 41),
            "Event time $t_0$ [ns]",
            lambda v: np.full_like(v, 1 / (t_hi - t_lo)),
            "per ns",
        ),
    ]
    for ax, (values, bins, label, pdf, unit) in zip(axes, panels, strict=True):
        ax.hist(values, bins=bins, density=True, histtype="step", color=C["blue"], alpha=0.6, lw=1.5)
        grid = np.linspace(bins[0], bins[-1], 400)
        ax.plot(grid, pdf(grid), color=C["blue"], lw=1.25, ls="--")
        ax.set_xlabel(label)
        ax.set_ylabel(f"Density [{unit}]")
        ax.set_ylim(0, None)
    handles = [
        plt.Line2D([], [], color=C["blue"], alpha=0.6, lw=1.5, label=f"simulated ({n:,} events)"),
        plt.Line2D([], [], color=C["blue"], ls="--", lw=1.25, label="expected density"),
    ]
    figure_legend(fig, handles, ncol=2)
    savefig(fig, "t1_1_prior_marginals")

    det = data.detector
    k = 5000
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.3))
    naive_phi = 2 * np.pi * rng.random(k)
    for ax, (px, py), title in [
        (axes[0], (x[:k], y[:k]), "As simulated: uniform in the disc\n$r_0 = R\\sqrt{u}$, $u \\sim U(0, 1)$"),
        (
            axes[1],
            (naive_r[:k] * np.cos(naive_phi), naive_r[:k] * np.sin(naive_phi)),
            "Wrong: $r_0$ uniform\n$r_0 = R\\,u$",
        ),
    ]:
        ax.plot(px, py, ".", ms=1.6, color=C["blue"], alpha=0.6, ls="none", rasterized=True)
        draw_detector(ax, det.sensor_x, det.sensor_y)
        ax.add_patch(Circle((0, 0), R / np.sqrt(2), fc="none", ec=C["red"], lw=1.3, ls="--"))
        inside = np.mean(np.hypot(px, py) < R / np.sqrt(2))
        ax.set_title(title + f"\n{100 * inside:.0f} % of the points inside the dashed circle", fontsize=10)
    figure_legend(
        fig,
        [
            plt.Line2D([], [], color=C["blue"], marker=".", ls="none", ms=6, label=f"{k:,} vertices"),
            plt.Line2D([], [], color=C["red"], ls="--", label="$r = R/\\sqrt{2}$ = 0.71 m (encloses half the area)"),
            plt.Line2D([], [], color=C["ink"], marker="s", ls="none", ms=4, label="sensors"),
        ],
        ncol=3,
    )
    savefig(fig, "t1_1_vertices")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    grid = np.linspace(0, R, 200)
    ax = axes[0]
    ax.hist(r, bins=np.linspace(0, R, 41), density=True, histtype="step", color=C["blue"], alpha=0.6, lw=1.5)
    ax.plot(grid, 2 * grid / R**2, color=C["blue"], ls="--", lw=1.25)
    ax.set_xlabel("Vertex radius $r_0$ [m]")
    ax.set_ylabel("Density [per m]")
    ax.set_title("$p(r_0) = 2 r_0 / R^2$")
    ax = axes[1]
    ax.hist(r**2, bins=np.linspace(0, R**2, 41), density=True, histtype="step", color=C["blue"], alpha=0.6, lw=1.5)
    ax.plot([0, R**2], [1 / R**2, 1 / R**2], color=C["blue"], ls="--", lw=1.25)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Vertex radius squared $r_0^2$ [m$^2$]")
    ax.set_ylabel("Density [per m$^2$]")
    ax.set_title("$p(r_0^2) = 1 / R^2$ (uniform)")
    ax = axes[2]
    r_sorted = np.sort(r)
    ax.plot(r_sorted, np.arange(1, n + 1) / n, color=C["blue"], lw=2.2, alpha=0.6)
    ax.plot(grid, grid**2 / R**2, color=C["blue"], ls="--", lw=1.25)
    ax.plot(grid, grid / R, color=C["grey"], ls=":", lw=1.3)
    for rv in (0.5, R / np.sqrt(2), 0.9):
        ax.plot([rv], [rv**2], "o", color=C["blue"], ms=4)
        ax.annotate(
            f"{rv:.2f} m: {100 * rv**2:.0f} %", (rv, rv**2), xytext=(7, -11), textcoords="offset points", fontsize=8.5
        )
    ax.set_xlabel("Vertex radius $r_0$ [m]")
    ax.set_ylabel("Fraction of events with $r < r_0$")
    ax.set_title("Cumulative: $F(r_0) = r_0^2 / R^2$")
    figure_legend(
        fig,
        [
            plt.Line2D([], [], color=C["blue"], alpha=0.6, lw=1.5, label=f"simulated ({n:,} events)"),
            plt.Line2D([], [], color=C["blue"], ls="--", lw=1.25, label="expected for a vertex uniform in the disc"),
            plt.Line2D([], [], color=C["grey"], ls=":", lw=1.3, label="cumulative if $r_0$ were uniform (wrong)"),
        ],
        ncol=3,
    )
    savefig(fig, "t1_1_radius")

    # ------------------------------------------------------------------------------------------------ T1.2
    charge = data.response.charge.sum(axis=1).astype(np.float64)
    q_obs = 600.0
    window = (charge > 0.98 * q_obs) & (charge < 1.02 * q_obs)
    full_disc = energy[window]
    fiducial = energy[window & (r <= 0.5)]
    low = dataset("t1_2_full_lowE_100k", 100_000, "full", seed=23, energy_min=0.3, energy_max=0.8)
    q_low = low.response.charge.sum(axis=1).astype(np.float64)
    e_low = low.truth.energy.astype(np.float64)
    q_edge = 160.0
    window_low = (q_low > 0.98 * q_edge) & (q_low < 1.02 * q_edge)
    wide_prior = e_low[window_low]
    default_edge = e_low[window_low & (e_low >= e_lo)]

    # royalblue = the default prior, firebrick = the modified prior, in both panels
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    bins = np.linspace(1.2, 2.8, 41)
    ax.hist(full_disc, bins=bins, density=True, histtype="step", lw=1.5, alpha=0.6, color=C["blue"])
    ax.hist(fiducial, bins=bins, density=True, histtype="step", lw=1.5, alpha=0.6, color=C["red"])
    ax.set_xlabel("True energy $E$ of the selected events [MeV]")
    ax.set_ylabel("Posterior density [per MeV]")
    ax.set_title(f"Observed total charge $Q$ = {q_obs:.0f} p.e. ($\\pm$2 %)")
    ax.set_ylim(0, 1.4 * ax.get_ylim()[1])  # room for the legend above the histograms
    legend_inside(
        ax,
        loc="upper left",
        title="Vertex prior",
        handles=[
            plt.Line2D(
                [],
                [],
                color=C["blue"],
                lw=1.5,
                alpha=0.6,
                label=f"disc $r \\leq$ 1 m, default ({len(full_disc)} events)",
            ),
            plt.Line2D(
                [], [], color=C["red"], lw=1.5, alpha=0.6, label=f"disc $r \\leq$ 0.5 m ({len(fiducial)} events)"
            ),
        ],
    )
    ax = axes[1]
    bins = np.linspace(0.3, 0.75, 46)
    ax.hist(wide_prior, bins=bins, density=True, histtype="step", lw=1.5, alpha=0.6, color=C["red"])
    ax.hist(default_edge, bins=bins, density=True, histtype="step", lw=1.5, alpha=0.6, color=C["blue"])
    ax.set_ylim(0, 1.4 * ax.get_ylim()[1])
    ax.axvline(e_lo, color=C["grey"], ls=":", lw=1.3)
    ax.text(
        e_lo - 0.006,
        ax.get_ylim()[1] * 0.66,
        "edge of the\ndefault prior\n0.5 MeV",
        fontsize=8.5,
        color=C["ink2"],
        va="top",
        ha="right",
    )
    ax.set_xlabel("True energy $E$ of the selected events [MeV]")
    ax.set_ylabel("Posterior density [per MeV]")
    ax.set_title(f"Observed total charge $Q$ = {q_edge:.0f} p.e. ($\\pm$2 %)")
    legend_inside(
        ax,
        loc="upper right",
        title="Energy prior",
        handles=[
            plt.Line2D([], [], color=C["red"], lw=1.5, alpha=0.6, label=f"[0.3, 0.8] MeV ({len(wide_prior)} events)"),
            plt.Line2D(
                [],
                [],
                color=C["blue"],
                lw=1.5,
                alpha=0.6,
                label=f"cut at $E \\geq$ 0.5 MeV, default ({len(default_edge)} events)",
            ),
        ],
    )
    savefig(fig, "t1_2_prior_matters")

    t12: dict[str, Any] = {
        "prior_density": "p(x0, y0, E) = 1 / (pi R^2) * 1 / (E_max - E_min) inside the disc and [E_min, E_max], else 0",
        "prior_density_value_per_m2_MeV": round(1 / (np.pi * R**2) / (e_hi - e_lo), 5),
        "t0_prior": [t_lo, t_hi],
        "fraction_of_prior_within_5cm_of_wall": round(1 - 0.95**2, 4),
        "fraction_of_prior_within_10cm_of_wall": round(1 - 0.9**2, 4),
        "fraction_of_prior_within_0.1MeV_of_an_energy_edge": round(0.2 / (e_hi - e_lo), 4),
        "abc_Q600": {
            "window_pe": [0.98 * q_obs, 1.02 * q_obs],
            "vertex_prior_disc_r_le_1": quantiles(full_disc),
            "vertex_prior_disc_r_le_0.5": quantiles(fiducial),
        },
        "abc_Q160_low_energy_edge": {
            "window_pe": [0.98 * q_edge, 1.02 * q_edge],
            "data": "100000 events, energy prior [0.3, 0.8] MeV, seed 23",
            "energy_prior_0.3_to_0.8": quantiles(wide_prior),
            "energy_prior_from_0.5": quantiles(default_edge),
            "fraction_of_wide_posterior_below_0.5MeV": round(float(np.mean(wide_prior < e_lo)), 4),
        },
    }
    path = save_results("T1", {"T1.1": t11, "T1.2": t12})

    print(f"T1.1: {n} events; mean r = {r.mean():.4f} m (2/3), median r = {np.median(r):.4f} m (0.7071)")
    print(f"      fraction r > 0.9 m: {np.mean(r > 0.9):.4f} (0.19); r < 0.5 m: {np.mean(r < 0.5):.4f} (0.25)")
    for key, value in ks.items():
        print(f"      KS {key:<28s} D = {value['D']:.5f}, p = {value['p_value']:.3f}")
    q = t12["abc_Q600"]
    print(
        f"T1.2: Q = 600 p.e.: E = {q['vertex_prior_disc_r_le_1']['median']:.3f} "
        f"[{q['vertex_prior_disc_r_le_1']['q16']:.3f}, {q['vertex_prior_disc_r_le_1']['q84']:.3f}] MeV (disc prior), "
        f"{q['vertex_prior_disc_r_le_0.5']['median']:.3f} [{q['vertex_prior_disc_r_le_0.5']['q16']:.3f}, "
        f"{q['vertex_prior_disc_r_le_0.5']['q84']:.3f}] MeV (r <= 0.5 m)"
    )
    edge = t12["abc_Q160_low_energy_edge"]
    print(
        f"      Q = 160 p.e.: {100 * edge['fraction_of_wide_posterior_below_0.5MeV']:.1f} % of the wide-prior "
        "posterior below 0.5 MeV"
    )
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
