"""T5.1 (bonus): the simulator is stochastic -- the same theta simulated 200 000 times.

theta = (x0, y0, E) = (0.3 m, 0.4 m, 2 MeV) as in TASKS.md (r = 0.5 m), full model, t0 ~ U(0, 10) ns as usual.
Two photon statistics: 'poisson' (N ~ Poisson(Y E), the default) and 'fixed' (N = Y E = 2000 for every event).

Run from the repository root::

    python solutions/stage1_flash/task5_fluctuations.py

Figures (solutions/stage1_flash/figures):
    t5_1_charge_distributions   three sensor charges vs Poisson; Q for both photon statistics; variance / mean
    t5_1_correlations           sensor-sensor correlation matrices (poisson, fixed) and measured vs expected
    t5_1_time_vs_charge         spread of the mean and first hit time of a sensor vs its charge (1 / sqrt(n))
Numbers: solutions/stage1_flash/results/T5.json (key T5.1).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from common import C, dataset, figure_legend, legend_inside, save_results, savefig, style
from event_display import sensor_distance
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

THETA = {"x": 0.3, "y": 0.4, "energy": 2.0}  # the theta of the TASKS.md example
N_REP = 200_000
CORR_CMAP = LinearSegmentedColormap.from_list("corr", [C["blue"], "white", C["red"]])  # diverging, paper colours


def poisson_pmf(k: np.ndarray, mu: float) -> np.ndarray:
    """Poisson probabilities P(k; mu), computed with log-gamma (no SciPy needed)."""
    k = np.asarray(k, dtype=np.float64)
    lg = np.array([math.lgamma(v + 1) for v in k.ravel()]).reshape(k.shape)
    return np.exp(k * np.log(mu) - mu - lg)


def var_over_mean(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Variance / mean along axis 0, with its standard error from the 4th central moment."""
    n = np.asarray(n, dtype=np.float64)
    mean = n.mean(0)
    dev = n - mean
    var = (dev**2).sum(0) / (len(n) - 1)
    m4 = (dev**4).mean(0)
    se_var = np.sqrt(np.maximum(m4 - var**2 * (len(n) - 3) / (len(n) - 1), 0.0) / len(n))
    return var / mean, se_var / mean


def sd_relative_error(values: np.ndarray) -> float:
    """Relative standard error of a sample standard deviation, including the excess kurtosis of the sample.

    Var(s^2) = sigma^4 (2 / (N - 1) + kappa / N), so se(s) / s = sqrt(2 / (N - 1) + kappa / N) / 2: the times have
    a long tail of scattered and reflected light, so the Gaussian value 1 / sqrt(2 (N - 1)) would be too small.
    """
    v = np.asarray(values, dtype=np.float64)
    dev = v - v.mean()
    kappa = (dev**4).mean() / (dev**2).mean() ** 2 - 3
    return float(0.5 * np.sqrt(max(2 / (len(v) - 1) + kappa / len(v), 0.0)))


def main() -> None:
    """Simulate the fixed theta, draw the three figures and save the numbers."""
    style()
    fixed_theta = {"fixed_vertex": (THETA["x"], THETA["y"]), "fixed_energy": THETA["energy"]}
    pois = dataset("t5_1_poisson", N_REP, "full", seed=51, **fixed_theta)
    fixd = dataset("t5_1_fixed", N_REP, "full", seed=52, photon_statistics="fixed", **fixed_theta)
    hits = dataset("t5_1_hits", 5000, "full", seed=53, save_hits=True, **fixed_theta)
    det = pois.detector
    m = det.n_sensors
    n_p = pois.response.charge.astype(np.int64)
    n_f = fixd.response.charge.astype(np.int64)
    q_p, q_f = n_p.sum(1), n_f.sum(1)
    n_gamma = round(1000 * THETA["energy"])  # photons per event with fixed statistics
    assert np.all(fixd.photons.n_emitted == n_gamma)
    d = sensor_distance(THETA["x"], THETA["y"], det)
    near, far = int(np.argmin(d)), int(np.argmax(d))
    # nearest, side, farthest sensor: darkgreen, firebrick, indigo (royalblue and peru mean Poisson N and fixed N)
    sensors = {near: C["green"], (near + m // 4) % m: C["red"], far: C["violet"]}

    # ---------------------------------------------------------------- distributions and variance / mean
    vm_p, se_p = var_over_mean(n_p)
    vm_f, se_f = var_over_mean(n_f)
    p_j = n_f.mean(0) / n_gamma  # probability that one photon is detected by sensor j
    vmq_p, seq_p = var_over_mean(q_p[:, None])
    vmq_f, seq_f = var_over_mean(q_f[:, None])
    f_det = q_f.mean() / n_gamma

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    a = axes[0]
    for j, col in sensors.items():
        mu = n_p[:, j].mean()
        k = np.arange(max(0, int(mu - 5 * mu**0.5)), int(mu + 5 * mu**0.5) + 2)
        freq = np.bincount(n_p[:, j], minlength=k.max() + 1)[k] / N_REP
        a.step(k, freq, where="mid", color=col, lw=1.5, alpha=0.6, label=f"sensor {j}: mean {mu:.1f} p.e.")
        a.plot(k, poisson_pmf(k, mu), ls="--", lw=1.25, color=col, zorder=3)  # the model, dashed, same colour
    a.plot([], [], ls="--", lw=1.25, color=C["ink"], label="Poisson, same mean")
    a.set_xlabel("Sensor charge $n_j$ [p.e.]")
    a.set_ylabel("Fraction of events")
    legend_inside(a, loc="upper right")

    a = axes[1]
    bins = np.arange(min(q_p.min(), q_f.min()) - 0.5, max(q_p.max(), q_f.max()) + 1.5, 4)
    for q, col, lab in [(q_p, C["blue"], "N ~ Poisson(Y E)"), (q_f, C["orange"], "N = Y E fixed")]:
        a.hist(q, bins=bins, density=True, histtype="step", color=col, lw=1.5, alpha=0.6, label=lab)
    x = np.linspace(bins[0], bins[-1], 400)
    mu_q = q_p.mean()
    a.plot(  # the model of the Poisson sample, dashed in its colour
        x,
        np.exp(-0.5 * (x - mu_q) ** 2 / mu_q) / np.sqrt(2 * np.pi * mu_q),
        color=C["blue"],
        ls="--",
        lw=1.25,
        label="Gaussian, variance = mean",
    )
    a.set_xlabel("Total charge Q [p.e.]")
    a.set_ylabel("Probability density [1/p.e.]")
    a.set_ylim(0.0, 1.5 * a.get_ylim()[1])  # headroom for the legend above the peaks
    legend_inside(a, loc="upper center")

    a = axes[2]
    j = np.arange(m)
    a.errorbar(j, vm_p, yerr=se_p, color=C["blue"], marker="o", ms=3.5, lw=0, elinewidth=1, label="Poisson N")
    a.errorbar(j, vm_f, yerr=se_f, color=C["orange"], marker="s", ms=3.5, lw=0, elinewidth=1, label="fixed N")
    a.axhline(1.0, color=C["blue"], lw=1.25, ls="--", label="Poisson: 1")
    a.plot(j, 1 - p_j, color=C["orange"], lw=1.25, ls="--", label="multinomial: $1 - p_j$")
    a.set_xticks([0, 8, 16, 24, 31])
    a.set_xlabel("Sensor number $j$")
    a.set_ylabel("Variance / mean of $n_j$")
    lo_y, hi_y = a.get_ylim()
    a.set_ylim(lo_y, hi_y + 0.45 * (hi_y - lo_y))  # headroom for the legend: the points span the whole width
    legend_inside(a, loc="upper center", ncol=2)
    fig.suptitle(
        f"The same theta {N_REP:,} times: E = {THETA['energy']:.0f} MeV at (x0, y0) = ({THETA['x']}, {THETA['y']}) m",
        fontsize=11,
    )
    savefig(fig, "t5_1_charge_distributions")

    # ---------------------------------------------------------------- correlations
    corr_p = np.corrcoef(n_p.T)
    corr_f = np.corrcoef(n_f.T)
    expect_f = -np.sqrt(np.outer(p_j, p_j) / np.outer(1 - p_j, 1 - p_j))
    off = ~np.eye(m, dtype=bool)
    lim = 0.04
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    norm = Normalize(-lim, lim)
    for a, cm, lab in [(axes[0], corr_p, "N ~ Poisson(Y E)"), (axes[1], corr_f, "N = Y E fixed")]:
        shown = np.where(off, cm, np.nan)
        im = a.imshow(shown, cmap=CORR_CMAP, norm=norm, origin="lower", interpolation="nearest")
        a.set_title(f"Correlation of $n_i$ and $n_j$, {lab}", fontsize=10)
        a.set_xlabel("Sensor number $j$")
        a.set_ylabel("Sensor number $i$")
        a.set_xticks([0, 8, 16, 24, 31])
        a.set_yticks([0, 8, 16, 24, 31])
    bar = fig.colorbar(im, ax=axes[:2], shrink=0.85, pad=0.02, extend="both")
    bar.set_label("Correlation coefficient (diagonal not shown)")
    a = axes[2]
    a.plot(expect_f[off], corr_f[off], ls="none", marker="s", ms=2.2, color=C["orange"], alpha=0.5, label="fixed N")
    a.plot(
        np.zeros(off.sum()),
        corr_p[off],
        ls="none",
        marker="o",
        ms=2.2,
        color=C["blue"],
        alpha=0.5,
        label="Poisson N (expected 0)",
    )
    lo = min(expect_f[off].min(), corr_f[off].min()) - 0.005
    a.plot([lo, 0.01], [lo, 0.01], color=C["ink"], lw=1.25, ls="--", label="measured = expected")
    a.set_xlabel("Expected correlation, multinomial")
    a.set_ylabel("Measured correlation")
    a.set_title(f"All {off.sum() // 2} sensor pairs", fontsize=10)
    legend_inside(a, loc="upper left")
    savefig(fig, "t5_1_correlations")

    # ---------------------------------------------------------------- times vs charge
    h = hits.hits
    assert h is not None
    sig = ~h.is_noise
    t_rel = h.time[sig].astype(np.float64) - hits.truth.t0[h.event_id[sig] - hits.truth.event_id[0]]
    sensor = h.sensor[sig]
    sigma_1 = np.array([t_rel[sensor == k].std(ddof=1) for k in range(m)])  # single-photon spread per sensor
    mean_1 = np.array([t_rel[sensor == k].mean() for k in range(m)])
    tau = pois.response.time_mean.astype(np.float64) - pois.truth.t0[:, None]
    first = pois.response.time_first.astype(np.float64) - pois.truth.t0[:, None]
    fig, (a_m, a_f) = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True, sharey=True)
    out_t: dict[str, object] = {}
    n_line = np.geomspace(0.8, 90, 50)
    for jj, col in sensors.items():
        n = n_p[:, jj]
        ks, sd_mean, sd_first, err_mean, err_first = [], [], [], [], []
        for k in range(1, n.max() + 1):
            sel = n == k
            if sel.sum() < 300:
                continue
            ks.append(k)
            for values, sd_list, err_list in ((tau[sel, jj], sd_mean, err_mean), (first[sel, jj], sd_first, err_first)):
                sd_list.append(values.std(ddof=1))
                err_list.append(sd_relative_error(values))
        ks_a = np.array(ks)
        sd_m, sd_f = np.array(sd_mean), np.array(sd_first)
        err, err_f = np.array(err_mean), np.array(err_first)
        a_m.errorbar(ks_a, sd_m, yerr=sd_m * err, ls="none", marker="o", ms=3.6, mfc="white", color=col, elinewidth=0.8)
        a_f.errorbar(
            ks_a, sd_f, yerr=sd_f * err_f, ls="none", marker="v", ms=3.8, mfc="white", color=col, elinewidth=0.8
        )
        for a, ls, alpha in ((a_m, "--", 1.0), (a_f, ":", 0.7)):  # the model, dashed in the colour of its sample
            a.plot(n_line, sigma_1[jj] / np.sqrt(n_line), color=col, lw=1.25, ls=ls, alpha=alpha, zorder=1)
        ratio = sd_m * np.sqrt(ks_a) / sigma_1[jj]
        # power law sd = c n^b, weighted least squares in log-log (relative errors = errors of the logarithm)
        design = np.column_stack([np.ones(len(ks_a)), np.log(ks_a)])
        w = 1 / err**2
        cov_fit = np.linalg.inv(design.T @ (design * w[:, None]))
        coef = cov_fit @ design.T @ (w * np.log(sd_m))
        chi2_fit = float((w * (np.log(sd_m) - design @ coef) ** 2).sum())
        pick = [0, len(ks_a) // 2, len(ks_a) - 1]
        out_t[str(jj)] = {
            "distance_m": float(d[jj]),
            "mean_charge_pe": float(n.mean()),
            "single_photon_time_spread_ns": float(sigma_1[jj]),
            "single_photon_mean_delay_ns": float(mean_1[jj] - d[jj] / pois.config.optics.speed_of_light),
            "charge_range_used": [int(ks_a.min()), int(ks_a.max())],
            "min_events_per_charge_value": 300,
            "sd_mean_time_times_sqrt_n_over_sigma1": {
                "min": float(ratio.min()),
                "max": float(ratio.max()),
                "mean": float(ratio.mean()),
                "chi2_vs_1": float((((ratio - 1) / err) ** 2).sum()),
                "n_points": len(ratio),
            },
            "power_law_fit_sd_mean_time": {
                "exponent": float(coef[1]),
                "exponent_err": float(np.sqrt(cov_fit[1, 1])),
                "prefactor_ns": float(np.exp(coef[0])),
                "chi2": chi2_fit,
                "ndof": len(ks_a) - 2,
            },
            "sd_first_time_ns_at": {str(int(ks_a[i])): float(sd_f[i]) for i in pick},
            "sd_mean_time_ns_at": {str(int(ks_a[i])): float(sd_m[i]) for i in pick},
        }
    for a in (a_m, a_f):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("Charge of the sensor $n_j$ [p.e.]")
        a.xaxis.set_minor_formatter(NullFormatter())
        a.yaxis.set_minor_formatter(NullFormatter())
        a.set_xlim(0.8, 90)
        a.set_ylim(0.1, 10)
    a_m.set_ylabel("Spread over events [ns]")
    a_m.set_title(r"Mean hit time $\tau_j - t_0$", fontsize=10.5)
    a_f.set_title(
        r"First hit time $- \ t_0$ (dotted: the $\sigma_1 / \sqrt{n}$ lines of the left panel)", fontsize=10.5
    )
    handles = [
        Line2D([], [], color=col, lw=1.6, label=f"sensor {jj} ({d[jj]:.1f} m from the vertex)")
        for jj, col in sensors.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            color=C["ink"],
            lw=1.25,
            ls="--",
            label=r"$\sigma_1 / \sqrt{n}$, $\sigma_1$: spread of single photons",
        ),
        Line2D([], [], ls="none", marker="o", mfc="white", color=C["ink"], label="mean time"),
        Line2D([], [], ls="none", marker="v", mfc="white", color=C["ink"], label="first time"),
    ]
    figure_legend(fig, handles, ncol=3, fontsize=9)
    savefig(fig, "t5_1_time_vs_charge")

    corr_off_p, corr_off_f, exp_off = corr_p[off], corr_f[off], expect_f[off]
    results = {
        "theta": THETA,
        "n_events_each": N_REP,
        "n_photons_fixed": n_gamma,
        "mean_n_emitted_poisson": float(pois.photons.n_emitted.mean()),
        "var_over_mean_n_emitted_poisson": float(pois.photons.n_emitted.var(ddof=1) / pois.photons.n_emitted.mean()),
        "sensors": {
            str(jj): {
                "distance_m": float(d[jj]),
                "mean_poisson_pe": float(n_p[:, jj].mean()),
                "var_over_mean_poisson": float(vm_p[jj]),
                "var_over_mean_poisson_se": float(se_p[jj]),
                "var_over_mean_fixed": float(vm_f[jj]),
                "var_over_mean_fixed_se": float(se_f[jj]),
                "expected_fixed_1_minus_p": float(1 - p_j[jj]),
                "p_detect_by_this_sensor": float(p_j[jj]),
                "fraction_zero_poisson": float((n_p[:, jj] == 0).mean()),
                "poisson_prob_zero_same_mean": float(np.exp(-n_p[:, jj].mean())),
            }
            for jj in sensors
        },
        "var_over_mean_all_sensors_poisson": {
            "min": float(vm_p.min()),
            "max": float(vm_p.max()),
            "mean": float(vm_p.mean()),
        },
        "var_over_mean_all_sensors_fixed": {
            "min": float(vm_f.min()),
            "max": float(vm_f.max()),
            "mean": float(vm_f.mean()),
        },
        "pull_fixed_vs_multinomial": {"rms": float(np.sqrt(np.mean(((vm_f - (1 - p_j)) / se_f) ** 2)))},
        "pull_poisson_vs_1": {"rms": float(np.sqrt(np.mean(((vm_p - 1) / se_p) ** 2)))},
        "Q": {
            "mean_poisson": float(q_p.mean()),
            "sd_poisson": float(q_p.std(ddof=1)),
            "var_over_mean_poisson": float(vmq_p[0]),
            "var_over_mean_poisson_se": float(seq_p[0]),
            "mean_fixed": float(q_f.mean()),
            "sd_fixed": float(q_f.std(ddof=1)),
            "var_over_mean_fixed": float(vmq_f[0]),
            "var_over_mean_fixed_se": float(seq_f[0]),
            "collected_fraction_f": float(f_det),
            "expected_fixed_1_minus_f": float(1 - f_det),
            "relative_resolution_poisson": float(q_p.std(ddof=1) / q_p.mean()),
            "relative_resolution_fixed": float(q_f.std(ddof=1) / q_f.mean()),
        },
        "correlations": {
            "poisson_offdiag_mean": float(corr_off_p.mean()),
            "poisson_offdiag_sd": float(corr_off_p.std()),
            "poisson_offdiag_max_abs": float(np.abs(corr_off_p).max()),
            "expected_statistical_sd": float(1 / np.sqrt(N_REP)),
            "fixed_offdiag_mean": float(corr_off_f.mean()),
            "fixed_offdiag_min": float(corr_off_f.min()),
            "fixed_expected_mean": float(exp_off.mean()),
            "fixed_expected_min": float(exp_off.min()),
            "fixed_rms_measured_minus_expected": float(np.sqrt(np.mean((corr_off_f - exp_off) ** 2))),
            "fixed_most_negative_expected_pairs": [
                {"i": int(a), "j": int(b), "expected": float(expect_f[a, b]), "measured": float(corr_f[a, b])}
                for a, b in zip(
                    *np.unravel_index(np.argsort(np.where(off, expect_f, 0), axis=None)[:6:2], (m, m)), strict=True
                )
            ],
            "fixed_most_negative_measured_pair": {
                "i": int(np.unravel_index(np.argmin(np.where(off, corr_f, 1)), (m, m))[0]),
                "j": int(np.unravel_index(np.argmin(np.where(off, corr_f, 1)), (m, m))[1]),
                "measured": float(np.where(off, corr_f, 1).min()),
                "expected": float(expect_f[np.unravel_index(np.argmin(np.where(off, corr_f, 1)), (m, m))]),
            },
        },
        "times": out_t,
    }
    path = save_results("T5", {"T5.1": results})
    print(f"T5.1: Q var/mean poisson {vmq_p[0]:.4f} +- {seq_p[0]:.4f}, fixed {vmq_f[0]:.4f} (expected {1 - f_det:.4f})")
    print(
        f"      corr off-diagonal: poisson {corr_off_p.mean():+.5f} (sd {corr_off_p.std():.4f}), "
        f"fixed {corr_off_f.mean():+.5f} (expected {exp_off.mean():+.5f})"
    )
    for jj, v in out_t.items():
        print(
            f"      sensor {jj}: sigma_1 = {v['single_photon_time_spread_ns']:.2f} ns, "
            f"sd(tau) sqrt(n)/sigma_1 in [{v['sd_mean_time_times_sqrt_n_over_sigma1']['min']:.3f}, "
            f"{v['sd_mean_time_times_sqrt_n_over_sigma1']['max']:.3f}]"
        )
    print(f"numbers -> {path}")


if __name__ == "__main__":
    main()
