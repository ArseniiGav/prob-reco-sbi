"""T5.2 (bonus): degeneracy -- two different (E, r) with the same mean total charge Q.

Event A sits at the centre with E_A = 3 MeV. Event B sits at radius r_B on the x axis with the energy that gives the
same mean Q: E_B = E_A f(0) / f(r_B), where f(r) = <Q> / (Y E) is the light collection measured at that vertex
(step 1, calibration runs). Then 20 000 events of A and of each B are simulated (step 2) and three features are
compared: the total charge Q, the radius of the charge barycentre (charge pattern) and the charge-weighted spread of
the relative mean times (timing). Separation is measured by the AUC = P(feature of B > feature of A) (0.5: the
feature cannot tell A from B; 1: it always can).

Run from the repository root::

    python solutions/stage1_flash/task5_degeneracy.py

Figures: t5_2_degeneracy (iso-Q line, Q, charge and time patterns of A and B for r_B = 0.8 m),
t5_2_separation (feature distributions and AUC vs r_B). Numbers: solutions/stage1_flash/results/T5.json (key T5.2).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from common import C, dataset, figure_legend, legend_inside, round_label, save_results, savefig, style
from event_display import barycentre, relative_times
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from prob_reco_sbi.flash import SimResult

E_A = 3.0  # MeV, event A at the centre
R_SCAN = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9)  # radii of event B [m]
R_SHOW = 0.8  # the pair shown in detail
N_CAL = 50_000  # calibration events per vertex of B (step 1)
N_CAL_A = 200_000  # calibration events at the centre: f(0) enters every E_B
N_A = 100_000  # events of A (step 2); A is compared with every B, so its sample is larger
N_EV = 20_000  # events of each B (step 2)
# event A is royalblue (solid, circle) and event B peru (dashed, square) in both figures; the three features of the
# separation scan have their own colours and marker shapes
FEATURES = {
    "Q": ("Total charge Q [p.e.]", C["ink"]),
    "r_bary": ("Barycentre radius [m]", C["green"]),
    "s_tau": ("Relative-time spread [ns]", C["red"]),
}


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """Area under the ROC curve, P(b > a) + P(b = a) / 2 (Mann-Whitney U with average ranks for ties)."""
    x = np.concatenate([a, b])
    order = np.argsort(x, kind="mergesort")
    _, start, counts = np.unique(x[order], return_index=True, return_counts=True)
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(start + (counts + 1) / 2, counts)  # 1-based average rank of each tie group
    u = ranks[len(a) :].sum() - len(b) * (len(b) + 1) / 2
    return float(u / (len(a) * len(b)))


def features(data: SimResult) -> dict[str, np.ndarray]:
    """Total charge, barycentre radius and charge-weighted RMS of the relative mean times of every event."""
    n = data.response.charge.astype(np.float64)
    q = n.sum(1)
    bx, by = barycentre(n, data.detector)
    rel = relative_times(n, data.response.time_mean)
    s_tau = np.sqrt(np.nansum(n * np.nan_to_num(rel) ** 2, 1) / q)
    return {"Q": q, "r_bary": np.hypot(bx, by), "s_tau": s_tau}


def fixed(name: str, n: int, r: float, energy: float, seed: int) -> SimResult:
    """Events at the vertex (r, 0) with a fixed energy (full model, cached)."""
    return dataset(name, n, "full", seed=seed, fixed_vertex=(float(r), 0.0), fixed_energy=float(energy))


def main() -> None:
    """Run the calibration and the comparison, draw the figures and save the numbers."""
    style()
    y = 1000.0  # light yield [photons/MeV] of the full preset
    # step 1: light collection f(r) at each vertex, from calibration runs at E_A
    radii = (0.0, *R_SCAN)
    f = {}
    for k, r in enumerate(radii):
        n_cal = N_CAL_A if r == 0.0 else N_CAL
        cal = fixed(f"t5_2_cal_r{r:.2f}", n_cal, r, E_A, seed=520 + k)
        q = cal.response.charge.sum(1)
        f[r] = (q.mean() / (y * E_A), q.std(ddof=1) / np.sqrt(n_cal) / (y * E_A))
    e_b = {r: E_A * f[0.0][0] / f[r][0] for r in R_SCAN}

    # step 2: A and every B with the same mean Q
    ev_a = fixed("t5_2_A_centre", N_A, 0.0, E_A, seed=540)
    feat_a = features(ev_a)
    scan = {}
    feats = {}
    for k, r in enumerate(R_SCAN):
        ev_b = fixed(f"t5_2_B_r{r:.2f}", N_EV, r, e_b[r], seed=550 + k)
        fb = features(ev_b)
        feats[r] = (ev_b, fb)
        dq = fb["Q"].mean() - feat_a["Q"].mean()
        dq_err = np.sqrt(fb["Q"].var(ddof=1) / N_EV + feat_a["Q"].var(ddof=1) / N_A)
        scan[f"{r:.2f}"] = {
            "r_B_m": r,
            "E_B_MeV": float(e_b[r]),
            "f_B": float(f[r][0]),
            "f_B_err": float(f[r][1]),
            "mean_Q_B": float(fb["Q"].mean()),
            "sd_Q_B": float(fb["Q"].std(ddof=1)),
            "delta_mean_Q": float(dq),
            "delta_mean_Q_err": float(dq_err),
            "auc": {name: auc(feat_a[name], fb[name]) for name in FEATURES},
            "median_r_bary_B_m": float(np.median(fb["r_bary"])),
            "median_s_tau_B_ns": float(np.median(fb["s_tau"])),
        }
    auc_err = float(np.sqrt((N_A + N_EV + 1) / (12 * N_A * N_EV)))  # standard error of an AUC of 0.5

    # ------------------------------------------------------------------------------------------ figure 1: the pair
    ev_b, feat_b = feats[R_SHOW]
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.15, 1.15]})
    a = axes[0]
    rr = np.array(radii)
    ff = np.array([f[r][0] for r in radii])
    q_a = y * E_A * f[0.0][0]
    a.plot(rr, q_a / (y * ff), color=C["grey"], lw=1.4, marker="s", ms=3.5, mfc="white", zorder=2)
    a.plot([0.0], [E_A], ls="none", marker="o", ms=9, color=C["blue"], zorder=3)
    a.plot([R_SHOW], [e_b[R_SHOW]], ls="none", marker="s", ms=9, color=C["orange"], zorder=3)
    a.set_xlabel("Vertex radius $r_0$ [m]")
    a.set_ylabel("Energy E [MeV]")
    a.set_xlim(-0.03, 1.0)

    a = axes[1]
    lo, hi = np.percentile(np.concatenate([feat_a["Q"], feat_b["Q"]]), [0.05, 99.95])
    bins = np.arange(np.floor(lo), np.ceil(hi) + 4, 4)
    a.hist(feat_a["Q"], bins=bins, histtype="step", lw=1.5, alpha=0.6, color=C["blue"], density=True)
    a.hist(feat_b["Q"], bins=bins, histtype="step", lw=1.5, alpha=0.6, ls="--", color=C["orange"], density=True)
    a.set_xlabel("Total charge Q [p.e.]")
    a.set_ylabel("Probability density [1/p.e.]")
    round_label(a, f"AUC = {scan[f'{R_SHOW:.2f}']['auc']['Q']:.3f}", 0.06, 0.92, fontsize=9.5, ha="left", va="top")

    j = np.arange(ev_a.n_sensors)
    for a, kind in ((axes[2], "charge"), (axes[3], "time")):
        for ev, col, ls in ((ev_a, C["blue"], "-"), (ev_b, C["orange"], "--")):
            n = ev.response.charge.astype(np.float64)
            v = n if kind == "charge" else relative_times(n, ev.response.time_mean)
            lo_, hi_ = np.nanpercentile(v, [16, 84], axis=0)
            a.fill_between(j, lo_, hi_, color=col, alpha=0.18, lw=0)
            a.plot(j, np.nanmean(v, 0), color=col, ls=ls, lw=1.6)
        a.set_xlabel("Sensor number $j$")
        a.set_xticks([0, 8, 16, 24, 31])
    axes[2].set_ylabel("Sensor charge [p.e.]")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1.0, 1000.0)  # fixed floor and headroom
    axes[3].set_ylabel(r"Relative mean time $\tau_j - \bar\tau$ [ns]")
    handles = [  # one legend of proxies for the four panels, above them
        Line2D([], [], color=C["blue"], lw=1.6, marker="o", ms=7, label=f"A: centre, E = {E_A:.2f} MeV"),
        Line2D(
            [],
            [],
            color=C["orange"],
            lw=1.6,
            ls="--",
            marker="s",
            ms=7,
            label=f"B: r = {R_SHOW} m, E = {e_b[R_SHOW]:.2f} MeV",
        ),
        Line2D(
            [],
            [],
            color=C["grey"],
            lw=1.4,
            marker="s",
            ms=3.5,
            mfc="white",
            label=f"same mean Q = {q_a:.0f} p.e.: the B points",
        ),
        Patch(facecolor=C["grey"], alpha=0.3, label="mean and 16-84 % band over events"),
    ]
    figure_legend(fig, handles, where="bottom", ncol=4, fontsize=9.5)  # below: the suptitle holds the top
    fig.suptitle(
        f"A: E = {E_A:.1f} MeV at the centre;  B: E = {e_b[R_SHOW]:.2f} MeV at r = {R_SHOW} m;  "
        f"{N_A:,} and {N_EV:,} events",
        fontsize=11,
    )
    savefig(fig, "t5_2_degeneracy")

    # ------------------------------------------------------------------------------------------ figure 2: separation
    fig = plt.figure(figsize=(13.0, 4.3))
    left, right = fig.subfigures(1, 2, width_ratios=[2.0, 1.0])
    axes = [*left.subplots(1, 2), right.subplots()]
    shown = [0.1, 0.3, R_SHOW]
    styles = {0.1: ":", 0.3: "-.", R_SHOW: "--"}
    for a, name in ((axes[0], "r_bary"), (axes[1], "s_tau")):
        hi = max(np.percentile(feats[r][1][name], 99.9) for r in shown)
        bins = np.linspace(0, hi, 80)
        a.hist(
            feat_a[name],
            bins=bins,
            histtype="stepfilled",
            color=C["blue"],
            alpha=0.3,
            density=True,
            label="A: centre, E = 3.00 MeV",
        )
        for r in shown:
            a.hist(
                feats[r][1][name],
                bins=bins,
                histtype="step",
                lw=1.5,
                alpha=0.6,
                ls=styles[r],
                color=C["orange"],
                density=True,
                label=f"B: r = {r} m, E = {e_b[r]:.2f} MeV",
            )
        a.set_xlabel(FEATURES[name][0])
        a.set_ylabel("Probability density [1/m]" if name == "r_bary" else "Probability density [1/ns]")
    handles, labels_ = axes[0].get_legend_handles_labels()
    left.legend(handles, labels_, loc="outside upper center", ncol=2, fontsize=9, frameon=False)
    a = axes[2]
    markers = {"Q": "o", "r_bary": "s", "s_tau": "^"}
    labels = {
        "Q": "total charge Q",
        "r_bary": "charge pattern (barycentre radius)",
        "s_tau": "timing (relative-time spread)",
    }
    for name in FEATURES:
        a.plot(
            R_SCAN,
            [scan[f"{r:.2f}"]["auc"][name] for r in R_SCAN],
            color=FEATURES[name][1],
            marker=markers[name],
            ms=5,
            lw=1.4,
            label=labels[name],
        )
    a.axhline(0.5, color=C["grey"], lw=0.8, ls=":")
    a.set_ylim(0.4, 1.02)
    a.set_xlabel("Radius of B, $r_B$ [m] (A at the centre, same mean Q)")
    a.set_ylabel("AUC: P(feature of B > feature of A)")
    legend_inside(a, loc="center right", fontsize=9)
    savefig(fig, "t5_2_separation")

    results = {
        "E_A_MeV": E_A,
        "light_yield": y,
        "n_calibration_events": N_CAL,
        "n_events_A": N_A,
        "n_events_each_B": N_EV,
        "n_calibration_events_centre": N_CAL_A,
        "f_centre": float(f[0.0][0]),
        "f_centre_err": float(f[0.0][1]),
        "mean_Q_A": float(feat_a["Q"].mean()),
        "sd_Q_A": float(feat_a["Q"].std(ddof=1)),
        "median_r_bary_A_m": float(np.median(feat_a["r_bary"])),
        "median_s_tau_A_ns": float(np.median(feat_a["s_tau"])),
        "auc_standard_error_at_0p5": auc_err,
        "scan": scan,
        "shown_pair_r_B": R_SHOW,
    }
    path = save_results("T5", {"T5.2": results})
    print(f"T5.2: A at the centre, E_A = {E_A} MeV, <Q_A> = {feat_a['Q'].mean():.1f} p.e.; AUC error ~ {auc_err:.4f}")
    for key, v in scan.items():
        print(
            f"  r_B = {key} m  E_B = {v['E_B_MeV']:.3f} MeV  "
            f"dQ = {v['delta_mean_Q']:+.2f} +- {v['delta_mean_Q_err']:.2f}  "
            + "  ".join(f"AUC({k}) = {x:.3f}" for k, x in v["auc"].items())
        )
    print(f"numbers -> {path}")


if __name__ == "__main__":
    main()
