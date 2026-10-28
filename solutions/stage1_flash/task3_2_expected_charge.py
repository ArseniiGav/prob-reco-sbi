"""T3.2 - expected charge of every sensor for straight light (subtended angle) and with absorption, vs simulation.

For a vertex p the photons that hit sensor j are those emitted into the angle Delta_j under which the sensor arc is
seen from p, so ``E[n_j] = Y E Delta_j / (2 pi)``; with absorption every direction alpha is weighted with the survival
probability ``exp(-s(alpha) / lambda_a)`` along its path s(alpha) to the wall. The predictions are compared with
50 000 repeated events at each of five fixed vertices (E = 2 MeV) for the presets "straight" and "absorb".

Run from the repository root::

    python solutions/stage1_flash/task3_2_expected_charge.py

Figures: t3_2_subtended_angle, t3_2_expected_charge, t3_2_pull_distribution. Numbers: results/T3.json, key "T3.2".
"""

from __future__ import annotations

import logging
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import task3_common as t3
from common import C, dataset, figure_legend, legend_inside, save_results, savefig, style
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Wedge

from prob_reco_sbi.flash import sensor_arcs

TESTED = ["straight", "absorb"]  # configurations with an analytic prediction
SENSOR_COLOR = "#3b3a37"
TASKS_VERTEX, TASKS_ENERGY, TASKS_N, TASKS_SEED = (0.5, 0.2), 2.0, 1000, 8  # the T3.2 commands of TASKS.md


def draw_detector(ax: Any, geometry: Any, lw: float = 3.0) -> None:
    """Wall (thin) and sensor arcs (thick)."""
    phi = np.linspace(0, 2 * np.pi, 721)
    ax.plot(geometry.radius * np.cos(phi), geometry.radius * np.sin(phi), color=C["ink2"], lw=0.8, zorder=1)
    for arc in sensor_arcs(geometry, n_points=24):
        ax.plot(arc[:, 0], arc[:, 1], color=SENSOR_COLOR, lw=lw, solid_capstyle="butt", zorder=2)


def sketch(results: dict[str, Any]) -> None:
    """Draw (a) the subtended angle (schematic), (b) the detector with the test vertices and (c) a zoom."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.0, 4.3), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    # (a) schematic with an enlarged sensor
    phi = np.linspace(0, 2 * np.pi, 721)
    ax1.plot(np.cos(phi), np.sin(phi), color=C["ink2"], lw=0.8)
    phi1, phi2 = np.deg2rad(20), np.deg2rad(62)
    arc = np.linspace(phi1, phi2, 50)
    ax1.plot(np.cos(arc), np.sin(arc), color=SENSOR_COLOR, lw=4, solid_capstyle="butt")
    p = np.array([-0.25, -0.35])
    a_pt, b_pt = np.array([np.cos(phi1), np.sin(phi1)]), np.array([np.cos(phi2), np.sin(phi2)])
    ang_a = math.degrees(math.atan2(*(a_pt - p)[::-1]))
    ang_b = math.degrees(math.atan2(*(b_pt - p)[::-1]))
    ax1.add_patch(Wedge((float(p[0]), float(p[1])), 0.6, ang_a, ang_b, color=C["aqua"], alpha=0.25, lw=0))
    for q in (a_pt, b_pt):
        ax1.plot([p[0], q[0]], [p[1], q[1]], color=C["ink"], lw=1.0)
    # one example direction inside the wedge, with its path length s(alpha)
    alpha = math.radians(0.8 * ang_a + 0.2 * ang_b)
    u = np.array([math.cos(alpha), math.sin(alpha)])
    b_ = p @ u
    s = -b_ + math.sqrt(b_ * b_ - (p @ p - 1))
    w = p + s * u
    ax1.annotate("", xy=w, xytext=p, arrowprops={"arrowstyle": "-|>", "color": C["blue"], "lw": 1.3})
    ax1.plot(*p, "o", color=C["ink"], ms=5)
    ax1.text(p[0] - 0.05, p[1] - 0.1, "vertex $p$", ha="right", va="top", fontsize=9.5)
    ax1.text(*(a_pt * 1.07), "$A_j$", ha="left", va="center", fontsize=10)
    ax1.text(*(b_pt * 1.08), "$B_j$", ha="left", va="bottom", fontsize=10)
    mid = math.radians(0.25 * ang_a + 0.75 * ang_b)
    ax1.text(
        p[0] + 0.45 * math.cos(mid), p[1] + 0.45 * math.sin(mid), r"$\Delta_j$", fontsize=11, ha="center", va="center"
    )
    ax1.text(*(p + 0.6 * s * u + np.array([0.04, -0.06])), r"$s(\alpha)$", color=C["blue"], fontsize=10, va="top")
    ax1.text(
        math.cos(np.deg2rad(41)) * 1.1,
        math.sin(np.deg2rad(41)) * 1.1,
        "sensor $j$",
        fontsize=9.5,
        ha="left",
        va="bottom",
    )
    ax1.text(
        -1.0,
        -1.22,
        r"$\langle n_j \rangle = \frac{Y E}{2\pi}\, \Delta_j$ (straight light)" + "\n"
        r"$\langle n_j \rangle = \frac{Y E}{2\pi} \int_{\Delta_j} e^{-s(\alpha)/\lambda_a}\, d\alpha$"
        " (with absorption)",
        fontsize=9.5,
        va="top",
    )
    ax1.set_title("(a) Subtended angle (schematic, sensor enlarged)", loc="left", fontsize=10.5)
    # (b) to scale, with the test vertices
    geometry = t3.config("straight").geometry
    draw_detector(ax2, geometry)
    labels = {}
    for number, (key, v) in enumerate(t3.VERTICES.items(), start=1):
        labels[key] = number
        for ax in (ax2, ax3):
            ax.plot(v["x"], v["y"], "o", color=C["red"], ms=5, mec="white", mew=0.8, zorder=4)
    ax2.text(0.04, 0.05, "1", fontsize=10, color=C["ink"])
    v2 = t3.VERTICES["r0.5"]
    ax2.text(v2["x"] + 0.04, v2["y"] + 0.04, "2", fontsize=10, color=C["ink"])
    box = (-1.03, -0.26, 0.2, 0.33)
    ax2.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, ec=C["ink2"], lw=0.9, ls="-"))
    ax2.text(-0.80, -0.22, "3-5: see (c)", fontsize=9, color=C["ink2"])
    ax2.text(0, -1.2, "32 sensors, photocoverage 0.509", ha="center", va="top", fontsize=9.5, color=C["ink2"])
    ax2.set_title("(b) Detector and the five test vertices", loc="left", fontsize=10.5)
    # (c) zoom near sensors 16 and 17
    draw_detector(ax3, geometry, lw=6)
    for key in ("r0.9-sensor", "r0.9-gap", "r0.98-edge"):
        v = t3.VERTICES[key]
        ax3.text(v["x"] + 0.008, v["y"] + 0.006, str(labels[key]), fontsize=10)
    ax3.text(-1.005, 0.0, "sensor 16", rotation=90, ha="right", va="center", fontsize=9, color=C["ink2"])
    ax3.text(-0.992, -0.195, "sensor 17", rotation=90, ha="right", va="center", fontsize=9, color=C["ink2"])
    ax3.set_xlim(box[0], box[0] + box[2])
    ax3.set_ylim(box[1], box[1] + box[3])
    ax3.set_title("(c) Zoom: vertices 3-5 near the wall", loc="left", fontsize=10.5)
    legend_lines = [f"{labels[k]}: {v['label']}" for k, v in t3.VERTICES.items()]
    ax3.text(1.04, 0.5, "\n".join(legend_lines), transform=ax3.transAxes, fontsize=9, va="center", ha="left")
    for ax in (ax1, ax2):
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
    for ax in (ax1, ax2, ax3):
        ax.set_aspect("equal")
        ax.axis("off")
    ax3.axis("on")
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color(C["ink2"])
    ax3.set_xlabel("zoom window 0.20 m x 0.33 m", fontsize=9, color=C["ink2"])
    savefig(fig, "t3_2_subtended_angle")
    results["vertices"] = {k: {"number": labels[k], **v} for k, v in t3.VERTICES.items()}


def compare(results: dict[str, Any]) -> dict[str, list[float]]:
    """Per-sensor mean charge vs prediction for every vertex: charges (log y) and pulls."""
    keys = list(t3.VERTICES)
    fig, axes = plt.subplots(2, len(keys), figsize=(14.0, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    pulls_all: dict[str, list[float]] = {ckey: [] for ckey in TESTED}
    out: dict[str, Any] = {}
    j = np.arange(32)
    for col, vkey in enumerate(keys):
        v = t3.VERTICES[vkey]
        out[vkey] = {}
        chi_text = []
        for ckey in TESTED:
            data = t3.vertex_run(vkey, ckey)
            x0, y0 = float(data.truth.x[0]), float(data.truth.y[0])  # the float32 vertex that was simulated
            energy = float(data.truth.energy[0])
            pred = t3.expected_charge(x0, y0, energy, data.config)
            charge = data.response.charge.astype(np.float64)
            k = len(data)
            mean = charge.mean(axis=0)
            pull = (mean - pred) / np.sqrt(pred / k)
            chi2 = float(np.sum(pull**2))
            ndf = int(pred.size)
            pulls_all[ckey].extend(pull.tolist())
            st = t3.CONFIGS[ckey]
            axes[0, col].plot(j, pred, color=st["color"], ls="--", lw=1.25, zorder=2)
            axes[0, col].plot(j, mean, "o", color=st["color"], ms=3.6, mec="white", mew=0.5, zorder=3)
            axes[1, col].plot(j, pull, "o", color=st["color"], ms=3.6, mec="white", mew=0.5, zorder=3)
            var_over_mean = charge.var(axis=0, ddof=1) / mean
            out[vkey][ckey] = {
                "n_events": k,
                "energy_MeV": energy,
                "vertex_float32": [x0, y0],
                "chi2": chi2,
                "ndf": ndf,
                "p_value": t3.chi2_sf(chi2, ndf),
                "max_abs_pull": float(np.abs(pull).max()),
                "pull_mean": float(pull.mean()),
                "pull_std": float(pull.std(ddof=1)),
                "collected_fraction_predicted": float(pred.sum() / (data.config.optics.light_yield * energy)),
                "collected_fraction_simulated": float(mean.sum() / (data.config.optics.light_yield * energy)),
                "predicted_charge": pred,
                "simulated_mean_charge": mean,
                "max_sensor": int(np.argmax(pred)),
                "max_sensor_predicted": float(pred.max()),
                "max_sensor_simulated": float(mean[np.argmax(pred)]),
                "min_sensor_predicted": float(pred.min()),
                "var_over_mean_range": [float(var_over_mean.min()), float(var_over_mean.max())],
            }
            chi_text.append(f"{st['short']} {chi2:.0f}/{ndf}")
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylim(3, 1500)
        axes[0, col].set_title(f"{list(t3.VERTICES).index(vkey) + 1}: {v['label']}", loc="left", fontsize=9.5)
        axes[1, col].set_title(r"$\chi^2$/ndf: " + ", ".join(chi_text), loc="left", fontsize=9)
        axes[1, col].axhline(0, color=C["ink2"], lw=0.8)
        axes[1, col].set_ylim(-4.5, 4.5)
        axes[1, col].set_yticks([-4, -2, 0, 2, 4])
        axes[1, col].set_xlabel("Sensor index $j$")
        axes[1, col].set_xticks([0, 8, 16, 24, 31])
    axes[0, 0].set_ylabel(r"Mean charge $\langle n_j \rangle$ [p.e.]" + "\n(E = 2 MeV)")
    axes[1, 0].set_ylabel("Pull")
    handles = [
        Line2D([], [], color=C["ink2"], ls="--", lw=1.25, label="prediction (dashed line)"),
        Line2D([], [], ls="", marker="o", color=C["ink2"], ms=4.5, label="simulation (markers)"),
        Line2D([], [], ls="", marker="s", color=t3.CONFIGS["straight"]["color"], ms=7, label="straight"),
        Line2D([], [], ls="", marker="s", color=t3.CONFIGS["absorb"]["color"], ms=7, label="absorb"),
    ]
    figure_legend(fig, handles=handles, where="bottom", ncol=4)  # the top holds the title
    fig.suptitle(
        "Expected vs simulated charge per sensor (50 000 events per vertex and preset; "
        r"pull = (mean $-$ prediction) / $\sqrt{\mathrm{prediction} / N_{\mathrm{events}}}$)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    savefig(fig, "t3_2_expected_charge")
    results["comparison"] = out
    return pulls_all


def pull_distribution(results: dict[str, Any], pulls: dict[str, list[float]]) -> None:
    """Compare the pulls of all sensors and vertices (5 x 32 per preset) with the standard normal distribution.

    The runs of one preset are independent (one seed per vertex). The two presets at the same vertex share the seed,
    hence the photon numbers and emission directions, so their pulls are correlated and are not pooled.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    edges = np.linspace(-4, 4, 17)
    summary = {}
    for ckey, values in pulls.items():
        p = np.asarray(values)
        st = t3.CONFIGS[ckey]
        chi2 = float(np.sum(p**2))
        ax.hist(p, edges.tolist(), histtype="step", color=st["color"], lw=1.5, alpha=0.6, zorder=3)
        summary[ckey] = {
            "n": int(p.size),
            "mean": float(p.mean()),
            "std": float(p.std(ddof=1)),
            "chi2": chi2,
            "ndf": int(p.size),
            "p_value": t3.chi2_sf(chi2, int(p.size)),
            "fraction_abs_above_2": float(np.mean(np.abs(p) > 2)),
            "fraction_abs_above_3": float(np.mean(np.abs(p) > 3)),
            "max_abs": float(np.abs(p).max()),
        }
    n = len(next(iter(pulls.values())))
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, n * (edges[1] - edges[0]) * np.exp(-0.5 * xs**2) / math.sqrt(2 * math.pi), **t3.ANALYTIC)
    ax.set_xlabel("Pull of the per-sensor mean charge")
    ax.set_ylabel("Sensors per bin")
    ax.set_ylim(0, 54)  # room for the legend above the histograms
    ax.set_title(f"Pulls of 5 vertices x 32 sensors per preset ({n} each)", loc="left", fontsize=10.5)
    handles = [
        Line2D(
            [],
            [],
            color=t3.CONFIGS[k]["color"],
            lw=1.5,
            label=f"{t3.CONFIGS[k]['short']}: mean {v['mean']:+.2f}, std {v['std']:.2f},\n"
            + rf"$\chi^2$ = {v['chi2']:.0f}/{v['ndf']} (p = {v['p_value']:.2f})",
        )
        for k, v in summary.items()
    ]
    handles.append(Line2D([], [], **{k: v for k, v in t3.ANALYTIC.items() if k != "zorder"}, label="standard normal"))
    legend_inside(ax, handles=handles, loc="upper center", ncol=2, labelspacing=0.9)
    savefig(fig, "t3_2_pull_distribution")
    results["pulls_per_preset"] = summary


def hint_approximation(x0: float, y0: float, energy: float, cfg: Any) -> np.ndarray:
    """Compute the approximation of the TASKS hint: ``Y E Delta_j / (2 pi) exp(-d_j / lambda_a)``, d_j to the centre."""
    delta, _ = t3.subtended_angle(np.float64(x0), np.float64(y0), cfg.geometry)
    scale = cfg.readout.detection_efficiency * cfg.optics.light_yield * energy / (2 * np.pi)
    if not cfg.optics.absorption:
        return scale * delta
    d = t3.sensor_distance(np.float64(x0), np.float64(y0), cfg.geometry)
    return scale * delta * np.exp(-d / cfg.optics.absorption_length)


def tasks_setup(results: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the set-up of the T3.2 commands of TASKS.md: (0.5, 0.2) m, 2 MeV, fixed photon numbers, 1000 events.

    With a fixed number of photons N per event the sensor charges are binomial, ``Var(n_j) = N p_j (1 - p_j)``, and
    the pulls use that variance. The hint's approximation (subtended angle times the survival probability along the
    distance to the sensor centre) is compared with the quadrature here and at the near-wall vertex 5.
    """
    out: dict[str, Any] = {}
    for ckey in TESTED:
        spec = t3.CONFIGS[ckey]
        data = dataset(
            f"t3_2_tasks_vertex_{ckey}",
            TASKS_N,
            spec["preset"],
            TASKS_SEED,
            fixed_vertex=TASKS_VERTEX,
            fixed_energy=TASKS_ENERGY,
            photon_statistics="fixed",
        )
        x0, y0, energy = float(data.truth.x[0]), float(data.truth.y[0]), float(data.truth.energy[0])
        n_photons = int(data.photons.n_emitted[0])
        pred = t3.expected_charge(x0, y0, energy, data.config)
        simple = hint_approximation(x0, y0, energy, data.config)
        charge = data.response.charge.astype(np.float64)
        mean = charge.mean(axis=0)
        p = pred / n_photons
        sigma = np.sqrt(n_photons * p * (1 - p) / len(data))  # binomial variance of n_j at fixed N
        pull, pull_simple = (mean - pred) / sigma, (mean - simple) / sigma
        chi2 = float(np.sum(pull**2))
        out[ckey] = {
            "command": f"flashsim generate --preset {spec['preset']} --fixed-vertex 0.5 0.2 --fixed-energy 2 "
            f"--photon-statistics fixed -n {TASKS_N} --seed {TASKS_SEED}",
            "n_events": len(data),
            "photons_per_event": n_photons,
            "chi2": chi2,
            "ndf": int(pred.size),
            "p_value": t3.chi2_sf(chi2, int(pred.size)),
            "collected_fraction_predicted": float(pred.sum() / (data.config.optics.light_yield * energy)),
            "collected_fraction_simulated": float(mean.sum() / (data.config.optics.light_yield * energy)),
            "max_sensor": int(np.argmax(pred)),
            "max_sensor_predicted": float(pred.max()),
            "max_sensor_simulated": float(mean[np.argmax(pred)]),
            "var_over_mean_measured_mean": float(np.mean(charge.var(axis=0, ddof=1) / mean)),
            "var_over_mean_expected_1_minus_p_mean": float(np.mean(1 - p)),
            "hint_approximation_chi2": float(np.sum(pull_simple**2)),
            "hint_approximation_max_relative_difference": float(np.max(np.abs(simple / pred - 1))),
            "predicted_charge": pred,
            "simulated_mean_charge": mean,
        }
    # the hint's approximation next to the wall: vertex 5 (r = 0.98 m, 2 cm in front of a sensor edge), with absorption
    v = t3.VERTICES["r0.98-edge"]
    absorb = t3.config("absorb")
    exact = t3.expected_charge(v["x"], v["y"], TASKS_ENERGY, absorb)
    approx = hint_approximation(v["x"], v["y"], TASKS_ENERGY, absorb)
    out["hint_approximation_at_vertex_5_absorb"] = {
        "max_relative_difference": float(np.max(np.abs(approx / exact - 1))),
        "nearest_sensor_relative_difference": float(approx[np.argmax(exact)] / exact[np.argmax(exact)] - 1),
    }
    results["tasks_setup"] = out
    return out


def main() -> None:
    """Make the T3.2 figures and store the numbers."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    style()
    results: dict[str, Any] = {}
    sketch(results)
    pulls = compare(results)
    pull_distribution(results, pulls)
    tasks = tasks_setup(results)
    results["energy_MeV"] = t3.VERTEX_ENERGY
    results["quadrature_nodes"] = 256
    absorb = t3.config("absorb")
    results["quadrature_check_max_relative_difference_256_vs_2048_nodes"] = max(
        float(
            np.max(
                np.abs(
                    t3.expected_charge(v["x"], v["y"], 1.0, absorb, 256)
                    / t3.expected_charge(v["x"], v["y"], 1.0, absorb, 2048)
                    - 1
                )
            )
        )
        for v in t3.VERTICES.values()
    )
    save_results("T3", {"T3.2": results})
    print("T3.2 chi2/ndf (p) per vertex: straight | absorb; collected fraction predicted / simulated")
    for vkey, entry in results["comparison"].items():
        text = " | ".join(
            f"{e['chi2']:.1f}/{e['ndf']} (p={e['p_value']:.2f}) {e['collected_fraction_predicted']:.4f}/"
            f"{e['collected_fraction_simulated']:.4f}"
            for e in entry.values()
        )
        print(f"  {vkey:12s} {text}")
    for ckey, a in results["pulls_per_preset"].items():
        print(
            f"  {ckey}: n={a['n']} mean={a['mean']:+.3f} std={a['std']:.3f} chi2={a['chi2']:.1f} p={a['p_value']:.3f}"
        )
    print("T3.2 TASKS.md set-up (vertex (0.5, 0.2) m, 2 MeV, fixed N, 1000 events): chi2/ndf (p), collected fraction")
    for ckey in TESTED:
        e = tasks[ckey]
        print(
            f"  {ckey:9s} {e['chi2']:.1f}/{e['ndf']} (p={e['p_value']:.2f}) {e['collected_fraction_predicted']:.4f}/"
            f"{e['collected_fraction_simulated']:.4f}; hint approximation: max relative difference "
            f"{100 * e['hint_approximation_max_relative_difference']:.2f} %, chi2 {e['hint_approximation_chi2']:.1f}"
        )


if __name__ == "__main__":
    main()
