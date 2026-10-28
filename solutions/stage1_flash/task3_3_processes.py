"""T3.3 - which process changes the light, which one destroys the timing, and why the light grows near the wall.

Run from the repository root (after or without task3_1_ladder.py; the data sets are cached)::

    python solutions/stage1_flash/task3_3_processes.py

Figures: t3_3_process_effects, t3_3_collection_by_category, t3_3_timing_residuals, t3_3_charge_pattern.
Numbers: results/T3.json, key "T3.3".
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import task3_common as t3
from common import C, figure_legend, save_results, savefig, style
from matplotlib.lines import Line2D

R_EDGES = t3.R_EDGES
RES_EDGES = np.concatenate([np.arange(-6.0, 8.0, 0.25), np.arange(8.0, 20.0, 1.0), np.arange(20.0, 30.0 + 1e-9, 2.0)])
LATE = 3.0  # ns: a photon or a sensor time more than 3 sigma_t later than the direct flight counts as late


def centres(edges: np.ndarray) -> np.ndarray:
    """Bin centres."""
    return 0.5 * (edges[1:] + edges[:-1])


def config_handles(keys: list[str]) -> list[Line2D]:
    """Legend handles of the configurations."""
    return [
        Line2D([], [], color=t3.CONFIGS[k]["color"], ls=t3.CONFIGS[k]["ls"], label=t3.CONFIGS[k]["label"]) for k in keys
    ]


def analytic_handle(label: str = "analytic expectation") -> Line2D:
    """Legend handle of the analytic bands."""
    return Line2D([], [], **{k: v for k, v in t3.ANALYTIC.items() if k != "zorder"}, label=label)


def direct_length(key: str) -> float:
    """Attenuation length of the direct light: a photon stays direct only without any bulk interaction."""
    optics = t3.config(key).optics
    mu = (1 / optics.absorption_length if optics.absorption else 0.0) + (
        1 / optics.scattering_length if optics.scattering else 0.0
    )
    return math.inf if mu == 0 else 1 / mu


def residuals(key: str) -> dict[str, np.ndarray]:
    """First-hit and mean-time residuals t - t0 - d_j / c_n of every hit sensor of the main data set [ns]."""
    data = t3.main(key)
    d = t3.sensor_distance(data.truth.x, data.truth.y, data.config.geometry)
    flight = data.truth.t0.astype(np.float64)[:, None] + d / data.config.optics.speed_of_light
    hit = data.response.charge > 0
    return {
        "first": (data.response.time_first.astype(np.float64) - flight)[hit],
        "mean": (data.response.time_mean.astype(np.float64) - flight)[hit],
        "charge": data.response.charge[hit],
    }


def collection_by_category(results: dict[str, Any]) -> dict[str, Any]:
    """Plot the collected fraction per photon category vs r, with the analytic expectation of the direct light."""
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), sharex=True)
    panels = {
        "all detected photons": "n_detected",
        "direct": "n_detected_direct",
        "scattered": "n_detected_scattered",
        "reflected": "n_detected_reflected",
    }
    rc = centres(R_EDGES)
    coverage = t3.config("straight").geometry.photocoverage
    out: dict[str, Any] = {}
    for key in t3.CONFIGS:
        data = t3.main(key)
        ye = data.config.optics.light_yield * data.truth.energy.astype(np.float64)
        r = np.hypot(data.truth.x.astype(np.float64), data.truth.y.astype(np.float64))
        st = t3.CONFIGS[key]
        out[key] = {}
        for ax, (title, attr) in zip(axes.flat, panels.items(), strict=True):
            values = getattr(data.photons, attr).astype(np.float64) / ye
            mean, sem, _ = t3.binned_mean(r, values, R_EDGES)
            out[key][title.replace(" ", "_")] = {
                "per_bin": mean,
                "centre_bin": float(mean[0]),
                "wall_bin": float(mean[-1]),
                "disc_average": float(values.mean()),
                "max_sem": float(np.nanmax(sem)),
            }
            if values.max() == 0:
                continue
            ax.plot(rc, mean, color=st["color"], ls=st["ls"], zorder=3)
    rf = np.linspace(0, 0.999, 200)
    seen = set()
    analytic = {}
    for key in t3.CONFIGS:
        length = direct_length(key)
        curve = coverage * t3.mean_transmission(rf, length)
        analytic[key] = {
            "direct_attenuation_length_m": None if math.isinf(length) else length,
            "c_T_centre": float(coverage * t3.mean_transmission(0.0, length)[0]),
            "c_T_r0.995": float(coverage * t3.mean_transmission(0.995, length)[0]),
            "c_T_bin_averages": [
                float(
                    np.average(
                        coverage * t3.mean_transmission(np.linspace(a, b, 41), length), weights=np.linspace(a, b, 41)
                    )
                )
                for a, b in itertools.pairwise(R_EDGES)
            ],
        }
        if length not in seen:
            seen.add(length)
            axes[0, 1].plot(rf, curve, **t3.ANALYTIC)
    labels = [
        (coverage + 0.03, r"$c$ (straight, straight + reflection)"),
        (0.352, r"$c\,\langle e^{-s/\lambda_s}\rangle$ (straight + scattering)"),
        (0.225, r"$c\,\langle e^{-s/\lambda_a}\rangle$ (absorb)"),
        (0.115, r"$c\,\langle e^{-s/\lambda_a - s/\lambda_s}\rangle$ (absorb-scatter = full: indigo on firebrick)"),
    ]
    for y, text in labels:
        axes[0, 1].text(0.03, y, text, fontsize=8.8, color=C["ink2"], va="center")
    axes[1, 0].text(
        0.03,
        0.035,
        "absorb-scatter = full (indigo on firebrick):\nphotons that are also reflected count as reflected",
        fontsize=8.8,
        color=C["ink2"],
        va="center",
    )
    limits = {
        "all detected photons": (0.0, 0.75),
        "direct": (0.0, 0.6),
        "scattered": (0.0, 0.22),
        "reflected": (0.0, 0.22),
    }
    for ax, title in zip(axes.flat, panels, strict=True):
        ax.set_title(title, loc="left")
        ax.set_ylim(*limits[title])
        ax.set_xlim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Vertex radius $r_0$ [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Collected fraction $\langle n / (Y E) \rangle$")
    figure_legend(  # below the panels: the top holds the title
        fig,
        handles=[*config_handles(list(t3.CONFIGS)), analytic_handle("analytic expectation (direct light)")],
        where="bottom",
        ncol=4,
    )
    fig.suptitle(
        "Collected light per photon category vs vertex radius (100 000 events per configuration)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    savefig(fig, "t3_3_collection_by_category")
    # growth of the collected light from the centre bin to the wall bin, split by category
    growth = {}
    for key in t3.CONFIGS:
        entry = out[key]
        total = entry["all_detected_photons"]["wall_bin"] - entry["all_detected_photons"]["centre_bin"]
        growth[key] = {"total": total}
        for cat in t3.CATEGORIES:
            growth[key][cat] = entry[cat]["wall_bin"] - entry[cat]["centre_bin"]
    results["collection_by_category"] = out
    results["direct_light_analytic"] = analytic
    results["growth_centre_to_wall_bin"] = growth
    return out


def timing(results: dict[str, Any]) -> dict[str, Any]:
    """First-hit and mean-time residuals of the hit sensors (log y), and photon-level late fractions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    width = np.diff(RES_EDGES)
    out: dict[str, Any] = {}
    for key in t3.CONFIGS:
        res = residuals(key)
        st = t3.CONFIGS[key]
        out[key] = {
            "first_hit": t3.quantile_summary(res["first"], LATE),
            "mean_time": t3.quantile_summary(res["mean"], LATE),
        }
        for ax, name in ((ax1, "first"), (ax2, "mean")):
            counts, _ = np.histogram(res[name], RES_EDGES)
            density = counts / res[name].size / width
            density[density == 0] = np.nan
            ax.stairs(density, RES_EDGES, color=st["color"], ls=st["ls"], lw=1.5, alpha=0.6, baseline=None)
        # photon level, from the hit lists
        ph = t3.photon_delays(t3.hits(key))
        out[key]["photons"] = {
            f"fraction_delay_above_{LATE:g}ns": float(np.mean(ph["delay"] > LATE)),
            "fraction_delay_above_10ns": float(np.mean(ph["delay"] > 10.0)),
            "direct_share": float(np.mean(ph["category"] == 0)),
            "mean_delay_ns": float(ph["delay"].mean()),
        }
    for ax, title in ((ax1, "(a) First hit of each sensor"), (ax2, "(b) Mean hit time of each sensor")):
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2.0)
        ax.set_xlim(RES_EDGES[0], RES_EDGES[-1])
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"Residual $t - t_0 - d_j / c_n$ [ns]")
        ax.axvline(LATE, color=C["ink2"], lw=0.8, ls=":")
    ax1.text(LATE + 0.6, 0.6, f"{LATE:g} ns", fontsize=9, color=C["ink2"])
    ax1.set_ylabel("Fraction of hit sensors per ns")
    figure_legend(fig, handles=config_handles(list(t3.CONFIGS)), where="bottom", ncol=3)  # the top holds the title
    fig.suptitle(
        "Timing residuals of the hit sensors (100 000 events, 32 sensors; time jitter 1 ns)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    savefig(fig, "t3_3_timing_residuals")
    results["timing"] = out
    return out


def charge_pattern(results: dict[str, Any]) -> None:
    """Per-sensor charge at two fixed vertices for all configurations: scattering and reflection flatten the pattern."""
    shown = ["r0.5", "r0.9-sensor"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    out: dict[str, Any] = {}
    j = np.arange(32)
    for vkey in t3.VERTICES:
        out[vkey] = {}
        for key in t3.CONFIGS:
            data = t3.vertex_run(vkey, key)
            ye = data.config.optics.light_yield * float(data.truth.energy[0])
            mean = data.response.charge.astype(np.float64).mean(axis=0) / ye
            photons = data.photons
            detected = photons.n_detected.astype(np.float64).sum()
            out[vkey][key] = {
                "n_events": len(data),
                "composition_of_detected": {
                    cat: float(getattr(photons, f"n_detected_{cat}").astype(np.float64).sum() / detected)
                    for cat in t3.CATEGORIES
                },
                "mean_fraction_per_sensor": mean,
                "max_over_min": float(mean.max() / mean.min()),
                "brightest_sensor_share": float(mean.max() / mean.sum()),
                "collected_fraction": float(mean.sum()),
            }
            if vkey in shown:
                ax = axes[shown.index(vkey)]
                st = t3.CONFIGS[key]
                ax.plot(j, mean, color=st["color"], ls=st["ls"], marker="o", ms=2.8, lw=1.3, zorder=3)
        if vkey in shown:
            ax = axes[shown.index(vkey)]
            v = t3.VERTICES[vkey]
            pred = t3.expected_charge(v["x"], v["y"], 1.0, t3.config("absorb")) / t3.config("absorb").optics.light_yield
            ax.plot(j, pred, **t3.ANALYTIC)
            ax.set_yscale("log")
            ax.set_title(f"{v['label']} (E = 2 MeV)", loc="left")
            ax.set_xlabel("Sensor index $j$")
            ax.set_xticks([0, 8, 16, 24, 31])
    axes[0].set_ylabel(r"Mean sensor charge $\langle n_j \rangle / (Y E)$")
    figure_legend(
        fig,
        handles=[*config_handles(list(t3.CONFIGS)), analytic_handle("analytic expectation for absorb (T3.2)")],
        ncol=4,
    )
    savefig(fig, "t3_3_charge_pattern")
    results["charge_pattern"] = out


def process_effects(results: dict[str, Any], timing_out: dict[str, Any]) -> None:
    """One row per configuration: collected light, direct share, late photons and the spread of the sensor times."""
    keys = list(t3.CONFIGS)
    metrics = {}
    for key in keys:
        data = t3.main(key)
        ye = data.config.optics.light_yield * data.truth.energy.astype(np.float64)
        p = data.photons
        metrics[key] = {
            "collected_fraction": float((data.response.charge.sum(axis=1) / ye).mean()),
            "direct_share_of_detected": float(p.n_detected_direct.sum() / p.n_detected.sum()),
            "late_photon_fraction": timing_out[key]["photons"][f"fraction_delay_above_{LATE:g}ns"],
            "first_hit_half_width_68": timing_out[key]["first_hit"]["half_width_68"],
            "mean_time_half_width_68": timing_out[key]["mean_time"]["half_width_68"],
            "mean_time_late_fraction": timing_out[key]["mean_time"][f"fraction_above_{LATE:g}ns"],
        }
    straight = metrics["straight"]["collected_fraction"]
    for key in keys:
        metrics[key]["collected_relative_to_straight"] = metrics[key]["collected_fraction"] / straight - 1
    ladder_steps = {
        "absorption (straight -> absorb)": ("straight", "absorb"),
        "scattering (absorb -> absorb-scatter)": ("absorb", "absorb-scatter"),
        "reflection (absorb-scatter -> full)": ("absorb-scatter", "full"),
        "scattering alone (straight -> straight + scattering)": ("straight", "scatter-only"),
        "reflection alone (straight -> straight + reflection)": ("straight", "reflect-only"),
    }
    steps = {
        name: metrics[b]["collected_fraction"] / metrics[a]["collected_fraction"] - 1
        for name, (a, b) in ladder_steps.items()
    }
    panels = [
        ("collected_fraction", r"Collected fraction $\langle Q/(YE) \rangle$", (0.0, 0.75)),
        ("direct_share_of_detected", "Direct share of the detected light", (0.0, 1.05)),
        ("late_photon_fraction", f"Detected photons with a delay\n$t - t_0 - d_j / c_n$ > {LATE:g} ns", (0.0, 0.3)),
        ("mean_time_late_fraction", f"Sensors whose mean delay\n$\\tau_j - t_0 - d_j / c_n$ > {LATE:g} ns", (0.0, 0.3)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(12.4, 3.4), sharey=True)
    y = np.arange(len(keys))
    for ax, (name, label, lim) in zip(axes, panels, strict=True):
        for k, key in enumerate(keys):
            st = t3.CONFIGS[key]
            ax.plot([0, metrics[key][name]], [k, k], color=C["grid"], lw=1.0, zorder=1)
            ax.plot(metrics[key][name], k, "o", color=st["color"], ms=7, mec="white", mew=0.8, zorder=3)
        ax.set_xlim(*lim)
        ax.set_xlabel(label)
    axes[0].axvline(metrics["straight"]["collected_fraction"], color=C["ink2"], lw=0.8, ls=":")
    axes[0].set_yticks(y, [t3.CONFIGS[k]["label"] for k in keys])
    axes[0].invert_yaxis()
    fig.suptitle("What each process does (disc averages; dotted line: straight light)", x=0.01, ha="left", fontsize=11)
    savefig(fig, "t3_3_process_effects")
    results["process_effects"] = metrics
    results["collected_light_change_per_step"] = steps


def main() -> None:
    """Make the T3.3 figures and store the numbers."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    style()
    results: dict[str, Any] = {}
    collection_by_category(results)
    timing_out = timing(results)
    process_effects(results, timing_out)
    charge_pattern(results)
    results["late_threshold_ns"] = LATE
    save_results("T3", {"T3.3": results})
    print("T3.3 change of the collected light per step:")
    for name, value in results["collected_light_change_per_step"].items():
        print(f"  {name:55s} {100 * value:+.1f} %")
    print(
        "T3.3 per configuration: collected | direct share | photons > 3 ns | first-hit 68% hw | mean-time 68% hw | "
        "sensors with mean time > 3 ns"
    )
    for key, m in results["process_effects"].items():
        print(
            f"  {key:15s} {m['collected_fraction']:.4f} | {m['direct_share_of_detected']:.3f} | "
            f"{m['late_photon_fraction']:.4f} | {m['first_hit_half_width_68']:.3f} | "
            f"{m['mean_time_half_width_68']:.3f} | {m['mean_time_late_fraction']:.4f}"
        )
    print("T3.3 growth centre bin -> wall bin (collected fraction): total / direct / scattered / reflected")
    for key, g in results["growth_centre_to_wall_bin"].items():
        print(f"  {key:15s} {g['total']:+.4f} / {g['direct']:+.4f} / {g['scattered']:+.4f} / {g['reflected']:+.4f}")


if __name__ == "__main__":
    main()
