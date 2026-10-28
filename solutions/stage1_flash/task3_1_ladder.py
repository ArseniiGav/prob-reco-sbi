"""T3.1 - the physics ladder: light collection, charge vs distance, photon delays and photon fates per configuration.

Run from the repository root::

    python solutions/stage1_flash/task3_1_ladder.py

Figures (solutions/stage1_flash/figures): t3_1_light_collection, t3_1_charge_vs_distance, t3_1_delays,
t3_1_photon_fates. Numbers: solutions/stage1_flash/results/T3.json, key "T3.1".
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import task3_common as t3
from common import C, figure_legend, legend_inside, save_results, savefig, style
from matplotlib.lines import Line2D

R_EDGES = t3.R_EDGES  # vertex radius bins [m]
D_EDGES_LOG = np.geomspace(0.01, 2.0, 31)  # vertex-sensor distance bins for the log-log panel [m]
D_EDGES_LIN = np.linspace(0.0, 2.0, 41)  # vertex-sensor distance bins for the attenuation panel [m]
# photon delay bins [ns]: 0.25 ns around the direct peak, wider in the tails (fewer photons per ns there)
DELAY_EDGES = np.concatenate([np.arange(-4.0, 8.0, 0.25), np.arange(8.0, 20.0, 1.0), np.arange(20.0, 60.0 + 1e-9, 2.0)])
FATES = {  # label -> PhotonCounts attribute
    "detected (all)": "n_detected",
    "detected: direct": "n_detected_direct",
    "detected: scattered": "n_detected_scattered",
    "detected: reflected": "n_detected_reflected",
    "absorbed in the bulk": "n_absorbed_bulk",
    "absorbed at the wall": "n_absorbed_wall",
}


def centres(edges: np.ndarray) -> np.ndarray:
    """Bin centres."""
    return 0.5 * (edges[1:] + edges[:-1])


def config_handles(keys: list[str]) -> list[Line2D]:
    """Legend handles of the configurations (colour and line style)."""
    return [
        Line2D([], [], color=t3.CONFIGS[k]["color"], ls=t3.CONFIGS[k]["ls"], label=t3.CONFIGS[k]["label"]) for k in keys
    ]


def analytic_handle(label: str = "analytic expectation") -> Line2D:
    """Legend handle of the analytic bands."""
    return Line2D([], [], **{k: v for k, v in t3.ANALYTIC.items() if k != "zorder"}, label=label)


def light_collection(results: dict[str, Any]) -> None:
    """Mean collected fraction Q / (Y E) vs vertex radius, all configurations, with the analytic expectations."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    rc = centres(R_EDGES)
    out = {}
    for key in t3.CONFIGS:
        data = t3.main(key)
        ye = data.config.optics.light_yield * data.truth.energy.astype(np.float64)
        frac = data.response.charge.sum(axis=1) / ye
        r = np.hypot(data.truth.x.astype(np.float64), data.truth.y.astype(np.float64))
        mean, sem, count = t3.binned_mean(r, frac, R_EDGES)
        style_ = t3.CONFIGS[key]
        ax.plot(rc, mean, color=style_["color"], ls=style_["ls"], zorder=3)
        out[key] = {
            "disc_average": float(frac.mean()),
            "disc_average_sem": float(frac.std() / np.sqrt(frac.size)),
            "centre_bin_0_to_0.05m": float(mean[0]),
            "wall_bin_0.99_to_1m": float(mean[-1]),
            "bin_0.85_to_0.9m": float(mean[17]),
            "wall_over_centre": float(mean[-1] / mean[0]),
            "max_sem_in_bins": float(np.nanmax(sem)),
            "wall_bin_sem": float(sem[-1]),
            "per_bin": mean,
            "sem_per_bin": sem,
            "events_per_bin_min": int(count.min()),
        }
    geometry = t3.config("straight").geometry
    coverage = geometry.photocoverage
    optics = t3.config("full").optics
    rf = np.linspace(0.0, 0.999, 200)
    absorb_curve = coverage * t3.mean_transmission(rf, optics.absorption_length)
    rho = optics.reflectivity
    reflect_estimate = coverage / (1 - rho * (1 - coverage))
    for curve in (np.full_like(rf, coverage), absorb_curve, np.full_like(rf, reflect_estimate)):
        ax.plot(rf, curve, **t3.ANALYTIC)
    label: dict[str, Any] = {"fontsize": 9, "color": C["ink2"], "ha": "left"}
    ax.text(0.02, reflect_estimate + 0.012, r"straight + reflection: $c / (1 - \rho (1 - c)) = 0.675$", **label)
    ax.text(
        0.02,
        coverage - 0.012,
        f"straight = straight + scattering = photocoverage $c$ = {coverage:.3f}",
        va="top",
        **label,
    )
    ax.text(
        0.30,
        0.212,
        r"grey band under the blue line (absorb): $c \, \langle e^{-s/\lambda_a} \rangle$ over directions",
        **label,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.75)
    ax.set_xlabel("Vertex radius $r_0$ [m]")
    ax.set_ylabel(r"Collected fraction $\langle Q / (Y E) \rangle$")
    ax.set_title("Light collection vs vertex radius (100 000 events per configuration)", loc="left")
    legend_inside(  # the empty band between the photocoverage and the reflection lines
        ax,
        handles=[*config_handles(list(t3.CONFIGS)), analytic_handle()],
        loc="center",
        bbox_to_anchor=(0.5, 0.715),
        ncol=3,
    )
    savefig(fig, "t3_1_light_collection")
    results["light_collection"] = out
    results["analytic"] = {
        "photocoverage_c": coverage,
        "absorb_centre_c_exp(-R/lambda_a)": coverage * float(np.exp(-geometry.radius / optics.absorption_length)),
        "absorb_c_T(r=0.995)": float(coverage * t3.mean_transmission(0.995, optics.absorption_length)[0]),
        "absorb_c_T(r->R)": float(coverage * t3.mean_transmission(0.9999, optics.absorption_length)[0]),
        "absorb_c_T_bin_averages": [
            float(
                np.average(
                    coverage * t3.mean_transmission(np.linspace(a, b, 41), optics.absorption_length),
                    weights=np.linspace(a, b, 41),
                )
            )
            for a, b in itertools.pairwise(R_EDGES)
        ],
        "reflect_only_estimate_c/(1-rho(1-c))": reflect_estimate,
    }


def charge_vs_distance(results: dict[str, Any]) -> None:
    """Per-sensor charge vs vertex-sensor distance, and the ratio to the straight-light (geometric) expectation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))
    out = {}
    nb = len(D_EDGES_LIN) - 1
    for key in t3.CONFIGS:
        data = t3.main(key)
        geometry = data.config.geometry
        x0, y0 = data.truth.x.astype(np.float64), data.truth.y.astype(np.float64)
        ye = data.config.optics.light_yield * data.truth.energy.astype(np.float64)
        d = t3.sensor_distance(x0, y0, geometry).ravel()
        delta, _ = t3.subtended_angle(x0, y0, geometry)
        n = data.response.charge.astype(np.float64).ravel()
        frac = n / np.repeat(ye, geometry.n_sensors)
        geometric = (ye[:, None] * delta / (2 * np.pi)).ravel()  # straight-light expectation per sensor
        style_ = t3.CONFIGS[key]
        mean, _, _ = t3.binned_mean(d, frac, D_EDGES_LOG)
        ax1.plot(centres(D_EDGES_LOG), mean, color=style_["color"], ls=style_["ls"], zorder=3)
        idx = np.digitize(d, D_EDGES_LIN) - 1
        ok = (idx >= 0) & (idx < nb)
        ratio = np.bincount(idx[ok], n[ok], nb) / np.bincount(idx[ok], geometric[ok], nb)
        ax2.plot(centres(D_EDGES_LIN), ratio, color=style_["color"], ls=style_["ls"], zorder=3)
        # effective attenuation length: straight-line fit of log(ratio) vs d for 0.2 m < d < 1.8 m
        dc = centres(D_EDGES_LIN)
        sel = (dc > 0.2) & (dc < 1.8) & np.isfinite(ratio) & (ratio > 0)
        slope, intercept = np.polyfit(dc[sel], np.log(ratio[sel]), 1)
        out[key] = {
            "ratio_to_geometric_in_d_bins": ratio,
            "fit_0.2_1.8m_slope_per_m": float(slope),
            "fit_0.2_1.8m_intercept": float(intercept),
            "effective_attenuation_length_m": float(-1 / slope) if slope < 0 else None,
            "ratio_at_d_1.00_1.05": float(ratio[20]),
            "ratio_at_d_1.95_2.00": float(ratio[-1]),
            "mean_charge_fraction_log_d_bins": mean,
        }
    a = t3.config("straight").geometry.sensor_width
    optics = t3.config("full").optics
    df = np.geomspace(0.05, 2.0, 100)
    ax1.plot(df, a / (2 * np.pi * df), **t3.ANALYTIC)
    ax1.text(0.11, 0.21, r"$a / (2\pi d)$" + "\n(normal incidence)", fontsize=9, color=C["ink2"], ha="left")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Distance vertex - sensor centre $d_j$ [m]")
    ax1.set_ylabel(r"Mean sensor charge $\langle n_j / (Y E) \rangle$")
    ax1.set_title("(a) Charge vs distance", loc="left")
    dl = np.linspace(0.0, 2.0, 100)
    lengths = {
        r"$e^{-d/\lambda_s}$": optics.scattering_length,
        r"$e^{-d/\lambda_a}$": optics.absorption_length,
        r"$e^{-d/\lambda_a - d/\lambda_s}$": 1 / (1 / optics.absorption_length + 1 / optics.scattering_length),
    }
    for text, length in lengths.items():
        ax2.plot(dl, np.exp(-dl / length), **t3.ANALYTIC)
        ax2.text(2.03, np.exp(-2.0 / length), text, fontsize=9.5, color=C["ink2"], va="center", ha="left")
    ax2.set_yscale("log")
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0.08, 3.0)
    ax2.set_xlabel("Distance vertex - sensor centre $d_j$ [m]")
    ax2.set_ylabel(r"Charge / straight-light expectation")
    ax2.set_title(r"(b) Ratio to the straight-light expectation $Y E\, \Delta_j / (2\pi)$", loc="left")
    handles = [*config_handles(list(t3.CONFIGS)), analytic_handle()]
    figure_legend(fig, handles=handles, ncol=4)
    savefig(fig, "t3_1_charge_vs_distance")
    results["charge_vs_distance"] = out
    results["distance_bins_linear_m"] = D_EDGES_LIN
    results["distance_bins_log_m"] = D_EDGES_LOG


def delays(results: dict[str, Any]) -> None:
    """Photon arrival delays t - t0 - d_j / c_n, all photons and per category, per configuration (log y)."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), sharex=True, sharey=True)
    panels = {"all detected photons": None, "direct": 0, "scattered": 1, "reflected": 2}
    out: dict[str, Any] = {}
    width = np.diff(DELAY_EDGES)
    for key in t3.CONFIGS:
        data = t3.hits(key)
        ph = t3.photon_delays(data)
        style_ = t3.CONFIGS[key]
        total = ph["delay"].size
        out[key] = {"n_photons": total, "n_events": len(data)}
        for ax, (title, cat) in zip(axes.flat, panels.items(), strict=True):
            sel = np.ones(total, bool) if cat is None else ph["category"] == cat
            if sel.sum() < 100:
                continue
            counts, _ = np.histogram(ph["delay"][sel], DELAY_EDGES)
            density = counts / total / width
            density[density == 0] = np.nan
            ax.stairs(density, DELAY_EDGES, color=style_["color"], ls=style_["ls"], lw=1.5, alpha=0.6, baseline=None)
            stats = t3.quantile_summary(ph["delay"][sel])
            stats["fraction_of_detected"] = float(sel.mean())
            stats["fraction_beyond_60ns"] = float(np.mean(ph["delay"][sel] > DELAY_EDGES[-1]))
            out[key][title.replace(" ", "_")] = stats
    for ax, title in zip(axes.flat, panels, strict=True):
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.0)
        ax.set_xlim(DELAY_EDGES[0], DELAY_EDGES[-1])
        ax.set_title(title, loc="left")
    for ax in axes[1]:
        ax.set_xlabel(r"Delay $t - t_0 - d_j / c_n$ [ns]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of detected photons per ns")
    figure_legend(fig, handles=config_handles(list(t3.CONFIGS)), where="bottom", ncol=3)  # the top holds the title
    fig.suptitle(
        f"Photon arrival delays ({t3.N_HITS} events per configuration; time jitter 1 ns; "
        "$d_j$ = distance to the hit sensor's centre)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    savefig(fig, "t3_1_delays")
    results["delays"] = out
    results["delay_bins_ns"] = DELAY_EDGES


def photon_fates(results: dict[str, Any]) -> None:
    """Where the photons end: fractions of the emitted photons per fate, per configuration (disc average)."""
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    keys = list(t3.CONFIGS)
    offsets = np.linspace(-0.3, 0.3, len(keys))
    out = {}
    for k, key in enumerate(keys):
        p = t3.main(key).photons
        emitted = p.n_emitted.astype(np.float64).sum()
        values = {label: getattr(p, attr).astype(np.float64).sum() / emitted for label, attr in FATES.items()}
        values["absorbed at a sensor"] = p.n_absorbed_sensor.astype(np.float64).sum() / emitted
        values["lost"] = p.n_lost.astype(np.float64).sum() / emitted
        values["n_lost_photons"] = int(p.n_lost.sum())
        detected = p.n_detected.astype(np.float64).sum()
        values["composition_of_detected"] = {
            cat: getattr(p, f"n_detected_{cat}").astype(np.float64).sum() / detected for cat in t3.CATEGORIES
        }
        out[key] = values
        style_ = t3.CONFIGS[key]
        x = np.array([values[f] for f in FATES])
        y = np.arange(len(FATES)) + offsets[k]
        shown = x > 0  # a process that is switched off gives exactly zero: no marker
        ax.scatter(x[shown], y[shown], s=40, color=style_["color"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(np.arange(len(FATES)), list(FATES))
    ax.invert_yaxis()
    ax.set_xlim(0, 0.72)
    ax.set_xlabel("Fraction of the emitted photons (disc average)")
    for y in np.arange(len(FATES) - 1) + 0.5:
        ax.axhline(y, color=C["grid"], lw=0.6)
    ax.set_title(
        "Photon fates (100 000 events per configuration)\n"
        "no photon was lost; none was absorbed by a sensor (efficiency 1)",
        loc="left",
        fontsize=10.5,
    )
    handles = [
        Line2D(
            [], [], ls="", marker="o", ms=6.5, color=t3.CONFIGS[k]["color"], mec="white", label=t3.CONFIGS[k]["label"]
        )
        for k in keys
    ]
    legend_inside(  # the empty right half of the two middle rows
        ax,
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.985, 0.5),
        ncol=2,
        title="top to bottom in each row",
        title_fontsize=9,
    )
    savefig(fig, "t3_1_photon_fates")
    results["photon_fates"] = out


def main() -> None:
    """Make the T3.1 figures and store the numbers."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    style()
    results: dict[str, Any] = {}
    light_collection(results)
    charge_vs_distance(results)
    delays(results)
    photon_fates(results)
    results["n_events_main"] = t3.N_MAIN
    results["seed_main"] = t3.MAIN_SEED
    results["n_events_hits"] = t3.N_HITS
    results["seed_hits"] = t3.HITS_SEED
    results["r_bins_m"] = R_EDGES
    save_results("T3", {"T3.1": results})
    lc = results["light_collection"]
    print("T3.1 collected fraction <Q/(YE)>: disc average | centre bin | wall bin")
    for key, v in lc.items():
        print(
            f"  {key:15s} {v['disc_average']:.4f} | {v['centre_bin_0_to_0.05m']:.4f} | {v['wall_bin_0.99_to_1m']:.4f}"
        )
    print("T3.1 photon fates (fraction of emitted): detected / absorbed in bulk / absorbed at wall")
    for key, v in results["photon_fates"].items():
        print(
            f"  {key:15s} {v['detected (all)']:.4f} / {v['absorbed in the bulk']:.4f} / {v['absorbed at the wall']:.4f}"
        )
    print("T3.1 detected-light composition direct / scattered / reflected:")
    for key, v in results["photon_fates"].items():
        print(f"  {key:15s} " + " / ".join(f"{v['composition_of_detected'][c]:.3f}" for c in t3.CATEGORIES))
    print("T3.1 fitted attenuation length of the charge ratio [m]:")
    for key, v in results["charge_vs_distance"].items():
        print(f"  {key:15s} {v['effective_attenuation_length_m']}")


if __name__ == "__main__":
    main()
