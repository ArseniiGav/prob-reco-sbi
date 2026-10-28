"""T5.3 (bonus): detector design -- 8, 16, 32, 48 sensors at fixed sensor width and at fixed photocoverage.

Two series of detectors (full model, everything else at the defaults):

* fixed sensor width a = 0.10 m: M = 8, 16, 32 and 48 sensors (photocoverage 0.13 ... 0.76).
* fixed photocoverage 0.509: M = 8, 16, 32, 48 with a = 0.40, 0.20, 0.10, 0.0667 m.

For each detector 40 000 events are simulated from the default prior with the same seed, so all detectors see the
same events (same vertices, energies and t0). Measured: the fraction of empty sensors vs E, the light collection
Q / (Y E) vs r, the radial response of the charge barycentre and the vertex resolution of the barycentre after a
radial calibration (the median barycentre radius vs the true radius, inverted; calibrated on the even events and
evaluated on the odd ones).

Run from the repository root::

    python solutions/stage1_flash/task5_detector_design.py

Figures: t5_3_detector_design (the four measurements, one row per series) and t5_3_layouts (one event on every
detector). Numbers: solutions/stage1_flash/results/T5.json (key T5.3).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from common import C, dataset, round_label, save_results, savefig, style
from event_display import barycentre, charge_colorbar, draw_barycentre, draw_detector, draw_vertex
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator, NullFormatter

from prob_reco_sbi.flash import SimResult, preset, simulate

N_EV = 40_000
SEED = 53
WIDTH = 0.10
COVERAGE = 32 * 0.10 / (2 * math.pi)  # 0.509, the default detector
SERIES = {
    "fixed width": [(8, WIDTH), (16, WIDTH), (32, WIDTH), (48, WIDTH)],
    "fixed coverage": [(8, 0.40), (16, 0.20), (32, 0.10), (48, 0.2 / 3)],
}
SERIES_LABEL = {"fixed width": "Fixed width, a = 10 cm", "fixed coverage": "Fixed coverage 0.51"}
#: M = 8, 16, 32, 48 sensors: the paper's colours in order (darkgreen, royalblue, firebrick, indigo), one marker each
COLOURS = [C["green"], C["blue"], C["red"], C["violet"]]
MARKERS = ["o", "s", "^", "D"]
R_EDGES = np.linspace(0.0, 1.0, 21)
E_EDGES = np.linspace(0.5, 5.0, 10)
SHOW_EVENT = {"x": 0.45, "y": 0.30, "energy": 1.0, "t0": 3.0}


def binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray, fn: object = np.mean) -> np.ndarray:
    """``fn(y)`` in bins of ``x``."""
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    return np.array([fn(y[idx == k]) for k in range(len(edges) - 1)])  # type: ignore[operator]


def analyse(data: SimResult) -> dict[str, object]:
    """Empty sensors, light collection and barycentre performance of one detector."""
    t = data.truth
    n = data.response.charge.astype(np.float64)
    r = np.hypot(t.x, t.y).astype(np.float64)
    energy = t.energy.astype(np.float64)
    y = data.config.optics.light_yield
    q = n.sum(1)
    bx, by = barycentre(n, data.detector)
    rb = np.hypot(bx, by)
    centres = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    # radial calibration on the even events: median barycentre radius vs true radius, made monotone and inverted
    cal = np.arange(len(r)) % 2 == 0
    med = np.maximum.accumulate(binned(r[cal], rb[cal], R_EDGES, np.median))
    test = ~cal
    r_corr = np.interp(rb[test], med, centres)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = bx[test] / rb[test], by[test] / rb[test]
    miss = np.hypot(r_corr * ux - t.x[test], r_corr * uy - t.y[test])
    raw_miss = np.hypot(bx[test] - t.x[test], by[test] - t.y[test])
    q68 = lambda v: np.percentile(v, 68.0)  # noqa: E731
    return {
        "n_sensors": data.n_sensors,
        "sensor_width_m": data.config.geometry.sensor_width,
        "photocoverage": data.config.geometry.photocoverage,
        "empty_fraction_all": float((n == 0).mean()),
        "empty_fraction_vs_E": binned(energy, (n == 0).mean(1), E_EDGES),
        "collection_mean": float((q / (y * energy)).mean()),
        "collection_vs_r": binned(r, q / (y * energy), R_EDGES),
        "collection_centre_bin": float(binned(r, q / (y * energy), R_EDGES)[0]),
        "collection_wall_bin": float(binned(r, q / (y * energy), R_EDGES)[-1]),
        "mean_charge_per_sensor_pe": float(n.mean()),
        "reflected_share_of_detected": float(data.photons.n_detected_reflected.sum() / data.photons.n_detected.sum()),
        "rb_median_vs_r": binned(r, rb, R_EDGES, np.median),
        "rb_over_r_median_at_r0p5": float(np.median((rb / r)[(r > 0.45) & (r < 0.55)])),
        "resolution68_vs_r_cm": 100 * binned(r[test], miss, R_EDGES, q68),
        "resolution68_all_cm": float(100 * np.percentile(miss, 68)),
        "resolution68_r_lt_0p8_cm": float(100 * np.percentile(miss[r[test] < 0.8], 68)),
        "raw_miss68_all_cm": float(100 * np.percentile(raw_miss, 68)),
        "resolution68_E_1p5_2p5_cm": float(100 * np.percentile(miss[(energy[test] > 1.5) & (energy[test] < 2.5)], 68)),
    }


def main() -> None:
    """Simulate every detector, draw the figures and save the numbers."""
    style()
    results: dict[str, dict[str, object]] = {}
    for series, designs in SERIES.items():
        results[series] = {}
        for m, a in designs:
            data = dataset(f"t5_3_m{m}_a{a:.2f}", N_EV, "full", seed=SEED, n_sensors=m, sensor_width=a)
            results[series][f"M{m}"] = analyse(data)
            print(
                f"T5.3 {series:15s} M = {m:2d}, a = {a:.2f} m: coverage {data.config.geometry.photocoverage:.3f}, "
                f"empty {results[series][f'M{m}']['empty_fraction_all']:.3f}, "
                f"collection {results[series][f'M{m}']['collection_mean']:.3f}, "
                f"R68 {results[series][f'M{m}']['resolution68_all_cm']:.1f} cm"
            )

    # ---------------------------------------------------------------------------------- figure: the measurements
    rc = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    ec = 0.5 * (E_EDGES[1:] + E_EDGES[:-1])
    fig = plt.figure(figsize=(14.0, 7.4))
    rows = fig.subfigures(2, 1, hspace=0.04)
    for row, (series, designs) in enumerate(SERIES.items()):
        axes = rows[row].subplots(1, 4)
        for k, (m, a) in enumerate(designs):
            v = results[series][f"M{m}"]
            st = {"color": COLOURS[k], "marker": MARKERS[k], "ms": 3.8, "lw": 1.4}
            lab = f"M = {m}, a = {100 * a:.0f} cm, coverage {v['photocoverage']:.2f}"
            axes[0].plot(ec, v["empty_fraction_vs_E"], label=lab, **st)
            axes[1].plot(rc, v["collection_vs_r"], **st)
            axes[2].plot(rc, v["rb_median_vs_r"], **st)
            axes[3].plot(rc, v["resolution68_vs_r_cm"], **st)
        axes[2].plot([0, 1], [0, 1], color=C["grey"], lw=0.8, ls=":", label="barycentre = truth")
        handles, labels = axes[0].get_legend_handles_labels()
        rows[row].legend(handles, labels, loc="outside upper center", ncol=4, fontsize=9.5, frameon=False)
        round_label(axes[0], SERIES_LABEL[series], 0.95, 0.92, fontsize=10, ha="right", va="top")
        axes[0].set_ylabel("Fraction of empty sensors")
        axes[1].set_ylabel("Light collection Q / (Y E)")
        axes[2].set_ylabel("Median barycentre radius [m]")
        axes[3].set_ylabel("Vertex resolution, 68 % [cm]")
        axes[0].set_xlabel("Energy E [MeV]")
        axes[1].set_xlabel("Vertex radius $r_0$ [m]")
        axes[2].set_xlabel("Vertex radius $r_0$ [m] (dotted: = truth)")
        axes[3].set_xlabel("Vertex radius $r_0$ [m]")
        axes[0].set_yscale("log")
        axes[0].set_ylim(1e-4, 1)
        axes[0].set_xlim(0.4, 5.1)
        axes[1].set_ylim(0.0, 0.75 if series == "fixed width" else 0.65)
        axes[2].set_ylim(0.0, 1.0)
        axes[3].set_yscale("log")
        axes[3].set_ylim(1.0, 100.0)
        for a in axes[1:]:
            a.set_xlim(0.0, 1.0)
        for a in (axes[0], axes[3]):
            a.yaxis.set_minor_formatter(NullFormatter())
    savefig(fig, "t5_3_detector_design")

    # ---------------------------------------------------------------------------------- figure: one event, 7 designs
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.6))
    ev_out: dict[str, object] = {}
    for row, (series, designs) in enumerate(SERIES.items()):
        for k, (m, a) in enumerate(designs):
            ev = simulate(
                SHOW_EVENT["x"],
                SHOW_EVENT["y"],
                SHOW_EVENT["energy"],
                SHOW_EVENT["t0"],
                config=preset("full", n_sensors=m, sensor_width=a),
                seed=531,
                device="cpu",
            )
            ax = axes[row, k]
            n = ev.response.charge[0].astype(float)
            norm = PowerNorm(0.5, vmin=0.0, vmax=max(n.max(), 1.0))
            draw_detector(ax, ev.detector, n, norm=norm, labels=())
            bx, by = barycentre(n, ev.detector)
            draw_vertex(ax, SHOW_EVENT["x"], SHOW_EVENT["y"])
            draw_barycentre(ax, float(bx), float(by))
            ax.set_title(
                f"{series}: M = {m}, a = {100 * a:.0f} cm\n"
                f"Q = {n.sum():.0f} p.e.; empty sensors: {int((n == 0).sum())}",
                fontsize=9.5,
            )
            bar = charge_colorbar(
                fig,
                ax,
                norm,
                label="Charge [p.e.], square-root scale",
                orientation="horizontal",
                shrink=0.7,
                pad=0.02,
                aspect=20,
                ticks=MaxNLocator(nbins=4, integer=True),
            )
            bar.ax.tick_params(labelsize=8.5)
            ev_out[f"{series} M{m}"] = {"Q_pe": int(n.sum()), "empty": int((n == 0).sum()), "max_pe": int(n.max())}
    r0 = math.hypot(SHOW_EVENT["x"], SHOW_EVENT["y"])
    fig.suptitle(
        f"One event on every detector: E = {SHOW_EVENT['energy']:.2f} MeV at (x0, y0) = "
        f"({SHOW_EVENT['x']:.2f}, {SHOW_EVENT['y']:.2f}) m (r = {r0:.2f} m), same random seed",
        fontsize=11,
    )
    savefig(fig, "t5_3_layouts")

    payload = {
        "n_events_each": N_EV,
        "seed": SEED,
        "r_bins_m": R_EDGES,
        "E_bins_MeV": E_EDGES,
        "designs": results,
        "example_event": {"theta": SHOW_EVENT, "per_design": ev_out},
    }
    path = save_results("T5", {"T5.3": payload})
    print(f"numbers -> {path}")


if __name__ == "__main__":
    main()
