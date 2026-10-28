"""T3.4 - how hard is the inverse problem? Naive estimators per configuration.

Energy: ``E_hat = Q / k`` with one calibration constant per configuration, ``k = median(Q / E)`` over an independent
calibration sample from the same prior (50 000 events, seed 32), as the TASKS.md hint says; in every bin of r the
bias is the median of ``E_hat / E - 1`` and the resolution half the width of its central 68 % interval (the robust
measures of the hint). The mean-based variants (``k = sum(Q) / sum(E)``, mean and standard deviation per bin) are
stored as well: the two differ where the distribution of Q / E is skewed or bimodal, i.e. next to the wall.
Vertex: the charge barycentre ``sum_j n_j (x_j, y_j) / sum_j n_j`` of the sensor centres (mean and standard
deviation per bin).

For the summary two references are added, both calibrated on the same calibration sample:
* "vertex known": ``E_hat = Q / (Y f(r, psi))`` with the mean collected fraction f in bins of r^2 and of the azimuth
  psi relative to the nearest sensor centre (the best a reconstruction of the vertex could do for the energy scale);
* Poisson floor: ``1 / sqrt(Y E f)``, the relative spread of Q itself (Q is Poisson at fixed theta), which no
  estimator of E can beat when the vertex is known;
* "radially calibrated barycentre": the barycentre radius mapped back through its mean response curve.

Run from the repository root::

    python solutions/stage1_flash/task3_4_naive_estimators.py

Figures: t3_4_energy_estimator, t3_4_barycentre, t3_4_inverse_problem. Numbers: results/T3.json, key "T3.4".
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
from common import C, figure_legend, legend_inside, save_results, savefig, style
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from prob_reco_sbi.flash import SimResult, sensor_positions

R_EDGES = t3.R_EDGES
N_R2_MAP, N_PSI_MAP = 80, 6  # vertex-known energy map: bins of r^2 (equal population) and of the folded azimuth
NAMED_BINS = {"0_to_0.05m": 0, "0.45_to_0.5m": 9, "0.85_to_0.9m": 17, "0.95_to_0.96m": 23, "0.99_to_1m": -1}


def centres(edges: np.ndarray) -> np.ndarray:
    """Bin centres."""
    return 0.5 * (edges[1:] + edges[:-1])


def binned_quantiles(x: np.ndarray, y: np.ndarray, edges: np.ndarray, q: tuple[float, ...]) -> np.ndarray:
    """Percentiles ``q`` of y in bins of x, shape ``[len(q), n_bins]`` (NaN for empty bins)."""
    idx = np.digitize(x, edges) - 1
    out = np.full((len(q), len(edges) - 1), np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[:, b] = np.percentile(y[sel], q)
    return out


def config_handles(keys: list[str]) -> list[Line2D]:
    """Legend handles of the configurations."""
    return [
        Line2D([], [], color=t3.CONFIGS[k]["color"], ls=t3.CONFIGS[k]["ls"], label=t3.CONFIGS[k]["label"]) for k in keys
    ]


def folded_azimuth(x: np.ndarray, y: np.ndarray, n_sensors: int) -> np.ndarray:
    """Azimuth of the vertex relative to the nearest sensor centre, folded into [0, pi / M] (detector symmetry)."""
    period = 2 * np.pi / n_sensors
    phi = np.mod(np.arctan2(y, x), period)
    return np.minimum(phi, period - phi)


def summaries(data: SimResult) -> dict[str, np.ndarray]:
    """Per-event truth, total charge and charge barycentre."""
    x = data.truth.x.astype(np.float64)
    y = data.truth.y.astype(np.float64)
    n = data.response.charge.astype(np.float64)
    q = n.sum(axis=1)
    sx, sy = sensor_positions(data.config.geometry)
    bx, by = n @ sx / q, n @ sy / q
    return {
        "x": x,
        "y": y,
        "r": np.hypot(x, y),
        "psi": folded_azimuth(x, y, data.config.geometry.n_sensors),
        "energy": data.truth.energy.astype(np.float64),
        "q": q,
        "bx": bx,
        "by": by,
        "rb": np.hypot(bx, by),
    }


class Calibration:
    """Calibration constants of one configuration from its calibration sample."""

    def __init__(self, cal: dict[str, np.ndarray], light_yield: float, n_sensors: int) -> None:
        self.k = float(np.median(cal["q"] / cal["energy"]))  # photo-electrons per MeV: median of Q / E (TASKS hint)
        self.k_mean = float(cal["q"].sum() / cal["energy"].sum())  # the mean-based constant, for comparison
        self.light_yield = light_yield
        # vertex-known map of the collected fraction in bins of r^2 (equal population for the uniform prior) and psi
        self.r2_edges = np.linspace(0, 1, N_R2_MAP + 1)
        self.psi_edges = np.linspace(0, np.pi / n_sensors, N_PSI_MAP + 1)
        i, j = self._bins(cal["r"], cal["psi"])
        flat = i * N_PSI_MAP + j
        qsum = np.bincount(flat, cal["q"], N_R2_MAP * N_PSI_MAP)
        esum = np.bincount(flat, cal["energy"], N_R2_MAP * N_PSI_MAP)
        self.map_counts = np.bincount(flat, minlength=N_R2_MAP * N_PSI_MAP)
        self.f_map = qsum / esum / light_yield  # mean collected fraction per bin
        # mean barycentre radius vs true radius (response curve), made monotonic for the inversion
        edges = np.sqrt(np.linspace(0, 1, 41))
        mean, _, _ = t3.binned_mean(cal["r"], cal["rb"], edges)
        self.g_r = np.concatenate([[0.0], centres(edges), [1.0]])
        self.g = np.maximum.accumulate(np.concatenate([[mean[0]], mean, [mean[-1]]]))

    def _bins(self, r: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.clip(np.digitize(r**2, self.r2_edges) - 1, 0, N_R2_MAP - 1)
        j = np.clip(np.digitize(psi, self.psi_edges) - 1, 0, N_PSI_MAP - 1)
        return i, j

    def collected_fraction(self, r: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Mean collected fraction f(r, psi) of the calibration map."""
        i, j = self._bins(r, psi)
        return self.f_map[i * N_PSI_MAP + j]

    def radius(self, rb: np.ndarray) -> np.ndarray:
        """Invert the barycentre response curve: radius whose mean barycentre radius is rb."""
        return np.interp(rb, self.g, self.g_r)


def noise_free_barycentre(key: str, r_values: np.ndarray, n_psi: int = 16) -> np.ndarray | None:
    """Azimuth-averaged radius of the barycentre of the expected charges (straight light and absorb only)."""
    cfg = t3.config(key)
    if cfg.optics.scattering or cfg.optics.reflection:
        return None
    sx, sy = sensor_positions(cfg.geometry)
    out = []
    psi = (np.arange(n_psi) + 0.5) * (np.pi / cfg.geometry.n_sensors) / n_psi  # from a sensor centre to a gap centre
    for r in r_values:
        radii = []
        for a in psi:
            mu = t3.expected_charge(r * math.cos(a), r * math.sin(a), 1.0, cfg, n_nodes=64)
            radii.append(math.hypot(mu @ sx / mu.sum(), mu @ sy / mu.sum()))
        out.append(np.mean(radii))
    return np.asarray(out)


def evaluate(results: dict[str, Any]) -> dict[str, Any]:
    """Apply the estimators to the main data sets of all configurations."""
    out: dict[str, Any] = {}
    for key in t3.CONFIGS:
        cfg = t3.config(key)
        light_yield = cfg.optics.light_yield
        cal = Calibration(summaries(t3.calib(key)), light_yield, cfg.geometry.n_sensors)
        ev = summaries(t3.main(key))
        rel = ev["q"] / cal.k / ev["energy"] - 1  # naive energy, relative residual
        rel_mean_k = ev["q"] / cal.k_mean / ev["energy"] - 1  # the same with the mean-based constant
        f = cal.collected_fraction(ev["r"], ev["psi"])
        rel_oracle = ev["q"] / (light_yield * f) / ev["energy"] - 1
        poisson = 1 / np.sqrt(light_yield * ev["energy"] * f)
        dr = ev["rb"] - ev["r"]
        err2d = np.hypot(ev["bx"] - ev["x"], ev["by"] - ev["y"])
        r_cal = cal.radius(ev["rb"])
        scale = np.where(ev["rb"] > 0, r_cal / ev["rb"], 0.0)
        err2d_cal = np.hypot(scale * ev["bx"] - ev["x"], scale * ev["by"] - ev["y"])
        q16, bias, q84 = binned_quantiles(ev["r"], rel, R_EDGES, (16, 50, 84))  # robust: median, central 68 %
        resolution = (q84 - q16) / 2
        _, _, count = t3.binned_mean(ev["r"], rel, R_EDGES)
        bias_mean, _, _ = t3.binned_mean(ev["r"], rel_mean_k, R_EDGES)
        bias_mean_sq, _, _ = t3.binned_mean(ev["r"], rel_mean_k**2, R_EDGES)
        resolution_std = np.sqrt(np.maximum(bias_mean_sq - bias_mean**2, 0))
        rb_bias, _, _ = t3.binned_mean(ev["r"], dr, R_EDGES)
        rb_sq, _, _ = t3.binned_mean(ev["r"], dr**2, R_EDGES)
        e2_sq, _, _ = t3.binned_mean(ev["r"], err2d**2, R_EDGES)
        out[key] = {
            "k_pe_per_MeV": float(cal.k),
            "k_over_Y": float(cal.k / light_yield),
            "k_mean_pe_per_MeV": float(cal.k_mean),
            "k_mean_over_Y": float(cal.k_mean / light_yield),
            "estimator_definition": "k = median(Q/E); bias = median of E_hat/E - 1 per bin; resolution = half of "
            "the central 68 % interval per bin; the *_mean_* / *_std_* keys use k = sum(Q)/sum(E), mean and std",
            "energy_bias_per_r_bin": bias,
            "energy_resolution_per_r_bin": resolution,
            "energy_bias_mean_per_r_bin": bias_mean,
            "energy_resolution_std_per_r_bin": resolution_std,
            "barycentre_radial_bias_per_r_bin": rb_bias,
            "barycentre_radial_resolution_per_r_bin": np.sqrt(np.maximum(rb_sq - rb_bias**2, 0)),
            "barycentre_rms_2d_error_per_r_bin": np.sqrt(e2_sq),
            "events_per_r_bin": count,
            "energy_rms_naive": float(np.sqrt(np.mean(rel**2))),
            "energy_bias_mean_naive": float(rel.mean()),
            "energy_rms_vertex_known": float(np.sqrt(np.mean(rel_oracle**2))),
            "energy_poisson_floor_rms": float(np.sqrt(np.mean(poisson**2))),
            "energy_bias_in_bin": {name: float(bias[i]) for name, i in NAMED_BINS.items()},
            "energy_resolution_in_bin": {name: float(resolution[i]) for name, i in NAMED_BINS.items()},
            "energy_bias_mean_in_bin": {name: float(bias_mean[i]) for name, i in NAMED_BINS.items()},
            "energy_resolution_std_in_bin": {name: float(resolution_std[i]) for name, i in NAMED_BINS.items()},
            "energy_rms_naive_r_below_0.9m": float(np.sqrt(np.mean(rel[ev["r"] < 0.9] ** 2))),
            "energy_rms_vertex_known_r_below_0.9m": float(np.sqrt(np.mean(rel_oracle[ev["r"] < 0.9] ** 2))),
            "energy_poisson_floor_rms_r_below_0.9m": float(np.sqrt(np.mean(poisson[ev["r"] < 0.9] ** 2))),
            "centre_calibration_wall_bias": float((1 + bias[-1]) / (1 + bias[0]) - 1),
            "barycentre_rms_2d_error": float(np.sqrt(np.mean(err2d**2))),
            "barycentre_rms_2d_error_calibrated": float(np.sqrt(np.mean(err2d_cal**2))),
            "barycentre_rms_2d_error_calibrated_r_below_0.9m": float(np.sqrt(np.mean(err2d_cal[ev["r"] < 0.9] ** 2))),
            "barycentre_radial_bias_in_bin": {name: float(rb_bias[i]) for name, i in NAMED_BINS.items()},
            "barycentre_radial_resolution_in_bin": {
                name: float(np.sqrt(max(rb_sq[i] - rb_bias[i] ** 2, 0))) for name, i in NAMED_BINS.items()
            },
            "barycentre_mean_radius_over_r_in_bin_0.45_to_0.5m": float(1 + rb_bias[9] / centres(R_EDGES)[9]),
            "fraction_of_events_r_above_0.9m": float(np.mean(ev["r"] > 0.9)),
            "barycentre_response_curve": {"r": cal.g_r, "mean_rb": cal.g},
            "vertex_known_map_min_events_per_bin": int(cal.map_counts.min()),
        }
    results["estimators"] = out
    return out


def energy_figure(out: dict[str, Any]) -> None:
    """Relative bias and resolution of the naive energy vs vertex radius."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    rc = centres(R_EDGES)
    for key in t3.CONFIGS:
        st = t3.CONFIGS[key]
        ax1.plot(rc, 100 * out[key]["energy_bias_per_r_bin"], color=st["color"], ls=st["ls"], zorder=3)
        ax2.plot(rc, 100 * out[key]["energy_resolution_per_r_bin"], color=st["color"], ls=st["ls"], zorder=3)
    ax1.axhline(0, color=C["ink2"], lw=0.8)
    ax1.set_ylabel(r"Bias: median of $\hat E / E - 1$ [%]")
    ax1.set_title(r"(a) Bias of $\hat E = Q / k$", loc="left")
    ax2.set_ylabel(r"Resolution: 68 % half-width of $\hat E / E$ [%]")
    ax2.set_title("(b) Resolution (pooled over the energy prior)", loc="left")
    ax2.set_yscale("log")
    ax2.set_ylim(1, 100)
    ax2.yaxis.set_minor_formatter(NullFormatter())
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_xlabel("Vertex radius $r_0$ [m]")
    figure_legend(fig, handles=config_handles(list(t3.CONFIGS)), where="bottom", ncol=3)  # the top holds the title
    fig.suptitle(
        r"Naive energy from the total charge with one calibration constant $k = \mathrm{median}(Q / E)$ "
        "per configuration",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    savefig(fig, "t3_4_energy_estimator")


def barycentre_figure(out: dict[str, Any], results: dict[str, Any]) -> None:
    """Radial bias and resolution of the charge barycentre vs vertex radius."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    rc = centres(R_EDGES)
    rf = np.linspace(0.0, 0.99, 34)
    analytic = {}
    for key in ("straight", "absorb"):
        g = noise_free_barycentre(key, rf)
        assert g is not None
        analytic[key] = {"r": rf, "noise_free_barycentre_radius": g}
        ax1.plot(rf, g - rf, **t3.ANALYTIC)
    for key in t3.CONFIGS:
        st = t3.CONFIGS[key]
        ax1.plot(rc, out[key]["barycentre_radial_bias_per_r_bin"], color=st["color"], ls=st["ls"], zorder=3)
        ax2.plot(rc, out[key]["barycentre_radial_resolution_per_r_bin"], color=st["color"], ls=st["ls"], zorder=3)
    ax1.plot([0, 1], [0, -1], color=C["ink2"], lw=0.8, ls=":")
    ax1.text(
        0.72,
        -0.76,
        r"$\hat r = 0$ (no information)",
        fontsize=8.8,
        color=C["ink2"],
        rotation=-25,
        rotation_mode="anchor",
        ha="center",
        va="top",
    )
    ax1.axhline(0, color=C["ink2"], lw=0.8)
    ax1.set_ylabel(r"Radial bias $\langle \hat r - r \rangle$ [m]")
    ax1.set_title("(a) Bias of the barycentre radius", loc="left")
    ax1.set_ylim(-1.0, 0.1)
    ax2.set_ylabel(r"Radial resolution: std of $\hat r - r$ [m]")
    ax2.set_title("(b) Resolution of the barycentre radius", loc="left")
    ax2.set_yscale("log")
    ax2.set_ylim(0.01, 1.0)
    ax2.yaxis.set_minor_formatter(NullFormatter())
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_xlabel("Vertex radius $r_0$ [m]")
    handles = [
        *config_handles(list(t3.CONFIGS)),
        Line2D(
            [],
            [],
            **{k: v for k, v in t3.ANALYTIC.items() if k != "zorder"},
            label="barycentre of the expected charges (straight, absorb)",
        ),
    ]
    figure_legend(fig, handles=handles, where="bottom", ncol=3)  # the top holds the title
    fig.suptitle(
        r"Charge barycentre $\sum_j n_j \vec{x}_j / \sum_j n_j$ as the vertex estimate", x=0.01, ha="left", fontsize=11
    )
    savefig(fig, "t3_4_barycentre")
    results["noise_free_barycentre"] = analytic


def summary_figure(out: dict[str, Any]) -> None:
    """How hard is the inverse problem: energy and vertex resolution of the naive estimators per configuration."""
    keys = list(t3.CONFIGS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 3.7), sharey=True)
    y = np.arange(len(keys))
    energy_variants = [
        ("energy_rms_naive", "o", "one constant: $Q / k$"),
        ("energy_rms_vertex_known", "s", "vertex known: $Q / (Y f(r, \\psi))$"),
        ("energy_poisson_floor_rms", "^", "Poisson floor $1/\\sqrt{Y E f}$"),
    ]
    vertex_variants = [
        ("barycentre_rms_2d_error", "o", "barycentre"),
        ("barycentre_rms_2d_error_calibrated", "s", "barycentre, radius calibrated"),
    ]
    # the legends go into the empty part of each panel: lower right of (a), right of the two middle rows of (b)
    placement = {ax1: {"loc": "lower right"}, ax2: {"loc": "center right", "bbox_to_anchor": (0.985, 0.5)}}
    for ax, variants, scale, x_max in ((ax1, energy_variants, 100.0, 24.0), (ax2, vertex_variants, 100.0, 52.0)):
        for k, key in enumerate(keys):
            st = t3.CONFIGS[key]
            values = [scale * out[key][name] for name, _, _ in variants]
            ax.plot([min(values), max(values)], [k, k], color=C["grid"], lw=1.2, zorder=1)
            for (_name, marker, _label), value in zip(variants, values, strict=True):
                ax.plot(value, k, marker, color=st["color"], ms=7.5, mec="white", mew=0.8, zorder=3)
        ax.set_xlim(0, x_max)
        handles = [Line2D([], [], ls="", marker=m, color=C["ink2"], ms=7, label=lab) for _, m, lab in variants]
        legend_inside(ax, handles=handles, ncol=1, **placement[ax])
    ax1.set_xlabel(r"Energy error: RMS of $\hat E / E - 1$ [%]")
    ax2.set_xlabel(r"Vertex error: RMS of $|\hat{\vec x} - \vec x|$ [cm]")
    ax1.set_yticks(y, [t3.CONFIGS[k]["label"] for k in keys])
    ax1.invert_yaxis()
    ax1.set_title("(a) Energy from the total charge", loc="left")
    ax2.set_title("(b) Vertex from the charge barycentre", loc="left")
    fig.suptitle(
        "How hard is the inverse problem? Naive estimators over the whole prior (100 000 events each)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    savefig(fig, "t3_4_inverse_problem")


def main() -> None:
    """Make the T3.4 figures and store the numbers."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    style()
    results: dict[str, Any] = {}
    out = evaluate(results)
    energy_figure(out)
    barycentre_figure(out, results)
    summary_figure(out)
    results["r_bins_m"] = R_EDGES
    results["calibration"] = {
        "sample": f"{t3.N_CALIB} events per configuration from the prior, seed {t3.CALIB_SEED}",
        "energy_constant": "k = median(Q / E) over the calibration sample (k_mean = sum(Q) / sum(E) for comparison)",
        "vertex_known_map": f"{N_R2_MAP} bins of r^2 x {N_PSI_MAP} bins of the azimuth folded into [0, pi/M]",
        "barycentre_response": "mean barycentre radius in 40 bins of r^2, made monotonic, inverted by interpolation",
    }
    save_results("T3", {"T3.4": results})
    print(
        "T3.4 k/Y | energy RMS: naive / vertex known / Poisson | bias centre / wall | barycentre RMS raw / calibrated"
    )
    for key, v in out.items():
        bias = v["energy_bias_in_bin"]
        energy = " / ".join(
            f"{100 * v[name]:5.2f}"
            for name in ("energy_rms_naive", "energy_rms_vertex_known", "energy_poisson_floor_rms")
        )
        vertex = " / ".join(
            f"{100 * v[name]:5.1f}" for name in ("barycentre_rms_2d_error", "barycentre_rms_2d_error_calibrated")
        )
        edges = f"{100 * bias['0_to_0.05m']:+6.2f} / {100 * bias['0.99_to_1m']:+6.2f}"
        print(f"  {key:15s} {v['k_over_Y']:.4f} | {energy} % | {edges} % | {vertex} cm")


if __name__ == "__main__":
    main()
