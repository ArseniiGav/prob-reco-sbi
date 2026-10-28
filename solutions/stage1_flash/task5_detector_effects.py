"""T5.4 (bonus): detector effects (time jitter, detection efficiency, dark noise) and the features they hurt.

Settings (one effect changed at a time on top of the default full model: sigma_t = 1 ns, epsilon = 1, no noise):
    time jitter          sigma_t = 0, 1, 3 ns
    detection efficiency epsilon = 1, 0.5, 0.25
    dark noise           0, 1e5, 1e6, 1e7 Hz per sensor (0, 0.01, 0.1, 1 noise hits per sensor in the 100 ns window)

Feature groups (32 numbers each, empty sensors set to 0 as in ``network_input``):
    charges              log(1 + n_j)
    relative mean times  tau_j - tau_bar                       (tau_bar: charge-weighted mean of the tau_j)
    relative first times t_first_j - t_bar_first               (t_bar_first: charge-weighted mean of the first times)
    network input        charges and relative mean times together (64 numbers, what the NPE will see)

How much does each group tell about the vertex? For a vertex theta_0 we simulate events at theta_0 and at
theta_0 +- delta (delta = 5 cm along x or y) and compute the linear Fisher information I = mu'^T Sigma^-1 mu'
(mu' = d<features>/dx from the shifted samples, Sigma = covariance of the features at theta_0; corrected for the
noise of mu' and of Sigma^-1). sigma = 1 / sqrt(I) is the vertex resolution an ideal linear fit of that feature group
would reach near theta_0 (a Cramer-Rao bound for the Gaussian approximation; a learned posterior can do better if
the features are not Gaussian).
Two vertices: (0.1, 0) m and (0.6, 0) m (radial = x, tangential = y); E = 2 MeV.

Run from the repository root::

    python solutions/stage1_flash/task5_detector_effects.py

Figures: t5_4_detector_effects (resolution per feature group and setting), t5_4_feature_distributions (how each
effect changes one feature). Numbers: solutions/stage1_flash/results/T5.json (key T5.4).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from common import C, dataset, figure_legend, legend_inside, save_results, savefig, style
from event_display import barycentre

from prob_reco_sbi.flash import SimResult

E = 2.0  # MeV
DELTA = 0.05  # m, vertex shift for the derivatives
N0, N_SHIFT = 20_000, 10_000
N_BOOT = 20
# Not the exact centre: there, every wall point is at the same distance R, so without jitter the first times of all
# sensors would be identical in every event (zero variance, an infinite and numerically useless bound).
VERTICES = {"r = 0.1 m": (0.1, 0.0), "r = 0.6 m": (0.6, 0.0)}
DIRECTIONS = {"r = 0.1 m": ("x", "y"), "r = 0.6 m": ("x", "y")}
SETTINGS = {
    "base": {},
    "jitter 0 ns": {"time_jitter": 0.0},
    "jitter 3 ns": {"time_jitter": 3.0},
    "efficiency 0.5": {"detection_efficiency": 0.5},
    "efficiency 0.25": {"detection_efficiency": 0.25},
    "dark 1e5 Hz": {"dark_rate": 1e5},
    "dark 1e6 Hz": {"dark_rate": 1e6},
    "dark 1e7 Hz": {"dark_rate": 1e7},
}
EFFECTS = {
    "Time jitter $\\sigma_t$": (["jitter 0 ns", "base", "jitter 3 ns"], ["0 ns", "1 ns", "3 ns"]),
    "Detection efficiency $\\varepsilon$": (["base", "efficiency 0.5", "efficiency 0.25"], ["1", "0.5", "0.25"]),
    "Dark-noise rate per sensor": (
        ["base", "dark 1e5 Hz", "dark 1e6 Hz", "dark 1e7 Hz"],
        ["0", "$10^5$ Hz", "$10^6$ Hz", "$10^7$ Hz"],
    ),
}
#: One colour and marker per feature group in both figures: charges darkgreen, mean times royalblue, first times
#: peru, the network input black (dashed); the settings of one effect differ by line style only.
GROUPS = {
    "charges": ("charges $\\log(1 + n_j)$", C["green"], "s", "-"),
    "rel_mean": ("relative mean times", C["blue"], "o", "-"),
    "rel_first": ("relative first times", C["orange"], "v", "-"),
    "network": ("charges + mean times (network input)", C["ink"], "D", "--"),
}


def features(data: SimResult) -> dict[str, np.ndarray]:
    """Return the feature groups of every event (see the module docstring)."""
    n = data.response.charge.astype(np.float64)
    hit = n > 0
    w = np.where(hit, n, 0.0)
    q = np.maximum(w.sum(1, keepdims=True), 1.0)
    out = {"charges": np.log1p(n)}
    for name, t in (("rel_mean", data.response.time_mean), ("rel_first", data.response.time_first)):
        t = np.where(hit, t.astype(np.float64), 0.0)
        ref = (w * t).sum(1, keepdims=True) / q
        out[name] = np.where(hit, t - ref, 0.0)
    out["network"] = np.concatenate([out["charges"], out["rel_mean"]], axis=1)
    return out


def linear_fisher(f0: np.ndarray, fp: np.ndarray, fm: np.ndarray, delta: float) -> float:
    """Linear Fisher information mu'^T Sigma^-1 mu' with the bias corrections for estimated mu' and Sigma^-1."""
    n0, d = f0.shape
    keep = f0.std(0) > 0  # features that never vary carry no information (and make Sigma singular)
    f0, fp, fm = f0[:, keep], fp[:, keep], fm[:, keep]
    d = int(keep.sum())
    cov = np.cov(f0, rowvar=False)
    cov += 1e-12 * np.trace(cov) / d * np.eye(d)  # guard against exact singularity; negligible otherwise
    dmu = (fp.mean(0) - fm.mean(0)) / (2 * delta)
    hartlap = (n0 - d - 2) / (n0 - 1)  # unbiased Sigma^-1 from an estimated Sigma
    noise = d * (1 / len(fp) + 1 / len(fm)) / (4 * delta**2)  # what the noise of mu' adds on average
    return float(hartlap * dmu @ np.linalg.solve(cov, dmu) - noise)


def sigma_cm(info: float) -> float:
    """Resolution 1 / sqrt(I) in cm (inf if the information is not positive)."""
    return float(100 / np.sqrt(info)) if info > 0 else float("inf")


def sample(setting: str, vertex: str, shift: str, sign: int, n: int, seed: int) -> SimResult:
    """Events at a (shifted) vertex for one detector setting (cached)."""
    x, y = VERTICES[vertex]
    if shift == "x":
        x += sign * DELTA
    elif shift == "y":
        y += sign * DELTA
    tag = f"{setting.replace(' ', '_')}_{vertex.replace(' ', '').replace('=', '').replace('.', 'p')}_{shift}{sign:+d}"
    return dataset(
        f"t5_4_{tag}",
        n,
        "full",
        seed=seed,
        fixed_vertex=(round(x, 6), round(y, 6)),
        fixed_energy=E,
        **SETTINGS[setting],
    )


def main() -> None:
    """Simulate every setting and vertex, compute the bounds, draw the figures and save the numbers."""
    style()
    rng = np.random.default_rng(54)
    results: dict[str, object] = {}
    seed = 5400
    store: dict[tuple[str, str], dict[str, object]] = {}
    for setting in SETTINGS:
        for vertex in VERTICES:
            seed += 1
            ev0 = sample(setting, vertex, "none", 0, N0, seed)
            f0 = features(ev0)
            entry: dict[str, object] = {}
            for direction in DIRECTIONS[vertex]:
                seed += 2
                evp = sample(setting, vertex, direction, +1, N_SHIFT, seed)
                evm = sample(setting, vertex, direction, -1, N_SHIFT, seed + 1)
                fp, fm = features(evp), features(evm)
                res = {}
                for g in GROUPS:
                    info = linear_fisher(f0[g], fp[g], fm[g], DELTA)
                    boot = []
                    for _ in range(N_BOOT):
                        i0 = rng.integers(0, N0, N0)
                        ip = rng.integers(0, N_SHIFT, N_SHIFT)
                        im = rng.integers(0, N_SHIFT, N_SHIFT)
                        boot.append(sigma_cm(linear_fisher(f0[g][i0], fp[g][ip], fm[g][im], DELTA)))
                    res[g] = {"sigma_cm": sigma_cm(info), "sigma_cm_err": float(np.std(boot, ddof=1))}
                # a simple estimator for comparison: the charge barycentre, sigma(b) / (d<b>/dx)
                axis = 0 if direction == "x" else 1
                b0 = barycentre(ev0.response.charge, ev0.detector)[axis]
                bp = barycentre(evp.response.charge, evp.detector)[axis]
                bm = barycentre(evm.response.charge, evm.detector)[axis]
                slope = (bp.mean() - bm.mean()) / (2 * DELTA)
                res["barycentre_estimator"] = {"sigma_cm": float(100 * b0.std(ddof=1) / slope), "slope": float(slope)}
                entry[direction] = res
            q = ev0.response.charge.sum(1).astype(np.float64)
            n = ev0.response.charge
            entry["Q_mean_pe"] = float(q.mean())
            # E from Q with the pedestal of the known mean noise subtracted: sigma_E / E = sigma_Q / (<Q> - <noise>)
            entry["energy_resolution_from_Q"] = float(q.std(ddof=1) / (q.mean() - ev0.photons.n_noise.mean()))
            entry["mean_noise_hits_per_event"] = float(ev0.photons.n_noise.mean())
            entry["empty_sensor_fraction"] = float((n == 0).mean())
            store[(setting, vertex)] = {"features": f0, "data": ev0}
            results[f"{setting} | {vertex}"] = entry
            print(
                f"T5.4 {setting:16s} {vertex:10s} "
                + "  ".join(f"{g}: {entry['x'][g]['sigma_cm']:6.2f} cm" for g in GROUPS)
                + f"  sigma_E/E = {entry['energy_resolution_from_Q']:.4f}"
            )

    # ---------------------------------------------------------------------------------- figure: resolution
    panels = [
        ("r = 0.1 m", "x", "r = 0.1 m, radial"),
        ("r = 0.6 m", "x", "r = 0.6 m, radial"),
        ("r = 0.6 m", "y", "r = 0.6 m, tangential"),
    ]
    fig, axes = plt.subplots(len(panels), 3, figsize=(12.5, 9.6), sharey=True)
    for row, (vertex, direction, label) in enumerate(panels):
        for col, (effect, (keys, ticks)) in enumerate(EFFECTS.items()):
            a = axes[row, col]
            xs = np.arange(len(keys))
            for g, (glabel, colour, marker, ls) in GROUPS.items():
                vals = [results[f"{k} | {vertex}"][direction][g]["sigma_cm"] for k in keys]
                errs = [results[f"{k} | {vertex}"][direction][g]["sigma_cm_err"] for k in keys]
                a.errorbar(
                    xs, vals, yerr=errs, color=colour, marker=marker, ms=5, ls=ls, lw=1.4, capsize=2, label=glabel
                )
            a.set_xticks(xs, ticks)
            a.set_xlim(-0.3, len(keys) - 0.7)
            for k, key in enumerate(keys):  # values too small to read on the linear axis are written out
                v = results[f"{key} | {vertex}"][direction]["rel_first"]["sigma_cm"]
                if v < 0.5:
                    a.annotate(
                        f"{v:.2f} cm",
                        (k, v),
                        xytext=(8, 9),
                        textcoords="offset points",
                        fontsize=8.5,
                        color=C["orange"],
                    )
            if row == 0:
                a.set_title(effect, fontsize=11)
            if col == 0:
                a.set_ylabel(f"Vertex resolution [cm]\n{label}")
    axes[0, 0].set_ylim(0.0, 8.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure_legend(
        fig,
        handles,
        labels,
        ncol=4,
        fontsize=9.5,
        title=f"Linear Fisher bound 1/sqrt(I) near the vertex, E = {E:.0f} MeV",
        title_fontsize=10,
    )
    savefig(fig, "t5_4_detector_effects")

    # ---------------------------------------------------------------------------------- figure: distributions
    # one feature per panel in the colour of its group (GROUPS); the settings of the effect differ by line style,
    # the default setting solid; the two log panels share the floor 1e-4
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    near, far = 0, 16
    vertex = "r = 0.6 m"
    a = axes[0]
    bins = np.linspace(-9, 3, 121)
    colour = GROUPS["rel_first"][1]
    for key, lab, ls in (("jitter 0 ns", "0 ns", ":"), ("base", "1 ns", "-"), ("jitter 3 ns", "3 ns", "--")):
        v = store[(key, vertex)]["features"]["rel_first"][:, near]
        a.hist(
            v,
            bins=bins,
            histtype="step",
            lw=1.5,
            alpha=0.6,
            ls=ls,
            color=colour,
            density=True,
            label=f"$\\sigma_t$ = {lab}",
        )
    a.set_xlabel(f"Relative first time of sensor {near} (nearest) [ns]")
    a.set_ylabel("Probability density [1/ns]")
    a.set_yscale("log")
    a.set_ylim(1e-4, 30.0)
    legend_inside(a, loc="upper left")
    a = axes[1]
    bins = np.linspace(-5, 50, 111)
    colour = GROUPS["rel_mean"][1]
    for key, lab, ls in (
        ("base", "no noise", "-"),
        ("dark 1e5 Hz", "$10^5$ Hz", "--"),
        ("dark 1e6 Hz", "$10^6$ Hz", ":"),
        ("dark 1e7 Hz", "$10^7$ Hz", "-."),
    ):
        v = store[(key, vertex)]["features"]["rel_mean"][:, far]
        a.hist(v, bins=bins, histtype="step", lw=1.5, alpha=0.6, ls=ls, color=colour, density=True, label=lab)
    a.set_xlabel(f"Relative mean time of sensor {far} (farthest) [ns]")
    a.set_ylabel("Probability density [1/ns]")
    a.set_yscale("log")
    a.set_ylim(1e-4, 30.0)
    legend_inside(a, loc="upper right")
    a = axes[2]
    colour = GROUPS["charges"][1]
    for key, lab, ls in (("base", "1", "-"), ("efficiency 0.5", "0.5", "--"), ("efficiency 0.25", "0.25", ":")):
        n = store[(key, vertex)]["data"].response.charge[:, far]
        k = np.arange(0, n.max() + 2)
        a.step(
            k,
            np.bincount(n, minlength=len(k))[: len(k)] / len(n),
            where="mid",
            lw=1.5,
            alpha=0.6,
            ls=ls,
            color=colour,
            label=f"$\\varepsilon$ = {lab}",
        )
    a.set_xlabel(f"Charge of sensor {far} (farthest) [p.e.]")
    a.set_ylabel("Fraction of events")
    a.set_xlim(-0.5, 25)
    legend_inside(a, loc="upper right")
    fig.suptitle(f"One feature per effect, E = {E:.0f} MeV at (0.6, 0) m, {N0:,} events each", fontsize=11)
    savefig(fig, "t5_4_feature_distributions")

    # numbers of the diagnostic figure
    diag = {}
    for key in SETTINGS:
        fz = store[(key, vertex)]["features"]
        diag[key] = {
            "rel_first_near_sd_ns": float(fz["rel_first"][:, near].std(ddof=1)),
            "rel_mean_far_sd_ns": float(fz["rel_mean"][:, far].std(ddof=1)),
            "rel_mean_far_frac_beyond_10ns": float((np.abs(fz["rel_mean"][:, far]) > 10).mean()),
            "charge_far_mean_pe": float(store[(key, vertex)]["data"].response.charge[:, far].mean()),
            "charge_far_frac_zero": float((store[(key, vertex)]["data"].response.charge[:, far] == 0).mean()),
        }
    payload = {
        "energy_MeV": E,
        "delta_m": DELTA,
        "n_events_centre_point": N0,
        "n_events_shifted_points": N_SHIFT,
        "n_bootstrap": N_BOOT,
        "settings": SETTINGS,
        "vertices": VERTICES,
        "results": results,
        "distributions_r0p6": diag,
    }
    path = save_results("T5", {"T5.4": payload})
    print(f"numbers -> {path}")


if __name__ == "__main__":
    main()
