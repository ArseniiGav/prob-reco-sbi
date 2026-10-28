"""T4 event displays: one event (T4.1, with a gallery), timing (T4.2), photon paths (T4.3), presets (T4.4).

Run from the repository root::

    python solutions/stage1_flash/task4_event_displays.py

Figures (solutions/stage1_flash/figures, PDF + PNG):
    t4_1_event_display   one event: sensors coloured by charge, true vertex, charge barycentre
    t4_1_gallery         four instructive events: centre, mid-radius, near the wall facing a sensor, low energy
    t4_2_timing          charge display, relative-time display, charge and times per sensor
    t4_3_photon_paths    150 photon paths of one event with every interaction point marked
    t4_4_presets         the same theta under the four presets (same colour scale)
    t4_4_profiles        mean charge and relative time per sensor for the four presets (2000 events each)
Numbers: solutions/stage1_flash/results/T4.json (keys T4.1 ... T4.4).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from common import CHARGE_CMAP, PRESET_COLOURS, C, dataset, figure_legend, legend_inside, save_results, savefig, style
from event_display import (
    TIME_CMAP,
    barycentre,
    charge_colorbar,
    draw_barycentre,
    draw_detector,
    draw_tracks,
    draw_vertex,
    event_title,
    marker_handles,
    path_handles,
    relative_times,
    sensor_distance,
    sensor_patch_handle,
    vertex_handles,
)
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from prob_reco_sbi.flash import Interaction, SimResult, preset, simulate

PRESETS = ("straight", "absorb", "absorb-scatter", "full")
PRESET_STYLE = {  # the colour per preset is shared with T0.2 and T3 (common.PRESET_COLOURS)
    "straight": {"color": PRESET_COLOURS["straight"], "ls": ":", "marker": "o"},
    "absorb": {"color": PRESET_COLOURS["absorb"], "ls": "--", "marker": "s"},
    "absorb-scatter": {"color": PRESET_COLOURS["absorb-scatter"], "ls": "-.", "marker": "^"},
    "full": {"color": PRESET_COLOURS["full"], "ls": "-", "marker": "D"},
}
MAIN_EVENT = {"x": 0.45, "y": 0.25, "energy": 2.0, "t0": 3.0}  # r = 0.51 m, used in T4.1 and T4.2
PRESET_EVENT = {"x": 0.6, "y": 0.3, "energy": 2.0, "t0": 0.0}  # r = 0.67 m, T4.4: the example call of TASKS.md
PRESET_SEED = 10  # as in TASKS.md: simulate(0.6, 0.3, 2.0, t0=0.0, config=preset(name), seed=10, save_tracks=1)
SENSOR_TICKS = [0, 8, 16, 24, 31]


def one_event(
    x: float,
    y: float,
    energy: float,
    t0: float,
    *,
    seed: int,
    name: str = "full",
    save_tracks: int = 0,
    track_photons: int | None = 100,
    **overrides: object,
) -> SimResult:
    """Simulate one event at a given theta (on the CPU, deterministic for a given seed)."""
    config = preset(name, **overrides)
    return simulate(
        x,
        y,
        energy,
        t0,
        config=config,
        seed=seed,
        device="cpu",
        save_tracks=save_tracks,
        track_photons=track_photons,
    )


def summary(data: SimResult, i: int = 0) -> dict[str, object]:
    """Numbers of one event: theta, total charge, hit sensors, barycentre, photon bookkeeping."""
    det = data.detector
    n = data.response.charge[i]
    bx, by = barycentre(n, det)
    x0, y0 = float(data.truth.x[i]), float(data.truth.y[i])
    ph = data.photons
    emitted = int(ph.n_emitted[i])
    return {
        "theta": {"x0_m": x0, "y0_m": y0, "r_m": float(np.hypot(x0, y0)), "energy_MeV": float(data.truth.energy[i])},
        "t0_ns": float(data.truth.t0[i]),
        "Q_pe": int(n.sum()),
        "max_sensor_charge_pe": int(n.max()),
        "max_sensor": int(n.argmax()),
        "max_sensor_fraction_of_Q": float(n.max() / n.sum()),
        "n_hit_sensors": int((n > 0).sum()),
        "n_empty_sensors": int((n == 0).sum()),
        "barycentre_m": [float(bx), float(by)],
        "barycentre_r_m": float(np.hypot(bx, by)),
        "barycentre_offset_cm": float(100 * np.hypot(bx - x0, by - y0)),
        "photons": {
            "emitted": emitted,
            "detected": int(ph.n_detected[i]),
            "absorbed_bulk": int(ph.n_absorbed_bulk[i]),
            "absorbed_wall": int(ph.n_absorbed_wall[i]),
            "lost": int(ph.n_lost[i]),
            "detected_direct": int(ph.n_detected_direct[i]),
            "detected_scattered": int(ph.n_detected_scattered[i]),
            "detected_reflected": int(ph.n_detected_reflected[i]),
            "collected_fraction": float(ph.n_detected[i] / emitted),
        },
    }


def display(
    ax: object, data: SimResult, i: int = 0, *, norm: Normalize | None = None, labels: bool = True
) -> Normalize:
    """Charge display of event ``i``: sensors by charge, true vertex, barycentre; returns the colour scale used."""
    det = data.detector
    n = data.response.charge[i].astype(float)
    norm = norm or Normalize(0.0, max(n.max(), 1.0))
    draw_detector(ax, det, n, norm=norm, labels=(0, 8, 16, 24) if labels else ())
    bx, by = barycentre(n, det)
    draw_vertex(ax, float(data.truth.x[i]), float(data.truth.y[i]))
    draw_barycentre(ax, float(bx), float(by))
    return norm


def title_of(data: SimResult, i: int = 0) -> str:
    """E, r and Q of event ``i``."""
    r = float(np.hypot(data.truth.x[i], data.truth.y[i]))
    return event_title(float(data.truth.energy[i]), r, float(data.response.charge[i].sum()))


# ----------------------------------------------------------------------------------------------------------------------
# T4.1  one event, and a gallery of four instructive events


def t4_1() -> dict[str, object]:
    """T4.1: one event display and a gallery of four instructive events."""
    ev = one_event(**MAIN_EVENT, seed=401)
    s = summary(ev)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    norm = display(ax, ev)
    ax.set_title(title_of(ev))
    charge_colorbar(fig, ax, norm, shrink=0.62, pad=0.02)
    star, cross = vertex_handles()
    cross.set_label(f"charge barycentre (naive vertex), {s['barycentre_offset_cm']:.0f} cm from the true vertex")
    numbers = Line2D([], [], ls="none", marker="$8$", ms=7, color=C["ink2"], label="numbers: sensor index j")
    figure_legend(fig, [star, cross, sensor_patch_handle(), numbers], where="bottom", ncol=1)
    savefig(fig, "t4_1_event_display")

    gallery = [
        ("centre", {"x": 0.0, "y": 0.0, "energy": 2.0, "t0": 3.0}),
        ("mid-radius", {"x": -0.5 * np.cos(np.pi / 4), "y": 0.5 * np.sin(np.pi / 4), "energy": 2.0, "t0": 3.0}),
        (
            "near the wall, facing sensor 4",
            {"x": 0.92 * np.cos(np.pi / 4), "y": 0.92 * np.sin(np.pi / 4), "energy": 2.0, "t0": 3.0},
        ),
        ("low energy", {"x": 0.6 * np.cos(4.4), "y": 0.6 * np.sin(4.4), "energy": 0.5, "t0": 3.0}),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.3))
    out = {"main_event": s, "gallery": {}}
    for k, (ax, (label, theta)) in enumerate(zip(axes, gallery, strict=True)):
        ev_k = one_event(**theta, seed=411 + k)
        # square-root colour scale: the few very bright sensors of a near-wall event do not hide the others
        norm = PowerNorm(0.5, vmin=0.0, vmax=float(ev_k.response.charge[0].max()))
        display(ax, ev_k, norm=norm)
        ax.set_title(f"{label}\n{title_of(ev_k)}", fontsize=10)
        bar = charge_colorbar(
            fig,
            ax,
            norm,
            label="Sensor charge [p.e.], square-root scale",
            orientation="horizontal",
            shrink=0.75,
            pad=0.02,
            aspect=22,
        )
        bar.ax.tick_params(labelsize=8.5)
        out["gallery"][label] = summary(ev_k)
    figure_legend(fig, [*vertex_handles(), sensor_patch_handle()], where="bottom", ncol=3)
    savefig(fig, "t4_1_gallery")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# T4.2  timing: a second display coloured by the relative mean time, and charge / times per sensor


def t4_2() -> dict[str, object]:
    """T4.2: the same event with its timing (time display and per-sensor profiles)."""
    ev = one_event(**MAIN_EVENT, seed=401)  # the same event as in T4.1
    det = ev.detector
    n = ev.response.charge[0].astype(float)
    tau = ev.response.time_mean[0].astype(float)
    first = ev.response.time_first[0].astype(float)
    x0, y0, t0 = MAIN_EVENT["x"], MAIN_EVENT["y"], MAIN_EVENT["t0"]
    rel_mean = relative_times(n, tau)
    hit = n > 0
    tau_bar = float((n[hit] * tau[hit]).sum() / n[hit].sum())
    rel_first = np.where(hit, first - tau_bar, np.nan)
    d = sensor_distance(x0, y0, det)
    c_n = ev.config.optics.speed_of_light
    straight = t0 + d / c_n - tau_bar  # straight flight from the vertex to the sensor centre, same reference

    fig = plt.figure(figsize=(13.0, 4.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.45], height_ratios=[1, 1.25])
    a_q = fig.add_subplot(gs[:, 0])
    a_t = fig.add_subplot(gs[:, 1])
    p_q = fig.add_subplot(gs[0, 2])
    p_t = fig.add_subplot(gs[1, 2], sharex=p_q)

    norm_q = display(a_q, ev)
    a_q.set_title("Charge per sensor")
    charge_colorbar(fig, a_q, norm_q, orientation="horizontal", shrink=0.8, pad=0.02, aspect=25)

    lim = float(np.nanmax(np.abs(rel_mean)))
    norm_t = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    draw_detector(a_t, det, rel_mean, norm=norm_t, cmap=TIME_CMAP)
    draw_vertex(a_t, x0, y0)
    a_t.set_title("Relative mean time per sensor")
    bar = charge_colorbar(
        fig,
        a_t,
        norm_t,
        label=r"Relative mean time $\tau_j - \bar\tau$ [ns]",
        cmap=TIME_CMAP,
        orientation="horizontal",
        shrink=0.8,
        pad=0.02,
        aspect=25,
    )
    bar.ax.tick_params(labelsize=8.5)

    j = np.arange(det.n_sensors)
    p_q.bar(j, n, color=CHARGE_CMAP(norm_q(n)), width=0.8, edgecolor=C["ink2"], linewidth=0.4)  # the display's scale
    p_q.set_ylabel("Charge [p.e.]")
    p_q.tick_params(labelbottom=False)
    p_q.set_title(title_of(ev))
    p_t.plot(j[hit], rel_mean[hit], color=C["blue"], marker="o", ms=4, lw=1.3, label=r"mean time $\tau_j - \bar\tau$")
    p_t.plot(j[hit], rel_first[hit], color=C["orange"], marker="v", ms=4.5, lw=1.0, label=r"first time $-\ \bar\tau$")
    p_t.plot(j, straight, color=C["ink"], ls="--", lw=1.2, label=r"straight flight $t_0 + d_j / c_n - \bar\tau$")
    p_t.axhline(0, color=C["grey"], lw=0.8, ls=":")
    p_t.set_ylabel("Time [ns]")
    p_t.set_xlabel("Sensor number $j$")
    p_t.set_xticks(SENSOR_TICKS)
    p_t.set_xlim(-0.8, det.n_sensors - 0.2)
    p_t.set_ylim(-6.5, 9.5)  # room for the legend above the early sensors (upper left is free of data)
    legend_inside(p_t, loc="upper left", ncol=1, handlelength=1.8)
    savefig(fig, "t4_2_timing")

    near, far = int(np.argmin(d)), int(np.argmax(d))
    resid_first = (first - t0 - d / c_n)[hit]
    return {
        "event": summary(ev),
        "tau_bar_ns": tau_bar,
        "relative_mean_time_range_ns": [float(np.nanmin(rel_mean)), float(np.nanmax(rel_mean))],
        "nearest_sensor": {
            "j": near,
            "d_m": float(d[near]),
            "charge": int(n[near]),
            "rel_mean_ns": float(rel_mean[near]),
            "rel_first_ns": float(rel_first[near]),
        },
        "farthest_sensor": {
            "j": far,
            "d_m": float(d[far]),
            "charge": int(n[far]),
            "rel_mean_ns": float(rel_mean[far]),
            "rel_first_ns": float(rel_first[far]),
        },
        "first_minus_straight_ns": {
            "median": float(np.median(resid_first)),
            "min": float(resid_first.min()),
            "max": float(resid_first.max()),
        },
        "corr_rel_mean_vs_distance": float(np.corrcoef(rel_mean[hit], d[hit])[0, 1]),
        "corr_rel_first_vs_distance": float(np.corrcoef(rel_first[hit], d[hit])[0, 1]),
    }


# ----------------------------------------------------------------------------------------------------------------------
# T4.3  photon paths of one event


def t4_3() -> dict[str, object]:
    """T4.3: photon paths of one event, read back from the HDF5 file."""
    # through the HDF5 file, as with `flashsim generate --save-tracks 1 --track-photons 150 --fixed-vertex -0.3 0.35
    # --fixed-energy 2`: the paths are read back from /tracks
    n_tracks = 150
    ev = dataset(
        "t4_3_photon_paths",
        1,
        "full",
        seed=431,
        save_tracks=1,
        track_photons=n_tracks,
        fixed_vertex=(-0.30, 0.35),
        fixed_energy=2.0,
    )
    assert ev.tracks is not None
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    norm = display(ax, ev)
    fates = draw_tracks(ax, ev.tracks, int(ev.truth.event_id[0]), lw=1.0)
    ax.set_title(f"{title_of(ev)}: {n_tracks} of {int(ev.photons.n_emitted[0])} photons drawn")
    charge_colorbar(fig, ax, norm, orientation="horizontal", shrink=0.45, pad=0.01, aspect=25)
    kinds = [
        Interaction.SCATTER,
        Interaction.REFLECT,
        Interaction.DETECT,
        Interaction.ABSORB_BULK,
        Interaction.ABSORB_WALL,
    ]
    handles = [*path_handles(), *marker_handles(kinds), *vertex_handles()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.8)  # beside the disc

    # interaction counts of the drawn photons
    tr = ev.tracks
    kind = tr.kind
    n_scatter_points = int((kind == Interaction.SCATTER).sum())
    n_reflect_points = int((kind == Interaction.REFLECT).sum())
    savefig(fig, "t4_3_photon_paths")
    s = summary(ev)
    drawn = {name: int(v) for name, v in fates.items()}
    return {
        "event": s,
        "drawn_photons": drawn,
        "drawn_detected_fraction": drawn.get("DETECT", 0) / n_tracks,
        "scatter_points_drawn": n_scatter_points,
        "mean_path_length_drawn_m": float(tr.path_length[tr.offsets[1:] - 1].mean()),
        "median_path_length_drawn_m": float(np.median(tr.path_length[tr.offsets[1:] - 1])),
        "reflect_points_drawn": n_reflect_points,
        "event_fate_fractions": {
            "detected": s["photons"]["detected"] / s["photons"]["emitted"],
            "absorbed_bulk": s["photons"]["absorbed_bulk"] / s["photons"]["emitted"],
            "absorbed_wall": s["photons"]["absorbed_wall"] / s["photons"]["emitted"],
        },
    }


# ----------------------------------------------------------------------------------------------------------------------
# T4.4  the same theta under the four presets


def t4_4() -> dict[str, object]:
    """T4.4: one theta under the four presets, single events and mean profiles."""
    # the call of TASKS.md (100 recorded photons per event, of which the first 40 are drawn)
    events = {name: one_event(**PRESET_EVENT, name=name, seed=PRESET_SEED, save_tracks=1) for name in PRESETS}
    vmax = max(float(ev.response.charge[0].max()) for ev in events.values())
    norm = Normalize(0.0, vmax)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.5))
    for ax, (name, ev) in zip(axes, events.items(), strict=True):
        display(ax, ev, norm=norm, labels=False)
        assert ev.tracks is not None
        draw_tracks(ax, ev.tracks, int(ev.truth.event_id[0]), limit=40, lw=0.9, marker_scale=0.9)
        q = int(ev.response.charge[0].sum())
        ax.set_title(f"{name}\nQ = {q} p.e., {int((ev.response.charge[0] == 0).sum())} empty sensors", fontsize=10)
    charge_colorbar(fig, axes, norm, shrink=0.7, pad=0.01, label="Sensor charge [p.e.] (same scale)")
    kinds = [
        Interaction.SCATTER,
        Interaction.REFLECT,
        Interaction.DETECT,
        Interaction.ABSORB_BULK,
        Interaction.ABSORB_WALL,
    ]
    figure_legend(
        fig, [*path_handles(), *marker_handles(kinds), *vertex_handles()], where="bottom", ncol=5, handlelength=1.6
    )
    r = float(np.hypot(PRESET_EVENT["x"], PRESET_EVENT["y"]))
    fig.suptitle(
        f"One theta, four presets: E = {PRESET_EVENT['energy']:.0f} MeV at r = {r:.2f} m (40 photon paths per panel)"
    )
    savefig(fig, "t4_4_presets")

    # mean pattern over many events at the same theta (2000 per preset)
    n_rep = 2000
    rep = {
        name: dataset(
            f"t4_4_{name}_fixed_theta",
            n_rep,
            name,
            seed=442,
            fixed_vertex=(PRESET_EVENT["x"], PRESET_EVENT["y"]),
            fixed_energy=PRESET_EVENT["energy"],
        )
        for name in PRESETS
    }
    fig, (p_q, p_t) = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True)
    det = events["full"].detector
    j = np.arange(det.n_sensors)
    out: dict[str, object] = {}
    for name, data in rep.items():
        n = data.response.charge.astype(float)
        rel = relative_times(n, data.response.time_mean)
        st = PRESET_STYLE[name]
        p_q.plot(j, n.mean(0), color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.3, label=name)
        p_t.plot(j, np.nanmean(rel, 0), color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.3)
        q = n.sum(1)
        ph = data.photons
        det_tot = ph.n_detected.sum()
        out[name] = {
            "single_event_Q_pe": int(events[name].response.charge[0].sum()),
            "mean_Q_pe": float(q.mean()),
            "sd_Q_pe": float(q.std(ddof=1)),
            "collected_fraction": float(ph.n_detected.sum() / ph.n_emitted.sum()),
            "composition_detected": {
                "direct": float(ph.n_detected_direct.sum() / det_tot),
                "scattered": float(ph.n_detected_scattered.sum() / det_tot),
                "reflected": float(ph.n_detected_reflected.sum() / det_tot),
            },
            "fate_fractions": {
                "absorbed_bulk": float(ph.n_absorbed_bulk.sum() / ph.n_emitted.sum()),
                "absorbed_wall": float(ph.n_absorbed_wall.sum() / ph.n_emitted.sum()),
                "lost": float(ph.n_lost.sum() / ph.n_emitted.sum()),
            },
            "mean_charge_nearest_farthest_pe": [float(n.mean(0).max()), float(n.mean(0).min())],
            "max_over_min_mean_charge": float(n.mean(0).max() / n.mean(0).min()),
            "rel_mean_time_range_ns": [float(np.nanmean(rel, 0).min()), float(np.nanmean(rel, 0).max())],
            "mean_empty_sensors": float((n == 0).sum(1).mean()),
            "barycentre_r_mean_m": float(np.hypot(*barycentre(n, data.detector)).mean()),
            "barycentre_r_sd_m": float(np.hypot(*barycentre(n, data.detector)).std(ddof=1)),
            "true_r_m": float(np.hypot(PRESET_EVENT["x"], PRESET_EVENT["y"])),
        }
    p_q.set_ylabel("Mean charge [p.e.]")
    p_q.set_yscale("log")
    p_q.set_ylim(3.0, 150.0)
    p_q.yaxis.set_minor_formatter(NullFormatter())
    p_t.axhline(0, color=C["grey"], lw=0.8, ls=":")
    p_t.set_ylabel(r"Mean relative time $\tau_j - \bar\tau$ [ns]")
    p_t.set_xlabel("Sensor number $j$")
    p_t.set_xticks(SENSOR_TICKS)
    legend_inside(p_q, loc="upper center", ncol=4)  # the top band of the log panel is empty away from the peak
    p_q.set_title(
        f"Mean of {n_rep} events at E = {PRESET_EVENT['energy']:.0f} MeV, "
        f"(x0, y0) = ({PRESET_EVENT['x']}, {PRESET_EVENT['y']}) m"
    )
    savefig(fig, "t4_4_profiles")
    return out


def straight_light_harmonics(r0: float, phi0: float = 0.0, k_max: int = 3) -> float:
    """Deviation of the straight-light collected fraction from the photocoverage, for a vertex at (r0, phi0).

    For isotropic emission the density of wall hits is the mean of the disc's Poisson kernel and the uniform density.
    Summed over M equally spaced sensors of width a, only the azimuthal harmonics of order k M survive:
    f - c = (1/pi) sum_k (r0/R)^(kM) sin(kMa/2R) cos(kM phi0) / k, with c = M a / (2 pi R) the photocoverage.
    """
    g = preset("straight").geometry
    m, half = g.n_sensors, g.sensor_width / (2 * g.radius)
    terms = [(r0 / g.radius) ** (k * m) * np.sin(k * m * half) * np.cos(k * m * phi0) / k for k in range(1, k_max + 1)]
    return float(np.sum(terms) / np.pi)


def main() -> None:
    """Make all T4.1-T4.4 figures and save their numbers."""
    style()
    results = {"T4.1": t4_1(), "T4.2": t4_2(), "T4.3": t4_3(), "T4.4": t4_4()}
    r_event = float(np.hypot(PRESET_EVENT["x"], PRESET_EVENT["y"]))
    phi_event = float(np.arctan2(PRESET_EVENT["y"], PRESET_EVENT["x"]))
    results["T4.4_straight_light_harmonics"] = {  # analytic, quoted in answers/T4.4.md
        "deviation_at_preset_event_r0": straight_light_harmonics(r_event, phi_event),
        "deviation_r0_0.9_facing_sensor": straight_light_harmonics(0.9, 0.0),
        "deviation_r0_0.9_facing_gap": straight_light_harmonics(0.9, np.pi / preset("straight").geometry.n_sensors),
    }
    path = save_results("T4", results)
    s1 = results["T4.1"]["main_event"]
    print(f"T4.1 main event: Q = {s1['Q_pe']} p.e., barycentre {s1['barycentre_offset_cm']:.1f} cm off")
    for name, v in results["T4.4"].items():
        print(f"T4.4 {name:15s} mean Q = {v['mean_Q_pe']:7.1f} +- {v['sd_Q_pe']:5.1f} p.e.")
    print(f"numbers -> {path}")


if __name__ == "__main__":
    main()
