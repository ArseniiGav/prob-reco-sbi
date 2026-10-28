"""Make the README figures: docs/img/detector.png (sketch of the detector) and docs/img/flash_photons.gif (animation).

Run from the repository root::

    python solutions/stage1_flash/make_readme_figures.py

Both pictures use real simulated photons (public API, CPU, fixed seeds):

* detector.png: one 2 MeV event with every photon path recorded; five photons are picked automatically, one of each
  kind (direct, scattered, reflected, absorbed in the liquid, absorbed by the wall), and drawn with the detector.
  About 1600 px wide, with large labels so that it stays readable when the README shows it at 400-700 px.
* flash_photons.gif: the T4.5 animation (a 0.3 MeV flash at (0.25, 0.15) m, all of its photons), 660 px wide,
  looping, below 3 MB.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from common import CHARGE_CMAP, REPO, C, style
from event_display import SENSOR_DEPTH, draw_detector, draw_vertex
from matplotlib.patches import FancyArrowPatch
from photon_animation import animate_event

from prob_reco_sbi.flash import Interaction, Tracks, preset, simulate

IMG = REPO / "docs" / "img"
SKETCH_EVENT = {"x": -0.25, "y": 0.10, "energy": 2.0, "t0": 0.0}
SKETCH_SEED = 7
N_SENSORS = 32  # sensors of the sketch detector (preset "full")
LIT = CHARGE_CMAP(1.0)  # the green of the charge scale: sensors hit by the drawn photons
# kind of photon -> (emission direction wanted [deg], colour, label, label position [m])
PICKS = {
    "direct": (8.0, C["blue"], "direct", (0.52, 0.05)),
    "scattered": (115.0, C["violet"], "scattered", (-0.27, 0.50)),
    "reflected": (297.0, C["aqua"], "reflected", (-0.33, -0.50)),
    "bulk": (200.0, C["ink"], "absorbed\nin the liquid", (-0.55, -0.26)),
    "wall": (55.0, C["ink2"], "absorbed\nby the wall", (0.45, 0.55)),
}


def classify(kind: np.ndarray) -> str | None:
    """Kind of photon of one path (only paths with at most one interaction before their fate)."""
    inner, fate = kind[1:-1], kind[-1]
    n_s, n_r = int((inner == Interaction.SCATTER).sum()), int((inner == Interaction.REFLECT).sum())
    if fate == Interaction.DETECT:
        return {(0, 0): "direct", (1, 0): "scattered", (0, 1): "reflected"}.get((n_s, n_r))
    if n_s or n_r:
        return None
    return {Interaction.ABSORB_BULK: "bulk", Interaction.ABSORB_WALL: "wall"}.get(Interaction(int(fate)))


def pick_paths(tracks: Tracks, event_id: int) -> dict[str, int]:
    """Choose one photon of each kind: emitted close to the wanted direction, with clearly visible segments."""
    best: dict[str, tuple[float, int]] = {}
    for k in tracks.of_event(event_id):
        x, y, kind = tracks.polyline(int(k))
        what = classify(kind)
        if what is None:
            continue
        seg = np.hypot(np.diff(x), np.diff(y))
        if seg.min() < 0.35:
            continue
        if what == "scattered":  # a clear kink
            u = np.array([x[1] - x[0], y[1] - y[0]]) / seg[0]
            v = np.array([x[2] - x[1], y[2] - y[1]]) / seg[1]
            if np.degrees(np.arccos(np.clip(u @ v, -1, 1))) < 55:
                continue
        if what == "bulk" and not 0.45 < seg[0] < 0.65:
            continue
        if what == "wall":  # end point well inside a gap between two sensors, not next to a sensor edge
            phase = (np.arctan2(y[-1], x[-1]) / (2 * np.pi / N_SENSORS)) % 1.0  # 0.5 = middle of a gap
            if abs(phase - 0.5) > 0.12:
                continue
        angle = np.degrees(np.arctan2(y[1] - y[0], x[1] - x[0])) % 360
        miss = abs((angle - PICKS[what][0] + 180) % 360 - 180)
        if what not in best or miss < best[what][0]:
            best[what] = (miss, int(k))
    missing = set(PICKS) - set(best)
    if missing:
        raise RuntimeError(f"no photon of kind {sorted(missing)} in the sketch event: change SKETCH_SEED")
    return {what: k for what, (_, k) in best.items()}


def detector_sketch(path: Path) -> dict[str, object]:
    """Draw the detector with five example photons and save it as a PNG."""
    ev = simulate(
        SKETCH_EVENT["x"],
        SKETCH_EVENT["y"],
        SKETCH_EVENT["energy"],
        SKETCH_EVENT["t0"],
        config=preset("full"),
        seed=SKETCH_SEED,
        device="cpu",
        save_tracks=1,
        track_photons=None,
    )
    assert ev.tracks is not None
    picks = pick_paths(ev.tracks, int(ev.truth.event_id[0]))
    det = ev.detector
    m = det.n_sensors
    lit = np.full(m, np.nan)
    for what in ("direct", "scattered", "reflected"):
        x, y, _ = ev.tracks.polyline(picks[what])
        lit[int(np.round(np.arctan2(y[-1], x[-1]) / (2 * np.pi / m))) % m] = 1.0

    fig = plt.figure(figsize=(8.4, 6.0), layout="none")
    ax = fig.add_axes((0.0, 0.0, 0.70, 1.0))
    patches = draw_detector(ax, det, None, labels=(), margin=0.02, wall_lw=2.2)
    for j, patch in enumerate(patches):
        patch.set_facecolor(LIT if np.isfinite(lit[j]) else "#3a3a38")
    marker_style = {
        Interaction.SCATTER: {"marker": "o", "ms": 12, "color": C["violet"], "mec": "white", "mew": 1.2},
        Interaction.REFLECT: {"marker": "s", "ms": 12, "color": C["aqua"], "mec": "white", "mew": 1.2},
        Interaction.ABSORB_BULK: {"marker": "X", "ms": 15, "color": C["ink"], "mec": "white", "mew": 1.2},
        Interaction.ABSORB_WALL: {"marker": "X", "ms": 15, "color": C["ink2"], "mec": "white", "mew": 1.2},
    }
    box = {"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.0}
    for what, k in picks.items():
        x, y, kind = ev.tracks.polyline(k)
        _, colour, label, (lx, ly) = PICKS[what]
        ax.plot(x, y, color=colour, lw=3.2, solid_capstyle="round", zorder=3)
        for code, style_ in marker_style.items():
            sel = kind == code
            ax.plot(x[sel], y[sel], ls="none", zorder=5, **style_)
        ax.text(
            lx,
            ly,
            label,
            color=colour,
            fontsize=17,
            ha="center",
            va="center",
            zorder=7,
            fontweight="bold",
            bbox=box,
            linespacing=0.95,
        )
    draw_vertex(ax, SKETCH_EVENT["x"], SKETCH_EVENT["y"], ms=30, zorder=8)
    ax.text(-0.53, 0.17, "vertex", fontsize=17, ha="center", va="center", color=C["ink"], zorder=7, bbox=box)
    phi = np.radians(-40)  # the radius, drawn along a direction free of photon paths
    ax.add_patch(
        FancyArrowPatch(
            (0, 0),
            (np.cos(phi), np.sin(phi)),
            arrowstyle="-|>",
            mutation_scale=22,
            lw=1.8,
            color=C["ink2"],
            zorder=2,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.plot([0], [0], marker="o", ms=6, color=C["ink2"], zorder=2)
    ax.text(
        0.55 * np.cos(phi) + 0.05,
        0.55 * np.sin(phi) + 0.06,
        "R = 1 m",
        fontsize=16,
        color=C["ink2"],
        rotation=np.degrees(phi),
        rotation_mode="anchor",
        ha="center",
        va="bottom",
    )
    lim = det.radius + SENSOR_DEPTH + 0.04
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # the right column: sensors and markers
    x_txt = 0.715
    fig.text(x_txt, 0.86, "32 photosensors", fontsize=16, color=C["ink"], va="top")
    fig.text(x_txt, 0.80, "10 cm arcs,\n51 % of the wall", fontsize=15, color=C["ink2"], va="top", linespacing=1.15)
    fig.text(x_txt, 0.66, "green: hit by\nthese photons", fontsize=15, color=LIT, va="top", linespacing=1.15)
    fig.text(x_txt, 0.47, "markers", fontsize=16, color=C["ink"], va="top")
    for i, (txt, colour, marker) in enumerate(
        [("scattering", C["violet"], "o"), ("wall reflection", C["aqua"], "s"), ("absorption", C["ink"], "X")]
    ):
        yy = 0.39 - 0.07 * i
        fig.lines.append(
            plt.Line2D(
                [x_txt + 0.012],
                [yy],
                transform=fig.transFigure,
                ls="none",
                marker=marker,
                ms=12,
                color=colour,
                mec="white",
                mew=1.0,
            )
        )
        fig.text(x_txt + 0.035, yy, txt, fontsize=15, color=colour, va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    kinds = {what: Interaction(int(ev.tracks.polyline(k)[2][-1])).name for what, k in picks.items()}
    return {
        "photons_in_event": len(ev.tracks),
        "picked": {w: int(ev.tracks.photon[k]) for w, k in picks.items()},
        "fates": kinds,
    }


def main() -> None:
    """Make both README figures and print their sizes."""
    style()
    sketch = detector_sketch(IMG / "detector.png")
    ev = simulate(
        0.25, 0.15, 0.3, 0.0, config=preset("full"), seed=451, device="cpu", save_tracks=1, track_photons=None
    )
    gif = animate_event(
        ev,
        IMG / "flash_photons.gif",
        width=6.6,
        dpi=100,
        dt=0.12,
        font=13.5,
        title="One 0.3 MeV flash, every photon",
        colors=96,
    )
    from PIL import Image

    with Image.open(IMG / "detector.png") as im:
        size = im.size
    print(
        f"docs/img/detector.png: {size[0]} x {size[1]} px, {sketch['photons_in_event']} photons, "
        f"picked {sketch['picked']}"
    )
    print(
        f"docs/img/flash_photons.gif: {gif['size_px'][0]} x {gif['size_px'][1]} px, {gif['frames']} frames, "
        f"{gif['duration_s']:.1f} s, {gif['file_MB']:.2f} MB"
    )


if __name__ == "__main__":
    main()
