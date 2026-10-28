"""Animation of the photons of one event flying through the detector (GIF), for T4.5 and the README.

A port of the seminar's ``flash_animation.py`` to the public API: the event is simulated with ``simulate(...,
save_tracks=1, track_photons=None)``, so every photon path is recorded with the path length at each point. The frames
advance in physical time: a photon at path length ``s`` is at time ``s / c_n`` after the emission. Paths grow at the
speed of light in the medium; Rayleigh scatterings (indigo dots) and wall reflections (teal squares) appear when they
happen; absorbed photons turn grey; detected photons stay peru (orange-brown) and light up their sensor (black = no
light, grey = little, green = most, on the scale of the final charges). Drawn in the paper style of ``common.style``
with the figure's own fixed layout (nothing may move between frames).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common import CHARGE_CMAP, C
from event_display import DISC_FILL, SENSOR_DEPTH, draw_detector, draw_vertex
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

from prob_reco_sbi.flash import Interaction, SimResult

__all__ = ["PhotonPath", "animate_event", "photon_paths"]


@dataclass(frozen=True)
class PhotonPath:
    """One recorded photon: its points, the time at each point and what happened there."""

    xy: np.ndarray  #: point coordinates [m], shape [k, 2]
    t: np.ndarray  #: time since the emission [ns], shape [k]
    kind: np.ndarray  #: Interaction code of each point, shape [k]
    sensor: int  #: sensor that detected the photon, -1 if it was not detected

    @property
    def detected(self) -> bool:
        """Whether the photon ended on a sensor and was detected."""
        return self.sensor >= 0

    @property
    def t_end(self) -> float:
        """Time of the last point [ns]."""
        return float(self.t[-1])


def photon_paths(result: SimResult, event_index: int = 0) -> list[PhotonPath]:
    """Return the recorded photon paths of one event, with the time of every point (``path_length / c_n``)."""
    if result.tracks is None:
        raise ValueError("the result has no tracks: simulate with save_tracks >= 1")
    tracks = result.tracks
    c_n = result.config.optics.speed_of_light
    m = result.n_sensors
    dphi = 2 * np.pi / m
    paths = []
    for k in tracks.of_event(int(result.truth.event_id[event_index])):
        part = slice(tracks.offsets[k], tracks.offsets[k + 1])
        xy = np.column_stack([tracks.x[part], tracks.y[part]]).astype(np.float64)
        kind = tracks.kind[part].astype(np.int64)
        sensor = -1
        if kind[-1] == Interaction.DETECT:
            sensor = int(np.round(np.arctan2(xy[-1, 1], xy[-1, 0]) / dphi)) % m
        paths.append(PhotonPath(xy=xy, t=tracks.path_length[part].astype(np.float64) / c_n, kind=kind, sensor=sensor))
    return paths


def animate_event(
    result: SimResult,
    out: Path,
    *,
    event_index: int = 0,
    width: float = 7.2,
    dpi: int = 110,
    fps: int = 25,
    dt: float = 0.12,
    t_end: float | None = None,
    hold: int = 40,
    font: float = 13.0,
    title: str | None = None,
    poster: Any = None,
    colors: int = 128,
    writer: str = "palette",
) -> dict[str, Any]:
    """Write a looping GIF of the photons of one event; optionally save the final frame with ``poster(fig)``.

    Parameters
    ----------
    result
        Simulation with recorded tracks (``save_tracks >= 1``; ``track_photons=None`` records every photon).
    out
        Output GIF file.
    event_index
        Event of ``result`` to animate.
    width, dpi
        Figure width [inch] and resolution: the GIF is ``width * dpi`` pixels wide.
    fps, dt
        Frames per second and physical time per frame [ns].
    t_end
        Last animated time [ns]; default: the end of the last photon path.
    hold
        Number of extra frames that show the final state before the GIF loops.
    font
        Base font size [pt].
    title
        Optional title above the clock.
    poster
        Called as ``poster(fig)`` with the final frame drawn (e.g. to save it as PDF and PNG).
    colors
        Number of colours of the GIF palette (at most 256).
    writer
        ``"palette"`` (default): one shared palette without dithering, a small file; ``"pillow"``: matplotlib's
        ``FuncAnimation`` with ``PillowWriter`` (one palette per frame), for comparison.

    Returns
    -------
    dict
        Numbers of the animation: photons, detected, absorbed, frames, duration and file size.
    """
    paths = photon_paths(result, event_index)
    detector = result.detector
    m = detector.n_sensors
    final = np.bincount([p.sensor for p in paths if p.detected], minlength=m)
    t_last = max(p.t_end for p in paths)
    t_end = t_last if t_end is None else t_end
    n_steps = int(np.ceil(t_end / dt))
    cmax = max(int(final.max()), 1)
    norm = Normalize(0, cmax)

    height = width * 1.08 + (0.35 if title else 0.0)
    fig = plt.figure(figsize=(width, height), dpi=dpi, layout="none")  # fixed layout: nothing moves between frames
    top = 1 - (0.6 + (0.35 if title else 0.0)) / height
    ax = fig.add_axes((0.02, 0.18, 0.96, top - 0.18))
    sensors = draw_detector(ax, detector, np.zeros(m), norm=norm, labels=(), margin=0.03)
    lim = detector.radius + SENSOR_DEPTH + 0.03
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    x0, y0 = float(result.truth.x[event_index]), float(result.truth.y[event_index])
    draw_vertex(ax, x0, y0, ms=font * 1.4)

    alive = LineCollection([], colors=C["orange"], linewidths=1.2, alpha=0.9, zorder=3)
    dead = LineCollection([], colors=C["ink2"], linewidths=0.7, alpha=0.35, zorder=2)
    ax.add_collection(dead)
    ax.add_collection(alive)
    heads = ax.plot([], [], ls="none", marker="o", ms=font * 0.3, color=C["orange"], zorder=5)[0]
    scat = ax.plot([], [], ls="none", marker="o", ms=font * 0.26, color=C["violet"], alpha=0.75, zorder=4)[0]
    bounce = ax.plot([], [], ls="none", marker="s", ms=font * 0.32, color=C["aqua"], zorder=4)[0]

    y_text = 1 - 0.32 / height
    if title:
        fig.text(0.5, 1 - 0.22 / height, title, ha="center", va="center", fontsize=font * 0.95, color=C["ink"])
        y_text = 1 - 0.62 / height
    clock = fig.text(0.04, y_text, "", fontsize=font * 1.25, color=C["ink"], ha="left", va="center")
    counter = fig.text(0.96, y_text, "", fontsize=font * 0.95, color=C["ink2"], ha="right", va="center")
    handles = [
        Line2D([], [], color=C["orange"], lw=1.8, label="photon in flight / detected"),
        Line2D([], [], color=C["ink2"], lw=1.2, alpha=0.6, label="photon absorbed"),
        Line2D([], [], ls="none", marker="o", ms=font * 0.45, color=C["violet"], label="Rayleigh scattering"),
        Line2D([], [], ls="none", marker="s", ms=font * 0.5, color=C["aqua"], label="wall reflection"),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.005),
        fontsize=font * 0.88,
        frameon=True,
        handlelength=1.5,
        borderaxespad=0.2,
    )
    cax = fig.add_axes((0.57, 0.085, 0.38, 0.022))
    ticks = MaxNLocator(nbins=6, integer=True)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=CHARGE_CMAP), cax=cax, orientation="horizontal", ticks=ticks)
    bar.set_label("Photons per sensor", fontsize=font * 0.88, color=C["ink2"])
    bar.outline.set_linewidth(0.5)
    cax.tick_params(labelsize=font * 0.8, colors=C["ink2"], length=3)

    def draw(t: float) -> None:
        live, gone, hx, hy, sx, sy, bx, by = [], [], [], [], [], [], [], []
        counts = np.zeros(m, dtype=int)
        n_absorbed = 0
        for p in paths:
            if t <= 0:
                break
            i = int(np.searchsorted(p.t, t, side="right"))  # points already reached
            if i >= len(p.t):  # path complete
                (live if p.detected else gone).append(p.xy)
                if p.detected:
                    counts[p.sensor] += 1
                else:
                    n_absorbed += 1
                reached = p.kind[1:-1]
                pts = p.xy[1:-1]
            else:
                f = (t - p.t[i - 1]) / max(p.t[i] - p.t[i - 1], 1e-12)
                head = p.xy[i - 1] + f * (p.xy[i] - p.xy[i - 1])
                live.append(np.vstack([p.xy[:i], head]))
                hx.append(head[0])
                hy.append(head[1])
                reached = p.kind[1:i]
                pts = p.xy[1:i]
            sel_s, sel_b = reached == Interaction.SCATTER, reached == Interaction.REFLECT
            sx.extend(pts[sel_s, 0])
            sy.extend(pts[sel_s, 1])
            bx.extend(pts[sel_b, 0])
            by.extend(pts[sel_b, 1])
        alive.set_segments(live)
        dead.set_segments(gone)
        heads.set_data(hx, hy)
        scat.set_data(sx, sy)
        bounce.set_data(bx, by)
        for j, patch in enumerate(sensors):
            patch.set_facecolor(CHARGE_CMAP(norm(counts[j])))
        clock.set_text(f"t = {t:.1f} ns")
        counter.set_text(f"detected {counts.sum()}   absorbed {n_absorbed}   in flight {len(hx)}")
        state.update(detected=int(counts.sum()), absorbed=n_absorbed, in_flight=len(hx))

    state: dict[str, int] = {}
    times = np.arange(n_steps + 1) * dt
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [round(1000 / fps)] * len(times)
    durations[-1] += hold * round(1000 / fps)  # the final state stays on screen before the GIF loops
    if writer == "pillow":
        # the plain matplotlib way (FuncAnimation + PillowWriter): every frame gets its own palette
        from matplotlib.animation import FuncAnimation, PillowWriter

        anim = FuncAnimation(fig, lambda k: draw(float(times[min(k, n_steps)])), frames=n_steps + 1 + hold)
        anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
        n_written = n_steps + 1 + hold
        draw(float(times[-1]))
    else:
        # One palette for all frames (taken from a few frames spread over the animation) and no dithering: unchanged
        # pixels stay identical from frame to frame, so the GIF stores only the small changed regions.
        sample = [_render(fig, draw, float(t)) for t in times[np.linspace(0, n_steps, 6).round().astype(int)]]
        montage = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for i, im in enumerate(sample):
            montage.paste(im, (0, i * im.height))
        fixed = [C[k] for k in ("blue", "orange", "aqua", "yellow", "violet", "red", "ink", "ink2", "green")]
        fixed += ["#ffffff", DISC_FILL] + [CHARGE_CMAP(norm(c)) for c in range(cmax + 1)]
        fixed += [CHARGE_CMAP(v) for v in np.linspace(0.0, 1.0, 40)]  # the colour bar
        palette = _palette(montage, fixed, colors)
        frames = [_render(fig, draw, float(t)).quantize(palette=palette, dither=Image.Dither.NONE) for t in times]
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=1)
        n_written = len(frames)
    if poster is not None:
        poster(fig)  # the figure shows the final frame
    plt.close(fig)
    n_frames = n_written
    return {
        "photons": len(paths),
        "detected": int(sum(p.detected for p in paths)),
        "absorbed": len(paths) - int(sum(p.detected for p in paths)),
        "final_frame": dict(state),
        "t_end_ns": float(n_steps * dt),
        "t_last_photon_ns": float(t_last),
        "frames": n_frames,
        "duration_s": sum(durations) / 1000,
        "palette_colors": colors if writer == "palette" else None,
        "writer": writer,
        "size_px": [round(width * dpi), round(height * dpi)],
        "file_MB": out.stat().st_size / 1e6,
        "max_sensor_photons": cmax,
        "t_first_detection_ns": float(min(p.t_end for p in paths if p.detected)),
        "median_end_time_ns": float(np.median([p.t_end for p in paths])),
        "median_detection_time_ns": float(np.median([p.t_end for p in paths if p.detected])),
    }


def _render(fig: Any, draw: Any, t: float) -> Image.Image:
    """Draw the frame at time ``t`` and return it as an RGB image."""
    draw(t)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(np.ascontiguousarray(rgba[..., :3]))


def _palette(montage: Image.Image, fixed: list[Any], colors: int) -> Image.Image:
    """GIF palette: the ``fixed`` colours (so that small markers keep their exact colour) plus adaptive colours."""
    fixed_rgb = list(dict.fromkeys(tuple(round(255 * v) for v in to_rgb(c)) for c in fixed))
    n_adaptive = max(colors - len(fixed_rgb), 8)
    adaptive = montage.quantize(colors=n_adaptive, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    flat = adaptive.getpalette()[: 3 * n_adaptive]
    extra = [tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)]
    # drop adaptive colours that nearly duplicate a fixed one: Pillow's palette lookup is approximate and could pick
    # the near-duplicate (e.g. an off-white instead of the white background)
    extra = [c for c in extra if min(max(abs(a - b) for a, b in zip(c, f, strict=True)) for f in fixed_rgb) > 6]
    values = [v for rgb in fixed_rgb for v in rgb] + [v for rgb in extra for v in rgb]
    values = values[: 3 * 256]
    image = Image.new("P", (1, 1))
    image.putpalette(values + [0] * (3 * 256 - len(values)))
    return image
