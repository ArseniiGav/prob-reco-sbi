"""Event displays of the toy detector, shared by the T4 and T5 solutions and by ``make_readme_figures.py``.

Conventions (those of the seminar, drawn in the paper's named colours of ``common.C``):

* the detector is a light-blue disc with a royal-blue wall; the identical sensors are drawn as annular sectors just
  outside the wall, coloured by their charge (``CHARGE_CMAP``: black = no light, grey = little, green = most);
* the true vertex is a gold star, the charge barycentre (the naive vertex estimate) a firebrick cross;
* photon paths are peru (orange-brown) for detected photons and grey for absorbed ones; the points along a path have
  their own markers: Rayleigh scatterings indigo dots, wall reflections teal squares, detections peru diamonds,
  absorptions black crosses (in the bulk) or black triangles (on the wall).

Run the solution scripts from the repository root (``python solutions/stage1_flash/task4_event_displays.py``); this
module is imported like ``common``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from common import CHARGE_CMAP, C
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Wedge

from prob_reco_sbi.flash import Detector, Interaction, Tracks

__all__ = [
    "DISC_FILL",
    "POINT_STYLE",
    "SENSOR_DEPTH",
    "TIME_CMAP",
    "barycentre",
    "charge_colorbar",
    "draw_barycentre",
    "draw_detector",
    "draw_tracks",
    "draw_vertex",
    "event_title",
    "marker_handles",
    "path_handles",
    "relative_times",
    "sensor_distance",
    "sensor_patch_handle",
    "vertex_handles",
]

DISC_FILL = "#eef4fc"  #: inside of the detector
SENSOR_DEPTH = 0.07  #: radial size of the drawn sensors [m] (drawing only; the sensors are arcs on the wall)
#: Relative times: early = royal blue, average = light grey, late = peru (diverging).
TIME_CMAP = LinearSegmentedColormap.from_list("relative_time", [C["blue"], "#ecebe7", C["orange"]])

#: Marker of every point type on a photon path (EMIT is the vertex itself and gets no marker).
POINT_STYLE: dict[int, dict[str, Any]] = {
    Interaction.SCATTER: {
        "marker": "o",
        "ms": 3.2,
        "color": C["violet"],
        "mec": "none",
        "label": "Rayleigh scattering",
    },
    Interaction.REFLECT: {"marker": "s", "ms": 3.6, "color": C["aqua"], "mec": "none", "label": "wall reflection"},
    Interaction.DETECT: {
        "marker": "D",
        "ms": 3.8,
        "color": C["orange"],
        "mec": C["ink"],
        "mew": 0.5,
        "label": "detection on a sensor",
    },
    Interaction.ABSORB_BULK: {
        "marker": "x",
        "ms": 4.2,
        "color": C["ink"],
        "mew": 1.1,
        "label": "absorption in the liquid",
    },
    Interaction.ABSORB_WALL: {
        "marker": "v",
        "ms": 4.2,
        "color": C["ink"],
        "mec": "none",
        "label": "absorption on the wall",
    },
    Interaction.ABSORB_SENSOR: {
        "marker": "^",
        "ms": 4.2,
        "color": C["ink2"],
        "mec": "none",
        "label": "absorbed by a sensor (not detected)",
    },
    Interaction.LOST: {"marker": "P", "ms": 4.5, "color": C["red"], "mec": "none", "label": "lost (max_steps)"},
}


def barycentre(charge: np.ndarray, detector: Detector) -> tuple[np.ndarray, np.ndarray]:
    """Charge-weighted mean of the sensor positions, ``sum_j n_j s_j / sum_j n_j`` (NaN for events without charge).

    Parameters
    ----------
    charge
        Sensor charges, shape ``[..., M]``.
    detector
        Sensor layout (``result.detector``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)`` of the barycentre [m], shape ``[...]``.
    """
    n = np.asarray(charge, dtype=np.float64)
    total = n.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bx = (n * detector.sensor_x).sum(axis=-1) / total
        by = (n * detector.sensor_y).sum(axis=-1) / total
    return bx, by


def relative_times(charge: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return times relative to the charge-weighted mean over the hit sensors (NaN where a sensor has no hit).

    This is the second half of ``network_input`` except that empty sensors are NaN instead of 0 (for plotting).
    """
    n = np.asarray(charge, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    hit = n > 0
    w = np.where(hit, n, 0.0)
    mean = (w * np.where(hit, t, 0.0)).sum(axis=-1, keepdims=True) / np.maximum(w.sum(axis=-1, keepdims=True), 1.0)
    return np.where(hit, t - mean, np.nan)


def sensor_distance(x: float | np.ndarray, y: float | np.ndarray, detector: Detector) -> np.ndarray:
    """Distance from the vertex ``(x, y)`` to the centre of every sensor [m], shape ``[..., M]``."""
    x = np.asarray(x, dtype=np.float64)[..., None]
    y = np.asarray(y, dtype=np.float64)[..., None]
    return np.hypot(detector.sensor_x - x, detector.sensor_y - y)


def draw_detector(
    ax: Any,
    detector: Detector,
    values: Sequence[float] | np.ndarray | None = None,
    *,
    norm: Normalize | None = None,
    cmap: Any = CHARGE_CMAP,
    missing_color: str = "white",
    depth: float = SENSOR_DEPTH,
    labels: Iterable[int] = (0, 8, 16, 24),
    label_size: float = 8.5,
    fill: str = DISC_FILL,
    wall_lw: float = 1.6,
    margin: float = 0.04,
) -> list[Wedge]:
    """Draw the disc and its sensors; colour the sensors by ``values`` (e.g. the charges).

    Parameters
    ----------
    ax
        Matplotlib axes (made equal-aspect and frameless).
    detector
        Sensor layout (``result.detector``).
    values
        One value per sensor (charge, relative time, ...); NaN values get ``missing_color``. ``None`` draws all
        sensors in dark grey.
    norm, cmap
        Colour scale of the values (default: 0 to the maximum value, black -> grey -> green).
    missing_color
        Colour of sensors with a NaN value (e.g. a time display: sensors without hits).
    depth
        Radial size of the drawn sensors [m].
    labels
        Sensors whose number ``j`` is written next to them.
    label_size
        Font size of those numbers.
    fill, wall_lw
        Colour of the inside of the disc and line width of the wall.
    margin
        Free space around the drawing [m].

    Returns
    -------
    list of matplotlib.patches.Wedge
        The sensor patches, in sensor order (to recolour them in an animation).
    """
    radius = detector.radius
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0.0, 0.0), radius, fc=fill, ec=C["blue"], lw=wall_lw, zorder=0))
    if values is not None:
        vals = np.asarray(values, dtype=np.float64)
        if norm is None:
            finite = vals[np.isfinite(vals)]
            norm = Normalize(0.0, max(float(finite.max()) if finite.size else 1.0, 1.0))
    half = np.degrees(detector.sensor_width / (2 * radius))
    patches = []
    for j, phi in enumerate(np.degrees(detector.sensor_phi)):
        if values is None:
            colour: Any = C["ink2"]
        elif not np.isfinite(vals[j]):
            colour = missing_color
        else:
            colour = cmap(norm(vals[j]))
        wedge = Wedge(
            (0.0, 0.0), radius + depth, phi - half, phi + half, width=depth, fc=colour, ec=C["ink2"], lw=0.4, zorder=4
        )
        ax.add_patch(wedge)
        patches.append(wedge)
    labels = list(labels)
    r_text = radius + depth + 0.085
    for j in labels:
        phi = detector.sensor_phi[j]
        ax.text(
            r_text * np.cos(phi),
            r_text * np.sin(phi),
            f"{j}",
            ha="center",
            va="center",
            fontsize=label_size,
            color=C["ink2"],
            zorder=4,
        )
    lim = radius + depth + margin + (0.15 if labels else 0.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return patches


def draw_vertex(ax: Any, x: float, y: float, ms: float = 14.0, **kwargs: Any) -> Any:
    """Mark the true vertex (gold star)."""
    style = {"ls": "none", "marker": "*", "ms": ms, "color": C["yellow"], "mec": C["ink"], "mew": 0.8, "zorder": 8}
    return ax.plot(x, y, **(style | kwargs))[0]


def draw_barycentre(ax: Any, x: float, y: float, ms: float = 9.0, **kwargs: Any) -> Any:
    """Mark the charge barycentre, the naive vertex estimate (firebrick cross)."""
    style = {"ls": "none", "marker": "X", "ms": ms, "color": C["red"], "mec": "white", "mew": 0.7, "zorder": 9}
    return ax.plot(x, y, **(style | kwargs))[0]


def vertex_handles(ms: float = 11.0) -> list[Line2D]:
    """Legend entries of the true vertex and the charge barycentre."""
    return [
        Line2D([], [], ls="none", marker="*", ms=ms, color=C["yellow"], mec=C["ink"], mew=0.8, label="true vertex"),
        Line2D(
            [],
            [],
            ls="none",
            marker="X",
            ms=ms * 0.7,
            color=C["red"],
            mec="white",
            mew=0.7,
            label="charge barycentre (naive vertex)",
        ),
    ]


def charge_colorbar(fig: Any, ax: Any, norm: Normalize, label: str = "Sensor charge [p.e.]", **kwargs: Any) -> Any:
    """Colour bar of the sensor charge (black -> grey -> green) next to ``ax``."""
    cmap = kwargs.pop("cmap", CHARGE_CMAP)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, **kwargs)
    bar.set_label(label)
    bar.outline.set_linewidth(0.5)
    return bar


def draw_tracks(
    ax: Any,
    tracks: Tracks,
    event_id: int,
    *,
    limit: int | None = None,
    markers: Iterable[int] = (
        Interaction.SCATTER,
        Interaction.REFLECT,
        Interaction.DETECT,
        Interaction.ABSORB_BULK,
        Interaction.ABSORB_WALL,
        Interaction.ABSORB_SENSOR,
        Interaction.LOST,
    ),
    lw: float = 1.0,
    marker_scale: float = 1.0,
) -> dict[str, int]:
    """Draw the recorded photon paths of one event with a marker at every interaction point.

    Parameters
    ----------
    ax
        Axes of an event display.
    tracks
        Recorded paths (``result.tracks``).
    event_id
        Event whose paths are drawn.
    limit
        Draw at most this many photons (the first ones recorded).
    markers
        Point types that get a marker (see ``POINT_STYLE``).
    lw
        Line width of the paths of detected photons (absorbed ones are thinner).
    marker_scale
        Factor on the marker sizes.

    Returns
    -------
    dict
        Number of drawn photons per fate (``Interaction`` names) and in total (``"all"``).
    """
    ids = tracks.of_event(event_id)[:limit]
    detected, absorbed = [], []
    points: dict[int, list[np.ndarray]] = {int(k): [] for k in markers}
    fates: dict[str, int] = {"all": len(ids)}
    for k in ids:
        x, y, kind = tracks.polyline(int(k))
        xy = np.column_stack([x, y]).astype(np.float64)
        fate = Interaction(int(kind[-1]))
        fates[fate.name] = fates.get(fate.name, 0) + 1
        (detected if fate == Interaction.DETECT else absorbed).append(xy)
        for code in points:
            sel = kind[1:] == code
            if sel.any():
                points[code].append(xy[1:][sel])
    ax.add_collection(LineCollection(absorbed, colors=C["ink2"], linewidths=0.6 * lw, alpha=0.45, zorder=2))
    ax.add_collection(LineCollection(detected, colors=C["orange"], linewidths=lw, alpha=0.9, zorder=3))
    for code, chunks in points.items():
        if not chunks:
            continue
        xy = np.concatenate(chunks)
        style = {k: v for k, v in POINT_STYLE[code].items() if k != "label"}
        style["ms"] = style["ms"] * marker_scale
        ax.plot(xy[:, 0], xy[:, 1], ls="none", zorder=6, **style)
    return fates


def path_handles() -> list[Line2D]:
    """Legend entries of the two kinds of photon path."""
    return [
        Line2D([], [], color=C["orange"], lw=1.6, label="path of a detected photon"),
        Line2D([], [], color=C["ink2"], lw=1.0, alpha=0.6, label="path of an absorbed photon"),
    ]


def marker_handles(kinds: Iterable[int], scale: float = 1.4) -> list[Line2D]:
    """Legend entries of the interaction markers ``kinds`` (see ``POINT_STYLE``)."""
    handles = []
    for code in kinds:
        style = dict(POINT_STYLE[code])
        label = style.pop("label")
        style["ms"] = style["ms"] * scale
        handles.append(Line2D([], [], ls="none", label=label, **style))
    return handles


def sensor_patch_handle(label: str = "sensor: black = no light, green = most light") -> Patch:
    """Legend entry explaining the sensor colours."""
    return Patch(fc=CHARGE_CMAP(0.8), ec=C["ink2"], lw=0.5, label=label)


def event_title(energy: float, r: float, charge_total: float) -> str:
    """Title of an event display: energy, vertex radius and total charge."""
    return f"E = {energy:.2f} MeV, r = {r:.2f} m, Q = {charge_total:.0f} p.e."
