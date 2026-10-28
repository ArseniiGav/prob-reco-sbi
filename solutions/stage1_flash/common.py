"""Shared helpers of the reference solutions: paths, figure style, cached data sets and results.

Every solution script is run from the repository root (``python solutions/stage1_flash/task2_response.py``) and starts
with::

    from common import FIG, C, dataset, save_results, savefig, style

    style()
    data = dataset("prior_full", 20_000)  # generated once, then read from solutions/stage1_flash/data/
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ...
    savefig(fig, "t1_prior_r2")                                # figures/t1_prior_r2.pdf and .png
    save_results("T1", {"T1.1": {"mean_r2": float(r2.mean())}})  # merged into results/T1.json

Figure conventions (the style of the NeuroMCT paper, applied by :func:`style` with the author's figure-style module
``arsenii_style``, which is not in the repository; without it the same calls draw plain matplotlib figures): white
background, no grid, ticks pointing in on all four sides with minor ticks; the paper's named colours (:data:`C`,
:data:`PRESET_COLOURS`); samples as unfilled step histograms (:func:`stairs_pdf`, alpha 0.6) and models as dashed
lines of the same colour (:func:`model_curve`); framed legends inside the axes in an empty corner
(:func:`legend_inside`), never over the data, or one unframed legend above the panels of a multi-panel figure
(:func:`figure_legend`); group labels in rounded white boxes (:func:`round_label`); sentence-case axis labels with
units; standard log axes (10^n labels) with a fixed floor per panel type; every figure readable on its own and saved
as PDF + PNG (:func:`savefig`). Event displays keep their own conventions: identical sensors coloured by charge
(:data:`CHARGE_CMAP`, black -> grey -> green), separate markers for scattering and wall reflection, the charge
barycentre as the naive vertex. Every number quoted in an answer must come from ``results/*.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

from prob_reco_sbi.flash import SimResult, generate, load, save
from prob_reco_sbi.flash import preset as make_preset

if str(Path(__file__).resolve().parents[1]) not in sys.path:  # the author's figure-style module, where present, is
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # solutions/arsenii_style.py (not in the repository)

try:  # the author's figure style: the figures in figures/ were made with it
    from arsenii_style import (
        PAPER_BASE,
        PAPER_MAP,
        PAPER_SOURCES,
        model_curve,
        paper_style,
        pdf_histogram,
        round_label,
        save_pdf_png,
        stairs_pdf,
        two_legends,
    )
except ImportError:  # without it: plain matplotlib, with the same names and calls
    PAPER_SOURCES = mpl.rcParamsDefault["axes.prop_cycle"].by_key()["color"][:5]
    PAPER_BASE = mpl.rcParamsDefault["axes.prop_cycle"].by_key()["color"][5]
    PAPER_MAP = "black"

    def paper_style(tick_labelsize: float = 16, axes_labelsize: float = 18, legend_fontsize: float = 14) -> None:
        """Set the font sizes of tick labels, axis labels and legends."""
        mpl.rcParams.update(
            {
                "xtick.labelsize": tick_labelsize,
                "ytick.labelsize": tick_labelsize,
                "axes.labelsize": axes_labelsize,
                "legend.fontsize": legend_fontsize,
            }
        )

    def pdf_histogram(samples: Any, bins: Any) -> tuple[np.ndarray, np.ndarray]:
        """Histogram of ``samples`` normalised to a probability density; returns the density and the bin edges."""
        counts, edges = np.histogram(np.asarray(samples), bins=bins)
        return counts / (counts.sum() * np.diff(edges)), edges

    def stairs_pdf(ax: Any, samples: Any, bins: Any, color: str, label: str = "", alpha: float = 0.6, **kw: Any) -> Any:
        """Draw the density histogram of ``samples`` as an unfilled step line."""
        pdf, edges = pdf_histogram(samples, bins)
        return ax.stairs(pdf, edges, color=color, label=label, alpha=alpha, **kw)

    def model_curve(ax: Any, x: Any, pdf: Any, color: str, label: str = "", **kw: Any) -> Any:
        """Draw a model density as a dashed line."""
        kw.setdefault("linestyle", "--")
        return ax.plot(x, pdf, color=color, label=label, **kw)[0]

    def two_legends(
        ax: Any,
        n_first: int,
        proxies: Any = (),
        loc_first: Any = "upper right",
        loc_second: Any = "upper left",
        fontsize: Any = None,
        proxy_color: str = "black",
        proxy_linewidth: float = 1.5,
    ) -> Any:
        """Legend of the first ``n_first`` entries, and a second legend of ``(label, linestyle)`` proxies."""
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        first = ax.legend(handles[:n_first], labels[:n_first], loc=loc_first, fontsize=fontsize)
        ax.add_artist(first)
        lines = [Line2D([], [], color=proxy_color, linestyle=ls, linewidth=proxy_linewidth) for _, ls in proxies]
        second = ax.legend(lines, [name for name, _ in proxies], loc=loc_second, fontsize=fontsize)
        return first, second

    def round_label(ax: Any, text: str, x: float, y: float, fontsize: float = 12, **kw: Any) -> Any:
        """Write ``text`` at axes coordinates ``(x, y)`` in a rounded white box."""
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        bbox = {"boxstyle": "round", "facecolor": "white"}
        return ax.text(x, y, text, fontsize=fontsize, transform=ax.transAxes, bbox=bbox, **kw)

    def save_pdf_png(fig: Any, stem: str | Path, dpi: int = 200, tight: bool = True) -> list[Path]:
        """Save ``<stem>.pdf`` and ``<stem>.png``; returns both paths."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        if tight:
            fig.tight_layout()
        paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
        for path in paths:
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
        return paths


__all__ = [
    "CHARGE_CMAP",
    "DATA",
    "FIG",
    "LINE_STYLES",
    "PAPER_BASE",
    "PAPER_MAP",
    "PAPER_SOURCES",
    "PRESET_COLOURS",
    "PROCESS_COLOURS",
    "REPO",
    "RES",
    "SOL",
    "C",
    "dataset",
    "figure_legend",
    "legend_inside",
    "model_curve",
    "pdf_histogram",
    "round_label",
    "save_results",
    "savefig",
    "stairs_pdf",
    "style",
    "two_legends",
]

log = logging.getLogger("solutions")

REPO = Path(__file__).resolve().parents[2]  #: repository root (this file is solutions/stage1_flash/common.py)
SOL = Path(__file__).resolve().parent  #: this directory, solutions/stage1_flash
FIG = SOL / "figures"  #: figures (PDF + PNG)
RES = SOL / "results"  #: numbers quoted in the answers, one JSON file per task
DATA = SOL / "data"  #: cached HDF5 data sets (never committed)

#: The paper's named matplotlib colours under the semantic keys the scripts use: the five source colours of the paper
#: (darkgreen, royalblue, firebrick, indigo, peru), the base-distribution teal, black for maps, references and ink.
#: ``yellow`` (gold) is only the true-vertex star of the event displays.
C = {
    "green": "darkgreen",
    "blue": "royalblue",
    "red": "firebrick",
    "violet": "indigo",
    "orange": "peru",
    "aqua": "teal",
    "yellow": "gold",
    "ink": "black",
    "ink2": "dimgray",
    "grey": "gray",
    "grid": "lightgray",
    "surface": "white",
}

#: Sensor charge in event displays: black (no light) -> grey -> green (most light), the seminar's convention, kept.
CHARGE_CMAP = LinearSegmentedColormap.from_list("charge", ["#000000", "#8a8a8a", "#1baf7a"])

#: One colour per physics preset in every figure of every task (T0.2, T3, T4.4), in the paper's order of colours:
#: straight darkgreen, absorb royalblue, absorb-scatter firebrick, full indigo.
PRESET_COLOURS = {"straight": C["green"], "absorb": C["blue"], "absorb-scatter": C["red"], "full": C["violet"]}
#: The single-process variants of T3 (straight light plus one process): scattering peru, reflection teal.
PROCESS_COLOURS = {"scatter-only": C["orange"], "reflect-only": C["aqua"]}
#: Line styles for variants of one thing (depths, bin counts, samples), as in the paper.
LINE_STYLES = ["-", "--", ":", "-."]


def style() -> None:
    """Apply the paper style at the font sizes of these small figures: 9.5-11 pt, DejaVu Sans, embedded fonts."""
    paper_style(tick_labelsize=9.5, axes_labelsize=10.5, legend_fontsize=9)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",  # the author: the matplotlib default everywhere, also on slides
            "font.size": 10.5,
            "axes.titlesize": 11,
            "figure.titlesize": 11,
            "axes.prop_cycle": cycler(color=[*PAPER_SOURCES, PAPER_BASE]),
            "axes.formatter.use_mathtext": True,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "legend.frameon": True,
            "patch.force_edgecolor": False,  # filled patches (detector, bands) keep the edges they ask for
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def legend_inside(ax: Any, loc: str | tuple[float, float] = "best", **kwargs: Any) -> Any:
    """Put a framed legend inside the axes, in an empty corner (``loc="best"`` picks the one with the least data)."""
    kwargs.setdefault("frameon", True)
    return ax.legend(loc=loc, **kwargs)


def figure_legend(fig: Any, handles: Any = None, labels: Any = None, where: str = "top", **kwargs: Any) -> Any:
    """One unframed legend for all panels of a figure, above them (``where="top"``, the paper's way) or below."""
    kwargs.setdefault("frameon", False)
    loc = "outside upper center" if where == "top" else "outside lower center"
    if handles is None:
        return fig.legend(loc=loc, **kwargs)
    if labels is None:
        return fig.legend(handles=handles, loc=loc, **kwargs)
    return fig.legend(handles, labels, loc=loc, **kwargs)


def savefig(fig: Any, name: str, close: bool = True) -> list[Path]:
    """Save ``fig`` as ``figures/<name>.pdf`` and ``.png`` (200 dpi, tight bbox); close it unless told not to."""
    import matplotlib.pyplot as plt

    paths = save_pdf_png(fig, FIG / name, dpi=200, tight=False)  # the figures use constrained layout
    if close:
        plt.close(fig)
    return paths


def dataset(
    name: str,
    n_events: int,
    preset: str = "full",
    seed: int = 22,
    *,
    save_hits: bool = False,
    save_tracks: int = 0,
    track_photons: int | None = 100,
    **overrides: Any,
) -> SimResult:
    """Return simulated events, generated once (CPU, public API) and cached in ``solutions/stage1_flash/data``.

    Parameters
    ----------
    name
        File name of the cache, ``solutions/stage1_flash/data/<name>.h5``.
    n_events, seed
        Number of events and seed.
    preset, **overrides
        Configuration: ``prob_reco_sbi.flash.preset(preset, **overrides)``.
    save_hits, save_tracks, track_photons
        Optional hit list and photon paths, as for ``generate``.

    Returns
    -------
    SimResult
        The cached data if the file matches the request (configuration, seed, number of events and the optional
        groups), otherwise freshly generated data, which then replace the file.
    """
    config = make_preset(preset, **overrides)
    path = DATA / f"{name}.h5"
    wanted = {"seed": seed, "save_hits": save_hits, "save_tracks": save_tracks, "track_photons": track_photons}
    if path.exists():
        cached = load(path)
        stored = {key: cached.provenance.get(key) for key in wanted}
        if cached.config == config and len(cached) == n_events and stored == wanted:
            return cached
        log.warning("%s does not match the request (configuration, seed, size or options): regenerating", path)
    log.warning("generating %s: %d events, preset %s %s, seed %d", path.name, n_events, preset, overrides, seed)
    data = generate(
        n_events,
        config=config,
        seed=seed,
        device="cpu",
        save_hits=save_hits,
        save_tracks=save_tracks,
        track_photons=track_photons,
        progress=True,
    )
    data.provenance["dataset"] = name
    save(data, path)
    return data


def _plain(value: Any) -> Any:
    """Convert NumPy scalars and arrays to plain Python values for JSON."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def save_results(task: str, values: dict[str, Any]) -> Path:
    """Merge ``values`` into ``results/<task>.json`` (top-level keys are replaced) and return the path."""
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / f"{task}.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(_plain(values))
    path.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    return path
