"""T4.5 (bonus): animation of the photons of one event.

Run from the repository root::

    python solutions/stage1_flash/task4_animation.py

With ``--compare-pillowwriter`` it also writes the same animation with matplotlib's FuncAnimation + PillowWriter (as in
the TASKS.md hint), records its size and deletes it.

Output: solutions/stage1_flash/figures/t4_5_photons.gif (looping GIF) and t4_5_photons_frame.pdf/.png (its final
frame); numbers in solutions/stage1_flash/results/T4.json (key T4.5). The event is a 0.3 MeV flash at (0.25, 0.15) m,
as in the seminar animation: about 300 photons, all of which are tracked and drawn (a 2 MeV event would have about
2000).
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")

from common import DATA, FIG, RES, save_results, savefig, style
from photon_animation import animate_event

from prob_reco_sbi.flash import preset, simulate

EVENT = {"x": 0.25, "y": 0.15, "energy": 0.3, "t0": 0.0}
SEED = 451


def save_poster(fig: object) -> None:
    """Save the final frame as PDF and PNG (without a layout engine: the frame has a fixed layout)."""
    with matplotlib.rc_context({"figure.constrained_layout.use": False}):
        savefig(fig, "t4_5_photons_frame", close=False)


def main() -> None:
    """Simulate the event, write the GIF and its final frame, save the numbers."""
    style()
    ev = simulate(
        EVENT["x"],
        EVENT["y"],
        EVENT["energy"],
        EVENT["t0"],
        config=preset("full"),
        seed=SEED,
        device="cpu",
        save_tracks=1,
        track_photons=None,
    )
    info = animate_event(
        ev,
        FIG / "t4_5_photons.gif",
        width=7.2,
        dpi=110,
        dt=0.12,
        title=f"A {EVENT['energy']} MeV flash at ({EVENT['x']}, {EVENT['y']}) m: every photon, full model",
        poster=save_poster,
    )
    info |= {"event": EVENT, "seed": SEED, "Q_pe": int(ev.response.charge[0].sum())}
    if "--compare-pillowwriter" in sys.argv:
        # the same animation written the plain matplotlib way (FuncAnimation + PillowWriter), only to compare the size
        tmp = DATA / "t4_5_pillowwriter_comparison.gif"
        plain = animate_event(ev, tmp, width=7.2, dpi=110, dt=0.12, title="comparison", writer="pillow")
        tmp.unlink()
        info["pillowwriter_comparison"] = {"file_MB": plain["file_MB"], "frames": plain["frames"]}
    else:  # keep the comparison of an earlier run with the flag
        previous = json.loads((RES / "T4.json").read_text()).get("T4.5", {}) if (RES / "T4.json").exists() else {}
        if "pillowwriter_comparison" in previous:
            info["pillowwriter_comparison"] = previous["pillowwriter_comparison"]
    path = save_results("T4", {"T4.5": info})
    print(
        f"T4.5: {info['photons']} photons, {info['detected']} detected, {info['frames']} frames "
        f"({info['duration_s']:.1f} s), {info['file_MB']:.2f} MB -> {FIG / 't4_5_photons.gif'}; numbers -> {path}"
    )


if __name__ == "__main__":
    main()
