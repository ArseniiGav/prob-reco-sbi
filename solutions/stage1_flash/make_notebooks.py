"""Build the solution notebooks, one per task, from the scripts, figures and answers; execute them to keep the outputs.

Run from the repository root (after the scripts, or let the notebooks run them)::

    python solutions/stage1_flash/make_notebooks.py                # build T0_setup.ipynb ... T5_bonus.ipynb, execute
    python solutions/stage1_flash/make_notebooks.py --no-run       # only build (no outputs)
    python solutions/stage1_flash/make_notebooks.py --keep-timing  # T0: redraw the timing figure, do not re-measure

Every notebook has, per question ID, a heading, a code cell that shows the figures of that question inline (from
``figures/<name>.png``) and the answer from ``answers/<ID>.md``. The first code cell runs the scripts of the task
(one subprocess each, as ``run_all.sh``), so the notebook makes its own figures and numbers; the scripts stay the
reference implementation and are also
run by ``run_all.sh``. Executing a notebook takes as long as its scripts (seconds with cached data, minutes without).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SOL = Path(__file__).resolve().parent  #: this directory, solutions/stage1_flash
REPO = SOL.parents[1]  #: repository root
FIGURES = SOL / "figures"
ANSWERS = SOL / "answers"

#: Task notebooks: file name, title, scripts to run (in order) and the question IDs.
TASKS = [
    (
        "T0_setup",
        "T0 Set-up",
        ["task0_setup.py", "task0_timing.py", "task0_timing.py --threads", "task0_inspect.py"],
        ["T0.1", "T0.2", "T0.3"],
    ),
    ("T1_prior", "T1 What did we simulate? The prior", ["task1_prior.py"], ["T1.1", "T1.2"]),
    (
        "T2_response",
        "T2 The detector response: what the network will see",
        ["task2_response.py", "task2_proposals.py"],
        ["T2.1", "T2.2", "T2.3", "T2.4", "T2.5"],
    ),
    (
        "T3_physics_ladder",
        "T3 Switch the physics on step by step",
        ["task3_1_ladder.py", "task3_2_expected_charge.py", "task3_3_processes.py", "task3_4_naive_estimators.py"],
        ["T3.1", "T3.2", "T3.3", "T3.4"],
    ),
    (
        "T4_event_displays",
        "T4 Event displays",
        ["task4_event_displays.py", "task4_animation.py"],
        ["T4.1", "T4.2", "T4.3", "T4.4", "T4.5"],
    ),
    (
        "T5_bonus",
        "T5 Bonus: towards inference",
        ["task5_fluctuations.py", "task5_degeneracy.py", "task5_detector_design.py", "task5_detector_effects.py"],
        ["T5.1", "T5.2", "T5.3", "T5.4"],
    ),
]

KERNEL = {"display_name": "Python (prob-reco-sbi)", "language": "python", "name": "prob-reco-sbi"}
SHOW = '''import os
from pathlib import Path

import IPython.display as ipd

here = Path.cwd()
os.chdir(next(p for p in (here, *here.parents) if (p / "pyproject.toml").exists()))  # the repository root
FIGURES = Path("solutions/stage1_flash/figures")


def show(*names):
    """Display the figures of a question (GIF animates, else the PNG)."""
    for name in names:
        path = next(p for p in (FIGURES / f"{name}.gif", FIGURES / f"{name}.png") if p.exists())
        ipd.display(ipd.Image(filename=str(path), width=900))
'''


FIGURE_MARKER = re.compile(r"^!\[fig\]\(([^)]+)\)\s*$", re.M)


def split_at_figures(body: str) -> list[tuple[str, str]]:
    """Split an answer at its figure markers ``![fig](name ...)``: a list of ("text", ...) and ("figure", names)."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in FIGURE_MARKER.finditer(body):
        parts.append(("text", body[pos : m.start()]))
        parts.append(("figure", m.group(1)))
        pos = m.end()
    parts.append(("text", body[pos:]))
    return parts


def build(name: str, title: str, scripts: list[str], ids: list[str], note: str = "") -> nbformat.NotebookNode:
    """Assemble the notebook of one task (not executed); ``note`` is added to the "Run the scripts" cell."""
    nb = nbformat.v4.new_notebook(metadata={"kernelspec": KERNEL, "language_info": {"name": "python"}})
    cells = nb.cells
    cells.append(
        nbformat.v4.new_markdown_cell(
            f"# {title}: reference solution\n\n"
            f"Questions {', '.join(ids)} of [TASKS.md](../../TASKS.md). The code is in the scripts named below "
            "(run from the repository root); this notebook runs them, shows their figures and gives the answers. "
            "Every number quoted comes from `results/`."
        )
    )
    cells.append(nbformat.v4.new_code_cell(SHOW))
    # one subprocess per script, as run_all.sh does
    run_lines = "\n".join(f"!python solutions/stage1_flash/{s}" for s in scripts)
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Run the scripts\n\nData sets are generated on first use and cached in "
            "`solutions/stage1_flash/data/`." + note
        )
    )
    cells.append(nbformat.v4.new_code_cell(run_lines))
    for qid in ids:
        answer = (ANSWERS / f"{qid}.md").read_text(encoding="utf-8").strip()
        heading, _, body = answer.partition("\n")
        cells.append(nbformat.v4.new_markdown_cell(f"## {heading.lstrip('# ').strip()}"))
        for kind, text in split_at_figures(body):
            if kind == "figure":
                cells.append(nbformat.v4.new_code_cell("show(" + ", ".join(f'"{f}"' for f in text.split()) + ")"))
            elif text.strip():
                cells.append(nbformat.v4.new_markdown_cell(text.strip()))
    return nb


def main() -> int:
    """Build every notebook (or those given with --only) and execute them unless --no-run."""
    parser = argparse.ArgumentParser(description="Build (and execute) the solution notebooks.")
    parser.add_argument("--no-run", action="store_true", help="build without executing")
    parser.add_argument("--only", nargs="*", help="notebook names to build, e.g. T2_response")
    parser.add_argument(
        "--keep-timing",
        action="store_true",
        help="T0: do not re-measure the T0.2 timing (run task0_timing.py --plot: the figure from results/T0.json)",
    )
    args = parser.parse_args()
    for name, title, scripts, ids in TASKS:
        if args.only and name not in args.only:
            continue
        note = ""
        if args.keep_timing and name == "T0_setup":  # the timing depends on the machine load: keep the stored run
            scripts = [s for s in scripts if not s.startswith("task0_timing.py")]
            scripts.insert(1, "task0_timing.py --plot")
            note = (
                " The T0.2 timing was not re-measured in this build: `task0_timing.py --plot` redraws its figure "
                "from the stored numbers of `results/T0.json`."
            )
        nb = build(name, title, scripts, ids, note)
        path = SOL / f"{name}.ipynb"
        if not args.no_run:
            resources = {"metadata": {"path": str(SOL)}}
            client = NotebookClient(nb, timeout=3600, kernel_name=KERNEL["name"], resources=resources)
            client.execute()
        for cell in nb.cells:  # the scripts print absolute paths: keep the notebooks free of this machine's paths
            for out in cell.get("outputs", []):
                if "text" in out:
                    out["text"] = out["text"].replace(str(REPO), ".")
        nbformat.write(nb, path)
        n_fig = sum(c.source.startswith("show(") for c in nb.cells)
        print(
            f"wrote {path.relative_to(REPO)}: {len(ids)} questions, {n_fig} figure cells"
            + ("" if args.no_run else ", executed")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
