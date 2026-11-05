"""Assemble ``solutions/stage1_flash/ANSWERS.md`` from ``answers/T*.md`` in question-ID order, figure names linked.

Run from the repository root (it is also the last step of ``run_all.sh``)::

    python solutions/stage1_flash/make_answers.py

A figure line ``![fig](../figures/<name>.png)`` (one or more images; ``.gif`` for an animation) shows the figure when
the answer file is opened on GitHub and becomes an inline image of ``ANSWERS.md``. Every figure name quoted in an
answer (``t2_2_total_charge`` and the like) or named in a figure line must exist in ``figures/`` as a PNG or a GIF, and
every image path of a figure line must point at an existing file; the script fails otherwise, so that the answers and
the figures cannot drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

from common import FIG, SOL

ANSWERS = SOL / "answers"
OUTPUT = SOL / "ANSWERS.md"
FIGURE_NAME = re.compile(r"`(t\d_\d_[a-z0-9_]+)`")
#: A figure line of an answer: one or more images ``![fig](../figures/<name>.png)`` (``.gif`` for an animation), which
#: GitHub shows when the answer file is opened; the older form ``![fig](name ...)`` with bare names is also read.
FIGURE_MARKER = re.compile(r"^(!\[fig\]\([^)]+\)(?:[ \t]+!\[fig\]\([^)]+\))*)\s*$", re.M)
FIGURE_LINK = re.compile(r"!\[fig\]\(([^)]+)\)")


def figure_names(marker: str) -> list[str]:
    """Return the figure names of a figure line, the file stems: ``../figures/t1_1_radius.png`` -> ``t1_1_radius``."""
    return [Path(target).stem for link in FIGURE_LINK.findall(marker) for target in link.split()]


def question_key(path: Path) -> tuple[int, int]:
    """Sort key of an answer file: (task, question)."""
    match = re.fullmatch(r"T(\d+)\.(\d+)\.md", path.name)
    if match is None:
        raise ValueError(f"unexpected answer file name {path.name}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    """Write ANSWERS.md and check the figure references."""
    files = sorted(ANSWERS.glob("T*.md"), key=question_key)
    figures = {p.stem: p.name for p in FIG.glob("*.png")} | {p.stem: p.name for p in FIG.glob("*.gif")}
    contents, sections, missing = [], [], []
    for path in files:
        text = path.read_text().strip()
        first, body = text.split("\n", 1)
        title = first.lstrip("# ").strip()
        anchor = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")  # the heading anchor of GitHub
        contents.append(f"- [{title}](#{anchor})")
        for name in FIGURE_NAME.findall(text) + [n for m in FIGURE_MARKER.findall(text) for n in figure_names(m)]:
            if name not in figures:
                missing.append((path.name, name))
        for marker in FIGURE_MARKER.findall(text):  # an image path must resolve from answers/, where GitHub reads it
            targets = [t for link in FIGURE_LINK.findall(marker) for t in link.split() if "/" in t]
            missing += [(path.name, t) for t in targets if not (ANSWERS / t).is_file()]
        body = FIGURE_NAME.sub(lambda m: f"[`{m.group(1)}`](figures/{figures.get(m.group(1), m.group(1))})", body)
        body = FIGURE_MARKER.sub(  # a figure line becomes one inline image per figure, linked from this directory
            lambda m: "\n".join(f"![{n}](figures/{figures.get(n, n)})" for n in figure_names(m.group(1))), body
        )
        sections.append(f"## {title}\n{body}")
    if missing:
        raise SystemExit(f"figures named in the answers but missing in {FIG}: {missing}")
    header = (
        "# Reference answers: A Flash in the Dark\n\n"
        "Assembled from `answers/T*.md` in question-ID order by `make_answers.py`. The figures are in `figures/` "
        "(PDF and PNG, linked below), every number comes from `results/*.json`, and `README.md` says which script "
        "makes what.\n\n"
        "## Contents\n\n" + "\n".join(contents) + "\n\n"
    )
    OUTPUT.write_text(header + "\n\n".join(sections) + "\n")
    print(f"wrote {OUTPUT} ({len(files)} answers, {len(figures)} figures available)")


if __name__ == "__main__":
    main()
