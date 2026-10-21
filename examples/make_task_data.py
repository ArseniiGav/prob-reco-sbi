"""Make the data sets of TASKS.md ("Data for T1 to T4") on any operating system, also on Windows without a Unix shell.

The script runs the same ``flashsim generate`` commands as the bash block in TASKS.md, through the entry point of the
command line interface, so it writes the same files. Run it from anywhere inside the repository::

    python examples/make_task_data.py              # every file of the block; files that exist are kept
    python examples/make_task_data.py --overwrite  # make every file again
    python examples/make_task_data.py --dry-run    # only print the commands

It takes a few minutes on a laptop and writes about 0.4 GB to ``data/`` in the repository root.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

from prob_reco_sbi.flash.cli import main as flashsim

ROOT = Path(__file__).resolve().parents[1]  # the repository root: every path of TASKS.md is relative to it
PRESETS = ("straight", "absorb", "absorb-scatter", "full")
PROCESSES = ("scattering", "reflection")  # absorption alone on top of straight light is the preset absorb


def task_commands() -> list[list[str]]:
    """Return the ``flashsim`` commands of the data block in TASKS.md, in the same order."""
    commands = [["generate", "-n", "50000", "--seed", "5", "-o", "data/t2_full.h5"]]  # T1 and T2: the default model
    for p in PRESETS:  # T3: the physics ladder, the same events for every preset, and small files with every hit
        commands.append(["generate", "--preset", p, "-n", "20000", "--seed", "6", "-o", f"data/t3_{p}.h5"])
        commands.append(
            ["generate", "--preset", p, "-n", "2000", "--seed", "7", "--save-hits", "-o", f"data/t3_{p}_hits.h5"]
        )
    for process in PROCESSES:  # T3: one process alone on top of straight light
        straight = ["generate", "--preset", "straight", f"--{process}"]
        commands.append([*straight, "-n", "20000", "--seed", "6", "-o", f"data/t3_straight_{process}.h5"])
        commands.append(
            [*straight, "-n", "2000", "--seed", "7", "--save-hits", "-o", f"data/t3_straight_{process}_hits.h5"]
        )
    # T4: a few events with every hit and the paths of 200 photons of each of the first five events
    tracks = ["--save-hits", "--save-tracks", "5", "--track-photons", "200"]
    commands.append(["generate", "-n", "20", "--seed", "9", *tracks, "-o", "data/t4_events.h5"])
    return commands


def output_of(command: list[str]) -> Path:
    """Return the output file of a ``generate`` command (the value after ``-o``)."""
    return Path(command[command.index("-o") + 1])


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Make the data sets of TASKS.md (T1 to T4) on any operating system.")
    parser.add_argument("--overwrite", action="store_true", help="make every file again, also those that exist")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    return parser.parse_args()


def main() -> int:
    """Run the commands of the data block; return the exit status of the first command that fails, else 0."""
    args = parse_args()
    os.chdir(ROOT)
    commands = task_commands()
    print(f"{len(commands)} data sets for TASKS.md, written to {ROOT / 'data'}")
    start = time.perf_counter()
    for command in commands:
        line = "flashsim " + shlex.join(command)
        if args.dry_run:
            print(line)
            continue
        if output_of(command).exists() and not args.overwrite:
            print(f"exists, kept: {output_of(command)}")
            continue
        print(line, flush=True)
        status = flashsim([*command, "-q", "--no-progress", *(["--overwrite"] if args.overwrite else [])])
        if status != 0:
            print(f"failed with exit status {status}: {line}", file=sys.stderr)
            return status
    if not args.dry_run:
        print(f"done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
