"""Make the data sets of the stage-2b tasks (neural posterior estimation) on any system, also on Windows.

The script runs ``flashsim generate`` commands through the entry point of the command line interface, with fixed
seeds, so every student gets the same files. Run it from anywhere inside the repository::

    python examples/make_npe_data.py              # every file; files that exist are kept
    python examples/make_npe_data.py --overwrite  # make every file again
    python examples/make_npe_data.py --dry-run    # only print the commands

It takes a few minutes on a laptop and writes about 0.2 GB to ``data/`` in the repository root:

==========================  ========  ======================================  ===============================
file                        preset    events                                  used in
==========================  ========  ======================================  ===============================
``data/n1_center.h5``       full      60 000 at the center, ``(0, 0)``        N1
``data/n2_absorb_train.h5`` absorb    310 000 (300 000 training, 10 000 val)  N2, N5.3
``data/n2_absorb_test.h5``  absorb    2000                                    N2, N5.3
``data/n3_full_train.h5``   full      310 000 (300 000 training, 10 000 val)  N3, N4 (ganged), N5
``data/n3_full_test.h5``    full      5000                                    N3, N4 (ganged), N5
==========================  ========  ======================================  ===============================

The short budget uses the first 100 000 events of a training file, and the last 10 000 events are the validation set.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

from prob_reco_sbi.flash.cli import main as flashsim

ROOT = Path(__file__).resolve().parents[1]  # the repository root: every path of the tasks is relative to it


def npe_commands() -> list[list[str]]:
    """Return the ``flashsim`` commands of the stage-2b data, in the order of the sections."""
    return [
        ["generate", "--fixed-vertex", "0", "0", "-n", "60000", "--seed", "21", "-o", "data/n1_center.h5"],
        ["generate", "--preset", "absorb", "-n", "310000", "--seed", "22", "-o", "data/n2_absorb_train.h5"],
        ["generate", "--preset", "absorb", "-n", "2000", "--seed", "23", "-o", "data/n2_absorb_test.h5"],
        ["generate", "-n", "310000", "--seed", "24", "-o", "data/n3_full_train.h5"],
        ["generate", "-n", "5000", "--seed", "25", "-o", "data/n3_full_test.h5"],
    ]


def output_of(command: list[str]) -> Path:
    """Return the output file of a ``generate`` command (the value after ``-o``)."""
    return Path(command[command.index("-o") + 1])


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Make the data sets of the stage-2b tasks (N1 to N5) on any system.")
    parser.add_argument("--overwrite", action="store_true", help="make every file again, also those that exist")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads of the simulation (default: torch's)")
    return parser.parse_args()


def main() -> int:
    """Run the commands; return the exit status of the first command that fails, else 0."""
    args = parse_args()
    os.chdir(ROOT)
    commands = npe_commands()
    print(f"{len(commands)} data sets for the stage-2b tasks, written to {ROOT / 'data'}")
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
        extra = ["-q", "--no-progress", *(["--overwrite"] if args.overwrite else [])]
        if args.threads is not None:
            extra += ["--threads", str(args.threads)]
        begin = time.perf_counter()
        status = flashsim([*command, *extra])
        if status != 0:
            print(f"failed with exit status {status}: {line}", file=sys.stderr)
            return status
        print(f"  {time.perf_counter() - begin:.0f} s", flush=True)
    if not args.dry_run:
        print(f"done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
