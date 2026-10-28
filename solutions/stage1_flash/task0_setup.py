"""T0.1 Set-up: run the fast test suite and record the software versions.

Run from the repository root::

    python solutions/stage1_flash/task0_setup.py

The students' steps (TASKS.md T0.1) are: clone, create a virtual environment, install the CPU build of PyTorch,
``python -m pip install -e ".[analysis,dev]"`` and ``python -m pytest``. This script repeats the last step in the
current environment (``python -m pytest -q`` from the repository root, without writing a pytest cache or bytecode) and
stores the outcome and the versions in ``results/T0.json``. It also checks whether the ``analysis`` extra alone would
provide ``pytest`` (it does not: ``pytest`` comes with the ``dev`` extra, which is why TASKS.md installs both).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, requires

import h5py
import matplotlib
import numpy as np
import torch
from common import REPO, save_results
from packaging.requirements import Requirement

import prob_reco_sbi


def dependency_closure(names: list[str]) -> set[str]:
    """Names of all installed distributions that ``pip install`` of ``names`` pulls in (without extras)."""
    seen: set[str] = set()
    stack = list(names)
    while stack:
        name = stack.pop().lower().replace("_", "-")
        if name in seen:
            continue
        seen.add(name)
        try:
            reqs = requires(name) or []
        except PackageNotFoundError:
            continue
        for text in reqs:
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                stack.append(req.name)
    return seen


def main() -> None:
    """Run the fast test suite and store its outcome and the software versions."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - start
    summary = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr.strip()
    counts = {key: int(value) for value, key in re.findall(r"(\d+) (passed|failed|skipped|deselected|error)", summary)}
    reported = re.search(r"in ([\d.]+)s", summary)

    # pyproject: analysis = [ipykernel, matplotlib, pandas]; the package itself needs numpy, torch, h5py
    closure = dependency_closure(["ipykernel", "matplotlib", "pandas", "numpy", "torch", "h5py"])

    values = {
        "pytest_command": "python -m pytest -q (from the repository root; fast suite, -m 'not slow' is the default)",
        "pytest_exit_code": done.returncode,
        "pytest_summary": summary,
        "pytest_counts": counts,
        "pytest_reported_seconds": float(reported.group(1)) if reported else None,
        "pytest_wall_seconds": round(wall, 1),
        "pytest_from_analysis_extra": "pytest" in closure,
        "versions": {
            "prob_reco_sbi": prob_reco_sbi.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    path = save_results("T0", {"T0.1": values})
    print(f"pytest: {summary}  (exit code {done.returncode}, {wall:.1f} s wall)")
    print(f"pytest installed by the analysis extra alone: {values['pytest_from_analysis_extra']}")
    print(f"versions: {values['versions']}")
    print(f"results -> {path}")


if __name__ == "__main__":
    main()
