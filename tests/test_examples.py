"""The example scripts: the task-data script stays in step with the stage-1 tasks and the CLI; the flow example runs."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from prob_reco_sbi.flash.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
STAGE1_TASKS = ROOT / "tasks" / "stage1_flash" / "README.md"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_task_data", ROOT / "examples" / "make_task_data.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tasks_outputs(presets, processes) -> list[str]:
    """Output files of the bash block "Data for T1 to T4" of the stage-1 tasks, with the loop variables expanded."""
    text = STAGE1_TASKS.read_text(encoding="utf-8")
    block = text.split("## Data for T1 to T4", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
    outputs = []
    for target in re.findall(r'-o\s+"?([^"\s]+\.h5)"?', block):
        if "process" in target:  # test first: "$process" also starts with "$p"
            outputs += [target.replace("${process}", q).replace("$process", q) for q in processes]
        elif "${p}" in target or "$p" in target:
            outputs += [target.replace("${p}", p).replace("$p", p) for p in presets]
        else:
            outputs.append(target)
    return outputs


def test_task_data_commands_are_valid_cli_commands():
    script = load_script()
    parser = build_parser()
    for command in script.task_commands():
        args = parser.parse_args(command)
        assert args.command == "generate"
        assert script.output_of(command).as_posix().startswith("data/")  # as_posix: "/" also on Windows


def test_task_data_script_matches_the_tasks_sheet():
    script = load_script()
    from_script = [script.output_of(c).as_posix() for c in script.task_commands()]
    assert sorted(from_script) == sorted(tasks_outputs(script.PRESETS, script.PROCESSES))
    assert len(set(from_script)) == len(from_script) == 14


@pytest.mark.slow
def test_flows_quickstart(tmp_path):
    """The flow example runs end to end with a small budget and writes its figure where it is told to."""
    command = [sys.executable, str(ROOT / "examples" / "flows_quickstart.py"), "--steps", "50", "--samples", "2000"]
    run = subprocess.run([*command, "--out", str(tmp_path)], capture_output=True, text=True, timeout=300, check=False)
    assert run.returncode == 0, run.stderr
    assert "KL(p || q) = " in run.stdout
    if importlib.util.find_spec("matplotlib") is not None:  # without matplotlib the script skips the figure
        assert (tmp_path / "flows_quickstart.png").stat().st_size > 0
