import subprocess
import sys

import numpy as np
import pytest

from prob_reco_sbi.flash import load, preset
from prob_reco_sbi.flash.cli import build_parser, default_output, main
from prob_reco_sbi.flash.config import config_fields


def run(*args, cwd=None):
    """Run ``python -m prob_reco_sbi.flash`` in a subprocess (CPU only)."""
    return subprocess.run(
        [sys.executable, "-m", "prob_reco_sbi.flash", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
        check=False,
    )


def test_cli_end_to_end(tmp_path):
    """generate -> info -> merge through the command line."""
    common = ["generate", "--device", "cpu", "-n", "20", "--save-hits", "--save-tracks", "2", "-q"]
    a = run(*common, "--seed", "1", "-o", str(tmp_path / "a.h5"))
    assert a.returncode == 0, a.stderr
    b = run(*common, "--seed", "2", "--first-event-id", "20", "-o", str(tmp_path / "b.h5"))
    assert b.returncode == 0, b.stderr
    info = run("info", str(tmp_path / "a.h5"))
    assert info.returncode == 0, info.stderr
    assert "20 events" in info.stdout
    assert "/hits/time" in info.stdout
    merged = run("merge", "-o", str(tmp_path / "ab.h5"), str(tmp_path / "a.h5"), str(tmp_path / "b.h5"))
    assert merged.returncode == 0, merged.stderr
    assert len(load(tmp_path / "ab.h5")) == 40
    again = run(*common, "--seed", "1", "-o", str(tmp_path / "a.h5"))
    assert again.returncode == 1
    assert "exists" in again.stderr
    # the same command in another fresh process gives the same data (the files differ only in the provenance times)
    repeat = run(*common, "--seed", "1", "--threads", "16", "-o", str(tmp_path / "a_again.h5"))
    assert repeat.returncode == 0, repeat.stderr
    first, second = load(tmp_path / "a.h5"), load(tmp_path / "a_again.h5")
    for group in ("truth", "response", "photons", "hits", "tracks"):
        for name, values in getattr(first, group).arrays().items():
            np.testing.assert_array_equal(values, getattr(second, group).arrays()[name], err_msg=f"{group}/{name}")


def test_presets_and_help():
    out = run("presets")
    assert out.returncode == 0
    assert "absorb-scatter" in out.stdout
    help_text = run("generate", "--help").stdout
    for _, f in config_fields():
        assert "--" + f.name.replace("_", "-") in help_text, f.name
    assert "--no-reflection" in help_text


def test_overrides_reach_the_stored_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = ["generate", "--preset", "straight", "--reflection", "--reflectivity", "0.8", "--device", "cpu"]
    argv += ["-n", "8", "--fixed-vertex", "0.1", "0.2", "--n-sensors", "16", "--no-progress", "-q"]
    assert main(argv) == 0
    expected = preset("straight", reflection=True, reflectivity=0.8, fixed_vertex=(0.1, 0.2), n_sensors=16)
    out = default_output("straight", expected, 8, 0)
    assert "straight-custom-" in out.name
    data = load(out)
    assert data.config == expected
    assert data.provenance["preset"] == "straight"
    assert data.provenance["command"].startswith("flashsim generate --preset straight")
    assert data.response.charge.shape == (8, 16)


def test_default_output_name():
    assert default_output("full", preset("full"), 10000, 3).as_posix() == "data/flash_full_n10000_seed3.h5"


def test_invalid_values_give_an_error_message(tmp_path, capsys):
    assert main(["generate", "--reflectivity", "2", "-o", str(tmp_path / "x.h5"), "--device", "cpu", "-q"]) == 1
    assert "reflectivity must be in [0, 1]" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        build_parser().parse_args(["generate", "--preset", "nonsense"])
    assert main(["generate", "-n", "2", "-o", str(tmp_path / "y.h5"), "--device", "gpu", "-q"]) == 1
    assert "unsupported device 'gpu'" in capsys.readouterr().err
    assert main(["generate", "-n", "2", "-o", str(tmp_path / "z.h5"), "--device", "cpu", "--save-tracks", "-1"]) == 1
    assert "save_tracks must be >= 0" in capsys.readouterr().err
    assert not (tmp_path / "y.h5").exists()


def test_unreadable_inputs_give_an_error_message(tmp_path, capsys):
    text = tmp_path / "notes.txt"
    text.write_text("not an HDF5 file\n")
    assert main(["info", str(text)]) == 1
    assert "not a readable HDF5 file" in capsys.readouterr().err
    assert main(["info", str(tmp_path)]) == 1
    assert "not a readable HDF5 file" in capsys.readouterr().err
    assert main(["info", str(tmp_path / "missing.h5")]) == 1
    assert "missing.h5" in capsys.readouterr().err
    assert main(["merge", "-o", str(tmp_path / "m.h5"), str(text)]) == 1
    assert "not a readable HDF5 file" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # the output directory is checked before the simulation
        main(["generate", "-n", "2", "--device", "cpu", "-o", str(text / "sub" / "x.h5"), "-q"])
    assert "cannot write" in capsys.readouterr().err
