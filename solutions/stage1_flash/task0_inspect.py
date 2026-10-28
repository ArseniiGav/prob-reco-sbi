"""T0.3 Inspect a data file: datasets, shapes and the stored configuration.

Run from the repository root::

    python solutions/stage1_flash/task0_inspect.py

Uses the file of T0.2 (``flashsim generate -n 10000 --seed 1``, default physics; created here if missing) and a small
file with the optional ``/hits`` and ``/tracks`` groups. Prints ``flashsim info``, the HDF5 tree (shape, type and unit
of every dataset), the root attributes and the stored configuration, and shows that ``load`` gives back the same
configuration object. Everything printed is also stored in ``results/T0.json`` under ``T0.3``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
from common import DATA, REPO, dataset, save_results

from prob_reco_sbi.flash import SimConfig, load, matching_preset, preset

CLI_FILE = DATA / "t0_cli_full_n10000_seed1.h5"


def flashsim(*args: str) -> str:
    """Run the command line tool from the repository root and return its standard output."""
    exe = shutil.which("flashsim")
    command = [exe] if exe else [sys.executable, "-m", "prob_reco_sbi.flash"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([*command, *args], check=True, capture_output=True, text=True, env=env, cwd=REPO)
    return done.stdout


def tree(path: Path) -> dict[str, dict[str, Any]]:
    """Every dataset of an HDF5 file with its shape, type and unit."""
    out: dict[str, dict[str, Any]] = {}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out["/" + name] = {"shape": list(obj.shape), "dtype": str(obj.dtype), "units": obj.attrs.get("units", "")}

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


def root_attributes(path: Path) -> dict[str, Any]:
    """Root attributes (format version, events, configuration, provenance) and the group attributes."""
    with h5py.File(path, "r") as f:
        attrs = {key: f.attrs[key] for key in f.attrs}
        detector = {key: f["detector"].attrs[key] for key in f["detector"].attrs}
        groups = {name: dict(f[name].attrs) for name in f if f[name].attrs}
    out = {
        "format_version": int(attrs["format_version"]),
        "n_events": int(attrs["n_events"]),
        "config": json.loads(attrs["config"]),
        "provenance": json.loads(attrs["provenance"]),
        "detector_attrs": {k: float(v) for k, v in detector.items()},
    }
    if "tracks" in groups:
        out["tracks_attrs"] = {k: json.loads(v) for k, v in groups["tracks"].items()}
    return out


def main() -> None:
    """Describe the T0.2 file and a small file with hits and tracks; store the layout in results/T0.json."""
    relative = str(CLI_FILE.relative_to(REPO))
    if not CLI_FILE.exists():
        flashsim("generate", "-n", "10000", "--seed", "1", "-o", relative, "-q")
    info = flashsim("info", relative)
    print(f"$ flashsim info {relative}\n{info}")

    layout = tree(CLI_FILE)
    attrs = root_attributes(CLI_FILE)
    print("HDF5 tree (dataset, shape, type, unit):")
    for name, meta in layout.items():
        print(f"  {name:<32s} {meta['shape']!s:<12s} {meta['dtype']:<8s} {meta['units']}")
    print(f"\nroot attributes: format_version = {attrs['format_version']}, n_events = {attrs['n_events']}")
    print("stored configuration (attribute 'config', JSON):")
    print(json.dumps(attrs["config"], indent=2))
    print(f"provenance keys: {', '.join(attrs['provenance'])}")

    data = load(CLI_FILE)
    same = data.config == preset("full")
    from_json = SimConfig.from_dict(attrs["config"]) == data.config
    print(f"\nload(): {data!r}; config == preset('full'): {same}; matching_preset: {matching_preset(data.config)}")
    print(f"data.response.charge.shape = {data.response.charge.shape}, data.truth.x.dtype = {data.truth.x.dtype}")

    small = dataset("t0_3_hits_tracks_1k", 1000, "full", seed=22, save_hits=True, save_tracks=3, track_photons=100)
    small_path = DATA / "t0_3_hits_tracks_1k.h5"
    small_layout = {k: v for k, v in tree(small_path).items() if k.startswith(("/hits", "/tracks"))}
    print("\noptional groups (1000 events, --save-hits --save-tracks 3 --track-photons 100):")
    for name, meta in small_layout.items():
        print(f"  {name:<32s} {meta['shape']!s:<12s} {meta['dtype']:<8s} {meta['units']}")
    assert small.hits is not None
    assert small.tracks is not None
    hits_per_event = len(small.hits) / len(small)

    save_results(
        "T0",
        {
            "T0.3": {
                "file": CLI_FILE.name,
                "file_size_MB": round(CLI_FILE.stat().st_size / 1e6, 2),
                "datasets": layout,
                "format_version": attrs["format_version"],
                "n_events": attrs["n_events"],
                "config": attrs["config"],
                "provenance_keys": list(attrs["provenance"]),
                "provenance_command": attrs["provenance"].get("command"),
                "detector_attrs": attrs["detector_attrs"],
                "config_equals_full_preset": same,
                "config_round_trip_from_json": from_json,
                "flashsim_info": info.splitlines(),
                "optional_groups_1k_events": small_layout,
                "optional_file_size_MB": round(small_path.stat().st_size / 1e6, 2),
                "hits_per_event_1k": round(hits_per_event, 1),
                "tracks_point_types": root_attributes(small_path).get("tracks_attrs", {}),
            }
        },
    )


if __name__ == "__main__":
    main()
