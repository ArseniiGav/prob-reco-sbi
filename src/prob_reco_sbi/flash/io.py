"""Reading, writing and merging simulated data: one HDF5 file per run.

File layout (format version 1; ``N`` events, ``M`` sensors)::

    /            attrs: format_version, n_events, config (JSON), provenance (JSON)
    /truth       event_id (int64); x, y [m], energy [MeV], t0 [ns] (float32)              shape [N]
    /response    charge (int32); time_mean, time_first [ns] (float32, NaN = no hit)        shape [N, M]
    /photons     n_emitted, n_detected, n_absorbed_bulk, n_absorbed_wall, n_absorbed_sensor,
                 n_lost, n_detected_direct, n_detected_scattered, n_detected_reflected,
                 n_noise (int32)                                                            shape [N]
    /hits        optional: event_id (int64), sensor (int32), time [ns] (float32),
                 n_scatter, n_reflect (int32, -1 for noise), path_length [m] (float32,
                 NaN for noise), is_noise (bool)                                            shape [n_hits]
    /tracks      optional: event_id (int64), photon (int32)                                 shape [n_tracks]
                           offsets (int64)                                                  shape [n_tracks + 1]
                           x, y, path_length [m] (float32), kind (int8)                     shape [n_points]
                 attrs: point_types (JSON, name -> code)
    /detector    sensor_x, sensor_y [m], sensor_phi [rad] (float64)                         shape [M]
                 attrs: radius, n_sensors, sensor_width, photocoverage

Every dataset has a ``units`` attribute (empty for counts and codes). Files are written to ``<name>.part`` first and
renamed when complete, so an interrupted job never leaves a truncated file under the final name.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import SimConfig, matching_preset
from .result import POINT_TYPES, Hits, PhotonCounts, Response, SimResult, Tracks, Truth

__all__ = ["FORMAT_VERSION", "describe", "load", "merge", "save"]

FORMAT_VERSION = 1

_GROUPS: dict[str, type[Any]] = {
    "truth": Truth,
    "response": Response,
    "photons": PhotonCounts,
    "hits": Hits,
    "tracks": Tracks,
}
_UNITS = {
    "x": "m",
    "y": "m",
    "energy": "MeV",
    "t0": "ns",
    "time_mean": "ns",
    "time_first": "ns",
    "time": "ns",
    "path_length": "m",
    "sensor_x": "m",
    "sensor_y": "m",
    "sensor_phi": "rad",
}
_COPY_BLOCK = 1 << 22  # elements copied at a time when merging


def save(result: SimResult, path: str | os.PathLike[str], *, overwrite: bool = True) -> Path:
    """Write a simulation result to an HDF5 file.

    Parameters
    ----------
    result
        Output of :func:`~prob_reco_sbi.flash.generate`, :func:`~prob_reco_sbi.flash.simulate` or :func:`load`.
    path
        Output file; missing parent directories are created.
    overwrite
        Replace an existing file (otherwise raise :class:`FileExistsError`).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with h5py.File(part, "w") as f:
            _write_header(f, result.config, result.provenance, len(result))
            for name in _GROUPS:
                container = getattr(result, name)
                if container is not None:
                    group = f.create_group(name)
                    for key, values in container.arrays().items():
                        _create(group, key, values)
            if result.tracks is not None:
                f["tracks"].attrs["point_types"] = json.dumps(POINT_TYPES)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
    return path


def load(path: str | os.PathLike[str], *, hits: bool = True, tracks: bool = True) -> SimResult:
    """Read a data file written by :func:`save`, :func:`merge` or ``flashsim``.

    Parameters
    ----------
    path
        HDF5 file.
    hits, tracks
        Read the optional ``/hits`` and ``/tracks`` groups if present (set ``False`` to save memory).

    Returns
    -------
    SimResult
        The stored events, with the configuration and provenance.
    """
    with _open(path) as f:
        _check_format(f, path)
        groups: dict[str, Any] = {}
        for name, cls in _GROUPS.items():
            wanted = {"hits": hits, "tracks": tracks}.get(name, True)
            if name in f and wanted:
                groups[name] = cls(**{fld.name: f[name][fld.name][()] for fld in fields(cls)})
        return SimResult(
            config=SimConfig.from_json(f.attrs["config"]),
            provenance=json.loads(f.attrs["provenance"]),
            **groups,
        )


def merge(paths: Sequence[str | os.PathLike[str]], out: str | os.PathLike[str], *, overwrite: bool = True) -> Path:
    """Concatenate data files of the same configuration (e.g. cluster shards with different seeds).

    The inputs must have the same configuration and the same optional groups, distinct seeds and distinct event ids
    (use ``first_event_id`` / ``--first-event-id`` to give shards disjoint ranges). Data are copied block by block,
    so the inputs need not fit in memory together.

    Parameters
    ----------
    paths
        Input files, in the order in which their events are concatenated.
    out
        Output file.
    overwrite
        Replace an existing output file.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    sources, target = [Path(p) for p in paths], Path(out)
    if not sources:
        raise ValueError("no input files")
    if target.resolve() in {p.resolve() for p in sources}:
        raise ValueError(f"the output {target} is also an input")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (pass overwrite=True to replace it)")
    inputs = [_inspect(p) for p in sources]
    _check_mergeable(inputs)
    config = SimConfig.from_json(inputs[0]["config"])
    from .. import __version__

    provenance = {
        "merged_from": [{"file": item["path"].name, **item["provenance"]} for item in inputs],
        "package_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with h5py.File(part, "w") as dst:
            n_events = sum(item["n_events"] for item in inputs)
            _write_header(dst, config, provenance, n_events)
            for name in inputs[0]["groups"]:
                _merge_group(dst, name, [item["path"] for item in inputs])
            if "tracks" in dst:
                dst["tracks"].attrs["point_types"] = json.dumps(POINT_TYPES)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target


def describe(path: str | os.PathLike[str]) -> str:
    """Human-readable summary of a data file: datasets, configuration, provenance and a few numbers."""
    path = Path(path)
    with _open(path) as f:
        _check_format(f, path)
        lines = [f"file: {path}  ({path.stat().st_size / 1e6:.2f} MB)"]
        config = SimConfig.from_json(f.attrs["config"])
        provenance = json.loads(f.attrs["provenance"])
        n = int(f.attrs["n_events"])
        lines.append(f"format version {f.attrs['format_version']}, {n} events, {config.geometry.n_sensors} sensors")
        lines.append("")
        lines.append(f"configuration (preset: {matching_preset(config) or 'modified'}):")
        lines += ["  " + line for line in config.summary().splitlines()]
        lines.append("")
        lines.append("provenance:")
        for key, value in provenance.items():
            if key == "merged_from":
                lines.append(f"  merged from {len(value)} files: {', '.join(s['file'] for s in value)}")
                lines.append(f"  seeds: {list(_seeds(provenance))}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append("datasets:")

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                unit = obj.attrs.get("units", "")
                lines.append(f"  /{name:<30s} {obj.dtype!s:<8s} {obj.shape!s:<16s} {unit}".rstrip())

        f.visititems(visit)
        photons = f["photons"]
        emitted = photons["n_emitted"][()].sum()
        if emitted > 0:
            lines.append("")
            lines.append("photon bookkeeping (all events):")
            for key in ("n_detected", "n_absorbed_bulk", "n_absorbed_wall", "n_absorbed_sensor", "n_lost"):
                lines.append(f"  {key[2:]:<16s} {photons[key][()].sum() / emitted:8.4f} of the emitted photons")
            charge = f["response/charge"][()]
            lines.append(
                f"  mean total charge {charge.sum(axis=1).mean():.1f} p.e. per event, "
                f"{100 * np.mean(charge == 0):.1f} % of the sensors empty"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# Helpers


def _write_header(f: h5py.File, config: SimConfig, provenance: dict[str, Any], n_events: int) -> None:
    from .geometry import sensor_phi, sensor_positions

    f.attrs["format_version"] = FORMAT_VERSION
    f.attrs["n_events"] = n_events
    f.attrs["config"] = config.to_json()
    f.attrs["provenance"] = json.dumps(provenance)
    detector = f.create_group("detector")
    g = config.geometry
    x, y = sensor_positions(g)
    for key, values in (("sensor_x", x), ("sensor_y", y), ("sensor_phi", sensor_phi(g))):
        _create(detector, key, values)
    detector.attrs.update(
        radius=g.radius, n_sensors=g.n_sensors, sensor_width=g.sensor_width, photocoverage=g.photocoverage
    )


def _create(group: h5py.Group, name: str, values: np.ndarray) -> h5py.Dataset:
    values = np.asarray(values)
    options: dict[str, Any] = {}
    if values.size > 1024:
        options = {"compression": "gzip", "compression_opts": 4, "shuffle": True, "chunks": True}
    dataset = group.create_dataset(name, data=values, **options)
    dataset.attrs["units"] = _UNITS.get(name, "")
    return dataset


def _open(path: str | os.PathLike[str]) -> h5py.File:
    """Open a file for reading; a file that is not HDF5 (or a directory) gives a clear ``ValueError``."""
    try:
        return h5py.File(path, "r")
    except FileNotFoundError:
        raise
    except OSError as error:
        raise ValueError(f"{path} is not a readable HDF5 file ({error})") from error


def _check_format(f: h5py.File, path: str | os.PathLike[str]) -> None:
    version = f.attrs.get("format_version")
    if version is None:
        raise ValueError(f"{path} is not a flash data file (no format_version attribute)")
    if int(version) > FORMAT_VERSION:
        raise ValueError(f"{path} has format version {version}; this package reads up to {FORMAT_VERSION}")


def _inspect(path: Path) -> dict[str, Any]:
    with _open(path) as f:
        _check_format(f, path)
        return {
            "path": path,
            "config": f.attrs["config"],
            "provenance": json.loads(f.attrs["provenance"]),
            "n_events": int(f.attrs["n_events"]),
            "groups": [name for name in _GROUPS if name in f],
            "event_id": f["truth/event_id"][()],
        }


def _seeds(provenance: dict[str, Any]) -> Iterable[Any]:
    if "merged_from" in provenance:
        for source in provenance["merged_from"]:
            yield from _seeds(source)
    else:
        yield provenance.get("seed")


def _check_mergeable(inputs: list[dict[str, Any]]) -> None:
    first = inputs[0]
    reference = SimConfig.from_json(first["config"])
    for item in inputs[1:]:
        if SimConfig.from_json(item["config"]) != reference:
            raise ValueError(f"{item['path']} has a different configuration than {first['path']}")
        if item["groups"] != first["groups"]:
            raise ValueError(
                f"{item['path']} has the groups {item['groups']}, {first['path']} has {first['groups']}; "
                "generate all shards with the same --save-hits / --save-tracks options"
            )
    seeds = [seed for item in inputs for seed in _seeds(item["provenance"])]
    repeated = sorted({s for s in seeds if seeds.count(s) > 1 and s is not None})
    if repeated:
        raise ValueError(f"seed(s) {repeated} occur in more than one input: those events would be identical copies")
    ids = np.concatenate([item["event_id"] for item in inputs])
    if len(np.unique(ids)) != len(ids):
        raise ValueError(
            "the inputs share event ids; give each shard its own range with first_event_id / --first-event-id"
        )


def _merge_group(dst: h5py.File, name: str, paths: list[Path]) -> None:
    """Concatenate one group of all inputs into ``dst`` (track offsets are shifted)."""
    sources = [h5py.File(p, "r") for p in paths]
    try:
        group = dst.create_group(name)
        for key in sources[0][name]:
            parts = [src[name][key] for src in sources]
            if name == "tracks" and key == "offsets":
                _merge_offsets(group, parts)
                continue
            total = sum(part.shape[0] for part in parts)
            shape = (total, *parts[0].shape[1:])
            options: dict[str, Any] = {}
            if total * int(np.prod(shape[1:], dtype=np.int64)) > 1024:
                options = {"compression": "gzip", "compression_opts": 4, "shuffle": True, "chunks": True}
            out = group.create_dataset(key, shape=shape, dtype=parts[0].dtype, **options)
            out.attrs["units"] = _UNITS.get(key, "")
            start = 0
            for part in parts:
                n = part.shape[0]
                rows = max(1, _COPY_BLOCK // max(1, int(np.prod(part.shape[1:], dtype=np.int64))))
                for a in range(0, n, rows):
                    b = min(n, a + rows)
                    out[start + a : start + b] = part[a:b]
                start += n
    finally:
        for src in sources:
            src.close()


def _merge_offsets(group: h5py.Group, parts: list[h5py.Dataset]) -> None:
    pieces, shift = [], 0
    for part in parts:
        offsets = part[()]
        pieces.append(offsets[:-1] + shift)
        shift += int(offsets[-1])
    pieces.append(np.array([shift], dtype=np.int64))
    _create(group, "offsets", np.concatenate(pieces).astype(np.int64))
