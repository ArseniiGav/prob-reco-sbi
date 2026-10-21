"""Event simulation: sample the prior, transport the photons of every event and read out the sensors.

:func:`generate` draws the event parameters from the prior (:func:`sample_prior`) and calls :func:`simulate`, which
works for any given events. One simulation runs these steps::

    1. photons per event         N ~ Poisson(Y E)  (or N = round(Y E))
    2. photon transport          transport.transport, in chunks of about photons_per_chunk photons
    3. time of detected photons  t = t0 + path / c_n + N(0, sigma_t^2)
    4. dark noise                Poisson(dark_rate * window) hits per event and sensor, times ~ U(0, window)
    5. readout per sensor        charge n_j, mean time tau_j and first time (NaN where n_j = 0)

Random numbers come from independent streams derived from the seed with :class:`numpy.random.SeedSequence`: a NumPy
stream for what is drawn per event (prior, t0, number of photons, dark noise), which is therefore the same on every
device and for every chunking, and a :class:`torch.Generator` on the compute device for what is drawn per photon
(directions, free paths, interactions, time jitter). All output arrays are identical for the same seed, device type,
chunking, PyTorch build and hardware (see :func:`~prob_reco_sbi.flash.transport.warm_up_cpu_kernels` for the CPU
detail that keeps the first run of a process identical to later ones).
"""

from __future__ import annotations

import functools
import itertools
import logging
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from .config import SimConfig
from .result import Hits, PhotonCounts, Response, SimResult, Tracks, Truth
from .transport import Interaction, TransportResult, transport

__all__ = ["DEFAULT_PHOTONS_PER_CHUNK", "generate", "resolve_device", "sample_prior", "simulate"]

log = logging.getLogger(__name__)

#: Default number of photons transported at once, per device type (sets the memory use: about 200 bytes per photon).
DEFAULT_PHOTONS_PER_CHUNK = {"cpu": 2**21, "cuda": 2**23}

#: Hit times are summed per sensor as integers in units of 2**-20 ns (about 1 fs): integer addition does not depend
#: on the order, so the parallel sums on a GPU are bit-identical from run to run (float sums are not).
TICKS_PER_NS = 2**20

_N_CODES = len(Interaction)


# ----------------------------------------------------------------------------------------------------------------------
# Public API


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Turn ``"auto"`` (CUDA if available, else CPU), ``"cpu"``, ``"cuda"`` or ``"cuda:N"`` into a torch device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and not re.fullmatch(r"(cpu|cuda(:\d+)?)", device.strip()):
        raise ValueError(f"unsupported device {device!r}: use 'auto', 'cpu', 'cuda' or 'cuda:N'")
    dev = torch.device(device)
    if dev.type not in ("cpu", "cuda"):
        raise ValueError(f"unsupported device {device!r}: use 'auto', 'cpu', 'cuda' or 'cuda:N'")
    if dev.type == "cuda" and not torch.cuda.is_available():
        if torch.version.cuda is None:
            raise RuntimeError(
                f"device 'cuda' requested, but the installed PyTorch ({torch.__version__}) is a CPU-only build: "
                "install a CUDA build (see 'CPU or GPU' in README.md) or use device='cpu'"
            )
        raise RuntimeError(
            "device 'cuda' requested, but CUDA is not available (no usable GPU or driver; use device='cpu' or 'auto')"
        )
    return dev


def sample_prior(
    n: int, config: SimConfig | None = None, rng: np.random.Generator | int | None = None
) -> dict[str, np.ndarray]:
    """Draw event parameters from the prior.

    The vertex is uniform in the disc ``r <= r_max`` (the same density per unit area everywhere), the energy and t0 are
    uniform in their ranges. Fixed values in the prior replace the corresponding distributions (the random numbers are
    drawn anyway, so the other parameters do not change).

    Parameters
    ----------
    n
        Number of events.
    config
        Configuration whose ``prior`` is used (default: the ``"full"`` preset).
    rng
        NumPy generator, or an integer seed (then the parameters equal those of ``generate(n, seed=rng)``), or
        ``None`` for fresh entropy.

    Returns
    -------
    dict
        Float32 arrays ``"x"``, ``"y"`` [m], ``"energy"`` [MeV] and ``"t0"`` [ns], each of length ``n``.
    """
    prior = (config or SimConfig()).prior
    if not isinstance(rng, np.random.Generator):
        rng = _streams(rng)[0] if rng is not None else np.random.default_rng()
    r = prior.r_max * np.sqrt(rng.random(n))
    phi = 2 * np.pi * rng.random(n)
    energy = rng.uniform(prior.energy_min, prior.energy_max, n)
    t0 = rng.uniform(prior.t0_min, prior.t0_max, n)
    x, y = r * np.cos(phi), r * np.sin(phi)
    if prior.fixed_vertex is not None:
        x = np.full(n, prior.fixed_vertex[0])
        y = np.full(n, prior.fixed_vertex[1])
    if prior.fixed_energy is not None:
        energy = np.full(n, prior.fixed_energy)
    return {name: v.astype(np.float32) for name, v in (("x", x), ("y", y), ("energy", energy), ("t0", t0))}


def generate(
    n_events: int,
    *,
    config: SimConfig | None = None,
    seed: int = 0,
    device: str | torch.device | None = "auto",
    save_hits: bool = False,
    save_tracks: int = 0,
    track_photons: int | None = 100,
    photons_per_chunk: int | None = None,
    first_event_id: int = 0,
    progress: bool = False,
) -> SimResult:
    """Simulate ``n_events`` events with parameters drawn from the prior.

    Parameters
    ----------
    n_events
        Number of events.
    config
        Configuration (default: the ``"full"`` preset); see :func:`~prob_reco_sbi.flash.preset`.
    seed
        Seed of all random numbers (non-negative integer).
    device, save_hits, save_tracks, track_photons, photons_per_chunk, first_event_id, progress
        As for :func:`simulate`.

    Returns
    -------
    SimResult
        Truth, response, photon bookkeeping and, if requested, hits and tracks.

    Examples
    --------
    >>> from prob_reco_sbi.flash import generate, preset
    >>> data = generate(1000, config=preset("full"), seed=1)  # doctest: +SKIP
    >>> data.response.charge.shape  # doctest: +SKIP
    (1000, 32)
    """
    config = config or SimConfig()
    _check_count(n_events)
    theta = sample_prior(n_events, config, _streams(_check_seed(seed))[0])
    return simulate(
        theta["x"],
        theta["y"],
        theta["energy"],
        theta["t0"],
        config=config,
        seed=seed,
        device=device,
        save_hits=save_hits,
        save_tracks=save_tracks,
        track_photons=track_photons,
        photons_per_chunk=photons_per_chunk,
        first_event_id=first_event_id,
        progress=progress,
    )


def simulate(
    x: ArrayLike,
    y: ArrayLike,
    energy: ArrayLike,
    t0: ArrayLike | None = None,
    *,
    config: SimConfig | None = None,
    seed: int = 0,
    device: str | torch.device | None = "auto",
    save_hits: bool = False,
    save_tracks: int = 0,
    track_photons: int | None = 100,
    photons_per_chunk: int | None = None,
    first_event_id: int = 0,
    progress: bool = False,
) -> SimResult:
    """Simulate given events.

    Parameters
    ----------
    x, y
        Vertex coordinates [m] (scalars or arrays of the same length; inside the detector).
    energy
        Event energies [MeV].
    t0
        Emission times [ns]; ``None`` draws them from the prior's t0 range.
    config
        Configuration (default: the ``"full"`` preset). Its prior is only used for ``t0`` when ``t0`` is ``None``.
    seed
        Seed of all random numbers (non-negative integer).
    device
        ``"auto"`` (CUDA if available, else CPU), ``"cpu"`` or ``"cuda"``.
    save_hits
        Keep every hit (detected photons and dark-noise hits) in ``result.hits``.
    save_tracks
        Record the photon paths of the first ``save_tracks`` events in ``result.tracks``.
    track_photons
        Record at most this many photons per recorded event (``None``: all of them).
    photons_per_chunk
        Approximate number of photons transported at once (memory control); default from
        :data:`DEFAULT_PHOTONS_PER_CHUNK`. The output depends on it through the order of the random numbers.
    first_event_id
        Event number of the first event (use different ranges for shards that will be merged).
    progress
        Report the progress on the standard error stream.

    Returns
    -------
    SimResult
        Truth, response, photon bookkeeping and, if requested, hits and tracks.
    """
    start = time.perf_counter()
    config = config or SimConfig()
    seed = _check_seed(seed)
    save_hits, save_tracks, track_photons, first_event_id = _check_options(
        save_hits, save_tracks, track_photons, first_event_id
    )
    # copies, so that the result never shares memory with the caller's arrays
    x, y, energy = (np.array(v, dtype=np.float32).ravel() for v in np.broadcast_arrays(x, y, energy))
    n = _check_count(x.size)
    _check_events(x, y, energy, config)
    _, event_rng, photon_seed = _streams(seed)
    if t0 is None:
        t0 = event_rng.uniform(config.prior.t0_min, config.prior.t0_max, n)
    t_emit = np.broadcast_to(np.asarray(t0, dtype=np.float32).ravel(), (n,)).copy()
    if not np.all(np.isfinite(t_emit)):
        raise ValueError("t0 must be finite")
    n_photons = _photon_numbers(energy, config, event_rng)
    noise = _dark_noise(n, config, event_rng)

    dev = resolve_device(device)
    generator = torch.Generator(device=dev)
    generator.manual_seed(photon_seed)
    budget = DEFAULT_PHOTONS_PER_CHUNK[dev.type] if photons_per_chunk is None else _check_int(photons_per_chunk, 1)
    chunks = _plan_chunks(n_photons, budget)
    log.info("simulating %d events (%.3g photons) on %s in %d chunk(s)", n, n_photons.sum(), dev, len(chunks))

    collector = _Collector(n, config.geometry.n_sensors, save_hits)
    bar = _Progress(n, enabled=progress)
    with torch.inference_mode():
        for a, b in chunks:
            fates = transport(
                torch.as_tensor(x[a:b], device=dev),
                torch.as_tensor(y[a:b], device=dev),
                torch.as_tensor(n_photons[a:b], device=dev),
                config,
                generator,
                track_events=max(0, min(save_tracks - a, b - a)),
                track_photons=track_photons,
            )
            t0_chunk = torch.as_tensor(t_emit[a:b], device=dev)
            collector.add(a, _detect(fates, t0_chunk, config, generator, save_hits))
            if fates.tracks is not None:
                collector.add_tracks(a, _tracks(fates, torch.as_tensor(n_photons[a:b], device=dev)))
            bar.update(b)
    bar.close()

    event_id = first_event_id + np.arange(n, dtype=np.int64)
    result = SimResult(
        truth=Truth(event_id=event_id, x=x, y=y, energy=energy, t0=t_emit),
        response=collector.response(noise, config),
        photons=collector.photon_counts(n_photons, noise),
        config=config,
        hits=collector.hits(noise, event_id) if save_hits else None,
        tracks=collector.tracks(event_id) if save_tracks > 0 else None,
    )
    result.provenance = _provenance(
        seed=seed,
        device=dev,
        photons_per_chunk=budget,
        n_chunks=len(chunks),
        first_event_id=first_event_id,
        save_hits=save_hits,
        save_tracks=save_tracks,
        track_photons=track_photons,
        wall_time=time.perf_counter() - start,
    )
    log.info("done in %.1f s (%.0f events/s)", result.provenance["wall_time_s"], n / result.provenance["wall_time_s"])
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Steps per event


def _streams(seed: int | None) -> tuple[np.random.Generator, np.random.Generator, int]:
    """Independent random streams of one seed: prior (NumPy), per-event quantities (NumPy), photons (torch seed)."""
    prior, events, photons = np.random.SeedSequence(seed).spawn(3)
    photon_seed = int(photons.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))
    return np.random.default_rng(prior), np.random.default_rng(events), photon_seed


def _photon_numbers(energy: np.ndarray, config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Photons per event: N ~ Poisson(Y E), or N = round(Y E) for fixed photon statistics."""
    mean = config.optics.light_yield * energy.astype(np.float64)
    if config.optics.photon_statistics == "fixed":
        return np.rint(mean).astype(np.int64)
    return rng.poisson(mean).astype(np.int64)


def _dark_noise(n_events: int, config: SimConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Dark-noise hits: Poisson(rate * window) per event and sensor, times uniform in the readout window."""
    m = config.geometry.n_sensors
    mean = config.readout.mean_noise_hits
    counts = rng.poisson(mean, size=(n_events, m)) if mean > 0 else np.zeros((n_events, m), dtype=np.int64)
    flat = np.repeat(np.arange(n_events * m), counts.ravel())  # one entry per hit: event * m + sensor
    times = rng.uniform(0.0, config.readout.readout_window, flat.size)
    return {"event": flat // m, "sensor": flat % m, "time": times, "counts": counts}


def _plan_chunks(n_photons: np.ndarray, budget: int) -> list[tuple[int, int]]:
    """Consecutive event ranges ``(start, stop)`` with about ``budget`` photons each.

    An event goes to chunk ``k`` if its first photon has a position in ``[k budget, (k + 1) budget)`` in the list of
    all photons, so a chunk holds at most ``budget`` photons plus those of one event.
    """
    first_photon = np.cumsum(n_photons) - n_photons
    chunk = first_photon // budget
    edges = np.concatenate([[0], np.flatnonzero(np.diff(chunk)) + 1, [len(n_photons)]])
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges)]


def _detect(
    fates: TransportResult, t0: torch.Tensor, config: SimConfig, generator: torch.Generator, save_hits: bool
) -> dict[str, np.ndarray]:
    """Compute the times of the detected photons of one chunk and their sums per event and sensor (on the device)."""
    n_events, m = t0.numel(), config.geometry.n_sensors
    fate_counts = torch.bincount(fates.event * _N_CODES + fates.fate.long(), minlength=n_events * _N_CODES)
    i = (fates.fate == Interaction.DETECT).nonzero().squeeze(1)
    event, sensor, path = fates.event[i], fates.sensor[i], fates.path_length[i]
    n_scatter, n_reflect = fates.n_scatter[i], fates.n_reflect[i]
    # arrival time = emission time + time of flight + Gaussian jitter
    hit_time = t0[event] + path / config.optics.speed_of_light
    if config.readout.time_jitter > 0:
        jitter = torch.randn(hit_time.shape, generator=generator, device=generator.device)
        hit_time = hit_time + config.readout.time_jitter * jitter
    # direct (0), scattered (1: scattered, never reflected) or reflected (2: reflected at least once)
    category = torch.where(n_reflect > 0, 2, torch.where(n_scatter > 0, 1, 0))
    composition = torch.bincount(event * 3 + category, minlength=n_events * 3)
    # per-sensor sums: number of photons, time sum (integer ticks, see TICKS_PER_NS) and earliest time
    key = event * m + sensor
    charge = torch.bincount(key, minlength=n_events * m)
    ticks = torch.round(hit_time.double() * TICKS_PER_NS).long()
    tick_sum = torch.zeros(n_events * m, dtype=torch.int64, device=key.device).index_add_(0, key, ticks)
    first = torch.full((n_events * m,), math.inf, device=key.device)
    first = first.scatter_reduce_(0, key, hit_time, reduce="amin")
    out = {
        "fate_counts": fate_counts.view(n_events, _N_CODES),
        "composition": composition.view(n_events, 3),
        "charge": charge.view(n_events, m),
        "tick_sum": tick_sum.view(n_events, m),
        "first": first.view(n_events, m),
    }
    if save_hits:
        out |= {
            "event": event,
            "sensor": sensor,
            "time": hit_time,
            "n_scatter": n_scatter,
            "n_reflect": n_reflect,
            "path_length": path,
        }
    return {name: value.cpu().numpy() for name, value in out.items()}


def _tracks(fates: TransportResult, n_photons: torch.Tensor) -> dict[str, np.ndarray]:
    """Collect the recorded points of one chunk, with the event and the number within the event of their photon."""
    assert fates.tracks is not None
    points = fates.tracks
    event = fates.event[points.photon]
    first_photon = torch.cumsum(n_photons, 0) - n_photons
    rank = points.photon - first_photon[event]
    out = {"photon": points.photon, "event": event, "rank": rank, "x": points.x, "y": points.y}
    out |= {"path_length": points.path_length, "kind": points.kind}
    return {name: value.cpu().numpy() for name, value in out.items()}


class _Collector:
    """Collects the per-chunk outputs into arrays for all events."""

    def __init__(self, n_events: int, n_sensors: int, save_hits: bool) -> None:
        shape = (n_events, n_sensors)
        self.fate_counts = np.zeros((n_events, _N_CODES), dtype=np.int64)
        self.composition = np.zeros((n_events, 3), dtype=np.int64)
        self.charge = np.zeros(shape, dtype=np.int64)
        self.tick_sum = np.zeros(shape, dtype=np.int64)
        self.first = np.full(shape, np.inf, dtype=np.float32)
        self.hit_parts: list[dict[str, np.ndarray]] = []
        self.track_parts: list[dict[str, np.ndarray]] = []
        self.save_hits = save_hits

    def add(self, start: int, chunk: dict[str, np.ndarray]) -> None:
        stop = start + len(chunk["charge"])
        for name in ("fate_counts", "composition", "charge", "tick_sum", "first"):
            getattr(self, name)[start:stop] = chunk[name]
        if self.save_hits:
            hits = {k: chunk[k] for k in ("sensor", "time", "n_scatter", "n_reflect", "path_length")}
            self.hit_parts.append(hits | {"event": chunk["event"] + start})

    def add_tracks(self, start: int, points: dict[str, np.ndarray]) -> None:
        self.track_parts.append(points | {"event": points["event"] + start})

    def response(self, noise: dict[str, np.ndarray], config: SimConfig) -> Response:
        """Charge, mean time and first time per sensor: detected photons plus dark noise."""
        m = config.geometry.n_sensors
        n_events = len(self.charge)
        key = noise["event"] * m + noise["sensor"]
        noise_sum = np.bincount(key, weights=noise["time"], minlength=n_events * m).reshape(n_events, m)
        noise_first = np.full(n_events * m, np.inf)
        np.minimum.at(noise_first, key, noise["time"])
        charge = self.charge + noise["counts"]
        time_sum = self.tick_sum / TICKS_PER_NS + noise_sum
        with np.errstate(invalid="ignore", divide="ignore"):
            time_mean = np.where(charge > 0, time_sum / charge, np.nan)
        time_first = np.fmin(self.first, noise_first.reshape(n_events, m))
        time_first[~np.isfinite(time_first)] = np.nan
        return Response(
            charge=charge.astype(np.int32),
            time_mean=time_mean.astype(np.float32),
            time_first=time_first.astype(np.float32),
        )

    def photon_counts(self, n_photons: np.ndarray, noise: dict[str, np.ndarray]) -> PhotonCounts:
        """Per-event photon bookkeeping from the fate counts."""
        fate = self.fate_counts
        counts = {
            "n_emitted": n_photons,
            "n_detected": fate[:, Interaction.DETECT],
            "n_absorbed_bulk": fate[:, Interaction.ABSORB_BULK],
            "n_absorbed_wall": fate[:, Interaction.ABSORB_WALL],
            "n_absorbed_sensor": fate[:, Interaction.ABSORB_SENSOR],
            "n_lost": fate[:, Interaction.LOST],
            "n_detected_direct": self.composition[:, 0],
            "n_detected_scattered": self.composition[:, 1],
            "n_detected_reflected": self.composition[:, 2],
            "n_noise": noise["counts"].sum(axis=1),
        }
        return PhotonCounts(**{name: np.ascontiguousarray(v, dtype=np.int32) for name, v in counts.items()})

    def hits(self, noise: dict[str, np.ndarray], event_id: np.ndarray) -> Hits:
        """All hits grouped by event: the detected photons of an event first, then its dark-noise hits."""
        n_noise = len(noise["time"])
        noise_hits = {
            "event": noise["event"],
            "sensor": noise["sensor"],
            "time": noise["time"],
            "n_scatter": np.full(n_noise, -1),
            "n_reflect": np.full(n_noise, -1),
            "path_length": np.full(n_noise, np.nan),
        }
        parts = [*self.hit_parts, noise_hits]
        column = {name: np.concatenate([part[name] for part in parts]) for name in noise_hits}
        is_noise = np.zeros(len(column["time"]), dtype=bool)
        is_noise[len(is_noise) - n_noise :] = True
        # Signal and noise hits are each ordered by event already: a stable sort by event merges them in linear time
        # and keeps the detected photons of every event before its noise hits.
        order = np.argsort(column["event"], kind="stable") if n_noise else slice(None)
        return Hits(
            event_id=event_id[column["event"][order]],
            sensor=column["sensor"][order].astype(np.int32),
            time=column["time"][order].astype(np.float32),
            n_scatter=column["n_scatter"][order].astype(np.int32),
            n_reflect=column["n_reflect"][order].astype(np.int32),
            path_length=column["path_length"][order].astype(np.float32),
            is_noise=is_noise[order],
        )

    def tracks(self, event_id: np.ndarray) -> Tracks:
        """Assemble the recorded photon paths as flattened polylines."""
        parts = self.track_parts
        if parts:
            cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        else:
            cat = {k: np.zeros(0) for k in ("photon", "event", "rank", "x", "y", "path_length", "kind")}
        photon = cat["event"].astype(np.int64) * (1 << 31) + cat["rank"]  # unique photon key across chunks
        starts = np.flatnonzero(np.r_[True, photon[1:] != photon[:-1]]) if len(photon) else np.zeros(0, int)
        return Tracks(
            event_id=event_id[cat["event"][starts].astype(np.int64)],
            photon=cat["rank"][starts].astype(np.int32),
            offsets=np.r_[starts, len(photon)].astype(np.int64),
            x=cat["x"].astype(np.float32),
            y=cat["y"].astype(np.float32),
            path_length=cat["path_length"].astype(np.float32),
            kind=cat["kind"].astype(np.int8),
        )


# ----------------------------------------------------------------------------------------------------------------------
# Checks, progress and provenance


def _check_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    return int(seed)


def _check_count(n: int) -> int:
    if int(n) < 1:
        raise ValueError(f"need at least one event, got {n}")
    return int(n)


def _check_int(value: Any, minimum: int, name: str = "photons_per_chunk") -> int:
    """Return ``value`` as a plain int (NumPy integers included), at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_options(
    save_hits: Any, save_tracks: Any, track_photons: Any, first_event_id: Any
) -> tuple[bool, int, int | None, int]:
    """Validate the run options and turn them into plain Python values (they end up in the JSON provenance)."""
    return (
        bool(save_hits),
        _check_int(save_tracks, 0, "save_tracks"),
        None if track_photons is None else _check_int(track_photons, 1, "track_photons"),
        _check_int(first_event_id, 0, "first_event_id"),
    )


def _check_events(x: np.ndarray, y: np.ndarray, energy: np.ndarray, config: SimConfig) -> None:
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(energy))):
        raise ValueError("vertex coordinates and energies must be finite")
    if np.any(energy < 0):
        raise ValueError("energies must be >= 0")
    radius = config.geometry.radius
    r = np.hypot(x.astype(np.float64), y.astype(np.float64))
    if np.any(r > radius * (1 + 1e-6)):
        raise ValueError(f"{int(np.sum(r > radius * (1 + 1e-6)))} vertices lie outside the detector (R = {radius} m)")


class _Progress:
    """Minimal progress report on stderr: a bar on a terminal, a log line every ~10 % otherwise."""

    def __init__(self, total: int, enabled: bool) -> None:
        self.total, self.enabled = total, enabled
        self.start = time.perf_counter()
        self.tty = enabled and sys.stderr.isatty()
        self.next_report = 0.1

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        rate = done / max(elapsed, 1e-9)
        frac = done / self.total
        text = f"{done}/{self.total} events, {rate:.3g} events/s, {elapsed:.0f} s"
        if self.tty:
            bar = "#" * int(30 * frac)
            sys.stderr.write(f"\r[{bar:<30}] {100 * frac:3.0f} %  {text}")
            sys.stderr.flush()
        elif frac >= self.next_report or done == self.total:
            log.info("progress: %3.0f %% (%s)", 100 * frac, text)
            self.next_report = math.floor(frac * 10 + 1) / 10

    def close(self) -> None:
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


@functools.lru_cache(maxsize=1)
def _git_commit() -> str | None:
    """Commit of the source checkout this package runs from (with ``-dirty`` for local changes), else None."""
    package_dir = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=package_dir, capture_output=True, text=True, timeout=10, check=True)
        return done.stdout.strip()

    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        if (top / "src" / "prob_reco_sbi").resolve() != package_dir:
            return None
        commit = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return commit + ("-dirty" if dirty else "")


def _provenance(*, device: torch.device, wall_time: float, **settings: Any) -> dict[str, Any]:
    """Seed, device, chunking, versions, creation time and wall time of a run."""
    import h5py

    from .. import __version__

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine()
    return {
        **settings,
        "device": str(device),
        "device_name": device_name,
        "torch_threads": torch.get_num_threads(),
        "package_version": __version__,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "h5py_version": h5py.__version__,
        "platform": platform.platform(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(wall_time, 3),
    }
