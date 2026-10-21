"""Photon transport in the toy detector "A Flash in the Dark".

All photons of a batch of events are tracked together: every tensor below has one entry per photon, and one pass of
the loop in :func:`transport` moves every photon still in flight to its next interaction point.

The physics (2-D, a single medium, no refraction)::

    emission    at the vertex, isotropic: direction angle ~ U(0, 2 pi)
    flight      free path to the next bulk interaction  s_free ~ Exp(mu),  mu = 1/lambda_a + 1/lambda_s
                (a process that is off adds nothing to mu; mu = 0 means straight flight to the wall);
                distance to the wall along the direction s_wall (geometry.distance_to_wall);
                the photon flies s = min(s_free, s_wall) and its path length grows by s
    bulk        absorbed with probability (1/lambda_a) / mu, otherwise Rayleigh-scattered:
                the direction turns by theta with p(theta) = (1 + cos^2 theta) / (3 pi)
    wall        on a sensor arc:   detected with probability epsilon, otherwise absorbed by the sensor
                between sensors:   reflected with probability rho, otherwise absorbed by the wall
                                   diffuse:  angle alpha to the inward normal, p(alpha) = cos(alpha) / 2
                                   specular: mirror image of the direction, u' = u - 2 (u . n) n
    end         photons still in flight after max_steps interactions are lost

For every photon the transport returns its fate (the type of its last interaction), the sensor it ended on, its path
length (the time of flight is path / c_n) and how often it was scattered and reflected. The steps that concern whole
events (number of photons, emission time, time jitter, dark noise and the per-sensor readout) are in
:mod:`prob_reco_sbi.flash.simulate`.

Every random number is drawn from the ``generator`` argument, a :class:`torch.Generator` on the device of the inputs,
so the result is reproducible for a given seed, device type and batch on the same hardware and PyTorch build (see
:func:`warm_up_cpu_kernels` for a CPU detail).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from .config import Geometry, SimConfig
from .geometry import distance_to_wall, sensor_at

__all__ = [
    "FATES",
    "Interaction",
    "Photons",
    "TrackPoints",
    "TransportResult",
    "emit",
    "free_path",
    "reflect_specular",
    "rotate",
    "sample_lambertian_angle",
    "sample_rayleigh_angle",
    "transport",
    "warm_up_cpu_kernels",
]


class Interaction(IntEnum):
    """Type of a point on a photon path; the type of the last point is the photon's fate."""

    EMIT = 0  #: emission at the vertex
    SCATTER = 1  #: Rayleigh scattering in the bulk
    REFLECT = 2  #: reflection on the wall between the sensors
    DETECT = 3  #: detection by a sensor
    ABSORB_BULK = 4  #: absorption in the bulk
    ABSORB_WALL = 5  #: absorption by the wall between the sensors
    ABSORB_SENSOR = 6  #: reached a sensor but not detected (detection efficiency < 1)
    LOST = 7  #: still in flight after max_steps interactions


#: The interactions that end a photon path.
FATES = (
    Interaction.DETECT,
    Interaction.ABSORB_BULK,
    Interaction.ABSORB_WALL,
    Interaction.ABSORB_SENSOR,
    Interaction.LOST,
)

#: Reflected photons restart this far [m] inside the wall, so that the next distance to the wall is well defined.
NUDGE = 1e-6

_cpu_kernels_warm = False


def warm_up_cpu_kernels() -> None:
    """Run the vectorised CPU math kernels once, single-threaded, before the first multi-threaded transport.

    In a fresh process, the first multi-threaded evaluation of the float32 kernels ``sin``, ``cos``, ``sqrt``, ``log``
    and ``asin`` can compute one thread's block of elements with a different code path than the other blocks (the
    one-time initialisation of the vector math library races between the threads; seen with the MKL-backed CPU
    builds of PyTorch). That flips the fates of a few photons at the sensor edges, and since the number of random
    draws of the time jitter equals the number of detected photons, every later chunk then gets other random numbers:
    the whole realisation differs from a run in a process that has used these kernels before. One small
    single-threaded call of each kernel beforehand initialises the library without a race. :func:`transport` calls
    this function for CPU tensors; it does its work once per process and is harmless to call again.
    """
    global _cpu_kernels_warm
    if _cpu_kernels_warm:
        return
    x = torch.linspace(0.1, 0.9, 8)
    for fn in (torch.sin, torch.cos, torch.sqrt, torch.log, torch.asin, torch.exp):
        fn(x)
    torch.atan2(x, x)
    torch.hypot(x, x)
    _cpu_kernels_warm = True


@dataclass
class Photons:
    """Photons in flight: one entry per photon in every tensor."""

    index: torch.Tensor  #: position of the photon in the batch's photon list (int64)
    x: torch.Tensor  #: position [m]
    y: torch.Tensor
    ux: torch.Tensor  #: unit direction
    uy: torch.Tensor
    path: torch.Tensor  #: path length so far [m]
    n_scatter: torch.Tensor  #: number of Rayleigh scatterings so far (int32)
    n_reflect: torch.Tensor  #: number of wall reflections so far (int32)
    tracked: torch.Tensor | None = None  #: whether the path is recorded (bool); None when nothing is recorded

    def __len__(self) -> int:
        return self.index.numel()

    def __getitem__(self, keep: torch.Tensor) -> Photons:
        """Select photons with an index tensor."""
        return Photons(
            index=self.index[keep],
            x=self.x[keep],
            y=self.y[keep],
            ux=self.ux[keep],
            uy=self.uy[keep],
            path=self.path[keep],
            n_scatter=self.n_scatter[keep],
            n_reflect=self.n_reflect[keep],
            tracked=None if self.tracked is None else self.tracked[keep],
        )


@dataclass
class TrackPoints:
    """Points of the recorded photon paths, ordered by photon and, for each photon, along its path."""

    photon: torch.Tensor  #: position of the photon in the batch's photon list (int64)
    x: torch.Tensor  #: point coordinates [m]
    y: torch.Tensor
    path_length: torch.Tensor  #: path length from the vertex to the point [m]
    kind: torch.Tensor  #: Interaction code of the point (int8)


@dataclass
class TransportResult:
    """Fate of every photon of a batch (tensors of length n_photons, photons ordered by event)."""

    event: torch.Tensor  #: event of each photon, as an index into the batch (int64)
    fate: torch.Tensor  #: Interaction code of the last point (int8)
    sensor: torch.Tensor  #: sensor of the last point, -1 if the photon did not end on a sensor (int64)
    path_length: torch.Tensor  #: total path length [m] (float32)
    n_scatter: torch.Tensor  #: number of Rayleigh scatterings (int32)
    n_reflect: torch.Tensor  #: number of wall reflections (int32)
    tracks: TrackPoints | None = None  #: recorded paths, if requested


def transport(
    x0: torch.Tensor,
    y0: torch.Tensor,
    n_photons: torch.Tensor,
    config: SimConfig,
    generator: torch.Generator,
    *,
    track_events: int = 0,
    track_photons: int | None = None,
) -> TransportResult:
    """Track all photons of a batch of events from emission to their fate.

    Parameters
    ----------
    x0, y0
        Vertex coordinates of the events [m], float tensors of shape ``[n_events]``.
    n_photons
        Number of photons emitted by each event, int64 tensor of shape ``[n_events]``.
    config
        Detector and optics configuration (the prior is not used here).
    generator
        Source of all random numbers, on the same device as the inputs.
    track_events
        Record the paths of the photons of the first ``track_events`` events of the batch.
    track_photons
        Record at most this many photons per recorded event (``None``: all of them).

    Returns
    -------
    TransportResult
        Fate, final sensor, path length and numbers of scatterings and reflections of every photon.
    """
    if x0.device.type == "cpu":
        warm_up_cpu_kernels()
    geometry, optics = config.geometry, config.optics
    mu_absorb = 1.0 / optics.absorption_length if optics.absorption else 0.0  # absorption coefficient [1/m]
    mu_scatter = 1.0 / optics.scattering_length if optics.scattering else 0.0  # scattering coefficient [1/m]
    mu = mu_absorb + mu_scatter
    p_absorb = mu_absorb / mu if mu > 0 else 0.0  # probability that a bulk interaction is an absorption
    rho = optics.reflectivity if optics.reflection else 0.0  # reflection off: black wall
    epsilon = config.readout.detection_efficiency

    photons, event = emit(x0, y0, n_photons, generator, track_events=track_events, track_photons=track_photons)
    result = _start_result(event)
    recorder = _TrackRecorder(photons) if photons.tracked is not None else None

    for _ in range(optics.max_steps):
        if len(photons) == 0:
            break
        # 1. Fly to the next interaction point: a bulk interaction or the wall, whichever comes first.
        s_wall = distance_to_wall(photons.x, photons.y, photons.ux, photons.uy, geometry.radius)
        s_free = free_path(len(photons), mu, generator)
        at_wall = s_wall <= s_free
        step = torch.where(at_wall, s_wall, s_free)
        photons.x += step * photons.ux
        photons.y += step * photons.uy
        photons.path += step

        # 2. Decide what happens there.
        what, sensor = interact(photons, at_wall, p_absorb, epsilon, rho, geometry, generator)
        if recorder is not None:
            recorder.add(photons, what)

        # 3. Photons that go on get a new direction.
        scatter(photons, what == Interaction.SCATTER, generator)
        reflect(photons, what == Interaction.REFLECT, optics.reflection_model, generator)

        # 4. Photons whose path ended here leave the loop.
        ended = (what != Interaction.SCATTER) & (what != Interaction.REFLECT)
        _finish(result, photons, ended.nonzero().squeeze(1), what, sensor)
        photons = photons[(~ended).nonzero().squeeze(1)]

    # 5. Photons still in flight after max_steps are lost (every photon's fate starts as LOST).
    _finish(result, photons, torch.arange(len(photons), device=event.device))
    if recorder is not None:
        recorder.add(photons, torch.full((len(photons),), Interaction.LOST, device=event.device))
        result.tracks = recorder.points()
    return result


# ----------------------------------------------------------------------------------------------------------------------
# The steps of the transport


def emit(
    x0: torch.Tensor,
    y0: torch.Tensor,
    n_photons: torch.Tensor,
    generator: torch.Generator,
    *,
    track_events: int = 0,
    track_photons: int | None = None,
) -> tuple[Photons, torch.Tensor]:
    """Create the photons of every event at its vertex, flying in isotropic directions.

    Returns
    -------
    photons : Photons
        The new photons, ordered by event.
    event : torch.Tensor
        Event index of every photon (int64).
    """
    device = x0.device
    event = torch.repeat_interleave(torch.arange(x0.numel(), device=device), n_photons)
    n = event.numel()
    angle = 2 * math.pi * torch.rand(n, generator=generator, device=device)
    tracked = None
    if track_events > 0:
        first = torch.cumsum(n_photons, 0) - n_photons  # position of every event's first photon
        rank = torch.arange(n, device=device) - first[event]  # number of the photon within its event
        limit = n if track_photons is None else track_photons
        tracked = (event < track_events) & (rank < limit)
    photons = Photons(
        index=torch.arange(n, device=device),
        x=x0[event],
        y=y0[event],
        ux=torch.cos(angle),
        uy=torch.sin(angle),
        path=torch.zeros(n, dtype=x0.dtype, device=device),
        n_scatter=torch.zeros(n, dtype=torch.int32, device=device),
        n_reflect=torch.zeros(n, dtype=torch.int32, device=device),
        tracked=tracked,
    )
    return photons, event


def free_path(n: int, mu: float, generator: torch.Generator) -> torch.Tensor:
    """Distances to the next bulk interaction, ``s ~ Exp(mu)``, i.e. ``p(s) = mu exp(-mu s)``.

    Inverse-transform sampling: the cumulative distribution ``F(s) = 1 - exp(-mu s)`` gives
    ``s = -log(1 - u) / mu`` for ``u ~ U(0, 1)`` (``1 - u`` is never 0, so ``s`` is finite). With ``mu = 0``
    (absorption and scattering both off) there is no bulk interaction: the distance is infinite, the photon flies
    straight to the wall, and no random number is drawn.
    """
    if mu == 0.0:
        return torch.full((n,), math.inf, device=generator.device)
    u = torch.rand(n, generator=generator, device=generator.device)
    return -torch.log(1 - u) / mu


def interact(
    photons: Photons,
    at_wall: torch.Tensor,
    p_absorb: float,
    epsilon: float,
    rho: float,
    geometry: Geometry,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decide what happens at the interaction point of every photon.

    One uniform random number ``u`` per photon decides::

        in the bulk:                 absorbed if u < p_absorb,  otherwise scattered
        at the wall, on a sensor:    detected if u < epsilon,   otherwise absorbed by the sensor
        at the wall, between them:   reflected if u < rho,      otherwise absorbed by the wall

    Returns
    -------
    what : torch.Tensor
        :class:`Interaction` code of every photon (int64).
    sensor : torch.Tensor
        Sensor at the interaction point, ``-1`` if the photon is not on a sensor (int64).
    """
    u = torch.rand(len(photons), generator=generator, device=generator.device)
    index, on_sensor = sensor_at(torch.atan2(photons.y, photons.x), geometry)
    on_sensor &= at_wall
    at_sensor = torch.where(u < epsilon, Interaction.DETECT, Interaction.ABSORB_SENSOR)
    between = torch.where(u < rho, Interaction.REFLECT, Interaction.ABSORB_WALL)
    in_bulk = torch.where(u < p_absorb, Interaction.ABSORB_BULK, Interaction.SCATTER)
    what = torch.where(at_wall, torch.where(on_sensor, at_sensor, between), in_bulk)
    return what, torch.where(on_sensor, index, -1)


def scatter(photons: Photons, selected: torch.Tensor, generator: torch.Generator) -> None:
    """Rayleigh scattering: turn the direction of the selected photons by ``theta ~ (1 + cos^2 theta) / (3 pi)``."""
    i = selected.nonzero().squeeze(1)
    theta = sample_rayleigh_angle(i.numel(), generator)
    photons.ux[i], photons.uy[i] = rotate(photons.ux[i], photons.uy[i], theta)
    photons.n_scatter[i] += 1


def reflect(photons: Photons, selected: torch.Tensor, model: str, generator: torch.Generator) -> None:
    """Wall reflection of the selected photons (which sit on the wall).

    ``model="diffuse"``: the new direction makes the angle ``alpha ~ cos(alpha) / 2`` with the inward normal
    (Lambert's law). ``model="specular"``: the direction is mirrored, ``u' = u - 2 (u . n) n``.
    """
    i = selected.nonzero().squeeze(1)
    x, y = photons.x[i], photons.y[i]
    r = torch.hypot(x, y)
    nx, ny = -x / r, -y / r  # inward normal of the wall
    if model == "diffuse":
        photons.ux[i], photons.uy[i] = rotate(nx, ny, sample_lambertian_angle(i.numel(), generator))
    else:
        photons.ux[i], photons.uy[i] = reflect_specular(photons.ux[i], photons.uy[i], nx, ny)
    photons.x[i] = x + NUDGE * nx
    photons.y[i] = y + NUDGE * ny
    photons.n_reflect[i] += 1


# ----------------------------------------------------------------------------------------------------------------------
# Angles and directions


def sample_rayleigh_angle(n: int, generator: torch.Generator) -> torch.Tensor:
    """Rayleigh scattering angles in 2-D, ``p(theta) = (1 + cos^2 theta) / (3 pi)`` on ``[0, 2 pi)``.

    Inverse-transform sampling: ``theta`` solves ``F(theta) = u`` for ``u ~ U(0, 1)`` and the cumulative distribution
    ``F(theta) = (3 theta / 2 + sin(2 theta) / 4) / (3 pi)``. Newton's method starts from ``theta = 2 pi u``, which is
    at most 1/4 rad off; since ``F'(theta) = p(theta) >= 1 / (3 pi)``, four iterations reach float32 precision.

    Parameters
    ----------
    n
        Number of angles.
    generator
        Source of the random numbers (sets the device).

    Returns
    -------
    torch.Tensor
        ``n`` angles [rad] (float32).
    """
    u = torch.rand(n, generator=generator, device=generator.device)
    theta = 2 * math.pi * u
    for _ in range(4):
        # Newton step for 3 pi F(theta) - 3 pi u = 0, whose derivative is 3 pi p(theta) = 1 + cos^2 theta
        residual = 1.5 * theta + 0.25 * torch.sin(2 * theta) - 3 * math.pi * u
        theta = theta - residual / (1 + torch.cos(theta) ** 2)
    return theta


def sample_lambertian_angle(n: int, generator: torch.Generator) -> torch.Tensor:
    """Angles to the wall normal for diffuse (Lambertian) reflection, ``p(alpha) = cos(alpha) / 2``.

    The cumulative distribution on ``[-pi/2, pi/2]`` is ``F(alpha) = (1 + sin alpha) / 2``, so
    ``alpha = arcsin(2 u - 1)`` with ``u ~ U(0, 1)``.

    Returns
    -------
    torch.Tensor
        ``n`` angles [rad] in ``[-pi/2, pi/2]`` (float32).
    """
    u = torch.rand(n, generator=generator, device=generator.device)
    return torch.asin(2 * u - 1)


def rotate(ux: torch.Tensor, uy: torch.Tensor, angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the vectors ``(ux, uy)`` counter-clockwise by ``angle`` [rad]."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    return ux * cos - uy * sin, ux * sin + uy * cos


def reflect_specular(
    ux: torch.Tensor, uy: torch.Tensor, nx: torch.Tensor, ny: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the directions ``u`` at a surface with unit normal ``n``: ``u' = u - 2 (u . n) n``."""
    dot = ux * nx + uy * ny
    return ux - 2 * dot * nx, uy - 2 * dot * ny


# ----------------------------------------------------------------------------------------------------------------------
# Bookkeeping


def _start_result(event: torch.Tensor) -> TransportResult:
    """Per-photon output arrays; every photon starts as LOST and is overwritten when its path ends."""
    n, device = event.numel(), event.device
    return TransportResult(
        event=event,
        fate=torch.full((n,), Interaction.LOST, dtype=torch.int8, device=device),
        sensor=torch.full((n,), -1, dtype=torch.int64, device=device),
        path_length=torch.zeros(n, dtype=torch.float32, device=device),
        n_scatter=torch.zeros(n, dtype=torch.int32, device=device),
        n_reflect=torch.zeros(n, dtype=torch.int32, device=device),
    )


def _finish(
    result: TransportResult,
    photons: Photons,
    i: torch.Tensor,
    what: torch.Tensor | None = None,
    sensor: torch.Tensor | None = None,
) -> None:
    """Store the final state of the photons ``i``; without ``what`` and ``sensor`` they stay LOST and -1."""
    k = photons.index[i]
    if what is not None and sensor is not None:
        result.fate[k] = what[i].to(torch.int8)
        result.sensor[k] = sensor[i]
    result.path_length[k] = photons.path[i]
    result.n_scatter[k] = photons.n_scatter[i]
    result.n_reflect[k] = photons.n_reflect[i]


class _TrackRecorder:
    """Collects the points of the recorded photon paths, one batch of points per step."""

    def __init__(self, photons: Photons) -> None:
        self.parts: list[tuple[torch.Tensor, ...]] = []
        self.add(photons, torch.full((len(photons),), Interaction.EMIT, device=photons.index.device))

    def add(self, photons: Photons, kind: torch.Tensor) -> None:
        assert photons.tracked is not None
        i = photons.tracked.nonzero().squeeze(1)
        self.parts.append((photons.index[i], photons.x[i], photons.y[i], photons.path[i], kind[i].to(torch.int8)))

    def points(self) -> TrackPoints:
        photon, x, y, path, kind = (torch.cat(column) for column in zip(*self.parts, strict=True))
        order = torch.sort(photon, stable=True).indices  # by photon; the steps of each photon stay in order
        return TrackPoints(photon[order], x[order], y[order], path[order], kind[order])
