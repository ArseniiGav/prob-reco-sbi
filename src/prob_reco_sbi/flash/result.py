"""Containers for simulated data: what :func:`generate` and :func:`simulate` return and :func:`load` reads back.

The attributes mirror the groups of the HDF5 file (see :mod:`prob_reco_sbi.flash.io`)::

    result.truth      event_id, x, y, energy, t0                       one entry per event
    result.response   charge, time_mean, time_first                    [n_events, n_sensors]
    result.photons    n_emitted, n_detected, ... (photon bookkeeping)  one entry per event
    result.hits       every detected photon and dark-noise hit         optional (save_hits)
    result.tracks     photon paths of the first events                 optional (save_tracks)
    result.detector   sensor positions                                 from the configuration
    result.config     the SimConfig used
    result.provenance seed, device, versions, creation time, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from .config import SimConfig
from .geometry import sensor_phi, sensor_positions
from .transport import Interaction

__all__ = ["Detector", "Hits", "PhotonCounts", "Response", "SimResult", "Tracks", "Truth"]


class _Arrays:
    """Shared helpers of the array containers."""

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the arrays of this container by name."""
        return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]


@dataclass(frozen=True)
class Truth(_Arrays):
    """True parameters of every event (arrays of length ``n_events``)."""

    event_id: np.ndarray  #: event number (int64), unique within a data set
    x: np.ndarray  #: vertex x0 [m] (float32)
    y: np.ndarray  #: vertex y0 [m] (float32)
    energy: np.ndarray  #: energy E [MeV] (float32)
    t0: np.ndarray  #: emission time t0 [ns] (float32), a nuisance parameter


@dataclass(frozen=True)
class Response(_Arrays):
    """Detector response per event and sensor (arrays of shape ``[n_events, n_sensors]``).

    ``charge`` counts the detected photons and dark-noise hits of a sensor ("photo-electrons"); ``time_mean`` and
    ``time_first`` are the mean and earliest time of those hits [ns], NaN for sensors without hits.
    """

    charge: np.ndarray  #: number of hits n_j (int32)
    time_mean: np.ndarray  #: mean hit time tau_j [ns] (float32, NaN where charge == 0)
    time_first: np.ndarray  #: first hit time [ns] (float32, NaN where charge == 0)


@dataclass(frozen=True)
class PhotonCounts(_Arrays):
    """Monte Carlo bookkeeping of the photons of every event (int32 arrays of length ``n_events``).

    Every emitted photon ends in exactly one way::

        n_emitted = n_detected + n_absorbed_bulk + n_absorbed_wall + n_absorbed_sensor + n_lost

    and every detected photon is either direct, scattered (at least one scattering, no reflection) or reflected (at
    least one reflection)::

        n_detected = n_detected_direct + n_detected_scattered + n_detected_reflected

    The total charge of an event is ``n_detected + n_noise``.
    """

    n_emitted: np.ndarray  #: photons emitted
    n_detected: np.ndarray  #: photons detected by a sensor
    n_absorbed_bulk: np.ndarray  #: photons absorbed in the bulk
    n_absorbed_wall: np.ndarray  #: photons absorbed by the wall between the sensors
    n_absorbed_sensor: np.ndarray  #: photons that reached a sensor but were not detected (efficiency < 1)
    n_lost: np.ndarray  #: photons still in flight after max_steps interactions
    n_detected_direct: np.ndarray  #: detected without scattering or reflection
    n_detected_scattered: np.ndarray  #: detected after at least one scattering and no reflection
    n_detected_reflected: np.ndarray  #: detected after at least one wall reflection
    n_noise: np.ndarray  #: dark-noise hits


@dataclass(frozen=True)
class Hits(_Arrays):
    """Every hit of every event: detected photons and dark-noise hits.

    The hits are grouped by event, the detected photons of an event first and then its dark-noise hits (not ordered
    in time). For dark-noise hits ``n_scatter`` and ``n_reflect`` are -1 and ``path_length`` is NaN.
    """

    event_id: np.ndarray  #: event of the hit (int64)
    sensor: np.ndarray  #: sensor index j (int32)
    time: np.ndarray  #: hit time [ns] (float32)
    n_scatter: np.ndarray  #: number of scatterings of the photon (int32)
    n_reflect: np.ndarray  #: number of wall reflections of the photon (int32)
    path_length: np.ndarray  #: path length of the photon [m] (float32)
    is_noise: np.ndarray  #: True for dark-noise hits (bool)

    def __len__(self) -> int:
        return len(self.event_id)


@dataclass(frozen=True)
class Tracks(_Arrays):
    """Recorded photon paths (polylines), flattened.

    Track ``k`` has the points ``offsets[k]:offsets[k + 1]``; each point has a type code ``kind`` (see
    :class:`~prob_reco_sbi.flash.transport.Interaction`: EMIT, SCATTER, REFLECT, DETECT, ABSORB_BULK, ABSORB_WALL,
    ABSORB_SENSOR, LOST). The first point of a track is its emission point, the last one its fate.

    Examples
    --------
    >>> for k in tracks.of_event(0):  # doctest: +SKIP
    ...     x, y, kind = tracks.polyline(k)
    ...     plt.plot(x, y)
    """

    event_id: np.ndarray  #: event of each track (int64), shape [n_tracks]
    photon: np.ndarray  #: number of the photon within its event (int32), shape [n_tracks]
    offsets: np.ndarray  #: start of each track in the point arrays (int64), shape [n_tracks + 1]
    x: np.ndarray  #: point coordinates [m] (float32), shape [n_points]
    y: np.ndarray
    path_length: np.ndarray  #: path length from the vertex to the point [m] (float32)
    kind: np.ndarray  #: Interaction code of each point (int8)

    def __len__(self) -> int:
        return len(self.event_id)

    def polyline(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the coordinates and point types ``(x, y, kind)`` of track ``k``."""
        part = slice(self.offsets[k], self.offsets[k + 1])
        return self.x[part], self.y[part], self.kind[part]

    def of_event(self, event_id: int) -> np.ndarray:
        """Return the indices of the tracks of one event."""
        return np.flatnonzero(self.event_id == event_id)

    @property
    def fate(self) -> np.ndarray:
        """Interaction code of the last point of every track (int8)."""
        return self.kind[self.offsets[1:] - 1]


@dataclass(frozen=True)
class Detector:
    """Sensor layout of the detector, for drawing (computed from the geometry)."""

    radius: float  #: detector radius R [m]
    sensor_width: float  #: arc length of a sensor [m]
    sensor_x: np.ndarray  #: sensor centres [m], shape [n_sensors]
    sensor_y: np.ndarray
    sensor_phi: np.ndarray  #: angles of the sensor centres [rad]

    @property
    def n_sensors(self) -> int:
        """Number of sensors M."""
        return len(self.sensor_phi)


@dataclass
class SimResult:
    """Simulated events: truth, detector response, photon bookkeeping, optional hits and tracks.

    Attributes
    ----------
    truth : Truth
        True event parameters.
    response : Response
        Charge and times per event and sensor.
    photons : PhotonCounts
        Photon bookkeeping per event.
    config : SimConfig
        Configuration used for the simulation.
    provenance : dict
        Seed, device, chunking, versions, creation time and wall time.
    hits : Hits or None
        Every hit, if the simulation was run with ``save_hits=True``.
    tracks : Tracks or None
        Photon paths of the first events, if the simulation was run with ``save_tracks > 0``.
    """

    truth: Truth
    response: Response
    photons: PhotonCounts
    config: SimConfig
    provenance: dict[str, Any] = field(default_factory=dict)
    hits: Hits | None = None
    tracks: Tracks | None = None

    def __len__(self) -> int:
        return len(self.truth.event_id)

    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self)

    @property
    def n_sensors(self) -> int:
        """Number of sensors M."""
        return self.config.geometry.n_sensors

    @property
    def detector(self) -> Detector:
        """Sensor layout, for drawing."""
        g = self.config.geometry
        x, y = sensor_positions(g)
        return Detector(radius=g.radius, sensor_width=g.sensor_width, sensor_x=x, sensor_y=y, sensor_phi=sensor_phi(g))

    def __repr__(self) -> str:
        extras = [name for name in ("hits", "tracks") if getattr(self, name) is not None]
        return (
            f"SimResult({self.n_events} events, {self.n_sensors} sensors"
            + (f", with {' and '.join(extras)}" if extras else "")
            + ")"
        )


#: Codes of the point types of :class:`Tracks` (``{"EMIT": 0, ...}``), also stored in the data files.
POINT_TYPES: dict[str, int] = {member.name: int(member) for member in Interaction}
