"""Detector geometry: sensor positions, the sensor lookup on the wall and the distance to the wall.

The detector is a disc of radius ``R``. Sensor ``j`` is the wall arc centred at ``phi_j = 2 pi j / M`` with angular
half-width ``a / (2 R)``. The NumPy helpers (:func:`sensor_phi`, :func:`sensor_positions`, :func:`sensor_edges`,
:func:`sensor_arcs`) are meant for drawing; the torch functions (:func:`sensor_at`, :func:`distance_to_wall`) are used
by the photon transport and work on any device.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .config import Geometry

__all__ = [
    "distance_to_wall",
    "photocoverage",
    "sensor_arcs",
    "sensor_at",
    "sensor_edges",
    "sensor_half_angle",
    "sensor_phi",
    "sensor_positions",
]


def photocoverage(geometry: Geometry) -> float:
    """Fraction of the wall covered by sensors, ``M a / (2 pi R)`` (0.509 for the default detector)."""
    return geometry.photocoverage


def sensor_half_angle(geometry: Geometry) -> float:
    """Angular half-width of a sensor arc, ``a / (2 R)`` [rad]."""
    return geometry.sensor_width / (2 * geometry.radius)


def sensor_phi(geometry: Geometry) -> np.ndarray:
    """Angles of the sensor centres, ``phi_j = 2 pi j / M`` [rad], shape ``[M]``."""
    return 2 * np.pi * np.arange(geometry.n_sensors) / geometry.n_sensors


def sensor_positions(geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates ``(x_j, y_j)`` of the sensor centres on the wall [m], each of shape ``[M]``."""
    phi = sensor_phi(geometry)
    return geometry.radius * np.cos(phi), geometry.radius * np.sin(phi)


def sensor_edges(geometry: Geometry) -> np.ndarray:
    """Angles of the two ends of every sensor arc [rad], shape ``[M, 2]`` (``phi_j -+ a / (2 R)``)."""
    phi = sensor_phi(geometry)
    half = sensor_half_angle(geometry)
    return np.stack([phi - half, phi + half], axis=1)


def sensor_arcs(geometry: Geometry, n_points: int = 16, radius: float | None = None) -> np.ndarray:
    """Points along every sensor arc, for drawing, shape ``[M, n_points, 2]``.

    Parameters
    ----------
    geometry
        Detector geometry.
    n_points
        Number of points per arc (the first and last are the arc ends).
    radius
        Radius at which the arcs are drawn (default: the detector radius; slightly larger values draw the sensors
        just outside the wall).

    Returns
    -------
    numpy.ndarray
        ``points[j, k] = (x, y)`` of the ``k``-th point of sensor ``j``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> for arc in sensor_arcs(Geometry()):  # doctest: +SKIP
    ...     plt.plot(arc[:, 0], arc[:, 1], lw=4)
    """
    edges = sensor_edges(geometry)
    t = np.linspace(0.0, 1.0, n_points)
    phi = edges[:, :1] + (edges[:, 1:] - edges[:, :1]) * t
    r = geometry.radius if radius is None else radius
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)


def sensor_at(phi: torch.Tensor, geometry: Geometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Which sensor, if any, covers the wall point at angle ``phi``.

    The nearest sensor centre is ``j = round(phi / dphi) mod M`` with ``dphi = 2 pi / M``; the point lies on that
    sensor if its angular distance to the centre, wrapped into ``[-pi, pi)``, is at most the half-width
    ``a / (2 R)``. The wrap makes ``phi = pi`` and ``phi = -pi`` the same point.

    Parameters
    ----------
    phi
        Angles of wall points [rad], any shape (e.g. from ``torch.atan2(y, x)``, which lies in ``(-pi, pi]``).
    geometry
        Detector geometry.

    Returns
    -------
    index : torch.Tensor
        Nearest sensor ``j`` in ``0 ... M - 1`` (int64), same shape as ``phi``.
    on_sensor : torch.Tensor
        Boolean, ``True`` where the point lies on that sensor's arc (edges included).
    """
    m = geometry.n_sensors
    dphi = 2 * math.pi / m
    nearest = torch.round(phi / dphi)
    offset = torch.remainder(phi - nearest * dphi + math.pi, 2 * math.pi) - math.pi
    index = torch.remainder(nearest.long(), m)
    return index, offset.abs() <= sensor_half_angle(geometry)


def distance_to_wall(
    x: torch.Tensor, y: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, radius: float
) -> torch.Tensor:
    """Distance from points inside the disc to the wall along unit directions.

    The wall point ``p + s u`` satisfies ``|p + s u|^2 = R^2``, i.e. ``s^2 + 2 b s + c = 0`` with ``b = p . u`` and
    ``c = |p|^2 - R^2 <= 0``, whose positive root is ``s = -b + sqrt(b^2 - c)``. The result is clipped at 0 so that a
    point that rounding has put just outside the wall meets it immediately.

    Parameters
    ----------
    x, y
        Positions [m].
    ux, uy
        Unit direction vectors.
    radius
        Detector radius ``R`` [m].

    Returns
    -------
    torch.Tensor
        Distance ``s >= 0`` [m] for every point.
    """
    b = x * ux + y * uy
    c = x * x + y * y - radius * radius
    return (torch.sqrt((b * b - c).clamp_min(0.0)) - b).clamp_min(0.0)
