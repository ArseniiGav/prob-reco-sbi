"""A Flash in the Dark: photon-transport simulation of a toy 2-D scintillation detector.

A disc of radius 1 m with 32 photosensors on its wall records the light of point-like events (vertex ``(x0, y0)``,
energy ``E``, unknown time ``t0``). Photons are tracked one by one through absorption, Rayleigh scattering and wall
reflection, each of which can be switched off (presets ``straight``, ``absorb``, ``absorb-scatter``, ``full``).

Quick start::

    from prob_reco_sbi.flash import generate, load, network_input, preset, save

    data = generate(10_000, config=preset("full"), seed=1)
    save(data, "data/flash_full.h5")
    data = load("data/flash_full.h5")
    x = network_input(data.response.charge, data.response.time_mean)  # [10000, 64]

The same is available on the command line as ``flashsim generate`` (``flashsim --help``).
"""

from .config import PRESETS, Geometry, Optics, Prior, Readout, SimConfig, matching_preset, preset
from .features import network_input
from .geometry import (
    distance_to_wall,
    photocoverage,
    sensor_arcs,
    sensor_at,
    sensor_edges,
    sensor_half_angle,
    sensor_phi,
    sensor_positions,
)
from .io import describe, load, merge, save
from .result import POINT_TYPES, Detector, Hits, PhotonCounts, Response, SimResult, Tracks, Truth
from .simulate import DEFAULT_PHOTONS_PER_CHUNK, generate, resolve_device, sample_prior, simulate
from .transport import Interaction, transport

__all__ = [
    "DEFAULT_PHOTONS_PER_CHUNK",
    "POINT_TYPES",
    "PRESETS",
    "Detector",
    "Geometry",
    "Hits",
    "Interaction",
    "Optics",
    "PhotonCounts",
    "Prior",
    "Readout",
    "Response",
    "SimConfig",
    "SimResult",
    "Tracks",
    "Truth",
    "describe",
    "distance_to_wall",
    "generate",
    "load",
    "matching_preset",
    "merge",
    "network_input",
    "photocoverage",
    "preset",
    "resolve_device",
    "sample_prior",
    "save",
    "sensor_arcs",
    "sensor_at",
    "sensor_edges",
    "sensor_half_angle",
    "sensor_phi",
    "sensor_positions",
    "simulate",
    "transport",
]
