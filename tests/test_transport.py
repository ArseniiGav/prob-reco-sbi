"""Unit tests of the photon transport: sampled angles, directions and the tracked paths."""

import math

import numpy as np
import pytest
import torch

from helpers import KS_LIMIT, ks_distance
from prob_reco_sbi.flash import Interaction, generate, preset, transport
from prob_reco_sbi.flash.transport import (
    emit,
    free_path,
    reflect_specular,
    rotate,
    sample_lambertian_angle,
    sample_rayleigh_angle,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_rayleigh_angles_follow_the_phase_function():
    theta = sample_rayleigh_angle(200_000, gen(1)).double().numpy()
    assert np.all((theta > -1e-5) & (theta < 2 * np.pi + 1e-5))
    cdf = lambda t: (1.5 * t + 0.25 * np.sin(2 * t)) / (3 * np.pi)  # noqa: E731
    assert ks_distance(np.clip(theta, 0, 2 * np.pi), cdf) < KS_LIMIT
    # moments of p(theta) = (1 + cos^2 theta) / (3 pi): <cos^2> = (pi + 3 pi / 4) / (3 pi) = 7/12, <cos> = 0
    assert np.mean(np.cos(theta) ** 2) == pytest.approx(7 / 12, abs=0.004)
    assert np.mean(np.cos(theta)) == pytest.approx(0.0, abs=0.009)


def test_rayleigh_angles_are_not_uniform():
    theta = sample_rayleigh_angle(200_000, gen(2)).double().numpy()
    uniform = lambda t: t / (2 * np.pi)  # noqa: E731
    assert ks_distance(theta, uniform) > 10


def test_lambertian_angles_follow_cosine_law():
    alpha = sample_lambertian_angle(200_000, gen(3)).double().numpy()
    assert np.all(np.abs(alpha) <= np.pi / 2)
    assert ks_distance(alpha, lambda a: (1 + np.sin(a)) / 2) < KS_LIMIT


def test_rotation_and_specular_reflection():
    ux, uy = rotate(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([math.pi / 2]))
    assert ux.item() == pytest.approx(0.0, abs=1e-7)
    assert uy.item() == pytest.approx(1.0)
    angle = 2 * math.pi * torch.rand(1000, generator=gen(4), dtype=torch.float64)
    ux, uy = torch.cos(angle), torch.sin(angle)
    nx, ny = -torch.cos(angle + 0.3), -torch.sin(angle + 0.3)  # inward normal at a nearby wall point
    vx, vy = reflect_specular(ux, uy, nx, ny)
    torch.testing.assert_close(torch.hypot(vx, vy), torch.ones_like(vx))
    torch.testing.assert_close(vx * nx + vy * ny, -(ux * nx + uy * ny))  # normal component flips
    torch.testing.assert_close(vx * ny - vy * nx, ux * ny - uy * nx)  # tangential component stays


def test_free_path_without_bulk_processes_is_infinite():
    assert torch.isinf(free_path(10, 0.0, gen())).all()
    s = free_path(100_000, 2.0, gen(5))
    assert s.mean().item() == pytest.approx(0.5, rel=0.02)


def test_emission_is_isotropic_and_ordered_by_event():
    photons, event = emit(
        torch.tensor([0.1, -0.2]), torch.tensor([0.0, 0.3]), torch.tensor([3, 50_000]), gen(6), track_events=1
    )
    np.testing.assert_array_equal(event[:4].numpy(), [0, 0, 0, 1])
    assert photons.tracked is not None
    assert photons.tracked.sum().item() == 3
    angle = torch.atan2(photons.uy[3:], photons.ux[3:]).double().numpy()
    assert ks_distance(angle, lambda a: (a + np.pi) / (2 * np.pi)) < KS_LIMIT
    torch.testing.assert_close(torch.hypot(photons.ux, photons.uy), torch.ones(50_003))


def test_transport_of_an_empty_batch():
    result = transport(torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.int64), preset("full"), gen())
    assert result.fate.numel() == 0


def segment_angles(data, kind):
    """Angles at the points of a given type: (incoming direction, outgoing direction, position)."""
    tracks = data.tracks
    incoming, outgoing, position = [], [], []
    for k in range(len(tracks)):
        x, y, point = tracks.polyline(k)
        # skip the point before a LOST point: that last segment is the 1 um step inside the wall after a reflection
        for i in np.flatnonzero((point[1:-1] == kind) & (point[2:] != Interaction.LOST)) + 1:
            incoming.append([x[i] - x[i - 1], y[i] - y[i - 1]])
            outgoing.append([x[i + 1] - x[i], y[i + 1] - y[i]])
            position.append([x[i], y[i]])
    return np.array(incoming), np.array(outgoing), np.array(position)


def test_reflection_angles_on_the_recorded_paths():
    """Diffuse reflections leave the wall with p(alpha) = cos(alpha) / 2 to the inward normal."""
    config = preset("full", reflectivity=1.0, scattering=False, absorption=False)
    data = generate(4, config=config, seed=7, device="cpu", save_tracks=4, track_photons=None)
    _, out, pos = segment_angles(data, Interaction.REFLECT)
    assert len(out) > 2000
    normal = -pos / np.hypot(pos[:, :1], pos[:, 1:])  # inward normal
    alpha = np.arctan2(normal[:, 0] * out[:, 1] - normal[:, 1] * out[:, 0], (out * normal).sum(1))
    assert ks_distance(alpha, lambda a: (1 + np.sin(a)) / 2) < KS_LIMIT


def test_specular_reflection_on_the_recorded_paths():
    config = preset("full", reflectivity=1.0, scattering=False, absorption=False, reflection_model="specular")
    data = generate(4, config=config, seed=8, device="cpu", save_tracks=4, track_photons=None)
    inc, out, pos = segment_angles(data, Interaction.REFLECT)
    assert len(out) > 500
    normal = pos / np.hypot(pos[:, :1], pos[:, 1:])
    cos_in = (inc * normal).sum(1) / np.hypot(inc[:, 0], inc[:, 1])
    cos_out = (out * normal).sum(1) / np.hypot(out[:, 0], out[:, 1])
    np.testing.assert_allclose(cos_out, -cos_in, atol=2e-3)


def test_scattering_angles_on_the_recorded_paths():
    """The turning angle |theta| in [0, pi] has p = 2 (1 + cos^2 theta) / (3 pi)."""
    config = preset("full", absorption=False, reflection=False, scattering_length=0.3)
    data = generate(2, config=config, seed=9, device="cpu", save_tracks=2, track_photons=None)
    inc, out, _ = segment_angles(data, Interaction.SCATTER)
    assert len(out) > 5000
    cos = (inc * out).sum(1) / (np.hypot(inc[:, 0], inc[:, 1]) * np.hypot(out[:, 0], out[:, 1]))
    theta = np.arccos(np.clip(cos, -1, 1))
    cdf = lambda t: (1.5 * t + 0.25 * np.sin(2 * t)) / (1.5 * np.pi)  # noqa: E731
    assert ks_distance(theta, cdf) < KS_LIMIT


def test_tracks_start_at_the_vertex_and_end_with_the_fate(full_small):
    tracks = full_small.tracks
    assert len(tracks) == 300  # 3 events x 100 photons
    assert np.all(tracks.kind[tracks.offsets[:-1]] == Interaction.EMIT)
    fates = {int(k) for k in tracks.fate}
    assert fates <= {Interaction.DETECT, Interaction.ABSORB_BULK, Interaction.ABSORB_WALL, Interaction.LOST}
    for k in tracks.of_event(full_small.truth.event_id[1]):
        x, y, _ = tracks.polyline(k)
        assert (x[0], y[0]) == (full_small.truth.x[1], full_small.truth.y[1])
        assert np.all(np.hypot(x, y) <= 1 + 1e-5)
    # path lengths grow along every track and match the distances between the points
    for k in range(0, len(tracks), 37):
        part = slice(tracks.offsets[k], tracks.offsets[k + 1])
        x, y, s = tracks.x[part], tracks.y[part], tracks.path_length[part]
        np.testing.assert_allclose(np.diff(s), np.hypot(np.diff(x), np.diff(y)), atol=1e-4)
