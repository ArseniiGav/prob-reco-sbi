import math

import numpy as np
import pytest
import torch

from prob_reco_sbi.flash import (
    Geometry,
    distance_to_wall,
    photocoverage,
    sensor_arcs,
    sensor_at,
    sensor_edges,
    sensor_half_angle,
    sensor_phi,
    sensor_positions,
)

G = Geometry()


def lookup(phi, geometry=G):
    index, on = sensor_at(torch.tensor(phi, dtype=torch.float64), geometry)
    return index.numpy(), on.numpy()


def test_photocoverage():
    assert photocoverage(G) == pytest.approx(0.5093, abs=1e-4)
    assert photocoverage(Geometry(n_sensors=16, sensor_width=0.2)) == pytest.approx(photocoverage(G))


def test_sensor_layout():
    phi = sensor_phi(G)
    np.testing.assert_allclose(phi, 2 * np.pi * np.arange(32) / 32)
    x, y = sensor_positions(G)
    np.testing.assert_allclose(np.hypot(x, y), 1.0)
    edges = sensor_edges(G)
    np.testing.assert_allclose(edges[:, 1] - edges[:, 0], 0.10)  # arc length a on a circle of radius 1
    arcs = sensor_arcs(G, n_points=5)
    assert arcs.shape == (32, 5, 2)
    np.testing.assert_allclose(np.hypot(arcs[..., 0], arcs[..., 1]), 1.0)
    np.testing.assert_allclose(np.arctan2(arcs[3, 0, 1], arcs[3, 0, 0]), edges[3, 0])


def test_lookup_at_sensor_centres():
    index, on = lookup(sensor_phi(G))
    np.testing.assert_array_equal(index, np.arange(32))
    assert on.all()


@pytest.mark.parametrize("j", [0, 1, 7, 16, 31])
def test_lookup_at_arc_edges(j):
    centre, half = 2 * np.pi * j / 32, sensor_half_angle(G)
    inside = [centre - half * (1 - 1e-9), centre + half * (1 - 1e-9)]
    outside = [centre - half * (1 + 1e-9), centre + half * (1 + 1e-9)]
    # angles as atan2 returns them, in (-pi, pi]
    wrap = lambda a: np.angle(np.exp(1j * np.asarray(a)))  # noqa: E731
    index, on = lookup(wrap(inside))
    np.testing.assert_array_equal(index, j)
    assert on.all()
    index, on = lookup(wrap(outside))
    np.testing.assert_array_equal(index, j)
    assert not on.any()


def test_lookup_at_the_wrap_around():
    half = sensor_half_angle(G)
    # phi = +pi and -pi are the same point, the centre of sensor 16 for M = 32
    index, on = lookup([math.pi, -math.pi, math.pi - 0.5 * half, -math.pi + 0.5 * half])
    np.testing.assert_array_equal(index, 16)
    assert on.all()
    index, on = lookup([-math.pi + 1.01 * half, math.pi - 1.01 * half])
    assert not on.any()
    # sensor 0 straddles phi = 0, also when angles are given in [0, 2 pi)
    index, on = lookup([-0.5 * half, 2 * math.pi - 0.5 * half, 0.5 * half])
    np.testing.assert_array_equal(index, 0)
    assert on.all()


def test_lookup_with_an_odd_number_of_sensors():
    g = Geometry(n_sensors=5, sensor_width=0.5)
    index, on = lookup([math.pi, -math.pi], g)  # half-way between sensors 2 and 3
    assert not on.any()
    assert set(index) <= {2, 3}
    index, on = lookup(sensor_phi(g) - 2 * np.pi * (sensor_phi(g) > np.pi), g)
    np.testing.assert_array_equal(index, np.arange(5))
    assert on.all()


def test_distance_to_wall():
    gen = torch.Generator().manual_seed(1)
    angle = 2 * math.pi * torch.rand(1000, generator=gen, dtype=torch.float64)
    ux, uy = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros(1000, dtype=torch.float64)
    torch.testing.assert_close(distance_to_wall(zero, zero, ux, uy, 1.0), torch.ones(1000, dtype=torch.float64))
    r = torch.sqrt(torch.rand(1000, generator=gen, dtype=torch.float64))
    x, y = r * torch.cos(angle + 1), r * torch.sin(angle + 1)
    s = distance_to_wall(x, y, ux, uy, 2.0)
    torch.testing.assert_close(torch.hypot(x + s * ux, y + s * uy), torch.full_like(s, 2.0))
    assert torch.all(s > 0)
    # a point just outside the wall that heads outwards meets the wall at once
    s = distance_to_wall(torch.tensor([1.0 + 1e-7]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]), 1.0)
    assert s.item() == 0.0
