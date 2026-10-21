"""Physics validations with statistical tolerances, for events at the centre of the detector, where the expectations
of the detected fractions and of the hit times are simple closed forms."""

import math

import numpy as np
import pytest

from helpers import N_SIGMA, assert_binomial, check_conservation
from prob_reco_sbi.flash import generate, network_input, preset, simulate

COVERAGE = 32 * 0.10 / (2 * math.pi)


def centre(name, n_events=200, **overrides):
    """Events at the centre with exactly 1000 photons each."""
    config = preset(name, fixed_vertex=(0.0, 0.0), fixed_energy=1.0, photon_statistics="fixed", **overrides)
    return generate(n_events, config=config, seed=21, device="cpu", save_hits=True)


def test_straight_light_detected_fraction_is_the_photocoverage():
    data = centre("straight")
    p = data.photons
    assert np.all(p.n_emitted == 1000)
    assert_binomial(p.n_detected.sum(), p.n_emitted.sum(), COVERAGE)
    np.testing.assert_array_equal(p.n_detected + p.n_absorbed_wall, p.n_emitted)
    assert p.n_absorbed_bulk.sum() == p.n_lost.sum() == p.n_detected_scattered.sum() == 0
    # every sensor sees the same share a / (2 pi R) of the light
    mean = data.response.charge.mean(axis=0)
    expected = 1000 * 0.10 / (2 * math.pi)
    assert np.all(np.abs(mean - expected) < N_SIGMA * math.sqrt(expected / len(data)))


def test_straight_light_times_are_the_flight_time_plus_jitter():
    data = centre("straight")
    hits = data.hits
    t0 = data.truth.t0[hits.event_id - data.truth.event_id[0]]
    np.testing.assert_allclose(hits.path_length, 1.0, atol=1e-5)
    delay = hits.time - t0 - 1.0 / 0.2  # t - t0 - R / c_n
    n = len(delay)
    assert abs(delay.mean()) < N_SIGMA / math.sqrt(n)
    assert abs(delay.std() - 1.0) < N_SIGMA / math.sqrt(2 * n)
    data = centre("straight", time_jitter=0.0, n_events=20)
    t0 = data.truth.t0[data.hits.event_id - data.truth.event_id[0]]
    np.testing.assert_allclose(data.hits.time, t0 + 5.0, atol=1e-5)


def test_absorption_at_the_centre():
    data = centre("absorb")
    p = data.photons
    survive = math.exp(-1.0 / 1.5)  # exp(-R / lambda_a)
    assert_binomial(p.n_detected.sum(), p.n_emitted.sum(), COVERAGE * survive)
    assert_binomial(p.n_absorbed_bulk.sum(), p.n_emitted.sum(), 1 - survive)


def test_scattering_alone_keeps_the_photocoverage_at_the_centre():
    data = centre("straight", scattering=True)
    p = data.photons
    assert p.n_absorbed_bulk.sum() == p.n_lost.sum() == 0  # every photon ends on the wall
    np.testing.assert_array_equal(p.n_detected + p.n_absorbed_wall, p.n_emitted)
    assert_binomial(p.n_detected.sum(), p.n_emitted.sum(), COVERAGE)
    # a detected photon reached the wall unscattered with probability exp(-R / lambda_s)
    assert_binomial(p.n_detected_direct.sum(), p.n_detected.sum(), math.exp(-1.0 / 2.0))
    assert p.n_detected_reflected.sum() == 0


@pytest.mark.parametrize("vertex", [(0.0, 0.0), (0.6, -0.5), (0.0, 0.999)])
def test_perfect_reflector_and_sensors_detect_every_photon(vertex):
    config = preset("straight", reflection=True, reflectivity=1.0, fixed_vertex=vertex)
    data = generate(50, config=config, seed=22, device="cpu")
    np.testing.assert_array_equal(data.photons.n_detected, data.photons.n_emitted)
    assert data.photons.n_detected_reflected.sum() > 0


def test_specular_reflection_from_the_centre_bounces_on_a_diameter():
    """Radial photons that miss the sensors come back through the centre forever and are lost."""
    config = preset("straight", reflection=True, reflectivity=1.0, reflection_model="specular", max_steps=20)
    data = generate(20, config=config, seed=23, device="cpu", photons_per_chunk=10**6).photons
    data_centre = simulate(np.zeros(20), 0.0, 1.0, config=config, seed=23, device="cpu").photons
    np.testing.assert_array_equal(data.n_detected + data.n_lost, data.n_emitted)
    assert_binomial(data_centre.n_lost.sum(), data_centre.n_emitted.sum(), 1 - COVERAGE)


def test_detection_efficiency():
    data = centre("straight", detection_efficiency=0.25)
    p = data.photons
    assert_binomial(p.n_detected.sum(), p.n_detected.sum() + p.n_absorbed_sensor.sum(), 0.25)
    check_conservation(data)


def test_relative_times_do_not_depend_on_t0():
    x, y, e = np.array([0.0, 0.5, -0.3]), np.array([0.0, 0.2, -0.7]), np.array([1.0, 2.0, 0.6])
    a = simulate(x, y, e, t0=0.0, seed=5, device="cpu")
    b = simulate(x, y, e, t0=7.5, seed=5, device="cpu")
    np.testing.assert_array_equal(a.response.charge, b.response.charge)
    hit = a.response.charge > 0
    np.testing.assert_allclose(b.response.time_mean[hit] - a.response.time_mean[hit], 7.5, atol=1e-4)
    np.testing.assert_allclose(
        network_input(a.response.charge, a.response.time_mean),
        network_input(b.response.charge, b.response.time_mean),
        atol=1e-4,
    )


def test_light_collection_grows_toward_the_wall(full_small):
    r = np.hypot(full_small.truth.x, full_small.truth.y)
    f = full_small.photons.n_detected / full_small.photons.n_emitted
    assert f[r < 0.4].mean() < f[r > 0.9].mean()
