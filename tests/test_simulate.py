"""Simulation driver: photon conservation, determinism, chunking, prior, fixed parameters and dark noise."""

import hashlib
import itertools
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from helpers import KS_LIMIT, N_SIGMA, check_conservation, ks_distance
from prob_reco_sbi.flash import Interaction, generate, load, preset, sample_prior, save, simulate
from prob_reco_sbi.flash.simulate import _plan_chunks


def arrays(result):
    """Every array of a result, by name."""
    out = {}
    for group in ("truth", "response", "photons", "hits", "tracks"):
        container = getattr(result, group)
        if container is not None:
            out |= {f"{group}/{k}": v for k, v in container.arrays().items()}
    return out


@pytest.mark.parametrize(("absorption", "scattering", "reflection"), list(itertools.product([False, True], repeat=3)))
def test_photon_conservation_for_every_process_combination(absorption, scattering, reflection):
    config = preset(
        "full",
        absorption=absorption,
        scattering=scattering,
        reflection=reflection,
        detection_efficiency=0.7,
        dark_rate=2e6,
    )
    data = generate(60, config=config, seed=1, device="cpu", save_hits=True, photons_per_chunk=40_000)
    check_conservation(data)
    p = data.photons
    assert p.n_noise.sum() > 0
    assert p.n_absorbed_sensor.sum() > 0
    assert (p.n_absorbed_bulk.sum() > 0) == absorption
    assert (p.n_detected_scattered.sum() > 0) == scattering
    assert (p.n_detected_reflected.sum() > 0) == reflection
    assert p.n_lost.sum() == 0
    signal = ~data.hits.is_noise
    assert np.all(np.isfinite(data.hits.path_length[signal]))  # also for straight flight (no inf * 0)
    assert np.all(np.isfinite(data.hits.time))
    assert (data.hits.n_scatter[signal].max() > 0) == scattering
    assert np.all(data.hits.n_scatter[~signal] == -1)
    assert np.all(np.isnan(data.hits.path_length[~signal]))
    for array in (data.response.time_mean, data.response.time_first, data.hits.time):
        assert not np.any(np.isinf(array))


def test_same_seed_gives_identical_output():
    kwargs = {"config": preset("full", dark_rate=1e6), "device": "cpu", "save_hits": True, "save_tracks": 2}
    a, b = generate(100, seed=3, **kwargs), generate(100, seed=3, **kwargs)
    c = generate(100, seed=4, **kwargs)
    for name, value in arrays(a).items():
        np.testing.assert_array_equal(value, arrays(b)[name], err_msg=name)
    assert not np.array_equal(a.truth.x, c.truth.x)
    assert not np.array_equal(a.response.charge, c.response.charge)


def test_event_quantities_do_not_depend_on_the_chunking():
    config = preset("full", dark_rate=1e6)
    a = generate(50, config=config, seed=6, device="cpu", photons_per_chunk=10_000)
    b = generate(50, config=config, seed=6, device="cpu", photons_per_chunk=10**7)
    assert a.provenance["n_chunks"] > 5
    assert b.provenance["n_chunks"] == 1
    for name in ("x", "y", "energy", "t0"):
        np.testing.assert_array_equal(getattr(a.truth, name), getattr(b.truth, name))
    np.testing.assert_array_equal(a.photons.n_emitted, b.photons.n_emitted)
    np.testing.assert_array_equal(a.photons.n_noise, b.photons.n_noise)
    check_conservation(a)


def test_chunk_plan():
    n = np.array([5, 5, 5, 20, 0, 3])
    # first photons at 0, 5, 10, 15, 35, 35 -> chunks 0, 0, 1, 1, 3, 3 (a chunk may exceed the budget by one event)
    assert _plan_chunks(n, 10) == [(0, 2), (2, 4), (4, 6)]
    assert _plan_chunks(n, 1000) == [(0, 6)]
    assert _plan_chunks(np.array([7]), 1) == [(0, 1)]


def test_tracks_across_chunk_boundaries():
    data = generate(6, seed=2, device="cpu", save_tracks=4, track_photons=50, photons_per_chunk=1500)
    tracks = data.tracks
    assert data.provenance["n_chunks"] >= 4
    np.testing.assert_array_equal(np.unique(tracks.event_id), [0, 1, 2, 3])
    np.testing.assert_array_equal(np.bincount(tracks.event_id), [50] * 4)
    np.testing.assert_array_equal(tracks.photon[:50], np.arange(50))
    assert np.all(tracks.kind[tracks.offsets[:-1]] == Interaction.EMIT)
    assert np.all(np.isin(tracks.fate, [int(f) for f in (3, 4, 5, 6, 7)]))


def test_generate_matches_sample_prior_and_simulate():
    config = preset("absorb")
    data = generate(30, config=config, seed=9, device="cpu")
    theta = sample_prior(30, config, 9)
    for name in ("x", "y", "energy", "t0"):
        np.testing.assert_array_equal(getattr(data.truth, name), theta[name])
    again = simulate(theta["x"], theta["y"], theta["energy"], theta["t0"], config=config, seed=9, device="cpu")
    np.testing.assert_array_equal(again.response.charge, data.response.charge)
    np.testing.assert_array_equal(again.response.time_mean, data.response.time_mean)


def test_prior_distributions():
    config = preset("full", r_max=0.8, energy_min=1.0, energy_max=2.0, t0_min=-5.0, t0_max=5.0)
    theta = sample_prior(50_000, config, np.random.default_rng(1))
    r2 = theta["x"].astype(float) ** 2 + theta["y"].astype(float) ** 2
    assert r2.max() <= 0.64 + 1e-6
    assert ks_distance(r2, lambda v: v / 0.64) < KS_LIMIT  # r^2 is uniform
    assert ks_distance(theta["energy"], lambda e: e - 1.0) < KS_LIMIT
    assert ks_distance(theta["t0"], lambda t: (t + 5.0) / 10.0) < KS_LIMIT
    angle = np.arctan2(theta["y"], theta["x"])
    assert ks_distance(angle, lambda a: (a + np.pi) / (2 * np.pi)) < KS_LIMIT


def test_fixed_vertex_energy_and_photon_number():
    config = preset("full", fixed_vertex=(0.3, -0.4), fixed_energy=2.5, photon_statistics="fixed")
    data = generate(40, config=config, seed=3, device="cpu")
    assert np.all(data.truth.x == np.float32(0.3))
    assert np.all(data.truth.y == np.float32(-0.4))
    assert np.all(data.truth.energy == np.float32(2.5))
    assert np.all(data.photons.n_emitted == 2500)
    assert len(np.unique(data.truth.t0)) == 40
    rounded = generate(5, config=config.replace(fixed_energy=1.0007), seed=3, device="cpu")  # Y E = 1000.7
    assert np.all(rounded.photons.n_emitted == 1001)
    poisson = generate(400, config=config.replace(photon_statistics="poisson"), seed=3, device="cpu")
    n = poisson.photons.n_emitted
    assert abs(n.mean() - 2500) < N_SIGMA * math.sqrt(2500 / 400)
    assert n.var() == pytest.approx(2500, rel=0.3)


def test_dark_noise_only():
    config = preset("full", light_yield=0.0, dark_rate=5e6, readout_window=200.0)  # 1 noise hit per sensor
    data = generate(300, config=config, seed=4, device="cpu", save_hits=True)
    check_conservation(data)
    assert data.photons.n_emitted.sum() == 0
    charge = data.response.charge
    assert abs(charge.mean() - 1.0) < N_SIGMA * math.sqrt(1.0 / charge.size)
    assert np.all(data.hits.is_noise)
    assert ks_distance(data.hits.time, lambda t: t / 200.0) < KS_LIMIT


def test_hits_are_grouped_by_event():
    data = generate(30, config=preset("full", dark_rate=3e6), seed=12, device="cpu", save_hits=True)
    h = data.hits
    assert h.is_noise.sum() > 0
    assert np.all(np.diff(h.event_id) >= 0)
    same_event = np.diff(h.event_id) == 0
    assert np.all(np.diff(h.is_noise.astype(int))[same_event] >= 0)  # detected photons before noise hits


def test_readout_matches_the_hits(full_small):
    h, response = full_small.hits, full_small.response
    i, j = 5, int(np.argmax(response.charge[5]))
    sel = (h.event_id == full_small.truth.event_id[i]) & (h.sensor == j)
    assert sel.sum() == response.charge[i, j]
    assert h.time[sel].mean() == pytest.approx(response.time_mean[i, j], abs=1e-4)
    assert h.time[sel].min() == response.time_first[i, j]


def test_inputs_are_copied():
    x = np.array([0.1, 0.2], dtype=np.float32)
    data = simulate(x, 0.0, np.float32(1.0), seed=1, device="cpu")
    x[0] = 0.9
    assert data.truth.x[0] == np.float32(0.1)
    assert data.truth.y.flags.writeable


def test_first_event_id_and_input_checks():
    data = simulate([0.0, 0.1], [0.0, 0.2], 1.0, seed=1, device="cpu", first_event_id=1000)
    np.testing.assert_array_equal(data.truth.event_id, [1000, 1001])
    with pytest.raises(ValueError, match="outside the detector"):
        simulate(1.5, 0.0, 1.0, device="cpu")
    with pytest.raises(ValueError, match="seed"):
        generate(10, seed=-1, device="cpu")
    with pytest.raises(ValueError, match="at least one event"):
        generate(0, device="cpu")
    with pytest.raises(ValueError, match="energies"):
        simulate(0.0, 0.0, -1.0, device="cpu")
    for bad in ({"photons_per_chunk": 0}, {"save_tracks": -1}, {"track_photons": 0}, {"first_event_id": -5}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            generate(2, device="cpu", **bad)
    for bad_device in ("gpu", "cuda0", "mps"):
        with pytest.raises(ValueError, match="unsupported device"):
            generate(2, device=bad_device)


def test_numpy_options_are_stored_as_plain_json_values(tmp_path):
    data = generate(
        3,
        seed=np.int64(7),
        device="cpu",
        save_tracks=np.int64(1),
        track_photons=np.int32(10),
        first_event_id=np.int64(100),
        save_hits=np.bool_(True),
    )
    assert data.truth.event_id[0] == 100
    save(data, tmp_path / "numpy_options.h5")  # the provenance is written as JSON
    p = load(tmp_path / "numpy_options.h5").provenance
    stored = (p["seed"], p["save_tracks"], p["track_photons"], p["first_event_id"], p["save_hits"])
    assert stored == (7, 1, 10, 100, True)


def _response_hash(result):
    return hashlib.sha1(result.response.charge.tobytes() + result.response.time_mean.tobytes()).hexdigest()


def _hash_of_fresh_process(env):
    """Start a fresh interpreter whose first simulation uses 16 threads; it prints a hash of the response arrays."""
    code = (
        "import hashlib, sys, torch\n"
        "torch.set_num_threads(16)\n"
        "from prob_reco_sbi.flash import generate\n"
        "d = generate(1000, seed=5, device='cpu')\n"
        "print(hashlib.sha1(d.response.charge.tobytes() + d.response.time_mean.tobytes()).hexdigest())\n"
    )
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)


@pytest.mark.slow
def test_first_simulation_of_a_fresh_process_is_reproducible():
    """Regression test for the multi-threaded first call of a process (see transport.warm_up_cpu_kernels).

    This process has simulated before, so its result is the canonical one; every fresh process must reproduce it.
    Marked slow because it starts six interpreters with 16 threads each; the continuous integration runs it.
    """
    reference = _response_hash(generate(1000, seed=5, device="cpu"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    processes = [_hash_of_fresh_process(env) for _ in range(6)]
    hashes = [p.communicate(timeout=300)[0].strip() for p in processes]
    assert all(p.returncode == 0 for p in processes)
    assert hashes == [reference] * len(hashes), hashes


def test_result_containers(full_small):
    assert len(full_small) == full_small.n_events == 200
    assert full_small.n_sensors == 32
    assert "with hits and tracks" in repr(full_small)
    det = full_small.detector
    assert det.n_sensors == 32
    assert det.sensor_x[0] == pytest.approx(1.0)
    p = full_small.provenance
    for key in ("seed", "device", "photons_per_chunk", "package_version", "torch_version", "created", "wall_time_s"):
        assert key in p
    assert p["device"] == "cpu"
    assert p["seed"] == 11
