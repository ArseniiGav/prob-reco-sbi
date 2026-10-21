"""Statistical regression against the original implementation (slow; run with ``pytest -m slow``).

The reference numbers come from the GPU implementation used in the SBI seminar of September 2026. They are kept by the
course team and are not distributed with this repository: the tests that need them run when the environment variable
``FLASH_REFERENCE_FILE`` points to the file (default: ``data/seminar_reference.json`` next to this module, ignored by
git) and are skipped otherwise. Every comparison is a set of pulls ``(new - reference) / sigma`` with the statistical
errors of both sides, estimated from the per-event spread of the new simulation.

If the environment variable ``FLASH_CORE_DIR`` points to a directory with the original ``flash_core.py``, the
transport is also compared with it directly, on the CPU and for the same event parameters.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from prob_reco_sbi.flash import SimResult, preset, sample_prior, sensor_positions, simulate

pytestmark = pytest.mark.slow

REFERENCE_FILE = Path(os.environ.get("FLASH_REFERENCE_FILE", Path(__file__).parent / "data" / "seminar_reference.json"))
REFERENCE = json.loads(REFERENCE_FILE.read_text()) if REFERENCE_FILE.is_file() else None
needs_reference = pytest.mark.skipif(
    REFERENCE is None, reason="the seminar reference numbers are not distributed; set FLASH_REFERENCE_FILE to use them"
)
EDGES = np.linspace(0.0, 1.0, 17)  # 16 radius bins of width 1/16 m, the binning of the reference numbers
MAX_PULL = 4.5  # largest allowed |pull| (about 50 comparisons: a false alarm has a probability below 1e-3)


def assert_pulls(new, new_err, ref, ref_err, label):
    """Assert that new and reference values agree within their combined statistical errors."""
    pulls = (np.asarray(new) - np.asarray(ref)) / np.sqrt(np.asarray(new_err) ** 2 + np.asarray(ref_err) ** 2)
    chi2, ndf = float(np.sum(pulls**2)), len(pulls)
    print(f"{label}: chi2/ndf = {chi2:.1f}/{ndf}, max |pull| = {np.max(np.abs(pulls)):.2f}")
    assert np.max(np.abs(pulls)) < MAX_PULL, f"{label}: pulls {np.round(pulls, 2)}"
    assert chi2 < ndf + 5 * math.sqrt(2 * ndf), f"{label}: chi2 {chi2:.1f} for {ndf} values"


def radius_bins(x, y):
    return np.clip(np.digitize(np.hypot(x, y), EDGES) - 1, 0, len(EDGES) - 2)


def binned_mean(values, k):
    """Mean of the values in every radius bin, its error and the per-event spread."""
    n_bins = len(EDGES) - 1
    mean = np.array([values[k == b].mean() for b in range(n_bins)])
    spread = np.array([values[k == b].std() for b in range(n_bins)])
    return mean, spread / np.sqrt(np.bincount(k, minlength=n_bins)), spread


def ratio(numerator, denominator):
    """Ratio of sums over events and its error from the per-event spread (delta method)."""
    value = numerator.sum() / denominator.sum()
    residual = numerator - value * denominator
    error = residual.std() * math.sqrt(len(numerator)) / denominator.sum()
    return value, error


@pytest.fixture(scope="module")
def uniform_events():
    """Vertices uniform in the disc."""
    theta = sample_prior(60_000, preset("full"), 101)
    return theta["x"], theta["y"]


@needs_reference
@pytest.mark.parametrize("name", ["absorb", "absorb-scatter", "full"])
def test_collected_fraction_versus_radius(name, uniform_events):
    assert REFERENCE is not None
    ref = REFERENCE["collection_vs_r"]
    assert np.allclose(ref["r_edges_m"], EDGES)
    x, y = uniform_events
    energy = np.full(len(x), ref["energy_MeV"], np.float32)
    data = simulate(x, y, energy, config=preset(name), seed=7, device="cpu")
    f = data.photons.n_detected / (1000.0 * energy)
    mean, err, spread = binned_mean(f, radius_bins(x, y))
    reference_counts = ref["n_events"] * np.diff(EDGES**2)  # expected events per bin in the reference
    assert_pulls(mean, err, ref[name], spread / np.sqrt(reference_counts), f"collection vs r, {name}")
    assert data.photons.n_lost.sum() == 0


def categories(data: SimResult):
    """Category of every hit (0 direct, 1 scattered, 2 reflected) and its delay t - t0 - d_j / c_n."""
    hits = data.hits
    assert hits is not None
    index = hits.event_id - data.truth.event_id[0]
    sx, sy = sensor_positions(data.config.geometry)
    distance = np.hypot(sx[hits.sensor] - data.truth.x[index], sy[hits.sensor] - data.truth.y[index])
    delay = hits.time - data.truth.t0[index] - distance / data.config.optics.speed_of_light
    category = np.where(hits.n_reflect > 0, 2, np.where(hits.n_scatter > 0, 1, 0))
    return index, category, delay


@needs_reference
def test_disc_composition_and_delays(uniform_events):
    """Shares of direct, scattered and reflected light and their mean delays, averaged over the disc at 2 MeV."""
    assert REFERENCE is not None
    ref = REFERENCE["disc_2MeV"]
    x, y = (v[:6000] for v in uniform_events)
    data = simulate(x, y, ref["energy_MeV"], t0=0.0, config=preset("full"), seed=8, device="cpu", save_hits=True)
    index, category, delay = categories(data)
    n_events = len(data)
    total = np.bincount(index, minlength=n_events).astype(float)
    shares, share_err, mean_delay, delay_err = [], [], [], []
    for c in range(3):
        in_c = category == c
        count = np.bincount(index[in_c], minlength=n_events).astype(float)
        value, error = ratio(count, total)
        shares.append(value)
        share_err.append(error)
        value, error = ratio(np.bincount(index[in_c], weights=delay[in_c], minlength=n_events), count)
        mean_delay.append(value)
        delay_err.append(error)
    scale = math.sqrt(n_events / ref["n_events"])  # the reference has fewer events: larger errors
    assert_pulls(shares, share_err, ref["shares_direct_scattered_reflected"], np.multiply(share_err, scale), "shares")
    assert_pulls(
        mean_delay,
        delay_err,
        ref["mean_delay_ns_direct_scattered_reflected"],
        np.multiply(delay_err, scale),
        "delays",
    )


def test_direct_light_at_the_centre():
    """At the centre a photon is detected without any interaction with probability coverage * exp(-R mu)."""
    data = simulate(np.zeros(1000), 0.0, 2.0, config=preset("full"), seed=9, device="cpu")
    p = data.photons
    expected = 32 * 0.10 / (2 * math.pi) * math.exp(-1.0 * (1 / 1.5 + 1 / 2.0))
    k, n = p.n_detected_direct.sum(), p.n_emitted.sum()
    assert abs(k / n - expected) < 5 * math.sqrt(expected * (1 - expected) / n)


# ----------------------------------------------------------------------------------------------------------------------
# Direct comparison with the original transport (only where it is available)

ORIGINAL_OPTICS = {
    "absorb": {"lam_a": 1.5, "lam_s": None, "r_wall": 0.0},
    "absorb-scatter": {"lam_a": 1.5, "lam_s": 2.0, "r_wall": 0.0},
    "full": {"lam_a": 1.5, "lam_s": 2.0, "r_wall": 0.5},
}


def load_original() -> ModuleType:
    directory = os.environ.get("FLASH_CORE_DIR")
    if not directory or not (Path(directory) / "flash_core.py").exists():
        pytest.skip("set FLASH_CORE_DIR to the directory of the original flash_core.py for this comparison")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # never write into the reference directory
    try:
        spec = importlib.util.spec_from_file_location("flash_core_original", Path(directory) / "flash_core.py")
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if module.dev != "cpu":
        pytest.skip("the original module selected a GPU; run with CUDA_VISIBLE_DEVICES=''")
    return module


def run_original(core, x, y, energy, name, hits=False, chunk=3000):
    """Collected photons per event (and the hits) of the original transport, with t0 = 0."""
    torch.manual_seed(12)
    counts, parts = [], []
    for a in range(0, len(x), chunk):
        b = min(len(x), a + chunk)
        nph = torch.poisson(1000.0 * torch.as_tensor(energy[a:b])).long()
        out = core.transport(
            torch.as_tensor(x[a:b]),
            torch.as_tensor(y[a:b]),
            nph,
            torch.zeros(b - a),
            hits=hits,
            **ORIGINAL_OPTICS[name],
        )
        counts.append(out[0].sum(1).numpy())
        if hits:
            parts.append(out[2])
    return np.concatenate(counts), parts


@pytest.mark.parametrize("name", list(ORIGINAL_OPTICS))
def test_collected_fraction_against_the_original(name, uniform_events):
    core = load_original()
    x, y = (v[:30_000] for v in uniform_events)
    energy = np.ones(len(x), np.float32)
    k = radius_bins(x, y)
    new = simulate(x, y, energy, config=preset(name), seed=12, device="cpu").photons.n_detected / 1000.0
    old = run_original(core, x, y, energy, name)[0] / 1000.0
    # both codes see the same vertices: compare per-event differences, whose spread includes the correlation
    difference, error, _ = binned_mean(new - old, k)
    assert_pulls(difference, error, np.zeros_like(difference), np.zeros_like(error), f"original transport, {name}")


def test_centre_composition_against_the_original():
    core = load_original()
    n = 1500
    zeros = np.zeros(n, np.float32)
    energy = np.full(n, 2.0, np.float32)
    data = simulate(zeros, zeros, energy, t0=0.0, config=preset("full"), seed=13, device="cpu", save_hits=True)
    _, category, delay = categories(data)
    _, parts = run_original(core, zeros, zeros, energy, "full", hits=True)
    nsc = np.concatenate([p["nsc"] for p in parts])
    nrf = np.concatenate([p["nrf"] for p in parts])
    old_category = np.where(nrf > 0, 2, np.where(nsc > 0, 1, 0))
    old_delay = np.concatenate([p["t"] for p in parts]) - 1.0 / 0.2
    new_shares = np.bincount(category, minlength=3) / len(category)
    old_shares = np.bincount(old_category, minlength=3) / len(old_category)
    # photons of centre events are independent: binomial errors
    assert_pulls(
        new_shares,
        np.sqrt(new_shares * (1 - new_shares) / len(category)),
        old_shares,
        np.sqrt(old_shares * (1 - old_shares) / len(old_category)),
        "centre shares vs original",
    )
    new_mean = [delay[category == c].mean() for c in range(3)]
    old_mean = [old_delay[old_category == c].mean() for c in range(3)]
    new_err = [delay[category == c].std() / math.sqrt((category == c).sum()) for c in range(3)]
    old_err = [old_delay[old_category == c].std() / math.sqrt((old_category == c).sum()) for c in range(3)]
    assert_pulls(new_mean, new_err, old_mean, old_err, "centre delays vs original")
