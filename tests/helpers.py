"""Statistical helpers and common checks of the test suite."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from prob_reco_sbi.flash import SimResult

#: Tolerance of the statistical checks in standard deviations (the tests must pass for any seed and torch version).
N_SIGMA = 5.0

#: sqrt(n) D of the Kolmogorov-Smirnov test exceeds this value with probability below 0.1 % for a correct pdf.
KS_LIMIT = 1.95


def assert_binomial(k: float, n: float, p: float, n_sigma: float = N_SIGMA) -> None:
    """Assert that ``k`` successes in ``n`` independent trials are compatible with the probability ``p``."""
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(k / n - p) < n_sigma * sigma, f"fraction {k / n:.5f}, expected {p:.5f} (sigma {sigma:.2g})"


def ks_distance(samples: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    """Kolmogorov-Smirnov distance ``sqrt(n) D`` between samples and a cumulative distribution function."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    n = len(x)
    f = cdf(x)
    d = max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(n) / n))
    return math.sqrt(n) * float(d)


def check_conservation(result: SimResult) -> None:
    """Exact photon bookkeeping of every event and consistency of the readout."""
    p = result.photons
    ended = p.n_detected + p.n_absorbed_bulk + p.n_absorbed_wall + p.n_absorbed_sensor + p.n_lost
    np.testing.assert_array_equal(p.n_emitted, ended)
    np.testing.assert_array_equal(p.n_detected, p.n_detected_direct + p.n_detected_scattered + p.n_detected_reflected)
    charge, time_mean, time_first = result.response.charge, result.response.time_mean, result.response.time_first
    np.testing.assert_array_equal(charge.sum(axis=1), p.n_detected + p.n_noise)
    np.testing.assert_array_equal(np.isnan(time_mean), charge == 0)
    np.testing.assert_array_equal(np.isnan(time_first), charge == 0)
    hit = charge > 0
    assert np.all(time_first[hit] <= time_mean[hit] + 1e-3)
    if result.hits is not None:
        h = result.hits
        index = h.event_id - result.truth.event_id[0]
        per_sensor = np.zeros_like(charge)
        np.add.at(per_sensor, (index, h.sensor), 1)
        np.testing.assert_array_equal(per_sensor, charge)
        np.testing.assert_array_equal(np.bincount(index[h.is_noise], minlength=len(result)), p.n_noise)
        # the per-event categories follow the rule: reflected if reflected at least once, else scattered if scattered
        # at least once, else direct (signal hits only)
        signal = ~h.is_noise
        reflected = signal & (h.n_reflect > 0)
        scattered = signal & ~reflected & (h.n_scatter > 0)
        direct = signal & ~reflected & ~scattered
        n = len(result)
        np.testing.assert_array_equal(np.bincount(index[reflected], minlength=n), p.n_detected_reflected)
        np.testing.assert_array_equal(np.bincount(index[scattered], minlength=n), p.n_detected_scattered)
        np.testing.assert_array_equal(np.bincount(index[direct], minlength=n), p.n_detected_direct)
