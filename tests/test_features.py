import doctest

import numpy as np
import pytest

from prob_reco_sbi.flash import config, features, network_input


def test_network_input_definition():
    charge = np.array([[0, 3, 1, 0], [0, 0, 0, 0]])
    tau = np.array([[np.nan, 12.0, 16.0, np.nan], [np.nan] * 4])
    x = network_input(charge, tau)
    assert x.shape == (2, 8)
    assert x.dtype == np.float32
    np.testing.assert_allclose(x[0, :4], np.log1p([0, 3, 1, 0]), rtol=1e-6)
    tau_bar = (3 * 12.0 + 1 * 16.0) / 4
    np.testing.assert_allclose(x[0, 4:], [0, 12.0 - tau_bar, 16.0 - tau_bar, 0])
    np.testing.assert_array_equal(x[1], 0)  # an event without hits
    np.testing.assert_array_equal(network_input(charge, use_time=False), x[:, :4])


def test_relative_times_are_invariant_under_a_common_shift():
    rng = np.random.default_rng(0)
    charge = rng.poisson(2.0, size=(100, 32))
    tau = np.where(charge > 0, rng.uniform(5, 30, size=charge.shape), np.nan)
    a = network_input(charge, tau)
    b = network_input(charge, tau + 123.4)
    np.testing.assert_allclose(a, b, atol=1e-4)
    # charge-weighted relative times average to zero
    np.testing.assert_allclose((charge * a[:, 32:]).sum(axis=1), 0, atol=1e-3)


def test_network_input_checks():
    with pytest.raises(ValueError, match="time_mean is required"):
        network_input(np.ones((2, 3)))
    with pytest.raises(ValueError, match="same shape"):
        network_input(np.ones((2, 3)), np.ones((2, 4)))


@pytest.mark.parametrize("module", [config, features])
def test_docstring_examples(module):
    failures, _ = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert failures == 0
