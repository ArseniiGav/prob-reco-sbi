"""Tests of neural posterior estimation: the batched spline, coordinates, models, training and the references."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import KS_LIMIT, ks_distance
from prob_reco_sbi.flash import ganged_channels, ganged_input, generate, preset, save, simulate
from prob_reco_sbi.flash.cli import build_parser
from prob_reco_sbi.flows1d import RQSpline
from prob_reco_sbi.npe import (
    AutoregressiveFlow,
    ConditionalFlow1D,
    GaussianChain,
    PosteriorEstimator,
    credible_levels,
    detection_probability,
    energy_evidence,
    energy_from_unbounded,
    energy_posterior,
    energy_to_unbounded,
    fit,
    from_unbounded,
    grid_posterior,
    inside_prior,
    load_model,
    load_pairs,
    mean_loss,
    rq_knots,
    rq_spline,
    rq_spline_inverse,
    save_model,
    to_pairs,
    to_unbounded,
    vertex_from_unbounded,
    vertex_to_unbounded,
)

ROOT = Path(__file__).resolve().parents[1]
SHIFT = math.log(math.expm1(1 - 1e-3))  # derivative offset between rq_spline and RQSpline


def randomize_heads(model, scale=0.3, seed=0):
    """Give the zero-initialized last layers random weights, so that the model is not the identity."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def standardized(model, n=2000, seed=0):
    """Set the standardization of a model from random parameters inside the prior and random observations."""
    g = torch.Generator().manual_seed(seed)
    if model.unbounded and model.theta_dim >= 2:
        r = torch.sqrt(torch.rand(n, generator=g, dtype=torch.float64))
        phi = 2 * math.pi * torch.rand(n, generator=g, dtype=torch.float64)
        cols = [r * torch.cos(phi), r * torch.sin(phi)]
        if model.theta_dim == 3:
            cols.append(0.5 + 4.5 * torch.rand(n, generator=g, dtype=torch.float64))
        theta = torch.stack(cols, -1)
    elif model.unbounded:
        theta = 0.5 + 4.5 * torch.rand(n, 1, generator=g, dtype=torch.float64)
    else:
        theta = torch.randn(n, model.theta_dim, generator=g, dtype=torch.float64)
    model.set_standardization(theta, torch.randn(n, model.obs_dim, generator=g, dtype=torch.float64))
    return model


# --------------------------------------------------------------------------------------------------------- spline
def random_rqspline(bins=8, bound=5.0, seed=0):
    spline = RQSpline(bins=bins, bound=bound).double()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in spline.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64))
    params = torch.cat([spline.widths, spline.heights, spline.derivatives - SHIFT]).detach()
    return spline, params


def test_batched_spline_equals_rqspline_for_shared_parameters():
    spline, params = random_rqspline()
    theta = torch.linspace(-7, 7, 2001, dtype=torch.float64)
    z_ref, log_det_ref = spline(theta)
    z, log_det = rq_spline(theta, params, 5.0)  # one row of parameters, broadcast to every value
    torch.testing.assert_close(z, z_ref.detach(), rtol=0, atol=1e-12)
    torch.testing.assert_close(log_det, log_det_ref.detach(), rtol=0, atol=1e-12)
    torch.testing.assert_close(rq_spline_inverse(z, params, 5.0), spline.inverse(z).detach(), rtol=0, atol=1e-12)


def test_zero_parameters_give_the_identity():
    theta = torch.linspace(-6, 6, 101, dtype=torch.float64)
    z, log_det = rq_spline(theta, torch.zeros(101, 3 * 16 - 1, dtype=torch.float64), 5.0)
    torch.testing.assert_close(z, theta, rtol=0, atol=1e-12)
    torch.testing.assert_close(log_det, torch.zeros_like(theta), rtol=0, atol=1e-12)
    x_knots, _, d = rq_knots(torch.zeros(3 * 16 - 1), 5.0)
    torch.testing.assert_close(x_knots, torch.linspace(-5, 5, 17), rtol=0, atol=1e-6)
    torch.testing.assert_close(d, torch.ones(17))


def test_inverse_of_forward_is_the_identity_with_one_spline_per_value():
    g = torch.Generator().manual_seed(1)
    params = 2 * torch.randn(5000, 3 * 8 - 1, generator=g, dtype=torch.float64)
    theta = 7 * torch.randn(5000, generator=g, dtype=torch.float64)  # inside and outside [-B, B]
    z, _ = rq_spline(theta, params, 5.0)
    torch.testing.assert_close(rq_spline_inverse(z, params, 5.0), theta, rtol=0, atol=1e-9)


def test_spline_log_derivative_against_autograd():
    g = torch.Generator().manual_seed(2)
    params = torch.randn(300, 3 * 8 - 1, generator=g, dtype=torch.float64)
    theta = (4 * torch.randn(300, generator=g, dtype=torch.float64)).requires_grad_(True)
    z, log_det = rq_spline(theta, params, 5.0)
    (grad,) = torch.autograd.grad(z.sum(), theta)  # every z depends on its own theta only
    torch.testing.assert_close(log_det, torch.log(grad), rtol=0, atol=1e-10)


def test_spline_checks_the_number_of_parameters():
    with pytest.raises(ValueError, match="3 K - 1"):
        rq_spline(torch.zeros(3), torch.zeros(3, 7))


# ---------------------------------------------------------------------------------------------------- coordinates
def test_coordinate_round_trip():
    g = torch.Generator().manual_seed(3)
    r = 0.999 * torch.sqrt(torch.rand(1000, generator=g, dtype=torch.float64))
    phi = 2 * math.pi * torch.rand(1000, generator=g, dtype=torch.float64)
    energy = 0.5 + 4.5 * torch.rand(1000, generator=g, dtype=torch.float64)
    theta = torch.stack([r * torch.cos(phi), r * torch.sin(phi), energy], -1)
    u, _ = to_unbounded(theta)
    torch.testing.assert_close(from_unbounded(u), theta, rtol=0, atol=1e-9)
    torch.testing.assert_close(vertex_from_unbounded(vertex_to_unbounded(theta[:, :2])[0]), theta[:, :2])
    torch.testing.assert_close(energy_from_unbounded(energy_to_unbounded(energy)[0]), energy)
    # every unbounded point maps into the prior (far out: onto its edge, up to rounding)
    wild = 30 * torch.randn(1000, 3, generator=g, dtype=torch.float64)
    assert inside_prior(from_unbounded(wild), tolerance=1e-12).all()


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_coordinate_log_jacobian_against_autograd(dim):
    points = [[0.3, -0.2, 1.7], [0.0, 0.0, 0.6], [-0.9, 0.35, 4.8], [1e-7, -2e-7, 2.5], [0.7, 0.69, 3.3]]
    for point in points:
        theta = torch.tensor(point[:2] + point[2:] if dim == 3 else (point[:2] if dim == 2 else point[2:]))
        theta = theta.double()
        jac = torch.autograd.functional.jacobian(lambda t: to_unbounded(t)[0], theta)
        expected = torch.linalg.slogdet(jac)[1]
        torch.testing.assert_close(to_unbounded(theta)[1], expected, rtol=0, atol=1e-8)


def test_energy_transform_works_on_plain_arrays():
    u, log_det = energy_to_unbounded(np.array([0.5 + 2.25]))
    assert u.item() == pytest.approx(0.0)
    assert log_det.item() == pytest.approx(math.log(4 / 4.5))


# ------------------------------------------------------------------------------------------------------- models
def test_default_chain_has_the_size_of_the_plan():
    assert AutoregressiveFlow(obs_dim=64).n_params == 52_453
    assert AutoregressiveFlow(obs_dim=64).parameter_names == ("x0", "y0", "E")


@pytest.mark.parametrize("cls", [AutoregressiveFlow, GaussianChain])
def test_untrained_models_are_the_standard_normal_in_flow_coordinates(cls):
    model = standardized(cls(obs_dim=5).double())
    theta = torch.tensor([[0.2, -0.4, 1.3], [0.0, 0.9, 4.2]], dtype=torch.float64)
    theta_t, log_det = model.theta_to_tilde(theta)
    expected = -0.5 * (theta_t**2).sum(-1) - 1.5 * math.log(2 * math.pi) + log_det
    torch.testing.assert_close(model.log_prob(theta, torch.randn(2, 5, dtype=torch.float64)), expected)


@pytest.mark.parametrize("unbounded", [True, False])
@pytest.mark.parametrize("cls", [ConditionalFlow1D, AutoregressiveFlow, GaussianChain])
def test_density_of_one_parameter_integrates_to_one(cls, unbounded):
    torch.manual_seed(0)
    kwargs = {"obs_dim": 2, "unbounded": unbounded}
    if cls is not ConditionalFlow1D:
        kwargs |= {"theta_dim": 1, "hidden": (16,), "n_embed": 8, "head": (16,)}
    model = randomize_heads(standardized(cls(**kwargs).double()))
    x = torch.tensor([0.3, -1.1], dtype=torch.float64)
    grid = torch.linspace(-30.0, 30.0, 400_001, dtype=torch.float64)
    if unbounded:  # uniform in the logit, so that narrow features of the random spline are resolved near the edges
        grid = energy_from_unbounded(grid)
    with torch.no_grad():
        density = torch.exp(model.log_prob(grid[:, None], x))
    assert torch.trapezoid(density, grid).item() == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("cls", [AutoregressiveFlow, GaussianChain])
def test_density_of_two_parameters_integrates_to_one_over_the_disc(cls):
    torch.manual_seed(0)
    model = cls(obs_dim=3, theta_dim=2, hidden=(16,), n_embed=8, head=(16,)).double()
    model = randomize_heads(standardized(model), scale=0.2)
    # polar grid, dense near the wall: r = tanh(s) with s uniform
    s = torch.linspace(0.0, 9.0, 1501, dtype=torch.float64)
    r = torch.tanh(s)
    phi = torch.linspace(0.0, 2 * math.pi, 257, dtype=torch.float64)[:-1]
    rr, pp = torch.meshgrid(r, phi, indexing="ij")
    theta = torch.stack([rr * torch.cos(pp), rr * torch.sin(pp)], -1)
    with torch.no_grad():
        density = torch.exp(model.log_prob(theta, torch.tensor([0.5, 0.1, -0.3], dtype=torch.float64)))
    ring = density.mean(-1) * 2 * math.pi * r  # integral over the angle at every radius
    assert torch.trapezoid(ring, r).item() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("cls", [AutoregressiveFlow, GaussianChain])
def test_samples_sent_back_are_standard_normal(cls):
    torch.manual_seed(0)
    model = randomize_heads(standardized(cls(obs_dim=4, hidden=(32,), n_embed=8, head=(32,)).double()))
    x = torch.tensor([0.1, 0.2, -0.5, 1.0], dtype=torch.float64)
    samples = model.sample(x, 20_000, seed=4)
    assert samples.shape == (20_000, 3)
    assert inside_prior(samples).all()
    with torch.no_grad():
        z, _ = model(samples, x)
    np.testing.assert_allclose(z.mean(0).numpy(), 0, atol=5 * 1 / math.sqrt(20_000))
    np.testing.assert_allclose(np.cov(z.numpy().T), np.eye(3), atol=0.05)
    normal_cdf = lambda v: 0.5 * (1 + np.vectorize(math.erf)(v / math.sqrt(2)))  # noqa: E731
    for i in range(3):
        assert ks_distance(z[:, i].numpy(), normal_cdf) < KS_LIMIT


def test_sample_shapes_and_seed():
    model = standardized(AutoregressiveFlow(obs_dim=6, hidden=(16,), n_embed=8, head=(16,)))
    x = torch.randn(4, 6)
    assert model.sample(x, 10, seed=1).shape == (10, 4, 3)
    assert model.sample(x[0], 10, seed=1).shape == (10, 3)
    torch.testing.assert_close(model.sample(x, 10, seed=1), model.sample(x, 10, seed=1))
    assert model.log_prob(model.sample(x, 10, seed=1), x).shape == (10, 4)  # samples broadcast against observations


def test_log_prob_is_minus_infinity_outside_the_prior():
    model = standardized(AutoregressiveFlow(obs_dim=3, hidden=(8,), n_embed=4, head=(8,)))
    theta = torch.tensor([[0.2, 0.1, 6.0], [1.2, 0.0, 2.0], [0.2, 0.1, 2.0]])
    lp = model.log_prob(theta, torch.zeros(3))
    assert torch.isinf(lp[:2]).all()
    assert torch.isfinite(lp[2])


def test_models_check_their_shapes():
    model = AutoregressiveFlow(obs_dim=3)
    with pytest.raises(ValueError, match="theta must have shape"):
        model.log_prob(torch.zeros(5, 2), torch.zeros(5, 3))
    with pytest.raises(ValueError, match="x must have shape"):
        model.log_prob(torch.zeros(5, 3), torch.zeros(5, 4))
    with pytest.raises(ValueError, match="theta_dim 1 to 3"):
        AutoregressiveFlow(obs_dim=3, theta_dim=4)
    assert AutoregressiveFlow(obs_dim=3, theta_dim=4, unbounded=False).theta_dim == 4
    assert isinstance(model, PosteriorEstimator)


# ------------------------------------------------------------------------------------------------------- training
def toy_pairs(n=1200, seed=0):
    """Pairs of a toy with a known answer: E uniform in the prior, x = (log E + noise, noise)."""
    g = torch.Generator().manual_seed(seed)
    energy = 0.5 + 4.5 * torch.rand(n, 1, generator=g)
    x = torch.cat([torch.log(energy) + 0.05 * torch.randn(n, 1, generator=g), torch.randn(n, 1, generator=g)], -1)
    return energy, x


def test_fit_records_the_run_and_restores_the_best_state():
    theta, x = toy_pairs()
    torch.manual_seed(0)
    model = ConditionalFlow1D(obs_dim=2, bins=8, hidden=(16,))
    history = fit(model, theta, x, epochs=3, batch_size=128, lr=3e-3, val_fraction=0.25, every=1, extra_steps=[2, 5])
    assert history.n_train == 900
    assert history.n_val == 300
    assert history.steps_per_epoch == 8
    assert history.loss.shape == history.learning_rate.shape == (24,)
    assert history.train_loss.shape == history.val_loss.shape == (4,)
    assert list(history.snapshot_step) == [0, 2, 5, 8, 16, 24]
    assert history.val_loss[1:].min() < history.val_loss[0]  # it learns
    assert mean_loss(model, theta[900:], x[900:]) == pytest.approx(history.best_val_loss, rel=1e-5)
    history.restore(model, 0)
    assert mean_loss(model, theta[900:], x[900:]) == pytest.approx(history.val_loss[0], rel=1e-5)


def test_fit_is_reproducible_with_a_seed():
    theta, x = toy_pairs(400)
    runs = []
    for _ in range(2):
        torch.manual_seed(3)
        model = ConditionalFlow1D(obs_dim=2, bins=4, hidden=(8,))
        runs.append(fit(model, theta, x, epochs=2, batch_size=64, seed=7).loss)
    np.testing.assert_array_equal(*runs)


def test_fit_with_explicit_validation_and_errors():
    theta, x = toy_pairs(400)
    model = ConditionalFlow1D(obs_dim=2, bins=4, hidden=(8,))
    history = fit(model, theta[:300], x[:300], epochs=1, validation=(theta[300:], x[300:]))
    assert (history.n_train, history.n_val) == (300, 100)
    bad = theta.clone()
    bad[5] = 7.0  # outside the prior of the model: log q^ = -inf
    with pytest.raises(FloatingPointError, match="loss is"):
        fit(model, bad, x, epochs=1, standardize=False, val_fraction=0)


def test_save_and_load_model(tmp_path):
    torch.manual_seed(0)
    model = standardized(AutoregressiveFlow(obs_dim=5, hidden=(16,), n_embed=8, head=(16,)))
    randomize_heads(model)
    theta, x = model.sample(torch.zeros(5), 7, seed=1), torch.randn(7, 5)
    path = save_model(tmp_path / "flow.pt", model, None, budget="test", seeds=[1, 2])
    loaded, history, info = load_model(path)
    assert history is None
    assert info == {"budget": "test", "seeds": [1, 2]}
    torch.testing.assert_close(loaded.log_prob(theta, x), model.log_prob(theta, x))

    model = ConditionalFlow1D(obs_dim=2, bins=4, hidden=(8,))
    pairs = toy_pairs(300)
    history = fit(model, *pairs, epochs=1)
    loaded, history_back, _ = load_model(save_model(tmp_path / "cf.pt", model, history))
    np.testing.assert_array_equal(history_back.val_loss, history.val_loss)
    assert len(history_back.states) == len(history.states)
    torch.testing.assert_close(loaded.log_prob(*pairs), model.log_prob(*pairs))


# ----------------------------------------------------------------------------------------------------------- data
def test_ganged_input_is_identical_for_a_response_and_its_mirror_image(full_small):
    charge, tau = full_small.response.charge, full_small.response.time_mean
    mirror = (32 - np.arange(32)) % 32
    a = ganged_input(charge, tau)
    assert a.shape == (len(charge), 34)
    np.testing.assert_array_equal(a, ganged_input(charge[:, mirror], tau[:, mirror]))
    assert not np.array_equal(charge, charge[:, mirror])
    np.testing.assert_array_equal(
        np.exp(a[:, :17]).round() - 1, charge @ (ganged_channels(32)[:, None] == np.arange(17))
    )
    assert ganged_input(charge, use_time=False).shape == (len(charge), 17)


def test_load_pairs_in_every_representation(tmp_path, full_small):
    path = save(full_small, tmp_path / "events.h5")
    sizes = {"charge+time": 64, "charge": 32, "q": 1, "ganged": 34}
    for representation, size in sizes.items():
        theta, x = load_pairs(path, representation)
        assert theta.shape == (len(full_small), 3)
        assert x.shape == (len(full_small), size)
        assert theta.dtype == x.dtype == torch.float32
    np.testing.assert_allclose(theta[:, 2].numpy(), full_small.truth.energy)
    np.testing.assert_allclose(load_pairs(path, "q")[1][:, 0].numpy(), np.log(full_small.response.charge.sum(1)))
    with pytest.raises(ValueError, match="unknown representation"):
        load_pairs(path, "hits")


def test_npe_data_commands_are_valid_cli_commands():
    spec = importlib.util.spec_from_file_location("make_npe_data", ROOT / "examples" / "make_npe_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = build_parser()
    sizes = {}
    for command in module.npe_commands():
        args = parser.parse_args(command)
        assert args.command == "generate"
        sizes[module.output_of(command).as_posix()] = args.n_events
    assert sizes == {
        "data/n1_center.h5": 60_000,
        "data/n2_absorb_train.h5": 310_000,
        "data/n2_absorb_test.h5": 2000,
        "data/n3_full_train.h5": 310_000,
        "data/n3_full_test.h5": 5000,
    }


def test_parameters_fixed_by_the_prior_are_not_inferred():
    at_center = generate(20, config=preset("full", fixed_vertex=(0.0, 0.0)), seed=1, device="cpu")
    theta, x = to_pairs(at_center, "q")
    assert theta.shape == (20, 1)
    assert x.shape == (20, 1)
    fixed_energy = generate(20, config=preset("absorb", fixed_energy=2.0), seed=1, device="cpu")
    assert to_pairs(fixed_energy, "charge")[0].shape == (20, 2)


# ----------------------------------------------------------------------------------------------------- references
def test_detection_probability_at_the_center():
    config = preset("absorb")
    a, radius, lam = config.geometry.sensor_width, config.geometry.radius, config.optics.absorption_length
    kappa = detection_probability(0.0, 0.0, config)
    np.testing.assert_allclose(kappa, a * math.exp(-radius / lam) / (2 * math.pi * radius), rtol=1e-12)
    assert detection_probability(0.0, 0.0, "straight").sum() == pytest.approx(config.geometry.photocoverage)


def test_detection_probability_symmetries_and_limits():
    x0, y0 = 0.42, -0.31
    turn = 2 * math.pi / 32
    rotated = detection_probability(
        x0 * math.cos(turn) - y0 * math.sin(turn), x0 * math.sin(turn) + y0 * math.cos(turn)
    )
    np.testing.assert_allclose(rotated, np.roll(detection_probability(x0, y0), 1), rtol=1e-12)
    np.testing.assert_allclose(detection_probability(x0, -y0), detection_probability(x0, y0)[(32 - np.arange(32)) % 32])
    # straight light is the limit of a long absorption length; the quadrature converges near the wall
    far = preset("absorb", absorption_length=1e9)
    np.testing.assert_allclose(detection_probability(0.97, 0.02, far), detection_probability(0.97, 0.02, "straight"))
    np.testing.assert_allclose(
        detection_probability(0.995, 0.01, n_nodes=16), detection_probability(0.995, 0.01, n_nodes=200), rtol=1e-6
    )
    assert detection_probability(np.zeros((4, 5)), 0.1).shape == (4, 5, 32)
    with pytest.raises(ValueError, match="straight light"):
        detection_probability(0.0, 0.0, "full")
    with pytest.raises(ValueError, match="inside the detector"):
        detection_probability(1.0, 0.0)


def test_energy_posterior_is_a_normalized_truncated_gamma():
    energy = np.linspace(0.5, 5.0, 40_001)
    for q, kappa in [(500, 0.26), (140, 0.27), (1290, 0.26), (2090, 0.26)]:  # the last: an event of 8 MeV
        post = energy_posterior(q, kappa)
        pdf = post.pdf(energy)
        assert np.trapezoid(pdf, energy) == pytest.approx(1.0, abs=2e-4)
        assert post.mean == pytest.approx(np.trapezoid(energy * pdf, energy), rel=1e-4)
        assert post.cdf(5.0) == pytest.approx(1.0)
        assert post.cdf(0.5) == 0.0
    post = energy_posterior(500, 0.26)
    mode = 500 / (1000 * 0.26)
    assert post.pdf(mode) == pytest.approx(post.pdf(energy).max(), rel=1e-6)
    assert post.pdf(6.0) == 0.0


def test_energy_evidence_is_the_integral_of_the_poisson_likelihood():
    q, kappa = 400, 0.24
    energy = np.linspace(0.5, 5.0, 40_001)
    rate = 1000 * kappa * energy
    poisson = np.exp(q * np.log(rate) - rate - math.lgamma(q + 1))
    expected = np.trapezoid(poisson, energy) / 4.5
    assert energy_evidence(q, kappa, log=False) == pytest.approx(expected, rel=1e-6)
    assert energy_evidence(q, kappa) == pytest.approx(math.log(expected), abs=1e-6)


def test_energy_references_take_a_preset_name_and_an_array_of_kappa():
    energy = np.linspace(0.5, 5.0, 4001)
    kappa = np.array([0.22, 0.26, 0.31])
    by_name = energy_posterior(600, kappa, "absorb").pdf(energy[:, None])
    assert by_name.shape == (len(energy), len(kappa))
    assert np.array_equal(by_name, energy_posterior(600, kappa, preset("absorb")).pdf(energy[:, None]))
    assert np.array_equal(by_name, energy_posterior(600, kappa).pdf(energy[:, None]))
    assert np.trapezoid(by_name, energy, axis=0) == pytest.approx(np.ones(3), abs=1e-4)
    assert np.array_equal(energy_evidence(600, kappa, "absorb"), energy_evidence(600, kappa, preset("absorb")))
    brighter = energy_evidence(600, kappa, preset("absorb", light_yield=1200.0))
    assert np.array_equal(brighter, energy_evidence(600, kappa, (0.5, 5.0), light_yield=1200.0))


def central_event(energy=2.0):
    """The expected charges of an event at the center, rounded: a deterministic event of preset absorb."""
    return np.round(1000 * energy * detection_probability(0.0, 0.0, "absorb"))


def test_grid_posterior_at_the_center_agrees_with_the_energy_posterior():
    charge = central_event()
    grid = grid_posterior(charge, "absorb", n_grid=101)
    exact = energy_posterior(charge.sum(), detection_probability(0.0, 0.0, "absorb").sum())
    np.testing.assert_allclose(grid.mean[:2], 0, atol=1e-6)
    assert grid.mean[2] == pytest.approx(exact.mean, abs=0.02 * exact.std)
    assert grid.energy_width == pytest.approx(exact.std, rel=0.02)
    np.testing.assert_allclose(grid.energy_density, exact.pdf(grid.energy), atol=0.02 * exact.pdf(exact.mean))
    assert grid.density.sum() * grid.cell_area == pytest.approx(1.0)
    assert grid.edge_mass < 1e-4


def test_grid_posterior_does_not_depend_on_the_grid_step():
    charge = np.round(1000 * 1.5 * detection_probability(0.62, -0.41, "absorb"))
    coarse, fine = grid_posterior(charge, n_grid=101), grid_posterior(charge, n_grid=161)
    np.testing.assert_allclose(coarse.mean, fine.mean, atol=1e-3 * fine.std.min())
    np.testing.assert_allclose(coarse.std, fine.std, rtol=1e-3)


def test_credible_levels_of_a_gaussian():
    axis = np.linspace(-6, 6, 601)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    density = np.exp(-(xx**2 + yy**2) / 2) / (2 * np.pi)
    levels = credible_levels(density, (0.68, 0.95))
    # for a 2D standard normal the region of probability p is bounded by the level (1 - p) / (2 pi)
    np.testing.assert_allclose(levels, (1 - np.array([0.68, 0.95])) / (2 * np.pi), rtol=1e-2)
    with pytest.raises(ValueError, match="probabilities"):
        credible_levels(density, 1.5)


# ----------------------------------------------------------------------------------------------------------- slow
@pytest.mark.slow
def test_detection_probability_against_the_simulation():
    n = 4000
    events = generate(n, config=preset("absorb", fixed_vertex=(0.5, 0.2), fixed_energy=2.0), seed=5, device="cpu")
    expected = 1000 * 2.0 * detection_probability(0.5, 0.2, "absorb")
    mean = events.response.charge.mean(0)
    chi2_per_sensor = float((((mean - expected) ** 2) / (expected / n)).sum() / 32)
    print(f"chi2 / 32 = {chi2_per_sensor:.3f}")
    assert 0.3 < chi2_per_sensor < 2.0, chi2_per_sensor


@pytest.mark.slow
def test_conditional_flow_reaches_the_exact_energy_posterior():
    """N1 in small: a 16-bin conditional flow on 20 000 events at the center, against the truncated Gamma."""
    torch.manual_seed(0)
    events = generate(22_000, config=preset("full", fixed_vertex=(0.0, 0.0)), seed=31, device="cpu")
    theta, x = to_pairs(events, "q")
    kappa = float(events.response.charge.sum() / (1000 * events.truth.energy.sum()))
    flow = ConditionalFlow1D()
    fit(flow, theta, x, epochs=30, batch_size=512, val_fraction=1 / 11, seed=1)
    grid = np.linspace(0.5 + 1e-4, 5.0 - 1e-4, 4001)
    worst = 0.0
    for energy in (1.0, 2.0, 3.0, 4.0):
        q = round(1000 * kappa * energy)
        exact = energy_posterior(q, kappa).pdf(grid)
        with torch.no_grad():
            model = torch.exp(flow.log_prob(torch.as_tensor(grid)[:, None], torch.tensor([math.log(q)]))).double()
        model = model.numpy()
        kl = np.trapezoid(exact * (np.log(np.maximum(exact, 1e-300)) - np.log(np.maximum(model, 1e-300))), grid)
        worst = max(worst, kl)
    print(f"largest KL in the bulk: {worst:.4f} nats")
    assert worst < 0.02, worst


@pytest.mark.slow
def test_chain_width_against_the_exact_posterior():
    """N2 in small: a chain on 20 000 absorb events is at most 2.5 times wider than the exact posterior.

    The median ratio is 1.05 to 1.13 for five seeds; the lower limit 0.9 leaves room for other platforms.
    """
    torch.manual_seed(0)
    config = preset("absorb")
    events = generate(22_000, config=config, seed=32, device="cpu")
    theta, x = to_pairs(events, "charge")
    flow = AutoregressiveFlow(obs_dim=32)
    fit(flow, theta, x, epochs=30, batch_size=256, val_fraction=1 / 11, seed=1)
    rng = np.random.default_rng(33)
    r, phi = 0.5 * np.sqrt(rng.random(50)), 2 * np.pi * rng.random(50)
    test = simulate(r * np.cos(phi), r * np.sin(phi), rng.uniform(0.5, 5.0, 50), config=config, seed=34, device="cpu")
    samples = flow.sample(to_pairs(test, "charge")[1], 1000, seed=2).numpy()
    width = np.sqrt(samples[..., 0].var(0) + samples[..., 1].var(0))
    exact = np.array([grid_posterior(c, config, n_grid=81).vertex_width for c in test.response.charge])
    ratio = float(np.median(width / exact))
    print(f"median width ratio to the exact posterior: {ratio:.3f}")
    assert 0.9 < ratio < 2.5, ratio


@pytest.mark.slow
def test_npe_quickstart(tmp_path):
    """The NPE example runs end to end with a small budget and writes its figure where it is told to."""
    script = ROOT / "examples" / "npe_quickstart.py"
    command = [sys.executable, str(script), "--n-train", "2000", "--epochs", "2", "--out", str(tmp_path)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    assert run.returncode == 0, run.stderr
    assert "posterior mean" in run.stdout
    if importlib.util.find_spec("matplotlib") is not None:
        assert (tmp_path / "npe_quickstart.png").stat().st_size > 0


@pytest.mark.slow
def test_zuko_flow_with_the_package_coordinates():
    """N5 in small: zuko's NSF of the sheet, wrapped in the package interface, trains with fit and samples."""
    zuko = pytest.importorskip("zuko")

    class ZukoNSF(PosteriorEstimator):
        def __init__(self, obs_dim):
            super().__init__(obs_dim, 3)
            self.flow = zuko.flows.NSF(features=3, context=obs_dim, transforms=3, hidden_features=[128, 128])

        def _forward_tilde(self, theta_t, context):
            return self.flow(context).transform.call_and_ladj(theta_t)

        def _inverse_tilde(self, z, context):
            return self.flow(context).transform.inv(z)

    assert ZukoNSF(64).flow.__class__.__name__ == "NSF"
    assert sum(p.numel() for p in ZukoNSF(64).flow.parameters()) == 102_351
    events = generate(600, config=preset("absorb"), seed=35, device="cpu")
    theta, x = to_pairs(events, "charge")
    torch.manual_seed(0)
    model = ZukoNSF(32)
    history = fit(model, theta, x, epochs=2, batch_size=128)
    assert np.all(np.isfinite(history.val_loss))
    samples = model.sample(x[:3], 50, seed=1)
    assert samples.shape == (50, 3, 3)
    assert inside_prior(samples).all()
    with torch.no_grad():
        direct = model.flow(model.x_to_tilde(x[:4])).log_prob(model.theta_to_tilde(theta[:4])[0])
    torch.testing.assert_close(model.log_prob(theta[:4], x[:4]), direct + model.theta_to_tilde(theta[:4])[1])
