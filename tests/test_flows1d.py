"""Unit tests of the one-dimensional normalizing flows: transforms, flows, targets, training and diagnostics."""

import math

import numpy as np
import pytest
import torch

from helpers import KS_LIMIT, ks_distance
from prob_reco_sbi.flows1d import (
    CAMEL,
    Affine,
    Flow,
    Mixture1D,
    Planar,
    Radial,
    RQSpline,
    Sigmoid,
    StandardNormal,
    Transform,
    Uniform,
    camel,
    fit,
    kl_to_truth,
    ks_statistic,
    oracle_map,
    qq_points,
    split,
)
from prob_reco_sbi.flows1d.diagnostics import _kolmogorov_sf


def randomised(transform, scale=1.0, seed=0):
    """The transform in float64 with every raw parameter drawn from N(0, scale^2)."""
    transform = transform.double()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in transform.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g, dtype=torch.float64))
    return transform


def all_transforms():
    return {
        "affine": randomised(Affine()),
        "planar": randomised(Planar(), scale=2.0),
        "radial": randomised(Radial(), scale=2.0),
        "spline": randomised(RQSpline(bins=8, bound=5.0), scale=2.0),
        "sigmoid": Sigmoid().double(),
        "oracle-normal": oracle_map(camel()),
        "oracle-uniform": oracle_map(camel(), "uniform"),
    }


GRID = torch.linspace(-9.0, 12.0, 2001, dtype=torch.float64)  # includes the spline tails beyond B = 5


@pytest.mark.parametrize("name", list(all_transforms()))
def test_inverse_undoes_forward(name):
    transform = all_transforms()[name]
    x = GRID if not name.startswith("oracle") else torch.linspace(-5.0, 10.0, 2001, dtype=torch.float64)
    if name == "sigmoid":
        x = torch.linspace(-15.0, 15.0, 2001, dtype=torch.float64)
    z, _ = transform(x)
    torch.testing.assert_close(transform.inverse(z), x, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", list(all_transforms()))
def test_log_derivative_matches_finite_differences(name):
    transform = all_transforms()[name]
    x = torch.linspace(-8.0, 10.0, 1001, dtype=torch.float64)
    h = 1e-6
    _, logdet = transform(x)
    numeric = torch.log((transform(x + h)[0] - transform(x - h)[0]) / (2 * h))
    torch.testing.assert_close(logdet, numeric, rtol=1e-5, atol=2e-5)  # rtol: F(x) close to 1 in the uniform oracle


@pytest.mark.parametrize("name", ["affine", "planar", "radial", "spline", "sigmoid"])
def test_log_derivative_matches_autograd(name):
    transform = all_transforms()[name]
    x = torch.linspace(-8.0, 10.0, 1001, dtype=torch.float64)
    if name == "spline":
        x = x[x.abs() != 5.0]  # at the bound the clamp makes the autograd derivative 0; the analytic one is right
    x.requires_grad_(True)
    z, logdet = transform(x)
    (slope,) = torch.autograd.grad(z.sum(), x)
    torch.testing.assert_close(logdet, torch.log(slope), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("make", [Affine, Planar, Radial, RQSpline], ids=["affine", "planar", "radial", "spline"])
def test_constrained_transforms_are_strictly_increasing(make, seed):
    transform = randomised(make(), scale=3.0, seed=seed)
    x = torch.linspace(-15.0, 15.0, 20_001, dtype=torch.float64)
    z, logdet = transform(x)
    assert torch.all(torch.diff(z) > 0)
    assert torch.all(torch.isfinite(logdet))
    if isinstance(transform, Planar):
        assert transform.u_hat * transform.w > -1
    if isinstance(transform, Radial):
        assert transform.alpha_hat > 0
        assert transform.beta_hat > -transform.alpha_hat


def test_planar_constraint_binds():
    raw = {"u": -3.0, "w": 2.0, "b": 0.0}  # u w = -6: the raw map folds back on itself
    x = torch.linspace(-3.0, 3.0, 6001, dtype=torch.float64)
    maps = {}
    for constrained in (True, False):
        planar = Planar(constrained=constrained).double()
        with torch.no_grad():
            for name, value in raw.items():
                getattr(planar, name).fill_(value)
        maps[constrained] = planar(x)[0]
        if constrained:
            assert (planar.u_hat * planar.w).item() == pytest.approx(-1 + math.log1p(math.exp(-6.0)))
            assert planar.u_hat.item() != pytest.approx(raw["u"])
    assert torch.all(torch.diff(maps[True]) > 0)
    assert torch.any(torch.diff(maps[False]) < 0)


def test_radial_constraint_binds():
    x = torch.linspace(-3.0, 3.0, 6001, dtype=torch.float64)
    maps = {}
    for constrained in (True, False):
        radial = Radial(constrained=constrained).double()
        with torch.no_grad():
            radial.x0.fill_(0.0)
            radial.alpha.fill_(0.0)  # alpha_hat = log 2
            radial.beta.fill_(-3.0)  # raw beta_hat = -3 < -alpha_hat: the raw map folds back
        maps[constrained] = radial(x)[0]
        if constrained:
            assert radial.beta_hat.item() == pytest.approx(-math.log(2) + math.log1p(math.exp(-3.0)))
    assert torch.all(torch.diff(maps[True]) > 0)
    assert torch.any(torch.diff(maps[False]) < 0)


def test_spline_constraint_binds():
    spline = RQSpline(bins=6, bound=4.0).double()
    with torch.no_grad():
        spline.widths.copy_(torch.tensor([30.0, -30, -30, -30, -30, -30]))  # one bin takes almost everything
        spline.heights.copy_(torch.tensor([-30.0, -30, -30, -30, -30, 30]))
        spline.derivatives.fill_(-50.0)
    cx, cy, d = spline.knots()
    assert torch.all(torch.diff(cx) >= 0.999 * spline.min_width * 2 * spline.bound)
    assert torch.all(torch.diff(cy) >= 0.999 * spline.min_height * 2 * spline.bound)
    assert torch.all(d >= spline.min_derivative)
    assert (cx[0].item(), cx[-1].item(), cy[0].item(), cy[-1].item()) == (-4.0, 4.0, -4.0, 4.0)
    x = torch.linspace(-6.0, 6.0, 12_001, dtype=torch.float64)
    z, logdet = spline(x)
    assert torch.all(torch.diff(z) > 0)
    assert torch.all(torch.isfinite(logdet))


def test_spline_is_identity_at_initialisation_and_outside_the_bound():
    spline = RQSpline(bins=16, bound=3.0).double()
    assert spline.n_params == 3 * 16 - 1
    x = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    z, logdet = spline(x)
    torch.testing.assert_close(z, x)
    torch.testing.assert_close(logdet, torch.zeros_like(x), rtol=0, atol=1e-7)  # float32 initial values
    spline = randomised(spline, scale=2.0)
    outside = torch.tensor([-5.0, -3.5, 3.5, 5.0], dtype=torch.float64)
    z, logdet = spline(outside)
    torch.testing.assert_close(z, outside)
    torch.testing.assert_close(logdet, torch.zeros(4, dtype=torch.float64))


def test_mixture_density_cdf_and_quantile():
    target = camel()
    assert target.params == CAMEL
    x = torch.linspace(*target.interval(), 40_001, dtype=torch.float64)
    p = target.pdf(x)
    assert torch.trapezoid(p, x).item() == pytest.approx(1.0, abs=1e-10)
    f = target.cdf(x)
    assert torch.all(torch.diff(f) >= 0)
    torch.testing.assert_close(f + target.sf(x), torch.ones_like(x))
    running = torch.cumulative_trapezoid(p, x)  # the CDF is the integral of the density
    torch.testing.assert_close(running, f[1:] - f[0], rtol=0, atol=1e-7)
    xs = torch.linspace(-4.0, 9.0, 501, dtype=torch.float64)
    torch.testing.assert_close(target.quantile(target.cdf(xs)), xs, rtol=0, atol=1e-9)
    assert target.mean == pytest.approx(0.3 * -2.2 + 0.5 * 0.5 + 0.2 * 3.5)
    assert target.cdf(-10.5).item() > 0  # accurate left tail (erfc, not 1 + erf)


def test_mixture_samples_follow_the_cdf():
    target = camel()
    x = target.sample(100_000, seed=3)
    assert x.dtype == torch.float64
    assert ks_distance(x.numpy(), lambda v: target.cdf(v).numpy()) < KS_LIMIT
    torch.testing.assert_close(target.sample(10, seed=4), target.sample(10, seed=4))


def test_mixture_is_one_line_and_validated():
    twins = Mixture1D(weights=[1, 1], means=[-1.0, 1.0], stds=[0.3, 0.3])
    assert twins.weights.tolist() == [0.5, 0.5]
    assert twins.cdf(0.0).item() == pytest.approx(0.5)
    assert Mixture1D(**twins.params).params == twins.params
    with pytest.raises(ValueError, match="same length"):
        Mixture1D(weights=[1.0], means=[0.0, 1.0], stds=[1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        Mixture1D(weights=[1.0], means=[0.0], stds=[0.0])


def test_oracle_map_gaussianizes_and_uniformizes_exact_samples():
    target = camel()
    x = target.sample(100_000, seed=5)
    z, _ = oracle_map(target)(x)
    u, _ = oracle_map(target, Uniform())(x)
    assert ks_distance(z.numpy(), lambda v: StandardNormal().double().cdf(v).numpy()) < KS_LIMIT
    assert ks_distance(u.numpy(), lambda v: v) < KS_LIMIT
    result = ks_statistic(z)  # the package's own test agrees with the helper of the suite
    assert math.sqrt(len(z)) * result.statistic == pytest.approx(ks_distance(z.numpy(), lambda v: target_normal(v)))
    assert result.pvalue > 1e-3


def target_normal(v):
    return 0.5 * (1 + np.vectorize(math.erf)(v / math.sqrt(2)))


def test_oracle_flow_has_the_target_density():
    target = camel()
    x = torch.linspace(-6.0, 12.0, 1001, dtype=torch.float64)
    for base in (StandardNormal(), Uniform()):
        flow = Flow([oracle_map(target, base)], base).double()
        torch.testing.assert_close(flow.log_prob(x), target.log_pdf(x))
        assert abs(kl_to_truth(flow, target)) < 1e-10


def test_quadrature_kl_of_known_gaussians():
    identity = Flow([Affine()])  # q = N(0, 1)
    assert kl_to_truth(identity, Mixture1D([1.0], [0.0], [1.0])) == pytest.approx(0.0, abs=1e-10)
    assert kl_to_truth(identity, Mixture1D([1.0], [1.0], [1.0])) == pytest.approx(0.5, abs=1e-8)
    # KL(N(0, s^2) || N(0, 1)) = (s^2 - 1) / 2 - log s
    assert kl_to_truth(identity, Mixture1D([1.0], [0.0], [2.0])) == pytest.approx(1.5 - math.log(2), abs=1e-8)
    assert identity.dtype == torch.float32  # the quadrature used a float64 copy


def test_spline_fit_reaches_the_truth():
    torch.manual_seed(0)
    target = camel()
    x = target.sample(20_000, seed=1)
    flow = Flow([RQSpline(bins=16)])
    history = fit(flow, x, steps=2000, lr=1e-2, every=None)
    kl = kl_to_truth(flow, target)
    assert kl < 0.01
    assert history.train_loss[-1] == pytest.approx(target.entropy() + kl, abs=0.03)  # NLL -> H(p) + KL
    assert history.loss[0] > history.loss[-1]


def test_history_records_snapshots():
    target = camel()
    x_train, x_val = split(target.sample(1000, seed=6), val_fraction=0.25, seed=7)
    assert (len(x_train), len(x_val)) == (750, 250)
    assert len(set(x_train.tolist()) & set(x_val.tolist())) == 0
    torch.manual_seed(1)
    flow = Flow([Planar(), Planar()])
    initial = {k: v.clone() for k, v in flow.state_dict().items()}
    grid = torch.linspace(-5.0, 9.0, 50)
    history = fit(flow, x_train, steps=23, lr=1e-2, batch=128, every=5, grid=grid, x_val=x_val)
    np.testing.assert_array_equal(history.step, [0, 5, 10, 15, 20, 23])
    assert history.loss.shape == (23,)
    assert history.train_loss.shape == history.val_loss.shape == (6,)
    assert history.q_grid.shape == history.z_grid.shape == (6, 50)
    assert history.params["transforms.0.u"].shape == (6,)
    torch.testing.assert_close(torch.as_tensor(history.z_grid[-1]), flow(grid)[0].detach())
    history.restore(flow, 0)
    for k, v in flow.state_dict().items():
        torch.testing.assert_close(v, initial[k])


@pytest.mark.parametrize(
    "layers",
    [lambda: [RQSpline(bins=8, bound=6.0)], lambda: [Planar(), Radial(), Planar(), Radial()]],
    ids=["spline", "planar-radial"],
)
def test_flow_samples_follow_its_density(layers):
    torch.manual_seed(2)
    flow = Flow([randomised(t, scale=1.0, seed=i) for i, t in enumerate(layers())]).double()
    samples = flow.sample(100_000, seed=8).numpy()
    edges = np.linspace(np.quantile(samples, 0.001), np.quantile(samples, 0.999), 41)
    counts, _ = np.histogram(samples, edges)
    fine = torch.linspace(edges[0], edges[-1], 40 * 200 + 1, dtype=torch.float64)
    density = torch.exp(flow.log_prob(fine)).detach()
    mass = [torch.trapezoid(density[i * 200 : (i + 1) * 200 + 1], fine[i * 200 : (i + 1) * 200 + 1]) for i in range(40)]
    expected = len(samples) * torch.stack(mass).numpy()
    pulls = (counts - expected) / np.sqrt(expected)
    assert np.max(np.abs(pulls)) < 5
    chi2 = np.sum(pulls**2)
    assert chi2 < 40 + 5 * math.sqrt(2 * 40)


def test_uniform_base_flow_maps_into_the_unit_interval():
    torch.manual_seed(3)
    flow = Flow([randomised(RQSpline(bins=8), scale=1.0), Sigmoid()], Uniform()).double()
    x = torch.linspace(-50.0, 50.0, 1001, dtype=torch.float64)
    u, _ = flow(x)
    assert torch.all((u >= 0) & (u <= 1))
    assert torch.all(torch.isfinite(flow.log_prob(torch.linspace(-20.0, 20.0, 101))))
    samples = flow.sample(10_000, seed=9)
    assert torch.all(torch.isfinite(samples))
    target = camel()
    history = fit(Flow([RQSpline(bins=8), Sigmoid()], Uniform()), target.sample(2000, seed=10), steps=20)
    assert np.all(np.isfinite(history.loss))
    with pytest.raises(FloatingPointError, match="Sigmoid"):
        fit(Flow([RQSpline(bins=8)], Uniform()), target.sample(2000, seed=10), steps=5)


def test_transform_points_follow_the_layers():
    torch.manual_seed(4)
    flow = Flow([Planar(), RQSpline(bins=4), Radial()])
    x = camel().sample(100, seed=11)
    path = flow.transform_points(x)
    assert path.shape == (4, 100)
    torch.testing.assert_close(path[0], x.float())
    torch.testing.assert_close(path[-1], flow(x)[0].detach())
    assert flow.n_params == 3 + (3 * 4 - 1) + 3


def test_qq_points_and_kolmogorov_distribution():
    z = StandardNormal().double().sample(50_000, seed=12)
    theory, data = qq_points(z, n_points=99)
    assert theory.shape == data.shape == (99,)
    np.testing.assert_allclose(data, theory, atol=0.05)
    theory_u, _ = qq_points(z, Uniform(), n_points=4)
    np.testing.assert_allclose(theory_u, [0.125, 0.375, 0.625, 0.875])
    # tabulated values of P(sqrt(n) D > lambda) and continuity where the two series meet
    assert _kolmogorov_sf(1.3581) == pytest.approx(0.05, abs=2e-4)
    assert _kolmogorov_sf(1.6276) == pytest.approx(0.01, abs=1e-4)
    assert _kolmogorov_sf(1.18 - 1e-13) == pytest.approx(_kolmogorov_sf(1.18 + 1e-13), abs=1e-11)
    assert _kolmogorov_sf(0.2) == pytest.approx(1.0)


def test_fit_detaches_samples_that_carry_a_graph():
    spline = randomised(RQSpline(bins=4), scale=0.5)
    x = spline.inverse(camel().sample(500, seed=13))  # differentiable inverse: x carries a graph
    assert x.requires_grad
    history = fit(Flow([RQSpline(bins=4)]), x, steps=3)
    assert history.loss.shape == (3,)


def test_standard_normal_cdf_keeps_the_left_tail():
    phi = StandardNormal().double().cdf(torch.tensor([-8.0, -9.0, -10.0, -30.0], dtype=torch.float64))
    exact = [0.5 * math.erfc(t / math.sqrt(2)) for t in (8.0, 9.0, 10.0, 30.0)]  # 1 + erf would give 0 from -8.3
    torch.testing.assert_close(phi, torch.tensor(exact, dtype=torch.float64), rtol=1e-12, atol=0)


def test_diagnostics_accept_samples_with_a_graph():
    flow = Flow([randomised(RQSpline(bins=4), scale=0.5).float()])
    x = camel().sample(2000, seed=14)
    z = flow(x)[0]
    assert z.requires_grad
    assert ks_statistic(z) == ks_statistic(z.detach())
    np.testing.assert_array_equal(qq_points(z)[1], qq_points(z.detach())[1])
    np.testing.assert_array_equal(qq_points(z, Uniform())[0], qq_points(z.detach(), Uniform())[0])


def test_ks_distance_is_invariant_under_the_oracle_map():
    target = camel()
    x = target.sample(20_000, seed=15)
    z, _ = oracle_map(target)(x)
    u, _ = oracle_map(target, Uniform())(x)
    d_x = ks_statistic(x, target).statistic
    assert ks_statistic(z).statistic == pytest.approx(d_x, rel=1e-12)
    assert ks_statistic(u, Uniform()).statistic == pytest.approx(d_x, rel=1e-12)  # Uniform() is float32: cdf in float64


def test_transform_without_inverse_says_so():
    class Shift(Transform):
        def __init__(self):
            super().__init__()
            self.t = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return x + self.t, torch.zeros_like(x)

    with pytest.raises(NotImplementedError, match="define inverse"):
        Flow([Shift()]).sample(5, seed=0)
