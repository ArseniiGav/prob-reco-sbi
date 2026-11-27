"""Neural posterior estimation (NPE) for the toy detector: conditional flows of the event parameters, from scratch.

A posterior model ``q^(theta | x)`` is a flow of the event parameters ``theta = (x0, y0, E)`` conditioned on the
observations ``x`` of an event. It is trained on simulated pairs ``{theta_i, x_i}`` by maximum likelihood, the forward
KL divergence of 2a averaged over the observations, and afterwards gives the posterior of any observation without new
simulations (amortized inference). Where the toy has a likelihood formula, the exact posterior is the reference.

Quick start::

    from prob_reco_sbi.npe import AutoregressiveFlow, fit, load_pairs

    theta, x = load_pairs("data/n3_full_train.h5", "charge+time")     # [310000, 3], [310000, 64]
    flow = AutoregressiveFlow(obs_dim=64)                                # three spline factors, product rule
    validation = (theta[-10_000:], x[-10_000:])                          # the last 10 000 events of the file
    history = fit(flow, theta[:100_000], x[:100_000], epochs=20, validation=validation)
    samples = flow.sample(x[-1], 500)                                    # [500, 3]: x0, y0 [m], E [MeV]

Modules
-------
spline
    :func:`rq_spline` and :func:`rq_spline_inverse`: the rational-quadratic spline of ``flows1d.RQSpline`` with one
    set of knots per value.
coordinates
    The fixed transforms to unbounded coordinates (artanh of the radius, logit of the energy) with their
    log-Jacobians.
flow
    The models: :class:`ConditionalFlow1D`, :class:`AutoregressiveFlow` (the chain), :class:`GaussianChain` (the
    Gaussian comparison model); their base :class:`PosteriorEstimator`.
data
    :func:`load_pairs`: the pairs ``(theta, x)`` of a data file in the representations of the course.
train
    :func:`fit`: minibatch maximum likelihood with a validation set, the best state and snapshots;
    :func:`save_model` and :func:`load_model`.
reference
    The exact references: :func:`detection_probability`, :func:`energy_posterior`, :func:`energy_evidence`,
    :func:`grid_posterior`, :func:`credible_levels`.
"""

from .coordinates import (
    ENERGY_RANGE,
    PARAMETER_NAMES,
    R_MAX,
    energy_from_unbounded,
    energy_to_unbounded,
    from_unbounded,
    inside_prior,
    to_unbounded,
    vertex_from_unbounded,
    vertex_to_unbounded,
)
from .data import REPRESENTATIONS, load_pairs, observations, parameter_names, to_pairs
from .flow import AutoregressiveFlow, ConditionalFlow1D, GaussianChain, PosteriorEstimator, mlp
from .reference import (
    EnergyPosterior,
    GridPosterior,
    credible_levels,
    detection_probability,
    energy_evidence,
    energy_posterior,
    grid_posterior,
)
from .spline import n_spline_params, rq_knots, rq_spline, rq_spline_inverse
from .train import History, fit, load_model, mean_loss, save_model

__all__ = [
    "ENERGY_RANGE",
    "PARAMETER_NAMES",
    "REPRESENTATIONS",
    "R_MAX",
    "AutoregressiveFlow",
    "ConditionalFlow1D",
    "EnergyPosterior",
    "GaussianChain",
    "GridPosterior",
    "History",
    "PosteriorEstimator",
    "credible_levels",
    "detection_probability",
    "energy_evidence",
    "energy_from_unbounded",
    "energy_posterior",
    "energy_to_unbounded",
    "fit",
    "from_unbounded",
    "grid_posterior",
    "inside_prior",
    "load_model",
    "load_pairs",
    "mean_loss",
    "mlp",
    "n_spline_params",
    "observations",
    "parameter_names",
    "rq_knots",
    "rq_spline",
    "rq_spline_inverse",
    "save_model",
    "to_pairs",
    "to_unbounded",
    "vertex_from_unbounded",
    "vertex_to_unbounded",
]
