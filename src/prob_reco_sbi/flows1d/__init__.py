"""Normalizing flows in one dimension, from scratch in PyTorch: transforms, flows, training and the exact truth.

A flow is a strictly increasing map ``z = f(x)`` of the real line, composed of simple layers, plus a base
distribution; the change of variables gives its density, ``log q(x) = log p_base(f(x)) + log f'(x)``. In one
dimension everything can be drawn and compared with the exact answer: the oracle map ``z = Phi^-1(F(x))`` of the
target and the KL divergence to the target by quadrature. The maths is in ``docs/flows.md``.

Quick start::

    from prob_reco_sbi.flows1d import Flow, RQSpline, camel, fit, kl_to_truth

    target = camel()                              # 0.30 N(-2.2, 0.25^2) + 0.50 N(0.5, 0.9^2) + 0.20 N(3.5, 1.4^2)
    x = target.sample(20_000, seed=1)
    flow = Flow([RQSpline(bins=16)])              # Gaussian base
    history = fit(flow, x, steps=2000, lr=1e-2)   # maximum likelihood (forward KL), Adam
    print(kl_to_truth(flow, target))              # exact KL(p || q) in nats
    samples = flow.sample(10_000, seed=2)
"""

from .diagnostics import KSResult, kl_to_truth, ks_statistic, qq_points
from .flow import Base, Flow, StandardNormal, Uniform
from .targets import CAMEL, Mixture1D, OracleMap, camel, oracle_map
from .train import History, fit, split
from .transforms import Affine, Planar, Radial, RQSpline, Sigmoid, Transform, invert_monotone

__all__ = [
    "CAMEL",
    "Affine",
    "Base",
    "Flow",
    "History",
    "KSResult",
    "Mixture1D",
    "OracleMap",
    "Planar",
    "RQSpline",
    "Radial",
    "Sigmoid",
    "StandardNormal",
    "Transform",
    "Uniform",
    "camel",
    "fit",
    "invert_monotone",
    "kl_to_truth",
    "ks_statistic",
    "oracle_map",
    "qq_points",
    "split",
]
