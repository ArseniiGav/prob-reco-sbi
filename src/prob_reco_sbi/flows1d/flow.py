"""A normalizing flow in one dimension: a stack of invertible transforms on top of a base distribution.

The flow maps a data value ``x`` to ``z = f(x) = f_L(... f_2(f_1(x)))`` and models the density of ``x`` through the
change of variables::

    log q(x) = log p_base(f(x)) + log |f'(x)|,    log |f'(x)| = sum over the layers of log |dz/dx|.

Sampling runs the other way: draw ``z`` from the base and return ``x = f^-1(z)``. Two bases are provided:
:class:`StandardNormal` (the default) and :class:`Uniform` on (0, 1); the latter needs a last layer that maps the
real line onto (0, 1), such as :class:`~prob_reco_sbi.flows1d.Sigmoid`::

    from prob_reco_sbi.flows1d import Flow, RQSpline, Sigmoid, Uniform

    flow = Flow([RQSpline(bins=16)])                         # Gaussian base
    flow = Flow([RQSpline(bins=16), Sigmoid()], Uniform())   # uniform base
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from ._utils import ArrayLike, Seed, as_tensor, make_generator
from .transforms import Transform

__all__ = ["Base", "Flow", "StandardNormal", "Uniform"]


class Base(nn.Module):
    """Base distribution of a flow: log-density, sampling, CDF and quantile function, elementwise.

    A base has no parameters. It carries an empty-shaped buffer that follows ``.double()``, ``.float()`` and ``.to()``,
    so that :attr:`dtype` tells the floating-point type of the flow it belongs to.
    """

    _reference: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_reference", torch.zeros(()), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point type of the base (and of the flow that owns it)."""
        return self._reference.dtype

    def log_prob(self, z: Tensor) -> Tensor:
        """Log-density at ``z``."""
        raise NotImplementedError

    def cdf(self, z: ArrayLike) -> Tensor:
        """Cumulative distribution function at ``z``."""
        raise NotImplementedError

    def quantile(self, p: ArrayLike) -> Tensor:
        """Quantile function (inverse CDF) at the probabilities ``p``."""
        raise NotImplementedError

    def sample(self, n: int, seed: Seed = None) -> Tensor:
        """Draw ``n`` values (``seed``: an integer, a :class:`torch.Generator` or ``None``)."""
        raise NotImplementedError


class StandardNormal(Base):
    """Standard normal base N(0, 1): ``log p(z) = -z^2 / 2 - log(2 pi) / 2``."""

    def log_prob(self, z: Tensor) -> Tensor:
        """Log-density at ``z``."""
        return -0.5 * z * z - 0.5 * math.log(2 * math.pi)

    def cdf(self, z: ArrayLike) -> Tensor:
        """``Phi(z)``, computed as ``erfc(-z / sqrt(2)) / 2``, which keeps its precision far in the left tail."""
        return 0.5 * torch.erfc(-as_tensor(z, self.dtype) / math.sqrt(2))

    def quantile(self, p: ArrayLike) -> Tensor:
        """``Phi^-1(p)``."""
        return torch.special.ndtri(as_tensor(p, self.dtype))

    def sample(self, n: int, seed: Seed = None) -> Tensor:
        """Draw ``n`` values (``seed``: an integer, a :class:`torch.Generator` or ``None``)."""
        return torch.randn(n, generator=make_generator(seed), dtype=self.dtype)


class Uniform(Base):
    """Uniform base on (0, 1): ``log p(u) = 0`` for ``0 <= u <= 1`` and ``-inf`` outside.

    A flow with this base must map the real line into (0, 1), e.g. with a final :class:`~prob_reco_sbi.flows1d.Sigmoid`
    layer; values that land outside get ``log q = -inf``.
    """

    def log_prob(self, z: Tensor) -> Tensor:
        """Log-density at ``z``."""
        inside = (z >= 0) & (z <= 1)
        return torch.where(inside, torch.zeros_like(z), torch.full_like(z, -math.inf))

    def cdf(self, z: ArrayLike) -> Tensor:
        """``clip(z, 0, 1)``."""
        return as_tensor(z, self.dtype).clamp(0, 1)

    def quantile(self, p: ArrayLike) -> Tensor:
        """``p`` itself."""
        return as_tensor(p, self.dtype).clone()

    def sample(self, n: int, seed: Seed = None) -> Tensor:
        """Draw ``n`` values (``seed``: an integer, a :class:`torch.Generator` or ``None``)."""
        return torch.rand(n, generator=make_generator(seed), dtype=self.dtype)


class Flow(nn.Module):
    """A normalizing flow: the transforms ``f_1 ... f_L`` (applied in this order to data) and a base distribution.

    Parameters
    ----------
    transforms
        The layers, from the data side to the base side.
    base
        Base distribution, :class:`StandardNormal` if omitted.

    Notes
    -----
    Inputs are converted to the flow's :attr:`dtype` (float32 unless the flow was converted with ``.double()``).
    Calling the flow, ``flow(x)``, returns ``(z, log|f'(x)|)``.
    """

    def __init__(self, transforms: Sequence[Transform], base: Base | None = None) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.base = base if base is not None else StandardNormal()

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point type of the flow."""
        return self.base.dtype

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: ArrayLike) -> tuple[Tensor, Tensor]:
        """Map data values to the base, ``z = f(x)``, and return ``(z, log|f'(x)|)``."""
        z = as_tensor(x, self.dtype)
        logdet = torch.zeros_like(z)
        for transform in self.transforms:
            z, ld = transform(z)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x: ArrayLike) -> Tensor:
        """Log-density of the model, ``log q(x) = log p_base(f(x)) + log|f'(x)|``."""
        z, logdet = self(x)
        return self.base.log_prob(z) + logdet

    def inverse(self, z: ArrayLike) -> Tensor:
        """Map base values back to data, ``x = f^-1(z)`` (no gradient)."""
        x = as_tensor(z, self.dtype)
        with torch.no_grad():
            for transform in reversed(self.transforms):
                x = cast(Transform, transform).inverse(x)
        return x

    def sample(self, n: int, seed: Seed = None) -> Tensor:
        """Draw ``n`` samples of the model: ``z`` from the base, then ``x = f^-1(z)`` (no gradient).

        ``seed``: an integer, a :class:`torch.Generator` or ``None`` (the global random state).
        """
        return self.inverse(self.base.sample(n, seed))

    def transform_points(self, x: ArrayLike) -> Tensor:
        """Positions of the points after every layer, for the transport picture.

        Returns
        -------
        Tensor
            Shape ``[L + 1, n]``: row 0 is ``x`` itself, row ``l`` the positions after the first ``l`` layers, and the
            last row ``z = f(x)`` (no gradient).
        """
        z = as_tensor(x, self.dtype).reshape(-1)
        rows = [z]
        with torch.no_grad():
            for transform in self.transforms:
                z = transform(z)[0]
                rows.append(z)
        return torch.stack(rows)
