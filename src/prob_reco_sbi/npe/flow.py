"""Posterior models: conditional densities of the event parameters theta given the observations x.

Every model is a :class:`PosteriorEstimator`: it takes and returns physical units (theta in m and MeV, x as given by
:func:`~prob_reco_sbi.npe.load_pairs`), and inside it works on transformed, standardized parameters ``theta~`` and
observations ``x~``::

    u        = to_unbounded(theta)              the fixed map of the coordinates (off with unbounded=False)
    theta~   = (u - mean_u) / std_u             standardization of the parameters, stored in buffers
    x~       = (x - mean_x) / std_x             standardization of the observations, stored in buffers
    z        = f_phi(theta~ | x~)               the learned transformation (normalizing direction)

    log q^(theta | x) = log p_Z(z) + log|det dz/dtheta~| + log|det dtheta~/dtheta|,    p_Z = N(0, I).

The standardization is set from the training pairs by :meth:`PosteriorEstimator.set_standardization` (which
:func:`~prob_reco_sbi.npe.fit` calls). The three models of the course:

* :class:`ConditionalFlow1D`: one parameter; a network maps x~ to the ``3 K - 1`` parameters of one spline
  (the reference of N1.5).
* :class:`AutoregressiveFlow`: the chain of the product rule, ``q^(theta | x) = prod_i q^(theta_i | psi_i)`` with
  ``psi_i = (h(x), theta~_1, ..., theta~_{i-1})``; ``h`` is the input embedding block, every factor a conditional
  spline whose parameters come from a parameters head.
* :class:`GaussianChain`: the comparison model, a multivariate normal in theta~ whose mean and covariance come from
  the network (what a regression network with a predicted uncertainty gives).

The network weights start from torch's global random state: call ``torch.manual_seed(seed)`` before building a model
to make it reproducible. The last layer of every head starts at zero, so a new flow is the identity in theta~ and its
density is the standardized prior shape ``N(0, I)`` in theta~.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ..flows1d._utils import Seed, make_generator
from .coordinates import ENERGY_RANGE, PARAMETER_NAMES, R_MAX, from_unbounded, inside_prior, to_unbounded
from .spline import n_spline_params, rq_spline, rq_spline_inverse

__all__ = ["AutoregressiveFlow", "ConditionalFlow1D", "GaussianChain", "PosteriorEstimator", "mlp"]

ArrayLike = Tensor | np.ndarray | Sequence[float] | float

#: Rows (samples times observations) mapped at once by :meth:`PosteriorEstimator.sample` (memory control).
SAMPLE_CHUNK = 1 << 18


def mlp(n_in: int, hidden: Sequence[int], n_out: int, zero_last: bool = False) -> nn.Sequential:
    """Build a multilayer perceptron with ReLU activations, ``n_in -> hidden[0] -> ... -> n_out``, linear output.

    With ``zero_last=True`` the weights and biases of the last layer start at zero, so the network starts by
    returning zeros for every input.
    """
    layers: list[nn.Module] = []
    width = n_in
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    last = nn.Linear(width, n_out)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class PosteriorEstimator(nn.Module):
    """Base class of the posterior models: ``q^(theta | x)`` in physical units.

    It holds what every model shares: the fixed map to unbounded coordinates, the standardization buffers, the
    broadcasting of parameters against observations, the density and the sampling. A model implements the learned
    part in the standardized coordinates, with ``context = self.context(x~)`` (by default ``x~`` itself)::

        _forward_tilde(theta_t, context) -> (z, log|det dz/dtheta~|)     rows [N, theta_dim] and [N, n_context]
        _inverse_tilde(z, context) -> theta_t

    Parameters
    ----------
    obs_dim
        Size of one observation ``x`` (e.g. 64 for charges and times of 32 sensors).
    theta_dim
        Number of parameters: 3 for ``(x0, y0, E)``, 2 for ``(x0, y0)``, 1 for ``(E,)``; with ``unbounded=False``
        any number.
    unbounded
        Map theta to unbounded coordinates before the flow (artanh of the radius, logit of the energy). With
        ``False`` the flow acts on standardized theta directly, and part of ``q^`` can fall outside the prior.
    r_max, energy_range
        The prior range of the vertex radius [m] and of the energy [MeV] (the stage-1 prior by default).
    """

    x_mean: Tensor
    x_std: Tensor
    theta_mean: Tensor
    theta_std: Tensor
    standardized: Tensor

    def __init__(
        self,
        obs_dim: int,
        theta_dim: int,
        *,
        unbounded: bool = True,
        r_max: float = R_MAX,
        energy_range: tuple[float, float] = ENERGY_RANGE,
    ) -> None:
        super().__init__()
        if obs_dim < 1 or theta_dim < 1:
            raise ValueError(f"obs_dim and theta_dim must be positive, got {obs_dim} and {theta_dim}")
        if unbounded and theta_dim not in PARAMETER_NAMES:
            raise ValueError(
                f"unbounded coordinates need theta = (x0, y0, E), (x0, y0) or (E,), i.e. theta_dim 1 to 3, got "
                f"{theta_dim}; use unbounded=False for other parameters"
            )
        self.obs_dim, self.theta_dim, self.unbounded = int(obs_dim), int(theta_dim), bool(unbounded)
        self.r_max, self.energy_range = float(r_max), (float(energy_range[0]), float(energy_range[1]))
        self.register_buffer("x_mean", torch.zeros(obs_dim))
        self.register_buffer("x_std", torch.ones(obs_dim))
        self.register_buffer("theta_mean", torch.zeros(theta_dim))
        self.register_buffer("theta_std", torch.ones(theta_dim))
        self.register_buffer("standardized", torch.tensor(False))
        #: The constructor arguments (used by :func:`~prob_reco_sbi.npe.save_model` to rebuild the model).
        self.hparams: dict[str, Any] = {
            "obs_dim": self.obs_dim,
            "theta_dim": self.theta_dim,
            "unbounded": self.unbounded,
            "r_max": self.r_max,
            "energy_range": self.energy_range,
        }

    # ------------------------------------------------------------------------------------------------ properties
    @property
    def dtype(self) -> torch.dtype:
        """Floating-point type of the model (float32 unless converted with ``.double()``)."""
        return self.x_mean.dtype

    @property
    def n_params(self) -> int:
        """Number of learnable parameters phi."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the parameters theta, e.g. ``("x0", "y0", "E")``."""
        if self.theta_dim in PARAMETER_NAMES:
            return PARAMETER_NAMES[self.theta_dim]
        return tuple(f"theta{i}" for i in range(self.theta_dim))

    # ------------------------------------------------------------------------------------------- standardization
    def _as(self, v: ArrayLike) -> Tensor:
        return torch.as_tensor(v, dtype=self.dtype, device=self.x_mean.device)

    def _unbound(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        if self.unbounded:
            return to_unbounded(theta, self.r_max, self.energy_range)
        return theta, theta.new_zeros(theta.shape[:-1])

    def set_standardization(self, theta: ArrayLike, x: ArrayLike) -> None:
        """Set the means and standard deviations of ``u`` (the unbounded theta) and of ``x`` from training pairs.

        Parameters
        ----------
        theta
            Training parameters, shape ``[N, theta_dim]`` (physical units).
        x
            Training observations, shape ``[N, obs_dim]``.
        """
        with torch.no_grad():
            theta_t, x_t = self._check(self._as(theta), self._as(x))
            u = self._unbound(theta_t.reshape(-1, self.theta_dim))[0]
            x_t = x_t.reshape(-1, self.obs_dim)
            self.theta_mean.copy_(u.mean(0))
            self.theta_std.copy_(u.std(0).clamp_min(1e-6))
            self.x_mean.copy_(x_t.mean(0))
            self.x_std.copy_(x_t.std(0).clamp_min(1e-6))
            self.standardized.fill_(True)

    def theta_to_tilde(self, theta: ArrayLike) -> tuple[Tensor, Tensor]:
        """Map theta to the coordinates of the flow: ``theta~`` and ``log|det dtheta~/dtheta|`` (shape ``[...]``)."""
        u, log_det = self._unbound(self._as(theta))
        return (u - self.theta_mean) / self.theta_std, log_det - torch.log(self.theta_std).sum()

    def theta_from_tilde(self, theta_t: ArrayLike) -> Tensor:
        """Map ``theta~`` back to physical units (inside the prior range when ``unbounded``)."""
        u = self._as(theta_t) * self.theta_std + self.theta_mean
        return from_unbounded(u, self.r_max, self.energy_range) if self.unbounded else u

    def x_to_tilde(self, x: ArrayLike) -> Tensor:
        """Standardize observations: ``x~ = (x - mean_x) / std_x``."""
        return (self._as(x) - self.x_mean) / self.x_std

    def context(self, x_t: Tensor) -> Tensor:
        """Conditioning variables computed once per observation from ``x~`` (default: ``x~`` itself)."""
        return x_t

    # ------------------------------------------------------------------------------------------ the learned part
    def _forward_tilde(self, theta_t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        """``z = f_phi(theta~ | context)`` and ``log|det dz/dtheta~|`` for rows ``[N, theta_dim]``, ``[N, c]``."""
        raise NotImplementedError

    def _inverse_tilde(self, z: Tensor, context: Tensor) -> Tensor:
        """``theta~ = f_phi^-1(z | context)`` for rows ``[N, theta_dim]``, ``[N, c]``."""
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------- interface
    def _check(self, theta: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        if theta.dim() == 0 or theta.shape[-1] != self.theta_dim:
            raise ValueError(f"theta must have shape [..., {self.theta_dim}], got {tuple(theta.shape)}")
        if x.dim() == 0 or x.shape[-1] != self.obs_dim:
            raise ValueError(f"x must have shape [..., {self.obs_dim}], got {tuple(x.shape)}")
        return theta, x

    def forward(self, theta: ArrayLike, x: ArrayLike) -> tuple[Tensor, Tensor]:
        """Map theta to the base in the normalizing direction and return ``(z, log|det dz/dtheta|)``.

        Parameters
        ----------
        theta
            Parameters, shape ``[..., theta_dim]`` (physical units).
        x
            Observations, shape ``[..., obs_dim]``; broadcast against theta (e.g. one observation ``[obs_dim]``
            with many values of theta).

        Returns
        -------
        tuple of Tensor
            ``z`` of shape ``[..., theta_dim]`` and the total log-determinant of shape ``[...]``, the fixed map and
            the standardization included.
        """
        theta_t, x_t = self._check(self._as(theta), self._as(x))
        tilde, log_det0 = self.theta_to_tilde(theta_t)
        ctx = self.context(self.x_to_tilde(x_t))
        batch = torch.broadcast_shapes(tilde.shape[:-1], ctx.shape[:-1])
        rows = tilde.expand(*batch, self.theta_dim).reshape(-1, self.theta_dim)
        ctx_rows = ctx.expand(*batch, ctx.shape[-1]).reshape(-1, ctx.shape[-1])
        z, log_det = self._forward_tilde(rows, ctx_rows)
        return z.reshape(*batch, self.theta_dim), log_det.reshape(batch) + log_det0.expand(batch)

    def log_prob(self, theta: ArrayLike, x: ArrayLike) -> Tensor:
        """Log-density ``log q^(theta | x)`` in physical units (per m^2 MeV for ``(x0, y0, E)``).

        ``theta`` and ``x`` broadcast as in :meth:`forward`. With ``unbounded=True`` parameters outside the prior range
        get ``-inf``.
        """
        z, log_det = self(theta, x)
        lp = -0.5 * (z * z).sum(-1) - 0.5 * self.theta_dim * math.log(2 * math.pi) + log_det
        if self.unbounded:
            inside = inside_prior(self._as(theta), self.r_max, self.energy_range).expand(lp.shape)
            lp = torch.where(inside, lp, torch.full_like(lp, -math.inf))
        return lp

    def inverse(self, z: ArrayLike, x: ArrayLike) -> Tensor:
        """Map base points ``z`` ``[..., theta_dim]`` to theta in the generative direction, given observations ``x``.

        The factors are inverted one after the other, each exactly. No gradient.
        """
        with torch.no_grad():
            z_t, x_t = self._check(self._as(z), self._as(x))
            ctx = self.context(self.x_to_tilde(x_t))
            batch = torch.broadcast_shapes(z_t.shape[:-1], ctx.shape[:-1])
            rows = z_t.expand(*batch, self.theta_dim).reshape(-1, self.theta_dim)
            ctx_rows = ctx.expand(*batch, ctx.shape[-1]).reshape(-1, ctx.shape[-1])
            tilde = self._inverse_tilde(rows, ctx_rows)
            return self.theta_from_tilde(tilde).reshape(*batch, self.theta_dim)

    def sample(self, x: ArrayLike, n: int, seed: Seed = None) -> Tensor:
        """Draw ``n`` posterior samples for every observation: ``z ~ N(0, I)``, then the generative direction.

        Parameters
        ----------
        x
            One observation ``[obs_dim]`` or ``M`` observations ``[M, obs_dim]`` (any leading shape).
        n
            Number of samples per observation.
        seed
            An integer, a :class:`torch.Generator` or ``None`` (the global random state).

        Returns
        -------
        Tensor
            Samples in physical units, shape ``[n, theta_dim]`` for one observation, ``[n, M, theta_dim]`` for ``M``
            (the convention of ``zuko``: samples first). No gradient.
        """
        generator = make_generator(seed)
        with torch.no_grad():
            x_t = self._check(self._as(x).new_zeros(self.theta_dim), self._as(x))[1]
            batch = x_t.shape[:-1]
            x_t = x_t.reshape(-1, self.obs_dim)
            ctx = self.context(self.x_to_tilde(x_t))
            m = len(x_t)
            out = torch.empty(n, m, self.theta_dim, dtype=self.dtype)
            per_chunk = max(1, SAMPLE_CHUNK // max(m, 1))
            for start in range(0, n, per_chunk):
                k = min(per_chunk, n - start)
                z = torch.randn(k * m, self.theta_dim, generator=generator, dtype=self.dtype).to(ctx.device)
                rows = ctx.unsqueeze(0).expand(k, m, ctx.shape[-1]).reshape(-1, ctx.shape[-1])
                theta = self.theta_from_tilde(self._inverse_tilde(z, rows))
                out[start : start + k] = theta.reshape(k, m, self.theta_dim).cpu()
            return out.reshape(n, *batch, self.theta_dim)


class ConditionalFlow1D(PosteriorEstimator):
    """A conditional flow for one parameter: ``q^(theta | x)`` with one spline whose parameters come from a network.

    ::

        params = network(x~)                                  3 K - 1 numbers per observation
        z, log f' = rq_spline(theta~, params, B)
        log q^(theta | x) = log p_Z(z) + log f' + log|dtheta~/dtheta|

    With the defaults it models the energy, ``q^(E | x)``, in the logit coordinate of
    :func:`~prob_reco_sbi.npe.energy_to_unbounded` (the reference of N1.5 and N1.6). Theta has the shape ``[..., 1]``.

    Parameters
    ----------
    obs_dim
        Size of one observation (1 for ``ln Q``).
    bins, bound
        Number of bins ``K`` and half-width ``B`` of the spline (on theta~).
    hidden
        Hidden layers of the network.
    unbounded
        Use the logit coordinate of the energy (``False``: the flow acts on the standardized energy itself).
    energy_range
        The prior range of the energy [MeV].
    """

    def __init__(
        self,
        obs_dim: int = 1,
        *,
        bins: int = 16,
        bound: float = 5.0,
        hidden: Sequence[int] = (64, 64),
        unbounded: bool = True,
        energy_range: tuple[float, float] = ENERGY_RANGE,
    ) -> None:
        super().__init__(obs_dim, 1, unbounded=unbounded, energy_range=energy_range)
        self.bins, self.bound = int(bins), float(bound)
        self.network = mlp(obs_dim, hidden, n_spline_params(bins), zero_last=True)
        self.hparams |= {"bins": self.bins, "bound": self.bound, "hidden": tuple(hidden)}
        del self.hparams["theta_dim"], self.hparams["r_max"]

    def spline_params(self, x: ArrayLike) -> Tensor:
        """Return the output of the network for observations ``x`` ``[..., obs_dim]``: the raw spline parameters."""
        return self.network(self.x_to_tilde(x))

    def _forward_tilde(self, theta_t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        z, log_det = rq_spline(theta_t[:, 0], self.network(context), self.bound)
        return z.unsqueeze(-1), log_det

    def _inverse_tilde(self, z: Tensor, context: Tensor) -> Tensor:
        return rq_spline_inverse(z[:, 0], self.network(context), self.bound).unsqueeze(-1)


class AutoregressiveFlow(PosteriorEstimator):
    """The chain of conditional one-dimensional spline flows: ``q^(theta | x) = prod_i q^(theta_i | psi_i)``.

    ::

        h      = embedding(x~)                                   input embedding block, obs_dim -> ... -> n_embed
        psi_i  = (h, theta~_1, ..., theta~_{i-1})                conditioning variables of factor i
        z_i    = rq_spline(theta~_i, heads[i](psi_i), B)         factor i: a spline from its parameters head
        log q^(theta | x) = sum_i [log p_Z(z_i) + log|dz_i/dtheta~_i|] + log|det dtheta~/dtheta|

    The normalizing direction (density) is one pass; the generative direction (samples) goes coordinate by
    coordinate, each factor inverted exactly with the coordinates before it already known. With the defaults
    (64 observations, three parameters, 8 bins) the model has 52 453 parameters.

    Parameters
    ----------
    obs_dim
        Size of one observation (64 for charges and times, 32 for charges, 34 for the ganged readout).
    theta_dim
        Number of parameters, in the order of the chain: 3 for ``(x0, y0, E)``, 2 for ``(x0, y0)``, 1 for ``(E,)``.
    bins, bound
        Number of bins ``K`` and half-width ``B`` of every spline (on theta~).
    hidden, n_embed
        Hidden layers and output size of the input embedding block ``h``.
    head
        Hidden layers of every parameters head.
    unbounded, r_max, energy_range
        As for :class:`PosteriorEstimator`.
    """

    def __init__(
        self,
        obs_dim: int,
        theta_dim: int = 3,
        *,
        bins: int = 8,
        bound: float = 5.0,
        hidden: Sequence[int] = (128, 128),
        n_embed: int = 32,
        head: Sequence[int] = (64, 64),
        unbounded: bool = True,
        r_max: float = R_MAX,
        energy_range: tuple[float, float] = ENERGY_RANGE,
    ) -> None:
        super().__init__(obs_dim, theta_dim, unbounded=unbounded, r_max=r_max, energy_range=energy_range)
        self.bins, self.bound = int(bins), float(bound)
        self.embedding = mlp(obs_dim, hidden, n_embed)
        self.heads = nn.ModuleList(
            [mlp(n_embed + i, head, n_spline_params(bins), zero_last=True) for i in range(theta_dim)]
        )
        self.hparams |= {
            "bins": self.bins,
            "bound": self.bound,
            "hidden": tuple(hidden),
            "n_embed": int(n_embed),
            "head": tuple(head),
        }

    def context(self, x_t: Tensor) -> Tensor:
        """Return the output of the input embedding block, ``h(x)``."""
        return self.embedding(x_t)

    def _forward_tilde(self, theta_t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        z, log_det = [], theta_t.new_zeros(len(theta_t))
        for i, head in enumerate(self.heads):
            params = head(torch.cat([context, theta_t[:, :i]], dim=-1))
            z_i, log_det_i = rq_spline(theta_t[:, i], params, self.bound)
            z.append(z_i)
            log_det = log_det + log_det_i
        return torch.stack(z, dim=-1), log_det

    def _inverse_tilde(self, z: Tensor, context: Tensor) -> Tensor:
        theta_t = z.new_zeros(z.shape[0], 0)
        for i, head in enumerate(self.heads):
            params = head(torch.cat([context, theta_t], dim=-1))
            theta_t = torch.cat([theta_t, rq_spline_inverse(z[:, i], params, self.bound).unsqueeze(-1)], dim=-1)
        return theta_t


class GaussianChain(PosteriorEstimator):
    """The Gaussian comparison model: ``q^(theta~ | x) = N(m(x), L(x) L(x)^T)`` with a full covariance.

    The network (the same input embedding block as :class:`AutoregressiveFlow` and one head) returns the mean ``m``,
    the log of the diagonal of the lower-triangular ``L`` and its entries below the diagonal. In the language of the
    chain, every factor ``q^(theta~_i | theta~_<i, x)`` is a Gaussian whose mean depends linearly on the parameters
    before it::

        theta~ = m + L z,    z ~ N(0, I),    log|det dz/dtheta~| = -sum_i log L_ii

    It is what a regression network with a predicted uncertainty gives: one mode, an ellipse. With
    ``theta_dim=1`` and ``unbounded=False`` it is the network of N1.3, ``q^(E | Q) = N(m(Q), s(Q)^2)``.

    Parameters
    ----------
    obs_dim, theta_dim, hidden, n_embed, head, unbounded, r_max, energy_range
        As for :class:`AutoregressiveFlow`.
    """

    def __init__(
        self,
        obs_dim: int,
        theta_dim: int = 3,
        *,
        hidden: Sequence[int] = (128, 128),
        n_embed: int = 32,
        head: Sequence[int] = (64, 64),
        unbounded: bool = True,
        r_max: float = R_MAX,
        energy_range: tuple[float, float] = ENERGY_RANGE,
    ) -> None:
        super().__init__(obs_dim, theta_dim, unbounded=unbounded, r_max=r_max, energy_range=energy_range)
        d = theta_dim
        self.embedding = mlp(obs_dim, hidden, n_embed)
        self.head = mlp(n_embed, head, 2 * d + d * (d - 1) // 2, zero_last=True)
        self.register_buffer("_rows", torch.tril_indices(d, d, -1)[0], persistent=False)
        self.register_buffer("_cols", torch.tril_indices(d, d, -1)[1], persistent=False)
        self.hparams |= {"hidden": tuple(hidden), "n_embed": int(n_embed), "head": tuple(head)}

    def context(self, x_t: Tensor) -> Tensor:
        """Return the output of the input embedding block, ``h(x)``."""
        return self.embedding(x_t)

    def _mean_and_scale(self, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Mean ``[N, d]``, log of the diagonal of ``L`` ``[N, d]`` and the matrix ``L`` ``[N, d, d]``."""
        d = self.theta_dim
        out = self.head(context)
        mean, log_scale, below = out[:, :d], out[:, d : 2 * d], out[:, 2 * d :]
        scale = torch.diag_embed(torch.exp(log_scale))
        if d > 1:
            rows, cols = self.get_buffer("_rows"), self.get_buffer("_cols")
            scale[:, rows, cols] = below
        return mean, log_scale, scale

    def mean_and_covariance(self, x: ArrayLike) -> tuple[Tensor, Tensor]:
        """Mean ``[..., d]`` and covariance ``[..., d, d]`` of the Gaussian in theta~ for observations ``x``."""
        x_t = self._as(x)
        batch, d = x_t.shape[:-1], self.theta_dim
        ctx = self.context(self.x_to_tilde(x_t))
        mean, _, scale = self._mean_and_scale(ctx.reshape(-1, ctx.shape[-1]))
        cov = scale @ scale.transpose(-1, -2)
        return mean.reshape(*batch, d), cov.reshape(*batch, d, d)

    def _forward_tilde(self, theta_t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_scale, scale = self._mean_and_scale(context)
        z = torch.linalg.solve_triangular(scale, (theta_t - mean).unsqueeze(-1), upper=False).squeeze(-1)
        return z, -log_scale.sum(-1)

    def _inverse_tilde(self, z: Tensor, context: Tensor) -> Tensor:
        mean, _, scale = self._mean_and_scale(context)
        return mean + (scale @ z.unsqueeze(-1)).squeeze(-1)
