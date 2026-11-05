"""Maximum-likelihood training of a flow on samples.

Training minimises the mean negative log-likelihood of the samples::

    loss = -(1/N) sum_i log q(x_i)  ->  E_p[-log q(x)] = H(p) + KL(p || q)    (many samples),

so it minimises the forward KL divergence ``KL(p || q)`` up to the entropy ``H(p)``, which does not depend on the flow.
It needs samples of ``p`` only, never its density: exactly the situation of neural posterior estimation, where the
simulator provides samples. :func:`fit` runs Adam and records the losses, the parameters and the model on a grid at
regular steps (the frames of the training animations)::

    from prob_reco_sbi.flows1d import Flow, RQSpline, camel, fit, split

    x_train, x_val = split(camel().sample(20_000, seed=1), val_fraction=0.2)
    history = fit(Flow([RQSpline(bins=16)]), x_train, steps=2000, lr=1e-2, x_val=x_val, grid=[-4.0, 0.0, 4.0])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ._utils import ArrayLike, Seed, as_tensor, make_generator
from .flow import Flow

__all__ = ["History", "fit", "split"]


@dataclass
class History:
    """Record of a training run of :func:`fit`; the snapshots are taken every ``every`` updates and after the last.

    Snapshot ``i`` describes the flow after ``step[i]`` updates (``step[0] = 0``: the initial flow).
    """

    step: np.ndarray  #: number of updates at each snapshot, shape [n_snapshots]
    loss: np.ndarray  #: loss of every update on its batch [nats], shape [steps]
    train_loss: np.ndarray  #: loss on the whole training set at the snapshots [nats], shape [n_snapshots]
    val_loss: np.ndarray | None  #: loss on the validation samples at the snapshots, if any were given
    params: dict[str, np.ndarray]  #: every parameter at the snapshots, shape [n_snapshots, *parameter shape]
    states: list[dict[str, Tensor]] = field(repr=False)  #: copies of the flow's state dict at the snapshots
    grid: np.ndarray | None = None  #: the grid of x values, if one was given
    q_grid: np.ndarray | None = None  #: model density q(grid) at the snapshots, shape [n_snapshots, n_grid]
    z_grid: np.ndarray | None = None  #: map f(grid) at the snapshots, shape [n_snapshots, n_grid]
    seconds: float = 0.0  #: wall-clock time of the run (including the snapshots)

    def restore(self, flow: Flow, index: int = -1) -> Flow:
        """Load snapshot ``index`` (default: the last) into ``flow`` and return it."""
        flow.load_state_dict(self.states[index])
        return flow


def split(x: ArrayLike, val_fraction: float = 0.2, seed: Seed = 0) -> tuple[Tensor, Tensor]:
    """Split samples at random into a training and a validation part.

    Parameters
    ----------
    x
        The samples (1-D).
    val_fraction
        Fraction of the samples in the validation part.
    seed
        Seed of the random permutation (an integer, a :class:`torch.Generator` or ``None``).

    Returns
    -------
    tuple of Tensor
        ``(x_train, x_val)``, float64.
    """
    x_t = as_tensor(x).reshape(-1)
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    order = torch.randperm(len(x_t), generator=make_generator(seed))
    n_val = round(val_fraction * len(x_t))
    return x_t[order[n_val:]], x_t[order[:n_val]]


def _mean_nll(flow: Flow, x: Tensor) -> Tensor:
    """Mean negative log-likelihood of the samples."""
    return -flow.log_prob(x).mean()


def fit(
    flow: Flow,
    x: ArrayLike,
    steps: int = 2000,
    lr: float = 1e-2,
    batch: int | None = None,
    every: int | None = 10,
    grid: ArrayLike | None = None,
    x_val: ArrayLike | None = None,
    seed: Seed = 0,
) -> History:
    """Train ``flow`` by maximum likelihood (forward KL) on the samples ``x`` with Adam.

    Parameters
    ----------
    flow
        The flow to train (in place).
    x
        Training samples of the target (1-D).
    steps
        Number of Adam updates.
    lr
        Learning rate.
    batch
        Samples per update, drawn at random with replacement; ``None`` uses all samples in every update (full batch,
        deterministic).
    every
        Take a snapshot every ``every`` updates (``None``: only before the first and after the last update).
    grid
        x values at which the density ``q`` and the map ``f`` are recorded in every snapshot.
    x_val
        Validation samples, whose loss is recorded in every snapshot (never used for the updates).
    seed
        Seed of the batch sampling (an integer, a :class:`torch.Generator` or ``None``).

    Returns
    -------
    History
        Losses, parameters and snapshots of the run.

    Raises
    ------
    FloatingPointError
        If the loss is not finite, e.g. with a uniform base when the last layer does not map into (0, 1).
    """
    x_t = as_tensor(x, flow.dtype).detach().reshape(-1)
    v_t = None if x_val is None else as_tensor(x_val, flow.dtype).detach().reshape(-1)
    g_t = None if grid is None else as_tensor(grid, flow.dtype).detach().reshape(-1)
    generator = make_generator(seed)
    every = every if every else steps
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    losses = np.empty(steps)
    snap_step: list[int] = []
    snap_train: list[float] = []
    snap_val: list[float] = []
    snap_q: list[np.ndarray] = []
    snap_z: list[np.ndarray] = []
    states: list[dict[str, Tensor]] = []

    def snapshot(step: int) -> None:
        with torch.no_grad():
            snap_step.append(step)
            snap_train.append(float(_mean_nll(flow, x_t)))
            if v_t is not None:
                snap_val.append(float(_mean_nll(flow, v_t)))
            if g_t is not None:
                z, logdet = flow(g_t)
                snap_q.append(torch.exp(flow.base.log_prob(z) + logdet).numpy().copy())
                snap_z.append(z.numpy().copy())
            states.append({k: v.detach().clone() for k, v in flow.state_dict().items()})

    start = time.perf_counter()
    snapshot(0)
    for step in range(steps):
        xb = x_t if batch is None else x_t[torch.randint(len(x_t), (batch,), generator=generator)]
        optimizer.zero_grad()
        loss = _mean_nll(flow, xb)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"the loss is {loss.item()} at update {step}: some samples have log q = -inf or nan (with a uniform "
                "base, the last layer must map into (0, 1), e.g. Sigmoid())"
            )
        loss.backward()
        optimizer.step()
        losses[step] = loss.item()
        if (step + 1) % every == 0 or step + 1 == steps:
            snapshot(step + 1)

    return History(
        step=np.asarray(snap_step),
        loss=losses,
        train_loss=np.asarray(snap_train),
        val_loss=np.asarray(snap_val) if v_t is not None else None,
        params={name: np.stack([state[name].numpy() for state in states]) for name in states[0]},
        states=states,
        grid=None if g_t is None else g_t.numpy().copy(),
        q_grid=np.stack(snap_q) if g_t is not None else None,
        z_grid=np.stack(snap_z) if g_t is not None else None,
        seconds=time.perf_counter() - start,
    )
