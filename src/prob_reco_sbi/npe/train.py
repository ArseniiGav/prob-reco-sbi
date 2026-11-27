"""Training of a posterior model on simulated pairs, and saving and loading trained models.

The loss is the negative log-density of the training parameters given their observations, averaged over a minibatch::

    L(phi) = -(1/n) sum_i log q^(theta_i | x_i)  ->  E_x[ KL(q(. | x) || q^(. | x)) ] + const   (many pairs),

the forward KL divergence of 2a averaged over the observations: it needs pairs drawn from the prior and the simulator,
never the posterior. :func:`fit` runs Adam with a cosine decay of the learning rate, records the losses per update and
per epoch, keeps the state with the best validation loss and takes snapshots of the model for the animations::

    from prob_reco_sbi.npe import AutoregressiveFlow, fit, load_pairs

    theta, x = load_pairs("data/n3_full_train.h5", "charge+time")      # 310 000 pairs
    flow = AutoregressiveFlow(obs_dim=64)
    history = fit(flow, theta, x, epochs=40, val_fraction=1 / 31)        # the last 10 000 pairs validate
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from . import flow as _flow

__all__ = ["History", "fit", "load_model", "mean_loss", "save_model"]

ArrayLike = Tensor | np.ndarray

#: Pairs evaluated at once when the loss of a whole set is computed (memory control).
EVAL_CHUNK = 8192


@dataclass
class History:
    """Record of a training run of :func:`fit`.

    Epoch 0 is the untrained model; entry ``k`` of the per-epoch arrays describes the model after ``k`` epochs. The
    snapshots are taken after every ``every`` epochs, before the first update and at the extra update counts asked for.
    """

    epoch: np.ndarray  #: 0, 1, ..., epochs
    train_loss: (
        np.ndarray
    )  #: mean minibatch loss of each epoch [nats]; entry 0: the untrained model on the training set
    val_loss: np.ndarray | None  #: loss on the validation pairs after each epoch [nats], if there are any
    loss: np.ndarray  #: loss of every update on its minibatch [nats], shape [steps]
    learning_rate: np.ndarray  #: learning rate of every update, shape [steps]
    snapshot_step: np.ndarray  #: number of updates at each snapshot
    snapshot_epoch: np.ndarray  #: the same in epochs (fractional for the extra snapshots)
    states: list[dict[str, Tensor]] = field(repr=False)  #: copies of the model's state dict at the snapshots
    best_epoch: int = 0  #: epoch of the lowest validation loss (the last epoch without validation pairs)
    best_state: dict[str, Tensor] = field(default_factory=dict, repr=False)  #: state dict at ``best_epoch``
    steps_per_epoch: int = 0  #: updates per epoch
    n_train: int = 0  #: training pairs
    n_val: int = 0  #: validation pairs
    batch_size: int = 0  #: pairs per update
    lr: float = 0.0  #: initial learning rate
    seed: int | None = None  #: seed of the minibatch order
    seconds: float = 0.0  #: wall-clock time of the run [s]
    epoch_seconds: np.ndarray = field(default_factory=lambda: np.zeros(0))  #: wall-clock time at the end of each epoch

    @property
    def best_val_loss(self) -> float:
        """The lowest validation loss (``nan`` without validation pairs)."""
        return float(self.val_loss[self.best_epoch]) if self.val_loss is not None else math.nan

    def restore(self, model: nn.Module, index: int = -1) -> nn.Module:
        """Load snapshot ``index`` (default: the last) into ``model`` and return it."""
        model.load_state_dict(self.states[index])
        return model

    def restore_best(self, model: nn.Module) -> nn.Module:
        """Load the state with the lowest validation loss into ``model`` and return it."""
        model.load_state_dict(self.best_state)
        return model

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary of tensors, numbers and lists (for :func:`torch.save` with ``weights_only`` loading)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = torch.from_numpy(np.asarray(value).copy()) if isinstance(value, np.ndarray) else value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> History:
        """Inverse of :meth:`to_dict`."""
        values: dict[str, Any] = {k: v.numpy() if isinstance(v, Tensor) else v for k, v in data.items()}
        return cls(**values)


def _as_tensor(v: ArrayLike, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(v, dtype=dtype)


def _log_prob(model: nn.Module, theta: Tensor, x: Tensor) -> Tensor:
    """``model.log_prob(theta, x)`` of any model with that method."""
    return cast(Tensor, cast(Any, model).log_prob(theta, x))


def mean_loss(model: nn.Module, theta: ArrayLike, x: ArrayLike, chunk: int = EVAL_CHUNK) -> float:
    """Mean negative log-density ``-(1/N) sum_i log q^(theta_i | x_i)`` of pairs, without gradient [nats]."""
    dtype = getattr(model, "dtype", torch.float32)
    theta_t, x_t = _as_tensor(theta, dtype), _as_tensor(x, dtype)
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(theta_t), chunk):
            total -= float(_log_prob(model, theta_t[start : start + chunk], x_t[start : start + chunk]).sum())
    return total / max(len(theta_t), 1)


def fit(
    model: nn.Module,
    theta: ArrayLike,
    x: ArrayLike,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    seed: int | None = 0,
    every: int | None = 1,
    *,
    validation: tuple[ArrayLike, ArrayLike] | None = None,
    standardize: bool = True,
    restore_best: bool = True,
    extra_steps: Sequence[int] = (),
    verbose: bool = False,
) -> History:
    """Train a posterior model by minibatch maximum likelihood (the NPE loss) with Adam and a cosine decay.

    Parameters
    ----------
    model
        The model to train, in place: a :class:`~prob_reco_sbi.npe.PosteriorEstimator`, or any module with a method
        ``log_prob(theta, x)`` that returns one log-density per pair.
    theta
        Parameters of the pairs, shape ``[N, theta_dim]`` (physical units).
    x
        Observations of the pairs, shape ``[N, obs_dim]``.
    epochs
        Passes over the training pairs.
    batch_size
        Pairs per update (the last minibatch of an epoch may be smaller).
    lr
        Initial learning rate of Adam; it decays to zero over the run as a cosine.
    val_fraction
        Fraction of the pairs kept for validation: the **last** ``round(val_fraction N)`` pairs (the events of a file
        are independent, so the last ones are as good as any). ``0`` trains on all pairs. Ignored if ``validation``
        is given.
    seed
        Seed of the minibatch order (``None``: the global random state). The initial weights come from the global
        random state when the model is built: call ``torch.manual_seed`` before building it.
    every
        Take a snapshot of the model every ``every`` epochs (``None``: only before and after training).
    validation
        Explicit validation pairs ``(theta_val, x_val)``; then every pair of ``theta``, ``x`` is used for training.
    standardize
        Set the standardization of the model from the training pairs first (if it has ``set_standardization``).
        Pass ``False`` to continue the training of a model.
    restore_best
        After training, load the state with the lowest validation loss into the model (with validation pairs).
    extra_steps
        Additional update counts at which a snapshot is taken (e.g. ``range(0, 200, 10)`` for the first epoch of an
        animation).
    verbose
        Print one line per epoch.

    Returns
    -------
    History
        Losses per update and per epoch, snapshots, the best epoch and its state, timing.

    Raises
    ------
    FloatingPointError
        If a minibatch loss is not finite.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError(f"need epochs >= 1, batch_size >= 1 and lr > 0, got {epochs}, {batch_size}, {lr}")
    dtype = getattr(model, "dtype", torch.float32)
    theta_t, x_t = _as_tensor(theta, dtype).detach(), _as_tensor(x, dtype).detach()
    if len(theta_t) != len(x_t):
        raise ValueError(f"theta and x must have the same number of pairs, got {len(theta_t)} and {len(x_t)}")
    if validation is not None:
        theta_v, x_v = (_as_tensor(v, dtype).detach() for v in validation)
    else:
        if not 0 <= val_fraction < 1:
            raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
        n_val = round(val_fraction * len(theta_t))
        n_train = len(theta_t) - n_val
        theta_t, theta_v = theta_t[:n_train], theta_t[n_train:]
        x_t, x_v = x_t[:n_train], x_t[n_train:]
    has_val = len(theta_v) > 0
    if len(theta_t) == 0:
        raise ValueError("no training pairs left")
    if standardize and hasattr(model, "set_standardization"):
        cast(Any, model).set_standardization(theta_t, x_t)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    steps_per_epoch = math.ceil(len(theta_t) / batch_size)
    steps = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    every = every if every else epochs
    extra = {int(s) for s in extra_steps if 0 < int(s) < steps}

    losses, rates = np.empty(steps), np.empty(steps)
    train_loss, val_loss, epoch_seconds = [mean_loss(model, theta_t, x_t)], [], [0.0]
    snap_step: list[int] = []
    states: list[dict[str, Tensor]] = []

    def state() -> dict[str, Tensor]:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def snapshot(step: int) -> None:
        snap_step.append(step)
        states.append(state())

    if has_val:
        val_loss.append(mean_loss(model, theta_v, x_v))
    best_epoch, best_value, best_state = 0, val_loss[0] if has_val else math.inf, state()
    snapshot(0)
    start = time.perf_counter()
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(theta_t), generator=generator)
        total = 0.0
        for begin in range(0, len(theta_t), batch_size):
            index = order[begin : begin + batch_size]
            loss = -_log_prob(model, theta_t[index], x_t[index]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"the loss is {loss.item()} at update {step} (epoch {epoch}): some pairs have log q^ = -inf or nan "
                    "(parameters outside the prior range of the model, or a learning rate that is too large)"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[step], rates[step] = loss.item(), scheduler.get_last_lr()[0]
            scheduler.step()
            total += loss.item() * len(index)
            step += 1
            if step in extra:
                snapshot(step)
        model.eval()
        train_loss.append(total / len(theta_t))
        if has_val:
            val_loss.append(mean_loss(model, theta_v, x_v))
            if val_loss[-1] < best_value:
                best_epoch, best_value, best_state = epoch, val_loss[-1], state()
        else:
            best_epoch, best_state = epoch, {}
        epoch_seconds.append(time.perf_counter() - start)
        if epoch % every == 0 or epoch == epochs:
            snapshot(step)
        if verbose:
            line = f"epoch {epoch:3d}: training loss {train_loss[-1]:.4f}"
            if has_val:
                line += f", validation loss {val_loss[-1]:.4f}"
            print(f"{line}  [{epoch_seconds[-1]:.0f} s]", flush=True)

    if not has_val:
        best_state = states[-1]
    elif restore_best:
        model.load_state_dict(best_state)
    order_snap = np.argsort(snap_step, kind="stable")
    snap = np.asarray(snap_step)[order_snap]
    return History(
        epoch=np.arange(epochs + 1),
        train_loss=np.asarray(train_loss),
        val_loss=np.asarray(val_loss) if has_val else None,
        loss=losses,
        learning_rate=rates,
        snapshot_step=snap,
        snapshot_epoch=snap / steps_per_epoch,
        states=[states[i] for i in order_snap],
        best_epoch=best_epoch,
        best_state=best_state,
        steps_per_epoch=steps_per_epoch,
        n_train=len(theta_t),
        n_val=len(theta_v),
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        seconds=time.perf_counter() - start,
        epoch_seconds=np.asarray(epoch_seconds),
    )


_MODELS: dict[str, type[_flow.PosteriorEstimator]] = {
    cls.__name__: cls for cls in (_flow.ConditionalFlow1D, _flow.AutoregressiveFlow, _flow.GaussianChain)
}


def save_model(
    path: str | os.PathLike[str], model: _flow.PosteriorEstimator, history: History | None = None, **info: Any
) -> Path:
    """Save a model of the package (its class, constructor arguments and weights), its history and notes.

    Parameters
    ----------
    path
        Output file (``.pt``); missing parent directories are created.
    model
        A :class:`~prob_reco_sbi.npe.ConditionalFlow1D`, :class:`~prob_reco_sbi.npe.AutoregressiveFlow` or
        :class:`~prob_reco_sbi.npe.GaussianChain`.
    history
        The training record, if it should be kept.
    **info
        Notes stored alongside, e.g. the data file, the budget and the seeds (numbers, strings, lists, dicts).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    name = type(model).__name__
    if name not in _MODELS:
        raise TypeError(f"save_model stores the models of the package ({', '.join(_MODELS)}), got {name}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class": name,
        "hparams": dict(model.hparams),
        "state": model.state_dict(),
        "history": history.to_dict() if history is not None else None,
        "info": info,
    }
    torch.save(payload, path)
    return path


def load_model(path: str | os.PathLike[str]) -> tuple[_flow.PosteriorEstimator, History | None, dict[str, Any]]:
    """Load a model written by :func:`save_model`.

    Returns
    -------
    tuple
        ``(model, history, info)``: the model with its weights and standardization (in evaluation mode), its training
        record (or ``None``) and the stored notes.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    hparams = dict(payload["hparams"])
    for key in ("hidden", "head", "energy_range"):
        if key in hparams:
            hparams[key] = tuple(hparams[key])
    model = _MODELS[payload["class"]](**hparams)
    model.load_state_dict(payload["state"])
    model.eval()
    history = History.from_dict(payload["history"]) if payload["history"] is not None else None
    return model, history, dict(payload["info"])
