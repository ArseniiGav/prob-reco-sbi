"""Training pairs ``{theta_i, x_i}`` from simulated data: the event parameters and what the network is given.

::

    theta, x = load_pairs("data/n3_full_train.h5", "charge+time")    # [310000, 3] and [310000, 64]

The parameters theta are those that the prior of the file draws at random, in the order ``(x0, y0, E)`` [m, MeV]: a
file simulated at a fixed vertex (``--fixed-vertex``) gives ``theta = (E,)``, one at a fixed energy
``theta = (x0, y0)``. The observations x, by representation:

==============  ==================================================================  =================
representation  x                                                                   size (32 sensors)
==============  ==================================================================  =================
"charge+time"   log charges and relative mean times, ``network_input(charge, tau)``  64
"charge"        log charges, ``network_input(charge, use_time=False)``               32
"q"             the log of the total charge, ``ln Q``                                1
"ganged"        log charges and relative times of the ganged readout,                34
                ``ganged_input(charge, tau)``
==============  ==================================================================  =================
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from ..flash import SimConfig, SimResult, ganged_input, load, network_input

__all__ = ["REPRESENTATIONS", "load_pairs", "observations", "parameter_names", "to_pairs"]

#: The representations of the observations, see the module docstring.
REPRESENTATIONS = ("charge+time", "charge", "q", "ganged")


def parameter_names(config: SimConfig) -> tuple[str, ...]:
    """Names of the parameters that the prior of ``config`` draws at random: ``("x0", "y0", "E")`` or a part of it."""
    names: tuple[str, ...] = ()
    if config.prior.fixed_vertex is None:
        names += ("x0", "y0")
    if config.prior.fixed_energy is None:
        names += ("E",)
    if not names:
        raise ValueError("the prior fixes both the vertex and the energy: there is no parameter to infer")
    return names


def observations(charge: np.ndarray, time_mean: np.ndarray | None, representation: str = "charge+time") -> np.ndarray:
    """Return the observations x of events from their sensor charges and mean times (float32, one row per event).

    Parameters
    ----------
    charge
        Sensor charges, shape ``[N, M]``.
    time_mean
        Mean hit times [ns], same shape (needed for ``"charge+time"`` and ``"ganged"``).
    representation
        One of :data:`REPRESENTATIONS`.

    Returns
    -------
    numpy.ndarray
        Shape ``[N, size]`` (see the table of the module).
    """
    if representation == "charge+time":
        return network_input(charge, time_mean)
    if representation == "charge":
        return network_input(charge, use_time=False)
    if representation == "q":
        q = np.asarray(charge, dtype=np.float64).sum(axis=-1, keepdims=True)
        return np.log(np.maximum(q, 1.0)).astype(np.float32)  # an event without any hit gets ln 1 = 0
    if representation == "ganged":
        return ganged_input(charge, time_mean)
    raise ValueError(f"unknown representation {representation!r}; choose from {', '.join(REPRESENTATIONS)}")


def to_pairs(
    result: SimResult, representation: str = "charge+time", dtype: torch.dtype = torch.float32
) -> tuple[Tensor, Tensor]:
    """Return the pairs ``(theta, x)`` of simulated events in memory (from ``generate``, ``simulate`` or ``load``).

    Parameters
    ----------
    result
        Simulated events.
    representation
        One of :data:`REPRESENTATIONS`.
    dtype
        Floating-point type of the tensors.

    Returns
    -------
    tuple of Tensor
        ``theta`` of shape ``[N, n_params]`` (the parameters of :func:`parameter_names`) and ``x`` of shape
        ``[N, size]``.
    """
    columns = {"x0": result.truth.x, "y0": result.truth.y, "E": result.truth.energy}
    theta = np.stack([columns[name] for name in parameter_names(result.config)], axis=-1)
    x = observations(result.response.charge, result.response.time_mean, representation)
    return torch.as_tensor(theta, dtype=dtype), torch.as_tensor(x, dtype=dtype)


def load_pairs(
    path: str | os.PathLike[str], representation: str = "charge+time", dtype: torch.dtype = torch.float32
) -> tuple[Tensor, Tensor]:
    """Read the pairs ``(theta, x)`` of a data file written by ``flashsim`` (see :func:`to_pairs`).

    Parameters
    ----------
    path
        HDF5 file.
    representation
        One of :data:`REPRESENTATIONS`.
    dtype
        Floating-point type of the tensors.

    Returns
    -------
    tuple of Tensor
        ``theta`` ``[N, n_params]`` and ``x`` ``[N, size]``.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}; choose from {', '.join(REPRESENTATIONS)}")
    return to_pairs(load(path, hits=False, tracks=False), representation, dtype)
