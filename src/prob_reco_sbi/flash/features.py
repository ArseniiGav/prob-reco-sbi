"""The network input: the fixed per-event representation that the neural posterior estimator will see."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["network_input"]


def network_input(charge: ArrayLike, time_mean: ArrayLike | None = None, use_time: bool = True) -> np.ndarray:
    """Per-event input of the neural network: log charges and charge-weighted relative mean times.

    For an event with sensor charges ``n_j`` and mean hit times ``tau_j`` the input is::

        log(1 + n_j)                       j = 1 ... M
        tau_j - tau_bar  (0 if n_j = 0)    j = 1 ... M,   tau_bar = sum_j n_j tau_j / sum_j n_j  (hit sensors)

    The unknown event time t0 shifts every ``tau_j`` and ``tau_bar`` alike, so it drops out of the relative times.
    This holds without dark noise: noise hits have times uniform in the fixed readout window, which do not move with
    t0, so with dark noise the relative times of noisy sensors depend on t0. An event without any hit gets zeros for
    all relative times.

    Parameters
    ----------
    charge
        Sensor charges, shape ``[..., M]`` (e.g. ``data.response.charge``).
    time_mean
        Mean hit times [ns], same shape, NaN where the charge is 0 (e.g. ``data.response.time_mean``). Not needed
        with ``use_time=False``.
    use_time
        Append the relative times; ``False`` returns the log charges only.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``[..., 2M]`` (``[..., M]`` with ``use_time=False``).

    Examples
    --------
    One event with three sensors, of which two are hit:

    >>> x = network_input([[0, 3, 1]], [[np.nan, 12.0, 16.0]])
    >>> x[0, :3]  # log(1 + n_j)
    array([0.       , 1.3862944, 0.6931472], dtype=float32)
    >>> x[0, 3:]  # tau_j - tau_bar, with tau_bar = (3 * 12 + 1 * 16) / 4 = 13 ns
    array([ 0., -1.,  3.], dtype=float32)
    """
    n = np.asarray(charge, dtype=np.float64)
    log_charge = np.log1p(n)
    if not use_time:
        return log_charge.astype(np.float32)
    if time_mean is None:
        raise ValueError("time_mean is required with use_time=True")
    tau = np.asarray(time_mean, dtype=np.float64)
    if tau.shape != n.shape:
        raise ValueError(f"charge and time_mean must have the same shape, got {n.shape} and {tau.shape}")
    hit = n > 0
    weight = np.where(hit, n, 0.0)
    tau_hit = np.where(hit, tau, 0.0)
    tau_bar = (weight * tau_hit).sum(axis=-1, keepdims=True) / np.maximum(weight.sum(axis=-1, keepdims=True), 1.0)
    relative = np.where(hit, tau - tau_bar, 0.0)
    return np.concatenate([log_charge, relative], axis=-1).astype(np.float32)
