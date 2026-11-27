"""The network input: the fixed per-event representation that the neural posterior estimator will see.

:func:`network_input` gives the log charges and relative times of all sensors; :func:`ganged_input` gives the same for
a ganged readout, in which sensor ``j`` and its mirror image ``M - j`` (mirror axis: the x axis) share one channel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ganged_channels", "ganged_input", "network_input"]


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


def ganged_channels(n_sensors: int = 32) -> np.ndarray:
    """Return the channel of every sensor of the ganged readout: sensors ``j`` and ``(M - j) mod M`` share one.

    Channel ``k = min(j, M - j)`` holds sensors ``k`` and ``M - k``.

    Sensor ``j`` sits at the angle ``2 pi j / M``, its mirror image under ``y0 -> -y0`` at ``-2 pi j / M``, which is
    sensor ``M - j``. Sensors 0 and ``M / 2`` (for even ``M``) lie on the mirror axis and keep a channel of their own,
    so there are ``M // 2 + 1`` channels (17 for 32 sensors).

    Parameters
    ----------
    n_sensors
        Number of sensors ``M``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``[M]`` with values ``0 ... M // 2``.

    Examples
    --------
    >>> ganged_channels(8)
    array([0, 1, 2, 3, 4, 3, 2, 1])
    """
    if n_sensors < 1:
        raise ValueError(f"n_sensors must be at least 1, got {n_sensors}")
    j = np.arange(n_sensors)
    return np.minimum(j, (n_sensors - j) % n_sensors)


def ganged_input(charge: ArrayLike, time_mean: ArrayLike | None = None, use_time: bool = True) -> np.ndarray:
    """Network input of a ganged readout: sensors ``j`` and ``M - j`` share one channel (see :func:`ganged_channels`).

    The charge of a channel is the sum of the charges of its sensors, its mean time the charge-weighted mean of their
    mean times. The result is :func:`network_input` of the channels. A response and its mirror image (the response of
    the event ``(x0, -y0, E)``) give exactly the same input, so the posterior of a ganged readout is symmetric under
    ``y0 -> -y0``.

    Parameters
    ----------
    charge
        Sensor charges, shape ``[..., M]``.
    time_mean
        Mean hit times [ns], same shape, NaN where the charge is 0. Not needed with ``use_time=False``.
    use_time
        Append the relative times of the channels; ``False`` returns the log charges only.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``[..., 2 C]`` (``[..., C]`` with ``use_time=False``), ``C = M // 2 + 1`` channels
        (34 values for 32 sensors).

    Examples
    --------
    Four sensors give three channels: sensor 0, sensors 1 and 3, sensor 2.

    >>> x = ganged_input([[2, 1, 0, 3]], [[10.0, 20.0, np.nan, 12.0]])
    >>> x[0, :3]  # log(1 + c_k) with the channel charges c = (2, 4, 0)
    array([1.0986123, 1.609438 , 0.       ], dtype=float32)
    >>> x[0, 3:]  # channel times (10, 14 ns) relative to their charge-weighted mean, 38 / 3 ns
    array([-2.6666667,  1.3333334,  0.       ], dtype=float32)
    """
    n = np.asarray(charge, dtype=np.float64)
    channels = ganged_channels(n.shape[-1])
    n_channels = int(channels.max()) + 1
    one_hot = (channels[:, None] == np.arange(n_channels)[None, :]).astype(np.float64)  # [M, C]
    c = n @ one_hot
    if not use_time:
        return network_input(c, use_time=False)
    if time_mean is None:
        raise ValueError("time_mean is required with use_time=True")
    tau = np.asarray(time_mean, dtype=np.float64)
    if tau.shape != n.shape:
        raise ValueError(f"charge and time_mean must have the same shape, got {n.shape} and {tau.shape}")
    weighted = np.where(n > 0, n * tau, 0.0) @ one_hot
    tau_c = np.where(c > 0, weighted / np.maximum(c, 1.0), np.nan)
    return network_input(c, tau_c)
