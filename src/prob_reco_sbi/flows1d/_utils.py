"""Small helpers shared by the modules of :mod:`prob_reco_sbi.flows1d` (not part of the public API)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

#: Anything that :func:`torch.as_tensor` turns into a tensor of numbers.
ArrayLike = torch.Tensor | np.ndarray | Sequence[float] | float

#: A seed: an integer, a ready :class:`torch.Generator` or ``None`` (the global random state).
Seed = int | torch.Generator | None


def as_tensor(x: ArrayLike, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert ``x`` to a tensor of the given floating-point type (no copy if it already is one)."""
    return torch.as_tensor(x, dtype=dtype)


def make_generator(seed: Seed) -> torch.Generator | None:
    """Return a CPU generator for an integer seed, the generator itself, or ``None`` for the global random state."""
    if seed is None or isinstance(seed, torch.Generator):
        return seed
    return torch.Generator().manual_seed(int(seed))
