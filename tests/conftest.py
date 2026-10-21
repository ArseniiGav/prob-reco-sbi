"""Shared fixtures of the test suite (all tests run on the CPU)."""

from __future__ import annotations

import pytest

from prob_reco_sbi.flash import SimResult, generate, preset


@pytest.fixture(scope="session")
def full_small() -> SimResult:
    """200 events of the full model with hits and the tracks of the first 3 events."""
    return generate(200, config=preset("full"), seed=11, device="cpu", save_hits=True, save_tracks=3)
