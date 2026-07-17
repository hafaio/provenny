"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Make a seeded generator so the solver's restart seeds are reproducible."""
    return np.random.default_rng(0)
