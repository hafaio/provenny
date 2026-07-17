"""Test the pairwise disconnection penalty."""

import numpy as np

from provenny._kernel import area_core, penalty_core

_TOL = 1e-9  # the kernel's tolerance, defaulted for us by the public entry points

# an ellipse and its 90-degree rotation cross four times, splitting each set
# difference into two equal lobes
_ROTATED_PAIR = np.array([[0.0, 0.0, 1.6, 0.8, 0.0], [0.0, 0.0, 1.6, 0.8, np.pi / 2]])


def test_penalty_zero_for_simple_lens() -> None:
    """Two ellipses meeting in a simple lens (two crossings) have zero penalty."""
    ellipses = np.array([[0.0, 0.0, 1.0, 0.7, 0.0], [1.0, 0.0, 1.0, 0.7, 0.0]])
    assert penalty_core(ellipses, 4.0, _TOL) == 0.0


def test_penalty_positive_for_four_crossings() -> None:
    """A four-crossing pair produces a positive penalty."""
    assert penalty_core(_ROTATED_PAIR, 4.0, _TOL) > 0.0


def test_penalty_matches_symmetric_lobe() -> None:
    """The penalty equals 2*(lobe/total)**2 with the analytically known lobe area."""
    total = 4.0
    overlap = area_core(_ROTATED_PAIR, _TOL)
    lobe = (np.pi * 1.6 * 0.8 - overlap) / 2  # two equal lobes, both directions
    expected = 2 * (lobe / total) ** 2
    assert np.isclose(penalty_core(_ROTATED_PAIR, total, _TOL), expected, atol=1e-9)
