"""Test region_center label points."""

import numpy as np
import pytest
from numpy.typing import NDArray

from provenny._region import region_center


def _circles(*rows: tuple[float, float, float]) -> NDArray[np.float64]:
    """Build a circle layout as canonical ellipse rows: equal axes, zero angle."""
    return np.array([[cx, cy, radius, radius, 0.0] for cx, cy, radius in rows])


def _in_ellipse(ellipse: NDArray[np.float64], x: float, y: float) -> bool:
    cx, cy, major, minor, angle = ellipse
    dx, dy = x - cx, y - cy
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    along_major = (cos_a * dx - sin_a * dy) / major
    along_minor = (sin_a * dx + cos_a * dy) / minor
    return bool(along_major**2 + along_minor**2 <= 1.0)


def _consistent(
    ellipses: NDArray[np.float64],
    inside: NDArray[np.bool_],
    point: tuple[float, float],
) -> bool:
    """Whether point is inside exactly the sets flagged in ``inside``."""
    x, y = point
    return all(
        _in_ellipse(ellipses[i], x, y) == bool(inside[i])
        for i in range(ellipses.shape[0])
    )


def test_overlapping_regions_are_consistent() -> None:
    """Two overlapping circles yield A-only, B-only, and A&B, each labeled inside it."""
    shapes = _circles((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))
    for inside in ([True, False], [False, True], [True, True]):
        flags = np.array(inside)
        assert _consistent(shapes, flags, region_center(shapes, flags))


def test_disjoint_intersection_is_empty() -> None:
    """Disjoint sets have no intersection region, so it raises rather than mislabeling."""
    shapes = _circles((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="empty"):
        region_center(shapes, np.array([True, True]))
    for inside in ([True, False], [False, True]):  # the single-set regions exist
        flags = np.array(inside)
        assert _consistent(shapes, flags, region_center(shapes, flags))


def test_subset_drops_the_empty_region() -> None:
    """A set contained in another has no set-only region of its own."""
    shapes = _circles((0.0, 0.0, 2.0), (0.0, 0.0, 0.5))
    with pytest.raises(ValueError, match="empty"):
        region_center(shapes, np.array([False, True]))  # B is inside A
    for inside in ([True, False], [True, True]):  # A-only and A&B
        flags = np.array(inside)
        assert _consistent(shapes, flags, region_center(shapes, flags))


def test_exterior_mask_raises() -> None:
    """The region inside no set (the exterior) is not a labelable region."""
    shapes = _circles((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="exterior"):
        region_center(shapes, np.array([False, False]))


def test_eccentric_rotated_ellipses() -> None:
    """The clearance search handles real eccentricity and rotation, not just circles."""
    shapes = np.array([[0.0, 0.0, 1.5, 0.7, 0.0], [1.0, 0.0, 1.5, 0.7, 2.07]])
    flags = np.array([True, True])  # the ellipses overlap
    assert _consistent(shapes, flags, region_center(shapes, flags))
