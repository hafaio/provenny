"""Test the zone arc-arrangement: areas, emptiness, and boundary loops."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from provenny import proportional_venn_array
from provenny._zone import zone


def _circles(*rows: tuple[float, float, float]) -> NDArray[np.float64]:
    """Build a circle layout as canonical ellipse rows: equal axes, zero angle."""
    return np.array([[cx, cy, radius, radius, 0.0] for cx, cy, radius in rows])


def _two_circles() -> NDArray[np.float64]:
    return _circles((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))


def test_zone_areas_match_targets() -> None:
    """Realized zone areas match the target subset areas the layout was built for."""
    areas = np.array([math.pi, math.pi, 0.6])  # |A|, |B|, |A & B|
    shapes = proportional_venn_array(areas, rng=np.random.default_rng(0))
    both = zone(shapes, np.array([True, True]))
    a_only = zone(shapes, np.array([True, False]))
    b_only = zone(shapes, np.array([False, True]))
    assert both is not None and a_only is not None and b_only is not None
    assert math.isclose(both.area, 0.6, rel_tol=1e-3)
    assert math.isclose(a_only.area, math.pi - 0.6, rel_tol=1e-3)
    assert math.isclose(b_only.area, math.pi - 0.6, rel_tol=1e-3)


def test_disjoint_intersection_zone_is_none() -> None:
    """A zone with no interior (disjoint sets' intersection) returns None."""
    shapes = _circles((0.0, 0.0, 1.0), (5.0, 0.0, 1.0))
    assert zone(shapes, np.array([True, True])) is None


def test_zone_rejects_a_layout_that_is_not_canonical_ellipses() -> None:
    """The canonical (n, 5) rows the solver returns are the only input; nothing is fixed up."""
    with pytest.raises(ValueError, match=r"\(n, 5\)"):
        zone(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]), np.array([True, True]))
    with pytest.raises(ValueError, match="canonical"):
        zone(np.array([[0.0, 0.0, 0.7, 1.5, 0.2]]), np.array([True]))


def test_zone_does_not_freeze_the_callers_layout() -> None:
    """The arcs own their rows, so a caller's array stays writable after a zone."""
    shapes = _two_circles()
    assert zone(shapes, np.array([True, True])) is not None
    shapes[0, 0] = 0.5  # still writable


def test_empty_collection_raises() -> None:
    """The all-false mask (the exterior) is not a zone."""
    with np.testing.assert_raises(ValueError):
        zone(_two_circles(), np.array([False, False]))


def test_zone_center_is_inside_the_zone() -> None:
    """The label point lands inside the intersection."""
    both = zone(_two_circles(), np.array([True, True]))
    assert both is not None
    center_x, center_y = both.center
    assert center_x**2 + center_y**2 <= 1.0
    assert (center_x - 1.0) ** 2 + center_y**2 <= 1.0


def test_near_coincident_sets_are_robust() -> None:
    """Near-coincident circles stay stable despite the kernel's near-singular pencil."""
    for eps in (1e-8, 1e-6, 1e-4):
        both = zone(_circles((0.0, 0.0, 1.0), (eps, 0.0, 1.0)), np.array([True, True]))
        assert both is not None
        assert math.isclose(both.area, math.pi, rel_tol=1e-3)


def test_coincident_sets_do_not_crash() -> None:
    """Two identical ellipses (equal sets) collapse rather than crashing the kernel."""
    circles = _circles((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))  # A == B
    both = zone(circles, np.array([True, True]))
    assert both is not None
    assert math.isclose(both.area, math.pi, rel_tol=1e-6)  # the shared disk
    # inside A and outside its twin B is a contradiction -> empty
    assert zone(circles, np.array([True, False])) is None


def test_coincident_detection_is_geometric() -> None:
    """The same circle stored with a different angle still collapses (form, not row)."""
    diff = np.array([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.7]])
    both = zone(diff, np.array([True, True]))
    assert both is not None
    assert math.isclose(both.area, math.pi, rel_tol=1e-6)
    assert zone(diff, np.array([True, False])) is None


def test_zone_center_is_lazy_and_cached() -> None:
    """The center is computed on first access and cached (same object thereafter)."""
    both = zone(_two_circles(), np.array([True, True]))
    assert both is not None
    first = both.center
    assert both.center is first


def test_zone_svg_path_is_wellformed() -> None:
    """The intersection boundary exports as a closed svg path."""
    both = zone(_two_circles(), np.array([True, True]))
    assert both is not None
    path = both.svg_path()
    assert path.startswith("M") and path.endswith("Z")
