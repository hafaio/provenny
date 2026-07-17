"""Test the Arc value type and its path/area conversions."""

import math
from itertools import pairwise

import numpy as np
import pytest

from provenny._shape import (
    Arc,
    Ellipse,
    loops_area,
    loops_matplotlib_path,
    loops_svg_path,
    sample_loop,
)

_MOVETO, _CURVE4, _CLOSEPOLY = 1, 4, 79
_SUBPIXEL = 1e-3  # cubic bezier stays this close to the true ellipse
_SLACK = 1e-12  # float slack for a "<=" on an exact quarter turn


def _arc(  # noqa: PLR0913
    cx: float,
    cy: float,
    major: float,
    minor: float,
    angle: float,
    start: float,
    end: float,
) -> Arc:
    """Build an Arc from scalar ellipse params (Arc holds an Ellipse)."""
    return Arc(Ellipse(np.array([cx, cy, major, minor, angle])), start, end)


def _rho(arc: Arc, x: float, y: float) -> float:
    """Frame radius of ``(x, y)`` for the arc's ellipse; 1 on the boundary."""
    along_major, along_minor = arc.ellipse.frame(x, y)
    return math.hypot(along_major, along_minor)


def _bezier_at(
    points: tuple[tuple[float, float], ...], t: float
) -> tuple[float, float]:
    weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3)
    return (
        sum(w * p[0] for w, p in zip(weights, points, strict=True)),
        sum(w * p[1] for w, p in zip(weights, points, strict=True)),
    )


def test_split_is_contiguous_and_bounded() -> None:
    """A full turn splits into four contiguous sub-arcs, each at most a quarter turn."""
    arc = _arc(0.0, 0.0, 1.0, 1.0, 0.0, 0.2, 0.2 + 2 * math.pi)
    quarter = math.pi / 2
    pieces = list(arc.split())
    assert len(pieces) == math.ceil((arc.end - arc.start) / quarter)
    assert pieces[0].start == arc.start
    assert math.isclose(pieces[-1].end, arc.end)
    for piece in pieces:
        assert abs(piece.end - piece.start) <= quarter + _SLACK
    for before, after in pairwise(pieces):
        assert math.isclose(before.end, after.start)


def test_bezier_endpoints_exact_and_midpoint_subpixel() -> None:
    """The bezier meets the arc exactly at the ends and stays sub-pixel off it between."""
    arc = _arc(0.5, -0.3, 2.0, 1.0, 0.4, 0.3, 0.3 + math.pi / 2)
    control = arc.bezier()
    assert np.allclose(control[0], arc.ellipse.point_at(arc.start))
    assert np.allclose(control[3], arc.ellipse.point_at(arc.end))
    assert abs(_rho(arc, *_bezier_at(control, 0.5)) - 1.0) < _SUBPIXEL


def test_arc_rejects_non_canonical_row() -> None:
    """An Arc cannot be built around a bad row: its Ellipse is the gate."""
    with pytest.raises(ValueError, match="canonical"):
        _arc(0.0, 0.0, 0.7, 1.5, 0.2, 0.0, 1.0)  # minor axis stored first


def test_arc_value_equality_and_hash() -> None:
    """Arcs compare and hash by value, from the dataclass defaults (Ellipse does too)."""
    arc = _arc(0.0, 0.0, 2.0, 1.0, 0.3, 0.0, 1.0)
    same = _arc(0.0, 0.0, 2.0, 1.0, 0.3, 0.0, 1.0)
    other = _arc(0.0, 0.0, 2.0, 1.0, 0.3, 0.0, 2.0)  # a different end
    assert arc == same and arc is not same
    assert hash(arc) == hash(same)
    assert arc != other
    assert {arc, same, other} == {arc, other}  # hashable and dedups by value


def test_area_of_ellipse_is_pi_a_b() -> None:
    """A full ellipse loop encloses exactly pi * major * minor, matching Ellipse.area."""
    arc = _arc(1.0, 2.0, 2.0, 1.3, 0.7, 0.0, 2 * math.pi)
    assert math.isclose(loops_area([[arc]]), math.pi * 2.0 * 1.3)
    assert math.isclose(loops_area([[arc]]), arc.ellipse.area)


def test_svg_path_is_wellformed() -> None:
    """The svg path opens with a move, has one cubic per quarter turn, and closes."""
    loop = [_arc(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2 * math.pi)]
    path = loops_svg_path([loop])
    assert path.startswith("M")
    assert "C" in path  # cubic segments
    assert path.endswith("Z")


def test_matplotlib_path_is_wellformed() -> None:
    """Vertices/codes form a closed cubic path matplotlib can fill."""
    loop = [_arc(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2 * math.pi)]
    vertices, codes = loops_matplotlib_path([loop])
    assert codes[0] == _MOVETO
    assert codes[-1] == _CLOSEPOLY
    assert _CURVE4 in codes  # cubic control points
    assert len(vertices) == len(codes)


def test_sample_loop_lies_on_ellipse() -> None:
    """Sampled boundary points lie on the ellipse."""
    arc = _arc(0.0, 0.0, 2.0, 1.0, 0.3, 0.0, 2 * math.pi)
    for x, y in sample_loop([arc], 40):
        assert math.isclose(_rho(arc, float(x), float(y)), 1.0)
