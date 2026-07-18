"""Test the Ellipse value type and its boundary exports."""

import numpy as np
import pytest

from provenny import Bounds, Ellipse
from provenny._shape import canonicalize


def test_ellipse_reads_named_parameters() -> None:
    """Ellipse exposes the row's parameters as properties without copying it out."""
    center_x, center_y, major, minor, angle = 0.5, -0.2, 1.5, 0.7, 0.6  # along = major
    row = np.array([center_x, center_y, major, minor, angle])
    ellipse = Ellipse(row)
    assert ellipse.array is row  # wraps the row, no copy
    assert ellipse.center == (center_x, center_y)
    assert ellipse.major == major
    assert ellipse.minor == minor
    assert np.isclose(ellipse.angle, angle)


def test_canonicalize_orders_axes_major_first() -> None:
    """A minor-first row is turned into the major-first convention, input untouched."""
    along, across, angle = 0.7, 1.5, 0.2  # the along-angle semi-axis is the smaller one
    rows = np.array([[0.0, 0.0, along, across, angle]])
    ellipse = Ellipse(canonicalize(rows)[0])
    assert ellipse.major == across
    assert ellipse.minor == along
    assert np.isclose(ellipse.angle, (angle + np.pi / 2.0) % np.pi)  # a quarter-turn on
    assert rows[0, 2] == along  # the input is left untouched


def test_ellipse_rejects_a_row_that_is_not_a_canonical_ellipse() -> None:
    """The constructor is the gate: it takes canonical (5,) rows and nothing else."""
    with pytest.raises(ValueError, match="canonical"):
        Ellipse(np.array([0.0, 0.0, 0.7, 1.5, 0.2]))  # minor axis stored first
    with pytest.raises(ValueError, match="canonical"):
        Ellipse(np.array([0.0, 0.0, 1.5, 0.7, 4.0]))  # angle outside [0, pi)
    with pytest.raises(ValueError, match=r"\(5,\) row"):
        Ellipse(np.array([0.0, 0.0, 1.5]))  # a circle row is not promoted


def test_ellipse_area() -> None:
    """Ellipse.area is pi * major * minor, matching what its arc boundary encloses."""
    ellipse = Ellipse(np.array([0.5, -0.2, 1.5, 0.7, 0.6]))
    assert np.isclose(ellipse.area, np.pi * 1.5 * 0.7)


def test_ellipse_bounds_are_the_tight_axis_extents() -> None:
    """An axis-aligned ellipse bounds to center +/- its semi-axes; a rotation swaps them."""
    flat = Ellipse(np.array([0.5, -0.2, 2.0, 1.0, 0.0]))
    assert np.allclose(flat.bounds, (0.5 - 2.0, -0.2 - 1.0, 0.5 + 2.0, -0.2 + 1.0))
    upright = Ellipse(np.array([0.0, 0.0, 2.0, 1.0, np.pi / 2.0]))  # major now vertical
    assert np.allclose(upright.bounds, (-1.0, -2.0, 1.0, 2.0))


def test_ellipse_bounds_enclose_the_outline_tightly() -> None:
    """For any rotation the box contains every boundary point and is touched on all sides."""
    ellipse = Ellipse(np.array([0.5, -0.3, 2.0, 1.0, 0.7]))
    points = ellipse.sample(2000)
    min_x, min_y, max_x, max_y = ellipse.bounds
    assert points[:, 0].min() >= min_x and points[:, 0].max() <= max_x
    assert points[:, 1].min() >= min_y and points[:, 1].max() <= max_y
    assert np.isclose(points[:, 0].min(), min_x, atol=1e-3)
    assert np.isclose(points[:, 0].max(), max_x, atol=1e-3)
    assert np.isclose(points[:, 1].min(), min_y, atol=1e-3)
    assert np.isclose(points[:, 1].max(), max_y, atol=1e-3)


def test_bounds_is_a_named_tuple() -> None:
    """Bounds is a real tuple with names and convenience helpers on top."""
    box = Ellipse(np.array([0.0, 0.0, 2.0, 1.0, 0.0])).bounds
    assert isinstance(box, tuple)
    assert box == (-2.0, -1.0, 2.0, 1.0)  # compares equal to the plain 4-tuple
    min_x, min_y, max_x, max_y = box  # unpacks
    assert (min_x, min_y, max_x, max_y) == (box.min_x, box.min_y, box.max_x, box.max_y)
    assert box[0] == box.min_x  # indexes and names the same slot
    assert (box.width, box.height, box.center) == (4.0, 2.0, (0.0, 0.0))
    other = Bounds(1.0, -3.0, 5.0, 0.5)
    assert box.union(other) == (-2.0, -3.0, 5.0, 1.0)


def test_ellipse_array_is_readonly() -> None:
    """The wrapped row cannot be mutated through the Ellipse."""
    ellipse = Ellipse(np.array([0.0, 0.0, 1.5, 0.7, 0.6]))
    with np.testing.assert_raises(ValueError):
        ellipse.array[0] = 1.0


def test_ellipse_value_equality_and_hash() -> None:
    """Ellipses compare and hash by value (equal rows)."""
    row = [0.0, 0.0, 1.5, 0.7, 0.6]
    one, same, other = (Ellipse(np.array(r)) for r in (row, row, [1.0, *row[1:]]))
    assert one == same and one is not same
    assert hash(one) == hash(same)
    assert one != other


def test_ellipse_sample_lies_on_the_ellipse() -> None:
    """Ellipse.sample returns points on the outline (frame radius 1)."""
    ellipse = Ellipse(np.array([0.5, -0.2, 1.5, 0.7, 0.6]))
    points = ellipse.sample(40)
    center_x, center_y = ellipse.center
    dx, dy = points[:, 0] - center_x, points[:, 1] - center_y
    cos_a, sin_a = np.cos(-ellipse.angle), np.sin(-ellipse.angle)
    along_major = (cos_a * dx - sin_a * dy) / ellipse.major
    along_minor = (sin_a * dx + cos_a * dy) / ellipse.minor
    assert np.allclose(along_major**2 + along_minor**2, 1.0)


def test_ellipse_paths_are_wellformed() -> None:
    """The path exports give a closed svg string and matching matplotlib arrays."""
    ellipse = Ellipse(np.array([0.5, -0.2, 1.5, 0.7, 0.6]))
    path = ellipse.svg_path()
    assert path.startswith("M") and path.endswith("Z")
    vertices, codes = ellipse.matplotlib_path()
    assert len(vertices) == len(codes)


def test_ellipse_point_at_lies_on_the_boundary() -> None:
    """point_at returns boundary points (frame radius 1)."""
    ellipse = Ellipse(np.array([0.5, -0.3, 2.0, 1.0, 0.4]))
    for anomaly in np.linspace(0.0, 2 * np.pi, 12).tolist():
        along_major, along_minor = ellipse.frame(*ellipse.point_at(anomaly))
        assert np.isclose(np.hypot(along_major, along_minor), 1.0)


def test_ellipse_contains_and_anomaly_roundtrip() -> None:
    """Interior/exterior classification, and anomaly inverting point_at on the boundary."""
    ellipse = Ellipse(np.array([0.5, -0.3, 2.0, 1.0, 0.4]))
    center_x, center_y = ellipse.center
    assert ellipse.contains(center_x, center_y)  # the center is inside
    assert not ellipse.contains(center_x + 10.0, center_y)  # far outside
    for anomaly in (0.3, 1.2, 3.0):
        assert np.isclose(ellipse.anomaly(*ellipse.point_at(anomaly)), anomaly)
