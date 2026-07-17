"""Test that ellipse intersection works."""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import optimize as spo

from provenny._kernel import area_core

_TOL = 1e-9  # the kernel's tolerance, defaulted for us by the public entry points


def _circle_overlap(d: float, r1: float, r2: float) -> float:
    """Closed-form overlap area of two circles, an independent test oracle."""
    return (
        r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        + r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        - 0.5 * ((r1 + r2 - d) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)) ** 0.5
    )


def _ellipse_forms(ellipses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Homogeneous forms of ellipse rows, normalized so ``det == -1``.

    A self-contained oracle so the tests never lean on the package internals they
    are meant to check.
    """
    cx, cy, major, minor, angle = np.moveaxis(ellipses, -1, 0)
    sin, cos = np.sin(angle), np.cos(angle)
    maj2, min2 = major**2, minor**2
    a = maj2 * sin**2 + min2 * cos**2
    b = (min2 - maj2) * sin * cos
    d = maj2 * cos**2 + min2 * sin**2
    e = -a * cx - b * cy
    f = -b * cx - d * cy
    g = a * cx**2 + 2 * b * cx * cy + d * cy**2 - maj2 * min2
    rows = [
        np.stack([a, b, e], -1),
        np.stack([b, d, f], -1),
        np.stack([e, f, g], -1),
    ]
    matrices = np.stack(rows, -2)
    scale = np.cbrt(-1.0 / np.linalg.det(matrices))
    return matrices * scale[..., None, None]


def _evaluate(
    forms: NDArray[np.float64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the form ``p^T C p`` at points, broadcasting batch dims."""
    return np.einsum("...i,...ij,...j->...", points, forms, points)


def _homogenize(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Append a unit homogeneous coordinate to each point."""
    return np.concatenate([points, np.ones((*points.shape[:-1], 1))], -1)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("num", [2, 3, 4])
def test_no_intersections_no_noverlap(num: int, seed: int) -> None:
    """Test that if there's no overlap the area is zero.

    We guarantee this by ensuring at least one ellipse is more than the num of majors away
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((num, 2))
    angles = rng.uniform(-np.pi, np.pi, num)
    majors = rng.random(num)
    minors = rng.random(num)
    majors += minors  # majors bounded by 2
    # majors are bounded by 2, so moving a center 4 away never intersects
    angle = rng.uniform(-np.pi, np.pi)
    centers[1, 0] = centers[0, 0] + 4 * np.cos(angle)
    centers[1, 1] = centers[0, 1] + 4 * np.sin(angle)
    ellipses = np.concat(
        [centers, majors[:, None], minors[:, None], angles[:, None]], 1
    )

    res = area_core(rng.permutation(ellipses), _TOL)
    assert np.isclose(res, 0)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("num", [1, 2, 3])
def test_no_intersections_overlap(num: int, seed: int) -> None:
    """Test that when there's no intersections, it's the area of the smallest.

    This does this by guaranteeing that the minors of all but one are greater
    than the majors of the others.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((1, 2)).repeat(num, 0)
    angles = rng.uniform(-np.pi, np.pi, num)
    majors = rng.standard_exponential(num)
    minors = rng.standard_exponential(num)
    majors[0] += minors[0]
    minors[1:] += majors[0]
    majors[1:] += minors[1:]
    ellipses = np.concat(
        [centers, majors[:, None], minors[:, None], angles[:, None]], 1
    )

    res = area_core(rng.permutation(ellipses), _TOL)
    assert np.isclose(res, majors[0] * minors[0] * np.pi)


@pytest.mark.parametrize("seed", range(10))
def test_circle_intersections(seed: int) -> None:
    """Test that intersection of two circles matches the closed form."""
    rng = np.random.default_rng(seed)
    c1x, c1y = rng.standard_normal(2)
    a1, a2 = rng.uniform(-np.pi, np.pi, 2)
    r1, r2 = rng.standard_exponential(2)
    angle = rng.uniform(-np.pi, np.pi)
    d = cast(float, rng.uniform(np.abs(r1 - r2), r1 + r2))  # any overlap
    c2x = c1x + d * np.cos(angle)
    c2y = c1y + d * np.sin(angle)
    ellipses = np.array([c1x, c1y, r1, r1, a1, c2x, c2y, r2, r2, a2]).reshape((2, 5))

    res = area_core(rng.permutation(ellipses), _TOL)
    expected = _circle_overlap(d, r1, r2)
    assert np.isclose(res, expected, atol=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_circle_intersections_inside(seed: int) -> None:
    """Test that intersection of two circles matches the closed form.

    Circles are always inside each other.
    """
    rng = np.random.default_rng(seed)
    c1x, c1y = rng.standard_normal(2)
    a1, a2 = rng.uniform(-np.pi, np.pi, 2)
    r1, r2 = rng.standard_exponential(2)
    angle = rng.uniform(-np.pi, np.pi)
    d = cast(float, rng.uniform(np.abs(r1 - r2), max(r1, r2)))  # centers are inside
    c2x = c1x + d * np.cos(angle)
    c2y = c1y + d * np.sin(angle)
    ellipses = np.array([c1x, c1y, r1, r1, a1, c2x, c2y, r2, r2, a2]).reshape((2, 5))

    res = area_core(rng.permutation(ellipses), _TOL)
    expected = _circle_overlap(d, r1, r2)
    assert np.isclose(res, expected, atol=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_circle_intersections_outside(seed: int) -> None:
    """Test that intersection of two circles matches the closed form.

    Circles are always outside each other.
    """
    rng = np.random.default_rng(seed)
    c1x, c1y = rng.standard_normal(2)
    a1, a2 = rng.uniform(-np.pi, np.pi, 2)
    r1, r2 = rng.standard_exponential(2)
    angle = rng.uniform(-np.pi, np.pi)
    d = cast(float, rng.uniform(max(r1, r2), r1 + r2))  # centers are outside
    c2x = c1x + d * np.cos(angle)
    c2y = c1y + d * np.sin(angle)
    ellipses = np.array([c1x, c1y, r1, r1, a1, c2x, c2y, r2, r2, a2]).reshape((2, 5))

    res = area_core(rng.permutation(ellipses), _TOL)
    expected = _circle_overlap(d, r1, r2)
    assert np.isclose(res, expected, atol=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_circle_intersections_subsumed(seed: int) -> None:
    """Test that intersection of two circles matches the closed form.

    One circle is subsumed inside a larger ellipse.
    """
    rng = np.random.default_rng(seed)
    c1x, c1y = rng.standard_normal(2)
    a1, a2, a3 = rng.uniform(-np.pi, np.pi, 3)
    r1, r2 = rng.standard_exponential(2)
    angle = rng.uniform(-np.pi, np.pi)
    d = cast(float, rng.uniform(np.abs(r1 - r2), r1 + r2))  # any overlap
    c2x = c1x + d * np.cos(angle)
    c2y = c1y + d * np.sin(angle)

    c3x = c1x + d * np.cos(angle) / 2
    c3y = c1y + d * np.sin(angle) / 2
    maj3, min3 = rng.standard_exponential(2)
    min3 += d / 2 + 2 * max(r1, r2)
    maj3 += min3

    ellipses = np.array(
        [c1x, c1y, r1, r1, a1, c2x, c2y, r2, r2, a2, c3x, c3y, maj3, min3, a3]
    ).reshape((3, 5))

    res = area_core(rng.permutation(ellipses), _TOL)
    expected = _circle_overlap(d, r1, r2)
    assert np.isclose(res, expected, atol=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_double_ellipse_intersection(seed: int) -> None:
    """Test two concentric perpendicular ellipses overlap within known bounds."""
    rng = np.random.default_rng(seed)
    cx, cy = rng.standard_normal(2)
    a1 = rng.uniform(-np.pi, np.pi)
    a2 = a1 + np.pi / 2
    minor, maj1, maj2 = rng.standard_exponential(3)
    minor += 1
    maj1 += 2 * minor
    maj2 += 2 * minor

    ellipses = np.array([cx, cy, maj1, minor, a1, cx, cy, maj2, minor, a2]).reshape(
        (2, 5)
    )

    res = area_core(rng.permutation(ellipses), _TOL)
    lower = minor**2 * np.pi
    upper = minor**2 * 4
    assert lower < res < upper


def _fun(x: NDArray[np.float64], val: NDArray[np.float64]) -> float:
    return float(x @ val)


def _jac(x: NDArray[np.float64], val: NDArray[np.float64]) -> NDArray[np.float64]:
    return val


def _cons_fun(
    x: NDArray[np.float64], forms: NDArray[np.float64]
) -> NDArray[np.float64]:
    return _evaluate(forms, _homogenize(x)[None])


def _cons_jac(
    x: NDArray[np.float64], forms: NDArray[np.float64]
) -> NDArray[np.float64]:
    arr = _homogenize(x)
    return 2 * (forms @ arr[:, None])[..., :2, 0]


def get_btree_approximation(
    ellipses: NDArray[np.float64], accuracy: float = 1e-3, *, tol: float = 1e-9
) -> tuple[float, float]:
    """Approximate the intersection area using a btree search."""
    forms = _ellipse_forms(ellipses)
    center = ellipses[:, :2].mean(0)

    points: list[NDArray[np.float64]] = []
    for score in [[1, 0], [0, 1], [-1, 0], [0, -1]]:

        def cons_fun(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return _cons_fun(x, forms)

        def cons_jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return _cons_jac(x, forms)

        res = spo.minimize(
            _fun,
            center,
            (np.array(score, "f8"),),
            jac=_jac,
            constraints=spo.NonlinearConstraint(
                cons_fun,
                -np.inf,
                0.0,
                cons_jac,
            ),
        )
        # failed to find a feasible point, therefore there's no intersection
        if not res.success:
            return 0.0, 0.0
        points.append(res.x)
    extents = np.stack(points)

    starts = extents.min(0)[None]
    expand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    widths = extents.max(0) - starts[0]
    remaining = widths.prod()
    total = 0

    # initial split
    widths /= 2
    starts = (starts[:, None] + widths * expand).reshape(-1, 2)

    while remaining - total > accuracy:
        corners = widths * expand + starts[:, None]
        caug = _homogenize(corners)
        mat_belong = _evaluate(forms[None, None], caug[:, :, None]) < tol
        belonging = mat_belong.all(-1)

        # pull out complete rects
        empty = ~belonging.any(-1)
        remaining -= empty.sum() * widths.prod()
        full = belonging.all(-1)
        total += full.sum() * widths.prod()
        starts = starts[~empty & ~full]

        # also update
        widths /= 2
        starts = (starts[:, None] + widths * expand).reshape(-1, 2)
    return total, remaining


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("num", [1, 2, 3, 4])
def test_btree_comparison(seed: int, num: int) -> None:
    """Test that this agrees with the btree approximation."""
    rng = np.random.default_rng(seed)
    cx, cy = rng.standard_normal((2, num))
    angles = rng.uniform(-np.pi, np.pi, num)
    majors = rng.random(num)
    minors = rng.random(num)
    majors += minors
    ellipses = np.stack([cx, cy, majors, minors, angles], 1)

    low, high = get_btree_approximation(ellipses)
    actual = area_core(ellipses, _TOL)
    # finding intersctions is approximate, so we can miss some areas
    assert low - 0.1 <= actual <= high + 0.1


def test_many_four_crossing_pairs_are_order_invariant() -> None:
    """Every pair crossing at four points fills the crossing buffer without corruption.

    Three elongated ellipses in a rotated star, each pair meeting at four points, is the
    case that overflows a too-small per-ellipse crossing buffer; a stray write into a
    neighbouring row would make the area depend on ellipse order, so pin order-invariance.
    """
    angles = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    ellipses = np.stack(
        [np.zeros(3), np.zeros(3), np.full(3, 1.6), np.full(3, 0.3), angles], axis=1
    )
    areas = [
        area_core(ellipses[list(order)], _TOL)
        for order in ((0, 1, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1))
    ]
    assert np.isfinite(areas).all()
    assert np.allclose(areas, areas[0])  # identical regardless of ellipse order


def test_duplicate_ellipses() -> None:
    """Identical ellipses intersect in their shared area, not garbage."""
    ellipses = np.array([[0.0, 0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0, 0.0]])
    res = area_core(ellipses, _TOL)
    assert np.isclose(res, 4 * np.pi)


def test_near_duplicate_ellipses() -> None:
    """A near-coincident duplicate is dropped, leaving the single area."""
    ellipses = np.array([[0.0, 0.0, 2.0, 2.0, 0.0], [1e-11, 0.0, 2.0, 2.0, 0.0]])
    res = area_core(ellipses, _TOL)
    assert np.isclose(res, 4 * np.pi)


@pytest.mark.parametrize("seed", range(10))
def test_near_tangent_circles(seed: int) -> None:
    """Two nearly-externally-tangent circles have a tiny closed-form overlap."""
    rng = np.random.default_rng(seed)
    r1, r2 = rng.uniform(0.5, 2, 2)
    gap = 1e-5
    d = cast(float, r1 + r2 - gap)  # just inside external tangency
    c1x, c1y = rng.standard_normal(2)
    angle = rng.uniform(-np.pi, np.pi)
    c2x = c1x + d * np.cos(angle)
    c2y = c1y + d * np.sin(angle)
    ellipses = np.array([[c1x, c1y, r1, r1, 0.0], [c2x, c2y, r2, r2, 0.0]])

    res = area_core(rng.permutation(ellipses), _TOL)
    expected = _circle_overlap(d, r1, r2)
    assert np.isclose(res, expected, atol=1e-8)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("offset", [0.0, 100.0, 1000.0])
def test_offset_circle_intersections(seed: int, offset: float) -> None:
    """The overlap area is invariant to translating far off the origin."""
    rng = np.random.default_rng(seed)
    r1, r2 = rng.uniform(0.5, 2, 2)
    d = cast(float, rng.uniform(np.abs(r1 - r2) + 0.1, r1 + r2 - 0.1))
    angle = rng.uniform(-np.pi, np.pi)
    c1x, c1y = offset, offset
    c2x = c1x + d * np.cos(angle)
    c2y = c1y + d * np.sin(angle)
    ellipses = np.array([[c1x, c1y, r1, r1, 0.0], [c2x, c2y, r2, r2, 0.0]])

    res = area_core(rng.permutation(ellipses), _TOL)
    expected = _circle_overlap(d, r1, r2)
    assert np.isclose(res, expected, atol=1e-6)
