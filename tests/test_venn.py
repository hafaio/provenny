"""Test the top-level proportional_venn layout."""

import numpy as np
import pytest
from numpy.typing import NDArray

from provenny import proportional_venn, proportional_venn_array, zone
from provenny._kernel import area_core, penalty_core

_TOL = 1e-9  # the kernel's tolerance, defaulted for us by the public entry points


def _subset_area(layout: NDArray[np.float64], mask: list[int]) -> float:
    """Intersection area of the shapes selected by mask."""
    return area_core(layout[mask], _TOL)


def test_circle_layout_shape(rng: np.random.Generator) -> None:
    """Circle mode returns the same ellipse rows as the other modes: equal axes, no angle."""
    areas = np.array([np.pi, np.pi, 0.4])
    layout = proportional_venn_array(areas, rng=rng)
    assert layout.shape == (2, 5)
    assert np.allclose(layout[:, 2], layout[:, 3])  # major == minor: a circle
    assert np.all(layout[:, 4] == 0.0)


def test_optimal_layout_shape(rng: np.random.Generator) -> None:
    """Optimal mode returns one (cx, cy, major, minor, angle) row per set."""
    areas = np.array([np.pi, np.pi, 0.4])
    layout = proportional_venn_array(areas, mode="optimal", rng=rng)
    assert layout.shape == (2, 5)


def test_two_disjoint_sets(rng: np.random.Generator) -> None:
    """Two equal disjoint sets get unit radii and are placed apart."""
    areas = np.array([np.pi, np.pi, 0.0])
    layout = proportional_venn_array(areas, rng=rng)
    radii = layout[:, 2]
    assert np.allclose(radii, 1.0, atol=1e-2)
    dist = np.linalg.norm(layout[0, :2] - layout[1, :2])
    assert dist >= radii.sum() - 1e-2


def test_two_overlapping_sets(rng: np.random.Generator) -> None:
    """The requested pairwise overlap is realized by the layout."""
    overlap = 0.6
    areas = np.array([np.pi, np.pi, overlap])
    layout = proportional_venn_array(areas, rng=rng)
    assert np.isclose(_subset_area(layout, [0]), np.pi, atol=5e-2)
    assert np.isclose(_subset_area(layout, [1]), np.pi, atol=5e-2)
    assert np.isclose(_subset_area(layout, [0, 1]), overlap, atol=5e-2)


def test_subset_layout(rng: np.random.Generator) -> None:
    """A smaller set fully contained in a larger one overlaps by its own area."""
    small, big = np.pi, 4 * np.pi
    areas = np.array([big, small, small])
    layout = proportional_venn_array(areas, rng=rng)
    assert np.isclose(_subset_area(layout, [0, 1]), small, atol=5e-2)


@pytest.mark.parametrize("seed", range(5))
def test_three_sets_areas(seed: int) -> None:
    """Three-set circle areas roughly match the requested individual areas."""
    areas = np.array([np.pi, np.pi, 0.5, np.pi, 0.5, 0.5, 0.2])
    layout = proportional_venn_array(areas, rng=np.random.default_rng(seed))
    assert layout.shape == (3, 5)
    for single in range(3):
        assert np.isclose(_subset_area(layout, [single]), np.pi, atol=1e-1)


def test_rejects_all_zero_areas() -> None:
    """All-zero areas are rejected with a clear error, not an assertion crash."""
    with pytest.raises(ValueError, match="positive"):
        proportional_venn_array(np.zeros(3))


def test_rejects_wrong_length_array() -> None:
    """An array whose length is not 2**n - 1 is rejected."""
    with pytest.raises(ValueError, match="2\\*\\*n"):
        proportional_venn_array(np.array([1.0, 1.0]))


def test_rejects_zero_set_area() -> None:
    """A set with zero area is rejected, not laid out with a degenerate/negative radius."""
    with pytest.raises(ValueError, match="positive"):
        proportional_venn_array(np.array([1.0, 0.0, 0.0]))


def test_rejects_nonfinite_area() -> None:
    """NaN/inf areas are rejected up front, not run into a bare assertion."""
    with pytest.raises(ValueError, match="finite"):
        proportional_venn_array(np.array([1.0, np.nan, 0.4]))


def test_rejects_nonpositive_restarts() -> None:
    """A restarts value below 1 is rejected with a message, not a bare assertion."""
    with pytest.raises(ValueError, match="restarts"):
        proportional_venn_array(np.array([1.0, 1.0, 0.4]), restarts=0)


def test_string_key_is_single_char_names() -> None:
    """A bare-string subset key is read as a collection of single-character names."""
    diagram = proportional_venn(
        {"A": 1.0, "B": 1.0, "AB": 0.4}, rng=np.random.default_rng(0)
    )
    assert diagram.names == ("A", "B")
    assert diagram.shapes.shape == (2, 5)


def test_rejects_inconsistent_named_areas() -> None:
    """Areas implying a negative region size are rejected."""
    with pytest.raises(ValueError, match="inconsistent"):
        proportional_venn({("A",): 1.0, ("B",): 1.0, ("A", "B"): 2.0})


_THREE = [[0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]


def _ellipse_rms(layout: NDArray[np.float64], areas: NDArray[np.float64]) -> float:
    realized = np.array([area_core(layout[m], _TOL) for m in _THREE])
    return float(np.sqrt(np.mean((realized - areas) ** 2)))


def test_optimal_stays_circular_when_circles_fit(rng: np.random.Generator) -> None:
    """Optimal mode adds no eccentricity when circles already fit (a 2-set target)."""
    areas = np.array([np.pi, np.pi, 0.6])
    layout = proportional_venn_array(areas, mode="optimal", rng=rng)
    assert layout.shape == (2, 5)
    ratio = layout[:, 3] / layout[:, 2]
    assert np.allclose(ratio, 1.0, atol=1e-2)


def test_optimal_fits_ellipse_target(rng: np.random.Generator) -> None:
    """Optimal mode fits a target that circles cannot, using real eccentricity."""
    theta = np.deg2rad(np.array([90.0, 210.0, 330.0]))
    src = np.stack(
        [
            0.55 * np.cos(theta),
            0.55 * np.sin(theta),
            np.full(3, 1.25),
            np.full(3, 0.6),
            theta,
        ],
        axis=1,
    )
    areas = np.array([area_core(src[m], _TOL) for m in _THREE])

    circle = proportional_venn_array(areas, mode="circle", rng=rng)
    ellipse = proportional_venn_array(areas, mode="optimal", rng=rng)

    fit_tol = 1e-2
    min_eccentricity = 0.1
    assert _ellipse_rms(ellipse, areas) < _ellipse_rms(circle, areas)
    assert _ellipse_rms(ellipse, areas) < fit_tol
    assert np.abs(ellipse[:, 3] / ellipse[:, 2] - 1).max() > min_eccentricity


def test_optimal_fits_four_set_ellipse_target(rng: np.random.Generator) -> None:
    """Optimal mode fits a 4-set ellipse target the round anneal alone stalls on.

    The radial structured seed escapes the circle basin that a warm-start-from-circles
    anneal cannot leave once four sets are involved.
    """
    theta = np.linspace(0, 2 * np.pi, 4, endpoint=False, dtype=np.float64)
    src = np.stack(
        [
            0.55 * np.cos(theta),
            0.55 * np.sin(theta),
            np.full(4, 1.25),
            np.full(4, 0.6),
            theta,
        ],
        axis=1,
    )
    masks = [[i for i in range(4) if mask & (1 << i)] for mask in range(1, 2**4)]
    areas = np.array([area_core(src[m], _TOL) for m in masks])

    ellipse = proportional_venn_array(areas, mode="optimal", rng=rng)
    realized = np.array([area_core(ellipse[m], _TOL) for m in masks])
    rms = float(np.sqrt(np.mean((realized - areas) ** 2)))
    fit_tol = 1e-2
    assert rms < fit_tol


def test_ellipse_mode_smoke(rng: np.random.Generator) -> None:
    """Ellipse mode runs end to end and returns connected (penalty-free) shapes.

    The four-set case where the penalty must actively remove a disconnection is
    verified separately; it is too slow for the default gate (the penalty evaluates
    every pair's crossings on each objective step).
    """
    areas = np.array([np.pi, np.pi, 0.6])
    layout = proportional_venn_array(areas, mode="ellipse", rng=rng)
    assert layout.shape == (2, 5)
    assert penalty_core(layout, float(areas.sum()), _TOL) == 0.0


def test_default_restarts_and_reproducibility() -> None:
    """restarts=None picks the n-scaled default, and a seeded generator is reproducible."""
    areas = np.array([np.pi, np.pi, 0.5, np.pi, 0.5, 0.5, 0.2])
    first = proportional_venn_array(areas, mode="optimal", rng=np.random.default_rng(0))
    second = proportional_venn_array(
        areas, mode="optimal", rng=np.random.default_rng(0)
    )
    assert first.shape == (3, 5)  # restarts=None resolved and solved
    assert np.array_equal(first, second)  # same seed -> identical layout


def test_zone_center_of_crescent_regions() -> None:
    """Every non-empty zone yields a label point, even thin single-set crescents.

    A heavily overlapped set's "only" region is a crescent that contains none of the sets'
    centers; the label search must seed from the boundary, or it falsely reports the (clearly
    non-empty, it has boundary loops) zone empty.
    """
    generator = np.random.default_rng(50)
    spread = generator.uniform(0.4, 1.1)
    source = np.stack(
        [
            generator.uniform(-spread, spread, 5),
            generator.uniform(-spread, spread, 5),
            generator.uniform(0.7, 1.6, 5),
            generator.uniform(0.35, 1.1, 5),
            generator.uniform(0, 2 * np.pi, 5),
        ],
        axis=1,
    )
    masks = [[i for i in range(5) if mask & (1 << i)] for mask in range(1, 2**5)]
    areas = np.array([area_core(source[m], _TOL) for m in masks])
    shapes = proportional_venn_array(
        areas, mode="optimal", rng=np.random.default_rng(1)
    )
    for mask in masks:
        inside = np.array([i in mask for i in range(5)], dtype=np.bool_)
        region = zone(shapes, inside)
        if region is not None:
            x, y = region.center  # must not raise "empty" for a zone that exists
            assert np.isfinite([x, y]).all()


def test_near_degenerate_target_does_not_crash() -> None:
    """A near-identical pair (99.9999% overlap) yields a layout, not an assertion.

    Such a target places two near-coincident circles; the intersection area stays finite
    (it falls back to the containment estimate) so the objective never becomes NaN.
    """
    target = np.array([1.0, 1.0, 0.999999])
    for seed in range(20):  # the failure was seed-dependent, so sweep a few
        shapes = proportional_venn_array(
            target, mode="optimal", rng=np.random.default_rng(seed)
        )
        assert shapes.shape == (2, 5)
