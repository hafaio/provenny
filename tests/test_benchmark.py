# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeStubs=false
"""Timing benchmarks for ``proportional_venn_array`` across modes and set counts.

Skipped by default (``--benchmark-skip`` is in the project's pytest options). To run them::

    uv run pytest tests/test_benchmark.py --benchmark-only --no-cov

``--benchmark-only`` overrides the default skip, and ``--no-cov`` drops coverage tracing so
the timings are trustworthy. Each case builds target areas from a reference layout, then
times the solve; the assertion just guards that the solve stays finite.

The pyright directives above relax only the unknown-type rules: pytest-benchmark ships no
stubs, so strict mode floods this file with third-party type noise.
"""

from collections.abc import Callable

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from provenny import proportional_venn_array
from provenny._kernel import area_core
from provenny._types import F64Array

_TOL = 1e-9  # the kernel tolerance the public entry points default to


def _subset_areas(shapes: F64Array) -> F64Array:
    """Return the exact area of every non-empty subset, bitmask-ordered."""
    num_sets = shapes.shape[0]
    areas = np.zeros(2**num_sets - 1)
    for mask in range(1, 2**num_sets):
        members = [index for index in range(num_sets) if mask & (1 << index)]
        areas[mask - 1] = area_core(shapes[members], _TOL)
    return areas


def _rosette(num_sets: int) -> F64Array:
    """``num_sets`` equal circles evenly spaced on a ring -- symmetric partial overlaps."""
    theta = np.linspace(0, 2 * np.pi, num_sets, endpoint=False)
    return np.stack(
        [
            0.7 * np.cos(theta),
            0.7 * np.sin(theta),
            np.full(num_sets, 1.0),
            np.full(num_sets, 1.0),
            np.zeros(num_sets),
        ],
        axis=1,
        dtype="f8",
    )


def _nested(num_sets: int) -> F64Array:
    """``num_sets`` concentric circles of decreasing radius -- a subset chain."""
    radii = np.linspace(1.6, 0.5, num_sets)
    zeros = np.zeros(num_sets)
    return np.stack([zeros, zeros, radii, radii, zeros], axis=1, dtype="f8")


def _ellipse_rosette(num_sets: int) -> F64Array:
    """``num_sets`` rotated ellipses on a ring -- targets that need real eccentricity."""
    theta = np.linspace(0, 2 * np.pi, num_sets, endpoint=False)
    return np.stack(
        [
            0.55 * np.cos(theta),
            0.55 * np.sin(theta),
            np.full(num_sets, 1.25),
            np.full(num_sets, 0.6),
            theta,
        ],
        axis=1,
        dtype="f8",
    )


_SCENARIOS: dict[str, Callable[[int], F64Array]] = {
    "nested": _nested,
    "rosette": _rosette,
    "ellipse": _ellipse_rosette,
}


@pytest.mark.parametrize("scenario", _SCENARIOS)
@pytest.mark.parametrize("num_sets", [2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["circle", "ellipse", "optimal"])
def test_solve_speed(
    benchmark: BenchmarkFixture, scenario: str, num_sets: int, mode: str
) -> None:
    """Time a single mode/scenario/set-count solve from a fixed seed."""
    areas = _subset_areas(_SCENARIOS[scenario](num_sets))
    shapes = benchmark(
        proportional_venn_array,
        areas,
        mode=mode,
        restarts=5,
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(shapes).all()
