"""Representative interior point for a region of a laid-out diagram.

A region is one in/out combination of the sets -- inside some, outside the rest. Its label
point is the region's chebyshev center: the interior point of greatest clearance to the
surrounding ellipses, which stays well inside even a non-convex region. Each region is solved
on its own, so only the ones asked for are computed.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numba as nb
import numpy as np
from scipy import optimize as spo

from ._types import BoolArray, F64Array

# below this frame radius the point is the center, where the gradient is undefined
_CENTER_EPS = 1e-12

# readonly params: a writable arg converts to readonly, so one signature takes frozen and fresh
RO1 = nb.types.Array(nb.float64, 1, "A", readonly=True)
RO2 = nb.types.Array(nb.float64, 2, "A", readonly=True)
BO1 = nb.types.Array(nb.boolean, 1, "A", readonly=True)


@nb.njit(nb.float64[:](RO2, nb.float64, nb.float64), cache=True, error_model="numpy")
def _signed_distances(ellipses: F64Array, x: float, y: float) -> F64Array:
    """Signed distance from ``(x, y)`` to each ellipse boundary, positive inside.

    First-order, not exact -- all the clearance search needs.
    """
    # first-order distance to the level set: (1 - rho) / |grad rho|, rho the frame radius
    cx, cy, major, minor, angle = ellipses.T
    dx = x - cx
    dy = y - cy
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    along_major = (cos_a * dx + sin_a * dy) / major
    along_minor = (-sin_a * dx + cos_a * dy) / minor
    rho = np.hypot(along_major, along_minor)
    grad_x = (along_major * cos_a / major - along_minor * sin_a / minor) / rho
    grad_y = (along_major * sin_a / major + along_minor * cos_a / minor) / rho
    distance = (1.0 - rho) / np.hypot(grad_x, grad_y)
    # at the center (rho -> 0) the gradient is undefined; nearest boundary is the minor axis
    return np.where(rho < _CENTER_EPS, np.minimum(major, minor), distance)


@nb.njit(nb.float64(RO1, RO2, BO1), cache=True, error_model="numpy")
def _neg_clearance(point: F64Array, ellipses: F64Array, inside: BoolArray) -> float:
    """Minus the region clearance: min distance inside the in-sets, outside the rest.

    Positive clearance means ``point`` sits inside every in-set and outside every out-set with
    room to spare, so a negative return means the point is in the region.
    """
    x, y = point
    distance = _signed_distances(ellipses, x, y)
    return -np.where(inside, distance, -distance).min()


_EMPTY_SEEDS = np.empty((0, 2), dtype="f8")


def region_center(
    ellipses: F64Array, inside: BoolArray, seeds: F64Array = _EMPTY_SEEDS
) -> tuple[float, float]:
    """Label point for the region inside the flagged sets.

    The region is the area inside every set flagged in ``inside`` and outside every set left
    false; the point returned is its interior spot of greatest clearance to the surrounding
    boundaries. Only this region is solved.

    Parameters
    ----------
    ellipses
        An ``(n, 5)`` layout of canonical ``(cx, cy, major, minor, angle)`` rows, as
        :func:`~provenny.proportional_venn_array` returns.
    inside
        A boolean mask over the ``n`` sets; ``inside[i]`` true places the region inside
        set ``i``, false outside it.
    seeds
        Extra ``(k, 2)`` start points for the clearance search, on top of the sets' centers.
        A crescent region contains none of its sets' centers, so a caller holding the
        region's boundary (:class:`~provenny._zone.Zone`) should pass points on it.

    Returns
    -------
    The ``(x, y)`` label point, inside exactly the flagged sets.

    Raises
    ------
    ValueError
        If no set is flagged (the exterior is not a region), or the flagged region is
        empty -- the sets do not intersect that way, so it has no interior.
    """
    flags = np.asarray(inside, dtype="?")
    if flags.shape != (ellipses.shape[0],):
        raise ValueError(f"inside must have one flag per set ({ellipses.shape[0]})")
    elif not flags.any():
        raise ValueError("no set flagged: the exterior is not a region")
    in_centers = ellipses[flags, :2]

    # derivative-free: the min-of-distances objective is kinked at its optimum, where gradient
    # methods converge to a worse-clearance point -- Nelder-Mead is slower but more robust
    def solves() -> Iterator[tuple[float, float, float]]:
        starts = itertools.chain(in_centers, in_centers.mean(0, keepdims=True), seeds)
        for start in starts:
            res = spo.minimize(
                _neg_clearance, start, args=(ellipses, flags), method="Nelder-Mead"
            )
            yield res.fun, *res.x

    fun, x, y = min(solves())
    if fun >= 0:  # no interior clears every boundary: empty
        raise ValueError(
            f"the region inside sets {np.flatnonzero(flags).tolist()} is empty"
        )
    return (float(x), float(y))
