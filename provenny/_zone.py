r"""Zone boundaries: the arc arrangement of one in/out set combination.

A zone is the region inside a chosen set of ellipses and outside the rest. Its boundary
runs along the ellipses, switching from one to another wherever two cross, so it is a set
of closed loops of :class:`~provenny._shape.Arc`\ s -- a lobe per outer loop, a nested hole
per inner one. This module builds that arrangement, and :class:`Zone` reads the area, the
label point, and the path exports off it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from ._kernel import collapse_coincident, ellipse_crossings
from ._region import region_center
from ._shape import (
    Arc,
    Ellipse,
    Point,
    loops_area,
    loops_matplotlib_path,
    loops_svg_path,
    require_canonical,
    sample_loop,
)
from ._types import BoolArray, F64Array, U8Array

# merge crossing points closer than this (in layout units) into one boundary node
_NODE_TOL = 1e-6


def _full_arcs(ellipses: F64Array) -> list[Arc]:
    """Each ellipse row as a full-turn :class:`Arc`."""
    # no copy: ellipses is the fresh, never-mutated array from collapse_coincident, so the
    # Ellipses can freeze row views into it
    return [Arc(Ellipse(row), 0.0, math.tau) for row in ellipses]


def _borders_zone(  # noqa: PLR0913
    arcs: list[Arc], flags: BoolArray, on_arc: Arc, x: float, y: float, tol: float
) -> bool:
    """Whether a point on ``on_arc``'s ellipse borders the zone: every other ellipse agrees."""
    return all(
        arc is on_arc or arc.ellipse.contains(x, y, tol) == bool(flag)
        for arc, flag in zip(arcs, flags, strict=True)
    )


def _boundary_loops(
    ellipses: F64Array, flags: BoolArray, tol: float
) -> Iterator[list[Arc]]:
    """Yield the zone's boundary as closed loops of arcs, from coincidence-free ellipses."""
    arcs = _full_arcs(ellipses)
    marks: list[list[tuple[float, Point]]] = [[] for _ in arcs]
    # each crossing solved once, shared: the endpoint is byte-identical, so stitches without a merge
    for (ell_a, arc_a, marks_a), (ell_b, arc_b, marks_b) in itertools.combinations(
        zip(ellipses, arcs, marks, strict=True), 2
    ):
        for crossing in ellipse_crossings(ell_a, ell_b, tol=tol):
            crossing_x, crossing_y = crossing
            point = (float(crossing_x), float(crossing_y))
            marks_a.append((arc_a.ellipse.anomaly(*point), point))
            marks_b.append((arc_b.ellipse.anomaly(*point), point))

    open_arcs: list[tuple[Arc, Point, Point]] = []
    for arc, arc_marks, flag in zip(arcs, marks, flags, strict=True):
        if not arc_marks:
            # uncrossed: constant membership, so one test decides the whole boundary (lobe, or hole)
            if _borders_zone(arcs, flags, arc, *arc.ellipse.point_at(0.0), tol):
                yield [arc if flag else replace(arc, start=math.tau, end=0.0)]
            continue
        arc_marks.sort()
        rotated = arc_marks[1:] + arc_marks[:1]
        for (anomaly, point), (next_anomaly, next_point) in zip(
            arc_marks, rotated, strict=True
        ):
            end = next_anomaly if next_anomaly > anomaly else next_anomaly + math.tau
            midpoint = arc.ellipse.point_at((anomaly + end) / 2.0)
            if not _borders_zone(arcs, flags, arc, *midpoint, tol):
                continue
            # orient interior-on-left: ccw where inside, reversed where outside, so arcs chain
            if flag:
                open_arcs.append(
                    (replace(arc, start=anomaly, end=end), point, next_point)
                )
            else:
                open_arcs.append(
                    (replace(arc, start=end, end=anomaly), next_point, point)
                )
    yield from _stitch(open_arcs)


def _stitch(open_arcs: list[tuple[Arc, Point, Point]]) -> Iterator[list[Arc]]:
    """Chain oriented arcs head to tail (at shared crossing points) into closed loops.

    Assumes generic position: each crossing node joins exactly one incoming and one outgoing
    kept arc. A tangency, or three ellipses through one point, raises rather than yielding a
    silently-open boundary.
    """
    nodes: list[Point] = []

    def node_of(point: Point) -> int:
        x, y = point
        for index, (other_x, other_y) in enumerate(nodes):
            if math.hypot(x - other_x, y - other_y) < _NODE_TOL:
                return index
        nodes.append(point)
        return len(nodes) - 1

    ends = [(node_of(start), node_of(end)) for _, start, end in open_arcs]
    outgoing: dict[int, list[int]] = {}
    for edge, (start_node, _) in enumerate(ends):
        outgoing.setdefault(start_node, []).append(edge)

    used = [False] * len(open_arcs)
    for first, (first_start, first_end) in enumerate(ends):
        if used[first]:
            continue
        loop: list[Arc] = []
        edge: int | None = first
        end_node = first_end
        while edge is not None and not used[edge]:
            used[edge] = True
            loop.append(open_arcs[edge][0])
            end_node = ends[edge][1]
            edge = next(
                (nxt for nxt in outgoing.get(end_node, []) if not used[nxt]), None
            )
        if end_node != first_start:
            raise ValueError("zone boundary did not close (degenerate crossing?)")
        yield loop


# not slotted: cached_property needs __dict__, which slots remove; too few Zones to matter
@dataclass(frozen=True, eq=False)
class Zone:
    """A laid-out zone: its boundary ``loops`` and the interior point to label it.

    ``loops`` are the closed arc loops (outer lobes, nested holes). :attr:`area` is exact and
    :attr:`center` is the chebyshev (max-clearance) interior point, always inside even a
    crescent.
    """

    loops: list[list[Arc]]
    _ellipses: F64Array
    _inside: BoolArray

    def __post_init__(self) -> None:
        """Reject a layout that is not one ellipse row per in/out flag."""
        if self._ellipses.shape != (self._inside.size, 5):
            raise ValueError(
                f"ellipses must be one (cx, cy, major, minor, angle) row per flag: "
                f"expected {(self._inside.size, 5)}, got {self._ellipses.shape}"
            )

    @cached_property
    def center(self) -> Point:
        """The chebyshev (max-clearance) label point, computed on first access."""
        # seed from the boundary: a thin crescent holds none of its sets' centers
        boundary = np.concatenate([sample_loop(loop, 8) for loop in self.loops])
        return region_center(self._ellipses, self._inside, seeds=boundary)

    @property
    def area(self) -> float:
        """The exact enclosed area (holes subtracted), via green's theorem on the arcs."""
        return loops_area(self.loops)

    def svg_path(self) -> str:
        """Return the boundary as an svg ``<path>`` d-string."""
        return loops_svg_path(self.loops)

    def matplotlib_path(self) -> tuple[F64Array, U8Array]:
        """Return the boundary as ``(vertices, codes)`` for ``matplotlib.path.Path``."""
        return loops_matplotlib_path(self.loops)

    def sample(self, num: int = 100) -> list[F64Array]:
        """Sample about ``num`` points around each boundary loop."""
        return [sample_loop(loop, num) for loop in self.loops]


def zone(ellipses: F64Array, inside: BoolArray, *, tol: float = 1e-9) -> Zone | None:
    """Build the zone inside the flagged sets and outside the rest, or ``None`` if empty.

    Parameters
    ----------
    ellipses
        An ``(n, 5)`` layout of canonical ``(cx, cy, major, minor, angle)`` rows
        (``major >= minor``, ``angle`` in ``[0, pi)``), as
        :func:`~provenny.proportional_venn_array` returns.
    inside
        A boolean mask over the sets; ``inside[i]`` true keeps the zone inside set ``i``.
    tol
        Numeric tolerance for the boundary tests.

    Returns
    -------
    The zone, or ``None`` when it is empty in this layout.

    Raises
    ------
    ValueError
        If the rows are not canonical ellipses, or no set is flagged -- the exterior is
        not a zone.
    """
    # own the rows: the arcs freeze what they're handed; freezing the caller's array would leak
    rows = np.array(ellipses, dtype="f8")
    if rows.shape[1:] != (5,) or rows.shape[0] == 0:
        raise ValueError("ellipses must have shape (n, 5) with n >= 1")
    require_canonical(rows)  # the same gate Ellipse checks, before any collapse away
    flags = np.asarray(inside, dtype="?")
    if flags.shape != (rows.shape[0],):
        raise ValueError(f"inside must have one flag per set ({rows.shape[0]})")
    elif not flags.any():
        raise ValueError("no set flagged: the exterior is not a zone")
    # identical ellipses have no transversal crossing; merge them for generic position, or bail
    # if a coincident pair is flagged inside one twin and outside the other
    collapsed = collapse_coincident(rows, flags, tol)
    if collapsed is None:
        return None
    loops = [*_boundary_loops(*collapsed, tol)]
    if not loops:
        return None
    return Zone(loops, rows, flags)
