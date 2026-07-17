r"""The canonical ellipse-row convention, the Ellipse and Arc value types, and path exports.

An :class:`Ellipse` wraps one placed shape; an :class:`Arc` is a piece of one, and holds the
ellipse it runs along.

A boundary -- an ellipse outline or a zone border -- is a list of closed loops, each a list of
:class:`Arc`\ s met end to end. Arcs carry the exact geometry; the ``loops_*`` converters turn
them into the cubic-bezier paths the plotting libraries take, to well under a pixel.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ._kernel import arc_green, eccentric_angle, frame, point_at
from ._types import F64Array, U8Array

_QUARTER_TURN = math.pi / 2.0

# matplotlib.path.Path command codes (stable public constants, no import needed)
_MOVETO = 1
_CURVE4 = 4
_CLOSEPOLY = 79

Point = tuple[float, float]
Cubic = tuple[Point, Point, Point, Point]


def canonicalize(ellipses: F64Array) -> F64Array:
    """Put every ``(n, 5)`` ellipse row in the ``major >= minor`` convention.

    Angles come out in ``[0, pi)``. Returns a fresh array; the input is untouched.
    """
    # swap + quarter-turn names the same ellipse; so does the [0, pi) wrap (orientation is pi-periodic)
    rows = np.array(ellipses, dtype="f8")
    swap = rows[:, 3] > rows[:, 2]
    rows[swap, 2:4] = rows[swap, 3:1:-1].copy()
    rows[swap, 4] += math.pi / 2.0
    rows[:, 4] %= math.pi
    return rows


def require_canonical(ellipses: F64Array) -> None:
    """Raise unless every ``(..., 5)`` row of ``ellipses`` is a canonical ellipse.

    Canonical is ``(cx, cy, major, minor, angle)`` with ``major >= minor`` and ``angle`` in
    ``[0, pi)``. Accepts a single ``(5,)`` row or an ``(n, 5)`` array -- the one gate
    :class:`Ellipse` and :func:`~provenny.zone` both check.
    """
    if ellipses.shape[-1] != 5:  # noqa: PLR2004
        raise ValueError("an ellipse is a (5,) row of (cx, cy, major, minor, angle)")
    elif np.any(ellipses[..., 2] < ellipses[..., 3]):
        raise ValueError("ellipse row is not canonical: major (index 2) < minor")
    elif np.any(ellipses[..., 4] < 0.0) or np.any(math.pi <= ellipses[..., 4]):
        raise ValueError("ellipse row is not canonical: angle outside [0, pi)")


@dataclass(frozen=True, slots=True)
class Ellipse:
    """A placed shape wrapping its ``(5,)`` layout row ``[*center, major, minor, angle]``.

    Indexing a :class:`~provenny.Diagram` returns one of these.
    """

    array: F64Array

    def __post_init__(self) -> None:
        """Reject a row that is not a canonical ellipse, then freeze it read-only."""
        require_canonical(self.array)
        self.array.flags.writeable = False

    def __eq__(self, other: object) -> bool:
        """Equal when the wrapped parameter rows are equal."""
        if not isinstance(other, Ellipse):
            return NotImplemented
        return bool(np.array_equal(self.array, other.array))

    def __hash__(self) -> int:
        """Hash the wrapped row's bytes."""
        return hash(self.array.tobytes())

    @property
    def center(self) -> Point:
        """The ``(x, y)`` center."""
        x, y = self.array[:2]
        return (float(x), float(y))

    @property
    def major(self) -> float:
        """The larger semi-axis."""
        return float(self.array[2])

    @property
    def minor(self) -> float:
        """The smaller semi-axis."""
        return float(self.array[3])

    @property
    def angle(self) -> float:
        """The major axis's orientation from the x-axis, in radians, in ``[0, pi)``."""
        return float(self.array[4])

    @property
    def area(self) -> float:
        """The enclosed area, ``pi * major * minor``."""
        return math.pi * self.major * self.minor

    def point_at(self, anomaly: float) -> Point:
        """Return the ``(x, y)`` boundary point at eccentric ``anomaly``."""
        x, y = point_at(self.array, anomaly)
        return (float(x), float(y))

    def frame(self, x: float, y: float) -> Point:
        """``(x, y)`` in the ellipse's axis frame, scaled by the semi-axes (1 on boundary)."""
        along_major, along_minor = frame(self.array, x, y)
        return (float(along_major), float(along_minor))

    def contains(self, x: float, y: float, tol: float = 0.0) -> bool:
        """Whether ``(x, y)`` lies within the closed ellipse."""
        along_major, along_minor = self.frame(x, y)
        return along_major * along_major + along_minor * along_minor <= 1.0 + tol

    def anomaly(self, x: float, y: float) -> float:
        """Return the eccentric anomaly of a boundary point ``(x, y)``, in ``[0, tau)``."""
        return float(eccentric_angle(self.array, x, y)) % math.tau

    def _loops(self) -> Loops:
        """Return the outline as arc loops: a single loop of the one full-ellipse arc."""
        return ((Arc(self, 0.0, math.tau),),)

    def svg_path(self) -> str:
        """Return the outline as an svg ``<path>`` d-string."""
        return loops_svg_path(self._loops())

    def matplotlib_path(self) -> tuple[F64Array, U8Array]:
        """Return the outline as ``(vertices, codes)`` for ``matplotlib.path.Path``."""
        return loops_matplotlib_path(self._loops())

    def sample(self, num: int = 100) -> F64Array:
        """Sample about ``num`` ``(x, y)`` points evenly around the outline."""
        (loop,) = self._loops()
        return sample_loop(loop, num)


@dataclass(frozen=True, slots=True)
class Arc:
    """A piece of a placed :class:`Ellipse`, swept over eccentric anomaly ``[start, end]``.

    The angles are in the ellipse's own frame; their order carries direction, so ``end >
    start`` sweeps counter-clockwise and a full ellipse is one arc with ``end - start == 2*pi``.
    """

    ellipse: Ellipse
    start: float
    end: float

    def split(self, max_sweep: float = _QUARTER_TURN) -> Iterator[Arc]:
        """Yield sub-arcs each sweeping at most ``max_sweep`` (default a quarter turn).

        At most four for a full ellipse, one for anything already within ``max_sweep``.
        """
        # a quarter turn keeps one cubic bezier per sub-arc within ~1e-4 of the radius
        sweep = self.end - self.start
        count = max(1, math.ceil(abs(sweep) / max_sweep))
        step = sweep / count
        for index in range(count):
            yield replace(
                self,
                start=self.start + index * step,
                end=self.start + (index + 1) * step,
            )

    def bezier(self) -> Cubic:
        """Return the four cubic-bezier control points approximating this arc.

        Accurate to ~1e-4 of the radius over a quarter turn, so call it on the pieces from
        :meth:`split` rather than on a whole ellipse.
        """
        # the exact affine image of the standard circular-arc bezier
        major = self.ellipse.major
        minor = self.ellipse.minor
        cos_a = math.cos(self.ellipse.angle)
        sin_a = math.sin(self.ellipse.angle)
        center_x, center_y = self.ellipse.center

        def world(local_x: float, local_y: float) -> Point:
            return (
                center_x + local_x * cos_a - local_y * sin_a,
                center_y + local_x * sin_a + local_y * cos_a,
            )

        offset = (4.0 / 3.0) * math.tan((self.end - self.start) / 4.0)
        cos0 = math.cos(self.start)
        sin0 = math.sin(self.start)
        cos1 = math.cos(self.end)
        sin1 = math.sin(self.end)
        return (
            world(major * cos0, minor * sin0),
            world(major * (cos0 - offset * sin0), minor * (sin0 + offset * cos0)),
            world(major * (cos1 + offset * sin1), minor * (sin1 - offset * cos1)),
            world(major * cos1, minor * sin1),
        )


Loops = Sequence[Sequence[Arc]]


def _bezier_segments(loop: Iterable[Arc]) -> Iterator[Cubic]:
    """Yield the cubic bezier control quadruples tracing one closed loop of arcs."""
    for arc in loop:
        for piece in arc.split():
            yield piece.bezier()


def _svg_pieces(loops: Loops) -> Iterator[str]:
    """Yield the svg ``d``-string tokens (one move, cubics, and a close per loop)."""
    for loop in loops:
        first = True
        for (x0, y0), (x1, y1), (x2, y2), (x3, y3) in _bezier_segments(loop):
            if first:
                first = False
                yield f"M{x0:.10g},{y0:.10g}"
            yield f"C{x1:.10g},{y1:.10g} {x2:.10g},{y2:.10g} {x3:.10g},{y3:.10g}"
        yield "Z"


def loops_svg_path(loops: Loops) -> str:
    """Build an svg ``<path>`` ``d`` string (cubic Beziers) for the boundary.

    Disconnected loops and holes become separate subpaths, filled even-odd.
    """
    return "".join(_svg_pieces(loops))


def loops_matplotlib_path(loops: Loops) -> tuple[F64Array, U8Array]:
    """``(vertices, codes)`` for ``matplotlib.path.Path`` (cubic Beziers).

    Feed straight to ``Path(vertices, codes)``; multiple loops fill even-odd.
    """
    vertices: list[Point] = []
    codes: list[int] = []
    for loop in loops:
        start = (0.0, 0.0)
        first = True
        for p0, p1, p2, p3 in _bezier_segments(loop):
            if first:
                first = False
                start = p0
                vertices.append(p0)
                codes.append(_MOVETO)
            vertices.extend((p1, p2, p3))
            codes.extend((_CURVE4, _CURVE4, _CURVE4))
        vertices.append(start)  # CLOSEPOLY convention: repeat the subpath's start
        codes.append(_CLOSEPOLY)
    return np.array(vertices, dtype="f8"), np.array(codes, dtype="u1")


def sample_loop(loop: Sequence[Arc], num: int) -> F64Array:
    """Sample about ``num`` points around one closed loop, spaced by arc sweep."""
    total_sweep = sum(abs(arc.end - arc.start) for arc in loop)
    points: list[Point] = []
    for arc in loop:
        count = max(1, round(num * abs(arc.end - arc.start) / total_sweep))
        for step in range(count):
            points.append(
                arc.ellipse.point_at(arc.start + (arc.end - arc.start) * step / count)
            )
    return np.array(points, dtype="f8")


def loops_area(loops: Loops) -> float:
    """Signed area enclosed by the boundary loops."""
    # green's theorem: each arc contributes ``integral of (1/2)(x dy - y dx)`` in closed form
    return float(
        sum(
            arc_green(arc.ellipse.array, arc.start, arc.end)
            for loop in loops
            for arc in loop
        )
    )
