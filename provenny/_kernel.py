"""numba geometry kernel: exact ellipse-intersection area and disconnection penalty.

``@njit`` (nopython) code, called thousands of times per solve. Two quantities, each as a
value-only core and an analytic-gradient one: the intersection area (:func:`area_core`,
:func:`area_grad_core`) and the disconnection penalty (:func:`penalty_core`,
:func:`penalty_grad_core`). Both work through an ellipse's homogeneous quadratic-form matrix
and the pencil-based intersection of two such forms.

Nothing here is exported, and nothing here validates: rows arriving here have already been
checked at the package's entry points (:func:`~provenny.proportional_venn_array`,
:func:`~provenny.zone`) or built by the solver.
"""

from __future__ import annotations

import cmath
import math

import numba as nb
import numpy as np

from ._types import (
    BoolArray,
    C128Array,
    CMat3,
    CVec3,
    F64Array,
    I64Array,
    Mat3,
)

# readonly params: a writable arg converts to readonly, so one signature takes both. Kernels
# only read their array inputs and write fresh arrays, so returns stay writable.
RO1 = nb.types.Array(nb.float64, 1, "A", readonly=True)
RO2 = nb.types.Array(nb.float64, 2, "A", readonly=True)
RO3 = nb.types.Array(nb.float64, 3, "A", readonly=True)
BO1 = nb.types.Array(nb.boolean, 1, "A", readonly=True)

# 3-vectors and 3x3 forms are tuples, not arrays: a tuple is registers, an array a heap alloc
# (~30x the arithmetic here). Only arrays that outlive a call (a layout, an output buffer) stay arrays.
F3 = nb.types.UniTuple(nb.float64, 3)
M3 = nb.types.UniTuple(F3, 3)
C3 = nb.types.UniTuple(nb.complex128, 3)
CM3 = nb.types.UniTuple(C3, 3)
PAIR3 = nb.types.UniTuple(C3, 2)

PI = math.pi
# node-merge distance. The area kernel dedups tightly: pencil copies agree to machine precision,
# and genuinely-distinct close crossings must be kept for exactness.
_DEDUP_TOL = 1e-9
# the crossing/penalty arrangement dedups loosely: a near-singular pencil spreads copies, and
# fusing them beats resolving two crossings a hair apart.
_CROSSING_TOL = 1e-4
# merge nodes within this anomaly. Per-pair point-space dedup runs first, so this only fuses
# a concurrent point reached via different pairs, never two distinct crossings.
_ARC_NODE_TOL = 1e-9


@nb.njit(nb.float64(nb.float64), cache=True, error_model="numpy")
def _real_cbrt(x: float) -> float:
    """Sign-preserving real cube root."""
    if x >= 0.0:
        return x ** (1.0 / 3.0)
    else:
        return -((-x) ** (1.0 / 3.0))


@nb.njit(nb.complex128(nb.complex128), cache=True, error_model="numpy")
def _complex_cbrt(z: complex) -> complex:
    """Principal complex cube root."""
    magnitude = abs(z)
    if magnitude == 0.0:
        return complex(0.0, 0.0)
    third = math.atan2(z.imag, z.real) / 3.0
    root_mag = magnitude ** (1.0 / 3.0)
    return complex(root_mag * math.cos(third), root_mag * math.sin(third))


@nb.njit(
    nb.float64(*(nb.float64,) * 9), cache=True, error_model="numpy", inline="always"
)
def _det3_entries(  # noqa: PLR0913
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    g: float,
    h: float,
    i: float,
) -> float:
    """Expand the 3x3 determinant from its entries, in row-major order."""
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


@nb.njit(nb.float64(M3), cache=True, error_model="numpy", inline="always")
def _det3(matrix: Mat3) -> float:
    """Expand the 3x3 determinant of a matrix."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return _det3_entries(a, b, c, d, e, f, g, h, i)


@nb.njit(CM3(M3), cache=True, error_model="numpy", inline="always")
def _complex3(matrix: Mat3) -> CMat3:
    """Widen a real 3x3 form to complex."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return (
        (complex(a, 0.0), complex(b, 0.0), complex(c, 0.0)),
        (complex(d, 0.0), complex(e, 0.0), complex(f, 0.0)),
        (complex(g, 0.0), complex(h, 0.0), complex(i, 0.0)),
    )


@nb.njit(CM3(CM3), cache=True, error_model="numpy", inline="always")
def _transpose3(matrix: CMat3) -> CMat3:
    """Transpose a complex 3x3."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return ((a, d, g), (b, e, h), (c, f, i))


@nb.njit(M3(RO3, nb.int64), cache=True, error_model="numpy", inline="always")
def _form_at(forms: F64Array, index: int) -> Mat3:
    """Read form ``index`` out of an ``(n, 3, 3)`` stack as a tuple."""
    return (
        (forms[index, 0, 0], forms[index, 0, 1], forms[index, 0, 2]),
        (forms[index, 1, 0], forms[index, 1, 1], forms[index, 1, 2]),
        (forms[index, 2, 0], forms[index, 2, 1], forms[index, 2, 2]),
    )


@nb.njit(
    M3(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
)
def _ellipse_form(
    cx: float, cy: float, major: float, minor: float, angle: float
) -> Mat3:
    """Build the homogeneous quadratic-form matrix of an ellipse, normalized to det -1."""
    sin = math.sin(angle)
    cos = math.cos(angle)
    maj2 = major * major
    min2 = minor * minor
    a = maj2 * sin * sin + min2 * cos * cos
    b = (min2 - maj2) * sin * cos
    c = maj2 * cos * cos + min2 * sin * sin
    d = -a * cx - b * cy
    e = -b * cx - c * cy
    f = a * cx * cx + 2.0 * b * cx * cy + c * cy * cy - maj2 * min2
    scale = _real_cbrt(-1.0 / _det3(((a, b, d), (b, c, e), (d, e, f))))
    return (
        (a * scale, b * scale, d * scale),
        (b * scale, c * scale, e * scale),
        (d * scale, e * scale, f * scale),
    )


@nb.njit(
    nb.float64(M3, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
    inline="always",
)
def _evaluate(form: Mat3, x: float, y: float) -> float:
    """Evaluate the quadratic form ``p^T M p`` at the point ``(x, y, 1)``."""
    # not ``p @ form @ p``: that heap-allocates p and hits blas twice for a 3x3, 143x slower
    (a, b, c), (d, e, f), (g, h, i) = form
    return x * (a * x + b * y + c) + y * (d * x + e * y + f) + (g * x + h * y + i)


@nb.njit(
    nb.complex128(nb.complex128, nb.complex128, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
    inline="always",
)
def _cardano_root(cube: complex, unit: complex, d0: float, b: float) -> complex:
    """One root of the depressed cubic, from the cube root turned by a cube-root of unity."""
    shifted = cube * unit
    if abs(shifted) == 0.0:
        return complex(b, 0.0)
    else:
        return complex(b, 0.0) + shifted + complex(d0, 0.0) / shifted


@nb.njit(
    C3(nb.float64, nb.float64, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
)
def _cubic_roots(a: float, b: float, c: float, d: float) -> CVec3:
    """Solve ``a x^3 + b x^2 + c x + d`` for its three complex roots, by cardano's method."""
    d0 = b * b - 3.0 * a * c
    d1 = 2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d
    discriminant = cmath.sqrt(complex(d1 * d1 - 4.0 * d0 * d0 * d0, 0.0))
    lo = complex(d1, 0.0) - discriminant
    hi = complex(d1, 0.0) + discriminant
    # take the larger-magnitude branch so the cube root stays well conditioned
    if abs(lo) >= abs(hi):
        cube = _complex_cbrt(lo / 2.0)
    else:
        cube = _complex_cbrt(hi / 2.0)
    omega = complex(-0.5, math.sqrt(3.0) / 2.0)
    inv = complex(-1.0 / (3.0 * a), 0.0)
    return (
        inv * _cardano_root(cube, complex(1.0, 0.0), d0, b),
        inv * _cardano_root(cube, omega, d0, b),
        inv * _cardano_root(cube, omega * omega, d0, b),
    )


@nb.njit(CM3(C3), cache=True, error_model="numpy", inline="always")
def _skew3(v: CVec3) -> CMat3:
    """Build the skew matrix ``S`` with ``S u = v x u`` (rows are ``v x e_i``)."""
    x, y, z = v
    zero = complex(0.0, 0.0)
    return ((zero, z, -y), (-z, zero, x), (y, -x, zero))


@nb.njit(CM3(CM3, CM3), cache=True, error_model="numpy", inline="always")
def _cmatmul3(a: CMat3, b: CMat3) -> CMat3:
    """Multiply two complex 3x3 matrices."""
    (a0, a1, a2), (a3, a4, a5), (a6, a7, a8) = a
    (b0, b1, b2), (b3, b4, b5), (b6, b7, b8) = b
    return (
        (
            a0 * b0 + a1 * b3 + a2 * b6,
            a0 * b1 + a1 * b4 + a2 * b7,
            a0 * b2 + a1 * b5 + a2 * b8,
        ),
        (
            a3 * b0 + a4 * b3 + a5 * b6,
            a3 * b1 + a4 * b4 + a5 * b7,
            a3 * b2 + a4 * b5 + a5 * b8,
        ),
        (
            a6 * b0 + a7 * b3 + a8 * b6,
            a6 * b1 + a7 * b4 + a8 * b7,
            a6 * b2 + a7 * b5 + a8 * b8,
        ),
    )


@nb.njit(nb.int64(C3), cache=True, error_model="numpy", inline="always")
def _vec_argmax_abs3(v: CVec3) -> int:
    """Index of the largest-magnitude entry of a complex 3-vector."""
    x, y, z = v
    a = abs(x)
    b = abs(y)
    c = abs(z)
    if a >= b and a >= c:
        return 0
    elif b >= c:
        return 1
    else:
        return 2


@nb.njit(
    nb.types.UniTuple(nb.int64, 2)(CM3),
    cache=True,
    error_model="numpy",
    inline="always",
)
def _mat_argmax_abs3(matrix: CMat3) -> tuple[int, int]:
    """Row and column of the largest-magnitude entry of a complex 3x3."""
    row = 0
    col = 0
    best = abs(matrix[0][0])
    for i in range(3):
        for j in range(3):
            mag = abs(matrix[i][j])
            if mag > best:
                best = mag
                row = i
                col = j
    return row, col


@nb.njit(CM3(CM3), cache=True, error_model="numpy", inline="always")
def _adjugate3(m: CMat3) -> CMat3:
    """Expand the adjugate (transposed cofactor matrix) of a complex 3x3."""
    (a, b, c), (d, e, f), (g, h, i) = m
    return (
        (e * i - f * h, h * c - i * b, b * f - c * e),
        (f * g - d * i, i * a - g * c, c * d - a * f),
        (d * h - e * g, g * b - h * a, a * e - b * d),
    )


# flat indices of the 2x2 minor omitting row/col k, for line-argmax k
_MINOR_INDS = ((4, 5, 7, 8), (0, 2, 6, 8), (0, 1, 3, 4))


@nb.njit(PAIR3(M3, C3), cache=True, error_model="numpy")
def _intersect_line(form: Mat3, line: CVec3) -> tuple[CVec3, CVec3]:
    """Intersect a form with a complex line, giving the two complex homogeneous crossings."""
    skew = _skew3(line)
    # the degenerate (rank-two) form S^T M S encoding the two line-form crossings
    point_pair = _cmatmul3(_cmatmul3(_transpose3(skew), _complex3(form)), skew)
    ind = _vec_argmax_abs3(line)
    (p0, p1, p2), (p3, p4, p5), (p6, p7, p8) = point_pair
    flat = (p0, p1, p2, p3, p4, p5, p6, p7, p8)
    i0, i1, i2, i3 = _MINOR_INDS[ind]
    minor_det = flat[i0] * flat[i3] - flat[i1] * flat[i2]
    # zero strongest component = degenerate line; keep mult at 0 (numba raises on complex /0),
    # the caller's imaginary-part and on-boundary filters drop the spurious points
    if abs(line[ind]) == 0.0:
        mult = complex(0.0, 0.0)
    else:
        mult = cmath.sqrt(-minor_det) / line[ind]
    (s0, s1, s2), (s3, s4, s5), (s6, s7, s8) = skew
    rank_one = (
        (p0 + s0 * mult, p1 + s1 * mult, p2 + s2 * mult),
        (p3 + s3 * mult, p4 + s4 * mult, p5 + s5 * mult),
        (p6 + s6 * mult, p7 + s7 * mult, p8 + s8 * mult),
    )
    row, col = _mat_argmax_abs3(rank_one)
    return (
        rank_one[row],
        (rank_one[0][col], rank_one[1][col], rank_one[2][col]),
    )


@nb.njit(nb.float64(M3, M3, M3), cache=True, error_model="numpy", inline="always")
def _bracket(one: Mat3, two: Mat3, three: Mat3) -> float:
    """Mixed determinant with column ``k`` taken from the ``k``-th argument.

    The polarization term of ``det(x*L + R)``.
    """
    # straight into the determinant: a mixed scratch matrix would be a heap allocation
    a = one[0][0]
    d = one[1][0]
    g = one[2][0]
    b = two[0][1]
    e = two[1][1]
    h = two[2][1]
    c = three[0][2]
    f = three[1][2]
    i = three[2][2]
    return _det3_entries(a, b, c, d, e, f, g, h, i)


@nb.njit(nb.complex128[:, ::1](M3, M3), cache=True, error_model="numpy")
def _intersect(left: Mat3, right: Mat3) -> C128Array:
    """Up to twelve candidate crossings of two forms, via their pencil."""
    alpha = _det3(left)
    beta = (
        _bracket(left, left, right)
        + _bracket(left, right, left)
        + _bracket(right, left, left)
    )
    gamma = (
        _bracket(left, right, right)
        + _bracket(right, left, right)
        + _bracket(right, right, left)
    )
    delta = _det3(right)

    (l0, l1, l2), (l3, l4, l5), (l6, l7, l8) = _complex3(left)
    (r0, r1, r2), (r3, r4, r5), (r6, r7, r8) = _complex3(right)
    out = np.empty((12, 3), "c16")
    count = 0
    for root in _cubic_roots(alpha, beta, gamma, delta):
        # degenerate pencil member: a pair of lines
        member = (
            (l0 * root + r0, l1 * root + r1, l2 * root + r2),
            (l3 * root + r3, l4 * root + r4, l5 * root + r5),
            (l6 * root + r6, l7 * root + r7, l8 * root + r8),
        )
        # its adjugate is rank one; its strongest diagonal locates the line crossing
        adj = _adjugate3(member)
        ind = _vec_argmax_abs3((adj[0][0], adj[1][1], adj[2][2]))
        norm = cmath.sqrt(-adj[ind][ind])
        # zero-diagonal adjugate = coincident lines; skip (other roots recover the crossings).
        # numba raises on complex /0 even under the numpy error model, so guard it
        if abs(norm) == 0.0:
            continue
        (a0, a1, a2), (a3, a4, a5), (a6, a7, a8) = _skew3(
            (adj[0][ind] / norm, adj[1][ind] / norm, adj[2][ind] / norm)
        )
        (m0, m1, m2), (m3, m4, m5), (m6, m7, m8) = member
        split = (
            (m0 + a0, m1 + a1, m2 + a2),
            (m3 + a3, m4 + a4, m5 + a5),
            (m6 + a6, m7 + a7, m8 + a8),
        )
        # the two lines are the strongest row and column of the split matrix
        row, col = _mat_argmax_abs3(split)
        lines = (split[row], (split[0][col], split[1][col], split[2][col]))
        for line in lines:
            for point in _intersect_line(left, line):
                out[count, 0], out[count, 1], out[count, 2] = point
                count += 1
    return out[:count]


@nb.njit(
    nb.boolean(RO3, nb.int64, nb.float64, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
)
def _inside_all(forms: F64Array, count: int, x: float, y: float, tol: float) -> bool:
    for i in range(count):
        if _evaluate(_form_at(forms, i), x, y) > tol:
            return False
    return True


@nb.njit(nb.float64(nb.float64, nb.float64), cache=True, error_model="numpy")
def _pymod(a: float, b: float) -> float:
    """Python-style modulo: result lands in ``[0, b)`` for positive ``b``."""
    return ((a % b) + b) % b


@nb.njit(
    nb.int64(nb.float64[:, ::1], nb.int64, nb.float64), cache=True, error_model="numpy"
)
def _dedup_points(points: F64Array, count: int, tol: float) -> int:
    """Compact points within ``tol`` (per coordinate) of an earlier kept one to the front.

    Mutates ``points``; the caller reads ``points[:kept]``.
    """
    # in place, no alloc in the hot loop; safe: each slot written was already read
    kept = 0
    for i in range(count):
        px = points[i, 0]
        py = points[i, 1]
        dup = False
        for k in range(kept):
            if abs(points[k, 0] - px) <= tol and abs(points[k, 1] - py) <= tol:
                dup = True
                break
        if not dup:
            points[kept, 0] = px
            points[kept, 1] = py
            kept += 1
    return kept


@nb.njit(
    nb.types.UniTuple(nb.float64, 2)(RO1, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
)
def frame(ellipse: F64Array, px: float, py: float) -> tuple[float, float]:
    """Coordinates of a point in the ellipse's axis frame, scaled by the semi-axes."""
    cx, cy, major, minor, angle = ellipse
    dx = px - cx
    dy = py - cy
    sin = math.sin(-angle)
    cos = math.cos(-angle)
    return (cos * dx - sin * dy) / major, (sin * dx + cos * dy) / minor


@nb.njit(nb.boolean(RO1, nb.float64, nb.float64), cache=True, error_model="numpy")
def _contains(ellipse: F64Array, px: float, py: float) -> bool:
    along_major, along_minor = frame(ellipse, px, py)
    return along_major * along_major + along_minor * along_minor <= 1.0


@nb.njit(nb.float64(RO1, nb.float64, nb.float64), cache=True, error_model="numpy")
def eccentric_angle(ellipse: F64Array, px: float, py: float) -> float:
    along_major, along_minor = frame(ellipse, px, py)
    return math.atan2(along_minor, along_major)


@nb.njit(
    nb.types.UniTuple(nb.float64, 2)(RO1, nb.float64), cache=True, error_model="numpy"
)
def point_at(ellipse: F64Array, angle: float) -> tuple[float, float]:
    """Return the boundary point at eccentric anomaly ``angle``."""
    cx, cy, major, minor, rotation = ellipse
    sin_r = math.sin(rotation)
    cos_r = math.cos(rotation)
    mc = major * math.cos(angle)
    ms = minor * math.sin(angle)
    return cx + mc * cos_r - ms * sin_r, cy + mc * sin_r + ms * cos_r


@nb.njit(nb.float64(RO1, nb.float64, nb.float64), cache=True, error_model="numpy")
def arc_green(ellipse: F64Array, start: float, end: float) -> float:
    """Contribution of the arc ``start -> end`` to ``(1/2) integral of (x dy - y dx)``."""
    x0, y0 = point_at(ellipse, start)
    x1, y1 = point_at(ellipse, end)
    cx, cy, major, minor, _ = ellipse
    return 0.5 * (major * minor * (end - start) + cx * (y1 - y0) - cy * (x1 - x0))


@nb.njit(
    nb.void(nb.float64[::1], RO1, nb.float64, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
)
def _arc_shape_grad(
    grad_row: F64Array, ellipse: F64Array, start: float, end: float, weight: float
) -> None:
    """Add ``weight`` times the arc ``start -> end``'s shape derivative into ``grad_row``.

    ``weight`` carries the outer chain-rule factor: 1 for the intersection area, the penalty
    factor and lobe sign for the disconnection penalty.
    """
    # boundary integral ``integral of (V . n) ds``, V = d point / d param. Crossing-motion
    # terms cancel over a closed loop, leaving only the arc's endpoints and its ellipse's params.
    major = ellipse[2]
    minor = ellipse[3]
    x0, y0 = point_at(ellipse, start)
    x1, y1 = point_at(ellipse, end)
    half_sweep = (end - start) / 2.0
    quarter_sin = (math.sin(2.0 * end) - math.sin(2.0 * start)) / 4.0
    grad_row[0] += weight * (y1 - y0)
    grad_row[1] += weight * (x0 - x1)
    grad_row[2] += weight * minor * (half_sweep + quarter_sin)
    grad_row[3] += weight * major * (half_sweep - quarter_sin)
    grad_row[4] += weight * (
        -0.5
        * (
            major * major * (math.cos(end) ** 2 - math.cos(start) ** 2)
            + minor * minor * (math.sin(end) ** 2 - math.sin(start) ** 2)
        )
    )


@nb.njit(nb.boolean(M3, M3, nb.float64), cache=True, error_model="numpy")
def _same_form(left: Mat3, right: Mat3, tol: float) -> bool:
    """Whether two 3x3 ellipse forms coincide entrywise within ``tol`` (same ellipse)."""
    for row in range(3):
        for col in range(3):
            if abs(left[row][col] - right[row][col]) > tol:
                return False
    return True


@nb.njit(
    nb.float64(RO3, RO2, nb.int64[::1], nb.int64, nb.float64, nb.float64[:, ::1]),
    cache=True,
    error_model="numpy",
)
def _containment_area(  # noqa: PLR0913
    forms: F64Array,
    kept: F64Array,
    keep: I64Array,
    m: int,
    tol: float,
    grad: F64Array,
) -> float:
    """Return the smallest ellipse's area if its center is inside every ellipse, else zero.

    Writes that ellipse's own ``d(pi a b)`` into ``grad``, at its original row (``keep`` maps
    back to it).
    """
    # no clean lens: one ellipse is inside all the others (containment, or an unresolvable
    # near-coincident stack)
    best_area = np.inf
    best_idx = 0
    for i in range(m):
        area = PI * kept[i, 2] * kept[i, 3]
        if area < best_area:
            best_area = area
            best_idx = i
    cx, cy, major, minor, _ = kept[best_idx]
    if _inside_all(forms, m, cx, cy, tol):
        original = keep[best_idx]
        grad[original, 2] = PI * minor
        grad[original, 3] = PI * major
        return best_area
    else:
        return 0.0


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[:, ::1]))(RO2, nb.float64, nb.boolean),
    cache=True,
    error_model="numpy",
)
def area_grad_core(  # noqa: PLR0912, PLR0915
    ellipses: F64Array, tol: float, want_grad: bool
) -> tuple[float, F64Array]:
    """Compute the intersection area, and (when ``want_grad``) its gradient per set.

    The gradient is ``d(area)/d(cx,cy,major,minor,angle)``, one row per ellipse, aligned with
    ``ellipses``; a coincidence-dropped duplicate keeps a zero row. ``want_grad`` false skips
    the per-arc shape-derivative work.
    """
    n = ellipses.shape[0]
    grad = np.zeros((n, 5), "f8")
    # recenter so the absolute-tolerance tests stay meaningful far from the origin
    shift_x = 0.0
    shift_y = 0.0
    for i in range(n):
        shift_x += ellipses[i, 0]
        shift_y += ellipses[i, 1]
    shift_x /= n
    shift_y /= n

    all_forms = np.empty((n, 3, 3), "f8")
    recentered = np.empty((n, 5), "f8")
    for i in range(n):
        cx, cy, major, minor, angle = ellipses[i]
        cx -= shift_x
        cy -= shift_y
        recentered[i, 0] = cx
        recentered[i, 1] = cy
        recentered[i, 2] = major
        recentered[i, 3] = minor
        recentered[i, 4] = angle
        form = _ellipse_form(cx, cy, major, minor, angle)
        for row in range(3):
            for col in range(3):
                all_forms[i, row, col] = form[row][col]

    # drop forms coinciding with an earlier one (no constraint). Same predicate as same_ellipse,
    # so the two never disagree.
    keep = np.empty(n, "i8")
    m = 0
    for i in range(n):
        coincident = False
        for kk in range(m):
            if _same_form(_form_at(all_forms, keep[kk]), _form_at(all_forms, i), tol):
                coincident = True
                break
        if not coincident:
            keep[m] = i
            m += 1

    forms = np.empty((m, 3, 3), "f8")
    kept_ellipses = np.empty((m, 5), "f8")
    for i in range(m):
        forms[i] = all_forms[keep[i]]
        kept_ellipses[i] = recentered[keep[i]]

    # mark each boundary at its crossings that lie inside all ellipses (bordering the
    # intersection), on both ellipses. The pencil emits each crossing up to 3x, so dedup per
    # pair in point space first: bounds each row at 4 and makes ncross count distinct crossings.
    two_pi = 2.0 * PI
    marks = np.empty((m, 4 * m), "f8")
    mcount = np.zeros(m, "i8")
    ncross = 0
    pair_points = np.empty((12, 2), "f8")
    for i in range(m):
        for j in range(i):
            candidates = _intersect(_form_at(forms, i), _form_at(forms, j))
            found = 0
            for c in range(candidates.shape[0]):
                if (
                    abs(candidates[c, 0].imag) <= tol
                    and abs(candidates[c, 1].imag) <= tol
                    and abs(candidates[c, 2].imag) <= tol
                ):
                    w = candidates[c, 2].real
                    x = candidates[c, 0].real / w
                    y = candidates[c, 1].real / w
                    if _inside_all(forms, m, x, y, tol):
                        pair_points[found, 0] = x
                        pair_points[found, 1] = y
                        found += 1
            # <= 4 crossings per pair; cap so a near-tangent pencil's spread duplicates can't
            # overrun a marks row (width 4*m)
            found = min(_dedup_points(pair_points, found, _DEDUP_TOL), 4)
            for c in range(found):
                x, y = pair_points[c]
                marks[i, mcount[i]] = eccentric_angle(kept_ellipses[i], x, y)
                mcount[i] += 1
                marks[j, mcount[j]] = eccentric_angle(kept_ellipses[j], x, y)
                mcount[j] += 1
                ncross += 1

    # < 2 crossings = no lens: the smallest ellipse (if its center is inside all) is the whole intersection
    if ncross < 2:  # noqa: PLR2004
        return _containment_area(forms, kept_ellipses, keep, m, tol, grad), grad

    # convex intersection = one arc loop; green's theorem over the kept arcs (interior on the
    # left) gives the area, each arc's shape derivative onto its ellipse's gradient row
    total = 0.0
    nodes: F64Array = np.empty(4 * m, "f8")  # reused across the ellipses
    for i in range(m):
        angle_count = int(mcount[i])
        if angle_count == 0:
            # uncrossed: its whole boundary borders the intersection, or none of it
            px, py = point_at(kept_ellipses[i], 0.0)
            if _inside_all(forms, m, px, py, tol):
                total += arc_green(kept_ellipses[i], 0.0, two_pi)
                if want_grad:
                    _arc_shape_grad(grad[keep[i]], kept_ellipses[i], 0.0, two_pi, 1.0)
            continue
        marks[i, :angle_count].sort()  # in place; the row is not read again
        # fuse concurrent-point copies into one node, including one split across the +/-pi seam
        node_count = 0
        for idx in range(angle_count):
            angle = marks[i, idx]
            if node_count == 0 or angle - nodes[node_count - 1] > _ARC_NODE_TOL:
                nodes[node_count] = angle
                node_count += 1
        if (
            node_count > 1
            and nodes[0] + two_pi - nodes[node_count - 1] <= _ARC_NODE_TOL
        ):
            node_count -= 1  # the last node wraps onto the first across the seam
        for idx in range(node_count):
            start = nodes[idx]
            end = nodes[(idx + 1) % node_count]
            if end <= start:
                end += two_pi
            mid_x, mid_y = point_at(kept_ellipses[i], (start + end) / 2.0)
            if _inside_all(forms, m, mid_x, mid_y, tol):
                total += arc_green(kept_ellipses[i], start, end)
                if want_grad:
                    _arc_shape_grad(grad[keep[i]], kept_ellipses[i], start, end, 1.0)

    # near-coincident boundaries can make the arc arithmetic inf/nan; fall back to the
    # smallest (near-contained) ellipse
    if not math.isfinite(total):
        grad = np.zeros((n, 5), "f8")
        return _containment_area(forms, kept_ellipses, keep, m, tol, grad), grad
    else:
        return total, grad


@nb.njit(nb.float64(RO2, nb.float64), cache=True, error_model="numpy")
def area_core(ellipses: F64Array, tol: float) -> float:
    """Exact area of the region common to every one of the ``(n, 5)`` ellipses.

    ``tol`` is the tolerance of the interior tests. For the gradient too, call
    ``area_grad_core``.
    """
    area, _ = area_grad_core(ellipses, tol, False)
    return area


@nb.njit(
    nb.types.Tuple((nb.float64[:, :], nb.int64))(RO1, RO1, nb.float64),
    cache=True,
    error_model="numpy",
)
def _crossings(left: F64Array, right: F64Array, tol: float) -> tuple[F64Array, int]:
    """Distinct real boundary crossings of two ellipses."""
    left_cx, left_cy, left_major, left_minor, left_angle = left
    right_cx, right_cy, right_major, right_minor, right_angle = right
    left_form = _ellipse_form(left_cx, left_cy, left_major, left_minor, left_angle)
    right_form = _ellipse_form(
        right_cx, right_cy, right_major, right_minor, right_angle
    )
    candidates = _intersect(left_form, right_form)
    buffer = np.empty((candidates.shape[0], 2), "f8")
    found = 0
    for c in range(candidates.shape[0]):
        if (
            abs(candidates[c, 0].imag) <= tol
            and abs(candidates[c, 1].imag) <= tol
            and abs(candidates[c, 2].imag) <= tol
        ):
            w = candidates[c, 2].real
            buffer[found, 0] = candidates[c, 0].real / w
            buffer[found, 1] = candidates[c, 1].real / w
            found += 1
    kept = _dedup_points(buffer, found, _CROSSING_TOL)
    return buffer[:kept], kept


@nb.njit(
    nb.float64(
        RO1, RO1, nb.float64[::1], nb.float64[::1], nb.float64, nb.float64, nb.boolean
    ),
    cache=True,
    error_model="numpy",
)
def _smaller_lobe_grad(  # noqa: PLR0912, PLR0913, PLR0915
    left: F64Array,
    right: F64Array,
    grad_left: F64Array,
    grad_right: F64Array,
    total: float,
    tol: float,
    want_grad: bool,
) -> float:
    """Area of the smaller lobe of ``left`` minus ``right``; zero unless the pair crosses 4x.

    When ``want_grad``, also accumulates ``d((lobe/total)**2)/d(param)`` into ``grad_left``
    and ``grad_right``. For the value alone, call ``_smaller_lobe_core``.
    """
    # the lobe is a two-arc loop: _arc_shape_grad over each arc, scaled by 2*(lobe/total)/total
    # and the loop's sign
    # coincident boundaries never disconnect and the pencil is singular for them; short-circuit
    left_cx, left_cy, left_major, left_minor, left_angle = left
    right_cx, right_cy, right_major, right_minor, right_angle = right
    left_form = _ellipse_form(left_cx, left_cy, left_major, left_minor, left_angle)
    right_form = _ellipse_form(
        right_cx, right_cy, right_major, right_minor, right_angle
    )
    if _same_form(left_form, right_form, tol):
        return 0.0

    points, kept = _crossings(left, right, tol)
    # a set difference only splits into lobes at exactly four boundary crossings
    if kept != 4:  # noqa: PLR2004
        return 0.0

    # walk the crossings in order around the left boundary
    raw_angles = np.empty(4, "f8")
    for i in range(4):
        raw_angles[i] = eccentric_angle(left, points[i, 0], points[i, 1])
    order = np.argsort(raw_angles, kind="mergesort")
    angles_left = np.empty(4, "f8")
    angles_right = np.empty(4, "f8")
    for i in range(4):
        angles_left[i] = raw_angles[order[i]]
        angles_right[i] = eccentric_angle(
            right, points[order[i], 0], points[order[i], 1]
        )

    two_pi = 2.0 * PI
    # four crossings bound two lobes; one buffer slot per lobe
    areas = np.empty(2, "f8")
    left_arc = np.empty((2, 2), "f8")
    right_arc = np.empty((2, 2), "f8")
    signs = np.empty(2, "f8")
    lobe_count = 0
    for i in range(4):
        start = angles_left[i]
        sweep = _pymod(angles_left[(i + 1) % 4] - start, two_pi)
        mid_x, mid_y = point_at(left, start + sweep / 2.0)
        if _contains(right, mid_x, mid_y):
            continue  # this arc of the left boundary is inside right, not a lobe edge
        area_left = arc_green(left, start, start + sweep)
        # close with the arc of the right boundary whose interior holds no crossing
        head = (i + 1) % 4
        tail = i
        begin = angles_right[head]
        span = _pymod(angles_right[tail] - begin, two_pi)
        holds_crossing = False
        for other in range(4):
            if other not in (head, tail) and (
                _pymod(angles_right[other] - begin, two_pi) < span
            ):
                holds_crossing = True
        if holds_crossing:
            span -= two_pi
        signed = area_left + arc_green(right, begin, begin + span)
        if lobe_count < 2:  # noqa: PLR2004
            areas[lobe_count] = abs(signed)
            left_arc[lobe_count, 0], left_arc[lobe_count, 1] = start, start + sweep
            right_arc[lobe_count, 0], right_arc[lobe_count, 1] = begin, begin + span
            signs[lobe_count] = 1.0 if signed >= 0.0 else -1.0
        lobe_count += 1

    if lobe_count != 2:  # noqa: PLR2004
        return 0.0
    if not (math.isfinite(areas[0]) and math.isfinite(areas[1])):
        return 0.0
    smaller = 0 if areas[0] <= areas[1] else 1
    lobe = areas[smaller]
    if want_grad:
        weight = 2.0 * lobe / (total * total) * signs[smaller]
        _arc_shape_grad(
            grad_left, left, left_arc[smaller, 0], left_arc[smaller, 1], weight
        )
        _arc_shape_grad(
            grad_right, right, right_arc[smaller, 0], right_arc[smaller, 1], weight
        )
    return lobe


@nb.njit(nb.float64(RO1, RO1, nb.float64), cache=True, error_model="numpy")
def _smaller_lobe_core(left: F64Array, right: F64Array, tol: float) -> float:
    """Area of the smaller lobe of ``left`` minus ``right``; zero unless 4 crossings."""
    unused = np.zeros(5)
    return _smaller_lobe_grad(left, right, unused, unused, 1.0, tol, False)


@nb.njit(nb.float64(RO2, nb.float64, nb.float64), cache=True, error_model="numpy")
def penalty_core(ellipses: F64Array, total: float, tol: float) -> float:
    """Sum over pairs of the squared spurious-lobe area, normalized by ``total``, both ways.

    Zero when every pair overlaps in a simple lens (at most two crossings); positive and
    smoothly growing once a pair crosses four times and disconnects. ``tol`` is the tolerance
    of the crossing tests. For the gradient too, call ``penalty_grad_core``.
    """
    n = ellipses.shape[0]
    result = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            forward = _smaller_lobe_core(ellipses[i], ellipses[j], tol) / total
            backward = _smaller_lobe_core(ellipses[j], ellipses[i], tol) / total
            result += forward * forward + backward * backward
    return result


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[:, ::1]))(RO2, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
)
def penalty_grad_core(
    ellipses: F64Array, total: float, tol: float
) -> tuple[float, F64Array]:
    """Compute the disconnection penalty and its gradient per ``(cx, cy, major, minor, angle)``."""
    n = ellipses.shape[0]
    grad = np.zeros((n, 5))
    result = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            forward = _smaller_lobe_grad(
                ellipses[i], ellipses[j], grad[i], grad[j], total, tol, True
            )
            backward = _smaller_lobe_grad(
                ellipses[j], ellipses[i], grad[j], grad[i], total, tol, True
            )
            result += (forward * forward + backward * backward) / (total * total)
    return result, grad


def same_ellipse(left: F64Array, right: F64Array, *, tol: float = 1e-9) -> bool:
    """Whether two ellipse rows describe the same ellipse, in any parameterization.

    Invariant to a circle's arbitrary angle, and to how the axes and rotation are otherwise
    spelled.
    """
    # compares the normalized quadratic forms, which erase the parameterization
    left_cx, left_cy, left_major, left_minor, left_angle = left
    right_cx, right_cy, right_major, right_minor, right_angle = right
    left_form = _ellipse_form(left_cx, left_cy, left_major, left_minor, left_angle)
    right_form = _ellipse_form(
        right_cx, right_cy, right_major, right_minor, right_angle
    )
    return _same_form(left_form, right_form, tol)


@nb.njit(
    nb.optional(nb.types.Tuple((nb.float64[:, :], nb.boolean[::1])))(
        RO2, BO1, nb.float64
    ),
    cache=True,
    error_model="numpy",
)
def collapse_coincident(
    ellipses: F64Array, flags: BoolArray, tol: float
) -> tuple[F64Array, BoolArray] | None:
    """Merge coincident ellipses to one representative each, or ``None`` on a flag conflict.

    Two ellipses coincide when their normalized forms match within ``tol``. ``None`` means a
    coincident pair is flagged inside one twin and outside the other -- a contradiction, so the
    zone is empty.
    """
    n = ellipses.shape[0]
    forms = np.empty((n, 3, 3), "f8")
    for i in range(n):
        cx, cy, major, minor, angle = ellipses[i]
        form = _ellipse_form(cx, cy, major, minor, angle)
        for row in range(3):
            for col in range(3):
                forms[i, row, col] = form[row][col]
    keep = np.empty(n, "i8")
    kept = 0
    for i in range(n):
        rep = -1
        for k in range(kept):
            if _same_form(_form_at(forms, i), _form_at(forms, keep[k]), tol):
                rep = keep[k]
                break
        if rep >= 0:
            if flags[i] != flags[rep]:
                return None
        else:
            keep[kept] = i
            kept += 1
    idx = keep[:kept]
    return ellipses[idx], flags[idx]


def _on_boundary(row: F64Array, points: F64Array, tol: float) -> BoolArray:
    """Whether each point lies on the ellipse boundary (frame radius 1 within tol)."""
    with np.errstate(
        invalid="ignore", divide="ignore"
    ):  # non-finite roots fall out below
        cx, cy, major, minor, angle = row
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        along_major = (cos_a * dx + sin_a * dy) / major
        along_minor = (-sin_a * dx + cos_a * dy) / minor
        return np.abs(along_major**2 + along_minor**2 - 1.0) < tol


def ellipse_crossings(
    left: F64Array, right: F64Array, *, tol: float = 1e-9
) -> F64Array:
    """Find the distinct real boundary crossing points of two ellipses, as a ``(k, 2)`` array.

    ``left`` and ``right`` are ellipse rows ``(cx, cy, major, minor, angle)``; ``tol`` is the
    tolerance of the crossing tests. Coincident ellipses cross nowhere, and give ``(0, 2)``.
    """
    left_row = np.ascontiguousarray(left, dtype="f8")
    right_row = np.ascontiguousarray(right, dtype="f8")
    if left_row.shape != (5,) or right_row.shape != (5,):
        raise ValueError("each ellipse must be a (5,) row")
    if same_ellipse(left_row, right_row, tol=tol):
        # coincident boundaries share every point; the pencil is singular for them, so short-circuit
        return np.empty((0, 2), dtype="f8")
    points, count = _crossings(left_row, right_row, tol)
    points = points[:count]
    # a near-singular pencil returns spurious/non-finite roots; keep only points on both boundaries
    keep = _on_boundary(left_row, points, _CROSSING_TOL) & _on_boundary(
        right_row, points, _CROSSING_TOL
    )
    return points[keep]
