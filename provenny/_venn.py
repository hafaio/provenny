"""Lay out area-proportional euler/venn diagrams for any number of sets."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numba as nb
import numpy as np
from scipy import optimize as spo

from ._kernel import area_core, area_grad_core, penalty_core, penalty_grad_core
from ._shape import Ellipse, canonicalize
from ._types import BoolArray, F64Array, I64Array
from ._zone import Zone
from ._zone import zone as _zone

_Name = TypeVar("_Name", bound=Hashable)

# eccentricity-penalty weights, annealed high -> 0: relax round -> elliptical while staying in
# the basin. Adds no eccentricity when circles are optimal (moving off a circle only raises the error).
_ANNEAL_SCHEDULE: tuple[float, ...] = (0.3, 0.1, 0.03, 0.01, 0.003, 0.0)
# default disconnection-penalty weight ("ellipse" mode). The penalty is a step, so any
# suprathreshold value works; this sits well above the threshold (~max area error).
_DEFAULT_DISCONNECT_WEIGHT = 100.0
# the (inds, sub_mat, sizes, tol) tail shared by every area objective's njit signature
_PROBLEM_ARGS = (nb.boolean[:, ::1], nb.float64[:, ::1], nb.float64[::1], nb.float64)


@nb.njit(
    nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
    cache=True,
    error_model="numpy",
)
def _circle_overlap_area(d: float, r1: float, r2: float, target: float) -> float:
    """Overlap area of two circles at center distance ``d``, minus ``target``."""
    # closed-form lens, not the general kernel: stays finite across the whole bracket, where
    # the kernel is ambiguous for near-coincident circles at d ~ 0
    lens = (
        r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        + r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        - 0.5 * math.sqrt((r1 + r2 - d) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    )
    return lens - target


@nb.njit(
    nb.float64(
        nb.float64[::1],
        nb.int64[:, ::1],
        nb.boolean[::1],
        nb.boolean[::1],
        nb.float64[::1],
    ),
    cache=True,
    error_model="numpy",
)
def _cons_mds_fun(  # noqa: PLR0913
    x_flat: F64Array,
    pinds: I64Array,
    disjoint: BoolArray,
    subset: BoolArray,
    norm_target: F64Array,
) -> float:
    """Constrained-mds stress: honor each pair's target distance where its rule is live."""
    x = x_flat.reshape((-1, 2))
    total = 0.0
    for p in range(pinds.shape[0]):
        i, j = pinds[p]
        (xi, yi), (xj, yj) = x[i], x[j]
        dx = xi - xj
        dy = yi - yj
        dist = math.sqrt(dx * dx + dy * dy)
        target = norm_target[p]
        # one-sided: a disjoint pair only pushes when too close, a subset pair when too far
        skip = (disjoint[p] and dist > target) or (subset[p] and dist < target)
        if not skip:
            error = dist - target
            total += error * error
    return total / 2.0


@nb.njit(
    nb.float64[::1](
        nb.float64[::1],
        nb.int64[:, ::1],
        nb.boolean[::1],
        nb.boolean[::1],
        nb.float64[::1],
    ),
    cache=True,
    error_model="numpy",
)
def _cons_mds_jac(  # noqa: PLR0913
    x_flat: F64Array,
    pinds: I64Array,
    disjoint: BoolArray,
    subset: BoolArray,
    norm_target: F64Array,
) -> F64Array:
    """Gradient of :func:`_cons_mds_fun`, scatter-added onto each pair's two centers."""
    x = x_flat.reshape((-1, 2))
    jac = np.zeros(x_flat.size)
    for p in range(pinds.shape[0]):
        i, j = pinds[p]
        (xi, yi), (xj, yj) = x[i], x[j]
        dx = xi - xj
        dy = yi - yj
        dist = math.sqrt(dx * dx + dy * dy)
        target = norm_target[p]
        skip = (disjoint[p] and dist > target) or (subset[p] and dist < target)
        # coincident centers contribute no gradient (not a NaN); subset pairs pull them in
        if not skip and dist > 0.0:
            ratio = (dist - target) / dist
            jac[2 * i] += ratio * dx
            jac[2 * i + 1] += ratio * dy
            jac[2 * j] -= ratio * dx
            jac[2 * j + 1] -= ratio * dy
    return jac


@dataclass(frozen=True, slots=True, eq=False)
class RegionProblem:
    """The disjoint-region machinery the area objectives read, precomputed once."""

    inds: BoolArray
    sub_mat: F64Array
    sizes: F64Array
    num_sets: int
    tol: float


@dataclass(frozen=True, slots=True, eq=False)
class MdsTargets:
    """The precomputed pairwise targets the constrained-mds circle seed reads."""

    single_areas: F64Array
    pinds: I64Array
    subset: BoolArray
    disjoint: BoolArray
    target_dists: F64Array
    num_sets: int


def _pairwise_target_dists(
    single_areas: F64Array,
    pair_areas: F64Array,
    pinds: I64Array,
    tol: float,
) -> tuple[F64Array, BoolArray, BoolArray]:
    """Compute the target center distance, and the type, of each pair of sets."""
    num_pairs = pinds.shape[0]
    target_dists = np.empty(num_pairs)

    # one set is a subset of the other, so the centers are |r1 - r2| apart
    subset = single_areas[pinds].min(-1) <= pair_areas + tol
    subset_radii = (single_areas[pinds[subset]] / np.pi) ** 0.5
    target_dists[subset] = np.abs(np.diff(subset_radii, 1, -1)[:, 0])

    # the two sets are disjoint, so the centers are r1 + r2 apart
    disjoint = pair_areas <= tol
    disjoint_radii = (single_areas[pinds[disjoint]] / np.pi) ** 0.5
    target_dists[disjoint] = disjoint_radii.sum(-1)

    # all remaining pairs partially overlap, so solve for the overlap distance
    valid = ~subset & ~disjoint
    for i, (r1, r2), target in zip(
        np.arange(num_pairs)[valid],
        (single_areas[pinds[valid]] / np.pi) ** 0.5,
        pair_areas[valid],
        strict=True,
    ):
        # a bracketed (brentq) solve: a derivative exists, but bracketing is globally convergent
        res = spo.root_scalar(
            _circle_overlap_area,
            (r1, r2, target),
            bracket=(np.abs(r1 - r2) + tol, r1 + r2 - tol),
        )
        assert res.converged
        target_dists[i] = res.root
    return target_dists, subset, disjoint


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[::1]))(
        nb.float64[::1], nb.float64[:, ::1], nb.float64[::1]
    ),
    cache=True,
    error_model="numpy",
)
def _error_from_areas(
    subset_areas: F64Array, sub_mat: F64Array, sizes: F64Array
) -> tuple[float, F64Array]:
    """Compute the normalized region-size error and its per-region residual.

    ``errors`` are the realized-minus-target disjoint region sizes over the total area; the
    value is half their squared norm.
    """
    errors = (sub_mat @ subset_areas - sizes) / sizes.sum()
    return float(errors @ errors) / 2, errors


@nb.njit(
    nb.float64(nb.float64[:, ::1], *_PROBLEM_ARGS), cache=True, error_model="numpy"
)
def _area_error(
    ellipses: F64Array, inds: BoolArray, sub_mat: F64Array, sizes: F64Array, tol: float
) -> float:
    """Squared error between the realized and target disjoint region sizes."""
    subset_areas = np.empty(inds.shape[0])
    for i in range(inds.shape[0]):
        subset_areas[i] = area_core(ellipses[inds[i]], tol)
    value, _ = _error_from_areas(subset_areas, sub_mat, sizes)
    return value


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[:, ::1]))(
        nb.float64[:, ::1], *_PROBLEM_ARGS
    ),
    cache=True,
    error_model="numpy",
)
def _area_error_grad(
    ellipses: F64Array,
    inds: BoolArray,
    sub_mat: F64Array,
    sizes: F64Array,
    tol: float,
) -> tuple[float, F64Array]:
    """Compute the area error and its gradient per ``(cx, cy, major, minor, angle)``, per set."""
    num_regions = inds.shape[0]
    num_sets = ellipses.shape[0]
    subset_areas = np.empty(num_regions)
    # each region's area gradient, packed compactly (only its member sets, in order)
    region_grads = np.empty((num_regions, num_sets, 5))
    for region in range(num_regions):
        area, grad = area_grad_core(ellipses[inds[region]], tol, True)
        subset_areas[region] = area
        region_grads[region, : grad.shape[0]] = grad
    value, errors = _error_from_areas(subset_areas, sub_mat, sizes)
    # chain the squared error through the region areas: d(value)/d(area) = sub_mat.T @ err
    error_grad = (sub_mat.T @ errors) / sizes.sum()
    grad = np.zeros((num_sets, 5))
    for region in range(num_regions):
        member = 0
        for ellipse in range(num_sets):
            if inds[region, ellipse]:
                grad[ellipse] += error_grad[region] * region_grads[region, member]
                member += 1
    return value, grad


@nb.njit(nb.float64(nb.float64[:, ::1]), cache=True, error_model="numpy")
def _smooth_ecc_penalty(ellipses: F64Array) -> float:
    """Smooth, kink-free eccentricity penalty; zero and flat at a circle."""
    # quadratic in log-eccentricity: no slope discontinuity at a circle, unlike 1 - (minor/major)**2
    log_ratio = np.log(ellipses[:, 3] / ellipses[:, 2])
    return float(log_ratio @ log_ratio) / 2


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[::1]))(
        nb.float64[::1], nb.int64, *_PROBLEM_ARGS
    ),
    cache=True,
    error_model="numpy",
)
def _circle_value_grad(  # noqa: PLR0913
    x: F64Array,
    num_sets: int,
    inds: BoolArray,
    sub_mat: F64Array,
    sizes: F64Array,
    tol: float,
) -> tuple[float, F64Array]:
    """Area error and gradient of a flat ``(cx, cy, radius)`` circle vector."""
    params = x.reshape((3, num_sets))
    ell = np.empty((num_sets, 5))
    ell[:, 0] = params[0]
    ell[:, 1] = params[1]
    ell[:, 2] = params[2]
    ell[:, 3] = params[2]
    ell[:, 4] = 0.0
    value, grad = _area_error_grad(ell, inds, sub_mat, sizes, tol)
    flat = np.empty(3 * num_sets)
    flat[0:num_sets] = grad[:, 0]
    flat[num_sets : 2 * num_sets] = grad[:, 1]
    # a circle's single radius drives both semi-axes, so their gradients add
    flat[2 * num_sets : 3 * num_sets] = grad[:, 2] + grad[:, 3]
    return value, flat


def _mds_seed(targets: MdsTargets, rng: np.random.Generator) -> F64Array:
    """Seed circle parameters from a constrained mds on the target distances."""
    num_sets = targets.num_sets
    # all target distances zero only when all sets coincide; fall back to 1 to avoid /0
    norm_factor = float(np.linalg.norm(targets.target_dists)) or 1.0
    res = spo.minimize(
        _cons_mds_fun,
        rng.standard_normal(size=num_sets * 2) / 2,
        (
            targets.pinds,
            targets.disjoint,
            targets.subset,
            targets.target_dists / norm_factor,
        ),
        jac=_cons_mds_jac,
    )
    coords = res.x
    cx_init, cy_init = coords.reshape((num_sets, 2)).T * norm_factor
    radii_init = (targets.single_areas / np.pi) ** 0.5
    return np.stack([cx_init, cy_init, radii_init])


# below this the mds already realized the targets -- an exact fit, nothing for the area minimize
# to improve. Realizable targets bottom out ~1e-30, frustrated ones ~1e-4+.
_FIT_TOLERANCE = 1e-9


def _minimize_value_grad(
    objective: Callable[..., tuple[float, F64Array]],
    x0: F64Array,
    args: tuple[object, ...],
) -> tuple[float, F64Array]:
    """l-bfgs-B on a value-and-gradient objective; returns its ``(loss, solution)``."""
    res = spo.minimize(
        # jac=True: objective returns (value, grad); scipy's stub doesn't type this
        objective,  # type: ignore[arg-type]
        x0,
        args=args,
        jac=True,
        method="l-bfgs-B",
    )
    return float(res.fun), np.asarray(res.x, dtype="f8")


# within this, two layouts are the same basin (same areas up to rigid motion)
_BASIN_TOL = 1e-4

# circle mode is basin-insensitive (exact fit ends on restart 1; unfittable targets look the
# same from every basin), so a small constant, not n-scaled
_CIRCLE_RESTARTS = 2


def _default_restarts(num_sets: int) -> int:
    """Pick the n-scaled anneal restart count for the ellipse modes."""
    # ~n-1 distinct basins; 2*(n-1) draws give coupon-collector margin to land the rare ones
    return max(2, 2 * (num_sets - 1))


def _basin_signature(circle: F64Array) -> tuple[int, ...]:
    """Key a circle layout by its per-pair center gaps, invariant to rigid motion.

    The gap between each *specific* pair, in fixed order (not the sorted multiset).
    """
    # not the sorted multiset: that merges layouts assigning the same gaps to different pairs --
    # genuinely different basins when set sizes differ
    centers = circle[:, :2]
    gaps = (
        math.hypot(*(centers[i] - centers[j]))
        for i in range(len(centers))
        for j in range(i)
    )
    return (*(round(gap / _BASIN_TOL) for gap in gaps),)


def _mds_basins(
    targets: MdsTargets, restarts: int, rng: np.random.Generator
) -> Iterator[F64Array]:
    """Yield the distinct circle basins found by ``restarts`` random mds solves.

    Lazy: a caller that stops early never pays for the remaining restarts.
    """
    seen: set[tuple[int, ...]] = set()
    for _ in range(restarts):
        circle = _mds_seed(targets, rng).T
        signature = _basin_signature(circle)
        if signature not in seen:
            seen.add(signature)
            yield circle


def _circle_layout(
    problem: RegionProblem,
    targets: MdsTargets,
    restarts: int,
    rng: np.random.Generator,
) -> F64Array:
    """Fit the best circle layout over ``restarts`` mds-seeded area minimizes.

    Returns ``(n, 3)`` ``(cx, cy, radius)`` rows -- the parameterization it optimizes over,
    which :func:`_solve` promotes to canonical ellipse rows.
    """
    num_sets = problem.num_sets
    args = (num_sets, problem.inds, problem.sub_mat, problem.sizes, problem.tol)
    best_loss = np.inf
    best: F64Array | None = None
    for _ in range(restarts):
        params = _mds_seed(targets, rng).flatten()
        loss = _circle_value_grad(params, *args)[0]
        if loss > _FIT_TOLERANCE:
            loss, params = _minimize_value_grad(_circle_value_grad, params, args)
        if loss < best_loss:
            best_loss = loss
            best = params
        if best_loss <= _FIT_TOLERANCE:
            break
    assert best is not None
    layout = best.reshape(3, num_sets).T
    # area error is even in radius, so the solve may land on a negative-radius mirror; abs it
    layout[:, 2] = np.abs(layout[:, 2])
    return layout


# initial eccentricity of the stretched seed; the fit adjusts from here, so moderate is enough
_SEED_LOG_ECCENTRICITY = math.log(1.5)
# the stretched seed anneals only weights <= this: low enough not to squash its orientation
# round, still relaxing to zero
_SEED_TAIL_WEIGHT = 0.01
# the stretched seed replaces the round fit only if it beats it by this much (keep circles
# when they fit as well)
_SEED_FIT_MARGIN = 1e-6


def _ellipse_seeds(
    circles: F64Array,
) -> tuple[F64Array, F64Array]:
    """Build the round seed and one radially-stretched seed for the ellipse anneal.

    The round seed covers the circle optimum; the stretched seed points each ellipse's long
    axis radially from the layout centroid.
    """
    # one structured seed suffices: the radial stretch lifts off the flat circle shoulder, then
    # the optimizer rotates each orientation freely (a tangential variant never helped in testing)
    cx, cy, rad = circles.T
    zeros = np.zeros_like(rad)
    stretch = np.full_like(rad, _SEED_LOG_ECCENTRICITY)
    centroid = circles[:, :2].mean(0)
    centroid_x, centroid_y = centroid
    radial = np.arctan2(cy - centroid_y, cx - centroid_x)
    circle_seed = np.concatenate([cx, cy, rad, zeros, zeros])
    radial_seed = np.concatenate([cx, cy, rad, stretch, radial + np.pi / 2])
    return circle_seed, radial_seed


@nb.njit(nb.float64[:, ::1](nb.float64[::1], nb.int64), cache=True, error_model="numpy")
def _anneal_ellipses(x: F64Array, num_sets: int) -> F64Array:
    """Unpack a flat (cx, cy, radius, log-eccentricity, angle) vector into (n, 5) rows.

    The rows are **not** canonical: column 2 is ``radius / e`` and column 3 ``radius * e``, so
    for eccentricity ``e > 1`` the "major" column holds the smaller axis.
    """
    # safe: the kernels read the orientation-agnostic quadratic form; _solve canonicalizes on exit
    params = x.reshape((5, num_sets))
    excents = np.exp(params[3])
    ell = np.empty((num_sets, 5))
    ell[:, 0] = params[0]
    ell[:, 1] = params[1]
    ell[:, 2] = params[2] / excents
    ell[:, 3] = params[2] * excents
    ell[:, 4] = params[4]
    return ell


@nb.njit(
    nb.float64(
        nb.float64[::1], nb.float64, nb.int64, *_PROBLEM_ARGS, nb.float64, nb.float64
    ),
    cache=True,
    error_model="numpy",
)
def _anneal_objective(  # noqa: PLR0913
    x: F64Array,
    weight: float,
    num_sets: int,
    inds: BoolArray,
    sub_mat: F64Array,
    sizes: F64Array,
    tol: float,
    disconnect_weight: float,
    total: float,
) -> float:
    """Area error plus the annealed eccentricity and (where weighted) disconnection penalties."""
    ell = _anneal_ellipses(x, num_sets)
    loss = _area_error(ell, inds, sub_mat, sizes, tol)
    loss += weight * _smooth_ecc_penalty(ell)
    if disconnect_weight != 0.0:
        loss += disconnect_weight * penalty_core(ell, total, tol)
    return loss


@nb.njit(
    nb.float64[::1](
        nb.float64[:, ::1], nb.float64[:, ::1], nb.float64[::1], nb.float64, nb.int64
    ),
    cache=True,
    error_model="numpy",
)
def _pack_anneal_grad(
    row_grad: F64Array, ell: F64Array, x: F64Array, weight: float, num_sets: int
) -> F64Array:
    """Chain ``d(loss)/d(cx,cy,major,minor,angle)`` to the flat optimizer vector.

    Also adds the eccentricity penalty's ``d/d(log_ecc) = weight*4*log_ecc``.
    """
    # adjoint of the _anneal_ellipses map (major = r/e, minor = r*e); changing that packing moves this
    log_ecc = x.reshape((5, num_sets))[3]
    major = ell[:, 2]
    minor = ell[:, 3]
    excents = np.exp(log_ecc)
    flat = np.empty(5 * num_sets)
    flat[0:num_sets] = row_grad[:, 0]
    flat[num_sets : 2 * num_sets] = row_grad[:, 1]
    # the radius drives both semi-axes
    flat[2 * num_sets : 3 * num_sets] = (
        row_grad[:, 2] / excents + row_grad[:, 3] * excents
    )
    # log-eccentricity stretches major down and minor up; the penalty is 2*log_ecc**2
    flat[3 * num_sets : 4 * num_sets] = (
        -row_grad[:, 2] * major + row_grad[:, 3] * minor + weight * 4.0 * log_ecc
    )
    flat[4 * num_sets : 5 * num_sets] = row_grad[:, 4]
    return flat


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[::1]))(
        nb.float64[::1], nb.float64, nb.int64, *_PROBLEM_ARGS
    ),
    cache=True,
    error_model="numpy",
)
def _fit_value_grad(  # noqa: PLR0913
    x: F64Array,
    weight: float,
    num_sets: int,
    inds: BoolArray,
    sub_mat: F64Array,
    sizes: F64Array,
    tol: float,
) -> tuple[float, F64Array]:
    """Evaluate the plain fit objective -- area error + annealed eccentricity -- and its gradient.

    No disconnection term, so overlaps may split into lobes; :func:`_connected_value_grad`
    adds that term.
    """
    ell = _anneal_ellipses(x, num_sets)
    value, area_grad = _area_error_grad(ell, inds, sub_mat, sizes, tol)
    value += weight * _smooth_ecc_penalty(ell)
    return value, _pack_anneal_grad(area_grad, ell, x, weight, num_sets)


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[::1]))(
        nb.float64[::1], nb.float64, nb.int64, *_PROBLEM_ARGS, nb.float64, nb.float64
    ),
    cache=True,
    error_model="numpy",
)
def _connected_value_grad(  # noqa: PLR0913
    x: F64Array,
    weight: float,
    num_sets: int,
    inds: BoolArray,
    sub_mat: F64Array,
    sizes: F64Array,
    tol: float,
    disconnect_weight: float,
    total: float,
) -> tuple[float, F64Array]:
    """Evaluate the connected objective -- the plain fit plus the disconnection penalty -- and its gradient."""
    ell = _anneal_ellipses(x, num_sets)
    value, area_grad = _area_error_grad(ell, inds, sub_mat, sizes, tol)
    value += weight * _smooth_ecc_penalty(ell)
    penalty, penalty_grad = penalty_grad_core(ell, total, tol)
    value += disconnect_weight * penalty
    return value, _pack_anneal_grad(
        area_grad + disconnect_weight * penalty_grad, ell, x, weight, num_sets
    )


def _anneal_ellipse(
    problem: RegionProblem,
    circles: F64Array,
    schedule: Sequence[float],
    disconnect_weight: float,
) -> F64Array:
    """Annealed ellipse layout, warm-started from the raw mds ``circles`` (not area-fitted).

    Fits the round seed (the circles as zero-eccentricity ellipses), and where they do not
    already suffice also the eccentric seed from :func:`_ellipse_seeds`, keeping whichever
    wins. ``disconnect_weight`` (> 0 for ``"ellipse"`` mode) penalizes pairs of ellipses that
    cross four times, splitting a set-difference region into lobes.
    """
    num_sets = problem.num_sets
    total = float(problem.sizes.sum())
    shared = (
        num_sets,
        problem.inds,
        problem.sub_mat,
        problem.sizes,
        problem.tol,
        disconnect_weight,
        total,
    )
    grad_args = (num_sets, problem.inds, problem.sub_mat, problem.sizes, problem.tol)
    # the disconnection penalty is expensive and kinked; a fit not wanting it skips that objective
    penalty_free = disconnect_weight == 0.0

    def minimize_from(seed: F64Array, weights: Sequence[float]) -> F64Array:
        x = seed
        for weight in weights:
            if penalty_free:
                _, x = _minimize_value_grad(_fit_value_grad, x, (weight, *grad_args))
            else:
                _, x = _minimize_value_grad(_connected_value_grad, x, (weight, *shared))
        return x

    circle_seed, radial_seed = _ellipse_seeds(circles)
    # if the round circles already fit (and stay connected where penalized), skip the anneal
    if _anneal_objective(circle_seed, 0.0, *shared) <= _FIT_TOLERANCE:
        return _anneal_ellipses(circle_seed, num_sets)

    tail = (*(weight for weight in schedule if weight <= _SEED_TAIL_WEIGHT),)
    best = minimize_from(circle_seed, schedule)
    best_loss = _anneal_objective(best, 0.0, *shared)
    if best_loss > _SEED_FIT_MARGIN:
        candidate = minimize_from(radial_seed, tail)
        if _anneal_objective(candidate, 0.0, *shared) < best_loss - _SEED_FIT_MARGIN:
            best = candidate
    return _anneal_ellipses(best, num_sets)


def _build_problem(areas: F64Array, tol: float) -> tuple[RegionProblem, MdsTargets]:
    """Precompute the region machinery and mds targets shared across the solve."""
    # areas has length 2**n - 1 for n sets, whose bit_length is exactly n
    num_sets = areas.size.bit_length()

    # set index in areas; numpy uints are 1/2/4/8-byte, so round up to a valid width
    min_bytes = (num_sets + 7) // 8
    byte_width = 1 << (min_bytes - 1).bit_length()
    int_range = np.arange(1, 2**num_sets, dtype=f">u{byte_width:d}")
    int_bytes = int_range[:, None].view("B")
    inds = np.unpackbits(int_bytes, -1)[:, : -num_sets - 1 : -1].astype("?")

    # matrix that turns full disjunction sizes into set sizes, and its inverse
    sum_mat = (inds[:, None] <= inds[None]).all(-1)
    sub_mat = np.linalg.inv(sum_mat)
    sizes = sub_mat @ areas
    if np.any(sizes < -tol):
        raise ValueError(
            "subset areas are inconsistent: they imply a negative region size"
        )
    sizes = np.maximum(sizes, 0)

    # solo-set and pair indices. tril_indices returns platform intp; pin int64 for the njit sigs
    single_inds = (1 << np.arange(num_sets)) - 1
    pinds = np.stack(np.tril_indices(num_sets, -1), 1).astype("i8")
    pair_inds = (2**pinds).sum(-1) - 1

    single_areas = areas[single_inds]
    target_dists, subset, disjoint = _pairwise_target_dists(
        single_areas, areas[pair_inds], pinds, tol
    )
    region = RegionProblem(
        inds=inds, sub_mat=sub_mat, sizes=sizes, num_sets=num_sets, tol=tol
    )
    targets = MdsTargets(
        single_areas=single_areas,
        pinds=pinds,
        subset=subset,
        disjoint=disjoint,
        target_dists=target_dists,
        num_sets=num_sets,
    )
    return region, targets


def _mapping_to_areas(
    areas: Mapping[Collection[_Name], float],
) -> tuple[tuple[_Name, ...], F64Array]:
    """Turn named subset areas into set names and a bitmask-ordered area array.

    Each key is the collection of set names a subset lies inside; a bare string counts as its
    characters, so ``{"A": 1, "B": 1, "AB": 0.4}`` names two sets and gives their overlap.
    Omitted subsets are empty. Set names come out in order of first appearance, scanning keys
    in mapping order and names in each key's iteration order; a ``set``/``frozenset`` key that
    first introduces several names therefore leaves their relative order unspecified across
    processes, which -- with a seeded ``rng`` -- can change the layout. Give each set its own
    singleton key for a stable order.
    """
    names = (*dict.fromkeys(name for key in areas for name in key),)
    if not names:
        raise ValueError("the areas mapping must name at least one set")
    bits = {name: 1 << index for index, name in enumerate(names)}
    values = np.zeros(2 ** len(names) - 1)
    seen: set[int] = set()
    for key, value in areas.items():
        # the subset's bitmask index is the OR of its names' bits
        mask = 0
        for name in key:
            bit = bits[name]
            if mask & bit:
                raise ValueError(f"subset key {key!r} names {name!r} twice")
            mask |= bit
        if mask in seen:
            raise ValueError(f"duplicate subset in areas: {set(key)}")
        seen.add(mask)
        if mask:  # the empty subset is the exterior, which has no region slot
            values[mask - 1] = float(value)
    return names, values


def _validate_areas(areas: F64Array) -> None:
    """Reject a malformed subset-area array with a clear error."""
    if areas.ndim != 1 or areas.size == 0:
        raise ValueError("areas must be a non-empty 1-D array of subset areas")
    num_sets = areas.size.bit_length()
    if 2**num_sets != areas.size + 1:
        raise ValueError(
            f"areas must have length 2**n - 1 for n sets; got length {areas.size}"
        )
    single_areas = areas[(1 << np.arange(num_sets)) - 1]
    if not np.all(np.isfinite(areas)):
        raise ValueError("subset areas must all be finite")
    elif np.any(areas < 0):
        raise ValueError("subset areas must be non-negative")
    elif np.any(single_areas <= 0):
        raise ValueError("every set must have a positive area")


def _solve(  # noqa: PLR0913
    areas: F64Array,
    restarts: int | None,
    tol: float,
    anneal: bool,
    disconnect_weight: float,
    rng: np.random.Generator | None,
) -> F64Array:
    """Lay out the areas, from already-resolved parameters (see the public wrappers)."""
    generator = np.random.default_rng() if rng is None else rng
    _validate_areas(areas)
    # nondimensionalize to unit reference area so tol, the penalty, and the weight are scale-free
    # (fixed tol is otherwise swamped far from unit area). Lengths scaled back at the end.
    reference_area = float(areas.max())
    problem, targets = _build_problem(areas / reference_area, tol)
    # each solver uses its own parameterization (circles in 3 columns, the anneal axes in either
    # order); this is the one place both become the canonical (n, 5) rows downstream assumes
    if not anneal:
        circle_restarts = _CIRCLE_RESTARTS if restarts is None else restarts
        cx, cy, radius = _circle_layout(problem, targets, circle_restarts, generator).T
        # equal axes and no rotation: already canonical, no need to normalize
        shapes = np.stack([cx, cy, radius, radius, np.zeros_like(radius)], 1)
    else:
        # the anneal leaves the axes in either order; canonicalize here (Ellipse and Arc require it)
        shapes = canonicalize(
            _best_anneal(problem, targets, restarts, generator, disconnect_weight)
        )
    # restore scale: lengths scale by sqrt(area factor), the angle (last column) is scale-invariant
    shapes[:, :4] *= math.sqrt(reference_area)
    return shapes


def _best_anneal(
    problem: RegionProblem,
    targets: MdsTargets,
    restarts: int | None,
    rng: np.random.Generator,
    disconnect_weight: float,
) -> F64Array:
    """Anneal each distinct mds basin and return the ellipse layout that fits best."""
    if restarts is None:
        restarts = _default_restarts(problem.num_sets)
    total = float(problem.sizes.sum())
    best_shapes: F64Array | None = None
    best_loss = np.inf
    # distinct arrangements anneal to very different fits, unknown before the anneal, so each gets a turn
    for circle in _mds_basins(targets, restarts, rng):
        shapes = _anneal_ellipse(problem, circle, _ANNEAL_SCHEDULE, disconnect_weight)
        loss = _area_error(
            shapes, problem.inds, problem.sub_mat, problem.sizes, problem.tol
        )
        if disconnect_weight:
            loss += disconnect_weight * penalty_core(shapes, total, problem.tol)
        if loss < best_loss:
            best_loss = loss
            best_shapes = shapes
        if best_loss <= _FIT_TOLERANCE:
            break  # an exact fit; no basin can beat it, and eccentricity only hurts
    assert best_shapes is not None
    return best_shapes


@dataclass(frozen=True, slots=True, eq=False)
class Diagram(Generic[_Name]):
    """A solved layout: the set ``names`` and the ``(n, 5)`` ``shapes`` placed for them.

    Returned by :func:`proportional_venn`. ``diagram[name]`` (or :meth:`ellipse`) gives a
    set's shape as an :class:`Ellipse`, and the diagram acts as a mapping over the names --
    iteration, ``len``, and ``in`` -- in order of first appearance, generic over the
    (hashable) label type.
    """

    names: tuple[_Name, ...]
    shapes: F64Array

    def __post_init__(self) -> None:
        """Reject shapes that are not one ellipse row per name."""
        if self.shapes.shape != (len(self.names), 5):
            raise ValueError(
                f"shapes must be one (cx, cy, major, minor, angle) row per name: "
                f"expected {(len(self.names), 5)}, got {self.shapes.shape}"
            )

    def __len__(self) -> int:
        """Return the number of sets."""
        return len(self.names)

    def __iter__(self) -> Iterator[_Name]:
        """Iterate the set names, in the order of ``names``."""
        return iter(self.names)

    def ellipse(self, name: _Name) -> Ellipse:
        """Return the shape placed for a set name, as an :class:`Ellipse`.

        The spelled-out form of ``diagram[name]``; raises ``KeyError`` for an unknown name,
        as indexing does.
        """
        try:
            index = self.names.index(name)
        except ValueError:
            raise KeyError(name) from None
        return Ellipse(self.shapes[index])

    def __getitem__(self, name: _Name) -> Ellipse:
        """Return the shape placed for a set name, as an :class:`Ellipse`."""
        return self.ellipse(name)

    def zone(self, names: Collection[Hashable], *, tol: float = 1e-9) -> Zone | None:
        """Return the zone inside exactly ``names`` (its boundary, area, and label point).

        ``names`` are the sets the zone is *inside*; every other set is outside it, so
        ``{"A", "B"}`` is the ``A & B`` zone and ``{"A"}`` the "A only" zone. ``None``
        means that zone is empty in this layout. Its label point is ``zone(...).center``.
        ``tol`` is the boundary-test tolerance (as for :func:`zone`). Raises ``KeyError``
        for an unknown name and ``ValueError`` for an empty ``names`` (the exterior is not
        a zone).
        """
        # Collection[Hashable], not Collection[_Name]: {"A", "B"} widens to set[str], not
        # assignable to Collection[Literal["A", "B"]] when _Name is inferred from literal keys
        wanted = frozenset(names)
        unknown = wanted.difference(self.names)
        if unknown:
            raise KeyError(f"unknown set names: {[*unknown]}")
        inside = np.array([name in wanted for name in self.names], dtype="?")
        return _zone(self.shapes, inside, tol=tol)


def proportional_venn(
    areas: Mapping[Collection[_Name], float],
    *,
    mode: Literal["circle", "ellipse", "optimal"] = "circle",
    restarts: int | None = None,
    tol: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> Diagram[_Name]:
    """Lay out a venn/euler diagram from named subset areas.

    Parameters
    ----------
    areas
        A mapping from a subset of set names to that subset's area (e.g.
        ``{"A": 1.0, "B": 1.0, "AB": 0.4}``); omitted subsets are empty. A string key
        is a subset of single-character names, so ``"AB"`` is the pair ``{"A", "B"}``.
    mode
        ``"circle"`` places circles. ``"ellipse"`` places ellipses (annealed from
        pairwise-mds circle positions), which fit targets that circles cannot, while
        keeping every pairwise overlap a single connected lens -- the mode to reach for
        when you want ellipses. ``"optimal"`` drops the connectedness penalty for the
        lowest area error it can reach, letting a pair overlap in two disconnected lobes.
        All modes return circles when circles are already optimal.
    restarts
        The number of random mds restarts. ``"ellipse"``/``"optimal"`` anneal each distinct
        basin the restarts uncover and keep the best fit; ``"circle"`` keeps the best
        circle. ``None`` picks the default: ``2 * (n - 1)`` for the ellipse modes (about
        the basin count, with margin), a small constant for ``"circle"``.
    tol
        The numeric tolerance for area computations.
    rng
        Generator for the restart seeds; ``None`` uses a fresh
        :func:`numpy.random.default_rng`. Pass a seeded one for reproducibility.

    Returns
    -------
    A :class:`Diagram` pairing each set name with the canonical ellipse placed for it;
    index or iterate it to read the shapes.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, ``restarts`` is below 1, the mapping names no set, or a
        subset is malformed -- repeated across keys, naming a set twice, or with a
        non-finite or negative area.
    """
    names, values = _mapping_to_areas(areas)
    shapes = proportional_venn_array(
        values, mode=mode, restarts=restarts, tol=tol, rng=rng
    )
    return Diagram(names, shapes)


def proportional_venn_array(
    areas: F64Array,
    *,
    mode: Literal["circle", "ellipse", "optimal"] = "circle",
    restarts: int | None = None,
    tol: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> F64Array:
    """Lay out a venn/euler diagram from a raw subset-area array.

    Parameters
    ----------
    areas
        A 1-D array of the area of every non-empty subset of the ``n`` sets, length
        ``2**n - 1``, indexed by subset bitmask minus one (bit ``i`` marks set ``i``).
        For two sets this is ``[|A|, |B|, |A & B|]``.
    mode
        ``"circle"`` places circles. ``"ellipse"`` places ellipses (annealed from
        pairwise-mds circle positions), which fit targets that circles cannot, while
        keeping every pairwise overlap a single connected lens -- the mode to reach for
        when you want ellipses. ``"optimal"`` drops the connectedness penalty for the
        lowest area error it can reach, letting a pair overlap in two disconnected lobes.
        All modes return circles when circles are already optimal.
    restarts
        The number of random mds restarts. ``"ellipse"``/``"optimal"`` anneal each distinct
        basin the restarts uncover and keep the best fit; ``"circle"`` keeps the best
        circle. ``None`` picks the default: ``2 * (n - 1)`` for the ellipse modes (about
        the basin count, with margin), a small constant for ``"circle"``.
    tol
        The numeric tolerance for area computations.
    rng
        Generator for the restart seeds; ``None`` uses a fresh
        :func:`numpy.random.default_rng`. Pass a seeded one for reproducibility.

    Returns
    -------
    The shapes, one canonical ellipse row per set: an ``(n, 5)`` array of
    ``(cx, cy, major, minor, angle)`` with ``major >= minor`` and ``angle`` in
    ``[0, pi)``. Every mode returns this same ``(n, 5)`` layout, so a caller never
    branches on the mode.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, ``restarts`` is below 1, or the areas are malformed --
        the wrong length, non-finite, negative, or leaving a set with no area.
    """
    match mode:
        case "circle":
            anneal = False
            disconnect_weight = 0.0
        case "ellipse":
            anneal = True
            disconnect_weight = _DEFAULT_DISCONNECT_WEIGHT
        case "optimal":
            anneal = True
            disconnect_weight = 0.0
        case _:
            raise ValueError(f"invalid mode: {mode!r}")
    if restarts is not None and restarts < 1:
        raise ValueError(f"restarts must be at least 1; got {restarts}")
    array = np.asarray(areas, dtype="f8")
    shapes = _solve(array, restarts, tol, anneal, disconnect_weight, rng)
    # freeze the freshly-owned array so the public shapes (and Ellipse views) are read-only
    shapes.flags.writeable = False
    return shapes
