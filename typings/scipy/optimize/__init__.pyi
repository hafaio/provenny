"""Type stubs for the scipy.optimize surface provenny uses."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

class _MinimizeResult:
    x: NDArray[np.float64]
    fun: float
    success: bool

class _RootResult:
    root: float
    converged: bool

class NonlinearConstraint:
    """A nonlinear constraint ``lb <= fun(x) <= ub`` for :func:`minimize`."""

    def __init__(
        self,
        fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        lb: float,
        ub: float,
        jac: Callable[[NDArray[np.float64]], NDArray[np.float64]] | str = ...,
    ) -> None: ...

def minimize(
    fun: Callable[..., float],
    x0: NDArray[np.float64],
    args: tuple[object, ...] = ...,
    method: str | None = ...,
    jac: Callable[..., NDArray[np.float64]] | bool | None = ...,
    constraints: object = ...,
) -> _MinimizeResult:
    """Minimize a scalar function of one or more variables."""

def root_scalar(
    f: Callable[..., float],
    args: tuple[object, ...] = ...,
    method: str | None = ...,
    bracket: tuple[float, float] | None = ...,
) -> _RootResult:
    """Find a scalar root of ``f`` within a bracket."""
