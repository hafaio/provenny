"""Minimal type stubs for the subset of numba the geometry kernel uses.

Numba ships no type information, so this stub types just what ``provenny`` touches: the
``njit`` decorator (as a signature-preserving identity decorator), the scalar type
objects used to spell out compiled signatures, and the ``types`` submodule with the
array/tuple type constructors those signatures build on.
"""

from collections.abc import Callable
from typing import TypeVar

from . import types as types

_Fn = TypeVar("_Fn", bound=Callable[..., object])

class _NumbaType:
    """A numba scalar type: subscript for arrays, call to build a signature."""

    def __getitem__(self, item: object) -> _NumbaType: ...
    def __call__(self, *args: object) -> _NumbaType: ...

void: _NumbaType
boolean: _NumbaType
int64: _NumbaType
float64: _NumbaType
complex128: _NumbaType

def optional(typ: _NumbaType) -> _NumbaType: ...
def njit(
    *signature: object, cache: bool = ..., error_model: str = ..., **options: object
) -> Callable[[_Fn], _Fn]:
    """Compile in nopython mode; typed to return the decorated function unchanged."""
