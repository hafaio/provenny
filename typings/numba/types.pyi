"""Type stubs for the ``numba.types`` constructors the kernel signatures build on.

Each constructor returns a :class:`~numba._NumbaType`, so its result composes into a
compiled signature exactly like the scalar type objects (``nb.float64`` and friends):
subscript it for arrays or call it to build a function signature.
"""

from . import _NumbaType

def Array(
    dtype: _NumbaType, ndim: int, layout: str, readonly: bool = ...
) -> _NumbaType: ...
def UniTuple(dtype: _NumbaType, count: int) -> _NumbaType: ...
def Tuple(types: tuple[_NumbaType, ...]) -> _NumbaType: ...
