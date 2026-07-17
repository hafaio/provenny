"""Shared numpy array and fixed-size tuple aliases used across the internal modules."""

import numpy as np
from numpy.typing import NDArray

F64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
C128Array = NDArray[np.complex128]
U8Array = NDArray[np.uint8]
I64Array = NDArray[np.int64]

# tuples not arrays: a tuple is registers, an array a heap allocation (see provenny._kernel)
Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
CVec3 = tuple[complex, complex, complex]
CMat3 = tuple[CVec3, CVec3, CVec3]
