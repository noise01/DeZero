gpu_enable = True
import numpy as np

try:
    import cupy as cp
    import cupyx as cpx

    cupy = cp
except ImportError:
    gpu_enable = False
from dezero import Variable


def get_array_module(x: Variable | np.ndarray | cp.ndarray):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x: Variable | np.ndarray | cp.ndarray) -> np.ndarray:
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x: Variable | np.ndarray | cp.ndarray) -> cp.ndarray:
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception("CuPy cannot be loaded. Install CuPy!")
    return cp.asarray(x)


def scatter_add(a: cp.ndarray, slices: cp.ndarray, value: cp.ndarray) -> None:
    return cpx.scatter_add(a, slices, value)
