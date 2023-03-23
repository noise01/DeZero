from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dezero.core import Variable

from dezero import utils, cuda
from dezero.core import Function, as_variable


# =============================================================================
# Basic functions: sin, cos, tanh, exp, log
# =============================================================================
class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        return gy * cos(x)


def sin(x: Variable) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        return xp.cos(x)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        return gy * -sin(x)


def cos(x: Variable) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        return xp.tanh(x)

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * (1 - y**2)


def tanh(x: Variable) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * y


def exp(x: Variable) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        return xp.log(x)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        return gy / x


def log(x: Variable) -> Variable:
    return Log()(x)


# =============================================================================
# Tensor operations: reshape, transpose, get_item
# =============================================================================
class Reshape(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)


def reshape(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes: tuple[int] = None) -> None:
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.axes)

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x: Variable, axes: tuple[int] = None) -> Variable:
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices) -> None:
        self.slices = slices

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x[self.slices]

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices: np.ndarray, in_shape: tuple[int]) -> None:
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy: Variable) -> Variable:
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            cuda.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx: Variable) -> Variable:
        return get_item(ggx, self.slices)


def get_item(x: Variable, slices: np.ndarray) -> Variable:
    f = GetItem(slices)
    return f(x)


# =============================================================================
# sum, sum_to, broadcast_to, average, matmul, linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis: int, keepdims: bool) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy: Variable) -> Variable:
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


def sum(x: Variable, axis: int = None, keepdims=False) -> Variable:
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return broadcast_to(gy, self.x_shape)


def sum_to(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        return xp.broadcast_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return sum_to(gy, self.x_shape)


def broadcast_to(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        return x.dot(W)

    def backward(self, gy: Variable) -> tuple[Variable]:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x: Variable, W: Variable) -> Variable:
    return MatMul()(x, W)


def linear_simple(x: Variable, W: Variable, b: Variable = None) -> Variable:
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable) -> tuple[Variable]:
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x: Variable, W: Variable, b: Variable = None) -> Variable:
    return Linear()(x, W, b)


# =============================================================================
# Activation functions: sigmoid, softmax
# =============================================================================
def sigmoid_simple(x: Variable) -> Variable:
    x = as_variable(x)
    return 1 / (1 + exp(-x))


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        return xp.tanh(x + 0.5) * 0.5 + 0.5

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * y * (1 - y)


def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)


# ToDo: overflow
def softmax_simple(x: Variable, axis=1) -> Variable:
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1) -> None:
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y: np.ndarray = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy: Variable) -> Variable:
        y: Variable = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x: Variable, axis=1):
    return Softmax(axis)(x)


# =============================================================================
# Loss functions: mean_squared_error, softmax_cross_entropy_simple
# =============================================================================
def mean_squared_error_simple(x0: Variable, x1: Variable) -> Variable:
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    return sum(diff**2) / len(diff)


class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff = x0 - x1
        return (diff**2).sum() / len(diff)

    def backward(self, gy: Variable) -> tuple[Variable]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_simple(x: Variable, t: Variable) -> Variable:
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    return -1 * sum(tlog_p) / N


class SoftmaxCrossEntropy(Function):
    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p: np.ndarray = log_p[np.arange(N), t.ravel()]
        return -log_p.sum() / np.float32(N)

    def backward(self, gy: Variable) -> Variable:
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        return gy * (y - t_onehot)


def softmx_cross_entropy(x: Variable, t: Variable) -> Variable:
    return SoftmaxCrossEntropy()(x, t)


# =============================================================================
# max / min / clip
# =============================================================================
class Clip(Function):
    def __init__(self, x_min: float, x_max: float) -> None:
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = cuda.get_array_module(x)
        return xp.clip(x, self.x_min, self.x_max)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        return gy * mask


def clip(x: Variable, x_min: float, x_max: float) -> Variable:
    return Clip(x_min, x_max)(x)
