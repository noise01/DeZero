from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from dezero import utils
from dezero.core import Function, as_variable

if TYPE_CHECKING:
    from dezero.core import Variable


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        return gy * cos(x)


def sin(x: Variable) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        return gy * -sin(x)


def cos(x: Variable) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * (1 - y**2)


def tanh(x: Variable) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * y


def exp(x: Variable) -> Variable:
    return Exp()(x)


class Reshape(Function):
    def __init__(self, shape: tuple) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)


def reshape(x: Variable, shape: tuple) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes: tuple[int] = None) -> None:
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.transpose(x, self.axes)

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x: Variable, axes: tuple[int] = None) -> Variable:
    return Transpose(axes)(x)


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


class BroadcastTo(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy: Variable) -> Variable:
        return sum_to(gy, self.x_shape)


def broadcast_to(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


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


def linear(x: Variable, W: Variable, b: Variable = None):
    return Linear()(x, W, b)


def linear_simple(x: Variable, W: Variable, b: Variable = None):
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = as_variable(x)
        return 1 / (1 + exp(-x))

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        return gy * y * (1 - y)


def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)


def sigmoid_simple(x: Variable) -> Variable:
    x = as_variable(x)
    return 1 / (1 + exp(-x))


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


def mean_squared_error_simple(x0: Variable, x1: Variable) -> Variable:
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    return sum(diff**2) / len(diff)
