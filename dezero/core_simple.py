from __future__ import annotations
import contextlib
import weakref

import numpy as np


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: bool) -> None:
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad() -> None:
    return using_config("enable_backprop", False)


# =============================================================================
# Variable, Function
# =============================================================================
class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str = None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad: np.ndarray = None
        self.creator: Function = None
        self.generation = 0

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def __add__(self, other: Variable) -> Variable:
        return add(self, other)

    def __radd__(self, other: Variable) -> Variable:
        return add(self, other)

    def __mul__(self, other: Variable) -> Variable:
        return mul(self, other)

    def __rmul__(self, other: Variable) -> Variable:
        return mul(self, other)

    def __neg__(self) -> Variable:
        return neg(self)

    def __sub__(self, other: Variable) -> Variable:
        return sub(self, other)

    def __rsub__(self, other: Variable) -> Variable:
        return rsub(self, other)

    def __truediv__(self, other: Variable) -> Variable:
        return div(self, other)

    def __rtruediv__(self, other: Variable) -> Variable:
        return rdiv(self, other)

    def __pow__(self, other: float) -> Variable:
        return pow(self, other)

    def set_creator(self, f: Function) -> None:
        self.creator = f
        self.generation = f.generation + 1

    def clear_grad(self) -> None:
        self.grad = None

    def backward(self, retain_grad=False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        fs: list[Function] = []
        seen_set = set()

        def add_f(f: Function) -> None:
            if f not in seen_set:
                fs.append(f)
                seen_set.add(f)
                fs.sort(key=lambda x: x.generation)

        add_f(self.creator)
        while fs:
            f = fs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_f(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_variable(obj: Variable | np.ndarray) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x: int | float | np.ndarray) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray] | np.ndarray:
        raise NotImplementedError()


# =============================================================================
# Arithmetic operations: Add, Mul, Sub, Div, Neg, Pow
# =============================================================================
class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        return gy, gy


def add(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


def neg(x: Variable) -> Variable:
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        return gy, -gy


def sub(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 / x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c: int) -> None:
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**self.c

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        c = self.c
        return gy * c * x ** (c - 1)


def pow(x: Variable, c: float) -> Variable:
    return Pow(c)(x)
