from __future__ import annotations
from typing import Any
import contextlib
import weakref

import numpy as np


class Config:
    enable_backprop = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name=None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name: str = name
        self.grad: Variable = None
        self.creator: Function = None
        self.generation = 0

    def set_creator(self, f: Function) -> None:
        self.creator = f
        self.generation = f.generation + 1

    def backward(self, retain_grad=False, create_graph=False) -> None:
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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

            with using_config("enable_backprop", create_graph):
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

    def clear_grad(self) -> None:
        self.grad = None

    @property
    def shape(self) -> tuple:
        return self.data.shape

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

    def __pow__(self, other: int) -> Variable:
        return pow(self, other)


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

    def backward(self, gy: Variable) -> tuple[Variable] | Variable:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: Variable) -> tuple[Variable]:
        return gy, gy


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: Variable) -> tuple[Variable]:
        x0, x1 = self.inputs
        return gy * x1, gy * x0


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: Variable) -> Variable:
        return -gy


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: Variable) -> tuple[Variable]:
        return gy, -gy


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 / x1

    def backward(self, gy: Variable) -> tuple[Variable]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c: int) -> None:
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**self.c

    def backward(self, gy: Variable) -> Variable:
        x, _ = self.inputs
        c = self.c
        return gy * c * x ** (c - 1)


@contextlib.contextmanager
def using_config(name: str, value: bool) -> None:
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


# no type hint
def no_grad():
    print(type(using_config("enable_backprop", False)))
    return using_config("enable_backprop", False)


# check type hint
def as_array(x: Any) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj: Variable | np.ndarray) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def add(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


def neg(x: Variable) -> Variable:
    return Neg()(x)


def sub(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x: Variable, c: int) -> Variable:
    return Pow(c)(x)


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    z = Variable(np.array(1.0))
    y = x * np.array(3.0)

    print(y)
