from __future__ import annotations
from typing import TYPE_CHECKING
import weakref

import numpy as np

from dezero.core import Parameter
import dezero.functions as F

if TYPE_CHECKING:
    from dezero.core import Variable


class Layer:
    def __init__(self) -> None:
        self._params = set()

    def __setattr__(self, __name: str, __value: Variable | Parameter) -> None:
        if isinstance(__value, (Parameter, Layer)):
            self._params.add(__name)
        super().__setattr__(__name, __value)

    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x: Variable) -> Variable:
        raise NotImplementedError()

    def params(self) -> list[Parameter]:
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def clear_grads(self) -> None:
        for param in self.params():
            param.clear_grad()

    def to_cpu(self) -> None:
        for param in self.params():
            param.to_cpu()

    def to_gpu(self) -> None:
        for param in self.params():
            param.to_gpu()


class Linear(Layer):
    def __init__(
        self, out_size: int, nobias=False, dtype=np.float32, in_size: int = None
    ) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self) -> None:
        W_data = np.random.randn(self.in_size, self.out_size).astype(
            self.dtype
        ) * np.sqrt(1 / self.in_size)
        self.W.data = W_data

    def forward(self, x: Variable) -> Variable:
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        return F.linear(x, self.W, self.b)
