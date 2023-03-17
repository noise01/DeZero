from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np

if TYPE_CHECKING:
    from dezero import Layer
    from dezero import Model
    from dezero import Parameter


class Optimizer:
    def __init__(self) -> None:
        self.target: Model | Layer = None
        self.hooks: list[Callable] = []

    def setup(self, target: Model | Layer) -> Optimizer:
        self.target = target
        return self

    def update(self) -> None:
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param: Parameter) -> None:
        raise NotImplementedError()

    def add_hook(self, f: Callable) -> None:
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__()

        self.lr = lr

    def update_one(self, param: Parameter) -> None:
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9) -> None:
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param: Parameter) -> None:
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
