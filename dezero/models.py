from __future__ import annotations
from typing import TYPE_CHECKING

import dezero.functions as F
import dezero.layers as L

from dezero import Layer
from dezero import utils

if TYPE_CHECKING:
    from dezero.core import Variable


class Model(Layer):
    def plot(self, *inputs: Variable, to_file="model.png") -> None:
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(
        self, fc_output_sizes: tuple[int] | list[int], activation=F.sigmoid_simple
    ) -> None:
        super().__init__()

        self.activation = activation
        self.layers: list[Layer] = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x: Variable) -> Variable:
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
