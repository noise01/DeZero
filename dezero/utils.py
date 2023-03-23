from __future__ import annotations
from typing import TYPE_CHECKING
import os
import subprocess

import numpy as np

from dezero import Variable, cuda

if TYPE_CHECKING:
    from dezero import Function


def _dot_var(v: Variable, verbose=False) -> str:
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_f(f: Function) -> str:
    dot_f = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_f.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))

    return txt


def get_dot_graph(output: Variable, verbose=False) -> str:
    txt = ""
    fs: list[Function] = []
    seen_set = set()

    def add_f(f: Function) -> None:
        if f not in seen_set:
            fs.append(f)
            seen_set.add(f)

    add_f(output.creator)
    txt += _dot_var(output, verbose)

    while fs:
        f = fs.pop()
        txt += _dot_f(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_f(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output: Variable, verbose=True, to_file="graph.png") -> None:
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


def sum_to(x: np.ndarray, shape: tuple[int]) -> np.ndarray:
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y: np.ndarray = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(
    gy: Variable, x_shape: tuple[int], axis: tuple[int] | int, keepdims: bool
) -> Variable:
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy


def logsumexp(x: np.ndarray, axis=1) -> np.ndarray:
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m
