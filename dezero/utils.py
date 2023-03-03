from __future__ import annotations
from typing import TYPE_CHECKING
import os
import subprocess

if TYPE_CHECKING:
    from core_simple import Variable, Function


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


def get_dot_graph(output: Variable, verbose=False):
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
