import os
import subprocess

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]'

    name = "" if v.name is None else v.name

    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    # id()でPyObjectのIDを取得できる.
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"

    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # yはweakref

    return txt


def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            seen_set.add(f)
            funcs.append(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n " + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    os.makedirs(tmp_dir, exist_ok=True)

    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    # dotコマンドを実行
    extension = os.path.splitext(to_file)[1][1:]
    cmd = f"dot {graph_path} -T {extension} -o {to_file}"
    subprocess.run(cmd, shell=True)