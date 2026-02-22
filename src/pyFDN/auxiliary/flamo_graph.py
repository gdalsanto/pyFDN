"""
Traverse a FLAMO model and build a node tree or visualize it as a flowchart.

Flow: left-to-right. Series and Parallel are shown as boxes with nested modules.
Recursion is shown with forward path (fF) and feedback path (fB) inside a box,
with the feedback path drawn so the loop is visible (e.g. fB below fF).
"""

from __future__ import annotations

from typing import Any

# Traversal uses type(module).__name__ and getattr; no need to import flamo here.


def _get_typename(module: Any) -> str:
    return type(module).__name__


def _series_children(module: Any) -> list[tuple[str, Any]]:
    """Return list of (name, submodule) for Series-like module."""
    # FLAMO Series is nn.Sequential; direct children are in ._modules (OrderedDict)
    modules = getattr(module, "_modules", None)
    if modules is None:
        modules = getattr(module, "modules", None)
    if modules is None:
        return []
    if hasattr(modules, "items"):
        return list(modules.items())
    if hasattr(modules, "__iter__") and not isinstance(modules, (str, bytes)):
        return [(str(i), m) for i, m in enumerate(modules)]
    return []


def _is_shell(module: Any) -> bool:
    return _get_typename(module) == "Shell"


def _is_series(module: Any) -> bool:
    return _get_typename(module) == "Series"


def _is_parallel(module: Any) -> bool:
    return _get_typename(module) == "Parallel"


def _is_recursion(module: Any) -> bool:
    return _get_typename(module) == "Recursion"


def flamo_model_to_nodes(
    model: Any,
    name: str = "root",
    *,
    include_shell_io: bool = False,
) -> dict[str, Any]:
    """
    Traverse a FLAMO model and build a tree of nodes (nested dicts).

    Each node has:
    - type: "Shell" | "Series" | "Parallel" | "Recursion" | "Leaf"
    - name: str (from parent's dict key or assigned)
    - module: the raw FLAMO module (for Leaf, the actual dsp module)
    - children: list of child nodes (for Series, Parallel; order preserved)
    - fF, fB: only for Recursion — nodes for forward and feedback path
    - input_layer, output_layer: only if include_shell_io and type is Shell

    Parameters
    ----------
    model : FLAMO model (Shell, Series, Parallel, Recursion, or dsp module)
    name : str
        Name for the root node.
    include_shell_io : bool
        If True, include input_layer and output_layer as children for Shell.

    Returns
    -------
    node : dict
        Root node (nested tree). Use flamo_nodes_flat() to get a list of all nodes.
    """
    node: dict[str, Any] = {
        "type": "Leaf",
        "name": name,
        "module": model,
        "children": [],
    }
    tname = _get_typename(model)

    if _is_shell(model):
        node["type"] = "Shell"
        # FLAMO Shell stores core as __core; use get_core() if available
        core = None
        if callable(getattr(model, "get_core", None)):
            core = model.get_core()
        if core is None:
            core = getattr(model, "core", None)
        if core is None:
            core = getattr(model, "_Shell__core", None)
        if core is not None:
            node["children"] = [flamo_model_to_nodes(core, name="core", include_shell_io=include_shell_io)]
        # Always attach input/output layers for Shell so draw can use them as model I/O
        il = None
        if callable(getattr(model, "get_inputLayer", None)):
            il = model.get_inputLayer()
        if il is None:
            il = getattr(model, "input_layer", None) or getattr(model, "_Shell__input_layer", None)
        ol = None
        if callable(getattr(model, "get_outputLayer", None)):
            ol = model.get_outputLayer()
        if ol is None:
            ol = getattr(model, "output_layer", None) or getattr(model, "_Shell__output_layer", None)
        if il is not None:
            node.setdefault("input_layer", flamo_model_to_nodes(il, name="input_layer", include_shell_io=include_shell_io))
        if ol is not None:
            node.setdefault("output_layer", flamo_model_to_nodes(ol, name="output_layer", include_shell_io=include_shell_io))
        return node

    if _is_series(model):
        node["type"] = "Series"
        pairs = _series_children(model)
        node["children"] = [
            flamo_model_to_nodes(sub, name=nm, include_shell_io=include_shell_io)
            for nm, sub in pairs
        ]
        return node

    if _is_parallel(model):
        node["type"] = "Parallel"
        brA = getattr(model, "brA", None) or getattr(model, "branchA", None)
        brB = getattr(model, "brB", None) or getattr(model, "branchB", None)
        children = []
        if brA is not None:
            children.append(flamo_model_to_nodes(brA, name="brA", include_shell_io=include_shell_io))
        if brB is not None:
            children.append(flamo_model_to_nodes(brB, name="brB", include_shell_io=include_shell_io))
        node["children"] = children
        return node

    if _is_recursion(model):
        node["type"] = "Recursion"
        fF = getattr(model, "fF", None) or getattr(model, "feedforward", None)
        fB = getattr(model, "fB", None) or getattr(model, "feedback", None)
        node["fF"] = flamo_model_to_nodes(fF, name="fF", include_shell_io=include_shell_io) if fF is not None else None
        node["fB"] = flamo_model_to_nodes(fB, name="fB", include_shell_io=include_shell_io) if fB is not None else None
        node["children"] = []  # Recursion uses fF/fB, not children
        return node

    # Leaf (Gain, Delay, FFT, etc.)
    return node


def flamo_nodes_flat(
    root: dict[str, Any],
    path: str = "root",
) -> list[dict[str, Any]]:
    """
    Flatten the node tree into a list of nodes, each with a 'path' key.

    Parameters
    ----------
    root : dict
        Root node from flamo_model_to_nodes().
    path : str
        Path prefix for the root.

    Returns
    -------
    list of dict
        Each dict has keys from the node plus "path" (e.g. "root/core/feedback_loop/fF").
    """
    out: list[dict[str, Any]] = []
    node = {k: v for k, v in root.items() if k not in ("fF", "fB")}
    node["path"] = path
    out.append(node)

    for i, ch in enumerate(root.get("children") or []):
        subpath = f"{path}/{ch['name']}"
        out.extend(flamo_nodes_flat(ch, path=subpath))

    if root.get("type") == "Recursion":
        for key in ("fF", "fB"):
            child = root.get(key)
            if child is not None:
                subpath = f"{path}/{key}"
                out.extend(flamo_nodes_flat(child, path=subpath))
    return out


def draw_flamo_graph(
    model: Any,
    *,
    name: str = "flamo",
    direction: str = "TB",
    format: str = "png",
    include_shell_io: bool = False,
    engine: str = "dot",
) -> Any:
    """
    Draw a flowchart of the FLAMO model (top-to-bottom by default for compactness).

    Series and Parallel are boxes containing their child modules.
    Recursion is a box with forward path (fF) and feedback path (fB),
    with a dashed edge from end of fF back to start of fB to show the loop.

    Requires:
      1. The Python package: pip install graphviz
      2. The Graphviz system binaries (the pip package alone is not enough).
         Install the native package too, e.g.:
         - macOS: brew install graphviz
         - Ubuntu/Debian: sudo apt-get install graphviz
         - Windows: install from https://graphviz.org/download/ and add bin to PATH

    Parameters
    ----------
    model : FLAMO model
    name : str
        Name for the graph / output file.
    direction : str
        Graph direction: "TB" (top-bottom, default), "LR" (left-to-right), etc.
    format : str
        Output format: "png", "svg", "pdf", etc.
    include_shell_io : bool
        If True, include Shell's input_layer and output_layer in the graph.
    engine : str
        Graphviz engine: "dot", "neato", etc.

    Returns
    -------
    graph : graphviz.Digraph
        Call .render() or .view() to save/display.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "draw_flamo_graph requires the Python package graphviz (pip install graphviz)."
        )

    try:
        graphviz.Digraph().pipe(format="png")  # probe for dot executable
    except Exception as e:
        if "ExecutableNotFound" in type(e).__name__ or (
            "executable" in str(e).lower() and "path" in str(e).lower()
        ):
            raise RuntimeError(
                "Graphviz executables (e.g. dot) not found. "
                "Install the system Graphviz package: "
                "macOS: brew install graphviz; "
                "Ubuntu/Debian: sudo apt-get install graphviz; "
                "Windows: https://graphviz.org/download/"
            ) from e
        raise

    root = flamo_model_to_nodes(model, name=name, include_shell_io=include_shell_io)
    digraph = graphviz.Digraph(name=name, format=format, engine=engine)
    # Default layout: TB (top-bottom) for compactness; edge labels small
    digraph.attr(
        rankdir=direction,
        splines="polyline",
        nodesep="0.15",
        ranksep="0.2",
        fontsize="8",
        margin="0.1,0.05",
        pad="0.2",
    )
    _node_id = [0]

    def next_id() -> str:
        _node_id[0] += 1
        return f"n{_node_id[0]}"

    def add_node(
        dg: graphviz.Digraph,
        nid: str,
        label: str,
        shape: str = "box",
        style: str = "",
    ) -> None:
        attrs = {"label": label, "shape": shape, "fontsize": "8", "margin": "0.05,0.02"}
        if style:
            attrs["style"] = style
        dg.node(nid, **attrs)

    def build_graph(
        dg: graphviz.Digraph,
        node: dict[str, Any],
        parent_id: str | None,
        edge_label: str = "",
    ) -> tuple[str, str]:
        """Return (first_id, last_id) of the added subgraph for this node."""
        ntype = node.get("type", "Leaf")
        nname = node.get("name", "?")

        if ntype == "Shell":
            # Box: input_layer → core → output_layer (Shell layers = model I/O)
            cid = next_id()
            with dg.subgraph(name=f"cluster_{cid}") as sub:
                sub.attr(label=f"Shell: {nname}", style="rounded", margin="5", fontsize="8")
                first_id = last_id = None
                prev = None
                il_node = node.get("input_layer")
                if il_node is not None:
                    first_id, prev = build_graph(sub, il_node, None, edge_label="")
                for ch in node.get("children") or []:
                    ch_first, ch_last = build_graph(sub, ch, prev, edge_label="")
                    if first_id is None:
                        first_id = ch_first
                    prev = ch_last
                    last_id = ch_last
                ol_node = node.get("output_layer")
                if ol_node is not None:
                    _, last_id = build_graph(sub, ol_node, prev, edge_label="")
                if first_id is None:
                    first_id = next_id()
                    add_node(sub, first_id, "", shape="point")
                    last_id = last_id or first_id
                if last_id is None:
                    last_id = first_id
            if parent_id is not None:
                dg.edge(parent_id, first_id, label=edge_label, fontsize="7")
            return first_id, last_id

        if ntype == "Series":
            # Box: chain only (no in/out nodes)
            cid = next_id()
            with dg.subgraph(name=f"cluster_{cid}") as sub:
                sub.attr(label=f"Series: {nname}", style="rounded", margin="5", fontsize="8")
                first_id = last_id = None
                prev = parent_id
                for ch in node.get("children") or []:
                    ch_first, ch_last = build_graph(sub, ch, prev, edge_label="")
                    if first_id is None:
                        first_id = ch_first
                    prev = ch_last
                    last_id = ch_last
                if first_id is None:
                    first_id = next_id()
                    add_node(sub, first_id, "", shape="point")
                    if parent_id is not None:
                        dg.edge(parent_id, first_id, label=edge_label, fontsize="7")
                if last_id is None:
                    last_id = first_id
            return first_id, last_id

        if ntype == "Parallel":
            # Box: parent → brA, parent → brB; both → merge (no in/out labels)
            cid = next_id()
            with dg.subgraph(name=f"cluster_{cid}") as sub:
                sub.attr(label=f"Parallel: {nname}", style="rounded", margin="5", fontsize="8")
                merge_id = next_id()
                add_node(sub, merge_id, "Σ", shape="circle")
                for ch in node.get("children") or []:
                    ch_first, ch_last = build_graph(sub, ch, parent_id, edge_label=ch.get("name", ""))
                    sub.edge(ch_last, merge_id)
            return merge_id, merge_id

        if ntype == "Recursion":
            # Box: parent → fF; fF_last → fB → fF_first (feedback); no in/out nodes
            cid = next_id()
            with dg.subgraph(name=f"cluster_{cid}") as sub:
                sub.attr(label=f"Recursion: {nname}", style="rounded,dashed", margin="5", fontsize="8")
                fF_node = node.get("fF")
                fB_node = node.get("fB")
                fF_first = fF_last = None
                if fF_node is not None:
                    fF_first, fF_last = build_graph(sub, fF_node, parent_id, edge_label=edge_label)
                if fB_node is not None:
                    _, fB_last = build_graph(sub, fB_node, fF_last or parent_id, edge_label="fB")
                    if fF_first is not None:
                        sub.edge(fB_last, fF_first, label="feedback", style="dashed", color="gray", fontsize="7")
                first_id = fF_first if fF_first is not None else next_id()
                last_id = fF_last if fF_last is not None else first_id
                if fF_first is None and parent_id is not None:
                    add_node(sub, first_id, "", shape="point")
                    dg.edge(parent_id, first_id, label=edge_label, fontsize="7")
            return first_id, last_id

        # Leaf: single node; label with actual module type (Gain, Delay, FFT, etc.)
        nid = next_id()
        mod = node.get("module")
        mod_type = type(mod).__name__ if mod is not None else "?"
        label = f"{nname}\n({mod_type})"
        add_node(dg, nid, label, shape="box")
        if parent_id is not None:
            dg.edge(parent_id, nid, label=edge_label, fontsize="7")
        return nid, nid

    build_graph(digraph, root, None)
    return digraph
