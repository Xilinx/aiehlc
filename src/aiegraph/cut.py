###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Cut an ONNX graph before a tail op (``cut_before(model, "TopK")``).

The offloaded part of a model ends where its post-processing tail starts (e.g. a detector's
TopK / gather selection, which stays on the host). The cut tensor is found on the graph:

  1. the *dominators* of the tail: tensors every path from the graph inputs to every
     ``op_type`` node goes through (a node whose removal disconnects all of them);
  2. the latest dominator in graph order, then stepped back over pure layout ops
     (Transpose / Reshape / Split / Flatten / Squeeze / Unsqueeze / Identity), so the cut
     output is the last computed tensor, not a view of it.

The model up to that tensor is extracted with ``onnx.utils.Extractor`` (graph inputs kept).
"""

from __future__ import annotations

from pathlib import Path

from .errors import AiegraphError

__all__ = ["LAYOUT_OPS", "cut_tensor", "cut_before"]

LAYOUT_OPS = ("Transpose", "Reshape", "Split", "Flatten", "Squeeze", "Unsqueeze", "Identity")


def _model(model):
    import onnx
    if isinstance(model, (str, Path)):
        return onnx.load(str(model))
    return model


def cut_tensor(model, op_type: str = "TopK") -> str:
    """The tensor the graph is cut at, before every ``op_type`` node (see module doc)."""
    g = _model(model).graph
    nodes = list(g.node)
    prod = {o: i for i, n in enumerate(nodes) for o in n.output}
    cons = {}
    for i, n in enumerate(nodes):
        for x in n.input:
            cons.setdefault(x, []).append(i)
    tail = [i for i, n in enumerate(nodes) if n.op_type == op_type]
    if not tail:
        raise AiegraphError("cut", f"no {op_type} node in the graph", hint="nothing to cut")
    anc, st = set(), list(tail)
    while st:
        i = st.pop()
        if i in anc:
            continue
        anc.add(i)
        st += [prod[x] for x in nodes[i].input if x in prod]
    init = {t.name for t in g.initializer}
    starts = [j for v in g.input if v.name not in init for j in cons.get(v.name, [])]
    tset = set(tail)

    def reaches_tail(skip: int) -> bool:
        seen, st = set(), list(starts)
        while st:
            j = st.pop()
            if j == skip or j in seen:
                continue
            if j in tset:
                return True
            seen.add(j)
            for o in nodes[j].output:
                st += cons.get(o, [])
        return False

    doms = [i for i in sorted(anc - tset) if not reaches_tail(i)]
    if not doms:
        raise AiegraphError("cut", f"no single tensor dominates the {op_type} tail")
    i = doms[-1]
    while nodes[i].op_type in LAYOUT_OPS and nodes[i].input and nodes[i].input[0] in prod:
        i = prod[nodes[i].input[0]]
    return nodes[i].output[0]


def cut_before(model, op_type: str = "TopK", out=None):
    """``(ModelProto, cut tensor)``: *model* up to :func:`cut_tensor`; saved to *out* if given."""
    import onnx
    from onnx.utils import Extractor
    m = _model(model)
    t = cut_tensor(m, op_type)
    m = onnx.shape_inference.infer_shapes(m)
    init = {x.name for x in m.graph.initializer}
    sub = Extractor(m).extract_model([v.name for v in m.graph.input if v.name not in init], [t])
    if out is not None:
        onnx.save(sub, str(out))
    return sub, t
