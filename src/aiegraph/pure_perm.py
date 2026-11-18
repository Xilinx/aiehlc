###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Pure-permutation check on saved Relay op semantics (m4 gap 2).

The keep-layout codegen may only delete a TVM call whose work is a data permutation (reshape / transpose /
layout_transform / squeeze / expand_dims), optionally with a small set of element-wise ops the rewrite folds
elsewhere (the attention scale ``multiply``, the residual ``add``, the time-embedding SiLU). The decision used to parse
the fused *symbol name* (``tvmgen_default_fused_reshape_transpose_...``); names are truncated past 80
characters, carry no shapes / axes, and say nothing about the number of outputs.

Now the decision is made on the **Relay op semantics** of each fused function, recorded when the graph is
built (``frontend.tvmrelay.deploy_flow.build_c``, the module right before ``LowerTE``) and saved next to the
graph as ``<stem>_opsem.json`` -- beside ``<stem>_graph.json`` that ``group.CallGraph`` reads:

    {"version": 1, "nodes": {"<graph.json node id>": {
        "func_name": "tvmgen_default_fused_...",    # must equal the graph.json node (alignment check)
        "name_ok": true,                             # the readable name TVM derives from the ops matches
        "params": [[shape], ...], "outs": [[shape], ...],
        "ops": [{"op": "transpose", "ins": [[shape], ...], "out": [shape], "attrs": {"axes": [...]}}, ...]}}}

``check(cg, call, extra, need)`` is pure iff: the sidecar entry exists and matches the call; one output;
every op is a label op (element count conserved, transpose ``axes`` a permutation) or listed in ``extra``
(element-wise: output shape == its largest input); ``need`` ops are present; and the function output has as
many elements as its largest parameter. Anything else -> AiegraphError("... not a pure permutation ...").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AiegraphError

__all__ = ["LABEL", "FusedFn", "OpSem", "opsem_path", "load", "check", "why_impure", "index_map", "permutation",
           "record_relay", "save_relay", "relay_capture"]

_STAGE = "pure_perm"
VERSION = 1
LABEL = frozenset({"reshape", "transpose", "layout_transform", "squeeze", "expand_dims"})
ELEMWISE = frozenset({"multiply", "add", "subtract", "divide", "sigmoid"})


def _n(shape) -> int:
    n = 1
    for d in shape or ():
        n *= int(d)
    return n


@dataclass
class FusedFn:
    """One fused TVM function as Relay saw it: ops in lowering order, parameter / output shapes."""
    func_name: str
    ops: list                    # [{"op", "ins", "out", "attrs"}]
    params: list = field(default_factory=list)
    outs: list = field(default_factory=list)
    name_ok: bool = True

    @property
    def op_names(self) -> list:
        return [o["op"] for o in self.ops]

    @classmethod
    def from_json(cls, d: dict) -> "FusedFn":
        return cls(d["func_name"], list(d["ops"]), [list(p) for p in d.get("params", [])],
                   [list(o) for o in d.get("outs", [])], bool(d.get("name_ok", True)))

    def to_json(self) -> dict:
        return {"func_name": self.func_name, "name_ok": self.name_ok, "params": self.params, "outs": self.outs,
                "ops": self.ops}


def why_impure(fn: FusedFn, extra=(), need=()) -> str:
    """'' if ``fn`` only permutes data (+ the element-wise ops in ``extra``), else the reason."""
    allowed = LABEL | set(extra)
    names = fn.op_names
    if not names:
        return "no ops recorded"
    bad = [n for n in names if n not in allowed]
    if bad:
        return f"ops {names}: {sorted(set(bad))} not allowed (allowed {sorted(allowed)})"
    miss = [n for n in need if n not in names]
    if miss:
        return f"ops {names}: missing required {miss}"
    if len(fn.outs) != 1:
        return f"{len(fn.outs)} outputs (a permutation has one)"
    for i, o in enumerate(fn.ops):
        ins = [list(s) for s in o.get("ins", [])]
        out = list(o.get("out", []))
        if not ins:
            return f"op {i} {o['op']}: no inputs recorded"
        if o["op"] in LABEL:
            if _n(ins[0]) != _n(out):
                return f"op {i} {o['op']}: element count not conserved {ins[0]} -> {out}"
            if o["op"] == "transpose":
                axes = (o.get("attrs") or {}).get("axes")
                nd = len(ins[0])
                axes = list(range(nd))[::-1] if axes is None else [int(a) % max(nd, 1) for a in axes]
                if sorted(axes) != list(range(nd)):
                    return f"op {i} transpose: axes {axes} not a permutation of rank {nd}"
        else:   # element-wise op from ``extra``: no reduction, no broadcast growth beyond its largest input
            if _n(out) != max(_n(s) for s in ins):
                return f"op {i} {o['op']}: element count not conserved {ins} -> {out}"
    big = max((_n(p) for p in fn.params), default=0)
    if _n(fn.outs[0]) != big:
        return f"output {fn.outs[0]} has {_n(fn.outs[0])} elements, largest input {big}"
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  the saved semantics, next to the CallGraph
# ═══════════════════════════════════════════════════════════════════════════

def opsem_path(graph_json) -> Path:
    p = Path(graph_json)
    name = p.name[:-3] if p.name.endswith(".gz") else p.name
    stem = name[:-len("_graph.json")] if name.endswith("_graph.json") else Path(name).stem
    return p.with_name(stem + "_opsem.json")


class OpSem:
    """``<stem>_opsem.json``: graph.json node id -> FusedFn."""

    def __init__(self, doc: dict, path=None):
        self.path = path
        if doc.get("version") != VERSION:
            raise AiegraphError(_STAGE, f"{path}: opsem version {doc.get('version')} != {VERSION}",
                                hint="re-convert the model (ag.convert) to re-record the Relay op semantics")
        self.nodes = {int(k): FusedFn.from_json(v) for k, v in doc["nodes"].items()}

    @classmethod
    def for_graph(cls, cg) -> "OpSem":
        sem = getattr(cg, "_opsem", None)
        if sem is not None:
            return sem
        if cg.path is None:
            raise AiegraphError(_STAGE, "CallGraph has no path: cannot find its <stem>_opsem.json")
        p = opsem_path(cg.path)
        if not p.is_file():
            raise AiegraphError(_STAGE, f"no Relay op semantics {p.name} beside {cg.path.name}",
                                hint="re-convert the model: frontend.tvmrelay.deploy_flow.build_c records it "
                                     "next to <stem>_graph.json")
        sem = cls(json.loads(p.read_text()), p)
        cg._opsem = sem
        return sem

    def fn(self, cg, call: int) -> FusedFn:
        c = cg.calls[call]
        f = self.nodes.get(c.node)
        if f is None or f.func_name != c.symbol:
            raise AiegraphError(_STAGE, f"call [{call}] {c.fn}: {self.path.name if self.path else 'opsem'} has "
                                        f"{f.func_name if f else 'no entry'} for graph.json node {c.node}",
                                hint="the op semantics are stale: re-convert the model")
        return f


def load(cg) -> OpSem:
    return OpSem.for_graph(cg)


def check(cg, call: int, extra=(), need=(), stage: str = _STAGE) -> list:
    """The call only permutes data (+ ``extra`` element-wise ops, with every op of ``need`` present), decided on
    the saved Relay op semantics. Returns the op names (lowering order); raises AiegraphError otherwise."""
    c = cg.calls[call]
    f = OpSem.for_graph(cg).fn(cg, call)
    why = "" if f.name_ok else "recorded ops do not match the fused symbol (alignment)"
    why = why or why_impure(f, extra, need)
    if not why and len(c.outs) != 1:
        why = f"{len(c.outs)} outputs in graph.json"
    if why:
        raise AiegraphError(stage, f"call [{call}] {c.fn}: not a pure permutation ({why})",
                            hint="the call does not match the keep-layout pattern: build without keep_layout "
                                 "or fix the partition")
    return f.op_names


# ═══════════════════════════════════════════════════════════════════════════
#  the permutation itself: which input element lands where
# ═══════════════════════════════════════════════════════════════════════════

def _axes(layout: str) -> list:
    """``"NCHW4c"`` -> [("N", 0), ("C", 0), ("H", 0), ("W", 0), ("c", 4)] (sub-axis factor, 0 = primal)."""
    out, num = [], ""
    for ch in layout:
        if ch.isdigit():
            num += ch
            continue
        if ch.isupper() != (num == ""):
            raise ValueError(f"layout {layout!r}")
        out.append((ch, int(num) if num else 0))
        num = ""
    if num:
        raise ValueError(f"layout {layout!r}")
    return out


def _to_primal(a, layout: str):
    """array in *layout* -> (array in its primal axes, primal letters in src order)"""
    import numpy as np
    ax = _axes(layout)
    prim = [c for c, f in ax if f == 0]
    if sorted(c.upper() for c, f in ax if f) != sorted(set(c.upper() for c, f in ax if f)) or \
            any(c.upper() not in prim for c, f in ax if f):
        raise ValueError(f"layout {layout!r}")
    if a.ndim != len(ax):
        raise ValueError(f"layout {layout!r} vs rank {a.ndim}")
    order, shape = [], []
    for p in prim:
        i = [k for k, (c, f) in enumerate(ax) if c == p][0]
        sub = [k for k, (c, f) in enumerate(ax) if f and c == p.lower()]
        order.append(i)
        shape.append(a.shape[i])
        if sub:
            order.append(sub[0])
            shape[-1] *= a.shape[sub[0]]
    return np.ascontiguousarray(a.transpose(order)).reshape(shape), prim


def _from_primal(a, prim: list, layout: str):
    import numpy as np
    ax = _axes(layout)
    dprim = [c for c, f in ax if f == 0]
    if sorted(dprim) != sorted(prim):
        raise ValueError(f"layout {layout!r} vs primal {prim}")
    a = a.transpose([prim.index(c) for c in dprim])
    split, names = [], []
    for k, p in enumerate(dprim):
        sub = [f for c, f in ax if f and c == p.lower()]
        if sub:
            if a.shape[k] % sub[0]:
                raise ValueError(f"layout {layout!r}: {p}={a.shape[k]} not a multiple of {sub[0]}")
            split += [a.shape[k] // sub[0], sub[0]]
            names += [p, p.lower()]
        else:
            split.append(a.shape[k])
            names.append(p)
    a = a.reshape(split)
    return np.ascontiguousarray(a.transpose([names.index(c) for c, f in ax]))


def index_map(fn: FusedFn, param: int = 0):
    """Flat index map of a pure permutation: ``out.flat[i] == params[param].flat[m[i]]`` (int64 array), by
    running the recorded ops on ``arange`` of that parameter. Element-wise ops (``multiply`` / ``add`` with
    an operand of the same or a broadcast shape) leave the map as it is; ops on another operand (a scale chain
    whose shapes differ from the tracked one) are skipped. ``None`` when the ops do not give a map of the
    output (an op this simulator does not know, a shape that never reaches the output)."""
    import numpy as np
    if not 0 <= param < len(fn.params) or len(fn.outs) != 1:
        return None
    a = np.arange(_n(fn.params[param]), dtype=np.int64).reshape(fn.params[param] or [1])
    try:
        for o in fn.ops:
            ins = [list(x) for x in o.get("ins", [])]
            out = list(o.get("out", []))
            if not ins or list(a.shape) != ins[0]:
                if o["op"] in ELEMWISE and any(list(a.shape) == x for x in ins[1:]) and list(a.shape) == out:
                    continue            # the tracked tensor is a later operand of a same-shape element-wise op
                continue                # an op on another operand chain
            op, at = o["op"], o.get("attrs") or {}
            if op in ("reshape", "squeeze", "expand_dims"):
                a = a.reshape(out or [1])
            elif op == "transpose":
                axes = at.get("axes")
                a = np.ascontiguousarray(a.transpose(None if axes is None else [int(x) % a.ndim for x in axes]))
            elif op == "layout_transform":
                p, prim = _to_primal(a, at["src_layout"])
                a = _from_primal(p, prim, at["dst_layout"])
            elif op in ELEMWISE:
                if out != ins[0]:
                    return None         # broadcast growth of the tracked tensor
            else:
                return None
            if list(a.shape) != out:
                return None
    except (KeyError, ValueError):
        return None
    if list(a.shape) != list(fn.outs[0]):
        return None
    return a.reshape(-1)


def permutation(cg, call: int, param: int = 0):
    """``index_map`` of call *call*'s saved op semantics for its *param*-th input."""
    return index_map(OpSem.for_graph(cg).fn(cg, call), param)


# ═══════════════════════════════════════════════════════════════════════════
#  recording (TVM side, called from deploy_flow.build_c)
# ═══════════════════════════════════════════════════════════════════════════

def relay_capture():
    """A PassInstrument keeping the module right before LowerTE (``.module``): every fused primitive function
    with its Relay ops, types and attrs, in the structure the graph executor codegen walks."""
    import tvm

    @tvm.instrument.pass_instrument
    class _Capture:
        def __init__(self):
            self.module = None

        def run_before_pass(self, mod, info):
            if info.name == "LowerTE":
                self.module = mod

    return _Capture()


def _shape(ty) -> list:
    return [int(d) for d in ty.shape]


def _outs(ty) -> list:
    from tvm import relay
    if isinstance(ty, relay.TupleType):
        return [o for f in ty.fields for o in _outs(f)]
    return [_shape(ty)]


def _attrs(call) -> dict:
    a = call.attrs
    if a is None:
        return {}
    out = {}
    for k in ("axes", "newshape", "src_layout", "dst_layout", "axis", "indices_or_sections"):
        try:
            v = a[k] if k in a.keys() else None
        except Exception:   # noqa: BLE001 - attrs without keys()
            v = getattr(a, k, None)
        if v is None:
            continue
        try:
            out[k] = [int(x) for x in v]
        except TypeError:
            out[k] = str(v) if not isinstance(v, int) else int(v)
        except Exception:   # noqa: BLE001
            out[k] = str(v)
    return out


def _fused(func) -> tuple:
    """ops of one primitive function in lowering order (args first, as te_compiler_cache's readable name)"""
    import tvm
    from tvm import relay
    ops = []

    def visit(e):
        if isinstance(e, relay.Call) and isinstance(e.op, tvm.ir.Op):
            ins = []
            for a in e.args:
                t = a.checked_type
                ins.extend(_outs(t))
            out = _outs(e.checked_type)
            ops.append({"op": e.op.name, "ins": ins, "out": out[0] if len(out) == 1 else out, "attrs": _attrs(e)})

    relay.analysis.post_order_visit(func.body, visit)
    params = [o for p in func.params for o in _outs(p.checked_type)]
    return ops, params, _outs(func.ret_type)


def record_relay(module, graph_json: str) -> dict:
    """Pair every primitive call of ``module['main']`` (post-order, as the graph executor codegen numbers its
    nodes) with the ``tvm_op`` nodes of ``graph_json``; returns the opsem document."""
    import tvm
    from tvm import relay
    g = json.loads(graph_json)
    tnodes = [i for i, n in enumerate(g["nodes"]) if n.get("op") == "tvm_op"]
    prim = []

    def visit(e):
        if isinstance(e, relay.Call) and isinstance(e.op, relay.Function) and \
                int(getattr(e.op.attrs, "Primitive", 0) if e.op.attrs is not None and "Primitive" in e.op.attrs.keys()
                    else 0) == 1:
            prim.append(e.op)

    relay.analysis.post_order_visit(module["main"].body, visit)
    if len(prim) != len(tnodes):
        raise AiegraphError(_STAGE, f"Relay has {len(prim)} primitive calls, graph.json {len(tnodes)} tvm_op nodes",
                            hint="the recording no longer matches the graph executor codegen order")
    nodes = {}
    for f, nid in zip(prim, tnodes):
        fname = g["nodes"][nid]["attrs"]["func_name"]
        ops, params, outs = _fused(f)
        cand = ("fused_" + "_".join(o["op"] for o in ops)).replace(".", "_")
        if fname == "__nop":
            ok = all(o["op"] in LABEL for o in ops)
        else:
            sym = fname[len("tvmgen_default_"):] if fname.startswith("tvmgen_default_") else fname
            ok = sym.startswith(cand[:80])
        nodes[str(nid)] = FusedFn(fname, ops, params, outs, ok).to_json()
    bad = sum(1 for v in nodes.values() if not v["name_ok"])
    return {"version": VERSION, "source": "relay module before LowerTE (deploy_flow.build_c)",
            "n_tvm_op": len(tnodes), "name_mismatch": bad, "nodes": nodes}


def save_relay(module, graph_json: str, path) -> dict:
    if module is None:
        raise AiegraphError(_STAGE, "no Relay module captured before LowerTE", hint="relay.build did not run LowerTE")
    doc = record_relay(module, graph_json)
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))
    return doc
