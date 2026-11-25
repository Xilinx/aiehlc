###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Attention matcher for group offload (design 2026-11-16-aiegraph-group-offload §5.2 / §6, step S3).

``match_attention(cg, onnx=...)`` finds every attention core of the network as a group of
whole TVM calls -- ``[q_t, k_t, qk, softmax, v_t, pv, merge]`` on the diffusion example -- and gives the
user what a kernel template needs (design §5.2)::

    g.calls                 [31, 32, 33, 34, 35, 36, 37]
    g.q, g.k, g.v           boundary inputs: the [1][N][H*D] linear outputs (Buf, ld = H*D)
    g.o                     boundary output: the merged O [N][H*D] (tensors[70] on attn0)
    g.H, g.N, g.M, g.D      heads, query rows, key rows, head dim
    g.scale                 the scale TVM folded into the Q/K layout calls, as the deployed call sees it
    g.cls                   shape class (H, N, M, D): groups of one class share one kernel instance
    g.onnx                  {"qk", "softmax", "pv", "scale": [...]} ONNX node names
    g.bmm                   {"qk": {...}, "pv": {...}}: the two nn_batch_matmul calls (node, func, shapes)

This replaces the example's ``map_attn`` (ONNX bmm layers zipped to TVM nn_batch_matmul in
graph order) and the C-text regexes of ``keep_layout_rt.fa_pass``:

* **TVM side** (graph.json data flow): one core per ``nn_softmax`` call whose input comes from a
  ``batch_matmul`` and whose output feeds exactly one ``batch_matmul`` (``group._attention_core``).
* **ONNX side**: ``partition.match_attention`` (bmm -> softmax -> bmm), or -- design §9 (1), the
  fallback for models the pattern does not cover -- an explicit ONNX node list
  ``nodes=[{"qk": ..., "softmax": ..., "pv": ...}, ...]`` (or ``(qk, softmax, pv)`` tuples).
* **ONNX <-> TVM**: the k-th ONNX ``Softmax`` is the k-th TVM ``nn_softmax`` call (TVM keeps the
  ONNX topological order); the head shapes must agree or the match fails.
* **Q / K / V sources**: the activation a layout call reads. A self-attention layout call reads all
  three outputs of the fused qkv split; the one it uses is the ONNX ``Split`` output index the
  Q (K, V) path starts from -- no C code is read.
* **scale**: no Relay module is kept after convert (sprint P0c, relay.txt RELAY_AFTER_CONVERT=no), so
  the scale comes from the scalar ``Mul`` constant on the ONNX Q / K path (Steve's fallback); when the
  converted params are on disk it is cross-checked against the scalar param the TVM Q / K layout
  call takes, and rounded to bf16 (RNE) when the weights are stored as bf16 (that is the value the
  deployed TVM call reads after ``graph_bf16_expand``).

Every mismatch is an ``AiegraphError``: an attention core is never half-matched.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AiegraphError
from .group import CallGraph, Group, make_group

__all__ = ["AttentionGroup", "match_attention", "attention_classes", "onnx_attention_nodes",
           "partition_attention", "tvm_attention_cores"]

_STAGE = "match"
_PATH_OPS = ("Reshape", "Transpose", "Mul", "Div", "Squeeze", "Unsqueeze", "Identity", "Cast")


@dataclass(repr=False)
class AttentionGroup(Group):
    """A ``Group`` (usable with ``offload_group``) with the attention-specific fields."""
    index: int = 0
    qk: int = None              # call ids
    softmax: int = None
    pv: int = None
    merge: int = None
    q_call: int = None
    k_call: int = None
    v_call: int = None
    q: object = None            # contract.Buf with .tensor (boundary inputs / output)
    k: object = None
    v: object = None
    o: object = None
    H: int = 0
    N: int = 0
    M: int = 0
    D: int = 0
    cls: int = None
    scale: float = None         # as the deployed TVM call sees it (bf16-rounded with bf16 weights)
    scale_fp32: float = None    # the ONNX constant
    scale_k: float = None
    onnx: dict = field(default_factory=dict)
    bmm: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return (self.H, self.N, self.M, self.D)

    def __repr__(self) -> str:
        return (f"attention {self.name}: calls {', '.join(f'[{c}]' for c in self.calls)} "
                f"H={self.H} N={self.N} M={self.M} D={self.D} cls={self.cls} scale={self.scale!r} "
                + " ".join(f"{n}=" + (f"tensors[{b.tensor}]" if b is not None else "?")
                           for n, b in (("q", self.q), ("k", self.k), ("v", self.v), ("o", self.o)))
                + f" onnx={self.onnx.get('softmax', '-')}")


# ─────────────────────────────── TVM side ───────────────────────────────

def _acts(cg: CallGraph, call) -> list:
    return [t for t in call.ins if t not in cg.const]


def tvm_attention_cores(cg: CallGraph) -> list:
    """``[(softmax, qk, pv, members, merge)]`` per ``nn_softmax`` call, graph order. A softmax that
    does not sit between two batch_matmuls gives ``None`` in its slot (so ranks stay aligned with
    the ONNX ``Softmax`` nodes)."""
    out = []
    for c in cg.calls:
        if "nn_softmax" not in c.fn:
            continue
        a = _acts(cg, c)
        qk = cg.producer.get(a[0]) if a else None
        if qk is None or "batch_matmul" not in cg.calls[qk].fn:
            out.append(None)
            continue
        pv = [r for r in cg.readers.get(c.outs[0], []) if "batch_matmul" in cg.calls[r].fn]
        if len(pv) != 1:
            out.append(None)
            continue
        pv = pv[0]
        mem = {qk, c.call, pv}
        mem |= {cg.producer[t] for t in _acts(cg, cg.calls[qk]) if t in cg.producer}
        mem |= {cg.producer[t] for t in _acts(cg, cg.calls[pv]) if t in cg.producer and cg.producer[t] != c.call}
        merge = cg.readers.get(cg.calls[pv].outs[0], [])
        merge = merge[0] if len(merge) == 1 and "reshape" in cg.calls[merge[0]].fn else None
        if merge is not None:
            mem.add(merge)
        out.append((c.call, qk, pv, mem, merge))
    return out


# ─────────────────────────────── ONNX side ───────────────────────────────

def _onnx_model(onnx):
    if onnx is None:
        return None
    if isinstance(onnx, (str, Path)):
        import onnx as _o
        return _o.load(str(onnx), load_external_data=False)
    return onnx


class _OnnxView:
    def __init__(self, model):
        from onnx import numpy_helper
        self.np = numpy_helper
        self.g = model.graph if hasattr(model, "graph") else model
        self.init = {t.name: t for t in self.g.initializer}
        self.prod = {o: n for n in self.g.node for o in n.output}
        self.by_name = {n.name: n for n in self.g.node}
        self.cons = {}
        for n in self.g.node:
            for i in n.input:
                self.cons.setdefault(i, []).append(n)
        self.vi = {v.name: v for v in list(self.g.value_info) + list(self.g.input) + list(self.g.output)}

    def const(self, name):
        """value of an initializer / Constant output, else None"""
        if name in self.init:
            return self.np.to_array(self.init[name])
        p = self.prod.get(name)
        if p is not None and p.op_type == "Constant":
            for a in p.attribute:
                if a.name == "value":
                    return self.np.to_array(a.t)
        return None

    def shape(self, name):
        v = self.vi.get(name)
        if v is None or not v.type.tensor_type.HasField("shape"):
            return None
        d = [x.dim_value if x.HasField("dim_value") else None for x in v.type.tensor_type.shape.dim]
        return None if any(x is None for x in d) else d

    def node(self, ref, want: str):
        n = self.by_name.get(ref)
        if n is None:
            raise AiegraphError(_STAGE, f"ONNX node {ref!r} not in the model",
                                hint="ONNX node names, as Netron shows them")
        if n.op_type != want:
            raise AiegraphError(_STAGE, f"ONNX node {ref!r} is {n.op_type}, expected {want}")
        return n

    def path(self, name):
        """walk up a Q/K/V operand through layout / scale ops: (scale constants [(node, value)],
        source tensor name, split index or None)"""
        scales = []
        for _ in range(16):
            p = self.prod.get(name)
            if p is None or p.op_type not in _PATH_OPS:
                break
            acts = [i for i in p.input if i and self.const(i) is None]
            if p.op_type in ("Mul", "Div"):
                cs = [i for i in p.input if i and self.const(i) is not None]
                if len(cs) == 1 and self.const(cs[0]).size == 1 and len(acts) == 1:
                    v = float(self.const(cs[0]).reshape(-1)[0])
                    scales.append((p.name, v if p.op_type == "Mul" else 1.0 / v))
                else:
                    break
            if len(acts) != 1:
                break
            name = acts[0]
        p = self.prod.get(name)
        split = list(p.output).index(name) if p is not None and p.op_type == "Split" else None
        return scales, name, split


def onnx_attention_nodes(onnx) -> list:
    """``[{"qk", "softmax", "pv", "scale"}]`` from ``partition.match_attention`` (graph order)."""
    from .partition import match_attention
    return [{"qk": a.qk, "softmax": a.softmax, "pv": a.pv, "scale": list(a.scale)}
            for a in match_attention(_onnx_model(onnx))]


def _norm_nodes(nodes) -> list:
    out = []
    for i, e in enumerate(nodes):
        if isinstance(e, dict):
            d = dict(e)
        elif isinstance(e, (list, tuple)) and len(e) == 3:
            d = dict(zip(("qk", "softmax", "pv"), e))
        else:
            raise AiegraphError(_STAGE, f"nodes[{i}] = {e!r}",
                                hint='{"qk": <MatMul>, "softmax": <Softmax>, "pv": <MatMul>} or a 3-tuple')
        miss = [k for k in ("qk", "softmax", "pv") if not d.get(k)]
        if miss:
            raise AiegraphError(_STAGE, f"nodes[{i}]: missing {miss}")
        out.append(d)
    return out


# ─────────────────────────────── params (scale cross-check) ───────────────────────────────

def _param_values(cg: CallGraph, workdir) -> dict:
    """tensor id -> fp32 value of the 1-element params, from ``<stem>_params.bin`` (if present)"""
    if workdir is None:
        return {}
    blobs = sorted(Path(workdir).glob("*_params.bin"))
    if len(blobs) != 1:
        return {}
    try:
        from frontend.tvmrelay.arm_build import param_offsets, params_view
    except ImportError:
        return {}
    offs = param_offsets(blobs[0])
    data = params_view(blobs[0])    # F21: mmap, only the 4-byte scalars are touched
    out = {}
    for t, nm in cg.names.items():
        if nm in offs and cg.dtype[t] == "float32":
            off, nb = offs[nm]
            if nb == 4:
                out[t] = struct.unpack_from("<f", data, off)[0]
    return out


def _bf16(x: float) -> float:
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


# ─────────────────────────────── the matcher ───────────────────────────────

def _src(cg, ov, call: int, onnx_operand: str, who: str, gname: str):
    """the boundary tensor a Q/K/V layout call reads (see module doc)"""
    acts = _acts(cg, cg.calls[call])
    scales, root, split = ov.path(onnx_operand) if ov is not None else ([], None, None)
    if len(acts) == 1:
        return acts[0], scales
    prods = {cg.producer.get(t) for t in acts}
    if len(prods) != 1 or None in prods:
        raise AiegraphError(_STAGE, f"{gname}: {who} layout call [{call}] {cg.calls[call].fn} reads "
                            f"{len(acts)} activations from different producers",
                            hint="not a head split of one linear output")
    outs = cg.calls[prods.pop()].outs
    if split is None and ov is None:
        return None, scales          # no ONNX: the boundary input is ambiguous, left unresolved
    if split is None:
        raise AiegraphError(_STAGE, f"{gname}: {who} layout call [{call}] reads {len(acts)} split outputs "
                            f"and the ONNX {who} path does not start at a Split"
                            + ("" if ov is not None else " (no ONNX model given)"),
                            hint="match('attention', onnx=<model>) or give the node list")
    if split >= len(outs) or outs[split] not in acts:
        raise AiegraphError(_STAGE, f"{gname}: {who} = ONNX Split output {split}, but layout call [{call}] "
                            f"does not read output {split} of call [{cg.producer[acts[0]]}]")
    return outs[split], scales


def _b(buf, shape, ld, name, t):
    return None if t is None else buf(shape, "float32", ld, name, t)


def match_attention(cg: CallGraph, onnx=None, nodes=None, wdtype: str = None, workdir=None) -> list:
    """All attention groups (see module doc). *onnx*: path / ModelProto; *nodes*: the ONNX node
    list fallback (design §9 (1)); *wdtype*: ``"bf16"`` rounds the scale as the bf16 weight is read;
    *workdir*: converted params, for the scale cross-check."""
    from .contract import buf
    cores = tvm_attention_cores(cg)
    model = _onnx_model(onnx)
    ov = _OnnxView(model) if model is not None else None
    if nodes is not None and ov is None:
        raise AiegraphError(_STAGE, "match('attention', nodes=...) needs the ONNX model (onnx=)")
    want = None
    if ov is not None:
        want = _norm_nodes(nodes) if nodes is not None else onnx_attention_nodes(model)
        softmaxes = [n.name for n in ov.g.node if n.op_type == "Softmax"]
        if len(softmaxes) != len(cores):
            raise AiegraphError(_STAGE, f"ONNX has {len(softmaxes)} Softmax nodes, the TVM graph "
                                f"{len(cores)} nn_softmax calls", hint="the model was converted from another ONNX")
        rank = {s: i for i, s in enumerate(softmaxes)}
        pick = []
        for d in want:
            qk, sm, pv = ov.node(d["qk"], "MatMul"), ov.node(d["softmax"], "Softmax"), ov.node(d["pv"], "MatMul")
            if ov.path(sm.input[0])[1] not in qk.output:
                raise AiegraphError(_STAGE, f"{d['softmax']}: its input is not {d['qk']}")
            if pv.input[0] != sm.output[0]:
                raise AiegraphError(_STAGE, f"{d['pv']}: input 0 is not the output of {d['softmax']}")
            r = rank[sm.name]
            if cores[r] is None:
                raise AiegraphError(_STAGE, f"ONNX {sm.name}: TVM nn_softmax call #{r} is not between two "
                                    f"batch_matmuls")
            pick.append((cores[r], dict(d, qk=qk, softmax=sm, pv=pv)))
    else:
        pick = [(c, None) for c in cores if c is not None]

    pvals = _param_values(cg, workdir)
    out, classes = [], []
    for gi, ((smc, qk, pv, mem, merge), on) in enumerate(pick):
        name = f"attn{gi}"
        base = make_group(cg, mem, name)
        qa, pa = _acts(cg, cg.calls[qk]), _acts(cg, cg.calls[pv])
        if len(qa) != 2 or len(pa) != 2:
            raise AiegraphError(_STAGE, f"{name}: bmm [{qk}]/[{pv}] do not read two activations")
        jq, jk, jv = cg.producer.get(qa[0]), cg.producer.get(qa[1]), cg.producer.get(pa[1])
        if None in (jq, jk, jv) or merge is None:
            raise AiegraphError(_STAGE, f"{name}: no Q/K/V layout call or O merge around [{qk}]-[{pv}]")
        xs, ws, ys = cg.shape[qa[0]], cg.shape[qa[1]], cg.shape[cg.calls[qk].outs[0]]
        vs, os_ = cg.shape[pa[1]], cg.shape[cg.calls[pv].outs[0]]
        if len(xs) != 3 or len(ws) != 3:
            raise AiegraphError(_STAGE, f"{name}: qk bmm operands {xs} {ws}, expected [H][N][D] [H][M][D]")
        H, N, D = xs
        M = ws[1]
        if tuple(ws) != (H, M, D) or tuple(ys) != (H, N, M) or tuple(vs) != (H, D, M) or tuple(os_) != (H, N, D):
            raise AiegraphError(_STAGE, f"{name}: shapes q{list(xs)} k{list(ws)} s{list(ys)} v{list(vs)} o{list(os_)} "
                                f"are not one attention core (H={H} N={N} M={M} D={D})")
        onm = {}
        if on is not None:
            qkn, pvn = on["qk"], on["pv"]
            for tname, want_shape in ((qkn.input[0], [1, H, N, D]), (qkn.input[1], [1, H, D, M]),
                                      (pvn.input[1], [1, H, M, D])):
                s = ov.shape(tname)
                if s is not None and s not in (want_shape, want_shape[1:]):
                    raise AiegraphError(_STAGE, f"{name}: ONNX {tname} has shape {s}, TVM core wants {want_shape}")
            tq, sq = _src(cg, ov, jq, qkn.input[0], "Q", name)
            tk, sk = _src(cg, ov, jk, qkn.input[1], "K", name)
            tv, _ = _src(cg, ov, jv, pvn.input[1], "V", name)
            onm = {"qk": qkn.name, "softmax": on["softmax"].name, "pv": pvn.name,
                   "scale": [n for n, _ in sq] + [n for n, _ in sk]}
        else:
            tq, sq = _src(cg, None, jq, None, "Q", name)
            tk, sk = _src(cg, None, jk, None, "K", name)
            tv, _ = _src(cg, None, jv, None, "V", name)
        s_q = s_k = None
        if len(sq) > 1 or len(sk) > 1:
            raise AiegraphError(_STAGE, f"{name}: more than one scale op on the Q/K path {sq} {sk}")
        if sq:
            s_q = sq[0][1]
        if sk:
            s_k = sk[0][1]
        for j, s, who in ((jq, s_q, "Q"), (jk, s_k, "K")):
            pc = [pvals[t] for t in cg.calls[j].ins if t in pvals]
            if s is not None and pc and not any(abs(p - s) <= 1e-6 * max(1.0, abs(s)) for p in pc):
                raise AiegraphError(_STAGE, f"{name}: ONNX {who} scale {s!r} is not the scalar param of TVM "
                                    f"call [{j}] {cg.calls[j].fn} ({pc})")
        to = cg.calls[merge].outs[0]
        ld = H * D
        sh = {t: cg.shape[t] for t in (tq, tk, tv, to) if t is not None}
        for t, rows, who in ((tq, N, "Q"), (tk, M, "K"), (tv, M, "V"), (to, N, "O")):
            if t is None:
                continue
            if list(sh[t][-2:]) != [rows, ld] or any(d != 1 for d in sh[t][:-2]):
                raise AiegraphError(_STAGE, f"{name}: {who} boundary tensors[{t}] is {list(sh[t])}, "
                                    f"expected [1][{rows}][{ld}] (one column block of {D} per head)")
        bins = {t.tensor for t in base.inputs}
        for t, who in ((tq, "Q"), (tk, "K"), (tv, "V")):
            if t is not None and t not in bins:
                raise AiegraphError(_STAGE, f"{name}: {who} tensors[{t}] is not a boundary input")
        if to not in {t.tensor for t in base.outputs}:
            raise AiegraphError(_STAGE, f"{name}: O tensors[{to}] is not the group's output")
        key = (H, N, M, D)
        if key not in classes:
            classes.append(key)
        rnd = (lambda x: x) if wdtype in (None, "fp32") else _bf16
        if wdtype not in (None, "fp32", "bf16"):
            raise AiegraphError(_STAGE, f"wdtype={wdtype!r}", hint="fp32 | bf16")

        def bmm(call, x, w, y):
            # shapes as the TVM call sees its arguments (graph.json entries, before __nop aliasing:
            # P is the [H][N][M] reshape of the [1][H][N][M] softmax output)
            n = cg.calls[call]
            rp = cg.graph["node_row_ptr"]
            raw = [rp[s] + k for s, k, _ in cg.graph["nodes"][n.node]["inputs"]]
            return dict(call=call, node=n.node, func=n.fn if not n.symbol.startswith("tvmgen_default_") else
                        n.symbol[len("tvmgen_default_"):], symbol=n.symbol,
                        xshape=list(cg.shape[raw[0]]), wshape=list(cg.shape[raw[1]]), yshape=list(cg.shape[y]),
                        nargs=len(n.ins) + len(n.outs))
        g = AttentionGroup(
            base.name, base.calls, base.inputs, base.consts, base.outputs, base.internal, base.symbols,
            index=gi, qk=qk, softmax=smc, pv=pv, merge=merge, q_call=jq, k_call=jk, v_call=jv,
            q=_b(buf, (N, D), ld, "q", tq), k=_b(buf, (M, D), ld, "k", tk),
            v=_b(buf, (M, D), ld, "v", tv), o=_b(buf, (N, D), ld, "o", to),
            H=H, N=N, M=M, D=D, cls=classes.index(key),
            scale=None if s_q is None else rnd(s_q), scale_fp32=s_q,
            scale_k=None if s_k is None else rnd(s_k), onnx=onm,
            bmm={"qk": bmm(qk, qa[0], qa[1], cg.calls[qk].outs[0]),
                 "pv": bmm(pv, pa[0], pa[1], cg.calls[pv].outs[0])})
        out.append(g)
    return out


def attention_classes(groups) -> list:
    """``[(H, N, M, D)]`` in class-index order (design §9 (1): six on the diffusion example)."""
    cls = {}
    for g in groups:
        cls.setdefault(g.cls, g.key)
    return [cls[i] for i in sorted(cls)]


def partition_attention(cg: CallGraph, partition: dict, groups=None, onnx=None, wdtype: str = None,
                        workdir=None) -> list:
    """The partition's ``attn`` rows (``autopart --mode attn``: one per bmm, ``attn_group`` /
    ``attn_role`` qk|pv) joined to the matched groups BY ONNX NODE NAME (was the example's
    ``map_attn``, which zipped them in graph order). One dict per row, sorted by
    (part, idx): the TVM nn_batch_matmul call (node, func, symbol, eids, yeid, nargs, x/w/yshape),
    the row's cid / group / role / softmax, M/K/N (unpadded), H, and ``ag_group`` (the
    ``AttentionGroup``). A row with no group, a group missing a row, or a shape mismatch is an error."""
    if groups is None:
        groups = match_attention(cg, onnx=onnx or partition.get("onnx"),
                                 wdtype=wdtype or partition.get("wdtype"), workdir=workdir)
    rows = [L for L in partition.get("layers", []) if L.get("cls") == "attn"]
    by = {L["node"]: L for L in rows}
    if len(by) != len(rows):
        raise AiegraphError(_STAGE, "partition attn rows repeat an ONNX node")
    nodes, rp = cg.graph["nodes"], cg.graph["node_row_ptr"]
    out, used = [], set()
    for g in groups:
        if not g.onnx:
            raise AiegraphError(_STAGE, f"{g.name}: no ONNX node names (match with onnx=)")
        for role in ("qk", "pv"):
            L = by.get(g.onnx[role])
            if L is None:
                raise AiegraphError(_STAGE, f"{g.name}: ONNX {role} {g.onnx[role]} has no partition attn row",
                                    hint="partition from autopart --mode attn on the same ONNX")
            if L.get("attn_role") != role:
                raise AiegraphError(_STAGE, f"{g.name}: partition row {L['node']} is attn_role "
                                    f"{L.get('attn_role')}, the matcher says {role}")
            used.add(L["node"])
            M, K, N = L["pad"]["orig"] if "pad" in L else L["mkn"][0]
            H = int(L["batch"])
            b = g.bmm[role]
            want = ([H, M, K], [H, N, K], [H, M, N])
            if (b["xshape"], b["wshape"], b["yshape"]) != want:
                raise AiegraphError(_STAGE, f"{g.name}: partition {L['node']} (H={H} M={M} K={K} N={N}) vs TVM "
                                    f"call [{b['call']}] {b['func']} x{b['xshape']} w{b['wshape']} y{b['yshape']}")
            n = nodes[b["node"]]
            out.append(dict(node=b["node"], call=b["call"], symbol=b["symbol"], func=b["func"],
                            eids=[rp[s] + k for s, k, _ in n["inputs"]], yeid=rp[b["node"]],
                            nargs=b["nargs"], xshape=b["xshape"], wshape=b["wshape"], yshape=b["yshape"],
                            kind="attn", cid=L["idx"], onnx=L["node"], group=int(L["attn_group"]),
                            role=role, softmax=L.get("softmax"), part=L.get("part"), H=H, M=M, N=N, K=K,
                            ag_group=g))
    left = sorted(set(by) - used)
    if left:
        raise AiegraphError(_STAGE, f"{len(left)} partition attn rows match no attention group: {left[:3]}")
    return sorted(out, key=lambda c: (c["part"], c["cid"]))
