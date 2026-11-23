###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Folding groups: a layout / scale call next to an offloaded call becomes part of its group.

``emod.match(kind, base=...)`` finds them on the graph; each is a :class:`~aiegraph.group.Group`
whose extra calls are *removed* (their work goes into the steps of the calls around them):

``"conv+layout"``   an offloaded conv (one-call ``match("conv")`` group) whose only reader is a
                    pure permutation to NCHW or NHWC: the conv step writes that layout straight
                    into the permutation's output buffer (its ``y`` argument relabelled
                    ``(1, O/c', OH, OW, c')``, c' = 1 or O).
``"linear+layout"`` an offloaded linear whose ``[M][C]`` output, plus a residual of the same
                    shape, is permuted to the NCHWc input of an offloaded conv: the linear adds
                    the residual in its scatter (one more packed argument) and the conv reads the
                    ``[M][C]`` buffer relabelled NHWC ``(1, 1, H, W, C)``.
``"dense+silu"``    a dense whose packed weight is computed per call as ``layout(W * s)`` with a
                    per-K scale ``s`` (``sigmoid(x)`` of the dense input ``x``, or a vector): the
                    weight is pre-packed once in ``weights.bin`` and the dense reads ``x * s``
                    (``dense(x, W*s) == dense(x*s, W)``).

Every decision is made on the graph: the permutation is the one the saved Relay op semantics
give (``pure_perm.index_map`` on ``<stem>_opsem.json``, compared element by element with the
layout the kernel writes / reads), the removed intermediate has no other reader, and the
buffers involved are untouched in between. ``plan_group`` re-checks placement, early writes,
labels and in-place hazards when the group is offloaded. Overlapping folds of one call are
joined by :func:`merge` (one group per connected set of calls).

Kernel contract of the conv label (``aiednn/conv2d`` scatter): an epilogue operand of the
conv is per channel, or full size in the output's c-block, or the output is planar (c' = 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import pure_perm
from .contract import Arg
from .errors import AiegraphError
from .group import CallGraph, Group, make_group

__all__ = ["Fold", "conv_layout", "linear_layout", "dense_silu", "merge", "packed_args", "step_io",
           "keep_layout_plan"]

_STAGE = "group"


@dataclass
class Fold:
    """What a folding group changes at its calls (``g.info["fold"]``)."""
    kind: str
    removed: list                                   # calls whose work moved into a step
    edits: dict = field(default_factory=dict)       # call -> [("set", slot, Arg) | ("insert", slot, Arg)]
    writes: dict = field(default_factory=dict)      # call -> [tensor written there instead of its producer]
    base: dict = field(default_factory=dict)        # call -> the one-call group it extends
    params: dict = field(default_factory=dict)      # kind-specific numbers (labels, N, K, ...)


def _n(shape) -> int:
    return int(np.prod(shape)) if shape else 1


def _raw_args(cg: CallGraph, call: int) -> list:
    """packed argument tensor ids of a call as the driver binds them (inputs, then outputs)"""
    rp = cg.graph["node_row_ptr"]
    c = cg.calls[call]
    return [rp[s] + k for s, k, _ in cg.graph["nodes"][c.node]["inputs"]] + list(c.outs)


def _only_reader(cg: CallGraph, t: int):
    r = cg.readers.get(cg.canon(t), [])
    return r[0] if len(r) == 1 and cg.canon(t) not in cg.heads else None


def _untouched(cg: CallGraph, lo: int, hi: int, t: int, skip=()) -> bool:
    sid = cg.storage(t)
    return not any(cg.storage(x) == sid for c in range(lo + 1, hi) if c not in skip
                   for x in cg.touches(c) if x not in cg.const)


def _pure(cg, call, extra=(), need=()):
    try:
        return pure_perm.check(cg, call, extra=extra, need=need, stage=_STAGE)
    except AiegraphError:
        return None


def _nchwc(shape, c2: int):
    """flat order of an NCHW{c}c tensor ``shape`` when re-blocked to NCHW{c2}c"""
    _, cb, h, w, c = shape
    a = np.arange(_n(shape), dtype=np.int64).reshape(shape).transpose(0, 1, 4, 2, 3).reshape(1, cb * c, h, w)
    return a.reshape(1, cb * c // c2, c2, h, w).transpose(0, 1, 3, 4, 2).reshape(-1)


def _group(cg, calls, name, fold: Fold, info: dict) -> Group:
    g = make_group(cg, calls, name)
    g.info = dict(info, fold=fold, kind=fold.kind)
    return g


# ═══════════════════════════════════════════════════════════════════════════
#  conv + layout: the conv scatter writes the reader's layout
# ═══════════════════════════════════════════════════════════════════════════

def conv_layout(cg: CallGraph, base, labels=(1, "O"), x_first: bool = True, log=None) -> list:
    """Folds of every *base* conv group (one call, 5-D NCHWc output) whose only reader is a pure
    permutation that equals re-blocking the output to NCHW{c'}c, c' in *labels* (``"O"`` = all
    channels = NHWC). Skips (with a *log* line) a conv whose epilogue operand would break the label
    contract or lives in the buffer it would now write. *x_first*: the conv kernel reads its
    activation (argument 0) completely before it writes (aiednn conv2d: pack, gemm, then scatter),
    so the output may land in the activation's buffer (``Arg(early=True)``)."""
    out = []
    for g in base:
        c = g.calls[0]
        y = cg.canon(cg.calls[c].outs[0])
        ys = list(cg.shape[y])
        r = _only_reader(cg, y)
        if len(ys) != 5 or ys[0] != 1 or r is None:
            continue
        rc = cg.calls[r]
        acts = [t for t in rc.ins if t not in cg.const]
        if len(rc.ins) != 1 or acts != [y] or len(rc.outs) != 1:
            continue
        if _pure(cg, r, need=("layout_transform",)) is None:
            continue
        m = pure_perm.permutation(cg, r, 0)
        if m is None:
            continue
        O, OH, OW = ys[1] * ys[4], ys[2], ys[3]
        hit = None
        for c2 in labels:
            c2 = O if c2 == "O" else int(c2)
            if O % c2 == 0 and np.array_equal(m, _nchwc(ys, c2)):
                hit = c2
                break
        if hit is None:
            continue
        to = rc.outs[0]
        raw = _raw_args(cg, c)
        epi = [t for t in raw[2:-1]]
        bad = [t for t in epi if not (_n(cg.shape[t]) == O or (hit == 1 and _n(cg.shape[t]) == O * OH * OW)
                                     or (len(cg.shape[t]) == 5 and cg.shape[t][4] == hit))]
        clash = [t for t in raw[1 if x_first else 0:-1] if t not in cg.const and cg.storage(t) == cg.storage(to)]
        if bad or clash or not _untouched(cg, c, r, to):
            if log:
                log(f"fold conv+layout [{c}] -> [{r}] {rc.fn}: skipped ("
                    + (f"epilogue tensors[{bad[0]}] {list(cg.shape[bad[0]])} vs label c={hit}" if bad else
                       f"operand tensors[{clash[0]}] lives in the output buffer" if clash else
                       "output buffer touched in between") + ")")
            continue
        lab = (1, O // hit, OH, OW, hit)
        ed = [("set", len(raw) - 1, Arg(to, lab))]
        if cg.storage(raw[0]) == cg.storage(to):
            ed.append(("set", 0, Arg(raw[0], None, True)))
        f = Fold("conv+layout", [r], edits={c: ed}, writes={c: [to]},
                 base={c: g}, params=dict(label=lab, c=hit, layout="planar" if hit == 1 else
                                          "nhwc" if hit == O else f"NCHW{hit}c"))
        out.append(_group(cg, [c, r], f"{g.name}_lay", f, dict(g.info)))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  linear + layout: residual into the linear scatter, the conv reads NHWC
# ═══════════════════════════════════════════════════════════════════════════

def linear_layout(cg: CallGraph, base, convs, x_first: bool = True, log=None) -> list:
    """Folds of every *base* linear group whose ``[M][C]`` output's only reader adds a residual and
    permutes the sum to the NCHWc input of a *convs* group's call (and nothing else reads it).
    *x_first*: as in :func:`conv_layout` (the conv's relabelled activation is ``early``)."""
    conv_at = {g.calls[0]: g for g in convs}
    out = []
    for g in base:
        c = g.calls[0]
        tl = cg.canon(cg.calls[c].outs[0])
        sl = list(cg.shape[tl])
        r = _only_reader(cg, tl)
        if r is None or len(sl) not in (2, 3) or (len(sl) == 3 and sl[0] != 1):
            continue
        M, C = sl[-2], sl[-1]
        rc = cg.calls[r]
        if len(rc.ins) != 2 or len(rc.outs) != 1 or rc.ins[0] != tl:
            continue
        ops = _pure(cg, r, extra=("add",), need=("add", "layout_transform"))
        if ops is None or ops.count("add") != 1:
            continue
        tr = rc.ins[1]
        to = rc.outs[0]
        so = list(cg.shape[to])
        cv = _only_reader(cg, to)
        if tr in cg.const or _n(cg.shape[tr]) != M * C or cv not in conv_at or len(so) != 5 or so[0] != 1:
            continue
        if so[1] * so[4] != C or so[2] * so[3] != M or cg.canon(_raw_args(cg, cv)[0]) != to:
            continue
        m0, m1 = pure_perm.permutation(cg, r, 0), pure_perm.permutation(cg, r, 1)
        # [M][C] == NHWC (1, H, W, C) == NCHW{C}c; the conv reads NCHW{c}c of the same values
        want = _nchwc([1, 1, so[2], so[3], C], so[4])
        if m0 is None or m1 is None or not np.array_equal(m0, want) or not np.array_equal(m1, want):
            continue
        p = cg.producer.get(tr)
        if p is None or p >= c or not _untouched(cg, c, cv, tl, skip=(r,)):
            if log:
                log(f"fold linear+layout [{c}] -> [{r}] -> [{cv}]: skipped (buffers touched in between)")
            continue
        raw = _raw_args(cg, c)
        H, W = so[2], so[3]
        f = Fold("linear+layout", [r],
                 edits={c: [("set", len(raw) - 1, Arg(tr)), ("insert", len(raw), Arg(tl))],
                        cv: [("set", 0, Arg(tl, (1, 1, H, W, C), x_first))]},
                 base={c: g, cv: conv_at[cv]}, params=dict(M=M, C=C, H=H, W=W, residual=tr))
        out.append(_group(cg, [c, r, cv], f"{g.name}_lay", f, dict(g.info)))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  dense + silu: the per-K scale goes to the dense input, the weight is packed once
# ═══════════════════════════════════════════════════════════════════════════

def dense_silu(cg: CallGraph, kmax: int = 4096) -> list:
    """Every call A = ``layout(W * s)`` of a 2-D constant ``W [N][K]`` (read by A only) into the
    ``[N/b][K][b]`` packed weight of the dense call B that alone reads it (B's weight argument), where
    ``s`` is a K-vector or ``sigmoid`` of B's own input. ``g.info``: ``N K b`` and the tensors."""
    out = []
    for a in cg.calls:
        ops = _pure(cg, a.call, extra=("multiply", "sigmoid"), need=("multiply", "layout_transform"))
        if ops is None or ops.count("multiply") != 1 or len(a.ins) != 2 or len(a.outs) != 1:
            continue
        ws = [t for t in a.ins if t in cg.const and len(cg.shape[t]) == 2]
        if len(ws) != 1:
            continue
        w = ws[0]
        N, K = cg.shape[w]
        to = a.outs[0]
        so = list(cg.shape[to])
        b = _only_reader(cg, to)
        if b is None or len(so) != 3 or so[1] != K or so[0] * so[2] != N or cg.readers.get(w) != [a.call]:
            continue
        bc = cg.calls[b]
        raw = _raw_args(cg, b)
        if "dense" not in bc.fn or len(raw) < 3 or raw[1] != to or len(bc.outs) != 1:
            continue
        x = cg.canon(raw[0])
        s = [t for t in a.ins if t != w][0]
        fn = pure_perm.load(cg).fn(cg, a.call)
        pi = a.ins.index(w)
        m = pure_perm.permutation(cg, a.call, pi)
        blk = so[2]
        want = np.arange(N * K, dtype=np.int64).reshape(N // blk, blk, K).transpose(0, 2, 1).reshape(-1)
        if m is None or not np.array_equal(m, want) or _n(cg.shape[x]) != K or K > kmax:
            continue
        mul = [o for o in fn.ops if o["op"] == "multiply"][0]
        if [list(i) for i in mul["ins"]] not in ([[N, K], [K]], [[K], [N, K]]):
            continue                        # not a per-K scale of W
        if "sigmoid" in ops:
            sig = [o for o in fn.ops if o["op"] == "sigmoid"]
            if cg.canon(s) != x or len(sig) != 1 or _n(sig[0]["ins"][0]) != K:
                continue
            scale = None                    # s = sigmoid(x), computed by the step
        else:
            if _n(cg.shape[s]) != K or s in cg.const:
                continue
            scale = cg.canon(s)
        if not _untouched(cg, a.call, b, to) or (scale is not None and not _untouched(cg, a.call, b, scale)):
            continue
        f = Fold("dense+silu", [], params=dict(N=N, K=K, block=blk, w=w, packed=to, x=x, scale=scale,
                                               dense=b, symbol=bc.symbol, nargs=len(raw)))
        g = _group(cg, [a.call, b], f"silu{a.call}", f, dict(f.params))
        g.w = _weight_ref(cg, w)
        out.append(g)
    return out


def _weight_ref(cg, w):
    from .callsite import WeightRef
    return WeightRef(cg.names.get(w, ""), w, tuple(cg.shape[w]), cg.dtype[w])


# ═══════════════════════════════════════════════════════════════════════════
#  joining folds, and what a step of a joined group gets
# ═══════════════════════════════════════════════════════════════════════════

def merge(cg: CallGraph, groups) -> list:
    """One group per connected set of calls over *groups* (folds and the one-call groups they
    extend), graph order. ``g.info["fold"]`` of a joined group holds every edit / removal / moved
    write; ``g.info["base"]`` maps each kept call to its one-call group."""
    groups = list(groups)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for g in groups:
        for c in g.calls:
            parent[find(c)] = find(g.calls[0])
    sets = {}
    for g in groups:
        sets.setdefault(find(g.calls[0]), []).append(g)
    out = []
    for gs in sorted(sets.values(), key=lambda v: min(c for g in v for c in g.calls)):
        calls = sorted({c for g in gs for c in g.calls})
        f = Fold("+".join(sorted({g.info.get("kind", "one") for g in gs})), [])
        base = {}
        for g in gs:
            fd = g.info.get("fold")
            if fd is None:
                base[g.calls[0]] = g
                continue
            f.removed += fd.removed
            for k, v in fd.edits.items():
                f.edits.setdefault(k, []).extend(v)
            for k, v in fd.writes.items():
                f.writes.setdefault(k, []).extend(v)
            base.update(fd.base)
        both = sorted(set(f.removed) & set(base))
        if both:
            raise AiegraphError(_STAGE, f"merge: calls {both} are removed by one fold and kept by another")
        f.base = base
        if len(gs) == 1 and gs[0].info.get("fold") is None:
            out.append(gs[0])
            continue
        name = "_".join(sorted({g.name for g in gs}, key=lambda n: (len(n), n)))[:60]
        g0 = make_group(cg, calls, name)
        g0.info = dict(fold=f, kind=f.kind, base=base)
        out.append(g0)
    return out


def packed_args(cg: CallGraph, call: int, edits=()) -> list:
    """The call's packed arguments as ``Arg``s with the folds' edits applied (sets by original slot,
    then inserts)."""
    args = [Arg(t) for t in _raw_args(cg, call)]
    for e in edits:
        if e[0] == "set":
            args[e[1]] = e[2]
    for e in sorted((e for e in edits if e[0] == "insert"), key=lambda e: -e[1]):
        args.insert(e[1], e[2])
    return args


def step_io(cg: CallGraph, g: Group, call: int) -> tuple:
    """``(reads, writes, args)`` of the step at *call* of a (joined) folding group: the boundary inputs
    its packed arguments carry, the group outputs it writes (its own or moved there) -- as
    ``GroupTensor`` -- and its packed arguments (``Arg``)."""
    f = g.info.get("fold")
    args = packed_args(cg, call, f.edits.get(call, ()) if f else ())
    ins = {t.tensor for t in g.inputs}
    outs = {t.tensor for t in g.outputs}
    reads, writes = [], []
    for a in args:
        t = cg.canon(a.tensor)
        if t in ins and t not in reads:
            reads.append(t)
    for t in list(cg.calls[call].outs) + list((f.writes.get(call, ()) if f else ())):
        t = cg.canon(t)
        if t in outs and t not in writes:
            writes.append(t)
    return [cg.tensor(t) for t in reads], [cg.tensor(t) for t in writes], args


# ═══════════════════════════════════════════════════════════════════════════
#  partition keep_layout rows -> the layout folds that realize them
# ═══════════════════════════════════════════════════════════════════════════

_LAYOUT_FOLDS = ("conv+layout", "linear+layout")


def keep_layout_plan(model, partition, layers=(), folds=None) -> dict:
    """The partition rows with ``keep_layout`` (an AIE output stays in the AIE layout; the
    Transpose / Reshape ops to its AIE consumers are labels only), checked twice:

    * on the ONNX *model*: ``partition.plan_keep_layout`` (every consumer an AIE row, the view
      row-major in the kept layout);
    * on the TVM graph, when *folds* (the ``match("conv+layout")`` / ``match("linear+layout")``
      groups) is given: each kept producer -- the *layers* dict with ``onnx == row node``, its
      ``call`` -- must be a member call of a layout fold (the fold proved the permutation pure with
      ``pure_perm`` and removed it). A kept row that no fold realizes raises AiegraphError.

    Tags *layers* (dicts with ``onnx``): producer ``keep_layout=<layout>``, consumer ``keep_in=[producer
    node, ...]``. Returns ``plan_keep_layout``'s ``dict(edges, label_ops, kept)`` (+ ``folded`` = realized
    edges when *folds* is given). No row with the field -> ``kept == 0`` and nothing is touched."""
    from .partition import plan_keep_layout
    rep = plan_keep_layout(model, partition)
    by = {c.get("onnx"): c for c in layers}
    if folds is not None and rep["edges"]:
        folded = {k for g in folds if g.info.get("kind") in _LAYOUT_FOLDS for k in g.calls}
        miss = []
        for e in rep["edges"]:
            c = by.get(e["node"])
            if c is None or c.get("call") not in folded:
                miss.append(e["node"])
        if miss:
            raise AiegraphError(_STAGE, f"keep_layout rows not realized by a layout fold: {miss}",
                                hint="the producer must be an offloaded conv / linear whose permutation "
                                     "match('conv+layout' | 'linear+layout') folds")
        rep["folded"] = len(rep["edges"])
    for e in rep["edges"]:
        if e["node"] in by:
            by[e["node"]]["keep_layout"] = e["layout"]
        for u in e["consumers"]:
            if u in by:
                by[u].setdefault("keep_in", []).append(e["node"])
    return rep
