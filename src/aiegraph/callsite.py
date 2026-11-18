###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""One-call groups for weighted ops (design 2026-11-16-aiegraph-group-offload §5.2, step S1).

``emod.match("conv", partition=..., onnx=...)`` / ``emod.match("linear", ...)`` find the
TVM call of every partition row (ONNX ``Conv`` / ``MatMul`` / ``Gemm`` with a constant
weight) and return it as a :class:`~aiegraph.group.Group` of ONE call, with

``g.x``      the activation the call reads (arg 0),
``g.y``      the tensor it writes,
``g.w``      its weight (a graph parameter) -- ``g.w.as_bf16(layout=fn)`` is the weight
             contract of a step: stored in ``weights.bin`` as bf16, already in the
             kernel's layout, read in place (no ``graph_bf16_expand`` at that call),
``g.info``   geometry (``cid`` = partition row, ``M``/``N``/``K``, stride, pad, shapes).

Matching (the graph forms of what the old per-example ``map_layers`` / ``map_linears`` did):

conv     ONNX ``Conv`` order == TVM ``conv2d`` call order; every pair must agree on the
         weight shape (``OIHW`` vs TVM ``OIHW{i}{o}``). Weight statistics are compared
         as well: a mismatch means TVM folded a scale into the weight (``FoldScaleAxis``);
         such calls are listed in ``info["folded"]`` (the caller decides if that is ok).
linear   TVM ``dense`` calls with a parameter weight, matched by VALUE: the packed TVM
         weight unpacked to ``[N][K]`` must equal the ONNX weight exactly, one hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import AiegraphError

__all__ = ["WeightRef", "WeightSpec", "conv_groups", "linear_groups", "weight_bytes",
           "params_file"]

_STAGE = "group"


@dataclass(frozen=True)
class WeightSpec:
    """How a step wants its weight: bf16 in ``weights.bin``, ``layout(w_bf16) -> flat``
    (a permutation of the TVM-layout bf16 weight), optionally after a compile-time
    ``transform(w_fp32) -> w_fp32`` (design ``weight_transform``)."""
    ref: "WeightRef"
    dtype: str = "bfloat16"
    layout: object = None
    transform: object = None


@dataclass(frozen=True)
class WeightRef:
    """A graph parameter as a call reads it: ``tensors[tensor]``, blob name ``name``."""
    name: str
    tensor: int
    shape: tuple
    dtype: str = "float32"

    def as_bf16(self, layout=None, transform=None) -> WeightSpec:
        if layout is not None and not callable(layout):
            raise AiegraphError(_STAGE, f"weight {self.name}: layout={layout!r} is not callable",
                                hint="layout(w_bf16: uint16 ndarray of the TVM shape) -> flat ndarray")
        if transform is not None and not callable(transform):
            raise AiegraphError(_STAGE, f"weight {self.name}: transform={transform!r} is not callable")
        return WeightSpec(self, "bfloat16", layout, transform)

    def __str__(self) -> str:
        return f"{self.name} (tensors[{self.tensor}]) {self.dtype}{list(self.shape)}"


def params_file(workdir) -> Path:
    g = next(Path(workdir).glob("*_graph.json"), None)
    if g is None:
        raise AiegraphError(_STAGE, f"no *_graph.json in {workdir}")
    return g.with_name(g.name.replace("_graph.json", "_params.bin"))


_OFFS = {}


def _offsets(params: Path) -> dict:
    """``arm_build.param_offsets`` (reads the whole blob), cached per file + mtime + size."""
    from . import _env
    _env.frontend("")
    from frontend.tvmrelay.arm_build import param_offsets
    st = Path(params).stat()
    key = (str(Path(params).resolve()), st.st_mtime_ns, st.st_size)
    if key not in _OFFS:
        _OFFS.clear()
        _OFFS[key] = param_offsets(Path(params))
    return _OFFS[key]


def _fp32(blob, offs, name):
    import numpy as np
    off, nb = offs[name]
    return np.frombuffer(blob[off:off + nb].tobytes(), dtype="<f4")


def weight_bytes(spec: WeightSpec, params: Path, offs: dict = None) -> bytes:
    """The bf16 bytes *spec* puts into ``weights.bin`` for its parameter (same RNE
    rounding as ``arm_build.write_aligned_blob``). The layout must be a permutation."""
    import numpy as np
    from . import _env
    _env.frontend("")
    from frontend.tvmrelay.arm_build import f32_to_bf16_rne
    offs = offs or _offsets(params)
    ref = spec.ref
    if ref.name not in offs:
        raise AiegraphError(_STAGE, f"weight {ref.name}: not a parameter of {params.name}")
    blob = np.memmap(params, dtype=np.uint8, mode="r")
    w = _fp32(blob, offs, ref.name)
    n = int(np.prod(ref.shape))
    if w.size != n:
        raise AiegraphError(_STAGE, f"weight {ref}: blob holds {w.size} fp32, shape needs {n}")
    if spec.transform is not None:
        w = np.asarray(spec.transform(w.reshape(ref.shape).copy()), dtype=np.float32)
        if w.size != n:
            raise AiegraphError(_STAGE, f"weight {ref}: transform changed the size {n} -> {w.size}")
    b = f32_to_bf16_rne(w.reshape(-1)).reshape(ref.shape)
    if spec.layout is None:
        out = b.reshape(-1)
    else:
        out = np.ascontiguousarray(np.asarray(spec.layout(b))).reshape(-1)
        if out.dtype != np.uint16 or out.size != n:
            raise AiegraphError(_STAGE, f"weight {ref}: layout returned {out.dtype}[{out.size}], "
                                f"wanted uint16[{n}]", hint="layout permutes bf16 values only")
        if not np.array_equal(np.bincount(out, minlength=1 << 16),
                              np.bincount(b.reshape(-1), minlength=1 << 16)):
            raise AiegraphError(_STAGE, f"weight {ref}: layout is not a permutation of the weight",
                                hint="value changes belong in transform=, not layout=")
    return out.astype("<u2").tobytes()


# ═══════════════════════════════════════════════════════════════════════════
#  matching
# ═══════════════════════════════════════════════════════════════════════════

def _stats(a, chunk: int = 1 << 20):
    """(size, sum, sum of squares, sum of |a|) in float64, chunked through one reused
    buffer (no full float64 copy of a big weight)."""
    import numpy as np
    a = np.asarray(a).ravel()
    buf = np.empty(min(chunk, max(a.size, 1)), dtype=np.float64)
    s = s2 = sa = 0.0
    for i in range(0, a.size, chunk):
        b = buf[:min(chunk, a.size - i)]
        b[:] = a[i:i + chunk]
        s += float(b.sum())
        sa += float(np.abs(b).sum())
        np.multiply(b, b, out=b)
        s2 += float(b.sum())
    return a.size, s, s2, sa


def _partition(partition) -> dict:
    if isinstance(partition, dict):
        return partition
    p = Path(partition)
    if not p.is_file():
        raise AiegraphError(_STAGE, f"partition {partition}: no such file")
    return json.loads(p.read_text())


def _raw_ins(cg, gc) -> list:
    """The call's input tensor ids as the driver binds them (``args[k]``; no __nop folding --
    a reshape-only __nop keeps its own shape)."""
    rp = cg.graph["node_row_ptr"]
    return [rp[s] + k for s, k, _ in cg.graph["nodes"][gc.node]["inputs"]]


def _one_call(cg, gc, name, wt, info):
    from .group import make_group
    g = make_group(cg, [gc.call], name)
    g.x = cg.tensor(gc.ins[0])
    g.y = cg.tensor(gc.outs[0])
    g.w = WeightRef(cg.names.get(wt, ""), wt, tuple(cg.shape[wt]), cg.dtype[wt])
    g.info = info
    return g


def _onnx(onnx_path):
    import onnx
    return onnx.load(str(onnx_path))


def _groups_ok(groups, g: int, C: int, O: int) -> bool:
    """ONNX conv ``group`` attribute *g* accepted by ``match("conv", groups=...)``; *groups* is one of
    1 (or any int), ``"dw"`` (plain + depthwise, g == C == O) or ``"any"``, or a tuple/list of them."""
    if isinstance(groups, (tuple, list)):
        return any(_groups_ok(x, g, C, O) for x in groups)
    if groups == "any":
        return True
    if groups == "dw":
        return g == C == O or g == 1
    return g == int(groups)


def _groups_valid(groups) -> bool:
    if isinstance(groups, (tuple, list)):
        return len(groups) > 0 and all(_groups_valid(x) for x in groups)
    return groups in ("dw", "any") or (isinstance(groups, int) and not isinstance(groups, bool) and groups >= 1)


def conv_groups(cg, workdir, onnx_path, partition, model=None, groups=1) -> list:
    """One group per partition conv row (``cls`` not linear/attn), partition order. ``groups``: the
    ONNX ``group`` attribute accepted -- 1 (plain convs), ``"dw"`` (plain + depthwise, group == C ==
    O), ``"any"``, or a tuple of those (e.g. ``(1, "dw")``); ``g.info["groups"]`` carries it, ``K``
    is the per-group reduction ``C/g*KH*KW``.
    ONNX Convs are paired with TVM conv2d calls by unpacked weight shape and weight value
    (graph order only breaks ties), so a TVM schedule that reorders branches still matches."""
    if not _groups_valid(groups):
        raise AiegraphError(_STAGE, f"match('conv', groups={groups!r})", hint="1, 'dw', 'any' or a tuple of them")
    import numpy as np
    from onnx import helper, numpy_helper
    part = _partition(partition)
    params = params_file(workdir)
    offs = _offsets(params)
    blob = np.memmap(params, dtype=np.uint8, mode="r")
    convs = []
    for gc in cg.calls:
        if "conv2d" not in gc.symbol:
            continue
        pins = [t for t in gc.ins if t in cg.const and cg.names.get(t) in offs]
        if not pins:
            raise AiegraphError(_STAGE, f"call [{gc.call}] {gc.fn}: conv without a parameter weight")
        w = max(pins, key=lambda t: offs[cg.names[t]][1])
        raw = _raw_ins(cg, gc)
        convs.append(dict(call=gc.call, node=gc.node, wt=w, warg=raw.index(w),
                          wname=cg.names[w], wshape=list(cg.shape[w]),
                          xshape=list(cg.shape[raw[0]]), yshape=list(cg.shape[gc.outs[0]]),
                          stats=_stats(_fp32(blob, offs, cg.names[w]))))
    m = model if model is not None else _onnx(onnx_path)
    inits = {t.name: t for t in m.graph.initializer}
    oconv = [n for n in m.graph.node if n.op_type == "Conv"]
    if len(oconv) != len(convs):
        raise AiegraphError(_STAGE, f"match('conv'): {len(oconv)} ONNX Conv vs {len(convs)} TVM "
                            f"conv2d calls", hint="conv matching needs TVM to keep every Conv")
    def oihw(ws):
        return (ws[0] * ws[5], ws[1] * ws[4], ws[2], ws[3]) if len(ws) == 6 else None

    def same(a, b):
        return (a[0] == b[0] and abs(a[1] - b[1]) <= 1e-6 * b[3] + 1e-9
                and abs(a[2] - b[2]) <= 1e-6 * b[2] + 1e-12)

    by_shape = {}
    for c in convs:
        by_shape.setdefault(oihw(c["wshape"]), []).append(c)
    by_name, folded, taken = {}, [], set()
    for n in oconv:
        w = numpy_helper.to_array(inits[n.input[1]])
        cand = [c for c in by_shape.get(tuple(w.shape), []) if c["call"] not in taken]
        if not cand:
            raise AiegraphError(_STAGE, f"match('conv'): ONNX {n.name} weight {list(w.shape)}: no "
                                "unmatched TVM conv2d call with that (unpacked OIHW) weight")
        st = _stats(w)
        hit = [c for c in cand if same(c["stats"], st)]
        c = hit[0] if hit else cand[0]
        if not hit:
            folded.append(n.name)
        taken.add(c["call"])
        by_name[n.name] = c
    onodes = {n.name: n for n in m.graph.node}
    out, used = [], set()
    for L in part["layers"]:
        if L.get("cls") in ("linear", "attn"):
            continue
        n = onodes.get(L["node"])
        if n is None or n.op_type != "Conv" or n.input[1] != L["weight"]:
            raise AiegraphError(_STAGE, f"partition row {L.get('idx')} {L['node']}: not an ONNX Conv "
                                f"with weight {L.get('weight')}")
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        pads = list(at.get("pads", [0, 0, 0, 0]))
        c = by_name[L["node"]]
        ng = int(at.get("group", 1))
        ws = c["wshape"]
        Cx = c["xshape"][1] * c["xshape"][4] if len(c["xshape"]) == 5 else -1
        if not _groups_ok(groups, ng, Cx, ws[0] * ws[5]) or list(at.get("dilations", [1, 1])) != [1, 1] \
                or pads[0] != pads[2] or pads[1] != pads[3]:
            raise AiegraphError(_STAGE, f"{L['node']}: group={ng} / dilated / asymmetric-pad conv "
                                f"(match groups={groups!r})",
                                hint="one-call conv groups take 2-D convs; groups='dw' adds depthwise")
        if c["call"] in used:
            raise AiegraphError(_STAGE, f"{L['node']}: call [{c['call']}] matched twice")
        used.add(c["call"])
        gc = cg.calls[c["call"]]
        O, C, KH, KW = ws[0] * ws[5], ws[1] * ws[4] * ng, ws[2], ws[3]
        M = c["yshape"][2] * c["yshape"][3]
        if c["xshape"][1] * c["xshape"][4] != C or c["yshape"][1] * c["yshape"][4] != O \
                or [KH, KW] != list(at["kernel_shape"]):
            raise AiegraphError(_STAGE, f"{L['node']}: TVM call [{c['call']}] shapes x{c['xshape']} "
                                f"y{c['yshape']} w{ws} disagree with the ONNX Conv")
        if "mkn" in L and [M, C // ng * KH * KW, O] != list(L["mkn"][0]):
            raise AiegraphError(_STAGE, f"{L['node']}: M,K,N {[M, C // ng * KH * KW, O]} != partition "
                                f"mkn {L['mkn'][0]}")
        if c["warg"] != 1:
            raise AiegraphError(_STAGE, f"{L['node']}: weight is arg {c['warg']} of call "
                                f"[{c['call']}], expected arg 1")
        sym = gc.symbol if gc.symbol.startswith("tvmgen_") else "tvmgen_default_" + gc.symbol
        info = dict(kind="conv", call=c["call"], node=c["node"], symbol=sym, func=gc.symbol,
                    wname=c["wname"], warg=c["warg"], nargs=len(gc.ins) + len(gc.outs),
                    wshape=ws, xshape=c["xshape"], yshape=c["yshape"], cid=L["idx"],
                    onnx=L["node"], weight=L["weight"], macs=M * O * C // ng * KH * KW,
                    sh=at["strides"][0], sw=at["strides"][1], ph=pads[0], pw=pads[1],
                    M=M, N=O, K=C // ng * KH * KW, groups=ng, folded=L["node"] in folded)
        out.append(_one_call(cg, gc, f"conv{L['idx']}", c["wt"], info))
    return out


def all_conv_calls(cg) -> list:
    """``[{call, node, symbol}]`` of every TVM conv2d call (carrier choice)."""
    return [dict(call=c.call, node=c.node,
                 symbol=c.symbol if c.symbol.startswith("tvmgen_") else "tvmgen_default_" + c.symbol)
            for c in cg.calls if "conv2d" in c.symbol]


def linear_groups(cg, workdir, onnx_path, partition, model=None) -> list:
    """One group per partition ``cls == "linear"`` row, matched to a TVM dense call by
    weight value (packed ``[N/nb][K][nb]`` unpacked == ONNX ``[N][K]``)."""
    import numpy as np
    from onnx import helper, numpy_helper
    part = _partition(partition)
    params = params_file(workdir)
    offs = _offsets(params)
    blob = np.memmap(params, dtype=np.uint8, mode="r")
    dense = {}
    for gc in cg.calls:
        if "dense" not in gc.symbol or len(gc.ins) < 2:
            continue
        raw = _raw_ins(cg, gc)
        w = raw[1]
        if w not in cg.const or cg.names.get(w) not in offs:
            continue        # weight computed at run time: not a weighted call
        ws, xs, ys = list(cg.shape[w]), list(cg.shape[raw[0]]), list(cg.shape[gc.outs[0]])
        if len(ws) != 3 or len(xs) != 2 or len(ys) != 2:
            continue
        dense.setdefault((ws[0] * ws[2], ws[1]), []).append(
            dict(call=gc.call, node=gc.node, wt=w, wname=cg.names[w], wshape=ws, xshape=xs, yshape=ys))
    m = model if model is not None else _onnx(onnx_path)
    inits = {t.name: t for t in m.graph.initializer}
    onodes = {n.name: n for n in m.graph.node}
    out, used = [], set()
    for L in part["layers"]:
        if L.get("cls") != "linear":
            continue
        n = onodes.get(L["node"])
        if n is None or n.op_type not in ("MatMul", "Gemm") or L["weight"] not in n.input:
            raise AiegraphError(_STAGE, f"partition row {L.get('idx')} {L['node']}: not an ONNX "
                                f"MatMul/Gemm with weight {L.get('weight')}")
        b = numpy_helper.to_array(inits[L["weight"]]).astype(np.float32)
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        wnk = b if (n.op_type == "Gemm" and at.get("transB", 0)) else b.T
        M, K, N = L["pad"]["orig"] if "pad" in L else L["mkn"][0]
        if wnk.shape != (N, K):
            raise AiegraphError(_STAGE, f"{L['node']}: weight {wnk.shape} != (N, K) = {(N, K)}")
        hit = []
        for c in dense.get((N, K), []):
            if c["call"] in used or c["xshape"][0] != M:
                continue
            on, kk, ob = c["wshape"]
            pk = _fp32(blob, offs, c["wname"]).reshape(on, kk, ob)
            if np.array_equal(pk.transpose(0, 2, 1).reshape(N, K), wnk):
                hit.append(c)
        if len(hit) != 1:
            raise AiegraphError(_STAGE, f"linear {L['node']} ({L['weight']} M={M} K={K} N={N}): "
                                f"{len(hit)} TVM dense calls carry this weight", hint="exactly one")
        c = hit[0]
        used.add(c["call"])
        if c["yshape"] != [M, N]:
            raise AiegraphError(_STAGE, f"{L['node']}: TVM output {c['yshape']} != {[M, N]}")
        gc = cg.calls[c["call"]]
        sym = gc.symbol if gc.symbol.startswith("tvmgen_") else "tvmgen_default_" + gc.symbol
        info = dict(kind="linear", call=c["call"], node=c["node"], symbol=sym, func=gc.symbol,
                    wname=c["wname"], warg=1, nargs=len(gc.ins) + len(gc.outs), wshape=c["wshape"],
                    xshape=c["xshape"], yshape=c["yshape"], cid=L["idx"], onnx=L["node"],
                    weight=L["weight"], macs=M * N * K, M=M, N=N, K=K, sh=1, sw=1, ph=0, pw=0)
        out.append(_one_call(cg, gc, f"lin{L['idx']}", c["wt"], info))
    return out
