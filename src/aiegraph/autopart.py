###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Automatic AIE / CPU partition of an ONNX model's GEMM layers (GAP-A, m2-s6).

Before this pass the diffusion example's partition.json was hand-kept (P35x: "Steve run E" excluded parts
0/1/21/23/24 by hand, then a launch budget N*6.9 ms <= 600 s cut parts 19/20/22). Here the
partition is *generated* from the graph:

  1. inventory   every Conv / MatMul / Gemm of the ONNX graph, classified
                   conv    Conv (im2col GEMM: M = pixels, K = Cin*KH*KW, N = Cout)
                   linear  MatMul / Gemm with a constant weight (initializer), M > 1
                   attn    MatMul with two activation inputs (QK^T, PV) -- stays CPU (PATN decides)
                   gemv    a constant-weight Gemm/MatMul with M == 1 (time embedding) -- not a GEMM
  2. parts       nodes -> parts from the module names (/u/<blocks.X>/<kind.Y>/...): one part per
                 sub-block, the stem (time embedding), conv_in, and the head after the last sub-block
  3. legality    a layer goes to the AIE iff its kincore plan (plan.kincore_geom, one launch per
                 layer, K in the core) is legal and K % 64 == N % 64 == 0. M (activation rows)
                 may be zero-padded up to the 64 tile (text K/V: M=77 -> 128), recorded in "pad".
                 K/N padding (conv_in K=36, conv_out N=4) needs weight repacking = GKP; off by
                 default (``pad_kn=True`` turns it on).
                 A layer whose single-launch walk does not fit is split along N into the fewest
                 equal 64-multiple pieces that fit (mkn gets one row per launch).
  4. mode        conv: every legal conv; l37: the legal convs of the board-proven --base set
                 (today's 37) + every legal linear; all: every legal conv and linear;
                 attn: all + every attention bmm of a partition.match_attention core
                 (bmm -> softmax -> bmm, GPAT, m2-s7). An attn row is one bmm, batch (heads)
                 looped, M/K/N zero-padded to 64 (activations: no weight repack), and carries
                 attn_group (one per softmax) + attn_role (qk | pv). Optional ``budget_ms`` trims greedily by MAC/launch (the P35x rule).

Output = the partition.json schema of the diffusion example's partition.json (layers[] with part, file,
node, op, cls, weight, macs, launches (64x64x64 tile launches), mkn, idx; N; aie_macs;
total_gemm_macs; aie_flop_frac; cpu_gemm_layers[]), so the example build script and kplan.py read it unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import AiegraphError
from .plan import kincore_geom

__all__ = ["MODES", "inventory", "assign_parts", "autopartition", "write_partition"]

MODES = ("conv", "l37", "all", "attn")
TILE = 64
_SUB = re.compile(r"^/u/((?:down|up|mid)_blocks?\.\d+/[a-z_]+\.\d+|mid_block)/")
_TOP = re.compile(r"^/u/([^/]+)/")


def _ceil(a, b):
    return -(-a // b)


def _up(v, t=TILE):
    return _ceil(v, t) * t


def _dims(vi):
    return [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]


def _load(model):
    import onnx
    if not isinstance(model, (str, Path)):
        return model
    return onnx.load(str(model), load_external_data=False)


def _shapes(m):
    import onnx
    shp = {v.name: _dims(v) for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)}
    shp.update({t.name: list(t.dims) for t in m.graph.initializer})
    need = [i for n in m.graph.node if n.op_type in ("Conv", "MatMul", "Gemm") for i in list(n.input[:2]) + [n.output[0]]]
    if any(shp.get(x) is None or None in shp[x] for x in need):
        m2 = onnx.shape_inference.infer_shapes(m)
        shp.update({v.name: _dims(v) for v in list(m2.graph.value_info) + list(m2.graph.output)})
    return shp


def _key(name: str):
    s = _SUB.match(name)
    if s:
        return s.group(1), True
    t = _TOP.match(name)
    return (t.group(1), False) if t else (None, False)


def assign_parts(m) -> dict:
    """{node name: (part index, part key)} from the module names, graph order.

    A part starts at each sub-block (<blocks.X>/<kind.Y>); glue that belongs to a bare block
    (/u/up_blocks.0/Concat) joins the next sub-block; before the first sub-block the stem (no
    conv) and every top-level conv module (conv_in) are parts; after the last sub-block
    everything is one head part. Unnamed nodes (MatMul_0, Gemm_3) take the part of their
    nearest named producer."""
    nodes = list(m.graph.node)
    keys = [_key(n.name) for n in nodes]
    prod = {o: i for i, n in enumerate(nodes) for o in n.output}
    sub_idx = [i for i, (_, s) in enumerate(keys) if s]
    if not sub_idx:
        raise AiegraphError("autopart", "no <blocks.X>/<kind.Y> sub-blocks in the node names",
                            hint="autopart parts a diffusers-style export (/u/<blocks>.0/<kind>.0/...)")
    first, last = sub_idx[0], sub_idx[-1]
    conv_top = {k for (k, s), n in zip(keys, nodes) if not s and k and n.op_type == "Conv"}

    def named(i, depth=0):
        k, s = keys[i]
        if k is not None or depth > 64:
            return k, s
        for x in nodes[i].input:
            if x in prod:
                return named(prod[x], depth + 1)
        return None, False

    out, part, cur = {}, -1, None
    pending = []                          # bare-block glue waiting for its sub-block
    for i, n in enumerate(nodes):
        k, s = named(i)
        if i > last and not s:
            k2 = "head"
        elif s:
            k2 = k
        elif i < first:
            k2 = k if k in conv_top else (cur if cur not in conv_top and cur is not None else "stem")
        else:                             # between sub-blocks, not a sub-block: glue -> next part
            pending.append(n.name)
            continue
        if k2 != cur:
            part += 1
            cur = k2
        out[n.name] = (part, cur)
        for p in pending:
            out[p] = (part, cur)
        pending = []
    return out


def inventory(model) -> list:
    """Every GEMM-like node -> dict(node, op, cls, weight, m, k, n, macs) in graph order."""
    m = _load(model)
    shp = _shapes(m)
    init = {t.name for t in m.graph.initializer}
    rows = []
    for n in m.graph.node:
        at = {a.name: __import__("onnx").helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type == "Conv":
            w, y = shp.get(n.input[1]), shp.get(n.output[0])
            O, CI, KH, KW = w
            why = []
            if at.get("group", 1) != 1:
                why.append(f"group{at['group']}")
            if list(at.get("dilations", [1, 1])) != [1, 1]:
                why.append("dilation")
            M, K, N = y[0] * y[2] * y[3], CI * KH * KW, O
            rows.append(dict(node=n.name, op="Conv", cls="conv", weight=n.input[1], m=M, k=K, n=N,
                             macs=M * K * N, why=why))
        elif n.op_type in ("MatMul", "Gemm"):
            a, b = shp.get(n.input[0]), shp.get(n.input[1])
            if n.input[1] not in init:
                # both inputs activations: batched QK^T / PV
                bat = 1
                for d in a[:-2]:
                    bat *= d
                M, K, N = a[-2], a[-1], b[-1]
                rows.append(dict(node=n.name, op=n.op_type, cls="attn", weight=None, m=M, k=K, n=N,
                                 batch=bat, macs=bat * M * K * N, why=["activation x activation"]))
                continue
            if n.op_type == "Gemm":
                if at.get("transA", 0):
                    raise AiegraphError("autopart", f"{n.name}: Gemm transA=1 not handled")
                M, K = a[0], a[1]
                N = b[0] if at.get("transB", 0) else b[1]
            else:
                M = 1
                for d in a[:-1]:
                    M *= d
                K, N = a[-1], b[-1]
            cls = "gemv" if M == 1 else "linear"
            rows.append(dict(node=n.name, op=n.op_type, cls=cls, weight=n.input[1], m=M, k=K, n=N,
                             macs=M * K * N, why=["M=1 (gemv)"] if cls == "gemv" else []))
    return rows


def _legal(r, pad_kn):
    """(legal, [mkn per launch], pad dict, why).

    One kincore launch per layer when its walk fits; otherwise the GEMM is split along N
    (weight columns) into the fewest equal 64-multiple pieces whose walk fits, one launch
    each (the 1024x320x960 fused QKV and the 1024x320x2560 gated-GELU projection need 3 / 4)."""
    if r["cls"] not in ("conv", "linear") or r["why"]:
        return False, None, {}, list(r["why"]) or [r["cls"]]
    M, K, N = r["m"], r["k"], r["n"]
    why = []
    if not pad_kn:
        why += [f"{nm}%64" for nm, v in (("K", K), ("N", N)) if v % TILE]
    if why:
        return False, None, {}, why
    Mp, Kp, Np = _up(M), _up(K), _up(N)
    pad = {nm: p for nm, v, p in (("M", M, Mp), ("K", K, Kp), ("N", N, Np)) if p != v}
    first = None
    for s in range(1, Np // TILE + 1):
        if Np % s or (Np // s) % TILE:
            continue
        g = kincore_geom(Np // s, Kp, Mp)
        if first is None:
            first = g
        if g["legal"]:
            if s > 1:
                pad["nsplit"] = s
            return True, [[Mp, Kp, Np // s]] * s, pad, []
    return False, None, {}, [f"kincore:{w}" for w in first["why"]]


def _fit(Mp, Kp, Np):
    """Fewest equal 64-multiple N pieces whose kincore walk fits: (pieces, first geom)."""
    first = None
    for s in range(1, Np // TILE + 1):
        if Np % s or (Np // s) % TILE:
            continue
        g = kincore_geom(Np // s, Kp, Mp)
        if first is None:
            first = g
        if g["legal"]:
            return s, g
    return None, first


def _legal_attn(r):
    """Attention bmm: per batch (head) one GEMM, M/K/N padded to 64 (both operands are
    activations, so padding is a zero-filled copy, not a weight repack)."""
    M, K, N, B = r["m"], r["k"], r["n"], r.get("batch", 1)
    Mp, Kp, Np = _up(M), _up(K), _up(N)
    s, g = _fit(Mp, Kp, Np)
    if s is None:
        return False, None, {}, [f"kincore:{w}" for w in g["why"]]
    pad = {nm: p for nm, v, p in (("M", M, Mp), ("K", K, Kp), ("N", N, Np)) if p != v}
    if s > 1:
        pad["nsplit"] = s
    pad["batch"] = B
    return True, [[Mp, Kp, Np // s]] * (s * B), pad, []


def autopartition(model, mode: str = "conv", base=None, ms_per_launch: float = 7.471,
                  pad_kn: bool = False, budget_ms: float = None,
                  log=print) -> dict:
    """Generate the partition dict for ``mode`` (conv | l37 | all | attn) from the ONNX ``model``.

    ``base``: an older partition.json (path or dict) -- only compared against (the hand
    excludes / budget cuts are re-checked and reported), never copied."""
    if mode not in MODES:
        raise AiegraphError("autopart", f"mode={mode!r}", hint=f"one of {MODES}")
    m = _load(model)
    rows = inventory(m)
    parts = assign_parts(m)
    agrp = {}
    if mode == "attn":
        from .partition import match_attention
        for a in match_attention(m):
            agrp[a.qk] = (a.group, "qk", a.softmax)
            agrp[a.pv] = (a.group, "pv", a.softmax)
    part_key = {}
    for nm, (p, k) in parts.items():
        part_key.setdefault(p, k)
    gemm = [r for r in rows if r["cls"] != "gemv"]
    bdoc = (json.loads(Path(base).read_text()) if isinstance(base, (str, Path)) else base) if base is not None else None
    proven = {l["node"] for l in bdoc.get("layers", []) if l.get("cls") == "conv"} if bdoc else set()
    if mode == "l37" and not proven:
        raise AiegraphError("autopart", "mode l37 needs --base (the board-proven conv set, P35x 37 convs)",
                            hint="--base <example dir>/partition.json")
    total = sum(r["macs"] for r in gemm)
    want_cls = {"conv": ("conv",), "attn": ("conv", "linear", "attn")}.get(mode, ("conv", "linear"))
    aie, cpu = [], []
    for r in gemm:
        if r["cls"] == "attn" and mode == "attn":
            if r["node"] in agrp:
                ok, mkn, pad, why = _legal_attn(r)
                r["attn_group"], r["attn_role"], r["softmax"] = agrp[r["node"]]
            else:
                ok, mkn, pad, why = False, None, {}, ["attn: not in a bmm->softmax->bmm core"]
        else:
            ok, mkn, pad, why = _legal(r, pad_kn)
        if ok and r["cls"] not in want_cls:
            ok, why = False, [f"mode {mode}: {r['cls']} on CPU"]
        if ok and mode == "l37" and r["cls"] == "conv" and r["node"] not in proven:
            ok, why = False, ["mode l37: conv not in the board-proven --base set"]
        p = parts[r["node"]][0]
        r.update(part=p, aie=ok, mkn=mkn, pad=pad, cpu_why=why)
        (aie if ok else cpu).append(r)
    if budget_ms is not None:
        aie.sort(key=lambda r: -r["macs"])
        keep, ms = [], 0.0
        for r in aie:
            if ms + ms_per_launch <= budget_ms:
                keep.append(r)
                ms += ms_per_launch
            else:
                r.update(aie=False, cpu_why=[f"budget {budget_ms} ms"])
                cpu.append(r)
        aie = sorted(keep, key=lambda r: gemm.index(r))
    order = {r["node"]: i for i, r in enumerate(gemm)}
    aie.sort(key=lambda r: order[r["node"]])
    cpu.sort(key=lambda r: order[r["node"]])

    def fname(p):
        return f"part_{p:02d}_{re.sub(r'[^0-9A-Za-z]+', '_', part_key[p]).strip('_')}.onnx"

    layers = []
    for i, r in enumerate(aie):
        lay = dict(part=r["part"], file=fname(r["part"]), node=r["node"], op=r["op"], cls=r["cls"],
                   weight=r["weight"], macs=r["macs"],
                   launches=sum(_ceil(a, TILE) * _ceil(c, TILE) * _ceil(b, TILE) for a, b, c in r["mkn"]),
                   mkn=r["mkn"], idx=i)
        if r["pad"]:
            lay["pad"] = dict(r["pad"], orig=[r["m"], r["k"], r["n"]])
        if r["cls"] == "attn":
            lay.update(attn_group=r["attn_group"], attn_role=r["attn_role"], softmax=r["softmax"],
                       batch=r.get("batch", 1))
        layers.append(lay)
    aie_macs = sum(r["macs"] for r in aie)
    kin = sum(len(l["mkn"]) for l in layers)   # kincore: one launch per (N-split) piece
    doc = dict(
        sprint="m2-s6", phase="GPA", generator="aiegraph.autopart", mode=mode,
        onnx=str(model) if isinstance(model, (str, Path)) else None, wdtype="bf16",
        rule=dict(classes=list(want_cls), legality="plan.kincore_geom legal and K%64==N%64==0; M padded to 64",
                  pad_kn=pad_kn, l37_convs="board-proven --base layers (P35x)" if mode == "l37" else None, budget_ms=budget_ms,
                  ms_per_launch=ms_per_launch,
                  attn=("AIE: partition.match_attention bmm->softmax->bmm, per-head, M/K/N padded to 64"
                        if mode == "attn" else "CPU (PATN)"), gemv="M==1, not a GEMM"),
        N=sum(l["launches"] for l in layers), kincore_launches=kin,
        aie_ms_est=round(kin * ms_per_launch, 3),
        aie_macs=aie_macs, total_gemm_macs=total, aie_flop_frac=aie_macs / total if total else 0.0,
        machine=__import__("platform").machine(),
        layers=layers,
        cpu_gemm_layers=[dict(part=r["part"], node=r["node"], cls=r["cls"], macs=r["macs"],
                              why=r["cpu_why"]) for r in cpu])
    _report(doc, gemm, parts, base, log)
    return doc


def _report(doc, gemm, parts, base, log):
    cls = {}
    for l in doc["layers"]:
        cls[l["cls"]] = cls.get(l["cls"], 0) + 1
    for r in gemm:
        if r["cls"] == "conv":
            log(f"AP_CONV part={r['part']} aie={int(r['aie'])} mkn={r['m']}x{r['k']}x{r['n']} "
                f"macs={r['macs']} node={r['node']}" + ("" if r["aie"] else " why=" + ",".join(r["cpu_why"])))
    if base is not None:
        b = json.loads(Path(base).read_text()) if isinstance(base, (str, Path)) else base
        bl = {l["node"]: l for l in b.get("layers", []) + b.get("cpu_gemm_layers", [])}
        mism = [n for n in bl if n in parts and parts[n][0] != bl[n]["part"]]
        missing = [n for n in bl if n not in parts]
        log(f"AP_BASE nodes={len(bl)} part_match={len(bl) - len(mism) - len(missing)} "
            f"part_mismatch={len(mism)} missing={len(missing)} base_total={b.get('total_gemm_macs')} "
            f"total_match={int(b.get('total_gemm_macs') == doc['total_gemm_macs'])}")
        bset = {l["node"] for l in b.get("layers", [])}
        now = {l["node"] for l in doc["layers"]}
        rule = b.get("rule", {})
        for p in sorted(set(rule.get("exclude_parts", [])) | {19, 20, 22}):
            pl = [r for r in gemm if r["part"] == p and r["cls"] == "conv"]
            on = sum(1 for r in pl if r["node"] in now)
            tag = "hand_exclude" if p in rule.get("exclude_parts", []) else "budget_cut"
            why = sorted({w for r in pl if r["node"] not in now for w in r["cpu_why"]})
            log(f"AP_RECHECK part={p} base={tag} convs={len(pl)} aie_now={on} "
                f"base_aie={sum(1 for r in pl if r['node'] in bset)}" + (f" cpu_why={';'.join(why)}" if why else ""))
    log(f"AP mode={doc['mode']} layers={len(doc['layers'])} conv={cls.get('conv', 0)} "
        f"linear={cls.get('linear', 0)} attn={cls.get('attn', 0)} "
        f"attn_groups={len({l['attn_group'] for l in doc['layers'] if 'attn_group' in l})} cpu_gemm={len(doc['cpu_gemm_layers'])} "
        f"cpu_conv={sum(1 for l in doc['cpu_gemm_layers'] if l['cls'] == 'conv')} "
        f"padded={sum(1 for l in doc['layers'] if 'pad' in l)} N={doc['N']} "
        f"kincore_launches={doc['kincore_launches']} aie_ms_est={doc['aie_ms_est']} "
        f"aie_macs={doc['aie_macs']} total_gemm_macs={doc['total_gemm_macs']} "
        f"frac={doc['aie_flop_frac']:.4f}")


def write_partition(doc: dict, out) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    return out
