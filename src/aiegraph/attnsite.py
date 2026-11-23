###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.attnsite -- attention bmm sites on gemm_kacc and the graph_driver.c glue of the AIE call sites.

    from aiegraph import attnsite
    att = attnsite.attn_sites(partition, workdir, onnx=...)    # partition attn rows + kernel geometry
    shapes = attnsite.attn_shapes(att)                         # gemm_kacc instances (ch, K, px) of the sites
    kw = attnsite.attn_kernel_parts(att)                       # kincore.write_source extra parts (no FA)
    text, n = attnsite.patch_driver(text, sel)                 # driver text path (no aiegraph group steps)

Generic over the model: every rule is keyed on the bmm shape (H, M, K, N); the rows come from
``attention.partition_attention`` (matched by ONNX node name, never by a model or layer name).

  kacc_plan     a GEMM (ch, K, px) on ONE gemm_kacc instance: whole, else S equal px launches, else S ch launches
  bmm_geom      geometry of one attention bmm Y[h] = X[h] (M x K) . W[h]^T on one instance (pad, swap, split)
  attn_sites    partition attn rows -> TVM nn_batch_matmul calls + geometry + launches + MACs
  attn_shapes   the distinct gemm_kacc instances of the sites
  attn_kernel_parts  the kacc bmm host entry + buffers for kincore.kernel_source
  patch_driver  the call sites aiegraph has not placed as group steps -> aiednn_aie_conv / _linear / _attn, plus
                the declarations and the PROF glue (aiednn attention/ + aiegraph runtime/ templates)
  prof_c_attn   the PROF glue with the attention branch (op=aie_host_attn<cid>)
"""
from __future__ import annotations

import re
from pathlib import Path

from aiednn import template as _tpl

from .cnn import kacc_geom
from .errors import AiegraphError
from .gemmpad import pad_mkn
from .kincore import SPLITS

_STAGE = "attnsite"
__all__ = ["kacc_plan", "bmm_geom", "attn_sites", "attn_shapes", "attn_kernel_parts", "patch_driver",
           "prof_c_attn", "ATTN_HOST", "DECL", "DECL_ATTN"]

ATTN_HOST = _tpl("attention/kacc_bmm_host.cc")
DECL = _tpl("gemm/kacc_site_decl.h")              # aiednn_aie_conv / aiednn_aie_linear (driver text path)
DECL_ATTN = _tpl("attention/kacc_bmm_decl.h")     # aiednn_aie_attn of a gemm_kacc bmm site
SWAP_MIN_OUT = 4 << 20                             # H*M*N from which a square bmm runs swapped (ch = M)
CALL_RE = re.compile(r"    GRAPH_PROF_BEGIN\((\d+)\);\n(#ifdef GRAPH_AIE_OFFLOAD\n.*?#endif\n|"
                     r"    \w+\(args, codes, \d+, 0, 0, 0\);\n    GRAPH_PROF_END\(\1, \"[^\"]*\"\);\n)", re.S)
_TVM_CALL = re.compile(r"(?:aie_)?(tvmgen_\w+)\(args, codes, (\d+), 0, 0, 0\);")


def _fits(ch: int, k: int, px: int) -> bool:
    try:
        kacc_geom(ch, k, px)
        return True
    except AiegraphError:
        return False


def kacc_plan(ch: int, k: int, px: int) -> tuple:
    """(mode, S, ch per launch, px per launch) of a GEMM on ONE gemm_kacc instance: ("none", 1, ch, px) when
    (ch, k, px) has a walk, else the smallest px split S in SPLITS (px % 64 S == 0), else the smallest ch split.
    No split fits -> AiegraphError."""
    if _fits(ch, k, px):
        return "none", 1, ch, px
    for s in SPLITS:
        if px % (64 * s) == 0 and _fits(ch, k, px // s):
            return "px", s, ch, px // s
    for s in SPLITS:
        if ch % (64 * s) == 0 and _fits(ch // s, k, px):
            return "ch", s, ch // s, px
    raise AiegraphError(_STAGE, f"kacc_plan: no split of ch={ch} K={k} px={px} fits gemm_kacc",
                        hint=f"px / ch splits tried: {SPLITS}")


def bmm_geom(H: int, M: int, K: int, N: int) -> dict:
    """Geometry of one attention bmm Y[h] = X[h] (M x K) . W[h]^T (W [N][K]) on ONE gemm_kacc instance:
    M / K / N zero-padded to 64 (gemmpad.pad_mkn), swapped (ch = M) for big square outputs, kacc_plan split,
    one more zero ch block while the a / b ports would have the same shape (aiehlc GemmSpace port sizing),
    K padded until kp/2 is not 32 / 64 (aiehlc 2-dim+iter b BD)."""
    if min(H, M, K, N) < 1:
        raise AiegraphError(_STAGE, f"bmm_geom: H={H} M={M} K={K} N={N} must all be >= 1")
    Mp, Kp, Np = pad_mkn(M, K, N)
    swap = int(Mp == Np and H * M * N >= SWAP_MIN_OUT)
    CHR, PXR = (M, N) if swap else (N, M)
    CHP, PXP = (Mp, Np) if swap else (Np, Mp)
    mode, S, chs, pxs = kacc_plan(CHP, Kp, PXP)
    geo = kacc_geom(chs, Kp, pxs)
    while geo["arows"] == geo["brows"] and mode == "none":
        CHP += 64
        mode, S, chs, pxs = kacc_plan(CHP, Kp, PXP)
        geo = kacc_geom(chs, Kp, pxs)
    while geo["kp"] // 2 in (32, 64):
        Kp += 64
        geo = kacc_geom(chs, Kp, pxs)
    return dict(H=H, M=M, K=K, N=N, swap=swap, CHR=CHR, PXR=PXR, CHP=CHP, PXP=PXP, Kp=Kp, kp=geo["kp"],
                kch=geo["kch"], tr=geo["tr"], splitpx=int(mode == "px"), S=S, chs=chs, pxs=pxs)


def attn_sites(partition: dict, workdir, onnx=None, log=None) -> list:
    """The partition's attn rows joined to their TVM nn_batch_matmul calls (attention.partition_attention on the
    workdir's graph.json, + ONNX for the Q / K / V sources and the scale), each with its gemm_kacc geometry
    ``geo`` (bmm_geom), ``launches`` (H * S) and ``macs``. A partition without attn rows is an AiegraphError."""
    from .attention import partition_attention
    from .group import CallGraph
    if not any(L.get("cls") == "attn" for L in partition.get("layers", [])):
        raise AiegraphError(_STAGE, "attn_sites: the partition has no attn rows", hint="autopart --mode attn")
    att = partition_attention(CallGraph.load(Path(workdir)), partition, onnx=onnx, workdir=workdir)
    groups = {c["ag_group"].name: c["ag_group"] for c in att}
    if log:
        log(f"TVM graph: {len(att)} attention bmm calls in {len(groups)} aiegraph attention "
            f"groups ({sum(len(g.calls) for g in groups.values())} calls)")
    out = []
    for c in att:
        geo = bmm_geom(c["H"], c["M"], c["K"], c["N"])
        out.append(dict(c, geo=geo, launches=c["H"] * geo["S"], macs=c["H"] * c["M"] * c["N"] * c["K"]))
    if log:
        log(f"mapped {len(out)} partition attn bmm layers ({len({c['group'] for c in out})} groups) to TVM "
            f"nn_batch_matmul calls, {sum(c['launches'] for c in out)} gemm_kacc launches")
    return out


def attn_shapes(att: list) -> list:
    """The distinct gemm_kacc instances (ch, K, px) of the attention sites, sorted. A row without ``geo``
    (not from attn_sites) is an AiegraphError."""
    bad = [c.get("onnx", c.get("cid")) for c in att if "geo" not in c]
    if bad:
        raise AiegraphError(_STAGE, f"attn_shapes: rows without geometry {bad[:3]}", hint="rows from attn_sites()")
    return sorted({(c["geo"]["chs"], c["geo"]["Kp"], c["geo"]["pxs"]) for c in att})


def attn_kernel_parts(att: list) -> dict:
    """The extra parts of ``kincore.kernel_source`` for gemm_kacc bmm sites (no flash attention): the
    aiednn_aie_attn export, the kacc bmm host entry and the per-site W / X / Y buffer sizes."""
    if not att:
        raise AiegraphError(_STAGE, "attn_kernel_parts: no attention site")
    attn_shapes(att)
    return dict(exports=("aiednn_aie_attn",), host_entries=ATTN_HOST.lstrip("\n"),
                bufs=[(g["CHP"] * g["kp"], g["PXP"] * g["kp"], g["CHP"] * g["PXP"]) for g in (c["geo"] for c in att)])


def prof_c_attn(kprefix: str = None) -> str:
    """The per-call-site PROF glue with the attention branch: attention sites print op=aie_host_attn<cid>
    (boardmain.prof_c(attn=True)); a site_prof.c without its branch anchor is an AiegraphError."""
    from .boardmain import prof_c
    try:
        return prof_c(attn=True, kprefix=kprefix)
    except ValueError as e:
        raise AiegraphError(_STAGE, f"prof_c_attn: {e}")


def _site(text: str, pos: int, after: int, c: dict, who) -> re.Match:
    cm = CALL_RE.search(text, pos)
    nxt = text.find("    args[0].v_handle", after)
    if not cm or not (nxt < 0 or cm.start() < nxt):
        raise AiegraphError(_STAGE, f"{who}: no TVM call site after its args block")
    calls = _TVM_CALL.findall(cm.group(2))
    if not calls or not all(s == c["symbol"] and int(k) == c["nargs"] for s, k in calls):
        raise AiegraphError(_STAGE, f"{who}: site calls {calls}, expected {c['symbol']} with {c['nargs']} args")
    return cm


def _patch_attn_site(text: str, c: dict) -> str:
    """One TVM nn_batch_matmul call site (located by its unique args[0..2] tensor block) -> aiednn_aie_attn.
    No weight: both operands are activations, packed at run time by the host (aie_kacc_pack.h)."""
    e0, e1 = c["eids"]
    blk = ("    args[0].v_handle = &tensors[%d];\n    args[1].v_handle = &tensors[%d];\n"
           "    args[2].v_handle = &tensors[%d];\n" % (e0, e1, c["yeid"]))
    if text.count(blk) != 1:
        raise AiegraphError(_STAGE, f"attn site {c['onnx']} (node {c['node']}): args block found "
                            f"{text.count(blk)} times, need 1")
    pos = text.index(blk)
    cm = _site(text, pos, pos + len(blk), c, f"attn site {c['onnx']}")
    k = cm.group(1)
    note = c["onnx"].replace("*/", "*_/")
    if c.get("fa") is not None:  # flash attention: role 0 = QK (pack Q, K), role 1 = PV (pack V, launch, write O)
        call = "aiednn_aie_attn(%d, args, %d, %d, %d);   /* fa %s g%d %s */" % (
            c["cid"], c["nargs"], int(c["role"] == "pv"), c["fa"], c["role"], c["group"], note)
    else:
        g = c["geo"]
        call = ("aiednn_aie_attn(%d, args, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);   /* %s g%d %s */"
                % (c["cid"], c["nargs"], int(c["role"] == "qk"), g["swap"], g["CHP"], g["PXP"], g["Kp"], g["kp"],
                   g["kch"], g["splitpx"], g["S"], g["chs"], g["pxs"], g["tr"], c["role"], c["group"], note))
    rep = ("    GRAPH_PROF_BEGIN(%s);\n    UP_AIE_BEGIN(%s);\n    %s\n"
           "    UP_AIE_END(%s);\n    GRAPH_PROF_END(%s, \"aie aiednn_aie_attn %d\");\n" % (k, k, call, k, k, c["cid"]))
    return text[:cm.start()] + rep + text[cm.end():]


def _patch_weight_site(text: str, c: dict) -> str:
    """One conv / linear call site whose bf16 weight is read in place: drop its graph_bf16_expand, call
    aiednn_aie_conv / aiednn_aie_linear with the weight's offset in g_params."""
    m = re.search(r"    tensors\[(\d+)\]\.data = 0;   /\* %s: bf16 @(\d+)\+(\d+) B" % re.escape(c["wname"]), text)
    if not m:
        raise AiegraphError(_STAGE, f"no bf16 init line for {c['wname']}", hint="AIEGRAPH_WDTYPE=bf16")
    eid, off, nb = int(m.group(1)), int(m.group(2)), int(m.group(3))
    c["woff"] = off   # wlayout.prelayout_weights rewrites this bf16 region of weights.bin
    wel = 1
    for d in c["wshape"]:
        wel *= d
    if nb != 2 * wel:
        raise AiegraphError(_STAGE, f"{c['wname']}: {nb} B bf16 for {wel} elements")
    exp = re.compile(r"    graph_bf16_expand\(\(float \*\)\(g_wscratch \+ (\d+)\), \(const uint16_t \*\)"
                     r"\(g_params \+ %d\), %dULL\);\n    tensors\[%d\]\.data = g_wscratch \+ \1;\n" % (off, wel, eid))
    if len(exp.findall(text)) != 1:
        raise AiegraphError(_STAGE, f"{c['wname']}: graph_bf16_expand found {len(exp.findall(text))} times, need 1")
    text = exp.sub("    tensors[%d].data = 0;   /* AIE conv %d: bf16 weight read in place by aiednn_aie_conv */\n"
                   % (eid, c["cid"]), text)
    a1 = "    args[1].v_handle = &tensors[%d];\n" % eid
    if text.count(a1) != 1:
        raise AiegraphError(_STAGE, f"{c['wname']}: weight args line found {text.count(a1)} times, need 1")
    pos = text.index(a1)
    cm = _site(text, pos, pos, c, c["wname"])
    k = cm.group(1)
    if c.get("kind") == "linear":   # PROF op=aie_host_linear<cid>
        call = "aiednn_aie_linear(%d, args, %d, (const uint16_t *)(g_params + %d), %d);" % (
            c["cid"], c["nargs"], off, c["S"])
        tag = "aie aiednn_aie_lin %d" % c["cid"]
    else:
        call = "aiednn_aie_conv(%d, args, %d, (const uint16_t *)(g_params + %d), %d, %d, %d, %d);" % (
            c["cid"], c["nargs"], off, c["sh"], c["sw"], c["ph"], c["pw"])
        tag = "aie aiednn_aie_conv %d" % c["cid"]
    rep = ("    GRAPH_PROF_BEGIN(%s);\n    UP_AIE_BEGIN(%s);\n    %s\n"
           "    UP_AIE_END(%s);\n    GRAPH_PROF_END(%s, \"%s\");\n" % (k, k, call, k, k, tag))
    return text[:cm.start()] + rep + text[cm.end():]


def patch_driver(text: str, sel: list) -> tuple:
    """The driver text patches aiegraph does not do as group steps. When aiegraph already placed the conv /
    linear group steps (their declarations sit before kernel_ws), *sel* must hold attention rows only and only
    those sites, their declaration and the PROF glue are added; without aiegraph steps the conv / linear sites
    are patched too. Flash-attention rows must already be aiegraph group steps (``fa_steps``): only their
    declarations are added. Returns (text, patched sites). A driver patched twice is an AiegraphError."""
    from .boardmain import EXPAND_ENTER, EXPAND_EXIT, RUN_ENTER, prof_c
    from .flashattn import DECL_FA, DECL_KL
    head = text.split("static uint8_t kernel_ws[")[0]
    if "UP_AIE_BEGIN(" in head or "up_pending_exp" in head:
        raise AiegraphError(_STAGE, "patch_driver: driver already patched")
    ag_steps = "int aiednn_aie_conv(int cid, void" in head
    att = [c for c in sel if c.get("kind") == "attn"]
    sel = [c for c in sel if c.get("kind") != "attn"]
    if ag_steps and sel:
        raise AiegraphError(_STAGE, "patch_driver: aiegraph placed the conv/linear steps, pass attn rows only")
    i = text.index("static uint8_t kernel_ws[")
    fam = bool(att) and all(c.get("fa") is not None for c in att)
    if fam and not all(c.get("fa_steps") for c in att):
        raise AiegraphError(_STAGE, "patch_driver: flash-attention sites must be aiegraph group steps")
    decl = ((DECL_FA + DECL_KL) if fam else DECL_ATTN) if att else ""
    text = text[:i] + ("" if ag_steps else DECL) + decl + (prof_c_attn() if att else prof_c()) + "\n" + text[i:]
    exp_body = "    for (uint64_t i = 0; i < n; ++i) d[i] = (uint32_t)src[i] << 16;\n}\n"
    if text.count(exp_body) != 1:
        raise AiegraphError(_STAGE, f"patch_driver: graph_bf16_expand body found {text.count(exp_body)} times")
    text = text.replace(exp_body, EXPAND_ENTER + exp_body[:-2] + EXPAND_EXIT + "}\n", 1)
    run = "float *graph_run(void) {\n"
    if text.count(run) != 1:
        raise AiegraphError(_STAGE, "patch_driver: graph_run not found once")
    text = text.replace(run, run + RUN_ENTER, 1)
    n = 0
    for c in sel:
        text = _patch_weight_site(text, c)
        n += 1
    nl = sum(1 for c in sel if c.get("kind") == "linear")
    if not ag_steps and (text.count("    aiednn_aie_conv(") != len(sel) - nl
                         or text.count("    aiednn_aie_linear(") != nl):
        raise AiegraphError(_STAGE, "patch_driver: conv / linear call site count mismatch")
    if fam:   # the FA group steps arm_build emitted: 2 per bmm site (pack_q+pack_k / pack_v+run_o)
        nkl = len(re.findall(r"^    aiednn_aie_attn_kl\(", text, re.M))
        if nkl != 2 * len(att):
            raise AiegraphError(_STAGE, f"patch_driver: {nkl} aiednn_aie_attn_kl steps for {len(att)} FA sites")
        if re.search(r"\btvmgen_default_fused_nn_(softmax|batch_matmul)\w*\(args, ",
                     text.split("float *graph_run(void)")[1]):
            raise AiegraphError(_STAGE, "patch_driver: attention call left in graph_run")
        return text, n + len(att)
    for c in att:   # PROF op=aie_host_attn<cid>
        text = _patch_attn_site(text, c)
        n += 1
    if text.count("    aiednn_aie_attn(") != len(att):
        raise AiegraphError(_STAGE, "patch_driver: attention call site count mismatch")
    return text, n
