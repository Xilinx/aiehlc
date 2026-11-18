###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Keep-layout + temb fold in the arm_build codegen (design 2026-11-16-aiegraph-group-offload, step S5).

``partition.plan_keep_layout`` decides on the ONNX graph which AIE outputs may stay in the AIE layout
(the ops on the way are labels only). This module is the codegen half (GAP 6a9d38: the plan was never
wired into code generation): a *driver pass* that ``arm_build.build_arm_elf`` runs on the freshly generated
``graph_driver.c`` / ``weights.bin`` of a workdir (``register(workdir, ...)``, or
``EmbeddedModel.keep_layout(...)``). It replaces the example-side ``keep_layout_rt.py``.

Four rewrites; each removes A78 layout calls that only permute data, folding the permutation into work that
already exists (no call moves, so TVM's storage plan still holds):

* temb (``[sigmoid_squeeze_]multiply_layout_transform*`` -> ``nn_contrib_dense_pack_add*``): the SiLU scale
  folded into the dense weight W*s and re-packed every call. The weight is pre-packed into weights.bin
  ([N][K] -> [N/8][K][8], bf16, bit-exact), the call only widens it, the dense reads x = t*s:
  dense(t, W*s) == dense(t*s, W).
* attention (head split Q / K / V^T, head merge O around every FA group): the FA packs read Q/K/V straight
  from the [rows][H*D] linear outputs (scale in the pack), the PV launch writes O into the merged buffer.
* transformer out (``reshape_add_reshape_transpose_layout_transform*``): the residual add goes into the
  linear's scatter, the following AIE 1x1 conv reads the [M][C] buffer as NHWC.
* conv out (``layout_transform[_transpose]_reshape*``): the AIE conv scatter writes the consumer's layout
  (planar NCHW / NHWC label) into the layout op's output buffer.

Every decision is checked on ``<stem>_graph.json`` (``group.CallGraph``; Relay is not reachable after
convert, relay.txt): purity = ``pure_perm.check`` on the Relay op semantics saved at convert
(``<stem>_opsem.json``): only reshape / transpose / layout_transform (+ the scale multiply / residual add / SiLU
of the temb fold), elements conserved, one output, the shapes match the group, the
removed intermediate has no other reader ("组外有读者"), the buffer written early is untouched in between
("被覆盖的 buffer"), and every planned ``keep_layout`` edge is realized. The TVM module C (when given) is
cross-checked as well. Anything else -> AiegraphError (the build fails, nothing is silently skipped).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import pure_perm
from .errors import AiegraphError
from .group import CallGraph

__all__ = ["apply", "keep_layout_driver", "register", "KeepLayoutPass"]

_STAGE = "keep_layout"


def _req(cond, *what):
    if not cond:
        raise AiegraphError(_STAGE, " ".join(str(w) for w in what),
                            hint="the call does not match the keep-layout pattern: build without keep_layout "
                                 "or fix the partition")


# ═══════════════════════════════════════════════════════════════════════════
#  purity of a fused op: saved Relay op semantics (pure_perm, <stem>_opsem.json)
# ═══════════════════════════════════════════════════════════════════════════

LABEL = pure_perm.LABEL


def _pure(cg: CallGraph, call: int, extra=(), need=()) -> list:
    """The call only permutes data (+ the ops in ``extra``): decided by ``pure_perm.check`` on the Relay op
    semantics recorded at convert (op sequence, shapes, transpose axes, element count, one output)."""
    return pure_perm.check(cg, call, extra=extra, need=need, stage=_STAGE)


def _elems(cg: CallGraph, t: int) -> int:
    return int(np.prod(cg.shape[cg.canon(t)])) if cg.shape[cg.canon(t)] else 1


def _readers(cg: CallGraph, t: int, removed: set) -> list:
    return [r for r in cg.readers.get(cg.canon(t), []) if r not in removed]


def _untouched_g(cg: CallGraph, lo: int, hi: int, t: int, removed: set, what: str) -> None:
    """no kept call strictly between calls lo and hi touches the buffer of tensor t (graph.json storage ids)"""
    sid = cg.storage(t)
    for k in range(lo + 1, hi):
        if k in removed:
            continue
        hit = [x for x in cg.touches(k) if x not in cg.const and cg.storage(x) == sid]
        _req(not hit, f"{what}: call [{k}] {cg.calls[k].fn} touches buffer sid {sid} of tensors[{t}] "
                      f"between [{lo}] and [{hi}] (overwritten buffer)")


# ═══════════════════════════════════════════════════════════════════════════
#  graph_driver.c structure (the text the codegen emits)
# ═══════════════════════════════════════════════════════════════════════════

PROF_RE = re.compile(r"    GRAPH_PROF_BEGIN\((\d+)\);\n(.*?)    GRAPH_PROF_END\(\1, \"([^\"]*)\"\);\n", re.S)
ARG_RE = re.compile(r"    args\[(\d+)\]\.v_handle = &(?:tensors\[(\d+)\]|(g_kl_\w+));\n")
LBL_RE = re.compile(r"    (g_kl_\w+) = tensors\[(\d+)\];(?: \1\.data = tensors\[(\d+)\]\.data;)?")
BUF_RE = re.compile(r"    tensors\[(\d+)\]\.data = (buf_\d+|g_wscratch \+ \d+);\n")
SHAPE_RE = re.compile(r"    shape_store\[(\d+)\]\[(\d+)\] = (\d+);\n")
FN_RE = re.compile(r"(?:aie_)?tvmgen_default_fused_(\w+)\(args, codes, (\d+), 0, 0, 0\);")
EXP_RE = re.compile(r"    graph_bf16_expand\(\(float \*\)\(g_wscratch \+ (\d+)\), \(const uint16_t \*\)\(g_params \+ (\d+)\), "
                    r"(\d+)ULL\);\n    tensors\[(\d+)\]\.data = g_wscratch \+ \1;\n")


def _calls(text: str) -> list:
    """graph_run call blocks in order: idx, prof span, op tag, fused fn, args (tensor ids), segment start."""
    body0 = text.index("float *graph_run(void)")
    out, seg = [], body0
    lbl = {"g_kl_x_t": -1}   # keep-layout labels -> the tensor whose buffer they point at
    for m in LBL_RE.finditer(text, body0):
        lbl[m.group(1)] = int(m.group(3) or m.group(2))
    for m in PROF_RE.finditer(text, body0):
        args = {}
        for a in ARG_RE.finditer(text, seg, m.start()):
            args[int(a.group(1))] = int(a.group(2)) if a.group(2) else lbl[a.group(3)]
        fm = FN_RE.search(m.group(2))
        site = re.search(r"unet_aie_(conv|linear|attn)\((\d+), args, (\d+)(?:, (\d+), (\d+))?", m.group(2))
        nargs = int(fm.group(2)) if fm else (int(site.group(3)) if site else 0)
        out.append({"idx": int(m.group(1)), "span": (m.start(), m.end()), "seg": seg, "tag": m.group(3),
                    "fn": fm.group(1) if fm else None, "site": site.group(0) if site else None,
                    "role": int(site.group(4)) if site and site.group(1) == "attn" and site.group(4) else None,
                    "args": [args[i] for i in range(nargs)] if all(i in args for i in range(nargs)) else
                    [args[i] for i in sorted(args)]})
        seg = m.end()
    return out


def _check_calls(cg: CallGraph, calls: list) -> None:
    """driver call [k] == graph.json call k (the TVM ones carry the fused name)"""
    for c in calls:
        _req(0 <= c["idx"] < len(cg.calls), f"driver call [{c['idx']}] not in graph.json ({len(cg.calls)} calls)")
        if c["fn"]:
            _req(cg.calls[c["idx"]].fn == c["fn"],
                 f"driver call [{c['idx']}] {c['fn']} != graph.json call {cg.calls[c['idx']].fn}")


def _tvm_body(csrc: str, fn: str) -> str:
    h = ("TVM_DLL int32_t tvmgen_default_fused_%s(void* args, int32_t* arg_type_ids, int32_t num_args, "
         "void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {\n" % fn)
    i = csrc.index(h)
    return csrc[i:csrc.index("\n}\n", i)]


def _remove(text: str, c: dict, why: str) -> str:
    a, b = c["span"]
    return text[:a] + "    /* keep-layout: call [%d] %s removed -- %s */\n" % (c["idx"], c["tag"][4:], why) + text[b:]


def _decl_point(text: str) -> int:
    i = text.index("float *graph_run(void)")
    return text.rindex("\n", 0, i) + 1


# ═══════════════════════════════════════════════════════════════════════════
#  temb: SiLU scale folded into the pre-packed dense weight
# ═══════════════════════════════════════════════════════════════════════════

TEMB_FN = re.compile(r"^(sigmoid_squeeze_)?multiply_layout_transform(_\d+)?$")
TEMB_BODY = re.compile(r"ax0_ax1_fused < (\d+); .*?ax2_inner < 8; .*?\(ax0_ax1_fused % (\d+)\).*?"
                       r"\(\(\(ax0_ax1_fused / (\d+)\) \* (\d+)\) \+ \(ax2_inner \* (\d+)\)\) \+ cse_var_1\)\]", re.S)

TEMB_C = r"""
/* keep-layout (aiegraph keep_layout_codegen): temb dense input x = t * sigmoid(t) (the weight is pre-packed in
 * weights.bin, so the old per-call W*s re-pack is gone) */
#include <math.h>
static float g_kl_x[4096] __attribute__((aligned(64)));
static DLTensor g_kl_x_t;
static void kl_temb_x(const DLTensor *t, const DLTensor *s, int k) {
    const float *tv = (const float *)t->data;
    g_kl_x_t = *t;
    g_kl_x_t.data = g_kl_x;
    if (s) {
        const float *sv = (const float *)s->data;
        for (int i = 0; i < k; i++) g_kl_x[i] = tv[i] * sv[i];
    } else {
        for (int i = 0; i < k; i++) g_kl_x[i] = tv[i] * (1.0f / (1.0f + expf(0.0f - tv[i])));
    }
}
"""


def temb_pass(text: str, cg: CallGraph, wbin: Path, csrc: str = None, removed: set = None, log=print) -> tuple:
    removed = set() if removed is None else removed
    calls = _calls(text)
    pos = {c["idx"]: i for i, c in enumerate(calls)}
    w = np.memmap(wbin, dtype=np.uint16, mode="r+")
    n = 0
    while True:
        todo = [cc for cc in calls if cc["fn"] and TEMB_FN.match(cc["fn"])]
        if not todo:
            break
        c = todo[0]
        g = cg.calls[c["idx"]]
        inline_sig = c["fn"].startswith("sigmoid_squeeze_")
        _pure(cg, c["idx"], extra=("multiply", "sigmoid") if inline_sig else ("multiply",),
              need=("multiply", "layout_transform"))
        _req(len(c["args"]) == 3 and len(g.ins) == 2, "temb call", c["idx"], c["args"])
        tw, ts, to = c["args"]
        if inline_sig:
            tw, ts = ts, tw
        _req(cg.canon(tw) in cg.const and len(cg.shape[cg.canon(tw)]) == 2,
             f"temb call [{c['idx']}]: weight tensors[{tw}] is not a 2-D constant")
        N, K = cg.shape[cg.canon(tw)]
        _req(N % 8 == 0 and tuple(cg.shape[cg.canon(to)]) == (N // 8, K, 8),
             f"temb call [{c['idx']}]: out {cg.shape[cg.canon(to)]} is not dense_pack [N/8][K][8] of W {N}x{K}")
        if csrc is not None:   # cross-check on the TVM C
            m = TEMB_BODY.search(_tvm_body(csrc, c["fn"]))
            _req(m, "temb body", c["fn"])
            nk8, K0, K_, K8, K__ = (int(x) for x in m.groups())
            _req(K0 == K_ == K__ == K and K8 == 8 * K and nk8 * 8 // K == N, "temb body", c["fn"], m.groups())
        seg = text[c["seg"]:c["span"][0]]
        em = [e for e in EXP_RE.finditer(seg) if int(e.group(4)) == tw]
        _req(len(em) == 1, "temb weight expand", c["idx"], tw)
        off, cnt = int(em[0].group(2)), int(em[0].group(3))
        _req(cnt == N * K, "temb weight size", c["idx"], cnt, N, K)
        _req(text.count("(g_params + %d)" % off) == 1, "temb weight shared", off)
        nxt = calls[pos[c["idx"]] + 1]
        _req(nxt["fn"] and nxt["fn"].startswith("nn_contrib_dense_pack_add") and nxt["args"][1] == to,
             "temb: the next call is not the dense_pack reading the packed weight", nxt["idx"], nxt["fn"])
        _req(_readers(cg, to, removed) == [nxt["idx"]], f"temb: packed weight tensors[{to}] has other readers "
             f"{_readers(cg, to, removed)}")
        tin = nxt["args"][0]
        if inline_sig:
            _req(ts == tin, "temb sigmoid input", c["idx"], ts, tin)
        else:
            _req(_elems(cg, ts) == K, f"temb scale tensors[{ts}] has {_elems(cg, ts)} elements, K={K}")
        sh_in = cg.shape[cg.canon(tin)]
        _req(sh_in[-1] == K and _elems(cg, tin) == K and K <= 4096, "temb dense input", tin, sh_in)
        # weights.bin: [N][K] -> [N/8][K][8] (bf16, a pure permutation: bit-exact)
        a = np.array(w[off // 2: off // 2 + N * K]).reshape(N // 8, 8, K).transpose(0, 2, 1)
        w[off // 2: off // 2 + N * K] = np.ascontiguousarray(a).reshape(-1)
        s0, s1 = c["seg"] + em[0].start(), c["seg"] + em[0].end()
        a0, a1 = c["span"]
        rep = ("    GRAPH_PROF_BEGIN(%d);\n"
               "    graph_bf16_expand((float *)tensors[%d].data, (const uint16_t *)(g_params + %d), %dULL);"
               "   /* keep-layout: pre-packed [N/8][K][8] */\n"
               "    kl_temb_x(&tensors[%d], %s, %d);\n"
               "    GRAPH_PROF_END(%d, \"cpu kl_temb_x_wexpand\");\n"
               % (c["idx"], to, off, N * K, tin, "0" if inline_sig else "&tensors[%d]" % ts, K, c["idx"]))
        old = "    args[0].v_handle = &tensors[%d];\n" % tin
        j = text.index(old, nxt["seg"], nxt["span"][0])
        text = (text[:s0] + text[s1:a0] + rep + text[a1:j] + "    args[0].v_handle = &g_kl_x_t;\n"
                + text[j + len(old):])
        calls = _calls(text)
        pos = {cc["idx"]: i for i, cc in enumerate(calls)}
        n += 1
        log(f"KL temb call [{c['idx']}] {c['fn']} -> pre-packed W {N}x{K} @weights.bin+{off}, dense [{nxt['idx']}] "
            f"reads x=t*s (t=tensors[{tin}])")
    w.flush()
    del w
    if n:
        i = _decl_point(text)
        text = text[:i] + TEMB_C + "\n" + text[i:]
    return text, n


# ═══════════════════════════════════════════════════════════════════════════
#  attention: FA packs at the head split ops, O by stride
# ═══════════════════════════════════════════════════════════════════════════

def _qk_src(cg: CallGraph, args: list, H: int, D: int) -> list:
    """head split op: args = (srcs..., [scale], out); the source = the [1,R,C] / [R,C] input with C == H*D"""
    out = []
    for t in args[:-1]:
        s = list(cg.shape[cg.canon(t)])
        if len(s) in (2, 3) and s[-1] == H * D and (len(s) == 2 or s[0] == 1):
            out.append(t)
    return out


def fa_pass(text: str, cg: CallGraph, wbin: Path, csrc: str = None, removed: set = None, log=print) -> tuple:
    removed = set() if removed is None else removed
    sh = lambda t: list(cg.shape[cg.canon(t)])   # noqa: E731
    w = np.memmap(wbin, dtype=np.uint16, mode="r")
    ngrp = 0
    prev_pv = -1
    while True:
        calls = _calls(text)
        qk = [i for i, c in enumerate(calls) if c["site"] and c["role"] == 0 and "KL" not in text[c["span"][0]:c["span"][1]]]
        if not qk:
            break
        iq = qk[0]
        cq = calls[iq]
        ip = next(i for i in range(iq + 1, len(calls)) if calls[i]["site"] and calls[i]["role"] == 1)
        cp = calls[ip]
        tq, tk = cq["args"][0], cq["args"][1]
        tv, to = cp["args"][1], cp["args"][2]
        H, N, D = sh(tq)
        M = sh(tk)[1]
        _req(sh(tk) == [H, M, D] and sh(tv) == [H, D, M] and sh(to) == [H, N, D], "FA shapes",
             sh(tq), sh(tk), sh(tv), sh(to))
        # the head split / merge calls: producers of Q, K, V^T and the only reader of O (graph.json)
        jq_, jk_, jv_ = (cg.producer.get(cg.canon(t)) for t in (tq, tk, tv))
        _req(None not in (jq_, jk_, jv_), "FA: Q/K/V^T without producer", tq, tk, tv)
        ro = _readers(cg, to, removed)
        ro = [r for r in ro if r != cp["idx"]]
        _req(len(ro) == 1, f"FA O tensors[{to}]: readers {ro} (expected only the head merge)")
        jo_ = ro[0]
        _pure(cg, jq_, extra=("multiply",), need=("multiply",))
        _pure(cg, jk_, extra=("multiply",), need=("multiply",))
        _pure(cg, jv_)
        _pure(cg, jo_)
        for j, t in ((jq_, tq), (jk_, tk), (jv_, tv), (jo_, cg.calls[jo_].outs[0])):
            _req(_elems(cg, t) == H * max(N, M) * D or _elems(cg, t) == H * M * D or _elems(cg, t) == H * N * D,
                 f"FA call [{j}]: element count {_elems(cg, t)}")
        scales = [t for t in cg.calls[jq_].ins if t in cg.const]
        _req(len(scales) == 1 and _elems(cg, scales[0]) == 1, f"FA Q op [{jq_}]: scale operand {scales}")
        # removed intermediates: no reader outside the group (graph.json)
        for t, who in ((tq, cq["idx"]), (tk, cq["idx"])):
            _req(_readers(cg, t, removed) == [who], f"FA: Q/K tensors[{t}] read outside the group: "
                                                    f"{_readers(cg, t, removed)}")
        _req(_readers(cg, tv, removed) == [cp["idx"]], f"FA: V^T tensors[{tv}] read outside the group: "
                                                       f"{_readers(cg, tv, removed)}")
        pos = {c["idx"]: i for i, c in enumerate(calls)}
        _req(all(x in pos for x in (jq_, jk_, jv_, jo_)), "FA: a head split/merge call is not in the driver",
             jq_, jk_, jv_, jo_)
        jq, jk, jv, jo = pos[jq_], pos[jk_], pos[jv_], pos[jo_]
        if csrc is not None:   # cross-check on the TVM C
            for j, kind in ((jq, "qk"), (jk, "qk"), (jv, "v"), (jo, "o")):
                b = _tvm_body(csrc, calls[j]["fn"])
                if kind == "qk":
                    ok = re.search(r"\* %d\) \+ \(\(ax0_ax1_fused (?:>> \d+|/ %d)\) \* %d\)\)"
                                   % (H * D, sh(tq if j == jq else tk)[1], D), b) and "p1_1)[0]" in b
                elif kind == "v":
                    ok = re.search(r"\(ax2_inner \* %d\)\) \+ ax0_ax1_fused\)\]" % (H * D), b)
                else:
                    ok = re.search(r"\* %d\) \+ \(ax0 \* %d\)\)" % (N * D, D), b) and \
                        re.search(r"T_reshape_1\)\[\(+ax0 \* %d\)" % (H * D), b)
                _req(ok, f"FA {kind} body", calls[j]["fn"])
        sq = _qk_src(cg, calls[jq]["args"], H, D)
        sk = _qk_src(cg, calls[jk]["args"], H, D)
        sv = _qk_src(cg, calls[jv]["args"], H, D)
        _req(sq and sk and sv, "FA sources", sq, sk, sv)

        def pick(j, cands):
            # self-attn: the Q/K/V op lists all 3 split outputs; which one it reads is in the TVM C only
            if len(cands) == 1:
                return cands[0]
            _req(csrc is not None, f"FA call [{calls[j]['idx']}]: {len(cands)} candidate sources, needs the TVM C")
            b = _tvm_body(csrc, calls[j]["fn"])
            argi = {m.group(1): int(m.group(2)) for m in
                    re.finditer(r"void\* (\w+) = \(\(\(TVMValue\*\)args\)\[(\d+)\]\.v_handle\);", b)}
            datav = {m.group(1): argi[m.group(2)] for m in
                     re.finditer(r"void\* (\w+) = \(\(\(DLTensor\*\)(\w+)\)\[0\]\.data\);", b)}
            used = set(re.findall(r"\(\(float\*\)(\w+)\)\[\(", b))
            src = [calls[j]["args"][datav[u]] for u in used if u in datav and calls[j]["args"][datav[u]] in cands]
            _req(len(set(src)) == 1, "pick src", calls[j]["fn"], src)
            return src[0]
        qs, ks, vs = pick(jq, sq), pick(jk, sk), pick(jv, sv)
        _req(sh(qs)[-2:] == [N, H * D] and sh(ks)[-2:] == [M, H * D] and sh(vs)[-2:] == [M, H * D],
             "FA source shapes", sh(qs), sh(ks), sh(vs))

        def scale(j):
            seg = text[calls[j]["seg"]:calls[j]["span"][0]]
            sc = [e for e in EXP_RE.finditer(seg) if int(e.group(3)) == 1]
            _req(len(sc) == 1, "scale", calls[j]["idx"])
            v = int(w[int(sc[0].group(2)) // 2])
            return float(np.array([v << 16], dtype=np.uint32).view(np.float32)[0])
        s_q, s_k = scale(jq), scale(jk)
        _req(s_q == s_k, "FA scales", s_q, s_k)
        to2 = calls[jo]["args"][-1]
        _req(sh(to2) in ([1, N, H * D], [N, H * D]), "FA O out", sh(to2))
        grp = {jq_, jk_, jv_, jo_}
        # early write: PV writes O into the merge's output -- untouched between PV and the removed merge
        _untouched_g(cg, cp["idx"], jo_, to2, removed | grp, "FA O early write")
        _req(min(jq, jk, jv) > prev_pv and max(jq, jk) < iq and jv < ip, "FA order", prev_pv, jq, jk, jv, iq, ip)
        # the sources are still in their buffers at the pack slots (the removed ops read them there)
        for j_, t in ((jq_, qs), (jk_, ks), (jv_, vs)):
            _req(cg.canon(t) in cg.calls[j_].ins, f"FA pack source tensors[{t}] not read by call [{j_}]")
        cid_q = re.search(r"unet_aie_attn\((\d+), args, 3, 0, (\d+)\);", text[cq["span"][0]:cq["span"][1]])
        cid_p = re.search(r"unet_aie_attn\((\d+), args, 3, 1, (\d+)\);", text[cp["span"][0]:cp["span"][1]])
        _req(cid_q and cid_p and cid_q.group(2) == cid_p.group(2), "FA site cls", cid_q, cid_p)
        cls = int(cid_q.group(2))

        def site(j, cid, role, src, s):
            i = calls[j]["idx"]
            return ("    GRAPH_PROF_BEGIN(%d);\n    UP_AIE_BEGIN(%d);\n"
                    "    unet_aie_attn_kl(%s, %d, %d, (const float *)tensors[%d].data, %d, 0, 0, %.9ef);   /* KL %s */\n"
                    "    UP_AIE_END(%d);\n    GRAPH_PROF_END(%d, \"aie unet_aie_attn %s\");\n"
                    % (i, i, cid, role, cls, src, H * D, s, calls[j]["fn"], i, i, cid))
        edits = [(calls[jq]["span"], site(jq, cid_q.group(1), 2, qs, s_q)),
                 (calls[jk]["span"], site(jk, cid_q.group(1), 3, ks, s_k)),
                 (calls[jv]["span"], site(jv, cid_p.group(1), 5, vs, 1.0)),
                 (cq["span"], None), (calls[jo]["span"], None)]
        a0, a1 = cp["span"]
        edits.append(((a0, a1), text[a0:a1].replace(
            cid_p.group(0), "unet_aie_attn_kl(%s, 6, %d, 0, 0, (float *)tensors[%d].data, %d, 1.0f);   /* KL */"
            % (cid_p.group(1), cls, to2, H * D))))
        span2call = {c["span"]: c for c in calls}
        for (b0, b1), rep in sorted(edits, key=lambda e: -e[0][0]):
            if rep is None:
                text = _remove(text, span2call[(b0, b1)], "FA keep-layout (packs at the head split ops, O by stride)")
            else:
                text = text[:b0] + rep + text[b1:]
        removed |= {cq["idx"], jo_}
        prev_pv = ip - 1
        ngrp += 1
        log(f"KL attn group: qk [{cq['idx']}] pv [{cp['idx']}] H={H} N={N} M={M} D={D} scale={s_q:.6g}: removed "
            f"[{calls[jq]['idx']}] {calls[jq]['fn']}, [{calls[jk]['idx']}] {calls[jk]['fn']}, "
            f"[{calls[jv]['idx']}] {calls[jv]['fn']}, [{calls[jo]['idx']}] {calls[jo]['fn']}; "
            f"q=t{qs} k=t{ks} v=t{vs} o=t{to2}")
    del w
    if ngrp:
        text = text.replace("int unet_aie_attn(int cid, void *args, int nargs, int role, int cls);",
                            "int unet_aie_attn(int cid, void *args, int nargs, int role, int cls);\n"
                            "int unet_aie_attn_kl(int cid, int role, int cls, const float *a, int lda, float *b, int ldb,\n"
                            "                     float s);   /* keep-layout (fa_attn.py) */", 1)
    return text, ngrp


# ═══════════════════════════════════════════════════════════════════════════
#  transformer out: residual into the linear scatter, conv reads NHWC
# ═══════════════════════════════════════════════════════════════════════════

PO_FN = re.compile(r"^reshape_add_reshape_transpose_layout_transform(_\d+)?$")
PO_BODY = re.compile(r"T_layout_trans_1\)\[.*\] = \(\(\(float\*\)p0_1\)\[cse_var_1\] \+ \(\(float\*\)p1_1\)\[cse_var_1\]\);")


def projout_pass(text: str, cg: CallGraph, csrc: str = None, removed: set = None, log=print) -> tuple:
    removed = set() if removed is None else removed
    sh = lambda t: list(cg.shape[cg.canon(t)])   # noqa: E731
    n = 0
    decl = []
    while True:
        calls = _calls(text)
        js = [i for i, c in enumerate(calls) if c["fn"] and PO_FN.match(c["fn"])]
        if not js:
            break
        j = js[0]
        c = calls[j]
        ops = _pure(cg, c["idx"], extra=("add",), need=("add", "layout_transform"))
        _req(ops.count("add") == 1, "proj_out ops", c["fn"], ops)
        if csrc is not None:
            _req(PO_BODY.search(_tvm_body(csrc, c["fn"])), "proj_out body", c["fn"])
        _req(len(c["args"]) == 3, "proj_out args", c["idx"], c["args"])
        tl, tr_, to = c["args"]
        lin, cv = calls[j - 1], calls[j + 1]
        _req(lin["site"] and lin["site"].startswith("unet_aie_linear(") and lin["args"][-1] == tl,
             "proj_out: not after its AIE linear", lin["idx"])
        _req(cv["site"] and cv["site"].startswith("unet_aie_conv(") and cv["args"][0] == to,
             "proj_out: not before its AIE conv", cv["idx"])
        _req(_readers(cg, tl, removed) == [c["idx"]], f"proj_out: linear out tensors[{tl}] read outside the group: "
                                                      f"{_readers(cg, tl, removed)}")
        _req(_readers(cg, to, removed) == [cv["idx"]], f"proj_out: out tensors[{to}] read outside the group: "
                                                       f"{_readers(cg, to, removed)}")
        Mx, C = sh(tl)[-2], sh(tl)[-1]
        _req(sh(tr_)[-2:] == [Mx, C] and _elems(cg, tr_) == Mx * C, "proj_out residual", sh(tr_), sh(tl))
        o = sh(to)
        _req(len(o) == 5 and o[0] == 1 and o[1] * o[4] == C and o[2] * o[3] == Mx, "proj_out out", o)
        H_, W_ = o[2], o[3]
        na = len(lin["args"])
        seg0, seg1 = lin["seg"], lin["span"][1]
        blk = text[seg0:seg1]
        old_out = "    args[%d].v_handle = &tensors[%d];\n" % (na - 1, tl)
        _req(blk.count(old_out) == 1, "lin out arg", lin["idx"])
        blk = blk.replace(old_out, "    args[%d].v_handle = &tensors[%d];   /* keep-layout: residual into the scatter */\n"
                          "    args[%d].v_handle = &tensors[%d];\n" % (na - 1, tr_, na, tl))
        m = re.search(r"unet_aie_linear\((\d+), args, %d, " % na, blk)
        _req(m, "lin call", lin["idx"])
        blk = blk[:m.start()] + "unet_aie_linear(%s, args, %d, " % (m.group(1), na + 1) + blk[m.end():]
        cblk = text[cv["seg"]:cv["span"][1]]
        old_x = "    args[0].v_handle = &tensors[%d];\n" % to
        _req(cblk.count(old_x) == 1, "conv x arg", cv["idx"])
        nm = "g_kl_nhwc_%d" % cv["idx"]
        decl.append("static DLTensor %s;\nstatic int64_t %s_shape[5] = {1, 1, %d, %d, %d};\n" % (nm, nm, H_, W_, C))
        cblk = cblk.replace(old_x, "    %s = tensors[%d]; %s.ndim = 5; %s.shape = %s_shape;   /* keep-layout: [M][C] as NHWC */\n"
                            "    args[0].v_handle = &%s;\n" % (nm, tl, nm, nm, nm, nm))
        rblk = "    /* keep-layout: call [%d] %s removed -- residual add in the linear scatter, conv reads NHWC */\n" % (
            c["idx"], c["fn"])
        text = text[:seg0] + blk + text[seg1:c["span"][0]] + rblk + text[c["span"][1]:cv["seg"]] + cblk + \
            text[cv["span"][1]:]
        removed.add(c["idx"])
        n += 1
        log(f"KL proj_out [{c['idx']}] {c['fn']}: linear [{lin['idx']}] += residual t{tr_}, conv [{cv['idx']}] reads "
            f"t{tl} as NHWC 1x1x{H_}x{W_}x{C}")
    if n:
        i = _decl_point(text)
        text = text[:i] + "/* keep-layout: NHWC labels of linear outputs read by AIE convs */\n" + "".join(decl) + \
            "\n" + text[i:]
    return text, n


# ═══════════════════════════════════════════════════════════════════════════
#  conv out: the AIE conv scatter writes the consumer's layout
# ═══════════════════════════════════════════════════════════════════════════

CO_FN = re.compile(r"^layout_transform(_transpose)?_reshape(_\d+)?$")


def convout_pass(text: str, cg: CallGraph, csrc: str = None, removed: set = None, log=print) -> tuple:
    removed = set() if removed is None else removed
    sh = lambda t: list(cg.shape[cg.canon(t)])   # noqa: E731
    n = 0
    decl = []
    skipped = set()
    done = []
    while True:
        calls = _calls(text)
        js = [i for i, c in enumerate(calls) if c["fn"] and CO_FN.match(c["fn"]) and i > 0 and calls[i - 1]["site"]
              and calls[i - 1]["site"].startswith("unet_aie_conv(") and calls[i - 1]["args"][-1] == c["args"][0]
              and _readers(cg, c["args"][0], removed) == [c["idx"]] and c["idx"] not in skipped]
        if not js:
            break
        j = js[0]
        c, cv = calls[j], calls[j - 1]
        tin, tout = c["args"][0], c["args"][-1]
        _req(len(c["args"]) == 2, "conv out args", c["idx"], c["args"])
        ops = _pure(cg, c["idx"], need=("layout_transform",))
        sid_out = cg.storage(tout)
        if any(t >= 0 and cg.canon(t) not in cg.const and cg.storage(t) == sid_out for t in cv["args"][1:-1]):
            # the conv's epilogue operand lives in the buffer it would now write (TVM reused it for the layout op
            # output): writing early would clobber it -> leave this layout op in place
            skipped.add(c["idx"])
            log(f"KL conv out [{cv['idx']}] -> [{c['idx']}] {c['fn']}: SKIP (epilogue operand shares buffer "
                f"buf_{sid_out})")
            continue
        xi = sh(tin)
        _req(len(xi) == 5 and xi[0] == 1, "conv out NCHWc", xi)
        O, OH, OW = xi[1] * xi[4], xi[2], xi[3]
        _req(_elems(cg, tout) == O * OH * OW, "conv out elements", sh(tout), xi)
        nhwc = "transpose" in ops
        body = _tvm_body(csrc, c["fn"]) if csrc is not None else None
        if nhwc:
            _req(sh(tout)[-2:] == [OH * OW, O], "nhwc out", c["fn"], sh(tout))
            if body is not None:
                _req(re.search(r"T_reshape_1\)\[\(+ax0_ax1_fused \* %d\)" % O, body), "nhwc body", c["fn"])
            for t in cv["args"][2:-1]:
                _req(_elems(cg, t) == O, "conv residual with NHWC out", cv["idx"], t)
            lab = (1, 1, OH, OW, O)
        else:
            _req(len(sh(tout)) == 3 and sh(tout)[0] == 1, "planar out", c["fn"], sh(tout))
            if body is not None:
                _req(re.search(r"T_reshape_1\)\[\(+(?:\(ax0_ax1_fused \* %d\)|cse_var_1)" % (sh(tout)[2]), body),
                     "planar body", c["fn"], sh(tout))
            lab = (1, O, OH, OW, 1)
        cblk = text[cv["seg"]:cv["span"][1]]
        na = len(cv["args"])
        old_y = "    args[%d].v_handle = &tensors[%d];\n" % (na - 1, tin)
        _req(cblk.count(old_y) == 1, "conv y arg", cv["idx"])
        nm = "g_kl_y_%d" % cv["idx"]
        decl.append("static DLTensor %s;\nstatic int64_t %s_shape[5] = {%d, %d, %d, %d, %d};\n" % ((nm, nm) + lab))
        cblk = cblk.replace(old_y, "    %s = tensors[%d]; %s.data = tensors[%d].data; %s.shape = %s_shape;   /* keep-layout: %s */\n"
                            "    args[%d].v_handle = &%s;\n"
                            % (nm, tin, nm, tout, nm, nm, "NHWC out" if nhwc else "planar NCHW out", na - 1, nm))
        rblk = "    /* keep-layout: call [%d] %s removed -- the AIE conv [%d] writes this layout */\n" % (
            c["idx"], c["fn"], cv["idx"])
        text = text[:cv["seg"]] + cblk + text[cv["span"][1]:c["span"][0]] + rblk + text[c["span"][1]:]
        removed.add(c["idx"])
        done.append((cv["idx"], c["idx"]))
        n += 1
        log(f"KL conv out [{cv['idx']}] -> [{c['idx']}] {c['fn']}: {'NHWC' if nhwc else 'planar'} label {lab} into t{tout}")
    if n:
        i = _decl_point(text)
        text = text[:i] + "/* keep-layout: AIE conv output labels */\n" + "".join(decl) + "\n" + text[i:]
    return text, n, done


# ═══════════════════════════════════════════════════════════════════════════
#  entry points
# ═══════════════════════════════════════════════════════════════════════════

def _plan_calls(cg: CallGraph, plan) -> list:
    """Producer calls of the planned keep_layout edges (plan_keep_layout edges tagged with the graph.json node
    id under ``node_id`` / ``call``, as the mapped partition layers carry it)."""
    out = []
    for e in (plan or {}).get("edges", []):
        if "call" in e:
            out.append(int(e["call"]))
        elif "node_id" in e:
            hit = [c.call for c in cg.calls if c.node == int(e["node_id"])]
            _req(hit, f"keep_layout plan edge {e.get('node')}: graph.json node {e['node_id']} is not a call")
            out.append(hit[0])
        else:
            raise AiegraphError(_STAGE, f"keep_layout plan edge {e.get('node')}: no graph.json node id",
                                hint="tag the plan_keep_layout edges with the mapped layer's node (node_id)")
    return out


def group_removed(graph_json) -> set:
    """Calls the recorded aiegraph groups (``layers/partition.json`` beside graph.json) removed from the driver
    (S3 FA groups: QK bmm, softmax, O merge) -- they are gone before the keep-layout passes run."""
    part = Path(graph_json).parent / "layers" / "partition.json"
    if not part.is_file():
        return set()
    import json
    doc = json.loads(part.read_text())
    return {int(m["call"]) for g in doc.get("groups", []) for m in g.get("members", [])
            if m.get("action") == "removed"}


def keep_layout_driver(text: str, cg: CallGraph, wbin: Path, csrc: str = None, plan=None, log=print,
                       pre_removed=()) -> tuple:
    """All four rewrites on the driver text; ``plan`` = ``partition.plan_keep_layout`` result (edges with
    ``node_id`` / ``call``): every planned edge must be realized (its conv's layout call removed).
    ``pre_removed``: calls already removed by aiegraph group offloads (``group_removed``)."""
    _check_calls(cg, _calls(text))
    removed = set(pre_removed)
    text, nt = temb_pass(text, cg, wbin, csrc, removed, log)
    text, na = fa_pass(text, cg, wbin, csrc, removed, log)
    text, npo = projout_pass(text, cg, csrc, removed, log)
    text, nco, done = convout_pass(text, cg, csrc, removed, log)
    want = _plan_calls(cg, plan)
    got = {p for p, _ in done}
    for r in removed:   # a removed layout call realizes the edge of the call producing its data input
        ins = cg.calls[r].ins
        if ins and ins[0] in cg.producer:
            got.add(cg.producer[ins[0]])
    miss = [k for k in want if k not in got]
    _req(not miss, f"planned keep_layout edges not realized by the codegen: producer calls {miss}")
    log(f"KEEP_LAYOUT_RT temb_sites={nt} attn_groups={na} proj_out={npo} conv_out={nco}")
    log(f"KEEP_LAYOUT_CODEGEN planned_edges={len(want)} realized={len(want) - len(miss)} removed_calls={len(removed)}")
    return text, dict(temb=nt, attn=na, proj_out=npo, conv_out=nco, planned=len(want), removed=sorted(removed))


class KeepLayoutPass:
    """``arm_build`` driver pass: ``fn(driver_c, graph_json, weights_bin, c_source)``."""

    name = "keep_layout"

    def __init__(self, plan=None, log=print):
        self.plan, self.log, self.last = plan, log, None

    def __call__(self, driver_c, graph_json, weights_bin, c_source=None):
        driver_c = Path(driver_c)
        cg = CallGraph.load(graph_json)
        csrc = Path(c_source).read_text(errors="replace") if c_source and Path(c_source).is_file() else None
        text, rep = keep_layout_driver(driver_c.read_text(), cg, Path(weights_bin), csrc, self.plan, self.log,
                                       pre_removed=group_removed(graph_json))
        driver_c.write_text(text)
        self.last = rep
        return rep


def register(workdir, plan=None, log=print) -> KeepLayoutPass:
    """Run the keep-layout codegen on every driver ``arm_build`` generates for ``workdir``."""
    from . import _env
    p = KeepLayoutPass(plan, log)
    _env.frontend("arm_build").register_driver_pass(workdir, p.name, p)
    return p


def apply(driver_c, graph_json, weights_bin, c_source=None, plan=None, log=print) -> dict:
    """One-shot: rewrite ``driver_c`` (and the temb weights in ``weights_bin``) in place."""
    return KeepLayoutPass(plan, log)(driver_c, graph_json, weights_bin, c_source)
