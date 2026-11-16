"""P4a (sprint unet_m3-s1): keep the AIE layout between layers -- graph_driver.c pass (build_unet.py --keep-layout).

The A78 layout ops of the P2 UNet build (PROF class layout, 61 ms) are TVM fused functions that only permute data
between producers and consumers. This pass removes them from graph_run where the permutation can be folded into
work that already exists, so no A78 transpose pass is left:

* temb (``[sigmoid_squeeze_]multiply_layout_transform*`` -> ``nn_contrib_dense_pack_add*``): TVM folded the SiLU
  multiplier of the time embedding into the dense WEIGHT and re-packs W*s into dense_pack's [N/8][K][8] every call
  (3.2 MB per resnet). Here the weight is pre-packed into weights.bin at build time ([N][K] -> [N/8][K][8], bf16,
  exact), the call only widens it (graph_bf16_expand, as every host dense does), and the dense reads x = t * s
  (1280 values): dense(t, W*s) == dense(t*s, W).
* attention (``reshape_transpose_multiply_reshape*`` Q / ``..._transpose*`` K / ``reshape_transpose_reshape_transpose*``
  V^T / ``reshape_transpose_reshape*`` O around every FA group): the FA host pack/unpack (fa_attn.py, which already
  re-tiles every element) reads Q/K/V straight from the [rows][H*D] linear outputs (head = column block) with the
  D^-1/4 scale applied in the pack, and writes O straight into the [N][H*D] buffer the next linear reads.

Every rewrite is checked: TVM body pattern of the removed function, shapes, and buffer liveness (the source buffer
of a moved read / the target buffer of a moved write is not touched by any call in between); anything else ->
AssertionError (build fails, nothing silently skipped).
"""

import re
from pathlib import Path

import numpy as np

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
    lbl = {"g_kl_x_t": -1}   # P4a labels -> the tensor whose buffer they point at
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


def _shapes(text: str) -> dict:
    sh = {}
    for m in SHAPE_RE.finditer(text):
        sh.setdefault(int(m.group(1)), {})[int(m.group(2))] = int(m.group(3))
    return {t: [d[i] for i in sorted(d)] for t, d in sh.items()}


def _bufs(text: str) -> dict:
    """tensor id -> storage buffer name (activations only; params live in g_wscratch/g_params)"""
    b = {}
    for m in BUF_RE.finditer(text[:text.index("float *graph_run(void)")]):
        if m.group(2).startswith("buf_"):
            b[int(m.group(1))] = m.group(2)
    return b


def _tvm_body(csrc: str, fn: str) -> str:
    h = ("TVM_DLL int32_t tvmgen_default_fused_%s(void* args, int32_t* arg_type_ids, int32_t num_args, "
         "void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {\n" % fn)
    i = csrc.index(h)
    return csrc[i:csrc.index("\n}\n", i)]


def _untouched(calls: list, lo: int, hi: int, buf: str, bufs: dict, skip: set) -> None:
    """no call strictly between list positions lo and hi (except removed ones) has an argument in buffer buf"""
    for c in calls[lo + 1:hi]:
        if c["idx"] in skip:
            continue
        hit = [t for t in c["args"] if bufs.get(t) == buf]
        assert not hit, ("keep-layout liveness", c["idx"], c["tag"], buf, hit)


def _remove(text: str, c: dict, why: str) -> str:
    a, b = c["span"]
    return text[:a] + "    /* P4a keep-layout: call [%d] %s removed -- %s */\n" % (c["idx"], c["tag"][4:], why) + text[b:]


# ------------------------------------------------------------------------------------------------ temb
TEMB_FN = re.compile(r"^(sigmoid_squeeze_)?multiply_layout_transform(_\d+)?$")
TEMB_BODY = re.compile(r"ax0_ax1_fused < (\d+); .*?ax2_inner < 8; .*?\(ax0_ax1_fused % (\d+)\).*?"
                       r"\(\(\(ax0_ax1_fused / (\d+)\) \* (\d+)\) \+ \(ax2_inner \* (\d+)\)\) \+ cse_var_1\)\]", re.S)

TEMB_C = r"""
/* P4a keep-layout (keep_layout_rt.py): temb dense input x = t * sigmoid(t) (the weight is pre-packed in
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


def temb_pass(text: str, csrc: str, wbin: Path, log=print) -> tuple:
    calls = _calls(text)
    pos = {c["idx"]: i for i, c in enumerate(calls)}
    sh = _shapes(text)
    w = np.memmap(wbin, dtype=np.uint16, mode="r+")
    n = 0
    saved = []
    while True:
        todo = [cc for cc in calls if cc["fn"] and TEMB_FN.match(cc["fn"])]
        if not todo:
            break
        c = todo[0]
        inline_sig = c["fn"].startswith("sigmoid_squeeze_")
        body = _tvm_body(csrc, c["fn"])
        m = TEMB_BODY.search(body)
        assert m, ("temb body", c["fn"])
        nk8, K, K_, K8, K__ = (int(x) for x in m.groups())
        assert K == K_ == K__ and K8 == 8 * K, ("temb body", c["fn"], m.groups())
        N = nk8 * 8 // K
        # inline sigmoid: p0 = t (sigmoid input), p1 = W; else p0 = W, p1 = s
        wi = 1 if inline_sig else 0
        assert len(c["args"]) == 3, c
        tw, ts, to = c["args"]
        if inline_sig:
            tw, ts = ts, tw
        seg = text[c["seg"]:c["span"][0]]
        em = [e for e in EXP_RE.finditer(seg) if int(e.group(4)) == tw]
        assert len(em) == 1, ("temb weight expand", c["idx"], tw)
        off, cnt = int(em[0].group(2)), int(em[0].group(3))
        assert cnt == N * K, (c["idx"], cnt, N, K)
        assert text.count("(g_params + %d)" % off) == 1, ("temb weight shared", off)
        nxt = calls[pos[c["idx"]] + 1]
        assert nxt["fn"] and nxt["fn"].startswith("nn_contrib_dense_pack_add") and nxt["args"][1] == to, nxt
        tin = nxt["args"][0]
        if inline_sig:
            assert ts == tin, (c["idx"], ts, tin)
        assert sh[tin][-1] == K and int(np.prod(sh[tin])) == K and K <= 4096, (tin, sh.get(tin))
        # weights.bin: [N][K] -> [N/8][K][8] (bf16, a pure permutation: bit-exact)
        a = np.array(w[off // 2: off // 2 + N * K]).reshape(N // 8, 8, K).transpose(0, 2, 1)
        w[off // 2: off // 2 + N * K] = np.ascontiguousarray(a).reshape(-1)
        # rewrite: widen the pre-packed weight straight into the dense's weight tensor; x = t*s for the dense
        s0, s1 = c["seg"] + em[0].start(), c["seg"] + em[0].end()
        a0, a1 = c["span"]
        rep = ("    GRAPH_PROF_BEGIN(%d);\n"
               "    graph_bf16_expand((float *)tensors[%d].data, (const uint16_t *)(g_params + %d), %dULL);"
               "   /* P4a: pre-packed [N/8][K][8] */\n"
               "    kl_temb_x(&tensors[%d], %s, %d);\n"
               "    GRAPH_PROF_END(%d, \"cpu kl_temb_x_wexpand\");\n"
               % (c["idx"], to, off, N * K, tin, "0" if inline_sig else "&tensors[%d]" % ts, K, c["idx"]))
        # the dense reads x: its args[0] line
        nseg0 = nxt["seg"]
        old = "    args[0].v_handle = &tensors[%d];\n" % tin
        j = text.index(old, nseg0, nxt["span"][0])
        new_text = (text[:s0] + text[s1:a0] + rep + text[a1:j] + "    args[0].v_handle = &g_kl_x_t;\n"
                    + text[j + len(old):])
        text = new_text
        calls = _calls(text)
        pos = {cc["idx"]: i for i, cc in enumerate(calls)}
        n += 1
        saved.append((c["idx"], c["fn"], N, K, off))
        log(f"KL temb call [{c['idx']}] {c['fn']} -> pre-packed W {N}x{K} @weights.bin+{off}, dense [{nxt['idx']}] "
            f"reads x=t*s (t=tensors[{tin}])")
    w.flush()
    del w
    if n:
        i = text.index("float *graph_run(void)")
        i = text.rindex("\n", 0, i) + 1
        text = text[:i] + TEMB_C + "\n" + text[i:]
    return text, n


# ------------------------------------------------------------------------------------------------ attention
Q_FN = re.compile(r"^reshape_transpose_multiply_reshape(_\d+)?$")
K_FN = re.compile(r"^reshape_transpose_multiply_reshape_transpose(_\d+)?$")
V_FN = re.compile(r"^reshape_transpose_reshape_transpose(_\d+)?$")
O_FN = re.compile(r"^reshape_transpose_reshape(_\d+)?$")


def _producer(calls: list, upto: int, t: int):
    """last call before list position upto whose OUTPUT (last arg) is tensor t"""
    for i in range(upto - 1, -1, -1):
        if calls[i]["args"] and calls[i]["args"][-1] == t:
            return i
    raise AssertionError(("no producer", t))


def _consumers(calls: list, frm: int, t: int) -> list:
    return [i for i in range(frm + 1, len(calls)) if t in calls[i]["args"][:-1]]


def _qk_src(c: dict, sh: dict, H: int, D: int):
    """Q/K transpose op: args = (srcs..., [scale], out); the source = the [1,R,C] input with C == H*D"""
    srcs = [t for t in c["args"][:-1] if len(sh.get(t, [])) in (2, 3) and sh[t][-1] == H * D
            and (len(sh[t]) == 2 or sh[t][0] == 1)]
    return srcs


def fa_pass(text: str, csrc: str, wbin: Path, log=print) -> tuple:
    sh = _shapes(text)
    bufs = _bufs(text)
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
        H, N, D = sh[tq]
        M = sh[tk][1]
        assert sh[tk] == [H, M, D] and sh[tv] == [H, D, M] and sh[to] == [H, N, D], (sh[tq], sh[tk], sh[tv], sh[to])
        jq, jk, jv = _producer(calls, iq, tq), _producer(calls, iq, tk), _producer(calls, ip, tv)
        assert Q_FN.match(calls[jq]["fn"] or "") and K_FN.match(calls[jk]["fn"] or "") and \
            V_FN.match(calls[jv]["fn"] or ""), (calls[jq]["fn"], calls[jk]["fn"], calls[jv]["fn"])
        co = _consumers(calls, ip, to)
        assert len(co) == 1 and O_FN.match(calls[co[0]]["fn"] or ""), ("O consumer", [calls[i]["fn"] for i in co])
        jo = co[0]
        for t in (tq, tk):
            assert _consumers(calls, iq - 1, t) == [iq], ("Q/K used elsewhere", t)
        assert _consumers(calls, ip - 1, tv) == [ip], ("V^T used elsewhere", tv)
        # the body of every removed op must be the plain head split / merge (checked on the TVM C)
        for j, kind in ((jq, "qk"), (jk, "qk"), (jv, "v"), (jo, "o")):
            b = _tvm_body(csrc, calls[j]["fn"])
            if kind == "qk":
                assert re.search(r"\* %d\) \+ \(\(ax0_ax1_fused (?:>> \d+|/ %d)\) \* %d\)\)" % (H * D, sh[tq if j == jq else tk][1], D), b) and \
                    "p1_1)[0]" in b, ("Q/K body", calls[j]["fn"])
            elif kind == "v":
                assert re.search(r"\(ax2_inner \* %d\)\) \+ ax0_ax1_fused\)\]" % (H * D), b), ("V body", calls[j]["fn"])
            else:
                assert re.search(r"\* %d\) \+ \(ax0 \* %d\)\)" % (N * D, D), b) and \
                    re.search(r"T_reshape_1\)\[\(+ax0 \* %d\)" % (H * D), b), \
                    ("O body", calls[j]["fn"])
        sq = _qk_src(calls[jq], sh, H, D)
        sk = _qk_src(calls[jk], sh, H, D)
        sv = _qk_src(calls[jv], sh, H, D)
        assert len(sk) >= 1 and len(sq) >= 1 and len(sv) >= 1, (sq, sk, sv)
        # split outputs (self-attn): Q/K/V op all list the 3 split outputs: pick by the body offset (+0/+C/+2C)
        def pick(j, cands):
            if len(cands) == 1:
                return cands[0]
            b = _tvm_body(csrc, calls[j]["fn"])
            argi = {m.group(1): int(m.group(2)) for m in
                    re.finditer(r"void\* (\w+) = \(\(\(TVMValue\*\)args\)\[(\d+)\]\.v_handle\);", b)}
            datav = {m.group(1): argi[m.group(2)] for m in
                     re.finditer(r"void\* (\w+) = \(\(\(DLTensor\*\)(\w+)\)\[0\]\.data\);", b)}
            used = set(re.findall(r"\(\(float\*\)(\w+)\)\[\(", b))   # indexed loads (scale is [0], no paren)
            src = [calls[j]["args"][datav[u]] for u in used if u in datav and calls[j]["args"][datav[u]] in cands]
            assert len(set(src)) == 1, ("pick src", calls[j]["fn"], src)
            return src[0]
        qs, ks, vs = pick(jq, sq), pick(jk, sk), pick(jv, sv)
        assert sh[qs][-2:] == [N, H * D] and sh[ks][-2:] == [M, H * D] and sh[vs][-2:] == [M, H * D], (sh[qs], sh[ks], sh[vs])
        # scale param (bf16 in weights.bin) of the Q and K ops
        def scale(j):
            seg = text[calls[j]["seg"]:calls[j]["span"][0]]
            sc = [e for e in EXP_RE.finditer(seg) if int(e.group(3)) == 1]
            assert len(sc) == 1, ("scale", calls[j]["idx"])
            v = int(w[int(sc[0].group(2)) // 2])
            return float(np.array([v << 16], dtype=np.uint32).view(np.float32)[0])
        s_q, s_k = scale(jq), scale(jk)
        assert s_q == s_k, (s_q, s_k)
        # placement: the Q / K / V packs run at the positions of the removed Q / K / V ops (their sources are live
        # there by construction: those ops read them); the QK site goes; the PV site launches and writes O straight
        # into the O op's output, so that buffer must be untouched between the PV site and the removed O op
        to2 = calls[jo]["args"][-1]
        assert sh[to2] in ([1, N, H * D], [N, H * D]), sh[to2]
        skip = {calls[j]["idx"] for j in (jq, jk, jv, jo)}
        _untouched(calls, ip, jo, bufs[to2], bufs, skip)
        assert min(jq, jk, jv) > prev_pv and max(jq, jk) < iq and jv < ip, (prev_pv, jq, jk, jv, iq, ip)
        cid_q = re.search(r"unet_aie_attn\((\d+), args, 3, 0, (\d+)\);", text[cq["span"][0]:cq["span"][1]])
        cid_p = re.search(r"unet_aie_attn\((\d+), args, 3, 1, (\d+)\);", text[cp["span"][0]:cp["span"][1]])
        assert cid_q and cid_p and cid_q.group(2) == cid_p.group(2), (cid_q, cid_p)
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
        for (a0, a1), rep in sorted(edits, key=lambda e: -e[0][0]):
            if rep is None:
                text = _remove(text, span2call[(a0, a1)], "FA keep-layout (packs at the head split ops, O by stride)")
            else:
                text = text[:a0] + rep + text[a1:]
        prev_pv = ip - 1   # list position of the PV site after the QK site block is removed
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
                            "                     float s);   /* P4a keep-layout (fa_attn.py) */", 1)
    return text, ngrp


# ------------------------------------------------------------------------------------------------ transformer out
PO_FN = re.compile(r"^reshape_add_reshape_transpose_layout_transform(_\d+)?$")
PO_BODY = re.compile(r"T_layout_trans_1\)\[.*\] = \(\(\(float\*\)p0_1\)\[cse_var_1\] \+ \(\(float\*\)p1_1\)\[cse_var_1\]\);")


def projout_pass(text: str, csrc: str, log=print) -> tuple:
    """linear (ff out, [M][C]) -> reshape_add_reshape_transpose_layout_transform (+ residual [M][C], -> NCHWc) -> AIE
    1x1 conv (proj_out): the residual add goes into the linear site's scatter epilogue ([M][N] residual, argument
    order kept: (y + bias) + res as TVM), the conv reads the [M][C] buffer as NHWC (= NCHWc with one channel block
    of C: a DLTensor label, its im2col already re-tiles every element) -> the layout op is gone."""
    sh = _shapes(text)
    n = 0
    decl = []
    while True:
        calls = _calls(text)
        js = [i for i, c in enumerate(calls) if c["fn"] and PO_FN.match(c["fn"])]
        if not js:
            break
        j = js[0]
        c = calls[j]
        assert PO_BODY.search(_tvm_body(csrc, c["fn"])), ("proj_out body", c["fn"])
        tl, tr_, to = c["args"]
        lin, cv = calls[j - 1], calls[j + 1]
        assert lin["site"] and lin["site"].startswith("unet_aie_linear(") and lin["args"][-1] == tl, lin
        assert cv["site"] and cv["site"].startswith("unet_aie_conv(") and cv["args"][0] == to, cv
        assert _consumers(calls, j - 1, tl) == [j] and _consumers(calls, j, to) == [j + 1], (tl, to)
        Mx, C = sh[tl][-2], sh[tl][-1]
        assert sh[tr_][-2:] == [Mx, C] and int(np.prod(sh[tr_])) == Mx * C, (sh[tr_], sh[tl])
        o = sh[to]
        assert len(o) == 5 and o[0] == 1 and o[1] * o[4] == C and o[2] * o[3] == Mx, o
        # body: the out NCHWc element (cb, p, ci) = in[p][cb*CB + ci]  (check one index expression)
        H_, W_ = o[2], o[3]
        # linear site: one more epilogue arg (the residual) before the output
        na = len(lin["args"])
        seg0, seg1 = lin["seg"], lin["span"][1]
        blk = text[seg0:seg1]
        old_out = "    args[%d].v_handle = &tensors[%d];\n" % (na - 1, tl)
        assert blk.count(old_out) == 1, ("lin out arg", lin["idx"])
        blk = blk.replace(old_out, "    args[%d].v_handle = &tensors[%d];   /* P4a: residual into the scatter */\n"
                          "    args[%d].v_handle = &tensors[%d];\n" % (na - 1, tr_, na, tl))
        m = re.search(r"unet_aie_linear\((\d+), args, %d, " % na, blk)
        assert m, ("lin call", lin["idx"])
        blk = blk[:m.start()] + "unet_aie_linear(%s, args, %d, " % (m.group(1), na + 1) + blk[m.end():]
        # conv site: x = NHWC label of the linear output
        cblk = text[cv["seg"]:cv["span"][1]]
        old_x = "    args[0].v_handle = &tensors[%d];\n" % to
        assert cblk.count(old_x) == 1, ("conv x arg", cv["idx"])
        nm = "g_kl_nhwc_%d" % cv["idx"]
        decl.append("static DLTensor %s;\nstatic int64_t %s_shape[5] = {1, 1, %d, %d, %d};\n" % (nm, nm, H_, W_, C))
        cblk = cblk.replace(old_x, "    %s = tensors[%d]; %s.ndim = 5; %s.shape = %s_shape;   /* P4a: [M][C] as NHWC */\n"
                            "    args[0].v_handle = &%s;\n" % (nm, tl, nm, nm, nm, nm))
        rblk = "    /* P4a keep-layout: call [%d] %s removed -- residual add in the linear scatter, conv reads NHWC */\n" % (
            c["idx"], c["fn"])
        text = text[:seg0] + blk + text[seg1:c["span"][0]] + rblk + text[c["span"][1]:cv["seg"]] + cblk + \
            text[cv["span"][1]:]
        n += 1
        log(f"KL proj_out [{c['idx']}] {c['fn']}: linear [{lin['idx']}] += residual t{tr_}, conv [{cv['idx']}] reads "
            f"t{tl} as NHWC 1x1x{H_}x{W_}x{C}")
    if n:
        i = text.index("float *graph_run(void)")
        i = text.rindex("\n", 0, i) + 1
        text = text[:i] + "/* P4a keep-layout: NHWC labels of linear outputs read by AIE convs */\n" + "".join(decl) + \
            "\n" + text[i:]
    return text, n


# ------------------------------------------------------------------------------------------------ conv outputs
CO_FN = re.compile(r"^layout_transform(_transpose)?_reshape(_\d+)?$")


def convout_pass(text: str, csrc: str, log=print) -> tuple:
    """AIE conv -> layout_transform_reshape (NCHWc -> planar NCHW, GroupNorm input) or layout_transform_transpose_reshape
    (NCHWc -> [HW][C], transformer input): the conv's scatter (which already re-tiles every output element) writes
    the consumer's layout directly -- planar = NCHWc label with block 1 (unet_aie uc_scatter16 yb == 1 branch),
    [HW][C] = NHWC = NCHWc label with one block of C -- into the layout op's output buffer; the op is gone."""
    sh = _shapes(text)
    n = 0
    decl = []
    skipped = set()
    while True:
        calls = _calls(text)
        js = [i for i, c in enumerate(calls) if c["fn"] and CO_FN.match(c["fn"]) and i > 0 and calls[i - 1]["site"]
              and calls[i - 1]["site"].startswith("unet_aie_conv(") and calls[i - 1]["args"][-1] == c["args"][0]
              and _consumers(calls, i - 1, c["args"][0]) == [i] and c["idx"] not in skipped]
        if not js:
            break
        j = js[0]
        c, cv = calls[j], calls[j - 1]
        tin, tout = c["args"][0], c["args"][-1]
        assert len(c["args"]) == 2, c
        bufs = _bufs(text)
        if any(bufs.get(t) == bufs[tout] for t in cv["args"][1:-1]):
            # the conv's epilogue operand lives in the buffer it would now write (TVM reused it for the layout op
            # output): writing early would clobber it -> leave this layout op in place
            skipped.add(c["idx"])
            log(f"KL conv out [{cv['idx']}] -> [{c['idx']}] {c['fn']}: SKIP (epilogue operand shares buffer {bufs[tout]})")
            continue
        assert _consumers(calls, j - 1, tin) == [j], ("conv out used elsewhere", tin)
        xi = sh[tin]
        assert len(xi) == 5 and xi[0] == 1, xi
        O, OH, OW = xi[1] * xi[4], xi[2], xi[3]
        assert int(np.prod(sh[tout])) == O * OH * OW, (sh[tout], xi)
        body = _tvm_body(csrc, c["fn"])
        nhwc = "_transpose_" in c["fn"]
        if nhwc:
            # out [HW][C] row-major: T_reshape_1[(p * C) + c]
            assert sh[tout][-2:] == [OH * OW, O] and re.search(r"T_reshape_1\)\[\(+ax0_ax1_fused \* %d\)" % O, body), \
                ("nhwc body", c["fn"], sh[tout])
            # epilogue of an NHWC-labelled output: per-channel only (a residual would need the same block)
            for t in cv["args"][2:-1]:
                assert int(np.prod(sh[t])) == O, ("conv residual with NHWC out", cv["idx"], t)
            lab = (1, 1, OH, OW, O)
        else:
            # out = planar NCHW (+ reshape label [1, G, C/G*HW]): T_reshape_1[(g * X) + i] walks the output in order
            assert len(sh[tout]) == 3 and sh[tout][0] == 1 and re.search(r"T_reshape_1\)\[\(+(?:\(ax0_ax1_fused \* %d\)|cse_var_1)"
                                                                          % (sh[tout][2]), body), ("planar body", c["fn"], sh[tout])
            lab = (1, O, OH, OW, 1)
        cblk = text[cv["seg"]:cv["span"][1]]
        na = len(cv["args"])
        old_y = "    args[%d].v_handle = &tensors[%d];\n" % (na - 1, tin)
        assert cblk.count(old_y) == 1, ("conv y arg", cv["idx"])
        nm = "g_kl_y_%d" % cv["idx"]
        decl.append("static DLTensor %s;\nstatic int64_t %s_shape[5] = {%d, %d, %d, %d, %d};\n" % ((nm, nm) + lab))
        cblk = cblk.replace(old_y, "    %s = tensors[%d]; %s.data = tensors[%d].data; %s.shape = %s_shape;   /* P4a: %s */\n"
                            "    args[%d].v_handle = &%s;\n"
                            % (nm, tin, nm, tout, nm, nm, "NHWC out" if nhwc else "planar NCHW out", na - 1, nm))
        rblk = "    /* P4a keep-layout: call [%d] %s removed -- the AIE conv [%d] writes this layout */\n" % (
            c["idx"], c["fn"], cv["idx"])
        text = text[:cv["seg"]] + cblk + text[cv["span"][1]:c["span"][0]] + rblk + text[c["span"][1]:]
        n += 1
        log(f"KL conv out [{cv['idx']}] -> [{c['idx']}] {c['fn']}: {'NHWC' if nhwc else 'planar'} label {lab} into t{tout}")
    if n:
        i = text.index("float *graph_run(void)")
        i = text.rindex("\n", 0, i) + 1
        text = text[:i] + "/* P4a keep-layout: AIE conv output labels */\n" + "".join(decl) + "\n" + text[i:]
    return text, n


def keep_layout_driver(text: str, csrc: str, wbin: Path, log=print) -> tuple:
    text, nt = temb_pass(text, csrc, wbin, log)
    text, na = fa_pass(text, csrc, wbin, log)
    text, npo = projout_pass(text, csrc, log)
    text, nco = convout_pass(text, csrc, log)
    log(f"KEEP_LAYOUT_RT temb_sites={nt} attn_groups={na} proj_out={npo} conv_out={nco}")
    return text, nt, na
