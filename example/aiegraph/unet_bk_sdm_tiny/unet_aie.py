"""--aie conv for build_unet.py (sprint unet-bf16-streamload, P35w).

The 37 conv layers of p35/partition.json go to the AIE; everything else stays
TVM C on the A78 (bf16 weights expanded into g_wscratch, Steve's .bss scratch).

  * map     each partition layer (ONNX Conv, by weight name) -> the TVM call that
            carries the same weight: fp32 weight statistics (count, sum, sum of
            squares) of the ONNX initializer == those of the TVM param (the TVM
            OIHW{i}i{o}o weight is a permutation). Unique match or the build fails.
  * kernel  gemm_bf16 (p2k/bf16_gemm_convin.cc, verbatim): one launch = one 64x64x64
            tile (bf16 x bf16 -> fp32). Host function unet_aie_conv() (C linkage)
            runs any NCHWc conv as ceil(M/64)*ceil(N/64)*ceil(K/64) launches:
            activation fp32 -> bf16 RNE im2col tile, weight tile gathered straight
            from the COMPACT bf16 blob (weights.bin, never expanded), each 64-K slice
            rounded to bf16, fp32 K-sum + bias/temb/residual adds on the APU.
            g_aie_launches++ per real launch; MACs counted per launch.
  * archive built by aiegraph's own offload flow on ONE carrier layer (a TVM function
            all of whose call sites are partition layers); calibrate/emulate (whole
            UNet on x86) are skipped -- force=True ignores their verdicts anyway.
  * driver  graph_driver.c is post-processed: each partition call site calls
            unet_aie_conv(cid, args, nargs, g_params+off, sh, sw, ph, pw) instead of
            the TVM kernel, and its weight is NOT expanded into g_wscratch.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_SRC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../src"))
if _SRC not in _sys.path:  # C text lives in src/aiednn/<cat>/ (kernels) and src/aiegraph/runtime/
    _sys.path.insert(0, _SRC)
from aiednn import path as _tplpath, runtime_source as _rt, template as _tpl  # noqa: E402

P2K = _tplpath("conv2d/bf16_gemm_convin.cc")  # 64x64 ports (8e7254d, board PASS p0_board.log)
P2K_KT = _tplpath("conv2d/bf16_gemm_convin_ktimer.cc")  # --ktimer: core timer tail, b 80x64 / c 64x80
ENTRY = "aiek_unetconv"


def log(msg):
    print(f"[build_unet/aie] {msg}", flush=True)


def ceil64(v):
    return (v + 63) // 64


# ─────────────────────────────── mapping ───────────────────────────────
def _stats(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return a.size, float(a.sum()), float((a * a).sum()), float(np.abs(a).sum())


def map_layers(onnx_path: Path, partition: dict, work: Path, offsets_fn) -> tuple:
    """partition layers -> TVM conv calls: (one dict per layer in partition order, all conv calls)."""
    import onnx
    from onnx import numpy_helper
    graphs = sorted(work.glob("*_graph.json"))
    assert len(graphs) == 1, graphs
    g = json.loads(graphs[0].read_text())
    params = graphs[0].with_name(graphs[0].name.replace("_graph.json", "_params.bin"))
    offs = offsets_fn(params)
    nodes, row_ptr = g["nodes"], g["node_row_ptr"]
    shapes = g["attrs"]["shape"][1]
    blob = np.memmap(params, dtype=np.uint8, mode="r")
    convs = []
    for i, n in enumerate(nodes):
        if n.get("op") != "tvm_op" or "conv2d_NCHWc" not in n["attrs"]["func_name"]:
            continue
        ins = [s for s, _, _ in n["inputs"]]
        pins = [s for s in ins if nodes[s]["op"] == "null" and nodes[s]["name"] in offs]
        w = max(pins, key=lambda s: offs[nodes[s]["name"]][1])
        wname = nodes[w]["name"]
        off, nb = offs[wname]
        st = _stats(np.frombuffer(blob[off:off + nb].tobytes(), dtype="<f4"))
        convs.append(dict(node=i, symbol="tvmgen_default_" + n["attrs"]["func_name"]
                          if not n["attrs"]["func_name"].startswith("tvmgen_") else n["attrs"]["func_name"],
                          func=n["attrs"]["func_name"], wname=wname, warg=ins.index(w),
                          nargs=len(ins) + int(n["attrs"].get("num_outputs", 1)),
                          wshape=shapes[row_ptr[w]], xshape=shapes[row_ptr[ins[0]]],
                          yshape=shapes[row_ptr[i]], stats=st))
    log(f"TVM graph {graphs[0].name}: {len(convs)} conv2d_NCHWc calls")

    m = onnx.load(str(onnx_path))
    inits = {t.name: t for t in m.graph.initializer}
    oconv = [n for n in m.graph.node if n.op_type == "Conv"]
    # ONNX Conv order == TVM conv call order (checked: same count, same O/I/KH/KW per
    # position, and the fp32 weight statistics agree wherever TVM did not fold a scale
    # into the weight -- FoldScaleAxis folds the GroupNorm affine into proj_in).
    assert len(oconv) == len(convs), (len(oconv), len(convs))
    by_name, folded = {}, []
    for n, c in zip(oconv, convs):
        w = numpy_helper.to_array(inits[n.input[1]])
        ws = c["wshape"]
        assert list(w.shape) == [ws[0] * ws[5], ws[1] * ws[4], ws[2], ws[3]], (n.name, w.shape, ws)
        st = _stats(w)
        same = (c["stats"][0] == st[0] and abs(c["stats"][1] - st[1]) <= 1e-6 * st[3] + 1e-9
                and abs(c["stats"][2] - st[2]) <= 1e-6 * st[2] + 1e-12)
        if not same:
            folded.append(n.name)
        by_name[n.name] = c
    assert all("proj_in" in x for x in folded), f"weight mismatch outside proj_in: {folded}"
    log(f"ONNX Conv -> TVM call by graph order: {len(convs)} pairs, shapes equal, weight stats equal "
        f"for {len(convs) - len(folded)} (the other {len(folded)} are proj_in, GN affine folded in by TVM)")
    onodes = {n.name: n for n in m.graph.node}
    out, used = [], set()
    for L in partition["layers"]:
        if L.get("cls") in ("linear", "attn"):
            continue   # PDN: map_linears; GATT: map_attn (autopart --mode attn partitions)
        n = onodes[L["node"]]
        assert n.op_type == "Conv" and n.input[1] == L["weight"], (L["node"], n.op_type)
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        assert at.get("group", 1) == 1 and list(at.get("dilations", [1, 1])) == [1, 1], at
        pads = list(at.get("pads", [0, 0, 0, 0]))
        assert pads[0] == pads[2] and pads[1] == pads[3], pads
        cand = [by_name[L["node"]]]
        c = cand[0]
        assert c["node"] not in used, c
        used.add(c["node"])
        ob, ib = c["wshape"][5], c["wshape"][4]
        O, C, KH, KW = c["wshape"][0] * ob, c["wshape"][1] * ib, c["wshape"][2], c["wshape"][3]
        M = c["yshape"][2] * c["yshape"][3]
        assert c["xshape"][1] * c["xshape"][4] == C and c["yshape"][1] * c["yshape"][4] == O, c
        assert [KH, KW] == list(at["kernel_shape"]), (KH, KW, at)
        launches = ceil64(M) * ceil64(O) * ceil64(C * KH * KW)
        mkn = L["mkn"][0]
        assert [M, C * KH * KW, O] == list(mkn), (L["node"], M, C * KH * KW, O, mkn)
        assert launches == L["launches"], (L["node"], launches, L["launches"])
        assert c["warg"] == 1, c
        out.append(dict(c, kind="conv", cid=L["idx"], onnx=L["node"], weight=L["weight"], launches=launches,
                        macs=M * O * C * KH * KW, sh=at["strides"][0], sw=at["strides"][1],
                        ph=pads[0], pw=pads[1], M=M, N=O, K=C * KH * KW))
    log(f"mapped {len(out)} partition layers to TVM calls")
    return out, convs


def kacc_split(N: int, K: int, Mp: int) -> tuple:
    """PDN: launches of ONE gemm_kacc instance for a dense GEMM (= kacc_gemm.plan, PDK board-proven c009):
    the whole GEMM if (ch N, k K, px Mp) fits, else S launches along px (X rows / Y blocks contiguous).
    Returns (S, px per launch)."""
    def fits(ch, k, px):
        try:
            kacc_geom(ch, k, px)
            return True
        except AssertionError:
            return False
    if fits(N, K, Mp):
        return 1, Mp
    for s in (2, 4, 8, 16):
        if Mp % (64 * s) == 0 and fits(N, K, Mp // s):
            return s, Mp // s
    raise SystemExit(f"kacc_split: no px split of N={N} K={K} M={Mp} fits gemm_kacc")


def map_linears(onnx_path: Path, partition: dict, work: Path, offsets_fn) -> list:
    """PDN: partition 'linear' layers (ONNX MatMul/Gemm with a constant weight) -> TVM nn_contrib_dense_pack
    calls (weight packed [N/nb][K][nb], arg 1). Matched by weight VALUE: the TVM fp32 param unpacked to [N][K]
    must equal the ONNX weight (MatMul B^T / Gemm B or B^T) exactly; a unique match per layer or the build fails."""
    import onnx
    from onnx import numpy_helper
    graphs = sorted(work.glob("*_graph.json"))
    assert len(graphs) == 1, graphs
    g = json.loads(graphs[0].read_text())
    params = graphs[0].with_name(graphs[0].name.replace("_graph.json", "_params.bin"))
    offs = offsets_fn(params)
    nodes, row_ptr = g["nodes"], g["node_row_ptr"]
    shapes = g["attrs"]["shape"][1]
    blob = np.memmap(params, dtype=np.uint8, mode="r")
    dense = {}   # (N, K) -> list of TVM calls
    ndense = 0
    for i, n in enumerate(nodes):
        if n.get("op") != "tvm_op" or "dense" not in n["attrs"]["func_name"]:
            continue
        ndense += 1
        ins = [s for s, _, _ in n["inputs"]]
        w = ins[1]
        if nodes[w]["op"] != "null" or nodes[w]["name"] not in offs:
            continue   # weight computed at run time (time embedding, M=1): stays on the CPU
        ws, xs, ys = shapes[row_ptr[w]], shapes[row_ptr[ins[0]]], shapes[row_ptr[i]]
        if len(ws) != 3 or len(xs) != 2 or len(ys) != 2:
            continue
        N, K = ws[0] * ws[2], ws[1]
        f = n["attrs"]["func_name"]
        dense.setdefault((N, K), []).append(dict(
            node=i, symbol=f if f.startswith("tvmgen_") else "tvmgen_default_" + f, func=f,
            wname=nodes[w]["name"], warg=1, nargs=len(ins) + int(n["attrs"].get("num_outputs", 1)),
            wshape=ws, xshape=xs, yshape=ys))
    log(f"TVM graph {graphs[0].name}: {ndense} dense calls, {sum(len(v) for v in dense.values())} with a param weight")
    m = onnx.load(str(onnx_path))
    inits = {t.name: t for t in m.graph.initializer}
    onodes = {n.name: n for n in m.graph.node}
    out, used = [], set()
    for L in partition["layers"]:
        if L.get("cls") != "linear":
            continue
        n = onodes[L["node"]]
        assert n.op_type in ("MatMul", "Gemm") and L["weight"] in n.input, (L["node"], n.op_type)
        b = numpy_helper.to_array(inits[L["weight"]]).astype(np.float32)
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        wnk = b if (n.op_type == "Gemm" and at.get("transB", 0)) else b.T   # natural W [N][K]
        M, K, N = L["pad"]["orig"] if "pad" in L else L["mkn"][0]
        assert wnk.shape == (N, K), (L["node"], wnk.shape, N, K)
        hit = []
        for c in dense.get((N, K), []):
            if c["node"] in used or c["xshape"][0] != M:
                continue
            off, nb = offs[c["wname"]]
            on, kk, ob = c["wshape"]
            pk = np.frombuffer(blob[off:off + nb].tobytes(), dtype="<f4").reshape(on, kk, ob)
            if np.array_equal(pk.transpose(0, 2, 1).reshape(N, K), wnk):
                hit.append(c)
        assert len(hit) == 1, f"linear {L['node']} ({L['weight']} M={M} K={K} N={N}): {len(hit)} TVM matches"
        c = hit[0]
        used.add(c["node"])
        assert c["yshape"] == [M, N], (L["node"], c["yshape"])
        S, pxs = kacc_split(N, K, -(-M // 64) * 64)
        out.append(dict(c, kind="linear", cid=L["idx"], onnx=L["node"], weight=L["weight"], launches=S, S=S,
                        pxs=pxs, macs=M * N * K, M=M, N=N, K=K, sh=1, sw=1, ph=0, pw=0))
    log(f"mapped {len(out)} partition linear layers to TVM dense calls (weights equal by value), "
        f"{sum(c['S'] for c in out)} gemm_kacc launches")
    return out


# ─────────────────────── GATT: attention bmm sites (--aie conv,linear,attn) ───────────────────────
def bmm_geom(H: int, M: int, K: int, N: int) -> dict:
    """GATT: geometry of one attention bmm Y[h] = X[h] (M x K) . W[h]^T (W [N][K]) on ONE gemm_kacc instance --
    the board-proven PBK rules of kacc_gemm.main_bmm (fd4cb96): M/K/N zero-padded to 64 (aiegraph.gemmpad.pad_mkn),
    swap (ch = M) for big square outputs, kacc_gemm.plan split, one more zero ch block when the a/b ports would
    have the same shape (aiehlc GemmSpace port sizing), K padded until kp/2 is not 32/64 (aiehlc 2-dim+iter b BD)."""
    import sys
    src = str(Path(__file__).resolve().parents[3] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from aiegraph.gemmpad import pad_mkn
    import kacc_gemm
    Mp, Kp, Np = pad_mkn(M, K, N)
    swap = int(Mp == Np and H * M * N >= (4 << 20))
    CHR, PXR = (M, N) if swap else (N, M)
    CHP, PXP = (Mp, Np) if swap else (Np, Mp)
    mode, S, chs, pxs = kacc_gemm.plan(CHP, Kp, PXP)
    geo = kacc_geom(chs, Kp, pxs)
    while geo["arows"] == geo["brows"] and mode == "none":
        CHP += 64
        mode, S, chs, pxs = kacc_gemm.plan(CHP, Kp, PXP)
        geo = kacc_geom(chs, Kp, pxs)
    while geo["kp"] // 2 in (32, 64):
        Kp += 64
        geo = kacc_geom(chs, Kp, pxs)
    return dict(H=H, M=M, K=K, N=N, swap=swap, CHR=CHR, PXR=PXR, CHP=CHP, PXP=PXP, Kp=Kp, kp=geo["kp"],
                kch=geo["kch"], tr=geo["tr"], splitpx=int(mode == "px"), S=S, chs=chs, pxs=pxs)


def map_attn(partition: dict, work: Path, onnx=None) -> list:
    """GATT (GAP-aiegraph-attn-site), sprint unet-m3 S3: the matching moved into aiegraph --
    ``aiegraph.attention.partition_attention`` finds the attention groups on graph.json (+ ONNX for the
    Q/K/V sources and the scale) and joins the partition 'attn' rows to their TVM nn_batch_matmul calls by
    ONNX node name. Here only the kernel geometry is added (bmm_geom). ``ag_group`` = the AttentionGroup."""
    import sys
    src = str(Path(__file__).resolve().parents[3] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from aiegraph.attention import partition_attention
    from aiegraph.group import CallGraph
    cg = CallGraph.load(work)
    att = partition_attention(cg, partition, onnx=onnx, workdir=work)
    log(f"TVM graph: {len(att)} attention bmm calls in {len({c['ag_group'].name for c in att})} aiegraph attention "
        f"groups ({sum(len(g.calls) for g in {c['ag_group'].name: c['ag_group'] for c in att}.values())} calls)")
    out = []
    for c in att:
        H, M, K, N = c["H"], c["M"], c["K"], c["N"]
        geo = bmm_geom(H, M, K, N)
        out.append(dict(c, geo=geo, launches=H * geo["S"], macs=H * M * N * K))
    log(f"mapped {len(out)} partition attn bmm layers ({len({c['group'] for c in out})} groups) to TVM "
        f"nn_batch_matmul calls, {sum(c['launches'] for c in out)} gemm_kacc launches")
    return out


def attn_shapes(att: list) -> list:
    """gemm_kacc instances (ch, K, px) of the attention sites"""
    return sorted({(c["geo"]["chs"], c["geo"]["Kp"], c["geo"]["pxs"]) for c in att})


def keep_layout_plan(onnx_path, partition: dict, layers: list = ()) -> dict:
    """GAP-B: the partition rows with ``keep_layout`` (aiegraph.partition.plan_keep_layout: output stays in the
    AIE layout, Transpose/Reshape on the way are labels only, every consumer is a partition layer).
    Tags the mapped ``layers`` (map_layers / map_linears / map_attn dicts, by ``onnx`` node name):
    producer ``keep_layout=<layout>``, consumer ``keep_in=[producer node, ...]``. No row with the field ->
    ``{"edges": [], ...}`` and the layers are not touched (the generated code stays byte-identical)."""
    from aiegraph.partition import plan_keep_layout
    rep = plan_keep_layout(onnx_path, partition)
    by = {c.get("onnx"): c for c in layers}
    for e in rep["edges"]:
        if e["node"] in by:
            by[e["node"]]["keep_layout"] = e["layout"]
        for u in e["consumers"]:
            if u in by:
                by[u].setdefault("keep_in", []).append(e["node"])
    return rep


def check_data_arg0(layers_dir: Path, sel: list) -> None:
    """The TVM kernel's conv reads its activation from p0 (not a residual/temb arg)."""
    for sym in sorted({c["symbol"] for c in sel}):
        hits = [p for p in layers_dir.glob("*/*.c") if f" {sym}(void* args" in p.read_text()]
        assert len(hits) == 1, (sym, hits)
        t = hits[0].read_text()
        src = set()
        for line in t.splitlines():
            if "p1_1)[" in line and "*" in line:
                src |= {x for x in re.findall(r"\(\(float\*\)(\w+)\)\[", line) if x != "p1_1"}
        if "data_pad" in src:
            src.discard("data_pad")
            for line in t.splitlines():
                if re.search(r"condval = |data_pad\)\[.*\] = ", line):
                    src |= set(re.findall(r"\(\(float\*\)(p\d+_1)\)\[", line))
        assert src == {"p0_1"}, f"{sym} ({hits[0].name}): conv activation from {src}, expected p0_1"
    log(f"checked {len({c['symbol'] for c in sel})} TVM conv kernels: activation = arg 0, weight = arg 1")


def pick_carrier(sel: list, all_calls: list, layers_dir: Path):
    selnodes = {c["node"] for c in sel}
    for c in sel:
        same = [x for x in all_calls if x["symbol"] == c["symbol"]]
        if all(x["node"] in selnodes for x in same):
            d = [p.parent for p in layers_dir.glob("*/*.c") if f" {c['symbol']}(void* args" in p.read_text()]
            assert len(d) == 1, d
            return int(d[0].name.split("_")[0]), c
    raise SystemExit("no carrier: every selected TVM conv function also has CPU call sites")


# ─────────────────────────────── kernel source ───────────────────────────────
HOST_CC = (_tpl("conv2d/tile64_conv_host.cc") + _tpl("elementwise/epilogue_bias_residual.cc") + _tpl("conv2d/tile64_conv_entry.cc"))

REF_C = _tpl("conv2d/conv_bf16_carrier_ref.c")


def kernel_text(ktimer: bool = False) -> str:
    src = (P2K_KT if ktimer else P2K).read_text()
    m = re.search(r"(__global__ void gemm_bf16\(.*?\n\})\s*\n\s*int main\(", src, re.S)
    assert m, "gemm_bf16 not found"
    return m.group(1)


SOFTFLOAT_RM = _tpl("conv2d/softfloat_rm_reset.cc")


CTRLPKT_PRAGMAS = (  # --ctrlpkt aot (P2a): spelled as aiehlc prints them (src/llvm/aiehlc.cc); no KERNELCONFIGOFFLOAD (P3)
    _tpl("gemm/ctrlpkt_aot_pragmas.h"))
# --ctrlpkt aot-kco (build_unet.py) / UNET_CTRLPKT_KCO=1 in the env (kacc_bench.sh -> kacc_gemm.py): the same 5
# pragmas + KERNELCONFIGOFFLOAD (the core programs its own DMA, .TM in kernel.lst); one switch for probe and full net
CTRLPKT_KCO = os.environ.get("UNET_CTRLPKT_KCO", "0") == "1"
if CTRLPKT_KCO:
    CTRLPKT_PRAGMAS += _tpl("gemm/ctrlpkt_kco_pragma.h")
CTRLPKT_NPRAGMAS = CTRLPKT_PRAGMAS.count("#pragma")


def kt_select(text: str, ktimer: bool) -> str:
    """keep /*KT+*/../*KT-*/ spans with --ktimer, /*NK+*/../*NK-*/ spans without; drop the other"""
    keep, drop = (("KT", "NK") if ktimer else ("NK", "KT"))
    text = re.sub(r"/\*%s\+\*/.*?/\*%s-\*/" % (drop, drop), "", text, flags=re.S)
    text = text.replace("/*%s+*/" % keep, "").replace("/*%s-*/" % keep, "")
    assert "/*KT" not in text and "/*NK" not in text
    return text


def write_kernel(kdst: Path, spec: str, maxacc: int, warm: bool = False, ctrlpkt: str = "off",
                 ktimer: bool = False) -> Path:
    spec = spec.replace('"out ', '"export    unet_aie_conv\\n"\n        "export    unet_aie_stats\\n"\n'
                        '        "export    unet_aie_launch_stats\\n"\n'
                        '        "export    unet_aie_host_split_site\\n"\n'
                        '        "export    unet_aie_host_split_line\\n"\n' +
                        ('        "export    unet_aie_ktimer_stats\\n"\n' if ktimer else '') +
                        '        "export    unet_aie_split_stats\\n"\n        "out ', 1)
    assert "export    unet_aie_conv" in spec, spec
    kdst.mkdir(parents=True, exist_ok=True)
    kcc = kdst / f"{ENTRY}.cc"
    head = _tpl("conv2d/tile64_conv_head.cc").replace("@E@", ENTRY).replace("@SRC@", str(P2K_KT if ktimer else P2K))
    if ctrlpkt == "aot":
        head = CTRLPKT_PRAGMAS + head
    ktext = kernel_text(ktimer)
    if warm:
        i = ktext.index(") {\n") + len(") {\n")
        ktext = ktext[:i] + SOFTFLOAT_RM + ktext[i:]
    kcc.write_text(head + ktext + "\n" + kt_select(HOST_CC, ktimer).replace("@MAXACC@", str(maxacc)).replace("@WARM@", "1" if warm else "0")
                   .replace("@CTRLPKT@", "1" if ctrlpkt == "aot" else "0")
                   .replace("@PART_C0@", "2" if ctrlpkt == "aot" else "3")
                   .replace("@E@", ENTRY).replace("@SPEC@", spec))
    (kdst / f"{ENTRY}_ref.c").write_text(REF_C.replace("@E@", ENTRY))
    return kcc


# ─────────────────────────────── driver patch ───────────────────────────────
DECL = ("int unet_aie_conv(int cid, void *args, int nargs, const uint16_t *wbf, "
        "int sh, int sw, int ph, int pw);   /* build_unet.py --aie conv */\n"
        "int unet_aie_linear(int cid, void *args, int nargs, const uint16_t *wbf, int S);   /* --aie conv,linear */\n")
DECL_ATTN = ("int unet_aie_attn(int cid, void *args, int nargs, int qk, int swap, int chp, int pxp, int kk, int kp, "
             "int kch, "
             "int splitpx, int S, int chs, int pxs, int tr);   /* --aie conv,linear,attn (GATT) */\n")
CALL_RE = re.compile(r"    GRAPH_PROF_BEGIN\((\d+)\);\n(#ifdef GRAPH_AIE_OFFLOAD\n.*?#endif\n|"
                     r"    \w+\(args, codes, \d+, 0, 0, 0\);\n    GRAPH_PROF_END\(\1, \"[^\"]*\"\);\n)", re.S)


def patch_driver(text: str, sel: list) -> tuple:
    """The text patches aiegraph does not do yet. S1 (unet-m3): when aiegraph already placed the
    conv / linear group steps (its declarations sit before kernel_ws), only the attention sites,
    the attention declaration and the PROF glue are added here; *sel* must then hold attn layers
    only. Without aiegraph steps (test_attn_sites.py) the conv / linear sites are patched as before."""
    head = text.split("static uint8_t kernel_ws[")[0]
    if "UP_AIE_BEGIN(" in head or "up_pending_exp" in head:
        raise RuntimeError("driver already patched")
    ag_steps = "int unet_aie_conv(int cid, void" in head
    att = [c for c in sel if c.get("kind") == "attn"]
    sel = [c for c in sel if c.get("kind") != "attn"]
    if ag_steps and sel:
        raise RuntimeError("aiegraph placed the conv/linear steps: patch_driver takes attn layers only")
    i = text.index("static uint8_t kernel_ws[")
    # GATT: the attn declaration / PROF branch only when there are attn sites (--aie conv,linear stays byte-identical)
    fam = bool(att) and all(c.get("fa") is not None for c in att)   # GFA: fused flash-attention sites
    if fam:
        import fa_attn
        # S3: the FA groups are aiegraph group offloads (build_unet.fa_offload_groups): arm_build already placed
        # their steps and removed the other calls (softmax included); only the declarations are added here
        assert all(c.get("fa_steps") for c in att), "GFA sites must be aiegraph group steps"
    text = text[:i] + ("" if ag_steps else DECL) + ((fa_attn.DECL_FA + fa_attn.DECL_KL if fam else DECL_ATTN) if att else "") + \
        (prof_c_attn() if att else PROF_C) + "\n" + text[i:]
    exp_body = "    for (uint64_t i = 0; i < n; ++i) d[i] = (uint32_t)src[i] << 16;\n}\n"
    assert text.count(exp_body) == 1, text.count(exp_body)
    text = text.replace(exp_body, "    XTime e0_, e1_;\n    XTime_GetTime(&e0_);\n" + exp_body[:-2] +
                        "    XTime_GetTime(&e1_);\n    up_pending_exp += e1_ - e0_;\n    up_exp_calls++;\n}\n", 1)
    run = "float *graph_run(void) {\n"
    assert text.count(run) == 1
    text = text.replace(run, run + "    up_pending_exp = 0; /* P1 PROF: expands before graph_run are not graph_run time */\n", 1)
    n = 0
    for c in sel:
        m = re.search(r"    tensors\[(\d+)\]\.data = 0;   /\* %s: bf16 @(\d+)\+(\d+) B" % re.escape(c["wname"]), text)
        assert m, f"no bf16 init line for {c['wname']}"
        eid, off, nb = int(m.group(1)), int(m.group(2)), int(m.group(3))
        c["woff"] = off   # P4w: prelayout_weights rewrites this bf16 region of weights.bin
        wel = 1
        for d in c["wshape"]:
            wel *= d
        assert nb == 2 * wel, (c["wname"], nb, wel)
        exp = re.compile(r"    graph_bf16_expand\(\(float \*\)\(g_wscratch \+ (\d+)\), \(const uint16_t \*\)"
                         r"\(g_params \+ %d\), %dULL\);\n    tensors\[%d\]\.data = g_wscratch \+ \1;\n" % (off, wel, eid))
        hits = exp.findall(text)
        assert len(hits) == 1, (c["wname"], len(hits))
        text = exp.sub("    tensors[%d].data = 0;   /* AIE conv %d: bf16 weight read in place by unet_aie_conv */\n"
                       % (eid, c["cid"]), text)
        a1 = "    args[1].v_handle = &tensors[%d];\n" % eid
        assert text.count(a1) == 1, (c["wname"], text.count(a1))
        pos = text.index(a1)
        cm = CALL_RE.search(text, pos)
        nxt = text.find("    args[0].v_handle", pos)
        assert cm and (nxt < 0 or cm.start() < nxt), c["wname"]
        calls = re.findall(r"(?:aie_)?(tvmgen_\w+)\(args, codes, (\d+), 0, 0, 0\);", cm.group(2))
        assert calls and all(s == c["symbol"] and int(k) == c["nargs"] for s, k in calls), (calls, c["symbol"])
        if c.get("kind") == "linear":   # PDN: tag "aie unet_aie_lin <cid>" -> PROF op=aie_host_linear<cid>
            rep = ("    GRAPH_PROF_BEGIN(%s);\n    UP_AIE_BEGIN(%s);\n"
                   "    unet_aie_linear(%d, args, %d, (const uint16_t *)(g_params + %d), %d);\n"
                   "    UP_AIE_END(%s);\n    GRAPH_PROF_END(%s, \"aie unet_aie_lin %d\");\n"
                   % (cm.group(1), cm.group(1), c["cid"], c["nargs"], off, c["S"], cm.group(1), cm.group(1), c["cid"]))
        else:
            rep = ("    GRAPH_PROF_BEGIN(%s);\n    UP_AIE_BEGIN(%s);\n"
                   "    unet_aie_conv(%d, args, %d, (const uint16_t *)(g_params + %d), %d, %d, %d, %d);\n"
                   "    UP_AIE_END(%s);\n    GRAPH_PROF_END(%s, \"aie unet_aie_conv %d\");\n"
                   % (cm.group(1), cm.group(1), c["cid"], c["nargs"], off, c["sh"], c["sw"], c["ph"], c["pw"],
                      cm.group(1), cm.group(1), c["cid"]))
        text = text[:cm.start()] + rep + text[cm.end():]
        n += 1
    nl = sum(1 for c in sel if c.get("kind") == "linear")
    if not ag_steps:
        assert text.count("    unet_aie_conv(") == len(sel) - nl and text.count("    unet_aie_linear(") == nl
    if fam:   # S3: count the group steps arm_build emitted (2 per bmm site: pack_q+pack_k / pack_v+run_o)
        nkl = len(re.findall(r"^    unet_aie_attn_kl\(", text, re.M))
        assert nkl == 2 * len(att), (nkl, len(att))
        assert not re.search(r"\btvmgen_default_fused_nn_(softmax|batch_matmul)\w*\(args, ",
                             text.split("float *graph_run(void)")[1]), "attention call left in graph_run"
        return text, n + len(att)
    for c in att:   # GATT: tag "aie unet_aie_attn <cid>" -> PROF op=aie_host_attn<cid>
        text = _patch_attn_site(text, c)
        n += 1
    assert text.count("    unet_aie_attn(") == len(att)
    return text, n


def _patch_attn_site(text: str, c: dict) -> str:
    """GATT: one TVM nn_batch_matmul call site (located by its unique args[0..2] tensor block) -> unet_aie_attn.
    No weight: both operands are activations, packed at run time by the host (aie_kacc_pack.h)."""
    e0, e1 = c["eids"]
    blk = ("    args[0].v_handle = &tensors[%d];\n    args[1].v_handle = &tensors[%d];\n"
           "    args[2].v_handle = &tensors[%d];\n" % (e0, e1, c["yeid"]))
    assert text.count(blk) == 1, (c["onnx"], c["node"], text.count(blk))
    pos = text.index(blk)
    cm = CALL_RE.search(text, pos)
    nxt = text.find("    args[0].v_handle", pos + len(blk))
    assert cm and (nxt < 0 or cm.start() < nxt), (c["onnx"], c["node"])
    calls = re.findall(r"(?:aie_)?(tvmgen_\w+)\(args, codes, (\d+), 0, 0, 0\);", cm.group(2))
    assert calls and all(s == c["symbol"] and int(k) == c["nargs"] for s, k in calls), (calls, c["symbol"])
    if c.get("fa") is not None:  # GFA: role 0 = QK (pack Q, K), role 1 = PV (pack V, launch, write O)
        rep = ("    GRAPH_PROF_BEGIN(%s);\n    UP_AIE_BEGIN(%s);\n"
               "    unet_aie_attn(%d, args, %d, %d, %d);   /* fa %s g%d %s */\n"
               "    UP_AIE_END(%s);\n    GRAPH_PROF_END(%s, \"aie unet_aie_attn %d\");\n"
               % (cm.group(1), cm.group(1), c["cid"], c["nargs"], int(c["role"] == "pv"), c["fa"], c["role"],
                  c["group"], c["onnx"].replace("*/", "*_/"), cm.group(1), cm.group(1), c["cid"]))
        return text[:cm.start()] + rep + text[cm.end():]
    g = c["geo"]
    rep = ("    GRAPH_PROF_BEGIN(%s);\n    UP_AIE_BEGIN(%s);\n"
           "    unet_aie_attn(%d, args, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);   /* %s g%d %s */\n"
           "    UP_AIE_END(%s);\n    GRAPH_PROF_END(%s, \"aie unet_aie_attn %d\");\n"
           % (cm.group(1), cm.group(1), c["cid"], c["nargs"], int(c["role"] == "qk"), g["swap"], g["CHP"], g["PXP"],
              g["Kp"], g["kp"], g["kch"], g["splitpx"], g["S"], g["chs"], g["pxs"], g["tr"], c["role"], c["group"],
              c["onnx"].replace("*/", "*_/"), cm.group(1), cm.group(1), c["cid"]))
    return text[:cm.start()] + rep + text[cm.end():]


# P1 (b) per-call-site A78 time: inserted into graph_driver.c by patch_driver
PROF_C = _rt("site_prof.c")

PROF_ATTN_BRANCH = (
    _rt("site_prof_attn_branch.c"))


def prof_c_attn() -> str:
    """PROF_C with the GATT branch: attention sites print op=aie_host_attn<cid> (counted as aie_host by hsplit)"""
    k = '        if (is_aie && strncmp(up_tag[i], "aie unet_aie_lin ", 17) == 0)'
    assert PROF_C.count(k) == 1
    return PROF_C.replace(k, PROF_ATTN_BRANCH + k.lstrip(), 1)


MAIN_EXTRA_DECL = (
    _rt("main_extra_decl.h"))

MAIN_CMP = _rt("main_cmp.c")


LAUNCH_LINE = _rt("main_launch_line.c")


PROF_LINE = _rt("main_prof_line.c")


def main_c(base: str, ktimer: bool = False) -> str:
    old = '    int rc = p1_cmp("unet-" UNET_AIE_MODE, out, UNET_WDTYPE "w/" UNET_AIE_MODE, 0, 0, 0, 0.0);\n'
    assert old in base
    s = base.replace(old, kt_select(MAIN_CMP, ktimer))
    s = s.replace("extern const uint8_t _binary_weights_bin_start[];\n",
                  "extern const uint8_t _binary_weights_bin_start[];\n" + kt_select(MAIN_EXTRA_DECL, ktimer), 1)
    s = s.replace("init (aiegraph fp32 C on A78, weights=%s, aie=%s)",
                  "init (aiegraph C on A78 + conv on AIE gemm_bf16, weights=%s, aie=%s)")
    anchor = ' n, REF_UNET_OUT_N);\n'
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + LAUNCH_LINE + PROF_LINE, 1)
    run = ("    XTime_GetTime(&t0);\n    float *out = graph_run();\n    XTime_GetTime(&t1);\n")
    assert s.count(run) == 1
    s = s.replace(run, "    unet_pmc_enable();\n    unsigned long long pmc0 = unet_pmc();\n" + run.replace(
        "    XTime_GetTime(&t1);\n", "    XTime_GetTime(&t1);\n    unsigned long long pmc1 = unet_pmc();\n"), 1)
    assert "unet_aie_stats(&launches" in s and "_Z22__Runtime_phase_cyclesPyPj" in s
    return s


# ─────────────────────────────── --kincore (P4bg) ───────────────────────────────
# One launch per conv layer, K accumulated in the core: Maxwell's gemm_kacc (bf16_gemm_kacc.cc,
# P4bk) instantiated per distinct layer GEMM shape. The kernel runs the transposed GEMM
# Y[ch][px] = W[ch,:] . X[px,:]; Maxwell's c013 kernel walks 16-row channel tiles on each core
# (px = 64 = 4 mesh cols x 16); here each core also walks px/64 16-wide pixel tiles (kplan's
# per-layer walk of 64x64 output macro tiles). Host: im2col + pre-tiling (bf16) only; NO K-sum.
KACC_CC = _tplpath("gemm/bf16_gemm_kacc.cc")


def kacc_wlayout(w6, N: int, K: int, M: int):
    """P4w: bf16 weight [O/ob][C/ib][KH][KW][ib][ob] (TVM OIHW{i}{o}) -> the kacc W port chunk layout for K
    (what uc_conv re-tiled on every call until P4w): W natural [N][K] (k = c*KH*KW + t), each 16-row block
    = 16 rows x K, chunk j (kch bf16) at j*kch of every row, in-chunk flat f = mi*8*kch + kb*64 + mm*8 + kk
    (n = 16*nb + 8*mi + mm, k = j*kch + 8*kb + kk). The Kp zero chunk (kacc_geom kp > k) is NOT stored: uc_conv
    appends it per row."""
    import numpy as np
    on, cn, kh, kw, ib, ob = w6.shape
    assert on * ob == N and cn * ib * kh * kw == K, (w6.shape, N, K)
    kch = kacc_geom(N, K, M)["kch"]
    wn = w6.transpose(0, 5, 1, 4, 2, 3).reshape(N, K)                      # [n][c][kh][kw]
    t = wn.reshape(N // 16, 2, 8, K // kch, kch // 8, 8)                    # nb mi mm j kb kk
    t = t.transpose(0, 3, 1, 4, 2, 5).reshape(N // 16, K // kch, 16, kch)   # nb j (f//kch) (f%kch)
    return np.ascontiguousarray(t.transpose(0, 2, 1, 3)).reshape(-1)       # nb row j q


def kacc_dense_wlayout(wp, N: int, K: int, pxs: int):
    """PDN: TVM dense_pack weight [N/nb][K][nb] (bf16) -> the kacc W chunk layout of instance (N, K, pxs), same
    order as kacc_wlayout / kacc_gemm.layout(wside=True): nb row j q (16-row block, chunk j of kch bf16)."""
    on, k, ob = wp.shape
    assert on * ob == N and k == K, (wp.shape, N, K)
    kch = kacc_geom(N, K, pxs)["kch"]
    wn = wp.transpose(0, 2, 1).reshape(N, K)
    t = wn.reshape(N // 16, 2, 8, K // kch, kch // 8, 8).transpose(0, 3, 1, 4, 2, 5)
    t = t.reshape(N // 16, K // kch, 16, kch).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t).reshape(-1)


def prelayout_weights(wbin: Path, sel: list) -> int:
    """P4w: rewrite every AIE conv layer's bf16 weight in weights.bin (in place, same size) into the kacc
    chunk layout, so the board pre step is only im2col. Returns the bytes rewritten."""
    import numpy as np
    blob = np.memmap(wbin, dtype=np.uint8, mode="r+")
    tot = 0
    for c in sel:
        nel = int(np.prod(c["wshape"]))
        w = np.frombuffer(blob[c["woff"]:c["woff"] + 2 * nel].tobytes(), dtype="<u2").reshape(c["wshape"])
        if c.get("kind") == "linear":   # PDN: TVM dense_pack [N/nb][K][nb] -> natural [N][K] -> kacc chunks
            lw = kacc_dense_wlayout(w, c["N"], c["K"], c["pxs"])
        else:
            lw = kacc_wlayout(w, c["N"], c["K"], c["M"])
        blob[c["woff"]:c["woff"] + 2 * nel] = lw.view(np.uint8)
        tot += 2 * nel
    blob.flush()
    del blob
    return tot


def kacc_name(ch: int, k: int, px: int) -> str:
    return f"gemm_kacc_{ch}x{k}x{px}"


def kacc_geom(ch: int, k: int, px: int) -> dict:
    """walk of one layer on the 4x4 mesh (aiehlc's trace fitter takes a 2-level affine walk per port):
    core (pm, pn) owns ch/4 channels (ntile tiles of 16) x px/4 pixels (S sub-tiles of 16);
      a (W)  [ch][k/2]  block {16, tkw}      rounds (t, j)  -- Maxwell's c013 walk
      b (X)  [px][k/2]  block {px/4, tkw}    rounds (t, j), step_i 0 -- the core's whole pixel range per chunk
      c (Y)  [ch][px]   block {16, px/4} fp32  rounds (t) -- the core's whole pixel range, held over the K walk
    P4bx: the c window IS the fp32 accumulator (S 16x16 tiles, [s][mi][ni][8x8] order; the host un-tiles it).
    A {16, 16} c block with a {px/4, tkw} b block made aiehlc take tileN=16 < tileCols=px/4 (the C port's d2
    overrides B's) and emit the N-sub-tiled walk: A re-sent nRounds=px/64 times per K walk, the core takes it
    once -> the px=1024 layer hung (P4bq probe_0). With c = {16, px/4}, tileN == tileCols: the "no N sub-tiling"
    walk of the board-proven px=64 layers (P4bq probe_16). K chunk kch = 2*tkw bf16, tkw = min(32, 4096/px)
    so the b block stays <= 4 KiB. px == 64: Maxwell's c013 walk."""
    assert ch % 64 == 0 and px % 64 == 0 and k % 64 == 0, (ch, k, px)
    tkw = min(32, 4096 // px)
    # aiehlc's port-buffer region is 29696 B (trace_routing kPortBufferBytes) and an output window is always
    # ping-pong: px=1024 (c block {16,256} fp32 = 16 KiB) cannot fit. Those layers run TRANSPOSED (tr=1):
    # a = X [px][k/2] block {16, tkw} walked over px tiles, b = W [ch][k/2] block {ch/4, tkw}, c = Y^T [px][ch]
    # block {16, ch/4}; same kernel walk, same host pre-tiling (W stays the mmul's left operand).
    tr = 2 * (16 * tkw * 4 + (px // 4) * tkw * 4 + 16 * (px // 4) * 4) > 29696
    if tr:
        tkw = 32
        while (ch // 4) * tkw > 1024 or (k // 2) % tkw:
            tkw //= 2
    assert tkw >= 4 and (k // 2) % tkw == 0, (ch, k, px, tkw)
    # K padded by one zero chunk when the W port would be square (ch == k/2 words): aiehlc's GemmSpace
    # derivation then mis-sizes the A window (gemm_kacc_1280x2560x64: 24 elements instead of 512)
    kp = k + 2 * tkw if ch == k // 2 else k
    rows, cols = (px, ch) if tr else (ch, px)   # a-port rows (walked by t) / b-port rows (whole range per core)
    l1 = 2 * (16 * tkw * 4 + (cols // 4) * tkw * 4 + 16 * (cols // 4) * 4)
    assert l1 <= 29696, (ch, k, px, l1)
    return dict(ntile=rows // 64, S=cols // 64, tkw=tkw, kch=2 * tkw, nkc=(kp // 2) // tkw, kp=kp, tr=int(tr),
                arows=rows, brows=cols, l1=l1)


KACC_KERNEL = _tpl("gemm/gemm_kacc_kernel.cc")


def kacc_kernel_text(ch: int, k: int, px: int) -> str:
    g = kacc_geom(ch, k, px)
    t = KACC_KERNEL
    for key, v in (("@NAME@", kacc_name(ch, k, px)), ("@SID@", f"{ch}x{k}x{px}"), ("@CH@", ch), ("@KW@", g["kp"] // 2), ("@PX@", px), ("@K@", k),
                   ("@TKW@", g["tkw"]), ("@KCH@", g["kch"]), ("@NKC@", g["nkc"]), ("@NT@", g["ntile"]),
                   ("@S@", g["S"]), ("@AROWS@", g["arows"]), ("@BROWS@", g["brows"]), ("@TR@", g["tr"]),
                   ("@LHS@", "(const uint16_t *)w + s * 16 * kch" if g["tr"] else "x"),
                   ("@RHS@", "x" if g["tr"] else "(const uint16_t *)w + s * 16 * kch")):
        t = t.replace(key, str(v))
    assert "@" not in t
    return t


def kacc_host(shapes: list, lin: list = (), att: list = ()) -> str:
    """host part of the --kincore aiek source: uc_conv with ONE gemm_kacc launch per layer (no host K-sum);
    lin (PDN): (N, K, Mp, pxs) of the linear layers -- buffers hold the whole GEMM (all S px launches)"""
    sw = "\n".join("    if (N == %d && K == %d && M == %d) {\n        %s<<<mesh>>>(%s, ky);\n        return 0;\n    }"
                   % (ch, k, px, kacc_name(ch, k, px), "kx, kw" if kacc_geom(ch, k, px)["tr"] else "kw, kx")
                   for ch, k, px in shapes)
    return KACC_HOST.replace("@DISPATCH@", sw) \
        .replace("@MAXW@", str(max([sh[0] * kacc_geom(*sh)["kp"] for sh in shapes] +
                                   [n * kacc_geom(n, k, p)["kp"] for n, k, mp, p in lin] +
                                   [g["CHP"] * g["kp"] for g in att]))) \
        .replace("@MAXX@", str(max([sh[2] * kacc_geom(*sh)["kp"] for sh in shapes] +
                                   [mp * kacc_geom(n, k, p)["kp"] for n, k, mp, p in lin] +
                                   [g["PXP"] * g["kp"] for g in att]))) \
        .replace("@MAXY@", str(max([ch * px for ch, k, px in shapes] + [n * mp for n, k, mp, p in lin] +
                                   [g["CHP"] * g["PXP"] for g in att])))


KACC_HOST = (_tpl("conv2d/kacc_conv_host_head.cc") + _tpl("layout/neon_tr4_f32.cc") + _tpl("conv2d/kacc_conv_host.cc"))

LINEAR_HOST = _tpl("gemm/kacc_linear_host.cc")


ATTN_HOST = _tpl("attention/kacc_bmm_host.cc")


def write_kernel_kacc(kdst: Path, spec: str, shapes: list, warm: bool = False, ctrlpkt: str = "off",
                      lin: list = (), att: list = (), fa: list = ()) -> Path:
    """--kincore: aiek_unetconv.cc = one gemm_kacc instance per distinct (ch, K, px) + host unet_aie_conv;
    lin (PDN): (N, K, Mp, pxs) per linear layer -> + host unet_aie_linear"""
    spec = spec.replace('"out ', '"export    unet_aie_conv\\n"\n        "export    unet_aie_stats\\n"\n' +
                        ('        "export    unet_aie_linear\\n"\n' if lin else '') +
                        ('        "export    unet_aie_attn\\n"\n' if att else '') +
                        ('        "export    unet_aie_attn_kl\\n"\n' if fa else '') +  # P4a keep-layout
                        '        "export    unet_aie_launch_stats\\n"\n'
                        '        "export    unet_aie_host_split_site\\n"\n'
                        '        "export    unet_aie_host_split_line\\n"\n'
                        '        "export    unet_aie_split_stats\\n"\n        "out ', 1)
    assert "export    unet_aie_conv" in spec, spec
    kdst.mkdir(parents=True, exist_ok=True)
    kcc = kdst / f"{ENTRY}.cc"
    head = (_tpl("gemm/kacc_unit_head.cc").replace("@E@", ENTRY).replace("@SRC@", str(KACC_CC))
            .replace("@NSHAPES@", str(len(shapes))))
    if ctrlpkt == "aot":
        head = CTRLPKT_PRAGMAS + head
    ktext = "\n\n".join(kacc_kernel_text(*s) for s in shapes)
    if fa:  # GFA (--attn-fa): flash-attention prologue (static helpers) before the kernels, one kernel per class
        import fa_attn
        head = head + fa_attn.fa_prologue(fa) + "\n"
        ktext = ktext + "\n\n" + "\n\n".join(fa_attn.fa_kernel_text(c) for c in fa)
    host = kt_select(HOST_CC, False)
    # replace the 64x64-tile uc_conv (host K-sum) by the one-launch version up to its epilogue
    i0 = host.index("static int uc_conv(")
    i1 = host.index('extern "C" int unet_aie_conv(')  # GPOOL: epilogue fused into the kacc scatter
    host = host[:i0] + (kacc_host(shapes, lin, [c["geo"] for c in att]) if att and not fa else kacc_host(shapes, lin)) + host[i1:]
    if lin:  # PDN: unet_aie_linear before unet_aie_conv
        j = host.index('extern "C" int unet_aie_conv(int cid')
        host = host[:j] + LINEAR_HOST.lstrip("\n") + host[j:]
    if att:  # GATT: unet_aie_attn before unet_aie_conv
        j = host.index('extern "C" int unet_aie_conv(int cid')
        if fa:
            import fa_attn
            host = host[:j] + fa_attn.fa_host(fa).lstrip("\n") + host[j:]
        else:
            host = host[:j] + ATTN_HOST.lstrip("\n") + host[j:]
    # drop the 64x64-port state the one-launch path does not use (no unused-static warnings)
    a0, a1 = host.index("static int uc_alloc(void) {"), host.index("static long uc_numel(")
    host = host[:a0] + host[a1:]
    for dead in ("static float g_uc_acc[UC_MAX_ACC] __attribute__((aligned(64))); // .bss, never malloc\n",
                 "static int32_t *g_uc_a = nullptr; // port a: bf16 [64][128]\n",
                 "static int32_t *g_uc_b = nullptr; // port b: bf16 [UC_BR][128]\n",
                 "static int32_t *g_uc_c = nullptr; // port c: f32  [64][UC_CW]\n"):
        assert host.count(dead) == 1, dead
        host = host.replace(dead, "")
    assert "gemm_bf16<<<" not in host and "g_uc_acc[(size_t)(mb" not in host
    kcc.write_text(head + ktext + "\n" + host.replace("@MAXACC@", "1").replace("@WARM@", "1" if warm else "0")
                   .replace("@CTRLPKT@", "1" if ctrlpkt == "aot" else "0")
                   .replace("@PART_C0@", "2" if ctrlpkt == "aot" else "3")
                   .replace("@E@", ENTRY).replace("@SPEC@", spec))
    (kdst / f"{ENTRY}_ref.c").write_text(REF_C.replace("@E@", ENTRY))
    return kcc


# S6 (unet-m3): the driver template -- the C glue aiegraph places in graph_driver.c (emod.driver_template).
# Same text, same anchors as the old patch_driver residue: declarations + PROF glue before kernel_ws,
# graph_bf16_expand timed into the following site, graph_run resets the pending expand time.
EXPAND_ENTER = "    XTime e0_, e1_;\n    XTime_GetTime(&e0_);\n"
EXPAND_EXIT = "    XTime_GetTime(&e1_);\n    up_pending_exp += e1_ - e0_;\n    up_exp_calls++;\n"
RUN_ENTER = "    up_pending_exp = 0; /* P1 PROF: expands before graph_run are not graph_run time */\n"


def driver_template(att: list) -> dict:
    """``emod.driver_template(**driver_template(att))``: conv / linear / FA steps are aiegraph group
    steps (their own declarations come from aiegraph); only the FA kernel declarations and the PROF
    glue are template text. *att* must be the GFA (``fa``) attention layers, or empty."""
    if att and not all(c.get("fa") is not None and c.get("fa_steps") for c in att):
        raise ValueError("driver_template: attention layers must be aiegraph FA group steps (--attn-fa)")
    if att:
        import fa_attn
        decls = fa_attn.DECL_FA + fa_attn.DECL_KL + prof_c_attn() + "\n"
    else:
        decls = PROF_C + "\n"
    return dict(decls=decls, expand_enter=EXPAND_ENTER, expand_exit=EXPAND_EXIT, run_enter=RUN_ENTER)
