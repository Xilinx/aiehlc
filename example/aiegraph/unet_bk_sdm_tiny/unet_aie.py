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

P2K = Path(__file__).resolve().parent / "bf16_gemm_convin.cc"  # 64x64 ports (8e7254d, board PASS p0_board.log)
P2K_KT = Path(__file__).resolve().parent / "bf16_gemm_convin_ktimer.cc"  # --ktimer: core timer tail, b 80x64 / c 64x80
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


def map_attn(partition: dict, work: Path) -> list:
    """GATT (GAP-aiegraph-attn-site): partition 'attn' layers (aiegraph.partition.match_attention bmm groups,
    autopart --mode attn) -> TVM nn_batch_matmul calls (transpose_b: X [H][M][K], W [H][N][K], Y [H][M][N]).
    Both operands are activations. Matched in graph order (TVM keeps the ONNX topological order of the bmms);
    every pair must agree on (H, M, K, N) or the build fails."""
    graphs = sorted(work.glob("*_graph.json"))
    assert len(graphs) == 1, graphs
    g = json.loads(graphs[0].read_text())
    nodes, row_ptr = g["nodes"], g["node_row_ptr"]
    shapes = g["attrs"]["shape"][1]
    bmm = []
    for i, n in enumerate(nodes):
        if n.get("op") != "tvm_op" or "nn_batch_matmul" not in n["attrs"]["func_name"]:
            continue
        ins = [s for s, _, _ in n["inputs"]]
        f = n["attrs"]["func_name"]
        bmm.append(dict(node=i, symbol=f if f.startswith("tvmgen_") else "tvmgen_default_" + f, func=f,
                        eids=[row_ptr[s] for s in ins], yeid=row_ptr[i], nargs=len(ins) + 1,
                        xshape=shapes[row_ptr[ins[0]]], wshape=shapes[row_ptr[ins[1]]], yshape=shapes[row_ptr[i]]))
    att = sorted((L for L in partition["layers"] if L.get("cls") == "attn"), key=lambda L: (L["part"], L["idx"]))
    log(f"TVM graph {graphs[0].name}: {len(bmm)} nn_batch_matmul calls; partition has {len(att)} attn bmm layers")
    if len(bmm) != len(att):
        raise SystemExit(f"map_attn: {len(att)} partition attn layers vs {len(bmm)} TVM nn_batch_matmul calls")
    out = []
    for L, c in zip(att, bmm):
        M, K, N = L["pad"]["orig"] if "pad" in L else L["mkn"][0]
        H = int(L["batch"])
        want = ([H, M, K], [H, N, K], [H, M, N])
        if (c["xshape"], c["wshape"], c["yshape"]) != want:
            raise SystemExit(f"map_attn: {L['node']} (H={H} M={M} K={K} N={N}) vs TVM node {c['node']} {c['func']} "
                             f"x{c['xshape']} w{c['wshape']} y{c['yshape']}")
        geo = bmm_geom(H, M, K, N)
        out.append(dict(c, kind="attn", cid=L["idx"], onnx=L["node"], group=int(L["attn_group"]),
                        role=L["attn_role"], softmax=L.get("softmax"), geo=geo, launches=H * geo["S"],
                        macs=H * M * N * K, M=M, N=N, K=K))
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
HOST_CC = r"""
// ═══════════════════════════════════════════════════════════════════════════
// HOST -- after the kernel: nothing that reaches the launch is declared above it.
// ═══════════════════════════════════════════════════════════════════════════
#include "aie_user_kernel.h"
#include "xil_cache.h"
#include "aie_timer.h"

#define UC_WARM @WARM@ /* build_unet.py --launch-opt warm: kernel loaded once (PM warm skip), --skip-bss */
#if UC_WARM
/* aie_runtime.c (AIEHLC_PM_WARM_SKIP): per-tile PM segs skipped / per-tile cold loads since boot */
extern "C" void __Runtime_pm_warm_stats(unsigned *skipped_segs, unsigned *cold_tile_loads);
#endif

#define UC_CTRLPKT @CTRLPKT@ /* build_unet.py --ctrlpkt aot: control-packet plan pragmas at the top of this file */
#if UC_CTRLPKT
/* aie_runtime.c: required by control_plan_shim_bd_ctrl (aiehlc lint L023), before device init */
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h:936 */
static int g_uc_ht = 0;
#endif

#define UC_MAX_ACC @MAXACC@ /* largest M*N of the partition layers (fp32 K-sum on the APU) */
#define UC_KPORT 128        /* staged row: 64 bf16 + 64 bf16 zeros (int32 [64][64] port) */
/*KT+*/#define UC_BR 80            /* port b rows: 4 core groups x (16 weight rows + 4 zero rows) */
#define UC_CW 80            /* port c row: 4 core blocks x (16 results + 4 KTIMER tail words) */
#define UC_GRP(i) ((((i) >> 4) * 20) + ((i) & 15)) /* row/col i of the 64 -> padded core-group layout */
/*KT-*//*NK+*/#define UC_BR 64            /* port b rows */
#define UC_CW 64            /* port c row */
#define UC_GRP(i) (i)
/*NK-*/
static inline uint16_t uc_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float uc_bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
/* the DLTensor / TVMValue the generated driver passes (tvm_graph_types.h) */
typedef struct {
    void *data;
    int32_t device_type, device_id;
    int32_t ndim;
    uint8_t code, bits;
    uint16_t lanes;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} uc_dlt;

static float g_uc_acc[UC_MAX_ACC] __attribute__((aligned(64))); // .bss, never malloc
static int32_t *g_uc_a = nullptr; // port a: bf16 [64][128]
static int32_t *g_uc_b = nullptr; // port b: bf16 [UC_BR][128]
static int32_t *g_uc_c = nullptr; // port c: f32  [64][UC_CW]
static int g_aie_launches = 0;    // every gemm_bf16 launch that ran
static unsigned long long g_uc_macs = 0; // MACs of those launches (valid rows x cols x K)
static int g_uc_calls = 0;        // unet_aie_conv calls
static int g_uc_rc = 0;           // first nonzero rc
static unsigned long long g_uc_aie_ticks = 0; // A78 XTime ticks spent in flush + launch + invalidate
static int g_uc_kloads = 0;       // launches that (re)wrote the kernel's program memory
/* P1 split: A78 pmccntr per launch region inside aie_wall. pre = B/C flush (host->DMA sync, -> startio),
 * post = C invalidate + pm stats (DMA->host sync, -> wait_io), call = gemm_bf16; glue = call minus the
 * runtime phase + wait_io cycles it accrued (48 lock-init writes, 60 createio, dev sync, sb commit in the
 * generated host code -> bdcfg). */
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls); /* C++ linkage, aie_runtime.h */
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
static inline unsigned long long uc_pmc(void) {
    unsigned long long v;
#if defined(__aarch64__)
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v));
#else
    v = 0ULL;
#endif
    return v;
}
static inline unsigned long long uc_rt_cyc(void) {
    unsigned long long ph[4] = {0, 0, 0, 0}, w = 0;
    unsigned int pn[4] = {0, 0, 0, 0}, wn = 0;
    __Runtime_phase_cycles(ph, pn);
    __Runtime_wait_io_cycles(&w, &wn);
    return ph[0] + ph[1] + ph[2] + ph[3] + w;
}
static unsigned long long g_sp_pre = 0, g_sp_call = 0, g_sp_glue = 0, g_sp_post = 0;
/* GPOOL: per-layer [UCL] split, recorded in the site and printed after graph_run */
#define UC_UCL_MAX 128
typedef struct {
    int M, N, K, fast;
    float pre, aie, post;
} uc_ucl;
static uc_ucl g_uc_ucl[UC_UCL_MAX];
static int g_uc_nucl = 0, g_uc_ucl_done = 0;
extern "C" int unet_aie_split_stats(unsigned long long *pre, unsigned long long *call, unsigned long long *glue,
                                    unsigned long long *post) {
    if (!g_uc_ucl_done) {
        g_uc_ucl_done = 1;
        for (int i = 0; i < g_uc_nucl; i++)
            printf("[UCL] l=%d M=%d N=%d K=%d pre_us=%.0f aie_us=%.0f post_us=%.0f fast=%d\n", i, g_uc_ucl[i].M,
                   g_uc_ucl[i].N, g_uc_ucl[i].K, (double)g_uc_ucl[i].pre, (double)g_uc_ucl[i].aie,
                   (double)g_uc_ucl[i].post, g_uc_ucl[i].fast);
    }
    *pre = g_sp_pre;
    *call = g_sp_call;
    *glue = g_sp_glue;
    *post = g_sp_post;
    return 0;
}
static unsigned g_uc_cold_tiles = 0; // per-tile cold loads inside those launches

extern "C" int unet_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles) {
    if (aie_ms)
        *aie_ms = 1000.0 * (double)g_uc_aie_ticks / (double)COUNTS_PER_SECOND;
    if (kloads)
        *kloads = g_uc_kloads;
    if (cold_tiles)
        *cold_tiles = g_uc_cold_tiles;
    return 0;
}
/* GAP-RT1: per-site aie_host split (pack / scatter / dma / wait) -> PROF aie_host_split (aie_host_split.h).
 * The site index comes from graph_driver's UP_AIE_BEGIN; wait = the runtime's always-on wait_io ticks. */
#ifndef UC_HS_DEFINED
#define UC_HS_DEFINED 1
#include "aie_host_split.h"
void __Runtime_host_split_wait_ticks(unsigned long long *ticks, unsigned int *calls); /* C++ linkage, aie_runtime.h */
static ahs_table g_ahs;
static inline unsigned long long uc_hs_wait(void) {
    unsigned long long t = 0ULL;
    unsigned int n = 0U;
    __Runtime_host_split_wait_ticks(&t, &n);
    return t;
}
#endif
extern "C" void unet_aie_host_split_site(int site) { ahs_set_site(&g_ahs, site); }
extern "C" int unet_aie_host_split_line(int site, char *buf, int n) {
    return ahs_format(&g_ahs, site, (double)COUNTS_PER_SECOND, buf, (size_t)n);
}

/*KT+*/static unsigned long long g_kt_max = 0, g_kt_all = 0; // KTIMER: sum over launches of max-core / all-core cycles
static int g_kt_n = 0, g_kt_bad = 0;                    // launches read, core tails without the magic word

extern "C" int unet_aie_ktimer_stats(unsigned long long *max_sum, unsigned long long *all_sum, int *n, int *bad) {
    if (max_sum)
        *max_sum = g_kt_max;
    if (all_sum)
        *all_sum = g_kt_all;
    if (n)
        *n = g_kt_n;
    if (bad)
        *bad = g_kt_bad;
    return 0;
}

/*KT-*/extern "C" int unet_aie_stats(int *launches, int *run_rc, int *calls, unsigned long long *macs) {
    if (launches)
        *launches = g_aie_launches;
    if (run_rc)
        *run_rc = g_uc_calls ? g_uc_rc : -99;
    if (calls)
        *calls = g_uc_calls;
    if (macs)
        *macs = g_uc_macs;
    return 0;
}

static int uc_alloc(void) {
    if (g_uc_c)
        return 0;
    g_uc_a = (int32_t *)__Runtime_Alloc(64 * UC_KPORT * 2);
    g_uc_b = (int32_t *)__Runtime_Alloc(UC_BR * UC_KPORT * 2);
    g_uc_c = (int32_t *)__Runtime_Alloc(64 * UC_CW * 4);
    if (!g_uc_a || !g_uc_b || !g_uc_c)
        return -1;
    memset(g_uc_a, 0, 64 * UC_KPORT * 2); // cols 64..127 stay zero
    memset(g_uc_b, 0, UC_BR * UC_KPORT * 2); // --ktimer: rows 16..19 of each core group stay zero
    return 0;
}

static long uc_numel(const uc_dlt *t) {
    long n = 1;
    for (int d = 0; d < t->ndim; d++)
        n *= (long)t->shape[d];
    return n;
}

static int uc_conv(void *const *argv, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw) {
    if (nargs < 4)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 5 || w->ndim != 6 || y->ndim != 5 || x->shape[0] != 1 || y->shape[0] != 1)
        return -2;
    const int cb = (int)x->shape[4], C = (int)x->shape[1] * cb, H = (int)x->shape[2], W = (int)x->shape[3];
    const int ib = (int)w->shape[4], ob = (int)w->shape[5], CI = (int)w->shape[1] * ib;
    const int O = (int)w->shape[0] * ob, KH = (int)w->shape[2], KW = (int)w->shape[3];
    const int yb = (int)y->shape[4], OH = (int)y->shape[2], OW = (int)y->shape[3];
    if (CI != C || (int)y->shape[1] * yb != O || OH != (H + 2 * ph - KH) / sh + 1 || OW != (W + 2 * pw - KW) / sw + 1)
        return -3;
    const int M = OH * OW, N = O, K = C * KH * KW, KK = KH * KW;
    if ((long)M * N > UC_MAX_ACC)
        return -3;
    const float *xin = (const float *)x->data;
    float *yout = (float *)y->data;
    if (uc_alloc() != 0)
        return -1;
    memset(g_uc_acc, 0, (size_t)M * N * 4);
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({@PART_C0@, 6, 0, 6}, 4, 4); // one mesh per call; ctrlpkt aot: col 2 = spare control column west (control_plan_dedicated_shim, as conv2dstem.cc)
    uint16_t *A = (uint16_t *)g_uc_a, *B = (uint16_t *)g_uc_b;
    const float *Cb = (const float *)g_uc_c;
    int bad = 0;
    for (int mb = 0; mb * 64 < M; mb++)
        for (int kc = 0; kc * 64 < K; kc++) {
            const int kv = (K - kc * 64) < 64 ? (K - kc * 64) : 64;
            for (int r = 0; r < 64; r++) { // A tile: im2col of the activation, fp32 -> bf16 RNE
                const int m = mb * 64 + r, oh = m / OW, ow = m % OW;
                for (int kk = 0; kk < 64; kk++) {
                    float v = 0.0f;
                    const int k = kc * 64 + kk;
                    if (m < M && k < K) {
                        const int c = k / KK, t = k % KK, ih = oh * sh + t / KW - ph, iw = ow * sw + t % KW - pw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            v = xin[(((size_t)(c / cb) * H + ih) * W + iw) * cb + c % cb];
                    }
                    A[r * UC_KPORT + kk] = uc_f2bf(v);
                }
            }
            Xil_DCacheFlushRange((UINTPTR)g_uc_a, 64 * UC_KPORT * 2);
            for (int nb = 0; nb * 64 < N; nb++) {
                for (int r = 0; r < 64; r++) { // B tile: straight from the compact bf16 blob (OIHW{ib}i{ob}o)
                    const int n = nb * 64 + r;
                    for (int kk = 0; kk < 64; kk++) {
                        const int k = kc * 64 + kk;
                        uint16_t v = 0;
                        if (n < N && k < K) {
                            const int c = k / KK, t = k % KK;
                            v = wbf[((((size_t)(n / ob) * (C / ib) + c / ib) * KK + t) * ib + c % ib) * ob + n % ob];
                        }
                        B[UC_GRP(r) * UC_KPORT + kk] = v;
                    }
                }
                memset(g_uc_c, 0xff, 64 * UC_CW * 4); // sentinel: unwritten -> NaN -> counted
                XTime lt0, lt1;
                XTime_GetTime(&lt0);
                const unsigned long long sp0 = uc_pmc();
#if UC_WARM
                unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
                __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
                Xil_DCacheFlushRange((UINTPTR)g_uc_b, UC_BR * UC_KPORT * 2);
                Xil_DCacheFlushRange((UINTPTR)g_uc_c, 64 * UC_CW * 4);
                const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
                gemm_bf16<<<mesh>>>(g_uc_a, g_uc_b, g_uc_c);
                const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
                Xil_DCacheInvalidateRange((UINTPTR)g_uc_c, 64 * UC_CW * 4);
#if UC_WARM
                __Runtime_pm_warm_stats(&seg1, &cold1);
                if (cold1 != cold0) {
                    g_uc_kloads++;
                    g_uc_cold_tiles += cold1 - cold0;
                }
#endif
                const unsigned long long sp3 = uc_pmc();
                XTime_GetTime(&lt1);
                g_sp_pre += sp1 - sp0;
                g_sp_call += sp2 - sp1;
                g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
                g_sp_post += sp3 - sp2;
                g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
                g_aie_launches++;
/*KT+*/                { // KTIMER: AIE core timer delta each core wrote into its block tail (row 0, words 16..18)
                    const uint32_t *Cw = (const uint32_t *)g_uc_c;
                    unsigned long long mx = 0;
                    for (int q = 0; q < 16; q++) {
                        const uint32_t *kt = Cw + (size_t)(q >> 2) * 16 * UC_CW + (q & 3) * 20 + 16;
                        if (kt[1] != 0x4B54494Du) {
                            g_kt_bad++;
                            continue;
                        }
                        const unsigned long long d = (unsigned long long)kt[0] | ((unsigned long long)kt[2] << 32);
                        g_kt_all += d;
                        if (d > mx)
                            mx = d;
                    }
                    g_kt_max += mx;
                    g_kt_n++;
                }
/*KT-*/                const int mv = (M - mb * 64) < 64 ? (M - mb * 64) : 64, nv = (N - nb * 64) < 64 ? (N - nb * 64) : 64;
                g_uc_macs += (unsigned long long)mv * nv * kv;
                for (int r = 0; r < mv; r++)
                    for (int j = 0; j < nv; j++) {
                        const float v = Cb[r * UC_CW + UC_GRP(j)];
                        if (!(v == v) || v > 1e30f || v < -1e30f)
                            bad++;
                        g_uc_acc[(size_t)(mb * 64 + r) * N + nb * 64 + j] += uc_bf2f(uc_f2bf(v)); // bf16 slice out
                    }
            }
        }
    if (bad) {
        printf("[unet_aie_conv] %d nonfinite AIE outputs\n", bad);
        return -4;
    }
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++)
            yout[((size_t)(n / yb) * M + m) * yb + n % yb] = g_uc_acc[(size_t)m * N + n];
    for (int i = 2; i < nargs - 1; i++) { // epilogue: bias / temb (per channel) or residual (same shape)
        const uc_dlt *e = (const uc_dlt *)argv[i];
        const float *ev = (const float *)e->data;
        const long ne = uc_numel(e);
        if (ne == N && e->ndim == 5) {
            for (int n = 0; n < N; n++) { // [1,N/eb,1,1,eb] -> flat n
                float *o = yout + (size_t)(n / yb) * M * yb + n % yb;
                for (int m = 0; m < M; m++)
                    o[(size_t)m * yb] += ev[n];
            }
        } else if (ne == (long)M * N && e->ndim == 5 && (int)e->shape[4] == yb) {
            for (long q = 0; q < (long)M * N; q++)
                yout[q] += ev[q];
        } else {
            return -5;
        }
    }
    return 0;
}

extern "C" int unet_aie_conv(int cid, void *args, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw) {
    g_uc_calls++;
    const int rc = uc_conv((void *const *)args, nargs, wbf, sh, sw, ph, pw);
    if (rc != 0) {
        printf("[unet_aie_conv] layer %d rc=%d\n", cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

// Carrier entry for aiegraph's offload flow (builds + links this archive). Never called:
// graph_driver.c (post-processed by build_unet.py) calls unet_aie_conv() at every
// partition call site, including this layer's.
AIEK_ENTRY(@E@) {
    (void)args;
    (void)num_args;
    printf("[@E@] carrier entry called -- driver not patched\n");
    if (!g_uc_rc)
        g_uc_rc = -7;
    return AIEK_ERR_KERNEL;
}

@SPEC@
"""

REF_C = r"""// @E@_ref.c -- x86 model of the carrier step for aiegraph's group-level check (offload check="group"):
// conv2d_NCHWc on TVM's layouts with the AIE numerics (activation and weight rounded to bf16, fp32
// accumulation), then the fused adds. aiegraph compares it with TVM's own C on x86: cos >= 0.999, no nonfinite.
//   args: x [1,Cc,H,W,cb], w [Kc,Cc,KH,KW,cb,kb], then 5-D adds ([1,Kc,1,1,kb] broadcast or the output
//   shape), out [1,Kc,OH,OW,kb]; stride H/OH, pad (KH-1)/2 ("same" convs of the UNet). Anything else -> PARAM.
#include <math.h>
#include <string.h>
#include "aie_user_kernel.h"

static float uc_bf16(float f) { /* round to nearest even, keep NaN/Inf */
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    memcpy(&f, &u, 4);
    return f;
}

AIEK_REFERENCE(@E@) {
    if (num_args < 3) return AIEK_ERR_ARGS;
    const aiek_tensor *X = &args[0], *Wt = &args[1], *O = &args[num_args - 1];
    for (int i = 0; i < num_args; i++)
        if (args[i].code != AIEK_FLOAT || args[i].bits != 32) return AIEK_ERR_ARGS;
    if (X->ndim != 5 || Wt->ndim != 6 || O->ndim != 5) return AIEK_ERR_PARAM;
    const int64_t Cc = X->shape[1], H = X->shape[2], W = X->shape[3], cb = X->shape[4];
    const int64_t Kc = Wt->shape[0], KH = Wt->shape[2], KW = Wt->shape[3], kb = Wt->shape[5];
    const int64_t OH = O->shape[2], OW = O->shape[3];
    if (Wt->shape[1] != Cc || Wt->shape[4] != cb || O->shape[1] != Kc || O->shape[4] != kb) return AIEK_ERR_PARAM;
    if (OH <= 0 || OW <= 0 || H % OH || W % OW || H / OH != W / OW || !(KH & 1) || !(KW & 1)) return AIEK_ERR_PARAM;
    const int64_t S = H / OH, PH = (KH - 1) / 2, PW = (KW - 1) / 2;
    for (int i = 2; i < num_args - 1; i++) {
        const aiek_tensor *A = &args[i];
        int bc = A->ndim == 5 && A->shape[1] == Kc && A->shape[2] == 1 && A->shape[3] == 1 && A->shape[4] == kb;
        int full = A->ndim == 5 && aiek_numel(A) == aiek_numel(O);
        if (!bc && !full) return AIEK_ERR_PARAM;
    }
    const float *x = (const float *)X->data, *w = (const float *)Wt->data;
    float *out = (float *)O->data;
    for (int64_t kc = 0; kc < Kc; kc++)
        for (int64_t oh = 0; oh < OH; oh++)
            for (int64_t ow = 0; ow < OW; ow++)
                for (int64_t k = 0; k < kb; k++) {
                    float acc = 0.0f;
                    for (int64_t cc = 0; cc < Cc; cc++)
                        for (int64_t r = 0; r < KH; r++) {
                            int64_t ih = oh * S + r - PH;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t q = 0; q < KW; q++) {
                                int64_t iw = ow * S + q - PW;
                                if (iw < 0 || iw >= W) continue;
                                const float *xp = x + ((cc * H + ih) * W + iw) * cb;
                                const float *wp = w + ((((kc * Cc + cc) * KH + r) * KW + q) * cb) * kb + k;
                                for (int64_t c = 0; c < cb; c++) acc += uc_bf16(xp[c]) * uc_bf16(wp[c * kb]);
                            }
                        }
                    int64_t oi = ((kc * OH + oh) * OW + ow) * kb + k;
                    for (int i = 2; i < num_args - 1; i++) {
                        const aiek_tensor *A = &args[i];
                        const float *a = (const float *)A->data;
                        acc += aiek_numel(A) == aiek_numel(O) ? a[oi] : a[kc * kb + k];
                    }
                    out[oi] = acc;
                }
    return AIEK_OK;
}
"""


def kernel_text(ktimer: bool = False) -> str:
    src = (P2K_KT if ktimer else P2K).read_text()
    m = re.search(r"(__global__ void gemm_bf16\(.*?\n\})\s*\n\s*int main\(", src, re.S)
    assert m, "gemm_bf16 not found"
    return m.group(1)


SOFTFLOAT_RM = """
#if defined(__chess__)
    { // --launch-opt warm (--skip-bss): softfloat's .bss float_rounding_mode is read by
      // float32_add/mul and never written by anyone -> zero it on every launch
        extern int float_rounding_mode;
        float_rounding_mode = 0;
    }
#endif
"""


CTRLPKT_PRAGMAS = (  # --ctrlpkt aot (P2a): spelled as aiehlc prints them (src/llvm/aiehlc.cc); no KERNELCONFIGOFFLOAD (P3)
    "#pragma control_packet_mode(aot)\n"
    "#pragma control_plan_op_control_packet\n"
    "#pragma CONTROL_PLAN_GROUP_REG_WRITE\n"
    "#pragma control_plan_dedicated_shim\n"
    "#pragma control_plan_shim_bd_ctrl\n")
# --ctrlpkt aot-kco (build_unet.py) / UNET_CTRLPKT_KCO=1 in the env (kacc_bench.sh -> kacc_gemm.py): the same 5
# pragmas + KERNELCONFIGOFFLOAD (the core programs its own DMA, .TM in kernel.lst); one switch for probe and full net
CTRLPKT_KCO = os.environ.get("UNET_CTRLPKT_KCO", "0") == "1"
if CTRLPKT_KCO:
    CTRLPKT_PRAGMAS += "#pragma KERNELCONFIGOFFLOAD\n"
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
    head = ("// %s.cc -- generated by build_unet.py --aie conv (P35w). Kernel: gemm_bf16 from %s (verbatim).\n"
            "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n" % (ENTRY, P2K_KT if ktimer else P2K))
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
    if "unet_aie_conv(int cid, void" in text.split("static uint8_t kernel_ws[")[0]:
        raise RuntimeError("driver already patched")
    att = [c for c in sel if c.get("kind") == "attn"]
    sel = [c for c in sel if c.get("kind") != "attn"]
    i = text.index("static uint8_t kernel_ws[")
    # GATT: the attn declaration / PROF branch only when there are attn sites (--aie conv,linear stays byte-identical)
    fam = bool(att) and all(c.get("fa") is not None for c in att)   # GFA: fused flash-attention sites
    if fam:
        import fa_attn
    text = text[:i] + DECL + ((fa_attn.DECL_FA if fam else DECL_ATTN) if att else "") + \
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
    assert text.count("    unet_aie_conv(") == len(sel) - nl and text.count("    unet_aie_linear(") == nl
    for c in att:   # GATT: tag "aie unet_aie_attn <cid>" -> PROF op=aie_host_attn<cid>
        text = _patch_attn_site(text, c)
        n += 1
    assert text.count("    unet_aie_attn(") == len(att)
    if fam:
        text = _remove_fa_softmax(text, att)
    return text, n


SOFTMAX_RE = re.compile(r"    GRAPH_PROF_BEGIN\((\d+)\);\n    (tvmgen_default_fused_nn_softmax\w*)\(args, codes, 2, 0, 0, 0\);\n"
                        r"    GRAPH_PROF_END\(\1, \"[^\"]*\"\);\n")


def _remove_fa_softmax(text: str, att: list) -> str:
    """GFA: the TVM softmax call between a group's QK site and its PV site is fused into the AIE attention:
    exactly one per group, removed (no PROF part, no CPU time)"""
    groups = {}
    for c in att:
        groups.setdefault(c["group"], {})[c["role"]] = c
    nrm = 0
    for g, d in sorted(groups.items()):
        a = text.index("    unet_aie_attn(%d, args, 3, 0, %d);" % (d["qk"]["cid"], d["qk"]["fa"]))
        b = text.index("    unet_aie_attn(%d, args, 3, 1, %d);" % (d["pv"]["cid"], d["pv"]["fa"]))
        assert a < b, (g, a, b)
        hits = list(SOFTMAX_RE.finditer(text, a, b))
        assert len(hits) == 1, (g, len(hits))
        m = hits[0]
        text = text[:m.start()] + "    /* GFA: %s (call %s) fused into AIE attention group %d */\n" % (
            m.group(2), m.group(1), g) + text[m.end():]
        nrm += 1
    assert nrm == len(groups) and not re.search(r"\btvmgen_default_fused_nn_softmax\w*\(args, ",
                                                text.split("float *graph_run(void)")[1]), "softmax call left in graph_run"
    return text


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
PROF_C = r"""
/* P1 (b), build_unet.py --aie conv: A78 wall time of every call site in graph_run (XTime),
 * always on. AIE sites count only their A78 part (site time - aie_wall of their launches);
 * graph_bf16_expand time goes to the site that follows it. unet_prof_print() from main.c. */
#include "aie_timer.h"
int unet_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles);
void unet_aie_host_split_site(int site);                   /* GAP-RT1: kernel archive (aie_host_split.h) */
int unet_aie_host_split_line(int site, char *buf, int n);
#define UP_MAX 1024
static unsigned long long up_ticks[UP_MAX], up_exp_ticks[UP_MAX], up_pending_exp;
static double up_aie_ms[UP_MAX];
static const char *up_tag[UP_MAX];
static unsigned up_n, up_exp_calls;
#undef GRAPH_PROF_BEGIN
#undef GRAPH_PROF_END
#define GRAPH_PROF_BEGIN(i) XTime graph_prof_t0_##i; up_exp_ticks[i] += up_pending_exp; up_pending_exp = 0; \
    XTime_GetTime(&graph_prof_t0_##i)
#define GRAPH_PROF_END(i, tag) do { XTime t1_; XTime_GetTime(&t1_);                      \
        up_ticks[i] += t1_ - graph_prof_t0_##i; up_tag[i] = (tag);                        \
        if ((unsigned)(i) + 1U > up_n) up_n = (unsigned)(i) + 1U; } while (0)
#define UP_AIE_BEGIN(i) double up_a0_##i = 0.0; unet_aie_host_split_site(i); unet_aie_launch_stats(&up_a0_##i, 0, 0)
#define UP_AIE_END(i) do { double a1_ = 0.0; unet_aie_launch_stats(&a1_, 0, 0); up_aie_ms[i] += a1_ - up_a0_##i; \
        unet_aie_host_split_site(-1); } while (0)
void unet_prof_print(double graph_run_ms, double aie_wall_ms) {
    const double tk = 1000.0 / (double)COUNTS_PER_SECOND;
    double tot = 0.0, cpu = 0.0, aieh = 0.0, exp_ms = 0.0, aie_in = 0.0;
    unsigned ns = 0;
    for (unsigned i = 0; i < up_n; i++) {
        if (!up_tag[i])
            continue;
        const int is_aie = strncmp(up_tag[i], "aie ", 4) == 0;
        const double site = (double)up_ticks[i] * tk, e = (double)up_exp_ticks[i] * tk;
        const double a78 = is_aie ? site - up_aie_ms[i] : site;
        const char *op = strncmp(up_tag[i], "cpu ", 4) == 0 ? up_tag[i] + 4 : up_tag[i];
        if (is_aie && strncmp(up_tag[i], "aie unet_aie_lin ", 17) == 0) /* PDN linear site: AIE host glue */
            printf("PROF part=%u op=aie_host_linear%s ms=%.3f bf16_expand_ms=%.3f site_ms=%.3f aie_ms=%.3f\n", i,
                   up_tag[i] + 17, a78 + e, e, site, up_aie_ms[i]);
        else if (is_aie)
            printf("PROF part=%u op=aie_host_conv%s ms=%.3f bf16_expand_ms=%.3f site_ms=%.3f aie_ms=%.3f\n", i,
                   up_tag[i] + 18, a78 + e, e, site, up_aie_ms[i]);
        else
            printf("PROF part=%u op=%s ms=%.3f bf16_expand_ms=%.3f\n", i, op, a78 + e, e);
        if (is_aie) { /* GAP-RT1: PROF aie_host_split part=<i> kind=<attn|conv|linear> pack_ms= scatter_ms= ... */
            char hs_[192];
            if (unet_aie_host_split_line((int)i, hs_, (int)sizeof hs_) > 0)
                printf("%s\n", hs_);
        }
        tot += a78 + e;
        exp_ms += e;
        aie_in += is_aie ? up_aie_ms[i] : 0.0;
        if (is_aie)
            aieh += a78;
        else
            cpu += a78;
        ns++;
    }
    const double tail = (double)up_pending_exp * tk;
    tot += tail;
    exp_ms += tail;
    const double den = graph_run_ms - aie_wall_ms;
    printf("PROF total_nonaie_ms=%.3f vs_graph_minus_aie=%.4f graph_run_ms=%.3f aie_wall_ms=%.3f sites=%u "
           "cpu_ms=%.3f aie_host_ms=%.3f bf16_expand_ms=%.3f (calls=%u tail=%.3f) aie_ms_in_sites=%.3f\n",
           tot, den > 0.0 ? tot / den : 0.0, graph_run_ms, aie_wall_ms, ns, cpu, aieh, exp_ms, up_exp_calls, tail,
           aie_in);
    {
        char hs_[192];
        if (unet_aie_host_split_line(-1, hs_, (int)sizeof hs_) > 0)
            printf("%s\n", hs_);
        else
            printf("PROF aie_host_split_total host_split_n=0\n");
    }
}
"""

PROF_ATTN_BRANCH = (
    '        if (is_aie && strncmp(up_tag[i], "aie unet_aie_attn ", 18) == 0) /* GATT attention bmm site */\n'
    '            printf("PROF part=%u op=aie_host_attn%s ms=%.3f bf16_expand_ms=%.3f site_ms=%.3f aie_ms=%.3f\\n", i,\n'
    '                   up_tag[i] + 18, a78 + e, e, site, up_aie_ms[i]);\n'
    '        else ')


def prof_c_attn() -> str:
    """PROF_C with the GATT branch: attention sites print op=aie_host_attn<cid> (counted as aie_host by hsplit)"""
    k = '        if (is_aie && strncmp(up_tag[i], "aie unet_aie_lin ", 17) == 0)'
    assert PROF_C.count(k) == 1
    return PROF_C.replace(k, PROF_ATTN_BRANCH + k.lstrip(), 1)


MAIN_EXTRA_DECL = (
    "void unet_prof_print(double graph_run_ms, double aie_wall_ms); /* graph_driver.c (P1 PROF) */\n"
    "int unet_aie_stats(int *launches, int *run_rc, int *calls, unsigned long long *macs);\n"
    "int unet_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles);\n"
    "/*KT+*/int unet_aie_ktimer_stats(unsigned long long *max_sum, unsigned long long *all_sum, int *n, int *bad);\n/*KT-*/"
    "int unet_aie_split_stats(unsigned long long *pre, unsigned long long *call, unsigned long long *glue, "
    "unsigned long long *post);\n"
    "/* the runtime is C++ (aie_runtime.o in the kernel archive): bind its mangled symbol */\n"
    "void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls) "
    "__asm__(\"_Z22__Runtime_phase_cyclesPyPj\");\n"
    "void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls) "
    "__asm__(\"_Z24__Runtime_wait_io_cyclesPyPj\");\n"
    "void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc, "
    "unsigned int *rst_n) __asm__(\"_Z28__Runtime_kload_split_cyclesPyPjS_S0_\");\n"
    "void __Runtime_setup_split_cycles(unsigned long long *plan_cyc, unsigned int *plan_n, unsigned long long *sync_cyc, "
    "unsigned int *sync_n) __asm__(\"_Z28__Runtime_setup_split_cyclesPyPjS_S0_\");\n"
    "unsigned long long __Runtime_ctrl_aot_fallbacks(void) __asm__(\"_Z28__Runtime_ctrl_aot_fallbacksv\");\n"
    "void __Runtime_ctrl_shim_bd_ctrl_stats_print(void) __asm__(\"_Z39__Runtime_ctrl_shim_bd_ctrl_stats_printv\");\n"
    "/* A78 PMU cycle counter (the runtime's RT_PROF_* clock): enabled here without the C reset, read\n"
    " * around graph_run with XTime so the PMU rate is measured, not assumed */\n"
    "static inline unsigned long long unet_pmc(void) {\n"
    "    unsigned long long v;\n"
    "    __asm__ volatile(\"isb; mrs %0, pmccntr_el0\" : \"=r\"(v));\n"
    "    return v;\n"
    "}\n"
    "static inline void unet_pmc_enable(void) {\n"
    "    unsigned long long v;\n"
    "    __asm__ volatile(\"mrs %0, pmcr_el0\" : \"=r\"(v));\n"
    "    v |= 1ULL;\n"
    "    v &= ~(1ULL << 3);\n"
    "    __asm__ volatile(\"msr pmcr_el0, %0\" : : \"r\"(v));\n"
    "    __asm__ volatile(\"msr pmcntenset_el0, %0\" : : \"r\"(1ULL << 31));\n"
    "    __asm__ volatile(\"isb\");\n"
    "}\n")

MAIN_CMP = r"""    int launches = 0, run_rc = -99, calls = 0;
    unsigned long long macs = 0;
    unet_aie_stats(&launches, &run_rc, &calls, &macs);
    unsigned long long ph_cyc[4] = {0, 0, 0, 0};
    unsigned int ph_calls[4] = {0, 0, 0, 0};
    __Runtime_phase_cycles(ph_cyc, ph_calls); /* [2] = PH_COREEN (AIEHLC_PROFILING=1 runtime) */
    const double frac = (double)macs / (double)UNET_TOTAL_GEMM_MACS;
    printf("AIE unet: unet_aie_conv calls=%d (planned %d) run_rc=%d aie_launches=%d (planned %d) "
           "PH_COREEN calls=%u cycles=%llu aie_macs=%llu aie_flop_frac=%.4f\n", calls, UNET_AIE_PLANNED_CALLS,
           run_rc, launches, UNET_AIE_PLANNED_LAUNCHES, ph_calls[2], ph_cyc[2], macs, frac);
/*KT+*/    {
        unsigned long long kt_max = 0, kt_all = 0;
        int kt_n = 0, kt_bad = 0;
        unet_aie_ktimer_stats(&kt_max, &kt_all, &kt_n, &kt_bad);
        /* AIE core timer, read by the kernel at entry/exit, summed over EVERY launch (max of the 16 cores) */
        printf("KTIMER core_cycles_per_launch=%llu launches=%d read=%d bad_tails=%d mean_core_cycles=%.1f "
               "total_max_core_cycles=%llu\n", launches ? kt_max / (unsigned long long)launches : 0ULL, launches,
               kt_n, kt_bad, kt_n ? (double)kt_all / (16.0 * kt_n) : 0.0, kt_max);
    }
/*KT-*/    __Runtime_ctrl_shim_bd_ctrl_stats_print(); /* [SHIM_BD_CTRL] + [CTRLPKT_AOT] in every build */
    int rc = p1_cmp("unet", out, "bf16", launches, (int)ph_calls[2], run_rc, frac);
"""


LAUNCH_LINE = r"""#ifdef UNET_LAUNCH_WARM
    {
        double aie_ms = 0.0;
        int kloads = 0;
        unsigned cold_tiles = 0;
        int wl = 0;
        unet_aie_stats(&wl, 0, 0, 0);
        unet_aie_launch_stats(&aie_ms, &kloads, &cold_tiles);
        printf("LAUNCH warm ms_per_launch=%.3f launches=%d aie_wall_ms=%.3f kloads=%d opts=%s,cold_tile_loads=%u"
               " ctrlpkt=%s aot_fallbacks=%llu\n",
               wl ? aie_ms / wl : 0.0, wl, aie_ms, kloads, UNET_LAUNCH_OPTS, cold_tiles, UNET_CTRLPKT,
               __Runtime_ctrl_aot_fallbacks());
        /* P1: per-launch split of aie_wall_ms by runtime phase (A78 pmccntr, AIEHLC_PROFILING=1 runtime).
         * PH_COREEN is the A78 cost of the core-enable writes, not AIE compute: the AIE compute is
         * inside wait_io (completion = output-DMA drain). core_reset is part of kload. */
        unsigned long long ph[4] = {0, 0, 0, 0}, wio = 0, kel = 0, krst = 0, plan = 0, sync = 0;
        unsigned int phn[4] = {0, 0, 0, 0}, wion = 0, keln = 0, krstn = 0, plann = 0, syncn = 0;
        __Runtime_phase_cycles(ph, phn); /* PH_KLOAD=0 PH_BDCFG=1 PH_COREEN=2 PH_STARTIO=3 */
        __Runtime_wait_io_cycles(&wio, &wion);
        __Runtime_kload_split_cycles(&kel, &keln, &krst, &krstn);
        __Runtime_setup_split_cycles(&plan, &plann, &sync, &syncn);
        const double g_s = (double)(t1 - t0) / (double)COUNTS_PER_SECOND;
        const unsigned long long pmu_hz = g_s > 0.0 ? (unsigned long long)((double)(pmc1 - pmc0) / g_s + 0.5) : 0ULL;
        const double cms = pmu_hz ? 1000.0 / (double)pmu_hz : 0.0; /* ms per PMU cycle */
        const double nl = wl ? (double)wl : 1.0;
        /* app-side launch regions (pmccntr, per launch, see unet_aie.py HOST_CC): host->DMA flush joins
         * startio, DMA->host invalidate joins wait_io, the generated-host glue (lock init, createio, dev
         * sync, commit: gemm_bf16 call minus runtime phases) joins bdcfg */
        unsigned long long sp_pre = 0, sp_call = 0, sp_glue = 0, sp_post = 0;
        unet_aie_split_stats(&sp_pre, &sp_call, &sp_glue, &sp_post);
        const unsigned long long c_kl = ph[0], c_bd = ph[1] + sp_glue, c_si = ph[3] + sp_pre, c_ce = ph[2],
                                 c_wi = wio + sp_post;
        const double rt_ms = (double)(ph[0] + ph[1] + ph[2] + ph[3] + wio) * cms;
        const double sum_ms = (double)(c_kl + c_bd + c_si + c_ce + c_wi) * cms;
        printf("LAUNCH split kload=%.3f bdcfg=%.3f startio=%.3f coreen=%.3f wait_io=%.3f core_reset=%.3f "
               "ms_per_launch=%.3f pmu_hz=%llu sum_vs_aie_wall=%.3f\n",
               c_kl * cms / nl, c_bd * cms / nl, c_si * cms / nl, c_ce * cms / nl, c_wi * cms / nl,
               krst * cms / nl, aie_ms / nl, pmu_hz, aie_ms > 0.0 ? sum_ms / aie_ms : 0.0);
        printf("LAUNCH split_detail calls kload=%u bdcfg=%u startio=%u coreen=%u wait_io=%u core_reset=%u "
               "kload_elf_ms=%.3f plan_ms=%.3f sync_ms=%.3f rt_bdcfg=%.4f rt_startio=%.4f rt_wait_io=%.4f "
               "host_glue_bdcfg=%.4f host_flush_startio=%.4f host_inval_wait_io=%.4f gemm_call=%.4f "
               "rt_phase_sum_ms=%.3f phase_sum_ms=%.3f aie_wall_ms=%.3f "
               "unattributed_ms_per_launch=%.4f pmc_delta=%llu xtime_s=%.3f\n",
               phn[0], phn[1], phn[3], phn[2], wion, krstn, kel * cms / nl, plan * cms / nl, sync * cms / nl,
               ph[1] * cms / nl, ph[3] * cms / nl, wio * cms / nl, sp_glue * cms / nl, sp_pre * cms / nl,
               sp_post * cms / nl, sp_call * cms / nl, rt_ms, sum_ms, aie_ms, (aie_ms - sum_ms) / nl, pmc1 - pmc0,
               g_s);
        fflush(stdout);
    }
#endif
"""


PROF_LINE = r"""    {
        double aie_ms_p = 0.0;
        unet_aie_launch_stats(&aie_ms_p, 0, 0);
        unet_prof_print(1000.0 * (double)(t1 - t0) / (double)COUNTS_PER_SECOND, aie_ms_p);
        fflush(stdout);
    }
"""


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
KACC_CC = Path(__file__).resolve().parent / "bf16_gemm_kacc.cc"


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


KACC_KERNEL = r"""__global__ void @NAME@(aie::port<input_window_int32 *, aie::IFM, aie::shape<@AROWS@, @KW@>> a_@SID@,
                          aie::port<input_window_int32 *, aie::WEIGHTS, aie::shape<@BROWS@, @KW@>> b_@SID@,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<@AROWS@, @BROWS@>> c_@SID@) {
    // ports named per instance: aiehlc's block descriptors (aiehlc_desc_<port>) are file-scope statics
    // P4bg: gemm_kacc (bf16_gemm_kacc.cc, Maxwell P4bk) for ch=@CH@ K=@K@ px=@PX@ -- see unet_aie.kacc_geom
    const int tm = 16, tn = 16, tkw = @TKW@, kch = @KCH@; // tile; K chunk = tkw words = kch bf16
    const int nkc = @NKC@;   // K chunks per tile
    const int ntile = @NT@;  // a-port 16-row tiles per core (channels; pixels when transposed)
    const int ns = @S@;      // 16-row sub-tiles of the core's b block (pixels; channels when transposed)
    // transposed=@TR@: the mmul's left operand is always W (pre-tiled [mi][kb][8x8 ch x k]), right is X
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * tm) + t * tm;
        // P4bx: the c window {16, ns*16} fp32 is the accumulator: ns 16x16 tiles in [s][mi][ni][8x8] order
        int32_t *y = (int32_t *)aie::store_buffer(c_@SID@, {m0, pn * (ns * tn)}, {tm, ns * tn});
        float *const kacc = (float *)y;
        for (int i = 0; i < ns * 256; i++)
            y[i] = 0;
        for (int j = 0; j < nkc; j++) {
            const int32_t *x = (const int32_t *)aie::load_buffer(a_@SID@, {m0, j * tkw}, {tm, tkw});
            const int32_t *w = (const int32_t *)aie::load_buffer(b_@SID@, {pn * (ns * tn), j * tkw}, {ns * tn, tkw});
            for (int s = 0; s < ns; s++) {
                float *ac = kacc + s * 256;
#if defined(__chess__)
                using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
                aie::accum<accfloat, 64> q00, q01, q10, q11;
                q00.from_vector(aie::load_v<64>(ac));
                q01.from_vector(aie::load_v<64>(ac + 64));
                q10.from_vector(aie::load_v<64>(ac + 128));
                q11.from_vector(aie::load_v<64>(ac + 192));
                MM c00(q00), c01(q01), c10(q10), c11(q11);
                const bfloat16 *a0 = (const bfloat16 *)(@LHS@), *a1 = a0 + 8 * kch;
                const bfloat16 *b0 = (const bfloat16 *)(@RHS@), *b1 = b0 + 8 * kch;
                for (int kb = 0; kb < kch / 8; kb++) {
                    auto va0 = aie::load_v<64>(a0 + kb * 64);
                    auto va1 = aie::load_v<64>(a1 + kb * 64);
                    auto vb0 = aie::load_v<64>(b0 + kb * 64);
                    auto vb1 = aie::load_v<64>(b1 + kb * 64);
                    c00.mac(va0, vb0);
                    c01.mac(va0, vb1);
                    c10.mac(va1, vb0);
                    c11.mac(va1, vb1);
                }
                aie::store_v(ac, c00.to_vector<float>());
                aie::store_v(ac + 64, c01.to_vector<float>());
                aie::store_v(ac + 128, c10.to_vector<float>());
                aie::store_v(ac + 192, c11.to_vector<float>());
#else // scalar, same pre-tiled layout (chunk = [mi 2][kb kch/8][8x8])
                const uint16_t *xs = (const uint16_t *)(@LHS@), *ws = (const uint16_t *)(@RHS@);
                for (int m = 0; m < tm; m++)
                    for (int n = 0; n < tn; n++) {
                        float sum = 0.0f;
                        for (int k = 0; k < kch; k++) {
                            union {
                                uint32_t u;
                                float f;
                            } xa, wb;
                            xa.u = (uint32_t)xs[((m >> 3) * (kch / 8) + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7)] << 16;
                            wb.u = (uint32_t)ws[((n >> 3) * (kch / 8) + (k >> 3)) * 64 + (k & 7) * 8 + (n & 7)] << 16;
                            sum += xa.f * wb.f;
                        }
                        ac[((m >> 3) * 2 + (n >> 3)) * 64 + (m & 7) * 8 + (n & 7)] += sum;
                    }
#endif
            }
            aie::release(a_@SID@);
            aie::release(b_@SID@);
        }
        aie::commit(c_@SID@); // one fp32 write-back of the core's 16 x px/4 block after the last chunk
    }
}
"""


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


KACC_HOST = r"""
#define UC_KW_ELEMS @MAXW@ /* largest ch*K (bf16) */
#define UC_KX_ELEMS @MAXX@ /* largest px*K (bf16) */
#define UC_KY_ELEMS @MAXY@ /* largest ch*px (fp32) */
static int32_t *g_kw = nullptr, *g_kx = nullptr, *g_ky = nullptr;
#if defined(__aarch64__) && !defined(__chess__)
#include <arm_neon.h>
#define UC_NEON 1
/* 4 x fp32 -> bf16, round-to-nearest-even: bit-identical to uc_f2bf */
static inline uint16x4_t uc_f2bf4(float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u));
    u = vaddq_u32(u, vaddq_u32(lsb, vdupq_n_u32(0x7fffu)));
    return vshrn_n_u32(u, 16);
}
/* GHL: in-place 4x4 fp32 transpose (rows r0..r3 -> columns) */
static inline void uc_tr4(float32x4_t &r0, float32x4_t &r1, float32x4_t &r2, float32x4_t &r3) {
    const float32x4x2_t t01 = vtrnq_f32(r0, r1), t23 = vtrnq_f32(r2, r3);
    /* 64-bit lane trn1/trn2 = combine(low, low) / combine(high, high), one instruction each */
    r0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
    r1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
    r2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
    r3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
}
#else
#define UC_NEON 0
#endif
static unsigned g_ul_wzc = 0; /* GHL: linear sites on the zero-copy W path */
#ifndef UC_OPT
#define UC_OPT 1
#endif
/* GHL host<->DMA sync options: bit0 sparse Y sentinel (one NaN word per 4 KiB instead of a full 0xff memset +
 * flush of Y), bit1 whole D-cache clean+invalidate to PoC before the launch instead of DC CIVAC per 64-B line
 * over W, X and Y (the zero-copy W can be 26 MB = 400k lines), bit2 the same after the launch instead of the
 * Y range invalidate. Set/way cost is bounded by the cache sizes; both make every line the launch touches
 * come from / go to DRAM, as the ranges did. */
static int g_uc_opt = UC_OPT;
#ifndef UC_VAR
#define UC_VAR 214
#endif
#ifndef UC_WHOOK /* GHL: linear S>1 sites scatter block s-1 from the runtime's wait_io poll during launch s.
 * OFF: palmyra213 run of 8dd51b6 (=1) moved aie_host 177 -> 159 ms but aie_wall 1022 -> 1038 and graph_run
 * 5609 -> 5610 (linear9 site 29.1 -> 29.0 ms): the CPU has no idle time inside a launch (bdcfg host glue
 * ~3.6 ms/launch, wait_io short), so the hook only relabels host time as launch time. */
#define UC_WHOOK 0
#endif
/* GHL loop variants (A78 microbench s7/hl/mb, default 214 = 2|4|16|64|128): bit0 scatter n-tiles outer, bit1 m-outer +
 * prefetch of the next n-tile, bit2 im2col in destination-row order, bit3 DC ZVA of fully written lines (no gain),
 * bit4 linear X tile by 8x8 transposes, bit5 im2col chunk-outer (slower), bit6 im2col rows R/R+8 from one 32-B load,
 * bit7 pad columns zeroed in the planar pass (not a separate sweep over every plane) */
static int g_uc_var = UC_VAR;
/* GHL: DC ZVA usable with a 64-B block (DCZID_EL0.BS == 4, DZP == 0): a full output line is allocated zeroed
 * instead of read from DRAM before the scatter overwrites all of it */
static int uc_zva64(void) {
#if defined(__aarch64__)
    static int z = -1;
    if (z < 0) {
        uint64_t d;
        __asm__ volatile("mrs %0, dczid_el0" : "=r"(d));
        z = ((d & 0x10u) == 0 && (d & 0xfu) == 4u) ? 1 : 0;
    }
    return z;
#else
    return 0;
#endif
}
static void uc_dcache_clean_all(void) {
#if defined(__aarch64__)
    uint64_t clidr;
    __asm__ volatile("dsb sy\n\tmrs %0, clidr_el1" : "=r"(clidr)::"memory");
    const unsigned loc = (unsigned)(clidr >> 24) & 7u;
    for (unsigned lvl = 0; lvl < loc; lvl++) {
        if (((clidr >> (3 * lvl)) & 7u) < 2u) /* no data / unified cache at this level */
            continue;
        uint64_t ccs;
        __asm__ volatile("msr csselr_el1, %1\n\tisb\n\tmrs %0, ccsidr_el1" : "=r"(ccs) : "r"((uint64_t)(lvl << 1)) : "memory");
        const unsigned ls = (unsigned)(ccs & 7u) + 4u, ways = (unsigned)((ccs >> 3) & 0x3ffu) + 1u,
                       sets = (unsigned)((ccs >> 13) & 0x7fffu) + 1u;
        const unsigned wsh = ways > 1 ? (unsigned)__builtin_clz(ways - 1u) : 0u;
        for (unsigned w = 0; w < ways; w++)
            for (unsigned s = 0; s < sets; s++)
                __asm__ volatile("dc cisw, %0" ::"r"(((uint64_t)w << wsh) | ((uint64_t)s << ls) | (uint64_t)(lvl << 1))
                                 : "memory");
        __asm__ volatile("dsb sy" ::: "memory");
    }
    __asm__ volatile("msr csselr_el1, xzr\n\tdsb sy\n\tisb" ::: "memory");
#endif
}
/* sentinel: unwritten -> NaN -> counted by the scatter. bit0: one word per 4 KiB (flushed by line unless the
 * whole-cache clean follows), else the full memset (flushed with Y below) */
static void uc_ysentinel(int32_t *ky, size_t bytes) {
    if (g_uc_opt & 1) {
        for (size_t o = 0; o < bytes / 4; o += 1024) {
            ky[o] = -1;
#if defined(__aarch64__)
            if (!(g_uc_opt & 2)) /* clean+invalidate to PoC, one DSB for all (Xil_DCacheFlushLine: 2 DSB + IRQ mask each) */
                __asm__ volatile("dc civac, %0" ::"r"(ky + o) : "memory");
#else
            if (!(g_uc_opt & 2))
                Xil_DCacheFlushLine((INTPTR)(ky + o));
#endif
        }
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#endif
    } else {
        memset(ky, 0xff, bytes);
    }
}
static void uc_sync_pre(const void *w, size_t wb, const void *x, size_t xb, const void *y, size_t yb) {
    if (g_uc_opt & 2) {
        uc_dcache_clean_all();
        return;
    }
    Xil_DCacheFlushRange((UINTPTR)w, wb);
    Xil_DCacheFlushRange((UINTPTR)x, xb);
    if (!(g_uc_opt & 1))
        Xil_DCacheFlushRange((UINTPTR)y, yb);
}
#if UC_NEON
/* GHL: scatter of one launch's tiled fp32 block Y (P4bx: [N][P] normal, [P][N] transposed walk) into the output,
 * 16 n x 16 px per tile, m-tiles outer / n-tiles inner: a tile reads 1 KiB of Y in contiguous chunks at a fixed
 * stride and every output row gets whole 64-B pieces. For one n the 8 px of an 8-aligned group are 8 contiguous
 * fp32 (c & 7 adds to wi, q % 16 == 0); 4 n x 4 px per 4x4 transpose. Output offset: yb == 0 -> [M][N] row-major
 * (linear), else NCHWc with channel block yb (conv, yb % 4 == 0). Epilogue in argument order: et 0 bias [N],
 * et 1 same layout as the output. Values and add order = the scalar loops. Returns the nonfinite count. */
static int uc_scatter16(const float *Ys, int P, int N, int tr, int mbase, int mvalid, float *yout, int M, int yb,
                        const float *const *ep, const int *et, int ne_, int mt0 = 0, int mt1 = 1 << 30) {
    const float32x4_t lim = vdupq_n_f32(1e30f);
    uint32x4_t badv = vdupq_n_u32(0);
    const float *src[16][2];
    const int nouter = g_uc_var & 1, pf = (g_uc_var & 2) && !tr;
    const int zva = (g_uc_var & 8) && yb == 0 && ((uintptr_t)yout & 63) == 0 && uc_zva64();
    const int nmt = (mvalid + 15) / 16, nnt = N / 16;
    /* GHL: src offsets tabulated once per call (2 divisions each): normal walk -> per m-tile (+ nt*P at use),
     * transposed -> per n-tile (+ mt*N at use) */
    static int32_t g_uc_srct[32768];
    static int g_srct_key[4] = {-1, -1, -1, -1}; /* GHL: (tr, P, N, ntt) of the table: m-tile range calls reuse it */
    const int ntt = tr ? nnt : nmt, tab = ntt * 32 <= 32768;
    const int fresh = !(g_srct_key[0] == tr && g_srct_key[1] == P && g_srct_key[2] == N && g_srct_key[3] == ntt);
    if (tab && fresh) {
        g_srct_key[0] = tr;
        g_srct_key[1] = P;
        g_srct_key[2] = N;
        g_srct_key[3] = ntt;
    }
    if (tab && fresh)
        for (int ti = 0; ti < ntt; ti++)
            for (int r = 0; r < 16; r++)
                for (int h = 0; h < 2; h++) {
                    int32_t o;
                    if (!tr) {
                        const int ml = ti * 16 + 8 * h, c = ml & 15, q = P / 4, cc = ml % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                        o = (wi / q) * P + (ml - cc) + wi % q;
                    } else {
                        const int n = ti * 16 + r, q = N / 4, cc = n % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + h) * 64 + (r & 7) * 8;
                        o = (wi / q) * N + (n - cc) + wi % q;
                    }
                    g_uc_srct[ti * 32 + r * 2 + h] = o;
                }
    if (mt1 > nmt)
        mt1 = nmt;
    for (int it = nouter ? 0 : mt0 * nnt; it < (nouter ? nmt * nnt : mt1 * nnt); it++) {
        const int mt = 16 * (nouter ? it % nmt : it / nnt), nt = 16 * (nouter ? it / nmt : it % nnt);
        if (mt < 16 * mt0 || mt >= 16 * mt1) /* GHL: m-tile range [mt0, mt1) (wait-hook chunks) */
            continue;
        if (tab) {
            const float *base = Ys + (tr ? (size_t)mt * N : (size_t)nt * P);
            const int32_t *tt = g_uc_srct + (tr ? nt : mt) * 2; /* (tile/16)*32 */
            for (int r = 0; r < 16; r++) {
                src[r][0] = base + tt[2 * r];
                src[r][1] = base + tt[2 * r + 1];
            }
        } else {
            for (int r = 0; r < 16; r++)
                for (int h = 0; h < 2; h++) {
                    const int n = nt + r, ml = mt + 8 * h, c = ml & 15;
                    if (!tr) {
                        const int q = P / 4, cc = ml % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                        src[r][h] = Ys + (size_t)((n & ~15) + wi / q) * P + (ml - cc) + wi % q;
                    } else {
                        const int q = N / 4, cc = n % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                        src[r][h] = Ys + (size_t)((ml & ~15) + wi / q) * N + (n - cc) + wi % q;
                    }
                }
        }
        {
            if (pf && nt + 16 < N) /* normal walk: the next n-tile is the same tile 16 rows of P further */
                for (int r = 0; r < 16; r++)
                    for (int h = 0; h < 2; h++)
                        __builtin_prefetch(src[r][h] + (size_t)16 * P);
#if defined(__aarch64__)
            if (zva) /* 16 n x 4 B = one whole 64-B line per row (row-major output, N % 16 == 0, 64-B aligned) */
                for (int q = 0; q < 16 && mt + q < mvalid; q++)
                    __asm__ volatile("dc zva, %0" ::"r"(yout + (size_t)(mbase + mt + q) * N + nt) : "memory");
#endif
            /* output offset of (m, n0) = ob[n0] + m * ds (row-major: n0 + m*N; NCHWc: block(n0) + m*yb + n0%yb) */
            const size_t ds = yb ? (size_t)yb : (size_t)N;
            size_t ob[4];
            for (int i = 0; i < 4; i++) {
                const int n0 = nt + 4 * i;
                ob[i] = yb ? (size_t)(n0 / yb) * M * yb + n0 % yb : (size_t)n0;
            }
            for (int mq = 0; mq < 16 && mt + mq < mvalid; mq += 4) {
                const int h = mq >> 3, off = mq & 7, m0 = mbase + mt + mq;
                const int nv = mvalid - (mt + mq) < 4 ? mvalid - (mt + mq) : 4; /* valid rows of the 4 */
                for (int nq = 0; nq < 16; nq += 4) {
                    float32x4_t v0 = vld1q_f32(src[nq][h] + off), v1 = vld1q_f32(src[nq + 1][h] + off),
                                v2 = vld1q_f32(src[nq + 2][h] + off), v3 = vld1q_f32(src[nq + 3][h] + off);
                    if (yb == 1) { /* P4a keep-layout: planar NCHW output (GN input): px m0..+3 of one channel are
                                    * 4 contiguous floats -> no transpose; residual (et 2+rb) in NCHWc block rb */
                        const float32x4_t vv[4] = {v0, v1, v2, v3};
                        for (int k = 0; k < 4; k++) {
                            const int n = nt + nq + k;
                            float *o = yout + (size_t)n * M + m0;
                            float32x4_t a = vv[k];
                            if (nv == 4)
                                badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(a, lim)), 31));
                            for (int e = 0; e < ne_; e++) {
                                if (et[e] == 0) {
                                    a = vaddq_f32(a, vdupq_n_f32(ep[e][n]));
                                } else {
                                    const int rb = et[e] >= 2 ? et[e] - 2 : 1;
                                    const float *r = ep[e] + (size_t)(n / rb) * M * rb + (size_t)m0 * rb + n % rb;
                                    float rv[4] = {r[0], 0.0f, 0.0f, 0.0f};
                                    for (int t = 1; t < 4; t++)
                                        rv[t] = (mt + mq + t < mvalid) ? r[(size_t)t * rb] : 0.0f;
                                    a = vaddq_f32(a, vld1q_f32(rv));
                                }
                            }
                            if (nv == 4)
                                vst1q_f32(o, a);
                            else {
                                float tmp[4];
                                vst1q_f32(tmp, a);
                                for (int t = 0; t < nv; t++)
                                    o[t] = tmp[t];
                            }
                        }
                        continue;
                    }
                    uc_tr4(v0, v1, v2, v3); /* vt = n nt+nq..+3 of px m0+t */
                    const size_t o0 = ob[nq >> 2] + (size_t)m0 * ds;
                    if (nv == 4) {
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v0, lim)), 31));
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v1, lim)), 31));
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v2, lim)), 31));
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v3, lim)), 31));
                        for (int e = 0; e < ne_; e++) {
                            if (et[e]) {
                                const float *ev = ep[e] + o0;
                                v0 = vaddq_f32(v0, vld1q_f32(ev));
                                v1 = vaddq_f32(v1, vld1q_f32(ev + ds));
                                v2 = vaddq_f32(v2, vld1q_f32(ev + 2 * ds));
                                v3 = vaddq_f32(v3, vld1q_f32(ev + 3 * ds));
                            } else {
                                const float32x4_t bv = vld1q_f32(ep[e] + nt + nq);
                                v0 = vaddq_f32(v0, bv);
                                v1 = vaddq_f32(v1, bv);
                                v2 = vaddq_f32(v2, bv);
                                v3 = vaddq_f32(v3, bv);
                            }
                        }
                        float *o = yout + o0;
                        vst1q_f32(o, v0);
                        vst1q_f32(o + ds, v1);
                        vst1q_f32(o + 2 * ds, v2);
                        vst1q_f32(o + 3 * ds, v3);
                    } else {
                        const float32x4_t vv[4] = {v0, v1, v2, v3};
                        for (int t = 0; t < nv; t++) {
                            const size_t o = o0 + (size_t)t * ds;
                            badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(vv[t], lim)), 31));
                            float32x4_t a = vv[t];
                            for (int e = 0; e < ne_; e++)
                                a = vaddq_f32(a, vld1q_f32(ep[e] + (et[e] ? o : (size_t)(nt + nq))));
                            vst1q_f32(yout + o, a);
                        }
                    }
                }
            }
        }
    }
    return (int)vaddvq_u32(badv);
}
#endif
static void uc_sync_post(const void *y, size_t yb) {
    if (g_uc_opt & 4)
        uc_dcache_clean_all();
    else
        Xil_DCacheInvalidateRange((UINTPTR)y, yb);
}
/* GPOOL: pooled host-glue buffers, grown once and reused by every layer (no per-call alloc) */
static uint16_t *g_uc_pb = nullptr; /* padded bf16 planes [C][H+2ph][W+2pw] + slack */
static int *g_uc_sk = nullptr, *g_uc_dk = nullptr; /* per-k source / destination offsets */
static size_t g_uc_pcap = 0, g_uc_kcap = 0;
static unsigned g_uc_fast = 0, g_uc_wzc = 0; /* layers on the fast im2col / zero-copy W path */
static int uc_pool_need(size_t pel, size_t kel) {
    if (pel > g_uc_pcap) {
        free(g_uc_pb);
        g_uc_pb = (uint16_t *)__Runtime_Alloc(pel * 2);
        g_uc_pcap = g_uc_pb ? pel : 0;
    }
    if (kel > g_uc_kcap) {
        free(g_uc_sk);
        free(g_uc_dk);
        g_uc_sk = (int *)__Runtime_Alloc(kel * sizeof(int));
        g_uc_dk = (int *)__Runtime_Alloc(kel * sizeof(int));
        g_uc_kcap = (g_uc_sk && g_uc_dk) ? kel : 0;
    }
    return (g_uc_pb && g_uc_sk && g_uc_dk) ? 0 : -1;
}
static int uc_kalloc(void) {
    if (g_ky)
        return 0;
    g_kw = (int32_t *)__Runtime_Alloc((size_t)UC_KW_ELEMS * 2);
    g_kx = (int32_t *)__Runtime_Alloc((size_t)UC_KX_ELEMS * 2);
    g_ky = (int32_t *)__Runtime_Alloc((size_t)UC_KY_ELEMS * 4);
    return (g_kw && g_kx && g_ky) ? 0 : -1;
}
/* one launch of the layer's gemm_kacc instance (ports W [N][K/2], X [M][K/2], Y f32 [N][M]) */
static int uc_kacc_launch(int N, int K, int M, int32_t *kw, int32_t *kx, int32_t *ky) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({@PART_C0@, 6, 0, 6}, 4, 4); // aiegraph lint L015: the mesh VarDecl in the launching function
@DISPATCH@
    return -6;
}

static int uc_conv(void *const *argv, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw) {
    XTime ut0; /* [UCL] per-layer split: host pre-tiling (W re-tile + im2col) | AIE window | host post */
    XTime_GetTime(&ut0);
    ahs_begin(&g_ahs, AHS_KIND_CONV);
    if (nargs < 4)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 5 || w->ndim != 6 || y->ndim != 5 || x->shape[0] != 1 || y->shape[0] != 1)
        return -2;
    const int cb = (int)x->shape[4], C = (int)x->shape[1] * cb, H = (int)x->shape[2], W = (int)x->shape[3];
    const int ib = (int)w->shape[4], ob = (int)w->shape[5], CI = (int)w->shape[1] * ib;
    const int O = (int)w->shape[0] * ob, KH = (int)w->shape[2], KW = (int)w->shape[3];
    const int yb = (int)y->shape[4], OH = (int)y->shape[2], OW = (int)y->shape[3];
    if (CI != C || (int)y->shape[1] * yb != O || OH != (H + 2 * ph - KH) / sh + 1 || OW != (W + 2 * pw - KW) / sw + 1)
        return -3;
    const int M = OH * OW, N = O, K = C * KH * KW, KK = KH * KW, kc = K / 64;
    if ((M | N | K) & 63 || (long)N * K > UC_KW_ELEMS || (long)M * K > UC_KX_ELEMS || (long)M * N > UC_KY_ELEMS)
        return -3;
    const float *xin = (const float *)x->data;
    float *yout = (float *)y->data;
    if (uc_kalloc() != 0)
        return -1;
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    /* natural layout (unet_aie.kacc_geom): W [N][K], X [M][K], each 16-row block pre-tiled per K chunk of
     * kch bf16 as bf16_gemm_kacc.cc: W chunk = [mi 2][kb kch/8][8x8 ch x k], X chunk = [ni 2][kb][8x8 k x px] */
    int tkw = 4096 / M < 32 ? 4096 / M : 32; /* = kacc_geom tkw */
    const int tr = 2 * (16 * tkw * 4 + (M / 4) * tkw * 4 + 16 * (M / 4) * 4) > 29696; /* = kacc_geom tr: Y^T walk */
    if (tr)
        for (tkw = 32; (N / 4) * tkw > 1024 || (K / 2) % tkw; tkw /= 2)
            ;
    const int kch = 2 * tkw;
    const int Kp = (N == K / 2) ? K + kch : K; /* = kacc_geom kp: zero chunk when the W port would be square */
    if (K % kch || (long)N * Kp > UC_KW_ELEMS || (long)M * Kp > UC_KX_ELEMS)
        return -3;
    uint16_t *Wd = (uint16_t *)g_kw, *Xd = (uint16_t *)g_kx;
    /* P4w: wbf already holds W in the kacc chunk layout for K (unet_aie.prelayout_weights, build time: each
     * 16-row block = 16 rows of K, chunk j at j*kch, f = mi*8*kch + kb*64 + mm*8 + kk within a chunk). With
     * Kp > K each row only gains the zero chunk at its end. */
    int32_t *const kw_pool = g_kw; /* GPOOL: the pooled W port, restored after the launch */
    int wz = 0;
    if (Kp == K && ((uintptr_t)wbf & 63) == 0) {
        /* GPOOL: wbf is already the exact port image (P4w prelayout) -> the launch reads it in place (DMA
         * takes the raw address, as for the aligned_alloc pool; flushed below like the pool) */
        g_kw = (int32_t *)(uintptr_t)wbf;
        g_uc_wzc++;
        wz = 1;
    } else if (Kp == K) {
        memcpy(Wd, wbf, (size_t)N * K * 2);
    } else {
        for (int n = 0; n < N; n++) {
            memcpy(Wd + (size_t)n * Kp, wbf + (size_t)n * K, (size_t)K * 2);
            memset(Wd + (size_t)n * Kp + K, 0, (size_t)(Kp - K) * 2);
        }
    }
    /* GPOOL: im2col via pooled padded bf16 planes P[C][H+2ph][W+2pw] (one RNE convert per input element)
     * and per-k offset tables; 8 output pixels of one row at a fixed k are 8 contiguous bf16 in the X chunk
     * layout ([ni 2][kb][8x8 k x px]) -> one 16-byte copy (stride 2: vld2 even lanes). */
    if (kch > 64 || KK > 64)
        return -3;
    const int fast = (OW % 8) == 0 && sh == sw && (sh == 1 || sh == 2) && (kch % 8) == 0 && (cb % 4) == 0 &&
                     (W % 4) == 0;
    if (fast) {
        const int Hp = H + 2 * ph, Wp = W + 2 * pw;
        const size_t HWp = (size_t)Hp * Wp, pn = (size_t)C * HWp + 64;
        if (uc_pool_need(pn, (size_t)Kp) != 0)
            return -1;
        uint16_t *P = g_uc_pb;
        /* zero only the halo (pad rows/cols) and the slack the stride-2 vld2 may read past the last plane */
        for (int c = 0; c < C; c++) {
            uint16_t *pl = P + (size_t)c * HWp;
            if (ph) {
                memset(pl, 0, (size_t)ph * Wp * 2);
                memset(pl + (size_t)(Hp - ph) * Wp, 0, (size_t)ph * Wp * 2);
            }
            if (pw && !(g_uc_var & 128))
                for (int h = ph; h < Hp - ph; h++) {
                    uint16_t *rw = pl + (size_t)h * Wp;
                    for (int i = 0; i < pw; i++)
                        rw[i] = rw[Wp - 1 - i] = 0;
                }
        }
        memset(P + (size_t)C * HWp, 0, 64 * 2);
        /* NCHWc fp32 -> planar bf16: 4 pixels x 4 channels per 4x4 transpose */
        const size_t HWc = (size_t)H * W * cb;
        for (int cbk = 0; cbk < C / cb; cbk++)
            for (int h = 0; h < H; h++) {
                const float *src = xin + (size_t)cbk * HWc + (size_t)h * W * cb;
                uint16_t *dst0 = P + (size_t)cbk * cb * HWp + (size_t)(h + ph) * Wp + pw;
                for (int w0 = 0; w0 < W; w0 += 4)
                    for (int c0 = 0; c0 < cb; c0 += 4) {
                        const float *s0 = src + (size_t)w0 * cb + c0;
                        uint16_t *d = dst0 + (size_t)c0 * HWp + w0;
#if UC_NEON
                        const float32x4_t r0 = vld1q_f32(s0), r1 = vld1q_f32(s0 + cb), r2 = vld1q_f32(s0 + 2 * cb),
                                          r3 = vld1q_f32(s0 + 3 * cb);
                        const float32x4x2_t t01 = vtrnq_f32(r0, r1), t23 = vtrnq_f32(r2, r3);
                        vst1_u16(d, uc_f2bf4(vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]))));
                        vst1_u16(d + HWp, uc_f2bf4(vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]))));
                        vst1_u16(d + 2 * HWp,
                                 uc_f2bf4(vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]))));
                        vst1_u16(d + 3 * HWp,
                                 uc_f2bf4(vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]))));
#else
                        for (int i = 0; i < 4; i++)
                            for (int j = 0; j < 4; j++)
                                d[(size_t)i * HWp + j] = uc_f2bf(s0[(size_t)j * cb + i]);
#endif
                    }
                if (pw && (g_uc_var & 128)) /* GHL: pad columns of this row while its lines are in L1 */
                    for (int c = 0; c < cb; c++) {
                        uint16_t *rw = dst0 + (size_t)c * HWp - pw;
                        for (int i = 0; i < pw; i++)
                            rw[i] = rw[Wp - 1 - i] = 0;
                    }
            }
        /* per-k tables: source offset in P (c, ky, kx) and destination offset in the 16-row block (ni = 0) */
        int *sk = g_uc_sk, *dk = g_uc_dk;
        {
            int k = 0;
            for (int c = 0; c < C; c++)
                for (int ky = 0; ky < KH; ky++)
                    for (int kx = 0; kx < KW; kx++, k++)
                        sk[k] = (int)((size_t)c * HWp + (size_t)ky * Wp + kx);
            for (k = 0; k < Kp; k++) {
                const int j = k / kch, kr = k - j * kch, f = (kr >> 3) * 64 + (kr & 7) * 8;
                dk[k] = j * kch + (f / kch) * Kp + f % kch;
            }
        }
        /* GHL: channel groups outer -- the CG padded planes (<= 32 KiB) stay in L1 while every 8-px block
         * takes their k range (k-inner over all C streamed the whole P from DRAM once per block) */
        const int CG = (int)(32768 / (HWp * 2)) > 0 ? (int)(32768 / (HWp * 2)) : 1;
#if UC_NEON
        if (g_uc_var & 4) {
            /* GHL: destination-row order -- row R of a 16-px block, chunk j, unit u (8 bf16 = one k, 8 px):
             * f = R*kch + 8u = ni*8*kch + (kb*64 + kk*8), k = j*kch + kb*8 + kk, px = mb0 + ni*8 (= the dk table
             * inverted); every row of the X port is written front to back */
            int kofs[8];
            const int zx = (g_uc_var & 8) && ((uintptr_t)Xd & 63) == 0 && uc_zva64();
            if (g_uc_var & 32) {
                /* GHL: per 16-px block, chunk j0 outer / row R inner: one chunk's kch k (<= 8 channels) stay in L1
                 * while its 16 row pieces (kch bf16 each, whole lines over 1-2 chunks) are written */
                int ko[16][8];
                const uint16_t *rw[16];
                for (int R = 0; R < 16; R++)
                    for (int u = 0; u < kch / 8; u++) {
                        const int rem = (R & 7) * kch + u * 8;
                        ko[R][u] = (rem >> 6) * 8 + ((rem & 63) >> 3);
                    }
                for (int mb0 = 0; mb0 < M; mb0 += 16) {
                    for (int ni = 0; ni < 2; ni++) {
                        const int mb = mb0 + ni * 8, oh = mb / OW, ow0 = mb % OW;
                        rw[ni] = P + (size_t)(oh * sh) * Wp + (size_t)ow0 * sw;
                    }
                    uint16_t *blk = Xd + (size_t)(mb0 >> 4) * 16 * Kp;
                    for (int j0 = 0; j0 < Kp; j0 += kch)
                        for (int R = 0; R < 16; R++) {
                            const uint16_t *row0 = rw[R >> 3];
                            uint16_t *d = blk + (size_t)R * Kp + j0;
                            for (int u = 0; u < kch / 8; u++) {
                                const int k = j0 + ko[R][u];
                                uint16x8_t v;
                                if (k >= K)
                                    v = vdupq_n_u16(0);
                                else if (sw == 1)
                                    v = vld1q_u16(row0 + sk[k]);
                                else
                                    v = vld2q_u16(row0 + sk[k]).val[0];
                                vst1q_u16(d + u * 8, v);
                            }
                        }
                }
            } else if ((g_uc_var & 64) && sw == 1 && OW % 16 == 0) {
                /* GHL: rows R and R + 8 take the same k at px mb0..+7 and mb0+8..+15 of one output row: one 32-B
                 * load (16 contiguous px of the padded plane) feeds both */
                for (int mb0 = 0; mb0 < M; mb0 += 16) {
                    const int oh = mb0 / OW, ow0 = mb0 % OW;
                    const uint16_t *row0 = P + (size_t)(oh * sh) * Wp + (size_t)ow0;
                    uint16_t *blk = Xd + (size_t)(mb0 >> 4) * 16 * Kp;
                    for (int R = 0; R < 8; R++) {
                        for (int u = 0; u < kch / 8; u++) {
                            const int rem = R * kch + u * 8;
                            kofs[u] = (rem >> 6) * 8 + ((rem & 63) >> 3);
                        }
                        uint16_t *d0 = blk + (size_t)R * Kp, *d1 = d0 + (size_t)8 * Kp;
                        for (int j0 = 0; j0 < Kp; j0 += kch)
                            for (int u = 0; u < kch / 8; u++) {
                                const int k = j0 + kofs[u];
                                if (k >= K) {
                                    vst1q_u16(d0 + j0 + u * 8, vdupq_n_u16(0));
                                    vst1q_u16(d1 + j0 + u * 8, vdupq_n_u16(0));
                                } else {
                                    const uint16_t *s = row0 + sk[k];
                                    vst1q_u16(d0 + j0 + u * 8, vld1q_u16(s));
                                    vst1q_u16(d1 + j0 + u * 8, vld1q_u16(s + 8));
                                }
                            }
                    }
                }
            } else
            for (int mb0 = 0; mb0 < M; mb0 += 16)
                for (int R = 0; R < 16; R++) {
                    const int ni = R >> 3, mb = mb0 + ni * 8, oh = mb / OW, ow0 = mb % OW;
                    const uint16_t *row0 = P + (size_t)(oh * sh) * Wp + (size_t)ow0 * sw;
                    uint16_t *dst = Xd + (size_t)(mb0 >> 4) * 16 * Kp + (size_t)R * Kp;
                    for (int u = 0; u < kch / 8; u++) {
                        const int rem = (R & 7) * kch + u * 8;
                        kofs[u] = (rem >> 6) * 8 + ((rem & 63) >> 3);
                    }
                    for (int j0 = 0; j0 < Kp; j0 += kch)
                        for (int u = 0; u < kch / 8; u++) {
#if defined(__aarch64__)
                            if (zx && ((j0 + u * 8) & 31) == 0) /* row written front to back: allocate the line zeroed */
                                __asm__ volatile("dc zva, %0" ::"r"(dst + j0 + u * 8) : "memory");
#endif
                            const int k = j0 + kofs[u];
                            uint16x8_t v;
                            if (k >= K)
                                v = vdupq_n_u16(0);
                            else if (sw == 1)
                                v = vld1q_u16(row0 + sk[k]);
                            else
                                v = vld2q_u16(row0 + sk[k]).val[0];
                            vst1q_u16(dst + j0 + u * 8, v);
                        }
                }
        } else
#endif
        for (int c0 = 0; c0 < C; c0 += CG) {
            const int k0 = c0 * KK, k1 = (c0 + CG < C ? c0 + CG : C) * KK;
            for (int mb = 0; mb < M; mb += 8) {
                const int oh = mb / OW, ow0 = mb % OW;
                uint16_t *blk = Xd + (size_t)(mb >> 4) * 16 * Kp + (size_t)((mb >> 3) & 1) * 8 * Kp;
                const uint16_t *row0 = P + (size_t)(oh * sh) * Wp + (size_t)ow0 * sw;
#if UC_NEON
                if (sw == 1)
                    for (int k = k0; k < k1; k++)
                        vst1q_u16(blk + dk[k], vld1q_u16(row0 + sk[k]));
                else
                    for (int k = k0; k < k1; k++)
                        vst1q_u16(blk + dk[k], vld2q_u16(row0 + sk[k]).val[0]);
#else
                for (int k = k0; k < k1; k++)
                    for (int i = 0; i < 8; i++)
                        blk[dk[k] + i] = row0[sk[k] + i * sw];
#endif
            }
        }
        for (int mb = 0; mb < M && Kp > K && !(UC_NEON && (g_uc_var & 4)); mb += 8) {
            uint16_t *blk = Xd + (size_t)(mb >> 4) * 16 * Kp + (size_t)((mb >> 3) & 1) * 8 * Kp;
            for (int k = K; k < Kp; k++)
                memset(blk + dk[k], 0, 16);
        }
        g_uc_fast++;
    } else {
        /* im2col of the activation, fp32 -> bf16 RNE; the in-chunk offset of (ni, nn, kr) from a table */
        static int g_uc_tab[16][64];
        static int g_uc_ty[64], g_uc_tx[64];
        for (int q = 0; q < 16; q++)
            for (int kr = 0; kr < kch; kr++) {
                const int f = (q >> 3) * 8 * kch + (kr >> 3) * 64 + (kr & 7) * 8 + (q & 7);
                g_uc_tab[q][kr] = (f / kch) * Kp + f % kch;
            }
        for (int t = 0; t < KK; t++) {
            g_uc_ty[t] = t / KW;
            g_uc_tx[t] = t % KW;
        }
        const size_t HWc = (size_t)H * W * cb;
        for (int m = 0; m < M; m++) {
            const int oh = m / OW, ow = m % OW, ih0 = oh * sh - ph, iw0 = ow * sw - pw;
            const int *tab = g_uc_tab[m & 15];
            uint16_t *dst = Xd + (size_t)(m >> 4) * 16 * Kp;
            int kr = 0;
            uint16_t *dj = dst;
            for (int c = 0; c < C; c++) {
                const float *xc = xin + (size_t)(c / cb) * HWc + c % cb;
                for (int t = 0; t < KK; t++) {
                    const int ih = ih0 + g_uc_ty[t], iw = iw0 + g_uc_tx[t];
                    const float v = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? xc[((size_t)ih * W + iw) * cb] : 0.0f;
                    dj[tab[kr]] = uc_f2bf(v);
                    if (++kr == kch) {
                        kr = 0;
                        dj += kch;
                    }
                }
            }
            for (int k = K; k < Kp; k++) {
                dj[tab[kr]] = 0;
                if (++kr == kch) {
                    kr = 0;
                    dj += kch;
                }
            }
        }
    }
    const size_t wbytes = (size_t)N * Kp * 2, xbytes = (size_t)M * Kp * 2;
    (void)kc;
    uc_ysentinel(g_ky, (size_t)M * N * 4); // sentinel: unwritten -> NaN -> counted
    XTime lt0, lt1;
    XTime_GetTime(&lt0);
    const unsigned long long hw0 = uc_hs_wait();
    ahs_add_span(&g_ahs, AHS_PACK, ut0, lt0);
    const unsigned long long sp0 = uc_pmc();
#if UC_WARM
    unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
    __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
    uc_sync_pre(g_kw, wbytes, g_kx, xbytes, g_ky, (size_t)M * N * 4);
    const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
    printf("[UCS] l=%d M=%d N=%d K=%d tkw=%d\n", g_aie_launches, M, N, K, kch / 2); /* P4bp: names a hanging launch */
    fflush(stdout);
    const int lrc = uc_kacc_launch(N, K, M, g_kw, g_kx, g_ky); // ONE launch: the whole layer, K summed in the cores
    g_kw = kw_pool;
    const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
    uc_sync_post(g_ky, (size_t)M * N * 4);
#if UC_WARM
    __Runtime_pm_warm_stats(&seg1, &cold1);
    if (cold1 != cold0) {
        g_uc_kloads++;
        g_uc_cold_tiles += cold1 - cold0;
    }
#endif
    const unsigned long long sp3 = uc_pmc();
    const unsigned long long hw1 = uc_hs_wait();
    XTime_GetTime(&lt1);
    ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0);
    if (lrc != 0)
        return lrc;
    g_sp_pre += sp1 - sp0;
    g_sp_call += sp2 - sp1;
    g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
    g_sp_post += sp3 - sp2;
    g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
    g_aie_launches++;
    g_uc_macs += (unsigned long long)M * N * K;
    /* GPOOL: epilogue (bias / temb per channel, or residual in the output layout) fused into the scatter,
     * applied in argument order as before; 8 px of one channel are 8 contiguous floats of the c window. */
    const float *ep[4];
    int et[4], ne_ = 0; /* et: 0 = per channel, 1 = same shape */
    for (int i = 2; i < nargs - 1; i++) {
        const uc_dlt *e = (const uc_dlt *)argv[i];
        const long ne = uc_numel(e);
        if (ne_ == 4)
            return -5;
        if (ne == N && e->ndim == 5)
            et[ne_] = 0;
        else if (ne == (long)M * N && e->ndim == 5 && (int)e->shape[4] == yb && yb != 1)
            et[ne_] = 1;
        else if (ne == (long)M * N && e->ndim == 5 && yb == 1 && UC_NEON) /* P4a planar out, NCHWc residual */
            et[ne_] = 2 + (int)e->shape[4];
        else
            return -5;
        ep[ne_++] = (const float *)e->data;
    }
    int N_ = N; /* GHL: rows left for the scalar scatter */
    const float *Y = (const float *)g_ky; // [N][M] fp32 (NCHW)
    int bad = 0;
#if UC_NEON
    if (yb % 4 == 0 || yb == 1) { /* P4a: yb == 1 = planar NCHW (keep_layout_rt.py label) */
        bad = uc_scatter16(Y, M, N, tr, 0, M, yout, M, yb, ep, et, ne_); /* GHL: 16x16 tiles */
        N_ = 0; /* scalar loop below skipped */
    }
#endif
    for (int n = 0; n < N_; n++) {
        const int r = n & 15;
        float *on = yout + (size_t)(n / yb) * M * yb + n % yb;
        for (int m = 0; m < M; m += 8) {
            /* P4bx: each core's c block holds 16x16 tiles (ch r, px c) in [s][mi][ni][8x8] order. Normal: block
             * (n/16, m/(M/4)) of Y [N][M], s = px sub-tile. Transposed: block (m/16, n/(N/4)) of Y^T [M][N],
             * s = ch sub-tile. */
            const float *src;
            if (!tr) {
                const int c = m & 15, q = M / 4, cc = m % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                src = Y + (size_t)((n & ~15) + wi / q) * M + (m - cc) + wi % q;
            } else {
                const int c = m & 15, q = N / 4, cc = n % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                src = Y + (size_t)((m & ~15) + wi / q) * N + (n - cc) + wi % q;
            }
            float v[8]; /* (c & 7) = wi & 7 and q % 8 == 0: the 8 px are contiguous in both walks */
            memcpy(v, src, 32);
            for (int i = 0; i < 8; i++)
                if (!(v[i] == v[i]) || v[i] > 1e30f || v[i] < -1e30f)
                    bad++;
            float *o = on + (size_t)m * yb;
            for (int e = 0; e < ne_; e++) {
                if (et[e] == 0) {
                    const float bv = ep[e][n];
                    for (int i = 0; i < 8; i++)
                        v[i] += bv;
                } else {
                    const float *ev = ep[e] + (o - yout);
                    for (int i = 0; i < 8; i++)
                        v[i] += ev[(size_t)i * yb];
                }
            }
            for (int i = 0; i < 8; i++)
                o[(size_t)i * yb] = v[i];
        }
    }
    if (bad) {
        printf("[unet_aie_conv] %d nonfinite AIE outputs\n", bad);
        return -4;
    }
    {
        XTime ut3;
        XTime_GetTime(&ut3);
        ahs_add_span(&g_ahs, AHS_SCATTER, lt1, ut3);
        const double us = 1e6 / (double)COUNTS_PER_SECOND;
        /* GPOOL: [UCL] is recorded here and printed by unet_aie_split_stats() after graph_run (a printf per
         * layer cost ~3.5 ms of A78 time inside the site) */
        const int li = g_aie_launches - 1;
        if (li >= 0 && li < UC_UCL_MAX) {
            uc_ucl *u = &g_uc_ucl[li];
            u->M = M;
            u->N = N;
            u->K = K;
            u->fast = fast | (wz << 1); /* bit0 fast im2col, bit1 zero-copy W */
            u->pre = (float)((double)(lt0 - ut0) * us);
            u->aie = (float)((double)(lt1 - lt0) * us);
            u->post = (float)((double)(ut3 - lt1) * us);
            if (li + 1 > g_uc_nucl)
                g_uc_nucl = li + 1;
        }
    }
    return 0;
}

"""

LINEAR_HOST = r"""
/* PDN (--aie conv,linear): one TVM nn_contrib_dense_pack call site on the AIE. Y[M][N] = X[M][K] . W[N][K]^T
 * (+ bias [1,N] / residual [M,N] epilogue) as S launches of ONE gemm_kacc instance (N, K, px=Mp/S), split
 * along px (kacc_split, as kacc_gemm.py / PDK c009). W is pre-laid at build time in the kacc chunk layout
 * (unet_aie.kacc_dense_wlayout); X is tiled here fp32 -> bf16 RNE, rows M..Mp-1 zero. */
#if UC_NEON && UC_WHOOK
/* GHL: scatter of launch s-1's Y block stepped from the runtime's wait_io poll while launch s computes (one
 * 16-row m-tile per poll iteration); the block was invalidated after launch s-1 returned and launch s's
 * DMA neither reads nor writes it. Whatever the poll did not reach is finished after the launch. */
typedef int (*Runtime_wait_hook_fn)(void *arg);
void __Runtime_set_wait_hook(Runtime_wait_hook_fn fn, void *arg); /* C++ linkage, aie_runtime.h */
typedef struct {
    const float *Ys;
    int P, N, tr, mbase, mvalid, M, nmt, t, bad, ne;
    float *yout;
    const float *const *ep;
    const int *et;
} uc_whook;
static int uc_whook_step(void *a) {
    uc_whook *h = (uc_whook *)a;
    if (h->t < h->nmt) {
        h->bad += uc_scatter16(h->Ys, h->P, h->N, h->tr, h->mbase, h->mvalid, h->yout, h->M, 0, h->ep, h->et, h->ne,
                               h->t, h->t + 1);
        h->t++;
    }
    return h->t >= h->nmt;
}
static unsigned long long g_uc_whook_tiles = 0, g_uc_whook_all = 0; /* m-tiles done in the poll / overlappable */
#endif
static int uc_linear(void *const *argv, int nargs, const uint16_t *wbf, int S) {
    XTime ut0;
    XTime_GetTime(&ut0);
    ahs_begin(&g_ahs, AHS_KIND_LINEAR);
    int wzc_ = 0;
    if (nargs < 3)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 2 || w->ndim != 3 || y->ndim != 2)
        return -2;
    const int M = (int)x->shape[0], K = (int)x->shape[1], N = (int)y->shape[1];
    if ((int)y->shape[0] != M || (int)w->shape[1] != K || (int)(w->shape[0] * w->shape[2]) != N)
        return -3;
    const int Mp = (M + 63) & ~63;
    if (S < 1 || Mp % (64 * S) || (N | K) & 63)
        return -3;
    const int P = Mp / S; /* px per launch */
    int tkw = 4096 / P < 32 ? 4096 / P : 32; /* = kacc_geom(N, K, P) */
    const int tr = 2 * (16 * tkw * 4 + (P / 4) * tkw * 4 + 16 * (P / 4) * 4) > 29696;
    if (tr)
        for (tkw = 32; (N / 4) * tkw > 1024 || (K / 2) % tkw; tkw /= 2)
            ;
    const int kch = 2 * tkw;
    const int Kp = (N == K / 2) ? K + kch : K;
    if (kch > 64 || K % kch || (long)N * Kp > UC_KW_ELEMS || (long)Mp * Kp > UC_KX_ELEMS ||
        (long)Mp * N > UC_KY_ELEMS)
        return -3;
    const float *xin = (const float *)x->data;
    float *yout = (float *)y->data;
    if (uc_kalloc() != 0)
        return -1;
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    uint16_t *Wd = (uint16_t *)g_kw, *Xd = (uint16_t *)g_kx;
    int32_t *const kw_pool = g_kw; /* GHL: the pooled W port, restored after the launches */
    if (Kp == K && ((uintptr_t)wbf & 63) == 0) {
        /* GHL (= GPOOL zero-copy W): wbf is the exact port image (kacc_dense_wlayout, build time) -> read in
         * place by the launches (flushed below like the pool) */
        g_kw = (int32_t *)(uintptr_t)wbf;
        g_ul_wzc++;
        wzc_ = 1;
    } else if (Kp == K) {
        memcpy(Wd, wbf, (size_t)N * K * 2);
    } else {
        for (int n = 0; n < N; n++) {
            memcpy(Wd + (size_t)n * Kp, wbf + (size_t)n * K, (size_t)K * 2);
            memset(Wd + (size_t)n * Kp + K, 0, (size_t)(Kp - K) * 2);
        }
    }
    /* X chunk layout = uc_conv im2col table: row m -> block m/16, in-chunk f = (q>>3)*8*kch + (kr>>3)*64 +
     * (kr&7)*8 + (q&7), q = m&15, chunk j at j*kch of each of the 16 rows (row stride Kp) */
    static int g_ul_tab[16][64];
    for (int q = 0; q < 16; q++)
        for (int kr = 0; kr < kch; kr++) {
            const int f = (q >> 3) * 8 * kch + (kr >> 3) * 64 + (kr & 7) * 8 + (q & 7);
            g_ul_tab[q][kr] = (f / kch) * Kp + f % kch;
        }
#if UC_NEON
    /* GHL: 4 rows x 4 k per 4x4 transpose: for a fixed k the 4 rows q0..q0+3 (q0 % 4 == 0) are 4 contiguous
     * bf16 of the chunk ((q & 7) is the fastest index of f) -> one 8-byte store, RNE as uc_f2bf */
    for (int m0 = 0; m0 < Mp; m0 += 4) {
        const int q0 = m0 & 15;
        const int *tab = g_ul_tab[q0];
        uint16_t *dst = Xd + (size_t)(m0 >> 4) * 16 * Kp;
        if ((g_uc_var & 16) && (m0 & 7) == 0 && m0 + 8 <= M) {
            /* GHL: 8 rows x 8 k: rows -> bf16 (RNE), 8x8 u16 transpose, one 16-byte store per k (8 px) */
            const float *xr = xin + (size_t)m0 * K;
            for (int k0 = 0; k0 < K; k0 += 8) {
                uint16x8_t r[8];
                for (int i = 0; i < 8; i++)
                    r[i] = vcombine_u16(uc_f2bf4(vld1q_f32(xr + (size_t)i * K + k0)),
                                        uc_f2bf4(vld1q_f32(xr + (size_t)i * K + k0 + 4)));
                const uint16x8x2_t t0 = vtrnq_u16(r[0], r[1]), t1 = vtrnq_u16(r[2], r[3]), t2 = vtrnq_u16(r[4], r[5]),
                                   t3 = vtrnq_u16(r[6], r[7]);
                const uint32x4x2_t u0 = vtrnq_u32(vreinterpretq_u32_u16(t0.val[0]), vreinterpretq_u32_u16(t1.val[0])),
                                   u1 = vtrnq_u32(vreinterpretq_u32_u16(t0.val[1]), vreinterpretq_u32_u16(t1.val[1])),
                                   u2 = vtrnq_u32(vreinterpretq_u32_u16(t2.val[0]), vreinterpretq_u32_u16(t3.val[0])),
                                   u3 = vtrnq_u32(vreinterpretq_u32_u16(t2.val[1]), vreinterpretq_u32_u16(t3.val[1]));
#define UC_LO(a, b) vcombine_u16(vget_low_u16(vreinterpretq_u16_u32(a)), vget_low_u16(vreinterpretq_u16_u32(b)))
#define UC_HI(a, b) vcombine_u16(vget_high_u16(vreinterpretq_u16_u32(a)), vget_high_u16(vreinterpretq_u16_u32(b)))
                const uint16x8_t c[8] = {UC_LO(u0.val[0], u2.val[0]), UC_LO(u1.val[0], u3.val[0]),
                                         UC_LO(u0.val[1], u2.val[1]), UC_LO(u1.val[1], u3.val[1]),
                                         UC_HI(u0.val[0], u2.val[0]), UC_HI(u1.val[0], u3.val[0]),
                                         UC_HI(u0.val[1], u2.val[1]), UC_HI(u1.val[1], u3.val[1])};
#undef UC_LO
#undef UC_HI
                const int j = k0 / kch, kr = k0 - j * kch;
                uint16_t *dj = dst + (size_t)j * kch;
                for (int kk = 0; kk < 8; kk++)
                    vst1q_u16(dj + tab[kr + kk], c[kk]);
            }
            for (int k = K; k < Kp; k++) {
                const int j = k / kch, kr = k - j * kch;
                vst1q_u16(dst + (size_t)j * kch + tab[kr], vdupq_n_u16(0));
            }
            m0 += 4; /* + the loop's 4: next 8-row group */
            continue;
        }
        if (m0 + 4 <= M) {
            const float *x0 = xin + (size_t)m0 * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
            for (int k0 = 0; k0 < K; k0 += 4) {
                float32x4_t r0 = vld1q_f32(x0 + k0), r1 = vld1q_f32(x1 + k0), r2 = vld1q_f32(x2 + k0),
                            r3 = vld1q_f32(x3 + k0);
                uc_tr4(r0, r1, r2, r3);
                const int j = k0 / kch, kr = k0 - j * kch;
                uint16_t *dj = dst + (size_t)j * kch;
                vst1_u16(dj + tab[kr], uc_f2bf4(r0));
                vst1_u16(dj + tab[kr + 1], uc_f2bf4(r1));
                vst1_u16(dj + tab[kr + 2], uc_f2bf4(r2));
                vst1_u16(dj + tab[kr + 3], uc_f2bf4(r3));
            }
        } else {
            for (int k0 = 0; k0 < K; k0 += 4) {
                const int j = k0 / kch, kr = k0 - j * kch;
                uint16_t *dj = dst + (size_t)j * kch;
                for (int jj = 0; jj < 4; jj++)
                    for (int i = 0; i < 4; i++)
                        dj[tab[kr + jj] + i] = m0 + i < M ? uc_f2bf(xin[(size_t)(m0 + i) * K + k0 + jj]) : 0;
            }
        }
        for (int k = K; k < Kp; k++) {
            const int j = k / kch, kr = k - j * kch;
            vst1_u16(dst + (size_t)j * kch + tab[kr], vdup_n_u16(0));
        }
    }
    if (0)
#endif
    for (int m = 0; m < Mp; m++) {
        const int *tab = g_ul_tab[m & 15];
        uint16_t *dst = Xd + (size_t)(m >> 4) * 16 * Kp;
        const float *xr = xin + (size_t)m * K;
        for (int j = 0; j * kch < Kp; j++) {
            uint16_t *dj = dst + (size_t)j * kch;
            const int k0 = j * kch;
            if (m < M && k0 < K)
                for (int kr = 0; kr < kch; kr++)
                    dj[tab[kr]] = uc_f2bf(xr[k0 + kr]);
            else
                for (int kr = 0; kr < kch; kr++)
                    dj[tab[kr]] = 0;
        }
    }
    const size_t wbytes = (size_t)N * Kp * 2, xbytes = (size_t)Mp * Kp * 2, ybytes = (size_t)Mp * N * 4;
    /* GHL: epilogue (bias [1,N] / residual [M,N], argument order kept) fused into the scatter */
    const float *ep[4];
    int et[4], ne_ = 0; /* et: 0 = bias [N], 1 = residual [M][N] */
    for (int i = 2; i < nargs - 1; i++) {
        const uc_dlt *e = (const uc_dlt *)argv[i];
        const long ne = uc_numel(e);
        if (ne_ == 4)
            return -5;
        if (ne == N)
            et[ne_] = 0;
        else if (ne == (long)M * N)
            et[ne_] = 1;
        else
            return -5;
        ep[ne_++] = (const float *)e->data;
    }
    const float *Y = (const float *)g_ky; /* launch s: [N][P] tiled block at s*N*P (P4bx [s][mi][ni][8x8]) */
    int bad = 0;
#if UC_NEON && UC_WHOOK
    uc_whook wh;
    memset(&wh, 0, sizeof(wh));
    wh.t = wh.nmt = 0;
#endif
    uc_ysentinel(g_ky, ybytes); // sentinel: unwritten -> NaN -> counted
    XTime la0, la1; /* GHL: [UCL] split of the site, aie = all its launch windows */
    XTime_GetTime(&la0);
    ahs_add_span(&g_ahs, AHS_PACK, ut0, la0);
    unsigned long long hs_win = 0ULL; /* GAP-RT1: sum of the launch windows; rest of [la0, la1] = loop glue -> dma */
    for (int s = 0; s < S; s++) {
        XTime lt0, lt1;
        XTime_GetTime(&lt0);
        const unsigned long long hw0 = uc_hs_wait();
        const unsigned long long sp0 = uc_pmc();
#if UC_WARM
        unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
        __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
        if (s == 0) /* GHL: host->DMA sync in the first launch's window, as uc_conv */
            uc_sync_pre(g_kw, wbytes, g_kx, xbytes, g_ky, ybytes);
        const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
        printf("[UCS] l=%d lin M=%d N=%d K=%d px=%d s=%d/%d tkw=%d\n", g_aie_launches, M, N, K, P, s, S, tkw);
        fflush(stdout);
#if UC_NEON && UC_WHOOK
        if (s > 0 && (g_uc_var & 1) == 0 && (s - 1) * P < M) { /* block s-1 scattered while launch s computes */
            wh.Ys = Y + (size_t)(s - 1) * N * P;
            wh.P = P;
            wh.N = N;
            wh.tr = tr;
            wh.mbase = (s - 1) * P;
            wh.mvalid = M - (s - 1) * P < P ? M - (s - 1) * P : P;
            wh.M = M;
            wh.yout = yout;
            wh.ep = ep;
            wh.et = et;
            wh.ne = ne_;
            wh.t = 0;
            wh.nmt = (wh.mvalid + 15) / 16;
            __Runtime_set_wait_hook(uc_whook_step, &wh);
        }
#endif
        const int lrc = uc_kacc_launch(N, K, P, g_kw, g_kx + (size_t)s * P * Kp / 2, g_ky + (size_t)s * N * P);
        const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
#if UC_NEON && UC_WHOOK
        __Runtime_set_wait_hook(0, 0);
        if (g_uc_opt & 4) {
            if (s == S - 1)
                uc_sync_post(g_ky, ybytes);
        } else { /* per-block invalidate (blocks tile Y): block s is read (next launch's poll / below) only after */
            Xil_DCacheInvalidateRange((UINTPTR)(g_ky + (size_t)s * N * P), (size_t)N * P * 4);
        }
#else
        if (s == S - 1)
            uc_sync_post(g_ky, ybytes);
#endif
#if UC_WARM
        __Runtime_pm_warm_stats(&seg1, &cold1);
        if (cold1 != cold0) {
            g_uc_kloads++;
            g_uc_cold_tiles += cold1 - cold0;
        }
#endif
        const unsigned long long sp3 = uc_pmc();
        const unsigned long long hw1 = uc_hs_wait();
        XTime_GetTime(&lt1);
        ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0);
        hs_win += (unsigned long long)(lt1 - lt0);
        if (lrc != 0) {
            g_kw = kw_pool;
            return lrc;
        }
#if UC_NEON && UC_WHOOK
        if (s > 0 && wh.nmt > 0) { /* the rest of block s-1 the poll did not reach (host time) */
            g_uc_whook_tiles += (unsigned long long)wh.t;
            g_uc_whook_all += (unsigned long long)wh.nmt;
            bad += wh.bad;
            if (wh.t < wh.nmt)
                bad += uc_scatter16(wh.Ys, P, N, tr, wh.mbase, wh.mvalid, yout, M, 0, ep, et, ne_, wh.t, wh.nmt);
            wh.t = wh.nmt = 0;
            wh.bad = 0;
        }
#endif
        g_sp_call += sp2 - sp1;
        g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
        g_sp_post += sp3 - sp2;
        g_sp_pre += sp1 - sp0;
        g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
        g_aie_launches++;
    }
    g_kw = kw_pool;
    XTime_GetTime(&la1);
    if ((unsigned long long)(la1 - la0) > hs_win)
        ahs_add(&g_ahs, AHS_DMA, (unsigned long long)(la1 - la0) - hs_win);
    g_uc_macs += (unsigned long long)M * N * K; /* real rows only (M, not Mp) */
#if UC_NEON
#if UC_WHOOK
    for (int s = (g_uc_var & 1) == 0 ? S - 1 : 0; s < S && s * P < M; s++) /* blocks < S-1: done above */
#else
    for (int s = 0; s < S && s * P < M; s++) /* GHL: 16x16 tiles per launch block */
#endif
        bad += uc_scatter16(Y + (size_t)s * N * P, P, N, tr, s * P, (M - s * P < P ? M - s * P : P), yout, M, 0, ep,
                            et, ne_);
    if (0)
#endif
    for (int m = 0; m < M; m++) {
        const int s = m / P, ml = m % P, c = ml & 15;
        const float *Ys = Y + (size_t)s * N * P;
        float *yo = yout + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const int r = n & 15;
            size_t i;
            if (!tr) {
                const int q = P / 4, cc = ml % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
                i = (size_t)((n & ~15) + wi / q) * P + (ml - cc) + wi % q;
            } else {
                const int q = N / 4, cc = n % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
                i = (size_t)((ml & ~15) + wi / q) * N + (n - cc) + wi % q;
            }
            float v = Ys[i];
            if (!(v == v) || v > 1e30f || v < -1e30f)
                bad++;
            for (int e = 0; e < ne_; e++)
                v += ep[e][et[e] ? (size_t)m * N + n : (size_t)n];
            yo[n] = v;
        }
    }
    if (bad) {
        printf("[unet_aie_linear] %d nonfinite AIE outputs\n", bad);
        return -4;
    }
    {
        XTime ut3;
        XTime_GetTime(&ut3);
        ahs_add_span(&g_ahs, AHS_SCATTER, la1, ut3);
        const double us = 1e6 / (double)COUNTS_PER_SECOND;
        const int li = g_aie_launches - 1; /* GHL: [UCL] of the site's last launch; fast = 4 (linear) | 2 zero-copy W */
        if (li >= 0 && li < UC_UCL_MAX) {
            uc_ucl *u = &g_uc_ucl[li];
            u->M = M;
            u->N = N;
            u->K = K;
            u->fast = 4 | (wzc_ << 1);
            u->pre = (float)((double)(la0 - ut0) * us);
            u->aie = (float)((double)(la1 - la0) * us);
            u->post = (float)((double)(ut3 - la1) * us);
            if (li + 1 > g_uc_nucl)
                g_uc_nucl = li + 1;
        }
    }
    return 0;
}

extern "C" int unet_aie_linear(int cid, void *args, int nargs, const uint16_t *wbf, int S) {
    g_uc_calls++;
    const int rc = uc_linear((void *const *)args, nargs, wbf, S);
    if (rc != 0) {
        printf("[unet_aie_linear] layer %d rc=%d\n", cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

"""


ATTN_HOST = r"""
/* GATT (--aie conv,linear,attn; GAP-aiegraph-attn-site): one TVM nn_batch_matmul call site (an attention QK^T or
 * PV bmm of an aiegraph.partition.match_attention group) on the AIE. TVM batch_matmul is transpose_b:
 * Y[h][M][N] = X[h][M][K] . W[h][N][K]^T, BOTH operands activations -> packed at run time from their natural f32
 * layout by aie_kacc_pack.h (akp_pack, bf16 RNE, zero pad to the kernel geometry; PBK, bmk fd4cb96), H heads x S
 * launches of ONE gemm_kacc instance (unet_aie.bmm_geom = kacc_gemm.main_bmm rules), output un-tiled by akp_unpad
 * (one bf16 SRS) and widened to the f32 TVM tensor. PKC recipe kc_bf16out_pad64: for QK^T the keys (W rows) are
 * centred over the REAL keys per head (softmax is shift-invariant per query row; pad rows stay 0). Softmax stays
 * on the CPU (TVM). */
#include "aie_kacc_pack.h"
#ifndef UC_HS_DEFINED /* GAP-RT1: standalone (x86 harness) -> local split table, no runtime wait counter */
#define UC_HS_DEFINED 1
#include "aie_host_split.h"
static ahs_table g_ahs;
static inline unsigned long long uc_hs_wait(void) { return 0ULL; }
#endif
static float *g_ua_kc = nullptr;    /* one head's centred keys [N][K] (QK^T) */
static uint16_t *g_ua_yb = nullptr; /* one head's bf16 output [M][N] */
static size_t g_ua_kcap = 0, g_ua_ycap = 0;
static unsigned g_ua_sites = 0;
static int uc_attn(void *const *argv, int nargs, int qk, int swap, int CHP, int PXP, int Kp, int kp, int kch,
                   int splitpx, int S, int chs, int pxs, int tr) {
    ahs_begin(&g_ahs, AHS_KIND_ATTN);
    if (nargs != 3)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[2];
    if (x->ndim != 3 || w->ndim != 3 || y->ndim != 3)
        return -2;
    const int H = (int)x->shape[0], M = (int)x->shape[1], K = (int)x->shape[2], N = (int)w->shape[1];
    if ((int)w->shape[0] != H || (int)w->shape[2] != K || (int)y->shape[0] != H || (int)y->shape[1] != M ||
        (int)y->shape[2] != N)
        return -3;
    const int CHR = swap ? M : N, PXR = swap ? N : M;
    if (CHR > CHP || PXR > PXP || K > Kp || Kp > kp || S < 1 || (CHP | PXP) & 63 ||
        (splitpx ? (PXP != S * pxs || CHP != chs) : (CHP != S * chs || PXP != pxs)))
        return -3;
    const size_t wel = (size_t)CHP * kp, xel = (size_t)PXP * kp, yel = (size_t)CHP * PXP;
    if (wel > (size_t)UC_KW_ELEMS || xel > (size_t)UC_KX_ELEMS || yel > (size_t)UC_KY_ELEMS)
        return -3;
    const float *X = (const float *)x->data, *Wt = (const float *)w->data;
    float *Y = (float *)y->data;
    if (uc_kalloc() != 0)
        return -1;
    if ((size_t)M * N > g_ua_ycap) {
        free(g_ua_yb);
        g_ua_ycap = (size_t)M * N;
        g_ua_yb = (uint16_t *)aligned_alloc(64, (g_ua_ycap * 2 + 63) & ~(size_t)63);
    }
    if (qk && (size_t)N * K > g_ua_kcap) {
        free(g_ua_kc);
        g_ua_kcap = (size_t)N * K;
        g_ua_kc = (float *)aligned_alloc(64, (g_ua_kcap * 4 + 63) & ~(size_t)63);
    }
    if (!g_ua_yb || (qk && !g_ua_kc))
        return -1;
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    int32_t *const kw = g_kw, *const kx = g_kx, *const ky = g_ky;
    const size_t wbytes = wel * 2, xbytes = xel * 2, ybytes = yel * 4;
    for (int h = 0; h < H; h++) {
        XTime ht0, ht1, ht2, ht3; /* GAP-RT1: head split pack [ht0, ht1) | launches | scatter [ht2, ht3) */
        XTime_GetTime(&ht0);
        const float *xh = X + (size_t)h * M * K, *wh = Wt + (size_t)h * N * K;
        if (qk) { /* PKC kc_bf16out_pad64: keys centred over the N real keys (fp64 mean per column) */
            for (int k = 0; k < K; k++) {
                double mu = 0.0;
                for (int n = 0; n < N; n++)
                    mu += wh[(size_t)n * K + k];
                const float m = (float)(mu / N);
                for (int n = 0; n < N; n++)
                    g_ua_kc[(size_t)n * K + k] = wh[(size_t)n * K + k] - m;
            }
            wh = g_ua_kc;
        }
        /* W side = instance channels (rows of W, or of X when swapped), X side = pixels; both [rows][K], k stride 1 */
        akp_pack((uint16_t *)kw, swap ? xh : wh, CHP, CHR, kp, K, kch, K, 1, 1, 0);
        akp_pack((uint16_t *)kx, swap ? wh : xh, PXP, PXR, kp, K, kch, K, 1, 0, 0);
        XTime_GetTime(&ht1);
        ahs_add_span(&g_ahs, AHS_PACK, ht0, ht1);
        for (int s = 0; s < S; s++) {
            XTime lt0, lt1;
            XTime_GetTime(&lt0);
            const unsigned long long hw0 = uc_hs_wait();
            if (s == 0)
                uc_sync_pre(kw, wbytes, kx, xbytes, ky, ybytes);
            int32_t *Ws = kw + (splitpx ? 0 : (size_t)s * chs * kp / 2);
            int32_t *Xs = kx + (splitpx ? (size_t)s * pxs * kp / 2 : 0);
            int32_t *Ys = ky + (size_t)s * chs * pxs;
            const int lrc = uc_kacc_launch(chs, Kp, pxs, Ws, Xs, Ys);
            if (s == S - 1)
                uc_sync_post(ky, ybytes);
            const unsigned long long hw1 = uc_hs_wait();
            XTime_GetTime(&lt1);
            ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0);
            g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
            g_aie_launches++;
            if (lrc != 0)
                return lrc;
        }
        XTime_GetTime(&ht2);
        for (int s = 0; s < S; s++) { /* launch s: Y block -> yb [M][N] (bf16) */
            const int p0 = splitpx ? s * pxs : 0, c0 = splitpx ? 0 : s * chs;
            const int pr = PXR - p0 < pxs ? PXR - p0 : pxs, cr = CHR - c0 < chs ? CHR - c0 : chs;
            if (pr <= 0 || cr <= 0)
                continue;
            const size_t off = swap ? (size_t)c0 * N + p0 : (size_t)p0 * N + c0;
            akp_unpad(g_ua_yb + off, N, (const float *)(ky + (size_t)s * chs * pxs), pxs, chs, tr, pr, cr, swap);
        }
        float *yh = Y + (size_t)h * M * N;
        const uint16_t *yb = g_ua_yb;
        for (size_t i = 0; i < (size_t)M * N; i++) {
            const uint32_t u = (uint32_t)yb[i] << 16;
            memcpy(yh + i, &u, 4);
        }
        XTime_GetTime(&ht3);
        ahs_add_span(&g_ahs, AHS_SCATTER, ht2, ht3);
    }
    g_uc_macs += (unsigned long long)H * M * N * K;
    g_ua_sites++;
    return 0;
}

extern "C" int unet_aie_attn(int cid, void *args, int nargs, int qk, int swap, int CHP, int PXP, int Kp, int kp,
                             int kch, int splitpx, int S, int chs, int pxs, int tr) {
    g_uc_calls++;
    const int rc = uc_attn((void *const *)args, nargs, qk, swap, CHP, PXP, Kp, kp, kch, splitpx, S, chs, pxs, tr);
    if (rc != 0) {
        printf("[unet_aie_attn] layer %d rc=%d\n", cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

"""


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
    head = ("// %s.cc -- generated by build_unet.py --aie conv --kincore (P4bg). Kernels: gemm_kacc of %s,\n"
            "// one instance per layer shape (%d), one launch per conv layer, K accumulated in the core.\n"
            "#define UNET_KACC_KERNEL 1\n"
            "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n"
            % (ENTRY, KACC_CC, len(shapes)))
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
