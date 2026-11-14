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
import re
from pathlib import Path

import numpy as np

P2K = Path(__file__).resolve().parent / "bf16_gemm_convin.cc"
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
        out.append(dict(c, cid=L["idx"], onnx=L["node"], weight=L["weight"], launches=launches,
                        macs=M * O * C * KH * KW, sh=at["strides"][0], sw=at["strides"][1],
                        ph=pads[0], pw=pads[1], M=M, N=O, K=C * KH * KW))
    log(f"mapped {len(out)} partition layers to TVM calls")
    return out, convs


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

#define UC_MAX_ACC @MAXACC@ /* largest M*N of the partition layers (fp32 K-sum on the APU) */
#define UC_KPORT 128        /* staged row: 64 bf16 + 64 bf16 zeros (int32 [64][64] port) */

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
static int32_t *g_uc_b = nullptr; // port b: bf16 [64][128]
static int32_t *g_uc_c = nullptr; // port c: f32  [64][64]
static int g_aie_launches = 0;    // every gemm_bf16 launch that ran
static unsigned long long g_uc_macs = 0; // MACs of those launches (valid rows x cols x K)
static int g_uc_calls = 0;        // unet_aie_conv calls
static int g_uc_rc = 0;           // first nonzero rc
static unsigned long long g_uc_aie_ticks = 0; // A78 XTime ticks spent in flush + launch + invalidate
static int g_uc_kloads = 0;       // launches that (re)wrote the kernel's program memory
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

extern "C" int unet_aie_stats(int *launches, int *run_rc, int *calls, unsigned long long *macs) {
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
    g_uc_b = (int32_t *)__Runtime_Alloc(64 * UC_KPORT * 2);
    g_uc_c = (int32_t *)__Runtime_Alloc(64 * 64 * 4);
    if (!g_uc_a || !g_uc_b || !g_uc_c)
        return -1;
    memset(g_uc_a, 0, 64 * UC_KPORT * 2); // cols 64..127 stay zero
    memset(g_uc_b, 0, 64 * UC_KPORT * 2);
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
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 4); // one mesh per call
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
                        B[r * UC_KPORT + kk] = v;
                    }
                }
                memset(g_uc_c, 0xff, 64 * 64 * 4); // sentinel: unwritten -> NaN -> counted
                XTime lt0, lt1;
                XTime_GetTime(&lt0);
#if UC_WARM
                unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
                __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
                Xil_DCacheFlushRange((UINTPTR)g_uc_b, 64 * UC_KPORT * 2);
                Xil_DCacheFlushRange((UINTPTR)g_uc_c, 64 * 64 * 4);
                gemm_bf16<<<mesh>>>(g_uc_a, g_uc_b, g_uc_c);
                Xil_DCacheInvalidateRange((UINTPTR)g_uc_c, 64 * 64 * 4);
#if UC_WARM
                __Runtime_pm_warm_stats(&seg1, &cold1);
                if (cold1 != cold0) {
                    g_uc_kloads++;
                    g_uc_cold_tiles += cold1 - cold0;
                }
#endif
                XTime_GetTime(&lt1);
                g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
                g_aie_launches++;
                const int mv = (M - mb * 64) < 64 ? (M - mb * 64) : 64, nv = (N - nb * 64) < 64 ? (N - nb * 64) : 64;
                g_uc_macs += (unsigned long long)mv * nv * kv;
                for (int r = 0; r < mv; r++)
                    for (int j = 0; j < nv; j++) {
                        const float v = Cb[r * 64 + j];
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

REF_C = r"""// @E@_ref.c -- carrier reference (calibrate/emulate are not run for the UNet build).
#include "aie_user_kernel.h"
AIEK_REFERENCE(@E@) {
    (void)args;
    (void)num_args;
    return AIEK_ERR_KERNEL;
}
"""


def kernel_text() -> str:
    src = P2K.read_text()
    m = re.search(r"(__global__ void gemm_bf16\(.*?\n\})\s*\n\s*int main\(", src, re.S)
    assert m, f"gemm_bf16 not found in {P2K}"
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


def write_kernel(kdst: Path, spec: str, maxacc: int, warm: bool = False) -> Path:
    spec = spec.replace('"out ', '"export    unet_aie_conv\\n"\n        "export    unet_aie_stats\\n"\n'
                        '        "export    unet_aie_launch_stats\\n"\n        "out ', 1)
    assert "export    unet_aie_conv" in spec, spec
    kdst.mkdir(parents=True, exist_ok=True)
    kcc = kdst / f"{ENTRY}.cc"
    head = ("// %s.cc -- generated by build_unet.py --aie conv (P35w). Kernel: gemm_bf16 from %s (verbatim).\n"
            "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\n" % (ENTRY, P2K))
    ktext = kernel_text()
    if warm:
        i = ktext.index(") {\n") + len(") {\n")
        ktext = ktext[:i] + SOFTFLOAT_RM + ktext[i:]
    kcc.write_text(head + ktext + "\n" + HOST_CC.replace("@MAXACC@", str(maxacc)).replace("@WARM@", "1" if warm else "0")
                   .replace("@E@", ENTRY).replace("@SPEC@", spec))
    (kdst / f"{ENTRY}_ref.c").write_text(REF_C.replace("@E@", ENTRY))
    return kcc


# ─────────────────────────────── driver patch ───────────────────────────────
DECL = ("int unet_aie_conv(int cid, void *args, int nargs, const uint16_t *wbf, "
        "int sh, int sw, int ph, int pw);   /* build_unet.py --aie conv */\n")
CALL_RE = re.compile(r"    GRAPH_PROF_BEGIN\((\d+)\);\n(#ifdef GRAPH_AIE_OFFLOAD\n.*?#endif\n|"
                     r"    \w+\(args, codes, \d+, 0, 0, 0\);\n    GRAPH_PROF_END\(\1, \"[^\"]*\"\);\n)", re.S)


def patch_driver(text: str, sel: list) -> tuple:
    if "unet_aie_conv(int cid" in text:
        raise RuntimeError("driver already patched")
    i = text.index("static uint8_t kernel_ws[")
    text = text[:i] + DECL + "\n" + text[i:]
    n = 0
    for c in sel:
        m = re.search(r"    tensors\[(\d+)\]\.data = 0;   /\* %s: bf16 @(\d+)\+(\d+) B" % re.escape(c["wname"]), text)
        assert m, f"no bf16 init line for {c['wname']}"
        eid, off, nb = int(m.group(1)), int(m.group(2)), int(m.group(3))
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
        rep = ("    GRAPH_PROF_BEGIN(%s);\n    unet_aie_conv(%d, args, %d, (const uint16_t *)(g_params + %d), %d, %d, %d, %d);\n"
               "    GRAPH_PROF_END(%s, \"aie unet_aie_conv %d\");\n"
               % (cm.group(1), c["cid"], c["nargs"], off, c["sh"], c["sw"], c["ph"], c["pw"], cm.group(1), c["cid"]))
        text = text[:cm.start()] + rep + text[cm.end():]
        n += 1
    assert text.count("    unet_aie_conv(") == len(sel)
    return text, n


MAIN_EXTRA_DECL = (
    "int unet_aie_stats(int *launches, int *run_rc, int *calls, unsigned long long *macs);\n"
    "int unet_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles);\n"
    "/* the runtime is C++ (aie_runtime.o in the kernel archive): bind its mangled symbol */\n"
    "void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls) "
    "__asm__(\"_Z22__Runtime_phase_cyclesPyPj\");\n"
    "void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls) "
    "__asm__(\"_Z24__Runtime_wait_io_cyclesPyPj\");\n"
    "void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc, "
    "unsigned int *rst_n) __asm__(\"_Z28__Runtime_kload_split_cyclesPyPjS_S0_\");\n"
    "void __Runtime_setup_split_cycles(unsigned long long *plan_cyc, unsigned int *plan_n, unsigned long long *sync_cyc, "
    "unsigned int *sync_n) __asm__(\"_Z28__Runtime_setup_split_cyclesPyPjS_S0_\");\n"
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
        printf("LAUNCH warm ms_per_launch=%.3f launches=%d aie_wall_ms=%.3f kloads=%d opts=%s,cold_tile_loads=%u\n",
               wl ? aie_ms / wl : 0.0, wl, aie_ms, kloads, UNET_LAUNCH_OPTS, cold_tiles);
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
        const double sum_ms = (double)(ph[0] + ph[1] + ph[2] + ph[3] + wio) * cms;
        printf("LAUNCH split kload=%.3f bdcfg=%.3f startio=%.3f coreen=%.3f wait_io=%.3f core_reset=%.3f "
               "ms_per_launch=%.3f pmu_hz=%llu sum_vs_aie_wall=%.3f\n",
               ph[0] * cms / nl, ph[1] * cms / nl, ph[3] * cms / nl, ph[2] * cms / nl, wio * cms / nl,
               krst * cms / nl, aie_ms / nl, pmu_hz, aie_ms > 0.0 ? sum_ms / aie_ms : 0.0);
        printf("LAUNCH split_detail calls kload=%u bdcfg=%u startio=%u coreen=%u wait_io=%u core_reset=%u "
               "kload_elf_ms=%.3f plan_ms=%.3f sync_ms=%.3f phase_sum_ms=%.3f aie_wall_ms=%.3f "
               "unattributed_ms_per_launch=%.4f pmc_delta=%llu xtime_s=%.3f\n",
               phn[0], phn[1], phn[3], phn[2], wion, krstn, kel * cms / nl, plan * cms / nl, sync * cms / nl,
               sum_ms, aie_ms, (aie_ms - sum_ms) / nl, pmc1 - pmc0, g_s);
        fflush(stdout);
    }
#endif
"""


def main_c(base: str) -> str:
    old = '    int rc = p1_cmp("unet-" UNET_AIE_MODE, out, UNET_WDTYPE "w/" UNET_AIE_MODE, 0, 0, 0, 0.0);\n'
    assert old in base
    s = base.replace(old, MAIN_CMP)
    s = s.replace("extern const uint8_t _binary_weights_bin_start[];\n",
                  "extern const uint8_t _binary_weights_bin_start[];\n" + MAIN_EXTRA_DECL, 1)
    s = s.replace("init (aiegraph fp32 C on A78, weights=%s, aie=%s)",
                  "init (aiegraph C on A78 + conv on AIE gemm_bf16, weights=%s, aie=%s)")
    anchor = ' n, REF_UNET_OUT_N);\n'
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + LAUNCH_LINE, 1)
    run = ("    XTime_GetTime(&t0);\n    float *out = graph_run();\n    XTime_GetTime(&t1);\n")
    assert s.count(run) == 1
    s = s.replace(run, "    unet_pmc_enable();\n    unsigned long long pmc0 = unet_pmc();\n" + run.replace(
        "    XTime_GetTime(&t1);\n", "    XTime_GetTime(&t1);\n    unsigned long long pmc1 = unet_pmc();\n"), 1)
    assert "unet_aie_stats(&launches" in s and "_Z22__Runtime_phase_cyclesPyPj" in s
    return s
