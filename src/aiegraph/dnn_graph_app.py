###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""A graph built through aiegraph.dnn_graph -> a C app that runs it through include/aiednn_graph.h"""

from __future__ import annotations

import ctypes as C
from pathlib import Path

import numpy as np

from .dnn_ctypes import DT
from .dnn_graph import GOP, Graph, attr

_NP = {k: v[1] for k, v in DT.items()}


def _hx(v) -> str:
    return float(np.float32(v)).hex() + "f"


def _attr_lines(lib, kw: dict, arrays: list) -> list:
    """C assignments for the fields of attr(**kw) that differ from aiednn_graph_op_attr_init"""
    a, d = attr(lib, **kw), attr(lib)
    out = []
    for name, ctype in a._fields_:
        if name in ("scales", "zero_points"):
            continue
        va, vd = getattr(a, name), getattr(d, name)
        if isinstance(va, C.Structure):
            for f, _ in va._fields_:
                x, y = getattr(va, f), getattr(vd, f)
                if hasattr(x, "__len__"):
                    for i in range(len(x)):
                        if x[i] != y[i]:
                            out.append(f"a.{name}.{f}[{i}] = {int(x[i])};")
                elif x != y:
                    out.append(f"a.{name}.{f} = {int(x)};")
        elif hasattr(va, "__len__"):
            for i in range(len(va)):
                if va[i] != vd[i]:
                    out.append(f"a.{name}[{i}] = {int(va[i])};")
        elif va != vd:
            if ctype is C.c_float:
                out.append(f"a.{name} = {_hx(va)};")
            elif name == "alg":
                out.append(f"a.alg = (aiednn_alg_kind_t){int(va)};")
            else:
                out.append(f"a.{name} = {int(va)};")
    if "scales" in kw:
        k = len(arrays)
        s = np.asarray(kw["scales"], np.float32).reshape(-1)
        arrays.append(f"static const float S{k}[{s.size}] = {{{', '.join(_hx(v) for v in s)}}};")
        out.append(f"a.n_scales = {s.size}; a.scales = S{k};")
        if kw.get("zero_points") is not None:
            z = np.asarray(kw["zero_points"], np.int32).reshape(-1)
            arrays.append(f"static const int32_t Z{k}[{z.size}] = {{{', '.join(str(int(v)) for v in z)}}};")
            out.append(f"a.zero_points = Z{k};")
    return out


def emit_graph_app(g: Graph, consts: dict, feeds: dict, output, golden: np.ndarray, out: Path, name: str,
                   step: float = 0.0, fuse: int = None, layouts: bool = None, top1_range=None,
                   tie_steps: float = 2.0, ref2: np.ndarray = None, ref2_name: str = "ref2",
                   part_ref: dict = None) -> Path:
    """*g*: a Graph whose compile_all() ran (its partitions say which tensors cross a boundary)"""
    if g.tensors[output][1] != "f32":
        raise ValueError(f"emit_graph_app: the checked output {output!r} must be f32 (is {g.tensors[output][1]})")
    lo, hi, grp = top1_range if top1_range is not None else (0, int(np.prod(golden.shape)), 1)
    fuse = g.fuse if fuse is None else int(fuse)
    layouts = getattr(g, "layouts", True) if layouts is None else bool(layouts)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tid = g.tensor_id
    names = sorted(tid, key=tid.get)                       # ids 1..n in first-use order
    data = {**consts, **feeds}
    boundary = {t for p in g.parts for t in p.inputs + p.outputs} | {output}
    blob, boff = bytearray(), {}
    for n in names:
        if n in data and n in boundary:
            shape, dt, _ = g.tensors[n]
            boff[n] = len(blob)
            blob += np.ascontiguousarray(data[n], _NP[dt]).reshape(shape).tobytes()
            blob += b"\0" * (-len(blob) % 64)
    gold_off = len(blob)
    blob += np.ascontiguousarray(golden, np.float32).tobytes()
    ref2_off = len(blob)
    if ref2 is not None:
        assert ref2.size == golden.size, "ref2 must have the golden's shape"
        blob += np.ascontiguousarray(ref2, np.float32).tobytes()
    pref = []                                              # (id, offset, elements, 0 f32 / 1 s8 / 2 u8)
    for p in g.parts if part_ref else []:
        for o in p.outputs:
            dt = g.tensors[o][1]
            if o in part_ref and dt in ("f32", "s8", "u8") and o not in [r[4] for r in pref]:
                blob += b"\0" * (-len(blob) % 64)
                pref.append((tid[o], len(blob), int(np.prod(g.tensors[o][0])), {"f32": 0, "s8": 1, "u8": 2}[dt], o))
                blob += np.ascontiguousarray(part_ref[o], _NP[dt]).reshape(g.tensors[o][0]).tobytes()
    (out / f"{name}_blob.bin").write_bytes(bytes(blob))
    aoff, arena = {}, 0
    for n in names:
        if n in boundary and n not in boff:
            shape, dt, _ = g.tensors[n]
            aoff[n] = arena
            arena += -(-int(np.prod(shape)) * np.dtype(_NP[dt]).itemsize // 64) * 64
    lt = []
    for n in names:
        shape, dt, const = g.tensors[n]
        dims = ", ".join(str(int(d)) for d in shape) or "0"
        lt.append(f"    {{{tid[n]}, {{{len(shape)}, {{{dims}}}, (aiednn_data_type_t){DT[dt][0]}}}, "
                  f"(aiednn_lt_property_t){int(const)}}},")
    arrays, body = [], []
    ops = [p for p in g.pending if p[0] != "__mark__"]
    for k, (kind, ins, o, opname, kw) in enumerate(g.pending):
        if kind == "__mark__":
            body.append(f"    CK(aiednn_graph_mark_output(g, {tid[o]}));")
            continue
        body.append(f"    /* {opname} */ aiednn_graph_op_attr_init(&a);")
        body += ["    " + s for s in _attr_lines(g.lib, kw, arrays)]
        body += [f"    in[{i}] = LT[{tid[t] - 1}];" for i, t in enumerate(ins)]
        body.append(f"    CK(aiednn_graph_add_op(g, {k + 1}, (aiednn_graph_op_kind_t){GOP[kind]}, &a, {len(ins)}, in, 1, "
                    f"&LT[{tid[o] - 1}]));")
    ptr = [f"    case {tid[n]}: return (void *)(g_blob + {boff[n]});" for n in names if n in boff]
    ptr += [f"    case {tid[n]}: return (void *)(g_arena + {aoff[n]});" for n in names if n in aoff]
    gshape = tuple(golden.shape)
    src = f"""/* Generated by aiegraph/dnn_graph_app.py: {name}, {len(ops)} graph ops through include/aiednn_graph.h.
 * Primitive kinds the partitions create, for dnn_build's kernel selection:
 * aiednn_convolution_forward_create( aiednn_matmul_forward_create( aiednn_inner_product_forward_create( */
#include "aiednn_graph.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define INCBIN(sym, path) \\
    __asm__(".section .rodata\\n.balign 64\\n.global " #sym "\\n" #sym ":\\n.incbin \\"" path "\\"\\n.previous\\n")
INCBIN(g_blob_data, "{(out / f"{name}_blob.bin").resolve()}");
extern "C" const unsigned char g_blob_data[];
#define g_blob ((unsigned char *)g_blob_data)
static unsigned char g_arena[{max(arena, 64)}] __attribute__((aligned(64)));
#define NGOLD {int(np.prod(gshape))}
#define GOLD_OFF {gold_off}
#define HAS_REF2 {int(ref2 is not None)}
#define REF2_OFF {ref2_off}
#define OUT_ID {tid[output]}
#define STEP {_hx(step)}
#define TOP_LO {int(lo)}
#define TOP_HI {int(hi)}
#define TOP_GROUP {int(grp)}
#define TIE_TOL {_hx(tie_steps * (step if step > 0 else 1e-3 * float(np.ptp(golden)) if golden.size else 0.0))}
#define MAXP 1024
#define NPREF {len(pref)}
static const struct {{ uint64_t id; size_t off; long n; int dt; }} PREF[{max(len(pref), 1)}] = {{
{chr(10).join(f"    {{{r[0]}, {r[1]}, {r[2]}, {r[3]}}}," for r in pref) or "    {0, 0, 0, 0},"}
}};
#if defined(__aarch64__)
static uint64_t now_ticks(void) {{ uint64_t v; __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v)); return v; }}
static double ticks_ms(uint64_t t) {{ uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); return 1e3 * (double)t / (double)f; }}
#else
#include <time.h>
static uint64_t now_ticks(void) {{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }}
static double ticks_ms(uint64_t t) {{ return (double)t / 1e6; }}
#endif
#define CK(x) do {{ aiednn_status_t st_ = (x); if (st_ != aiednn_success) {{ printf("FAIL: %s: %s\\n", #x, aiednn_status_str(st_)); return 1; }} }} while (0)

static aiednn_logical_tensor_t LT[{len(names)}] = {{
{chr(10).join(lt)}
}};
{chr(10).join(arrays)}

static void *ptr_of(uint64_t id) {{
    switch (id) {{
{chr(10).join(ptr)}
    default: return nullptr;
    }}
}}

static int build(aiednn_graph_t g) {{
    aiednn_graph_op_attr_t a;
    static aiednn_logical_tensor_t in[{max([len(p[1]) for p in ops] + [1])}];
{chr(10).join(body)}
    return 0;
}}

extern "C" long aiednn_megakernel_switches __attribute__((weak)); /* gemm_uk.cc: ELF reloads (~0.5 ms each) */
static long mk_switches(void) {{ return &aiednn_megakernel_switches ? aiednn_megakernel_switches : 0; }}

static aiednn_compiled_partition_t g_cp[MAXP];
static int g_nin[MAXP], g_nout[MAXP];
static aiednn_tensor_t *g_in[MAXP], *g_out[MAXP];
static uint64_t g_t[MAXP];

static int bind(aiednn_partition_t p, int i) {{
    static aiednn_logical_tensor_t lts[AIEDNN_MAX_SRCS + 8];
    const int ni = aiednn_partition_get_inputs(p, lts, AIEDNN_MAX_SRCS + 8);
    g_in[i] = (aiednn_tensor_t *)malloc((size_t)(ni ? ni : 1) * sizeof(aiednn_tensor_t));
    for (int k = 0; k < ni; k++)
        if (!(g_in[i][k].data = ptr_of(g_in[i][k].id = lts[k].id))) {{
            printf("FAIL: partition %d input %llu has no buffer\\n", i, (unsigned long long)lts[k].id);
            return 1;
        }}
    const int no = aiednn_partition_get_outputs(p, lts, AIEDNN_MAX_SRCS + 8); /* training ops: several */
    g_out[i] = (aiednn_tensor_t *)malloc((size_t)(no ? no : 1) * sizeof(aiednn_tensor_t));
    for (int k = 0; k < no; k++)
        if (!(g_out[i][k].data = ptr_of(g_out[i][k].id = lts[k].id))) {{
            printf("FAIL: partition %d output %llu has no buffer\\n", i, (unsigned long long)lts[k].id);
            return 1;
        }}
    g_nin[i] = ni, g_nout[i] = no;
    return 0;
}}

static int pref_of(uint64_t id) {{
    for (int r = 0; r < NPREF; r++)
        if (PREF[r].id == id) return r;
    return -1;
}}

/* each partition on the emulation's inputs, its output against the emulation's: AIE vs ref: per partition */
static int part_check(int n, aiednn_partition_t *parts) {{
    static double rel[MAXP]; static double mxd[MAXP]; static long mis[MAXP], tot[MAXP];
    int differ = 0;
    for (int i = 0; i < n; i++) {{
        for (int k = 0; k < g_nin[i]; k++) {{
            const int r = pref_of(g_in[i][k].id);
            if (r >= 0) memcpy(g_in[i][k].data, g_blob + PREF[r].off, (size_t)PREF[r].n * (PREF[r].dt ? 1 : 4));
        }}
        CK(aiednn_compiled_partition_execute(g_cp[i], nullptr, g_nin[i], g_in[i], g_nout[i], g_out[i], nullptr, 0));
        rel[i] = mxd[i] = 0, mis[i] = tot[i] = 0;
        for (int k = 0; k < g_nout[i]; k++) {{
            const int r = pref_of(g_out[i][k].id);
            if (r < 0) continue;
            double num = 0, den = 0;
            for (long e = 0; e < PREF[r].n; e++) {{
                const double a = PREF[r].dt == 0 ? ((const float *)g_out[i][k].data)[e]
                                 : PREF[r].dt == 1 ? ((const int8_t *)g_out[i][k].data)[e] : ((const uint8_t *)g_out[i][k].data)[e];
                const double b = PREF[r].dt == 0 ? ((const float *)(g_blob + PREF[r].off))[e]
                                 : PREF[r].dt == 1 ? ((const int8_t *)(g_blob + PREF[r].off))[e] : ((const uint8_t *)(g_blob + PREF[r].off))[e];
                const double d = fabs(a - b);
                num += d * d, den += b * b, mxd[i] = d > mxd[i] ? d : mxd[i];
                mis[i] += PREF[r].dt ? d > 0.5 : d > 1e-5 * (1 + fabs(b));
            }}
            tot[i] += PREF[r].n;
            rel[i] = sqrt(num / (den > 0 ? den : 1));
        }}
        differ += mis[i] > 0;
    }}
    int worst = -1;
    for (int shown = 0; shown < 10; shown++) {{ /* the 10 largest relative differences */
        int b = -1;
        for (int i = 0; i < n; i++)
            if (mis[i] > 0 && (b < 0 || rel[i] > rel[b])) b = i;
        if (b < 0) break;
        if (worst < 0) worst = b;
        printf("PARTCHK %3d relerr=%.2e max_diff=%g mismatched=%ld/%ld %s\\n", b, rel[b], mxd[b], mis[b], tot[b],
               aiednn_partition_info(parts[b]));
        mis[b] = -mis[b]; /* shown */
    }}
    printf("PARTCHK_SUMMARY partitions=%d checked_with_ref=%d differing=%d worst=%d (reported, not checked)\\n", n, NPREF,
           differ, worst);
    return 0;
}}

static int run_all(int n, int timed) {{
    for (int i = 0; i < n; i++) {{
        const uint64_t t0 = now_ticks();
        CK(aiednn_compiled_partition_execute(g_cp[i], nullptr, g_nin[i], g_in[i], g_nout[i], g_out[i], nullptr, 0));
        if (timed) g_t[i] += now_ticks() - t0;
    }}
    return 0;
}}

int main() {{
    printf("=== aiednn graph app {name} ===\\n");
    aiednn_graph_t g;
    CK(aiednn_graph_create(&g));
    aiednn_graph_set_fusion(g, {fuse}); /* the level the graph was partitioned with on the host */
    aiednn_graph_set_layout_propagation(g, {int(layouts)}); /* nhwc on internal edges, as on the host */
    const uint64_t tb = now_ticks();
    if (build(g)) return 1;
    CK(aiednn_graph_finalize(g));
    static aiednn_partition_t parts[MAXP];
    const int n = aiednn_graph_get_partitions(g, parts, MAXP);
    if (n > MAXP) {{ printf("FAIL: %d partitions > MAXP\\n", n); return 1; }}
    int unsup = 0;
    for (int i = 0; i < n; i++) {{
        if (!aiednn_partition_is_supported(parts[i])) {{ printf("UNSUPPORTED %s\\n", aiednn_partition_info(parts[i])); unsup++; continue; }}
        CK(aiednn_partition_compile(parts[i], &g_cp[i]));
        if (bind(parts[i], i)) return 1;
    }}
    if (unsup) {{ printf("FAIL: %d unsupported partitions (no framework to run them here)\\n", unsup); return 1; }}
    const double build_ms = ticks_ms(now_ticks() - tb);
    const uint64_t tc = now_ticks();
    if (run_all(n, 0)) return 1;                       /* cold: weight packing, ELF load, tables */
    const double cold = ticks_ms(now_ticks() - tc);
    aiednn_stats_reset();
    const long sw0 = mk_switches();
    const uint64_t ts = now_ticks();
    if (run_all(n, 1)) return 1;
    const double ms = ticks_ms(now_ticks() - ts);
    printf("MEGAKERNEL switches=%ld in the timed run (~0.5 ms each)\\n", mk_switches() - sw0);
    printf("GRAPH_BUILD time_ms=%.1f (add ops, partition = create + destroy every candidate, compile)\\n", build_ms);
    printf("MODEL_COLD time_ms=%.1f\\n", cold);
    static const char *seen[MAXP]; static double tms[MAXP]; static int cnt[MAXP]; int ns = 0;
    for (int i = 0; i < n; i++) {{
        const char *im = aiednn_compiled_partition_impl_info(g_cp[i]);
        int k = 0;
        while (k < ns && strcmp(seen[k], im)) k++;
        if (k == ns) seen[ns++] = im;
        tms[k] += ticks_ms(g_t[i]); cnt[k]++;
    }}
    for (int k = 0; k < ns; k++) printf("PROF %-44s %4d parts %9.2f ms\\n", seen[k], cnt[k], tms[k]);
    for (int i = 0; i < n; i++) printf("PART %3d %8.3f ms  %s\\n", i, ticks_ms(g_t[i]), aiednn_partition_info(parts[i]));
    aiednn_print_stats();
    const float *y = (const float *)ptr_of(OUT_ID), *ref = (const float *)(g_blob + GOLD_OFF);
    double num = 0, den = 0, mx = 0; int top = TOP_LO, rtop = TOP_LO;
    for (int i = 0; i < NGOLD; i++) {{
        const double d = (double)y[i] - ref[i];
        num += d * d, den += (double)ref[i] * ref[i], mx = fabs(d) > mx ? fabs(d) : mx;
        if (i >= TOP_LO && i < TOP_HI && y[i] > y[top]) top = i;
        if (i >= TOP_LO && i < TOP_HI && ref[i] > ref[rtop]) rtop = i;
    }}
    const double tie_gap = (double)ref[rtop] - ref[top]; /* the golden's own score of the app's pick */
    top = (top - TOP_LO) / TOP_GROUP, rtop = (rtop - TOP_LO) / TOP_GROUP;
    printf("MODEL {name} partitions=%d time_ms=%.2f relerr=%.3e top1=%d ref_top1=%d max_diff_codes=%.1f\\n", n, ms,
           sqrt(num / (den > 0 ? den : 1)), top, rtop, STEP > 0 ? mx / STEP : mx);
    if (HAS_REF2) {{ /* reported only: e.g. the framework's eager output */
        const float *r2 = (const float *)(g_blob + REF2_OFF);
        double n2 = 0, d2 = 0, m2 = 0; int t2 = TOP_LO;
        for (int i = 0; i < NGOLD; i++) {{
            const double d = (double)y[i] - r2[i];
            n2 += d * d, d2 += (double)r2[i] * r2[i], m2 = fabs(d) > m2 ? fabs(d) : m2;
            if (i >= TOP_LO && i < TOP_HI && r2[i] > r2[t2]) t2 = i;
        }}
        printf("REF2 {ref2_name} relerr=%.3e top1=%d max_diff_codes=%.1f (reported, not checked)\\n",
               sqrt(n2 / (d2 > 0 ? d2 : 1)), (t2 - TOP_LO) / TOP_GROUP, STEP > 0 ? m2 / STEP : m2);
    }}
    if (NPREF && part_check(n, parts)) return 1; /* last: it overwrites partition buffers */
    if (top != rtop && tie_gap > TIE_TOL) {{
        printf("FAIL: top-1 differs (golden scores the pick %.4f below its top-1, tolerance %.4f)\\n", tie_gap, TIE_TOL);
        return 1;
    }}
    if (top != rtop)
        printf("NEAR-TIE: top-1 %d vs golden %d, golden gap %.4f <= %.4f\\n", top, rtop, tie_gap, TIE_TOL);
    printf("PASS: aiednn graph app {name}, top-1 %d\\n", top);
    return 0;
}}
"""
    path = out / f"{name}_app.cc"
    path.write_text(src.replace("\n\n\n", "\n\n"))
    return path
