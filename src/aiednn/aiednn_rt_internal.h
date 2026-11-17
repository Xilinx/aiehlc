/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn runtime internals, shared by aiednn_rt.cc and the implementation
 * headers it includes (aiednn_rt_gemm.h, aiednn_rt_ops.h). One translation
 * unit: everything here is static. Spliced into the aiehlc entry by the
 * builder, so never write the aiehlc kernel keyword or the launch chevrons.
 ******************************************************************************/
#ifndef AIEDNN_RT_INTERNAL_H
#define AIEDNN_RT_INTERNAL_H

#include "aiednn_impl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiednn_rt_dtypes.h" /* bf16 / f16 / f8 <-> f32 */
#include "aiednn_rt_phi.h"    /* GELU: rt_phi, used by the epilogue fast path */
#include "aiednn_rt_geluq.h"  /* GELU / swish -> int8 by table (rt_epi_row) */

#define RT_MAX_MD (AIEDNN_MAX_SRCS + 2 + AIEDNN_MAX_POST_OPS)
enum { RT_FWD_INF = 0, RT_FWD_TRAIN, RT_BWD_DATA, RT_BWD_WEIGHTS, RT_BWD };

struct rt_impl;

/* One problem: what create() validated. md[] slots are in the kind's slot order
 * (rt_slots_of in aiednn_rt.cc); arg[i] is the AIEDNN_ARG_* bound to slot i. */
typedef struct {
    aiednn_primitive_kind_t kind;
    int alg;
    int nmd;
    int arg[RT_MAX_MD];
    aiednn_memory_desc_t md[RT_MAX_MD];
    aiednn_conv_params_t conv;
    aiednn_pool_params_t pool;
    /* 3-D convolution / pooling: the depth dim's {stride, pad_l, pad_r, dilate, kernel}; conv / pool
     * above carry H and W (aiednn_rt_nd.h). Unused (0) for 1-D / 2-D. */
    int64_t dpth[5];
    /* training (aiednn_rt_bwd.h): RT_FWD_INF / RT_FWD_TRAIN / RT_BWD_DATA / RT_BWD_WEIGHTS / RT_BWD,
     * and the N-D geometry of a training convolution / pooling */
    int prop;
    aiednn_conv_nd_params_t cnd;
    aiednn_pool_nd_params_t pnd;
    /* per output channel, added to the exact accumulator before the epilogue (zp+: (shift - zp) * sum(w));
     * owned by the composite for one execute; NULL = none */
    const int64_t *acc_add;
    float alpha, beta, eps, pnorm, lrn_k, scale;
    int axis, transposed, causal, nsrc;
    unsigned flags;
    int64_t groups, local_size, group_size;
    int perm[AIEDNN_MAX_NDIMS];
    int64_t slice_off[AIEDNN_MAX_NDIMS], slice_step[AIEDNN_MAX_NDIMS]; /* slice (aiednn_rt_slice.h) */
    float scales[AIEDNN_MAX_SRCS];
    int32_t zps[AIEDNN_MAX_SRCS]; /* quantized concat: each source's zero point (scales[] its scale) */
    int32_t zero_point;
    /* fused epilogue (aiednn_attr_t): output scales, post-ops, integer dst requantize */
    int has_attr;
    int64_t n_oscales;
    const float *oscales; /* owned by the primitive (copied at create) */
    float dst_scale;
    int32_t dst_zp;
    int32_t src_zp, wei_zp; /* GEMM family, per tensor (aiednn_attr_t); nonzero: ref: only */
    float bin_sc[2];        /* quantized binary (aiednn_binary_create_attr): the sources' scales (0: as is) */
    int32_t bin_zp[2];      /* and zero points */
    /* dequantized sources (aiednn_rt_phead.h): source i < ndq with dq_sc[i] > 0 is s8 / u8, read as
     * f32((code - dq_zp[i]) * dq_sc[i]); the dispatched problem sees f32 there */
    int ndq, dq_packed;
    unsigned attr_flags; /* aiednn_attr_t.flags (AIEDNN_ATTR_TOLERANCE_1CODE) */
    float dq_sc[4];
    int32_t dq_zp[4];
    int npo;
    aiednn_post_op_t po[AIEDNN_MAX_POST_OPS];
} rt_prob;

typedef struct {
    const struct rt_impl *inner;  /* composite: the implementation it runs on */
    const aiednn_impl_t *blk;     /* blocked: the AIE GEMM block kernel */
    const struct rt_impl *inner2; /* a composite over a composite (zp+ over im2col): the inner's inner, for impl_info */
} rt_bind;

#define RT_TUNE_MAX 8 /* implementations an autotuned primitive times (aiednn_rt_tune.h) */
/* aiednn_set_preference: 0 = AIE whenever an AIE implementation can compute the problem; 1 = fastest
 * (acceptors may decline a shape where ref: was measured faster: rt_cost_refusals) */
static int g_prefer;
static inline int rt_cost_refusals(void) { return g_prefer == 1; }

struct aiednn_primitive {
    int in_use;
    rt_prob prob;
    const struct rt_impl *impl;
    rt_bind bind;
    /* every implementation that accepted the problem, for the autotuner (aiednn_set_autotune) */
    int ncand, tuned;
    int sw_checked; /* the megakernel-switch trial ran (execute_args_body, aiednn_prefer_fastest) */
    const struct rt_impl *cand[RT_TUNE_MAX];
    rt_bind cbind[RT_TUNE_MAX];
    uint64_t sig;
    int const_w; /* aiednn_primitive_set_constant_weights */
    char info[160];
    /* aiednn_primitive_get_workspace_size: measured for ws_impl (a sizing run), valid while it is chosen */
    const struct rt_impl *ws_impl;
    size_t ws_size;
    /* zp+ (aiednn_rt_gemmx.h): the per-output-channel weight sums of constant weights (const_w), computed
     * once for the weights at wsum_key -- O x K adds per execute otherwise (resnet18: ~11 M on the A72) */
    int64_t *wsum;
    const void *wsum_key;
    int64_t wsum_n;
    /* a requant_binary post-op (aiednn_post_op_requant_binary): prob is the GEMM with its int8 intermediate as
     * dst, qtail the quantized binary that follows (src = that intermediate, src1 = arg qtail_src1) */
    int qres;
    rt_prob qtail;
    int qtail_src1;
    /* a requant_eltwise tail: its 256-entry table (the eltwise-attr ref on every code), built on first use */
    int qlut_ok;
    float qlut[256];
    /* a requant_binary tail with 8-bit src1 / dst: dst = qlut2[code][src1 code], 64 KB, the ref chain on every
     * pair (built on first use; NULL until then, freed by destroy) */
    uint8_t *qlut2;
    /* memory formats (aiednn_rt_layout.h): slots whose nhwc buffer execute converts around the dense run
     * (wrap), with their scratch; freed by destroy */
    unsigned fmt_mask;
    void *fmt_buf[RT_MAX_MD];
    size_t fmt_cap[RT_MAX_MD];
    /* a requant post-op whose src1 is in the other format than dst: converted per execute (1: to nhwc, 2: to
     * dense), qs1_md its descriptor as created */
    int qs1_cvt;
    aiednn_memory_desc_t qs1_md;
    void *qs1_buf;
    size_t qs1_cap;
    /* post-ops on a non-GEMM primitive (aiednn_rt_ptail.h): prob is the op writing f32, pt_ep the epilogue */
    int ptail;
    rt_prob pt_ep;
    /* dequantized sources (aiednn_rt_phead.h): their s8 / u8 descriptors and scales; prob has them as f32 */
    int ndq;
    int dq_packed; /* sdpa AIEDNN_SDPA_QKV_PACKED: sources 0..2 are rows of one [B, S, 3, H, D] buffer */
    aiednn_memory_desc_t dq_md[4];
    float dq_sc[4];
    int32_t dq_zp[4];
};

static int g_exec_const_w; /* the executing primitive's WEIGHTS are a model constant */

/* a requant_* post-op tail fused into the GEMM half's own output write (aiednn_primitive_execute_args sets it up;
 * an impl that writes EVERY dst element through pipe_work / dw_put_item sets claim, the others leave the tail to
 * a separate in-place pass). dst is the int8 intermediate and the final 8-bit result at once. */
typedef struct {
    int on, claim;
    uint8_t *base;       /* dst */
    const uint8_t *lut;  /* eltwise tail: final code of each intermediate code */
    const uint8_t *lut2; /* binary tail: final code of [intermediate code << 8 | src1 code] */
    const uint8_t *src1; /* binary tail: src1, laid out as dst */
    uint64_t t;          /* ticks spent (reported as RT_PH_QTAIL) */
} rt_qfuse;
static rt_qfuse g_qfuse;
static unsigned long g_qtail_n, g_qtail_fused; /* requant tails run / fused into the output write */

/* n elements at dst, stride apart (just written, still in cache): intermediate code -> final code */
static inline void rt_qfuse_run(void *dst, int64_t n, int64_t stride = 1) {
    if (!g_qfuse.on || !g_qfuse.claim || n <= 0)
        return;
    uint8_t *d = (uint8_t *)dst;
    const int64_t off = d - g_qfuse.base;
    if (g_qfuse.lut2) {
        const uint8_t *s = g_qfuse.src1 + off, *t = g_qfuse.lut2;
        for (int64_t i = 0; i < n * stride; i += stride)
            d[i] = t[(unsigned)d[i] << 8 | s[i]];
    } else {
        const uint8_t *t = g_qfuse.lut;
        for (int64_t i = 0; i < n * stride; i += stride)
            d[i] = t[d[i]];
    }
}

#define RT_SDPA_VAR 0x100 /* rt_prob.flags of an sdpa: seq lens / page table (aiednn_rt_sdpa_var.h) */
enum {
    RT_CAP_3D = 1,
    RT_CAP_TRAIN = 2,
    RT_CAP_ZP = 4,
    RT_CAP_WQ = 8 /* weight group scales */,
    RT_CAP_NHWC = 16 /* reads / writes nhwc operands itself (or converts internally) */,
    RT_CAP_Q8SRC = 32 /* reads 8-bit sources with their attr scales (no dequantized copies) */
};                    /* rt_impl.caps: problem classes an impl takes */
typedef struct rt_impl {
    char name[64];
    aiednn_primitive_kind_t kind;
    int specificity;
    int (*accepts)(const struct rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
    int (*run)(const struct rt_impl *self, aiednn_primitive *prim, void *const *ptr);
    const aiednn_impl_t *gen; /* generated kernel behind aie: / blocked: */
    unsigned caps;            /* RT_CAP_*: problem classes dispatch otherwise keeps from non-ref impls */
    unsigned long execs, launches;
    uint64_t t_total, t_entry; /* ticks: whole run; inside the AIE kernel entry (copies + launch) */
} rt_impl;

static int dispatch(const rt_prob *p, const rt_impl **out, rt_bind *bind, int quiet,
                    struct aiednn_primitive *cands = nullptr);

/* memory formats (aiednn_rt_layout.h has the rest) */
static inline int md_is_nhwc(const aiednn_memory_desc_t *m) { return m->ndims == 4 && m->format == aiednn_format_nhwc; }

static inline int rt_prob_has_fmt(const rt_prob *p) {
    for (int s = 0; s < p->nmd; s++)
        if (p->md[s].format != aiednn_format_dense)
            return 1;
    return 0;
}

/* an nhwc-capable impl on a branch without an nhwc path: run itself on dense copies */
static int rt_fmt_rerun_dense(const rt_impl *self, aiednn_primitive *prim, void *const *ptr0,
                              int (*run)(const rt_impl *, aiednn_primitive *, void *const *));

/* phase timers: the architectural counter on aarch64 (no BSP), CLOCK_MONOTONIC else */
#if defined(__aarch64__)
static inline uint64_t rt_ticks(void) {
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static double rt_ticks_ms(uint64_t t) {
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return 1e3 * (double)t / (double)f;
}
#else
#include <time.h>
static inline uint64_t rt_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static double rt_ticks_ms(uint64_t t) { return (double)t / 1e6; }
#endif
static double eltwise_f(int alg, double x, double a, double b); /* aiednn_rt_ops.h */
static int run_bwd(const rt_prob *p, void *const *ptr);         /* aiednn_rt_bwd.h: training */
static double binary_f(int alg, double a, double b);

/* ── slots ──────────────────────────────────────────────────────────────── */

static int rt_slot(const rt_prob *p, int arg) {
    for (int i = 0; i < p->nmd; i++)
        if (p->arg[i] == arg)
            return i;
    return -1;
}

static const aiednn_memory_desc_t *rt_md(const rt_prob *p, int arg) {
    const int s = rt_slot(p, arg);
    return s < 0 ? nullptr : &p->md[s];
}

static void *rt_ptr(const rt_prob *p, void *const *ptr, int arg) {
    const int s = rt_slot(p, arg);
    return s < 0 ? nullptr : ptr[s];
}

/* ── data types ─────────────────────────────────────────────────────────── */

static const char *dt_str(aiednn_data_type_t dt) {
    switch (dt) {
    case aiednn_s8:
        return "s8";
    case aiednn_u8:
        return "u8";
    case aiednn_s32:
        return "s32";
    case aiednn_f32:
        return "f32";
    case aiednn_bf16:
        return "bf16";
    case aiednn_f16:
        return "f16";
    case aiednn_f8_e5m2:
        return "f8_e5m2";
    case aiednn_f8_e4m3:
        return "f8_e4m3";
    case aiednn_s4:
        return "s4";
    case aiednn_u4:
        return "u4";
    case aiednn_e8m0:
        return "e8m0";
    case aiednn_f4_e2m1:
        return "f4_e2m1";
    default:
        return "undef";
    }
}

static int dt_bytes(aiednn_data_type_t dt) {
    return dt == aiednn_s8 || dt == aiednn_u8 || dt == aiednn_f8_e5m2 || dt == aiednn_f8_e4m3 || dt == aiednn_e8m0 ? 1
           : dt == aiednn_bf16 || dt == aiednn_f16                                                                 ? 2
                                                                                                                   : 4;
}

static int dt_is_int(aiednn_data_type_t dt) {
    return dt == aiednn_s8 || dt == aiednn_u8 || dt == aiednn_s32 || dt == aiednn_s4 || dt == aiednn_u4;
}
static int dt_is_4bit(aiednn_data_type_t dt) { return dt == aiednn_s4 || dt == aiednn_u4 || dt == aiednn_f4_e2m1; }
/* bytes of n elements (4-bit types: two per byte) */
static int64_t dt_size(aiednn_data_type_t dt, int64_t n) { return dt_is_4bit(dt) ? (n + 1) / 2 : n * dt_bytes(dt); }
static int dt_is_float(aiednn_data_type_t dt) {
    return (dt >= aiednn_f32 && dt <= aiednn_f8_e4m3) || dt == aiednn_e8m0 || dt == aiednn_f4_e2m1;
}

static double rt_ld(aiednn_data_type_t dt, const void *p, int64_t i) {
    switch (dt) {
    case aiednn_s8:
        return ((const int8_t *)p)[i];
    case aiednn_u8:
        return ((const uint8_t *)p)[i];
    case aiednn_s32:
        return ((const int32_t *)p)[i];
    case aiednn_bf16:
        return rt_bf16_to_f32(((const uint16_t *)p)[i]);
    case aiednn_f16:
        return rt_f16_to_f32(((const uint16_t *)p)[i]);
    case aiednn_f8_e5m2:
        return rt_e5m2_to_f32(((const uint8_t *)p)[i]);
    case aiednn_f8_e4m3:
        return rt_e4m3_to_f32(((const uint8_t *)p)[i]);
    case aiednn_s4: {
        const int v = (((const uint8_t *)p)[i >> 1] >> ((i & 1) * 4)) & 0xf;
        return v >= 8 ? v - 16 : v;
    }
    case aiednn_u4:
        return (((const uint8_t *)p)[i >> 1] >> ((i & 1) * 4)) & 0xf;
    case aiednn_e8m0:
        return rt_e8m0_to_f32(((const uint8_t *)p)[i]);
    case aiednn_f4_e2m1:
        return rt_e2m1_to_f32((((const uint8_t *)p)[i >> 1] >> ((i & 1) * 4)) & 0xf);
    default:
        return ((const float *)p)[i];
    }
}

/* integer destinations: round to nearest even, then saturate */
/* host-side phase timers of the AIE composites (aiednn_print_stats: aiednn_phase line) */
/* RT_PH_CUNPACK: an n-tile / MMT / depthwise / items launch's result tiles copied out of the device
 * buffer into [M][N] / item order (the int8 scatter to channel planes and host epilogues stay RT_PH_EPI) */
/* RT_PH_FMT: nhwc <-> dense conversions of a wrapped primitive (aiednn_rt_layout.h) */
/* RT_PH_ZPPAD / ZPWSUM: zp+'s x - shift pre-pad copy and its weight sums; RT_PH_QTAIL: a requant_* post-op's
 * tail on the int8 tile (aiednn_post_op_requant_binary / _eltwise) */
enum {
    RT_PH_IM2COL,
    RT_PH_APACK,
    RT_PH_BPACK,
    RT_PH_ACCUM,
    RT_PH_EPI,
    RT_PH_DWPACK,
    RT_PH_HWC,
    RT_PH_REGION,
    RT_PH_CUNPACK,
    RT_PH_ZPPAD,
    RT_PH_ZPWSUM,
    RT_PH_QTAIL,
    RT_PH_FMT,
    RT_PH_SDPA_SM,
    RT_PH_SDPA_ROWS,
    RT_PH_SDPA_GEMM,
    RT_PH_SDPA_OUT, /* gemm+sdpa's parts (aiednn_rt_gemmx.h) */
    RT_PH_N
};
static uint64_t g_phase[RT_PH_N];
#define RT_PHASE(ph, stmt)                                                                                             \
    do {                                                                                                               \
        const uint64_t ph_t0_ = rt_ticks();                                                                            \
        stmt;                                                                                                          \
        g_phase[ph] += rt_ticks() - ph_t0_;                                                                            \
    } while (0)

static void rt_st(aiednn_data_type_t dt, void *p, int64_t i, double v) {
    switch (dt) {
    case aiednn_f32:
        ((float *)p)[i] = (float)v;
        return;
    case aiednn_bf16:
        ((uint16_t *)p)[i] = rt_f32_to_bf16((float)v);
        return;
    case aiednn_f16:
        ((uint16_t *)p)[i] = rt_f32_to_f16((float)v);
        return;
    case aiednn_f8_e5m2:
        ((uint8_t *)p)[i] = rt_f32_to_e5m2((float)v);
        return;
    case aiednn_f8_e4m3:
        ((uint8_t *)p)[i] = rt_f32_to_e4m3((float)v);
        return;
    case aiednn_e8m0:
        ((uint8_t *)p)[i] = rt_f32_to_e8m0((float)v);
        return;
    case aiednn_f4_e2m1: {
        uint8_t *b = (uint8_t *)p + (i >> 1);
        const int sh = (int)(i & 1) * 4;
        *b = (uint8_t)((*b & ~(0xf << sh)) | (rt_f32_to_e2m1((float)v) << sh));
        return;
    }
    default:
        break;
    }
    double lo = dt == aiednn_s8   ? -128.0
                : dt == aiednn_u8 ? 0.0
                : dt == aiednn_s4 ? -8.0
                : dt == aiednn_u4 ? 0.0
                                  : -2147483648.0;
    double hi = dt == aiednn_s8   ? 127.0
                : dt == aiednn_u8 ? 255.0
                : dt == aiednn_s4 ? 7.0
                : dt == aiednn_u4 ? 15.0
                                  : 2147483647.0;
    v = v != v ? 0.0 : nearbyint(v);
    v = v < lo ? lo : v > hi ? hi : v;
    if (dt_is_4bit(dt)) {
        uint8_t *b = (uint8_t *)p + (i >> 1);
        const int sh = (int)(i & 1) * 4;
        *b = (uint8_t)((*b & ~(0xf << sh)) | ((((int)v) & 0xf) << sh));
        return;
    }
    if (dt == aiednn_s8)
        ((int8_t *)p)[i] = (int8_t)v;
    else if (dt == aiednn_u8)
        ((uint8_t *)p)[i] = (uint8_t)v;
    else
        ((int32_t *)p)[i] = (int32_t)v;
}

/* ── shapes ─────────────────────────────────────────────────────────────── */

static int64_t md_numel(const aiednn_memory_desc_t *md) {
    int64_t n = 1;
    for (int i = 0; i < md->ndims; i++)
        n *= md->dims[i];
    return n;
}

static int md_same_shape(const aiednn_memory_desc_t *a, const aiednn_memory_desc_t *b) {
    if (a->ndims != b->ndims)
        return 0;
    for (int i = 0; i < a->ndims; i++)
        if (a->dims[i] != b->dims[i])
            return 0;
    return 1;
}

/* src broadcastable to dst: same rank, each dim equal or 1 */
static int md_bcast_ok(const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst) {
    if (src->ndims != dst->ndims)
        return 0;
    for (int i = 0; i < src->ndims; i++)
        if (src->dims[i] != dst->dims[i] && src->dims[i] != 1)
            return 0;
    return 1;
}

/* offset in a (broadcast) src of the dst element with flat index i */
static int64_t bcast_off(const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *dst, int64_t i) {
    int64_t off = 0, stride = 1;
    for (int d = dst->ndims - 1; d >= 0; d--) {
        const int64_t c = i % dst->dims[d];
        i /= dst->dims[d];
        if (src->dims[d] != 1)
            off += c * stride;
        stride *= src->dims[d];
    }
    return off;
}

/* grow-only scratch buffer (process lifetime) */
/* ── execute-time memory ─────────────────────────────────────────────────────
 * An execute's temporaries come from rt_alloc / rt_calloc / scratch: the heap normally, the caller's
 * workspace (aiednn_primitive_execute_async) when g_ws.base is set -- a bump arena, so that execute
 * makes no heap allocation -- and with g_ws.sizing the heap serves while g_ws.high records what an
 * arena would have needed (aiednn_primitive_get_workspace_size). Caches that outlive an execute
 * (packed constant weights, step tables, the tune table) stay on the heap. */
#define RT_WS_ALIGN 64
typedef struct {
    char *base;
    size_t size, off, high;
    int sizing, overflow;
} rt_ws_t;
static rt_ws_t g_ws;

static size_t rt_ws_take(size_t n) { /* the arena offset of the next n bytes, moving the high-water mark */
    const size_t at = (g_ws.off + RT_WS_ALIGN - 1) & ~(size_t)(RT_WS_ALIGN - 1);
    g_ws.off = at + n;
    g_ws.high = g_ws.off > g_ws.high ? g_ws.off : g_ws.high;
    return at;
}
static void *rt_alloc(size_t n) {
    n = n ? n : 1;
    if (g_ws.base) {
        const size_t at = rt_ws_take(n);
        if (g_ws.off > g_ws.size) {
            g_ws.overflow = 1;
            return nullptr;
        }
        return g_ws.base + at;
    }
    if (g_ws.sizing)
        rt_ws_take(n);
    return malloc(n);
}
static void *rt_calloc(size_t n, size_t sz) {
    void *p = rt_alloc(n * sz);
    if (p)
        memset(p, 0, n * sz);
    return p;
}
static void rt_free(void *p) { /* arena memory is released with the execute */
    if (p && !(g_ws.base && (char *)p >= g_ws.base && (char *)p < g_ws.base + g_ws.size))
        free(p);
}

/* a buffer that grows across executes on the heap; from the arena when a caller workspace is set */
static void *scratch(void **buf, size_t *cap, size_t need) {
    if (g_ws.base)
        return rt_alloc(need);
    if (g_ws.sizing)
        rt_ws_take(need ? need : 1);
    if (need > *cap) {
        free(*buf);
        *buf = malloc(need ? need : 1);
        *cap = *buf ? need : 0;
    }
    return *buf;
}

static void fill_tensor(aiek_tensor *t, const aiednn_memory_desc_t *md, const void *data) {
    t->data = (void *)data;
    t->ndim = md->ndims;
    t->code = md->data_type == aiednn_u8 ? AIEK_UINT : md->data_type == aiednn_f32 ? AIEK_FLOAT : AIEK_INT;
    t->bits = (uint8_t)(dt_bytes(md->data_type) * 8);
    t->lanes = 1;
    t->shape = md->dims;
    t->nbytes = (uint64_t)md_numel(md) * (uint64_t)dt_bytes(md->data_type);
}

/* run another implementation on a sub-problem (composites) */
static int run_sub(const rt_impl *impl, const rt_prob *p, const rt_bind *bind, void *const *ptr) {
    aiednn_primitive sub;
    memset(&sub, 0, sizeof sub);
    sub.prob = *p;
    sub.impl = impl;
    if (bind)
        sub.bind = *bind;
    const uint64_t t0 = rt_ticks();
    const int rc = impl->run(impl, &sub, ptr);
    ((rt_impl *)impl)->t_total += rt_ticks() - t0;
    ((rt_impl *)impl)->execs++;
    return rc;
}

static void md2(aiednn_memory_desc_t *md, int64_t a, int64_t b, aiednn_data_type_t dt) {
    memset(md, 0, sizeof *md);
    md->ndims = 2;
    md->dims[0] = a;
    md->dims[1] = b;
    md->data_type = dt;
}

/* the [M,K] x [N,K]^T -> s32 sub-problem composites hand to the matmul dispatch */
static void gemm_prob(rt_prob *g, int64_t M, int64_t N, int64_t K, aiednn_data_type_t a_dt) {
    memset(g, 0, sizeof *g);
    g->kind = aiednn_matmul;
    g->nmd = 3;
    g->arg[0] = AIEDNN_ARG_SRC;
    g->arg[1] = AIEDNN_ARG_WEIGHTS;
    g->arg[2] = AIEDNN_ARG_DST;
    md2(&g->md[0], M, K, a_dt);
    md2(&g->md[1], N, K, aiednn_s8);
    md2(&g->md[2], M, N, aiednn_s32);
    g->transposed = 1;
}

/* ── the fused epilogue of the GEMM family ───────────────────────────────────
 * Every implementation of convolution / deconvolution / matmul / inner product --
 * AIE composites and ref: alike -- ends in this, so an AIE result and the ref:
 * result agree bit for bit whenever the accumulators do. acc: the exact
 * accumulator of dst element di, whose output channel is ch. */
/* a binary post-op operand: an s8 / u8 src1 with alpha > 0 is quantized, real = (code - beta) * alpha
 * (aiednn.h aiednn_post_op_t); otherwise the stored value */
static inline double rt_po_src1(aiednn_data_type_t dt, const void *src, int64_t i, double alpha, double beta) {
    const double v = rt_ld(dt, src, i);
    return (dt == aiednn_s8 || dt == aiednn_u8) && alpha > 0 ? (v - beta) * alpha : v;
}

static void rt_epilogue(const rt_prob *p, void *const *ptr, int64_t di, int64_t ch, double acc) {
    const aiednn_memory_desc_t *d = &p->md[2];
    double v = acc;
    if (p->n_oscales)
        v *= (double)p->oscales[p->n_oscales == 1 ? 0 : ch];
    const int bs = rt_slot(p, AIEDNN_ARG_BIAS);
    if (bs >= 0) {
        const aiednn_memory_desc_t *b = &p->md[bs];
        v += rt_ld(b->data_type, ptr[bs], b->ndims == d->ndims ? bcast_off(b, d, di) : ch);
    }
    for (int k = 0; k < p->npo; k++) {
        const aiednn_post_op_t *po = &p->po[k];
        if (po->kind == aiednn_post_op_eltwise) {
            v = eltwise_f(po->alg, v, po->alpha, po->beta);
        } else {
            const int s = rt_slot(p, AIEDNN_ARG_ATTR_POST_OP_SRC(k));
            v = binary_f(po->alg, v,
                         rt_po_src1(po->src1.data_type, ptr[s], bcast_off(&po->src1, d, di), po->alpha, po->beta));
        }
    }
    if (dt_is_int(d->data_type) && p->has_attr)
        v = v / (double)p->dst_scale + (double)p->dst_zp;
    rt_st(d->data_type, ptr[2], di, v);
}

/* rt_epilogue with everything per-call resolved once (composites call it per
 * output element): bias / post-op source slots found, and a source whose shape is
 * dst's indexes with di directly instead of a per-dimension div/mod. The double
 * operations and their order are rt_epilogue's, so the two agree bit for bit. */
typedef struct {
    const rt_prob *p;
    void *const *ptr;
    const aiednn_memory_desc_t *d;
    void *dst;
    const float *osc;
    int64_t nosc;
    const void *bias;
    const aiednn_memory_desc_t *bmd;
    int bias_mode; /* 0 none, 1 per channel, 2 same shape as dst, 3 broadcast */
    int npo;
    struct {
        int kind, alg;
        double a, b;
        const void *src;
        const aiednn_memory_desc_t *md;
        int same;
        int perch; /* binary src1 is one value per output channel (1 everywhere else): index ch */
    } po[AIEDNN_MAX_POST_OPS];
    int requant;
    double dscale, dzp;
    int fast;            /* rt_epi_row's tight loop applies: bias none / per channel, eltwise post-ops only */
    const rt_actq *actq; /* one GELU / swish post-op -> s8 / u8: the step table, NULL = the formula */
    /* fixed-point requant (rt_fx_*): the integer epilogue the AIE core runs too */
    int fx;
    const int64_t *fx_b;  /* per output channel: rint(bias / dst_scale * 2^sh) + 2^(sh-1) */
    const int32_t *fx_m;  /* per output channel: multiplier in [2^30, 2^31) */
    const int8_t *fx_sh;  /* per output channel: right shift in [1, 62] */
    int32_t fx_lo, fx_hi; /* final clamp: dst type range intersected with the activation */
    const void *fx_r;     /* fused residual add (s8 / u8, dst's shape) or NULL */
    aiednn_data_type_t fx_rdt;
    int32_t fx_rzp;
    const int64_t *fx_rm;   /* per output channel: rint(src1_scale / dst_scale * 2^sh) */
    const int64_t *acc_add; /* rt_prob.acc_add: folded into fx_b, else added to acc */
} rt_epi;

static int same_dims(const aiednn_memory_desc_t *a, const aiednn_memory_desc_t *b) {
    if (a->ndims != b->ndims)
        return 0;
    for (int i = 0; i < a->ndims; i++)
        if (a->dims[i] != b->dims[i])
            return 0;
    return 1;
}

/* ── fixed-point requantization ────────────────────────────────────────────
 * For an int8 / uint8 destination with output scales, bias none or per channel and at
 * most one post-op that is ReLU (alpha 0) or clip, the epilogue is integer:
 *   q = clamp(((int64)acc * M[ch] + B[ch]) >> sh[ch] + zp, lo, hi)
 * with M * 2^-sh = scales[ch] / dst_scale (M in [2^30, 2^31)), B = rint(bias / dst_scale *
 * 2^sh) + 2^(sh-1) (bias at full precision plus the rounding half) and lo / hi the dst range
 * intersected with the activation's quantized bounds. It is the DEFINITION of the result for
 * these primitives -- ref: and the AIE core both compute it, so the on-device AIE-vs-ref check
 * stays bit-exact. Within one code of the double formula. A fused residual add whose src1 is
 * s8 / u8 (dst's shape, scale alpha, zero point beta) adds (r - beta) * R[ch] to the sum,
 * R = rint(alpha / dst_scale * 2^sh); the activation clamp then applies to acc + bias + residual. Do NOT round the bias
 * to accumulator units (rint(bias / scales)): that is off by up to m/2 codes per output, and the flips compound --
 * mobilenet_v3_small int8 went from rel 2e-7 to 1.8e-1. */
static int rt_fx_mult(double m, int32_t *M, int8_t *sh) {
    if (m != m || m < 0)
        return 0;
    int e = 0;
    const double f = frexp(m, &e); /* m = f * 2^e, f in [0.5, 1) */
    int64_t q = (int64_t)nearbyint(f * 2147483648.0);
    if (q == 2147483648LL) {
        q /= 2;
        e++;
    }
    const int s = 31 - e;
    if (m == 0 || s > 62) {
        /* a dead channel (zero weight scale) or m < 2^-31: |acc * m| < 1 for every int32 acc, so the
         * output is the rounded bias (+ zp). M = 0 says that exactly; before, one such channel sent
         * a whole conv to the host epilogue (resnet18's stem: 802,816 outputs on the A72) */
        *M = 0;
        *sh = 31;
        return 1;
    }
    if (s < 1)
        return 0;
    *M = (int32_t)q;
    *sh = (int8_t)s;
    return 1;
}

#define RT_FX_MAX_DROP 8 /* bits of M a lowered shift may give up (M keeps >= 23) */
static inline int rt_fx_min(int a, int b) { return a < b ? a : b; }
/* the largest shift s with x * 2^s <= lim (x 0: no limit) */
static inline int rt_fx_cap(double x, double lim) {
    if (!(x > 0))
        return 127;
    const double s = floor(log2(lim / x));
    return s > 126 ? 127 : s < -126 ? -127 : (int)s;
}

/* q = clamp((acc * M + B) >> sh + zp): B carries the bias at full precision and the rounding
 * half, so the bias is not rounded to accumulator units (that cost +-m/2 codes when the
 * accumulator step m = osc/dst_scale is coarse). */
static inline int32_t rt_fx_q(int64_t acc, int64_t B, int32_t M, int8_t sh, int32_t zp, int32_t lo, int32_t hi) {
    int64_t q = ((acc * (int64_t)M + B) >> sh) + zp;
    return (int32_t)(q < lo ? lo : q > hi ? hi : q);
}

static void *g_fxbuf;
static size_t g_fxbuf_cap;
static uint64_t g_fx_execs, g_epi_execs; /* epilogues run integer / in total (aiednn_phase) */
static uint64_t g_fx_core;               /* (image, group) GEMMs whose requant ran on the AIE core */

static void rt_fx_init(rt_epi *e, const rt_prob *p, int64_t C) {
#if defined(AIEDNN_NO_FX) /* A/B: the double epilogue everywhere */
    return;
#endif
    const aiednn_data_type_t dt = e->d->data_type;
    if ((dt != aiednn_s8 && dt != aiednn_u8) || !p->has_attr || !e->nosc || e->bias_mode > 1)
        return;
    /* post-ops: [residual add of a quantized dst-shaped src1] then [ReLU (alpha 0) | clip] */
    int k = 0, res = 0;
    if (e->npo > k && e->po[k].kind == aiednn_post_op_binary) {
        const aiednn_data_type_t rdt = e->po[k].md->data_type;
        if (e->po[k].alg != aiednn_binary_add || (rdt != aiednn_s8 && rdt != aiednn_u8) || !(e->po[k].a > 0) ||
            !e->po[k].same)
            return;
        res = 1, k++;
    }
    double plo = -1e300, phi = 1e300; /* activation bounds in the real (pre-requant) domain */
    /* any run of ReLU (alpha 0) / clip is one clamp: clamp(clamp(v, a1, b1), a2, b2) = clamp(v, lo', hi') with
     * lo' = clamp(a1, a2, b2), hi' = clamp(b1, a2, b2) -- the same function. A fusion that keeps a folded
     * requantize's saturation as a clip before the activation (graph fusion level 2: clip, clip(0, 6)) used to
     * send every such conv / depthwise to the double epilogue on the A72 (mobilenet_v2: 318 ms) */
    for (; k < e->npo && e->po[k].kind == aiednn_post_op_eltwise; k++) {
        double a, b;
        if (e->po[k].alg == aiednn_eltwise_relu && e->po[k].a == 0)
            a = 0, b = 1e300;
        else if (e->po[k].alg == aiednn_eltwise_clip)
            a = e->po[k].a, b = e->po[k].b;
        else
            return;
        if (a > b)
            return; /* an empty clip: not a clamp */
        const double nlo = plo < a ? a : plo > b ? b : plo, nhi = phi < a ? a : phi > b ? b : phi;
        plo = nlo, phi = nhi;
    }
    if (k != e->npo || C < 1)
        return;
    size_t need = (size_t)C * (2 * sizeof(int64_t) + sizeof(int32_t) + 1);
    char *buf = (char *)scratch(&g_fxbuf, &g_fxbuf_cap, need);
    if (!buf)
        return;
    int64_t *B = (int64_t *)buf, *RM = B + C;
    int32_t *M = (int32_t *)(RM + C);
    int8_t *SH = (int8_t *)(M + C);
    /* shifts: the full-precision one per channel (rt_fx_mult), then lowered where the int64 sums would overflow
     * (a bias or residual scale large against a tiny accumulator step: the shift is 40-52 there) and to within
     * 14 of the smallest live one (the core's vector requant). Lowering a shift by d keeps 31 - d bits of M;
     * at most RT_FX_MAX_DROP. resnet50: 10 of 53 convs used to fall back to the double epilogue on the host. */
    int8_t shlo = 127;
    for (int64_t ch = 0; ch < C; ch++) {
        if (!rt_fx_mult((double)e->osc[e->nosc == 1 ? 0 : ch] / e->dscale, &M[ch], &SH[ch]))
            return;
        const double bias = (e->bias_mode ? rt_ld(e->bmd->data_type, e->bias, ch) : 0.0) / e->dscale;
        int cap = rt_fx_cap(fabs(bias), 2.0e18);
        if (e->acc_add)
            cap = rt_fx_min(
                cap,
                rt_fx_cap(fabs((double)e->acc_add[ch]) * (double)e->osc[e->nosc == 1 ? 0 : ch] / e->dscale, 2.0e17));
        if (res)
            cap = rt_fx_min(cap, rt_fx_cap(fabs(e->po[0].a / e->dscale), 4.0e15));
        if (SH[ch] > cap) {
            if (M[ch] && SH[ch] - cap > RT_FX_MAX_DROP)
                return;
            SH[ch] = (int8_t)(cap < 1 ? 1 : cap);
        }
        if (M[ch] && SH[ch] < shlo)
            shlo = SH[ch];
    }
    for (int64_t ch = 0; ch < C; ch++) {
        const double osc = (double)e->osc[e->nosc == 1 ? 0 : ch];
        if (!M[ch]) /* a dead channel: any shift; the live ones' smallest keeps the spread */
            SH[ch] = shlo < 127 && shlo < SH[ch] ? shlo : SH[ch];
        else if (shlo < 127 && SH[ch] > shlo + 14) {
            if (SH[ch] - shlo - 14 > RT_FX_MAX_DROP)
                return;
            SH[ch] = (int8_t)(shlo + 14);
        }
        if (M[ch]) { /* M at the final shift (unchanged where the shift is) */
            const double mq = nearbyint(osc / e->dscale * ldexp(1.0, SH[ch]));
            if (!(mq >= 1) || mq >= 2147483648.0)
                return;
            M[ch] = (int32_t)mq;
        }
        const double b =
            nearbyint((e->bias_mode ? rt_ld(e->bmd->data_type, e->bias, ch) : 0.0) / e->dscale * ldexp(1.0, SH[ch]));
        if (b != b || fabs(b) > 2.3e18) /* acc * M + B + residual stays inside int64 */
            return;
        B[ch] = (int64_t)b + ((int64_t)1 << (SH[ch] - 1));
        if (e->acc_add) { /* (acc + add) * M = acc * M + add * M: exact, also for the AIE core's requant */
            const double am = (double)e->acc_add[ch] * (double)M[ch];
            if (fabs(am) + fabs(b) > 2.3e18)
                return;
            B[ch] += e->acc_add[ch] * (int64_t)M[ch];
        }
        if (res) {
            const double rm = nearbyint(e->po[0].a / e->dscale * ldexp(1.0, SH[ch]));
            if (rm != rm || fabs(rm) > 4.5e15) /* |r - zr| <= 255 */
                return;
            RM[ch] = (int64_t)rm;
        }
    }
    int8_t shhi = 0;
    shlo = 127;
    for (int64_t ch = 0; ch < C; ch++)
        shlo = SH[ch] < shlo ? SH[ch] : shlo, shhi = SH[ch] > shhi ? SH[ch] : shhi;
    if (shhi - shlo > 14) /* the core's vector requant (aie_modes_gemmq.h): 2^(spread) must fit int16 */
        return;
    const double tlo = dt == aiednn_s8 ? -128 : 0, thi = dt == aiednn_s8 ? 127 : 255;
    const double qlo = plo <= -1e300 ? tlo : floor(plo / e->dscale + 0.5) + e->dzp;
    const double qhi = phi >= 1e300 ? thi : floor(phi / e->dscale + 0.5) + e->dzp;
    e->fx_lo = (int32_t)(qlo > tlo ? qlo : tlo);
    e->fx_hi = (int32_t)(qhi < thi ? qhi : thi);
    e->fx_m = M, e->fx_b = B, e->fx_sh = SH;
    if (res)
        e->fx_r = e->po[0].src, e->fx_rdt = e->po[0].md->data_type, e->fx_rzp = (int32_t)e->po[0].b, e->fx_rm = RM;
    e->fx = 1;
    g_fx_execs++;
}

static void rt_epi_init(rt_epi *e, const rt_prob *p, void *const *ptr) {
    memset(e, 0, sizeof *e);
    e->p = p;
    e->ptr = ptr;
    e->d = &p->md[2];
    e->dst = ptr[2];
    e->osc = p->oscales;
    e->nosc = p->n_oscales;
    const int cax = p->kind == aiednn_matmul || p->kind == aiednn_inner_product ? e->d->ndims - 1 : 1;
    const int bs = rt_slot(p, AIEDNN_ARG_BIAS);
    if (bs >= 0) {
        e->bias = ptr[bs];
        e->bmd = &p->md[bs];
        e->bias_mode = e->bmd->ndims != e->d->ndims ? 1 : same_dims(e->bmd, e->d) ? 2 : 3;
        if (e->bias_mode == 3) { /* [1,..,C,..,1] (a matmul's [1,1,1,N]): one value per channel, at offset ch */
            int perch = e->bmd->dims[cax] == e->d->dims[cax];
            for (int i = 0; i < e->bmd->ndims && perch; i++)
                perch = i == cax || e->bmd->dims[i] == 1;
            if (perch)
                e->bias_mode = 1;
        }
    }
    e->npo = p->npo;
    for (int k = 0; k < p->npo; k++) {
        const aiednn_post_op_t *po = &p->po[k];
        e->po[k].kind = po->kind;
        e->po[k].alg = po->alg;
        e->po[k].a = po->alpha;
        e->po[k].b = po->beta;
        if (po->kind != aiednn_post_op_eltwise) {
            e->po[k].src = ptr[rt_slot(p, AIEDNN_ARG_ATTR_POST_OP_SRC(k))];
            e->po[k].md = &po->src1;
            e->po[k].same = same_dims(&po->src1, e->d);
        }
    }
    e->requant = dt_is_int(e->d->data_type) && p->has_attr;
    e->dscale = (double)p->dst_scale;
    e->dzp = (double)p->dst_zp;
    const aiednn_memory_desc_t *d = e->d;
    const int64_t C = d->dims[cax];
    e->fast = e->bias_mode <= 1;
    for (int k = 0; k < e->npo; k++) {
        if (e->po[k].kind == aiednn_post_op_eltwise)
            continue;
        const aiednn_memory_desc_t *m = e->po[k].md;
        int perch = m->ndims == d->ndims && m->dims[cax] == C;
        for (int i = 0; i < m->ndims && perch; i++)
            perch = i == cax || m->dims[i] == 1;
        e->po[k].perch = perch && !e->po[k].same;
        if (!e->po[k].same && !e->po[k].perch) /* any other broadcast: the general path */
            e->fast = 0;
    }
    if (e->fast && e->npo == 1 && e->po[0].kind == aiednn_post_op_eltwise && e->requant &&
        (d->data_type == aiednn_s8 || d->data_type == aiednn_u8) &&
        (e->po[0].alg == aiednn_eltwise_gelu_erf || e->po[0].alg == aiednn_eltwise_gelu_tanh ||
         e->po[0].alg == aiednn_eltwise_swish)) {
        const int u8 = d->data_type == aiednn_u8, al = e->po[0].alg;
        e->actq = rt_actq_get(al == aiednn_eltwise_gelu_erf    ? RT_ACTQ_GELU
                              : al == aiednn_eltwise_gelu_tanh ? RT_ACTQ_GELU_TANH
                                                               : RT_ACTQ_SWISH,
                              e->po[0].a, e->dscale, e->dzp, u8 ? 0.0 : -128.0, u8 ? 255.0 : 127.0);
    }
    e->acc_add = p->acc_add;
    g_epi_execs++;
    rt_fx_init(e, p, C);
}

static inline void rt_epi_put(const rt_epi *e, int64_t di, int64_t ch, double acc) {
    if (e->fx) { /* integer requant: the definition for these primitives (rt_fx_init) */
        int64_t B = e->fx_b[ch];
        if (e->fx_r) /* fused residual: + (r - zr) * R, at the same 2^sh scale as acc * M */
            B += ((int64_t)(e->fx_rdt == aiednn_s8 ? ((const int8_t *)e->fx_r)[di] : ((const uint8_t *)e->fx_r)[di]) -
                  e->fx_rzp) *
                 e->fx_rm[ch];
        const int32_t q = rt_fx_q((int64_t)acc, B, e->fx_m[ch], e->fx_sh[ch], (int32_t)e->dzp, e->fx_lo, e->fx_hi);
        if (e->d->data_type == aiednn_s8)
            ((int8_t *)e->dst)[di] = (int8_t)q;
        else
            ((uint8_t *)e->dst)[di] = (uint8_t)q;
        return;
    }
    if (e->acc_add)
        acc += (double)e->acc_add[ch];
#if defined(AIEDNN_EPI_SLOW) /* A/B check: the per-element path */
    rt_epilogue(e->p, e->ptr, di, ch, acc);
    return;
#endif
    double v = acc;
    if (e->nosc)
        v *= (double)e->osc[e->nosc == 1 ? 0 : ch];
    if (e->bias_mode)
        v += rt_ld(e->bmd->data_type, e->bias,
                   e->bias_mode == 1   ? ch
                   : e->bias_mode == 2 ? di
                                       : bcast_off(e->bmd, e->d, di));
    for (int k = 0; k < e->npo; k++) {
        if (e->po[k].kind == aiednn_post_op_eltwise)
            v = eltwise_f(e->po[k].alg, v, e->po[k].a, e->po[k].b);
        else
            v = binary_f(e->po[k].alg, v,
                         rt_po_src1(e->po[k].md->data_type, e->po[k].src,
                                    e->po[k].same ? di : bcast_off(e->po[k].md, e->d, di), e->po[k].a, e->po[k].b));
    }
    if (e->requant)
        v = v / e->dscale + e->dzp;
    rt_st(e->d->data_type, e->dst, di, v);
}

/* n outputs: dst[di0 + i] from acc[i * acc_stride], channel ch0 (+ i if ch_step).
 * The fast loop performs rt_epilogue's double operations in the same order (scale,
 * bias, relu / clip, requant, round-to-nearest-even + saturate), only with the
 * per-call and per-channel lookups hoisted: bit-identical (AIEDNN_EPI_SLOW checks). */
static void rt_epi_row(const rt_epi *e, int64_t di0, int64_t ch0, int ch_step, const int32_t *acc, int64_t acc_stride,
                       int64_t n) {
    if (e->fx && !e->fx_r && ch_step && acc_stride == 1) { /* a matmul row: rt_fx_q per channel, nothing else */
        const int64_t *B = e->fx_b + ch0;
        const int32_t *M = e->fx_m + ch0;
        const int8_t *sh = e->fx_sh + ch0;
        const int32_t zp = (int32_t)e->dzp, lo = e->fx_lo, hi = e->fx_hi;
        uint8_t *o = (uint8_t *)e->dst + di0; /* s8 / u8 codes: the same byte */
        for (int64_t i = 0; i < n; i++)
            o[i] = (uint8_t)rt_fx_q((int64_t)acc[i], B[i], M[i], sh[i], zp, lo, hi);
        return;
    }
    if (e->fx) {
        for (int64_t i = 0; i < n; i++)
            rt_epi_put(e, di0 + i, ch0 + (ch_step ? i : 0), (double)acc[i * acc_stride]);
        return;
    }
    if (e->acc_add) { /* corrected accumulators, then the loops below unchanged */
        rt_epi e2 = *e;
        e2.acc_add = nullptr;
        int32_t buf[256];
        for (int64_t i0 = 0; i0 < n; i0 += 256) {
            const int64_t m = n - i0 < 256 ? n - i0 : 256;
            for (int64_t i = 0; i < m; i++)
                buf[i] = acc[(i0 + i) * acc_stride] + (int32_t)e->acc_add[ch0 + (ch_step ? i0 + i : 0)];
            rt_epi_row(&e2, di0 + i0, ch0 + (ch_step ? i0 : 0), ch_step, buf, 1, m);
        }
        return;
    }
#if !defined(AIEDNN_EPI_SLOW)
    /* an int8 linear layer into f32 (every int8 transformer projection): per-channel scales, an f32 bias or
     * none, no post-ops -- the generic loop's double operations without its per-element dispatch */
    if (e->fast && ch_step && acc_stride == 1 && e->nosc > 1 && e->npo == 0 && !e->requant &&
        e->d->data_type == aiednn_f32 && (!e->bias_mode || (e->bias_mode == 1 && e->bmd->data_type == aiednn_f32))) {
        const float *osc = e->osc + ch0, *bs = e->bias_mode ? (const float *)e->bias + ch0 : nullptr;
        float *o = (float *)e->dst + di0;
        if (bs)
            for (int64_t i = 0; i < n; i++)
                o[i] = (float)((double)acc[i] * (double)osc[i] + (double)bs[i]);
        else
            for (int64_t i = 0; i < n; i++)
                o[i] = (float)((double)acc[i] * (double)osc[i]);
        return;
    }
    /* the same with one residual add of a tensor shaped like dst (o / down projections + the skip): f32, or
     * s8 / u8 dequantized by the post-op's scale and zero point (rt_po_src1; the int8 skips of
     * --aiednn-quantize-residual) */
    if (e->fast && ch_step && acc_stride == 1 && e->nosc > 1 && e->npo == 1 && !e->requant &&
        e->d->data_type == aiednn_f32 && (!e->bias_mode || (e->bias_mode == 1 && e->bmd->data_type == aiednn_f32)) &&
        e->po[0].kind == aiednn_post_op_binary && e->po[0].alg == aiednn_binary_add && e->po[0].same &&
        (e->po[0].md->data_type == aiednn_f32 || e->po[0].md->data_type == aiednn_s8 ||
         e->po[0].md->data_type == aiednn_u8)) {
        const float *osc = e->osc + ch0, *bs = e->bias_mode ? (const float *)e->bias + ch0 : nullptr;
        const aiednn_data_type_t rdt = e->po[0].md->data_type;
        const double ra = e->po[0].a, rb = e->po[0].b;
        float *o = (float *)e->dst + di0;
        for (int64_t i = 0; i < n; i++) {
            double v = (double)acc[i] * (double)osc[i];
            if (bs)
                v += (double)bs[i];
            const double r = rdt == aiednn_f32 ? (double)((const float *)e->po[0].src)[di0 + i]
                                               : rt_po_src1(rdt, e->po[0].src, di0 + i, ra, rb);
            o[i] = (float)(v + r);
        }
        return;
    }
    /* the two hot row shapes of a transformer-style MLP (convnext), channel per element, per-channel
     * scales and f32 bias: the same double operations as below, no per-element dispatch */
    if (e->fast && ch_step && acc_stride == 1 && e->nosc > 1 && e->npo == 1 &&
        (!e->bias_mode || e->bmd->data_type == aiednn_f32)) {
        const float *osc = e->osc + ch0, *bs = e->bias_mode ? (const float *)e->bias + ch0 : nullptr;
        const aiednn_data_type_t dt = e->d->data_type;
        if (e->po[0].kind == aiednn_post_op_eltwise && e->po[0].alg == aiednn_eltwise_gelu_erf && e->requant &&
            dt == aiednn_s8) { /* GELU -> int8 */
            const double ds = e->dscale, zp = e->dzp;
            int8_t *o = (int8_t *)e->dst + di0;
            if (e->actq) { /* the same codes, by table (aiednn_rt_geluq.h) */
                const rt_actq *g = e->actq;
                for (int64_t i = 0; i < n; i++) {
                    double v = (double)acc[i] * (double)osc[i];
                    if (bs)
                        v += (double)bs[i];
                    o[i] = (int8_t)rt_actq_code(g, v);
                }
                return;
            }
            for (int64_t i = 0; i < n; i++) {
                double v = (double)acc[i] * (double)osc[i];
                if (bs)
                    v += (double)bs[i];
                v = v * rt_phi(v);
                v = v / ds + zp;
                v = v != v ? 0.0 : nearbyint(v);
                o[i] = (int8_t)(v < -128.0 ? -128.0 : v > 127.0 ? 127.0 : v);
            }
            return;
        }
        if (e->po[0].kind == aiednn_post_op_binary && e->po[0].perch && e->po[0].alg == aiednn_binary_mul &&
            e->po[0].md->data_type == aiednn_f32 && !e->requant && dt == aiednn_f32) { /* layer scale -> f32 */
            const float *g = (const float *)e->po[0].src + ch0;
            float *o = (float *)e->dst + di0;
            for (int64_t i = 0; i < n; i++) {
                double v = (double)acc[i] * (double)osc[i];
                if (bs)
                    v += (double)bs[i];
                o[i] = (float)(v * (double)g[i]);
            }
            return;
        }
    }
    if (e->fast) {
        const aiednn_data_type_t dt = e->d->data_type;
        const double lo = dt == aiednn_s8 ? -128.0 : dt == aiednn_u8 ? 0.0 : -2147483648.0;
        const double hi = dt == aiednn_s8 ? 127.0 : dt == aiednn_u8 ? 255.0 : 2147483647.0;
        double sc = 0, bi = 0;
        if (e->actq && !ch_step) { /* one channel (an NCHW plane row): rt_actq_code inlined, its lookups hoisted */
            const rt_actq *g = e->actq;
            const rt_actq_cell *cells = g->cell;
            const double ya = g->ya, ih = g->inv_h, ncl = (double)g->ncell;
            const double s1 = e->nosc ? (double)e->osc[e->nosc == 1 ? 0 : ch0] : 1.0;
            const double b1 = e->bias_mode ? rt_ld(e->bmd->data_type, e->bias, ch0) : 0.0;
            const int has_s = e->nosc != 0, has_b = e->bias_mode != 0;
            uint8_t *o = (uint8_t *)e->dst + di0; /* s8 / u8 codes: the same byte */
            for (int64_t i = 0; i < n; i++) {
                double y = (double)acc[i * acc_stride];
                if (has_s)
                    y *= s1;
                if (has_b)
                    y += b1;
                double d = (y - ya) * ih;
                d = d < -1.0 ? -1.0 : d > ncl ? ncl : d;
                int q;
                if (d != d) {
                    q = rt_actq_direct(g->kind, g->a, y, g->ds, g->zp, g->lo, g->hi);
                } else {
                    const rt_actq_cell *c = &cells[1 + (int)d];
                    q = fabs(y - c->t) <= c->w ? rt_actq_direct(g->kind, g->a, y, g->ds, g->zp, g->lo, g->hi)
                                               : (y >= c->t ? c->c1 : c->c0);
                }
                o[i] = (uint8_t)q;
            }
            return;
        }
        if (e->actq && ch_step && acc_stride == 1 && (!e->bias_mode || e->bmd->data_type == aiednn_f32)) {
            /* a matmul row into GELU / swish -> s8 / u8: rt_actq_code inlined, scales and f32 bias by channel */
            const rt_actq *g = e->actq;
            const rt_actq_cell *cells = g->cell;
            const double ya = g->ya, ih = g->inv_h, ncl = (double)g->ncell;
            const float *osc = e->nosc ? e->osc + (e->nosc == 1 ? 0 : ch0) : nullptr;
            const int64_t ost = e->nosc > 1 ? 1 : 0;
            const float *bs = e->bias_mode ? (const float *)e->bias + ch0 : nullptr;
            uint8_t *o = (uint8_t *)e->dst + di0;
            for (int64_t i = 0; i < n; i++) {
                double y = (double)acc[i];
                if (osc)
                    y *= (double)osc[i * ost];
                if (bs)
                    y += (double)bs[i];
                double d = (y - ya) * ih;
                d = d < -1.0 ? -1.0 : d > ncl ? ncl : d;
                int q;
                if (d != d) {
                    q = rt_actq_direct(g->kind, g->a, y, g->ds, g->zp, g->lo, g->hi);
                } else {
                    const rt_actq_cell *c = &cells[1 + (int)d];
                    q = fabs(y - c->t) <= c->w ? rt_actq_direct(g->kind, g->a, y, g->ds, g->zp, g->lo, g->hi)
                                               : (y >= c->t ? c->c1 : c->c0);
                }
                o[i] = (uint8_t)q;
            }
            return;
        }
        if (e->actq) { /* one GELU / swish -> s8 / u8: the same codes, by table (aiednn_rt_geluq.h) */
            for (int64_t i = 0; i < n; i++) {
                const int64_t ch = ch0 + (ch_step ? i : 0);
                if (i == 0 || ch_step) {
                    if (e->nosc)
                        sc = (double)e->osc[e->nosc == 1 ? 0 : ch];
                    if (e->bias_mode)
                        bi = rt_ld(e->bmd->data_type, e->bias, ch);
                }
                double v = (double)acc[i * acc_stride];
                if (e->nosc)
                    v *= sc;
                if (e->bias_mode)
                    v += bi;
                const int q = rt_actq_code(e->actq, v);
                if (dt == aiednn_s8)
                    ((int8_t *)e->dst)[di0 + i] = (int8_t)q;
                else
                    ((uint8_t *)e->dst)[di0 + i] = (uint8_t)q;
            }
            return;
        }
        for (int64_t i = 0; i < n; i++) {
            const int64_t ch = ch0 + (ch_step ? i : 0);
            if (i == 0 || ch_step) {
                if (e->nosc)
                    sc = (double)e->osc[e->nosc == 1 ? 0 : ch];
                if (e->bias_mode)
                    bi = rt_ld(e->bmd->data_type, e->bias, ch);
            }
            double v = (double)acc[i * acc_stride];
            if (e->nosc)
                v *= sc;
            if (e->bias_mode)
                v += bi;
            for (int k = 0; k < e->npo; k++) { // relu / clip / GELU inline, any other eltwise through eltwise_f
                if (e->po[k].kind != aiednn_post_op_eltwise) { /* same-shape or per-channel src1: no index math */
                    v = binary_f(e->po[k].alg, v,
                                 rt_po_src1(e->po[k].md->data_type, e->po[k].src, e->po[k].same ? di0 + i : ch,
                                            e->po[k].a, e->po[k].b));
                    continue;
                }
                v = e->po[k].alg == aiednn_eltwise_relu       ? (v > 0 ? v : e->po[k].a * v)
                    : e->po[k].alg == aiednn_eltwise_clip     ? (v < e->po[k].a   ? e->po[k].a
                                                                 : v > e->po[k].b ? e->po[k].b
                                                                                  : v)
                    : e->po[k].alg == aiednn_eltwise_gelu_erf ? v * rt_phi(v)
                                                              : eltwise_f(e->po[k].alg, v, e->po[k].a, e->po[k].b);
            }
            if (e->requant)
                v = v / e->dscale + e->dzp;
            const int64_t di = di0 + i;
            if (dt == aiednn_f32) {
                ((float *)e->dst)[di] = (float)v;
                continue;
            }
            v = v != v ? 0.0 : nearbyint(v);
            v = v < lo ? lo : v > hi ? hi : v;
            if (dt == aiednn_s8)
                ((int8_t *)e->dst)[di] = (int8_t)v;
            else if (dt == aiednn_u8)
                ((uint8_t *)e->dst)[di] = (uint8_t)v;
            else
                ((int32_t *)e->dst)[di] = (int32_t)v;
        }
        return;
    }
#endif
    for (int64_t i = 0; i < n; i++)
        rt_epi_put(e, di0 + i, ch0 + (ch_step ? i : 0), (double)acc[i * acc_stride]);
}

#endif /* AIEDNN_RT_INTERNAL_H */
