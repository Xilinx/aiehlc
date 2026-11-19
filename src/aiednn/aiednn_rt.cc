/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn runtime: descriptors, validation, dispatch over an implementation
 * list, execute. The implementations live in aiednn_rt_gemm.h (GEMM family +
 * AIE composites) and aiednn_rt_ops.h (everything else, ref:).
 *
 * The list (rt_impls) is built once from three sources, in this order:
 *   1. the generated aiednn_impl_list[] -- one aie:<kernel> per AIEK_SPEC,
 *      accepting exactly the dims the spec fixes (specificity = fixed dims);
 *   2. composites over those kernels (specificity 0):
 *        blocked:<aie gemm>  any s8 x s8 -> s32|s8 matmul (batched, bias,
 *                            either weight layout) tiled onto an AIE GEMM
 *                            block, K summed on the host in int32 -- exact;
 *        im2col+ / col2im+ / ip+  convolution / deconvolution / inner product
 *                            as GEMMs, accepted only when the matmul dispatch
 *                            for that GEMM lands on a non-ref implementation;
 *   3. ref: CPU implementations (specificity -1), so any valid problem runs.
 * Dispatch asks every implementation of the problem's kind; the most specific
 * acceptor wins and a tie is aiednn_unimplemented, never first-wins (the
 * list order does not decide).
 *
 * Spliced into the aiehlc entry translation unit by the builder
 * (aiegraph/dnn.py), with the implementation headers copied next to it: never
 * write the aiehlc kernel keyword or the launch chevrons here, even in a
 * comment. Also builds with a plain host g++ (x86 target, ctypes tests).
 ******************************************************************************/
#include "aiednn_rt_internal.h"
#include "aiednn_rt_phead.h" /* dequantized sources (attr.src_scales on layer norm / softmax / sdpa) */
#include "aiednn_rt_gemm.h"
#include "aiednn_rt_eager.h" /* eager mode: one kernel instance per layer shape */
#include "aiednn_rt_ntile.h"
#include "aiednn_rt_dwn.h"
#include "aiednn_rt_itemsn.h"
#include "aie_modes_mmt.h"
#include "aiednn_rt_mmt.h"
#include "aiednn_rt_ops.h"
#include "aiednn_rt_bwd.h" /* training: backward passes, normalization training forward */
#include "aie_modes_pool.h"
#include "aiednn_rt_pool.h"
#include "aie_modes_lut.h"
#include "aiednn_rt_lut.h"
#include "aiednn_rt_eager_lut.h"  /* eager mode: table lookups on the lut template */
#include "aiednn_rt_eager_norm.h" /* eager mode: layer / RMS norm rows on the lnorm template */
#include "aiednn_rt_eltblk.h"
#include "aiednn_rt_red.h"
#include "aiednn_rt_steps.h"
#include "aie_modes_sdpa.h"
#include "aiednn_rt_sdpa.h"
#include "aie_modes_bf16.h"
#include "aiednn_rt_bf16.h"
#include "aiednn_rt_eager_float.h" /* eager mode: float matmul on the gemm_bf16 template */
#include "aiednn_rt_wq.h"
#include "aie_modes_fpool.h"
#include "aie_modes_fvec.h"
#include "aiednn_rt_fvec.h"

#define RT_MAX_IMPLS 256

static rt_impl g_impls[RT_MAX_IMPLS];
static int g_nimpls = -1;
static int g_verbose = 0;
static const char *g_filter = nullptr;

static const char *kind_str(aiednn_primitive_kind_t k) {
    static const char *names[] = {"?",
                                  "matmul",
                                  "convolution",
                                  "deconvolution",
                                  "inner_product",
                                  "eltwise",
                                  "binary",
                                  "reduction",
                                  "pooling",
                                  "resampling",
                                  "softmax",
                                  "batch_normalization",
                                  "layer_normalization",
                                  "group_normalization",
                                  "lrn",
                                  "prelu",
                                  "concat",
                                  "sum",
                                  "reorder",
                                  "shuffle",
                                  "sdpa",
                                  "rnn",
                                  "grouped_matmul",
                                  "dropout",
                                  "grid_sample",
                                  "ctc_loss",
                                  "block_quantize",
                                  "sparse_matmul",
                                  "slice",
                                  "gated_mlp",
                                  "rope",
                                  "gen_index",
                                  "dynamic_quantize",
                                  "linear_attention",
                                  "custom"};
    return (int)k >= 1 && (int)k < (int)(sizeof names / sizeof names[0]) ? names[k] : "?";
}

static void prob_str(const rt_prob *p, char *buf, int cap) {
    int n = 0;
    for (int a = 0; a < p->nmd && n < cap; a++) {
        n += snprintf(buf + n, cap - n, "%s%s:", a ? " " : "", dt_str(p->md[a].data_type));
        for (int i = 0; i < p->md[a].ndims && n < cap; i++)
            n += snprintf(buf + n, cap - n, i ? "x%lld" : "%lld", (long long)p->md[a].dims[i]);
    }
    if ((p->kind == aiednn_convolution || p->kind == aiednn_deconvolution) && n < cap)
        snprintf(buf + n, cap - n, " s%lldx%lld p%lldx%lld d%lldx%lld g%lld", (long long)p->conv.strides[0],
                 (long long)p->conv.strides[1], (long long)p->conv.padding_l[0], (long long)p->conv.padding_l[1],
                 (long long)p->conv.dilates[0], (long long)p->conv.dilates[1], (long long)p->conv.groups);
}

// ── the implementation list ────────────────────────────────────────────────

typedef int (*rt_acc_fn)(const rt_impl *, const rt_prob *, rt_bind *, char *, int);
typedef int (*rt_run_fn)(const rt_impl *, aiednn_primitive *, void *const *);

static int run_ref_rnn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);         /* aiednn_rt_rnn.h */
static int run_grouped(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);         /* aiednn_rt_grouped.h */
static int run_ref_dropout(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);     /* aiednn_rt_misc.h */
static int run_ref_grid_sample(const rt_impl *self, aiednn_primitive *prim, void *const *ptr); /* aiednn_rt_misc.h */
static int run_ref_ctc(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);         /* aiednn_rt_misc.h */
static int run_ref_blockq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);      /* aiednn_rt_mxq.h */
/* memory formats: aiednn_rt_layout.h */
static int rt_fmt_canon(rt_prob *p);
static void rt_fmt_strip(rt_prob *p);
static void rt_fmt_wrap_plan(aiednn_primitive *prim, const rt_prob *orig);
static int rt_fmt_in(aiednn_primitive *prim, void **ptr, void **user);
static void rt_fmt_out(aiednn_primitive *prim, void *const *ptr, void *const *user);
static int ptail_split(const rt_prob *p0, rt_prob *inner, rt_prob *ep); /* aiednn_rt_ptail.h */
static float *ptail_scratch(const aiednn_primitive *prim);
static int ptail_run(const aiednn_primitive *prim, const float *tile, void *dst, int nargs,
                     const aiednn_exec_arg_t *args);
static inline int md_fmt_free(const aiednn_memory_desc_t *m);
static void rt_fmt_convert(const aiednn_memory_desc_t *m, const void *src, void *dst, int to_nhwc);
static int acc_nhwc_pool(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_nhwc_pool(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_conv(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why,
                       int cap); /* aiednn_rt_gemmx.h */
static int run_gx_conv(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_deconv(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_deconv(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_ip(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_ip(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_sdpa(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_sdpa(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_zp(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int is_fpool_block(const aiednn_impl_t *g); /* aiednn_rt_fpool.h, after aiednn_rt_nd.h (RT_D_*) */
static int acc_fpool(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_fpool(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int run_gx_zp(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_mm(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_mm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_rnn(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_rnn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_rnnbwd(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_rnnbwd(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_sdpabwd(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_sdpabwd(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_sdpavar(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int acc_sp_densify(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why,
                          int cap); /* aiednn_rt_sparse.h */
static int run_sp_densify(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int run_ref_sparse_mm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int run_ref_slice(const rt_impl *self, aiednn_primitive *prim, void *const *ptr); /* aiednn_rt_slice.h */
static int run_ref_custom(const rt_impl *self, aiednn_primitive *prim, void *const *ptr); /* aiednn_rt_custom.h */
static int run_ref_gmlp(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);  /* aiednn_rt_fused.h */
static int acc_gx_gmlp(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_gmlp(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int run_ref_rope(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int run_ref_gen_index(const rt_impl *self, aiednn_primitive *prim, void *const *ptr); /* aiednn_rt_dynq.h */
static int run_ref_linattn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);   /* aiednn_rt_linattn.h */
static int acc_gx_linattn(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_linattn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_sdpa8(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why,
                        int cap); /* aiednn_rt_sdpa8.h */
static int run_gx_sdpa8(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int run_ref_dynq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_dynq_reorder(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_dynq_reorder(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_dcb(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why,
                   int cap); /* aiednn_rt_deconvbwd.h */
static int run_dcb(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_apb(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_apb(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_rsb(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_rsb(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_rsl(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_rsl(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int run_gx_sdpavar(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_dequant(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);
static int run_gx_dequant(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);
static int acc_gx_cvt(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap);

static void add_impl(const char *name, aiednn_primitive_kind_t kind, int spec, rt_acc_fn acc, rt_run_fn run,
                     const aiednn_impl_t *gen) {
    if (g_nimpls >= RT_MAX_IMPLS) { /* never silently: a dropped impl changes dispatch */
        static int warned;
        if (!warned++)
            printf("aiednn: ERROR: more than RT_MAX_IMPLS (%d) implementations; %s and later ones dropped\n",
                   RT_MAX_IMPLS, name);
        return;
    }
    rt_impl *r = &g_impls[g_nimpls++];
    memset(r, 0, sizeof *r);
    snprintf(r->name, sizeof r->name, "%s", name);
    r->kind = kind;
    r->specificity = spec;
    r->accepts = acc;
    r->run = run;
    r->gen = gen;
}

static void rt_init(void) {
    if (g_nimpls >= 0)
        return;
    g_nimpls = 0;
    char name[64];
    for (int i = 0; i < aiednn_impl_count; i++) {
        const aiednn_impl_t *g = &aiednn_impl_list[i];
        if (strncmp(g->name, "aie:eager_", 10)) /* an eager instance runs only through its eager: impl */
            add_impl(g->name, g->kind, g->specificity, acc_gen, run_gen, g);
        if (is_gemm_block(g) && strncmp(g->name, "aie:eager_", 10)) { /* eager instances: only their eager: impl */
            snprintf(name, sizeof name, "blocked:%s", g->name);
            add_impl(name, aiednn_matmul, 0, acc_blocked, run_blocked, g);
        }
        if (is_gemm_q_block(g))
            g_qblk = g; // GEMM + integer requant, used by im2col when the epilogue is integer
        if (is_dwq_block(g))
            g_dwqblk = g; // depthwise + requant, three items per core (run_dwblk with an integer epilogue)
        if (is_gemm_kc_block(g))
            g_kcblk = g; // K block of a GEMM chained on the core (persistent kernel keeps the sum)
        if (is_conv_kc_block(g))
            g_ckcblk = g; // direct conv: the cores gather the im2col tile from an input region (CONV)
        if (is_conv_q_block(g))
            g_cqblk = g; // direct conv + requant on the core
        if (is_mmt_block(g))
            g_mmtblk = g;       // short-K int8 conv on mlir-aie's matmul micro-kernel (aiednn_rt_mmt.h)
        if (is_pool_block(g)) { // max / avg pooling in 16-band launches of the AIE block (aiednn_rt_pool.h)
            snprintf(name, sizeof name, "poolblk:%s", g->name);
            add_impl(name, aiednn_pooling, 1, acc_poolblk, run_poolblk, g);
            snprintf(name, sizeof name, "redblk:%s", g->name); // reductions over trailing axes (aiednn_rt_red.h)
            add_impl(name, aiednn_reduction, 1, acc_redblk, run_redblk, g);
        }
        if (is_lut_block(g)) { // any per-channel function of an 8-bit value: table lookup (aiednn_rt_lut.h)
            static const aiednn_primitive_kind_t lut_kinds[] = {
                aiednn_eltwise, aiednn_prelu, aiednn_batch_normalization, aiednn_binary, aiednn_reorder};
            snprintf(name, sizeof name, "lut:%s", g->name);
            for (aiednn_primitive_kind_t lk : lut_kinds)
                add_impl(name, lk, 1, acc_lut, run_lut, g);
            snprintf(name, sizeof name, "steps:%s", g->name); // f32 / s32 -> 8-bit: step tables (aiednn_rt_steps.h)
            for (aiednn_primitive_kind_t lk : lut_kinds)
                add_impl(name, lk, 1, acc_steps, run_steps, g);
        }
        if (is_elt_block(g)) { // binary of two 8-bit tensors: exact int32 a op b (aiednn_rt_eltblk.h)
            snprintf(name, sizeof name, "eltblk:%s", g->name);
            add_impl(name, aiednn_binary, 1, acc_eltblk, run_eltblk, g);
        }
        if (is_bf16_block(g)) { // float matmul on the bf16 mode (aiednn_rt_bf16.h): bf16 x bf16, f32 as bf16x3
            snprintf(name, sizeof name, "bf16blk:%s", g->name);
            add_impl(name, aiednn_matmul, 1, acc_bf16blk, run_bf16blk, g);
            snprintf(name, sizeof name, "bf16x3:%s", g->name);
            add_impl(name, aiednn_matmul, 1, acc_bf16x3, run_bf16x3, g);
        }
        if (is_fpool_block(g)) { // f32 2-D / 3-D windowed pooling on the FPOOL mode (aiednn_rt_fpool.h)
            snprintf(name, sizeof name, "fpool:%s", g->name);
            add_impl(name, aiednn_pooling, 1, acc_fpool, run_fpool, g);
            g_impls[g_nimpls - 1].caps = RT_CAP_3D | RT_CAP_TRAIN; /* forward, and max backward (argmax on the AIE) */
        }
        if (is_wq_block(g)) { // 8-bit src x grouped 8 / 4-bit weights on the WQ mode (aiednn_rt_wq.h)
            snprintf(name, sizeof name, "wq:%s", g->name);
            add_impl(name, aiednn_matmul, 1, acc_wq, run_wq, g);
            g_impls[g_nimpls - 1].caps = RT_CAP_ZP | RT_CAP_WQ;
            add_impl(name, aiednn_inner_product, 1, acc_wq_ip, run_wq_ip, g);
            g_impls[g_nimpls - 1].caps = RT_CAP_ZP | RT_CAP_WQ;
        }
        if (is_fvec_block(g)) { // f32 pointwise / row ops on the FVEC mode (aiednn_rt_fvec.h)
            static const aiednn_primitive_kind_t fk[] = {aiednn_eltwise,
                                                         aiednn_binary,
                                                         aiednn_softmax,
                                                         aiednn_layer_normalization,
                                                         aiednn_reduction,
                                                         aiednn_pooling,
                                                         aiednn_prelu,
                                                         aiednn_batch_normalization,
                                                         aiednn_group_normalization};
            snprintf(name, sizeof name, "fvec:%s", g->name);
            for (aiednn_primitive_kind_t k : fk) { /* TRAIN: eltwise backward (acc_fvec gates the prop per kind) */
                /* pooling: global only (one reduction per row), more specific than a windowed pooling mode */
                add_impl(name, k, k == aiednn_pooling ? 2 : 1, acc_fvec, run_fvec, g);
                g_impls[g_nimpls - 1].caps = RT_CAP_TRAIN;
            }
        }
        if (is_sdpa_block(g)) { // attention per (batch, head, query block) on the SDPA mode (aiednn_rt_sdpa.h)
            snprintf(name, sizeof name, "sdpablk:%s", g->name);
            add_impl(name, aiednn_sdpa, 1, acc_sdpablk, run_sdpablk, g);
        }
        if (is_dw_block(g)) { // depthwise convolution in 16-band launches of the AIE block
            snprintf(name, sizeof name, "dwblk:%s", g->name);
            add_impl(name, aiednn_convolution, 1, acc_dwblk, run_dwblk, g);
            g_impls[g_nimpls - 1].caps = RT_CAP_NHWC; /* items gather / scatter nhwc channels (dw_item) */
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++)
        if (is_eager_inst(&aiednn_impl_list[i])) { /* eager mode: the instance of the layer's exact shape */
            add_impl("eager:mm", aiednn_matmul, 2, acc_eager_mm, run_blocked, nullptr);
            break;
        }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_q q;
        if (rt_eager_q_of(&aiednn_impl_list[i], &q)) { /* eager mode, requant on the core (gemm_q instances) */
            add_impl("eager:mmq", aiednn_matmul, 3, acc_eager_mmq, run_eager_mmq, nullptr);
            add_impl("eager:convq", aiednn_convolution, 3, acc_eager_convq, run_eager_convq, nullptr);
            break;
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_bf q;
        if (rt_eager_bf_of(&aiednn_impl_list[i], &q)) { /* eager mode: float matmul (gemm_bf16 instances) */
            add_impl("eager:bf16", aiednn_matmul, 3, acc_eager_bf16, run_eager_bf16, nullptr);
            add_impl("eager:bf16x3", aiednn_matmul, 3, acc_eager_bf16x3, run_eager_bf16x3, nullptr);
            break;
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_rope q;
        if (rt_eager_rope_of(&aiednn_impl_list[i], &q)) { /* eager mode: rotary position embedding (rope instances) */
            add_impl("eager:rope", aiednn_rope, 3, acc_eager_rope, run_eager_rope, nullptr);
            break;
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_lnorm q;
        if (rt_eager_lnorm_of(&aiednn_impl_list[i], &q)) { /* eager mode: layer / RMS norm rows (lnorm instances) */
            add_impl("eager:lnorm", aiednn_layer_normalization, 3, acc_eager_lnorm, run_eager_lnorm, nullptr);
            break;
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_quant q;
        if (rt_eager_quant_of(&aiednn_impl_list[i], &q)) { /* eager mode: f32 -> 8-bit quantize (quant instances) */
            add_impl("eager:quant", aiednn_reorder, 3, acc_eager_quant, run_eager_quant, nullptr);
            break;
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_lut q;
        if (rt_eager_lut_of(&aiednn_impl_list[i], &q)) { /* eager mode: per-channel 8-bit functions (lut instances) */
            static const aiednn_primitive_kind_t kinds[] = {aiednn_eltwise, aiednn_prelu, aiednn_batch_normalization,
                                                            aiednn_binary, aiednn_reorder};
            for (aiednn_primitive_kind_t k : kinds)
                add_impl("eager:lut", k, 3, acc_eager_lut, run_eager_lut, nullptr);
            break;
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_dw q;
        if (rt_eager_dw_of(&aiednn_impl_list[i], &q)) { /* eager mode: depthwise + requant (dwconv_q instances) */
            add_impl("eager:dwq", aiednn_convolution, 3, acc_eager_dwq, run_eager_dwq, nullptr);
            break;
        }
    }
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_pool q;
        if (rt_eager_pool_of(&aiednn_impl_list[i], &q)) { /* eager mode: 8-bit pooling (pool instances) */
            add_impl("eager:pool", aiednn_pooling, 3, acc_eager_pool, run_eager_pool, nullptr);
            break;
        }
    }
    add_impl("nhwc:pooling", aiednn_pooling, 0, acc_nhwc_pool, run_nhwc_pool, nullptr); /* aiednn_rt_layout.h */
    g_impls[g_nimpls - 1].caps = RT_CAP_NHWC;
    add_impl("im2col", aiednn_convolution, 0, acc_im2col, run_im2col, nullptr);
    g_impls[g_nimpls - 1].caps = RT_CAP_NHWC; /* the pipes / stem read and write nhwc (aiednn_rt_gemm.h) */
    add_impl("col2im", aiednn_deconvolution, 0, acc_col2im, run_col2im, nullptr);
    add_impl("ip", aiednn_inner_product, 0, acc_ip, run_ip, nullptr);
    add_impl("grouped", aiednn_grouped_matmul, 0, acc_any, run_grouped, nullptr);    /* per group: matmul dispatch */
    add_impl("gemm+conv", aiednn_convolution, 0, acc_gx_conv, run_gx_conv, nullptr); /* aiednn_rt_gemmx.h */
    g_impls[g_nimpls - 1].caps = RT_CAP_3D | RT_CAP_TRAIN | RT_CAP_ZP;
    add_impl("gemm+deconv", aiednn_deconvolution, 0, acc_gx_deconv, run_gx_deconv, nullptr);
    g_impls[g_nimpls - 1].caps = RT_CAP_3D;
    add_impl("gemm+ip", aiednn_inner_product, 0, acc_gx_ip, run_gx_ip, nullptr);
    g_impls[g_nimpls - 1].caps = RT_CAP_TRAIN;
    static const aiednn_primitive_kind_t zks[3] = {aiednn_convolution, aiednn_matmul, aiednn_inner_product};
    for (int z = 0; z < 3; z++) { /* zp+: the symmetric s8 problem plus an accumulator correction */
        add_impl("zp", zks[z], 1, acc_gx_zp, run_gx_zp, nullptr);
        g_impls[g_nimpls - 1].caps = RT_CAP_ZP | RT_CAP_NHWC; /* nhwc: when its symmetric problem's impl takes it */
    }
    add_impl("gemm+mm", aiednn_matmul, 0, acc_gx_mm, run_gx_mm, nullptr);
    g_impls[g_nimpls - 1].caps = RT_CAP_ZP;
    add_impl("gemm+mm", aiednn_inner_product, 0, acc_gx_mm, run_gx_mm, nullptr);
    g_impls[g_nimpls - 1].caps = RT_CAP_ZP;
    add_impl("gemm+sdpa", aiednn_sdpa, 0, acc_gx_sdpa, run_gx_sdpa, nullptr);
    add_impl("dynq", aiednn_dynamic_quantize, 0, acc_dynq_reorder, run_dynq_reorder, nullptr); /* per tensor */
    add_impl("conv+deconvbwd", aiednn_deconvolution, 0, acc_dcb, run_dcb, nullptr); /* backward as convolutions */
    g_impls[g_nimpls - 1].caps = RT_CAP_3D | RT_CAP_TRAIN;
    add_impl("deconv+avgpoolbwd", aiednn_pooling, 0, acc_apb, run_apb, nullptr); /* avg backward, depthwise */
    g_impls[g_nimpls - 1].caps = RT_CAP_3D | RT_CAP_TRAIN;
    add_impl("conv+resampbwd", aiednn_resampling, 0, acc_rsb, run_rsb, nullptr); /* nearest backward, depthwise */
    g_impls[g_nimpls - 1].caps = RT_CAP_3D | RT_CAP_TRAIN;
    add_impl("gemm+resampbwd", aiednn_resampling, 0, acc_rsl, run_rsl, nullptr); /* linear backward, per axis */
    g_impls[g_nimpls - 1].caps = RT_CAP_3D | RT_CAP_TRAIN;
    add_impl("gemm+rnn", aiednn_rnn, 0, acc_gx_rnn, run_gx_rnn, nullptr);
    add_impl("gemm+rnnbwd", aiednn_rnn, 0, acc_gx_rnnbwd, run_gx_rnnbwd, nullptr); /* BPTT's GEMMs on the AIE */
    g_impls[g_nimpls - 1].caps = RT_CAP_TRAIN;
    add_impl("gemm+sdpabwd", aiednn_sdpa, 0, acc_gx_sdpabwd, run_gx_sdpabwd, nullptr); /* attention backward */
    g_impls[g_nimpls - 1].caps = RT_CAP_TRAIN; /* gemm+sdpabwd: this line sets the impl just added */
    add_impl("gemm+linattn", aiednn_linear_attention, 0, acc_gx_linattn, run_gx_linattn,
             nullptr);                                                           /* chunked gated delta rule */
    add_impl("gemm+sdpa8", aiednn_sdpa, 0, acc_gx_sdpa8, run_gx_sdpa8, nullptr); /* int8 attention (opt-in tolerance) */
    g_impls[g_nimpls - 1].caps = RT_CAP_Q8SRC;                                   /* gemm+sdpa8 */
    add_impl("gemm+sdpavar", aiednn_sdpa, 0, acc_gx_sdpavar, run_gx_sdpavar, nullptr); /* varlen / paged K V */
    add_impl("densify", aiednn_sparse_matmul, 0, acc_sp_densify, run_sp_densify, nullptr); /* CSR -> dense GEMM */
    add_impl("gemm+gmlp", aiednn_gated_mlp, 0, acc_gx_gmlp, run_gx_gmlp, nullptr); /* gated MLP's GEMMs on the AIE */
    add_impl("dequant", aiednn_matmul, 0, acc_gx_dequant, run_gx_dequant, nullptr);
    add_impl("dequant", aiednn_inner_product, 0, acc_gx_dequant, run_gx_dequant, nullptr);
    add_impl("cvt", aiednn_matmul, 0, acc_gx_cvt, run_gx_dequant, nullptr);
    add_impl("cvt", aiednn_inner_product, 0, acc_gx_cvt, run_gx_dequant, nullptr);
    static const struct {
        const char *name;
        aiednn_primitive_kind_t kind;
        rt_run_fn run;
    } refs[] = {
        {"ref:matmul", aiednn_matmul, run_ref_matmul},
        {"ref:convolution", aiednn_convolution, run_ref_conv},
        {"ref:deconvolution", aiednn_deconvolution, run_ref_deconv},
        {"ref:inner_product", aiednn_inner_product, run_ref_ip},
        {"ref:eltwise", aiednn_eltwise, run_ref_eltwise},
        {"ref:binary", aiednn_binary, run_ref_binary},
        {"ref:prelu", aiednn_prelu, run_ref_prelu},
        {"ref:reduction", aiednn_reduction, run_ref_reduction},
        {"ref:pooling", aiednn_pooling, run_ref_pooling},
        {"ref:resampling", aiednn_resampling, run_ref_resampling},
        {"ref:lrn", aiednn_lrn, run_ref_lrn},
        {"ref:softmax", aiednn_softmax, run_ref_softmax},
        {"ref:batch_normalization", aiednn_batch_normalization, run_ref_bnorm},
        {"ref:layer_normalization", aiednn_layer_normalization, run_ref_lnorm},
        {"ref:group_normalization", aiednn_group_normalization, run_ref_gnorm},
        {"ref:concat", aiednn_concat, run_ref_concat},
        {"ref:sum", aiednn_sum, run_ref_sum},
        {"ref:reorder", aiednn_reorder, run_ref_reorder},
        {"ref:shuffle", aiednn_shuffle, run_ref_shuffle},
        {"ref:sdpa", aiednn_sdpa, run_ref_sdpa},
        {"ref:rnn", aiednn_rnn, run_ref_rnn},
        {"ref:dropout", aiednn_dropout, run_ref_dropout},
        {"ref:grid_sample", aiednn_grid_sample, run_ref_grid_sample},
        {"ref:ctc_loss", aiednn_ctc_loss, run_ref_ctc},
        {"ref:block_quantize", aiednn_block_quantize, run_ref_blockq},
        {"ref:sparse_matmul", aiednn_sparse_matmul, run_ref_sparse_mm},
        {"ref:slice", aiednn_slice, run_ref_slice},
        {"ref:gated_mlp", aiednn_gated_mlp, run_ref_gmlp},
        {"ref:rope", aiednn_rope, run_ref_rope},
        {"ref:gen_index", aiednn_gen_index, run_ref_gen_index},
        {"ref:dynamic_quantize", aiednn_dynamic_quantize, run_ref_dynq},
        {"ref:linear_attention", aiednn_linear_attention, run_ref_linattn},
        {"ref:custom", aiednn_custom, run_ref_custom},
    };
    for (unsigned i = 0; i < sizeof refs / sizeof refs[0]; i++)
        add_impl(refs[i].name, refs[i].kind, -1, acc_any, refs[i].run, nullptr);
}

#include "aiednn_rt_tune.h" /* measured choice among the acceptors (aiednn_set_autotune) */

static int rt_prob_is_3d_(const rt_prob *p) { /* = rt_prob_is_3d (aiednn_rt_nd.h, included later) */
    return (p->kind == aiednn_convolution || p->kind == aiednn_deconvolution || p->kind == aiednn_pooling) &&
           p->md[0].ndims == 5;
}

/* quiet dispatches (composites' inner GEMMs, zp+, im2col's matmul) repeat for the same problem on every
 * execute; acceptance is a function of the problem's bytes, the impl filter, the preference and a few
 * environment switches, so the choice is memoized on exactly those (one full dispatch walks every impl's
 * acceptor: ~20 ms per ResNet-18 inference on the A72 through zp+ alone) */
#define RT_DCACHE_N 64
typedef struct {
    rt_prob p;
    uint64_t sc_hash, env;
    char filter[64];
    int prefer, used;
    const rt_impl *impl;
    rt_bind bind;
} rt_dcache_entry;
static rt_dcache_entry g_dcache[RT_DCACHE_N];
static int g_dcache_next;

static int rt_env_on(const char *name); /* aiednn_rt_ntile.h */
static uint64_t rt_dcache_env(void) {
    static const char *const names[] = {"AIEDNN_GEMMX_ANY_INNER", "AIEDNN_NO_NTILE",   "AIEDNN_NO_DWN",
                                        "AIEDNN_NO_MMT",          "AIEDNN_NO_SDPABLK", "AIEDNN_WQ_DYNQ"};
    uint64_t e = 0;
    for (unsigned i = 0; i < sizeof names / sizeof names[0]; i++)
        e |= (uint64_t)rt_env_on(names[i]) << i;
    return e;
}
static uint64_t rt_dcache_sc(const rt_prob *p) { /* the values behind the oscales pointer */
    uint64_t h = 1469598103934665603ULL;
    for (int64_t i = 0; p->oscales && i < p->n_oscales; i++) {
        uint32_t b;
        memcpy(&b, &p->oscales[i], 4);
        h = (h ^ b) * 1099511628211ULL;
    }
    return h;
}

static int dispatch_full(const rt_prob *p, const rt_impl **out, rt_bind *bind, int quiet, aiednn_primitive *cands);

static int dispatch(const rt_prob *p, const rt_impl **out, rt_bind *bind, int quiet, aiednn_primitive *cands) {
    if (!quiet || cands)
        return dispatch_full(p, out, bind, quiet, cands);
    const uint64_t sc = rt_dcache_sc(p), env = rt_dcache_env();
    const char *flt = g_filter ? g_filter : "";
    for (int i = 0; i < RT_DCACHE_N; i++) {
        const rt_dcache_entry *e = &g_dcache[i];
        if (e->used && e->sc_hash == sc && e->env == env && e->prefer == g_prefer && !strcmp(e->filter, flt) &&
            !memcmp(&e->p, p, sizeof *p)) {
            *out = e->impl, *bind = e->bind;
            return aiednn_success;
        }
    }
    const int st = dispatch_full(p, out, bind, quiet, cands);
    if (st == aiednn_success && strlen(flt) < sizeof g_dcache[0].filter) {
        rt_dcache_entry *e = &g_dcache[g_dcache_next];
        g_dcache_next = (g_dcache_next + 1) % RT_DCACHE_N;
        memcpy(&e->p, p, sizeof *p);
        e->sc_hash = sc, e->env = env, e->prefer = g_prefer, e->used = 1;
        snprintf(e->filter, sizeof e->filter, "%s", flt);
        e->impl = *out, e->bind = *bind;
    }
    return st;
}

static int dispatch_full(const rt_prob *p, const rt_impl **out, rt_bind *bind, int quiet, aiednn_primitive *cands) {
    rt_init();
    char desc[256] = "", why[160];
    if (!quiet && g_verbose) /* only the verbose lines print it */
        prob_str(p, desc, sizeof desc);
    const rt_impl *best = nullptr;
    rt_bind best_bind = {nullptr, nullptr};
    int tie = 0;
    for (int i = 0; i < g_nimpls; i++) {
        const rt_impl *r = &g_impls[i];
        rt_bind b = {nullptr, nullptr};
        if (r->kind != p->kind)
            continue;
        int ok;
        if (g_filter && strncmp(r->name, g_filter, strlen(g_filter)) != 0) {
            ok = 0;
            snprintf(why, sizeof why, "impl filter '%s'", g_filter);
        } else if (r->specificity >= 0 && rt_prob_is_3d_(p) && !(r->caps & RT_CAP_3D)) { /* 2-D AIE paths */
            ok = 0;
            snprintf(why, sizeof why, "3-D: ref: only");
        } else if (r->specificity >= 0 && (((p->src_zp || p->wei_zp) && !(r->caps & RT_CAP_ZP)) ||
                                           (rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES) >= 0 && !(r->caps & RT_CAP_WQ) &&
                                            strcmp(r->name, "dequant")))) {
            /* the AIE GEMMs are symmetric, with per-channel scales (dequant removes the group scales) */
            ok = 0;
            snprintf(why, sizeof why, "zero points / weight group scales: ref: only");
        } else if (r->specificity >= 0 && p->prop != RT_FWD_INF && !(r->caps & RT_CAP_TRAIN)) { /* inference paths */
            ok = 0;
            snprintf(why, sizeof why, "training: ref: only");
        } else if (p->ndq && !(r->caps & RT_CAP_Q8SRC)) { /* 8-bit sources: dequantized first (create) */
            ok = 0;
            snprintf(why, sizeof why, "8-bit sources with scales: dequantized first");
        } else if (!(r->caps & RT_CAP_NHWC) && rt_prob_has_fmt(p)) { /* create's native pass (aiednn_rt_layout.h) */
            ok = 0;
            snprintf(why, sizeof why, "nhwc operands: no nhwc path");
        } else {
            why[0] = '\0';
            ok = r->accepts(r, p, &b, why, sizeof why);
        }
        if (!quiet && g_verbose >= 2)
            printf("aiednn_verbose,dispatch,%s,%s,%s%s,%s\n", kind_str(p->kind), r->name,
                   ok ? "accepts" : "skipping: ", ok ? "" : why, desc);
        if (!ok)
            continue;
        if (cands && cands->ncand < RT_TUNE_MAX) { // every acceptor: the autotuner may time them
            cands->cand[cands->ncand] = r;
            cands->cbind[cands->ncand++] = b;
        }
        if (!best || r->specificity > best->specificity) {
            tie = 0;
            best = r;
            best_bind = b;
        } else if (r->specificity == best->specificity) {
            tie = 1;
        }
    }
    *out = best;
    if (!best || tie) {
        if (!quiet && g_verbose)
            printf("aiednn_verbose,dispatch,%s,%s,%s\n", kind_str(p->kind),
                   best ? "ambiguous: equally specific implementations" : "unimplemented", desc);
        return aiednn_unimplemented;
    }
    *bind = best_bind;
    return aiednn_success;
}

// ── validation helpers ─────────────────────────────────────────────────────

static int g_why_line;

static int md_ok(const aiednn_memory_desc_t *md) {
    if (!md || md->ndims < 1 || md->ndims > AIEDNN_MAX_NDIMS || md->data_type < aiednn_s8 ||
        md->data_type > aiednn_f4_e2m1 || (md->format != aiednn_format_dense && md->format != aiednn_format_nhwc) ||
        (md->format == aiednn_format_nhwc && md->ndims != 4))
        return 0;
    for (int i = 0; i < md->ndims; i++)
        if (md->dims[i] <= 0)
            return 0;
    return 1;
}

static aiednn_status_t bad(const char *kind, const char *msg) {
    if (g_verbose)
        printf("aiednn_verbose,create,%s,invalid: %s\n", kind, msg);
    return aiednn_invalid_arguments;
}

#define REQUIRE(cond, kind, msg)                                                                                       \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            g_why_line = __LINE__;                                                                                     \
            return bad(kind, msg);                                                                                     \
        }                                                                                                              \
    } while (0)

static void prob_init(rt_prob *p, aiednn_primitive_kind_t kind, int alg) {
    memset(p, 0, sizeof *p);
    p->kind = kind;
    p->alg = alg;
    p->scale = 1.0f;
    p->dst_scale = 1.0f;
}

static void prob_add(rt_prob *p, int arg, const aiednn_memory_desc_t *md);

static void prob_add(rt_prob *p, int arg, const aiednn_memory_desc_t *md);

/* attr.wei_scales / wei_zero_points: grouped weight decompression of a matmul / inner product */
static const char *attach_wei_groups(rt_prob *p, const aiednn_attr_t *a) {
    const aiednn_memory_desc_t *s = &a->wei_scales, *z = &a->wei_zero_points, *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    if (p->kind != aiednn_matmul && p->kind != aiednn_inner_product)
        return "weight group scales: matmul / inner product only";
    if (!s->ndims)
        return "weight zero points need weight scales";
    const aiednn_data_type_t sd = p->md[0].data_type;
    if (!(dt_is_float(sd) || sd == aiednn_s8 || sd == aiednn_u8) || w->data_type == aiednn_s32 || a->wei_zero_point)
        return "weight group scales: floating or 8-bit src, weights not s32, no per-tensor weight zero point";
    const int64_t N = p->kind == aiednn_matmul ? w->dims[w->ndims - 1] : w->dims[0];
    const int64_t K = p->kind == aiednn_matmul ? w->dims[w->ndims - 2] : md_numel(w) / w->dims[0];
    const aiednn_data_type_t st = s->data_type;
    if (!md_ok(s) || s->ndims != 2 || s->dims[1] != N || s->dims[0] < 1 || K % s->dims[0] ||
        !(st == aiednn_f32 || st == aiednn_bf16 || st == aiednn_f16 || st == aiednn_e8m0))
        return "weight scales: f32 / bf16 / f16 / e8m0 [K / group, N], the group dividing K";
    prob_add(p, AIEDNN_ARG_ATTR_WEI_SCALES, s);
    if (z->ndims) {
        if (!md_ok(z) || !md_same_shape(s, z) || !dt_is_int(z->data_type))
            return "weight zero points: s8 / u8 / s4 / u4 / s32, the scales' shape";
        prob_add(p, AIEDNN_ARG_ATTR_WEI_ZERO_POINTS, z);
    }
    return nullptr;
}

/* validate an attr against dst (channel axis ch_axis) and fold it into p; scales are
 * copied (freed at destroy). Returns NULL or the reason it is refused. */
static const char *attach_attr(rt_prob *p, const aiednn_attr_t *a, int ch_axis) {
    if (!a)
        return nullptr;
    const aiednn_memory_desc_t *d = rt_md(p, AIEDNN_ARG_DST);
    const int64_t C = d->dims[ch_axis];
    if (a->n_scales < 0 || (a->n_scales > 1 && a->n_scales != C) || (a->n_scales && !a->scales))
        return "attr scales: 0, 1 or one per output channel";
    if (a->n_post_ops < 0 || a->n_post_ops > AIEDNN_MAX_POST_OPS)
        return "attr: at most AIEDNN_MAX_POST_OPS post-ops";
    if (a->dst_scale < 0)
        return "attr dst_scale must be >= 0 (0 means 1)";
    for (int k = 0; k < a->n_post_ops; k++) {
        const aiednn_post_op_t *po = &a->post_ops[k];
        if (po->kind == aiednn_post_op_eltwise) {
            if (po->alg < aiednn_eltwise_relu || po->alg > aiednn_eltwise_selu)
                return "eltwise post-op: not an eltwise algorithm";
        } else if (po->kind == aiednn_post_op_requant_eltwise) {
            if (po->alg < aiednn_eltwise_relu || po->alg > aiednn_eltwise_selu)
                return "requant_eltwise post-op: not an eltwise algorithm";
            if (!(po->q_scale > 0) || (po->q_data_type != aiednn_s8 && po->q_data_type != aiednn_u8))
                return "requant_eltwise post-op: q_scale > 0, q_data_type s8 / u8";
        } else if (po->kind == aiednn_post_op_requant_binary) {
            const aiednn_alg_kind_t al = po->alg;
            if (al != aiednn_binary_add && al != aiednn_binary_sub && al != aiednn_binary_mul &&
                al != aiednn_binary_max && al != aiednn_binary_min)
                return "requant_binary post-op: add / sub / mul / max / min";
            if (!(po->q_scale > 0) || (po->q_data_type != aiednn_s8 && po->q_data_type != aiednn_u8) ||
                !(po->alpha >= 0))
                return "requant_binary post-op: q_scale > 0, q_data_type s8 / u8, src1 scale (alpha) >= 0";
            if (!md_ok(&po->src1) || !md_bcast_ok(&po->src1, d) ||
                (po->src1.data_type != aiednn_s8 && po->src1.data_type != aiednn_u8 &&
                 po->src1.data_type != aiednn_f32))
                return "requant_binary post-op src1: s8 / u8 / f32, broadcast to dst";
            prob_add(p, AIEDNN_ARG_ATTR_POST_OP_SRC(k), &po->src1);
        } else if (po->kind == aiednn_post_op_binary) {
            if (po->alg < aiednn_binary_add || po->alg == aiednn_binary_select || po->alg > aiednn_binary_add_square)
                return "binary post-op: add .. le (not select)";
            if (!md_ok(&po->src1) || !md_bcast_ok(&po->src1, d))
                return "binary post-op src1 must broadcast to dst";
            prob_add(p, AIEDNN_ARG_ATTR_POST_OP_SRC(k), &po->src1);
        } else {
            return "post-op kind: eltwise or binary";
        }
        p->po[k] = *po;
    }
    p->npo = a->n_post_ops;
    p->n_oscales = a->n_scales;
    if (a->n_scales) {
        float *sc = (float *)malloc((size_t)a->n_scales * sizeof(float));
        if (!sc)
            return "out of memory";
        memcpy(sc, a->scales, (size_t)a->n_scales * sizeof(float));
        p->oscales = sc;
    }
    p->dst_scale = a->dst_scale > 0 ? a->dst_scale : 1.0f;
    p->dst_zp = a->dst_zero_point;
    p->has_attr = 1;
    if (a->src_zero_point || a->wei_zero_point) {
        const int gemm = p->kind == aiednn_matmul || p->kind == aiednn_inner_product || p->kind == aiednn_convolution ||
                         p->kind == aiednn_deconvolution || p->kind == aiednn_grouped_matmul;
        if (!gemm)
            return "src / weights zero points: GEMM family only";
        if (!dt_is_int(p->md[0].data_type) || !dt_is_int(p->md[1].data_type))
            return "src / weights zero points need integer src and weights";
        p->src_zp = a->src_zero_point, p->wei_zp = a->wei_zero_point;
    }
    return a->wei_scales.ndims || a->wei_zero_points.ndims ? attach_wei_groups(p, a) : nullptr;
}

static void prob_add(rt_prob *p, int arg, const aiednn_memory_desc_t *md) {
    p->arg[p->nmd] = arg;
    p->md[p->nmd] = *md;
    p->nmd++;
}

static void vec_md(aiednn_memory_desc_t *md, int64_t n) {
    memset(md, 0, sizeof *md);
    md->ndims = 1;
    md->dims[0] = n;
    md->data_type = aiednn_f32;
}

/* a requant_binary post-op at kr: inner = the problem up to it with its int8 intermediate as dst, tail = the
 * quantized binary after it (aiednn_binary_create_attr's semantics). 0: none, 1: split, -1: not allowed here */
static int qres_split(const rt_prob *p, rt_prob *inner, rt_prob *tail, int *src1_arg) {
    int kr = -1;
    for (int k = 0; k < p->npo; k++)
        if (p->po[k].kind == aiednn_post_op_requant_binary || p->po[k].kind == aiednn_post_op_requant_eltwise) {
            if (kr >= 0)
                return -1;
            kr = k;
        } else if (kr >= 0 && p->po[k].kind != aiednn_post_op_eltwise) {
            return -1;
        }
    if (kr < 0)
        return 0;
    const int gemm = p->kind == aiednn_convolution || p->kind == aiednn_deconvolution || p->kind == aiednn_matmul ||
                     p->kind == aiednn_inner_product;
    const aiednn_post_op_t *po = &p->po[kr];
    const int elt = po->kind == aiednn_post_op_requant_eltwise;
    const int ds = rt_slot(p, AIEDNN_ARG_DST), s1 = elt ? -1 : rt_slot(p, AIEDNN_ARG_ATTR_POST_OP_SRC(kr));
    const aiednn_data_type_t dt = ds >= 0 ? p->md[ds].data_type : aiednn_f32;
    if (!gemm || ds < 0 || (!elt && s1 < 0) || p->n_oscales < 0 ||
        (dt != aiednn_s8 && dt != aiednn_u8 && dt != aiednn_f32))
        return -1;
    *inner = *p;
    inner->npo = kr;
    inner->md[ds].data_type = po->q_data_type;
    inner->dst_scale = po->q_scale, inner->dst_zp = po->q_zero_point;
    if (!elt) {
        for (int i = s1; i + 1 < inner->nmd; i++) /* drop src1's slot */
            inner->arg[i] = inner->arg[i + 1], inner->md[i] = inner->md[i + 1];
        inner->nmd--;
    }
    prob_init(tail, elt ? aiednn_eltwise : aiednn_binary, po->alg);
    prob_add(tail, AIEDNN_ARG_SRC, &inner->md[rt_slot(inner, AIEDNN_ARG_DST)]);
    if (!elt)
        prob_add(tail, AIEDNN_ARG_SRC_1, &p->md[s1]);
    prob_add(tail, AIEDNN_ARG_DST, &p->md[ds]);
    tail->has_attr = 1;
    tail->bin_sc[0] = po->q_scale, tail->bin_zp[0] = po->q_zero_point;
    if (elt)
        tail->alpha = po->alpha, tail->beta = po->beta;
    else
        tail->bin_sc[1] = po->alpha, tail->bin_zp[1] = (int32_t)po->beta;
    tail->npo = p->npo - kr - 1;
    for (int k = 0; k < tail->npo; k++)
        tail->po[k] = p->po[kr + 1 + k];
    tail->dst_scale = p->dst_scale > 0 ? p->dst_scale : 1.0f, tail->dst_zp = p->dst_zp;
    *src1_arg = elt ? 0 : AIEDNN_ARG_ATTR_POST_OP_SRC(kr);
    return 1;
}

static aiednn_status_t create(aiednn_primitive_t *primitive, const rt_prob *p0) {
    const rt_impl *impl = nullptr;
    rt_bind bind = {nullptr, nullptr};
    aiednn_primitive cands; /* every acceptor (autotuning) */
    cands.ncand = 0;
    rt_prob inner, tail, pin, pep, pdq;
    int qs1 = 0;
    const rt_prob *porig = p0;
    int q8src = 0; /* an impl reads the 8-bit sources itself (RT_CAP_Q8SRC): no dequantized copies */
    if (p0->ndq && !p0->dq_packed) {
        const rt_impl *qi = nullptr;
        rt_bind qb = {nullptr, nullptr};
        q8src = dispatch_full(p0, &qi, &qb, 1, nullptr) == aiednn_success;
    }
    if (!q8src &&
        (p0->ndq ||
         p0->dq_packed)) { /* 8-bit / packed sources (aiednn_rt_phead.h): dispatch the f32 / unpacked problem */
        phead_split(p0, &pdq);
        p0 = &pdq;
    }
    const int pt = ptail_split(p0, &pin, &pep); /* post-ops on a non-GEMM op (aiednn_rt_ptail.h) */
    if (pt < 0) {
        free((void *)p0->oscales);
        return bad(kind_str(p0->kind),
                   "post-ops: per-tensor output scale only; nhwc dst needs nhwc / format-free sources");
    }
    if (pt)
        p0 = &pin;
    const int qr = qres_split(p0, &inner, &tail, &qs1);
    if (qr < 0) {
        free((void *)p0->oscales);
        return aiednn_invalid_arguments;
    }
    /* the tail: src = the intermediate in dst's format, src1, dst. A src1 in the other format (a skip edge the
     * graph kept dense while dst went nhwc, or the reverse) is converted to dst's per execute */
    int qs1_cvt = 0;
    aiednn_memory_desc_t qs1_md;
    memset(&qs1_md, 0, sizeof qs1_md);
    if (qr && !rt_fmt_canon(&tail)) {
        const int t1 = rt_slot(&tail, AIEDNN_ARG_SRC_1), td = rt_slot(&tail, AIEDNN_ARG_DST);
        qs1_md = tail.md[t1];
        qs1_cvt = t1 >= 0 && qs1_md.ndims == 4 && !md_fmt_free(&qs1_md) ? (md_is_nhwc(&tail.md[td]) ? 1 : 2) : 0;
        if (qs1_cvt)
            tail.md[t1].format = tail.md[td].format;
        if (!qs1_cvt || !rt_fmt_canon(&tail)) {
            free((void *)p0->oscales);
            return bad(kind_str(p0->kind), "requant post-op: dst and its src1 in formats the tail cannot reconcile");
        }
    }
    const rt_prob *pf = qr ? &inner : p0; /* the GEMM half dispatches exactly as the standalone primitive */
    /* memory formats (aiednn_rt_layout.h): dispatch sees the dense problem; canon leaves no format behind */
    rt_prob dense = *pf;
    const int canon = rt_fmt_canon(&dense);
    /* native first: the formatted problem among the impls with an nhwc path; else wrap the dense one */
    const int native = !canon && dispatch(pf, &impl, &bind, 1, nullptr) == aiednn_success;
    if (!canon && !native)
        rt_fmt_strip(&dense);
    const rt_prob *p = native ? pf : &dense;
    const int st = native ? (int)aiednn_success : dispatch(p, &impl, &bind, 0, &cands);
    if (st != aiednn_success) {
        if (pt)
            free((void *)pep.oscales);
        free((void *)p->oscales); // the copied attr scales are the primitive's; there is none
        return (aiednn_status_t)st;
    }
    do { /* one heap object per primitive (no cap: a model compiles hundreds); freed by destroy */
        aiednn_primitive *prim = (aiednn_primitive *)calloc(1, sizeof(aiednn_primitive));
        if (!prim)
            break;
        prim->in_use = 1;
        prim->prob = *p;
        prim->impl = impl;
        prim->bind = bind;
        prim->ncand = cands.ncand;
        for (int c = 0; c < cands.ncand; c++)
            prim->cand[c] = cands.cand[c], prim->cbind[c] = cands.cbind[c];
        prim->sig = rt_tune_sig(p);
        if (native) /* the impl reads the formats itself; no autotuning among dense-only impls */
            prim->tuned = 1;
        else
            rt_tune_apply(prim); /* a decision measured before (aiednn_autotune_load / an earlier execute) */
        impl = prim->impl, bind = prim->bind;
        if (!canon && !native) /* wrap: dense run, execute converts the nhwc operands */
            rt_fmt_wrap_plan(prim, pf);
        if (bind.inner && bind.inner2)
            snprintf(prim->info, sizeof prim->info, "%s+%s+%s", impl->name, bind.inner->name, bind.inner2->name);
        else if (bind.inner)
            snprintf(prim->info, sizeof prim->info, "%s+%s", impl->name, bind.inner->name);
        else
            snprintf(prim->info, sizeof prim->info, "%s", impl->name);
        if (qr) { /* the tail runs after the GEMM half (aiednn_primitive_execute_args); no autotuning of the half */
            prim->qres = 1, prim->qtail = tail, prim->qtail_src1 = qs1, prim->tuned = 1;
            prim->qs1_cvt = qs1_cvt, prim->qs1_md = qs1_md; /* 1: dense -> nhwc, 2: nhwc -> dense */
            const size_t l = strlen(prim->info);
            snprintf(prim->info + l, sizeof prim->info - l, "+qres");
        }
        if (pt) {
            prim->ptail = 1, prim->pt_ep = pep, prim->tuned = 1;
            const size_t l = strlen(prim->info);
            snprintf(prim->info + l, sizeof prim->info - l, "+pt");
        }
        if (!q8src && (porig->ndq || porig->dq_packed)) {
            phead_keep(prim, porig);
            const size_t l = strlen(prim->info);
            snprintf(prim->info + l, sizeof prim->info - l, "+dq");
        }
        if (prim->fmt_mask) { /* the nhwc operands are converted around a dense run (aiednn_rt_layout.h) */
            const size_t l = strlen(prim->info);
            snprintf(prim->info + l, sizeof prim->info - l, "+fmtwrap");
        }
        if (g_verbose) {
            char desc[256];
            prob_str(p, desc, sizeof desc);
            printf("aiednn_verbose,create,%s,%s,%s\n", kind_str(p->kind), prim->info, desc);
        }
        *primitive = prim;
        return aiednn_success;
    } while (0);
    free((void *)p->oscales);
    return aiednn_out_of_memory;
}

// int8 GEMM-family typing: s8|u8 x s8 -> s32 (or s8 for matmul), s32 bias; or all f32
// no attr: s8|u8 x s8 -> s32 (or s8 for matmul), s32 bias; or all f32.
// with an attr (fused epilogue): s8|u8 x s8 -> s32|s8|u8|f32, bias s32|f32; or all f32.
// attr == 2 (weight group scales): also s8|u8 x s8|u8|s4|u4 -> floating dst.
static int gemm_types_ok(const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
                         const aiednn_memory_desc_t *d, int allow_s8_dst, int attr) {
    if (dt_is_float(s->data_type)) /* any floating types, mixed (bf16 x bf16 -> f32, ...); integer weights
                                      (s8 / u8 / s4 / u4) are decompressed: value x the output scale */
        return (dt_is_float(w->data_type) || (dt_is_int(w->data_type) && w->data_type != aiednn_s32)) &&
               dt_is_float(d->data_type) && (!b || dt_is_float(b->data_type));
    if (attr == 2) /* weight group scales (attach_wei_groups): 8-bit src x 4/8-bit integer weights -> float */
        return (s->data_type == aiednn_s8 || s->data_type == aiednn_u8) && dt_is_int(w->data_type) &&
               w->data_type != aiednn_s32 && dt_is_float(d->data_type) && (!b || dt_is_float(b->data_type));
    if (dt_is_4bit(w->data_type))
        return 0;
    if (!((s->data_type == aiednn_s8 || s->data_type == aiednn_u8) && w->data_type == aiednn_s8))
        return 0;
    if (attr)
        return !b || b->data_type == aiednn_s32 || b->data_type == aiednn_f32;
    return (d->data_type == aiednn_s32 || (allow_s8_dst && d->data_type == aiednn_s8)) &&
           (!b || b->data_type == aiednn_s32);
}

// ── create: GEMM family ────────────────────────────────────────────────────

static aiednn_status_t conv_like_create(aiednn_primitive_t *pr, aiednn_primitive_kind_t kind,
                                        const aiednn_memory_desc_t *src, const aiednn_memory_desc_t *w,
                                        const aiednn_memory_desc_t *b, const aiednn_memory_desc_t *dst,
                                        const aiednn_conv_params_t *cp, const aiednn_attr_t *attr) {
    const char *k = kind_str(kind);
    REQUIRE(pr && cp && md_ok(src) && md_ok(w) && md_ok(dst) && (!b || md_ok(b)), k, "null or malformed descriptor");
    REQUIRE(src->ndims == 4 && w->ndims == 4 && dst->ndims == 4, k, "want src/weights/dst rank 4 (NCHW, OIHW)");
    REQUIRE(gemm_types_ok(src, w, b, dst, 0, attr != nullptr), k,
            "types: s8|u8 x s8 -> s32 (bias s32; with an attr also s8|u8|f32 dst, f32 bias), or f32");
    const int64_t G = cp->groups;
    REQUIRE(G >= 1 && src->dims[1] % G == 0 && dst->dims[1] % G == 0, k, "groups must divide C and O");
    REQUIRE(w->dims[0] == dst->dims[1] && w->dims[1] == src->dims[1] / G && dst->dims[0] == src->dims[0], k,
            "weights must be [O, C/groups, KH, KW], dst [N, O, ...]");
    REQUIRE(!b || (b->ndims == 1 && b->dims[0] == dst->dims[1]), k, "bias must be [O]");
    for (int i = 0; i < 2; i++) {
        REQUIRE(cp->strides[i] >= 1 && cp->dilates[i] >= 0 && cp->padding_l[i] >= 0 && cp->padding_r[i] >= 0, k,
                "strides >= 1, dilates and padding >= 0");
        const int64_t ext = (w->dims[2 + i] - 1) * (cp->dilates[i] + 1) + 1;
        const int64_t want = kind == aiednn_convolution
                                 ? (src->dims[2 + i] + cp->padding_l[i] + cp->padding_r[i] - ext) / cp->strides[i] + 1
                                 : (src->dims[2 + i] - 1) * cp->strides[i] - cp->padding_l[i] - cp->padding_r[i] + ext;
        REQUIRE(dst->dims[2 + i] == want, k, "dst spatial size does not match strides / padding / dilation");
    }
    rt_prob p;
    prob_init(&p, kind, 0);
    prob_add(&p, AIEDNN_ARG_SRC, src);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    if (b)
        prob_add(&p, AIEDNN_ARG_BIAS, b);
    p.conv = *cp;
    const char *why = attach_attr(&p, attr, 1);
    if (why) {
        free((void *)p.oscales);
        return bad(k, why);
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_convolution_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                             const aiednn_memory_desc_t *w,
                                                             const aiednn_memory_desc_t *b,
                                                             const aiednn_memory_desc_t *dst,
                                                             const aiednn_conv_params_t *cp) {
    return conv_like_create(p, aiednn_convolution, src, w, b, dst, cp, nullptr);
}

extern "C" aiednn_status_t
aiednn_convolution_forward_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                       const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
                                       const aiednn_memory_desc_t *dst, const aiednn_conv_params_t *cp,
                                       const aiednn_attr_t *attr) {
    return conv_like_create(p, aiednn_convolution, src, w, b, dst, cp, attr);
}

extern "C" aiednn_status_t aiednn_deconvolution_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                               const aiednn_memory_desc_t *w,
                                                               const aiednn_memory_desc_t *b,
                                                               const aiednn_memory_desc_t *dst,
                                                               const aiednn_conv_params_t *cp) {
    return conv_like_create(p, aiednn_deconvolution, src, w, b, dst, cp, nullptr);
}

extern "C" aiednn_status_t
aiednn_deconvolution_forward_create_attr(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                         const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
                                         const aiednn_memory_desc_t *dst, const aiednn_conv_params_t *cp,
                                         const aiednn_attr_t *attr) {
    return conv_like_create(p, aiednn_deconvolution, src, w, b, dst, cp, attr);
}

extern "C" aiednn_status_t aiednn_matmul_forward_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                        const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
                                                        const aiednn_memory_desc_t *d, int transposed) {
    return aiednn_matmul_forward_create_attr(pr, s, w, b, d, transposed, nullptr);
}

extern "C" aiednn_status_t aiednn_matmul_forward_create_attr(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                             const aiednn_memory_desc_t *w,
                                                             const aiednn_memory_desc_t *b,
                                                             const aiednn_memory_desc_t *d, int transposed,
                                                             const aiednn_attr_t *attr) {
    const char *k = "matmul";
    REQUIRE(pr && md_ok(s) && md_ok(w) && md_ok(d) && (!b || md_ok(b)), k, "null or malformed descriptor");
    const int nd = d->ndims;
    REQUIRE(nd >= 2 && s->ndims == nd && w->ndims == nd, k, "src, weights, dst: same rank >= 2");
    REQUIRE(gemm_types_ok(s, w, b, d, 1,
                          !attr                    ? 0
                          : attr->wei_scales.ndims ? 2
                                                   : 1),
            k, "types: s8|u8 x s8 -> s32|s8 (bias s32; with an attr also u8|f32 dst, f32 bias), or f32");
    const int64_t M = d->dims[nd - 2], N = d->dims[nd - 1], K = s->dims[nd - 1];
    REQUIRE(s->dims[nd - 2] == M, k, "src [..., M, K], dst [..., M, N]");
    REQUIRE(transposed ? (w->dims[nd - 2] == N && w->dims[nd - 1] == K)
                       : (w->dims[nd - 2] == K && w->dims[nd - 1] == N),
            k, "weights must be [..., K, N] ([..., N, K] when transposed)");
    for (int i = 0; i < nd - 2; i++)
        REQUIRE((s->dims[i] == d->dims[i] || s->dims[i] == 1) && (w->dims[i] == d->dims[i] || w->dims[i] == 1), k,
                "batch dims must equal dst's or be 1");
    REQUIRE(!b || md_bcast_ok(b, d), k, "bias must broadcast to dst (same rank)");
    rt_prob p;
    prob_init(&p, aiednn_matmul, 0);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    prob_add(&p, AIEDNN_ARG_DST, d);
    if (b)
        prob_add(&p, AIEDNN_ARG_BIAS, b);
    p.transposed = transposed != 0;
    const char *why = attach_attr(&p, attr, nd - 1);
    if (why) {
        free((void *)p.oscales);
        return bad(k, why);
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_inner_product_forward_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                               const aiednn_memory_desc_t *w,
                                                               const aiednn_memory_desc_t *b,
                                                               const aiednn_memory_desc_t *d) {
    return aiednn_inner_product_forward_create_attr(pr, s, w, b, d, nullptr);
}

extern "C" aiednn_status_t
aiednn_inner_product_forward_create_attr(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                         const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
                                         const aiednn_memory_desc_t *d, const aiednn_attr_t *attr) {
    const char *k = "inner_product";
    REQUIRE(pr && md_ok(s) && md_ok(w) && md_ok(d) && (!b || md_ok(b)), k, "null or malformed descriptor");
    REQUIRE(s->ndims >= 2 && w->ndims == s->ndims && d->ndims == 2, k,
            "src [N, IC...], weights [OC, IC...], dst [N, OC]");
    for (int i = 1; i < s->ndims; i++)
        REQUIRE(w->dims[i] == s->dims[i], k, "weights' input dims must equal src's");
    REQUIRE(d->dims[0] == s->dims[0] && d->dims[1] == w->dims[0], k, "dst must be [N, OC]");
    REQUIRE(!b || (b->ndims == 1 && b->dims[0] == w->dims[0]), k, "bias must be [OC]");
    REQUIRE(gemm_types_ok(s, w, b, d, 1,
                          !attr                    ? 0
                          : attr->wei_scales.ndims ? 2
                                                   : 1),
            k, "types: s8|u8 x s8 -> s32|s8 (bias s32; with an attr also u8|f32 dst, f32 bias), or f32");
    rt_prob p;
    prob_init(&p, aiednn_inner_product, 0);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    prob_add(&p, AIEDNN_ARG_DST, d);
    if (b)
        prob_add(&p, AIEDNN_ARG_BIAS, b);
    const char *why = attach_attr(&p, attr, 1);
    if (why) {
        free((void *)p.oscales);
        return bad(k, why);
    }
    return create(pr, &p);
}

static aiednn_status_t sdpa_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *q,
                                   const aiednn_memory_desc_t *kk, const aiednn_memory_desc_t *v,
                                   const aiednn_memory_desc_t *mask, const aiednn_memory_desc_t *d, float scale,
                                   int causal, const aiednn_attr_t *attr) {
    const char *k = "sdpa";
    REQUIRE(pr && md_ok(q) && md_ok(kk) && md_ok(v) && md_ok(d) && (!mask || md_ok(mask)), k,
            "null or malformed descriptor");
    REQUIRE(q->ndims == 4 && kk->ndims == 4 && v->ndims == 4 && d->ndims == 4, k, "Q, K, V, dst rank 4 [B,H,S,D]");
    /* causal bit 0; AIEDNN_SDPA_QKV_BSHD / _DST_BSHD: those tensors are [B,S,H,D] (the head split /
     * merge permutes folded in, --aiednn-fold-head-permute); the checks below see [B,H,S,D] */
    aiednn_memory_desc_t qc = *q, kc = *kk, vc = *v, dc = *d;
    const auto swap12 = [](aiednn_memory_desc_t *m) {
        const int64_t t = m->dims[1];
        m->dims[1] = m->dims[2], m->dims[2] = t;
    };
    const int packed = (causal & AIEDNN_SDPA_QKV_PACKED) != 0;
    REQUIRE(!packed || !(causal & AIEDNN_SDPA_QKV_BSHD), k, "AIEDNN_SDPA_QKV_PACKED: descriptors are [B,H,S,D]");
    if (causal & AIEDNN_SDPA_QKV_BSHD)
        swap12(&qc), swap12(&kc), swap12(&vc);
    if (causal & AIEDNN_SDPA_DST_BSHD)
        swap12(&dc);
    q = &qc, kk = &kc, v = &vc;
    const aiednn_memory_desc_t *dd = d;
    d = &dc;
    auto qkv_ok = [&](const aiednn_memory_desc_t *m,
                      int i) { /* floating, or 8-bit read through its attr scale */
                               return dt_is_float(m->data_type) ||
                                      (attr && attr_src_sc(attr, i) > 0 &&
                                       (m->data_type == aiednn_s8 || m->data_type == aiednn_u8));
    };
    REQUIRE(qkv_ok(q, 0) && qkv_ok(kk, 1) && qkv_ok(v, 2) &&
                (dt_is_float(d->data_type) || (attr && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8))) &&
                (!mask || dt_is_float(mask->data_type)),
            k, "floating types (s8 / u8 Q / K / V through attr src_scales; dst s8 / u8 with attr)");
    REQUIRE(!attr || (!attr->n_scales && !attr->n_post_ops), k, "attr: dst_scale / dst_zero_point only");
    /* grouped-query / multi-query attention: K and V may carry Hk heads, Hk dividing H; query head h reads
     * key / value head h / (H / Hk) */
    REQUIRE(q->dims[0] == d->dims[0] && kk->dims[0] == d->dims[0] && v->dims[0] == d->dims[0] &&
                q->dims[1] == d->dims[1] && kk->dims[1] == v->dims[1] && kk->dims[1] >= 1 &&
                d->dims[1] % kk->dims[1] == 0,
            k, "Q, dst [B, H]; K, V [B, Hk] with Hk dividing H (GQA / MQA)");
    REQUIRE(kk->dims[2] == v->dims[2] && kk->dims[3] == q->dims[3] && d->dims[2] == q->dims[2] &&
                d->dims[3] == v->dims[3],
            k, "K [B,H,Skv,D], V [B,H,Skv,Dv], dst [B,H,Sq,Dv]");
    rt_prob p;
    prob_init(&p, aiednn_sdpa, 0);
    // the problem keeps the [B,H,S,D] view; p.flags says where the memory is [B,S,H,D]
    prob_add(&p, AIEDNN_ARG_QUERY, q);
    prob_add(&p, AIEDNN_ARG_KEY, kk);
    prob_add(&p, AIEDNN_ARG_VALUE, v);
    prob_add(&p, AIEDNN_ARG_DST, d);
    (void)dd;
    if (mask) {
        aiednn_memory_desc_t sc = *d;
        sc.dims[3] = kk->dims[2];
        REQUIRE(md_bcast_ok(mask, &sc), k, "mask must broadcast to [B, H, Sq, Skv]");
        prob_add(&p, AIEDNN_ARG_ATTN_MASK, mask);
    }
    p.scale = scale;
    p.causal = (causal & 1) != 0;
    p.flags = (unsigned)(causal & (AIEDNN_SDPA_QKV_BSHD | AIEDNN_SDPA_DST_BSHD));
    if (packed) { /* one [B, S, 3, H, D] buffer: unpacked per execute (aiednn_rt_phead.h) */
        REQUIRE(md_same_shape(q, kk) && md_same_shape(q, v) && q->data_type == kk->data_type &&
                    q->data_type == v->data_type,
                k, "AIEDNN_SDPA_QKV_PACKED: Q, K, V of one shape and type");
        p.dq_packed = 1;
    }
    if (attr) { /* dequantized Q / K / V: src_scales[0], [1], src_scales_23[0] (aiednn_rt_phead.h) */
        const char *why = attach_src_dq(&p, attr, 3);
        REQUIRE(!why, k, why ? why : "");
    }
    if (attr && !attr_only_src_dq(attr)) { // folded quantizing reorder: only the dst quantization
        const char *err = attach_attr(&p, attr, 1);
        REQUIRE(!err, k, err ? err : "");
    }
    if (attr) { /* AIEDNN_ATTR_TOLERANCE_INT8_ATTN: gemm+sdpa8 may serve s8 Q / K / V */
        REQUIRE(!(attr->flags & ~(AIEDNN_ATTR_TOLERANCE_1CODE | AIEDNN_ATTR_TOLERANCE_INT8_ATTN)), k,
                "attr flags: AIEDNN_ATTR_TOLERANCE_1CODE / _INT8_ATTN only");
        p.attr_flags = attr->flags;
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_sdpa_forward_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *q,
                                                      const aiednn_memory_desc_t *kk, const aiednn_memory_desc_t *v,
                                                      const aiednn_memory_desc_t *mask, const aiednn_memory_desc_t *d,
                                                      float scale, int causal) {
    return sdpa_create(pr, q, kk, v, mask, d, scale, causal, nullptr);
}

extern "C" aiednn_status_t
aiednn_sdpa_forward_create_attr(aiednn_primitive_t *pr, const aiednn_memory_desc_t *q, const aiednn_memory_desc_t *kk,
                                const aiednn_memory_desc_t *v, const aiednn_memory_desc_t *mask,
                                const aiednn_memory_desc_t *d, float scale, int causal, const aiednn_attr_t *attr) {
    return sdpa_create(pr, q, kk, v, mask, d, scale, causal, attr);
}

// ── create: pointwise / reduction ──────────────────────────────────────────

extern "C" aiednn_status_t aiednn_eltwise_forward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, float alpha,
                                                         float beta, const aiednn_memory_desc_t *s,
                                                         const aiednn_memory_desc_t *d) {
    const char *k = "eltwise";
    REQUIRE(pr && md_ok(s) && md_ok(d) && md_same_shape(s, d), k, "src and dst: same shape");
    REQUIRE(alg >= aiednn_eltwise_relu && alg <= aiednn_eltwise_selu, k, "not an eltwise algorithm");
    REQUIRE(alg != aiednn_eltwise_soft_relu || alpha != 0, k, "soft_relu needs alpha != 0");
    rt_prob p;
    prob_init(&p, aiednn_eltwise, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.alpha = alpha;
    p.beta = beta;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_eltwise_forward_create_attr(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                              float alpha, float beta, const aiednn_memory_desc_t *s,
                                                              const aiednn_memory_desc_t *d,
                                                              const aiednn_attr_t *attr) {
    const char *k = "eltwise";
    if (!attr)
        return aiednn_eltwise_forward_create(pr, alg, alpha, beta, s, d);
    REQUIRE(pr && md_ok(s) && md_ok(d) && md_same_shape(s, d), k, "src and dst: same shape");
    REQUIRE(alg >= aiednn_eltwise_relu && alg <= aiednn_eltwise_selu, k, "not an eltwise algorithm");
    REQUIRE(alg != aiednn_eltwise_soft_relu || alpha != 0, k, "soft_relu needs alpha != 0");
    const aiednn_memory_desc_t *m[2] = {s, d};
    for (int i = 0; i < 2; i++)
        REQUIRE(m[i]->data_type == aiednn_s8 || m[i]->data_type == aiednn_u8 || m[i]->data_type == aiednn_f32, k,
                "attr: s8 / u8 / f32 tensors");
    REQUIRE(!attr->n_scales && !attr->src_zero_point && !attr->wei_zero_point && !attr->wei_scales.ndims, k,
            "attr: src_scales[0] / src_zero_points[0] / dst_scale / dst_zero_point / eltwise post-ops only");
    for (int i = 0; i < attr->n_post_ops; i++)
        REQUIRE(attr->post_ops[i].kind == aiednn_post_op_eltwise, k, "attr: eltwise post-ops only");
    REQUIRE(attr->src_scales[0] >= 0, k, "src_scales[0] must be >= 0 (0: the codes as is)");
    rt_prob p;
    prob_init(&p, aiednn_eltwise, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.alpha = alpha, p.beta = beta;
    const char *err = attach_attr(&p, attr, d->ndims > 1 ? 1 : 0);
    REQUIRE(!err, k, err ? err : "");
    p.bin_sc[0] = attr->src_scales[0], p.bin_zp[0] = attr->src_zero_points[0];
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_binary_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                const aiednn_memory_desc_t *s0, const aiednn_memory_desc_t *s1,
                                                const aiednn_memory_desc_t *s2, const aiednn_memory_desc_t *d) {
    const char *k = "binary";
    REQUIRE(pr && md_ok(s0) && md_ok(s1) && md_ok(d), k, "null or malformed descriptor");
    REQUIRE(alg >= aiednn_binary_add && alg <= aiednn_binary_add_square, k, "not a binary algorithm");
    REQUIRE(md_bcast_ok(s0, d) && md_bcast_ok(s1, d), k, "sources must broadcast to dst (same rank)");
    REQUIRE((alg == aiednn_binary_select) == (s2 != nullptr) && (!s2 || (md_ok(s2) && md_bcast_ok(s2, d))), k,
            "select needs src2 (the condition), broadcastable to dst; other algorithms take none");
    rt_prob p;
    prob_init(&p, aiednn_binary, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s0);
    prob_add(&p, AIEDNN_ARG_SRC_1, s1);
    prob_add(&p, AIEDNN_ARG_DST, d);
    if (s2)
        prob_add(&p, AIEDNN_ARG_SRC_2, s2);
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_binary_create_attr(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                     const aiednn_memory_desc_t *s0, const aiednn_memory_desc_t *s1,
                                                     const aiednn_memory_desc_t *d, const aiednn_attr_t *attr) {
    const char *k = "binary";
    if (!attr)
        return aiednn_binary_create(pr, alg, s0, s1, nullptr, d);
    REQUIRE(pr && md_ok(s0) && md_ok(s1) && md_ok(d), k, "null or malformed descriptor");
    REQUIRE(alg == aiednn_binary_add || alg == aiednn_binary_sub || alg == aiednn_binary_mul ||
                alg == aiednn_binary_max || alg == aiednn_binary_min,
            k, "attr: add / sub / mul / max / min");
    REQUIRE(md_bcast_ok(s0, d) && md_bcast_ok(s1, d), k, "sources must broadcast to dst (same rank)");
    const aiednn_memory_desc_t *m[3] = {s0, s1, d};
    for (int i = 0; i < 3; i++)
        REQUIRE(m[i]->data_type == aiednn_s8 || m[i]->data_type == aiednn_u8 || m[i]->data_type == aiednn_f32, k,
                "attr: s8 / u8 / f32 tensors");
    REQUIRE(!attr->n_scales && !attr->src_zero_point && !attr->wei_zero_point && !attr->wei_scales.ndims, k,
            "attr: src_scales / src_zero_points / dst_scale / dst_zero_point / eltwise post-ops only");
    for (int i = 0; i < attr->n_post_ops; i++)
        REQUIRE(attr->post_ops[i].kind == aiednn_post_op_eltwise, k, "attr: eltwise post-ops only");
    REQUIRE(attr->src_scales[0] >= 0 && attr->src_scales[1] >= 0, k, "src_scales must be >= 0 (0: the codes as is)");
    rt_prob p;
    prob_init(&p, aiednn_binary, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s0);
    prob_add(&p, AIEDNN_ARG_SRC_1, s1);
    prob_add(&p, AIEDNN_ARG_DST, d);
    const char *err = attach_attr(&p, attr, d->ndims > 1 ? 1 : 0);
    REQUIRE(!err, k, err ? err : "");
    for (int i = 0; i < 2; i++)
        p.bin_sc[i] = attr->src_scales[i], p.bin_zp[i] = attr->src_zero_points[i];
    REQUIRE(!(attr->flags & ~AIEDNN_ATTR_TOLERANCE_1CODE), k, "attr flags: AIEDNN_ATTR_TOLERANCE_1CODE only");
    p.attr_flags = attr->flags;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_prelu_forward_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                       const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *d) {
    const char *k = "prelu";
    REQUIRE(pr && md_ok(s) && md_ok(w) && md_ok(d) && md_same_shape(s, d), k, "src and dst: same shape");
    REQUIRE(md_bcast_ok(w, d), k, "weights must broadcast to src (same rank)");
    rt_prob p;
    prob_init(&p, aiednn_prelu, 0);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    prob_add(&p, AIEDNN_ARG_DST, d);
    return create(pr, &p);
}

static aiednn_status_t aiednn_reduction_create_(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, float pnorm, float eps,
                                                const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                                                const aiednn_attr_t *attr) {
    const char *k = "reduction";
    REQUIRE(pr && md_ok(s) && md_ok(d) && s->ndims == d->ndims, k, "src and dst: same rank");
    REQUIRE(alg >= aiednn_reduction_sum && alg <= aiednn_reduction_mul_no_zeros, k, "not a reduction");
    for (int i = 0; i < s->ndims; i++)
        REQUIRE(d->dims[i] == s->dims[i] || d->dims[i] == 1, k, "each dst dim must equal src's or be 1 (reduced)");
    REQUIRE(alg < aiednn_reduction_norm_lp_sum || pnorm > 0, k, "lp norms need p > 0");
    rt_prob p;
    prob_init(&p, aiednn_reduction, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.pnorm = pnorm;
    p.eps = eps;
    if (attr) { /* post-ops / dst quantization (aiednn_rt_ptail.h) */
        const char *why = attach_attr(&p, attr, 1);
        if (why) {
            free((void *)p.oscales);
            return bad(k, why);
        }
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_reduction_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, float pnorm,
                                                   float eps, const aiednn_memory_desc_t *s,
                                                   const aiednn_memory_desc_t *d) {
    return aiednn_reduction_create_(pr, alg, pnorm, eps, s, d, nullptr);
}

extern "C" aiednn_status_t aiednn_reduction_create_attr(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, float pnorm,
                                                        float eps, const aiednn_memory_desc_t *s,
                                                        const aiednn_memory_desc_t *d, const aiednn_attr_t *attr) {
    return aiednn_reduction_create_(pr, alg, pnorm, eps, s, d, attr);
}

// ── create: spatial / normalization ────────────────────────────────────────

static aiednn_status_t aiednn_pooling_forward_create_(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                      const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                                                      const aiednn_pool_params_t *pp, const aiednn_attr_t *attr) {
    const char *k = "pooling";
    REQUIRE(pr && pp && md_ok(s) && md_ok(d) && s->ndims == 4 && d->ndims == 4, k, "src and dst rank 4 (NCHW)");
    REQUIRE(alg >= aiednn_pooling_max && alg <= aiednn_pooling_avg_exclude_padding, k, "not a pooling algorithm");
    REQUIRE(s->dims[0] == d->dims[0] && s->dims[1] == d->dims[1], k, "N and C must match");
    for (int i = 0; i < 2; i++) {
        REQUIRE(pp->kernel[i] >= 1 && pp->strides[i] >= 1 && pp->dilates[i] >= 0 && pp->padding_l[i] >= 0 &&
                    pp->padding_r[i] >= 0,
                k, "kernel, strides >= 1; dilates, padding >= 0");
        const int64_t ext = (pp->kernel[i] - 1) * (pp->dilates[i] + 1) + 1;
        REQUIRE(d->dims[2 + i] == (s->dims[2 + i] + pp->padding_l[i] + pp->padding_r[i] - ext) / pp->strides[i] + 1, k,
                "dst spatial size does not match kernel / strides / padding");
    }
    rt_prob p;
    prob_init(&p, aiednn_pooling, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.pool = *pp;
    if (attr) { /* post-ops / dst quantization (aiednn_rt_ptail.h) */
        const char *why = attach_attr(&p, attr, 1);
        if (why) {
            free((void *)p.oscales);
            return bad(k, why);
        }
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_pooling_forward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                         const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                                                         const aiednn_pool_params_t *pp) {
    return aiednn_pooling_forward_create_(pr, alg, s, d, pp, nullptr);
}

extern "C" aiednn_status_t aiednn_pooling_forward_create_attr(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                              const aiednn_memory_desc_t *s,
                                                              const aiednn_memory_desc_t *d,
                                                              const aiednn_pool_params_t *pp,
                                                              const aiednn_attr_t *attr) {
    return aiednn_pooling_forward_create_(pr, alg, s, d, pp, attr);
}

#include "aiednn_rt_nd.h"         /* 1-D / 3-D convolution, deconvolution and pooling */
#include "aiednn_rt_fpool.h"      /* f32 2-D / 3-D pooling on the AIE's FPOOL mode */
#include "aiednn_rt_rnn.h"        /* RNN / LSTM / GRU */
#include "aiednn_rt_rnn_bwd.h"    /* RNN backward (BPTT) */
#include "aiednn_rt_bwd_create.h" /* training primitives' create functions */
#include "aiednn_rt_grouped.h"    /* grouped (MoE) matmul */
#include "aiednn_rt_misc.h"       /* dropout, spatial transformer, CTC loss */
#include "aiednn_rt_gemmx.h"      /* gemm+ composites: operators as GEMMs on the matmul dispatch */
#include "aiednn_rt_linattn.h"    /* linear attention: Gated DeltaNet / KDA / GLA (ref: + gemm+linattn) */
#include "aiednn_rt_sdpa8.h"      /* int8 attention: two int8 GEMMs per head (gemm+sdpa8) */
#include "aiednn_rt_sdpa_bwd.h"   /* sdpa backward: ref: and gemm+sdpabwd */
#include "aiednn_rt_sdpa_var.h"   /* variable-length / paged-cache sdpa: ref: and gemm+sdpavar */
#include "aiednn_rt_mxq.h"        /* block-scale (MX) quantize / dequantize */
#include "aiednn_rt_layout.h"     /* memory formats: nhwc canon / wrap, set_format / get_size */
#include "aiednn_rt_sparse.h"     /* CSR src matmul: densify+ (AIE GEMM) and ref: */
#include "aiednn_rt_ptail.h"      /* post-ops on non-GEMM primitives */
#include "aiednn_rt_slice.h"      /* strided sub-tensor copy */
#include "aiednn_rt_custom.h"     /* a user operator run on the host (framework custom ops) */
#include "aiednn_rt_fused.h"      /* gated MLP, RoPE */
#include "aiednn_rt_dynq.h"       /* gen_index, dynamic quantize / dequantize */
#include "aiednn_rt_deconvbwd.h"  /* deconvolution backward as convolutions */

static aiednn_status_t aiednn_resampling_forward_create_(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                         const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                                                         const aiednn_attr_t *attr) {
    const char *k = "resampling";
    REQUIRE(pr && md_ok(s) && md_ok(d) && s->ndims == d->ndims && s->ndims >= 3 && s->ndims <= 5, k,
            "src and dst: same rank, 3..5 (N, C, spatial)");
    REQUIRE(alg == aiednn_resampling_nearest || alg == aiednn_resampling_linear, k, "nearest or linear");
    REQUIRE(s->dims[0] == d->dims[0] && s->dims[1] == d->dims[1], k, "N and C must match");
    rt_prob p;
    prob_init(&p, aiednn_resampling, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    if (attr) { /* post-ops / dst quantization (aiednn_rt_ptail.h) */
        const char *why = attach_attr(&p, attr, 1);
        if (why) {
            free((void *)p.oscales);
            return bad(k, why);
        }
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_resampling_forward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                            const aiednn_memory_desc_t *s,
                                                            const aiednn_memory_desc_t *d) {
    return aiednn_resampling_forward_create_(pr, alg, s, d, nullptr);
}

extern "C" aiednn_status_t aiednn_resampling_forward_create_attr(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                                 const aiednn_memory_desc_t *s,
                                                                 const aiednn_memory_desc_t *d,
                                                                 const aiednn_attr_t *attr) {
    return aiednn_resampling_forward_create_(pr, alg, s, d, attr);
}

extern "C" aiednn_status_t aiednn_lrn_forward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, int64_t local_size,
                                                     float alpha, float beta, float kk, const aiednn_memory_desc_t *s,
                                                     const aiednn_memory_desc_t *d) {
    const char *k = "lrn";
    REQUIRE(pr && md_ok(s) && md_ok(d) && md_same_shape(s, d) && s->ndims >= 2 && s->ndims <= 4, k,
            "src and dst: same shape, rank 2..4 (N, C, H, W)");
    REQUIRE(alg == aiednn_lrn_across_channels || alg == aiednn_lrn_within_channel, k, "not an lrn algorithm");
    REQUIRE(local_size >= 1 && dt_is_float(s->data_type) && dt_is_float(d->data_type), k,
            "local_size >= 1, floating types");
    rt_prob p;
    prob_init(&p, aiednn_lrn, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.local_size = local_size;
    p.alpha = alpha;
    p.beta = beta;
    p.lrn_k = kk;
    return create(pr, &p);
}

static aiednn_status_t aiednn_softmax_forward_create_(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, int axis,
                                                      const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                                                      const aiednn_attr_t *attr) {
    const char *k = "softmax";
    REQUIRE(pr && md_ok(s) && md_ok(d) && md_same_shape(s, d), k, "src and dst: same shape");
    REQUIRE(alg >= aiednn_softmax_accurate && alg <= aiednn_softmax_accurate_inf_as_zero, k, "not a softmax algorithm");
    REQUIRE(axis >= 0 && axis < s->ndims &&
                (dt_is_float(s->data_type) || (attr && (s->data_type == aiednn_s8 || s->data_type == aiednn_u8))) &&
                (dt_is_float(d->data_type) || (attr && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8))),
            k, "axis in range, floating types (with an attr: s8 / u8 src through src_scales, s8 / u8 dst)");
    rt_prob p;
    prob_init(&p, aiednn_softmax, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.axis = axis;
    if (attr) { /* dequantized src (aiednn_rt_phead.h) */
        const char *why = attach_src_dq(&p, attr, 1);
        REQUIRE(!why, k, why ? why : "");
    }
    if (attr && !attr_only_src_dq(attr)) { /* post-ops / dst quantization (aiednn_rt_ptail.h) */
        const char *why = attach_attr(&p, attr, 1);
        if (why) {
            free((void *)p.oscales);
            return bad(k, why);
        }
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_softmax_forward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, int axis,
                                                         const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d) {
    return aiednn_softmax_forward_create_(pr, alg, axis, s, d, nullptr);
}

extern "C" aiednn_status_t aiednn_softmax_forward_create_attr(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, int axis,
                                                              const aiednn_memory_desc_t *s,
                                                              const aiednn_memory_desc_t *d,
                                                              const aiednn_attr_t *attr) {
    return aiednn_softmax_forward_create_(pr, alg, axis, s, d, attr);
}

static aiednn_status_t norm_create(aiednn_primitive_t *pr, aiednn_primitive_kind_t kind, const aiednn_memory_desc_t *s,
                                   const aiednn_memory_desc_t *d, float eps, unsigned flags, int64_t groups,
                                   const aiednn_attr_t *attr = nullptr) {
    const char *k = kind_str(kind);
    REQUIRE(pr && md_ok(s) && md_ok(d) && md_same_shape(s, d), k, "src and dst: same shape");
    /* batch norm is per-channel affine: also 8-bit in (lut: on the AIE, aiednn_rt_lut.h), any out */
    REQUIRE((dt_is_float(s->data_type) && dt_is_float(d->data_type)) ||
                (kind == aiednn_batch_normalization &&
                 (s->data_type == aiednn_s8 || s->data_type == aiednn_u8 || dt_is_float(s->data_type))) ||
                (attr && dt_is_float(s->data_type) && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8)) ||
                (attr && kind == aiednn_layer_normalization && attr->src_scales[0] > 0 &&
                 (s->data_type == aiednn_s8 || s->data_type == aiednn_u8) &&
                 (dt_is_float(d->data_type) || d->data_type == aiednn_s8 || d->data_type == aiednn_u8)),
            k,
            "floating types (batch norm: also s8 / u8 source, any destination; layer norm with attr: s8 / u8 dst, "
            "s8 / u8 src through src_scales)");
    REQUIRE(kind == aiednn_layer_normalization || s->ndims >= 2, k, "src must be [N, C, ...]");
    REQUIRE(!(flags & ~(AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_FUSE_NORM_RELU | AIEDNN_RMS_NORM)), k,
            "unknown flag");
    REQUIRE(kind == aiednn_layer_normalization || !(flags & AIEDNN_RMS_NORM), k, "AIEDNN_RMS_NORM is for layer norm");
    const int64_t C = kind == aiednn_layer_normalization ? s->dims[s->ndims - 1] : s->dims[1];
    REQUIRE(kind != aiednn_group_normalization || (groups >= 1 && C % groups == 0), k, "groups must divide C");
    rt_prob p;
    prob_init(&p, kind, 0);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    aiednn_memory_desc_t c;
    vec_md(&c, C);
    if (kind == aiednn_batch_normalization) {
        prob_add(&p, AIEDNN_ARG_MEAN, &c);
        prob_add(&p, AIEDNN_ARG_VARIANCE, &c);
    }
    if (flags & AIEDNN_USE_SCALE)
        prob_add(&p, AIEDNN_ARG_SCALE, &c);
    if (flags & AIEDNN_USE_SHIFT)
        prob_add(&p, AIEDNN_ARG_SHIFT, &c);
    p.eps = eps;
    p.flags = flags;
    p.groups = groups;
    if (attr && kind == aiednn_layer_normalization) { /* dequantized src (aiednn_rt_phead.h) */
        const char *why = attach_src_dq(&p, attr, 1);
        REQUIRE(!why, k, why ? why : "");
    }
    if (attr && !(kind == aiednn_layer_normalization && attr_only_src_dq(attr))) {
        /* dst quantization alone: the norm's own fold (lut: / steps: / fvec:); post-ops: aiednn_rt_ptail.h */
        REQUIRE(dt_is_float(s->data_type) || p.ndq || kind == aiednn_batch_normalization, k, "attr: floating source");
        REQUIRE(attr->n_post_ops || attr->n_scales ||
                    (kind != aiednn_group_normalization && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8)),
                k, "attr without post-ops: batch / layer norm -> s8 / u8 (the dst quantization)");
        const char *err = attach_attr(&p, attr, 1);
        REQUIRE(!err, k, err ? err : "");
    }
    if (attr) { /* AIEDNN_ATTR_TOLERANCE_1CODE lets fvec: quantize its f32 rows (layer / RMS norm -> s8 / u8) */
        REQUIRE(!(attr->flags & ~AIEDNN_ATTR_TOLERANCE_1CODE), k, "attr flags: AIEDNN_ATTR_TOLERANCE_1CODE only");
        p.attr_flags = attr->flags;
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_batch_normalization_forward_inference_create(aiednn_primitive_t *p,
                                                                               const aiednn_memory_desc_t *s,
                                                                               const aiednn_memory_desc_t *d, float eps,
                                                                               unsigned flags) {
    return norm_create(p, aiednn_batch_normalization, s, d, eps, flags, 0);
}

extern "C" aiednn_status_t aiednn_layer_normalization_forward_inference_create_attr(aiednn_primitive_t *p,
                                                                                    const aiednn_memory_desc_t *s,
                                                                                    const aiednn_memory_desc_t *d,
                                                                                    float eps, unsigned flags,
                                                                                    const aiednn_attr_t *attr) {
    return norm_create(p, aiednn_layer_normalization, s, d, eps, flags, 0, attr);
}

extern "C" aiednn_status_t aiednn_batch_normalization_forward_inference_create_attr(aiednn_primitive_t *p,
                                                                                    const aiednn_memory_desc_t *s,
                                                                                    const aiednn_memory_desc_t *d,
                                                                                    float eps, unsigned flags,
                                                                                    const aiednn_attr_t *attr) {
    return norm_create(p, aiednn_batch_normalization, s, d, eps, flags, 0, attr);
}

extern "C" aiednn_status_t aiednn_layer_normalization_forward_inference_create(aiednn_primitive_t *p,
                                                                               const aiednn_memory_desc_t *s,
                                                                               const aiednn_memory_desc_t *d, float eps,
                                                                               unsigned flags) {
    return norm_create(p, aiednn_layer_normalization, s, d, eps, flags, 0);
}

extern "C" aiednn_status_t aiednn_group_normalization_forward_inference_create(aiednn_primitive_t *p, int64_t groups,
                                                                               const aiednn_memory_desc_t *s,
                                                                               const aiednn_memory_desc_t *d, float eps,
                                                                               unsigned flags) {
    return norm_create(p, aiednn_group_normalization, s, d, eps, flags, groups);
}

extern "C" aiednn_status_t
aiednn_group_normalization_forward_inference_create_attr(aiednn_primitive_t *p, int64_t groups,
                                                         const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                                                         float eps, unsigned flags, const aiednn_attr_t *attr) {
    return norm_create(p, aiednn_group_normalization, s, d, eps, flags, groups, attr);
}

// ── create: data movement ──────────────────────────────────────────────────

extern "C" aiednn_status_t aiednn_concat_create(aiednn_primitive_t *pr, int n, const aiednn_memory_desc_t *srcs,
                                                int axis, const aiednn_memory_desc_t *d) {
    const char *k = "concat";
    REQUIRE(!d || !dt_is_4bit(d->data_type), k, "4-bit types: matmul / inner product weights and reorder only");
    REQUIRE(pr && srcs && md_ok(d) && n >= 1 && n <= AIEDNN_MAX_SRCS && axis >= 0 && axis < d->ndims, k,
            "1..AIEDNN_MAX_SRCS sources, axis in range");
    int64_t total = 0;
    rt_prob p;
    prob_init(&p, aiednn_concat, 0);
    for (int i = 0; i < n; i++) {
        const aiednn_memory_desc_t *s = &srcs[i];
        REQUIRE(md_ok(s) && s->ndims == d->ndims && s->data_type == d->data_type, k, "sources: dst's rank and type");
        for (int j = 0; j < d->ndims; j++)
            REQUIRE(j == axis || s->dims[j] == d->dims[j], k, "sources must match dst except along axis");
        total += s->dims[axis];
        prob_add(&p, AIEDNN_ARG_MULTIPLE_SRC + i, s);
    }
    REQUIRE(total == d->dims[axis], k, "dst's axis must be the sum of the sources'");
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.axis = axis;
    p.nsrc = n;
    return create(pr, &p);
}

/* int8 concat (oneDNN's int8_concat_fusion: dequantize each source -> concat -> quantize): 8-bit sources with their
 * own (scale, zero point), an 8-bit dst with its own; a source already in dst's quantization is a plain copy */
extern "C" aiednn_status_t aiednn_concat_create_q(aiednn_primitive_t *pr, int n, const aiednn_memory_desc_t *srcs,
                                                  const float *scales, const int32_t *zero_points, int axis,
                                                  const aiednn_memory_desc_t *d, float dst_scale,
                                                  int32_t dst_zero_point) {
    const char *k = "concat";
    REQUIRE(pr && srcs && scales && md_ok(d) && n >= 1 && n <= AIEDNN_MAX_SRCS && axis >= 0 && axis < d->ndims &&
                dst_scale > 0 && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8),
            k, "1..AIEDNN_MAX_SRCS sources, axis in range, an s8 / u8 dst, dst_scale > 0");
    int64_t total = 0;
    rt_prob p;
    prob_init(&p, aiednn_concat, 0);
    for (int i = 0; i < n; i++) {
        const aiednn_memory_desc_t *s = &srcs[i];
        REQUIRE(md_ok(s) && s->ndims == d->ndims && (s->data_type == aiednn_s8 || s->data_type == aiednn_u8) &&
                    scales[i] > 0,
                k, "sources: s8 / u8 of dst's rank, scale > 0");
        for (int j = 0; j < d->ndims; j++)
            REQUIRE(j == axis || s->dims[j] == d->dims[j], k, "sources must match dst except along axis");
        total += s->dims[axis];
        prob_add(&p, AIEDNN_ARG_MULTIPLE_SRC + i, s);
        p.scales[i] = scales[i], p.zps[i] = zero_points ? zero_points[i] : 0;
    }
    REQUIRE(total == d->dims[axis], k, "dst's axis must be the sum of the sources'");
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.axis = axis, p.nsrc = n;
    p.has_attr = 1, p.dst_scale = dst_scale, p.dst_zp = dst_zero_point;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_sum_create(aiednn_primitive_t *pr, int n, const float *scales,
                                             const aiednn_memory_desc_t *srcs, const aiednn_memory_desc_t *d) {
    const char *k = "sum";
    REQUIRE(pr && srcs && md_ok(d) && n >= 1 && n <= AIEDNN_MAX_SRCS, k, "1..AIEDNN_MAX_SRCS sources");
    rt_prob p;
    prob_init(&p, aiednn_sum, 0);
    for (int i = 0; i < n; i++) {
        REQUIRE(md_ok(&srcs[i]) && md_same_shape(&srcs[i], d), k, "sources must have dst's shape");
        prob_add(&p, AIEDNN_ARG_MULTIPLE_SRC + i, &srcs[i]);
        p.scales[i] = scales ? scales[i] : 1.0f;
    }
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.nsrc = n;
    return create(pr, &p);
}

static aiednn_status_t reorder_create_(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                       const aiednn_memory_desc_t *d, const int *perm, float scale, int32_t zero_point,
                                       const aiednn_attr_t *attr);

extern "C" aiednn_status_t aiednn_reorder_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                 const aiednn_memory_desc_t *d, const int *perm, float scale,
                                                 int32_t zero_point) {
    return reorder_create_(pr, s, d, perm, scale, zero_point, nullptr);
}

/* reorder + post-ops (oneDNN's reorder + sum / binary, x8 reorder chains): the reorder (dequantizing an integer
 * src by scale / zero_point) into f32, then the attr's post-ops and dst quantization (aiednn_rt_ptail.h) */
extern "C" aiednn_status_t aiednn_reorder_create_attr(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                      const aiednn_memory_desc_t *d, const int *perm, float scale,
                                                      int32_t zero_point, const aiednn_attr_t *attr) {
    return reorder_create_(pr, s, d, perm, scale, zero_point, attr);
}

static aiednn_status_t reorder_create_(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                       const aiednn_memory_desc_t *d, const int *perm, float scale, int32_t zero_point,
                                       const aiednn_attr_t *attr) {
    const char *k = "reorder";
    REQUIRE(pr && md_ok(s) && md_ok(d) && s->ndims == d->ndims && scale != 0, k, "same rank, scale != 0");
    rt_prob p;
    prob_init(&p, aiednn_reorder, 0);
    int seen = 0;
    for (int i = 0; i < d->ndims; i++) {
        const int pi = perm ? perm[i] : i;
        REQUIRE(pi >= 0 && pi < s->ndims && !(seen & (1 << pi)), k, "perm must be a permutation");
        seen |= 1 << pi;
        REQUIRE(d->dims[i] == s->dims[pi], k, "dst dim i must be src dim perm[i]");
        p.perm[i] = pi;
    }
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.scale = scale;
    p.zero_point = zero_point;
    if (attr) {
        const char *why = attach_attr(&p, attr, 1);
        if (why) {
            free((void *)p.oscales);
            return bad(k, why);
        }
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_shuffle_forward_create(aiednn_primitive_t *pr, int axis, int64_t group_size,
                                                         const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d) {
    const char *k = "shuffle";
    REQUIRE(s && !dt_is_4bit(s->data_type), k, "4-bit types: matmul / inner product weights and reorder only");
    REQUIRE(pr && md_ok(s) && md_ok(d) && md_same_shape(s, d) && s->data_type == d->data_type, k,
            "src and dst: same shape and type");
    REQUIRE(axis >= 0 && axis < s->ndims && group_size >= 1 && s->dims[axis] % group_size == 0, k,
            "axis in range, group_size divides it");
    rt_prob p;
    prob_init(&p, aiednn_shuffle, 0);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.axis = axis;
    p.group_size = group_size;
    return create(pr, &p);
}

// ── shorthands ─────────────────────────────────────────────────────────────

extern "C" aiednn_status_t aiednn_matmul_primitive_create(aiednn_primitive_t *primitive,
                                                          const aiednn_memory_desc_t *src,
                                                          const aiednn_memory_desc_t *weights,
                                                          const aiednn_memory_desc_t *dst) {
    return aiednn_matmul_forward_create(primitive, src, weights, nullptr, dst, 1);
}

extern "C" aiednn_status_t aiednn_convolution_primitive_create(aiednn_primitive_t *primitive,
                                                               const aiednn_memory_desc_t *src,
                                                               const aiednn_memory_desc_t *weights,
                                                               const aiednn_memory_desc_t *dst, int stride) {
    aiednn_conv_params_t cp;
    memset(&cp, 0, sizeof cp);
    cp.strides[0] = cp.strides[1] = stride;
    cp.groups = 1;
    return aiednn_convolution_forward_create(primitive, src, weights, nullptr, dst, &cp);
}

// ── API: descriptors, execute, introspection ───────────────────────────────

extern "C" aiednn_status_t aiednn_memory_desc_init(aiednn_memory_desc_t *md, int ndims, const int64_t *dims,
                                                   aiednn_data_type_t data_type) {
    if (!md || !dims || ndims < 1 || ndims > AIEDNN_MAX_NDIMS || data_type < aiednn_s8 || data_type > aiednn_f4_e2m1)
        return aiednn_invalid_arguments;
    memset(md, 0, sizeof *md);
    md->ndims = ndims;
    md->data_type = data_type;
    for (int i = 0; i < ndims; i++) {
        if (dims[i] <= 0)
            return aiednn_invalid_arguments;
        md->dims[i] = dims[i];
    }
    return aiednn_success;
}

static aiednn_status_t execute_args_body(aiednn_primitive_t primitive, int nargs, const aiednn_exec_arg_t *args);

/* An execute under the "ref:" filter is a check (an app's VERIFY_GEMM rerun of a layer on the CPU): its
 * epilogue / packing time is kept out of aiednn_phase, which otherwise reported the checks' ref: epilogues
 * as the measured run's (yolov8n: "epilogue=222 ms" of which the AIE layers' own was a few ms), nor in
 * aiednn_epilogue's counts. */
extern "C" aiednn_status_t aiednn_primitive_execute_args(aiednn_primitive_t primitive, int nargs,
                                                         const aiednn_exec_arg_t *args) {
    const int check = g_filter && strcmp(g_filter, "ref:") == 0;
    uint64_t saved[RT_PH_N];
    const uint64_t fx = g_fx_execs, epi = g_epi_execs, core = g_fx_core; /* aiednn_epilogue's counts too */
    if (check)
        memcpy(saved, g_phase, sizeof saved);
    const aiednn_status_t st = execute_args_body(primitive, nargs, args);
    if (check)
        memcpy(g_phase, saved, sizeof saved), g_fx_execs = fx, g_epi_execs = epi, g_fx_core = core;
    return st;
}

/* Megakernel switches priced in (aiednn_prefer_fastest). A library build keeps its modes in several core ELFs;
 * an op whose mode is not in the resident one reloads an ELF first (aiednn_megakernel_need), and the next op
 * needing the old one reloads it back: ~0.6 ms each on AIE-ML (VEK280), 20-27 ms on VEK385 (phase2's graph
 * runs: tiny_llama's nine lut: calls 206 ms). Dispatch cannot see that -- which ELF is resident depends on
 * the op before. So the first execute that switched also runs the primitive's ref: candidate, timed, and
 * keeps ref: when it beats the AIE run (which included the switch in) plus one switch back. Only switching
 * executes pay the trial; outputs are valid either way (ref: wrote dst last). */
extern "C" long aiednn_megakernel_switches __attribute__((weak));
#if defined(AIE_GEN) && AIE_GEN >= 5
#define RT_MK_SWITCH_MS 25.0
#else
#define RT_MK_SWITCH_MS 0.6
#endif
static long g_sw_trials, g_sw_to_ref;

static long rt_mk_switches(void) {
    static long fake; /* tests: AIEDNN_TEST_MK_SWITCH makes every execute look like a switch (no ELFs off the board) */
    if (rt_env_on("AIEDNN_TEST_MK_SWITCH"))
        return ++fake;
    return &aiednn_megakernel_switches ? aiednn_megakernel_switches : 0;
}

static void rt_switch_trial(aiednn_primitive *prim, void *const *ptr, double aie_ms) {
    prim->sw_checked = 1;
    int c = -1;
    for (int i = 0; i < prim->ncand; i++)
        if (strncmp(prim->cand[i]->name, "ref:", 4) == 0 && prim->cand[i] != prim->impl)
            c = i;
    if (c < 0 || rt_tune_overlaps(&prim->prob, ptr))
        return;
    g_sw_trials++;
    const rt_impl *r = prim->cand[c];
    const rt_bind keep = prim->bind;
    prim->bind = prim->cbind[c];
    const uint64_t t0 = rt_ticks();
    const int rc = r->run(r, prim, ptr);
    const double ref_ms = rt_ticks_ms(rt_ticks() - t0);
    if (rc == AIEK_OK && ref_ms < aie_ms + RT_MK_SWITCH_MS) {
        prim->impl = r, g_sw_to_ref++;
        rt_prim_info(prim);
        if (g_verbose)
            printf("aiednn_verbose,switch,%s,ref: %.3f ms < %.3f + %.1f ms,%s\n", kind_str(prim->prob.kind), ref_ms,
                   aie_ms, RT_MK_SWITCH_MS, prim->info);
        return;
    }
    prim->bind = keep;
}

static aiednn_status_t execute_args_body(aiednn_primitive_t primitive, int nargs, const aiednn_exec_arg_t *args) {
    if (!primitive || !primitive->in_use || (nargs > 0 && !args))
        return aiednn_invalid_arguments;
    const rt_prob *p = &primitive->prob;
    void *ptr[RT_MAX_MD];
    void *qdst = nullptr, *qsrc1 = nullptr; /* a requant_binary primitive: the real dst and src1 (the tail's) */
    void *pdst = nullptr;                   /* a post-op tail (aiednn_rt_ptail.h): the caller's dst */
    for (int a = 0; primitive->qres && a < nargs; a++) {
        if (args[a].arg == AIEDNN_ARG_DST)
            qdst = args[a].data;
        if (args[a].arg == primitive->qtail_src1)
            qsrc1 = args[a].data;
    }
    if (primitive->qres && (!qdst || (primitive->qtail_src1 && !qsrc1)))
        return aiednn_invalid_arguments;
    for (int a = 0; a < nargs; a++) /* sdpa: a score scale known only at execute (a framework graph's constant input) */
        if (args[a].arg == AIEDNN_ARG_ATTR_SCALE && args[a].data && p->kind == aiednn_sdpa)
            primitive->prob.scale = *(const float *)args[a].data;
    if (primitive->qres && primitive->qs1_cvt) { /* src1 to dst's format (create: the tail's formats differed) */
        const aiednn_memory_desc_t *m = &primitive->qs1_md;
        void *buf = scratch(&primitive->qs1_buf, &primitive->qs1_cap, (size_t)dt_size(m->data_type, md_numel(m)));
        if (!buf)
            return aiednn_out_of_memory;
        const uint64_t tf = rt_ticks();
        rt_fmt_convert(m, qsrc1, buf, primitive->qs1_cvt == 1);
        g_phase[RT_PH_FMT] += rt_ticks() - tf;
        qsrc1 = buf;
    }
    /* an 8-bit tail fuses into the GEMM half's output write: dst holds the intermediate, then the final codes
     * (g_qfuse; an impl that does not claim it gets the tail as an in-place pass below) */
    uint8_t q8[256];
    const int qd8 =
        primitive->qres && dt_bytes(primitive->qtail.md[rt_slot(&primitive->qtail, AIEDNN_ARG_DST)].data_type) == 1;
    const int fuse =
        qd8 && !primitive->fmt_mask && !rt_env_on("AIEDNN_NO_QFUSE") &&
        (primitive->qtail.kind == aiednn_eltwise ? 1 : qres_lut2_ok(primitive) && qres_lut2(primitive) == AIEK_OK);
    if (fuse && primitive->qtail.kind == aiednn_eltwise)
        qelt_lut8(primitive, q8);
    for (int s = 0; s < p->nmd; s++) {
        ptr[s] = nullptr;
        for (int a = 0; a < nargs; a++)
            if (args[a].arg == p->arg[s])
                ptr[s] = args[a].data;
        if (fuse && p->arg[s] == AIEDNN_ARG_DST) /* the caller's dst: intermediate, then final codes in place */
            ptr[s] = qdst;
        else if (primitive->ptail && p->arg[s] == AIEDNN_ARG_DST) { /* the op writes an f32 tile, the tail dst */
            pdst = ptr[s];
            ptr[s] = ptail_scratch(primitive);
            if (!ptr[s] || !pdst)
                return !pdst ? aiednn_invalid_arguments : aiednn_out_of_memory;
        } else if (primitive->qres && p->arg[s] == AIEDNN_ARG_DST) { /* the GEMM half writes the int8 intermediate */
            ptr[s] = scratch(&g_qres_buf, &g_qres_cap, (size_t)md_numel(&p->md[s]));
            if (!ptr[s])
                return aiednn_out_of_memory;
        }
        if (!ptr[s]) {
            if (g_verbose)
                printf("aiednn_verbose,exec,%s,missing argument %d\n", kind_str(p->kind), p->arg[s]);
            return aiednn_invalid_arguments;
        }
    }
    if ((primitive->ndq || primitive->dq_packed) && phead_in(primitive, ptr) != AIEK_OK) /* aiednn_rt_phead.h */
        return aiednn_out_of_memory;
    void *user[RT_MAX_MD]; /* wrap: the caller's buffers while ptr[] holds the dense copies */
    if (primitive->fmt_mask) {
        const int rf = rt_fmt_in(primitive, ptr, user);
        if (rf != AIEK_OK)
            return aiednn_out_of_memory;
    }
    if (g_tune && !primitive->tuned && primitive->ncand > 1) { /* time every acceptor once, keep the fastest */
        const int rc = rt_tune_execute(primitive, ptr);
        if (rc != RT_NOT_TAKEN) {
            if (rc == AIEK_OK && primitive->fmt_mask)
                rt_fmt_out(primitive, ptr, user);
            return rc == AIEK_OK ? aiednn_success : rc == AIEK_ERR_ALLOC ? aiednn_out_of_memory : aiednn_runtime_error;
        }
    }
    const rt_impl *impl = primitive->impl;
    const uint64_t t0 = rt_ticks();
    g_exec_const_w = primitive->const_w;
    if (fuse) {
        const int el = primitive->qtail.kind == aiednn_eltwise;
        g_qfuse = {1,
                   0,
                   (uint8_t *)qdst,
                   el ? q8 : nullptr,
                   el ? nullptr : primitive->qlut2,
                   el ? nullptr : (const uint8_t *)qsrc1,
                   0};
    }
    const long sw0 = rt_mk_switches();
    int rc = impl->run(impl, primitive, ptr);
    g_exec_const_w = 0;
    if (rc == AIEK_OK && rt_cost_refusals() && !primitive->sw_checked && rt_mk_switches() != sw0 && !fuse &&
        primitive->ncand > 1) /* this execute reloaded a core ELF: is ref: cheaper than the switches? */
        rt_switch_trial(primitive, ptr, rt_ticks_ms(rt_ticks() - t0));
    const int fused = fuse && g_qfuse.claim;
    g_qtail_n += primitive->qres != 0, g_qtail_fused += fused;
    if (fuse)
        g_phase[RT_PH_QTAIL] += g_qfuse.t;
    g_qfuse.on = 0;
    if (rc == AIEK_OK && primitive->fmt_mask) {
        rt_fmt_out(primitive, ptr, user);
        for (int s = 0; s < p->nmd; s++)
            ptr[s] = user[s];
    }
    if (rc == AIEK_OK && primitive->ptail)
        rc = ptail_run(primitive, (const float *)ptr[rt_slot(p, AIEDNN_ARG_DST)], pdst, nargs, args);
    if (rc == AIEK_OK && primitive->qres && !fused) { /* fuse && !fused: the same tail, in place on dst */
        const uint64_t tq = rt_ticks();
        if (primitive->qtail.kind == aiednn_eltwise) {
            rc = qelt_tail(primitive, (const uint8_t *)ptr[rt_slot(p, AIEDNN_ARG_DST)], qdst);
        } else {
            void *tp[3] = {ptr[rt_slot(p, AIEDNN_ARG_DST)], qsrc1, qdst};
            rc = qres_tail2(primitive, tp);
        }
        g_phase[RT_PH_QTAIL] += rt_ticks() - tq;
    }
    ((rt_impl *)impl)->t_total += rt_ticks() - t0;
    ((rt_impl *)impl)->execs++;
    if (g_verbose)
        printf("aiednn_verbose,exec,%s,%s,rc=%d\n", kind_str(p->kind), primitive->info, rc);
    return rc == AIEK_OK ? aiednn_success : rc == AIEK_ERR_ALLOC ? aiednn_out_of_memory : aiednn_runtime_error;
}

extern "C" aiednn_status_t aiednn_primitive_set_constant_weights(aiednn_primitive_t primitive, int constant) {
    if (!primitive)
        return aiednn_invalid_arguments;
    primitive->const_w = constant != 0;
    return aiednn_success;
}

extern "C" aiednn_status_t aiednn_primitive_execute(aiednn_primitive_t primitive, const void *src, const void *weights,
                                                    void *dst) {
    const aiednn_exec_arg_t args[3] = {
        {AIEDNN_ARG_SRC, (void *)src}, {AIEDNN_ARG_WEIGHTS, (void *)weights}, {AIEDNN_ARG_DST, dst}};
    return aiednn_primitive_execute_args(primitive, 3, args);
}

extern "C" const char *aiednn_primitive_impl_info(aiednn_primitive_t primitive) {
    return primitive && primitive->impl ? primitive->info : "none";
}

extern "C" void aiednn_primitive_destroy(aiednn_primitive_t primitive) {
    if (!primitive)
        return;
    free((void *)primitive->prob.oscales);
    primitive->prob.oscales = nullptr;
    free(primitive->wsum);
    primitive->wsum = nullptr;
    free(primitive->qlut2);
    primitive->qlut2 = nullptr;
    for (int s = 0; s < RT_MAX_MD; s++)
        free(primitive->fmt_buf[s]);
    free(primitive->qs1_buf);
    free((void *)primitive->pt_ep.oscales);
    primitive->in_use = 0;
    free(primitive);
}

extern "C" void aiednn_set_verbose(int level) { g_verbose = level; }

extern "C" void aiednn_set_impl_filter(const char *prefix) { // copied: callers may pass a temporary
    static char buf[64];
    if (!prefix) {
        g_filter = nullptr;
        return;
    }
    snprintf(buf, sizeof buf, "%s", prefix);
    g_filter = buf;
}

extern "C" void aiednn_kernel_stats(void) __attribute__((weak)); // a kernel source may report its phases

extern "C" void aiednn_kernel_stats_reset(void) __attribute__((weak)); // a kernel source's own counters

/* Op boundaries (the emitted model app calls aiednn_op_begin(i) before op i executes). With several
 * megakernels per model (dnn_build modes="auto" splitting the ops into segments whose modes fit a
 * core's program memory), the generated implementation list carries op -> megakernel and the
 * persistent runtime's aiednn_megakernel_select reloads the cores' ELF when a segment starts. */
extern "C" int aiednn_cur_op = -1;
extern "C" const int aiednn_mk_nops __attribute__((weak));
extern "C" const unsigned char aiednn_mk_of_op[] __attribute__((weak));
extern "C" void aiednn_megakernel_select(int k) __attribute__((weak));
extern "C" void aiednn_op_begin(int i) {
    aiednn_cur_op = i;
    if (&aiednn_mk_nops && aiednn_mk_of_op && aiednn_megakernel_select && i >= 0 && i < aiednn_mk_nops)
        aiednn_megakernel_select(aiednn_mk_of_op[i]);
}

/* zero every counter aiednn_print_stats reports (an app's warm-up run, then the timed one) */
extern "C" void aiednn_stats_reset(void) {
    rt_init();
    for (int i = 0; i < g_nimpls; i++)
        g_impls[i].execs = 0, g_impls[i].launches = 0, g_impls[i].t_total = 0, g_impls[i].t_entry = 0;
    memset(g_phase, 0, sizeof g_phase);
    g_fx_execs = 0, g_epi_execs = 0, g_fx_core = 0, g_qtail_n = 0, g_qtail_fused = 0;
    if (aiednn_kernel_stats_reset)
        aiednn_kernel_stats_reset();
}

extern "C" void aiednn_print_stats(void) {
    rt_init();
    printf("aiednn_stats: implementation                     execs  aie_launches   total_ms  in_entry_ms\n");
    for (int i = 0; i < g_nimpls; i++)
        if (g_impls[i].execs || g_impls[i].launches)
            printf("aiednn_stats: %-32s %6lu %12lu %10.2f %12.2f\n", g_impls[i].name, g_impls[i].execs,
                   g_impls[i].launches, rt_ticks_ms(g_impls[i].t_total), rt_ticks_ms(g_impls[i].t_entry));
    printf("aiednn_phase: im2col_pack=%.2f a_pack=%.2f b_pack=%.2f c_accum=%.2f epilogue=%.2f dw_pack=%.2f (ms)\n",
           rt_ticks_ms(g_phase[RT_PH_IM2COL]), rt_ticks_ms(g_phase[RT_PH_APACK]), rt_ticks_ms(g_phase[RT_PH_BPACK]),
           rt_ticks_ms(g_phase[RT_PH_ACCUM]), rt_ticks_ms(g_phase[RT_PH_EPI]), rt_ticks_ms(g_phase[RT_PH_DWPACK]));
    printf("aiednn_phase: hwc_transpose=%.2f conv_region=%.2f c_unpack=%.2f (ms; NCHW -> HWC of conv inputs, "
           "direct-conv A packing, result tiles out of the device buffers)\n",
           rt_ticks_ms(g_phase[RT_PH_HWC]), rt_ticks_ms(g_phase[RT_PH_REGION]), rt_ticks_ms(g_phase[RT_PH_CUNPACK]));
    printf(
        "aiednn_phase: zp_pad=%.2f zp_wsum=%.2f qres_tail=%.2f fmt_convert=%.2f (ms; zp+'s shifted / padded src copy "
        "and weight sums, requant_* post-op tails, nhwc <-> dense around primitives without an nhwc path)\n",
        rt_ticks_ms(g_phase[RT_PH_ZPPAD]), rt_ticks_ms(g_phase[RT_PH_ZPWSUM]), rt_ticks_ms(g_phase[RT_PH_QTAIL]),
        rt_ticks_ms(g_phase[RT_PH_FMT]));
    if (g_phase[RT_PH_SDPA_SM] | g_phase[RT_PH_SDPA_GEMM])
        printf("aiednn_phase: sdpa_softmax=%.2f sdpa_rows=%.2f sdpa_gemm=%.2f sdpa_out=%.2f (ms; gemm+sdpa: host "
               "softmax rows, "
               "head row copies, its inner GEMM calls, the output scatter)\n",
               rt_ticks_ms(g_phase[RT_PH_SDPA_SM]), rt_ticks_ms(g_phase[RT_PH_SDPA_ROWS]),
               rt_ticks_ms(g_phase[RT_PH_SDPA_GEMM]), rt_ticks_ms(g_phase[RT_PH_SDPA_OUT]));
    if (g_sw_trials)
        printf("aiednn_switch_guard: %ld primitives reloaded a core ELF, %ld moved to ref: (switch priced %.1f ms)\n",
               g_sw_trials, g_sw_to_ref, RT_MK_SWITCH_MS);
    printf("aiednn_qtail: %lu of %lu requant post-op tails fused into the GEMM output write\n", g_qtail_fused,
           g_qtail_n);
    printf("aiednn_epilogue: integer (fixed-point) %llu of %llu GEMM-family epilogues, %llu GEMMs requantized on the "
           "AIE\n",
           (unsigned long long)g_fx_execs, (unsigned long long)g_epi_execs, (unsigned long long)g_fx_core);
    if (aiednn_kernel_stats)
        aiednn_kernel_stats();
}

extern "C" const char *aiednn_status_str(aiednn_status_t status) {
    switch (status) {
    case aiednn_success:
        return "success";
    case aiednn_unimplemented:
        return "unimplemented";
    case aiednn_invalid_arguments:
        return "invalid_arguments";
    case aiednn_runtime_error:
        return "runtime_error";
    case aiednn_out_of_memory:
        return "out_of_memory";
    }
    return "?";
}
#include "aiednn_rt_stream.h" /* streams, caller workspace, async execute, find / set_impl */
#if __has_include("aiednn_rt_graph.h")
#include "aiednn_rt_graph.h" /* the partition graph API (aiednn_graph.h), on the public calls above */
#endif
