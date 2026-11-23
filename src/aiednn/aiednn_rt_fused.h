/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_fused.h -- fused transformer primitives the graph fusion patterns land on:
 *   gated MLP   dst = (act(src W_gate) * (src W_up)) W_down          (LLaMA / Mistral SwiGLU, GeGLU)
 *   RoPE        dst = src * cos + rotate(src) * sin                   (half-split LLaMA / HF, interleaved GPT-J)
 * gated MLP: gemm+gmlp runs its three GEMMs on the AIE (f32 as bf16x3 through gx_gemm), the gating on the host;
 * ref: in double. RoPE: f32 arithmetic in the unfused chain's order (x*cos, rotated*sin, add -- each rounded to
 * f32), so it is bit-identical to the framework's three ops.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_gemmx.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_FUSED_H
#define AIEDNN_RT_FUSED_H

// ── gated MLP ─────────────────────────────────────────────────────────────

typedef struct {
    int64_t M, D, F;
} gmlp_geo;

static void gmlp_geo_of(const rt_prob *p, gmlp_geo *g) {
    const aiednn_memory_desc_t *wg = rt_md(p, AIEDNN_ARG_WEIGHTS);
    const int bt = p->transposed != 0; /* W_gate [F, D] */
    g->D = wg->dims[bt], g->F = wg->dims[!bt], g->M = md_numel(&p->md[0]) / g->D;
}

/* the three products through mm (C [M][N] = A [M][K] x B [K][N]), the gating in double between them */
/* bt: B is [N][K] (nn.Linear's [out, in]) instead of [K][N] */
typedef int (*gmlp_mm_fn)(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt);

static int gmlp_mm_ref(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double s = 0;
            for (int64_t k = 0; k < K; k++)
                s += (double)A[i * K + k] * (double)(bt ? B[j * K + k] : B[k * N + j]);
            C[i * N + j] = (float)s;
        }
    return AIEK_OK;
}

static int gmlp_mm_gx(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt) {
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32};
    return gx_gemm(&t, M, N, K, A, B, C, bt);
}

/* an operand as f32 (a view when it already is) */
static const float *gmlp_f32(aiednn_data_type_t dt, const void *src, int64_t n, float **own) {
    *own = nullptr;
    if (dt == aiednn_f32)
        return (const float *)src;
    float *t = (float *)rt_alloc((size_t)n * sizeof(float));
    if (t)
        for (int64_t i = 0; i < n; i++)
            t[i] = (float)rt_ld(dt, src, i);
    *own = t;
    return t;
}

static int gmlp_run(const rt_prob *p, void *const *ptr, gmlp_mm_fn mm) {
    gmlp_geo g;
    gmlp_geo_of(p, &g);
    const int sg = rt_slot(p, AIEDNN_ARG_WEIGHTS), su = rt_slot(p, AIEDNN_ARG_WEIGHTS_UP);
    const int sd = rt_slot(p, AIEDNN_ARG_WEIGHTS_DOWN), so = rt_slot(p, AIEDNN_ARG_DST);
    float *o0, *o1, *o2, *o3;
    const float *X = gmlp_f32(p->md[0].data_type, ptr[0], g.M * g.D, &o0);
    const float *Wg = gmlp_f32(p->md[sg].data_type, ptr[sg], g.D * g.F, &o1);
    const float *Wu = gmlp_f32(p->md[su].data_type, ptr[su], g.D * g.F, &o2);
    const float *Wd = gmlp_f32(p->md[sd].data_type, ptr[sd], g.F * g.D, &o3);
    float *H = (float *)rt_alloc((size_t)(2 * g.M * g.F + g.M * g.D) * sizeof(float));
    int rc = X && Wg && Wu && Wd && H ? AIEK_OK : AIEK_ERR_ALLOC;
    float *U = H + g.M * g.F, *Y = U + g.M * g.F;
    const int bt = p->transposed != 0;
    if (rc == AIEK_OK)
        rc = mm(g.M, g.F, g.D, X, Wg, H, bt);
    if (rc == AIEK_OK)
        rc = mm(g.M, g.F, g.D, X, Wu, U, bt);
    for (int64_t i = 0; rc == AIEK_OK && i < g.M * g.F; i++) /* act(gate) * up, rounded to f32 as the chain is */
        H[i] = (float)eltwise_f(p->alg, (double)H[i], p->alpha, p->beta) * U[i];
    if (rc == AIEK_OK)
        rc = mm(g.M, g.D, g.F, H, Wd, Y, bt);
    for (int64_t i = 0; rc == AIEK_OK && i < g.M * g.D; i++)
        rt_st(p->md[so].data_type, ptr[so], i, (double)Y[i]);
    rt_free(o0), rt_free(o1), rt_free(o2), rt_free(o3), rt_free(H);
    return rc;
}

static int run_ref_gmlp(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    return gmlp_run(&prim->prob, ptr, gmlp_mm_ref);
}

static int acc_gx_gmlp(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    gmlp_geo g;
    gmlp_geo_of(p, &g);
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32};
    rt_bind b2 = {nullptr, nullptr};
    const int bt = p->transposed != 0;
    return gx_inner_ok(&t, g.M, g.F, g.D, bind, why, cap, bt) && gx_inner_ok(&t, g.M, g.D, g.F, &b2, why, cap, bt);
}

static int run_gx_gmlp(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    return gmlp_run(&prim->prob, ptr, gmlp_mm_gx);
}

extern "C" aiednn_status_t aiednn_gated_mlp_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *src,
                                                   const aiednn_memory_desc_t *w_gate, const aiednn_memory_desc_t *w_up,
                                                   const aiednn_memory_desc_t *w_down, const aiednn_memory_desc_t *dst,
                                                   aiednn_alg_kind_t act, float alpha, float beta, unsigned flags) {
    const char *k = "gated_mlp";
    REQUIRE(pr && md_ok(src) && md_ok(w_gate) && md_ok(w_up) && md_ok(w_down) && md_ok(dst), k,
            "null or malformed descriptor");
    REQUIRE(!(flags & ~AIEDNN_GMLP_WEIGHTS_TRANSPOSED), k, "unknown flag");
    const int bt = (flags & AIEDNN_GMLP_WEIGHTS_TRANSPOSED) != 0;
    REQUIRE(w_gate->ndims == 2 && md_same_shape(w_gate, w_up) && w_down->ndims == 2 &&
                w_down->dims[0] == w_gate->dims[1] && w_down->dims[1] == w_gate->dims[0],
            k, "w_gate, w_up [D, F] and w_down [F, D] (transposed: [F, D] and [D, F])");
    REQUIRE(src->dims[src->ndims - 1] == w_gate->dims[bt] && md_same_shape(src, dst), k, "src, dst [..., D]");
    REQUIRE(dt_is_float(src->data_type) && dt_is_float(dst->data_type) && dt_is_float(w_gate->data_type) &&
                dt_is_float(w_up->data_type) && dt_is_float(w_down->data_type),
            k, "floating types");
    REQUIRE(act >= aiednn_eltwise_relu && act <= aiednn_eltwise_selu, k, "act: an eltwise algorithm (swish = SiLU)");
    rt_prob p;
    prob_init(&p, aiednn_gated_mlp, act);
    prob_add(&p, AIEDNN_ARG_SRC, src);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w_gate);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    prob_add(&p, AIEDNN_ARG_WEIGHTS_UP, w_up);
    prob_add(&p, AIEDNN_ARG_WEIGHTS_DOWN, w_down);
    p.alpha = alpha, p.beta = beta;
    p.transposed = bt;
    return create(pr, &p);
}

// ── RoPE ──────────────────────────────────────────────────────────────────

#define RT_ROPE_Q 2u /* rt_prob.flags: the attr's quantization (dq_sc / dq_zp of x, cos, sin; dst_scale / dst_zp) */

/* one element of a rope source: f32 as is, s8 / u8 codes dequantized as the reorder does (f32((code - zp) scale)) */
static inline float rope_ld(const rt_prob *p, int i, const void *base, int64_t off) {
    const aiednn_data_type_t dt = p->md[i == 0 ? 0 : i + 1].data_type; /* slots: src, dst, cos, sin */
    const double v = rt_ld(dt, base, off);
    if (dt != aiednn_s8 && dt != aiednn_u8)
        return (float)v;
    return (float)((v - (double)p->dq_zp[i]) * (double)p->dq_sc[i]);
}

/* the attr form: dequantizing reorders, the f32 rope (the same float operations), a quantizing reorder -- one pass,
 * bit-identical to those three primitives */
static int rope_quantized(const rt_prob *p, void *const *ptr, int64_t Dh, int64_t rows, int64_t h, int inter) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1], *cm = &p->md[2], *sm = &p->md[3];
    const int q8 = d->data_type == aiednn_s8 || d->data_type == aiednn_u8;
    const double ds = p->dst_scale, inv = 1.0 / ds, zp = p->dst_zp, lo = d->data_type == aiednn_s8 ? -128.0 : 0.0;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t j = 0; j < Dh; j++) {
            const int64_t i = r * Dh + j;
            const int64_t pj = inter ? (j ^ 1) : (j < h ? j + h : j - h);
            const float sign = inter ? ((j & 1) ? 1.0f : -1.0f) : (j < h ? -1.0f : 1.0f);
            const float x = rope_ld(p, 0, ptr[0], i), xr = sign * rope_ld(p, 0, ptr[0], r * Dh + pj);
            const float c = rope_ld(p, 1, ptr[2], bcast_off(cm, s, i)), sn = rope_ld(p, 2, ptr[3], bcast_off(sm, s, i));
            const float a = x * c, b = xr * sn, f = a + b;
            if (q8)
                ((uint8_t *)ptr[1])[i] = (uint8_t)(int)rt_quant8(f, ds, inv, zp, lo);
            else
                rt_st(d->data_type, ptr[1], i, (double)f);
        }
    return AIEK_OK;
}

static int run_ref_rope(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1], *cm = &p->md[2], *sm = &p->md[3];
    const int64_t Dh = s->dims[s->ndims - 1], rows = md_numel(s) / Dh, h = Dh / 2;
    const int inter = (p->flags & 1) != 0;
    if (p->flags & RT_ROPE_Q) /* aiednn_rope_create_attr: 8-bit sources read through, an 8-bit dst quantized */
        return rope_quantized(p, ptr, Dh, rows, h, inter);
    if (!inter && s->data_type == aiednn_f32 && d->data_type == aiednn_f32 && cm->data_type == aiednn_f32 &&
        sm->data_type == aiednn_f32 && cm->dims[cm->ndims - 1] == Dh && sm->dims[sm->ndims - 1] == Dh) {
        /* typed rows (the same float operations, in the same order): the cos / sin rows of a src row are contiguous */
        const float *x = (const float *)ptr[0], *cb = (const float *)ptr[2], *sb = (const float *)ptr[3];
        float *y = (float *)ptr[1];
        for (int64_t r = 0; r < rows; r++) {
            const float *xr = x + r * Dh, *c = cb + bcast_off(cm, s, r * Dh), *sn = sb + bcast_off(sm, s, r * Dh);
            float *o = y + r * Dh;
            for (int64_t j = 0; j < h; j++) {
                const float a = xr[j] * c[j], b = -xr[j + h] * sn[j];
                o[j] = a + b;
            }
            for (int64_t j = h; j < Dh; j++) {
                const float a = xr[j] * c[j], b = xr[j - h] * sn[j];
                o[j] = a + b;
            }
        }
        return AIEK_OK;
    }
    for (int64_t r = 0; r < rows; r++)
        for (int64_t j = 0; j < Dh; j++) {
            const int64_t i = r * Dh + j;
            /* the rotated partner: half-split cat(-x2, x1); interleaved (-x[2k+1], x[2k]) */
            const int64_t pj = inter ? (j ^ 1) : (j < h ? j + h : j - h);
            const float sign = inter ? ((j & 1) ? 1.0f : -1.0f) : (j < h ? -1.0f : 1.0f);
            const float x = (float)rt_ld(s->data_type, ptr[0], i),
                        xr = sign * (float)rt_ld(s->data_type, ptr[0], r * Dh + pj);
            const float c = (float)rt_ld(cm->data_type, ptr[2], bcast_off(cm, s, i));
            const float sn = (float)rt_ld(sm->data_type, ptr[3], bcast_off(sm, s, i));
            const float a = x * c, b = xr * sn;
            rt_st(d->data_type, ptr[1], i, (double)(a + b));
        }
    return AIEK_OK;
}

static aiednn_status_t rope_create_(aiednn_primitive_t *pr, const aiednn_memory_desc_t *src,
                                    const aiednn_memory_desc_t *cos, const aiednn_memory_desc_t *sin,
                                    const aiednn_memory_desc_t *dst, int interleaved, const aiednn_attr_t *attr) {
    const char *k = "rope";
    REQUIRE(pr && md_ok(src) && md_ok(cos) && md_ok(sin) && md_ok(dst) && md_same_shape(src, dst), k,
            "src, dst same shape");
    REQUIRE(src->ndims >= 2 && src->dims[src->ndims - 1] % 2 == 0, k, "rank >= 2, an even last dim");
    REQUIRE(md_bcast_ok(cos, src) && md_bcast_ok(sin, src), k, "cos / sin broadcast to src (same rank)");
    rt_prob p;
    prob_init(&p, aiednn_rope, 0);
    prob_add(&p, AIEDNN_ARG_SRC, src);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    prob_add(&p, AIEDNN_ARG_ROPE_COS, cos);
    prob_add(&p, AIEDNN_ARG_ROPE_SIN, sin);
    p.flags = interleaved ? 1u : 0u;
    if (!attr) {
        REQUIRE(dt_is_float(src->data_type) && dt_is_float(dst->data_type), k, "floating types");
        return create(pr, &p);
    }
    /* x / cos / sin: f32, or s8 / u8 codes with attr src_scales[0] / src_scales[1] / src_scales_23[0] (+ zero points);
     * dst: f32, or s8 / u8 quantized by dst_scale / dst_zero_point */
    REQUIRE(!attr->n_post_ops && !attr->n_scales && !attr->flags, k, "attr: source / dst quantization only");
    const aiednn_memory_desc_t *srcs[3] = {src, cos, sin};
    const float sc[3] = {attr->src_scales[0], attr->src_scales[1], attr->src_scales_23[0]};
    const int32_t zp[3] = {attr->src_zero_points[0], attr->src_zero_points[1], attr->src_zero_points_23[0]};
    for (int i = 0; i < 3; i++) {
        const int q = srcs[i]->data_type == aiednn_s8 || srcs[i]->data_type == aiednn_u8;
        REQUIRE(q ? sc[i] > 0 : dt_is_float(srcs[i]->data_type) && sc[i] == 0 && zp[i] == 0, k,
                "x / cos / sin: floating, or s8 / u8 with its attr scale > 0 (src_scales[0], [1], src_scales_23[0])");
        p.dq_sc[i] = sc[i], p.dq_zp[i] = zp[i];
    }
    const int q8 = dst->data_type == aiednn_s8 || dst->data_type == aiednn_u8;
    REQUIRE(q8 ? attr->dst_scale > 0 : dt_is_float(dst->data_type) && !(attr->dst_scale > 0) && !attr->dst_zero_point,
            k, "dst: floating, or s8 / u8 with dst_scale > 0");
    p.dst_scale = q8 ? attr->dst_scale : 1.0f, p.dst_zp = q8 ? attr->dst_zero_point : 0;
    p.flags |= RT_ROPE_Q;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_rope_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *src,
                                              const aiednn_memory_desc_t *cos, const aiednn_memory_desc_t *sin,
                                              const aiednn_memory_desc_t *dst, int interleaved) {
    return rope_create_(pr, src, cos, sin, dst, interleaved, nullptr);
}

extern "C" aiednn_status_t aiednn_rope_create_attr(aiednn_primitive_t *pr, const aiednn_memory_desc_t *src,
                                                   const aiednn_memory_desc_t *cos, const aiednn_memory_desc_t *sin,
                                                   const aiednn_memory_desc_t *dst, int interleaved,
                                                   const aiednn_attr_t *attr) {
    return rope_create_(pr, src, cos, sin, dst, interleaved, attr);
}

#endif /* AIEDNN_RT_FUSED_H */
