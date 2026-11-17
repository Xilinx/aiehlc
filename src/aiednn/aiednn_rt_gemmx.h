/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_gemmx.h -- gemm+ composites: operators reduced to plain GEMMs on the matmul dispatch.
 *
 * The operator's data movement (im2col / col2im, transposes, type widening) runs on the host; every
 * multiply-accumulate runs in an inner matmul A [M, K] x B [K, N] -> C [M, N] (C s32 for s8 x s8,
 * else f32) that dispatch picks for that shape. A composite accepts only when the inner matmul would
 * run on the AIE -- s8 x s8 on the GEMM block today, f32 / bf16 on a floating GEMM mode -- so with
 * the default preference (aiednn_prefer_aie) an operator the AIE can compute this way runs there.
 * AIEDNN_GEMMX_ANY_INNER=1 also accepts a ref: inner matmul: it checks the composites' data
 * movement in workstation emulation, where no floating GEMM mode exists.
 * Results: exact for s8 (int32 sums); within the inner floating GEMM's tolerance otherwise.
 *
 *   gemm+conv   convolution forward (floating 1-3 D, s8 3-D: 2-D s8 is im2col's), backward data and
 *               backward weights (+ bias), any groups / stride / padding / dilation
 *   gemm+deconv deconvolution forward (floating 1-3 D, s8 3-D: 2-D s8 is col2im's): per (image, group)
 *               W' [Og*nk, Cg] x src [Cg, positions], scattered into dst (col2im), then the epilogue
 *   gemm+ip     inner product backward data (dy W) and backward weights (dy^T x, + bias)
 *   zp+         2-D conv / matmul / inner product with a u8 src or an s8 src zero point: rewritten as the
 *               symmetric s8 problem (conv src pre-padded with zp - shift, padding 0) with the exact
 *               per-channel correction in rt_prob.acc_add, then dispatched like any s8 problem -- the tuned
 *               im2col / direct / depthwise / n-tile paths, on-core requant included (acc_add folds into
 *               the fixed-point bias). Ranks above gemm+conv / gemm+mm, which stay for 3-D and leftovers
 *   gemm+mm     matmul / inner product with a u8 src or an s8 src zero point (asymmetric int8):
 *               shifted to s8, the exact per-column correction added on the host (gx_zp)
 *   gemm+sdpa   attention with floating Q / K / V: per (batch, head) S = Q K^T and O = P V as GEMMs,
 *               scale / mask / causal / softmax on the host (sdpablk, the int8 mode, ranks above it)
 *   gemm+rnnbwd RNN / LSTM / GRU backward: BPTT (aiednn_rt_rnn_bwd.h) with W x, dW, dx and dU as GEMMs over all
 *               steps (only the recurrence stays per step)
 *   gemm+rnn    RNN / LSTM / GRU inference: per (layer, direction) one input-projection GEMM over all
 *               steps, one recurrent GEMM per step (+ U_o (r * h) for the reset-before GRU); the
 *               gates on the host (aiednn_rt_rnn.h's formulas)
 *   dequant     matmul / inner product with group-quantized (attr.wei_scales) weights and a floating
 *               src: the weights dequantized to f32 once per execute, then the same problem (epilogue
 *               included) on the f32 weights through dispatch -- int4 / int8 groups, MXFP4 / MXFP8
 *   cvt         matmul / inner product whose operand types no GEMM mode takes (f16, f8, mixed floating
 *               types, integer weights decompressed by the output scales): src and weights widened to
 *               f32, then the same problem through dispatch
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_ntile.h (rt_env_on),
 * aiednn_rt_bwd.h (bwd_geo), aiednn_rt_nd.h (RT_D_*), aiednn_rt_rnn.h (rnn_gates / rnn_sig).
 ******************************************************************************/
#ifndef AIEDNN_RT_GEMMX_H
#define AIEDNN_RT_GEMMX_H

typedef struct {
    aiednn_data_type_t a, b, c;
} gx_types;

/* the inner GEMM's types for two operand types; 0: not reducible (4-bit, mixed integer / float) */
static int gx_types_for(aiednn_data_type_t x, aiednn_data_type_t w, gx_types *t) {
    if ((x == aiednn_s8 && (w == aiednn_s8 || w == aiednn_u8)) ||
        (x == aiednn_u8 && w == aiednn_s8)) /* u8: shifted (gx_zp) */
        t->a = t->b = aiednn_s8, t->c = aiednn_s32;
    else if (x == aiednn_bf16 && w == aiednn_bf16)
        t->a = t->b = aiednn_bf16, t->c = aiednn_f32;
    else if (dt_is_float(x) && dt_is_float(w) && !dt_is_4bit(x) && !dt_is_4bit(w))
        t->a = t->b = t->c = aiednn_f32;
    else
        return 0;
    return 1;
}

static void gx_prob(rt_prob *g, int64_t M, int64_t N, int64_t K, const gx_types *t, int bt = 0) {
    memset(g, 0, sizeof *g);
    g->kind = aiednn_matmul;
    g->scale = g->dst_scale = 1.0f;
    g->nmd = 3;
    g->arg[0] = AIEDNN_ARG_SRC, g->arg[1] = AIEDNN_ARG_WEIGHTS, g->arg[2] = AIEDNN_ARG_DST;
    md2(&g->md[0], M, K, t->a);
    if (bt) /* B given as [N][K] */
        md2(&g->md[1], N, K, t->b), g->transposed = 1;
    else
        md2(&g->md[1], K, N, t->b);
    md2(&g->md[2], M, N, t->c);
}

/* would the inner GEMM of this shape run on the AIE? */
static int gx_inner_ok(const gx_types *t, int64_t M, int64_t N, int64_t K, rt_bind *bind, char *why, int cap,
                       int bt = 0) {
    rt_prob g;
    gx_prob(&g, M, N, K, t, bt);
    const rt_impl *in = nullptr;
    rt_bind b = {nullptr, nullptr};
    if (dispatch(&g, &in, &b, 1) != aiednn_success) {
        snprintf(why, cap, "no matmul for its %s GEMM %lldx%lldx%lld", dt_str(t->a), (long long)M, (long long)N,
                 (long long)K);
        return 0;
    }
    if (strncmp(in->name, "ref:", 4) == 0 && !rt_env_on("AIEDNN_GEMMX_ANY_INNER")) {
        snprintf(why, cap, "its %s GEMM %lldx%lldx%lld would run on %s", dt_str(t->a), (long long)M, (long long)N,
                 (long long)K, in->name);
        return 0;
    }
    bind->inner = in, bind->blk = b.blk;
    return 1;
}

/* C = A x B through the matmul dispatch (the shape may differ from the one accepted) */
static int gx_gemm(const gx_types *t, int64_t M, int64_t N, int64_t K, const void *A, const void *B, void *C,
                   int bt = 0, int b_is_weights = 0) {
    rt_prob g;
    gx_prob(&g, M, N, K, t, bt);
    const rt_impl *in = nullptr;
    rt_bind b = {nullptr, nullptr};
    if (dispatch(&g, &in, &b, 1) != aiednn_success)
        return AIEK_ERR_ARGS;
    void *ptr[3] = {(void *)A, (void *)B, C};
    /* the inner B is a temporary (im2col columns, converted copies), never the model's constant weights:
     * the constant-weight cache keys packed blocks by pointer, so a reused scratch address would hand a
     * later call stale blocks (the outer primitive's flag must not reach it) */
    const int cw = g_exec_const_w;
    g_exec_const_w = b_is_weights ? cw : 0; /* B the primitive's own (constant) weights: cacheable */
    const int rc = run_sub(in, &g, &b, ptr);
    g_exec_const_w = cw;
    return rc;
}

/* n elements of src (type sdt, from element off) as type ddt: src itself when the types agree,
 * else converted into buf (n * dt_bytes(ddt) bytes) */
static const void *gx_view(aiednn_data_type_t sdt, const void *src, int64_t off, int64_t n, aiednn_data_type_t ddt,
                           void *buf) {
    if (sdt == ddt && !dt_is_4bit(sdt))
        return (const char *)src + off * dt_bytes(sdt);
    for (int64_t i = 0; i < n; i++)
        rt_st(ddt, buf, i, rt_ld(sdt, src, off + i));
    return buf;
}

// ── src zero points and u8 src (frameworks' asymmetric int8: onnxruntime QDQ, PT2E) ─────────
/* The AIE GEMM takes s8 x s8. src x (u8 or s8, zero point zp): x' = x - shift (shift 128 for u8, else 0)
 * is s8; padding is filled with zp - shift. Then inner = sum_valid (x - shift) w + sum_pad (zp - shift) w,
 * and the exact accumulator sum_valid (x - zp) w = inner + (shift - zp) * sum_all w[o]: the padded taps
 * cancel, so the correction is one constant per output channel. The epilogue then runs on the exact
 * int32 -- bit-exact against ref:. Weight zero points stay ref:. */
typedef struct {
    int on; /* a shift or a zero point: the corrected path */
    int shift, zp;
    int8_t pad;       /* the s8 value padding takes */
    int64_t corr_mul; /* shift - zp */
} gx_zp;

static void gx_zp_make(const rt_prob *p, aiednn_data_type_t xt, gx_zp *z) {
    z->shift = xt == aiednn_u8 ? 128 : 0;
    z->zp = p->src_zp;
    z->on = z->shift || z->zp;
    z->pad = (int8_t)(z->zp - z->shift);
    z->corr_mul = (int64_t)z->shift - z->zp;
}

// ── gemm+conv ───────────────────────────────────────────────────────────────

enum { GX_FWD, GX_BWD_DATA, GX_BWD_WEIGHTS };

/* forward geometry in the backward's form (aiednn_rt_bwd.h bwd_geo): 2-D from p->conv, 3-D adds
 * depth from p->dpth */
static void gx_fwd_geo(const rt_prob *p, bwd_geo *g) {
    const aiednn_memory_desc_t *x = &p->md[0], *w = &p->md[1], *y = &p->md[2];
    const int d3 = x->ndims == 5;
    g->N = x->dims[0], g->C = x->dims[1], g->O = y->dims[1], g->G = p->conv.groups;
    g->Cg = g->C / g->G, g->Og = g->O / g->G;
    bwd_dims3(x, g->xs), bwd_dims3(y, g->ys), bwd_dims3(w, g->k);
    g->s[0] = d3 ? p->dpth[RT_D_S] : 1, g->pl[0] = d3 ? p->dpth[RT_D_PL] : 0,
    g->dil[0] = d3 ? p->dpth[RT_D_DIL] + 1 : 1;
    for (int i = 0; i < 2; i++)
        g->s[1 + i] = p->conv.strides[i], g->pl[1 + i] = p->conv.padding_l[i], g->dil[1 + i] = p->conv.dilates[i] + 1;
}

typedef struct {
    int mode;
    bwd_geo g;
    const aiednn_memory_desc_t *x, *w, *y; /* the conv's src / weights / dst sides (or their gradients) */
    int64_t nx, ny, nk, Kd;
    gx_zp z; /* forward only */
} gx_conv;

static int gx_conv_make(const rt_prob *p, gx_conv *c) {
    c->mode = p->prop == RT_FWD_INF       ? GX_FWD
              : p->prop == RT_BWD_DATA    ? GX_BWD_DATA
              : p->prop == RT_BWD_WEIGHTS ? GX_BWD_WEIGHTS
                                          : -1;
    if (c->mode < 0)
        return 0;
    if (c->mode == GX_FWD) {
        c->x = &p->md[0], c->w = &p->md[1], c->y = &p->md[2];
        gx_fwd_geo(p, &c->g);
    } else {
        const int data = c->mode == GX_BWD_DATA;
        c->x = rt_md(p, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
        c->w = rt_md(p, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
        c->y = rt_md(p, AIEDNN_ARG_DIFF_DST);
        bwd_conv_geo(p, c->x, c->w, c->y, &c->g);
    }
    const bwd_geo *g = &c->g;
    gx_zp_make(p, c->x->data_type, &c->z);
    if (c->mode != GX_FWD)
        c->z.on = 0;
    c->nx = g->xs[0] * g->xs[1] * g->xs[2], c->ny = g->ys[0] * g->ys[1] * g->ys[2];
    c->nk = g->k[0] * g->k[1] * g->k[2], c->Kd = g->Cg * c->nk;
    return 1;
}

/* inner GEMM shape and operand types per mode: fwd W[Og,Kd] x col[Kd,ny]; data W^T[Kd,Og] x dY[Og,ny];
 * weights dY[Og,ny] x col^T[ny,Kd] */
static int gx_conv_gemm(const gx_conv *c, gx_types *t, int64_t *M, int64_t *N, int64_t *K) {
    const aiednn_data_type_t xt = c->x->data_type, wt = c->w->data_type, yt = c->y->data_type;
    if (c->mode == GX_FWD)
        return gx_types_for(wt, xt, t) && (*M = c->g.Og, *N = c->ny, *K = c->Kd, 1);
    if (c->mode == GX_BWD_DATA)
        return gx_types_for(wt, yt, t) && (*M = c->Kd, *N = c->ny, *K = c->g.Og, 1);
    return gx_types_for(yt, xt, t) && (*M = c->g.Og, *N = c->Kd, *K = c->ny, 1);
}

static int acc_gx_conv(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    gx_conv c;
    if (!gx_conv_make(p, &c)) {
        snprintf(why, cap, "inference, backward data or backward weights only");
        return 0;
    }
    if (p->wei_zp || (p->src_zp && c.mode != GX_FWD)) {
        snprintf(why, cap, "weight zero points (or a backward pass with zero points): ref:");
        return 0;
    }
    if (c.mode == GX_FWD && c.x->ndims == 4 && c.x->data_type == aiednn_s8 && !p->src_zp) {
        snprintf(why, cap, "2-D symmetric s8: im2col's");
        return 0;
    }
    gx_types t;
    int64_t M, N, K;
    if (!gx_conv_gemm(&c, &t, &M, &N, &K)) {
        snprintf(why, cap, "operand types %s x %s: no s8 / floating GEMM", dt_str(c.w->data_type),
                 dt_str(c.x->data_type));
        return 0;
    }
    return gx_inner_ok(&t, M, N, K, bind, why, cap);
}

/* the patch matrix of (image n, group grp): element (j = ci_local * nk + kk, a) is src at the tap kk
 * of output position a (0 in the padding). kn: [Kd][ny] (forward), else [ny][Kd] (backward weights) */
static void gx_im2col(const gx_conv *c, const void *x, int64_t n, int64_t grp, int kn, aiednn_data_type_t dt,
                      void *col) {
    const bwd_geo *g = &c->g;
    const aiednn_data_type_t xt = c->x->data_type;
    for (int64_t ci = 0; ci < g->Cg; ci++) {
        const int64_t xb = (n * g->C + grp * g->Cg + ci) * c->nx;
        for (int64_t kk = 0; kk < c->nk; kk++) {
            const int64_t k0 = kk / (g->k[1] * g->k[2]), k1 = kk / g->k[2] % g->k[1], k2 = kk % g->k[2];
            const int64_t j = ci * c->nk + kk;
            for (int64_t a = 0; a < c->ny; a++) {
                const int64_t a0 = a / (g->ys[1] * g->ys[2]), a1 = a / g->ys[2] % g->ys[1], a2 = a % g->ys[2];
                const int64_t b0 = a0 * g->s[0] - g->pl[0] + k0 * g->dil[0];
                const int64_t b1 = a1 * g->s[1] - g->pl[1] + k1 * g->dil[1];
                const int64_t b2 = a2 * g->s[2] - g->pl[2] + k2 * g->dil[2];
                const int in = b0 >= 0 && b0 < g->xs[0] && b1 >= 0 && b1 < g->xs[1] && b2 >= 0 && b2 < g->xs[2];
                const double v = in ? rt_ld(xt, x, xb + (b0 * g->xs[1] + b1) * g->xs[2] + b2) - c->z.shift
                                    : (c->z.on ? (double)c->z.pad : 0.0);
                rt_st(dt, col, kn ? j * c->ny + a : a * c->Kd + j, v);
            }
        }
    }
}

/* backward data: scatter-add the column gradient dcol [Kd][ny] of (n, grp) into dx (double) */
static void gx_col2im(const gx_conv *c, const void *dcol, aiednn_data_type_t dt, int64_t n, int64_t grp, double *dx) {
    const bwd_geo *g = &c->g;
    for (int64_t ci = 0; ci < g->Cg; ci++) {
        double *xb = dx + (n * g->C + grp * g->Cg + ci) * c->nx;
        for (int64_t kk = 0; kk < c->nk; kk++) {
            const int64_t k0 = kk / (g->k[1] * g->k[2]), k1 = kk / g->k[2] % g->k[1], k2 = kk % g->k[2];
            const int64_t j = ci * c->nk + kk;
            for (int64_t a = 0; a < c->ny; a++) {
                const int64_t a0 = a / (g->ys[1] * g->ys[2]), a1 = a / g->ys[2] % g->ys[1], a2 = a % g->ys[2];
                const int64_t b0 = a0 * g->s[0] - g->pl[0] + k0 * g->dil[0];
                const int64_t b1 = a1 * g->s[1] - g->pl[1] + k1 * g->dil[1];
                const int64_t b2 = a2 * g->s[2] - g->pl[2] + k2 * g->dil[2];
                if (b0 >= 0 && b0 < g->xs[0] && b1 >= 0 && b1 < g->xs[1] && b2 >= 0 && b2 < g->xs[2])
                    xb[(b0 * g->xs[1] + b1) * g->xs[2] + b2] += rt_ld(dt, dcol, j * c->ny + a);
            }
        }
    }
}

static int gx_conv_fwd(const rt_prob *p, const gx_conv *c, const gx_types *t, void *const *ptr, char *A, char *B,
                       char *Cm) {
    const bwd_geo *g = &c->g;
    rt_epi E;
    rt_epi_init(&E, p, ptr);
    for (int64_t grp = 0; grp < g->G; grp++) {
        const void *Wg = gx_view(c->w->data_type, ptr[1], grp * g->Og * c->Kd, g->Og * c->Kd, t->a, A);
        for (int64_t n = 0; n < g->N; n++) {
            gx_im2col(c, ptr[0], n, grp, 1, t->b, B);
            const int rc = gx_gemm(t, g->Og, c->ny, c->Kd, Wg, B, Cm);
            if (rc != AIEK_OK)
                return rc;
            for (int64_t ol = 0; ol < g->Og; ol++) {
                const int64_t o = grp * g->Og + ol;
                double cr = 0;
                if (c->z.on) {
                    int64_t sw = 0;
                    for (int64_t j = 0; j < c->Kd; j++)
                        sw += (int64_t)rt_ld(c->w->data_type, ptr[1], o * c->Kd + j);
                    cr = (double)(c->z.corr_mul * sw);
                }
                for (int64_t a = 0; a < c->ny; a++)
                    rt_epi_put(&E, (n * g->O + o) * c->ny + a, o, rt_ld(t->c, Cm, ol * c->ny + a) + cr);
            }
        }
    }
    return AIEK_OK;
}

static int gx_conv_bwd(const rt_prob *p, const gx_conv *c, const gx_types *t, void *const *ptr, char *A, char *B,
                       char *Cm) {
    const bwd_geo *g = &c->g;
    const int data = c->mode == GX_BWD_DATA;
    void *px = rt_ptr(p, ptr, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
    void *pw = rt_ptr(p, ptr, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    void *py = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    const int64_t nout = data ? md_numel(c->x) : md_numel(c->w);
    double *acc = (double *)rt_calloc((size_t)nout, sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    int rc = AIEK_OK;
    for (int64_t grp = 0; grp < g->G && rc == AIEK_OK; grp++) {
        if (data) /* W_g^T [Kd][Og], once per group */
            for (int64_t ol = 0; ol < g->Og; ol++)
                for (int64_t j = 0; j < c->Kd; j++)
                    rt_st(t->a, A, j * g->Og + ol, rt_ld(c->w->data_type, pw, (grp * g->Og + ol) * c->Kd + j));
        for (int64_t n = 0; n < g->N && rc == AIEK_OK; n++) {
            const int64_t yo = (n * g->O + grp * g->Og) * c->ny; /* dY_g of image n: [Og][ny], contiguous */
            if (data) {
                const void *dy = gx_view(c->y->data_type, py, yo, g->Og * c->ny, t->b, B);
                rc = gx_gemm(t, c->Kd, c->ny, g->Og, A, dy, Cm);
                if (rc == AIEK_OK)
                    gx_col2im(c, Cm, t->c, n, grp, acc);
            } else {
                const void *dy = gx_view(c->y->data_type, py, yo, g->Og * c->ny, t->a, A);
                gx_im2col(c, px, n, grp, 0, t->b, B);
                rc = gx_gemm(t, g->Og, c->Kd, c->ny, dy, B, Cm);
                for (int64_t i = 0; rc == AIEK_OK && i < g->Og * c->Kd; i++)
                    acc[grp * g->Og * c->Kd + i] += rt_ld(t->c, Cm, i);
            }
        }
    }
    for (int64_t i = 0; rc == AIEK_OK && i < nout; i++)
        rt_st(data ? c->x->data_type : c->w->data_type, data ? px : pw, i, acc[i]);
    const aiednn_memory_desc_t *db = rt_md(p, AIEDNN_ARG_DIFF_BIAS);
    if (rc == AIEK_OK && !data && db) /* the bias gradient sums dst per channel (no GEMM) */
        for (int64_t o = 0; o < g->O; o++) {
            double s = 0;
            for (int64_t n = 0; n < g->N; n++)
                for (int64_t a = 0; a < c->ny; a++)
                    s += rt_ld(c->y->data_type, py, (n * g->O + o) * c->ny + a);
            rt_st(db->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_BIAS), o, s);
        }
    rt_free(acc);
    return rc;
}

static int run_gx_conv(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    gx_conv c;
    gx_types t;
    int64_t M, N, K;
    if (!gx_conv_make(p, &c) || !gx_conv_gemm(&c, &t, &M, &N, &K))
        return AIEK_ERR_ARGS;
    /* A, B, C of the largest GEMM operand any mode uses (data / weights reuse A for W^T / dY) */
    const int64_t na = (c.Kd > c.ny ? c.Kd : c.ny) * c.g.Og, nb = c.Kd * c.ny;
    const int64_t ncm = (c.Kd > c.g.Og ? c.Kd : c.g.Og) * (c.ny > c.Kd ? c.ny : c.Kd);
    char *A = (char *)rt_alloc((size_t)(na * 4)), *B = (char *)rt_alloc((size_t)(nb * 4)),
         *Cm = (char *)rt_alloc((size_t)(ncm * 4));
    int rc = AIEK_ERR_ALLOC;
    if (A && B && Cm)
        rc = c.mode == GX_FWD ? gx_conv_fwd(p, &c, &t, ptr, A, B, Cm) : gx_conv_bwd(p, &c, &t, ptr, A, B, Cm);
    rt_free(A), rt_free(B), rt_free(Cm);
    return rc;
}

// ── gemm+deconv ─────────────────────────────────────────────────────────────

static int gx_deconv_gemm(const rt_prob *p, const bwd_geo *g, gx_types *t, int64_t *M, int64_t *N, int64_t *K) {
    *M = g->Og * g->k[0] * g->k[1] * g->k[2], *N = g->xs[0] * g->xs[1] * g->xs[2], *K = g->Cg;
    return gx_types_for(p->md[1].data_type, p->md[0].data_type, t);
}

static int acc_gx_deconv(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (p->prop != RT_FWD_INF || p->src_zp || p->wei_zp || (p->md[0].ndims == 4 && p->md[0].data_type == aiednn_s8)) {
        snprintf(why, cap, "inference without zero points; 2-D s8 is col2im's");
        return 0;
    }
    bwd_geo g;
    gx_fwd_geo(p, &g);
    gx_types t;
    int64_t M, N, K;
    if (!gx_deconv_gemm(p, &g, &t, &M, &N, &K)) {
        snprintf(why, cap, "operand types: no s8 / floating GEMM");
        return 0;
    }
    return gx_inner_ok(&t, M, N, K, bind, why, cap);
}

/* deconv: src position a (xs side) and tap kk write dst position a * s - pl + kk * dil (ys side) */
static int run_gx_deconv(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    bwd_geo g;
    gx_fwd_geo(p, &g);
    gx_types t;
    int64_t M, Na, K;
    gx_deconv_gemm(p, &g, &t, &M, &Na, &K);
    const int64_t nk = g.k[0] * g.k[1] * g.k[2], ny = g.ys[0] * g.ys[1] * g.ys[2], nd = md_numel(&p->md[2]);
    char *A = (char *)rt_alloc((size_t)(M * K * 4)), *B = (char *)rt_alloc((size_t)(K * Na * 4)),
         *Cm = (char *)rt_alloc((size_t)(M * Na * 4));
    double *acc = (double *)rt_calloc((size_t)nd, sizeof(double));
    int rc = A && B && Cm && acc ? AIEK_OK : AIEK_ERR_ALLOC;
    for (int64_t grp = 0; rc == AIEK_OK && grp < g.G; grp++) {
        for (int64_t ol = 0; ol < g.Og; ol++) /* A [(ol, kk)][c] = w[grp*Og + ol][c][kk] */
            for (int64_t c = 0; c < g.Cg; c++)
                for (int64_t kk = 0; kk < nk; kk++)
                    rt_st(t.a, A, (ol * nk + kk) * K + c,
                          rt_ld(p->md[1].data_type, ptr[1], ((grp * g.Og + ol) * g.Cg + c) * nk + kk));
        for (int64_t n = 0; rc == AIEK_OK && n < g.N; n++) {
            const void *X = gx_view(p->md[0].data_type, ptr[0], (n * g.C + grp * g.Cg) * Na, g.Cg * Na, t.b, B);
            rc = gx_gemm(&t, M, Na, K, A, X, Cm);
            for (int64_t ol = 0; rc == AIEK_OK && ol < g.Og; ol++) {
                double *yb = acc + (n * g.O + grp * g.Og + ol) * ny;
                for (int64_t kk = 0; kk < nk; kk++) {
                    const int64_t k0 = kk / (g.k[1] * g.k[2]), k1 = kk / g.k[2] % g.k[1], k2 = kk % g.k[2];
                    for (int64_t a = 0; a < Na; a++) {
                        const int64_t a0 = a / (g.xs[1] * g.xs[2]), a1 = a / g.xs[2] % g.xs[1], a2 = a % g.xs[2];
                        const int64_t b0 = a0 * g.s[0] - g.pl[0] + k0 * g.dil[0];
                        const int64_t b1 = a1 * g.s[1] - g.pl[1] + k1 * g.dil[1];
                        const int64_t b2 = a2 * g.s[2] - g.pl[2] + k2 * g.dil[2];
                        if (b0 >= 0 && b0 < g.ys[0] && b1 >= 0 && b1 < g.ys[1] && b2 >= 0 && b2 < g.ys[2])
                            yb[(b0 * g.ys[1] + b1) * g.ys[2] + b2] += rt_ld(t.c, Cm, (ol * nk + kk) * Na + a);
                    }
                }
            }
        }
    }
    if (rc == AIEK_OK) {
        rt_epi E;
        rt_epi_init(&E, p, ptr);
        for (int64_t i = 0; i < nd; i++)
            rt_epi_put(&E, i, (i / ny) % g.O, acc[i]);
    }
    rt_free(A), rt_free(B), rt_free(Cm), rt_free(acc);
    return rc;
}

// ── gemm+ip (backward) ──────────────────────────────────────────────────────

/* x [N, K] (flattened), W [O, K], dy [N, O]: data dx = dy W; weights dW = dy^T x */
static int gx_ip_make(const rt_prob *p, gx_types *t, int64_t *Nb, int64_t *O, int64_t *K) {
    const int data = p->prop == RT_BWD_DATA;
    if (!data && p->prop != RT_BWD_WEIGHTS)
        return 0;
    const aiednn_memory_desc_t *x = rt_md(p, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
    const aiednn_memory_desc_t *w = rt_md(p, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    const aiednn_memory_desc_t *y = rt_md(p, AIEDNN_ARG_DIFF_DST);
    *Nb = x->dims[0], *O = w->dims[0], *K = md_numel(x) / x->dims[0];
    return gx_types_for(y->data_type, data ? w->data_type : x->data_type, t);
}

static int acc_gx_ip(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    gx_types t;
    int64_t Nb, O, K;
    if (!gx_ip_make(p, &t, &Nb, &O, &K)) {
        snprintf(why, cap, "backward data / weights with s8 or floating operands");
        return 0;
    }
    return p->prop == RT_BWD_DATA ? gx_inner_ok(&t, Nb, K, O, bind, why, cap)
                                  : gx_inner_ok(&t, O, K, Nb, bind, why, cap);
}

static int run_gx_ip(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    gx_types t;
    int64_t Nb, O, K;
    gx_ip_make(p, &t, &Nb, &O, &K);
    const int data = p->prop == RT_BWD_DATA;
    const aiednn_memory_desc_t *y = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const void *py = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    char *A = (char *)rt_alloc((size_t)(Nb * O * 4)), *B = (char *)rt_alloc((size_t)(O * K + Nb * K) * 4);
    float *Cm = (float *)rt_alloc((size_t)((data ? Nb : O) * K) * sizeof(int32_t));
    int rc = A && B && Cm ? AIEK_OK : AIEK_ERR_ALLOC;
    if (rc == AIEK_OK && data) {
        const aiednn_memory_desc_t *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
        const void *dy = gx_view(y->data_type, py, 0, Nb * O, t.a, A);
        const void *W = gx_view(w->data_type, rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS), 0, O * K, t.b, B);
        rc = gx_gemm(&t, Nb, K, O, dy, W, Cm);
        const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
        for (int64_t i = 0; rc == AIEK_OK && i < Nb * K; i++)
            rt_st(dx->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), i, rt_ld(t.c, Cm, i));
    } else if (rc == AIEK_OK) {
        const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC), *dw = rt_md(p, AIEDNN_ARG_DIFF_WEIGHTS);
        for (int64_t n = 0; n < Nb; n++) /* dy^T [O][N] */
            for (int64_t o = 0; o < O; o++)
                rt_st(t.a, A, o * Nb + n, rt_ld(y->data_type, py, n * O + o));
        const void *X = gx_view(x->data_type, rt_ptr(p, ptr, AIEDNN_ARG_SRC), 0, Nb * K, t.b, B);
        rc = gx_gemm(&t, O, K, Nb, A, X, Cm);
        for (int64_t i = 0; rc == AIEK_OK && i < O * K; i++)
            rt_st(dw->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_WEIGHTS), i, rt_ld(t.c, Cm, i));
        const aiednn_memory_desc_t *db = rt_md(p, AIEDNN_ARG_DIFF_BIAS);
        for (int64_t o = 0; rc == AIEK_OK && db && o < O; o++) {
            double sum = 0;
            for (int64_t n = 0; n < Nb; n++)
                sum += rt_ld(y->data_type, py, n * O + o);
            rt_st(db->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_BIAS), o, sum);
        }
    }
    rt_free(A), rt_free(B), rt_free(Cm);
    return rc;
}

// ── gemm+sdpa ───────────────────────────────────────────────────────────────

static int gx_sdpa_types(const rt_prob *p, gx_types *t) {
    const aiednn_data_type_t q = p->md[0].data_type, k = p->md[1].data_type, v = p->md[2].data_type;
    if (!dt_is_float(q) || !dt_is_float(k) || !dt_is_float(v) || dt_is_4bit(q) || dt_is_4bit(k) || dt_is_4bit(v))
        return 0;
    const int bf = q == aiednn_bf16 && k == aiednn_bf16 && v == aiednn_bf16;
    t->a = t->b = bf ? aiednn_bf16 : aiednn_f32, t->c = aiednn_f32;
    return 1;
}

static int acc_gx_sdpa(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    gx_types t;
    if (!gx_sdpa_types(p, &t) || (p->flags & RT_SDPA_VAR)) {
        snprintf(why, cap, "floating Q / K / V only, fixed lengths (varlen: gemm+sdpavar)");
        return 0;
    }
    const int64_t Sq = p->md[0].dims[2], Skv = p->md[1].dims[2], D = p->md[0].dims[3], Dv = p->md[2].dims[3];
    rt_bind b2 = {nullptr, nullptr};
    return gx_inner_ok(&t, Sq, Skv, D, bind, why, cap, 1) && gx_inner_ok(&t, Sq, Dv, Skv, &b2, why, cap);
}

/* one head's [S][W] rows of a [B,HH,S,W] or [B,S,HH,W] tensor (folded head permutes, as aiednn_rt_ops.h's
 * ref:sdpa) as type tt: the tensor itself when it is [B,HH,S,W] of that type, else rows copied into buf */
static const void *gx_head_rows(aiednn_data_type_t dt, const void *base, int bshd, int64_t HH, int64_t b, int64_t h,
                                int64_t S, int64_t W, aiednn_data_type_t tt, char *buf) {
    const int64_t es = dt_bytes(tt);
    if (dt == tt && !bshd)
        return (const char *)base + ((b * HH + h) * S * W) * es;
    for (int64_t s = 0; s < S; s++) {
        const int64_t at = bshd ? ((b * S + s) * HH + h) * W : ((b * HH + h) * S + s) * W;
        if (dt == tt)
            memcpy(buf + s * W * es, (const char *)base + at * es, (size_t)(W * es));
        else
            for (int64_t e = 0; e < W; e++)
                rt_st(tt, buf, s * W + e, rt_ld(dt, base, at + e));
    }
    return buf;
}

/* e^x for x <= 0 in float, as the FVEC core computes it (aie_modes_fvec.h aie_fv_exp): n = round(x / ln 2) by
 * the round-to-nearest trick, r = x - n ln 2 in two parts (Cody-Waite), e^r a degree-7 polynomial (truncation
 * < 3e-9, one more term than the core), 2^n from the exponent bits. Plain float code the A72 compiler vectorizes; -inf
 * (a masked score) gives exactly 0 */
static inline float gx_expf(float x) {
    const float xc = x < -87.3f ? -87.3f : x;
    const float tn = xc * 1.44269504089f + 12582912.0f, nf = tn - 12582912.0f;
    int32_t n;
    memcpy(&n, &tn, 4);
    n -= 0x4B400000;
    const float r = (xc - nf * 0.693359375f) - nf * -2.12194440e-4f;
    float q = 1.0f / 5040;
    q = q * r + 1.0f / 720;
    q = q * r + 1.0f / 120;
    q = q * r + 1.0f / 24;
    q = q * r + 1.0f / 6;
    q = q * r + 0.5f;
    q = q * r + 1.0f;
    q = q * r + 1.0f;
    const int32_t e = (n + 127) << 23;
    float s2;
    memcpy(&s2, &e, 4);
    return x == -INFINITY ? 0.0f : q * s2;
}

/* one f32 score row a[0..n) -> P = softmax(a) in float: within ~1e-6 relative of the double rows, inside the
 * f32 contract the bf16x3 GEMMs around it already set. On the A72 the max / exp / sum / scale passes are NEON,
 * 4 lanes: the compiler leaves the polynomial scalar, and scalar float rows measured what the double ones did
 * (vit_b_16 sdpa_softmax 190 ms either way, ~23 cycles per score at 675 MHz) */
#if defined(__aarch64__)
#include <arm_neon.h>
static inline float32x4_t gx_expf4(float32x4_t x) {
    const uint32x4_t ninf = vceqq_f32(x, vdupq_n_f32(-INFINITY));
    const float32x4_t xc = vmaxq_f32(x, vdupq_n_f32(-87.3f));
    const float32x4_t big = vdupq_n_f32(12582912.0f);
    const float32x4_t tn = vfmaq_f32(big, xc, vdupq_n_f32(1.44269504089f)), nf = vsubq_f32(tn, big);
    const int32x4_t n = vsubq_s32(vreinterpretq_s32_f32(tn), vdupq_n_s32(0x4B400000));
    const float32x4_t r = vfmsq_f32(vfmsq_f32(xc, nf, vdupq_n_f32(0.693359375f)), nf, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t q = vdupq_n_f32(1.0f / 5040);
    q = vfmaq_f32(vdupq_n_f32(1.0f / 720), q, r);
    q = vfmaq_f32(vdupq_n_f32(1.0f / 120), q, r);
    q = vfmaq_f32(vdupq_n_f32(1.0f / 24), q, r);
    q = vfmaq_f32(vdupq_n_f32(1.0f / 6), q, r);
    q = vfmaq_f32(vdupq_n_f32(0.5f), q, r);
    q = vfmaq_f32(vdupq_n_f32(1.0f), q, r);
    q = vfmaq_f32(vdupq_n_f32(1.0f), q, r);
    const float32x4_t s2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23));
    return vbslq_f32(ninf, vdupq_n_f32(0.0f), vmulq_f32(q, s2));
}
#endif

static void gx_softmax_row_f32(float *a, int64_t n, float *P) {
    float mx = -INFINITY;
    int64_t j = 0;
#if defined(__aarch64__)
    float32x4_t m4 = vdupq_n_f32(-INFINITY);
    for (; j + 4 <= n; j += 4)
        m4 = vmaxq_f32(m4, vld1q_f32(a + j));
    mx = vmaxvq_f32(m4);
#endif
    for (; j < n; j++)
        mx = a[j] > mx ? a[j] : mx;
    if (mx == -INFINITY) {
        memset(P, 0, (size_t)n * sizeof(float));
        return;
    }
    float sum = 0;
    j = 0;
#if defined(__aarch64__)
    const float32x4_t mx4 = vdupq_n_f32(mx);
    float32x4_t s4 = vdupq_n_f32(0.0f);
    for (; j + 4 <= n; j += 4) {
        const float32x4_t e = gx_expf4(vsubq_f32(vld1q_f32(a + j), mx4));
        vst1q_f32(a + j, e);
        s4 = vaddq_f32(s4, e);
    }
    sum = vaddvq_f32(s4);
#endif
    for (; j < n; j++) {
        a[j] = gx_expf(a[j] - mx);
        sum += a[j];
    }
    const float inv = 1.0f / sum;
    j = 0;
#if defined(__aarch64__)
    const float32x4_t inv4 = vdupq_n_f32(inv);
    for (; j + 4 <= n; j += 4)
        vst1q_f32(P + j, vmulq_f32(vld1q_f32(a + j), inv4));
#endif
    for (; j < n; j++)
        P[j] = a[j] * inv;
}

/* the softmax of the score rows S [Sq][Skv] of head (b, h) into P (type t): scale, mask, causal, in double */
static void gx_sdpa_softmax(const rt_prob *p, void *const *ptr, int64_t B, int64_t H, int64_t b, int64_t h, int64_t Sq,
                            int64_t Skv, const float *S, aiednn_data_type_t t, char *P, double *row) {
    const aiednn_memory_desc_t *mk = rt_md(p, AIEDNN_ARG_ATTN_MASK);
    const void *mp = rt_ptr(p, ptr, AIEDNN_ARG_ATTN_MASK);
    aiednn_memory_desc_t sc; /* the [B, H, Sq, Skv] score shape the mask broadcasts to */
    memset(&sc, 0, sizeof sc);
    sc.ndims = 4, sc.dims[0] = B, sc.dims[1] = H, sc.dims[2] = Sq, sc.dims[3] = Skv;
    const int64_t bh = b * H + h;
    const double scale = (double)p->scale;
    if (t == aiednn_f32 && !rt_env_on("AIEDNN_SDPA_DOUBLE_SOFTMAX")) { /* float rows (gx_softmax_row_f32) */
        float *fr = (float *)row;                                      /* row holds Skv doubles: room for Skv floats */
        const float fs = (float)p->scale;
        for (int64_t i = 0; i < Sq; i++) {
            const int64_t jn = p->causal ? (i + (Skv - Sq) + 1 < Skv ? i + (Skv - Sq) + 1 : Skv) : Skv;
            const float *Si = S + i * Skv;
            if (mk)
                for (int64_t j = 0; j < Skv; j++)
                    fr[j] = j >= jn ? -INFINITY
                                    : Si[j] * fs +
                                          (float)rt_ld(mk->data_type, mp, bcast_off(mk, &sc, (bh * Sq + i) * Skv + j));
            else { /* the row max subtracted before scaling: S - max is nearly exact, (S - max) * s rounds once */
                float smx = -INFINITY;
                for (int64_t j = 0; j < jn; j++)
                    smx = Si[j] > smx ? Si[j] : smx;
                for (int64_t j = 0; j < jn; j++)
                    fr[j] = fs > 0 ? (Si[j] - smx) * fs : Si[j] * fs;
                for (int64_t j = jn; j < Skv; j++)
                    fr[j] = -INFINITY;
            }
            gx_softmax_row_f32(fr, Skv, (float *)P + i * Skv);
        }
        return;
    }
    for (int64_t i = 0; i < Sq; i++) {
        double mx = -INFINITY, sum = 0;
        const int64_t jn = p->causal ? (i + (Skv - Sq) + 1 < Skv ? i + (Skv - Sq) + 1 : Skv) : Skv; /* j < jn kept */
        for (int64_t j = 0; j < Skv; j++) {
            double a = S[i * Skv + j] * scale;
            if (mk)
                a += rt_ld(mk->data_type, mp, bcast_off(mk, &sc, (bh * Sq + i) * Skv + j));
            if (j >= jn)
                a = -INFINITY;
            row[j] = a, mx = a > mx ? a : mx;
        }
        for (int64_t j = 0; j < Skv; j++)
            sum += row[j] = mx == -INFINITY ? 0 : exp(row[j] - mx);
        if (t == aiednn_f32) {
            float *pr = (float *)P + i * Skv;
            for (int64_t j = 0; j < Skv; j++)
                pr[j] = (float)(sum > 0 ? row[j] / sum : 0.0);
        } else
            for (int64_t j = 0; j < Skv; j++)
                rt_st(t, P, i * Skv + j, sum > 0 ? row[j] / sum : 0.0);
    }
}

/* the (b, h) heads [bh0, bh1): scores, softmax, P V -- each GEMM pass deferred (b16q_defer) so the bf16 mode
 * runs a group's heads in a few n-tile launches instead of several launches per head */
static int gx_sdpa_heads(const rt_prob *p, void *const *ptr, const gx_types *t, int64_t bh0, int64_t bh1, float *Sall,
                         float *Oall, char *Qb, char *Kb, char *Vb, char *Pb, double *row) {
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    const int64_t B = d->dims[0], H = d->dims[1], Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3], Dv = v->dims[3];
    const int qs = (p->flags & AIEDNN_SDPA_QKV_BSHD) != 0;
    const int64_t Hk = k->dims[1], Hg = H / Hk; /* GQA / MQA: K, V heads */
    int rc = AIEK_OK;
    b16q_defer(1);
    for (int64_t bh = bh0; rc == AIEK_OK && bh < bh1; bh++) { /* S = Q K^T, K [Skv][D] as B^T */
        const int64_t b = bh / H, h = bh - b * H;
        uint64_t t0 = rt_ticks();
        const void *Qp = gx_head_rows(q->data_type, ptr[0], qs, H, b, h, Sq, D, t->a, Qb);
        const void *Kp = gx_head_rows(k->data_type, ptr[1], qs, Hk, b, h / Hg, Skv, D, t->b, Kb);
        const uint64_t t1 = rt_ticks();
        g_phase[RT_PH_SDPA_ROWS] += t1 - t0;
        rc = gx_gemm(t, Sq, Skv, D, Qp, Kp, Sall + (bh - bh0) * Sq * Skv, 1);
        g_phase[RT_PH_SDPA_GEMM] += rt_ticks() - t1;
    }
    const int rf = b16q_defer(0);
    rc = rc != AIEK_OK ? rc : rf;
    b16q_defer(1);
    for (int64_t bh = bh0; rc == AIEK_OK && bh < bh1; bh++) { /* O = softmax(S) V */
        const int64_t b = bh / H, h = bh - b * H;
        const uint64_t t0 = rt_ticks();
        gx_sdpa_softmax(p, ptr, B, H, b, h, Sq, Skv, Sall + (bh - bh0) * Sq * Skv, t->a, Pb, row);
        const uint64_t t1 = rt_ticks();
        const void *Vp = gx_head_rows(v->data_type, ptr[2], qs, Hk, b, h / Hg, Skv, Dv, t->b, Vb);
        const uint64_t t2 = rt_ticks();
        g_phase[RT_PH_SDPA_SM] += t1 - t0, g_phase[RT_PH_SDPA_ROWS] += t2 - t1;
        rc = gx_gemm(t, Sq, Dv, Skv, Pb, Vp, Oall + (bh - bh0) * Sq * Dv);
        g_phase[RT_PH_SDPA_GEMM] += rt_ticks() - t2;
    }
    const int rv = b16q_defer(0);
    return rc != AIEK_OK ? rc : rv;
}

static int run_gx_sdpa(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob; /* Q, K, V, DST, [MASK] */
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    const int64_t B = d->dims[0], H = d->dims[1], Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3], Dv = v->dims[3];
    const int ds_ = (p->flags & AIEDNN_SDPA_DST_BSHD) != 0;
    gx_types t;
    gx_sdpa_types(p, &t);
    const int fq = p->has_attr && dt_is_int(d->data_type); /* the f32 dst, then reorder's quantize */
    /* heads per group: their scores (f32 [Sq][Skv] each) within 8 MB */
    /* without the bf16 n-tile launch (AIEDNN_B16N) one head at a time: its scores stay in cache */
    const int64_t per = Sq * Skv * 4, G0 = !rt_env_on("AIEDNN_B16N") ? 1 : per > 0 ? ((int64_t)8 << 20) / per : 1;
    const int64_t G = G0 < 1 ? 1 : G0 > B * H ? B * H : G0;
    char *Qb = (char *)rt_alloc((size_t)(Sq * D * 4)), *Kb = (char *)rt_alloc((size_t)(Skv * D * 4));
    char *Vb = (char *)rt_alloc((size_t)(Skv * Dv * 4)), *Pb = (char *)rt_alloc((size_t)(Sq * Skv * 4));
    float *Sall = (float *)rt_alloc((size_t)(G * Sq * Skv) * sizeof(float));
    float *Oall = (float *)rt_alloc((size_t)(G * Sq * Dv) * sizeof(float));
    double *row = (double *)rt_alloc((size_t)Skv * sizeof(double));
    int rc = Qb && Kb && Vb && Pb && Sall && Oall && row ? AIEK_OK : AIEK_ERR_ALLOC;
    for (int64_t bh0 = 0; rc == AIEK_OK && bh0 < B * H; bh0 += G) {
        const int64_t bh1 = bh0 + G < B * H ? bh0 + G : B * H;
        rc = gx_sdpa_heads(p, ptr, &t, bh0, bh1, Sall, Oall, Qb, Kb, Vb, Pb, row);
        const uint64_t to = rt_ticks();
        for (int64_t bh = bh0; rc == AIEK_OK && bh < bh1; bh++) {
            const int64_t b = bh / H, h = bh - b * H;
            const float *O = Oall + (bh - bh0) * Sq * Dv;
            for (int64_t i = 0; i < Sq; i++) {
                const int64_t at = ds_ ? ((b * Sq + i) * H + h) * Dv : ((b * H + h) * Sq + i) * Dv;
                if (d->data_type == aiednn_f32 && !fq)
                    memcpy((float *)ptr[3] + at, O + i * Dv, (size_t)Dv * sizeof(float));
                else
                    for (int64_t e = 0; e < Dv; e++) {
                        const double o = O[i * Dv + e];
                        rt_st(d->data_type, ptr[3], at + e,
                              fq ? nearbyint((double)(float)o / p->dst_scale) + p->dst_zp : o);
                    }
            }
        }
        g_phase[RT_PH_SDPA_OUT] += rt_ticks() - to;
    }
    rt_free(Qb), rt_free(Kb), rt_free(Vb), rt_free(Pb), rt_free(Sall), rt_free(Oall), rt_free(row);
    return rc;
}

// ── gemm+mm: matmul / inner product with a src zero point or a u8 src ─────────────

/* 2-D matmul A[M,K] x B[K,N] (or B^T for an inner product), s8 / u8 src with a zero point */
static int gx_mm_make(const rt_prob *p, int64_t *M, int64_t *N, int64_t *K, int *bt, gx_zp *z) {
    const aiednn_memory_desc_t *s = &p->md[0], *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    if (p->prop != RT_FWD_INF || p->wei_zp || w->data_type != aiednn_s8 ||
        (s->data_type != aiednn_u8 && s->data_type != aiednn_s8) || rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES) >= 0)
        return 0;
    gx_zp_make(p, s->data_type, z);
    if (!z->on)
        return 0; /* symmetric s8: the blocked / ip implementations */
    if (p->kind == aiednn_inner_product) {
        *M = s->dims[0], *K = md_numel(s) / s->dims[0], *N = w->dims[0], *bt = 1;
        return 1;
    }
    if (s->ndims != 2 || w->ndims != 2)
        return 0;
    *M = s->dims[0], *K = s->dims[1], *bt = p->transposed; /* weights [K, N], or [N, K] (ONNX Gemm transB) */
    *N = p->transposed ? w->dims[0] : w->dims[1];
    return 1;
}

static int acc_gx_mm(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    int64_t M, N, K;
    int bt;
    gx_zp z;
    if (!gx_mm_make(p, &M, &N, &K, &bt, &z)) {
        snprintf(why, cap, "2-D u8 / s8-with-zero-point src x s8 weights, no weight zero point");
        return 0;
    }
    const gx_types t = {aiednn_s8, aiednn_s8, aiednn_s32};
    return gx_inner_ok(&t, M, N, K, bind, why, cap, bt);
}

static int run_gx_mm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    int64_t M, N, K;
    int bt;
    gx_zp z;
    if (!gx_mm_make(p, &M, &N, &K, &bt, &z))
        return AIEK_ERR_ARGS;
    const gx_types t = {aiednn_s8, aiednn_s8, aiednn_s32};
    int8_t *A = (int8_t *)rt_alloc((size_t)(M * K));
    int32_t *Cm = (int32_t *)rt_alloc((size_t)(M * N) * sizeof(int32_t));
    int64_t *cr = (int64_t *)rt_calloc((size_t)N, sizeof(int64_t));
    int rc = A && Cm && cr ? AIEK_OK : AIEK_ERR_ALLOC;
    const void *W = rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS);
    if (rc == AIEK_OK) {
        const aiednn_data_type_t sdt = p->md[0].data_type;
        for (int64_t i = 0; i < M * K; i++) /* x - shift: s8 */
            A[i] = (int8_t)(rt_ld(sdt, ptr[0], i) - z.shift);
        for (int64_t k = 0; k < K; k++) /* (shift - zp) * the column sums of W */
            for (int64_t n = 0; n < N; n++)
                cr[n] += ((const int8_t *)W)[bt ? n * K + k : k * N + n];
        rc = gx_gemm(&t, M, N, K, A, W, Cm, bt, 1); /* W is the primitive's weights: packed once when constant */
    }
    if (rc == AIEK_OK) {
        rt_epi E;
        rt_epi_init(&E, p, ptr);
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++)
                rt_epi_put(&E, m * N + n, n, (double)Cm[m * N + n] + (double)(z.corr_mul * cr[n]));
    }
    rt_free(A), rt_free(Cm), rt_free(cr);
    return rc;
}

// ── zp+: a src zero point / u8 src as the symmetric s8 problem plus an accumulator correction ─────

/* the inner problem q of p (s8 src, no zero point, a conv's padding baked into its src), or 0 */
static int gx_zpq(const rt_prob *p, rt_prob *q, gx_zp *z, int64_t *pad /* t, b, l, r */) {
    const aiednn_memory_desc_t *s = &p->md[0], *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    if (p->prop != RT_FWD_INF || p->wei_zp || !w || w->data_type != aiednn_s8 ||
        (s->data_type != aiednn_u8 && s->data_type != aiednn_s8) || rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES) >= 0 ||
        p->acc_add)
        return 0;
    gx_zp_make(p, s->data_type, z);
    if (!z->on)
        return 0;
    *q = *p;
    q->src_zp = 0;
    q->md[0].data_type = aiednn_s8;
    pad[0] = pad[1] = pad[2] = pad[3] = 0;
    if (p->kind == aiednn_convolution) {
        if (s->ndims != 4)
            return 0;
        pad[0] = p->conv.padding_l[0], pad[1] = p->conv.padding_r[0], pad[2] = p->conv.padding_l[1],
        pad[3] = p->conv.padding_r[1];
        q->md[0].dims[2] += pad[0] + pad[1], q->md[0].dims[3] += pad[2] + pad[3];
        q->conv.padding_l[0] = q->conv.padding_l[1] = q->conv.padding_r[0] = q->conv.padding_r[1] = 0;
        return 1;
    }
    if (p->kind != aiednn_matmul)
        return p->kind == aiednn_inner_product;
    /* matmul: 2-D, or batched activations [.., M, K] with weights shared over the batch (an nn.Linear on
     * [B, S, D]: weights [1, .., K, N]) -- one correction per output channel either way */
    if (s->ndims < 2 || w->ndims != s->ndims)
        return 0;
    for (int i = 0; i < w->ndims - 2; i++)
        if (w->dims[i] != 1)
            return 0;
    return 1;
}

static int acc_gx_zp(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    rt_prob q;
    gx_zp z;
    int64_t pad[4];
    if (!gx_zpq(p, &q, &z, pad)) {
        snprintf(why, cap,
                 "u8 / zero-point s8 src x s8 weights: 2-D conv, inner product, matmul (batched: shared weights)");
        return 0;
    }
    const rt_impl *in = nullptr;
    rt_bind b = {nullptr, nullptr, nullptr};
    if (dispatch(&q, &in, &b, 1) != aiednn_success || in->specificity < 0) {
        snprintf(why, cap, "its symmetric s8 problem would run on %s", in ? in->name : "nothing");
        return 0;
    }
    bind->inner = in, bind->blk = b.blk, bind->inner2 = b.inner; /* impl_info: zp+im2col+blocked:aie:... */
    return 1;
}

static int run_gx_zp(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    rt_prob q;
    gx_zp z;
    int64_t pad[4];
    if (!gx_zpq(p, &q, &z, pad))
        return AIEK_ERR_ARGS;
    const aiednn_memory_desc_t *s = &p->md[0], *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    const int ip = p->kind == aiednn_inner_product, conv = p->kind == aiednn_convolution;
    /* the correction per output channel: (shift - zp) * the sum of that channel's weights; rows of W are
     * output channels for conv / inner product / a transposed matmul ([N, K], ONNX Gemm transB), columns
     * for a plain matmul ([K, N]) */
    const int rows = conv || ip || p->transposed;
    const int wl = p->kind == aiednn_matmul ? w->ndims - 2 : 0; /* a matmul's weight matrix: its last two dims */
    const int64_t O = rows ? w->dims[wl] : w->dims[wl + 1], Kw = md_numel(w) / O;
    int64_t *add = (int64_t *)rt_calloc((size_t)O, sizeof(int64_t));
    const int64_t nq = md_numel(&q.md[0]);
    const int copy = z.shift || pad[0] || pad[1] || pad[2] || pad[3];
    int8_t *xs = copy ? (int8_t *)rt_alloc((size_t)nq) : nullptr;
    int rc = add && (!copy || xs) ? AIEK_OK : AIEK_ERR_ALLOC;
    const int8_t *W = (const int8_t *)rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS);
    const uint64_t tw0 = rt_ticks();
    /* constant weights: their sums once per primitive (prim->wsum), not O x K adds per execute */
    const int cached = prim->const_w && prim->wsum && prim->wsum_key == (const void *)W && prim->wsum_n == O;
    int64_t *ws = cached ? prim->wsum : NULL;
    if (!cached && prim->const_w && rc == AIEK_OK) {
        free(prim->wsum);
        prim->wsum = (int64_t *)malloc((size_t)O * sizeof(int64_t)); /* the primitive's: freed by destroy */
        prim->wsum_key = W, prim->wsum_n = prim->wsum ? O : 0;
    }
    for (int64_t o = 0; rc == AIEK_OK && o < O; o++) {
        int64_t sw = 0;
        if (cached)
            sw = ws[o];
        else if (rows)
            for (int64_t k = 0; k < Kw; k++)
                sw += W[o * Kw + k];
        else
            for (int64_t k = 0; k < Kw; k++)
                sw += W[k * O + o];
        if (!cached && prim->const_w && prim->wsum)
            prim->wsum[o] = sw;
        add[o] = z.corr_mul * sw;
    }
    g_phase[RT_PH_ZPWSUM] += rt_ticks() - tw0;
    const uint64_t tp0 = rt_ticks();
    if (rc == AIEK_OK && copy) { /* x - shift, the padding filled with zp - shift */
        const uint8_t *xu = (const uint8_t *)ptr[0];
        if (!conv)
            for (int64_t i = 0; i < nq; i++)
                xs[i] = (int8_t)((int)xu[i] - z.shift);
        else if (md_is_nhwc(s)) { /* nhwc: a pixel row is W * C contiguous bytes */
            const int64_t N = s->dims[0], Cc = s->dims[1], H = s->dims[2], W_ = s->dims[3];
            const int64_t Hp = q.md[0].dims[2], Wp = q.md[0].dims[3], row = W_ * Cc;
            memset(xs, z.pad, (size_t)nq);
            for (int64_t n = 0; n < N; n++)
                for (int64_t h = 0; h < H; h++) {
                    const uint8_t *src = xu + (n * H + h) * row;
                    int8_t *dst = xs + ((n * Hp + h + pad[0]) * Wp + pad[2]) * Cc;
                    if (z.shift)
                        for (int64_t x = 0; x < row; x++)
                            dst[x] = (int8_t)((int)src[x] - 128);
                    else
                        memcpy(dst, src, (size_t)row);
                }
        } else {
            const int64_t Np = s->dims[0] * s->dims[1], H = s->dims[2], W_ = s->dims[3];
            const int64_t Hp = q.md[0].dims[2], Wp = q.md[0].dims[3];
            memset(xs, z.pad, (size_t)nq);
            for (int64_t c = 0; c < Np; c++)
                for (int64_t h = 0; h < H; h++) {
                    const uint8_t *src = xu + (c * H + h) * W_;
                    int8_t *dst = xs + (c * Hp + h + pad[0]) * Wp + pad[2];
                    if (z.shift)
                        for (int64_t x = 0; x < W_; x++)
                            dst[x] = (int8_t)((int)src[x] - 128);
                    else
                        memcpy(dst, src, (size_t)W_);
                }
        }
    }
    g_phase[RT_PH_ZPPAD] += rt_ticks() - tp0;
    if (rc == AIEK_OK) {
        q.acc_add = add;
        void *qp[RT_MAX_MD];
        for (int i = 0; i < p->nmd; i++)
            qp[i] = ptr[i];
        if (copy)
            qp[0] = xs;
        const rt_impl *in = nullptr; /* the inner problem's own binding (its own composite inner) */
        rt_bind b = {nullptr, nullptr};
        rc = dispatch(&q, &in, &b, 1) != aiednn_success ? AIEK_ERR_ARGS : run_sub(in, &q, &b, qp);
    }
    rt_free(add), rt_free(xs);
    return rc;
}

// ── gemm+rnnbwd: back-propagation through time with its GEMMs on the matmul dispatch ─────────────

/* the BPTT GEMM hook (aiednn_rt_rnn_bwd.h's rnnb_ctx.mm): f32 x f32 -> f32 through dispatch */
static int gx_rnnb_mm(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt) {
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32};
    return gx_gemm(&t, M, N, K, A, B, C, bt);
}

static int gx_rnnb_ctx(const rt_prob *p, void *const *ptr, rnnb_ctx *c) {
    if (p->prop != RT_BWD || !rt_md(p, AIEDNN_ARG_DIFF_DST)) /* (a forward problem has no DIFF_DST) */
        return 0;
    const aiednn_memory_desc_t *sl = rt_md(p, AIEDNN_ARG_SRC), *wl = rt_md(p, AIEDNN_ARG_WEIGHTS);
    c->p = p, c->ptr = ptr, c->cell = p->alg, c->G = rnn_gates(p->alg), c->Gb = c->G + rnn_is_lbr(p->alg);
    c->dir = (int)p->flags, c->T = sl->dims[0], c->N = sl->dims[1], c->SLC = sl->dims[2];
    c->L = wl->dims[0], c->D = wl->dims[1], c->H = wl->dims[4], c->DLC = rt_md(p, AIEDNN_ARG_DIFF_DST)->dims[2];
    c->mm = gx_rnnb_mm;
    return p->prop == RT_BWD && dt_is_float(sl->data_type) && dt_is_float(wl->data_type);
}

static int acc_gx_rnnbwd(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    rnnb_ctx c;
    if (p->kind != aiednn_rnn || !gx_rnnb_ctx(p, nullptr, &c)) {
        snprintf(why, cap, "floating RNN backward only");
        return 0;
    }
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32}; /* the biggest of the four: dA W^T */
    return gx_inner_ok(&t, c.T * c.N, c.SLC, c.G * c.H, bind, why, cap, 1);
}

static int run_gx_rnnbwd(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    rnnb_ctx c;
    if (!gx_rnnb_ctx(&prim->prob, ptr, &c))
        return AIEK_ERR_ARGS;
    return run_ref_rnn_bwd(&c);
}

// ── gemm+rnn ────────────────────────────────────────────────────────────────

typedef struct {
    const rt_prob *p;
    void *const *ptr;
    gx_types t;
    int cell, G, Gb, dir, gru;
    int64_t T, N, H, L, D, SLC, DLC;
} gx_rnn;

static int gx_rnn_make(const rt_prob *p, void *const *ptr, gx_rnn *r) {
    const aiednn_memory_desc_t *sl = rt_md(p, AIEDNN_ARG_SRC), *wl = rt_md(p, AIEDNN_ARG_WEIGHTS);
    r->p = p, r->ptr = ptr, r->cell = p->alg, r->G = rnn_gates(p->alg), r->Gb = r->G + rnn_is_lbr(p->alg);
    r->gru = rnn_is_gru(p->alg), r->dir = (int)p->flags;
    r->T = sl->dims[0], r->N = sl->dims[1], r->SLC = sl->dims[2], r->L = wl->dims[0], r->D = wl->dims[1];
    r->H = wl->dims[4], r->DLC = rt_md(p, AIEDNN_ARG_DST)->dims[2];
    const int bf = sl->data_type == aiednn_bf16 && wl->data_type == aiednn_bf16;
    r->t.a = r->t.b = bf ? aiednn_bf16 : aiednn_f32, r->t.c = aiednn_f32;
    return p->prop == RT_FWD_INF && dt_is_float(sl->data_type) && dt_is_float(wl->data_type);
}

static int acc_gx_rnn(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    gx_rnn r;
    if (!gx_rnn_make(p, nullptr, &r)) {
        snprintf(why, cap, "floating inference only");
        return 0;
    }
    rt_bind b2 = {nullptr, nullptr};
    return gx_inner_ok(&r.t, r.T * r.N, r.G * r.H, r.SLC, bind, why, cap) &&
           gx_inner_ok(&r.t, r.N, r.G * r.H, r.H, &b2, why, cap);
}

/* the gates of step t from the GEMM results: P = W x (all steps), R = U h, Ro = (r * h) U_o (the
 * reset-before GRU, filled by the caller between the two passes); writes the new state into hn */
static void gx_rnn_gates(const gx_rnn *r, int64_t ld, int64_t t, const float *P, const float *R, const double *h,
                         double *cs, double *hn, double *rg, int pass) {
    const rt_prob *p = r->p;
    const int64_t N = r->N, H = r->H, GH = r->G * H;
    const int sb = rt_slot(p, AIEDNN_ARG_BIAS);
    auto B = [&](int64_t g, int64_t j) {
        return sb >= 0 ? rt_ld(p->md[sb].data_type, r->ptr[sb], (ld * r->Gb + g) * H + j) : 0.0;
    };
    for (int64_t n = 0; n < N; n++)
        for (int64_t j = 0; j < H; j++) {
            const float *Pn = P + (t * N + n) * GH, *Rn = R + n * GH;
            auto a = [&](int64_t g) {
                return (double)Pn[g * H + j] + B(g, j) + (r->gru && g == 2 ? 0.0 : (double)Rn[g * H + j]);
            };
            const int64_t i = n * H + j;
            if (r->cell == aiednn_rnn_lstm) {
                cs[i] = rnn_sig(a(1)) * cs[i] + rnn_sig(a(0)) * tanh(a(2));
                hn[i] = rnn_sig(a(3)) * tanh(cs[i]);
            } else if (r->gru && pass == 0) { /* r first: the reset-before GRU's U_o GEMM needs r * h */
                rg[i] = rnn_sig(a(1));
            } else if (r->gru) {
                const double u = rnn_sig(a(0)) * rnn_att(p, r->ptr, t, n, N); /* AUGRU: (1 - a) u */
                const double o =
                    a(2) + (rnn_is_lbr(r->cell) ? rg[i] * ((double)Rn[2 * H + j] + B(3, j)) : (double)R[N * GH + i]);
                hn[i] = u * h[i] + (1 - u) * tanh(o);
            } else {
                const double x = a(0);
                hn[i] = r->cell == aiednn_rnn_vanilla_tanh       ? tanh(x)
                        : r->cell == aiednn_rnn_vanilla_logistic ? rnn_sig(x)
                        : x > 0                                  ? x
                                                                 : (double)p->alpha * x;
            }
        }
}

/* one (layer, direction): in [T][N][I] (inner operand type), out [T][N][DLC] (double) at column col,
 * added to (bidirectional sum) when acc; final h / c into h / cs */
static int gx_rnn_layer(const gx_rnn *r, int64_t l, int64_t d, int64_t I, const void *in, double *out, int64_t col,
                        int acc, char *Wb, char *Ub, char *Uo, float *P, float *R, char *hb, double *h, double *cs) {
    const rt_prob *p = r->p;
    void *const *ptr = r->ptr;
    const int64_t T = r->T, N = r->N, H = r->H, G = r->G, GH = G * H, ld = l * r->D + d;
    const int rev = r->dir == aiednn_rnn_right2left || d == 1;
    const aiednn_memory_desc_t *wl = rt_md(p, AIEDNN_ARG_WEIGHTS), *wi = rt_md(p, AIEDNN_ARG_WEIGHTS_ITER);
    const int si = rt_slot(p, AIEDNN_ARG_SRC_ITER), sc = rt_slot(p, AIEDNN_ARG_SRC_ITER_C);
    const void *W = gx_view(wl->data_type, rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS), ld * I * GH, I * GH, r->t.b, Wb);
    const void *U = gx_view(wi->data_type, rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS_ITER), ld * H * GH, H * GH, r->t.b, Ub);
    if (r->gru && !rnn_is_lbr(r->cell)) /* U_o = U[:, 2, :] for (r * h) U_o */
        for (int64_t k = 0; k < H; k++)
            for (int64_t j = 0; j < H; j++)
                rt_st(r->t.b, Uo, k * H + j, rt_ld(r->t.b, U, (k * G + 2) * H + j));
    double *hn = h + N * H, *rg = hn + N * H;
    for (int64_t i = 0; i < N * H; i++) {
        h[i] = si >= 0 ? rt_ld(p->md[si].data_type, ptr[si], ld * N * H + i) : 0.0;
        cs[i] = sc >= 0 ? rt_ld(p->md[sc].data_type, ptr[sc], ld * N * H + i) : 0.0;
    }
    /* W / U are the primitive's own weights unless converted (bf16 -> f32 copies): cacheable then */
    const int w_own =
        W == (const void *)((const char *)rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS) + ld * I * GH * dt_bytes(wl->data_type));
    const int u_own = U == (const void *)((const char *)rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS_ITER) +
                                          ld * H * GH * dt_bytes(wi->data_type));
    int rc = gx_gemm(&r->t, T * N, GH, I, in, W, P, 0, w_own); /* every step's W x */
    for (int64_t s = 0; s < T && rc == AIEK_OK; s++) {
        const int64_t t = rev ? T - 1 - s : s;
        for (int64_t i = 0; i < N * H; i++)
            rt_st(r->t.a, hb, i, h[i]);
        rc = gx_gemm(&r->t, N, GH, H, hb, U, R, 0, u_own); /* U h for every gate */
        if (rc == AIEK_OK && r->gru) {
            gx_rnn_gates(r, ld, t, P, R, h, cs, hn, rg, 0);
            if (!rnn_is_lbr(r->cell)) { /* (r * h) U_o */
                for (int64_t i = 0; i < N * H; i++)
                    rt_st(r->t.a, hb, i, rg[i] * h[i]);
                rc = gx_gemm(&r->t, N, H, H, hb, Uo, R + N * GH);
            }
        }
        if (rc != AIEK_OK)
            break;
        gx_rnn_gates(r, ld, t, P, R, h, cs, hn, rg, 1);
        for (int64_t i = 0; i < N * H; i++) {
            h[i] = hn[i];
            double *o = &out[(t * N + i / H) * r->DLC + col + i % H];
            *o = acc ? *o + h[i] : h[i];
        }
    }
    return rc;
}

static int run_gx_rnn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    gx_rnn r;
    gx_rnn_make(p, ptr, &r);
    const int64_t T = r.T, N = r.N, H = r.H, GH = r.G * H, W_ = r.SLC > r.DLC ? r.SLC : r.DLC;
    const int concat = r.dir == aiednn_rnn_bidirectional_concat, bisum = r.dir == aiednn_rnn_bidirectional_sum;
    double *seq = (double *)rt_calloc((size_t)(2 * T * N * W_ + 4 * N * H), sizeof(double));
    char *inb = (char *)rt_alloc((size_t)(T * N * W_ * 4)), *Wb = (char *)rt_alloc((size_t)(W_ * GH * 4));
    char *Ub = (char *)rt_alloc((size_t)(H * GH * 4)), *Uo = (char *)rt_alloc((size_t)(H * H * 4)),
         *hb = (char *)rt_alloc((size_t)(N * H * 4));
    float *P = (float *)rt_alloc((size_t)(T * N * GH) * sizeof(float)),
          *R = (float *)rt_alloc((size_t)(N * (GH + H)) * sizeof(float));
    int rc = seq && inb && Wb && Ub && Uo && hb && P && R ? AIEK_OK : AIEK_ERR_ALLOC;
    double *in = seq, *out = seq + T * N * W_, *h = out + T * N * W_, *cs = h + 3 * N * H;
    const int sdi = rt_slot(p, AIEDNN_ARG_DST_ITER), sdc = rt_slot(p, AIEDNN_ARG_DST_ITER_C);
    for (int64_t i = 0; rc == AIEK_OK && i < T * N * r.SLC; i++)
        in[i] = rt_ld(p->md[0].data_type, ptr[0], i);
    for (int64_t l = 0; rc == AIEK_OK && l < r.L; l++) {
        const int64_t I = l == 0 ? r.SLC : r.DLC;
        for (int64_t i = 0; i < T * N * I; i++)
            rt_st(r.t.a, inb, i, in[i]);
        for (int64_t d = 0; rc == AIEK_OK && d < r.D; d++) {
            rc = gx_rnn_layer(&r, l, d, I, inb, out, concat ? d * H : 0, bisum && d == 1, Wb, Ub, Uo, P, R, hb, h, cs);
            for (int64_t i = 0; rc == AIEK_OK && i < N * H; i++) {
                if (sdi >= 0)
                    rt_st(p->md[sdi].data_type, ptr[sdi], (l * r.D + d) * N * H + i, h[i]);
                if (sdc >= 0)
                    rt_st(p->md[sdc].data_type, ptr[sdc], (l * r.D + d) * N * H + i, cs[i]);
            }
        }
        double *tmp = in; /* this layer's output is the next one's input */
        in = out, out = tmp;
    }
    const int sd = rt_slot(p, AIEDNN_ARG_DST);
    for (int64_t i = 0; rc == AIEK_OK && i < T * N * r.DLC; i++)
        rt_st(p->md[sd].data_type, ptr[sd], i, in[i]);
    rt_free(seq), rt_free(inb), rt_free(Wb), rt_free(Ub), rt_free(Uo), rt_free(hb), rt_free(P), rt_free(R);
    return rc;
}

// ── dequant+ ────────────────────────────────────────────────────────────────

/* the problem with f32 weights (and an f32 src when cvt_src), no weight group scales / zero points;
 * slot map: m slot i is p slot map[i] */
static void gx_dequant_prob(const rt_prob *p, rt_prob *m, int *map, int cvt_src) {
    *m = *p;
    m->nmd = 0;
    for (int s = 0; s < p->nmd; s++) {
        if (p->arg[s] == AIEDNN_ARG_ATTR_WEI_SCALES || p->arg[s] == AIEDNN_ARG_ATTR_WEI_ZERO_POINTS)
            continue;
        m->arg[m->nmd] = p->arg[s], m->md[m->nmd] = p->md[s], map[m->nmd++] = s;
        if (p->arg[s] == AIEDNN_ARG_WEIGHTS || (cvt_src && p->arg[s] == AIEDNN_ARG_SRC))
            m->md[m->nmd - 1].data_type = aiednn_f32;
    }
    if (p->kind == aiednn_inner_product) { /* as its [N, IC] x [OC, IC]^T matmul: the inner is a GEMM impl */
        rt_prob mm;
        ip_as_matmul(m, &mm);
        *m = mm;
    }
}

/* cvt+'s problems: a floating src whose (src, weights) pair is neither f32 x f32 nor bf16 x bf16 */
static int gx_needs_cvt(const rt_prob *p) {
    const aiednn_data_type_t s = p->md[0].data_type, w = rt_md(p, AIEDNN_ARG_WEIGHTS)->data_type;
    return dt_is_float(s) && rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES) < 0 && w != aiednn_s32 &&
           !(s == aiednn_f32 && w == aiednn_f32) && !(s == aiednn_bf16 && w == aiednn_bf16);
}

static int gx_dequant_accept(const rt_prob *p, int cvt_src, rt_bind *bind, char *why, int cap) {
    rt_prob m;
    int map[RT_MAX_MD];
    gx_dequant_prob(p, &m, map, cvt_src);
    const rt_impl *in = nullptr;
    rt_bind b = {nullptr, nullptr};
    if (dispatch(&m, &in, &b, 1) != aiednn_success ||
        (strncmp(in->name, "ref:", 4) == 0 && !rt_env_on("AIEDNN_GEMMX_ANY_INNER"))) {
        snprintf(why, cap, "the dequantized f32 problem would run on %s", in ? in->name : "nothing");
        return 0;
    }
    bind->inner = in, bind->blk = b.blk;
    return 1;
}

static int acc_gx_dequant(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES) < 0 || !dt_is_float(p->md[0].data_type)) {
        snprintf(why, cap, "weight group scales with a floating src only");
        return 0;
    }
    return gx_dequant_accept(p, p->md[0].data_type != aiednn_f32, bind, why, cap);
}

static int acc_gx_cvt(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (!gx_needs_cvt(p)) {
        snprintf(why, cap, "f32 x f32 / bf16 x bf16 / group scales / integer src: served directly");
        return 0;
    }
    return gx_dequant_accept(p, p->md[0].data_type != aiednn_f32, bind, why, cap);
}

static int run_gx_dequant(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    const int ss = rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES), zs = rt_slot(p, AIEDNN_ARG_ATTR_WEI_ZERO_POINTS);
    const int ip = p->kind == aiednn_inner_product;
    const int64_t nw = md_numel(w), N = ip ? w->dims[0] : w->dims[w->ndims - 1];
    const int64_t K = ip ? nw / N : w->dims[w->ndims - 2], gk = ss >= 0 ? K / p->md[ss].dims[0] : K;
    const int cvt_src = p->md[0].data_type != aiednn_f32;
    const int64_t ns = cvt_src ? md_numel(&p->md[0]) : 0;
    float *wf = (float *)rt_alloc((size_t)(nw + ns) * sizeof(float)), *sf = wf + nw;
    if (!wf)
        return AIEK_ERR_ALLOC;
    const void *pw = rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS);
    for (int64_t i = 0; i < nw; i++) { /* matmul [.., K, N], inner product [N, K]: the (k, n) of element i */
        const int64_t k = ip ? i % K : i / N % K, n = ip ? i / K : i % N, gi = (k / gk) * N + n;
        const double z = zs >= 0 ? rt_ld(p->md[zs].data_type, ptr[zs], gi) : 0.0;
        wf[i] = (float)((rt_ld(w->data_type, pw, i) - z) * (ss >= 0 ? rt_ld(p->md[ss].data_type, ptr[ss], gi) : 1.0));
    }
    for (int64_t i = 0; i < ns; i++)
        sf[i] = (float)rt_ld(p->md[0].data_type, ptr[0], i);
    rt_prob m;
    int map[RT_MAX_MD];
    gx_dequant_prob(p, &m, map, cvt_src);
    void *mp[RT_MAX_MD];
    for (int i = 0; i < m.nmd; i++)
        mp[i] = m.arg[i] == AIEDNN_ARG_WEIGHTS          ? (void *)wf
                : cvt_src && m.arg[i] == AIEDNN_ARG_SRC ? (void *)sf
                                                        : ptr[map[i]];
    const rt_impl *in = nullptr; /* the inner problem's own binding (a composite inner has its own inner) */
    rt_bind b = {nullptr, nullptr};
    const int cw = g_exec_const_w; /* wf is a temporary: no constant-weight caching of it */
    g_exec_const_w = 0;
    const int rc = dispatch(&m, &in, &b, 1) != aiednn_success ? AIEK_ERR_ARGS : run_sub(in, &m, &b, mp);
    g_exec_const_w = cw;
    rt_free(wf);
    return rc;
}

#endif /* AIEDNN_RT_GEMMX_H */
