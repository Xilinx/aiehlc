/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_eager_fvec.h -- eager mode's f32 pointwise primitives on the feltw / fbinary templates
 * (src/aietensorop/eager/feltw_eager_kernel.cc, fbinary_eager_kernel.cc): one instance per algorithm, the
 * algorithm compiled in (aie:eager_feltw__R_L_TR_F, aie:eager_fbinary__R_L_TR_F, F = the FVEC AIE_FVF code).
 *   eager:feltw    f32 eltwise (every algorithm fvec_eltwise_fn maps, pow with an integer exponent in [-2, 4])
 *   eager:fbinary  f32 binary (add / sub / mul / div / max / min / comparisons / logical / add_square), either
 *                  operand broadcast (materialized here)
 * The tensor runs as rows of the instance's L floats, R rows per launch. Contract (FVEC's): within
 * 1e-5 + 1e-5 |ref| of ref:'s double arithmetic.
 *
 * Needs, from the including TU: aiednn_rt_fvec.h (fvec_eltwise_fn), aie_modes_fvec.h (AIE_FVF_*).
 ******************************************************************************/
#ifndef AIEDNN_RT_EAGER_FVEC_H
#define AIEDNN_RT_EAGER_FVEC_H

typedef struct {
    const aiednn_impl_t *g;
    int64_t R, L;
    int F;
} rt_eager_fp;

/* an instance of template tmpl ("feltw" / "fbinary") */
static int rt_eager_fp_of(const aiednn_impl_t *g, const char *tmpl, rt_eager_fp *q) {
    char pre[40];
    snprintf(pre, sizeof pre, "aie:eager_%s__", tmpl);
    const size_t n = strlen(pre);
    long long r, l, tr;
    int f;
    /* feltw: x, {alpha, beta}, y; fbinary / faffine: one packed input ([x | y], [x | c]) and the output */
    const int nargs = strcmp(tmpl, "feltw") ? 2 : 3;
    if (g->nargs != nargs || strncmp(g->name, pre, n) || sscanf(g->name + n, "%lld_%lld_%lld_%d", &r, &l, &tr, &f) != 4)
        return 0;
    q->g = g, q->R = r, q->L = l, q->F = f;
    return 1;
}

static int rt_eager_fp_find(const char *tmpl, int F, rt_eager_fp *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_fp q;
        if (rt_eager_fp_of(&aiednn_impl_list[i], tmpl, &q) && q.F == F && (!found || q.R * q.L > out->R * out->L))
            *out = q, found = 1;
    }
    return found;
}

/* the binary algorithm's AIE_FVF code, or 0 */
static int eager_fbinary_fn(int alg) {
    switch (alg) {
    case aiednn_binary_add:
        return AIE_FVF_ADD;
    case aiednn_binary_sub:
        return AIE_FVF_SUB;
    case aiednn_binary_mul:
        return AIE_FVF_MUL;
    case aiednn_binary_div:
        return AIE_FVF_DIV;
    case aiednn_binary_max:
        return AIE_FVF_MAX;
    case aiednn_binary_min:
        return AIE_FVF_MIN;
    case aiednn_binary_eq:
        return AIE_FVF_EQ;
    case aiednn_binary_ne:
        return AIE_FVF_NE;
    case aiednn_binary_gt:
        return AIE_FVF_GT;
    case aiednn_binary_ge:
        return AIE_FVF_GE;
    case aiednn_binary_lt:
        return AIE_FVF_LT;
    case aiednn_binary_le:
        return AIE_FVF_LE;
    case aiednn_binary_logical_and:
        return AIE_FVF_LAND;
    case aiednn_binary_logical_or:
        return AIE_FVF_LOR;
    case aiednn_binary_logical_xor:
        return AIE_FVF_LXOR;
    case aiednn_binary_add_square:
        return AIE_FVF_ADD_SQUARE;
    }
    return 0;
}

/* f32 eltwise: its F (0: not one) and alpha / beta */
static int eager_feltw_fn(const rt_prob *p, float *a, float *b) {
    if (p->kind != aiednn_eltwise || p->prop != RT_FWD_INF || p->md[0].data_type != aiednn_f32 ||
        p->md[1].data_type != aiednn_f32 || p->has_attr || p->npo || rt_prob_has_fmt(p))
        return 0;
    const int F = fvec_eltwise_fn(p, a, b);
    if (F == AIE_FVF_POW) {
        const int e = (int)*b;
        if ((float)e != *b || e < -2 || e > 4)
            return 0;
    }
    return F;
}

static int acc_eager_feltw(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    float a, b;
    const int F = eager_feltw_fn(p, &a, &b);
    rt_eager_fp q;
    if (!F || !rt_eager_fp_find("feltw", F, &q)) {
        snprintf(why, cap, F ? "no feltw instance of this algorithm" : "f32 -> f32 inference eltwise, no attr");
        return 0;
    }
    return 1;
}

static int acc_eager_fbinary(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const int F = eager_fbinary_fn(p->alg);
    rt_eager_fp q;
    if (p->kind != aiednn_binary || p->prop != RT_FWD_INF || !F || p->md[0].data_type != aiednn_f32 ||
        p->md[1].data_type != aiednn_f32 || p->md[2].data_type != aiednn_f32 || p->has_attr || p->npo ||
        rt_prob_has_fmt(p) || !rt_eager_fp_find("fbinary", F, &q)) {
        snprintf(why, cap, "f32 binary without attr, of an fbinary instance's operation");
        return 0;
    }
    return 1;
}

static void *g_efp_x, *g_efp_z;
static size_t g_efp_x_cap, g_efp_z_cap;

/* one launch of an fbinary instance on xy (R rows of [x | y], 2 L floats each) into z (R x L floats) */
static int eager_fp_launch(const rt_impl *self, const rt_eager_fp *q, float *xy, float *z) {
    const int64_t shp[2] = {q->R, q->L}, xshp[2] = {q->R, 2 * q->L};
    aiek_tensor t[2] = {{xy, 2, AIEK_FLOAT, 32, 1, xshp, (uint64_t)(q->R * q->L * 8)},
                        {z, 2, AIEK_FLOAT, 32, 1, shp, (uint64_t)(q->R * q->L * 4)}};
    const uint64_t te = rt_ticks();
    const int rc = q->g->entry(t, 2);
    ((rt_impl *)self)->t_entry += rt_ticks() - te;
    ((rt_impl *)self)->launches++;
    return rc;
}

static int run_eager_feltw(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    float a, b;
    const int F = eager_feltw_fn(p, &a, &b);
    rt_eager_fp q;
    if (!F || !rt_eager_fp_find("feltw", F, &q))
        return AIEK_ERR_PARAM;
    const int64_t n = md_numel(&p->md[1]), per = q.R * q.L;
    float *x = (float *)scratch(&g_efp_x, &g_efp_x_cap, (size_t)per * 4);
    float *z = (float *)scratch(&g_efp_z, &g_efp_z_cap, (size_t)per * 4);
    if (!x || !z)
        return AIEK_ERR_ALLOC;
    float ab[16];
    for (int r = 0; r < 4; r++) /* {alpha, beta} in every mesh column's row */
        ab[4 * r] = a, ab[4 * r + 1] = b, ab[4 * r + 2] = ab[4 * r + 3] = 0.0f;
    for (int64_t e0 = 0; e0 < n; e0 += per) {
        const int64_t cnt = n - e0 < per ? n - e0 : per;
        uint64_t t0 = rt_ticks();
        memcpy(x, (const float *)ptr[0] + e0, (size_t)cnt * 4);
        memset(x + cnt, 0, (size_t)(per - cnt) * 4);
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        const int64_t pshp[2] = {4, 4}, shp[2] = {q.R, q.L};
        aiek_tensor t[3] = {{x, 2, AIEK_FLOAT, 32, 1, shp, (uint64_t)per * 4},
                            {ab, 2, AIEK_FLOAT, 32, 1, pshp, 64},
                            {z, 2, AIEK_FLOAT, 32, 1, shp, (uint64_t)per * 4}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        memcpy((float *)ptr[1] + e0, z, (size_t)cnt * 4);
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

/* z = F(a, b) on fbinary instance q, either operand broadcast to d (materialized) */
static int eager_fbinary_run(const rt_impl *self, const rt_eager_fp *q, const float *a, const aiednn_memory_desc_t *s0,
                             const float *b, const aiednn_memory_desc_t *s1, const aiednn_memory_desc_t *d,
                             float *out) {
    const int64_t n = md_numel(d), per = q->R * q->L, L = q->L;
    float *xy = (float *)scratch(&g_efp_x, &g_efp_x_cap, (size_t)per * 8);
    float *z = (float *)scratch(&g_efp_z, &g_efp_z_cap, (size_t)per * 4);
    if (!xy || !z)
        return AIEK_ERR_ALLOC;
    const int af = same_dims(s0, d), bf = same_dims(s1, d);
    for (int64_t e0 = 0; e0 < n; e0 += per) {
        const int64_t cnt = n - e0 < per ? n - e0 : per;
        uint64_t t0 = rt_ticks();
        for (int64_t i = 0; i < per; i++) { /* row i / L: [x | y]; broadcasts materialized, the last tail zero */
            float *row = xy + i / L * 2 * L;
            row[i % L] = i < cnt ? a[af ? e0 + i : bcast_off(s0, d, e0 + i)] : 0.0f;
            row[L + i % L] = i < cnt ? b[bf ? e0 + i : bcast_off(s1, d, e0 + i)] : 1.0f; /* 1: no division by zero */
        }
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        const int rc = eager_fp_launch(self, q, xy, z);
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        memcpy(out + e0, z, (size_t)cnt * 4);
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

static int run_eager_fbinary(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_eager_fp q;
    if (!rt_eager_fp_find("fbinary", eager_fbinary_fn(p->alg), &q))
        return AIEK_ERR_PARAM;
    return eager_fbinary_run(self, &q, (const float *)ptr[0], &p->md[0], (const float *)ptr[1], &p->md[1], &p->md[2],
                             (float *)ptr[2]);
}

/* ── eager:faffine -- f32 per-channel ops on the faffine template (aie:eager_faffine__R_L_TR_F) ───────────────
 * inference batch norm (x m + b per channel, m = scale / sqrt(var + eps), b = shift - mean m; F 97 with the fused
 * ReLU) and PReLU with a slope per contiguous block of at least 16 (F 98). Each channel block of sp elements is cut
 * into rows of the instance's L, every row carrying its block's {c0, c1}. PReLU with a per-element slope (blocks
 * under 16) runs on an fbinary instance of F 98. FVEC's contract. */

#define RT_EAGER_AFFINE 96 /* = eager_fmath_ops.cc's EG_F_ROW_* (= FVEC's AIE_FVF_ROW_*) */
#define RT_EAGER_AFFINE_RELU 97
#define RT_EAGER_PRELU 98
#define EG_F_ADD_ 64 /* eager_fmath_ops.cc's EG_F_ADD .. (the runtime does not include the spliced text) */
#define EG_F_SUB_ 65
#define EG_F_MUL_ 66
#define EG_F_MAX_ 68
#define EG_F_MIN_ 69

/* the faffine instance of F for blocks of sp: the least padding, then the most rows */
static int rt_eager_faffine_find(int F, int64_t sp, rt_eager_fp *out) {
    int found = 0;
    int64_t best = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_fp q;
        if (!rt_eager_fp_of(&aiednn_impl_list[i], "faffine", &q) || q.F != F)
            continue;
        const int64_t waste = (sp + q.L - 1) / q.L * q.L - sp;
        if (!found || waste < best || (waste == best && q.R * q.L > out->R * out->L))
            *out = q, best = waste, found = 1;
    }
    return found;
}

/* the problem as channel blocks: F, block length sp, blocks; 0 if not one */
static int eager_faffine_plan(const rt_prob *p, int *F, int64_t *sp, int64_t *nblk) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = rt_md(p, AIEDNN_ARG_DST);
    if (p->prop != RT_FWD_INF || !d || s->data_type != aiednn_f32 || d->data_type != aiednn_f32 || p->has_attr ||
        p->npo || rt_prob_has_fmt(p))
        return 0;
    if (p->kind == aiednn_batch_normalization && s->ndims >= 2) {
        *F = (p->flags & AIEDNN_FUSE_NORM_RELU) ? RT_EAGER_AFFINE_RELU : RT_EAGER_AFFINE;
        *sp = md_numel(s) / (s->dims[0] * s->dims[1]), *nblk = s->dims[0] * s->dims[1];
        return 1;
    }
    const aiednn_memory_desc_t *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    if (p->kind == aiednn_prelu && w && w->data_type == aiednn_f32) {
        *F = RT_EAGER_PRELU, *sp = fvec_bcast_block(w, d), *nblk = md_numel(d) / *sp;
        return 1;
    }
    return 0;
}

static int acc_eager_faffine(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    int F;
    int64_t sp, nb;
    rt_eager_fp q;
    if (!eager_faffine_plan(p, &F, &sp, &nb)) {
        snprintf(why, cap, "f32 inference batch norm / PReLU (f32 weights), no attr");
        return 0;
    }
    if (F == RT_EAGER_PRELU && sp < 16 && rt_eager_fp_find("fbinary", RT_EAGER_PRELU, &q))
        return 1;
    if (!rt_eager_faffine_find(F, sp, &q)) {
        snprintf(why, cap, "no faffine instance of F %d", F);
        return 0;
    }
    return 1;
}

static void *g_efa_x, *g_efa_y;
static size_t g_efa_x_cap, g_efa_y_cap;

/* block k's constants */
static void eager_faffine_consts(const rt_prob *p, void *const *ptr, int64_t k, int64_t sp, float *c) {
    if (p->kind == aiednn_prelu) {
        c[0] = ((const float *)rt_ptr(
            p, ptr, AIEDNN_ARG_WEIGHTS))[bcast_off(rt_md(p, AIEDNN_ARG_WEIGHTS), rt_md(p, AIEDNN_ARG_DST), k * sp)];
        c[1] = 0.0f;
        return;
    }
    const int64_t ch = k % p->md[0].dims[1];
    const double mean = ld_opt(p, ptr, AIEDNN_ARG_MEAN, ch, 0), var = ld_opt(p, ptr, AIEDNN_ARG_VARIANCE, ch, 1);
    const double g = ld_opt(p, ptr, AIEDNN_ARG_SCALE, ch, 1), b = ld_opt(p, ptr, AIEDNN_ARG_SHIFT, ch, 0);
    const double m = g / sqrt(var + p->eps);
    c[0] = (float)m, c[1] = (float)(b - mean * m);
}

static int run_eager_faffine(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    int F;
    int64_t sp, nb;
    rt_eager_fp q;
    if (!eager_faffine_plan(p, &F, &sp, &nb))
        return AIEK_ERR_PARAM;
    const float *xs = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC);
    float *ds = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DST);
    if (F == RT_EAGER_PRELU && sp < 16 && rt_eager_fp_find("fbinary", RT_EAGER_PRELU, &q))
        return eager_fbinary_run(self, &q, xs, &p->md[0], (const float *)rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS),
                                 rt_md(p, AIEDNN_ARG_WEIGHTS), rt_md(p, AIEDNN_ARG_DST), ds);
    if (!rt_eager_faffine_find(F, sp, &q))
        return AIEK_ERR_PARAM;
    const int64_t rpb = (sp + q.L - 1) / q.L, rows = nb * rpb; /* rows per block, rows */
    const int64_t ld = q.L + 4;                                /* a row: its L values, then {c0, c1, -, -} */
    float *x = (float *)scratch(&g_efa_x, &g_efa_x_cap, (size_t)(q.R * ld) * 4);
    float *y = (float *)scratch(&g_efa_y, &g_efa_y_cap, (size_t)(q.R * q.L) * 4);
    if (!x || !y)
        return AIEK_ERR_ALLOC;
    const int64_t xshp[2] = {q.R, ld}, yshp[2] = {q.R, q.L};
    float kc[2] = {0.0f, 0.0f};
    int64_t kblk = -1;
    for (int64_t r0 = 0; r0 < rows; r0 += q.R) {
        const int64_t nr = rows - r0 < q.R ? rows - r0 : q.R;
        uint64_t t0 = rt_ticks();
        memset(x, 0, (size_t)(q.R * ld) * 4);
        for (int64_t r = 0; r < nr; r++) {
            const int64_t g = r0 + r, k = g / rpb, o = g % rpb * q.L, len = sp - o < q.L ? sp - o : q.L;
            if (k != kblk)
                eager_faffine_consts(p, ptr, k, sp, kc), kblk = k;
            memcpy(x + r * ld, xs + k * sp + o, (size_t)len * 4);
            x[r * ld + q.L] = kc[0], x[r * ld + q.L + 1] = kc[1];
        }
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        aiek_tensor t[2] = {{x, 2, AIEK_FLOAT, 32, 1, xshp, (uint64_t)(q.R * ld) * 4},
                            {y, 2, AIEK_FLOAT, 32, 1, yshp, (uint64_t)(q.R * q.L) * 4}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 2);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        for (int64_t r = 0; r < nr; r++) {
            const int64_t g = r0 + r, k = g / rpb, o = g % rpb * q.L, len = sp - o < q.L ? sp - o : q.L;
            memcpy(ds + k * sp + o, y + r * q.L, (size_t)len * 4);
        }
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

/* ── eager:frow -- f32 row operations on the frow template (aie:eager_frow__R_L_TR_OP_YB) ───────────────────
 * softmax / log softmax over any axis (rows gathered at the axis stride), reductions (sum / mean / max / min over
 * any reduced axes: each output's sub-tensor gathered into a row, ref:'s index order) and 2-D f32 pooling (each
 * window a row: taps, padding -inf for max or 0 for average, the divisor -- k k or the valid count -- in the row's
 * last float). Rows of up to the instance's L, R per launch. FVEC's contract. */

#define RT_FROW_SOFTMAX 1 /* = eager_fmath_ops.cc's EG_ROW_* */
#define RT_FROW_LOGSOFTMAX 2
#define RT_FROW_SUM 3
#define RT_FROW_MEAN 4
#define RT_FROW_MAX 5
#define RT_FROW_MIN 6
#define RT_FROW_SUMDIV 7

typedef struct {
    const aiednn_impl_t *g;
    int64_t R, L;
    int op;
} rt_eager_frow;

static int rt_eager_frow_of(const aiednn_impl_t *g, rt_eager_frow *q) {
    long long r, l, tr, yb;
    int op;
    if (g->nargs != 3 || sscanf(g->name, "aie:eager_frow__%lld_%lld_%lld_%d_%lld", &r, &l, &tr, &op, &yb) != 5)
        return 0;
    q->g = g, q->R = r, q->L = l, q->op = op;
    return 1;
}

/* the instance of op whose rows hold n floats, the narrowest, then the most rows */
static int rt_eager_frow_find(int op, int64_t n, rt_eager_frow *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_frow q;
        if (!rt_eager_frow_of(&aiednn_impl_list[i], &q) || q.op != op || q.L < n)
            continue;
        if (!found || q.L < out->L || (q.L == out->L && q.R > out->R))
            *out = q, found = 1;
    }
    return found;
}

/* the problem as rows: op, row length (floats the row needs), rows */
typedef struct {
    int op;
    int64_t n, rows, lt; /* n: floats per row incl. a divisor; lt: the valid length the core reduces */
} rt_frow_plan;

static int eager_frow_plan(const rt_prob *p, rt_frow_plan *f) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    if (p->prop != RT_FWD_INF || s->data_type != aiednn_f32 || d->data_type != aiednn_f32 || p->has_attr || p->npo ||
        rt_prob_has_fmt(p))
        return 0;
    if (p->kind == aiednn_softmax && (p->alg == aiednn_softmax_accurate || p->alg == aiednn_softmax_log)) {
        f->op = p->alg == aiednn_softmax_log ? RT_FROW_LOGSOFTMAX : RT_FROW_SOFTMAX;
        f->n = f->lt = s->dims[p->axis], f->rows = md_numel(s) / f->n;
        return 1;
    }
    if (p->kind == aiednn_reduction) {
        const int op = p->alg == aiednn_reduction_sum    ? RT_FROW_SUM
                       : p->alg == aiednn_reduction_mean ? RT_FROW_MEAN
                       : p->alg == aiednn_reduction_max  ? RT_FROW_MAX
                       : p->alg == aiednn_reduction_min  ? RT_FROW_MIN
                                                         : 0;
        if (!op)
            return 0;
        f->op = op, f->rows = md_numel(d), f->n = f->lt = md_numel(s) / f->rows;
        return 1;
    }
    if (p->kind == aiednn_pooling && s->ndims == 4) {
        const int64_t taps = p->pool.kernel[0] * p->pool.kernel[1];
        f->op = p->alg == aiednn_pooling_max ? RT_FROW_MAX : RT_FROW_SUMDIV;
        f->lt = taps, f->n = taps + (f->op == RT_FROW_SUMDIV), f->rows = md_numel(d);
        return 1;
    }
    return 0;
}

static int acc_eager_frow(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    rt_frow_plan f;
    rt_eager_frow q;
    if (!eager_frow_plan(p, &f)) {
        snprintf(why, cap, "f32 softmax / log softmax, sum / mean / max / min reduction or 2-D pooling, no attr");
        return 0;
    }
    if (!rt_eager_frow_find(f.op, f.op == RT_FROW_SUMDIV ? f.lt + 1 : f.n, &q)) {
        snprintf(why, cap, "no frow instance of op %d with rows >= %lld", f.op, (long long)f.n);
        return 0;
    }
    return 1;
}

/* row r (of the problem's rows) into dst [L] (padding past the row as the op needs) */
static void eager_frow_row(const rt_prob *p, const rt_frow_plan *f, const float *x, int64_t r, float *dst, int64_t L) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    if (p->kind == aiednn_softmax) { /* rows at the axis stride */
        int64_t inner = 1;
        for (int i = p->axis + 1; i < s->ndims; i++)
            inner *= s->dims[i];
        const int64_t base = (r / inner) * f->n * inner + r % inner;
        for (int64_t a = 0; a < f->n; a++)
            dst[a] = x[base + a * inner];
        return;
    }
    if (p->kind == aiednn_reduction) { /* the reduced sub-tensor of output r, ref:'s order */
        int64_t dc[AIEDNN_MAX_NDIMS], rem = r;
        for (int i = d->ndims - 1; i >= 0; i--)
            dc[i] = rem % d->dims[i], rem /= d->dims[i];
        for (int64_t k = 0; k < f->n; k++) {
            int64_t off = 0, rr = k;
            for (int i = s->ndims - 1, stride = 1; i >= 0; i--) {
                int64_t c = dc[i];
                if (d->dims[i] == 1 && s->dims[i] != 1)
                    c = rr % s->dims[i], rr /= s->dims[i];
                off += c * stride;
                stride *= (int)s->dims[i];
            }
            dst[k] = x[off];
        }
        return;
    }
    /* pooling window r = (nc, oh, ow) */
    const aiednn_pool_params_t *pp = &p->pool;
    const int64_t H = s->dims[2], W = s->dims[3], Ho = d->dims[2], Wo = d->dims[3];
    const int64_t nc = r / (Ho * Wo), oh = r / Wo % Ho, ow = r % Wo;
    const int mx = f->op == RT_FROW_MAX;
    int64_t t = 0, cnt = 0;
    for (int64_t kh = 0; kh < pp->kernel[0]; kh++)
        for (int64_t kw = 0; kw < pp->kernel[1]; kw++, t++) {
            const int64_t ih = oh * pp->strides[0] - pp->padding_l[0] + kh * (pp->dilates[0] + 1);
            const int64_t iw = ow * pp->strides[1] - pp->padding_l[1] + kw * (pp->dilates[1] + 1);
            const int in = ih >= 0 && ih < H && iw >= 0 && iw < W;
            dst[t] = in ? x[(nc * H + ih) * W + iw] : mx ? -INFINITY : 0.0f;
            cnt += in;
        }
    if (!mx) { /* the divisor: k k (include padding) or the valid count (exclude; 0 taps: 0 / 1 = 0, as ref:) */
        const int64_t dv = p->alg == aiednn_pooling_avg_include_padding ? t : cnt;
        dst[L - 1] = (float)(dv ? dv : 1);
    }
}

static void *g_efr_x, *g_efr_y;
static size_t g_efr_x_cap, g_efr_y_cap;

static int run_eager_frow(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_frow_plan f;
    rt_eager_frow q;
    if (!eager_frow_plan(p, &f) || !rt_eager_frow_find(f.op, f.op == RT_FROW_SUMDIV ? f.lt + 1 : f.n, &q))
        return AIEK_ERR_PARAM;
    const int red = f.op >= RT_FROW_SUM;
    const int64_t yw = red ? 4 : q.L; /* floats per output row */
    float *x = (float *)scratch(&g_efr_x, &g_efr_x_cap, (size_t)(q.R * q.L) * 4);
    float *y = (float *)scratch(&g_efr_y, &g_efr_y_cap, (size_t)(q.R * yw) * 4);
    if (!x || !y)
        return AIEK_ERR_ALLOC;
    int32_t pl[16] = {0};
    for (int c = 0; c < 4; c++)
        pl[4 * c] = (int32_t)f.lt;
    const float *xs = (const float *)ptr[0];
    float *ds = (float *)ptr[1];
    const int64_t xshp[2] = {q.R, q.L}, pshp[2] = {4, 4}, yshp[2] = {q.R, yw};
    for (int64_t r0 = 0; r0 < f.rows; r0 += q.R) {
        const int64_t nr = f.rows - r0 < q.R ? f.rows - r0 : q.R;
        uint64_t t0 = rt_ticks();
        memset(x, 0, (size_t)(q.R * q.L) * 4);
        for (int64_t r = 0; r < nr; r++)
            eager_frow_row(p, &f, xs, r0 + r, x + r * q.L, q.L);
        for (int64_t r = nr; r < q.R && f.op == RT_FROW_SUMDIV; r++)
            x[r * q.L + q.L - 1] = 1.0f; /* padding rows: no division by zero */
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        aiek_tensor t[3] = {{x, 2, AIEK_FLOAT, 32, 1, xshp, (uint64_t)(q.R * q.L) * 4},
                            {pl, 2, AIEK_INT, 32, 1, pshp, 64},
                            {y, 2, AIEK_FLOAT, 32, 1, yshp, (uint64_t)(q.R * yw) * 4}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        for (int64_t r = 0; r < nr; r++) {
            const int64_t g = r0 + r;
            if (red || p->kind != aiednn_softmax) {
                ds[g] = y[r * yw];
                continue;
            }
            int64_t inner = 1; /* softmax rows back at the axis stride */
            for (int i = p->axis + 1; i < p->md[0].ndims; i++)
                inner *= p->md[0].dims[i];
            const int64_t base = (g / inner) * f.n * inner + g % inner;
            for (int64_t a = 0; a < f.n; a++)
                ds[base + a * inner] = y[r * yw + a];
        }
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

/* ── eager:bin8 -- binary of two 8-bit tensors on the bin8 template (aie:eager_bin8__R_L_TR_OP_Q) ──────────────
 * Q 0: the exact int32 a OP b (add / sub / mul / max / min; comparisons through sub, as eltblk:), stored as ref:
 * does for any destination type. Q 1: the quantized binary (attr src scales / zero points, dst scale / zero
 * point, one relu / clip post-op): FVEC's program in the same float order on the core, the quantize on the host.
 * Either operand may broadcast (packed per element). */

typedef struct {
    const aiednn_impl_t *g;
    int64_t R, L;
    int op, Q;
} rt_eager_b8;

static int rt_eager_b8_of(const aiednn_impl_t *g, rt_eager_b8 *q) {
    long long r, l, tr;
    int op, qq;
    if (g->nargs != 3 || sscanf(g->name, "aie:eager_bin8__%lld_%lld_%lld_%d_%d", &r, &l, &tr, &op, &qq) != 5)
        return 0;
    q->g = g, q->R = r, q->L = l, q->op = op, q->Q = qq;
    return 1;
}

static int rt_eager_b8_find(int op, int Q, rt_eager_b8 *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_b8 q;
        if (rt_eager_b8_of(&aiednn_impl_list[i], &q) && q.op == op && q.Q == Q &&
            (!found || q.R * q.L > out->R * out->L))
            *out = q, found = 1;
    }
    return found;
}

static int is_8bit(aiednn_data_type_t t) { return t == aiednn_s8 || t == aiednn_u8; }

/* the problem's (OP, Q), 0 if bin8 cannot serve it */
static int eager_bin8_plan(const rt_prob *p, int *op, int *Q) {
    const aiednn_memory_desc_t *a = &p->md[0], *b = &p->md[1], *d = &p->md[2];
    if (p->kind != aiednn_binary || p->prop != RT_FWD_INF || !is_8bit(a->data_type) || !is_8bit(b->data_type) ||
        rt_prob_has_fmt(p))
        return 0;
    if (!p->has_attr) {
        const int e = eltblk_op(p->alg);
        *op = e == AIE_MODE_ELT_ADD   ? EG_F_ADD_
              : e == AIE_MODE_ELT_MUL ? EG_F_MUL_
              : e == AIE_MODE_ELT_MAX ? EG_F_MAX_
              : e == AIE_MODE_ELT_MIN ? EG_F_MIN_
              : e == AIE_MODE_ELT_SUB ? EG_F_SUB_
                                      : 0;
        lut_plan lq;
        char lw[8];
        *Q = 0;
        return *op && !p->npo && !lut_plan_make(p, &lq, lw, sizeof lw); /* one-axis broadcasts: lut: serves them */
    }
    const int tol1 = (p->attr_flags & AIEDNN_ATTR_TOLERANCE_1CODE) != 0;
    *op = p->alg == aiednn_binary_add           ? EG_F_ADD_
          : p->alg == aiednn_binary_sub         ? EG_F_SUB_
          : p->alg == aiednn_binary_max         ? EG_F_MAX_
          : p->alg == aiednn_binary_min         ? EG_F_MIN_
          : p->alg == aiednn_binary_mul && tol1 ? EG_F_MUL_
                                                : 0;
    *Q = 1;
    if (!*op || !md_same_shape(a, d) || p->npo > 1 || (d->data_type != aiednn_f32 && !is_8bit(d->data_type)))
        return 0;
    return !p->npo ||
           (p->po[0].kind == aiednn_post_op_eltwise &&
            ((p->po[0].alg == aiednn_eltwise_relu && p->po[0].alpha == 0.0f) || p->po[0].alg == aiednn_eltwise_clip));
}

static int acc_eager_bin8(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    int op, Q;
    rt_eager_b8 q;
    if (!eager_bin8_plan(p, &op, &Q)) {
        snprintf(why, cap,
                 "binary of two 8-bit tensors: add / sub / mul / max / min / compare without attr, or the "
                 "quantized add / sub / max / min (mul with AIEDNN_ATTR_TOLERANCE_1CODE), <= 1 relu / clip");
        return 0;
    }
    if (!rt_eager_b8_find(op, Q, &q)) {
        snprintf(why, cap, "no bin8 instance of op %d, Q %d", op, Q);
        return 0;
    }
    return 1;
}

static void *g_eb8_x, *g_eb8_z;
static size_t g_eb8_x_cap, g_eb8_z_cap;

static int run_eager_bin8(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *a = &p->md[0], *b = &p->md[1], *d = &p->md[2];
    int op, Q;
    rt_eager_b8 q;
    if (!eager_bin8_plan(p, &op, &Q) || !rt_eager_b8_find(op, Q, &q))
        return AIEK_ERR_PARAM;
    const int64_t n = md_numel(d), per = q.R * q.L;
    uint8_t *xy = (uint8_t *)scratch(&g_eb8_x, &g_eb8_x_cap, (size_t)per * 2);
    int32_t *z = (int32_t *)scratch(&g_eb8_z, &g_eb8_z_cap, (size_t)per * 4);
    if (!xy || !z)
        return AIEK_ERR_ALLOC;
    float pw[32] = {0};
    const int sg = (a->data_type == aiednn_s8) | (b->data_type == aiednn_s8) << 1;
    for (int c = 0; c < 4; c++) {
        float *w = pw + 8 * c;
        for (int s = 0; s < 2; s++) {
            const int dq = Q && p->bin_sc[s] != 0.0f;
            w[2 * s] = dq ? p->bin_sc[s] : 1.0f, w[2 * s + 1] = dq ? (float)-p->bin_zp[s] : 0.0f;
        }
        w[4] = -FLT_MAX, w[5] = FLT_MAX;
        if (Q && p->npo && p->po[0].alg == aiednn_eltwise_relu)
            w[4] = 0.0f;
        else if (Q && p->npo)
            w[4] = p->po[0].alpha, w[5] = p->po[0].beta;
        memcpy(w + 6, &sg, 4);
    }
    const uint8_t *x = (const uint8_t *)ptr[0], *y = (const uint8_t *)ptr[1];
    const int af = same_dims(a, d), bf = same_dims(b, d);
    const int64_t xshp[2] = {q.R, 2 * q.L}, pshp[2] = {4, 8}, zshp[2] = {q.R, q.L};
    for (int64_t e0 = 0; e0 < n; e0 += per) {
        const int64_t cnt = n - e0 < per ? n - e0 : per;
        uint64_t t0 = rt_ticks();
        memset(xy, 0, (size_t)per * 2);
        for (int64_t i = 0; i < cnt; i++) {
            const int64_t r = i / q.L, c = i % q.L, o = e0 + i;
            xy[r * 2 * q.L + c] = x[af ? o : bcast_off(a, d, o)];
            xy[r * 2 * q.L + q.L + c] = y[bf ? o : bcast_off(b, d, o)];
        }
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        aiek_tensor t[3] = {{xy, 2, AIEK_INT, 8, 1, xshp, (uint64_t)per * 2},
                            {pw, 2, AIEK_FLOAT, 32, 1, pshp, 128},
                            {z, 2, (uint8_t)(Q ? AIEK_FLOAT : AIEK_INT), 32, 1, zshp, (uint64_t)per * 4}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        if (Q)
            fvec_qstore(p, ptr, e0, (const float *)z, cnt);
        else {
            const int zsign = p->alg == aiednn_binary_mul && d->data_type == aiednn_f32;
            for (int64_t i = 0; i < cnt; i++) {
                double v = eltblk_value(p->alg, z[i]);
                if (zsign && z[i] == 0) { /* ref's double product, with its signed zero */
                    const int64_t r = i / q.L, c = i % q.L;
                    const uint8_t xa = xy[r * 2 * q.L + c], yb = xy[r * 2 * q.L + q.L + c];
                    v = (double)(sg & 1 ? (int8_t)xa : xa) * (double)(sg & 2 ? (int8_t)yb : yb);
                }
                rt_st(d->data_type, ptr[2], e0 + i, v);
            }
        }
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_EAGER_FVEC_H */
