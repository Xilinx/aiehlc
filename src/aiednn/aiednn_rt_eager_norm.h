/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_eager_norm.h -- eager mode's row ops: eager:lnorm (layer / RMS norm on the lnorm template) and
 * eager:rope (rotary position embedding on the rope template), src/aietensorop/eager/.
 *
 * Needs, from the including TU: aiednn_rt_ops.h (rt_quant8_row).
 ******************************************************************************/
#ifndef AIEDNN_RT_EAGER_NORM_H
#define AIEDNN_RT_EAGER_NORM_H

/* ── eager:lnorm -- layer / RMS norm rows (lnorm instances, aie:eager_lnorm__R_L_TR_RMS) ─────────────────────
 * f32 rows normalized on the core in single precision (src/aietensorop/eager/lnorm_eager_ref.h): within the 1e-5
 * contract against ref:'s double arithmetic, as fixed-megakernel fvec: is -- so the same rule: an f32 dst always,
 * an s8 / u8 dst only when the caller allows a code at a rounding boundary to differ (AIEDNN_ATTR_TOLERANCE_1CODE),
 * quantized here with the quantize reorder's arithmetic. Rows of any length <= the instance's L run in chunks of R. */

typedef struct {
    const aiednn_impl_t *g;
    int64_t R, L;
    int rms;
} rt_eager_lnorm;

static int rt_eager_lnorm_of(const aiednn_impl_t *g, rt_eager_lnorm *q) {
    long long r, l, tr;
    int rms;
    if (g->nargs != 3 || sscanf(g->name, "aie:eager_lnorm__%lld_%lld_%lld_%d", &r, &l, &tr, &rms) != 4)
        return 0;
    q->g = g, q->R = r, q->L = l, q->rms = rms;
    return 1;
}

/* the instance of the norm's kind with the narrowest row >= C, then the most rows per launch */
static int rt_eager_lnorm_find(int64_t C, int rms, rt_eager_lnorm *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_lnorm q;
        if (!rt_eager_lnorm_of(&aiednn_impl_list[i], &q) || q.rms != rms || q.L < C)
            continue;
        if (!found || q.L < out->L || (q.L == out->L && q.R > out->R))
            *out = q, found = 1;
    }
    return found;
}

static int acc_eager_lnorm(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const aiednn_memory_desc_t *gmd = rt_md(p, AIEDNN_ARG_SCALE), *bmd = rt_md(p, AIEDNN_ARG_SHIFT);
    const int q8 = (d->data_type == aiednn_s8 || d->data_type == aiednn_u8) && p->has_attr && !p->npo &&
                   !p->n_oscales && (p->attr_flags & AIEDNN_ATTR_TOLERANCE_1CODE);
    if (p->kind != aiednn_layer_normalization || p->prop != RT_FWD_INF || s->data_type != aiednn_f32 || p->ndq ||
        (d->data_type != aiednn_f32 && !q8) || (d->data_type == aiednn_f32 && p->has_attr) ||
        (p->flags & AIEDNN_FUSE_NORM_RELU) || rt_prob_has_fmt(p) || (gmd && gmd->data_type != aiednn_f32) ||
        (bmd && bmd->data_type != aiednn_f32)) {
        snprintf(why, cap,
                 "f32 layer / RMS norm inference -> f32 (-> s8 / u8 only with AIEDNN_ATTR_TOLERANCE_1CODE), "
                 "f32 scale / shift, no ReLU");
        return 0;
    }
    const int64_t C = s->dims[s->ndims - 1];
    rt_eager_lnorm q;
    if (!rt_eager_lnorm_find(C, (p->flags & AIEDNN_RMS_NORM) != 0, &q)) {
        snprintf(why, cap, "no lnorm instance with rows >= %lld", (long long)C);
        return 0;
    }
    return 1;
}

static void *g_eln_x, *g_eln_p, *g_eln_y;
static size_t g_eln_x_cap, g_eln_p_cap, g_eln_y_cap;

static int run_eager_lnorm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const int64_t C = s->dims[s->ndims - 1], rows = md_numel(s) / C;
    rt_eager_lnorm q;
    if (!rt_eager_lnorm_find(C, (p->flags & AIEDNN_RMS_NORM) != 0, &q))
        return AIEK_ERR_PARAM;
    const int64_t rb = 4 * q.L; /* bytes per row */
    uint8_t *x = (uint8_t *)scratch(&g_eln_x, &g_eln_x_cap, (size_t)(q.R * rb));
    uint8_t *pp = (uint8_t *)scratch(&g_eln_p, &g_eln_p_cap, (size_t)(12 * q.L));
    float *y = (float *)scratch(&g_eln_y, &g_eln_y_cap, (size_t)(q.R * rb));
    if (!x || !pp || !y)
        return AIEK_ERR_ALLOC;
    const float *gs = rt_md(p, AIEDNN_ARG_SCALE) ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SCALE) : nullptr;
    const float *bs = rt_md(p, AIEDNN_ARG_SHIFT) ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SHIFT) : nullptr;
    memset(pp, 0, (size_t)(12 * q.L)); /* pre-tiled: core c's rows 3c, 3c + 1, 3c + 2 = its gamma, beta, params */
    const int64_t qc = q.L / 4;
    const float eps = p->eps, rl = 1.0f / (float)C;
    const int32_t lt = (int32_t)C;
    for (int64_t c = 0; c < 4; c++) {
        float *gam = (float *)(pp + 3 * c * q.L), *bet = (float *)(pp + (3 * c + 1) * q.L);
        for (int64_t k = 0; k < qc && c * qc + k < C; k++)
            gam[k] = gs ? gs[c * qc + k] : 1.0f, bet[k] = bs ? bs[c * qc + k] : 0.0f;
        uint8_t *at = pp + (3 * c + 2) * q.L;
        memcpy(at, &eps, 4), memcpy(at + 4, &rl, 4), memcpy(at + 8, &lt, 4);
    }
    const int q8i = d->data_type != aiednn_f32;
    const double ds = p->dst_scale, zp = p->dst_zp, lo = d->data_type == aiednn_s8 ? -128.0 : 0.0;
    const int64_t shp[2] = {q.R, rb}, pshp[2] = {12, q.L};
    for (int64_t r0 = 0; r0 < rows; r0 += q.R) {
        const int64_t nr = rows - r0 < q.R ? rows - r0 : q.R;
        uint64_t t0 = rt_ticks();
        memset(x, 0, (size_t)(q.R * rb));
        for (int64_t r = 0; r < nr; r++)
            memcpy(x + r * rb, (const float *)ptr[0] + (r0 + r) * C, (size_t)(C * 4));
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        aiek_tensor t[3] = {{x, 2, AIEK_INT, 8, 1, shp, (uint64_t)(q.R * rb)},
                            {pp, 2, AIEK_INT, 8, 1, pshp, (uint64_t)(12 * q.L)},
                            {y, 2, AIEK_INT, 8, 1, shp, (uint64_t)(q.R * rb)}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        for (int64_t r = 0; r < nr; r++) {
            const float *yr = y + r * q.L;
            if (q8i)
                rt_quant8_row(yr, C, ds, zp, lo, (uint8_t *)ptr[1] + (r0 + r) * C);
            else
                memcpy((float *)ptr[1] + (r0 + r) * C, yr, (size_t)(C * 4));
        }
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

/* ── eager:rope -- rotary position embedding (rope instances, aie:eager_rope__R_D_TR_INTER) ────────────────────
 * f32 rows of the head dim D; the core does ref:'s float multiplies and add (src/aietensorop/eager/rope_eager_ref.h),
 * so f32 results are ref:'s bit for bit; an s8 / u8 dst (the attr form) is quantized here with ref:'s arithmetic.
 * The host packs each row with its broadcast cos / sin rows. Only f32 sources (quantized sources stay on ref:). */

#define RT_ROPE_Q 2u /* rt_prob.flags: the attr's quantization (dq_sc / dq_zp of x, cos, sin; dst_scale / dst_zp) */

typedef struct {
    const aiednn_impl_t *g;
    int64_t R, D;
    int inter;
} rt_eager_rope;

static int rt_eager_rope_of(const aiednn_impl_t *g, rt_eager_rope *q) {
    long long r, d, tr;
    int in;
    if (g->nargs != 2 || sscanf(g->name, "aie:eager_rope__%lld_%lld_%lld_%d", &r, &d, &tr, &in) != 4)
        return 0;
    q->g = g, q->R = r, q->D = d, q->inter = in;
    return 1;
}

static int rt_eager_rope_find(int64_t D, int inter, rt_eager_rope *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_rope q;
        if (rt_eager_rope_of(&aiednn_impl_list[i], &q) && q.D == D && q.inter == inter && (!found || q.R > out->R))
            *out = q, found = 1;
    }
    return found;
}

static int acc_eager_rope(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1], *cm = &p->md[2], *sm = &p->md[3];
    const int q8 = (p->flags & RT_ROPE_Q) && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8);
    if (p->kind != aiednn_rope || s->data_type != aiednn_f32 || cm->data_type != aiednn_f32 ||
        sm->data_type != aiednn_f32 || (d->data_type != aiednn_f32 && !q8) || rt_prob_has_fmt(p)) {
        snprintf(why, cap, "f32 x / cos / sin, an f32 dst or an 8-bit one (the attr form)");
        return 0;
    }
    rt_eager_rope q;
    if (!rt_eager_rope_find(s->dims[s->ndims - 1], (p->flags & 1) != 0, &q)) {
        snprintf(why, cap, "no rope instance with D %lld (%s)", (long long)s->dims[s->ndims - 1],
                 (p->flags & 1) ? "interleaved" : "half-split");
        return 0;
    }
    return 1;
}

static void *g_erp_x, *g_erp_y;
static size_t g_erp_x_cap, g_erp_y_cap;

static int run_eager_rope(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1], *cm = &p->md[2], *sm = &p->md[3];
    const int64_t D = s->dims[s->ndims - 1], rows = md_numel(s) / D;
    rt_eager_rope q;
    if (!rt_eager_rope_find(D, (p->flags & 1) != 0, &q))
        return AIEK_ERR_PARAM;
    float *x = (float *)scratch(&g_erp_x, &g_erp_x_cap, (size_t)(q.R * 12 * D));
    float *y = (float *)scratch(&g_erp_y, &g_erp_y_cap, (size_t)(q.R * 4 * D));
    if (!x || !y)
        return AIEK_ERR_ALLOC;
    const int q8 = d->data_type != aiednn_f32;
    const double ds = p->dst_scale, zp = p->dst_zp, lo = d->data_type == aiednn_s8 ? -128.0 : 0.0;
    const float *xs = (const float *)ptr[0], *cs = (const float *)ptr[2], *ss = (const float *)ptr[3];
    const int64_t xshp[2] = {q.R, 12 * D}, yshp[2] = {q.R, 4 * D};
    for (int64_t r0 = 0; r0 < rows; r0 += q.R) {
        const int64_t nr = rows - r0 < q.R ? rows - r0 : q.R;
        uint64_t t0 = rt_ticks();
        memset(x, 0, (size_t)(q.R * 12 * D));
        for (int64_t r = 0; r < nr; r++) { /* [x | cos | sin] */
            float *row = x + r * 3 * D;
            const int64_t e0 = (r0 + r) * D;
            memcpy(row, xs + e0, (size_t)(D * 4));
            for (int64_t j = 0; j < D; j++)
                row[D + j] = cs[bcast_off(cm, s, e0 + j)], row[2 * D + j] = ss[bcast_off(sm, s, e0 + j)];
        }
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        aiek_tensor t[2] = {{x, 2, AIEK_INT, 8, 1, xshp, (uint64_t)(q.R * 12 * D)},
                            {y, 2, AIEK_INT, 8, 1, yshp, (uint64_t)(q.R * 4 * D)}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 2);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        if (q8)
            rt_quant8_row(y, nr * D, ds, zp, lo, (uint8_t *)ptr[1] + r0 * D);
        else
            memcpy((float *)ptr[1] + r0 * D, y, (size_t)(nr * D * 4));
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_EAGER_NORM_H */
