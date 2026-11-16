/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_pool.h -- poolblk:<aie pool block>: pooling on the GEMM ELF's pool
 * mode (aie_modes_pool.h, opcode 0xB0), 16 output bands per launch.
 *
 * Item = (image*channel plane, band of output rows): input rows at pitch Wp = (Wo-1)*s + KW, padded
 * with the type minimum (max) or 0 (avg). Kernel block:
 *   items s8 [16][4096], geom s32 [8] = {op, signed, s, KH, KW, Wp, Rout, Wo},
 *   out s32 [16][1024] = per output the window's exact max or sum.
 * The host finishes as ref:pooling does. Windows with no valid element are not accepted.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aie_modes_pool.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_POOL_H
#define AIEDNN_RT_POOL_H

#define RT_PL_MAX_ITEMS 64 /* items per launch = the impl's arg 0 rows (16 on 4x4) */
#define RT_PL_BYTES AIE_MODE_IN
#define RT_PL_OUT AIE_MODE_OUT

static int is_pool_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_pooling && g->nargs == 3 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[0].dims[0] >= 1 && g->args[0].dims[0] <= RT_PL_MAX_ITEMS && g->args[0].dims[1] == RT_PL_BYTES &&
           g->args[1].data_type == aiednn_s32 && g->args[2].data_type == aiednn_s32 && g->args[2].ndims == 2 &&
           g->args[2].dims[1] == RT_PL_OUT;
}

typedef struct {
    int64_t NC, H, W, Ho, Wo, KH, KW, s, pt, pl;
    int64_t wp, rout, bands;
    int op, sgn;
} pool_plan;

/* a window of output position o covers [o*s - pad, o*s - pad + k); valid if it
 * meets [0, n). Checking the first and last windows covers all of them. */
static int pool_dim_ok(int64_t n, int64_t no, int64_t k, int64_t s, int64_t pad) {
    const int64_t first = -pad, last = (no - 1) * s - pad;
    return first + k > 0 && last < n;
}

static int pool_plan_make(const rt_prob *p, pool_plan *q, char *why, int cap) {
    const aiednn_memory_desc_t *src = &p->md[0], *dst = &p->md[1];
    const aiednn_pool_params_t *pp = &p->pool;
    if (src->data_type != aiednn_s8 && src->data_type != aiednn_u8) {
        snprintf(why, cap, "source must be s8 or u8");
        return 0;
    }
    if (pp->dilates[0] || pp->dilates[1] || pp->strides[0] != pp->strides[1] || pp->strides[0] > 255 ||
        pp->kernel[0] > 255 || pp->kernel[1] > 255) {
        snprintf(why, cap, "needs undilated, equal strides <= 255, kernel <= 255");
        return 0;
    }
    q->NC = src->dims[0] * src->dims[1], q->H = src->dims[2], q->W = src->dims[3];
    q->Ho = dst->dims[2], q->Wo = dst->dims[3];
    q->KH = pp->kernel[0], q->KW = pp->kernel[1], q->s = pp->strides[0];
    q->pt = pp->padding_l[0], q->pl = pp->padding_l[1];
    q->op = p->alg == aiednn_pooling_max ? AIE_MODE_POOL_MAX : AIE_MODE_POOL_SUM;
    q->sgn = src->data_type == aiednn_s8;
    if (!pool_dim_ok(q->H, q->Ho, q->KH, q->s, q->pt) || !pool_dim_ok(q->W, q->Wo, q->KW, q->s, q->pl)) {
        snprintf(why, cap, "a window lies entirely in the padding");
        return 0;
    }
    q->wp = (q->Wo - 1) * q->s + q->KW;
    for (int64_t r = q->Ho < RT_PL_OUT ? q->Ho : RT_PL_OUT; r >= 1; r--)
        if (r * q->Wo <= RT_PL_OUT && ((r - 1) * q->s + q->KH) * q->wp <= RT_PL_BYTES && q->wp < 65536) {
            q->rout = r;
            q->bands = (q->Ho + r - 1) / r;
            return 1;
        }
    snprintf(why, cap, "one output row and its input rows do not fit an item");
    return 0;
}

#define RT_PL_MIN_WORK 1024

static int acc_poolblk(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    (void)bind;
    pool_plan q;
    if (!pool_plan_make(p, &q, why, cap))
        return 0;
    /* one item per (plane, band): little window work per item makes launches cost more than ref: */
    if (rt_cost_refusals() && q.rout * q.Wo * q.KH * q.KW < RT_PL_MIN_WORK) { /* aiednn_prefer_fastest only */
        snprintf(why, cap, "%lld window ops per item (< %d): ref: is faster", (long long)(q.rout * q.Wo * q.KH * q.KW),
                 RT_PL_MIN_WORK);
        return 0;
    }
    return 1;
}

/* item (plane, band) -> its padded input rows */
static void pool_pack_item(const pool_plan *q, const uint8_t *plane, int64_t band, uint8_t *raw) {
    const uint8_t pad = q->op == AIE_MODE_POOL_MAX && q->sgn ? 0x80 : 0;
    const int64_t oh0 = band * q->rout, rin = (q->rout - 1) * q->s + q->KH;
    memset(raw, pad, RT_PL_BYTES);
    const int64_t c0 = q->pl > 0 ? q->pl : 0, c1 = q->pl + q->W < q->wp ? q->pl + q->W : q->wp;
    for (int64_t r = 0; r < rin; r++) {
        const int64_t ih = oh0 * q->s - q->pt + r;
        if (ih >= 0 && ih < q->H && c1 > c0) /* columns c0 .. c1 hold input columns c - pl */
            memcpy(raw + r * q->wp + c0, plane + ih * q->W + (c0 - q->pl), (size_t)(c1 - c0));
    }
}

static int64_t pool_valid(int64_t o, int64_t s, int64_t pad, int64_t k, int64_t n) {
    int64_t lo = o * s - pad, hi = lo + k;
    lo = lo < 0 ? 0 : lo;
    hi = hi > n ? n : hi;
    return hi > lo ? hi - lo : 0;
}

/* the item's exact window values -> dst, as ref:pooling finishes them */
static void pool_finish_item(const rt_prob *p, const pool_plan *q, int64_t nc, int64_t band, const int32_t *v,
                             void *dst) {
    const aiednn_data_type_t dt = p->md[1].data_type;
    const int64_t oh0 = band * q->rout, rows = q->Ho - oh0 < q->rout ? q->Ho - oh0 : q->rout;
    if (p->alg == aiednn_pooling_max && (dt == aiednn_s8 || dt == aiednn_u8)) {
        /* the window maximum of 8-bit codes is the output code itself: a typed row copy */
        for (int64_t r = 0; r < rows; r++) {
            const int32_t *vr = v + r * q->Wo;
            const int64_t o = (nc * q->Ho + oh0 + r) * q->Wo;
            if (dt == aiednn_s8)
                for (int64_t ow = 0; ow < q->Wo; ow++)
                    ((int8_t *)dst)[o + ow] = (int8_t)vr[ow];
            else
                for (int64_t ow = 0; ow < q->Wo; ow++)
                    ((uint8_t *)dst)[o + ow] = (uint8_t)vr[ow];
        }
        return;
    }
    for (int64_t r = 0; r < rows; r++)
        for (int64_t ow = 0; ow < q->Wo; ow++) {
            const int64_t oh = oh0 + r;
            double acc = (double)v[r * q->Wo + ow];
            if (p->alg == aiednn_pooling_avg_include_padding) {
                acc /= (double)(q->KH * q->KW);
            } else if (p->alg == aiednn_pooling_avg_exclude_padding) {
                const int64_t cnt = pool_valid(oh, q->s, q->pt, q->KH, q->H) * pool_valid(ow, q->s, q->pl, q->KW, q->W);
                acc = cnt ? acc / (double)cnt : 0.0;
            }
            rt_st(dt, dst, (nc * q->Ho + oh) * q->Wo + ow, acc);
        }
}

static void *g_pli, *g_plo;
static size_t g_pli_cap, g_plo_cap;

typedef struct {
    const rt_prob *p;
    const pool_plan *q;
    const uint8_t *x;
    void *dst;
} pool_itn_ctx;
static void pool_itn_pack(void *c, int64_t it, uint8_t *raw) {
    const pool_itn_ctx *k = (const pool_itn_ctx *)c;
    pool_pack_item(k->q, k->x + (it / k->q->bands) * k->q->H * k->q->W, it % k->q->bands, raw);
}
static void pool_itn_put(void *c, int64_t it, const int32_t *res) {
    const pool_itn_ctx *k = (const pool_itn_ctx *)c;
    pool_finish_item(k->p, k->q, it / k->q->bands, it % k->q->bands, res, k->dst);
}

static int run_poolblk(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    pool_plan q;
    char why[8];
    pool_plan_make(p, &q, why, sizeof why);
    { /* many launches' worth of items in one (aiednn_rt_itemsn.h) */
        const int32_t g[8] = {q.op,          q.sgn,         (int32_t)q.s,    (int32_t)q.KH,
                              (int32_t)q.KW, (int32_t)q.wp, (int32_t)q.rout, (int32_t)q.Wo};
        uint8_t d[11];
        AIE_MODE_POOL_DESC(d, g);
        pool_itn_ctx k = {p, &q, (const uint8_t *)ptr[0], ptr[1]};
        const int rc = rt_items_n(self, q.NC * q.bands, AIE_MODE_OP_POOL, d, 11, pool_itn_pack, pool_itn_put, &k);
        if (rc != RT_NOT_TAKEN)
            return rc;
    }
    const int64_t NI = self->gen->args[0].dims[0]; // items per launch
    uint8_t *items = (uint8_t *)scratch(&g_pli, &g_pli_cap, NI * RT_PL_BYTES);
    int32_t *out = (int32_t *)scratch(&g_plo, &g_plo_cap, NI * RT_PL_OUT * sizeof(int32_t));
    if (!items || !out)
        return AIEK_ERR_ALLOC;
    const uint8_t *x = (const uint8_t *)ptr[0];
    const int64_t total = q.NC * q.bands;
    int32_t geom[8] = {q.op,          q.sgn,         (int32_t)q.s,    (int32_t)q.KH,
                       (int32_t)q.KW, (int32_t)q.wp, (int32_t)q.rout, (int32_t)q.Wo};
    const int64_t shape_i[2] = {NI, RT_PL_BYTES}, shape_g[1] = {8}, shape_o[2] = {NI, RT_PL_OUT};
    for (int64_t base = 0; base < total; base += NI) {
        const int64_t n_items = total - base < NI ? total - base : NI;
        const uint64_t tp0 = rt_ticks();
        for (int64_t k = 0; k < n_items; k++) {
            const int64_t it = base + k;
            pool_pack_item(&q, x + (it / q.bands) * q.H * q.W, it % q.bands, items + k * RT_PL_BYTES);
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        aiek_tensor t[3] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * RT_PL_BYTES)},
                            {geom, 1, AIEK_INT, 32, 1, shape_g, sizeof geom},
                            {out, 2, AIEK_INT, 32, 1, shape_o, (uint64_t)NI * RT_PL_OUT * sizeof(int32_t)}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        const int rc = self->gen->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t te0 = rt_ticks();
        for (int64_t k = 0; k < n_items; k++) {
            const int64_t it = base + k;
            pool_finish_item(p, &q, it / q.bands, it % q.bands, out + k * RT_PL_OUT, ptr[1]);
        }
        g_phase[RT_PH_EPI] += rt_ticks() - te0;
    }
    return AIEK_OK;
}

/* x86 stand-in for the AIE pool block (same macro the core runs): items, geom, out as above. */
static int rt_pool_block_x86(const aiek_tensor *t, int n) {
    if (n != 3)
        return AIEK_ERR_ARGS;
    const int8_t *items = (const int8_t *)t[0].data;
    const int32_t *geom = (const int32_t *)t[1].data;
    int32_t *out = (int32_t *)t[2].data;
    uint8_t d[11];
    AIE_MODE_POOL_DESC(d, geom);
    for (int64_t k = 0; k < t[0].shape[0]; k++)
        AIE_MODE_POOL(items + k * RT_PL_BYTES, out + k * RT_PL_OUT, d);
    return AIEK_OK;
}

#endif /* AIEDNN_RT_POOL_H */
