/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_fpool.h -- f32 windowed pooling on the AIE (impl "fpool:", the GEMM ELF's FPOOL mode,
 * aie_modes_fpool.h): 2-D [N, C, H, W] and 3-D [N, C, D, H, W], max and average (padding included or
 * excluded). One item per (plane, output depth, band of output rows): its KD input slices, padded on the
 * host (-FLT_MAX for max, 0 for a sum); the core takes the window max / f32 sum, the host divides. Max is
 * exact; an average is within f32 rounding of ref:'s double sum (RT_FPOOL_RTOL, tests).
 *
 * Max pooling backward (impl "fpool:" for RT_BWD_DATA, max only): the core finds every output window's
 * first strict maximum (op ARGMAX, the same items from SRC); the host routes dy through those taps in
 * double, in ref:'s order (outputs row-major, then rounded once), so the gradient is bit-identical to
 * ref:. A tap in the padding (only possible when every valid value is -FLT_MAX / -inf) is redone with
 * ref:'s rule. Average backward has no search to move: it stays ref:.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_itemsn.h, aie_modes_fpool.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_FPOOL_H
#define AIEDNN_RT_FPOOL_H

#define RT_FPOOL_RTOL 1e-6

static int is_fpool_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_pooling && g->nargs == 3 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[0].dims[1] == 4096 && g->args[2].data_type == aiednn_f32 && g->args[2].ndims == 2 &&
           g->args[2].dims[1] == AIE_FPOOL_OUT;
}

typedef struct {
    int64_t NC, D, H, W, Do, Ho, Wo, KD, KH, KW, sd, sh, sw, pd, pt, pl;
    int64_t wp, rin, rout, bands;
    int op;
} fpool_plan;
static int fpool_plan_fit(fpool_plan *q, char *why, int cap);

static int fpool_plan_make(const rt_prob *p, fpool_plan *q, char *why, int cap) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const aiednn_pool_params_t *pp = &p->pool;
    const int nd = s->ndims;
    if (p->prop != RT_FWD_INF || s->data_type != aiednn_f32 || d->data_type != aiednn_f32 || (nd != 4 && nd != 5)) {
        snprintf(why, cap, "f32 2-D / 3-D inference pooling only");
        return 0;
    }
    q->NC = s->dims[0] * s->dims[1];
    q->D = nd == 5 ? s->dims[2] : 1, q->Do = nd == 5 ? d->dims[2] : 1;
    q->H = s->dims[nd - 2], q->W = s->dims[nd - 1], q->Ho = d->dims[nd - 2], q->Wo = d->dims[nd - 1];
    q->KD = nd == 5 ? p->dpth[RT_D_K] : 1, q->sd = nd == 5 ? p->dpth[RT_D_S] : 1,
    q->pd = nd == 5 ? p->dpth[RT_D_PL] : 0;
    q->KH = pp->kernel[0], q->KW = pp->kernel[1], q->sh = pp->strides[0], q->sw = pp->strides[1];
    q->pt = pp->padding_l[0], q->pl = pp->padding_l[1];
    q->op = p->alg == aiednn_pooling_max ? 0 : 1;
    if (pp->dilates[0] || pp->dilates[1] || (nd == 5 && p->dpth[RT_D_DIL]) || (q->sw != 1 && q->sw != 2) || q->sh < 1 ||
        q->sh > 255 || q->KD > 255 || q->KH > 255 || q->KW > 255) {
        snprintf(why, cap, "needs undilated windows, W stride 1 or 2, kernels and H stride <= 255");
        return 0;
    }
    if (!pool_dim_ok(q->D, q->Do, q->KD, q->sd, q->pd) || !pool_dim_ok(q->H, q->Ho, q->KH, q->sh, q->pt) ||
        !pool_dim_ok(q->W, q->Wo, q->KW, q->sw, q->pl)) {
        snprintf(why, cap, "a window lies entirely in the padding");
        return 0;
    }
    return fpool_plan_fit(q, why, cap);
}

/* one output row's window slices must fit an item: pick the most output rows per item */
static int fpool_plan_fit(fpool_plan *q, char *why, int cap) {
    q->wp = (q->Wo - 1) * q->sw + q->KW;
    for (int64_t r = q->Ho < AIE_FPOOL_OUT ? q->Ho : AIE_FPOOL_OUT; r >= 1; r--) {
        const int64_t rin = (r - 1) * q->sh + q->KH;
        if (r * q->Wo <= AIE_FPOOL_OUT && q->KD * rin * q->wp <= AIE_FPOOL_IN && q->wp < 65536) {
            q->rout = r, q->rin = rin, q->bands = (q->Ho + r - 1) / r;
            return 1;
        }
    }
    snprintf(why, cap, "one output row's window slices do not fit an item (KD x KH x Wp > 1024 floats)");
    return 0;
}

/* max pooling backward: src [N, C, (D,) H, W] and the N-D params (p->pnd) -> the forward's item plan, op ARGMAX */
static int fpool_plan_bwd(const rt_prob *p, fpool_plan *q, char *why, int cap) {
    const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const aiednn_pool_nd_params_t *pp = &p->pnd;
    if (p->alg != aiednn_pooling_max || !x || !dy || !dx || x->data_type != aiednn_f32 || dy->data_type != aiednn_f32 ||
        dx->data_type != aiednn_f32 || (x->ndims != 4 && x->ndims != 5)) {
        snprintf(why, cap, "f32 2-D / 3-D max pooling backward only (average backward: ref:)");
        return 0;
    }
    const int nd = x->ndims, sp = nd - 2, o = nd == 5 ? 1 : 0; /* pnd index of H */
    for (int i = 0; i < sp; i++)
        if (pp->dilates[i]) {
            snprintf(why, cap, "dilated windows");
            return 0;
        }
    q->NC = x->dims[0] * x->dims[1];
    q->D = nd == 5 ? x->dims[2] : 1, q->Do = nd == 5 ? dy->dims[2] : 1;
    q->H = x->dims[nd - 2], q->W = x->dims[nd - 1], q->Ho = dy->dims[nd - 2], q->Wo = dy->dims[nd - 1];
    q->KD = nd == 5 ? pp->kernel[0] : 1, q->sd = nd == 5 ? pp->strides[0] : 1, q->pd = nd == 5 ? pp->padding_l[0] : 0;
    q->KH = pp->kernel[o], q->KW = pp->kernel[o + 1], q->sh = pp->strides[o], q->sw = pp->strides[o + 1];
    q->pt = pp->padding_l[o], q->pl = pp->padding_l[o + 1];
    q->op = 2;
    if ((q->sw != 1 && q->sw != 2) || q->sh > 255 || q->KD > 255 || q->KH > 255 || q->KW > 255) {
        snprintf(why, cap, "needs W stride 1 or 2, kernels and H stride <= 255");
        return 0;
    }
    if (!pool_dim_ok(q->D, q->Do, q->KD, q->sd, q->pd) || !pool_dim_ok(q->H, q->Ho, q->KH, q->sh, q->pt) ||
        !pool_dim_ok(q->W, q->Wo, q->KW, q->sw, q->pl)) {
        snprintf(why, cap, "a window lies entirely in the padding");
        return 0;
    }
    return fpool_plan_fit(q, why, cap);
}

#ifndef RT_POOL_SMALL
#define RT_POOL_SMALL 65536 /* source elements below which prefer_fastest keeps a pooling on ref: */
#endif

static int acc_fpool(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    fpool_plan q;
    if (p->prop == RT_BWD_DATA)
        return fpool_plan_bwd(p, &q, why, cap);
    if (!fpool_plan_make(p, &q, why, cap))
        return 0;
    /* aiednn_prefer_fastest only: a small pooling is a few us on the host, a launch ~1 ms (resnet18's global
     * 512x7x7 average: 2.26 ms on fvec: vs ~0.05 on ref:, galerina13) */
    if (rt_cost_refusals() && md_numel(&p->md[0]) < RT_POOL_SMALL) {
        snprintf(why, cap, "ref: measured faster on a small pooling (aiednn_prefer_fastest)");
        return 0;
    }
    return 1;
}

/* item = (nc, od, band): KD slices of rin padded rows of wp floats */
static void fpool_pack_item(const fpool_plan *q, const float *x, int64_t it, uint8_t *raw) {
    const int64_t band = it % q->bands, od = it / q->bands % q->Do, nc = it / q->bands / q->Do;
    float *f = (float *)raw;
    const float pad = q->op == 1 ? 0.f : AIE_FPOOL_NEG; /* a sum pads 0; max and argmax -FLT_MAX */
    const int64_t oh0 = band * q->rout;
    for (int64_t i = 0; i < AIE_FPOOL_IN; i++)
        f[i] = pad;
    for (int64_t s = 0; s < q->KD; s++) {
        const int64_t id = od * q->sd - q->pd + s;
        if (id < 0 || id >= q->D)
            continue;
        for (int64_t r = 0; r < q->rin; r++) {
            const int64_t ih = oh0 * q->sh - q->pt + r;
            if (ih < 0 || ih >= q->H)
                continue;
            const float *src = x + ((nc * q->D + id) * q->H + ih) * q->W;
            float *dst = f + (s * q->rin + r) * q->wp;
            const int64_t c0 = q->pl > 0 ? q->pl : 0, c1 = q->pl + q->W < q->wp ? q->pl + q->W : q->wp;
            if (c1 > c0)
                memcpy(dst + c0, src + (c0 - q->pl), (size_t)(c1 - c0) * sizeof(float));
        }
    }
}

/* the item's max / sum -> dst, as ref:pooling finishes it */
static void fpool_finish_item(const rt_prob *p, const fpool_plan *q, int64_t it, const float *v, float *dst) {
    const int64_t band = it % q->bands, od = it / q->bands % q->Do, nc = it / q->bands / q->Do;
    const int64_t oh0 = band * q->rout, rows = q->Ho - oh0 < q->rout ? q->Ho - oh0 : q->rout;
    const int64_t vd = pool_valid(od, q->sd, q->pd, q->KD, q->D);
    for (int64_t r = 0; r < rows; r++)
        for (int64_t ow = 0; ow < q->Wo; ow++) {
            const int64_t oh = oh0 + r;
            double a = v[r * q->Wo + ow];
            if (p->alg == aiednn_pooling_avg_include_padding) {
                a /= (double)(q->KD * q->KH * q->KW);
            } else if (p->alg == aiednn_pooling_avg_exclude_padding) {
                const int64_t cnt =
                    vd * pool_valid(oh, q->sh, q->pt, q->KH, q->H) * pool_valid(ow, q->sw, q->pl, q->KW, q->W);
                a = cnt ? a / (double)cnt : 0.0;
            }
            dst[((nc * q->Do + od) * q->Ho + oh) * q->Wo + ow] = (float)a;
        }
}

typedef struct {
    const rt_prob *p;
    const fpool_plan *q;
    const float *x;
    float *dst;
} fpool_ctx;
static void fpool_itn_pack(void *c, int64_t it, uint8_t *raw) {
    const fpool_ctx *k = (const fpool_ctx *)c;
    fpool_pack_item(k->q, k->x, it, raw);
}
static void fpool_itn_put(void *c, int64_t it, const int32_t *res) {
    const fpool_ctx *k = (const fpool_ctx *)c;
    fpool_finish_item(k->p, k->q, it, (const float *)res, k->dst);
}

static void *g_fpi, *g_fpo;
static size_t g_fpi_cap, g_fpo_cap;

static int run_fpool_bwd(const rt_impl *self, aiednn_primitive *prim, void *const *ptr);

static int run_fpool(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    if (p->prop == RT_BWD_DATA)
        return run_fpool_bwd(self, prim, ptr);
    fpool_plan q;
    char why[8];
    fpool_plan_make(p, &q, why, sizeof why);
    const int32_t geom[8] = {q.op,          (int32_t)q.KD, (int32_t)q.KH,  (int32_t)q.KW, (int32_t)(q.sh | (q.sw << 8)),
                             (int32_t)q.Wo, (int32_t)q.wp, (int32_t)q.rout};
    const int64_t total = q.NC * q.Do * q.bands;
    fpool_ctx k = {p, &q, (const float *)ptr[0], (float *)ptr[1]};
    { /* many launches' worth of items in one (aiednn_rt_itemsn.h) */
        uint8_t d[12];
        AIE_FPOOL_DESC(d, geom);
        const int rc = rt_items_n(self, total, AIE_MODE_OP_FPOOL, d, 12, fpool_itn_pack, fpool_itn_put, &k);
        if (rc != RT_NOT_TAKEN)
            return rc;
    }
    const int64_t NI = self->gen->args[0].dims[0];
    uint8_t *items = (uint8_t *)scratch(&g_fpi, &g_fpi_cap, (size_t)NI * 4096);
    float *out = (float *)scratch(&g_fpo, &g_fpo_cap, (size_t)NI * AIE_FPOOL_OUT * sizeof(float));
    if (!items || !out)
        return AIEK_ERR_ALLOC;
    const int64_t shape_i[2] = {NI, 4096}, shape_g[1] = {8}, shape_o[2] = {NI, AIE_FPOOL_OUT};
    for (int64_t base = 0; base < total; base += NI) {
        const int64_t n = total - base < NI ? total - base : NI;
        const uint64_t tp = rt_ticks();
        for (int64_t i = 0; i < n; i++)
            fpool_pack_item(&q, k.x, base + i, items + i * 4096);
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp;
        aiek_tensor t[3] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * 4096)},
                            {(void *)geom, 1, AIEK_INT, 32, 1, shape_g, sizeof geom},
                            {out, 2, AIEK_FLOAT, 32, 1, shape_o, (uint64_t)(NI * AIE_FPOOL_OUT * sizeof(float))}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        const int rc = self->gen->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t t0 = rt_ticks();
        for (int64_t i = 0; i < n; i++)
            fpool_finish_item(p, &q, base + i, out + i * AIE_FPOOL_OUT, k.dst);
        g_phase[RT_PH_EPI] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

/* argmax items -> per-output tap index (int32), in output order */
typedef struct {
    const fpool_plan *q;
    const float *x;
    int32_t *tap;
} fpool_bwd_ctx;
static void fpool_bwd_put(void *c, int64_t it, const int32_t *res) {
    const fpool_bwd_ctx *k = (const fpool_bwd_ctx *)c;
    const fpool_plan *q = k->q;
    const int64_t band = it % q->bands, od = it / q->bands % q->Do, nc = it / q->bands / q->Do;
    const int64_t oh0 = band * q->rout, rows = q->Ho - oh0 < q->rout ? q->Ho - oh0 : q->rout;
    for (int64_t r = 0; r < rows; r++)
        memcpy(k->tap + ((nc * q->Do + od) * q->Ho + oh0 + r) * q->Wo, res + r * q->Wo,
               (size_t)q->Wo * sizeof(int32_t));
}
static void fpool_bwd_pack(void *c, int64_t it, uint8_t *raw) {
    const fpool_bwd_ctx *k = (const fpool_bwd_ctx *)c;
    fpool_pack_item(k->q, k->x, it, raw);
}

static void *g_fpb_tap, *g_fpb_acc;
static size_t g_fpb_tap_cap, g_fpb_acc_cap;

static int run_fpool_bwd(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    fpool_plan q;
    char why[8];
    fpool_plan_bwd(p, &q, why, sizeof why);
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC),
                *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const int64_t ny = q.Do * q.Ho * q.Wo, nx = q.D * q.H * q.W;
    int32_t *tap = (int32_t *)scratch(&g_fpb_tap, &g_fpb_tap_cap, (size_t)(q.NC * ny) * sizeof(int32_t));
    double *acc = (double *)scratch(&g_fpb_acc, &g_fpb_acc_cap, (size_t)(q.NC * nx) * sizeof(double));
    if (!tap || !acc)
        return AIEK_ERR_ALLOC;
    const int32_t geom[8] = {2,
                             (int32_t)q.KD,
                             (int32_t)q.KH,
                             (int32_t)q.KW,
                             (int32_t)(q.sh | (q.sw << 8)),
                             (int32_t)q.Wo,
                             (int32_t)q.wp,
                             (int32_t)q.rout};
    const int64_t total = q.NC * q.Do * q.bands;
    fpool_bwd_ctx k = {&q, x, tap};
    uint8_t d[12];
    AIE_FPOOL_DESC(d, geom);
    int rc = rt_items_n(self, total, AIE_MODE_OP_FPOOL, d, 12, fpool_bwd_pack, fpool_bwd_put, &k);
    if (rc == RT_NOT_TAKEN) { /* one launch of NI items at a time */
        const int64_t NI = self->gen->args[0].dims[0];
        uint8_t *items = (uint8_t *)scratch(&g_fpi, &g_fpi_cap, (size_t)NI * 4096);
        int32_t *out = (int32_t *)scratch(&g_fpo, &g_fpo_cap, (size_t)NI * AIE_FPOOL_OUT * sizeof(float));
        if (!items || !out)
            return AIEK_ERR_ALLOC;
        const int64_t shape_i[2] = {NI, 4096}, shape_g[1] = {8}, shape_o[2] = {NI, AIE_FPOOL_OUT};
        rc = AIEK_OK;
        for (int64_t base = 0; base < total && rc == AIEK_OK; base += NI) {
            const int64_t n = total - base < NI ? total - base : NI;
            for (int64_t i = 0; i < n; i++)
                fpool_pack_item(&q, x, base + i, items + i * 4096);
            aiek_tensor t[3] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * 4096)},
                                {(void *)geom, 1, AIEK_INT, 32, 1, shape_g, sizeof geom},
                                {out, 2, AIEK_FLOAT, 32, 1, shape_o, (uint64_t)(NI * AIE_FPOOL_OUT * sizeof(float))}};
            ((rt_impl *)self)->launches++;
            const uint64_t te = rt_ticks();
            rc = self->gen->entry(t, 3);
            ((rt_impl *)self)->t_entry += rt_ticks() - te;
            for (int64_t i = 0; rc == AIEK_OK && i < n; i++)
                fpool_bwd_put(&k, base + i, out + i * AIE_FPOOL_OUT);
        }
    }
    if (rc != AIEK_OK)
        return rc;
    /* route dy, as ref: does: outputs in order, double accumulators, one rounding at the end */
    const uint64_t t0 = rt_ticks();
    memset(acc, 0, (size_t)(q.NC * nx) * sizeof(double));
    const int64_t khw = q.KH * q.KW;
    for (int64_t nc = 0; nc < q.NC; nc++)
        for (int64_t a = 0; a < ny; a++) {
            const int64_t od = a / (q.Ho * q.Wo), oh = a / q.Wo % q.Ho, ow = a % q.Wo;
            const int32_t t = tap[nc * ny + a];
            int64_t b0 = od * q.sd - q.pd + t / khw, b1 = oh * q.sh - q.pt + t / q.KW % q.KH,
                    b2 = ow * q.sw - q.pl + t % q.KW;
            if (b0 < 0 || b0 >= q.D || b1 < 0 || b1 >= q.H || b2 < 0 || b2 >= q.W) { /* padding won: ref:'s rule */
                double bv = -INFINITY;
                int64_t best = -1;
                for (int64_t kk = 0; kk < q.KD * khw; kk++) {
                    const int64_t c0 = od * q.sd - q.pd + kk / khw, c1 = oh * q.sh - q.pt + kk / q.KW % q.KH;
                    const int64_t c2 = ow * q.sw - q.pl + kk % q.KW;
                    if (c0 < 0 || c0 >= q.D || c1 < 0 || c1 >= q.H || c2 < 0 || c2 >= q.W)
                        continue;
                    const double v = x[nc * nx + (c0 * q.H + c1) * q.W + c2];
                    if (best < 0 || v > bv)
                        bv = v, best = (c0 * q.H + c1) * q.W + c2;
                }
                if (best < 0)
                    continue;
                b0 = best / (q.H * q.W), b1 = best / q.W % q.H, b2 = best % q.W;
            }
            acc[nc * nx + (b0 * q.H + b1) * q.W + b2] += (double)dy[nc * ny + a];
        }
    for (int64_t i = 0; i < q.NC * nx; i++)
        dx[i] = (float)acc[i];
    g_phase[RT_PH_EPI] += rt_ticks() - t0;
    return AIEK_OK;
}

#endif /* AIEDNN_RT_FPOOL_H */
