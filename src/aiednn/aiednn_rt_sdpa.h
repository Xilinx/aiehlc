/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_sdpa.h -- sdpablk:<aie sdpa block>: scaled dot-product attention on the GEMM ELF's SDPA
 * mode (aie_modes_sdpa.h, opcode 0x5D), one (batch, head, query block) per core.
 *
 * Q, K, V quantized per tensor on the host (step = max|x| / 127; s8 taken as is). The core computes
 * exact int32 scores, softmax from a 2^x table and the weighted V; the host dequantizes as ref:sdpa.
 * Not bit-exact: tolerance RT_SDPA_TOL (codes).
 *
 * Accepted: f32 or s8 Q / K / V (f32 only into an int8 destination), no mask tensor (causal ok),
 * Skv <= AIE_SDPA_MAX_KV, D and Dv <= 255, one query row's K and V plus a query fitting an item.
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_itemsn.h, aie_modes_sdpa.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_SDPA_H
#define AIEDNN_RT_SDPA_H

#define RT_SDPA_TOL 4 /* int8 destination codes against ref:sdpa (tests) */

static int is_sdpa_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_sdpa && g->nargs == 3 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[0].dims[1] == RT_ITEM_BYTES && g->args[1].data_type == aiednn_s32 &&
           g->args[2].data_type == aiednn_s32 && g->args[2].ndims == 2 && g->args[2].dims[1] == RT_ITEM_OUT;
}

typedef struct {
    const rt_prob *p;
    void *const *ptr;
    int64_t B, H, Hk, Sq, Skv, D, Dv, qb, nqb; /* qb query rows per item, nqb blocks per head; Hk K / V heads */
    int qs_bshd, ds_bshd;
    double step[3]; /* Q, K, V quantization steps */
} sdpa_plan;

#define SDPA_IX(bshd, H, b, h, s, e, S, W)                                                                             \
    ((bshd) ? ((((b) * (S) + (s)) * (H) + (h)) * (W) + (e)) : (((((b) * (H) + (h)) * (S) + (s)) * (W) + (e))))

static int sdpa_plan_make(const rt_prob *p, sdpa_plan *s, char *why, int cap) {
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    for (int a = 0; a < 3; a++)
        if (p->md[a].data_type != aiednn_f32 && p->md[a].data_type != aiednn_s8) {
            snprintf(why, cap, "Q / K / V must be f32 or s8");
            return 0;
        }
    const int s8_in = q->data_type == aiednn_s8 && k->data_type == aiednn_s8 && v->data_type == aiednn_s8;
    if (!s8_in && !(p->has_attr && dt_is_int(d->data_type))) { // an f32 result keeps f32 precision: ref:
        snprintf(why, cap, "f32 Q / K / V need a quantized (int8 + scale) destination");
        return 0;
    }
    if (rt_md(p, AIEDNN_ARG_ATTN_MASK)) {
        snprintf(why, cap, "a mask tensor");
        return 0;
    }
    s->p = p, s->B = d->dims[0], s->H = d->dims[1], s->Sq = q->dims[2], s->Skv = k->dims[2], s->D = q->dims[3];
    s->Hk = k->dims[1]; /* GQA / MQA: K, V heads (dividing H) */
    s->Dv = v->dims[3];
    s->qs_bshd = (p->flags & AIEDNN_SDPA_QKV_BSHD) != 0, s->ds_bshd = (p->flags & AIEDNN_SDPA_DST_BSHD) != 0;
    const int64_t kv = s->D * ((s->Skv + 15) & ~15) + s->Skv * s->Dv; /* K^T padded to 16 keys, V */
    if (s->Skv < 1 || s->Skv > AIE_SDPA_MAX_KV || s->D > 255 || s->Dv > 255 || s->Dv > RT_ITEM_OUT ||
        kv + s->D > AIE_SDPA_COFF) {
        snprintf(why, cap, "Skv <= %d, D / Dv <= 255 and K + V + a query row in 4 KB", AIE_SDPA_MAX_KV);
        return 0;
    }
    int64_t qb = (AIE_SDPA_COFF - kv) / s->D;
    qb = qb < RT_ITEM_OUT / s->Dv ? qb : RT_ITEM_OUT / s->Dv;
    s->qb = qb < s->Sq ? qb : s->Sq;
    s->nqb = (s->Sq + s->qb - 1) / s->qb;
    return 1;
}

/* per-tensor step: max|x| / 127 for f32 (1 if all zero), 1 for s8 */
static double sdpa_step(const aiednn_memory_desc_t *m, const void *x) {
    if (m->data_type == aiednn_s8)
        return 1.0;
    const float *f = (const float *)x;
    float mx = 0;
    for (int64_t i = 0, n = md_numel(m); i < n; i++)
        mx = fabsf(f[i]) > mx ? fabsf(f[i]) : mx;
    return mx > 0 ? (double)mx / 127.0 : 1.0;
}

static inline int8_t sdpa_q8(aiednn_data_type_t dt, const void *x, int64_t i, double step) {
    if (dt == aiednn_s8)
        return ((const int8_t *)x)[i];
    const double v = nearbyint((double)((const float *)x)[i] / step);
    return (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
}

/* item it = ((b * H + h) * nqb + qblock): Q rows q0 .. q0+n, the head's K and V */
static void sdpa_pack(void *ctx, int64_t it, uint8_t *raw) {
    const sdpa_plan *s = (const sdpa_plan *)ctx;
    const rt_prob *p = s->p;
    const int64_t bh = it / s->nqb, b = bh / s->H, h = bh % s->H, q0 = (it % s->nqb) * s->qb;
    const int64_t n = s->Sq - q0 < s->qb ? s->Sq - q0 : s->qb;
    const int64_t skp = (s->Skv + 15) & ~15;
    int8_t *qd = (int8_t *)raw, *kd = qd + n * s->D, *vd = kd + s->D * skp; /* kd: K^T [D][skp] */
    const int32_t coff = (int32_t)(q0 + s->Skv - s->Sq); // causal: global query q0 + i sees keys <= q0 + i + Skv - Sq
    for (int b_ = 0; b_ < 4; b_++)
        raw[AIE_SDPA_COFF + b_] = (uint8_t)((uint32_t)coff >> (8 * b_));
    for (int64_t i = 0; i < n; i++)
        for (int64_t e = 0; e < s->D; e++)
            qd[i * s->D + e] = sdpa_q8(p->md[0].data_type, s->ptr[0],
                                       SDPA_IX(s->qs_bshd, s->H, b, h, q0 + i, e, s->Sq, s->D), s->step[0]);
    memset(kd, 0, (size_t)(s->D * skp));
    for (int64_t j = 0; j < s->Skv; j++) {
        for (int64_t e = 0; e < s->D; e++)
            kd[e * skp + j] =
                sdpa_q8(p->md[1].data_type, s->ptr[1],
                        SDPA_IX(s->qs_bshd, s->Hk, b, h / (s->H / s->Hk), j, e, s->Skv, s->D), s->step[1]);
        for (int64_t e = 0; e < s->Dv; e++)
            vd[j * s->Dv + e] =
                sdpa_q8(p->md[2].data_type, s->ptr[2],
                        SDPA_IX(s->qs_bshd, s->Hk, b, h / (s->H / s->Hk), j, e, s->Skv, s->Dv), s->step[2]);
    }
}

static void sdpa_put(void *ctx, int64_t it, const int32_t *res) {
    const sdpa_plan *s = (const sdpa_plan *)ctx;
    const rt_prob *p = s->p;
    const aiednn_memory_desc_t *d = &p->md[3];
    const int64_t bh = it / s->nqb, b = bh / s->H, h = bh % s->H, q0 = (it % s->nqb) * s->qb;
    const int64_t n = s->Sq - q0 < s->qb ? s->Sq - q0 : s->qb;
    const double vs = s->step[2] / 65536.0;
    for (int64_t i = 0; i < n; i++)
        for (int64_t e = 0; e < s->Dv; e++) {
            const double o = (double)res[i * s->Dv + e] * vs; // as ref:sdpa from here on
            const int64_t at = SDPA_IX(s->ds_bshd, s->H, b, h, q0 + i, e, s->Sq, s->Dv);
            if (p->has_attr && dt_is_int(d->data_type))
                rt_st(d->data_type, s->ptr[3], at, nearbyint((double)(float)o / p->dst_scale) + p->dst_zp);
            else
                rt_st(d->data_type, s->ptr[3], at, o);
        }
}

static int acc_sdpablk(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    if (rt_env_on("AIEDNN_NO_SDPABLK")) { // keep ref:sdpa (bit-exact comparisons of graph rewrites)
        snprintf(why, cap, "AIEDNN_NO_SDPABLK");
        return 0;
    }
    if (p->flags & RT_SDPA_VAR) {
        snprintf(why, cap, "variable lengths / paged K V: gemm+sdpavar");
        return 0;
    }
    sdpa_plan s;
    return sdpa_plan_make(p, &s, why, cap);
}

static void *g_sdi, *g_sdo;
static size_t g_sdi_cap, g_sdo_cap;

static int run_sdpablk(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    sdpa_plan s;
    char why[8];
    sdpa_plan_make(p, &s, why, sizeof why);
    s.ptr = ptr;
    const uint64_t tq = rt_ticks();
    for (int a = 0; a < 3; a++)
        s.step[a] = sdpa_step(&p->md[a], ptr[a]);
    g_phase[RT_PH_DWPACK] += rt_ticks() - tq;
    /* exponent in Q16 log2 units per score unit: scale * Q step * K step * log2(e) * 65536 = M / 2^SH */
    const double c = (double)p->scale * s.step[0] * s.step[1] * 1.4426950408889634 * 65536.0;
    int sh = 0;
    while (sh < 62 && c * ldexp(1.0, sh) < 1073741824.0)
        sh++;
    const double m = nearbyint(c * ldexp(1.0, sh));
    if (!(c > 0) || m >= 2147483648.0)
        return AIEK_ERR_PARAM; // a score step the int32 multiplier cannot hold
    const int32_t geom[8] = {(int32_t)s.qb,     (int32_t)s.Skv, (int32_t)s.D, (int32_t)s.Dv,
                             p->causal ? 1 : 0, (int32_t)m,     sh,           0};
    const int64_t total = s.B * s.H * s.nqb;
    uint8_t d[12];
    int32_t gq[8];
    memcpy(gq, geom, sizeof gq);
    if (s.Sq % s.qb == 0 || total == s.B * s.H) { /* every item has qb rows: one descriptor fits all */
        AIE_MODE_SDPA_DESC(d, gq);
        const int rc = rt_items_n(self, total, AIE_MODE_OP_SDPA, d, 12, sdpa_pack, sdpa_put, &s);
        if (rc != RT_NOT_TAKEN)
            return rc;
    }
    const int64_t NI = self->gen->args[0].dims[0]; // items per launch
    int8_t *items = (int8_t *)scratch(&g_sdi, &g_sdi_cap, NI * RT_ITEM_BYTES);
    int32_t *out = (int32_t *)scratch(&g_sdo, &g_sdo_cap, NI * RT_ITEM_OUT * sizeof(int32_t));
    if (!items || !out)
        return AIEK_ERR_ALLOC;
    const int64_t shape_i[2] = {NI, RT_ITEM_BYTES}, shape_g[1] = {8}, shape_o[2] = {NI, RT_ITEM_OUT};
    for (int64_t base = 0; base < total;) {
        /* one launch shares a descriptor: items with the same query-row count */
        const int64_t rows0 = s.Sq - (base % s.nqb) * s.qb < s.qb ? s.Sq - (base % s.nqb) * s.qb : s.qb;
        int64_t n_items = 0;
        const uint64_t tp0 = rt_ticks();
        while (base + n_items < total && n_items < NI) {
            const int64_t it = base + n_items,
                          rows = s.Sq - (it % s.nqb) * s.qb < s.qb ? s.Sq - (it % s.nqb) * s.qb : s.qb;
            if (rows != rows0)
                break;
            sdpa_pack(&s, it, (uint8_t *)items + n_items * RT_ITEM_BYTES);
            n_items++;
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        gq[0] = (int32_t)rows0;
        aiek_tensor t[3] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * RT_ITEM_BYTES)},
                            {gq, 1, AIEK_INT, 32, 1, shape_g, sizeof gq},
                            {out, 2, AIEK_INT, 32, 1, shape_o, (uint64_t)NI * RT_ITEM_OUT * sizeof(int32_t)}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        const int rc = self->gen->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t te0 = rt_ticks();
        for (int64_t k = 0; k < n_items; k++)
            sdpa_put(&s, base + k, out + k * RT_ITEM_OUT);
        g_phase[RT_PH_EPI] += rt_ticks() - te0;
        base += n_items;
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_SDPA_H */
