/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_red.h -- redblk:<aie pool block>: reduction of an 8-bit tensor over
 * its trailing axes on the GEMM ELF's pool mode (aie_modes_pool.h, 0xB0).
 *
 * src viewed as [outer][R]; each segment of R is laid out as a KH x KW window, Wo windows
 * per item; a segment longer than one item is cut into 4 KB chunks combined on the host.
 * The core returns exact int32 max / min / sum / sum|x| / sum x*x / max|x|; the host finishes
 * (mean, eps, p-th root) as ref: does. Accepted: sum, mean, max, min, amax, lp norms p = 1, 2.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aie_modes_pool.h,
 * aiednn_rt_pool.h (the block: is_pool_block).
 ******************************************************************************/
#ifndef AIEDNN_RT_RED_H
#define AIEDNN_RT_RED_H

typedef struct {
    int64_t outer, R;           /* segments, elements per segment */
    int64_t kh, kw, wo, chunks; /* window rows / cols, windows per item, items per segment (R > 4096) */
    int op, sgn;
} red_plan;

static int red_plan_make(const rt_prob *p, red_plan *q, char *why, int cap) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    if (s->data_type != aiednn_s8 && s->data_type != aiednn_u8) {
        snprintf(why, cap, "source must be s8 or u8");
        return 0;
    }
    const int lp = p->alg == aiednn_reduction_norm_lp_sum || p->alg == aiednn_reduction_norm_lp_power_p_sum;
    switch (p->alg) {
    case aiednn_reduction_sum:
    case aiednn_reduction_mean:
        q->op = AIE_MODE_POOL_SUM;
        break;
    case aiednn_reduction_max:
        q->op = AIE_MODE_POOL_MAX;
        break;
    case aiednn_reduction_min:
        q->op = AIE_MODE_POOL_MIN;
        break;
    case aiednn_reduction_amax:
        q->op = AIE_MODE_POOL_MAXABS;
        break;
    default:
        if (!lp || (p->pnorm != 1.0f && p->pnorm != 2.0f)) {
            snprintf(why, cap, "sum / mean / max / min / amax, lp norms with p = 1 or 2");
            return 0;
        }
        q->op = p->pnorm == 1.0f ? AIE_MODE_POOL_SUMABS : AIE_MODE_POOL_SUMSQ;
    }
    q->sgn = s->data_type == aiednn_s8;
    /* reduced dims (d == 1 < s) must all follow every kept dim of extent > 1 */
    int seen_red = 0;
    q->outer = 1, q->R = 1;
    for (int i = 0; i < s->ndims; i++) {
        if (s->dims[i] == 1)
            continue;
        if (d->dims[i] == 1) {
            seen_red = 1;
            q->R *= s->dims[i];
        } else if (seen_red) {
            snprintf(why, cap, "the reduced axes must be trailing");
            return 0;
        } else {
            q->outer *= s->dims[i];
        }
    }
    if (q->R <= AIE_MODE_IN) {
        q->kw = 1;
        while (q->kw * q->kw < q->R)
            q->kw++;
        q->kh = (q->R + q->kw - 1) / q->kw;
        q->wo = AIE_MODE_IN / (q->kh * q->kw);
        q->wo = q->wo > AIE_MODE_OUT ? AIE_MODE_OUT : q->wo;
        q->chunks = 1;
    } else {
        q->kw = 64, q->kh = 64, q->wo = 1;
        q->chunks = (q->R + AIE_MODE_IN - 1) / AIE_MODE_IN;
    }
    return 1;
}

static int acc_redblk(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    (void)bind;
    red_plan q;
    return red_plan_make(p, &q, why, cap);
}

static void *g_rdi, *g_rdo, *g_rdv;
static size_t g_rdi_cap, g_rdo_cap, g_rdv_cap;

/* the exact window value of every (segment, chunk), in segment-major order */
static int red_values(const rt_impl *self, const red_plan *q, const uint8_t *x, int64_t *val) {
    const int64_t NI = self->gen->args[0].dims[0];
    uint8_t *items = (uint8_t *)scratch(&g_rdi, &g_rdi_cap, (size_t)(NI * AIE_MODE_IN));
    int32_t *out = (int32_t *)scratch(&g_rdo, &g_rdo_cap, (size_t)(NI * AIE_MODE_OUT) * sizeof(int32_t));
    if (!items || !out)
        return AIEK_ERR_ALLOC;
    const uint8_t pad = q->op == AIE_MODE_POOL_MAX   ? (q->sgn ? 0x80 : 0x00)
                        : q->op == AIE_MODE_POOL_MIN ? (q->sgn ? 0x7f : 0xff)
                                                     : 0;
    const int64_t clen = q->chunks > 1 ? AIE_MODE_IN : q->R; /* elements per window */
    const int64_t wins = q->outer * q->chunks;
    int32_t geom[8] = {q->op, q->sgn,        (int32_t)q->kw, (int32_t)q->kh, (int32_t)q->kw, (int32_t)(q->wo * q->kw),
                       1,     (int32_t)q->wo};
    const int64_t shape_i[2] = {NI, AIE_MODE_IN}, shape_g[1] = {8}, shape_o[2] = {NI, AIE_MODE_OUT};
    for (int64_t w0 = 0; w0 < wins; w0 += NI * q->wo) {
        const uint64_t tp0 = rt_ticks();
        memset(items, pad, (size_t)(NI * AIE_MODE_IN));
        for (int64_t w = w0; w < wins && w < w0 + NI * q->wo; w++) {
            const int64_t seg = w / q->chunks, ch = w % q->chunks, k = (w - w0) / q->wo, j = (w - w0) % q->wo;
            const int64_t base = seg * q->R + ch * clen, len = q->R - ch * clen < clen ? q->R - ch * clen : clen;
            uint8_t *it = items + k * AIE_MODE_IN + j * q->kw;
            for (int64_t t = 0; t < len; t++)
                it[(t / q->kw) * q->wo * q->kw + t % q->kw] = x[base + t];
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        aiek_tensor t[3] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * AIE_MODE_IN)},
                            {geom, 1, AIEK_INT, 32, 1, shape_g, sizeof geom},
                            {out, 2, AIEK_INT, 32, 1, shape_o, (uint64_t)NI * AIE_MODE_OUT * sizeof(int32_t)}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        const int rc = self->gen->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        for (int64_t w = w0; w < wins && w < w0 + NI * q->wo; w++)
            val[w] = out[((w - w0) / q->wo) * AIE_MODE_OUT + (w - w0) % q->wo];
    }
    return AIEK_OK;
}

static int run_redblk(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    red_plan q;
    char why[160];
    if (!red_plan_make(p, &q, why, sizeof why))
        return AIEK_ERR_ARGS;
    int64_t *val = (int64_t *)scratch(&g_rdv, &g_rdv_cap, (size_t)(q.outer * q.chunks) * sizeof(int64_t));
    if (!val)
        return AIEK_ERR_ALLOC;
    const int rc = red_values(self, &q, (const uint8_t *)ptr[0], val);
    if (rc != AIEK_OK)
        return rc;
    const uint64_t te0 = rt_ticks();
    const int is_max = q.op == AIE_MODE_POOL_MAX || q.op == AIE_MODE_POOL_MAXABS, is_min = q.op == AIE_MODE_POOL_MIN;
    for (int64_t o = 0; o < q.outer; o++) {
        int64_t v = val[o * q.chunks];
        for (int64_t c = 1; c < q.chunks; c++) {
            const int64_t u = val[o * q.chunks + c];
            v = is_max ? (u > v ? u : v) : is_min ? (u < v ? u : v) : v + u;
        }
        double acc = (double)v; /* ref's double accumulation of these integers: exact */
        if (p->alg == aiednn_reduction_mean)
            acc /= (double)q.R;
        else if (p->alg == aiednn_reduction_norm_lp_power_p_sum)
            acc += p->eps;
        else if (p->alg == aiednn_reduction_norm_lp_sum)
            acc = pow(acc + p->eps, 1.0 / p->pnorm);
        rt_st(p->md[1].data_type, ptr[1], o, acc);
    }
    g_phase[RT_PH_EPI] += rt_ticks() - te0;
    return AIEK_OK;
}

#endif /* AIEDNN_RT_RED_H */
