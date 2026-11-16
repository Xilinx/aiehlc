/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_steps.h -- steps:<aie lut block>: f32 / s32 -> s8 / u8 per-element
 * ops on the GEMM ELF's step-table mode (aie_modes_lut.h, opcode 0x75).
 *
 * Accepted (source f32 or s32, destination s8 or u8): quantizing reorders, batch norm, PReLU, binary
 * with a broadcast operand and the arithmetic eltwise algorithms -- per channel monotone on either side of
 * zero, so the 8-bit result is a step function of the source's order-preserving key.
 *
 * The tables are ref: itself: the host bisects every step over the key space of each half line (32 rounds),
 * so the AIE's binary search reproduces ref bit for bit, NaN included. Tables are cached per problem +
 * operand contents. Kernel block: the lut: block, geom[3] flags steps / kind / stride; items hold
 * n <= 1024 four-byte values.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_ops.h,
 * aie_modes_lut.h, aiednn_rt_lut.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_STEPS_H
#define AIEDNN_RT_STEPS_H

#define RT_ST_MAXN 511 /* steps per channel: 255 per half line + the one at zero */

static int steps_alg_ok(const rt_prob *p, const lut_plan *q) {
    switch (p->kind) {
    case aiednn_reorder:
    case aiednn_batch_normalization:
    case aiednn_prelu:
        return 1;
    case aiednn_binary:
        return p->alg >= aiednn_binary_add && p->alg <= aiednn_binary_le &&
               p->alg != aiednn_binary_pow; /* monotone per half line */
    case aiednn_eltwise:
        switch (p->alg) {
        case aiednn_eltwise_relu:
        case aiednn_eltwise_clip:
        case aiednn_eltwise_linear:
        case aiednn_eltwise_abs:
        case aiednn_eltwise_square:
        case aiednn_eltwise_round:
        case aiednn_eltwise_floor:
        case aiednn_eltwise_ceil:
        case aiednn_eltwise_neg:
        case aiednn_eltwise_hardsigmoid:
            return 1;
        }
        return 0;
    default:
        (void)q;
        return 0;
    }
}

static int steps_plan_make(const rt_prob *p, lut_plan *q, char *why, int cap) {
    if (!lut_plan_shape(p, q, why, cap))
        return 0;
    const aiednn_data_type_t mt = p->md[q->main_slot].data_type, dt = p->md[q->dst_slot].data_type;
    if ((mt != aiednn_f32 && mt != aiednn_s32) || (dt != aiednn_s8 && dt != aiednn_u8)) {
        snprintf(why, cap, "f32 / s32 source to s8 / u8");
        return 0;
    }
    if (!steps_alg_ok(p, q)) {
        snprintf(why, cap, "not an arithmetic (monotone per half line) algorithm");
        return 0;
    }
    q->slots = AIE_LUT_TBYTES / ((AIE_STEPS_BYTES(RT_ST_MAXN) + 7) & ~7); /* worst case; the run may fit more */
    if (q->C > q->slots && q->inner * q->slots < AIE_LUT_ITEM / 4 / 4) {
        snprintf(why, cap, "%lld channels of %lld elements: too fine for the step tables", (long long)q->C,
                 (long long)q->inner);
        return 0;
    }
    return 1;
}

static int acc_steps(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    (void)bind;
    lut_plan q;
    if ((p->kind == aiednn_binary || p->kind == aiednn_eltwise) && p->has_attr) {
        snprintf(why, cap, "the quantized binary / eltwise (attr): fvec: / lut:");
        return 0;
    }
    if (rt_cost_refusals() && reorder_ref_is_fast(p)) { /* f32 -> s8 quantize: board 3.71 (AIE) vs 3.30 ms (ref) */
        snprintf(why, cap, "ref:reorder's contiguous loop is faster");
        return 0;
    }
    /* f32 batch norm: ref's per-channel loop is faster */
    if (rt_cost_refusals() && p->kind == aiednn_batch_normalization && p->md[0].data_type == aiednn_f32) {
        snprintf(why, cap, "ref:batch_normalization's per-channel f32 loop is faster");
        return 0;
    }
    return steps_plan_make(p, &q, why, cap);
}

/* key -> the source value's 4 bytes */
static inline uint32_t steps_word(int f32, int64_t key) {
    const int32_t k = (int32_t)key;
    return f32 ? (uint32_t)(k < 0 ? k ^ 0x7fffffff : k) : (uint32_t)k;
}

typedef struct {
    int64_t a, b; /* key bracket, codes differ at its ends */
    int16_t ca, cb;
} st_br;

typedef struct {
    int64_t t;
    int16_t code;
} st_step;

static void *g_stx, *g_sty, *g_stb, *g_stn, *g_sts;
static size_t g_stx_cap, g_sty_cap, g_stb_cap, g_stn_cap, g_sts_cap;

/* codes of P probe words per channel (xs channel-major) -> ys codes */
static int steps_eval(const rt_prob *p, const lut_plan *q, void *const *ptr, int64_t P, const uint32_t *xs,
                      int16_t *codes) {
    uint8_t *ys = (uint8_t *)scratch(&g_sty, &g_sty_cap, (size_t)(q->C * P));
    if (!ys)
        return AIEK_ERR_ALLOC;
    const int rc = lut_synth(p, q, ptr, P, xs, ys);
    const int sgn = p->md[q->dst_slot].data_type == aiednn_s8;
    for (int64_t i = 0; i < q->C * P; i++)
        codes[i] = sgn ? (int16_t)(int8_t)ys[i] : (int16_t)ys[i];
    return rc;
}

/* every step of every channel: steps[c * RT_ST_MAXN ...], nst[c], c0 = code at the lowest key,
 * nanc = code of NaN */
static int steps_find(const rt_prob *p, const lut_plan *q, void *const *ptr, st_step *steps, int *nst, int16_t *c0,
                      int16_t *nanc) {
    const int f32 = p->md[q->main_slot].data_type == aiednn_f32;
    const int64_t kmin = f32 ? (int32_t)(0xff800000u ^ 0x7fffffffu) : INT32_MIN, kmax = f32 ? 0x7f800000 : INT32_MAX;
    const int64_t C = q->C, MAXB = 2 * 256 + 2; /* open brackets per channel (<= steps found + 2) */
    uint32_t *xs = (uint32_t *)scratch(&g_stx, &g_stx_cap, (size_t)(C * MAXB) * 4);
    int16_t *cd = (int16_t *)scratch(&g_stn, &g_stn_cap, (size_t)(C * MAXB) * 2);
    st_br *br = (st_br *)scratch(&g_stb, &g_stb_cap, (size_t)(C * MAXB * 2) * sizeof(st_br));
    if (!xs || !cd || !br)
        return AIEK_ERR_ALLOC;
    /* lowest key, -1 / 0 (the two half lines), highest key, NaN */
    const int64_t probe[5] = {kmin, -1, 0, kmax, 0};
    for (int64_t c = 0; c < C; c++)
        for (int j = 0; j < 5; j++)
            xs[c * 5 + j] = j == 4 ? (f32 ? 0x7fc00000u : 0u) : steps_word(f32, probe[j]);
    int rc = steps_eval(p, q, ptr, 5, xs, cd);
    if (rc != AIEK_OK)
        return rc;
    int64_t *nb = (int64_t *)scratch(&g_sts, &g_sts_cap, (size_t)C * 2 * sizeof(int64_t)); /* open, next open */
    if (!nb)
        return AIEK_ERR_ALLOC;
    st_br *cur = br, *nxt = br + C * MAXB;
    for (int64_t c = 0; c < C; c++) {
        const int16_t *k = cd + c * 5;
        c0[c] = k[0], nanc[c] = f32 ? k[4] : 0, nst[c] = 0;
        st_br *o = cur + c * MAXB;
        int n = 0;
        if (k[0] != k[1])
            o[n++] = {kmin, -1, k[0], k[1]};
        if (k[1] != k[2]) /* the step at zero: the bracket (-1, 0] is already one key wide */
            o[n++] = {-1, 0, k[1], k[2]};
        if (k[2] != k[3])
            o[n++] = {0, kmax, k[2], k[3]};
        nb[c] = n;
    }
    for (;;) {
        int64_t P = 0;
        for (int64_t c = 0; c < C; c++)
            P = nb[c] > P ? nb[c] : P;
        if (!P)
            break;
        for (int64_t c = 0; c < C; c++) /* one-key brackets are steps; the rest probe their midpoint */
            for (int64_t j = 0; j < P; j++) {
                const st_br *b = cur + c * MAXB + (j < nb[c] ? j : 0);
                xs[c * P + j] = j < nb[c] ? steps_word(f32, b->a + (b->b - b->a) / 2) : 0u;
            }
        rc = steps_eval(p, q, ptr, P, xs, cd);
        if (rc != AIEK_OK)
            return rc;
        for (int64_t c = 0; c < C; c++) {
            st_br *o = nxt + c * MAXB;
            int n = 0;
            for (int64_t j = 0; j < nb[c]; j++) {
                const st_br b = cur[c * MAXB + j];
                if (b.b - b.a == 1) { /* code changes exactly at key b */
                    if (nst[c] >= RT_ST_MAXN)
                        return AIEK_ERR_ARGS;
                    steps[c * RT_ST_MAXN + nst[c]++] = {b.b, b.cb};
                    continue;
                }
                const int64_t m = b.a + (b.b - b.a) / 2;
                const int16_t cm = cd[c * P + j];
                if (n + 2 > MAXB)
                    return AIEK_ERR_ARGS;
                if (cm != b.ca)
                    o[n++] = {b.a, m, b.ca, cm};
                if (cm != b.cb)
                    o[n++] = {m, b.b, cm, b.cb};
            }
            nb[c + C] = n;
        }
        for (int64_t c = 0; c < C; c++)
            nb[c] = nb[c + C];
        st_br *t = cur;
        cur = nxt, nxt = t;
    }
    return AIEK_OK;
}

static int st_cmp(const void *x, const void *y) {
    const int64_t a = ((const st_step *)x)->t, b = ((const st_step *)y)->t;
    return a < b ? -1 : a > b;
}

/* the cache: tables of the last few problems, keyed by their parameters and operand contents */
#define RT_ST_CACHE 128 /* cached step tables */
typedef struct {
    uint64_t h;
    int64_t C;
    int stride;
    uint8_t *tab; /* C * stride bytes */
} st_cache;
static st_cache g_st_cache[RT_ST_CACHE];
static int g_st_next;

static uint64_t st_fnv(uint64_t h, const void *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        h = (h ^ ((const uint8_t *)p)[i]) * 0x100000001b3ull;
    return h;
}

static uint64_t steps_hash(const rt_prob *p, const lut_plan *q, void *const *ptr) {
    uint64_t h = 0xcbf29ce484222325ull;
    const int head[4] = {p->kind, p->alg, (int)p->flags, q->axis};
    const float fp[4] = {p->alpha, p->beta, p->eps, p->scale};
    h = st_fnv(h, head, sizeof head);
    h = st_fnv(h, fp, sizeof fp);
    h = st_fnv(h, &p->zero_point, sizeof p->zero_point);
    h = st_fnv(h, &q->C, sizeof q->C);
    /* the fused epilogue: layers with equal statistics differ only in their dst scale */
    const int attr_i[3] = {p->has_attr, p->dst_zp, p->npo};
    h = st_fnv(h, attr_i, sizeof attr_i);
    h = st_fnv(h, &p->dst_scale, sizeof p->dst_scale);
    h = st_fnv(h, &p->n_oscales, sizeof p->n_oscales);
    if (p->n_oscales)
        h = st_fnv(h, p->oscales, (size_t)p->n_oscales * sizeof(float));
    for (int k = 0; k < p->npo; k++) {
        const int pk[2] = {p->po[k].kind, p->po[k].alg};
        const float pf[2] = {p->po[k].alpha, p->po[k].beta};
        h = st_fnv(h, pk, sizeof pk);
        h = st_fnv(h, pf, sizeof pf);
    }
    for (int i = 0; i < p->nmd; i++) {
        h = st_fnv(h, &p->md[i].data_type, sizeof p->md[i].data_type);
        if (i != q->main_slot && i != q->dst_slot) /* weights, statistics, the broadcast side */
            h = st_fnv(h, ptr[i], (size_t)(md_numel(&p->md[i]) * dt_bytes(p->md[i].data_type)));
    }
    return h;
}

/* this problem's tables (cached): returns them and the per-channel stride, or NULL */
static const uint8_t *steps_tables(const rt_prob *p, const lut_plan *q, void *const *ptr, int *stride_out) {
    const uint64_t h = steps_hash(p, q, ptr);
    for (int i = 0; i < RT_ST_CACHE; i++)
        if (g_st_cache[i].tab && g_st_cache[i].h == h && g_st_cache[i].C == q->C) {
            *stride_out = g_st_cache[i].stride;
            return g_st_cache[i].tab;
        }
    st_step *steps = (st_step *)rt_alloc((size_t)(q->C * RT_ST_MAXN) * sizeof(st_step)); /* temporaries */
    int *nst = (int *)rt_alloc((size_t)q->C * sizeof(int));
    int16_t *c0 = (int16_t *)rt_alloc((size_t)q->C * 2 * sizeof(int16_t));
    if (!steps || !nst || !c0 || steps_find(p, q, ptr, steps, nst, c0, c0 + q->C) != AIEK_OK) {
        rt_free(steps), rt_free(nst), rt_free(c0);
        return nullptr;
    }
    int maxn = 0;
    for (int64_t c = 0; c < q->C; c++)
        maxn = nst[c] > maxn ? nst[c] : maxn;
    const int stride = (AIE_STEPS_BYTES(maxn) + 7) & ~7;
    uint8_t *tab = (uint8_t *)calloc((size_t)(q->C * stride), 1);
    for (int64_t c = 0; tab && c < q->C; c++) {
        st_step *s = steps + c * RT_ST_MAXN;
        qsort(s, (size_t)nst[c], sizeof *s, st_cmp);
        uint8_t *tb = tab + c * stride;
        *(int32_t *)tb = nst[c];
        *(int16_t *)(tb + 4) = c0[q->C + c];
        int32_t *T = (int32_t *)(tb + AIE_STEPS_HDR);
        int16_t *P = (int16_t *)(tb + AIE_STEPS_HDR + 4 * nst[c]);
        P[0] = c0[c];
        for (int k = 0; k < nst[c]; k++)
            T[k] = (int32_t)s[k].t, P[k + 1] = s[k].code;
    }
    rt_free(steps), rt_free(nst), rt_free(c0);
    if (!tab)
        return nullptr;
    st_cache *e = &g_st_cache[g_st_next];
    g_st_next = (g_st_next + 1) % RT_ST_CACHE;
    free(e->tab);
    *e = {h, q->C, stride, tab};
    *stride_out = stride;
    return tab;
}

static void *g_sti, *g_stt, *g_sto;
static size_t g_sti_cap, g_stt_cap, g_sto_cap;

static int run_steps(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    lut_plan q;
    char why[160];
    if (!steps_plan_make(p, &q, why, sizeof why))
        return AIEK_ERR_ARGS;
    const uint64_t tt0 = rt_ticks();
    int stride = 0;
    const uint8_t *tab = steps_tables(p, &q, ptr, &stride);
    g_phase[RT_PH_BPACK] += rt_ticks() - tt0;
    if (!tab)
        return AIEK_ERR_ALLOC;
    const int64_t NI = self->gen->args[0].dims[0], slots = AIE_LUT_TBYTES / stride;
    uint8_t *items = (uint8_t *)scratch(&g_sti, &g_sti_cap, (size_t)(NI * AIE_LUT_ITEM));
    uint8_t *tabs = (uint8_t *)scratch(&g_stt, &g_stt_cap, (size_t)(NI * AIE_LUT_TBYTES));
    uint8_t *out = (uint8_t *)scratch(&g_sto, &g_sto_cap, (size_t)(NI * AIE_LUT_ITEM));
    if (!items || !tabs || !out)
        return AIEK_ERR_ALLOC;
    const uint32_t *x = (const uint32_t *)ptr[q.main_slot];
    uint8_t *y = (uint8_t *)ptr[q.dst_slot];
    const int64_t cap = AIE_LUT_ITEM / 4, nt = q.C <= slots ? q.C : slots;
    const int f32 = p->md[q.main_slot].data_type == aiednn_f32;
    const uint32_t flags =
        1u << 8 | (uint32_t)(stride / 8) << 16 | (uint32_t)(f32 ? AIE_STEPS_F32 : AIE_STEPS_S32) << 26 | 1u << 27;
    int32_t geom[RT_LUT_MAX_ITEMS * 4];
    int64_t e0s[RT_LUT_MAX_ITEMS], ns[RT_LUT_MAX_ITEMS];
    const int64_t shape_i[2] = {NI, AIE_LUT_ITEM}, shape_t[2] = {NI, AIE_LUT_TBYTES}, shape_g[2] = {NI, 4};
    for (int64_t e = 0; e < q.numel;) {
        const uint64_t tp0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++) {
            const int64_t ch0 = (e / q.inner) % q.C, run0 = q.inner - e % q.inner;
            int64_t n = q.numel - e < cap ? q.numel - e : cap;
            if (q.C > slots && n > run0 + (slots - 1) * q.inner)
                n = run0 + (slots - 1) * q.inner;
            e0s[k] = e, ns[k] = n;
            memcpy(items + k * AIE_LUT_ITEM, x + e, (size_t)n * 4);
            for (int64_t s = 0; s < nt; s++)
                memcpy(tabs + k * AIE_LUT_TBYTES + s * stride, tab + ((ch0 + s) % q.C) * stride, (size_t)stride);
            geom[4 * k] = (int32_t)n;
            geom[4 * k + 1] = (int32_t)(run0 < 65535 ? run0 : 65535);
            geom[4 * k + 2] = (int32_t)(q.inner < 65535 ? q.inner : 65535);
            geom[4 * k + 3] = (int32_t)(flags | (uint32_t)nt);
            e += n;
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        aiek_tensor t[4] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * AIE_LUT_ITEM)},
                            {tabs, 2, AIEK_INT, 8, 1, shape_t, (uint64_t)(NI * AIE_LUT_TBYTES)},
                            {geom, 2, AIEK_INT, 32, 1, shape_g, (uint64_t)(NI * 4 * sizeof(int32_t))},
                            {out, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * AIE_LUT_ITEM)}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        const int rc = self->gen->entry(t, 4);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t te0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++)
            if (ns[k])
                memcpy(y + e0s[k], out + k * AIE_LUT_ITEM, (size_t)ns[k]);
        g_phase[RT_PH_EPI] += rt_ticks() - te0;
        for (int64_t k = 0; k < NI; k++)
            ns[k] = 0;
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_STEPS_H */
