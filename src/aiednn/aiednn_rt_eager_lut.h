/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_eager_lut.h -- eager:lut: any per-channel function of one 8-bit value with an 8-bit result on the
 * lut template's instances (aie:eager_lut__M_N_TM, src/aietensorop/eager/lut_eager_kernel.cc). The tables are
 * fixed-megakernel lut:'s (lut_tables: ref: run on all 256 codes of every channel), so the result is ref:'s bit
 * for bit. Each of the instance's N columns takes one run of up to M elements of one channel, with that
 * channel's table; a launch carries N runs. A 16- / 32-bit dst (u8 -> f32 eltwise, quantized eltwise into f32, ...)
 * runs on the lutw template (aie:eager_lutw__M_N_TM_ES), whose tables hold ES-byte entries.
 *
 * Needs, from the including TU: aiednn_rt_lut.h (lut_plan_shape, lut_tables).
 ******************************************************************************/
#ifndef AIEDNN_RT_EAGER_LUT_H
#define AIEDNN_RT_EAGER_LUT_H

#define RT_EAGER_LUT_TABLE 256 /* the template's table chunks (lut_eager_ref.h: EG_LUT_TABLE) */

typedef struct {
    const aiednn_impl_t *g;
    int64_t M, N, ES; /* ES: result bytes (lut 1, lutw 2 / 4) */
} rt_eager_lut;

static int rt_eager_lut_of(const aiednn_impl_t *g, rt_eager_lut *q) {
    long long m, n, tm, es;
    if (g->nargs != 3)
        return 0;
    (void)es;
    if (sscanf(g->name, "aie:eager_lut__%lld_%lld_%lld", &m, &n, &tm) != 3 || (n != 64 && n != 128))
        return 0;
    q->g = g, q->M = m, q->N = n, q->ES = 1;
    return 1;
}

/* the instance with es-byte results and the most columns x rows: fewest launches */
static int rt_eager_lut_find(int64_t es, rt_eager_lut *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_lut q;
        if (rt_eager_lut_of(&aiednn_impl_list[i], &q) && q.ES == es && (!found || q.M * q.N > out->M * out->N))
            *out = q, found = 1;
    }
    return found;
}

/* ── per-row table templates (lutw: 8-bit -> ES-byte entries; steps: 32-bit -> 8-bit step tables) ─────────────
 * aie:eager_lutw__R_V_TR_ES / aie:eager_steps__R_V_TR_S: every row one run of one channel, its table in front of
 * its values. */

typedef struct {
    const aiednn_impl_t *g;
    int64_t R, V, P; /* P: lutw's ES, steps' S (table bytes) */
} rt_eager_rowtab;

static int rt_eager_rowtab_of(const aiednn_impl_t *g, const char *tmpl, rt_eager_rowtab *q) {
    char fmt[64];
    snprintf(fmt, sizeof fmt, "aie:eager_%s__%%lld_%%lld_%%lld_%%lld", tmpl);
    long long r, v, tr, pp;
    if (g->nargs != 2 || sscanf(g->name, fmt, &r, &v, &tr, &pp) != 4)
        return 0;
    q->g = g, q->R = r, q->V = v, q->P = pp;
    return 1;
}

/* lutw: ES == P; steps: the table fits (P >= need); then the most values per launch */
static int rt_eager_rowtab_find(const char *tmpl, int64_t es, int64_t need, rt_eager_rowtab *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_rowtab q;
        if (!rt_eager_rowtab_of(&aiednn_impl_list[i], tmpl, &q) || (es ? q.P != es : q.P < need))
            continue;
        if (!found || q.R * q.V > out->R * out->V)
            *out = q, found = 1;
    }
    return found;
}

static void *g_erw_x, *g_erw_y;
static size_t g_erw_x_cap, g_erw_y_cap;

/* the per-row-table run: channel runs as rows of up to V elements, each with its channel's tb-byte table
 * (tab + ch * tstride); vb input bytes per element, ob output bytes per element */
static int run_eager_rowtab(const rt_impl *self, const rt_eager_rowtab *w, const lut_plan *q, const uint8_t *tab,
                            int64_t tstride, int64_t tb, int64_t vb, int64_t ob, const uint8_t *x, uint8_t *dst) {
    const int64_t ld = tb + vb * w->V;
    uint8_t *xs = (uint8_t *)scratch(&g_erw_x, &g_erw_x_cap, (size_t)(w->R * ld));
    uint8_t *ys = (uint8_t *)scratch(&g_erw_y, &g_erw_y_cap, (size_t)(w->R * w->V * ob));
    if (!xs || !ys)
        return AIEK_ERR_ALLOC;
    int64_t e0s[1024], ns[1024]; /* rows per launch: R <= 1024 */
    if (w->R > 1024)
        return AIEK_ERR_ARGS;
    const int64_t xshp[2] = {w->R, ld}, yshp[2] = {w->R, w->V * ob};
    for (int64_t el = 0; el < q->numel;) {
        uint64_t t0 = rt_ticks();
        int64_t r = 0;
        memset(xs, 0, (size_t)(w->R * ld));
        for (; r < w->R && el < q->numel; r++) { /* row r: one run of one channel */
            const int64_t run = q->inner - el % q->inner, n = run < w->V ? run : w->V;
            const int64_t ch = el / q->inner % q->C;
            memcpy(xs + r * ld, tab + ch * tstride, (size_t)tb);
            memcpy(xs + r * ld + tb, x + el * vb, (size_t)(n * vb));
            e0s[r] = el, ns[r] = n;
            el += n;
        }
        for (int64_t k = r; k < w->R; k++) /* padding rows: the first row's table (a valid one) */
            memcpy(xs + k * ld, xs, (size_t)tb);
        g_phase[RT_PH_DWPACK] += rt_ticks() - t0;
        aiek_tensor t[2] = {{xs, 2, AIEK_INT, 8, 1, xshp, (uint64_t)(w->R * ld)},
                            {ys, 2, AIEK_INT, 8, 1, yshp, (uint64_t)(w->R * w->V * ob)}};
        const uint64_t te = rt_ticks();
        const int rc = w->g->entry(t, 2);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        for (int64_t k = 0; k < r; k++)
            memcpy(dst + e0s[k] * ob, ys + k * w->V * ob, (size_t)(ns[k] * ob));
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

static int acc_eager_lut(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    lut_plan q;
    if (!lut_plan_shape(p, &q, why, cap))
        return 0;
    const aiednn_data_type_t mt = p->md[q.main_slot].data_type;
    if (mt != aiednn_s8 && mt != aiednn_u8) {
        snprintf(why, cap, "an 8-bit full-size source");
        return 0;
    }
    rt_eager_lut e;
    rt_eager_rowtab w;
    if (q.es == 1 ? !rt_eager_lut_find(1, &e) : !rt_eager_rowtab_find("lutw", q.es, 0, &w)) {
        snprintf(why, cap, "no lut / lutw instance with %d-byte results", q.es);
        return 0;
    }
    return 1;
}

static void *g_el_tab, *g_el_a, *g_el_w, *g_el_y;
static size_t g_el_tab_cap, g_el_a_cap, g_el_w_cap, g_el_y_cap;

static int rt_eager_lut_launch(const rt_impl *self, const rt_eager_lut *e, int8_t *a, int8_t *w, int8_t *y) {
    const int64_t ld = RT_EAGER_LUT_TABLE * e->ES + e->N;
    const int64_t a_shape[2] = {e->M, ld}, w_shape[2] = {e->N, ld}, y_shape[2] = {e->M, e->N * e->ES};
    aiek_tensor t[3] = {{a, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(e->M * ld)},
                        {w, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(e->N * ld)},
                        {y, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)(e->M * e->N * e->ES)}};
    const uint64_t te = rt_ticks();
    const int rc = e->g->entry(t, 3);
    ((rt_impl *)self)->t_entry += rt_ticks() - te;
    ((rt_impl *)self)->launches++;
    return rc;
}

static int run_eager_lut(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    lut_plan q;
    char why[160];
    rt_eager_lut e;
    if (!lut_plan_shape(p, &q, why, sizeof why))
        return AIEK_ERR_ARGS;
    if (q.es > 1) { /* a wide dst: lutw, the channel's table in front of every row */
        rt_eager_rowtab w;
        if (!rt_eager_rowtab_find("lutw", q.es, 0, &w))
            return AIEK_ERR_ARGS;
        uint8_t *wt = (uint8_t *)scratch(&g_el_tab, &g_el_tab_cap, (size_t)(q.C * 256 * q.es));
        if (!wt)
            return AIEK_ERR_ALLOC;
        const uint64_t tt = rt_ticks();
        const int rc = lut_tables(p, &q, ptr, wt);
        g_phase[RT_PH_BPACK] += rt_ticks() - tt;
        if (rc != AIEK_OK)
            return rc;
        return run_eager_rowtab(self, &w, &q, wt, 256 * q.es, 256 * q.es, 1, q.es, (const uint8_t *)ptr[q.main_slot],
                                (uint8_t *)ptr[q.dst_slot]);
    }
    if (!rt_eager_lut_find(q.es, &e))
        return AIEK_ERR_ARGS;
    const int64_t es = e.ES, tb = RT_EAGER_LUT_TABLE * es, ld = tb + e.N;
    uint8_t *tab = (uint8_t *)scratch(&g_el_tab, &g_el_tab_cap, (size_t)(q.C * tb));
    int8_t *a = (int8_t *)scratch(&g_el_a, &g_el_a_cap, (size_t)(e.M * ld));
    int8_t *w = (int8_t *)scratch(&g_el_w, &g_el_w_cap, (size_t)(e.N * ld));
    int8_t *y = (int8_t *)scratch(&g_el_y, &g_el_y_cap, (size_t)(e.M * e.N * es));
    if (!tab || !a || !w || !y)
        return AIEK_ERR_ALLOC;
    const uint64_t tt0 = rt_ticks();
    int rc = lut_tables(p, &q, ptr, tab);
    g_phase[RT_PH_BPACK] += rt_ticks() - tt0;
    if (rc != AIEK_OK)
        return rc;
    const uint8_t *x = (const uint8_t *)ptr[q.main_slot];
    uint8_t *dst = (uint8_t *)ptr[q.dst_slot];
    int64_t e0s[128], ns[128];
    memset(a, 0, (size_t)(e.M * ld));
    memset(w, 0, (size_t)(e.N * ld));
    for (int64_t el = 0; el < q.numel;) {
        const uint64_t tp0 = rt_ticks();
        int64_t col = 0;
        for (; col < e.N && el < q.numel; col++) { /* column col: one run of one channel */
            const int64_t run = q.inner - el % q.inner, n = run < e.M ? run : e.M;
            const int64_t ch = el / q.inner % q.C;
            memcpy(w + col * ld, tab + ch * tb, (size_t)tb);
            for (int64_t m = 0; m < n; m++)
                a[m * ld + tb + col] = (int8_t)x[el + m];
            e0s[col] = el, ns[col] = n;
            el += n;
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        if ((rc = rt_eager_lut_launch(self, &e, a, w, y)) != AIEK_OK)
            return rc;
        const uint64_t tu = rt_ticks();
        for (int64_t k = 0; k < col; k++)
            for (int64_t m = 0; m < ns[k]; m++)
                memcpy(dst + (e0s[k] + m) * es, y + (m * e.N + k) * es, (size_t)es);
        g_phase[RT_PH_CUNPACK] += rt_ticks() - tu;
    }
    return AIEK_OK;
}

/* ── eager:steps -- f32 / s32 -> s8 / u8 through fixed-megakernel steps:'s step tables (steps_tables: ref:
 * bisected over the key space, exact by construction), on the steps template ──────────────────────────────── */

static int acc_eager_steps(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    lut_plan q;
    rt_eager_rowtab w;
    if ((p->kind == aiednn_binary || p->kind == aiednn_eltwise) && p->has_attr) {
        snprintf(why, cap, "the quantized binary / eltwise (attr): bin8 / lut");
        return 0;
    }
    if (!steps_plan_make(p, &q, why, cap))
        return 0;
    char qw[8];
    if (p->kind == aiednn_reorder && acc_eager_quant(self, p, bind, qw, sizeof qw)) {
        snprintf(why, cap, "f32 -> 8-bit quantize: eager:quant serves it");
        return 0;
    }
    if (!rt_eager_rowtab_find("steps", 0, (AIE_STEPS_BYTES(RT_ST_MAXN) + 7) & ~7, &w)) {
        snprintf(why, cap, "no steps instance holding the largest table (%d bytes)",
                 (AIE_STEPS_BYTES(RT_ST_MAXN) + 7) & ~7);
        return 0;
    }
    return 1;
}

static void *g_est_tab;
static size_t g_est_tab_cap;

static int run_eager_steps(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    lut_plan q;
    char why[160];
    if (!steps_plan_make(p, &q, why, sizeof why))
        return AIEK_ERR_ARGS;
    const uint64_t tt0 = rt_ticks();
    int stride = 0;
    const uint8_t *tab = steps_tables(p, &q, ptr, &stride);
    g_phase[RT_PH_BPACK] += rt_ticks() - tt0;
    rt_eager_rowtab w;
    if (!tab || !rt_eager_rowtab_find("steps", 0, stride, &w))
        return AIEK_ERR_ALLOC;
    /* the instance's table: steps:'s slot with the key kind in the reserved int16 (bytes 6 .. 7) */
    uint8_t *et = (uint8_t *)scratch(&g_est_tab, &g_est_tab_cap, (size_t)(q.C * w.P));
    if (!et)
        return AIEK_ERR_ALLOC;
    const int16_t kind = p->md[q.main_slot].data_type == aiednn_f32 ? 0 : 1;
    for (int64_t c = 0; c < q.C; c++) {
        memset(et + c * w.P, 0, (size_t)w.P);
        memcpy(et + c * w.P, tab + c * stride, (size_t)stride);
        memcpy(et + c * w.P + 6, &kind, 2);
    }
    return run_eager_rowtab(self, &w, &q, et, w.P, w.P, 4, 1, (const uint8_t *)ptr[q.main_slot],
                            (uint8_t *)ptr[q.dst_slot]);
}

/* ── eager:red8 -- reduction of an 8-bit tensor over its trailing axes on the red8 template
 * (aie:eager_red8__R_L_TR_OP_SG): every segment cut into rows of the instance's L (neutral padding), the rows'
 * exact int32 partials combined on the host, then ref:reduction's double epilogue (fixed redblk:'s red_plan_make /
 * finish): ref:'s result bit for bit. ─────────────────────────────────────────────────────────────────────── */

typedef struct {
    const aiednn_impl_t *g;
    int64_t R, L;
    int op, sg;
} rt_eager_r8;

static int rt_eager_r8_of(const aiednn_impl_t *g, rt_eager_r8 *q) {
    long long r, l, tr;
    int op, sg;
    if (g->nargs != 2 || sscanf(g->name, "aie:eager_red8__%lld_%lld_%lld_%d_%d", &r, &l, &tr, &op, &sg) != 5)
        return 0;
    q->g = g, q->R = r, q->L = l, q->op = op, q->sg = sg;
    return 1;
}

/* red_plan's AIE_MODE_POOL_* op -> the template's OP */
static int eager_r8_op(int pool_op) {
    return pool_op == AIE_MODE_POOL_SUM      ? 1
           : pool_op == AIE_MODE_POOL_MAX    ? 2
           : pool_op == AIE_MODE_POOL_MIN    ? 3
           : pool_op == AIE_MODE_POOL_SUMABS ? 4
           : pool_op == AIE_MODE_POOL_SUMSQ  ? 5
           : pool_op == AIE_MODE_POOL_MAXABS ? 6
                                             : 0;
}

/* the instance of (op, sg): the fewest launches for segments of R elements */
static int rt_eager_r8_find(int op, int sg, int64_t segs, int64_t R, rt_eager_r8 *out) {
    int found = 0;
    int64_t best = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_r8 q;
        if (!rt_eager_r8_of(&aiednn_impl_list[i], &q) || q.op != op || q.sg != sg)
            continue;
        const int64_t rows = segs * ((R + q.L - 1) / q.L), launches = (rows + q.R - 1) / q.R;
        if (!found || launches < best)
            *out = q, best = launches, found = 1;
    }
    return found;
}

static int acc_eager_red8(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    red_plan q;
    rt_eager_r8 e;
    if (p->has_attr || p->npo || rt_prob_has_fmt(p) || !red_plan_make(p, &q, why, cap))
        return 0;
    if (!rt_eager_r8_find(eager_r8_op(q.op), q.sgn, q.outer, q.R, &e)) {
        snprintf(why, cap, "no red8 instance of this reduction and signedness");
        return 0;
    }
    return 1;
}

static void *g_er8_x, *g_er8_y, *g_er8_v;
static size_t g_er8_x_cap, g_er8_y_cap, g_er8_v_cap;

static int run_eager_red8(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    red_plan q;
    char why[160];
    rt_eager_r8 e;
    if (!red_plan_make(p, &q, why, sizeof why) || !rt_eager_r8_find(eager_r8_op(q.op), q.sgn, q.outer, q.R, &e))
        return AIEK_ERR_ARGS;
    const int64_t chunks = (q.R + e.L - 1) / e.L, rows = q.outer * chunks;
    uint8_t *xs = (uint8_t *)scratch(&g_er8_x, &g_er8_x_cap, (size_t)(e.R * e.L));
    int32_t *ys = (int32_t *)scratch(&g_er8_y, &g_er8_y_cap, (size_t)e.R * 16);
    int64_t *val = (int64_t *)scratch(&g_er8_v, &g_er8_v_cap, (size_t)rows * sizeof(int64_t));
    if (!xs || !ys || !val)
        return AIEK_ERR_ALLOC;
    const uint8_t pad = e.op == 2 ? (q.sgn ? 0x80 : 0x00) : e.op == 3 ? (q.sgn ? 0x7f : 0xff) : 0;
    const uint8_t *x = (const uint8_t *)ptr[0];
    const int64_t xshp[2] = {e.R, e.L}, yshp[2] = {e.R, 4};
    for (int64_t r0 = 0; r0 < rows; r0 += e.R) {
        const int64_t nr = rows - r0 < e.R ? rows - r0 : e.R;
        uint64_t t0 = rt_ticks();
        memset(xs, pad, (size_t)(e.R * e.L));
        for (int64_t r = 0; r < nr; r++) { /* row r0 + r: chunk c of segment o */
            const int64_t o = (r0 + r) / chunks, c = (r0 + r) % chunks, len = q.R - c * e.L < e.L ? q.R - c * e.L : e.L;
            memcpy(xs + r * e.L, x + o * q.R + c * e.L, (size_t)len);
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - t0;
        aiek_tensor t[2] = {{xs, 2, AIEK_INT, 8, 1, xshp, (uint64_t)(e.R * e.L)},
                            {ys, 2, AIEK_INT, 32, 1, yshp, (uint64_t)e.R * 16}};
        const uint64_t te = rt_ticks();
        const int rc = e.g->entry(t, 2);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        for (int64_t r = 0; r < nr; r++)
            val[r0 + r] = ys[4 * r];
    }
    const uint64_t te0 = rt_ticks();
    const int is_max = e.op == 2 || e.op == 6, is_min = e.op == 3;
    for (int64_t o = 0; o < q.outer; o++) {
        int64_t v = val[o * chunks];
        for (int64_t c = 1; c < chunks; c++) {
            const int64_t u = val[o * chunks + c];
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

#endif /* AIEDNN_RT_EAGER_LUT_H */
