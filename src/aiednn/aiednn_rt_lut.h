/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_lut.h -- lut:<aie lut block>: any per-channel function of one 8-bit
 * value on the GEMM ELF's table mode (aie_modes_lut.h, opcode 0x7A).
 *
 * Accepted: eltwise, PReLU, batch normalization, binary with one operand
 * broadcast (scalar or along one axis) and identity-permutation reorders, whose
 * full-size operand is s8 / u8; any destination type. The other operands
 * (weights, mean / variance, the broadcast side) may be any type.
 *
 * Exact by construction: the tables are the ref: implementation itself, run on
 * a synthetic problem of the same kind whose full operand holds all 256 byte
 * values of every channel (dims 1 except the channel axis and one axis of 256).
 * The AIE then only indexes them, so the result is ref's bit for bit.
 *
 * Kernel block: items s8 [NI][4096], tables s8 [NI][7680], geom s32 [NI][4] =
 * {n, run0, inner, nt | es << 8}, out s8 [NI][4096] (es = 4: n <= 1024 words).
 * Item = n consecutive elements of the tensor; its channels in table slots
 * 0, 1, ... (all C of them when C fits, so a run past channel C - 1 wraps).
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_ops.h,
 * aie_modes_lut.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_LUT_H
#define AIEDNN_RT_LUT_H

#define RT_LUT_MAX_ITEMS 64 /* items per launch = the impl's arg 0 rows (16 on 4x4) */

static int is_lut_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_eltwise && g->nargs == 4 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[0].dims[0] >= 1 && g->args[0].dims[0] <= RT_LUT_MAX_ITEMS && g->args[0].dims[1] == AIE_LUT_ITEM &&
           g->args[1].data_type == aiednn_s8 && g->args[1].dims[1] == AIE_LUT_TBYTES &&
           g->args[2].data_type == aiednn_s32 && g->args[2].dims[1] == 4 && g->args[3].data_type == aiednn_s8 &&
           g->args[3].dims[1] == AIE_LUT_ITEM;
}

typedef struct {
    int main_slot, dst_slot; /* md slots of the full 8-bit operand and of dst */
    int axis;                /* channel axis, -1 = one table */
    int64_t C, inner, numel; /* channels, elements per channel run, elements */
    int es, slots;           /* result bytes; tables per core */
} lut_plan;

/* the broadcast operand o of dst d: -2 = not scalar / one-axis, -1 = scalar, else its axis */
static int lut_bcast_axis(const aiednn_memory_desc_t *o, const aiednn_memory_desc_t *d) {
    int axis = -1;
    if (!md_bcast_ok(o, d))
        return -2;
    for (int i = 0; i < d->ndims; i++)
        if (o->dims[i] != 1) {
            if (axis >= 0)
                return -2;
            axis = i;
        }
    return axis;
}

/* kind, operands, channel axis and runs; no type checks (lut: and steps: add theirs) */
static int lut_plan_shape(const rt_prob *p, lut_plan *q, char *why, int cap) {
    const aiednn_memory_desc_t *d = rt_md(p, AIEDNN_ARG_DST);
    q->axis = -1;
    q->main_slot = rt_slot(p, AIEDNN_ARG_SRC);
    q->dst_slot = rt_slot(p, AIEDNN_ARG_DST);
    /* batch norm may carry a dst quantization (dst_scale / dst_zp only, --aiednn-fold-norm-quant):
     * still per-channel and monotone, and the synthetic ref run copies it with the prob */
    const int bn_dstq = p->kind == aiednn_batch_normalization && !p->npo && !p->n_oscales;
    /* the quantized eltwise (create_attr): its ref is the whole dequantize -> eltwise -> post-ops -> quantize
     * chain, so the table run on all 256 codes is exact for it too */
    const int elt_q = p->kind == aiednn_eltwise && p->has_attr && !p->n_oscales;
    if (!d || (p->has_attr && !bn_dstq && !elt_q) || (p->npo && !elt_q)) {
        snprintf(why, cap, "no attr / post-ops (batch norm: dst quantization only; eltwise: create_attr)");
        return 0;
    }
    switch (p->kind) {
    case aiednn_eltwise:
        break;
    case aiednn_reorder:
        for (int k = 0; k < d->ndims; k++)
            if (p->perm[k] != k) {
                snprintf(why, cap, "reorder: identity permutation only");
                return 0;
            }
        break;
    case aiednn_batch_normalization:
        if (d->ndims < 2) {
            snprintf(why, cap, "batch norm: rank >= 2");
            return 0;
        }
        q->axis = 1;
        break;
    case aiednn_prelu:
        q->axis = lut_bcast_axis(rt_md(p, AIEDNN_ARG_WEIGHTS), d);
        break;
    case aiednn_binary: {
        const aiednn_memory_desc_t *a = rt_md(p, AIEDNN_ARG_SRC), *b = rt_md(p, AIEDNN_ARG_SRC_1);
        if (p->alg == aiednn_binary_select) {
            snprintf(why, cap, "binary: not select");
            return 0;
        }
        const int a_full = same_dims(a, d), b_full = same_dims(b, d);
        if (a_full == b_full) {
            snprintf(why, cap, "binary: exactly one operand must be dst-sized (the other broadcast)");
            return 0;
        }
        if (b_full)
            q->main_slot = rt_slot(p, AIEDNN_ARG_SRC_1);
        q->axis = lut_bcast_axis(b_full ? a : b, d);
        break;
    }
    default:
        snprintf(why, cap, "not a per-element kind");
        return 0;
    }
    const aiednn_memory_desc_t *m = &p->md[q->main_slot];
    if (q->axis == -2) {
        snprintf(why, cap, "the broadcast operand must be scalar or vary along one axis");
        return 0;
    }
    if (!same_dims(m, d)) {
        snprintf(why, cap, "source and dst: same shape");
        return 0;
    }
    q->numel = md_numel(d);
    q->es = dt_bytes(d->data_type);
    q->C = q->axis < 0 ? 1 : d->dims[q->axis];
    q->inner = q->numel;
    if (q->axis >= 0) {
        q->inner = 1;
        for (int i = q->axis + 1; i < d->ndims; i++)
            q->inner *= d->dims[i];
    }
    if (q->C > 1 && d->ndims < 2) {
        snprintf(why, cap, "a per-channel table needs rank >= 2");
        return 0;
    }
    return 1;
}

static int lut_plan_make(const rt_prob *p, lut_plan *q, char *why, int cap) {
    if (!lut_plan_shape(p, q, why, cap))
        return 0;
    const aiednn_data_type_t mt = p->md[q->main_slot].data_type;
    if (mt != aiednn_s8 && mt != aiednn_u8) {
        snprintf(why, cap, "the full-size source must be s8 or u8");
        return 0;
    }
    q->slots = AIE_LUT_TBYTES / (256 * q->es);
    if (q->C > q->slots && q->inner * q->slots < AIE_LUT_ITEM / q->es / 4) { /* items < 1/4 full */
        snprintf(why, cap, "%lld channels of %lld elements: too fine for %d tables per core", (long long)q->C,
                 (long long)q->inner, q->slots);
        return 0;
    }
    return 1;
}

/* identity reorders ref: runs as a tight contiguous loop (reorder_contig: dequantize to f32, same
 * type): one AIE launch per 64 KB loses to them (board: s8 -> f32 2.97 vs 0.69 ms) */
static int reorder_ref_is_fast(const rt_prob *p) {
    const aiednn_data_type_t s = p->md[0].data_type, d = p->md[1].data_type;
    return p->kind == aiednn_reorder && (s == d || (s == aiednn_f32) != (d == aiednn_f32));
}

static int acc_lut(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    (void)bind;
    lut_plan q;
    if (rt_cost_refusals() && reorder_ref_is_fast(p)) { /* aiednn_prefer_fastest only */
        snprintf(why, cap, "ref:reorder's contiguous loop is faster");
        return 0;
    }
    if (rt_cost_refusals() && eltwise_ref_8bit_ok(p)) { /* ref: looks the same table up on the host: no launch, */
        snprintf(why, cap, "ref:eltwise's 256-entry table is faster"); /* no megakernel switch (VEK385 */
        return 0;                                                      /* tiny_llama: 17 lut: parts, 303 ms) */
    }
    return lut_plan_make(p, &q, why, cap);
}

static int lut_run_ref(const rt_prob *p, void *const *ptr) {
    aiednn_primitive sub;
    memset(&sub, 0, sizeof sub);
    sub.prob = *p;
    switch (p->kind) {
    case aiednn_eltwise:
        return run_ref_eltwise(nullptr, &sub, ptr);
    case aiednn_reorder:
        return run_ref_reorder(nullptr, &sub, ptr);
    case aiednn_batch_normalization:
        return run_ref_bnorm(nullptr, &sub, ptr);
    case aiednn_prelu:
        return run_ref_prelu(nullptr, &sub, ptr);
    case aiednn_binary:
        return run_ref_binary(nullptr, &sub, ptr);
    default:
        return AIEK_ERR_ARGS;
    }
}

static void *g_lut_src, *g_lut_dst, *g_lut_tab;
static size_t g_lut_src_cap, g_lut_dst_cap, g_lut_tab_cap;

/* ref on a synthetic problem of p's kind: dims 1 except the channel axis (C) and one axis of P
 * probes. xs = C*P main-operand values, ys = C*P results, both channel-major (c*P + j); the
 * other operands (weights, mean, the broadcast side) are p's own. */
static int lut_synth(const rt_prob *p, const lut_plan *q, void *const *ptr, int64_t P, const void *xs, void *ys) {
    rt_prob s = *p;
    aiednn_memory_desc_t *m = &s.md[q->main_slot], *d = &s.md[q->dst_slot];
    const int nd = d->ndims;
    const int L = q->axis < 0 || q->axis < nd - 1 ? nd - 1 : q->axis - 1; /* the probe axis */
    for (int i = 0; i < nd; i++)
        m->dims[i] = d->dims[i] = 1;
    if (q->axis >= 0)
        m->dims[q->axis] = d->dims[q->axis] = q->C;
    m->dims[L] = d->dims[L] = P;
    const int mb = dt_bytes(m->data_type), db = dt_bytes(d->data_type);
    void *sp[RT_MAX_MD];
    for (int i = 0; i < s.nmd; i++)
        sp[i] = ptr[i];
    if (L > q->axis) { /* channel-major already */
        sp[q->main_slot] = (void *)xs;
        sp[q->dst_slot] = ys;
        return lut_run_ref(&s, sp);
    }
    uint8_t *xt = (uint8_t *)scratch(&g_lut_src, &g_lut_src_cap, (size_t)(q->C * P * mb));
    uint8_t *yt = (uint8_t *)scratch(&g_lut_dst, &g_lut_dst_cap, (size_t)(q->C * P * db));
    if (!xt || !yt)
        return AIEK_ERR_ALLOC;
    for (int64_t c = 0; c < q->C; c++) /* probe-major: (c, j) at j*C + c */
        for (int64_t j = 0; j < P; j++)
            memcpy(xt + (j * q->C + c) * mb, (const uint8_t *)xs + (c * P + j) * mb, mb);
    sp[q->main_slot] = xt;
    sp[q->dst_slot] = yt;
    const int rc = lut_run_ref(&s, sp);
    for (int64_t c = 0; c < q->C; c++)
        for (int64_t j = 0; j < P; j++)
            memcpy((uint8_t *)ys + (c * P + j) * db, yt + (j * q->C + c) * db, db);
    return rc;
}

static void *g_lut_x;
static size_t g_lut_x_cap;

/* tab [C][256 * es]: ref on all 256 byte values of every channel */
static int lut_tables(const rt_prob *p, const lut_plan *q, void *const *ptr, uint8_t *tab) {
    uint8_t *xs = (uint8_t *)scratch(&g_lut_x, &g_lut_x_cap, (size_t)q->C * 256);
    if (!xs)
        return AIEK_ERR_ALLOC;
    for (int64_t c = 0; c < q->C; c++)
        for (int v = 0; v < 256; v++)
            xs[c * 256 + v] = (uint8_t)v;
    return lut_synth(p, q, ptr, 256, xs, tab);
}

static void *g_lui, *g_lut, *g_luo;
static size_t g_lui_cap, g_lut_cap, g_luo_cap;

static int run_lut(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    lut_plan q;
    char why[160];
    if (!lut_plan_make(p, &q, why, sizeof why))
        return AIEK_ERR_ARGS;
    const int64_t NI = self->gen->args[0].dims[0];
    const int tb = 256 * q.es;
    uint8_t *tab = (uint8_t *)scratch(&g_lut_tab, &g_lut_tab_cap, (size_t)(q.C * tb));
    uint8_t *items = (uint8_t *)scratch(&g_lui, &g_lui_cap, (size_t)(NI * AIE_LUT_ITEM));
    uint8_t *tabs = (uint8_t *)scratch(&g_lut, &g_lut_cap, (size_t)(NI * AIE_LUT_TBYTES));
    uint8_t *out = (uint8_t *)scratch(&g_luo, &g_luo_cap, (size_t)(NI * AIE_LUT_ITEM));
    if (!tab || !items || !tabs || !out)
        return AIEK_ERR_ALLOC;
    const uint64_t tt0 = rt_ticks();
    int rc = lut_tables(p, &q, ptr, tab);
    g_phase[RT_PH_BPACK] += rt_ticks() - tt0;
    if (rc != AIEK_OK)
        return rc;
    const uint8_t *x = (const uint8_t *)ptr[q.main_slot];
    uint8_t *y = (uint8_t *)ptr[q.dst_slot];
    const int64_t cap = AIE_LUT_ITEM / q.es, nt = q.C <= q.slots ? q.C : q.slots;
    int32_t geom[RT_LUT_MAX_ITEMS * 4];
    int64_t e0s[RT_LUT_MAX_ITEMS], ns[RT_LUT_MAX_ITEMS];
    const int64_t shape_i[2] = {NI, AIE_LUT_ITEM}, shape_t[2] = {NI, AIE_LUT_TBYTES}, shape_g[2] = {NI, 4};
    for (int64_t e = 0; e < q.numel;) {
        const uint64_t tp0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++) {
            const int64_t ch0 = (e / q.inner) % q.C, run0 = q.inner - e % q.inner;
            int64_t n = q.numel - e < cap ? q.numel - e : cap;
            if (q.C > q.slots && n > run0 + (q.slots - 1) * q.inner)
                n = run0 + (q.slots - 1) * q.inner;
            e0s[k] = e, ns[k] = n;
            memcpy(items + k * AIE_LUT_ITEM, x + e, (size_t)n);
            for (int64_t s = 0; s < nt; s++)
                memcpy(tabs + k * AIE_LUT_TBYTES + s * tb, tab + ((ch0 + s) % q.C) * tb, (size_t)tb);
            geom[4 * k] = (int32_t)n;
            geom[4 * k + 1] = (int32_t)(run0 < 65535 ? run0 : 65535);
            geom[4 * k + 2] = (int32_t)(q.inner < 65535 ? q.inner : 65535);
            geom[4 * k + 3] = (int32_t)(nt | (q.es << 8));
            e += n;
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        aiek_tensor t[4] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * AIE_LUT_ITEM)},
                            {tabs, 2, AIEK_INT, 8, 1, shape_t, (uint64_t)(NI * AIE_LUT_TBYTES)},
                            {geom, 2, AIEK_INT, 32, 1, shape_g, (uint64_t)(NI * 4 * sizeof(int32_t))},
                            {out, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * AIE_LUT_ITEM)}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        rc = self->gen->entry(t, 4);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t te0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++)
            if (ns[k])
                memcpy(y + e0s[k] * q.es, out + k * AIE_LUT_ITEM, (size_t)(ns[k] * q.es));
        g_phase[RT_PH_EPI] += rt_ticks() - te0;
        for (int64_t k = 0; k < NI; k++) /* a later launch's unused items: n = 0 */
            ns[k] = 0;
    }
    return AIEK_OK;
}

/* x86 stand-in for the AIE lut / steps block (the functions the core runs): items, tables, geom, out
 * as above. For workstation runs of lut: without a board. */
static int rt_lut_block_x86(const aiek_tensor *t, int n) {
    if (n != 4)
        return AIEK_ERR_ARGS;
    for (int64_t k = 0; k < t[0].shape[0]; k++) {
        const int32_t *g = (const int32_t *)t[2].data + 4 * k;
        uint8_t d[11];
        aie_lut_desc(d, g);
        const uint8_t *in = (const uint8_t *)t[0].data + k * AIE_LUT_ITEM;
        const uint8_t *tab = (const uint8_t *)t[1].data + k * AIE_LUT_TBYTES;
        uint8_t *out = (uint8_t *)t[3].data + k * AIE_LUT_ITEM;
        if (((uint32_t)g[3] >> 27) & 1) /* steps: (aiednn_rt_steps.h) */
            aie_steps_apply(in, out, d, tab);
        else
            aie_lut_apply(in, out, d, tab);
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_LUT_H */
