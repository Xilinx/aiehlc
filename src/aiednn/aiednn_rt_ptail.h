/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_ptail.h -- post-ops (aiednn_attr_t: output scale, eltwise / binary post-ops, dst quantization) on the
 * primitives outside the GEMM family: pooling, softmax, layer / group / batch norm, reduction, resampling, LRN,
 * PReLU. The graph fusion patterns "X + post-ops" (pool, softmax, norms, reduction, interpolate) land here.
 *
 * create splits the problem (ptail_split): the inner primitive (the op without its attr, writing f32)
 * dispatches as standalone; execute then runs the chain through rt_epilogue. impl_info ends in "+pt".
 *
 * Needs, from the including TU: aiednn_rt_internal.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_PTAIL_H
#define AIEDNN_RT_PTAIL_H

static int ptail_kind(aiednn_primitive_kind_t k) {
    return k == aiednn_pooling || k == aiednn_softmax || k == aiednn_layer_normalization ||
           k == aiednn_group_normalization || k == aiednn_batch_normalization || k == aiednn_reduction ||
           k == aiednn_resampling || k == aiednn_lrn || k == aiednn_prelu || k == aiednn_reorder || k == aiednn_slice;
}

/* p0 carries post-ops on a ptail kind: inner = the op alone writing f32, ep = an epilogue-only problem in the GEMM
 * slot layout rt_epilogue reads ({-, -, dst} + the post-op sources). 0: not applicable, 1: split, -1: refused */
static int ptail_split(const rt_prob *p0, rt_prob *inner, rt_prob *ep) {
    if (!ptail_kind(p0->kind) || !p0->has_attr || p0->prop != RT_FWD_INF)
        return 0;
    /* every ptail kind applies the attr here, except batch / layer norm's and slice's dst quantization
     * alone, which their own impls fold */
    if (!p0->npo && !p0->n_oscales &&
        (p0->kind == aiednn_batch_normalization || p0->kind == aiednn_layer_normalization || p0->kind == aiednn_slice))
        return 0;
    if (p0->n_oscales > 1)
        return -1; /* per-channel output scales: GEMM family only (which axis is the channel is op-specific) */
    const int ds = rt_slot(p0, AIEDNN_ARG_DST);
    const aiednn_memory_desc_t *d = &p0->md[ds];
    prob_init(ep, p0->kind, 0);
    prob_add(ep, AIEDNN_ARG_SRC, d); /* placeholders: rt_epilogue reads slot 2 as dst */
    prob_add(ep, AIEDNN_ARG_WEIGHTS, d);
    prob_add(ep, AIEDNN_ARG_DST, d);
    *inner = *p0;
    inner->nmd = 0;
    for (int s = 0; s < p0->nmd; s++) {
        if (p0->arg[s] >= AIEDNN_ARG_ATTR_POST_OP_SRC(0) &&
            p0->arg[s] < AIEDNN_ARG_ATTR_POST_OP_SRC(AIEDNN_MAX_POST_OPS)) {
            prob_add(ep, p0->arg[s], &p0->md[s]);
            continue;
        }
        inner->arg[inner->nmd] = p0->arg[s], inner->md[inner->nmd++] = p0->md[s];
    }
    for (int k = 0; k < p0->npo; k++)
        ep->po[k] = p0->po[k];
    ep->npo = p0->npo, ep->has_attr = 1, ep->dst_scale = p0->dst_scale, ep->dst_zp = p0->dst_zp;
    ep->n_oscales = p0->n_oscales, ep->oscales = p0->oscales; /* owned by the primitive (pt_ep) from here */
    if (md_is_nhwc(d)) { /* the tail walks dst's bytes: every 4-D operand in their physical (nhwc) order */
        for (int s = 0; s < ep->nmd; s++)
            if (ep->md[s].ndims == 4 && !md_is_nhwc(&ep->md[s]) && !md_fmt_free(&ep->md[s]))
                return -1;
        for (int s = 0; s < ep->nmd; s++)
            md_to_phys(&ep->md[s], 1);
        for (int k = 0; k < ep->npo; k++)
            if (ep->po[k].src1.ndims == 4)
                md_to_phys(&ep->po[k].src1, 1);
    }
    inner->has_attr = 0, inner->npo = 0, inner->n_oscales = 0, inner->oscales = nullptr;
    inner->dst_scale = 1.0f, inner->dst_zp = 0;
    inner->md[rt_slot(inner, AIEDNN_ARG_DST)].data_type = aiednn_f32;
    return 1;
}

static void *g_ptail_buf;
static size_t g_ptail_cap;

/* the f32 tile the inner primitive writes in place of dst */
static float *ptail_scratch(const aiednn_primitive *prim) {
    const aiednn_memory_desc_t *d = &prim->pt_ep.md[2];
    return (float *)scratch(&g_ptail_buf, &g_ptail_cap, (size_t)md_numel(d) * sizeof(float));
}

/* after the inner run: dst[i] = the post-op chain on tile[i] */
static int ptail_run(const aiednn_primitive *prim, const float *tile, void *dst, int nargs,
                     const aiednn_exec_arg_t *args) {
    const rt_prob *ep = &prim->pt_ep;
    void *eptr[RT_MAX_MD] = {nullptr, nullptr, dst};
    for (int s = 3; s < ep->nmd; s++) {
        eptr[s] = nullptr;
        for (int a = 0; a < nargs; a++)
            if (args[a].arg == ep->arg[s])
                eptr[s] = args[a].data;
        if (!eptr[s])
            return AIEK_ERR_ARGS;
    }
    const uint64_t t0 = rt_ticks();
    const int64_t n = md_numel(&ep->md[2]);
    for (int64_t i = 0; i < n; i++)
        rt_epilogue(ep, eptr, i, 0, (double)tile[i]);
    g_phase[RT_PH_QTAIL] += rt_ticks() - t0;
    return AIEK_OK;
}

#endif /* AIEDNN_RT_PTAIL_H */
