/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_grouped.h -- grouped matmul (mixture-of-experts): one matmul per group of rows, each
 * dispatched on its own [rows, K] x [K, N] shape at execute (the group sizes are run-time data), so
 * an int8 group takes the AIE blocked GEMM and a tiny one whatever dispatch prefers for it.
 *
 * Needs, from the including TU: aiednn_rt_internal.h and dispatch / the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_GROUPED_H
#define AIEDNN_RT_GROUPED_H

static int run_grouped(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = rt_md(p, AIEDNN_ARG_SRC), *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    const aiednn_memory_desc_t *d = rt_md(p, AIEDNN_ARG_DST), *b = rt_md(p, AIEDNN_ARG_BIAS);
    const int32_t *gs = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_GROUP_SIZES);
    const int64_t T = s->dims[0], K = s->dims[1], E = w->dims[0], N = d->dims[1];
    int64_t off = 0;
    for (int64_t e = 0; e < E; e++) {
        const int64_t rows = gs[e];
        if (rows < 0 || off + rows > T)
            return AIEK_ERR_PARAM;
        if (!rows)
            continue;
        rt_prob q = *p; /* the group's matmul: [rows, K] x [K, N] (+ bias [1, N]) */
        q.kind = aiednn_matmul;
        q.nmd = 0;
        aiednn_memory_desc_t ms = *s, mw = *w, md_ = *d, mb;
        ms.dims[0] = rows;
        md_.dims[0] = rows;
        mw.ndims = 2;
        mw.dims[0] = w->dims[1], mw.dims[1] = w->dims[2];
        prob_add(&q, AIEDNN_ARG_SRC, &ms);
        prob_add(&q, AIEDNN_ARG_WEIGHTS, &mw);
        prob_add(&q, AIEDNN_ARG_DST, &md_);
        void *sub[4] = {(char *)ptr[rt_slot(p, AIEDNN_ARG_SRC)] + dt_size(s->data_type, off * K),
                        (char *)ptr[rt_slot(p, AIEDNN_ARG_WEIGHTS)] + dt_size(w->data_type, e * K * N),
                        (char *)ptr[rt_slot(p, AIEDNN_ARG_DST)] + dt_size(d->data_type, off * N), nullptr};
        if (b) {
            mb = *b;
            mb.dims[0] = 1;
            prob_add(&q, AIEDNN_ARG_BIAS, &mb);
            sub[3] = (char *)ptr[rt_slot(p, AIEDNN_ARG_BIAS)] + dt_size(b->data_type, e * N);
        }
        aiednn_primitive sp;
        memset(&sp, 0, sizeof sp);
        sp.prob = q;
        if (dispatch(&q, &sp.impl, &sp.bind, 1) != aiednn_success)
            return AIEK_ERR_PARAM;
        const int rc = sp.impl->run(sp.impl, &sp, sub);
        if (rc != AIEK_OK)
            return rc;
        off += rows;
    }
    return AIEK_OK;
}

extern "C" aiednn_status_t aiednn_grouped_matmul_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                        const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
                                                        const aiednn_memory_desc_t *d, int transposed,
                                                        const aiednn_attr_t *attr) {
    const char *k = "grouped_matmul";
    REQUIRE(pr && md_ok(s) && md_ok(w) && md_ok(d) && (!b || md_ok(b)), k, "null or malformed descriptor");
    REQUIRE(s->ndims == 2 && w->ndims == 3 && d->ndims == 2, k, "src [T, K], weights [E, K, N], dst [T, N]");
    const int64_t T = s->dims[0], K = s->dims[1], E = w->dims[0], N = d->dims[1];
    REQUIRE(d->dims[0] == T &&
                (transposed ? (w->dims[1] == N && w->dims[2] == K) : (w->dims[1] == K && w->dims[2] == N)),
            k, "weights [E, K, N] ([E, N, K] when transposed), dst [T, N]");
    REQUIRE(!b || (b->ndims == 2 && b->dims[0] == E && b->dims[1] == N), k, "bias must be [E, N]");
    REQUIRE(gemm_types_ok(s, w, b, d, 1, attr != nullptr), k, "types: as matmul");
    REQUIRE(!attr || !attr->n_post_ops || attr->post_ops[0].kind == aiednn_post_op_eltwise, k,
            "attr: scales, dst quantization and eltwise post-ops (no binary post-op)");
    for (int i = 0; attr && i < attr->n_post_ops; i++)
        REQUIRE(attr->post_ops[i].kind == aiednn_post_op_eltwise, k, "no binary post-op");
    rt_prob p;
    prob_init(&p, aiednn_grouped_matmul, 0);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    prob_add(&p, AIEDNN_ARG_DST, d);
    if (b)
        prob_add(&p, AIEDNN_ARG_BIAS, b);
    aiednn_memory_desc_t g;
    memset(&g, 0, sizeof g);
    g.ndims = 1, g.dims[0] = E, g.data_type = aiednn_s32;
    prob_add(&p, AIEDNN_ARG_GROUP_SIZES, &g);
    p.transposed = transposed != 0;
    const char *why = attach_attr(&p, attr, 1);
    if (why) {
        free((void *)p.oscales);
        return bad(k, why);
    }
    return create(pr, &p);
}

#endif /* AIEDNN_RT_GROUPED_H */
