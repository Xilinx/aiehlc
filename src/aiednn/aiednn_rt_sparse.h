/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_sparse.h -- matmul with a CSR-encoded sparse src: dst [M, N] = A [M, K] x W [K, N] (+ bias), A given
 * as values [nnz], column indices [nnz] s32 and row pointers [M + 1] s32 (SRC, SRC_INDICES, SRC_POINTERS).
 *
 * Two implementations, both exact for integers:
 *   densify+<matmul impl> -- scatter A into a dense [M, K] block and run the matmul dispatch on it.
 *   ref:sparse_matmul     -- per row, each nonzero times its weight row (double accumulators).
 * Under aiednn_prefer_fastest densify+ refuses below RT_SPARSE_DENSIFY_MIN density (nnz / (M K)).
 *
 * Needs, from the including TU: aiednn_rt_internal.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_SPARSE_H
#define AIEDNN_RT_SPARSE_H

#ifndef RT_SPARSE_DENSIFY_MIN
#define RT_SPARSE_DENSIFY_MIN 0.05 /* below: the host's nnz x N loop beats a dense GEMM over mostly zeros */
#endif

typedef struct {
    int64_t M, K, N, nnz;
} sp_geo;

static void sp_geo_of(const rt_prob *p, sp_geo *g) {
    const aiednn_memory_desc_t *w = rt_md(p, AIEDNN_ARG_WEIGHTS), *rp = rt_md(p, AIEDNN_ARG_SRC_POINTERS);
    g->M = rp->dims[0] - 1, g->K = w->dims[0], g->N = w->dims[1], g->nnz = p->md[0].dims[0];
}

/* the dense matmul A [M, K] (src type) x W -> dst over the problem's own weights / bias / dst slots */
static void sp_dense_prob(const rt_prob *p, rt_prob *m) {
    sp_geo g;
    sp_geo_of(p, &g);
    aiednn_memory_desc_t a;
    md2(&a, g.M, g.K, p->md[0].data_type);
    prob_init(m, aiednn_matmul, 0);
    prob_add(m, AIEDNN_ARG_SRC, &a);
    prob_add(m, AIEDNN_ARG_WEIGHTS, rt_md(p, AIEDNN_ARG_WEIGHTS));
    prob_add(m, AIEDNN_ARG_DST, rt_md(p, AIEDNN_ARG_DST));
    if (rt_md(p, AIEDNN_ARG_BIAS))
        prob_add(m, AIEDNN_ARG_BIAS, rt_md(p, AIEDNN_ARG_BIAS));
}

/* row pointers non-decreasing from 0 to nnz, column indices in [0, K) */
static int sp_csr_ok(const sp_geo *g, const int32_t *ix, const int32_t *rp) {
    if (rp[0] != 0 || rp[g->M] != g->nnz)
        return 0;
    for (int64_t i = 0; i < g->M; i++)
        if (rp[i + 1] < rp[i])
            return 0;
    for (int64_t z = 0; z < g->nnz; z++)
        if (ix[z] < 0 || ix[z] >= g->K)
            return 0;
    return 1;
}

static int acc_sp_densify(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    sp_geo g;
    sp_geo_of(p, &g);
    if (rt_cost_refusals() && (double)g.nnz < RT_SPARSE_DENSIFY_MIN * (double)g.M * (double)g.K) {
        snprintf(why, cap, "density %.3f: the sparse host loop is faster (aiednn_prefer_fastest)",
                 (double)g.nnz / ((double)g.M * (double)g.K));
        return 0;
    }
    rt_prob m;
    sp_dense_prob(p, &m);
    const rt_impl *in = nullptr;
    rt_bind b = {nullptr, nullptr};
    if (dispatch(&m, &in, &b, 1) != aiednn_success || in->specificity < 0) {
        snprintf(why, cap, "its dense matmul would run on %s", in ? in->name : "nothing");
        return 0;
    }
    bind->inner = in, bind->blk = b.blk, bind->inner2 = b.inner;
    return 1;
}

static void *g_spa;
static size_t g_spa_cap;

static int run_sp_densify(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    sp_geo g;
    sp_geo_of(p, &g);
    const int32_t *ix = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_SRC_INDICES);
    const int32_t *rp = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_SRC_POINTERS);
    if (!sp_csr_ok(&g, ix, rp))
        return AIEK_ERR_ARGS;
    const int64_t es = dt_bytes(p->md[0].data_type);
    char *a = (char *)scratch(&g_spa, &g_spa_cap, (size_t)(g.M * g.K * es));
    if (!a)
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = rt_ticks();
    memset(a, 0, (size_t)(g.M * g.K * es));
    for (int64_t i = 0; i < g.M; i++) /* duplicates add, as the sparse loop does */
        for (int64_t z = rp[i]; z < rp[i + 1]; z++)
            rt_st(p->md[0].data_type, a, i * g.K + ix[z],
                  rt_ld(p->md[0].data_type, a, i * g.K + ix[z]) + rt_ld(p->md[0].data_type, ptr[0], z));
    g_phase[RT_PH_APACK] += rt_ticks() - t0;
    rt_prob m;
    sp_dense_prob(p, &m);
    void *mp[4] = {a, rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS), rt_ptr(p, ptr, AIEDNN_ARG_DST),
                   rt_ptr(p, ptr, AIEDNN_ARG_BIAS)};
    const rt_bind b = {prim->bind.inner2, prim->bind.blk};
    return run_sub(prim->bind.inner, &m, &b, mp);
}

/* per row: dst[i, :] = sum over its nonzeros v * W[k, :] (+ bias), double accumulators */
static int run_ref_sparse_mm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    sp_geo g;
    sp_geo_of(p, &g);
    const int32_t *ix = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_SRC_INDICES);
    const int32_t *rp = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_SRC_POINTERS);
    if (!sp_csr_ok(&g, ix, rp))
        return AIEK_ERR_ARGS;
    const aiednn_memory_desc_t *w = rt_md(p, AIEDNN_ARG_WEIGHTS), *d = rt_md(p, AIEDNN_ARG_DST);
    const aiednn_memory_desc_t *bm = rt_md(p, AIEDNN_ARG_BIAS);
    const void *W = rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS), *B = rt_ptr(p, ptr, AIEDNN_ARG_BIAS);
    double *acc = (double *)rt_alloc((size_t)g.N * sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    for (int64_t i = 0; i < g.M; i++) {
        for (int64_t j = 0; j < g.N; j++)
            acc[j] = 0.0;
        for (int64_t z = rp[i]; z < rp[i + 1]; z++) {
            const double v = rt_ld(p->md[0].data_type, ptr[0], z);
            for (int64_t j = 0; j < g.N; j++)
                acc[j] += v * rt_ld(w->data_type, W, ix[z] * g.N + j);
        }
        for (int64_t j = 0; j < g.N; j++) {
            const double bv = bm ? rt_ld(bm->data_type, B, bcast_off(bm, d, i * g.N + j)) : 0.0;
            rt_st(d->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DST), i * g.N + j, acc[j] + bv);
        }
    }
    rt_free(acc);
    return AIEK_OK;
}

extern "C" aiednn_status_t aiednn_sparse_matmul_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *values,
                                                       int64_t M, int64_t K, const aiednn_memory_desc_t *weights,
                                                       const aiednn_memory_desc_t *bias,
                                                       const aiednn_memory_desc_t *dst) {
    const char *k = "sparse_matmul";
    REQUIRE(pr && md_ok(values) && md_ok(weights) && md_ok(dst) && (!bias || md_ok(bias)) && M >= 1 && K >= 1, k,
            "null or malformed descriptor");
    REQUIRE(values->ndims == 1 && weights->ndims == 2 && dst->ndims == 2, k,
            "values [nnz], weights [K, N], dst [M, N]");
    REQUIRE(weights->dims[0] == K && dst->dims[0] == M && dst->dims[1] == weights->dims[1], k,
            "weights [K, N], dst [M, N]");
    REQUIRE(values->dims[0] <= M * K, k, "nnz <= M x K");
    aiednn_memory_desc_t a;
    md2(&a, M, K, values->data_type);
    REQUIRE(gemm_types_ok(&a, weights, bias, dst, 0, 0), k, "types: s8|u8 x s8 -> s32 (bias s32), or f32");
    REQUIRE(!bias || md_bcast_ok(bias, dst), k, "bias must broadcast to dst (rank 2)");
    aiednn_memory_desc_t ix, rp;
    vec_md(&ix, values->dims[0]);
    ix.data_type = aiednn_s32;
    vec_md(&rp, M + 1);
    rp.data_type = aiednn_s32;
    rt_prob p;
    prob_init(&p, aiednn_sparse_matmul, 0);
    prob_add(&p, AIEDNN_ARG_SRC, values);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, weights);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    prob_add(&p, AIEDNN_ARG_SRC_INDICES, &ix);
    prob_add(&p, AIEDNN_ARG_SRC_POINTERS, &rp);
    if (bias)
        prob_add(&p, AIEDNN_ARG_BIAS, bias);
    return create(pr, &p);
}

#endif /* AIEDNN_RT_SPARSE_H */
