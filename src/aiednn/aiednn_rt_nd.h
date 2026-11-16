/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_nd.h -- 3-D convolution, deconvolution and pooling (ref:), and the N-D create
 * functions (aiednn_*_create_nd).
 *
 * 1-D problems are rewritten as 2-D with H = 1 and reach the 2-D implementations, the AIE ones included.
 * 3-D problems keep their 5-D descriptors (depth in rt_prob.dpth[]) and run on ref: only.
 *
 * Needs, from the including TU: aiednn_rt_internal.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_ND_H
#define AIEDNN_RT_ND_H

enum { RT_D_S, RT_D_PL, RT_D_PR, RT_D_DIL, RT_D_K };

/* 3-D (de)convolution: src [N, C, D, H, W], weights [O, C/G, KD, KH, KW], dst [N, O, Do, Ho, Wo] */
static int run_ref_conv3d(const rt_prob *p, void *const *ptr, int deconv) {
    rt_epi E;
    rt_epi_init(&E, p, ptr);
    const aiednn_memory_desc_t *s = &p->md[0], *w = &p->md[1], *d = &p->md[2];
    const int64_t N = s->dims[0], C = s->dims[1], D = s->dims[2], H = s->dims[3], W = s->dims[4];
    const int64_t O = d->dims[1], Do = d->dims[2], Ho = d->dims[3], Wo = d->dims[4];
    const int64_t KD = w->dims[2], KH = w->dims[3], KW = w->dims[4], G = p->conv.groups, Cg = C / G, Og = O / G;
    const int64_t sd = p->dpth[RT_D_S], sh = p->conv.strides[0], sw = p->conv.strides[1];
    const int64_t pd = p->dpth[RT_D_PL], ph = p->conv.padding_l[0], pw = p->conv.padding_l[1];
    const int64_t dd = p->dpth[RT_D_DIL] + 1, dh = p->conv.dilates[0] + 1, dw = p->conv.dilates[1] + 1;
    const int64_t nd = md_numel(d);
    double *acc = (double *)rt_alloc((size_t)nd * sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    for (int64_t i = 0; i < nd; i++)
        acc[i] = 0;
    /* conv: dst position o, tap k reads src o * s - pad + k * dil; deconv: src position i, tap k
     * writes dst i * s - pad + k * dil */
    const int64_t Pd = deconv ? D : Do, Ph = deconv ? H : Ho, Pw = deconv ? W : Wo;
    const int64_t Qd = deconv ? Do : D, Qh = deconv ? Ho : H, Qw = deconv ? Wo : W;
    for (int64_t n = 0; n < N; n++)
        for (int64_t o = 0; o < O; o++) {
            const int64_t g = o / Og;
            for (int64_t c = 0; c < Cg; c++) {
                const int64_t ci = g * Cg + c;
                for (int64_t a = 0; a < Pd; a++)
                    for (int64_t b = 0; b < Ph; b++)
                        for (int64_t e = 0; e < Pw; e++)
                            for (int64_t kd = 0; kd < KD; kd++) {
                                const int64_t qd = a * sd - pd + kd * dd;
                                if (qd < 0 || qd >= Qd)
                                    continue;
                                for (int64_t kh = 0; kh < KH; kh++) {
                                    const int64_t qh = b * sh - ph + kh * dh;
                                    if (qh < 0 || qh >= Qh)
                                        continue;
                                    for (int64_t kw = 0; kw < KW; kw++) {
                                        const int64_t qw = e * sw - pw + kw * dw;
                                        if (qw < 0 || qw >= Qw)
                                            continue;
                                        const double wv = rt_ld(w->data_type, ptr[1],
                                                                (((o * Cg + c) * KD + kd) * KH + kh) * KW + kw) -
                                                          p->wei_zp;
                                        if (!deconv)
                                            acc[(((n * O + o) * Do + a) * Ho + b) * Wo + e] +=
                                                wv * (rt_ld(s->data_type, ptr[0],
                                                            (((n * C + ci) * D + qd) * H + qh) * W + qw) -
                                                      p->src_zp);
                                        else
                                            acc[(((n * O + o) * Do + qd) * Ho + qh) * Wo + qw] +=
                                                wv *
                                                (rt_ld(s->data_type, ptr[0], (((n * C + ci) * D + a) * H + b) * W + e) -
                                                 p->src_zp);
                                    }
                                }
                            }
            }
        }
    for (int64_t i = 0; i < nd; i++)
        rt_epi_put(&E, i, (i / (Do * Ho * Wo)) % O, acc[i]);
    rt_free(acc);
    return AIEK_OK;
}

static int run_ref_pool3d(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const int64_t NC = s->dims[0] * s->dims[1], D = s->dims[2], H = s->dims[3], W = s->dims[4];
    const int64_t Do = d->dims[2], Ho = d->dims[3], Wo = d->dims[4];
    const aiednn_pool_params_t *pp = &p->pool;
    const int64_t KD = p->dpth[RT_D_K];
    for (int64_t nc = 0; nc < NC; nc++)
        for (int64_t a = 0; a < Do; a++)
            for (int64_t b = 0; b < Ho; b++)
                for (int64_t e = 0; e < Wo; e++) {
                    double acc = p->alg == aiednn_pooling_max ? -INFINITY : 0.0;
                    int64_t cnt = 0;
                    for (int64_t kd = 0; kd < KD; kd++)
                        for (int64_t kh = 0; kh < pp->kernel[0]; kh++)
                            for (int64_t kw = 0; kw < pp->kernel[1]; kw++) {
                                const int64_t id =
                                    a * p->dpth[RT_D_S] - p->dpth[RT_D_PL] + kd * (p->dpth[RT_D_DIL] + 1);
                                const int64_t ih = b * pp->strides[0] - pp->padding_l[0] + kh * (pp->dilates[0] + 1);
                                const int64_t iw = e * pp->strides[1] - pp->padding_l[1] + kw * (pp->dilates[1] + 1);
                                if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= W)
                                    continue;
                                const double x = rt_ld(s->data_type, ptr[0], ((nc * D + id) * H + ih) * W + iw);
                                acc = p->alg == aiednn_pooling_max ? (x > acc ? x : acc) : acc + x;
                                cnt++;
                            }
                    if (p->alg == aiednn_pooling_avg_include_padding)
                        acc /= (double)(KD * pp->kernel[0] * pp->kernel[1]);
                    else if (p->alg == aiednn_pooling_avg_exclude_padding)
                        acc = cnt ? acc / (double)cnt : 0.0;
                    rt_st(d->data_type, ptr[1], ((nc * Do + a) * Ho + b) * Wo + e, acc);
                }
    return AIEK_OK;
}

/* a descriptor with a unit spatial dim inserted at index 2 (1-D -> 2-D, same bytes) */
static aiednn_memory_desc_t rt_md_unit_h(const aiednn_memory_desc_t *m) {
    aiednn_memory_desc_t r = *m;
    r.ndims = m->ndims + 1;
    r.dims[2] = 1;
    for (int i = 2; i < m->ndims; i++)
        r.dims[i + 1] = m->dims[i];
    return r;
}

static aiednn_status_t conv_nd_create(aiednn_primitive_t *pr, aiednn_primitive_kind_t kind,
                                      const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *w,
                                      const aiednn_memory_desc_t *b, const aiednn_memory_desc_t *d,
                                      const aiednn_conv_nd_params_t *cp, const aiednn_attr_t *attr) {
    const char *k = kind_str(kind);
    REQUIRE(pr && cp && md_ok(s) && md_ok(w) && md_ok(d) && (!b || md_ok(b)), k, "null or malformed descriptor");
    const int sp = s->ndims - 2;
    REQUIRE(sp >= 1 && sp <= 3 && w->ndims == s->ndims && d->ndims == s->ndims, k,
            "src / weights / dst: same rank, 3..5 (N, C, 1..3 spatial)");
    aiednn_conv_params_t c2;
    memset(&c2, 0, sizeof c2);
    c2.groups = cp->groups;
    const int off = sp == 1 ? 1 : sp == 2 ? 0 : -1; /* params index -> 2-D index (H, W) */
    for (int i = 0; i < sp; i++) {
        const int j = i + off;
        if (j < 0)
            continue;
        c2.strides[j] = cp->strides[i], c2.padding_l[j] = cp->padding_l[i], c2.padding_r[j] = cp->padding_r[i];
        c2.dilates[j] = cp->dilates[i];
    }
    if (sp == 1) {
        c2.strides[0] = 1;
        const aiednn_memory_desc_t s4 = rt_md_unit_h(s), w4 = rt_md_unit_h(w), d4 = rt_md_unit_h(d);
        return conv_like_create(pr, kind, &s4, &w4, b, &d4, &c2, attr);
    }
    if (sp == 2)
        return conv_like_create(pr, kind, s, w, b, d, &c2, attr);
    /* 3-D: validated here (conv_like_create is 4-D only), ref: only */
    REQUIRE(gemm_types_ok(s, w, b, d, 0, attr != nullptr), k, "types: as 2-D");
    const int64_t G = cp->groups;
    REQUIRE(G >= 1 && s->dims[1] % G == 0 && d->dims[1] % G == 0, k, "groups must divide C and O");
    REQUIRE(w->dims[0] == d->dims[1] && w->dims[1] == s->dims[1] / G && d->dims[0] == s->dims[0], k,
            "weights must be [O, C/groups, KD, KH, KW], dst [N, O, ...]");
    REQUIRE(!b || (b->ndims == 1 && b->dims[0] == d->dims[1]), k, "bias must be [O]");
    for (int i = 0; i < 3; i++) {
        REQUIRE(cp->strides[i] >= 1 && cp->dilates[i] >= 0 && cp->padding_l[i] >= 0 && cp->padding_r[i] >= 0, k,
                "strides >= 1, dilates and padding >= 0");
        const int64_t ext = (w->dims[2 + i] - 1) * (cp->dilates[i] + 1) + 1;
        const int64_t want = kind == aiednn_convolution
                                 ? (s->dims[2 + i] + cp->padding_l[i] + cp->padding_r[i] - ext) / cp->strides[i] + 1
                                 : (s->dims[2 + i] - 1) * cp->strides[i] - cp->padding_l[i] - cp->padding_r[i] + ext;
        REQUIRE(d->dims[2 + i] == want, k, "dst spatial size does not match strides / padding / dilation");
    }
    rt_prob p;
    prob_init(&p, kind, 0);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    prob_add(&p, AIEDNN_ARG_DST, d);
    if (b)
        prob_add(&p, AIEDNN_ARG_BIAS, b);
    p.conv = c2;
    p.dpth[RT_D_S] = cp->strides[0], p.dpth[RT_D_PL] = cp->padding_l[0], p.dpth[RT_D_PR] = cp->padding_r[0];
    p.dpth[RT_D_DIL] = cp->dilates[0];
    const char *why = attach_attr(&p, attr, 1);
    if (why) {
        free((void *)p.oscales);
        return bad(k, why);
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_convolution_forward_create_nd(
    aiednn_primitive_t *p, const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
    const aiednn_memory_desc_t *d, const aiednn_conv_nd_params_t *cp, const aiednn_attr_t *attr) {
    return conv_nd_create(p, aiednn_convolution, s, w, b, d, cp, attr);
}

extern "C" aiednn_status_t aiednn_deconvolution_forward_create_nd(
    aiednn_primitive_t *p, const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *b,
    const aiednn_memory_desc_t *d, const aiednn_conv_nd_params_t *cp, const aiednn_attr_t *attr) {
    return conv_nd_create(p, aiednn_deconvolution, s, w, b, d, cp, attr);
}

extern "C" aiednn_status_t aiednn_pooling_forward_create_nd(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                            const aiednn_memory_desc_t *s,
                                                            const aiednn_memory_desc_t *d,
                                                            const aiednn_pool_nd_params_t *pp) {
    const char *k = "pooling";
    REQUIRE(pr && pp && md_ok(s) && md_ok(d) && s->ndims == d->ndims && s->ndims >= 3 && s->ndims <= 5, k,
            "src and dst: same rank, 3..5 (N, C, 1..3 spatial)");
    const int sp = s->ndims - 2;
    aiednn_pool_params_t p2;
    memset(&p2, 0, sizeof p2);
    p2.kernel[0] = p2.kernel[1] = 1, p2.strides[0] = p2.strides[1] = 1;
    const int off = sp == 1 ? 1 : sp == 2 ? 0 : -1;
    for (int i = 0; i < sp; i++) {
        const int j = i + off;
        if (j < 0)
            continue;
        p2.kernel[j] = pp->kernel[i], p2.strides[j] = pp->strides[i], p2.padding_l[j] = pp->padding_l[i];
        p2.padding_r[j] = pp->padding_r[i], p2.dilates[j] = pp->dilates[i];
    }
    if (sp == 1) {
        const aiednn_memory_desc_t s4 = rt_md_unit_h(s), d4 = rt_md_unit_h(d);
        return aiednn_pooling_forward_create(pr, alg, &s4, &d4, &p2);
    }
    if (sp == 2)
        return aiednn_pooling_forward_create(pr, alg, s, d, &p2);
    REQUIRE(alg >= aiednn_pooling_max && alg <= aiednn_pooling_avg_exclude_padding, k, "not a pooling algorithm");
    REQUIRE(s->dims[0] == d->dims[0] && s->dims[1] == d->dims[1], k, "N and C must match");
    for (int i = 0; i < 3; i++) {
        REQUIRE(pp->kernel[i] >= 1 && pp->strides[i] >= 1 && pp->dilates[i] >= 0 && pp->padding_l[i] >= 0 &&
                    pp->padding_r[i] >= 0,
                k, "kernel, strides >= 1; dilates, padding >= 0");
        const int64_t ext = (pp->kernel[i] - 1) * (pp->dilates[i] + 1) + 1;
        REQUIRE(d->dims[2 + i] == (s->dims[2 + i] + pp->padding_l[i] + pp->padding_r[i] - ext) / pp->strides[i] + 1, k,
                "dst spatial size does not match kernel / strides / padding");
    }
    rt_prob p;
    prob_init(&p, aiednn_pooling, alg);
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.pool = p2;
    p.dpth[RT_D_S] = pp->strides[0], p.dpth[RT_D_PL] = pp->padding_l[0], p.dpth[RT_D_PR] = pp->padding_r[0];
    p.dpth[RT_D_DIL] = pp->dilates[0], p.dpth[RT_D_K] = pp->kernel[0];
    return create(pr, &p);
}

#endif /* AIEDNN_RT_ND_H */
