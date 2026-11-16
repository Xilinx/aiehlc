/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_bwd_create.h -- the create functions of the training primitives (aiednn.h "training");
 * their run functions are in aiednn_rt_bwd.h. Validation mirrors the forward creates.
 *
 * Needs, from the including TU: the create helpers of aiednn_rt.cc (REQUIRE, md_ok, prob_*, create).
 ******************************************************************************/
#ifndef AIEDNN_RT_BWD_CREATE_H
#define AIEDNN_RT_BWD_CREATE_H

static int all_float(const aiednn_memory_desc_t *const *m, int n) {
    for (int i = 0; i < n; i++)
        if (m[i] && !dt_is_float(m[i]->data_type))
            return 0;
    return 1;
}

/* x: the conv's src side, y: its dst side; NULL or the reason the shapes do not fit */
static const char *conv_nd_shapes(aiednn_primitive_kind_t kind, const aiednn_memory_desc_t *x,
                                  const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *y,
                                  const aiednn_conv_nd_params_t *cp) {
    if (!cp || !md_ok(x) || !md_ok(w) || !md_ok(y))
        return "null or malformed descriptor";
    const int sp = x->ndims - 2;
    if (sp < 1 || sp > 3 || w->ndims != x->ndims || y->ndims != x->ndims)
        return "src / weights / dst: same rank, 3..5 (N, C, 1..3 spatial)";
    const int64_t G = cp->groups;
    if (G < 1 || x->dims[1] % G || y->dims[1] % G)
        return "groups must divide C and O";
    if (w->dims[0] != y->dims[1] || w->dims[1] != x->dims[1] / G || y->dims[0] != x->dims[0])
        return "weights must be [O, C/groups, k...], dst [N, O, ...]";
    for (int i = 0; i < sp; i++) {
        if (cp->strides[i] < 1 || cp->dilates[i] < 0 || cp->padding_l[i] < 0 || cp->padding_r[i] < 0)
            return "strides >= 1, dilates and padding >= 0";
        const int64_t ext = (w->dims[2 + i] - 1) * (cp->dilates[i] + 1) + 1;
        const int64_t want = kind == aiednn_convolution
                                 ? (x->dims[2 + i] + cp->padding_l[i] + cp->padding_r[i] - ext) / cp->strides[i] + 1
                                 : (x->dims[2 + i] - 1) * cp->strides[i] - cp->padding_l[i] - cp->padding_r[i] + ext;
        if (y->dims[2 + i] != want)
            return "dst spatial size does not match strides / padding / dilation";
    }
    return nullptr;
}

static aiednn_status_t conv_bwd_create(aiednn_primitive_t *pr, aiednn_primitive_kind_t kind, int prop,
                                       const aiednn_memory_desc_t *x, const aiednn_memory_desc_t *w,
                                       const aiednn_memory_desc_t *db, const aiednn_memory_desc_t *y,
                                       const aiednn_conv_nd_params_t *cp) {
    const char *k = kind_str(kind);
    REQUIRE(pr, k, "null primitive");
    const char *err = conv_nd_shapes(kind, x, w, y, cp);
    REQUIRE(!err, k, err ? err : "");
    REQUIRE(!db || (md_ok(db) && db->ndims == 1 && db->dims[0] == y->dims[1]), k, "diff_bias must be [O]");
    const aiednn_memory_desc_t *m[4] = {x, w, db, y};
    REQUIRE(all_float(m, 4), k, "training: floating types");
    rt_prob p;
    prob_init(&p, kind, 0);
    p.prop = prop;
    if (prop == RT_BWD_DATA) {
        prob_add(&p, AIEDNN_ARG_DIFF_SRC, x);
        prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    } else {
        prob_add(&p, AIEDNN_ARG_SRC, x);
        prob_add(&p, AIEDNN_ARG_DIFF_WEIGHTS, w);
        if (db)
            prob_add(&p, AIEDNN_ARG_DIFF_BIAS, db);
    }
    prob_add(&p, AIEDNN_ARG_DIFF_DST, y);
    p.cnd = *cp;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_convolution_backward_data_create(aiednn_primitive_t *p,
                                                                   const aiednn_memory_desc_t *dx,
                                                                   const aiednn_memory_desc_t *w,
                                                                   const aiednn_memory_desc_t *dy,
                                                                   const aiednn_conv_nd_params_t *cp) {
    return conv_bwd_create(p, aiednn_convolution, RT_BWD_DATA, dx, w, nullptr, dy, cp);
}
extern "C" aiednn_status_t
aiednn_convolution_backward_weights_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *x,
                                           const aiednn_memory_desc_t *dw, const aiednn_memory_desc_t *db,
                                           const aiednn_memory_desc_t *dy, const aiednn_conv_nd_params_t *cp) {
    return conv_bwd_create(p, aiednn_convolution, RT_BWD_WEIGHTS, x, dw, db, dy, cp);
}
extern "C" aiednn_status_t aiednn_deconvolution_backward_data_create(aiednn_primitive_t *p,
                                                                     const aiednn_memory_desc_t *dx,
                                                                     const aiednn_memory_desc_t *w,
                                                                     const aiednn_memory_desc_t *dy,
                                                                     const aiednn_conv_nd_params_t *cp) {
    return conv_bwd_create(p, aiednn_deconvolution, RT_BWD_DATA, dx, w, nullptr, dy, cp);
}
extern "C" aiednn_status_t
aiednn_deconvolution_backward_weights_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *x,
                                             const aiednn_memory_desc_t *dw, const aiednn_memory_desc_t *db,
                                             const aiednn_memory_desc_t *dy, const aiednn_conv_nd_params_t *cp) {
    return conv_bwd_create(p, aiednn_deconvolution, RT_BWD_WEIGHTS, x, dw, db, dy, cp);
}

static aiednn_status_t ip_bwd_create(aiednn_primitive_t *pr, int prop, const aiednn_memory_desc_t *x,
                                     const aiednn_memory_desc_t *w, const aiednn_memory_desc_t *db,
                                     const aiednn_memory_desc_t *y) {
    const char *k = "inner_product";
    REQUIRE(pr && md_ok(x) && md_ok(w) && md_ok(y) && (!db || md_ok(db)), k, "null or malformed descriptor");
    REQUIRE(y->ndims == 2 && x->ndims == w->ndims && x->ndims >= 2 && y->dims[0] == x->dims[0] &&
                w->dims[0] == y->dims[1] && md_numel(x) / x->dims[0] == md_numel(w) / w->dims[0],
            k, "src [N, IC...], weights [O, IC...], dst [N, O]");
    REQUIRE(!db || (db->ndims == 1 && db->dims[0] == y->dims[1]), k, "diff_bias must be [O]");
    const aiednn_memory_desc_t *m[4] = {x, w, db, y};
    REQUIRE(all_float(m, 4), k, "training: floating types");
    rt_prob p;
    prob_init(&p, aiednn_inner_product, 0);
    p.prop = prop;
    prob_add(&p, prop == RT_BWD_DATA ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC, x);
    prob_add(&p, prop == RT_BWD_DATA ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS, w);
    if (db && prop == RT_BWD_WEIGHTS)
        prob_add(&p, AIEDNN_ARG_DIFF_BIAS, db);
    prob_add(&p, AIEDNN_ARG_DIFF_DST, y);
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_inner_product_backward_data_create(aiednn_primitive_t *p,
                                                                     const aiednn_memory_desc_t *dx,
                                                                     const aiednn_memory_desc_t *w,
                                                                     const aiednn_memory_desc_t *dy) {
    return ip_bwd_create(p, RT_BWD_DATA, dx, w, nullptr, dy);
}
extern "C" aiednn_status_t aiednn_inner_product_backward_weights_create(aiednn_primitive_t *p,
                                                                        const aiednn_memory_desc_t *x,
                                                                        const aiednn_memory_desc_t *dw,
                                                                        const aiednn_memory_desc_t *db,
                                                                        const aiednn_memory_desc_t *dy) {
    return ip_bwd_create(p, RT_BWD_WEIGHTS, x, dw, db, dy);
}

extern "C" aiednn_status_t aiednn_eltwise_backward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, float alpha,
                                                          float beta, const aiednn_memory_desc_t *dx,
                                                          const aiednn_memory_desc_t *dy,
                                                          const aiednn_memory_desc_t *x) {
    const char *k = "eltwise";
    REQUIRE(pr && md_ok(dx) && md_ok(dy) && md_ok(x) && md_same_shape(dx, dy) && md_same_shape(dx, x), k,
            "diff_src, diff_dst, src: same shape");
    REQUIRE(alg >= aiednn_eltwise_relu && alg <= aiednn_eltwise_selu, k, "not an eltwise algorithm");
    const aiednn_memory_desc_t *m[3] = {dx, dy, x};
    REQUIRE(all_float(m, 3), k, "training: floating types");
    rt_prob p;
    prob_init(&p, aiednn_eltwise, alg);
    p.prop = RT_BWD_DATA;
    prob_add(&p, AIEDNN_ARG_SRC, x);
    prob_add(&p, AIEDNN_ARG_DIFF_DST, dy);
    prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    p.alpha = alpha, p.beta = beta;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_softmax_backward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, int axis,
                                                          const aiednn_memory_desc_t *dx,
                                                          const aiednn_memory_desc_t *dy,
                                                          const aiednn_memory_desc_t *y) {
    const char *k = "softmax";
    REQUIRE(pr && md_ok(dx) && md_ok(dy) && md_ok(y) && md_same_shape(dx, dy) && md_same_shape(dx, y), k,
            "diff_src, diff_dst, dst: same shape");
    REQUIRE(alg >= aiednn_softmax_accurate && alg <= aiednn_softmax_accurate_inf_as_zero, k, "not a softmax algorithm");
    REQUIRE(axis >= 0 && axis < y->ndims, k, "axis in range");
    const aiednn_memory_desc_t *m[3] = {dx, dy, y};
    REQUIRE(all_float(m, 3), k, "training: floating types");
    rt_prob p;
    prob_init(&p, aiednn_softmax, alg);
    p.prop = RT_BWD_DATA;
    prob_add(&p, AIEDNN_ARG_DST, y);
    prob_add(&p, AIEDNN_ARG_DIFF_DST, dy);
    prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    p.axis = axis;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_pooling_backward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                          const aiednn_memory_desc_t *dx,
                                                          const aiednn_memory_desc_t *dy, const aiednn_memory_desc_t *x,
                                                          const aiednn_pool_nd_params_t *pp) {
    const char *k = "pooling";
    REQUIRE(pr && pp && md_ok(dx) && md_ok(dy) && dx->ndims == dy->ndims && dx->ndims >= 3 && dx->ndims <= 5, k,
            "diff_src and diff_dst: same rank, 3..5");
    REQUIRE(alg >= aiednn_pooling_max && alg <= aiednn_pooling_avg_exclude_padding, k, "not a pooling algorithm");
    REQUIRE(alg != aiednn_pooling_max || (x && md_ok(x) && md_same_shape(x, dx)), k,
            "max pooling needs src (the forward's input, diff_src's shape)");
    REQUIRE(dx->dims[0] == dy->dims[0] && dx->dims[1] == dy->dims[1], k, "N and C must match");
    for (int i = 0; i < dx->ndims - 2; i++) {
        const int64_t ext = (pp->kernel[i] - 1) * (pp->dilates[i] + 1) + 1;
        REQUIRE(pp->kernel[i] >= 1 && pp->strides[i] >= 1 &&
                    dy->dims[2 + i] ==
                        (dx->dims[2 + i] + pp->padding_l[i] + pp->padding_r[i] - ext) / pp->strides[i] + 1,
                k, "diff_dst spatial size does not match kernel / strides / padding");
    }
    const aiednn_memory_desc_t *m[3] = {dx, dy, alg == aiednn_pooling_max ? x : nullptr};
    REQUIRE(all_float(m, 3), k, "training: floating types");
    rt_prob p;
    prob_init(&p, aiednn_pooling, alg);
    p.prop = RT_BWD_DATA;
    if (alg == aiednn_pooling_max)
        prob_add(&p, AIEDNN_ARG_SRC, x);
    prob_add(&p, AIEDNN_ARG_DIFF_DST, dy);
    prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    p.pnd = *pp;
    return create(pr, &p);
}

/* the statistics' element count: batch norm C, layer norm rows, group norm N x groups */
static int64_t norm_stats_n(aiednn_primitive_kind_t kind, const aiednn_memory_desc_t *s, int64_t groups) {
    return kind == aiednn_batch_normalization   ? s->dims[1]
           : kind == aiednn_layer_normalization ? md_numel(s) / s->dims[s->ndims - 1]
                                                : s->dims[0] * groups;
}

static aiednn_status_t norm_train_create(aiednn_primitive_t *pr, aiednn_primitive_kind_t kind, int prop, int64_t groups,
                                         const aiednn_memory_desc_t *x, const aiednn_memory_desc_t *y,
                                         const aiednn_memory_desc_t *dx, float eps, unsigned flags) {
    const char *k = kind_str(kind);
    REQUIRE(pr && md_ok(x) && md_ok(y) && md_same_shape(x, y) && (!dx || (md_ok(dx) && md_same_shape(x, dx))), k,
            "tensors: same shape");
    REQUIRE(kind == aiednn_layer_normalization || x->ndims >= 2, k, "src must be [N, C, ...]");
    REQUIRE(!(flags & ~(AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_RMS_NORM)), k,
            "flags: scale / shift (/ RMS for layer norm); a fused ReLU is inference only");
    REQUIRE(kind == aiednn_layer_normalization || !(flags & AIEDNN_RMS_NORM), k, "AIEDNN_RMS_NORM is for layer norm");
    const int64_t C = kind == aiednn_layer_normalization ? x->dims[x->ndims - 1] : x->dims[1];
    REQUIRE(kind != aiednn_group_normalization || (groups >= 1 && C % groups == 0), k, "groups must divide C");
    const aiednn_memory_desc_t *m[3] = {x, y, dx};
    REQUIRE(all_float(m, 3), k, "training: floating types");
    rt_prob p;
    prob_init(&p, kind, 0);
    p.prop = prop;
    aiednn_memory_desc_t st, c;
    vec_md(&st, norm_stats_n(kind, x, groups));
    vec_md(&c, C);
    prob_add(&p, AIEDNN_ARG_SRC, x);
    prob_add(&p, AIEDNN_ARG_MEAN, &st);
    prob_add(&p, AIEDNN_ARG_VARIANCE, &st);
    if (flags & AIEDNN_USE_SCALE)
        prob_add(&p, AIEDNN_ARG_SCALE, &c);
    if (prop == RT_FWD_TRAIN) {
        prob_add(&p, AIEDNN_ARG_DST, y);
        if (flags & AIEDNN_USE_SHIFT)
            prob_add(&p, AIEDNN_ARG_SHIFT, &c);
    } else {
        prob_add(&p, AIEDNN_ARG_DIFF_DST, y);
        prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
        if (flags & AIEDNN_USE_SCALE)
            prob_add(&p, AIEDNN_ARG_DIFF_SCALE, &c);
        if (flags & AIEDNN_USE_SHIFT)
            prob_add(&p, AIEDNN_ARG_DIFF_SHIFT, &c);
    }
    p.eps = eps, p.flags = flags, p.groups = groups;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_batch_normalization_forward_training_create(aiednn_primitive_t *p,
                                                                              const aiednn_memory_desc_t *s,
                                                                              const aiednn_memory_desc_t *d, float eps,
                                                                              unsigned flags) {
    return norm_train_create(p, aiednn_batch_normalization, RT_FWD_TRAIN, 0, s, d, nullptr, eps, flags);
}
extern "C" aiednn_status_t aiednn_layer_normalization_forward_training_create(aiednn_primitive_t *p,
                                                                              const aiednn_memory_desc_t *s,
                                                                              const aiednn_memory_desc_t *d, float eps,
                                                                              unsigned flags) {
    return norm_train_create(p, aiednn_layer_normalization, RT_FWD_TRAIN, 0, s, d, nullptr, eps, flags);
}
extern "C" aiednn_status_t aiednn_group_normalization_forward_training_create(aiednn_primitive_t *p, int64_t groups,
                                                                              const aiednn_memory_desc_t *s,
                                                                              const aiednn_memory_desc_t *d, float eps,
                                                                              unsigned flags) {
    return norm_train_create(p, aiednn_group_normalization, RT_FWD_TRAIN, groups, s, d, nullptr, eps, flags);
}
extern "C" aiednn_status_t aiednn_batch_normalization_backward_create(aiednn_primitive_t *p,
                                                                      const aiednn_memory_desc_t *dx,
                                                                      const aiednn_memory_desc_t *dy,
                                                                      const aiednn_memory_desc_t *s, float eps,
                                                                      unsigned flags) {
    return norm_train_create(p, aiednn_batch_normalization, RT_BWD, 0, s, dy, dx, eps, flags);
}
extern "C" aiednn_status_t aiednn_layer_normalization_backward_create(aiednn_primitive_t *p,
                                                                      const aiednn_memory_desc_t *dx,
                                                                      const aiednn_memory_desc_t *dy,
                                                                      const aiednn_memory_desc_t *s, float eps,
                                                                      unsigned flags) {
    return norm_train_create(p, aiednn_layer_normalization, RT_BWD, 0, s, dy, dx, eps, flags);
}
extern "C" aiednn_status_t aiednn_group_normalization_backward_create(aiednn_primitive_t *p, int64_t groups,
                                                                      const aiednn_memory_desc_t *dx,
                                                                      const aiednn_memory_desc_t *dy,
                                                                      const aiednn_memory_desc_t *s, float eps,
                                                                      unsigned flags) {
    return norm_train_create(p, aiednn_group_normalization, RT_BWD, groups, s, dy, dx, eps, flags);
}

extern "C" aiednn_status_t aiednn_prelu_backward_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *dx,
                                                        const aiednn_memory_desc_t *dw, const aiednn_memory_desc_t *dy,
                                                        const aiednn_memory_desc_t *x, const aiednn_memory_desc_t *w) {
    const char *k = "prelu";
    REQUIRE(pr && md_ok(dx) && md_ok(dw) && md_ok(dy) && md_ok(x) && md_ok(w) && md_same_shape(dx, x) &&
                md_same_shape(dy, x) && md_same_shape(dw, w) && w->ndims == x->ndims,
            k, "diff_src / diff_dst like src, diff_weights like weights (broadcastable to src)");
    for (int i = 0; i < x->ndims; i++)
        REQUIRE(w->dims[i] == 1 || w->dims[i] == x->dims[i], k, "weights must broadcast to src");
    const aiednn_memory_desc_t *m[5] = {dx, dw, dy, x, w};
    REQUIRE(all_float(m, 5), k, "training: floating types");
    rt_prob p;
    prob_init(&p, aiednn_prelu, 0);
    p.prop = RT_BWD;
    prob_add(&p, AIEDNN_ARG_SRC, x);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, w);
    prob_add(&p, AIEDNN_ARG_DIFF_DST, dy);
    prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    prob_add(&p, AIEDNN_ARG_DIFF_WEIGHTS, dw);
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_lrn_backward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg, int64_t local_size,
                                                      float alpha, float beta, float kk, const aiednn_memory_desc_t *dx,
                                                      const aiednn_memory_desc_t *dy, const aiednn_memory_desc_t *x) {
    const char *k = "lrn";
    REQUIRE(pr && md_ok(dx) && md_ok(dy) && md_ok(x) && md_same_shape(dx, x) && md_same_shape(dy, x) && x->ndims >= 2 &&
                x->ndims <= 4,
            k, "diff_src, diff_dst, src: same shape, rank 2..4");
    REQUIRE(alg == aiednn_lrn_across_channels || alg == aiednn_lrn_within_channel, k, "not an lrn algorithm");
    const aiednn_memory_desc_t *m[3] = {dx, dy, x};
    REQUIRE(local_size >= 1 && all_float(m, 3), k, "local_size >= 1, floating types");
    rt_prob p;
    prob_init(&p, aiednn_lrn, alg);
    p.prop = RT_BWD_DATA;
    prob_add(&p, AIEDNN_ARG_SRC, x);
    prob_add(&p, AIEDNN_ARG_DIFF_DST, dy);
    prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    p.local_size = local_size, p.alpha = alpha, p.beta = beta, p.lrn_k = kk;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_resampling_backward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t alg,
                                                             const aiednn_memory_desc_t *dx,
                                                             const aiednn_memory_desc_t *dy) {
    const char *k = "resampling";
    REQUIRE(pr && md_ok(dx) && md_ok(dy) && dx->ndims == dy->ndims && dx->ndims >= 3 && dx->ndims <= 5, k,
            "diff_src and diff_dst: same rank, 3..5");
    REQUIRE(alg == aiednn_resampling_nearest || alg == aiednn_resampling_linear, k, "nearest or linear");
    REQUIRE(dx->dims[0] == dy->dims[0] && dx->dims[1] == dy->dims[1], k, "N and C must match");
    const aiednn_memory_desc_t *m[2] = {dx, dy};
    REQUIRE(all_float(m, 2), k, "training: floating types");
    rt_prob p;
    prob_init(&p, aiednn_resampling, alg);
    p.prop = RT_BWD_DATA;
    prob_add(&p, AIEDNN_ARG_DIFF_DST, dy);
    prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_shuffle_backward_create(aiednn_primitive_t *pr, int axis, int64_t group_size,
                                                          const aiednn_memory_desc_t *dx,
                                                          const aiednn_memory_desc_t *dy) {
    const char *k = "shuffle";
    REQUIRE(pr && md_ok(dx) && md_ok(dy) && md_same_shape(dx, dy) && axis >= 0 && axis < dx->ndims, k,
            "diff_src and diff_dst: same shape, axis in range");
    REQUIRE(group_size >= 1 && dx->dims[axis] % group_size == 0, k, "group_size must divide the axis");
    rt_prob p;
    prob_init(&p, aiednn_shuffle, 0);
    p.prop = RT_BWD_DATA;
    prob_add(&p, AIEDNN_ARG_DIFF_DST, dy);
    prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    p.axis = axis, p.group_size = group_size;
    return create(pr, &p);
}

#endif /* AIEDNN_RT_BWD_CREATE_H */
