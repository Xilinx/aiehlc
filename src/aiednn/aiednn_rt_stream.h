/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_stream.h -- streams, caller-owned workspace, async execute, find / set_impl.
 *
 *   workspace   an execute's temporaries come from the caller's workspace through the rt_alloc arena
 *               (any 64-byte aligned host memory; the AIE never reads it). Its size is the arena's
 *               high-water mark of one sizing run, cached per implementation.
 *   streams     in order; each execute runs at enqueue, wait returns the first error.
 *   find        every acceptor, by dispatch rank (HEURISTIC) or timed on library-owned buffers
 *               (MEASURE); the fastest allowed one goes into the autotune table.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_tune.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_STREAM_H
#define AIEDNN_RT_STREAM_H

struct aiednn_stream {
    unsigned flags;
    aiednn_status_t err; /* the first failure since the last wait */
};

extern "C" aiednn_status_t aiednn_stream_create(aiednn_stream_t *stream, unsigned flags) {
    if (!stream || flags)
        return aiednn_invalid_arguments;
    *stream = (aiednn_stream_t)calloc(1, sizeof(struct aiednn_stream));
    return *stream ? aiednn_success : aiednn_out_of_memory;
}

extern "C" aiednn_status_t aiednn_stream_wait(aiednn_stream_t stream) {
    if (!stream)
        return aiednn_invalid_arguments;
    const aiednn_status_t e = stream->err;
    stream->err = aiednn_success;
    return e;
}

extern "C" void aiednn_stream_destroy(aiednn_stream_t stream) {
    if (stream)
        aiednn_stream_wait(stream), free(stream);
}

/* library-owned buffers for every argument slot of p (a pattern of small values: no NaN / inf, a
 * nonzero divisor), or nullptr; free with rt_owned_free */
static void **rt_owned_args(const rt_prob *p) {
    void **ptr = (void **)calloc(RT_MAX_MD, sizeof(void *));
    for (int s = 0; ptr && s < p->nmd; s++) {
        const aiednn_memory_desc_t *m = &p->md[s];
        const int64_t n = md_numel(m);
        ptr[s] = malloc((size_t)dt_size(m->data_type, n) + 64);
        if (!ptr[s]) {
            for (int t = 0; t < s; t++)
                free(ptr[t]);
            free(ptr);
            return nullptr;
        }
        const int intish = dt_is_int(m->data_type) && m->data_type != aiednn_s32;
        for (int64_t i = 0; i < n; i++) /* s32 operands (group sizes, labels, lengths) stay 0 / 1 */
            rt_st(m->data_type, ptr[s], i,
                  m->data_type == aiednn_s32 ? (double)((i % 2) != 0)
                  : intish                   ? (double)((i * 7) % 11 - 5)
                                             : 0.25 + 0.5 * (double)((i * 13) % 7) / 7.0);
    }
    return ptr;
}
static void rt_owned_free(const rt_prob *p, void **ptr) {
    for (int s = 0; ptr && s < p->nmd; s++)
        free(ptr[s]);
    free(ptr);
}

/* one run of implementation (impl, bind) of prim on ptr, with the arena in sizing mode: rc, the
 * workspace it needed and its time */
static int rt_sized_run(aiednn_primitive *prim, const rt_impl *impl, rt_bind bind, void *const *ptr, size_t *ws,
                        double *ms) {
    const rt_impl *keep = prim->impl;
    const rt_bind kb = prim->bind;
    prim->impl = impl, prim->bind = bind;
    g_ws = rt_ws_t{nullptr, 0, 0, 0, 1, 0};
    const uint64_t t0 = rt_ticks();
    const int rc = impl->run(impl, prim, ptr); /* constant-weight caches stay off: these are not the model's */
    *ms = rt_ticks_ms(rt_ticks() - t0);
    *ws = g_ws.high;
    g_ws = rt_ws_t{};
    prim->impl = keep, prim->bind = kb;
    return rc;
}

extern "C" size_t aiednn_primitive_get_workspace_size(aiednn_primitive_t prim) {
    if (!prim || !prim->in_use || !prim->impl)
        return 0;
    if (prim->ws_impl == prim->impl)
        return prim->ws_size;
    void **ptr = rt_owned_args(&prim->prob);
    size_t ws = 0;
    double ms;
    if (ptr && rt_sized_run(prim, prim->impl, prim->bind, ptr, &ws, &ms) == AIEK_OK)
        prim->ws_impl = prim->impl, prim->ws_size = ws;
    rt_owned_free(&prim->prob, ptr);
    return prim->ws_impl == prim->impl ? prim->ws_size : 0;
}

extern "C" aiednn_status_t aiednn_primitive_execute_async(aiednn_primitive_t prim, aiednn_stream_t stream, int nargs,
                                                          const aiednn_exec_arg_t *args, void *workspace,
                                                          size_t workspace_size) {
    if (workspace && ((uintptr_t)workspace % RT_WS_ALIGN))
        return aiednn_invalid_arguments;
    if (workspace)
        g_ws = rt_ws_t{(char *)workspace, workspace_size, 0, 0, 0, 0};
    aiednn_status_t st = aiednn_primitive_execute_args(prim, nargs, args);
    if (workspace && g_ws.overflow && st == aiednn_success)
        st = aiednn_out_of_memory; /* the workspace was smaller than get_workspace_size said to give */
    g_ws = rt_ws_t{};
    if (stream && stream->err == aiednn_success)
        stream->err = st;
    return st;
}

static int rt_perf_cmp(const void *a, const void *b) {
    const aiednn_impl_perf_t *x = (const aiednn_impl_perf_t *)a, *y = (const aiednn_impl_perf_t *)b;
    const int xok = x->status == aiednn_success && x->time_us >= 0,
              yok = y->status == aiednn_success && y->time_us >= 0;
    if (xok != yok)
        return yok - xok;
    return x->time_us < y->time_us ? -1 : x->time_us > y->time_us;
}

extern "C" int aiednn_primitive_find(aiednn_primitive_t prim, int mode, aiednn_impl_perf_t *out, int cap) {
    if (!prim || !prim->in_use)
        return 0;
    int order[RT_TUNE_MAX], n = prim->ncand;
    for (int i = 0; i < n; i++) /* dispatch rank: most specific first */
        order[i] = i;
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && prim->cand[order[j]]->specificity > prim->cand[order[j - 1]]->specificity; j--) {
            const int t = order[j];
            order[j] = order[j - 1], order[j - 1] = t;
        }
    aiednn_impl_perf_t all[RT_TUNE_MAX];
    void **ptr = mode == AIEDNN_FIND_MEASURE ? rt_owned_args(&prim->prob) : nullptr;
    for (int i = 0; i < n; i++) {
        const int c = order[i];
        aiednn_impl_perf_t *r = &all[i];
        memset(r, 0, sizeof *r);
        snprintf(r->impl, sizeof r->impl, "%s", prim->cand[c]->name);
        r->on_aie = prim->cand[c]->specificity >= 0;
        r->time_us = -1;
        r->status = aiednn_success;
        if (prim->cand[c] == prim->ws_impl)
            r->workspace_size = prim->ws_size;
        if (mode != AIEDNN_FIND_MEASURE)
            continue;
        double ms = 0;
        size_t ws = 0;
        int rc = ptr ? AIEK_OK : AIEK_ERR_ALLOC;
        for (int rep = 0; rep < 2 && rc == AIEK_OK; rep++) /* the second run is timed: tables built, ELF loaded */
            rc = rt_sized_run(prim, prim->cand[c], prim->cbind[c], ptr, &ws, &ms);
        r->status = rc == AIEK_OK ? aiednn_success : rc == AIEK_ERR_ALLOC ? aiednn_out_of_memory : aiednn_runtime_error;
        r->time_us = rc == AIEK_OK ? (float)(ms * 1e3) : -1.0f;
        r->workspace_size = ws;
    }
    rt_owned_free(&prim->prob, ptr);
    if (mode == AIEDNN_FIND_MEASURE) {
        qsort(all, (size_t)n, sizeof all[0], rt_perf_cmp);
        for (int i = 0; i < n; i++) /* the autotune table: the fastest the preference allows */
            for (int c = 0; c < prim->ncand; c++)
                if (!strcmp(prim->cand[c]->name, all[i].impl) && all[i].status == aiednn_success &&
                    rt_cand_allowed(prim, c)) {
                    rt_tune_put(prim->sig, prim->prob.kind, all[i].impl);
                    i = n;
                    break;
                }
    }
    for (int i = 0; i < n && i < cap && out; i++)
        out[i] = all[i];
    return n;
}

extern "C" aiednn_status_t aiednn_primitive_set_impl(aiednn_primitive_t prim, const char *impl) {
    if (!prim || !prim->in_use || !impl)
        return aiednn_invalid_arguments;
    for (int c = 0; c < prim->ncand; c++)
        if (!strcmp(prim->cand[c]->name, impl)) {
            prim->impl = prim->cand[c], prim->bind = prim->cbind[c];
            prim->tuned = 1; /* pinned: the autotuner leaves it */
            rt_prim_info(prim);
            return aiednn_success;
        }
    return aiednn_unimplemented; /* not one of the implementations that accepted the problem */
}

#endif /* AIEDNN_RT_STREAM_H */
