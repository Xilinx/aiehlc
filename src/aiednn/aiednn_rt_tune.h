/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_tune.h -- measured implementation choice (autotuning).
 *
 * Dispatch ranks the implementations that accept a problem by specificity (exact-shape kernel >
 * composite > ref:). That says what *fits*, not what is *fastest*: a composite and ref: can both
 * accept, and which one wins depends on the shape (a global pool loses on the AIE, a 3x3 conv wins
 * 25x). With aiednn_set_autotune(1), the first execute of a problem signature runs every accepting
 * implementation (create records them, up to RT_TUNE_MAX), times each on its second run, keeps the
 * fastest for the primitive and records the choice per signature; later primitives with the same
 * signature take it at create, without timing. Every implementation is bit-exact against ref: (or
 * within its declared tolerance), so whichever ran last left a valid dst.
 *
 * aiednn_autotune_dump() prints the table ("AIEDNN_TUNE <sig> <kind> <impl> <us> | <impl> <us> ..."),
 * aiednn_autotune_load(text) preloads one (a board run's dump), so an app ships tuned choices.
 *
 * Not tuned (the dispatch choice runs): a dst that overlaps an input (an in-place primitive --
 * running it twice would read its own output), or a single acceptor.
 ******************************************************************************/
#ifndef AIEDNN_RT_TUNE_H
#define AIEDNN_RT_TUNE_H

#define RT_TUNE_SLOTS 512

typedef struct {
    uint64_t sig;
    int kind;
    char impl[64];
} rt_tune_entry;

static rt_tune_entry g_tune_tab[RT_TUNE_SLOTS];
static int g_tune_n, g_tune;

/* problem signature: kind, algorithm, every memory descriptor's type and dims, conv / pool
 * geometry, the epilogue's shape -- what decides which implementation is fastest, no pointers */
static uint64_t rt_tune_sig(const rt_prob *p) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](int64_t v) {
        for (int i = 0; i < 8; i++)
            h = (h ^ (uint64_t)((v >> (8 * i)) & 0xff)) * 1099511628211ULL;
    };
    mix(p->kind), mix(p->prop), mix(p->alg), mix(p->nmd), mix(p->flags), mix(p->axis), mix(p->transposed),
        mix(p->groups);
    for (int s = 0; s < p->nmd; s++) {
        mix(p->arg[s]), mix(p->md[s].data_type), mix(p->md[s].ndims);
        for (int d = 0; d < p->md[s].ndims; d++)
            mix(p->md[s].dims[d]);
    }
    for (int i = 0; i < 2; i++) {
        mix(p->conv.strides[i]), mix(p->conv.padding_l[i]), mix(p->conv.padding_r[i]), mix(p->conv.dilates[i]);
        mix(p->pool.kernel[i]), mix(p->pool.strides[i]), mix(p->pool.padding_l[i]), mix(p->pool.padding_r[i]);
    }
    mix(p->conv.groups), mix(p->has_attr), mix(p->n_oscales > 1), mix(p->npo);
    for (int k = 0; k < p->npo; k++)
        mix(p->po[k].kind), mix(p->po[k].alg);
    return h;
}

static const rt_tune_entry *rt_tune_find(uint64_t sig, int kind) {
    for (int i = 0; i < g_tune_n; i++)
        if (g_tune_tab[i].sig == sig && g_tune_tab[i].kind == kind)
            return &g_tune_tab[i];
    return nullptr;
}

static void rt_tune_put(uint64_t sig, int kind, const char *impl) {
    rt_tune_entry *e = (rt_tune_entry *)rt_tune_find(sig, kind);
    if (!e) {
        if (g_tune_n >= RT_TUNE_SLOTS)
            return;
        e = &g_tune_tab[g_tune_n++];
    }
    e->sig = sig, e->kind = kind;
    snprintf(e->impl, sizeof e->impl, "%s", impl);
}

static void rt_prim_info(aiednn_primitive *prim) {
    if (prim->bind.inner)
        snprintf(prim->info, sizeof prim->info, "%s+%s", prim->impl->name, prim->bind.inner->name);
    else
        snprintf(prim->info, sizeof prim->info, "%s", prim->impl->name);
}

/* may candidate c be chosen? under aiednn_prefer_aie a ref: candidate only when no AIE one accepted */
static int rt_cand_allowed(const aiednn_primitive *prim, int c) {
    if (rt_cost_refusals() || prim->cand[c]->specificity >= 0)
        return 1;
    for (int i = 0; i < prim->ncand; i++)
        if (prim->cand[i]->specificity >= 0)
            return 0;
    return 1;
}

/* at create: take the implementation a tuning run chose for this signature, if it accepted */
static void rt_tune_apply(aiednn_primitive *prim) {
    const rt_tune_entry *e = rt_tune_find(prim->sig, prim->prob.kind);
    if (!e)
        return;
    for (int c = 0; c < prim->ncand; c++)
        if (strcmp(prim->cand[c]->name, e->impl) == 0 && rt_cand_allowed(prim, c)) {
            prim->impl = prim->cand[c], prim->bind = prim->cbind[c], prim->tuned = 1;
            return;
        }
}

static int rt_tune_overlaps(const rt_prob *p, void *const *ptr) {
    int d = -1;
    for (int s = 0; s < p->nmd; s++)
        d = p->arg[s] == AIEDNN_ARG_DST ? s : d;
    if (d < 0)
        return 1;
    const char *d0 = (const char *)ptr[d], *d1 = d0 + md_numel(&p->md[d]) * dt_bytes(p->md[d].data_type);
    for (int s = 0; s < p->nmd; s++) {
        if (s == d)
            continue;
        const char *s0 = (const char *)ptr[s], *s1 = s0 + md_numel(&p->md[s]) * dt_bytes(p->md[s].data_type);
        if (s0 < d1 && d0 < s1)
            return 1;
    }
    return 0;
}

/* the first execute: every acceptor twice (the second timed), the fastest kept. RT_NOT_TAKEN: the
 * caller runs the dispatch choice */
static int rt_tune_execute(aiednn_primitive *prim, void *const *ptr) {
    prim->tuned = 1;
    if (rt_tune_overlaps(&prim->prob, ptr))
        return RT_NOT_TAKEN;
    int allowed = 0;
    for (int c = 0; c < prim->ncand; c++)
        allowed += rt_cand_allowed(prim, c);
    if (allowed < 2) /* nothing to choose between: the dispatch choice */
        return RT_NOT_TAKEN;
    double best_ms = 1e300, ms[RT_TUNE_MAX];
    int best = -1, last = -1, rc = AIEK_OK;
    for (int c = 0; c < prim->ncand; c++) {
        ms[c] = -1;
        if (!rt_cand_allowed(prim, c))
            continue;
        last = c;
        g_ws.off = 0; /* a caller workspace serves each candidate from its start */
        prim->impl = prim->cand[c], prim->bind = prim->cbind[c];
        int r = AIEK_OK;
        uint64_t t = 0;
        for (int rep = 0; rep < 2 && r == AIEK_OK; rep++) {
            g_exec_const_w = prim->const_w;
            const uint64_t t0 = rt_ticks();
            r = prim->impl->run(prim->impl, prim, ptr);
            t = rt_ticks() - t0;
            g_exec_const_w = 0;
        }
        if (r != AIEK_OK)
            continue; /* an implementation that fails at run time is never chosen */
        ms[c] = rt_ticks_ms(t);
        if (ms[c] < best_ms)
            best_ms = ms[c], best = c;
    }
    if (best < 0)
        return AIEK_ERR_KERNEL;
    if (best != last) { /* dst holds the last candidate's result: equal, but run the winner */
        prim->impl = prim->cand[best], prim->bind = prim->cbind[best];
        g_exec_const_w = prim->const_w;
        rc = prim->impl->run(prim->impl, prim, ptr);
        g_exec_const_w = 0;
    }
    prim->impl = prim->cand[best], prim->bind = prim->cbind[best];
    rt_prim_info(prim);
    rt_tune_put(prim->sig, prim->prob.kind, prim->impl->name);
    if (g_verbose) {
        printf("aiednn_verbose,autotune,%s,%016llx,chose %s", kind_str(prim->prob.kind), (unsigned long long)prim->sig,
               prim->info);
        for (int c = 0; c < prim->ncand; c++)
            if (ms[c] >= 0)
                printf(" | %s %.3f ms", prim->cand[c]->name, ms[c]);
        printf("\n");
    }
    return rc;
}

extern "C" void aiednn_set_autotune(int on) { g_tune = on != 0; }

extern "C" void aiednn_set_preference(aiednn_preference_t pref) { g_prefer = pref == aiednn_prefer_fastest; }
extern "C" aiednn_preference_t aiednn_get_preference(void) {
    return g_prefer ? aiednn_prefer_fastest : aiednn_prefer_aie;
}

extern "C" int aiednn_primitive_impl_candidates(aiednn_primitive_t prim, char *buf, size_t cap) {
    if (!prim)
        return 0;
    int order[RT_TUNE_MAX], n = prim->ncand;
    for (int i = 0; i < n; i++) /* most specific first, registration order within a rank */
        order[i] = i;
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && prim->cand[order[j]]->specificity > prim->cand[order[j - 1]]->specificity; j--) {
            const int t = order[j];
            order[j] = order[j - 1], order[j - 1] = t;
        }
    size_t used = 0;
    if (buf && cap)
        buf[0] = '\0';
    for (int i = 0; i < n && buf && cap; i++) {
        const int w = snprintf(buf + used, cap - used, "%s%s", i ? "," : "", prim->cand[order[i]]->name);
        if (w < 0 || (size_t)w >= cap - used)
            break;
        used += (size_t)w;
    }
    return n;
}

extern "C" void aiednn_autotune_dump(void) {
    for (int i = 0; i < g_tune_n; i++)
        printf("AIEDNN_TUNE %016llx %d %s\n", (unsigned long long)g_tune_tab[i].sig, g_tune_tab[i].kind,
               g_tune_tab[i].impl);
}

/* lines "AIEDNN_TUNE <sig hex> <kind> <impl>" (aiednn_autotune_dump's format; other lines ignored) */
extern "C" int aiednn_autotune_load(const char *text) {
    int n = 0;
    for (const char *l = text; l && *l;) {
        unsigned long long sig;
        int kind;
        char impl[64];
        if (sscanf(l, "AIEDNN_TUNE %llx %d %63s", &sig, &kind, impl) == 3) {
            rt_tune_put((uint64_t)sig, kind, impl);
            n++;
        }
        const char *nl = strchr(l, '\n');
        l = nl ? nl + 1 : nullptr;
    }
    return n;
}

#endif /* AIEDNN_RT_TUNE_H */
