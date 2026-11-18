/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* GAP-RT1: per-call-site split of the A78 time of an AIE offload site ("aie_host") into
 *   pack    - host operand tiling / bf16 packing before the first launch (im2col, akp_pack, ...)
 *   scatter - host output un-tiling / epilogue after the launches (uc_scatter16, akp_unpad, ...)
 *   dma     - launch windows minus the runtime's wait_io (cache sync, BD config, start, core enable)
 *   wait    - the runtime's __Runtime_wait_io poll inside the launch windows
 * Header-only (C and C++, no xaiengine dependency): x86 unit test src/mlir/runtime/test/test_host_split.py.
 * Ticks are whatever clock the caller uses (XTime on the A78); conversion to ms takes its rate.
 * The table is zero-initialisable (a static ahs_table needs no init call). */
#ifndef AIE_HOST_SPLIT_H
#define AIE_HOST_SPLIT_H

#include <stddef.h>
#include <stdio.h>

#ifndef AHS_MAX_SITES
#define AHS_MAX_SITES 1024
#endif

enum { AHS_PACK = 0, AHS_SCATTER = 1, AHS_DMA = 2, AHS_WAIT = 3, AHS_NPH = 4 };
enum { AHS_KIND_NONE = 0, AHS_KIND_CONV = 1, AHS_KIND_LINEAR = 2, AHS_KIND_ATTN = 3 };

typedef struct {
    unsigned long long t[AHS_NPH]; /* ticks per phase */
    unsigned calls;                /* site entries (graph_run calls) */
    int kind;                      /* AHS_KIND_* of the last entry */
} ahs_site;

typedef struct {
    ahs_site s[AHS_MAX_SITES];
    int cur1; /* current site + 1; 0 = no site (adds are dropped) */
} ahs_table;

/* free-running tick counter of the runtime side (A78 generic timer = XTime's counter); 0 where there is none */
static inline unsigned long long ahs_now(void) {
    unsigned long long v = 0ULL;
#if defined(__aarch64__) && !defined(__AIESIM__) && !defined(__chess__)
#if defined(__linux__)
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
#else
    __asm__ volatile("mrs %0, cntpct_el0" : "=r"(v));
#endif
#endif
    return v;
}

static inline void ahs_set_site(ahs_table *a, int site) { a->cur1 = (site >= 0 && site < AHS_MAX_SITES) ? site + 1 : 0; }

static inline ahs_site *ahs_cur(ahs_table *a) { return a->cur1 > 0 ? &a->s[a->cur1 - 1] : (ahs_site *)0; }

/* entry of an offload site of kind k (counts the call; the site index comes from ahs_set_site) */
static inline void ahs_begin(ahs_table *a, int kind) {
    ahs_site *s = ahs_cur(a);
    if (s) {
        s->kind = kind;
        s->calls++;
    }
}

static inline void ahs_add(ahs_table *a, int ph, unsigned long long ticks) {
    ahs_site *s = ahs_cur(a);
    if (s && ph >= 0 && ph < AHS_NPH)
        s->t[ph] += ticks;
}

/* [t0, t1) -> phase ph (a backwards pair adds 0) */
static inline void ahs_add_span(ahs_table *a, int ph, unsigned long long t0, unsigned long long t1) {
    ahs_add(a, ph, t1 > t0 ? t1 - t0 : 0ULL);
}

/* one launch window of win ticks of which wait ticks were spent in wait_io: wait -> WAIT (clamped to the
 * window), the rest -> DMA. dma + wait == win always, so the aie_ms window is split, never re-measured. */
static inline void ahs_add_window(ahs_table *a, unsigned long long win, unsigned long long wait) {
    if (wait > win)
        wait = win;
    ahs_add(a, AHS_WAIT, wait);
    ahs_add(a, AHS_DMA, win - wait);
}

static inline const char *ahs_kind_name(int k) {
    return k == AHS_KIND_CONV ? "conv" : k == AHS_KIND_LINEAR ? "linear" : k == AHS_KIND_ATTN ? "attn" : "none";
}

/* sites that were entered at least once (host_split_n) */
static inline int ahs_count(const ahs_table *a) {
    int n = 0;
    for (int i = 0; i < AHS_MAX_SITES; i++)
        n += a->s[i].calls > 0;
    return n;
}

/* ms[AHS_NPH] of one site (site < 0: sum of all sites); returns the number of sites summed (0 = no data) */
static inline int ahs_ms(const ahs_table *a, int site, double tick_hz, double *ms) {
    unsigned long long t[AHS_NPH] = {0, 0, 0, 0};
    int n = 0;
    for (int i = site < 0 ? 0 : site; i < (site < 0 ? AHS_MAX_SITES : site + 1) && i < AHS_MAX_SITES; i++) {
        if (!a->s[i].calls)
            continue;
        for (int p = 0; p < AHS_NPH; p++)
            t[p] += a->s[i].t[p];
        n++;
    }
    for (int p = 0; p < AHS_NPH; p++)
        ms[p] = tick_hz > 0.0 ? 1000.0 * (double)t[p] / tick_hz : 0.0;
    return n;
}

/* the PROF line of one site (no newline):
 *   PROF aie_host_split part=<n> kind=<attn|conv|linear> pack_ms= scatter_ms= dma_ms= wait_ms=
 * site < 0: PROF aie_host_split_total host_split_n=<sites> pack_ms= scatter_ms= dma_ms= wait_ms=
 * returns snprintf's length, 0 when the site has no data (nothing written but "") */
static inline int ahs_format(const ahs_table *a, int site, double tick_hz, char *buf, size_t n) {
    double ms[AHS_NPH];
    if (n)
        buf[0] = 0;
    if (site >= AHS_MAX_SITES)
        return 0;
    const int ns = ahs_ms(a, site, tick_hz, ms);
    if (!ns)
        return 0;
    if (site < 0)
        return snprintf(buf, n,
                        "PROF aie_host_split_total host_split_n=%d pack_ms=%.3f scatter_ms=%.3f dma_ms=%.3f "
                        "wait_ms=%.3f",
                        ns, ms[AHS_PACK], ms[AHS_SCATTER], ms[AHS_DMA], ms[AHS_WAIT]);
    return snprintf(buf, n, "PROF aie_host_split part=%d kind=%s pack_ms=%.3f scatter_ms=%.3f dma_ms=%.3f wait_ms=%.3f",
                    site, ahs_kind_name(a->s[site].kind), ms[AHS_PACK], ms[AHS_SCATTER], ms[AHS_DMA], ms[AHS_WAIT]);
}

#endif /* AIE_HOST_SPLIT_H */
