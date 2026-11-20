/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* GAP-RT1: per-call-site split of the A78 time of an AIE offload site ("aie_host") into
 *   pack    - host operand tiling / bf16 packing before the first launch (im2col, akp_pack, ...)
 *   scatter - host output un-tiling / epilogue after the launches (uc_scatter16, akp_unpad, ...)
 *   dma     - launch windows minus the runtime's wait_io (cache sync, BD config, start, core enable)
 *   wait    - the runtime's __Runtime_wait_io poll inside the launch windows
 * and (M2 N-segment cost model) the in-launch host glue of the same launches, in A78 PMU cycles (pmccntr), the
 * per-site share of the global g_sp_* split (split_detail host_glue_bdcfg / host_flush_startio / host_inval_wait_io):
 *   glue    - (sp2-sp1)-(rt1-rt0): the launch call minus the runtime phases + wait_io (lock init, createio, commit)
 *   flush   - sp1-sp0: host->DMA cache flush before the launch
 *   inval   - sp3-sp2: DMA->host invalidate (+ pm stats) after the launch
 *   launches- AIE launches of the site (ahs_launch)
 * The cycle rate for *_ms comes from ahs_cal (cycle/tick pairs taken at the launch windows), else cyc_hz, else tick_hz.
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

enum { AHS_PACK = 0, AHS_SCATTER = 1, AHS_DMA = 2, AHS_WAIT = 3, AHS_NTK = 4 }; /* XTime ticks */
enum { AHS_GLUE = 4, AHS_FLUSH = 5, AHS_INVAL = 6, AHS_NPH = 7 };              /* PMU cycles */
enum { AHS_KIND_NONE = 0, AHS_KIND_CONV = 1, AHS_KIND_LINEAR = 2, AHS_KIND_ATTN = 3, AHS_KIND_FA = 4 };

typedef struct {
    unsigned long long t[AHS_NPH]; /* ticks (AHS_PACK..AHS_WAIT) / PMU cycles (AHS_GLUE..AHS_INVAL) per phase */
    unsigned calls;                /* site entries (graph_run calls) */
    unsigned launches;             /* AIE launches inside the site */
    int kind;                      /* AHS_KIND_* of the last entry */
    unsigned perf_n;               /* M3.1 site-perf: launches with a core counter reading */
    unsigned long long io_act, lock_stall, stream_stall; /* slowest-core counter sums (core cycles) */
} ahs_site;

typedef struct {
    ahs_site s[AHS_MAX_SITES];
    int cur1; /* current site + 1; 0 = no site (adds are dropped) */
    double cyc_hz; /* PMU cycle rate override for the cycle phases (0 = from ahs_cal, else tick_hz) */
    unsigned long long cal_c0, cal_t0, cal_c1, cal_t1; /* first / last (cycles, ticks) pair of ahs_cal */
    unsigned cal_n;
} ahs_table;

/* M3.1 site-perf (-DAIEHLC_SITE_PERF=1): the runtime's slowest-core counter reading of the last launch
 * (aie_runtime.c; the x86 test links a stub). Returns 1 for a fresh reading, 0 for none. */
#if defined(AIEHLC_SITE_PERF) && AIEHLC_SITE_PERF
int __Runtime_core_perf_last_launch(unsigned long long *io_act_cyc, unsigned long long *lock_stall_cyc,
                                    unsigned long long *stream_stall_cyc);
#endif

/* aiednn_aie_host_split_line(AHS_PERF_LINE + part, ...) asks for the PROF aie_site_perf line of part
 * (same exported entry point; archives built without AIEHLC_SITE_PERF return 0 for it) */
#define AHS_PERF_LINE 0x10000

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

/* one AIE launch of the current site */
static inline void ahs_launch(ahs_table *a) {
    ahs_site *s = ahs_cur(a);
    if (s)
        s->launches++;
}

/* a (PMU cycles, ticks) pair read back to back (any site, also outside one): the cycle rate of the cycle phases */
static inline void ahs_cal(ahs_table *a, unsigned long long cyc, unsigned long long ticks) {
    if (!a->cal_n) {
        a->cal_c0 = cyc;
        a->cal_t0 = ticks;
    }
    a->cal_c1 = cyc;
    a->cal_t1 = ticks;
    a->cal_n++;
}

/* PMU cycles per second for ticks at tick_hz: cyc_hz, else the ahs_cal span, else tick_hz (same clock) */
static inline double ahs_cyc_hz(const ahs_table *a, double tick_hz) {
    if (a->cyc_hz > 0.0)
        return a->cyc_hz;
    if (a->cal_n > 1 && a->cal_t1 > a->cal_t0 && a->cal_c1 > a->cal_c0)
        return (double)(a->cal_c1 - a->cal_c0) * tick_hz / (double)(a->cal_t1 - a->cal_t0);
    return tick_hz;
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
#if defined(AIEHLC_SITE_PERF) && AIEHLC_SITE_PERF
    {
        unsigned long long ia = 0ULL, ls = 0ULL, ss = 0ULL;
        ahs_site *s = ahs_cur(a);
        if (__Runtime_core_perf_last_launch(&ia, &ls, &ss) && s) {
            s->perf_n++;
            s->io_act += ia;
            s->lock_stall += ls;
            s->stream_stall += ss;
        }
    }
#endif
}

static inline const char *ahs_kind_name(int k) {
    return k == AHS_KIND_CONV     ? "conv"
           : k == AHS_KIND_LINEAR ? "linear"
           : k == AHS_KIND_ATTN   ? "attn"
           : k == AHS_KIND_FA     ? "fa"
                                  : "none";
}

/* sites that were entered at least once (host_split_n) */
static inline int ahs_count(const ahs_table *a) {
    int n = 0;
    for (int i = 0; i < AHS_MAX_SITES; i++)
        n += a->s[i].calls > 0;
    return n;
}

/* ms[AHS_NPH] of one site (site < 0: sum of all sites); returns the number of sites summed (0 = no data);
 * *nl (if non-null) = launches summed */
static inline int ahs_ms_n(const ahs_table *a, int site, double tick_hz, double *ms, unsigned *nl) {
    unsigned long long t[AHS_NPH] = {0, 0, 0, 0, 0, 0, 0};
    unsigned l = 0;
    int n = 0;
    for (int i = site < 0 ? 0 : site; i < (site < 0 ? AHS_MAX_SITES : site + 1) && i < AHS_MAX_SITES; i++) {
        if (!a->s[i].calls)
            continue;
        for (int p = 0; p < AHS_NPH; p++)
            t[p] += a->s[i].t[p];
        l += a->s[i].launches;
        n++;
    }
    const double chz = ahs_cyc_hz(a, tick_hz);
    for (int p = 0; p < AHS_NPH; p++) {
        const double hz = p < AHS_NTK ? tick_hz : chz;
        ms[p] = hz > 0.0 ? 1000.0 * (double)t[p] / hz : 0.0;
    }
    if (nl)
        *nl = l;
    return n;
}

static inline int ahs_ms(const ahs_table *a, int site, double tick_hz, double *ms) {
    return ahs_ms_n(a, site, tick_hz, ms, (unsigned *)0);
}

/* the PROF line of one site (no newline, <= 191 chars for site_prof.c's hs_[192]):
 *   PROF aie_host_split part=<n> kind=<attn|conv|linear|fa> pack_ms= scatter_ms= dma_ms= wait_ms= glue_ms= flush_ms=
 *        inval_ms= launches=
 * site < 0: PROF aie_host_split_total host_split_n=<sites> pack_ms= scatter_ms= dma_ms= wait_ms= glue_ms= ...
 * returns snprintf's length, 0 when the site has no data (nothing written but "") */
/* M3.1: the counter line of one part, only for parts with launches > 0 (else 0, ""):
 *   PROF aie_site_perf part=<n> kind=<k> launches=<n> perf_n=<n> io_act_cyc=<sum> lock_stall_cyc=<sum>
 *        stream_stall_cyc=<sum>
 * A separate line: the aie_host_split line must keep ending in launches=<n> (n_fit.py / m3_site.py regexes). */
static inline int ahs_format_perf(const ahs_table *a, int site, char *buf, size_t n) {
    if (n)
        buf[0] = 0;
    if (site < 0 || site >= AHS_MAX_SITES || !a->s[site].calls || !a->s[site].launches)
        return 0;
    const ahs_site *s = &a->s[site];
    return snprintf(buf, n,
                    "PROF aie_site_perf part=%d kind=%s launches=%u perf_n=%u io_act_cyc=%llu lock_stall_cyc=%llu "
                    "stream_stall_cyc=%llu",
                    site, ahs_kind_name(s->kind), s->launches, s->perf_n, s->io_act, s->lock_stall, s->stream_stall);
}

static inline int ahs_format(const ahs_table *a, int site, double tick_hz, char *buf, size_t n) {
    double ms[AHS_NPH];
#if defined(AIEHLC_SITE_PERF) && AIEHLC_SITE_PERF
    if (site >= AHS_PERF_LINE) /* only site-perf builds print the counter line (else: "" below) */
        return ahs_format_perf(a, site - AHS_PERF_LINE, buf, n);
#endif
    if (n)
        buf[0] = 0;
    if (site >= AHS_MAX_SITES)
        return 0;
    unsigned nl = 0;
    const int ns = ahs_ms_n(a, site, tick_hz, ms, &nl);
    if (!ns)
        return 0;
    if (site < 0)
        return snprintf(buf, n,
                        "PROF aie_host_split_total host_split_n=%d pack_ms=%.3f scatter_ms=%.3f dma_ms=%.3f "
                        "wait_ms=%.3f glue_ms=%.3f flush_ms=%.3f inval_ms=%.3f launches=%d",
                        ns, ms[AHS_PACK], ms[AHS_SCATTER], ms[AHS_DMA], ms[AHS_WAIT], ms[AHS_GLUE], ms[AHS_FLUSH],
                        ms[AHS_INVAL], (int)nl);
    return snprintf(buf, n,
                    "PROF aie_host_split part=%d kind=%s pack_ms=%.3f scatter_ms=%.3f dma_ms=%.3f wait_ms=%.3f "
                    "glue_ms=%.3f flush_ms=%.3f inval_ms=%.3f launches=%d",
                    site, ahs_kind_name(a->s[site].kind), ms[AHS_PACK], ms[AHS_SCATTER], ms[AHS_DMA], ms[AHS_WAIT],
                    ms[AHS_GLUE], ms[AHS_FLUSH], ms[AHS_INVAL], (int)nl);
}

#endif /* AIE_HOST_SPLIT_H */
