/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* AIE-side cost-model microbench (segment K, costmodel K1a), bare-metal A78 + AIE2PS (palmyra / vek385).
 * Built on the example/perf probes (aieml_perf.cc init + routing style) and the board probes
 * amit/tasks/2026-11-14_palmyra-ddr-cap/ddrbw.cpp (DDR) and 2026-11-11_vek385-s2mm-bw (tile DMA, HW counters).
 * Driven by script/perf/run_microbench.sh, which fits the raw "MB ..." lines into params/<board>.json.
 *
 * Every tile-side number is timed IN HARDWARE by memory-module perf counters (AIE cycles), never by host polls:
 *   clk     shim PL timer vs CNTPCT over ~200 ms               -> aie_clk_mhz
 *   pmu     PMCCNTR_EL0 vs CNTPCT over ~200 ms                 -> pmu_hz (A78 cycle counter)
 *   dma     tile (MB_COL,3) MM2S0 -> circuit DMA0->DMA0 -> S2MM0, sizes 256..28672 B,
 *           ctr MM2S0_START_TASK -> S2MM0_FINISHED_BD          -> dma_b_per_cyc (slope), dma_first_byte_cyc (intercept)
 *   bd      MM2S0 chain of 8 x 64 B BDs vs 1 x 512 B BD          -> bd_switch_cyc = (chain - single) / 7
 *   lock    same chain, every BD acq lock0(>=1,-1) rel lock1(+1) -> lock_acq_cyc = (lock - chain) / 8
 *   bank    2 loopback pairs (ch0, ch1) gated by lock 15, 4 buffers in ONE 8 KiB bank vs 4 Chess banks,
 *           ctr LOCK_15_ACQ -> S2MMx_FINISHED_TASK, plus DMA_*_MEMORY_CONFLICT level counts
 *                                                              -> dm_bank_conflict_cyc (extra cyc per KiB per channel)
 *   hop     256 B through a 4-link and a 6-link stream-switch ring back into the same tile (no U-turn)
 *                                                              -> ss_hop_cyc = (ring6 - ring4) / 2
 *   ddr     ddrbw.cpp R4/W4: DDR <-> shim (cols 0..3) <-> memtile, 4 MiB BDs, data checked
 *                                                              -> ddr_rd_gbps / ddr_wr_gbps (wall clock, CNTPCT)
 * Shim timing rule (knowledge 2026-11-10-aie2ps-shim-perfctr-semantics): shim FINISHED_TASK fires early, so no shim
 * perf counter is used; DDR rates are end-to-end (memtile S2MM drained / shim S2MM drained) over 4 MiB tasks.
 */
#include "xaiengine.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef __AIESIM__
#include "xil_cache.h"
#include "xil_printf.h"
#endif

#define HW_GEN XAIE_DEV_GEN_AIE2PS
#ifndef CORE_IP_MEM
#define CORE_IP_MEM 0x8000
#endif
#ifndef CORE_OP_MEM
#define CORE_OP_MEM 0x8800
#endif

void routing(XAie_DevInst *) {}
__global__ void loop_kernel(int *in, int *out) { (void)in; (void)out; }

static void say(const char *fmt, ...) {
    char b[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    printf("%s\n", b);
}
static inline uint64_t cntpct(void) { uint64_t v; __asm__ volatile("isb; mrs %0, cntpct_el0" : "=r"(v)); return v; }
static inline uint64_t cntfrq(void) { uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }
static inline uint64_t pmccntr(void) { uint64_t v; __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v)); return v; }
static void pmu_enable(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, pmcr_el0" : "=r"(v));
    v |= 1ull;          /* E */
    v &= ~(1ull << 3);  /* D=0: count every cycle */
    __asm__ volatile("msr pmcr_el0, %0" : : "r"(v));
    __asm__ volatile("msr pmcntenset_el0, %0" : : "r"(1ull << 31));
    __asm__ volatile("isb");
}

static int g_fail = 0;
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        AieRC _r = (AieRC)(x);                                                                                         \
        if (_r != XAIE_OK) {                                                                                           \
            say("MB rcfail line=%d rc=%d", __LINE__, (int)_r);                                                          \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

/* ------------------------------------------------------------------ clocks */
static void bench_clk(XAie_DevInst *D) {
    const uint64_t f = cntfrq(), win = f / 5; /* 200 ms */
    u64 t0 = 0, t1 = 0;
    uint64_t c0 = cntpct();
    CK(XAie_ReadTimer(D, XAie_TileLoc(0, 0), XAIE_PL_MOD, &t0));
    uint64_t c0b = cntpct();
    while (cntpct() - c0 < win) {
    }
    uint64_t c1 = cntpct();
    CK(XAie_ReadTimer(D, XAie_TileLoc(0, 0), XAIE_PL_MOD, &t1));
    uint64_t c1b = cntpct();
    /* midpoints of the bracketing CNTPCT reads */
    uint64_t dt = ((c1 + c1b) - (c0 + c0b)) / 2;
    say("MB clk aie_cyc=%llu cntpct=%llu cntfrq=%llu", (unsigned long long)(t1 - t0), (unsigned long long)dt,
        (unsigned long long)f);
    pmu_enable();
    uint64_t p0 = pmccntr(), q0 = cntpct();
    while (cntpct() - q0 < win) {
    }
    uint64_t p1 = pmccntr(), q1 = cntpct();
    say("MB pmu pmccntr=%llu cntpct=%llu cntfrq=%llu", (unsigned long long)(p1 - p0), (unsigned long long)(q1 - q0),
        (unsigned long long)f);
}

/* ------------------------------------------------------------------ tile DMA helpers */
#define MB_ROW 3
#define MB_COL 4   /* dma / bd / lock tile */
#define BK_COL 5   /* bank-conflict tile */
#define R4_COL 6   /* ring4: (6,3)(6,4)(7,4)(7,3) */
#define R6_COL 8   /* ring6: (8,3)(8,4)(8,5)(9,5)(9,4)(9,3) */
#define POLL_MAX 2000000u

static void bd(XAie_DevInst *D, XAie_LocType t, u16 id, u32 addr, u32 len, int next, int acq, s8 acqv, int rel,
               s8 relv) {
    XAie_DmaDesc d;
    CK(XAie_DmaDescInit(D, &d, t));
    CK(XAie_DmaSetAddrLen(&d, addr, len));
    if (acq >= 0) CK(XAie_DmaSetLock(&d, XAie_LockInit((u16)acq, acqv), XAie_LockInit((u16)rel, relv)));
    if (next >= 0) CK(XAie_DmaSetNextBd(&d, (u16)next, XAIE_ENABLE));
    CK(XAie_DmaEnableBd(&d));
    CK(XAie_DmaWriteBd(D, &d, t, id));
}
static int pend(XAie_DevInst *D, XAie_LocType t, u8 ch, XAie_DmaDirection dir) {
    u8 p = 0xff;
    if (XAie_DmaGetPendingBdCount(D, t, ch, dir, &p) != XAIE_OK) return -1;
    return p;
}
static int wait_idle(XAie_DevInst *D, XAie_LocType t, int nch) {
    for (u32 i = 0; i < POLL_MAX; i++) {
        int busy = 0;
        for (u8 c = 0; c < nch; c++) {
            int a = pend(D, t, c, DMA_S2MM), b = pend(D, t, c, DMA_MM2S);
            if (a < 0 || b < 0) return -1;
            busy |= a | b;
        }
        if (!busy) return 0;
    }
    say("MB TIMEOUT col=%d row=%d s2mm0=%d mm2s0=%d s2mm1=%d mm2s1=%d", t.Col, t.Row, pend(D, t, 0, DMA_S2MM),
        pend(D, t, 0, DMA_MM2S), pend(D, t, 1, DMA_S2MM), pend(D, t, 1, DMA_MM2S));
    g_fail++;
    return -2;
}
static void ctr(XAie_DevInst *D, XAie_LocType t, u8 i, XAie_Events a, XAie_Events b) {
    CK(XAie_PerfCounterControlSet(D, t, XAIE_MEM_MOD, i, a, b));
    CK(XAie_PerfCounterSet(D, t, XAIE_MEM_MOD, i, 0));
}
static u32 ctr_get(XAie_DevInst *D, XAie_LocType t, u8 i) {
    u32 v = 0;
    CK(XAie_PerfCounterGet(D, t, XAIE_MEM_MOD, i, &v));
    return v;
}
static u32 g_buf[8192];
static void fill(XAie_DevInst *D, XAie_LocType t, u32 addr, u32 bytes, u32 salt) {
    for (u32 i = 0; i < bytes / 4; i++) g_buf[i] = salt ^ (i * 2654435761u);
    CK(XAie_DataMemBlockWrite(D, t, addr, g_buf, bytes));
}
static int check(XAie_DevInst *D, XAie_LocType t, u32 addr, u32 bytes, u32 salt) {
    CK(XAie_DataMemBlockRead(D, t, addr, g_buf, bytes));
    int bad = 0;
    for (u32 i = 0; i < bytes / 4; i++) bad += g_buf[i] != (salt ^ (i * 2654435761u));
    return bad;
}
static void clear(XAie_DevInst *D, XAie_LocType t, u32 addr, u32 bytes) {
    for (u32 i = 0; i < bytes / 4; i++) g_buf[i] = 0xDEADBEEFu;
    CK(XAie_DataMemBlockWrite(D, t, addr, g_buf, bytes));
}

/* one loopback transfer in tile t: S2MM0 BD0 single (dst, total), MM2S0 = chain of k BDs (1..) of total/k each.
 * mode 0: plain, 1: every MM2S BD acq lock0 (-1) rel lock1 (+1), lock0 preset to k. Returns ctr0 cycles. */
#define SRC 0x1000
#define DST 0x9000
static u32 xfer_local(XAie_DevInst *D, XAie_LocType t, u32 total, int k, int mode, u32 salt, int *bad) {
    fill(D, t, SRC, total, salt);
    clear(D, t, DST, total);
    if (mode == 1) {
        CK(XAie_LockSetValue(D, t, XAie_LockInit(0, (s8)k)));
        CK(XAie_LockSetValue(D, t, XAie_LockInit(1, 0)));
    }
    ctr(D, t, 0, XAIE_EVENT_DMA_MM2S_0_START_TASK_MEM, XAIE_EVENT_DMA_S2MM_0_FINISHED_BD_MEM);
    ctr(D, t, 1, XAIE_EVENT_DMA_MM2S_0_START_TASK_MEM, XAIE_EVENT_DMA_MM2S_0_FINISHED_TASK_MEM);
    bd(D, t, 0, DST, total, -1, -1, 0, -1, 0);
    const u32 part = total / (u32)k;
    for (int i = 0; i < k; i++)
        bd(D, t, (u16)(2 + i), SRC + part * i, part, i + 1 < k ? 2 + i + 1 : -1, mode == 1 ? 0 : -1, -1, 1, 1);
    CK(XAie_DmaChannelPushBdToQueue(D, t, 0, DMA_S2MM, 0)); /* receiver first */
    CK(XAie_DmaChannelPushBdToQueue(D, t, 0, DMA_MM2S, 2));
    wait_idle(D, t, 1);
    *bad = check(D, t, DST, total, salt);
    return ctr_get(D, t, 0);
}

static void bench_dma(XAie_DevInst *D) {
    XAie_LocType t = XAie_TileLoc(MB_COL, MB_ROW);
    CK(XAie_StrmConnCctEnable(D, t, DMA, 0, DMA, 0));
    CK(XAie_DmaChannelEnable(D, t, 0, DMA_S2MM));
    CK(XAie_DmaChannelEnable(D, t, 0, DMA_MM2S));
    static const u32 sizes[] = {256, 1024, 4096, 8192, 16384, 28672};
    for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
        u32 mn = 0xFFFFFFFFu;
        int badall = 0;
        for (int r = 0; r < 3; r++) {
            int bad = 0;
            u32 c = xfer_local(D, t, sizes[s], 1, 0, 0xA0000000u + 16 * s + r, &bad);
            badall += bad;
            if (c < mn) mn = c;
        }
        say("MB dma bytes=%u cyc=%u data=%s", sizes[s], mn, badall ? "BAD" : "OK");
        if (badall) g_fail++;
    }
    /* BD switch + lock: 8 x 64 B chained vs 1 x 512 B, same total */
    const u32 tot = 512;
    const int K = 8;
    u32 m1 = 0xFFFFFFFFu, mk = 0xFFFFFFFFu, ml = 0xFFFFFFFFu;
    int bad1 = 0, badk = 0, badl = 0;
    for (int r = 0; r < 5; r++) {
        int b;
        u32 c = xfer_local(D, t, tot, 1, 0, 0xB1000000u + r, &b); bad1 += b; if (c < m1) m1 = c;
        c = xfer_local(D, t, tot, K, 0, 0xB2000000u + r, &b); badk += b; if (c < mk) mk = c;
        c = xfer_local(D, t, tot, K, 1, 0xB3000000u + r, &b); badl += b; if (c < ml) ml = c;
    }
    say("MB bd single_cyc=%u chain_cyc=%u nbd=%d bytes=%u data=%s", m1, mk, K, tot, (bad1 | badk) ? "BAD" : "OK");
    /* uncontended lock on every BD of the chain: hidden by BD prefetch (report only) */
    say("MB lockchain chain_cyc=%u lock_cyc=%u nbd=%d bytes=%u data=%s", mk, ml, K, tot, badl ? "BAD" : "OK");
    if (bad1 | badk | badl) g_fail++;
}

/* Lock hand-off (producer release -> waiting consumer acquire), the case kernels pay:
 *   leg 1: MM2S0 (SRC) -> DMA0->DMA0 -> S2MM0 BD0 (DST), BD0 releases lock 2 (+1)
 *   leg 2: MM2S1 BD10 waits on lock 2 (acq >=1, -1), sends DST (the data leg 1 just wrote) -> DMA1->DMA1 -> S2MM1 (DST2)
 *   ctr0 = S2MM0_FINISHED_BD -> S2MM1_FINISHED_BD (lock hand-off + leg 2), ctr1 = MM2S0_START_TASK -> S2MM0_FINISHED_BD
 *   (leg 1, same size, no lock): lock_acq_cyc = ctr0 - ctr1. */
#define DST2 0xC000
static void bench_lock(XAie_DevInst *D) {
    XAie_LocType t = XAie_TileLoc(MB_COL, MB_ROW);
    CK(XAie_StrmConnCctEnable(D, t, DMA, 1, DMA, 1));
    CK(XAie_DmaChannelEnable(D, t, 1, DMA_S2MM));
    CK(XAie_DmaChannelEnable(D, t, 1, DMA_MM2S));
    const u32 n = 256;
    u32 mh = 0xFFFFFFFFu, ml = 0xFFFFFFFFu;
    int badall = 0;
    for (int r = 0; r < 5; r++) {
        const u32 salt = 0xE0000000u + r;
        fill(D, t, SRC, n, salt);
        clear(D, t, DST, n);
        clear(D, t, DST2, n);
        CK(XAie_LockSetValue(D, t, XAie_LockInit(2, 0)));
        ctr(D, t, 0, XAIE_EVENT_DMA_S2MM_0_FINISHED_BD_MEM, XAIE_EVENT_DMA_S2MM_1_FINISHED_BD_MEM);
        ctr(D, t, 1, XAIE_EVENT_DMA_MM2S_0_START_TASK_MEM, XAIE_EVENT_DMA_S2MM_0_FINISHED_BD_MEM);
        XAie_DmaDesc d;
        CK(XAie_DmaDescInit(D, &d, t)); /* S2MM0 BD0: DST, release lock 2 */
        CK(XAie_DmaSetAddrLen(&d, DST, n));
        CK(XAie_DmaLockControl(&d, XAie_LockInit(2, 0), XAie_LockInit(2, 1), XAIE_DISABLE, XAIE_ENABLE));
        CK(XAie_DmaEnableBd(&d));
        CK(XAie_DmaWriteBd(D, &d, t, 0));
        bd(D, t, 2, SRC, n, -1, -1, 0, -1, 0); /* MM2S0 BD2 */
        CK(XAie_DmaDescInit(D, &d, t));        /* MM2S1 BD10: DST, acquire lock 2 */
        CK(XAie_DmaSetAddrLen(&d, DST, n));
        CK(XAie_DmaLockControl(&d, XAie_LockInit(2, -1), XAie_LockInit(2, 0), XAIE_ENABLE, XAIE_DISABLE));
        CK(XAie_DmaEnableBd(&d));
        CK(XAie_DmaWriteBd(D, &d, t, 10));
        bd(D, t, 11, DST2, n, -1, -1, 0, -1, 0); /* S2MM1 BD11 */
        CK(XAie_DmaChannelPushBdToQueue(D, t, 1, DMA_S2MM, 11));
        CK(XAie_DmaChannelPushBdToQueue(D, t, 1, DMA_MM2S, 10)); /* waits on lock 2 */
        CK(XAie_DmaChannelPushBdToQueue(D, t, 0, DMA_S2MM, 0));
        CK(XAie_DmaChannelPushBdToQueue(D, t, 0, DMA_MM2S, 2));
        wait_idle(D, t, 2);
        badall += check(D, t, DST2, n, salt);
        u32 h = ctr_get(D, t, 0), l = ctr_get(D, t, 1);
        if (h < mh) mh = h;
        if (l < ml) ml = l;
    }
    say("MB lock handoff_cyc=%u leg_cyc=%u bytes=%u data=%s", mh, ml, n, badall ? "BAD" : "OK");
    if (badall) g_fail++;
}

/* ------------------------------------------------------------------ bank conflict */
/* 2 loopback pairs in one tile, MM2S BDs gated by lock 15 (acq -1, rel lock 14 +1), repeat R each */
static void bank_run(XAie_DevInst *D, XAie_LocType t, const u32 a[4], u32 len, u32 rep, int ctrmode, u32 *c0, u32 *c1,
                     int *bad) {
    for (int i = 0; i < 2; i++) {
        fill(D, t, a[i], len, 0xC0000000u + i);
        clear(D, t, a[2 + i], len);
    }
    CK(XAie_LockSetValue(D, t, XAie_LockInit(15, 0)));
    CK(XAie_LockSetValue(D, t, XAie_LockInit(14, 0)));
    /* AIE2PS mem module has no per-lock ACQ events (only LOCK_SELn) and no DMA_*_MEMORY_CONFLICT: use
     * channel elapsed minus the gate-lock stall (level), and the bank-0 conflict level event */
    if (ctrmode == 0) { /* ch0: MM2S0 task start -> S2MM0 task end, and MM2S0 lock-stall cycles (gate wait) */
        ctr(D, t, 0, XAIE_EVENT_DMA_MM2S_0_START_TASK_MEM, XAIE_EVENT_DMA_S2MM_0_FINISHED_TASK_MEM);
        ctr(D, t, 1, XAIE_EVENT_DMA_MM2S_0_STALLED_LOCK_MEM, XAIE_EVENT_DMA_MM2S_0_STALLED_LOCK_MEM);
    } else if (ctrmode == 1) { /* same for ch1 */
        ctr(D, t, 0, XAIE_EVENT_DMA_MM2S_1_START_TASK_MEM, XAIE_EVENT_DMA_S2MM_1_FINISHED_TASK_MEM);
        ctr(D, t, 1, XAIE_EVENT_DMA_MM2S_1_STALLED_LOCK_MEM, XAIE_EVENT_DMA_MM2S_1_STALLED_LOCK_MEM);
    } else { /* bank-0 conflict cycles and MM2S0 memory starvation (level events) */
        ctr(D, t, 0, XAIE_EVENT_CONFLICT_DM_BANK_0_MEM, XAIE_EVENT_CONFLICT_DM_BANK_0_MEM);
        ctr(D, t, 1, XAIE_EVENT_DMA_MM2S_0_MEMORY_STARVATION_MEM, XAIE_EVENT_DMA_MM2S_0_MEMORY_STARVATION_MEM);
    }
    for (int i = 0; i < 2; i++) {
        bd(D, t, (u16)(i), a[2 + i], len, -1, -1, 0, -1, 0);                 /* S2MM ch i BD i */
        bd(D, t, (u16)(4 + i), a[i], len, -1, 15, -1, 14, 1);                /* MM2S ch i BD 4+i */
    }
    for (u8 i = 0; i < 2; i++) CK(XAie_DmaChannelSetStartQueue(D, t, i, DMA_S2MM, i, rep, XAIE_DISABLE));
    for (u8 i = 0; i < 2; i++) CK(XAie_DmaChannelSetStartQueue(D, t, i, DMA_MM2S, (u16)(4 + i), rep, XAIE_DISABLE));
    CK(XAie_LockSetValue(D, t, XAie_LockInit(15, (s8)(2 * rep)))); /* open the gate: both MM2S start together */
    wait_idle(D, t, 2);
    *c0 = ctr_get(D, t, 0);
    *c1 = ctr_get(D, t, 1);
    *bad = check(D, t, a[2], len, 0xC0000000u) + check(D, t, a[3], len, 0xC0000001u);
}
static void bench_bank(XAie_DevInst *D) {
    XAie_LocType t = XAie_TileLoc(BK_COL, MB_ROW);
    for (u8 i = 0; i < 2; i++) {
        CK(XAie_StrmConnCctEnable(D, t, DMA, i, DMA, i));
        CK(XAie_DmaChannelEnable(D, t, i, DMA_S2MM));
        CK(XAie_DmaChannelEnable(D, t, i, DMA_MM2S));
    }
    const u32 len = 2048, rep = 16;
    static const u32 same[4] = {0x0000, 0x0800, 0x1000, 0x1800}; /* all in physical bank 0 (0x0000-0x1FFF) */
    static const u32 diff[4] = {0x0000, 0x4000, 0x8000, 0xC000}; /* 4 Chess banks = disjoint physical banks */
    const u32 *cfg[2] = {same, diff};
    const char *nm[2] = {"same", "diff"};
    for (int k = 0; k < 2; k++) {
        u32 e0 = 0xFFFFFFFFu, e1 = 0xFFFFFFFFu, s0 = 0, s1 = 0;
        int badall = 0;
        for (int r = 0; r < 3; r++) {
            u32 a, b;
            int bad;
            bank_run(D, t, cfg[k], len, rep, 0, &a, &b, &bad);
            badall += bad;
            if (a - b < e0) e0 = a - b;
            bank_run(D, t, cfg[k], len, rep, 1, &a, &b, &bad);
            badall += bad;
            if (a - b < e1) e1 = a - b;
            bank_run(D, t, cfg[k], len, rep, 2, &a, &b, &bad);
            badall += bad;
            if (a > s0) s0 = a;
            if (b > s1) s1 = b;
        }
        say("MB bank case=%s bytes_per_ch=%u elapsed0=%u elapsed1=%u conflict_bank0=%u starv_mm2s0=%u data=%s", nm[k],
            len * rep, e0, e1, s0, s1, badall ? "BAD" : "OK");
        if (badall) g_fail++;
    }
}

/* ------------------------------------------------------------------ stream-switch hop */
static u32 ring_xfer(XAie_DevInst *D, XAie_LocType t, u32 bytes, u32 salt, int *bad) {
    fill(D, t, SRC, bytes, salt);
    clear(D, t, DST, bytes);
    ctr(D, t, 0, XAIE_EVENT_DMA_MM2S_0_START_TASK_MEM, XAIE_EVENT_DMA_S2MM_0_FINISHED_BD_MEM);
    bd(D, t, 0, DST, bytes, -1, -1, 0, -1, 0);
    bd(D, t, 2, SRC, bytes, -1, -1, 0, -1, 0);
    CK(XAie_DmaChannelPushBdToQueue(D, t, 0, DMA_S2MM, 0));
    CK(XAie_DmaChannelPushBdToQueue(D, t, 0, DMA_MM2S, 2));
    wait_idle(D, t, 1);
    *bad = check(D, t, DST, bytes, salt);
    return ctr_get(D, t, 0);
}
static void bench_hop(XAie_DevInst *D) {
    const u32 bytes = 256;
    /* ring4: A(6,3) DMA0->N0 | (6,4) S0->E0 | (7,4) W0->S0 | (7,3) N0->W0 | A E0->DMA0 */
    XAie_LocType A = XAie_TileLoc(R4_COL, MB_ROW);
    CK(XAie_StrmConnCctEnable(D, A, DMA, 0, NORTH, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R4_COL, 4), SOUTH, 0, EAST, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R4_COL + 1, 4), WEST, 0, SOUTH, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R4_COL + 1, 3), NORTH, 0, WEST, 0));
    CK(XAie_StrmConnCctEnable(D, A, EAST, 0, DMA, 0));
    /* ring6: B(8,3) DMA0->N0 | (8,4) S0->N0 | (8,5) S0->E0 | (9,5) W0->S0 | (9,4) N0->S0 | (9,3) N0->W0 | B E0->DMA0 */
    XAie_LocType B = XAie_TileLoc(R6_COL, MB_ROW);
    CK(XAie_StrmConnCctEnable(D, B, DMA, 0, NORTH, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R6_COL, 4), SOUTH, 0, NORTH, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R6_COL, 5), SOUTH, 0, EAST, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R6_COL + 1, 5), WEST, 0, SOUTH, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R6_COL + 1, 4), NORTH, 0, SOUTH, 0));
    CK(XAie_StrmConnCctEnable(D, XAie_TileLoc(R6_COL + 1, 3), NORTH, 0, WEST, 0));
    CK(XAie_StrmConnCctEnable(D, B, EAST, 0, DMA, 0));
    XAie_LocType L = XAie_TileLoc(MB_COL, MB_ROW); /* local loop, already DMA0->DMA0 */
    XAie_LocType T[3] = {L, A, B};
    const int links[3] = {0, 4, 6};
    for (int k = 1; k < 3; k++) {
        CK(XAie_DmaChannelEnable(D, T[k], 0, DMA_S2MM));
        CK(XAie_DmaChannelEnable(D, T[k], 0, DMA_MM2S));
    }
    for (int k = 0; k < 3; k++) {
        u32 mn = 0xFFFFFFFFu;
        int badall = 0;
        for (int r = 0; r < 5; r++) {
            int bad;
            u32 c = ring_xfer(D, T[k], bytes, 0xD0000000u + 16 * k + r, &bad);
            badall += bad;
            if (c < mn) mn = c;
        }
        say("MB hop links=%d bytes=%u cyc=%u data=%s", links[k], bytes, mn, badall ? "BAD" : "OK");
        if (badall) g_fail++;
    }
}

/* ------------------------------------------------------------------ DDR (ddrbw.cpp R4 / W4) */
#define MT_ROW 1
#define MT_DMA_HI 0x80000u
#define MT_LEN (256u << 10)
#define MT_R_OFF 0u
#define MT_W_OFF (256u << 10)
#define DDR_BYTES (4u << 20)
#define DDR_NCOL 4
#define DDR_REPS 8
static AieRC shimbd(XAie_DevInst *D, u8 col, XAie_MemInst *m, uint32_t n, u16 b) {
    XAie_LocType S = XAie_TileLoc(col, 0);
    XAie_DmaDesc d;
    return (AieRC)(XAie_DmaDescInit(D, &d, S) | XAie_DmaSetAddrLen(&d, XAie_MemGetDevAddr(m), n) |
                   XAie_DmaSetAxi(&d, 0U, 16U, 0U, 0U, 0U) | XAie_DmaEnableBd(&d) | XAie_DmaWriteBd(D, &d, S, b));
}
static AieRC mtbd(XAie_DevInst *D, u8 col, uint32_t off, u16 b) {
    XAie_LocType M = XAie_TileLoc(col, MT_ROW);
    XAie_DmaDesc d;
    return (AieRC)(XAie_DmaDescInit(D, &d, M) | XAie_DmaSetAddrLen(&d, (u64)(MT_DMA_HI | off), MT_LEN) |
                   XAie_DmaEnableBd(&d) | XAie_DmaWriteBd(D, &d, M, b));
}
static XAie_MemInst *alloc64(XAie_DevInst *D, uint32_t cap) {
    XAie_MemInst *m = XAie_MemAllocate(D, cap + 64, XAIE_MEM_CACHEABLE);
    if (!m) return 0;
    u64 off = (64 - ((u64)(uintptr_t)m->VAddr & 63)) & 63;
    m->VAddr = (void *)((uintptr_t)m->VAddr + off);
    m->DevAddr += off;
    m->Size = cap;
    return m;
}
static inline uint32_t pat(uint32_t i, uint32_t salt) { return (i * 2654435761u) ^ salt; }
static uint32_t g_mt[MT_LEN / 4];
/* kind 0 = read (DDR -> memtile), 1 = write (memtile -> DDR); all DDR_NCOL columns concurrently */
static void ddr_point(XAie_DevInst *D, int kind, XAie_MemInst **buf) {
    const uint32_t n = DDR_BYTES, rep = n / MT_LEN;
    for (int c = 0; c < DDR_NCOL; c++) {
        uint32_t *p = (uint32_t *)XAie_MemGetVAddr(buf[c]);
        const uint32_t salt = 0x5C000000u + 64 * kind + c;
        if (kind == 0) {
            for (uint32_t i = 0; i < n / 4; i++) p[i] = pat(i, salt);
            XAie_MemSyncForDev(buf[c]);
        } else {
            for (uint32_t i = 0; i < n / 4; i++) p[i] = 0xDEADBEEFu;
            XAie_MemSyncForDev(buf[c]);
            for (uint32_t i = 0; i < MT_LEN / 4; i++) g_mt[i] = pat(i, salt);
            CK(XAie_DataMemBlockWrite(D, XAie_TileLoc(c, MT_ROW), MT_W_OFF, g_mt, MT_LEN));
        }
    }
    uint64_t tk = 0, cy = 0, tmin = ~0ull;
    int fails = 0;
    for (int r = 0; r < DDR_REPS && !fails; r++) {
        AieRC rc = XAIE_OK;
        for (int c = 0; c < DDR_NCOL; c++)
            rc = (AieRC)(rc | (kind == 0 ? (shimbd(D, c, buf[c], n, 0) | mtbd(D, c, MT_R_OFF, 0))
                                         : (shimbd(D, c, buf[c], n, 1) | mtbd(D, c, MT_W_OFF, 1))));
        u64 a0 = 0, a1 = 0;
        XAie_ReadTimer(D, XAie_TileLoc(0, 0), XAIE_PL_MOD, &a0);
        uint64_t c0 = cntpct();
        for (int c = 0; c < DDR_NCOL; c++) {
            XAie_LocType S = XAie_TileLoc(c, 0), M = XAie_TileLoc(c, MT_ROW);
            if (kind == 0)
                rc = (AieRC)(rc | XAie_DmaChannelSetStartQueue(D, M, 0, DMA_S2MM, 0, rep, XAIE_DISABLE) |
                             XAie_DmaChannelPushBdToQueue(D, S, 1, DMA_MM2S, 0));
            else
                rc = (AieRC)(rc | XAie_DmaChannelPushBdToQueue(D, S, 0, DMA_S2MM, 1) |
                             XAie_DmaChannelSetStartQueue(D, M, 0, DMA_MM2S, 1, rep, XAIE_DISABLE));
        }
        int left = DDR_NCOL;
        for (uint32_t p = 0; p < 200000000u && left; p++) {
            left = 0;
            for (int c = 0; c < DDR_NCOL; c++) {
                int x = kind == 0 ? pend(D, XAie_TileLoc(c, MT_ROW), 0, DMA_S2MM) : pend(D, XAie_TileLoc(c, 0), 0, DMA_S2MM);
                int y = kind == 0 ? pend(D, XAie_TileLoc(c, 0), 1, DMA_MM2S) : 0;
                left += (x | y) != 0;
            }
        }
        uint64_t c1 = cntpct();
        XAie_ReadTimer(D, XAie_TileLoc(0, 0), XAIE_PL_MOD, &a1);
        if (rc != XAIE_OK || left) {
            say("MB ddr TIMEOUT kind=%d rep=%d rc=%d left=%d", kind, r, (int)rc, left);
            fails++;
            break;
        }
        tk += c1 - c0;
        cy += a1 - a0;
        if (c1 - c0 < tmin) tmin = c1 - c0;
    }
    int bad = 0;
    if (!fails)
        for (int c = 0; c < DDR_NCOL; c++) {
            const uint32_t salt = 0x5C000000u + 64 * kind + c;
            if (kind == 0) {
                CK(XAie_DataMemBlockRead(D, XAie_TileLoc(c, MT_ROW), MT_R_OFF, g_mt, MT_LEN));
                const uint32_t base = (n - MT_LEN) / 4;
                for (uint32_t i = 0; i < MT_LEN / 4; i++) bad += g_mt[i] != pat(base + i, salt);
            } else {
                XAie_MemSyncForCPU(buf[c]);
                const uint32_t *o = (const uint32_t *)XAie_MemGetVAddr(buf[c]);
                for (uint32_t i = 0; i < n / 4; i++) bad += o[i] != pat(i % (MT_LEN / 4), salt);
            }
        }
    say("MB ddr kind=%s cols=%d bytes=%llu cntpct=%llu cntpct_min_rep=%llu cntfrq=%llu aie_cyc=%llu data=%s",
        kind == 0 ? "rd" : "wr", DDR_NCOL, (unsigned long long)n * DDR_NCOL * DDR_REPS, (unsigned long long)tk,
        (unsigned long long)tmin, (unsigned long long)cntfrq(), (unsigned long long)cy, (bad || fails) ? "BAD" : "OK");
    if (bad || fails) g_fail++;
}
static void bench_ddr(XAie_DevInst *D) {
    for (int c = 0; c < DDR_NCOL; c++) {
        XAie_LocType S = XAie_TileLoc(c, 0), M = XAie_TileLoc(c, MT_ROW);
        const u64 SB = (u64)c << 25;
        u32 v = 0;
        XAie_Write32(D, SB + 0x7FF10, 1); /* shim stream-switch reset pulse (vek385-3 rule, ddrbw.cpp) */
        XAie_Read32(D, SB + 0x7FF10, &v);
        XAie_Write32(D, SB + 0x7FF10, 0);
        XAie_Read32(D, SB + 0x7FF10, &v);
        CK(XAie_StrmConnCctEnable(D, S, SOUTH, 7, NORTH, 0));
        CK(XAie_StrmConnCctEnable(D, S, NORTH, 0, SOUTH, 1));
        CK(XAie_EnableShimDmaToAieStrmPort(D, S, 7));
        CK(XAie_EnableAieToShimDmaStrmPort(D, S, 1));
        CK(XAie_StrmConnCctEnable(D, M, SOUTH, 0, DMA, 0));
        CK(XAie_StrmConnCctEnable(D, M, DMA, 0, SOUTH, 0));
        CK(XAie_DmaChannelEnable(D, S, 0, DMA_S2MM));
        CK(XAie_DmaChannelEnable(D, S, 1, DMA_MM2S));
        CK(XAie_DmaChannelEnable(D, M, 0, DMA_S2MM));
        CK(XAie_DmaChannelEnable(D, M, 0, DMA_MM2S));
    }
    XAie_MemInst *b[DDR_NCOL];
    for (int c = 0; c < DDR_NCOL; c++)
        if (!(b[c] = alloc64(D, DDR_BYTES))) { say("MB ddr alloc failed"); g_fail++; return; }
    ddr_point(D, 0, b);
    ddr_point(D, 1, b);
}

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_AIE_TILE_NUM_ROWS 4
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
#if defined(__aarch64__) && !defined(__AIESIM__)
    {
        unsigned long midr;
        __asm__ volatile("mrs %0, midr_el1" : "=r"(midr));
        printf("BOARD machine=aarch64 midr=0x%08lx\n", midr);
    }
#else
    printf("BOARD machine=other midr=0x00000000\n");
#endif
#ifndef __AIESIM__
    Xil_DCacheDisable();
    Xil_ICacheDisable();
#endif
    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);
    XAie_InstDeclare(DevInst, &ConfigPtr);
    if (XAie_CfgInitialize(&DevInst, &ConfigPtr) != XAIE_OK) { say("MB END rc=-1 init FAIL"); return -1; }
#ifdef __AIESIM__
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_SIM);
    XAie_PartitionInitialize(&DevInst, NULL);
#else
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);
    if (XAie_UpdateNpiAddr(&DevInst, 0xF6D50000u) != XAIE_OK) { say("MB END rc=-1 npi FAIL"); return -1; }
    if (XAie_PartitionInitialize(&DevInst, NULL) != XAIE_OK) { say("MB END rc=-1 partinit FAIL"); return -1; }
#endif
    say("MB start aie_microbench v1 section=aie");
    bench_clk(&DevInst);
    bench_dma(&DevInst);
    bench_lock(&DevInst);
    bench_hop(&DevInst);
    bench_bank(&DevInst);
    bench_ddr(&DevInst);
    say("MB END rc=%d %s", g_fail, g_fail == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return g_fail ? 1 : 0;
}
