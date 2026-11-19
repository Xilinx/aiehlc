/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* DDR-side cost-model microbench (segment K, costmodel K2d), bare-metal A78 + AIE2PS (palmyra213).
 * Driven by script/perf/run_microbench.sh --section ddr; script/perf/mb_fit.py (section=ddr) merges the fitted keys
 * into the existing params/<board>.json (keeps every section=aie key).
 * Self-contained (aiehlc.sh rewrites only the runtime source file), DDR path copied from aie_microbench.cc bench_ddr
 * (amit/tasks/2026-11-14_palmyra-ddr-cap/ddrbw.cpp R4/W4).
 *
 *   ddr       kind=rd: DDR -> shim MM2S1 -> memtile (c,1) S2MM0, kind=wr: memtile MM2S0 -> shim S2MM0 -> DDR,
 *             4 MiB per column per rep, columns 0..cols-1 concurrently, cols = 1..4, DDR_REPS reps each.
 *             Every rep is timed separately on CNTPCT (push of all columns -> all columns drained).
 *     "MB ddr kind=rd|wr cols=<c> reps=<R> bytes=<total> cntpct=<t> cntpct_min_rep=<t> cntpct_med_rep=<t> cntfrq=<f>
 *      aie_cyc=<c> data=OK"
 *     bytes= is the TOTAL over all reps and all cols (DDR_BYTES * cols * reps); cntpct= is the SUM over reps;
 *     cntpct_min_rep / cntpct_med_rep are the fastest / median single rep (median of an odd DDR_REPS = one rep).
 *     One rep moves bytes/reps, so aggregate GB/s of a rep = bytes/reps / (cntpct_rep/cntfrq) / 1e9.
 *   ddrburst  shim (0,0) MM2S1 one 2-D BD: d0 = <bytes> contiguous (one burst), d1 = n bursts at stride 4096 B
 *             (every burst a different 4 KiB page) -> memtile (0,1) S2MM0 linear BD of bytes*n.
 *             Timed IN HARDWARE at the memtile S2MM (shim FINISHED_TASK fires early, knowledge
 *             2026-11-10-aie2ps-shim-perfctr-semantics): memtile perf ctr0 S2MM_SEL0_START_TASK -> S2MM_SEL0_FINISHED_TASK.
 *             Two n per size (DB_NLO, DB_NHI): the fit takes the slope (cyc_hi-cyc_lo)/(n_hi-n_lo) = AIE cycles per
 *             burst with the fixed start-up (first DDR access, push MMIO) cancelled.
 *     "MB ddrburst bytes=<b> stride=4096 n=<bursts> cyc=<aie cyc> data=OK"
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

void routing(XAie_DevInst *) {}
__global__ void loop_kernel(int *in, int *out) { (void)in; (void)out; }

static void say(const char *fmt, ...) {
    char b[320];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    printf("%s\n", b);
    fflush(stdout);
}
static inline uint64_t cntpct(void) { uint64_t v; __asm__ volatile("isb; mrs %0, cntpct_el0" : "=r"(v)); return v; }
static inline uint64_t cntfrq(void) { uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }

static int g_fail = 0;
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        AieRC _r = (AieRC)(x);                                                                                         \
        if (_r != XAIE_OK) {                                                                                           \
            say("MB rcfail line=%d rc=%d", __LINE__, (int)_r);                                                          \
            g_fail++;                                                                                                  \
        }                                                                                                              \
    } while (0)

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
    uint64_t dt = ((c1 + c1b) - (c0 + c0b)) / 2;
    say("MB clk aie_cyc=%llu cntpct=%llu cntfrq=%llu", (unsigned long long)(t1 - t0), (unsigned long long)dt,
        (unsigned long long)f);
}

static int pend(XAie_DevInst *D, XAie_LocType t, u8 ch, XAie_DmaDirection dir) {
    u8 p = 0xff;
    if (XAie_DmaGetPendingBdCount(D, t, ch, dir, &p) != XAIE_OK) return -1;
    return p;
}

#define MT_ROW 1
#define MT_DMA_HI 0x80000u
#define MT_LEN (256u << 10)
#define MT_R_OFF 0u
#define MT_W_OFF (256u << 10)
#define DDR_BYTES (4u << 20) /* per column per rep */
#define DDR_NCOL 4
#define DDR_REPS 9 /* odd: the median is one real rep */
static AieRC shimbd(XAie_DevInst *D, u8 col, XAie_MemInst *m, uint32_t n, u16 b) {
    XAie_LocType S = XAie_TileLoc(col, 0);
    XAie_DmaDesc d;
    return (AieRC)(XAie_DmaDescInit(D, &d, S) | XAie_DmaSetAddrLen(&d, XAie_MemGetDevAddr(m), n) |
                   XAie_DmaSetAxi(&d, 0U, 16U, 0U, 0U, 0U) | XAie_DmaEnableBd(&d) | XAie_DmaWriteBd(D, &d, S, b));
}
static AieRC mtbd(XAie_DevInst *D, u8 col, uint32_t off, uint32_t len, u16 b) {
    XAie_LocType M = XAie_TileLoc(col, MT_ROW);
    XAie_DmaDesc d;
    return (AieRC)(XAie_DmaDescInit(D, &d, M) | XAie_DmaSetAddrLen(&d, (u64)(MT_DMA_HI | off), len) |
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

/* kind 0 = read (DDR -> memtile), 1 = write (memtile -> DDR); columns 0..ncol-1 concurrently */
static void ddr_point(XAie_DevInst *D, int kind, int ncol, XAie_MemInst **buf) {
    const uint32_t n = DDR_BYTES, rep = n / MT_LEN;
    for (int c = 0; c < ncol; c++) {
        uint32_t *p = (uint32_t *)XAie_MemGetVAddr(buf[c]);
        const uint32_t salt = 0x5C000000u + 64 * kind + 8 * ncol + c;
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
    uint64_t tk = 0, cy = 0, tr[DDR_REPS];
    int fails = 0, nr = 0;
    for (int r = 0; r < DDR_REPS && !fails; r++) {
        AieRC rc = XAIE_OK;
        for (int c = 0; c < ncol; c++)
            rc = (AieRC)(rc | (kind == 0 ? (shimbd(D, c, buf[c], n, 0) | mtbd(D, c, MT_R_OFF, MT_LEN, 0))
                                         : (shimbd(D, c, buf[c], n, 1) | mtbd(D, c, MT_W_OFF, MT_LEN, 1))));
        u64 a0 = 0, a1 = 0;
        XAie_ReadTimer(D, XAie_TileLoc(0, 0), XAIE_PL_MOD, &a0);
        uint64_t c0 = cntpct();
        for (int c = 0; c < ncol; c++) {
            XAie_LocType S = XAie_TileLoc(c, 0), M = XAie_TileLoc(c, MT_ROW);
            if (kind == 0)
                rc = (AieRC)(rc | XAie_DmaChannelSetStartQueue(D, M, 0, DMA_S2MM, 0, rep, XAIE_DISABLE) |
                             XAie_DmaChannelPushBdToQueue(D, S, 1, DMA_MM2S, 0));
            else
                rc = (AieRC)(rc | XAie_DmaChannelPushBdToQueue(D, S, 0, DMA_S2MM, 1) |
                             XAie_DmaChannelSetStartQueue(D, M, 0, DMA_MM2S, 1, rep, XAIE_DISABLE));
        }
        int left = ncol;
        for (uint32_t p = 0; p < 200000000u && left; p++) {
            left = 0;
            for (int c = 0; c < ncol; c++) {
                int x = kind == 0 ? pend(D, XAie_TileLoc(c, MT_ROW), 0, DMA_S2MM) : pend(D, XAie_TileLoc(c, 0), 0, DMA_S2MM);
                int y = kind == 0 ? pend(D, XAie_TileLoc(c, 0), 1, DMA_MM2S) : 0;
                left += (x | y) != 0;
            }
        }
        uint64_t c1 = cntpct();
        XAie_ReadTimer(D, XAie_TileLoc(0, 0), XAIE_PL_MOD, &a1);
        if (rc != XAIE_OK || left) {
            say("MB ddr TIMEOUT kind=%d cols=%d rep=%d rc=%d left=%d", kind, ncol, r, (int)rc, left);
            fails++;
            break;
        }
        tk += c1 - c0;
        cy += a1 - a0;
        tr[nr++] = c1 - c0;
    }
    int bad = 0;
    if (!fails)
        for (int c = 0; c < ncol; c++) {
            const uint32_t salt = 0x5C000000u + 64 * kind + 8 * ncol + c;
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
    /* insertion sort of the per-rep times: min = tr[0], median = tr[nr/2] */
    for (int i = 1; i < nr; i++)
        for (int j = i; j > 0 && tr[j - 1] > tr[j]; j--) {
            uint64_t t = tr[j]; tr[j] = tr[j - 1]; tr[j - 1] = t;
        }
    say("MB ddr kind=%s cols=%d reps=%d bytes=%llu cntpct=%llu cntpct_min_rep=%llu cntpct_med_rep=%llu cntfrq=%llu "
        "aie_cyc=%llu data=%s",
        kind == 0 ? "rd" : "wr", ncol, nr, (unsigned long long)n * ncol * nr, (unsigned long long)tk,
        (unsigned long long)(nr ? tr[0] : 0), (unsigned long long)(nr ? tr[nr / 2] : 0), (unsigned long long)cntfrq(),
        (unsigned long long)cy, (bad || fails) ? "BAD" : "OK");
    if (bad || fails) g_fail++;
}

/* ------------------------------------------------------------------ strided small bursts */
#define DB_STRIDE 4096u
#define DB_NLO 64u
#define DB_NHI 512u
static u32 burst_one(XAie_DevInst *D, XAie_MemInst *m, u32 b, u32 nb, int *bad) {
    XAie_LocType S = XAie_TileLoc(0, 0), M = XAie_TileLoc(0, MT_ROW);
    const u32 tot = b * nb;
    for (u32 i = 0; i < tot / 4; i++) g_mt[i] = 0xDEADBEEFu;
    CK(XAie_DataMemBlockWrite(D, M, MT_R_OFF, g_mt, tot));
    XAie_DmaDesc d;
    XAie_DmaDimDesc dims[2];
    dims[0].AieMlDimDesc.StepSize = 1; /* 32-bit words */
    dims[0].AieMlDimDesc.Wrap = (u16)(b / 4);
    dims[1].AieMlDimDesc.StepSize = DB_STRIDE / 4;
    dims[1].AieMlDimDesc.Wrap = (u16)nb;
    XAie_DmaTensor ts = {2, dims};
    CK(XAie_DmaDescInit(D, &d, S));
    CK(XAie_DmaSetMultiDimAddr(&d, &ts, XAie_MemGetDevAddr(m), tot));
    CK(XAie_DmaSetAxi(&d, 0U, 16U, 0U, 0U, 0U));
    CK(XAie_DmaEnableBd(&d));
    CK(XAie_DmaWriteBd(D, &d, S, 2));
    CK(mtbd(D, 0, MT_R_OFF, tot, 2));
    CK(XAie_EventSelectDmaChannel(D, M, 0, DMA_S2MM, 0));
    CK(XAie_PerfCounterControlSet(D, M, XAIE_MEM_MOD, 0, XAIE_EVENT_DMA_S2MM_SEL0_START_TASK_MEM_TILE,
                                  XAIE_EVENT_DMA_S2MM_SEL0_FINISHED_TASK_MEM_TILE));
    CK(XAie_PerfCounterSet(D, M, XAIE_MEM_MOD, 0, 0));
    CK(XAie_DmaChannelPushBdToQueue(D, M, 0, DMA_S2MM, 2)); /* receiver first: ctr starts here */
    CK(XAie_DmaChannelPushBdToQueue(D, S, 1, DMA_MM2S, 2));
    int left = 1;
    for (u32 p = 0; p < 20000000u && left; p++) left = pend(D, M, 0, DMA_S2MM) != 0 || pend(D, S, 1, DMA_MM2S) != 0;
    if (left) {
        say("MB ddrburst TIMEOUT bytes=%u n=%u", b, nb);
        g_fail++;
        *bad = 1;
        return 0;
    }
    u32 cyc = 0;
    CK(XAie_PerfCounterGet(D, M, XAIE_MEM_MOD, 0, &cyc));
    CK(XAie_DataMemBlockRead(D, M, MT_R_OFF, g_mt, tot));
    const uint32_t *src = (const uint32_t *)XAie_MemGetVAddr(m);
    int e = 0;
    for (u32 i = 0; i < nb; i++)
        for (u32 j = 0; j < b / 4; j++) e += g_mt[i * (b / 4) + j] != src[i * (DB_STRIDE / 4) + j];
    *bad = e;
    return cyc;
}
static void bench_burst(XAie_DevInst *D, XAie_MemInst *m) {
    uint32_t *p = (uint32_t *)XAie_MemGetVAddr(m);
    for (uint32_t i = 0; i < DB_NHI * DB_STRIDE / 4; i++) p[i] = pat(i, 0x7B000000u);
    XAie_MemSyncForDev(m);
    static const u32 sizes[] = {32, 64, 128, 256};
    const u32 ns[2] = {DB_NLO, DB_NHI};
    for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; s++)
        for (int k = 0; k < 2; k++) {
            u32 mn = 0xFFFFFFFFu;
            int badall = 0;
            for (int r = 0; r < 3; r++) {
                int bad = 0;
                u32 c = burst_one(D, m, sizes[s], ns[k], &bad);
                badall += bad;
                if (c < mn) mn = c;
            }
            say("MB ddrburst bytes=%u stride=%u n=%u cyc=%u data=%s", sizes[s], DB_STRIDE, ns[k], mn,
                badall ? "BAD" : "OK");
            if (badall) g_fail++;
        }
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
    for (int kind = 0; kind < 2; kind++)
        for (int nc = 1; nc <= DDR_NCOL; nc++) ddr_point(D, kind, nc, b);
    bench_burst(D, b[0]);
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
    say("MB start ddr_microbench v1 section=ddr");
    bench_clk(&DevInst);
    bench_ddr(&DevInst);
    say("MB END rc=%d %s", g_fail, g_fail == 0 ? "PASS" : "FAIL");
    fflush(stdout);
    return g_fail ? 1 : 0;
}
