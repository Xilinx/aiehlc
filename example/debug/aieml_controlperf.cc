/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*
 * aieml_controlperf.cc — control-plane API microbenchmark.
 *
 * Measures the HOST-side control cost of each XAie / __Runtime_* API that
 * host_canonicalized() in aout/worklocal/host.cc issues, by calling each API
 * in a loop and timing it with XTime. Every __Runtime_* config call is a
 * series of AXI-MM register writes to the device, so the per-call number is the
 * host wall time spent programming the AIE array — the dominant control latency
 * behind a "slow" matmul.
 *
 * Lightweight config/createio/startio/wait calls run LIGHT_ITERS (default 1000);
 * heavy ELF-load / core-enable ops run HEAVY_ITERS (default 20). Both report a
 * per-call average. No real DMA completion or routing is set up: startio and
 * wait_io are measured as isolated host-issue cost only.
 *
 * Raw-write comparison: every register-write API is paired (via BENCH2) with a
 * burst of raw XAie_Write32 4-byte writes to the SAME BD/queue register file,
 * issuing the exact number of words the API programs internally (a shim BD = 9
 * words on AIE2PS / 8 on AIEML, a core BD = 6 words, a channel/queue/lock op =
 * 1 word). Each row prints: nW (register writes), api_us, raw_us, delta_us
 * (api-raw = the wrapper's host CPU overhead) and ns/wr (raw_us/nW = one real
 * AXI-MM write). This separates the true device-write cost from the descriptor-
 * building the __Runtime_* wrappers do on the host.
 *
 * Build (single-kernel flow — the dummy kernel below is extracted into a kernel
 * ELF; aie_runtime.c is linked in):
 *   source script/aiehlc.sh --aie-version 5 \
 *       --runtime-source-file example/debug/aieml_controlperf.cc
 * Run:
 *   python3 script/test/apppaltest.py aout/worklocal/build/host
 */

#include "xaiengine.h"
#include <math.h>
#include <stdio.h>

#ifndef __AIESIM__
#include "xil_cache.h"
#include "xil_printf.h"
#include "xil_io.h" /* Xil_Out32 / Xil_In32 — BSP register access layer */
#include "sleep.h"  /* usleep() in the standalone BSP */
#if AIE_GEN <= 2
#include "xtime_l.h"
#else
#include "xiltimer.h"
#endif
#else
#include <unistd.h> /* usleep() on the host (aiesim) */
#endif              /* __AIESIM__ */

#define uint_TYPE uint32_t

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#endif

#if AIE_GEN <= 2

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

#define XAIE_NUM_ROWS 11
#define XAIE_NUM_COLS 38
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8
#else

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 4

#endif

// Core trace API + __Runtime_* wrappers, linked in from src/mlir/runtime by
// aiehlc.sh's RUNTIME_SRCS. Included after xaiengine.h so the guarded
// <aie_codegen_inc/xaie_routing.h> include is skipped (XAIE_ROUTING_H already
// defined by xaiengine.h).
#include "aie_runtime.h"

// Iteration counts. Lightweight register-write / CPU-only APIs run LIGHT_ITERS;
// heavy ELF-load / core-enable APIs run HEAVY_ITERS.
#ifndef LIGHT_ITERS
#define LIGHT_ITERS 1000
#endif
#ifndef HEAVY_ITERS
#define HEAVY_ITERS 20
#endif

// Set to 1 to rewrite the AIE aperture's memory attribute (via EL3 page-table
// patch) BEFORE running the benchmark, so the register-write timings are taken
// under the NEW attribute. Default 0: probe/patch is only demonstrated, the
// benchmark runs under the original Device-nGnRnE mapping. See patch_aie_attr().
#ifndef AIE_ATTR_PATCH
#define AIE_ATTR_PATCH 0
#endif
// MAIR_EL3 attribute index to install into the AIE descriptor (bits [4:2]).
// Inspect the MAIR value printed by probe_bd_addr_attr() to pick a slot: MAIR
// byte i is the memory type selected by AttrIndx=i.
#ifndef AIE_NEW_ATTRINDX
#define AIE_NEW_ATTRINDX 3u
#endif

// Volatile sink: keeps the optimizer from eliding pure-CPU calls (createio,
// DmaDescInit, TileLoc) that have no device side effect. Register-write APIs
// (bd_config, startio, WriteBd, ...) already have observable MMIO side effects.
static volatile uint64_t g_sink = 0;

// Embedded kernel ELF symbol produced by aiehlc's `ld -r -b binary` step
// (name = _binary_kernel_<kernelname>_start). The generated host.cc also emits
// this extern; declaring it here lets us pass the ELF straight to
// __Runtime_set_kernel_elf (the frontend only rewrites kernel-name references
// that appear inside XAie_LoadElfMem, not this one).
extern unsigned char _binary_kernel_controlperf_dummy_start[];

// Dummy kernel so the aiehlc single-kernel flow still produces a kernel ELF.
// The cores never actually compute in this control-plane benchmark; the body is
// intentionally empty (load_kernel_group_16t just streams this ELF into tiles).
__global__ void controlperf_dummy(input_window_int32 *win
                                  __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:512"))),
                                  output_window_int32 *out
                                  __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:512")))) {
    (void)win;
    (void)out;
}

// Time `stmt` over `iters` iterations, print label / iters / total us / us-per-call.
#define BENCH(label, iters, stmt)                                                                                      \
    do {                                                                                                               \
        XTime _t0, _t1;                                                                                                \
        XTime_GetTime(&_t0);                                                                                           \
        for (long _i = 0; _i < (long)(iters); _i++) {                                                                  \
            stmt;                                                                                                      \
        }                                                                                                              \
        XTime_GetTime(&_t1);                                                                                           \
        double _tot_us = 1.0 * (double)(_t1 - _t0) / ((double)COUNTS_PER_SECOND / 1000000.0);                          \
        double _us_call = _tot_us / (double)(iters);                                                                   \
        printf("%-36s %6ld %12.2f %12.4f\n", (label), (long)(iters), _tot_us, _us_call);                               \
        fflush(stdout);                                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Raw AXI-MM register-write reference path
// ---------------------------------------------------------------------------
// Every __Runtime_* config API ultimately issues a fixed burst of 32-bit AXI-MM
// register writes: a shim BD is 9 words (AIE2PS) / 8 words (AIEML), a core BD is
// 6 words, and a channel / start-queue / lock op is a single word. The API's
// us/call therefore folds together (a) the host CPU cost of building the
// descriptor (XAie_DmaDescInit + the XAie_DmaSet* field packers) and (b) the
// actual bus-write cost. To separate the two we re-issue EXACTLY that many
// XAie_Write32 4-byte writes straight into the same BD / queue register file and
// time that alone. Then:
//     raw_us              = pure device-write cost of N AXI-MM writes
//     raw_us / N          = one real 4-byte AXI-MM write
//     api_us - raw_us     = host CPU overhead the wrapper adds on top of the bus
// The register-file geometry mirrors aie_runtime_debug.h and the aie-rt reginit
// tables (Aie2PS/AieMl {Shim,Tile}DmaMod). Offsets are tile-relative; the
// baremetal XAie_Write32 backend adds XAIE_BASE_ADDR and the (row,col) tile
// address we OR in below (same address math as AieRt_PrintShimBdRawAll).
#if AIE_GEN <= 2
#define RAW_SHIM_BD_BASE 0x0001D000u /* NOC_MODULE_DMA_BD0_0    */
#define RAW_SHIM_BD_STEP 0x20u
#define RAW_SHIM_BD_WORDS 8
#define RAW_SHIM_QUEUE 0x0001D204u /* NOC_MODULE_DMA_S2MM_0_TASK_QUEUE  */
#else
#define RAW_SHIM_BD_BASE 0x00009000u /* NOC_MODULE_DMA_BD0_0    */
#define RAW_SHIM_BD_STEP 0x30u
#define RAW_SHIM_BD_WORDS 9
#define RAW_SHIM_QUEUE 0x00009304u /* NOC_MODULE_DMA_S2MM_0_TASK_QUEUE  */
#endif
#define RAW_CORE_BD_BASE 0x0001D000u /* MEMORY_MODULE_DMA_BD0_0 (both gens) */
#define RAW_CORE_BD_STEP 0x20u
#define RAW_CORE_BD_WORDS 6
#define RAW_CORE_QUEUE 0x0001DE04u /* MEMORY_MODULE_DMA_S2MM_0_START_QUEUE */

// A high BD slot reserved for the raw path so its writes never overwrite the
// BD registers the real-API benchmarks program (all timing-only, but keep them
// disjoint for clarity). 16 BDs per tile, so slot 15 is always valid.
#define RAW_BD_SLOT 15

// Absolute register address of word `w` of BD `bd` on `tile`. row/col occupy the
// XAIE_{ROW,COL}_SHIFT fields; the backend adds XAIE_BASE_ADDR. Matches
// XAie_GetTileAddr(dev,row,col) + DmaMod->BaseAddr + bd*IdxOffset + w*4.
static inline uint64_t raw_bd_addr(XAie_LocType tile, uint32_t base, uint32_t step, int bd, int w) {
    return 0x2000031d000ULL;
    // return ((uint64_t)tile.Col << XAIE_COL_SHIFT) | ((uint64_t)tile.Row << XAIE_ROW_SHIFT) |
    //        (uint64_t)(base + (uint32_t)bd * step + (uint32_t)w * 4u);
}

// Issue `words` back-to-back 4-byte AXI-MM writes into `tile`'s BD `bd` register
// file — the raw analogue of one XAie_DmaWriteBd / __Runtime_dma_bd_config.
static inline void raw_bd_write(XAie_DevInst *dev, XAie_LocType tile, int bd, uint32_t base, uint32_t step, int words) {
    for (int w = 0; w < words; w++)
        XAie_Write32(dev, raw_bd_addr(tile, base, step, bd, w), 0xB0000000u | (uint32_t)w);
}

static inline void raw_bd_write_group_dsb(XAie_DevInst *dev, XAie_LocType tile, int bd, uint32_t base, uint32_t step,
                                          int words) {
    for (int w = 0; w < words; w++) {
        // XAie_Write32(dev, raw_bd_addr(tile, base, step, bd, w), 0xB0000000u | (uint32_t)w);
        *(volatile u32 *)(raw_bd_addr(tile, base, step, bd, w)) = 0xB0000000u | (uint32_t)w;
    }
    // dsb(sy);
}

// One 32-bit AXI-MM write to a tile control/queue register — the raw analogue of
// startio / channel-enable / lock-set (each one register write).
static inline void raw_reg_write(XAie_DevInst *dev, XAie_LocType tile, uint32_t reg_off, uint32_t val) {
    XAie_Write32(dev, /*((uint64_t)tile.Col << XAIE_COL_SHIFT) | ((uint64_t)tile.Row << XAIE_ROW_SHIFT) */ 0x300000 |
                          (uint64_t)reg_off,
                 val);
}

// ---------------------------------------------------------------------------
// 4-layer register access: SAME device register, three abstraction levels
// ---------------------------------------------------------------------------
// A runtime __Runtime_* config API ultimately becomes a burst of 32-bit stores
// through this software stack:
//     __Runtime_*  ->  XAie_Write32 (aie-rt)  ->  Xil_Out32 (BSP)  ->  *(volatile u32*)
// The helpers below re-issue EXACTLY the same words at each lower level, all
// hitting the identical device register, so BENCH4 can attribute cost per layer.
//
// Address convention (verified against xaie_baremetal.c:153,193):
//   XAie_Write32(dev, off, v) -> Xil_Out32(BaseAddr + off, v), BaseAddr == XAIE_BASE_ADDR.
// So XAie_Write32 takes the tile-relative OFFSET, while Xil_Out32 / the bare
// volatile store take the ABSOLUTE address = XAIE_BASE_ADDR + off. (The older
// raw_bd_write_group_dsb path wrote a bare low offset with no BaseAddr, i.e. a
// cached-DRAM alias rather than the device — do not use it for cost comparison.)

// Tile-relative register offset of word `w` of BD `bd` (argument to XAie_Write32).
static inline uint64_t reg_off_of(XAie_LocType tile, uint32_t base, uint32_t step, int bd, int w) {
    return ((uint64_t)tile.Col << XAIE_COL_SHIFT) | ((uint64_t)tile.Row << XAIE_ROW_SHIFT) |
           (uint64_t)(base + (uint32_t)bd * step + (uint32_t)w * 4u);
}
// Absolute CPU address of the same register (argument to Xil_Out32 / volatile).
static inline uint64_t reg_abs_of(XAie_LocType tile, uint32_t base, uint32_t step, int bd, int w) {
    return (uint64_t)XAIE_BASE_ADDR + reg_off_of(tile, base, step, bd, w);
}

// Layer 2: aie-rt driver write — XAie_Write32 -> backend -> Xil_Out32 -> volatile.
static inline void words_write_xaie(XAie_DevInst *dev, XAie_LocType tile, int bd, uint32_t base, uint32_t step,
                                    int words) {
    for (int w = 0; w < words; w++)
        XAie_Write32(dev, reg_off_of(tile, base, step, bd, w), 0xB0000000u | (uint32_t)w);
}
// Layer 3: BSP write — Xil_Out32 -> volatile store (one software frame less).
static inline void words_write_xil(XAie_LocType tile, int bd, uint32_t base, uint32_t step, int words) {
#ifndef __AIESIM__
    for (int w = 0; w < words; w++)
        Xil_Out32((UINTPTR)reg_abs_of(tile, base, step, bd, w), 0xB0000000u | (uint32_t)w);
#else
    (void)tile;
    (void)bd;
    (void)base;
    (void)step;
    (void)words;
#endif
}
// Layer 4: bare compiler-emitted volatile store, no wrapper at all.
static inline void words_write_vol(XAie_LocType tile, int bd, uint32_t base, uint32_t step, int words) {
#ifndef __AIESIM__
    for (int w = 0; w < words; w++)
        *(volatile u32 *)(uintptr_t)reg_abs_of(tile, base, step, bd, w) = 0xB0000000u | (uint32_t)w;
#else
    (void)tile;
    (void)bd;
    (void)base;
    (void)step;
    (void)words;
#endif
}

// Read counterparts (XAie_Read32 -> Xil_In32 -> volatile load). Accumulate into
// g_sink so the loads are not optimized away.
static inline void words_read_xaie(XAie_DevInst *dev, XAie_LocType tile, int bd, uint32_t base, uint32_t step,
                                   int words) {
    u32 v;
    for (int w = 0; w < words; w++) {
        XAie_Read32(dev, reg_off_of(tile, base, step, bd, w), &v);
        g_sink += v;
    }
}
static inline void words_read_xil(XAie_LocType tile, int bd, uint32_t base, uint32_t step, int words) {
#ifndef __AIESIM__
    for (int w = 0; w < words; w++)
        g_sink += Xil_In32((UINTPTR)reg_abs_of(tile, base, step, bd, w));
#else
    (void)tile;
    (void)bd;
    (void)base;
    (void)step;
    (void)words;
#endif
}
static inline void words_read_vol(XAie_LocType tile, int bd, uint32_t base, uint32_t step, int words) {
#ifndef __AIESIM__
    for (int w = 0; w < words; w++)
        g_sink += *(volatile u32 *)(uintptr_t)reg_abs_of(tile, base, step, bd, w);
#else
    (void)tile;
    (void)bd;
    (void)base;
    (void)step;
    (void)words;
#endif
}

// Time an API call and its raw N-write32 equivalent back-to-back over `iters`
// loops. Columns: API | nW | api_us | raw_us | delta_us(api-raw) | ns/write.
//   nW        register writes the API issues internally
//   api_us    per-call wall time of the real __Runtime_*/XAie_* wrapper
//   raw_us    per-call wall time of nW XAie_Write32 to the same register file
//   delta_us  api_us - raw_us  = host CPU overhead of the wrapper
//   ns/write  raw_us / nW      = one real 4-byte AXI-MM write
#define BENCH2(label, iters, nwrites, api_stmt, raw_stmt)                                                              \
    do {                                                                                                               \
        XTime _a0, _a1, _r0, _r1;                                                                                      \
        XTime_GetTime(&_a0);                                                                                           \
        for (long _i = 0; _i < (long)(iters); _i++) {                                                                  \
            api_stmt;                                                                                                  \
        }                                                                                                              \
        XTime_GetTime(&_a1);                                                                                           \
        XTime_GetTime(&_r0);                                                                                           \
        for (long _i = 0; _i < (long)(iters); _i++) {                                                                  \
            raw_stmt;                                                                                                  \
        }                                                                                                              \
        XTime_GetTime(&_r1);                                                                                           \
        double _cps = (double)COUNTS_PER_SECOND / 1000000.0;                                                           \
        double _api = (double)(_a1 - _a0) / _cps / (double)(iters);                                                    \
        double _raw = (double)(_r1 - _r0) / _cps / (double)(iters);                                                    \
        printf("%-30s %3d %10.4f %10.4f %10.4f %9.1f\n", (label), (int)(nwrites), _api, _raw, _api - _raw,             \
               (nwrites) ? _raw / (double)(nwrites) * 1000.0 : 0.0);                                                   \
        fflush(stdout);                                                                                                \
    } while (0)

// Time one runtime API against the SAME N register accesses issued three ways at
// descending abstraction levels, all hitting the identical device register:
//   api_us   full __Runtime_* / XAie_* wrapper
//   xaie_us  N x XAie_Write32/Read32  (aie-rt driver: wrapper + backend dispatch)
//   xil_us   N x Xil_Out32/Xil_In32   (BSP inline: one volatile access, no dispatch)
//   vol_us   N x bare *(volatile u32*) (compiler store/load, zero software layers)
//   ns/acc   vol_us / N = the pure AXI-MM bus cost with all software stripped
// Reading down api->xaie->xil->vol shows the host CPU overhead each layer adds on
// top of the identical bus transaction; the tail (xil vs vol) should be ~equal,
// confirming Xil_Out32 compiles to just the volatile store (no dsb / barrier).
#define BENCH4(label, iters, naccess, api_stmt, xaie_stmt, xil_stmt, vol_stmt)                                         \
    do {                                                                                                               \
        XTime _a0, _a1, _b0, _b1, _c0, _c1, _d0, _d1;                                                                  \
        XTime_GetTime(&_a0);                                                                                           \
        for (long _i = 0; _i < (long)(iters); _i++) {                                                                  \
            api_stmt;                                                                                                  \
        }                                                                                                              \
        XTime_GetTime(&_a1);                                                                                           \
        XTime_GetTime(&_b0);                                                                                           \
        for (long _i = 0; _i < (long)(iters); _i++) {                                                                  \
            xaie_stmt;                                                                                                 \
        }                                                                                                              \
        XTime_GetTime(&_b1);                                                                                           \
        XTime_GetTime(&_c0);                                                                                           \
        for (long _i = 0; _i < (long)(iters); _i++) {                                                                  \
            xil_stmt;                                                                                                  \
        }                                                                                                              \
        XTime_GetTime(&_c1);                                                                                           \
        XTime_GetTime(&_d0);                                                                                           \
        for (long _i = 0; _i < (long)(iters); _i++) {                                                                  \
            vol_stmt;                                                                                                  \
        }                                                                                                              \
        XTime_GetTime(&_d1);                                                                                           \
        double _cps = (double)COUNTS_PER_SECOND / 1000000.0;                                                           \
        double _api = (double)(_a1 - _a0) / _cps / (double)(iters);                                                    \
        double _xaie = (double)(_b1 - _b0) / _cps / (double)(iters);                                                   \
        double _xil = (double)(_c1 - _c0) / _cps / (double)(iters);                                                    \
        double _vol = (double)(_d1 - _d0) / _cps / (double)(iters);                                                    \
        printf("%-28s %3d %9.4f %9.4f %9.4f %9.4f %8.1f\n", (label), (int)(naccess), _api, _xaie, _xil, _vol,          \
               (naccess) ? _vol / (double)(naccess) * 1000.0 : 0.0);                                                   \
        fflush(stdout);                                                                                                \
    } while (0)

int run_control_perf(XAie_DevInst *dev) {
    // gen5 (AIE2PS): row 0 = shim, rows 1-2 = MemTile, rows 3-6 = AIE cores.
    XAie_LocType shim = XAie_TileLoc(0, 0);
    XAie_LocType core = XAie_TileLoc(0, 3);
    XAie_LocType memtile = XAie_TileLoc(0, 1);
    (void)memtile;

    // One DMA-capable DDR buffer for shim BD addresses.
    void *buf = __Runtime_alloc_buffer(dev, 16384);
    if (!buf) {
        printf("[controlperf] alloc_buffer failed\n");
        return -1;
    }
    // Core tiles take a DMA-view byte address (core_proc_addr - 0x70000); a small
    // in-range L1 offset is enough for the register-write path.
    void *core_buf = (void *)(uintptr_t)0x1000;

    printf("\n==== 1 AIE control-plane API microbenchmark (AIE_GEN=%d) ====\n", (int)AIE_GEN);
    printf("Each __Runtime_* config call = a burst of AXI-MM register writes to the device.\n");
    printf("%-36s %6s %12s %12s\n", "API", "iters", "total_us", "us/call");
    printf("-------------------------------------------------------------------------\n");

    // ---- Baseline / CPU-only (no device register writes) ---------------------
    BENCH("XAie_TileLoc (cpu)", LIGHT_ITERS, {
        XAie_LocType _l = XAie_TileLoc(0, 3);
        g_sink += (uint64_t)_l.Row;
    });
    BENCH("XAie_DmaDescInit (cpu)", LIGHT_ITERS, {
        XAie_DmaDesc _dd;
        XAie_DmaDescInit(dev, &_dd, core);
        g_sink += (uint64_t)((uint8_t *)&_dd)[0];
    });
    {
        XAie_DmaDesc _d0 = __Runtime_dma_bd_config(dev, shim, buf, 0, 1024, 0, 0, 0, 2, -1, 3, 1, -1);
        BENCH("dma_createio_4 (cpu struct)", LIGHT_ITERS, {
            io _x = __Runtime_dma_createio_4(shim, _d0, 1, 0, DMA_S2MM);
            g_sink += (uint64_t)_x.bd_id;
        });
    }

    // ---- Register-write config APIs vs raw AXI-MM writes ---------------------
    // Each row runs the real __Runtime_*/XAie_* wrapper AND an equal-count burst
    // of raw XAie_Write32 into the same BD/queue register file, so the API's
    // per-call cost is decomposed into bus writes (raw_us) + host CPU overhead
    // (delta_us). See BENCH2 / the raw helpers above.
    printf("-------------------------------------------------------------------------\n");
    printf("%-30s %3s %10s %10s %10s %9s\n", "API vs raw write32", "nW", "api_us", "raw_us", "delta_us", "ns/wr");
    printf("-------------------------------------------------------------------------\n");

    // Simple BD (len 1024) — the shim S2MM output BD host.cc programs 120x.
    // Shim BD = RAW_SHIM_BD_WORDS register writes.
    BENCH2("dma_bd_config (shim,1024)", LIGHT_ITERS, RAW_SHIM_BD_WORDS,
           __Runtime_dma_bd_config(dev, shim, buf, 0, 1024, 0, 0, 0, 2, -1, 3, 1, -1),
           raw_bd_write_group_dsb(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS));
    // Core tile BD = RAW_CORE_BD_WORDS register writes.
    BENCH2("dma_bd_config (core,1024)", LIGHT_ITERS, RAW_CORE_BD_WORDS,
           __Runtime_dma_bd_config(dev, core, core_buf, 0, 1024, 0, 0, 0, 2, -1, 3, 1, -1),
           raw_bd_write_group_dsb(dev, core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, RAW_CORE_BD_WORDS));

    // Multi-dim 3D BD (len 4096, dims stride/wrap 4/16, 256/16, 64/4) — host.cc.
    // Multi-dim just fills more fields of the SAME shim BD (still one BD write).
    BENCH2("dma_bd_config_multidim (3D)", LIGHT_ITERS, RAW_SHIM_BD_WORDS,
           __Runtime_dma_bd_config_multidim(dev, shim, buf, 1, 4096, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 16, 256, 16, 64, 4,
                                            0, 0),
           raw_bd_write_group_dsb(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS));

    // Multi-dim 3D BD + OOO iteration (iter_step 4096, iter_wrap 4) — host.cc.
    BENCH2("dma_bd_config_multidim_ooo", LIGHT_ITERS, RAW_SHIM_BD_WORDS,
           __Runtime_dma_bd_config_multidim_ooo(dev, shim, buf, 0, 4096, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, 16, 256, 16,
                                                64, 4, 4096, 4),
           raw_bd_write_group_dsb(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS));

    // Raw XAie_DmaWriteBd alone (desc built once, CPU side): isolates the driver
    // BD-write path against a plain N-word XAie_Write32 burst to the same region.
    {
        XAie_DmaDesc _raw;
        XAie_DmaDescInit(dev, &_raw, shim);
        XAie_DmaSetAddrLen(&_raw, (uint64_t)(uintptr_t)buf, 1024);
        XAie_DmaEnableBd(&_raw);
        BENCH2("XAie_DmaWriteBd (shim)", LIGHT_ITERS, RAW_SHIM_BD_WORDS, XAie_DmaWriteBd(dev, &_raw, shim, 0),
               raw_bd_write_group_dsb(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS));
    }

    // Lock init + set value (host.cc issues these per tile before launch) = 1 write.
    BENCH2("XAie_LockSetValue+LockInit", LIGHT_ITERS, 1, XAie_LockSetValue(dev, core, XAie_LockInit(0, 1)),
           raw_bd_write_group_dsb(dev, core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, 1));

    // Enable OOO BD execution on a DMA channel (host.cc: 4 calls) = 1 write.
    BENCH2("dma_channel_enable_ooo", LIGHT_ITERS, 1, __Runtime_dma_channel_enable_ooo(dev, shim, 0, DMA_S2MM),
           raw_bd_write_group_dsb(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, 1));

    // ---- startio: isolated host-issue cost (XAie_DmaChannelSetStartQueue) -----
    // The channel is NOT drained, so its start queue saturates; we measure only
    // the non-blocking register-write issue cost, not DMA completion. The raw
    // analogue is one write straight to the shim S2MM task-queue register.
    {
        XAie_DmaDesc _sd = __Runtime_dma_bd_config(dev, shim, buf, 0, 1024, 0, 0, 0, 2, -1, 3, 1, -1);
        io _sio = __Runtime_dma_createio_4(shim, _sd, 0, 0, DMA_S2MM);
        BENCH2("startio (SetStartQueue)", LIGHT_ITERS, 1, __Runtime_startio(dev, _sio, 0, 1),
               raw_reg_write(dev, shim, RAW_SHIM_QUEUE, 0x00000001u));
    }

    // ---- wait_io: one status poll on an IDLE channel (0 pending -> returns) ---
    // Pick a channel never started so XAie_DmaGetPendingBdCount reports 0 on the
    // first read and the wait returns after a single AXI-MM status read. This is
    // a READ path, so its raw analogue is one XAie_Read32 (delta_us then exposes
    // the wrapper cost over a single AXI-MM read; ns/wr is one 4-byte read).
    {
        XAie_DmaDesc _wd = __Runtime_dma_bd_config(dev, core, core_buf, 1, 1024, 0, 0, 0, 2, -1, 3, 1, -1);
        io _wio = __Runtime_dma_createio_4(core, _wd, 1, 1, DMA_S2MM);
        ioevent _wev;
        _wev.io = _wio;
        _wev.timeout_us = 10000;
        uint32_t _rdsink = 0;
        BENCH2("wait_io (idle poll, read)", LIGHT_ITERS, 1, __Runtime_wait_io(dev, _wev), {
            XAie_Read32(dev,
                        ((uint64_t)core.Col << XAIE_COL_SHIFT) | ((uint64_t)core.Row << XAIE_ROW_SHIFT) |
                            (uint64_t)RAW_CORE_QUEUE,
                        &_rdsink);
            g_sink += _rdsink;
        });
    }

    // ============================================================
    // 2. Same runtime API vs its lower-level implementations
    // ============================================================
    // Each row runs a real __Runtime_* / XAie_* API and then re-issues the SAME
    // number of 32-bit register accesses at three descending abstraction levels
    // (XAie_Write32 -> Xil_Out32 -> bare volatile), all to the identical device
    // register file. Comparing the columns isolates the per-layer host CPU cost
    // stacked on top of the one shared AXI-MM bus transaction. See BENCH4.
    printf("-------------------------------------------------------------------------\n");
    printf("%-28s %3s %9s %9s %9s %9s %8s\n", "API vs xaie/xil/volatile", "nW", "api_us", "xaie_us", "xil_us", "vol_us",
           "ns/acc");
    printf("-------------------------------------------------------------------------\n");

    // Shim S2MM output BD (RAW_SHIM_BD_WORDS words) — the dominant hot-path BD.
    BENCH4("dma_bd_config (shim,1024)", LIGHT_ITERS, RAW_SHIM_BD_WORDS,
           __Runtime_dma_bd_config(dev, shim, buf, 0, 1024, 0, 0, 0, 2, -1, 3, 1, -1),
           words_write_xaie(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS),
           words_write_xil(shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS),
           words_write_vol(shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS));

    // Core tile BD (RAW_CORE_BD_WORDS words).
    BENCH4("dma_bd_config (core,1024)", LIGHT_ITERS, RAW_CORE_BD_WORDS,
           __Runtime_dma_bd_config(dev, core, core_buf, 0, 1024, 0, 0, 0, 2, -1, 3, 1, -1),
           words_write_xaie(dev, core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, RAW_CORE_BD_WORDS),
           words_write_xil(core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, RAW_CORE_BD_WORDS),
           words_write_vol(core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, RAW_CORE_BD_WORDS));

    // Raw driver BD write (desc built once) — isolates XAie_DmaWriteBd's word
    // packing against the three low-level bursts to the same shim BD region.
    {
        XAie_DmaDesc _b4d;
        XAie_DmaDescInit(dev, &_b4d, shim);
        XAie_DmaSetAddrLen(&_b4d, (uint64_t)(uintptr_t)buf, 1024);
        XAie_DmaEnableBd(&_b4d);
        BENCH4("XAie_DmaWriteBd (shim)", LIGHT_ITERS, RAW_SHIM_BD_WORDS, XAie_DmaWriteBd(dev, &_b4d, shim, 0),
               words_write_xaie(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS),
               words_write_xil(shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS),
               words_write_vol(shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_SHIM_BD_WORDS));
    }

    // Single-register write op: channel-enable (1 word) at all layers. The raw
    // bursts target BD slot 15 word0 (a BD never started) to avoid clobbering
    // live queue state.
    BENCH4("dma_channel_enable_ooo (1w)", LIGHT_ITERS, 1, __Runtime_dma_channel_enable_ooo(dev, shim, 0, DMA_S2MM),
           words_write_xaie(dev, shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, 1),
           words_write_xil(shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, 1),
           words_write_vol(shim, RAW_BD_SLOT, RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, 1));

    // Single-register read op: runtime wait_io (one status read) vs XAie_Read32
    // -> Xil_In32 -> bare volatile load, all reading BD slot 15 word0.
    {
        XAie_DmaDesc _r4d = __Runtime_dma_bd_config(dev, core, core_buf, 1, 1024, 0, 0, 0, 2, -1, 3, 1, -1);
        io _r4io = __Runtime_dma_createio_4(core, _r4d, 1, 1, DMA_S2MM);
        ioevent _r4ev;
        _r4ev.io = _r4io;
        _r4ev.timeout_us = 10000;
        BENCH4("read32 (status, 1w)", LIGHT_ITERS, 1, __Runtime_wait_io(dev, _r4ev),
               words_read_xaie(dev, core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, 1),
               words_read_xil(core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, 1),
               words_read_vol(core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, 1));
    }

    // ---- Heavy ops: ELF load into 16 tiles + core enable ----------------------
    // The 16 compute tiles host.cc drives: cols 0-3 x rows 3-6.
    XAie_LocType kt[16] = {
        XAie_TileLoc(0, 3), XAie_TileLoc(0, 4), XAie_TileLoc(0, 5), XAie_TileLoc(0, 6),
        XAie_TileLoc(1, 3), XAie_TileLoc(1, 4), XAie_TileLoc(1, 5), XAie_TileLoc(1, 6),
        XAie_TileLoc(2, 3), XAie_TileLoc(2, 4), XAie_TileLoc(2, 5), XAie_TileLoc(2, 6),
        XAie_TileLoc(3, 3), XAie_TileLoc(3, 4), XAie_TileLoc(3, 5), XAie_TileLoc(3, 6),
    };
    __Runtime_set_kernel_elf(_binary_kernel_controlperf_dummy_start);
    kernel_group _kg = {0, 0, 0};
    BENCH("load_kernel_group_16t (ELF x16)", HEAVY_ITERS,
          _kg = __Runtime_load_kernel_group_16t(dev, kt[0], kt[1], kt[2], kt[3], kt[4], kt[5], kt[6], kt[7], kt[8],
                                                kt[9], kt[10], kt[11], kt[12], kt[13], kt[14], kt[15], 16));
    BENCH("launch_kernel_group (enable x16)", HEAVY_ITERS, {
        event _ev = __Runtime_launch_kernel_group(dev, _kg);
        g_sink += (uint64_t)_ev.num_tiles;
    });

    // ---- Aggregate: composite host control cost of one matmul iteration -------
    // Approximates host_canonicalized's per-iteration issue: 120 BD configs plus
    // 60 createio+startio pairs, reported as one wall-time number.
    {
        XAie_DmaDesc _ad = __Runtime_dma_bd_config(dev, core, core_buf, 0, 1024, 0, 0, 0, 2, -1, 3, 1, -1);
        XTime _g0, _g1;
        XTime_GetTime(&_g0);
        for (int i = 0; i < 120; i++) {
            __Runtime_dma_bd_config(dev, core, core_buf, i % 6, 1024, 0, 0, 0, 2, -1, 3, 1, -1);
        }
        for (int i = 0; i < 60; i++) {
            io _x = __Runtime_dma_createio_4(core, _ad, 0, 0, DMA_S2MM);
            __Runtime_startio(dev, _x, 0, 1);
        }
        XTime_GetTime(&_g1);
        double _us = 1.0 * (double)(_g1 - _g0) / ((double)COUNTS_PER_SECOND / 1000000.0);

        // Raw analogue of the same iteration's device traffic: 120 core-BD writes
        // (RAW_CORE_BD_WORDS each) + 60 core start-queue writes, no API/CPU work.
        XTime _rg0, _rg1;
        XTime_GetTime(&_rg0);
        for (int i = 0; i < 120; i++) {
            raw_bd_write(dev, core, RAW_BD_SLOT, RAW_CORE_BD_BASE, RAW_CORE_BD_STEP, RAW_CORE_BD_WORDS);
        }
        for (int i = 0; i < 60; i++) {
            raw_reg_write(dev, core, RAW_CORE_QUEUE, 0x00000001u);
        }
        XTime_GetTime(&_rg1);
        double _raw_us = 1.0 * (double)(_rg1 - _rg0) / ((double)COUNTS_PER_SECOND / 1000000.0);
        long _nwr = 120L * RAW_CORE_BD_WORDS + 60L;

        printf("-------------------------------------------------------------------------\n");
        printf("--- control overhead / matmul iter (120 bd_config + 60 createio+startio) ---\n");
        printf("      api=%.2f us   raw(%ld write32)=%.2f us   host_overhead=%.2f us   raw=%.1f ns/write\n", _us, _nwr,
               _raw_us, _us - _raw_us, _raw_us / (double)_nwr * 1000.0);
        fflush(stdout);
    }

    printf("==== control-plane microbenchmark done ====\n");
    fflush(stdout);
    return 0;
}

#ifndef __AIESIM__
// Decode the PAR_EL1 memory-attribute byte (bits [63:56]) produced by an AT
// (address-translation) instruction. Device memory has an upper nibble of 0 and
// bits [3:2] select the {nGnRnE, nGnRE, nGRE, GRE} gathering/ordering/early-ack
// variant; any other upper nibble is Normal (cacheable) memory.
static const char *par_mem_attr_str(uint64_t par) {
    if (par & 0x1ULL) /* F bit set => translation faulted (VA unmapped here) */
        return "FAULT (unmapped for this regime)";
    uint32_t attr = (uint32_t)((par >> 56) & 0xFFU);
    if ((attr & 0xF0U) == 0x00U) {
        switch (attr & 0x0CU) {
        case 0x00U:
            return "Device-nGnRnE";
        case 0x04U:
            return "Device-nGnRE";
        case 0x08U:
            return "Device-nGRE";
        default:
            return "Device-GRE";
        }
    }
    return "Normal memory (cacheable)";
}

// Translate one VA through the current EL1&0 regime (AT S1E1R) and, at EL3, also
// the EL3 regime (AT S1E3R), then print the decoded memory type from PAR_EL1.
// NOTE: every AT variant writes its result to PAR_EL1 — there is no separate
// PAR_EL3 register — so both reads use PAR_EL1. AT S1E3R is UNDEFINED below EL3,
// hence the CurrentEL guard.
static void probe_va(const char *label, uint64_t va, uint64_t cur_el) {
    uint64_t par_el1 = 0, par_el3 = 0;

    __asm__ volatile("at s1e1r, %1\n\t"
                     "isb\n\t"
                     "mrs %0, par_el1"
                     : "=r"(par_el1)
                     : "r"(va)
                     : "memory");
    if (cur_el >= 3ULL) {
        __asm__ volatile("at s1e3r, %1\n\t"
                         "isb\n\t"
                         "mrs %0, par_el1"
                         : "=r"(par_el3)
                         : "r"(va)
                         : "memory");
    }

    printf("[%-12s] VA=0x%012llx  S1E1R PAR=0x%016llx -> %-24s", label, (unsigned long long)va,
           (unsigned long long)par_el1, par_mem_attr_str(par_el1));
    if (cur_el >= 3ULL)
        printf("  |  S1E3R -> %s", par_mem_attr_str(par_el3));
    printf("\n");
    fflush(stdout);
}

// Compare the memory type of an AIE array register against a local DDR (stack)
// data address. ARM terminology: this is the "memory type / memory attributes"
// (Device vs Normal + cacheability/shareability), not a "memory-map attribute".
// Expected: the AIE BD register is Device-nGnRnE (non-posted, ~365 ns/write),
// while the DDR local is Normal cacheable (a fast cache hit) — which is exactly
// why raw_bd_addr()'s device write and a bare cached store differ ~400x.
static void probe_bd_addr_attr(void) {

    uint64_t mair, tcr, ttbr, sctlr;
    asm volatile("mrs %0, mair_el3" : "=r"(mair));
    asm volatile("mrs %0, tcr_el3" : "=r"(tcr));
    asm volatile("mrs %0, ttbr0_el3" : "=r"(ttbr));
    asm volatile("mrs %0, sctlr_el3" : "=r"(sctlr));

    uint64_t cur_el = 0;
    __asm__ volatile("mrs %0, CurrentEL" : "=r"(cur_el));
    cur_el = (cur_el >> 2) & 0x3ULL;

    // (1) AIE array BD register — expected Device-nGnRnE.
    uint64_t aie_va = raw_bd_addr(XAie_TileLoc(0, 0), RAW_SHIM_BD_BASE, RAW_SHIM_BD_STEP, RAW_BD_SLOT, 0);
    // (2) A local DDR data address (stack) — expected Normal cacheable.
    volatile uint32_t ddr_local = 0xA5A5A5A5u;
    uint64_t ddr_va = (uint64_t)(uintptr_t)&ddr_local;

    printf("\n==== memory-type probe (PAR_EL1 after AT translate) ====\n");
    printf("CurrentEL = EL%llu%s\n", (unsigned long long)cur_el,
           (cur_el >= 3ULL) ? "" : "  (AT S1E3R skipped below EL3)");
    printf("MAIR=%016llx TCR=%016llx TTBR0=%016llx SCTLR=%08llx\r\n", mair, tcr, ttbr, sctlr);

    probe_va("AIE BD reg", aie_va, cur_el);
    probe_va("DDR local", ddr_va, cur_el);
    g_sink += ddr_local; /* keep ddr_local live */
}

// ----------------------------------------------------------------------------
// EL3 page-table attribute patch for the AIE aperture.
//
// The AIE array registers are mapped (by the EL3 boot tables) as Device-nGnRnE,
// which makes every MMIO store a non-posted, ~365 ns round-trip. This routine
// walks the live EL3 stage-1 translation tables (TTBR0_EL3), finds the block
// descriptor that maps XAIE_BASE_ADDR, and rewrites its AttrIndx field (bits
// [4:2]) to AIE_NEW_ATTRINDX so the aperture is re-typed to whatever MAIR_EL3
// slot AIE_NEW_ATTRINDX names. It then cleans the patched descriptor line to
// PoC and invalidates the EL3 TLBs so the new attribute takes effect.
//
// This is a low-level, EL3-only, self-modifying-page-table operation. It only
// runs when AIE_ATTR_PATCH is compiled in (default off) and only when actually
// executing at EL3. It is intended purely as a bring-up/measurement experiment
// on this baremetal harness — it is not a production configuration path.

// Descriptor field masks (4 KB granule, 48-bit PA).
#define PTE_TYPE_MASK 0x3ULL // bits [1:0]: 3=table/page, 1=block
#define PTE_ADDR_MASK 0x0000FFFFFFFFF000ULL
#define PTE_ATTRINDX_MSK 0x1CULL // bits [4:2]

// Clean one cache line (containing va) to Point of Coherency, then barrier.
static inline void clean_dcache_line(uint64_t va) {
    __asm__ volatile("dc cvac, %0\n\t"
                     "dsb ish\n\t"
                     "isb"
                     :
                     : "r"(va)
                     : "memory");
}

// Invalidate all EL3 stage-1 TLB entries and re-synchronize.
static inline void tlb_flush_all_el3(void) {
    __asm__ volatile("dsb ish\n\t"
                     "tlbi alle3\n\t"
                     "dsb ish\n\t"
                     "isb"
                     :
                     :
                     : "memory");
}

// Read-back probe of the EL3 memory type for a VA using AT S1E3R -> PAR_EL1.
static void probe_attr_el3(const char *label, uint64_t va) {
    uint64_t par = 0;
    __asm__ volatile("at s1e3r, %1\n\t"
                     "isb\n\t"
                     "mrs %0, par_el1"
                     : "=r"(par)
                     : "r"(va)
                     : "memory");
    printf("[%-14s] VA=0x%012llx  PAR=0x%016llx -> %s\n", label, (unsigned long long)va, (unsigned long long)par,
           par_mem_attr_str(par));
    fflush(stdout);
}

// Walk the EL3 stage-1 tables for XAIE_BASE_ADDR and rewrite the AttrIndx of the
// descriptor that maps it. Returns:
//   0  success (block descriptor found and patched)
//  -1  L0 entry is not a valid table (aperture not mapped as expected)
//  -2  descriptor is a finer-grained table (L2/L3) — not handled here
//  -3  descriptor is invalid / faulting
static int patch_aie_attr(uint32_t new_attrindx) {
    const uint64_t va = (uint64_t)XAIE_BASE_ADDR;

    uint64_t ttbr = 0;
    __asm__ volatile("mrs %0, ttbr0_el3" : "=r"(ttbr));
    uint64_t l0_base = ttbr & PTE_ADDR_MASK;

    uint32_t i0 = (uint32_t)((va >> 39) & 0x1FFULL);
    uint32_t i1 = (uint32_t)((va >> 30) & 0x1FFULL);

    volatile uint64_t *l0 = (volatile uint64_t *)(uintptr_t)(l0_base + (uint64_t)i0 * 8u);
    uint64_t l0d = *l0;
    printf("patch: TTBR0_EL3=0x%016llx L0[%u]@0x%012llx=0x%016llx\n", (unsigned long long)ttbr, i0,
           (unsigned long long)(uintptr_t)l0, (unsigned long long)l0d);
    if ((l0d & PTE_TYPE_MASK) != 0x3ULL)
        return -1; // not a table pointer

    uint64_t l1_base = l0d & PTE_ADDR_MASK;
    volatile uint64_t *l1 = (volatile uint64_t *)(uintptr_t)(l1_base + (uint64_t)i1 * 8u);
    uint64_t l1d = *l1;
    printf("patch: L1[%u]@0x%012llx=0x%016llx  (type=%llu)\n", i1, (unsigned long long)(uintptr_t)l1,
           (unsigned long long)l1d, (unsigned long long)(l1d & PTE_TYPE_MASK));

    uint64_t type = l1d & PTE_TYPE_MASK;
    if (type == 0x0ULL)
        return -3; // invalid descriptor
    if (type == 0x3ULL)
        return -2; // table pointer -> mapped at L2/L3 granularity, not handled

    // type == 1: 1 GB block descriptor at L1. Rewrite AttrIndx bits [4:2].
    uint64_t old_ai = (l1d & PTE_ATTRINDX_MSK) >> 2;
    uint64_t nd = (l1d & ~PTE_ATTRINDX_MSK) | (((uint64_t)new_attrindx & 0x7ULL) << 2);
    *l1 = nd;
    printf("patch: rewrote AttrIndx %llu -> %u; new L1 desc=0x%016llx\n", (unsigned long long)old_ai, new_attrindx,
           (unsigned long long)nd);

    clean_dcache_line((uint64_t)(uintptr_t)l1);
    tlb_flush_all_el3();
    return 0;
}

// Demonstration wrapper: probe the AIE memory type, patch the descriptor, then
// probe again to confirm the new attribute is live. EL3-only.
static void aie_attr_patch_demo(void) {
    uint64_t cur_el = 0;
    __asm__ volatile("mrs %0, CurrentEL" : "=r"(cur_el));
    cur_el = (cur_el >> 2) & 0x3ULL;

    printf("\n==== AIE aperture attribute patch (AttrIndx -> %u) ====\n", (unsigned)AIE_NEW_ATTRINDX);
    if (cur_el < 3ULL) {
        printf("Not at EL3 (CurrentEL=EL%llu) — skipping page-table patch.\n", (unsigned long long)cur_el);
        return;
    }

    probe_attr_el3("AIE before", (uint64_t)XAIE_BASE_ADDR);
    int rc = patch_aie_attr((uint32_t)AIE_NEW_ATTRINDX);
    if (rc != 0) {
        printf("patch_aie_attr() failed rc=%d (aperture not a single L1 block?)\n", rc);
        return;
    }
    probe_attr_el3("AIE after", (uint64_t)XAIE_BASE_ADDR);
}
#endif /* __AIESIM__ */

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    XAie_InstDeclare(DevInst, &ConfigPtr);

    AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
    if (RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }

#ifdef __AIESIM__
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_SIM);
#else
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);

#if AIE_GEN >= 2
    if (DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
        printf("XAie_UpdateNpiAddr()\n");
#if AIE_GEN == 5
        printf("XAie_UpdateNpiAddr(0xf6d50000)\n");
        RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
#else
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
#endif
        if (RC != XAIE_OK) {
            printf("Failed to update NPI address\n");
            return -1;
        }
    }
    printf("before XAie_PartitionInitialize\n");
    RC = XAie_PartitionInitialize(&DevInst, NULL);
#else
    XAie_PmRequestTiles(&DevInst, NULL, 0);
#endif
#endif /* __AIESIM__ */

#ifndef __AIESIM__
    probe_bd_addr_attr();
#if AIE_ATTR_PATCH
    // Opt-in: re-type the AIE aperture via EL3 page-table patch, then re-probe.
    // The register-write benchmark below then runs under the NEW attribute.
    aie_attr_patch_demo();
#endif
#endif

    run_control_perf(&DevInst);
    return 0;
}
