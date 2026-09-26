/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "xaiengine.h"
#include <math.h>
#include <stdio.h>

#ifndef __AIESIM__
#include "xil_cache.h"
#include "xil_printf.h"
#include "sleep.h" /* usleep() in the standalone BSP */
#if AIE_GEN <= 2
#include "xtime_l.h"
#else
#include "xiltimer.h"
#endif
#else
#include <unistd.h> /* usleep() on the host (aiesim) */
#endif /* __AIESIM__ */

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

// real offsets are injected by aiehlc
#ifndef CORE_IP_MEM
#define CORE_IP_MEM 0x1000
#endif
#ifndef CORE_OP_MEM
#define CORE_OP_MEM 0x6000
#endif

// Master switch for the core event-trace flow. The trace unit is routed down
// from the core through the intervening core tiles into the same-column top
// MemTile's S2MM DMA. The __Runtime_core_trace_begin_ch/_end_into session
// helpers own the fixed-reserved MemTile drain convention (buffer offset, stream
// channel, S2MM channel, BD), so no buffer/channel macros are needed here; the
// _ch form only pins the trace stream channel away from this tile's data DMA. Set
// to 0 to fully compile out trace setup/read and isolate whether the hang is
// trace-induced; set to 1 to re-enable.
#ifndef TRACE_ENABLE
#ifndef __AIESIM__
// #define TRACE_ENABLE 1
#else
#define TRACE_ENABLE 0
#endif
#endif

// Host<->AIE time-sync instrumentation. Emits a machine-readable [TIMESYNC]
// block (host anchors, per-tile AIE-timer anchors, host phase events, raw trace
// hex) that src/tool/debug/host_aie_timeline.py correlates into one microsecond
// axis. Host-side only (XTime), so it is compiled out under the simulator.
#ifndef __AIESIM__
// #define TIMESYNC 1
#else
#define TIMESYNC 0
#endif

// Core trace API from src/mlir/runtime/aie_runtime.c (linked into this host by
// aiehlc.sh's RUNTIME_SRCS). Pulled in via the runtime header, which is on the
// host compile include path (-I src/mlir/runtime). aie_runtime.h's guarded
// <aie_codegen_inc/xaie_routing.h> include is skipped because xaiengine.h above
// already defined XAIE_ROUTING_H.
#include "aie_runtime.h"
// Reduced for simulator
// #define N 16
#define N 4
#define MAT_SIZE (N * N)

// Tile Memory-Mapped (TM) register access. The anchor that gives the compiler
// the TM memory space must live at FILE scope, and aiehlc regenerates the
// kernel from the function body plus #define lines only -- so the include is
// wrapped in the prologue markers, which aiehlc copies through verbatim.
// See src/mlir/runtime/kernel_tm.h for why a raw pointer cast does not work.
//
// AIEHLC_KERNEL_PROLOGUE_BEGIN
#include "kernel_tm.h"
// AIEHLC_KERNEL_PROLOGUE_END

// #define DISABLE_CACHE
//__attribute__((annotate("streaming")))
__global__ void perf(input_window_int32 *win __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:512"))),
                     output_window_int32 *out
                     __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:512")))) {
#define N 4
#define MAT_SIZE (N * N)
#define DATA_SIZE (MAT_SIZE * 2)
#define VECTOR_LENGTH 16
    // TM register access via kernel_tm.h. Three paths, so one HW run compares
    // them: (A) this header, (B) AMD's adf:: API, (C) a raw pointer cast as a
    // deliberate control -- C is expected to stay inside the core.
    TM_W(TM_MEM_SPARE_REG, 0x7234);
    chess_memory_fence();
    uint32 rb = TM_R(TM_MEM_SPARE_REG);
    uint32 corestatus = TM_R(TM_CORE_STATUS);
    /*
    adf::write(adf::reg_val{0x0001D100, 0x7235});
    chess_memory_fence();
    uint32 rb_adf = adf::read(0x0001D100);

    *((volatile int *)(0x0001D100 + 8)) = 0x7236;
    chess_memory_fence();
    uint32 rb_raw = *((volatile int *)(0x0001D100 + 8));

    // Park results in DM: data memory is passive storage, so the values survive
    // until the host reads them (a TM register may be volatile hardware state).
    *((volatile int *)(0x70000 + 0x0000FF04)) = (int)rb;         // expect 0x7234
    *((volatile int *)(0x70000 + 0x0000FF08)) = (int)corestatus;
    *((volatile int *)(0x70000 + 0x0000FF0C)) = (int)rb_adf;     // expect 0x7235
    *((volatile int *)(0x70000 + 0x0000FF10)) = (int)rb_raw;     // expect 0x7236 on-core
     */
    //aie::vector<int32_t, VECTOR_LENGTH> temp_a = window_readincr_v<VECTOR_LENGTH>(win);
	//aie::store_unaligned_v<VECTOR_LENGTH>(A_mat + (w*VECTOR_LENGTH), temp_a);
	uint32_t * ptr_out = (uint32_t *)(0x70000 + 0x6000);
	uint32_t * ptr_in = (uint32_t *)(0x70000 + 0x1000);

    uint32_t * vec1 = ((uint32_t*)ptr_in), * vec2 = ((uint32_t*)ptr_in + MAT_SIZE);

    for (int i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            uint32_t ret = 0;
            for (int k = 0; k < N; k++) {
                ret += vec1[i * N + k] * vec2[j * N + k];
            }
            ptr_out[i * N + j] = ret;
        }
    }
}
void blockread(XAie_DevInst *DevInst, uint64_t addr)
{
#define DSIZE  512
  uint32_t odata[DSIZE];
	XAie_DataMemBlockRead(DevInst, XAie_TileLoc(4,4),  addr,
			 (void*)odata, DSIZE * sizeof(uint32_t));

	// step 5 validate data
	printf("addr = 0x%lx\n", addr);
	for(uint32_t i = 0U; i < DSIZE; i++) {
		printf("odata[%d] = %x\n", i, odata[i]);
	}

}
void breakprint(char *info) {
	return;
	char input;
	printf("input key to run %s\n", info);
	scanf("%c", input);
	printf("log--- %s done\n", info);
}
int test_routing(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;
	XAie_RoutingInstance* routingInstance;
#ifndef __AIESIM__
    XTime tStart, tEnd;
#endif
    printf("Starting test_routing 08/14 -2\n");
    breakprint("core reset--");
#ifdef __AIESIM__
    int shimcol = 3;
#elif AIE_GEN == 5
    int shimcol = 10;
#else
    int shimcol = 6;
#endif
    XAie_CoreReset(DevInst, XAie_TileLoc(4, 4));
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);
    XAie_CoreUnreset(DevInst, XAie_TileLoc(4, 4));

    XAie_CoreProcessorBusEnable(DevInst, XAie_TileLoc(4, 4));

    routingInstance = XAie_InitRoutingHandler(DevInst);
    XAie_Route(routingInstance, NULL, XAie_TileLoc(shimcol, 0) /* Source*/, XAie_TileLoc(4, 4) /* destination*/);
    XAie_Route(routingInstance, NULL, XAie_TileLoc(4, 4) /* Source*/, XAie_TileLoc(shimcol, 0) /* destination*/);

    u32 mlen = MAT_SIZE * 2;
    const u32 recv_len = MAT_SIZE;
	
	//Prepare DDR data
    XAie_MemInst *in = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
    XAie_MemInst *out = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);

    breakprint(" XAie_MemSyncForDev---\n");

    int32_t *vmem = (int32_t *)XAie_MemGetVAddr(in);
    int32_t *vmem_out = (int32_t *)XAie_MemGetVAddr(out);

    for (int j = 0; j < mlen; j++) {
        vmem[j] = 1 + j;
        vmem_out[j] = j;
    }

    const int count = 4; // iterations for perf measurement
    printf("before runtimerace\n");

    // ---- Host<->AIE time-sync + AIE core-trace: one container -------------
    // trc_prof holds BOTH the host phase timeline (TS_EVT), the per-tile
    // host<->AIE anchors (TS_ANCHOR) and the decoded core-trace intervals. A
    // single __Runtime_aie_trace_profile_dump emits the whole [TIMESYNC] block.
    // Traced-tile list: only (4,4) computes here; to place more tiles on the
    // timeline, arm each via __Runtime_core_trace_setup with distinct trace
    // buffers/channels, read each back, add it to ts_tiles[], and decode each
    // into this same profile.
#if (TIMESYNC || TRACE_ENABLE)
    static AieTraceProfile trc_prof; // fixed-capacity sink (no malloc)
    __Runtime_aie_trace_profile_init(&trc_prof);
#endif
#if TIMESYNC
    XAie_LocType ts_tiles[] = {XAie_TileLoc(4, 4)};
    const int ts_ntiles = (int)(sizeof(ts_tiles) / sizeof(ts_tiles[0]));
    __Runtime_aie_trace_profile_set_clock(&trc_prof, (uint64_t)COUNTS_PER_SECOND);
// Record a host phase event into the profile (bounded internally).
#define TS_EVT(it, ph)                                                                                                 \
    do {                                                                                                               \
        XTime _h;                                                                                                      \
        XTime_GetTime(&_h);                                                                                            \
        __Runtime_aie_trace_profile_event(&trc_prof, (it), (ph), (uint64_t)_h);                                        \
    } while (0)
// Capture an anchor (which=0 before loop, 1 after): host-before -> read every
// tile timer -> host-after; the anchor's host value is the midpoint
// (XAie_ReadTimer has ~us AXI-MM latency).
#define TS_ANCHOR(which)                                                                                               \
    do {                                                                                                               \
        XTime _hb, _ha;                                                                                                \
        uint64_t _av[8] = {0};                                                                                         \
        XTime_GetTime(&_hb);                                                                                           \
        for (int _t = 0; _t < ts_ntiles; _t++) {                                                                       \
            uint64_t _v = 0;                                                                                           \
            __Runtime_read_aie_timer(DevInst, ts_tiles[_t], &_v);                                                      \
            _av[_t] = _v;                                                                                              \
        }                                                                                                              \
        XTime_GetTime(&_ha);                                                                                           \
        uint64_t _hm = (uint64_t)_hb + ((uint64_t)_ha - (uint64_t)_hb) / 2;                                            \
        for (int _t = 0; _t < ts_ntiles; _t++)                                                                         \
            __Runtime_aie_trace_profile_anchor(&trc_prof, (which), ts_tiles[_t], _av[_t], _hm);                        \
    } while (0)
#else
#define TS_EVT(it, ph)                                                                                                 \
    do {                                                                                                               \
    } while (0)
#endif

    // Arm the core trace unit on tile (4,4) BEFORE the core runs: capture the
    // ACTIVE/stall timeline and route it DOWN through the intervening core tiles
    // into the same-column top MemTile's S2MM DMA. The session helper owns the
    // fixed-reserved MemTile drain convention (buffer offset, stream channel,
    // S2MM channel, BD) that the hand code used to spell out. Must precede
    // XAie_Run (which makes the core active and fires ACTIVE_CORE, opening the
    // trace window).
#if TRACE_ENABLE
    // Pin the trace stream channel to 1: this tile's output data DMA egresses on
    // SOUTH channel 0, and the core-trace route is programmed directly (outside
    // the routing engine's resource manager), so letting it default to slot 0
    // would put both flows on SOUTH ch 0 and deadlock XAie_RouteDmaWait on the
    // output DMA. Channel 1 is free here (matches the old hand-coded TRC_STRM_CH).
    __Runtime_core_trace_begin_ch(DevInst, /*col=*/4, /*row=*/4, /*strm_ch=*/1);
#endif

    printf("after runtimerace\n");

#ifndef __AIESIM__
    XTime_GetTime(&tStart);
#endif

    // Anchor0: bracket each tile's free-running timer with the host clock right
    // before the run loop. Pairs with Anchor1 (after the loop) to fit AIE
    // cycles -> host microseconds per tile.
#if TIMESYNC
    TS_ANCHOR(0);
#endif

    for (int i = 0; i < count; i++) {
        TS_EVT(i, "iter_start");

#ifdef __AIESIM__
        if (i > 0) {
            XAie_CoreReset(DevInst, XAie_TileLoc(4, 4));
            XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);
            XAie_CoreUnreset(DevInst, XAie_TileLoc(4, 4));
        }
#endif

        XAie_MemSyncForDev(in);

        printf("after XAie_MemSyncForDev\n");

        breakprint("Starting to Move data\n");
        // step 3: move data to destination tile
        // XTime_GetTime(&tStart);
        // printf("vmem = 0x%p\n",vmem);

        TS_EVT(i, "dma_in_start");
        XAie_MoveDataExternal2Aie(routingInstance, /*src=*/XAie_TileLoc(shimcol, 0), in, mlen * sizeof(u32),
                                  CORE_IP_MEM, /*dest=*/XAie_TileLoc(4, 4));
        XAie_RouteDmaWait(routingInstance, XAie_TileLoc(shimcol, 0), XAie_TileLoc(4, 4), true);
        TS_EVT(i, "dma_in_done");
        TS_EVT(i, "run");

        XAie_Run(routingInstance, 1);
        printf("XAie_Run\n");
#ifdef __AIESIM__
        while (XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4, 4), 1) != XAIE_OK) {
        }
#else
        XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4, 4), 0);
#endif
        TS_EVT(i, "wait_done");

        breakprint("fflush\n");
        printf("after XAie_CoreWaitForDone\n");

#ifndef __AIESIM__
        Xil_DCacheFlushRange((INTPTR)vmem_out, mlen * sizeof(int32_t));
#endif
        TS_EVT(i, "dma_out_start");
        XAie_MoveDataAie2External(routingInstance, XAie_TileLoc(4, 4), CORE_OP_MEM, mlen * sizeof(u32), out,
                                  XAie_TileLoc(shimcol, 0));
        XAie_RouteDmaWait(routingInstance, XAie_TileLoc(4, 4), XAie_TileLoc(shimcol, 0), false);
        TS_EVT(i, "dma_out_done");
        XAie_MemSyncForCPU(out);
#ifndef __AIESIM__
        Xil_DCacheInvalidateRange((INTPTR)vmem_out, mlen * sizeof(int32_t));
#endif

        ///*
        // XTime_GetTime(&tEnd);
        // printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));

        //printf("\nFinished moving data back to DDR\n");
        // step 5 validate data
        int32_t vmem_out_cpu[recv_len];

        // vmem contains the input (128 samples, 64 of matrix A and 64 of matrix B, in row major and column major forms
        // respectively) and vmem_out contains the output samples (64 of result)
        // compute CPU Result for softmax
        int32_t A_mat[N][N];  // Matrix A
        int32_t B_mat[N][N];  // Matrix B
        int32_t result[N][N] = {0};  // Result matrix

        printf("after Xil_DCacheInvalidateRange\n");
        // Extract matrix A (row major)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_mat[i][j] = vmem[i * N + j];
            }
        }

        // Extract matrix B (column major)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B_mat[i][j] = vmem[MAT_SIZE + i * N + j];
            }
        }

        // Perform matrix multiplication
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    result[i][j] += A_mat[i][k] * B_mat[j][k];
                }
            }
        }

        // Store the result in vmem_out_cpu
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                vmem_out_cpu[i * N + j] = result[i][j];
            }
        }

        int mismatches = 0;
        for (int i = 0; i < MAT_SIZE; i++) {
            if (vmem_out_cpu[i] != vmem_out[i]) {
                if (mismatches < 4)
                    printf("[MISMATCH] [%d]: CPU=%d AIE=%d\n", i, vmem_out_cpu[i], vmem_out[i]);
                mismatches++;
            }
        }

        if (mismatches == 0) {
            printf("\n[PASS] Iteration %d: all %d outputs match CPU reference.\n\n", i, MAT_SIZE);
        } else {
            printf("\n[FAIL] Iteration %d: %d / %d mismatches.\n\n", i, mismatches, MAT_SIZE);
        }
        fflush(stdout);
        //*/
    }

    // Anchor1: bracket the tile timers again immediately after the run loop.
    // Together with Anchor0 this gives one continuous per-tile AIE-cycle ->
    // host-us fit spanning all iterations (tile timers free-run from device
    // init and are NOT reset between iterations on hardware).
#if TIMESYNC
    TS_ANCHOR(1);
#endif

    // Core finished: read back and decode every tile armed by
    // __Runtime_core_trace_begin into the SHARED profile (trc_prof), so the core
    // trace lands in the same container as the host clock, anchors and phase
    // events and a single dump below emits one coherent [TIMESYNC] block. The
    // session helper reads via the MemTile loc, attaches the (col,row) tag and
    // decodes; it does not init or dump (the caller owns both).
#if TRACE_ENABLE
    usleep(1000 * 1000 * 5); // give the S2MM DMA a moment to finish writing the last trace words
    __Runtime_core_trace_end_into(DevInst, &trc_prof);
#endif

    // Emit the whole run as ONE machine-readable [TIMESYNC] block straight from
    // the profile: host clock frequency, the two anchor records (host + per-tile
    // AIE timer), every host phase event, the effective AIE Hz, the trace-word
    // count, and one "[TIMESYNC] trace tile=c,r <start> -- <end> EVENT (N cyc)"
    // line per decoded core-trace interval. src/tool/debug/host_aie_timeline.py
    // parses these lines, fits each tile's AIE-cycle->host-us map from the
    // anchors, and emits a merged host+AIE timeline as CSV/JSON.
#if (TIMESYNC || TRACE_ENABLE)
    __Runtime_aie_trace_profile_dump(&trc_prof);
    fflush(stdout);
#endif

#ifndef __AIESIM__
    XTime_GetTime(&tEnd);
    printf("Transferred %u bytes (%d iter) in %.2f us.\n", (unsigned)(count * mlen * sizeof(u32)), count,
           1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND / 1000000));
#endif
    return 0;
}

int main(int argc, char* argv[]) {
#ifndef __AIESIM__
#ifdef DISABLE_CACHE
	Xil_DCacheDisable();
	Xil_ICacheDisable();
    printf("Cache disabled: expect a big perf drop (>350 us for 8x8 and 16x16).\n");
#else
    printf("Cache enabled: expect ~15 us for 8x8, ~215 us for 16x16.\n");
#endif
#endif /* __AIESIM__ */

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    ///*

	XAie_InstDeclare(DevInst, &ConfigPtr);

	int partitonnum = 34;
    int startcol = 1;
    int colnum = (startcol + partitonnum <= XAIE_NUM_COLS) ? partitonnum : (XAIE_NUM_COLS- startcol);

	AieRC RC;
    /*
    RC = XAie_SetupPartitionConfig(&DevInst, XAIE_BASE_ADDR + (startcol<<XAIE_COL_SHIFT),
                                       startcol, colnum);
    if(RC != XAIE_OK) {
        printf("Driver XAie_SetupPartitionConfig failed.\n");
        return -1;
    }
    //*/

    RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
	if(RC != XAIE_OK) {
		printf("Driver initialization failed.\n");
		return -1;
	}

#ifdef __AIESIM__
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_SIM);
#else
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);

#if AIE_GEN >= 2
	if(DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
		printf("XAie_UpdateNpiAddr()\n");
#if AIE_GEN == 5
        printf("XAie_UpdateNpiAddr(0xf6d50000)\n");
        RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
#else
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
#endif
        if(RC != XAIE_OK) {
			printf("Failed to update NPI address\n");
			return -1;
		}
	}
    printf("before XAie_PartitionInitialize-2--\n");
    RC = XAie_PartitionInitialize(&DevInst, NULL);
#else
    XAie_PmRequestTiles(&DevInst, NULL, 0);
#endif
#endif /* __AIESIM__ */

    test_routing(&DevInst);

    u32 v = 0;
    XAie_Read32(&DevInst, XAie_GetTileAddr(&DevInst, 4, 4) + 0x16000, &v);
    printf("Read value from 0x16000: 0x%x\n", v);
    return 1;
    RC = XAie_PartitionTeardown(&DevInst);
    if(RC != XAIE_OK) {
        printf("Failed to Teardown partition\n");
        return -1;
	}

	return 1;
}
