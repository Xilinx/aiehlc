/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host side of simplegemm_dataflow.cc (kernel-only GEMM with dataflow options).
 *
 *   C[GD_M, GD_N] = A[GD_M, GD_K] x B[GD_N, GD_K]^T   (int8 in, int32 acc, int8 out)
 *
 * GD_M = GD_P * 256: GD_P 256x256x256 panels that share B, multiplied in ONE
 * launch (the array is configured once for all of them).
 *
 * B is passed PACKED: mesh column pn reads the contiguous 64 x 256 block at
 * byte pn * 64 * 256 (rows pn*64.. of B), and inside that block the bytes are
 * in aie::mmul<8, 8, 8> B-operand order -- 8 x 8 (k x n) tiles, tile (ni, ki)
 * at (ni * k / 8 + ki) * 64, row-major k x n inside. An int8 transpose
 * is not something the DMA can do (it moves 32-bit words), so the host does
 * it once, like conv2dstem's filter pre-pack. gemm_df_pack_b() builds it.
 *
 * Checks: gemm_df_verify() recomputes every output on the CPU and compares
 * bit-exactly (the kernel keeps the low byte of the int32 sum, as (int8_t) does).
 ******************************************************************************/
#ifndef SIMPLEGEMM_DATAFLOW_H
#define SIMPLEGEMM_DATAFLOW_H

#include "unistd.h"
#include "xil_cache.h"
#include "xiltimer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GD_HW_ROWS 4
#define GD_HW_COLS 4

static inline uint32_t gd_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// A in [-4, 4], B in {-1, 0, 1}: a dot product stays within +-1024.
static void gemm_df_init(int8_t *A, int8_t *B, int m, int n, int k) {
    for (long i = 0; i < (long)m * k; i++)
        A[i] = (int8_t)((int)(gd_mix32((uint32_t)i + 1234u) % 9u) - 4);
    for (long i = 0; i < (long)n * k; i++)
        B[i] = (int8_t)((int)(gd_mix32((uint32_t)i + 5678u) % 3u) - 1);
}

// B [n, k] row-major -> the packed layout described above (64 rows per mesh column).
static void gemm_df_pack_b(const int8_t *B, int8_t *Bp, int n, int k) {
    const int cn = n / GD_HW_COLS, kt = k / 8;
    for (int pn = 0; pn < GD_HW_COLS; pn++)
        for (int ni = 0; ni < cn / 8; ni++)
            for (int ki = 0; ki < kt; ki++)
                for (int kk = 0; kk < 8; kk++)
                    for (int nn = 0; nn < 8; nn++)
                        Bp[(long)pn * cn * k + (long)(ni * kt + ki) * 64 + kk * 8 + nn] =
                            B[(long)(pn * cn + ni * 8 + nn) * k + ki * 8 + kk];
}

// Bit-exact check against a CPU GEMM; prints PASS / FAIL, returns the mismatch count.
static long gemm_df_verify(const int8_t *A, const int8_t *B, const int8_t *C, int m, int n, int k) {
    long mismatches = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int q = 0; q < k; q++)
                acc += A[(long)i * k + q] * B[(long)j * k + q];
            const int8_t want = (int8_t)acc, got = C[(long)i * n + j];
            if (got != want && ++mismatches <= 16)
                printf("MISMATCH C[%d][%d] (core %d,%d): got %d, expected %d\n", i, j, i / (m / GD_HW_ROWS),
                       j / (n / GD_HW_COLS), got, want);
        }
    if (mismatches == 0)
        printf("GOLDEN PASS: all %ld output elements match.\n", (long)m * n);
    else
        printf("GOLDEN FAIL: %ld mismatches out of %ld.\n", mismatches, (long)m * n);
    return mismatches;
}

static double gd_ms(XTime t0, XTime t1) { return t1 > t0 ? 1000.0 * (double)(t1 - t0) / COUNTS_PER_SECOND : 0.0; }

// The runtime's PMU phase counters (always linked; they only count in a
// `script/aiehlc.sh --profiling` build). C++ linkage, as in the archive.
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);

// Per-call split of the launches between two snapshots (ph0/wio0) -- kernel
// load, BD configuration, core enable, start_io, waiting for the DMA.
static void gemm_df_report_phases(const unsigned long long *ph0, unsigned long long wio0, int iters) {
    unsigned long long ph1[4] = {0, 0, 0, 0}, wio1 = 0, d[5], total = 0;
    unsigned int n[4] = {0, 0, 0, 0}, nw = 0;
    __Runtime_phase_cycles(ph1, n);
    __Runtime_wait_io_cycles(&wio1, &nw);
    for (int i = 0; i < 4; i++)
        d[i] = ph1[i] >= ph0[i] ? ph1[i] - ph0[i] : 0;
    d[4] = wio1 >= wio0 ? wio1 - wio0 : 0;
    for (int i = 0; i < 5; i++)
        total += d[i];
    if (total == 0) {
        printf("[gemm_df] PERF launch phases: counters read 0 -- build with `source script/aiehlc.sh --profiling` "
               "for the kload/bdcfg/coreen/startio/wait_io split\n");
        return;
    }
    const char *name[5] = {"kload  ", "bdcfg  ", "coreen ", "startio", "wait_io"};
    printf("[gemm_df] PERF launch phases (PMU cycles per call, %% of accounted):\n");
    for (int i = 0; i < 5; i++)
        printf("          %s %12.0f  (%5.1f%%)\n", name[i], (double)d[i] / iters, 100.0 * (double)d[i] / (double)total);
}

#endif // SIMPLEGEMM_DATAFLOW_H
