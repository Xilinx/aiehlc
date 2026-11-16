/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * gemm -- one block of the aiednn VEK280 int32 GEMM:
 * C[M][N] (int32) = A[M][K] (int8) * B[N][K]^T (int8), exact accumulators.
 *
 * The aiehlc program is all-int8 (one element type per window): its C is
 * 4 * GEMM_N bytes wide -- the bytes of the int32 result -- so B has that many
 * rows, each core column's first quarter real. M/N/K below are what aiehlc sees.
 ******************************************************************************/
#ifndef AIETENSOROP_GEMM_H
#define AIETENSOROP_GEMM_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef HW_ROWS /* the mesh: HW_ROWS x HW_COLS cores, a 32x32 int32 C tile each */
#define HW_ROWS 4
#endif
#ifndef HW_COLS
#define HW_COLS 4
#endif
#define GEMM_M (32 * HW_ROWS) /* logical block: int32 C[GEMM_M][GEMM_N] = A[GEMM_M][GEMM_K] * B[GEMM_N][GEMM_K]^T */
#define GEMM_N (32 * HW_COLS)
/* depthwise / pool overlay items: one per core, item r*HW_COLS+c = A rows 32r+8c .. +7 */
#define DW_ITEMS (HW_ROWS * HW_COLS)
#define DW_ITEM_BYTES 4096 /* = 8 A rows of K bytes */
#define DW_WBYTES 64       /* weights at the item's tail: KH*KW <= 49 (7x7); requant params in the 32 before */
#define DW_OUT 1024        /* int32 outputs per item (one core's C tile) */
#define GEMM_K 512         /* = N below: aiehlc offsets B by N per column (skill gemmnonsquareb) */
#if HW_COLS != 4
#error                                                                                                                 \
    "gemm_uk: HW_COLS must be 4 -- B rows N = 128 * HW_COLS must equal K = 512 (gemmnonsquareb), and a core row's 32 A rows hold HW_COLS 8-row items"
#endif
/* the partition: core rows 3 .. 2 + HW_ROWS (VEK280), enough NoC shims for the output and input streams */
#if defined(AIE_GEN) && AIE_GEN == 5 /* VEK385 / AIE2PS: a DMA shim in every column, cores from row 3 */
#if HW_ROWS == 4
#define GEMM_PARTITION {0, 3, 0, 6}
#else
#error "gemm_uk: VEK385 runs the 4x4 mesh only"
#endif
#elif HW_ROWS == 4
#define GEMM_PARTITION {0, 8, 0, 7}
#elif HW_ROWS == 6
#define GEMM_PARTITION {0, 16, 0, 8}
#elif HW_ROWS == 8 /* all 8 VEK280 core rows */
#define GEMM_PARTITION {0, 16, 0, 10}
#else
#error "gemm_uk: add the partition for this mesh (endRow = 2 + HW_ROWS, skill nonsquaremesh)"
#endif

#ifndef M
#define M GEMM_M
#endif
#ifndef K
#define K 512
#endif
#ifndef N
#define N 512 /* = 4 * GEMM_N: C bytes per row, B rows (padded) */
#endif

#endif
