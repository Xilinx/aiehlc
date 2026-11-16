/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * matmul -- shape of the aiednn VEK280 GEMM: C[M][N] = sat8(A[M][K] * B[N][K]^T).
 * B is stored [N][K] so B rows stream with the same DMA pattern as A rows.
 ******************************************************************************/
#ifndef AIETENSOROP_MATMUL_H
#define AIETENSOROP_MATMUL_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef M
#define M 256
#endif
#ifndef K
#define K 256
#endif
#ifndef N
#define N 256
#endif

#ifndef HW_ROWS
#define HW_ROWS 4
#endif
#ifndef HW_COLS
#define HW_COLS 4
#endif

/* The partition. AIE_GEN 5 (AIE2PS): a DMA shim in every column. AIE-ML: cores from row 3,
 * only cols 2,3,6,7 of 0..7 reach DDR. */
#if defined(AIE_GEN) && AIE_GEN == 5
#define MATMUL_PARTITION {0, 3, 0, 6}
#else
#define MATMUL_PARTITION {0, 8, 0, 7}
#endif

#endif
