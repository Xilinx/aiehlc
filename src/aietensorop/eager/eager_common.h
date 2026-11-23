/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: host code shared by every eager template (*_eager_kernel.cc). An instance's AIEK_ENTRY
 * stages each operand into a DMA buffer, launches, and copies the outputs back; the slots are shared by every
 * instance (launches run one at a time). __Runtime_Alloc buffers survive the partition re-initialization between
 * instances, aieArray::alloc ones do not.
 ******************************************************************************/
#ifndef EAGER_COMMON_H
#define EAGER_COMMON_H

#include "aie_user_kernel.h"
#include "xil_cache.h"
#include <stdint.h>
#include <string.h>

/* the mesh: HW_ROWS x HW_COLS cores. The kernels split rows by HW_ROWS, so the entry #defines it (only the kernel's
 * own text and the main / spliced .cc #defines reach the core compile and the trace); 4 x 4 by default. The
 * templates' per-column layouts (resident parameter rows, 16-byte reduction rows) take HW_COLS = 4. */
#ifndef HW_ROWS
#define HW_ROWS 4
#endif
#ifndef HW_COLS
#define HW_COLS 4
#endif
#define EG_HW_ROWS HW_ROWS
#define EG_HW_COLS HW_COLS
#if HW_COLS != 4
#error "eager templates: HW_COLS must be 4 (per-column parameter rows, 16-byte reduction rows)"
#endif

/* the partition: VEK280 core rows 3 .. 2 + HW_ROWS and enough NoC shims (fixed mode's gemm.h partitions) */
#if defined(AIE_GEN) && AIE_GEN == 5
#if HW_ROWS != 4
#error "eager templates: VEK385 runs the 4 x 4 mesh only"
#endif
#define EG_PARTITION {3, 6, 0, 6}
#elif HW_ROWS == 4
#define EG_PARTITION {0, 8, 0, 7}
#elif HW_ROWS == 6
#define EG_PARTITION {0, 16, 0, 8}
#elif HW_ROWS == 8
#define EG_PARTITION {0, 16, 0, 10}
#else
#error "eager templates: add the partition for this mesh (endRow = 2 + HW_ROWS)"
#endif

/* aie_runtime.h (C++ linkage, like the g++-compiled runtime): the instance launches call it before
 * device.partition(), whose routing is always the first kernel's -- without it every launch of another
 * instance re-initializes the partition there and again in __aie_launch */
void __Runtime_partition_reuse_next(void);

#define EG_MAX_PORTS 8

static void *g_eg_buf[EG_MAX_PORTS];
static size_t g_eg_cap[EG_MAX_PORTS];

/* port slot i's DMA buffer, at least `bytes` (NULL: allocation failed) */
static void *eager_slot(int i, size_t bytes) {
    if (bytes > g_eg_cap[i]) {
        void *q = __Runtime_Alloc(bytes);
        if (!q)
            return NULL;
        g_eg_buf[i] = q, g_eg_cap[i] = bytes;
    }
    return g_eg_buf[i];
}

/* an input operand into slot i, flushed for the DMA */
static void *eager_stage_in(int i, const aiek_tensor *t, size_t bytes) {
    void *d = eager_slot(i, bytes);
    if (d) {
        memcpy(d, t->data, bytes);
        Xil_DCacheFlushRange((UINTPTR)d, bytes);
    }
    return d;
}

/* slot i back into an output operand (the launch invalidates outputs only before the DMA) */
static void eager_fetch_out(int i, const aiek_tensor *t, size_t bytes) {
    Xil_DCacheInvalidateRange((UINTPTR)g_eg_buf[i], bytes);
    memcpy(t->data, g_eg_buf[i], bytes);
}

#endif /* EAGER_COMMON_H */
