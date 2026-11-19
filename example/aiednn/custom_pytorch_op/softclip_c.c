/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Case C custom op: aiednn_custom_custom_op_demo_c_softclip_c_default
 * Clips each float element to [-1, 1] on the A72 host CPU.
 *
 * The function name is derived from the op's qualified name:
 *   custom_op_demo_c::softclip_c.default  ->  aiednn_custom_custom_op_demo_c_softclip_c_default
 * (dnn_torch.custom_c_symbol: replace "::" and "." with "_", prefix "aiednn_custom_")
 *
 * Signature (aiednn_custom_fn_t):
 *   int fn(int nsrc,
 *           const aiednn_memory_desc_t *srcs,   -- [nsrc] input descriptors
 *           const void *const *src_data,         -- [nsrc] input pointers
 *           const aiednn_memory_desc_t *dst,     -- output descriptor
 *           void *dst_data,                      -- output pointer
 *           void *ctx)                           -- user context (NULL here)
 * Return 0 on success, non-zero to signal an error.
 ******************************************************************************/
#include "aiednn.h"
#include <stdint.h>

int aiednn_custom_custom_op_demo_c_softclip_c_default(int nsrc, const aiednn_memory_desc_t *srcs,
                                                      const void *const *src_data, const aiednn_memory_desc_t *dst,
                                                      void *dst_data, void *ctx) {
    (void)srcs;
    (void)ctx;
    if (nsrc != 1)
        return -1;
    const float *x = (const float *)src_data[0];
    float *y = (float *)dst_data;
    int64_t n = 1;
    for (int d = 0; d < dst->ndims; d++)
        n *= dst->dims[d];
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
    }
    return 0;
}
