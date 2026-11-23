/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: grouped weight-only quantization -- an int8 src times 4-bit weights travelling as nibbles, a
 * scale and a zero point per (group, column) riding in the group's W chunk (gemm_wq_eager_ref.h: the layout). The
 * group size GK is the K chunk, a template argument: each chunk's int32 sum is exact and is scaled into the f32
 * accumulator on the core. The host corrects a src zero point and applies the epilogue.
 *
 * Routing: the GEMM skeleton (gemm_s8's): each core owns a TM x N / 4 tile, K streamed in chunks of GK.
 ******************************************************************************/
#include "eager_common.h"
#include "gemm_wq_eager_ref.h" /* the layout, eager_gemm_wq_ref (the instances' AIEK_REFERENCE) */

// M a multiple of HW_ROWS * TM, N of 16, K of GK, GK of 16 and >= 16 (nibbles + scale + zero point fit a chunk), TM >=
// N / 4
template <int M, int N, int K, int GK, int TM = 16>
__global__ void gemm_wq(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, K>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, K>> w,
                        aie::port<output_window_int32 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4;
    const int ntile = M / (HW_ROWS * TM);
    const int nkc = K / GK;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        float acc[TM * TN];
        for (int i = 0; i < TM * TN; i++)
            acc[i] = 0.0f;
        for (int j = 0; j < nkc; j++) {
            const int8_t *x = aie::load_buffer(a, {m0, j * GK}, {TM, GK});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * GK}, {TN, GK});
            for (int n = 0; n < TN; n++) {
                const int8_t *ch = v + n * GK;
                const float s = *(const float *)(ch + GK / 2);
                const int32_t zw = *(const int32_t *)(ch + GK / 2 + 4);
                for (int m = 0; m < TM; m++) {
                    const int8_t *xr = x + m * GK;
                    int32_t p = 0;
                    for (int i = 0; i < GK / 2; i++) {
                        const int32_t lo = (int32_t)(int8_t)(ch[i] << 4) >> 4, hi = (int32_t)ch[i] >> 4;
                        p += (int32_t)xr[2 * i] * (lo - zw) + (int32_t)xr[2 * i + 1] * (hi - zw);
                    }
                    acc[m * TN + n] += s * (float)p;
                }
            }
            aie::release(a);
            aie::release(w);
        }
        int32_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int i = 0; i < TM * TN; i++)
            ((float *)y)[i] = acc[i];
        aie::commit(c);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_gemm_wq_launch_fn)(int8_t *A, int8_t *W, int32_t *C);

/* args: A [M][K] s8, W [N][K] (groups' nibbles + scale + zero point), C [M][N] f32 */
static int eager_gemm_wq_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K,
                               eager_gemm_wq_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * K));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(N * K));
    int32_t *y = (int32_t *)eager_slot(2, (size_t)(M * N) * 4);
    if (!a || !w || !y)
        return AIEK_ERR_ALLOC;
    launch(a, w, y);
    eager_fetch_out(2, &args[2], (size_t)(M * N) * 4);
    return AIEK_OK;
}
