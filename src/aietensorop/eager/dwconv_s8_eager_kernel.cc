/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: depthwise int8 convolution to exact int32 sums (dwconv_s8_eager_ref.h: the layout) -- the
 * depthwise layers dwconv_q cannot finish on the core (an s32 / f32 dst, per-channel float scales, a residual
 * add) keep the AIE for the sums and take the epilogue on the host. Y[m][c] = sum_t A[m][c, t] . W[c][t]; the host
 * gathers each pixel's taps (im2col per channel), so one template serves every kernel size, stride, padding and
 * dilation; TP (taps, padded) is a template argument.
 *
 * Routing: the GEMM skeleton, as dwconv_q. A's chunks are the K walk; mesh column pn owns channels pn C/4 .. and
 * computes only on its own chunks (pn C/16 ..), the others pass by.
 ******************************************************************************/
#include "eager_common.h"
#include "dwconv_s8_eager_ref.h" /* the layout, eager_dwconv_s8_ref (the instances' AIEK_REFERENCE) */

// M a multiple of HW_ROWS * TM, C of 16, TP of 4
template <int M, int C, int TP, int TM = 16>
__global__ void dwconv_s8(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, C / 4 * 4 * TP>> a,
                          aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<C, C / 4 * 4 * TP>> w,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<M, C>> c) {
    const int TN = C / 4, TK = 4 * TP;
    const int ntile = M / (HW_ROWS * TM);
    const int nkc = C / 4, own = TN / 4; /* chunks, and the chunks of one mesh column */
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        int32_t sum[TM * TN];
        for (int j = 0; j < nkc; j++) {
            const int8_t *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
            if (j / own == pn) {
                const int g = j - pn * own;
                for (int i = 0; i < 4; i++) {
                    const int n = g * 4 + i;
                    const int8_t *wr = v + n * TK;
                    for (int m = 0; m < TM; m++) {
                        const int8_t *xr = x + m * TK + i * TP;
                        int32_t s = 0;
                        for (int k = 0; k < TP; k++)
                            s += (int32_t)xr[k] * (int32_t)wr[k];
                        sum[m * TN + n] = s;
                    }
                }
            }
            aie::release(a);
            aie::release(w);
        }
        int32_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int i = 0; i < TM * TN; i++)
            y[i] = sum[i];
        aie::commit(c);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_dwconv_s8_launch_fn)(int8_t *A, int8_t *W, int32_t *Y);

/* args: A [M][K], W [C][K] (K = C/4 * 4 TP), Y [M][C] int32 */
static int eager_dwconv_s8_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t C, int64_t TP,
                                 eager_dwconv_s8_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int64_t K = C / 4 * 4 * TP;
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * K));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(C * K));
    int32_t *y = (int32_t *)eager_slot(2, (size_t)(M * C) * 4);
    if (!a || !w || !y)
        return AIEK_ERR_ALLOC;
    launch(a, w, y);
    eager_fetch_out(2, &args[2], (size_t)(M * C) * 4);
    return AIEK_OK;
}
