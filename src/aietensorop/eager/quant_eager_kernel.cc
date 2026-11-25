/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: f32 -> 8-bit quantize on the core (quant_eager_ref.h: the layout and the arithmetic -- exact
 * integer division of the two floats' mantissas, since AIE-ML's core has no double). U8 picks the destination type.
 *
 * Routing: the GEMM skeleton. K = 4 N bytes in 4 chunks of N bytes; mesh column pn owns output columns
 * pn N/4 .. -- exactly chunk pn's floats -- and lets the other chunks pass by; its W rows' chunk pn starts with
 * {scale bits, zp}.
 ******************************************************************************/
#include "eager_common.h"
#include "quant_eager_ref.h" /* the layout, eager_quant_ref (the instances' AIEK_REFERENCE) */

// M a multiple of HW_ROWS * TM, N of 16, N / 4 <= TM (W's block {N/4, N} must not exceed A's {TM, N})
template <int M, int N, int U8, int TM = 16>
__global__ void quant(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, 4 * N>> a,
                      aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, 4 * N>> w,
                      aie::port<output_window_int8 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4, TK = N;
    const int ntile = M / (HW_ROWS * TM);
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        int8_t code[TM * TN];
        for (int j = 0; j < 4; j++) {
            const int8_t *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
            if (j == pn) {
                const uint32_t su = ((const uint32_t *)v)[0];
                const int32_t zp = ((const int32_t *)v)[1], lo = U8 ? 0 : -128, hi = lo + 255;
                const int32_t es = (int32_t)((su >> 23) & 255u);
                const int64_t ms = (int64_t)(su & 0x7fffffu) | (es ? (int64_t)1 << 23 : 0);
                for (int m = 0; m < TM; m++)
                    for (int n = 0; n < TN;
                         n++) { /* eager_quant_code, written out: only kernel text reaches the core */
                        const uint32_t fu = ((const uint32_t *)(x + m * TK))[n];
                        const int32_t ef = (int32_t)((fu >> 23) & 255u);
                        const int neg = (fu >> 31) != 0;
                        int32_t qc;
                        if (ef == 255) {
                            qc = (fu & 0x7fffffu) ? 0 : neg ? lo : hi;
                        } else {
                            const int64_t mf = (int64_t)(fu & 0x7fffffu) | (ef ? (int64_t)1 << 23 : 0);
                            const int32_t D = (ef ? ef : 1) - (es ? es : 1);
                            int64_t k = 0;
                            int sat = 0;
                            if (mf != 0 && D >= -38) {
                                if (D >= 12) {
                                    sat = 1;
                                } else {
                                    int64_t num = D > 0 ? mf << D : mf;
                                    const int64_t den = D < 0 ? ms << -D : ms;
                                    for (int b = 12; b >= 0; b--)
                                        if ((num >> b) >= den) {
                                            num -= den << b;
                                            k |= (int64_t)1 << b;
                                        }
                                    if (2 * num > den || (2 * num == den && (k & 1)))
                                        k += 1;
                                }
                            }
                            const int64_t q = (neg ? -k : k) + zp;
                            qc = sat ? (neg ? lo : hi) : (int32_t)(q < lo ? lo : q > hi ? hi : q);
                        }
                        code[m * TN + n] = (int8_t)qc;
                    }
            }
            aie::release(a);
            aie::release(w);
        }
        int8_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int i = 0; i < TM * TN; i++)
            y[i] = code[i];
        aie::commit(c);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_quant_launch_fn)(int8_t *A, int8_t *W, int8_t *Y);

/* args: A [M][4 N] f32 bytes, W [N][4 N] ({scale, zp} at each chunk's start), Y [M][N] */
static int eager_quant_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t N,
                             eager_quant_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * 4 * N));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(N * 4 * N));
    int8_t *y = (int8_t *)eager_slot(2, (size_t)(M * N));
    if (!a || !w || !y)
        return AIEK_ERR_ALLOC;
    launch(a, w, y);
    eager_fetch_out(2, &args[2], (size_t)(M * N));
    return AIEK_OK;
}
