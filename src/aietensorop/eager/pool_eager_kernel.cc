/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: 8-bit max / average pooling over host-packed windows (pool_eager_ref.h: one window per
 * row, its divisor in the row's last word). The element type (U8: int8 / uint8) and the algorithm (AVG) are
 * template arguments, so each layer gets the instance of its own type, window size and window count.
 *
 * Routing: the GEMM skeleton, because kernel-only ports lower through the GEMM path only (a reduction's own
 * output walk does not lower yet: upstream simplereduce_rowsum.cc). The same bytes are viewed as 4 windows per
 * row, X [MW/4][4 KK] (the "A" operand, one K round of 4 KK), Y [MW/4][16] (the "C" operand, N = 16, so mesh
 * column pn owns bytes 4 pn .. 4 pn + 3 = window pn's code word), plus a 16 x 4 KK WEIGHTS port the kernel never
 * reads (Z: any bytes; it stands in for the B walk the lowering expects).
 ******************************************************************************/
#include "eager_common.h"
#include "pool_eager_ref.h" /* the layout, eager_div_rne, eager_pool_ref (the instances' AIEK_REFERENCE) */

// MW a multiple of 16 * TR (every mesh row gets whole TR-row tiles), KK of 4 (taps + the divisor word)
template <int MW, int KK, int TR, int AVG, int U8>
__global__ void pool(aie::port<input_window_int8 *, aie::IFM, aie::shape<MW / 4, 4 * KK>> x,
                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<16, 4 * KK>> z,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<MW / 4, 16>> y) {
    const int ntile = MW / (16 * TR), taps = KK - 4;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TR) + t * TR;
        int32_t code[TR];
        for (int j = 0; j < 1; j++) {
            const int8_t *a = aie::load_buffer(x, {m0, j * 4 * KK}, {TR, 4 * KK});
            const int8_t *u = aie::load_buffer(z, {pn * 4, j * 4 * KK}, {4, 4 * KK});
            for (int r = 0; r < TR;
                 r++) { /* eager_pool_row, written out: only the kernel's own text reaches the core */
                const int8_t *row = a + r * 4 * KK + pn * KK;
                const int32_t div = ((const int32_t *)(row + taps))[0];
                int32_t v = AVG || U8 ? 0 : -128;
                for (int k = 0; k < taps; k++) {
                    const int32_t e = U8 ? (int32_t)(uint8_t)row[k] : (int32_t)row[k];
                    v = AVG ? v + e : (e > v ? e : v);
                }
                if (AVG) {
                    int32_t q = div > 0 ? v / div : 0, rm = div > 0 ? v % div : 0;
                    if (rm < 0)
                        q -= 1, rm += div;
                    v = div > 0 && (2 * rm > div || (2 * rm == div && (q & 1))) ? q + 1 : q;
                }
                const int32_t lo = U8 ? 0 : -128, hi = lo + 255;
                code[r] = v < lo ? lo : v > hi ? hi : v;
            }
            (void)u;
            aie::release(x);
            aie::release(z);
        }
        int8_t *o = aie::store_buffer(y, {m0, pn * 4}, {TR, 4});
        for (int r = 0; r < TR; r++) {
            o[r * 4] = (int8_t)code[r];
            o[r * 4 + 1] = o[r * 4 + 2] = o[r * 4 + 3] = 0;
        }
        aie::commit(y);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_pool_launch_fn)(int8_t *X, int8_t *Z, int8_t *Y);

/* args: X [MW][KK] packed windows, Z [16][4 KK] (the WEIGHTS stand-in, any bytes), Y [MW][4] -- the instance's
 * own shapes */
static int eager_pool_entry(const aiek_tensor *args, int num_args, int64_t MW, int64_t KK,
                            eager_pool_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(MW * KK));
    int8_t *z = (int8_t *)eager_stage_in(1, &args[1], (size_t)(16 * KK * 4));
    int8_t *y = (int8_t *)eager_slot(2, (size_t)(MW * 4));
    if (!x || !z || !y)
        return AIEK_ERR_ALLOC;
    launch(x, z, y);
    eager_fetch_out(2, &args[2], (size_t)(MW * 4));
    return AIEK_OK;
}
