/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: reduction of 8-bit rows to an exact int32 -- one operation OP and the signedness SG compiled into
 * the instance (red8_eager_ref.h: the layout and the operations). The host cuts each reduced segment into rows of L,
 * pads with the op's neutral value and combines the rows' partials (sum / max / min of integers: exact), then
 * divides / adds eps / takes the root as ref: does in double -- the result is ref:'s bit for bit. 1-byte values in,
 * one int32 per row out.
 * A row op on the non-GEMM gather lowering: mesh row pm owns rows pm R / HW_ROWS .., each TR-row strip of whole rows
 *goes to the row's 4 cores; core pn writes the row's value into its own 4 bytes of the 16-byte output row.
 ******************************************************************************/
#include "eager_common.h"
#include "red8_eager_ref.h" /* the layout, eager_red8_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, L of 16; OP 1 sum, 2 max, 3 min, 4 sum |x|, 5 sum x x, 6 max |x|
template <int R, int L, int TR, int OP, int SG>
__global__ void red8(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, L>> x,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 16>> y) {
    const int rows = R / HW_ROWS;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *s = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, L});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * 4}, {TR, 4});
        for (int r = 0; r < TR; r++) {
            const int8_t *xr = s + r * L;
            int32_t acc = SG ? (int32_t)xr[0] : (int32_t)(uint8_t)xr[0];
            acc = OP == 1 || OP == 2 || OP == 3 ? acc : OP == 5 ? acc * acc : acc < 0 ? -acc : acc;
            for (int k = 1; k < L; k++) {
                const int32_t v = SG ? (int32_t)xr[k] : (int32_t)(uint8_t)xr[k], a = v < 0 ? -v : v;
                acc = OP == 1   ? acc + v
                      : OP == 2 ? (v > acc ? v : acc)
                      : OP == 3 ? (v < acc ? v : acc)
                      : OP == 4 ? acc + a
                      : OP == 5 ? acc + v * v
                                : (a > acc ? a : acc);
            }
            *(int32_t *)(o + r * 4) = acc;
        }
        aie::release(x);
        aie::commit(y);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_red8_launch_fn)(int8_t *X, int8_t *Y);

/* args: X [R][L] 8-bit, Y [R][4] int32 (the value in word 0) */
static int eager_red8_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t L, eager_red8_launch_fn launch) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * L));
    int8_t *y = (int8_t *)eager_slot(1, (size_t)(R * 16));
    if (!x || !y)
        return AIEK_ERR_ALLOC;
    launch(x, y);
    eager_fetch_out(1, &args[1], (size_t)(R * 16));
    return AIEK_OK;
}
