/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: table lookup with wide results, Y[r][j] = table_r[X[r][j]] as ES bytes (lutw_eager_ref.h: the
 * layout) -- 8-bit codes in, s32 / f32 (ES 4) or 16-bit (ES 2) values out. The host builds the tables by running
 * ref: on all 256 codes, so u8 -> f32 eltwise, quantized eltwise to f32, PReLU / batch norm of an 8-bit source
 * into a wide dst are exact by construction. A row is one channel's run: its table, then its codes.
 * A row op on the non-GEMM gather lowering: mesh row pm owns rows pm R / HW_ROWS .., each TR-row strip of whole rows
 * goes to the row's 4 cores; core pn writes its quarter of every row's values.
 ******************************************************************************/
#include "eager_common.h"
#include "lutw_eager_ref.h" /* the layout, eager_lutw_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, V (codes per row) of 16
template <int R, int V, int TR, int ES>
__global__ void lutw(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 256 * ES + V>> x,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<R, V * ES>> y) {
    const int rows = R / HW_ROWS, q = V / 4, TB = 256 * ES, ld = TB + V;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *s = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, ld});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * q * ES}, {TR, q * ES});
        for (int r = 0; r < TR; r++) {
            const uint8_t *tab = (const uint8_t *)(s + r * ld), *code = tab + TB + pn * q;
            int8_t *orow = o + r * q * ES;
            for (int j = 0; j < q; j++)
                for (int b = 0; b < ES; b++)
                    orow[j * ES + b] = (int8_t)tab[code[j] * ES + b];
        }
        aie::release(x);
        aie::commit(y);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_lutw_launch_fn)(int8_t *X, int8_t *Y);

/* args: X [R][256 ES + V] tables + codes, Y [R][V] ES-byte values */
static int eager_lutw_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t V, int64_t ES,
                            eager_lutw_launch_fn launch) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * (256 * ES + V)));
    int8_t *y = (int8_t *)eager_slot(1, (size_t)(R * V * ES));
    if (!x || !y)
        return AIEK_ERR_ALLOC;
    launch(x, y);
    eager_fetch_out(1, &args[1], (size_t)(R * V * ES));
    return AIEK_OK;
}
