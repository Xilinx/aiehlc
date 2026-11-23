/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: step tables, a 32-bit value -> 8-bit code (steps_eager_ref.h: the layout). The host finds every
 * step of ref: by bisection over the key space (fixed-megakernel steps:'s tables), so f32 / s32 -> s8 / u8
 * reorders, batch norm, PReLU, broadcast binaries and the arithmetic eltwise algorithms are exact by construction.
 * A row is one channel's run: its table (S bytes), then its 4-byte values; 1-byte codes out.
 * A row op on the non-GEMM gather lowering: mesh row pm owns rows pm R / HW_ROWS .., each TR-row strip of whole rows
 * goes to the row's 4 cores; core pn writes its quarter of every row's codes.
 ******************************************************************************/
#include "eager_common.h"
#include "steps_eager_ref.h" /* the layout, eager_steps_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, V (values per row) of 16, S (table bytes) of 16
template <int R, int V, int TR, int S>
__global__ void steps(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, S + 4 * V>> x,
                      aie::port<output_window_int8 *, aie::OFM, aie::shape<R, V>> y) {
    const int rows = R / HW_ROWS, q = V / 4, ld = S + 4 * V;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *s = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, ld});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * q}, {TR, q});
        for (int r = 0; r < TR; r++) {
            const int8_t *tb = s + r * ld;
            const int32_t ns0 = *(const int32_t *)tb, nmax = (S - 10) / 6; /* the trace runs on arbitrary bytes */
            const int32_t ns = ns0 < 0 ? 0 : ns0 > nmax ? nmax : ns0;
            const int16_t nanc = *(const int16_t *)(tb + 4), kind = *(const int16_t *)(tb + 6);
            /* the 8-byte header (steps_eager_ref.h's EG_STEPS_HDR: only the kernel's own text reaches the trace) */
            const int32_t *T = (const int32_t *)(tb + 8), *val = (const int32_t *)(tb + S) + pn * q;
            const int16_t *P = (const int16_t *)(tb + 8 + 4 * ns);
            for (int j = 0; j < q; j++) {
                const int32_t v = val[j];
                if (kind == 0 && (v & 0x7fffffff) > 0x7f800000) {
                    o[r * q + j] = (int8_t)nanc;
                    continue;
                }
                const int32_t key = kind == 0 && v < 0 ? v ^ 0x7fffffff : v;
                int lo = 0, hi = ns; /* first T > key */
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (T[mid] <= key)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                o[r * q + j] = (int8_t)P[lo];
            }
        }
        aie::release(x);
        aie::commit(y);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_steps_launch_fn)(int8_t *X, int8_t *Y);

/* args: X [R][S + 4 V] tables + values, Y [R][V] codes */
static int eager_steps_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t V, int64_t S,
                             eager_steps_launch_fn launch) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * (S + 4 * V)));
    int8_t *y = (int8_t *)eager_slot(1, (size_t)(R * V));
    if (!x || !y)
        return AIEK_ERR_ALLOC;
    launch(x, y);
    eager_fetch_out(1, &args[1], (size_t)(R * V));
    return AIEK_OK;
}
