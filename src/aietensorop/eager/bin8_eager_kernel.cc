/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: binary of two 8-bit tensors -- one operation OP compiled into the instance, Q 0 the exact
 * int32 a OP b (the plain 8-bit binary: any destination type is then exact on the host), Q 1 the quantized binary
 * in f32 (dequantize both, OP, the relu / clip post-op; the host quantizes as ref: does). bin8_eager_ref.h: the
 * layout; eager_fmath_ops.cc: the math. The 8-bit inputs travel as bytes and come back 4 bytes wide.
 * XY is a looped whole-row strip broadcast along the mesh row; core pn writes its quarter of the columns.
 ******************************************************************************/
#include "eager_common.h"
#include "eager_fmath_ops.cc" // EAGER_FMATH_OPS: the math as text (spliced; the kernel expands it)
#include "bin8_eager_ref.h"   /* the layout, eager_bin8_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, L (elements per row) of 16
template <int R, int L, int TR, int OP, int Q>
__global__ void bin8(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 2 * L>> xy,
                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<4, 32>> p,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 4 * L>> z) {
    const int rows = R / HW_ROWS, q = L / 4;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    EAGER_FMATH_OPS;
    const float *pw = (const float *)aie::load_buffer(p, {pn, 0}, {1, 32}, {.resident = true});
    const float sa = pw[0], nza = pw[1], sb = pw[2], nzb = pw[3], lo = pw[4], hi = pw[5];
    const int sg = ((const int32_t *)pw)[6];
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *s = aie::load_buffer(xy, {pm * rows + i * TR, 0}, {TR, 2 * L});
        int8_t *o = aie::store_buffer(z, {pm * rows + i * TR, pn * L}, {TR, L});
        for (int r = 0; r < TR; r++) {
            const int8_t *ar = s + r * 2 * L + pn * q, *br = s + r * 2 * L + L + pn * q;
            for (int c = 0; c < q; c++) {
                const int a = (sg & 1) ? (int)ar[c] : (int)(uint8_t)ar[c];
                const int b = (sg & 2) ? (int)br[c] : (int)(uint8_t)br[c];
                if (Q) {
                    const float v = eager_fop(OP, ((float)a + nza) * sa, ((float)b + nzb) * sb, 0.0f, 0.0f);
                    ((float *)(o + r * L))[c] = eg_fmin(eg_fmax(v, lo), hi);
                } else {
                    ((int32_t *)(o + r * L))[c] = OP == EG_F_ADD   ? a + b
                                                  : OP == EG_F_SUB ? a - b
                                                  : OP == EG_F_MUL ? a * b
                                                  : OP == EG_F_MAX ? (a > b ? a : b)
                                                                   : (a < b ? a : b);
                }
            }
        }
        aie::release(xy);
        aie::commit(z);
    }
    aie::release(p);
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_bin8_launch_fn)(int8_t *XY, int8_t *P, int8_t *Z);

/* args: XY [R][2 L] 8-bit, P [4][8] words, Z [R][L] int32 / f32 */
static int eager_bin8_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t L, eager_bin8_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    int8_t *xy = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * 2 * L));
    int8_t *p = (int8_t *)eager_stage_in(1, &args[1], 128);
    int8_t *z = (int8_t *)eager_slot(2, (size_t)(R * 4 * L));
    if (!xy || !p || !z)
        return AIEK_ERR_ALLOC;
    launch(xy, p, z);
    eager_fetch_out(2, &args[2], (size_t)(R * 4 * L));
    return AIEK_OK;
}
