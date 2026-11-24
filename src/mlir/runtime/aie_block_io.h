/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// Kernel-side block-load convention for __global__(op, map) kernels:
//
//   const int8_t *x = aie::load_buffer(win_a, {ih0, iw0, 0}, {IH_T, IW_T, C});
//   int8_t *y       = aie::store_buffer(win_c, {oh0, ow0, oc0}, {OH_T, OW_T, OC_T});
//   ... compute ...
//   aie::release(win_a);   aie::commit(win_c);
//
// Offsets/shapes are ELEMENT coordinates of the port's whole tensor (as in
// Triton's block pointers). The data still arrives by DMA in a fixed order, so
// a load/store does not choose WHAT arrives: it acquires the port's next window
// and CHECKS that the requested block is the one the DMA delivers there. The
// expected block comes from the compiler-derived aiehlc_desc_<port> (emitted
// by convautotile from the same numbers as the port's DMA descriptor), so a
// mismatch -- a wrong offset, shape, or loop order -- is reported with klog tag
// "BLKE" instead of silently computing on the wrong data.
//
// The aiehlc pipeline rewrites the aie:: calls to the functions below (the
// kernel TU's `namespace aie` belongs to the AIE API). Design:
// doc/design/op_mapping_programming_model.md.
#ifndef AIE_BLOCK_IO_H
#define AIE_BLOCK_IO_H

#include <stdint.h>

// bfloat16 ports: on the core a bfloat16 tensor travels in a 16-bit window
// (the pipeline carries it as i16; the kernel works on its blocks through
// `bfloat16` pointers, chess's native type). ADF has no bfloat16 window type.
#ifdef __chess__
typedef input_window_int16 input_window_bfloat16;
typedef output_window_int16 output_window_bfloat16;
#endif

// The kernel-side declarations (op, block sizes, one descriptor + cursor per
// port) are emitted by convautotile::kernelTUDecls from the same Derived
// numbers the ports' GemmSpace / DMA are built from.

// The op of __global__(conv, map): `conv.kh` etc. in the kernel body.
struct aiehlc_conv2d_t {
    int h, w, c, k, kh, kw, stride, pad;
};
// Per-port block descriptor: the block a port delivers in round (i, j) on core
// (p0, p1) is offset[d] = base_p0[d]*p0 + base_p1[d]*p1 + step_i[d]*i +
// step_j[d]*j, shape[d]; rounds run i outer, j inner (the DMA order).
struct aiehlc_blk_desc {
    int rank;
    int shape[4];
    int base_p0[4];
    int base_p1[4];
    int step_i[4];
    int step_j[4];
    int rounds_i;
    int rounds_j;
};
// `map.oh_t` etc. in the kernel body: the mapping's block sizes.
struct aiehlc_kernel_view_t {
    int oh, ow, oh_per_core, oc_per_core, oh_t, ow_t, ih_t, iw_t, c_align;
};
// Round index of the next block on one port, kept in [0, rounds_i*rounds_j)
// (persists across kernel calls and wraps per launch).
struct aiehlc_blk_cursor {
    int n;
};
// A block offset or shape: {d0, d1, d2[, d3]}.
struct aiehlc_blk {
    int v[4];
};

// 1 when (off, shape) is the block port `d` delivers next on core (p0, p1).
static inline int aiehlc_blk_check(const aiehlc_blk_desc &d, aiehlc_blk_cursor &cur, int p0, int p1,
                                   const aiehlc_blk &off, const aiehlc_blk &shape) {
    const int per = d.rounds_i * d.rounds_j;
    // Keep n in [0, per): a resident run(-1) graph would otherwise overflow the
    // int after 2^31 blocks and report false BLKE from then on. Also avoids a
    // per-call '%' (no hardware divide on the AIE scalar unit).
    if (per <= 0 || cur.n < 0 || cur.n >= per)
        cur.n = 0;
    const int round = cur.n;
    const int i = d.rounds_j > 0 ? round / d.rounds_j : 0;
    const int j = d.rounds_j > 0 ? round % d.rounds_j : 0;
    if (per > 0 && ++cur.n == per)
        cur.n = 0;
    int ok = 1;
    for (int k = 0; k < d.rank; ++k) {
        const int want = d.base_p0[k] * p0 + d.base_p1[k] * p1 + d.step_i[k] * i + d.step_j[k] * j;
        if (off.v[k] != want || shape.v[k] != d.shape[k]) {
            klog("BLKE", (k << 16) | round); // dim k, round
            klog("BLKW", want);              // expected offset
            klog("BLKG", off.v[k]);          // requested offset
            ok = 0;
        }
    }
    return ok;
}

// A block's address, converting to the pointer type the kernel assigns it to
// (`const int8_t *`, `const bfloat16 *`, ...) -- as in the kernel analysis and kemu.
struct aiehlc_block_ptr {
    int8_t *p;
    template <typename T> operator T *() const { return (T *)p; }
};
template <typename W>
static inline aiehlc_block_ptr aiehlc_load_buffer(W *win, const aiehlc_blk_desc &d, aiehlc_blk_cursor &cur, int p0,
                                                  int p1, aiehlc_blk off, aiehlc_blk shape) {
    aiehlc_blk_check(d, cur, p0, p1, off, shape);
    return {(int8_t *)acquire_input_window(win)};
}
template <typename W>
static inline aiehlc_block_ptr aiehlc_store_buffer(W *win, const aiehlc_blk_desc &d, aiehlc_blk_cursor &cur, int p0,
                                                   int p1, aiehlc_blk off, aiehlc_blk shape) {
    aiehlc_blk_check(d, cur, p0, p1, off, shape);
    return {(int8_t *)acquire_output_window(win)};
}
template <typename W> static inline void aiehlc_release(W *win) { release_input_window(win); }
template <typename W> static inline void aiehlc_commit(W *win) { release_output_window(win); }

#endif // AIE_BLOCK_IO_H
