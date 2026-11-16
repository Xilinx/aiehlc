/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * gemm_uk -- int8 x int8 -> int32 GEMM block for VEK280, as an aiednn kernel.
 *
 * As matmul_uk.cc (aie::mmul<8,8,8>), but the accumulators leave the array as
 * exact int32, row-major per core tile.
 *
 * Logical block (the AIEK_SPEC): A [32*HW_ROWS][512] s8, B [128][512] s8 -> C [32*HW_ROWS][128]
 * s32, on HW_ROWS x 4 cores -> a 32x32 int32 tile per core.
 *
 * K must equal the program's B row count N (512): aiehlc offsets each column's B chunk by
 * tile * N bytes while the BD strides by K (skill gemmnonsquareb).
 *
 * The program is all-int8, so C is the int32 result's bytes ([128][512] int8) and each
 * column's B tile must be 128 rows: the entry puts column j's 32 real B rows at rows
 * j*128 .. j*128+31 of a zeroed [512][512] staging buffer.
 ******************************************************************************/
#include "gemm.h"
// AIEHLC_KERNEL_PROLOGUE_BEGIN
#include "aie_modes_cfg.h"  // which optional modes this build compiles (a per-app copy shadows the default)
#include "aie_modes_pool.h" // src/aietensorop/ops (copied flat by dnn_build); the markers carry it into the core source
#include "aie_modes_lut.h"
#include "aie_modes_conv.h"  // direct conv: region gather (CONV mode)
#include "aie_modes_sdpa.h"  // attention per (batch, head) item (SDPA mode)
#include "aie_modes_poolv.h" // the pool mode's vector path (MAX / SUM, stride 1 / 2)
#include "aie_modes_bf16.h"  // bf16 GEMM, f32 accumulation (BF16 mode)
#include "aie_modes_fvec.h"  // f32 pointwise / row ops (FVEC mode)
#include "aie_modes_wq.h"    // grouped weight-only quant: f32 group-scaled sums (WQ mode)
#include "aie_modes_fpool.h" // f32 windowed pooling 2-D / 3-D (FPOOL mode)
#include "aie_modes_mmt.h"   // MMT: short-K GEMM on mlir-aie's matmul micro-kernel (the mlir-aie submodule)
#include "aie_modes_dwq.h"   // depthwise + requant (0xD8) in vector lanes
#include "aie_modes_gemmq.h" // GEMM + requant (0xF1) in vector lanes
/* 0xD8 on an 8-row mesh: the slot in B rows 120..127 loses its bytes 3584..3595 to the descriptor; they
 * travel in the item's tail past the <= 49 weight bytes (DW_WBYTES) instead */
#define DWQ_STASH 4084
#define DWQ_STASH_N 12
/* core cycle stamp (diagnostic, default off): a GEMMQ launch writes its cycles into C row 0 bytes 64..67 */
#ifndef AIEDNN_KCYC
#define AIEDNN_KCYC 0
#endif
#define KC_TIC() (AIEDNN_KCYC ? (uint64_t)aie::tile::current().cycles() : (uint64_t)0)
// AIEHLC_KERNEL_PROLOGUE_END

#include "aie_user_kernel.h"

#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)
#pragma KERNELCONFIGOFFLOAD      // the core programs its own DMA: needed by --persistent (host re-arms shim BDs only)
#pragma memtile_staging(weights) // B cached in the mem tiles across launches (aiednn_gemm_weights_key)

constexpr int kTileM = 32;   // A rows (= C rows) per core
constexpr int kTileNB = 128; // B rows (= C bytes per row) per core; a quarter are real
constexpr aie::Bytes kL1Budget{8192};
constexpr aie::GemmSpace RowBA = {.policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = kL1Budget}},
                                  .d1 = {.fullsize = M, .tile_size = kTileM, .stride = kTileM},
                                  .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}};
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = kL1Budget}},
                                  .d1 = {.fullsize = N, .tile_size = kTileNB, .stride = kTileNB},
                                  .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}};
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = kL1Budget}},
    .d1 = {.fullsize = M, .tile_size = kTileM, .stride = kTileM},
    .d2 = {.fullsize = N, .tile_size = kTileNB, .stride = kTileNB}};
constexpr aie::GlobalPolicy gemm_policy = {.fullconnect_auto = 1};
__global__(gemm_policy) void gemm_s8s32(aie::port<input_window_int8 *, RowBA> win_a,
                                        aie::port<input_window_int8 *, ColBB> win_b,
                                        aie::port<output_window_int8 *, LtoR_Merge> win_c) {
    const int k_rounds = aie::get_arg_total_rounds_in_dim(1, win_a);
    const int eff_k = aie::get_arg_per_round_size_in_dim(1, win_a);
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);
    const int tile_rows = aie::get_arg_per_round_size_in_dim(0, win_a);
    const int tile_cols = aie::get_arg_per_round_size_in_dim(0, win_b);
    const int rows_per_round = buf_sz_a / eff_k;
    const int cols_per_round = buf_sz_b / eff_k;

    const int real_cols = tile_cols / 4; // int32 outputs per C row of this core
    const int k_tiles = eff_k / 8;
    const int m_tiles = tile_rows / 8;
    const int n_tiles = real_cols / 8;
    const int b_strips = cols_per_round / 8;
    alignas(aie::vector_decl_align) int8_t a_pack[tile_rows * eff_k];
    alignas(aie::vector_decl_align) int8_t b_strip[eff_k * 8];
    alignas(aie::vector_decl_align) int32_t acc[tile_rows * real_cols];
    // K chaining (0xA0 / d[2] bit 0): acc carries a partial sum from the previous launch of the
    // persistent kernel. Its address escapes, so accumulate-only stores are not dead at return.
    static int32_t *volatile acc_keep;
    acc_keep = acc;
    alignas(aie::vector_decl_align) int32_t local_out[tile_rows * real_cols]; // = tile_rows * tile_cols bytes
    // WQ mode keeps its f32 tile here across launches (continue flag), as acc above
    static int32_t *volatile out_keep;
    out_keep = local_out;

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        uint64_t kc_t0 = 0; // AIEDNN_KCYC: set at the first window
        // Depthwise overlay (dw_blk_s8s32): B's padding row 127 -- always zero for a GEMM --
        // carries a descriptor; then this core's 4 KB slice of A (rows 8c..8c+7, c = its mesh
        // column, from the descriptor) is one depthwise item: zero-padded input rows plus the
        // KHxKW weights in its last 32 bytes, gathered into acc (unused in this mode).
        bool dw = false; // any non-GEMM mode: the item gather below is shared
        int mode = 0;    // 0 GEMM, 0xD7 depthwise, AIE_MODE_OP_POOL, AIE_MODE_OP_ELTWISE, AIE_MODE_OP_LUT
        uint8_t mdesc[16];
        // GEMM + integer requant (0xF1): row 127 of the B block carries {zp, lo, hi, M[32], b[32],
        // sh[32]} for this column's 32 channels; each k-round window holds eff_k bytes of it
        bool gq = false;
        bool bfcont = false, bfmore = false; // bf16 mode: accumulate onto the last launch's tile / C not final yet
        bool kcont = false;                  // K chain: start from acc (a previous launch's partial sum)
        bool kmore = false;                  // K chain: accumulate only (0xA0), the output is not the result
        bool wq = false;          // grouped weight-only quant (AIE_MODE_OP_WQ): f32 tile += scale_g * int32 partial
        int wq_gr = 1, wq_ph = 0; // rounds per group, rounds of the current group summed before this launch
        bool dwq =
            false; // depthwise + integer requant (0xD8): int32 {zp, lo, hi, M}, int64 B, int32 sh at item bytes 4000..
        alignas(16) uint8_t qrow[512];
        // fused residual (0xF1, d[3] bit 0): row 126 = {R int64[32], rzp int32} and this core's
        // 32x32 int8 residual tile in rows 96 + 2r, 97 + 2r of its column's B block (r = mesh row; rows
        // 96..127 are the per-launch tail, 0..95 depend only on the weights block: memtile-cacheable)
        bool gres = false;
        alignas(16) uint8_t rqrow[512];
        alignas(16) int8_t rtile[1024];
        const int my_r = (int)(get_coreid() & 0x1F) - 3;
        int dw_c = 0, dw_s = 1, dw_kh = 3, dw_kw = 3, dw_wp = 0, dw_rout = 0, dw_wo = 0;
        int8_t *dw_raw = (int8_t *)acc; // 4096 bytes
        // 0xD8 with d[11] = nb (<= 2): nb more items per core in B's rows 16r + 8q .. +7 (rows
        // 0..95 carry no weights in this mode); their int8 bands follow item 0's in the C tile
        int dw_nb = 0;
        // also the LUT mode's tables (AIE_LUT_TBYTES) and the direct-conv input region (16 KB, CONV)
        alignas(16) int8_t dwb[AIEK_MODE_CONV ? 4 * 4096 : 2 * 4096];
#if AIEK_MODE_CONV
        // direct conv (CONV_MAGIC in B row CONV_GEOM_ROW0, opcode 0 / 0xA0 / 0xF1): A carries this core row's raw HWC
        // input region (CONV_GEOM in B row CONV_GEOM_ROW0 + my_r); the core gathers the im2col tile itself
        bool creg = false, crun64 = false;
        int32_t cbase[32];  // per A row: region offset of its pixel (-1: past M)
        int32_t ctoff[128]; // per 4-byte K unit of the block: offset of its (tap, channel) (-1: past K)
#endif
        for (int kr = 0; kr < k_rounds; kr++) {
            const bool first0 = (kr == 0); // with kcont (read from this round's descriptor below) acc continues
            const bool last = (kr == k_rounds - 1);
            // B's first window before A, so the descriptor is known before A is packed
            const uint32_t *B_first = (const uint32_t *)acquire_input_window(win_b);
            if (AIEDNN_KCYC && kr == 0)
                kc_t0 = KC_TIC();
            if (kr == 0 && num_b_rounds == 1 && cols_per_round == 128) {
                const uint8_t *d = (const uint8_t *)B_first + 127 * eff_k;
                // a mode compiled out (aie_modes_cfg.h) decodes as plain GEMM: its host entry is not in the build
                gq = AIEK_MODE_GEMMQ && d[0] == 0xF1;
                dwq = AIEK_MODE_DWQ && d[0] == 0xD8;
                kmore = d[0] == 0xA0;
                gres = gq && (d[3] & 1);
                kcont = (d[0] == 0 || d[0] == 0xA0 || d[0] == 0xF1) && (d[2] & 1);
#if AIEK_MODE_WQ
                wq = d[0] == AIE_MODE_OP_WQ;
                if (wq) {
                    wq_gr = d[3] ? d[3] : 1;
                    wq_ph = d[4];
                    if (!(d[2] & 1)) /* a new f32 tile; with continue it holds the previous launches' sums */
                        for (int i = 0; i < tile_rows * real_cols; i += 16)
                            aie::store_v((float *)local_out + i, aie::zeros<float, 16>());
                }
#endif
#if AIEK_MODE_CONV
                creg = (d[0] == 0 || d[0] == 0xA0 || d[0] == 0xF1) &&
                       *(const int32_t *)((const uint8_t *)B_first + CONV_GEOM_ROW0 * eff_k) == CONV_MAGIC;
#endif
                mode = AIEK_MODE_DW && (d[0] == 0xD7 || dwq) ? 0xD7
                       : (AIEK_MODE_POOL && d[0] == AIE_MODE_OP_POOL) ||
                               (AIEK_MODE_ELTWISE && d[0] == AIE_MODE_OP_ELTWISE) ||
                               (AIEK_MODE_LUT && d[0] == AIE_MODE_OP_LUT) ||
                               (AIEK_MODE_STEPS && d[0] == AIE_MODE_OP_STEPS) ||
                               (AIEK_MODE_MMT && d[0] == AIE_MODE_OP_MMT) ||
                               (AIEK_MODE_SDPA && d[0] == AIE_MODE_OP_SDPA) ||
                               (AIEK_MODE_BF16 && d[0] == AIE_MODE_OP_BF16) ||
                               (AIEK_MODE_FVEC && d[0] == AIE_MODE_OP_FVEC) ||
                               (AIEK_MODE_FPOOL && d[0] == AIE_MODE_OP_FPOOL)
                           ? d[0]
                           : 0;
                dw = mode != 0;
                bfcont = mode == AIE_MODE_OP_BF16 && (d[2] & 1);
                bfmore = mode == AIE_MODE_OP_BF16 && (d[2] & 2);
                for (int i = 0; i < 16; i++)
                    mdesc[i] = d[2 + i];                                    // mode bytes after opcode + column
                if (mode == AIE_MODE_OP_LUT || mode == AIE_MODE_OP_STEPS) { // {n, run0, inner, nt}, ES, kind, stride
                    for (int i = 0; i < 7; i++)
                        mdesc[i] = d[AIE_LUT_ENTRY(my_r) + i];
                    for (int i = 0; i < 4; i++)
                        mdesc[7 + i] = d[2 + i];
                }
                if (mode == AIE_MODE_OP_FVEC) { /* this core's {n, rowlen} entry, then op / flags / alpha / beta */
                    for (int i = 0; i < 4; i++)
                        mdesc[i] = d[AIE_FVEC_ENTRY(my_r) + i];
                    for (int i = 0; i < 10; i++)
                        mdesc[4 + i] = d[2 + i];
                }
#if AIEK_MODE_MMT
                if (mode == AIE_MODE_OP_MMT) { /* this column's pre-tiled B + requant params stay for the launch */
                    const uint32_t *src = B_first;
                    uint32_t *dst = (uint32_t *)dwb;
                    for (int w = 0; w < AIE_MMT_RES / 4; w++)
                        dst[w] = src[w];
                }
#endif
                dw_c = d[1];
                if (mode == 0xD7) {
                    dw_c = d[1], dw_s = d[2], dw_kh = d[3], dw_kw = d[4];
                    dw_wp = d[5] | (d[6] << 8), dw_rout = d[7] | (d[8] << 8), dw_wo = d[9] | (d[10] << 8);
                    dw_nb = dwq ? (d[11] > 2 ? 2 : d[11]) : 0;
                }
            }
#if AIEK_MODE_GEMMQ
            if (gq && (kr + 1) * eff_k <= 512) {
                const uint8_t *d = (const uint8_t *)B_first + 127 * eff_k;
                for (int i = 0; i < eff_k; i++)
                    qrow[kr * eff_k + i] = d[i];
                if (gres) {
                    const uint8_t *rq = (const uint8_t *)B_first + 126 * eff_k;
                    const int8_t *r0 = (const int8_t *)B_first + (96 + 2 * my_r) * eff_k;
                    for (int i = 0; i < eff_k; i++) {
                        rqrow[kr * eff_k + i] = rq[i];
                        rtile[kr * eff_k + i] = r0[i];
                        rtile[512 + kr * eff_k + i] = r0[eff_k + i];
                    }
                }
            }
#endif
#if AIEK_MODE_CONV
            if (creg) {
                if (kr == 0) { // the whole region (geometry in its last 64 bytes), then the tables
                    conv_region_in(win_a, dwb, cbase, ctoff, k_rounds, num_a_rounds, rows_per_round, eff_k, tile_rows);
                    crun64 = eff_k == 64 && ((const int32_t *)(dwb + CONV_REGION_DATA))[CONV_GEOM_CG] % 64 == 0;
                }
                conv_pack_round(dwb, cbase, ctoff, a_pack, kr, eff_k, k_tiles, tile_rows, crun64);
            }
            const int a_rounds = creg ? 0 : num_a_rounds; // direct conv: A came in round 0
#else
            const int a_rounds = num_a_rounds;
#endif
            /* one loop header in both branches: lint's brace matching sees through #if */
            for (int ra = 0; ra < a_rounds; ra++) {
                const uint32_t *A_w = (const uint32_t *)acquire_input_window(win_a);
#if AIEK_MODE_BF16
                if (mode == AIE_MODE_OP_BF16) { /* this round's pre-tiled A (2 KB): used with B's window below */
                    uint32_t *dst = (uint32_t *)dwb;
                    for (int w = 0; w < rows_per_round * eff_k / 4; w++)
                        dst[w] = A_w[w];
                    release_input_window(win_a);
                    continue;
                }
#endif
#if AIEK_MODE_MMT
                if (mode == AIE_MODE_OP_MMT) { /* rows [kr * mr, +mr) of the core's 128, straight from the window */
                    aie_mmt_round((const int8_t *)A_w, (const int8_t *)dwb, mdesc[0], kr, acc, (int8_t *)local_out);
                    release_input_window(win_a);
                    continue;
                }
#endif
#if AIEK_MODE_ITEMS
                if (dw) {
                    for (int r = 0; r < rows_per_round; r++) {
                        const int row = ra * rows_per_round + r - 8 * dw_c; // row of this core's item
                        if (row < 0 || row >= 8)
                            continue;
                        uint32_t *dst = (uint32_t *)(dw_raw + row * (k_rounds * eff_k) + kr * eff_k); // A row = K bytes
                        const uint32_t *src = A_w + r * (eff_k / 4);
                        for (int w = 0; w < eff_k / 4; w++)
                            dst[w] = src[w];
                    }
                    release_input_window(win_a);
                    continue;
                }
#endif
                for (int r = 0; r < rows_per_round; r++) {
                    const int row = ra * rows_per_round + r;
                    uint32_t *dst = (uint32_t *)(a_pack + (row >> 3) * k_tiles * 64 + (row & 7) * 8);
                    const uint32_t *src = A_w + r * (eff_k / 4);
                    for (int kt = 0; kt < k_tiles; kt++) {
                        dst[kt * 16] = src[kt * 2];
                        dst[kt * 16 + 1] = src[kt * 2 + 1];
                    }
                }
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                const uint32_t *B_w = rb == 0 ? B_first : (const uint32_t *)acquire_input_window(win_b);
#if AIEK_MODE_BF16
                if (mode == AIE_MODE_OP_BF16 &&
                    rb == 0) /* C[32][32] f32 (acc) += A_round x B_round (aie_modes_bf16.h) */
                    aie_bf16_round((const bfloat16 *)dwb, (const bfloat16 *)B_w, (float *)acc, kr == 0 && !bfcont);
#endif
                for (int jt = 0; jt < b_strips && !dw; jt++) {
                    const int ct = rb * b_strips + jt;
                    if (ct >= n_tiles) // padding rows of the B tile: nothing to compute
                        continue;
                    for (int kt = 0; kt < k_tiles; kt++) {
                        uint32_t *dst = (uint32_t *)(b_strip + kt * 64);
                        for (int j = 0; j < 8; j++) {
                            const uint32_t *src = B_w + (jt * 8 + j) * (eff_k / 4) + kt * 2;
                            dst[j * 2] = src[0];
                            dst[j * 2 + 1] = src[1];
                        }
                        aie::store_v(b_strip + kt * 64, aie::transpose(aie::load_v<64>(b_strip + kt * 64), 8, 8));
                    }
                    for (int it = 0; it < m_tiles; it++) {
                        int32_t *acc_t = acc + (it * n_tiles + ct) * 64;
                        const int8_t *a_t = a_pack + it * k_tiles * 64;
                        aie::mmul<8, 8, 8, int8, int8> block;
                        if (wq ? (wq_ph + kr) % wq_gr == 0 : (first0 && !kcont)) { // WQ: a group's first round
                            block.mul(aie::load_v<64>(a_t), aie::load_v<64>(b_strip));
                        } else {
                            block = aie::mmul<8, 8, 8, int8, int8>(aie::load_v<64>(acc_t));
                            block.mac(aie::load_v<64>(a_t), aie::load_v<64>(b_strip));
                        }
                        for (int kt = 1; kt < k_tiles; kt++)
                            block.mac(aie::load_v<64>(a_t + kt * 64), aie::load_v<64>(b_strip + kt * 64));
                        if (!last || kmore || wq) {
                            aie::store_v(acc_t, block.to_vector<int32>());
                            continue;
                        }
                        // 8x8 int32 result, row-major: row i -> local_out row it*8+i, cols ct*8..+8
                        const auto c32 = block.to_vector<int32>();
                        for (int i = 0; i < 8; i++)
                            aie::store_v(local_out + (it * 8 + i) * real_cols + ct * 8, c32.extract<8>(i));
                    }
                }
#if AIEK_MODE_WQ
                if (wq && (wq_ph + kr) % wq_gr == wq_gr - 1) // a group ends: f32 tile += its scales x int32 partial
                    aie_wq_flush(acc, (float *)local_out, (const float *)(B_w + AIE_WQ_ROW_S0 * (eff_k / 4)),
                                 (const float *)(B_w + (AIE_WQ_ROW_S0 + 1) * (eff_k / 4)), m_tiles, n_tiles);
#endif
#if AIEK_MODE_TABLES
                for (int i = 0; i < AIE_LUT_ROWS &&
                                (mode == AIE_MODE_OP_LUT || mode == AIE_MODE_OP_STEPS || mode == AIE_MODE_OP_FVEC);
                     i++) { /* tables / FVEC aux into dwb */
                    const uint32_t *src = B_w + (AIE_LUT_ROWS * my_r + i) * (eff_k / 4);
                    uint32_t *dst = (uint32_t *)(dwb + i * (k_rounds * eff_k) + kr * eff_k);
                    for (int w = 0; w < eff_k / 4; w++)
                        dst[w] = src[w];
                }
#endif
#if AIEK_MODE_DWQ
                for (int q = 0; q < dw_nb; q++) // multi-item depthwise: this core's items in B
                    for (int i = 0; i < 8; i++) {
                        const uint32_t *src = B_w + (16 * my_r + 8 * q + i) * (eff_k / 4);
                        uint32_t *dst = (uint32_t *)(dwb + q * 4096 + i * (k_rounds * eff_k) + kr * eff_k);
                        for (int w = 0; w < eff_k / 4; w++)
                            dst[w] = src[w];
                    }
#endif
                release_input_window(win_b);
            }
        }

        if (!dw) {
        }
#if AIEK_MODE_POOL
        else if (mode == AIE_MODE_OP_POOL) {
            if (!aie_pool_vec(dw_raw, local_out, mdesc)) /* vector path first (MAX / SUM, stride 1 / 2) */
                AIE_MODE_POOL(dw_raw, local_out, mdesc);
        }
#endif
#if AIEK_MODE_FVEC
        else if (mode == AIE_MODE_OP_FVEC)
            aie_fvec_apply((const float *)dw_raw, (float *)local_out, mdesc, (const float *)dwb);
#endif
#if AIEK_MODE_BF16
        else if (mode == AIE_MODE_OP_BF16) {
            if (!bfmore) /* the f32 tile, row-major [32][32] = the core's 4 KB C tile */
                aie_bf16_out((const float *)acc, (float *)local_out);
        }
#endif
#if AIEK_MODE_ELTWISE
        else if (mode == AIE_MODE_OP_ELTWISE)
            AIE_MODE_ELTWISE(dw_raw, local_out, mdesc);
#endif
#if AIEK_MODE_LUT
        else if (mode == AIE_MODE_OP_LUT)
            aie_lut_apply((const uint8_t *)dw_raw, (uint8_t *)local_out, mdesc, (const uint8_t *)dwb);
#endif
#if AIEK_MODE_STEPS
        else if (mode == AIE_MODE_OP_STEPS)
            aie_steps_apply((const uint8_t *)dw_raw, (uint8_t *)local_out, mdesc, (const uint8_t *)dwb);
#endif
#if AIEK_MODE_MMT
        else if (mode == AIE_MODE_OP_MMT) { /* local_out was written round by round */
        }
#endif
#if AIEK_MODE_FPOOL
        else if (mode == AIE_MODE_OP_FPOOL) /* the item read as floats, the C tile written as floats */
            aie_fpool_item((const float *)dw_raw, (float *)local_out, mdesc);
#endif
#if AIEK_MODE_SDPA
        else if (mode == AIE_MODE_OP_SDPA) /* scores scratch: dwb (unused in this mode) */
            aie_sdpa_item(dw_raw, local_out, mdesc, (int32_t *)dwb);
#endif
#if AIEK_MODE_DW
        else { // int32 accumulators of the item, row-major [dw_rout][dw_wo], the rest zero
#if AIEK_MODE_DWQ
            if (dwq) { // int8 out: item q's band at bytes [q * 1024, +n_out) of the core's C tile
                // an 8-row mesh puts row 7's second item in B rows 120..127, whose last row starts with the
                // descriptor: the host moved those 12 input bytes to the item's unused tail (DWQ_STASH)
                for (int q = 0; q < dw_nb; q++)
                    if (16 * my_r + 8 * q + 7 == 127)
                        for (int j = 0; j < DWQ_STASH_N; j++)
                            dwb[q * 4096 + 3584 + j] = dwb[q * 4096 + DWQ_STASH + j];
                for (int iq = 0; iq <= dw_nb; iq++) // 32 output columns per step (aie_modes_dwq.h)
                    aie_dwq_item(iq == 0 ? dw_raw : dwb + (iq - 1) * 4096, dw_s, dw_kh, dw_kw, dw_wp, dw_rout, dw_wo,
                                 (int8_t *)local_out + iq * 1024);
            } else
#endif
                // 0xD7 stays scalar: its vector form cost ~0.9 KB of program memory for a rarely used path
                for (int i = 0; i < tile_rows * real_cols; i++) {
                    int32_t a = 0;
                    if (i < dw_rout * dw_wo) {
                        const int oh = i / dw_wo, ow = i - oh * dw_wo;
                        const int8_t *in = dw_raw + oh * dw_s * dw_wp + ow * dw_s, *wt = dw_raw + 4096 - 64;
                        for (int kh = 0; kh < dw_kh; kh++)
                            for (int kw = 0; kw < dw_kw; kw++)
                                a += (int32_t)wt[kh * dw_kw + kw] * (int32_t)in[kh * dw_wp + kw];
                    }
                    local_out[i] = a;
                }
        }
#endif

#if AIEK_MODE_GEMMQ
        if (gq) // the int32 tile -> int8 in place, in vector lanes (aie_modes_gemmq.h)
            aie_gq_tile((int32_t *)local_out, tile_rows, qrow, gres ? rqrow : nullptr, gres ? rtile : nullptr);
#endif
        if (AIEDNN_KCYC && gq) // C row 0 bytes 64..67: unused by the int8 C
            ((uint32_t *)local_out)[16] = (uint32_t)(KC_TIC() - kc_t0);
        for (int rc = 0; rc < num_c_rounds; rc++) { // buf_sz_c is in bytes (int8 window)
            uint32_t *out = (uint32_t *)acquire_output_window(win_c);
            const uint32_t *src = (const uint32_t *)local_out + rc * (buf_sz_c / 4);
            for (int w = 0; w < buf_sz_c / 4; w++)
                out[w] = src[w];
            release_output_window(win_c);
        }
    }
}

// Mesh: function-local, same function as the launch (aiehlc recovers it from
// there); buffers: process-lifetime device.alloc statics (see matmul_uk.cc).
static int8_t *g_gemm_a = nullptr, *g_gemm_b = nullptr, *g_gemm_c = nullptr;

// phase split of the entry, reported by aiednn_print_stats() through aiednn_kernel_stats()
static uint64_t g_gemm_t[3], g_gemm_calls;
static unsigned long long g_gemm_ph0[5]; // PMU phase baseline (--profiling), see aiednn_kernel_stats
static inline uint64_t gemm_ticks(void) {
#if defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;
#endif
}
static uint64_t g_kcyc[3]; // AIEDNN_KCYC: core (0, 0)'s cycles over GEMMQ launches, of which input waits; launches
extern "C" void aiednn_kernel_stats(void) {
#if defined(__aarch64__)
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    const double ms = 1e3 / (double)f;
    printf("aiednn_kernel_stats: gemm_s8s32 calls=%llu copy_in_ms=%.2f launch_ms=%.2f copy_out_ms=%.2f\n",
           (unsigned long long)g_gemm_calls, g_gemm_t[0] * ms, g_gemm_t[1] * ms, g_gemm_t[2] * ms);
    if (AIEDNN_KCYC && g_kcyc[2]) // AIE clock ~1 GHz: cycles ~ ns
        printf("aiednn_kernel_stats: core (0,0) per GEMMQ launch: %.0f cycles first window -> output (%llu launches)\n",
               (double)g_kcyc[0] / g_kcyc[2], (unsigned long long)g_kcyc[2]);
#if defined(AIEHLC_PROFILING) && AIEHLC_PROFILING
    // PMU phase shares (aiehlc.sh --profiling); PMU cycles are A72 clock, so only ratios compare to ms
    unsigned long long ph[4] = {0, 0, 0, 0}, wio = 0;
    unsigned int phc[4] = {0, 0, 0, 0}, wioc = 0;
    __Runtime_phase_cycles(ph, phc);
    __Runtime_wait_io_cycles(&wio, &wioc);
    unsigned long long d[5];
    for (int i = 0; i < 4; i++)
        d[i] = ph[i] - g_gemm_ph0[i];
    d[4] = wio - g_gemm_ph0[4];
    const double tot = (double)(d[0] + d[1] + d[2] + d[3] + d[4]) + 1e-9;
    printf("aiednn_kernel_stats: launch phases (PMU share) kload=%.1f%% bdcfg=%.1f%% coreen=%.1f%% startio=%.1f%% "
           "wait_io=%.1f%% (wait_io calls=%u, %.0f Mcyc total)\n",
           100 * d[0] / tot, 100 * d[1] / tot, 100 * d[2] / tot, 100 * d[3] / tot, 100 * d[4] / tot, wioc, tot / 1e6);
#endif
#endif
}
extern "C" void aiednn_kernel_stats_reset(void) {
    g_gemm_t[0] = g_gemm_t[1] = g_gemm_t[2] = 0, g_gemm_calls = 0;
    memset(g_kcyc, 0, sizeof g_kcyc);
#if defined(__aarch64__) && defined(AIEHLC_PROFILING) && AIEHLC_PROFILING
    unsigned int c[4];
    __Runtime_phase_cycles(g_gemm_ph0, c);
    __Runtime_wait_io_cycles(&g_gemm_ph0[4], c);
#endif
}

/* Memtile-resident B (#pragma memtile_staging(weights)): aiednn_gemm_weights_key names the next launch's
 * B block (0 = not cacheable). On a memtile hit only the per-launch tail rows 96..127 are written and
 * synced; on a miss the whole block is copied. */
static uint64_t g_wkey_next, g_wkey_last;
extern "C" void aiednn_gemm_weights_key(uint64_t key) { g_wkey_next = key; }

/* B's block rows into the launch buffer unless the mem tiles hold this block; returns "hit" */
static int b_rows_in(const int8_t *b) {
    const int real = GEMM_N / HW_COLS, rows_nb = N / HW_COLS;
    const uint64_t key = g_wkey_next;
    g_wkey_next = 0;
    int hit = 0;
#if defined(AIEHLC_ASYNC_LAUNCH)
    /* the runtime's LRU decides; a miss streams the whole block from DDR */
    hit = key && __Runtime_memtile_weights_cached(key);
    __Runtime_memtile_weights_key(key, 32);
    __Runtime_memtile_weights_head(real); /* a GEMM core skips rows real .. 95: a miss moves real + tail rows */
#endif
    g_wkey_last = key;
    if (!hit)
        for (int j = 0; j < HW_COLS; j++)
            memcpy(g_gemm_b + j * rows_nb * K, b + j * real * K, real * K);
    return hit;
}

/* after the tail rows are written: on a hit, write back only them and skip the 256 KB B sync */
static void b_tail_done(int hit) {
#if defined(AIEHLC_ASYNC_LAUNCH)
    if (!hit)
        return;
    const int rows_nb = N / HW_COLS;
    XAie_DevInst *dev = __Runtime_get_partition_dev(0);
    for (int j = 0; j < HW_COLS; j++)
        __Runtime_sync_for_dev(dev, g_gemm_b + (size_t)(j * rows_nb + 96) * K, (size_t)32 * K);
    __Runtime_sync_skip_next(g_gemm_b);
#else
    (void)hit;
#endif
}

/* a launch whose B carries items, not a weights block (dw / pool / eltwise modes) */
static void b_not_cached(void) {
    g_wkey_next = 0, g_wkey_last = 0;
#if defined(AIEHLC_ASYNC_LAUNCH)
    __Runtime_memtile_weights_key(0, 0);
#endif
}

static int gemm_s8s32_run(const int8_t *a, const int8_t *b, int32_t *c) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (!g_gemm_a) {
        g_gemm_a = (int8_t *)device.alloc(M * K * 4);
        g_gemm_b = (int8_t *)device.alloc(N * K * 4);
        g_gemm_c = (int8_t *)device.alloc(M * N * 4);
        if (!g_gemm_a || !g_gemm_b || !g_gemm_c)
            return AIEK_ERR_ALLOC;
        memset(g_gemm_b, 0, N * K); // the padding rows stay zero for the process lifetime
    }
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, a, M * K);
    b_tail_done(b_rows_in(b)); // 32 real of 128 B rows per core column; nothing in the tail
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    memcpy(c, g_gemm_c, GEMM_M * GEMM_N * sizeof(int32_t)); // C bytes = int32 [GEMM_M][GEMM_N]
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

/* One K block of a chained GEMM (opcodes 0 / 0xA0): flags bit 0 = start from the core's accumulators,
 * bit 1 = more blocks follow (C is not copied). Needs the persistent kernel. */
static int gemm_bufs(aieArray &device) { // the launch buffers, allocated once for the process
    if (!g_gemm_a) {
        g_gemm_a = (int8_t *)device.alloc(M * K * 4);
        g_gemm_b = (int8_t *)device.alloc(N * K * 4);
        g_gemm_c = (int8_t *)device.alloc(M * N * 4);
        if (!g_gemm_a || !g_gemm_b || !g_gemm_c)
            return AIEK_ERR_ALLOC;
        memset(g_gemm_b, 0, N * K);
    }
    return AIEK_OK;
}

/* direct conv (aie_modes_conv.h): mesh row r's geometry (16 int32 at geom + 16 r) into the last 64
 * bytes of its A slice (A is already in the launch buffer), CONV_MAGIC into word 0 of B row
 * CONV_GEOM_ROW0 of every column; geom NULL clears the magic (zero again for a GEMM launch) */
static void conv_geom_in(const int32_t *geom) {
    const int rows_nb = N / HW_COLS;
    if (geom)
        for (int r = 0; r < HW_ROWS; r++)
            memcpy(g_gemm_a + (size_t)r * 32 * K + CONV_REGION_DATA, geom + 16 * r, 64);
    for (int j = 0; j < HW_COLS; j++) {
        const int32_t magic = geom ? CONV_MAGIC : 0;
        memcpy(g_gemm_b + (size_t)(j * rows_nb + CONV_GEOM_ROW0) * K, &magic, 4);
    }
}

static void gemm_kc_prep(const int8_t *a, const int8_t *b, int32_t flags, const int32_t *geom = nullptr) {
    memcpy(g_gemm_a, a, M * K);
    const int rows_nb = N / HW_COLS, hit = b_rows_in(b);
    for (int j = 0; j < HW_COLS; j++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (j * rows_nb + rows_nb - 1) * K;
        d[0] = (flags & 2) ? 0xA0 : 0;
        d[1] = (uint8_t)j;
        d[2] = (uint8_t)(flags & 1);
    }
    if (geom)
        conv_geom_in(geom);
    b_tail_done(hit);
}

static void gemm_kc_post(int32_t flags, int32_t *c, int conv = 0) {
    const int rows_nb = N / HW_COLS;
    if (conv)
        conv_geom_in(nullptr);
    for (int j = 0; j < HW_COLS; j++)
        memset(g_gemm_b + (j * rows_nb + rows_nb - 1) * K, 0, 16);
    if (!(flags & 2) && !(flags & 4)) // bit 2: the caller reads C from aiednn_gemm_stage_c() itself
        memcpy(c, g_gemm_c, GEMM_M * GEMM_N * sizeof(int32_t));
}

/* the launch buffer of C (int32 [GEMM_M][GEMM_N], synced for the CPU after every launch): with
 * flags bit 2 the runtime copies the rows it needs straight from it, skipping a 64 KB copy */
extern "C" const int32_t *aiednn_gemm_stage_c(void) { return (const int32_t *)g_gemm_c; }

static int gemm_kc_run(const int8_t *a, const int8_t *b, int32_t flags, int32_t *c) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    gemm_kc_prep(a, b, flags);
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    gemm_kc_post(flags, c);
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(gemm_kc_s8s32) {
    if (num_args != 4)
        return AIEK_ERR_ARGS;
    return gemm_kc_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                       *AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], int32_t));
}

// clang-format off
AIEK_SPEC(gemm_kc_s8s32,
    "kernel    gemm_kc_s8s32\n"
    "op        gemm_kc\n"
    "match     dense\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "mesh      src[0]=32*rows dst[0]=32*rows\n"
    "in    int8  [128,512]  src\n"
    "param int8  [128,512]  weight\n"
    "param int32 [1]        flags\n"
    "out   int32 [128,128]  dst\n");
// clang-format on

/* bf16 GEMM (aie_modes_bf16.h, opcode AIE_MODE_OP_BF16): A bf16 [GEMM_M][256], W bf16 [GEMM_N][256]
 * (channels x K) -> C f32 [GEMM_M][GEMM_N], f32 accumulation on the core. flags bit 0: continue the
 * previous launch's tile (K chained), bit 1: more K follows (C is not written). The host pre-tiles each
 * mesh row's A and each column's W into the launch buffers' first 32 rows of every window. */
static int bf16_blk_run(const uint16_t *a, const uint16_t *w, int32_t flags, float *c) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    for (int r = 0; r < HW_ROWS; r++)
        aie_bf16_pack_a(a + (size_t)r * 32 * AIE_BF16_K, 32, AIE_BF16_K, 0, AIE_BF16_K,
                        (uint8_t *)g_gemm_a + (size_t)r * 32 * K);
    for (int j = 0; j < HW_COLS; j++) {
        uint8_t *blk = (uint8_t *)g_gemm_b + (size_t)j * rows_nb * K;
        aie_bf16_pack_b(w + (size_t)j * 32 * AIE_BF16_K, 32, AIE_BF16_K, 0, AIE_BF16_K, blk);
        uint8_t *d = blk + (size_t)(rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_BF16;
        d[1] = (uint8_t)j;
        d[2] = (uint8_t)(flags & 3);
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int j = 0; j < HW_COLS; j++)
        memset(g_gemm_b + (j * rows_nb + rows_nb - 1) * K, 0, 16);
    if (!(flags & 2)) /* core (r, c)'s f32 tile: rows r*32.., floats c*32.. of the [GEMM_M][GEMM_N] C */
        memcpy(c, g_gemm_c, GEMM_M * GEMM_N * sizeof(float));
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(bf16_blk_f32) {
    if (num_args != 4)
        return AIEK_ERR_ARGS;
    return bf16_blk_run(AIEK_PTR(&args[0], const uint16_t), AIEK_PTR(&args[1], const uint16_t),
                        *AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], float));
}

// clang-format off
AIEK_SPEC(bf16_blk_f32,
    "kernel    bf16_f32\n"
    "op        bf16_gemm\n"
    "match     dense\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     BF16\n"
    "mesh      src[0]=32*rows dst[0]=32*rows\n"
    "in    int8    [128,512]  src\n"
    "param int8    [128,512]  weight\n"
    "param int32   [1]        flags\n"
    "out   float32 [128,128]  dst\n");
// clang-format on

/* grouped weight-only quant (aie_modes_wq.h, opcode AIE_MODE_OP_WQ): A int8 [GEMM_M][512], W int8
 * [GEMM_N][512] (channels x K, as the plain GEMM), scales f32 [8][GEMM_N] (row kr: the scales of the group
 * that ends at round kr; other rows unread), ctl int32 {flags, gr, phase} (flags bit 0 continue, bit 1 more
 * K follows; gr = group size / 64; phase = rounds of the current group summed by earlier launches) -> C f32
 * [GEMM_M][GEMM_N] = sum over groups of scale x the group's int32 partial. */
static int wq_blk_run(const int8_t *a, const int8_t *w, const float *sc, const int32_t *ctl, float *c) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS, flags = ctl[0], gr = ctl[1] < 1 ? 1 : ctl[1], ph = ctl[2];
    if (gr > 255)
        return AIEK_ERR_PARAM;
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, a, M * K);
    const int hit = b_rows_in(w);
    for (int j = 0; j < HW_COLS; j++) {
        uint8_t *blk = (uint8_t *)g_gemm_b + (size_t)j * rows_nb * K;
        for (int kr = 0; kr < 8; kr++)
            if ((ph + kr) % gr == gr - 1)
                aie_wq_put_scales(blk, kr, sc + kr * GEMM_N + j * 32);
        uint8_t *d = blk + (size_t)(rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_WQ;
        d[1] = (uint8_t)j;
        d[2] = (uint8_t)(flags & 3);
        d[3] = (uint8_t)gr;
        d[4] = (uint8_t)(ph % gr);
    }
    b_tail_done(hit);
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int j = 0; j < HW_COLS; j++)
        memset(g_gemm_b + (j * rows_nb + rows_nb - 1) * K, 0, 16);
    if (!(flags & 2)) /* core (r, c)'s f32 tile: rows r*32.., floats c*32.. of the [GEMM_M][GEMM_N] C */
        memcpy(c, g_gemm_c, GEMM_M * GEMM_N * sizeof(float));
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(wq_blk_s8f32) {
    if (num_args != 5)
        return AIEK_ERR_ARGS;
    return wq_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                      AIEK_PTR(&args[2], const float), AIEK_PTR(&args[3], const int32_t), AIEK_PTR(&args[4], float));
}

// clang-format off
AIEK_SPEC(wq_blk_s8f32,
    "kernel    wq_s8f32\n"
    "op        wq_gemm\n"
    "match     dense\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     WQ\n"
    "mesh      src[0]=32*rows dst[0]=32*rows\n"
    "in    int8    [128,512]  src\n"
    "param int8    [128,512]  weight\n"
    "param float32 [8,128]    scales\n"
    "param int32   [3]        ctl\n"
    "out   float32 [128,128]  dst\n");
// clang-format on

/* GEMM + integer requant (opcode 0xF1): the core writes int8 C. params int32 [516] =
 * {zp, lo, hi, 0, B int64[GEMM_N], M[GEMM_N], sh[GEMM_N]} (aiednn_rt_internal.h rt_fx_q:
 * q = (acc * M + B) >> sh + zp); in row 127: zp/lo/hi at +4, B at +16, M at +272, sh at +400; column c's
 * 32 channels go to row 127 of its B block, which a GEMM never reads (padding rows). */
/* qp int32 [6 + 6 * GEMM_N]: ... sh[GEMM_N], then {has_res, rzp, R int64[GEMM_N]}; res int8
 * [GEMM_M][GEMM_N] (rows = A rows, cols = channels): with has_res the core adds (res - rzp) * R
 * (a fused int8 residual add, aiednn_rt_internal.h rt_epi_put) */
#define GQ_RES_OFF(n) (4 + 4 * (n)) /* has_res, rzp, then R int64[n] */
static void gemm_q_prep(const int8_t *a, const int8_t *b, const int32_t *qp, const int8_t *res,
                        const int32_t *geom = nullptr) {
    memcpy(g_gemm_a, a, M * K);
    const int real = GEMM_N / HW_COLS, rows_nb = N / HW_COLS, hit = b_rows_in(b);
    for (int j = 0; j < HW_COLS; j++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (j * rows_nb + rows_nb - 1) * K;
        int32_t *w = (int32_t *)(d + 4);
        d[0] = 0xF1;
        d[1] = (uint8_t)j;
        d[2] = (uint8_t)(qp[3] & 1); // K chain: continue from the accumulators (gemm_kc_run)
        const int has_res = qp[GQ_RES_OFF(GEMM_N)] & 1;
        d[3] = (uint8_t)has_res;
        if (has_res) { // row 126: R int64[32] + rzp; rows 96 + 2r .. 97 + 2r: core (r, j)'s 32x32 tile
            uint8_t *rq = d - K;
            memcpy(rq, qp + GQ_RES_OFF(GEMM_N) + 2 + 2 * j * real, (size_t)real * 8);
            memcpy(rq + 256, qp + GQ_RES_OFF(GEMM_N) + 1, 4);
            for (int rr = 0; rr < HW_ROWS; rr++)
                for (int mm = 0; mm < 32; mm++)
                    memcpy(g_gemm_b + (size_t)(j * rows_nb + 96 + 2 * rr + mm / 16) * K + (mm % 16) * 32,
                           res + (size_t)(rr * 32 + mm) * GEMM_N + j * real, (size_t)real);
        }
        w[0] = qp[0], w[1] = qp[1], w[2] = qp[2];
        int64_t *bb = (int64_t *)(d + 16);
        int32_t *m = (int32_t *)(d + 272);
        int8_t *sh = (int8_t *)(d + 400);
        for (int i = 0; i < real; i++) {
            memcpy(&bb[i], qp + 4 + 2 * (j * real + i), 8);
            m[i] = qp[4 + 2 * GEMM_N + j * real + i];
            sh[i] = (int8_t)qp[4 + 3 * GEMM_N + j * real + i];
        }
    }
    if (geom)
        conv_geom_in(geom);
    b_tail_done(hit);
}

static void gemm_q_post(int8_t *c, int conv = 0) {
    const int real = GEMM_N / HW_COLS, rows_nb = N / HW_COLS;
    if (AIEDNN_KCYC) {
        g_kcyc[0] += ((const uint32_t *)g_gemm_c)[16];
        g_kcyc[2]++;
    }
    if (conv)
        conv_geom_in(nullptr);
    for (int j = 0; j < HW_COLS; j++) // the padding row is zero again for a plain GEMM
        memset(g_gemm_b + (j * rows_nb + rows_nb - 1) * K, 0, 512);
    for (int r = 0; r < GEMM_M; r++) // C row r: core column j's int8 at bytes [j*128, +32)
        for (int j = 0; j < HW_COLS; j++)
            memcpy(c + r * GEMM_N + j * real, g_gemm_c + (size_t)r * GEMM_N * 4 + j * 128, real);
}

static int gemm_q_run(const int8_t *a, const int8_t *b, const int32_t *qp, const int8_t *res, int8_t *c) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    gemm_q_prep(a, b, qp, res);
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    gemm_q_post(c);
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

#if defined(AIEHLC_ASYNC_LAUNCH)
/* The persistent build's split launch (persist_transform.py): aiednn_gemm_async_start() fills the
 * launch buffers and starts the array, the host is free until aiednn_gemm_async_finish(), which
 * waits and copies C out. One launch in flight (the shim BDs are re-armed per launch). which: 0
 * = gemm_kc_s8s32 args (A, B, flags, C), 1 = gemm_q_s8s8 args (A, B, qparams, residual, C). */
static int g_async_which = -1;
static int32_t g_async_flags;
static void *g_async_c;
static uint64_t g_async_t0;

extern "C" int aiednn_gemm_async_start(const aiek_tensor *args, int num_args, int which) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    static const int nargs[4] = {4, 5, 5, 6}; // kc, q, conv kc, conv q
    if (g_async_which >= 0 || which < 0 || which > 3 || num_args != nargs[which] || gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ARGS;
    g_async_t0 = gemm_ticks();
    if (which & 1) { // q: A, B, qparams, residual, [geom], C
        gemm_q_prep(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                    AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], const int8_t),
                    which == 3 ? AIEK_PTR(&args[4], const int32_t) : nullptr);
        g_async_c = AIEK_PTR(&args[num_args - 1], int8_t);
    } else { // kc: A, B, flags, [geom], C
        g_async_flags = *AIEK_PTR(&args[2], const int32_t);
        gemm_kc_prep(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t), g_async_flags,
                     which == 2 ? AIEK_PTR(&args[3], const int32_t) : nullptr);
        g_async_c = AIEK_PTR(&args[num_args - 1], int32_t);
    }
    __aie_launch_start("gemm_s8s32", mesh, g_gemm_a, (size_t)M * K, g_gemm_b, (size_t)N * K, g_gemm_c, (size_t)M * N);
    g_async_which = which;
    return AIEK_OK;
}

extern "C" int aiednn_gemm_async_finish(void) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (g_async_which < 0)
        return AIEK_ERR_ARGS;
    __aie_launch_finish("gemm_s8s32", mesh, g_gemm_a, (size_t)M * K, g_gemm_b, (size_t)N * K, g_gemm_c, (size_t)M * N);
    const uint64_t t2 = gemm_ticks();
    if (g_async_which & 1)
        gemm_q_post((int8_t *)g_async_c, g_async_which == 3);
    else
        gemm_kc_post(g_async_flags, (int32_t *)g_async_c, g_async_which == 2);
    g_gemm_t[1] += t2 - g_async_t0, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    g_async_which = -1;
    return AIEK_OK;
}
#endif

AIEK_ENTRY(gemm_q_s8s8) {
    if (num_args != 5)
        return AIEK_ERR_ARGS;
    return gemm_q_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                      AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], const int8_t), AIEK_PTR(&args[4], int8_t));
}

// clang-format off
AIEK_SPEC(gemm_q_s8s8,
    "kernel    gemm_q_s8s8\n"
    "op        gemm_q\n"
    "match     dense\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     GEMMQ\n"
    "mesh      src[0]=32*rows residual[0]=32*rows dst[0]=32*rows\n"
    "in    int8  [128,512]  src\n"
    "param int8  [128,512]  weight\n"
    "param int32 [774]      qparams\n"
    "param int8  [128,128]  residual\n"
    "out   int8  [128,128]  dst\n");
// clang-format on

/* Direct conv (aie_modes_conv.h): the same launches as gemm_kc / gemm_q, with A a raw input region
 * per mesh row and geom int32 [16 * rows] its geometry (CONV_GEOM_*) */
static int conv_kc_run(const int8_t *a, const int8_t *b, int32_t flags, const int32_t *geom, int32_t *c) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    gemm_kc_prep(a, b, flags, geom);
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    gemm_kc_post(flags, c, 1);
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

static int conv_q_run(const int8_t *a, const int8_t *b, const int32_t *qp, const int8_t *res, const int32_t *geom,
                      int8_t *c) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    gemm_q_prep(a, b, qp, res, geom);
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    gemm_q_post(c, 1);
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(conv_kc_s8s32) {
    if (num_args != 5)
        return AIEK_ERR_ARGS;
    return conv_kc_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                       *AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], const int32_t),
                       AIEK_PTR(&args[4], int32_t));
}

// clang-format off
AIEK_SPEC(conv_kc_s8s32,
    "kernel    conv_kc_s8s32\n"
    "op        gemm_kc\n"
    "match     dense\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     CONV\n"
    "mesh      src[0]=32*rows geom[0]=16*rows dst[0]=32*rows\n"
    "in    int8  [128,512]  src\n"
    "param int8  [128,512]  weight\n"
    "param int32 [1]        flags\n"
    "param int32 [64]       geom\n"
    "out   int32 [128,128]  dst\n");
// clang-format on

AIEK_ENTRY(conv_q_s8s8) {
    if (num_args != 6)
        return AIEK_ERR_ARGS;
    return conv_q_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                      AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], const int8_t),
                      AIEK_PTR(&args[4], const int32_t), AIEK_PTR(&args[5], int8_t));
}

// clang-format off
AIEK_SPEC(conv_q_s8s8,
    "kernel    conv_q_s8s8\n"
    "op        gemm_q\n"
    "match     dense\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     GEMMQ CONV\n"
    "mesh      src[0]=32*rows residual[0]=32*rows geom[0]=16*rows dst[0]=32*rows\n"
    "in    int8  [128,512]  src\n"
    "param int8  [128,512]  weight\n"
    "param int32 [774]      qparams\n"
    "param int8  [128,128]  residual\n"
    "param int32 [64]       geom\n"
    "out   int8  [128,128]  dst\n");
// clang-format on

AIEK_ENTRY(gemm_blk_s8s32) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return gemm_s8s32_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                          AIEK_PTR(&args[2], int32_t));
}

/* Depthwise overlay on the same launch (same ELF, routing, DMA): DW_ITEMS items, one per
 * core, item i = r*HW_COLS+c on core (row r, column c). An item is 4096 bytes of A: the
 * zero-padded input rows of one output band of one channel, row pitch geom[3] (Wp),
 * and its KHxKW weights in the last DW_WBYTES. The descriptor rides in row 127 of
 * each column's B block (zero for a GEMM, and zeroed again after the launch).
 * Output: int32 [16][1024], item i's band row-major [Rout][Wo]. */
static int dw_blk_run(const int8_t *items, const int32_t *geom, int32_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (!g_gemm_a) {
        g_gemm_a = (int8_t *)device.alloc(M * K * 4);
        g_gemm_b = (int8_t *)device.alloc(N * K * 4);
        g_gemm_c = (int8_t *)device.alloc(M * N * 4);
        if (!g_gemm_a || !g_gemm_b || !g_gemm_c)
            return AIEK_ERR_ALLOC;
        memset(g_gemm_b, 0, N * K);
    }
    const int rows_nb = N / HW_COLS; // B rows per core column
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES); // = M * K: item i at A rows i*8 .. i*8+7
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        const int v[] = {geom[6] ? 0xD8 : 0xD7, c, geom[0], geom[1], geom[2]};
        for (int i = 0; i < 5; i++)
            d[i] = (uint8_t)v[i];
        for (int i = 0; i < 3; i++) { // Wp, Rout, Wo as little-endian 16-bit
            d[5 + 2 * i] = (uint8_t)(geom[3 + i] & 0xff);
            d[6 + 2 * i] = (uint8_t)(geom[3 + i] >> 8);
        }
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++) // a GEMM must see zero padding rows again
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 16);
    const int32_t *cm = (const int32_t *)g_gemm_c; // int32 [GEMM_M][GEMM_N]: core (r,c) tile at rows r*32, cols c*32
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++) {
            if (geom[6]) { // requantized on the core: int8 item k at bytes [k*DW_OUT, +DW_OUT) of out
                int8_t *o8 = (int8_t *)out + (r * HW_COLS + c) * DW_OUT;
                for (int i = 0; i < DW_OUT; i++)
                    o8[i] = ((const int8_t *)g_gemm_c)[(size_t)(r * 32 + i / 128) * GEMM_N * 4 + c * 128 + i % 128];
                continue;
            }
            for (int i = 0; i < DW_OUT; i++)
                out[(r * HW_COLS + c) * DW_OUT + i] = cm[(r * 32 + i / 32) * GEMM_N + c * 32 + i % 32];
        }
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

/* Depthwise + integer requant with 3 items per core (0xD8, d[11] = 2): item r*HW_COLS+c in A as
 * in dw_blk_run, items DW_ITEMS + 2*(r*HW_COLS+c) + q (q = 0, 1) in rows 16r + 8q .. +7 of column
 * c's B block (no weights in this mode). Output int8 [3*DW_ITEMS][DW_OUT]. */
static int dwq_blk_run(const int8_t *items, const int32_t *geom, int8_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES);
    for (int c = 0; c < HW_COLS; c++) {
        for (int r = 0; r < HW_ROWS; r++)
            for (int q = 0; q < 2; q++) {
                const int8_t *it = items + (size_t)(DW_ITEMS + 2 * (r * HW_COLS + c) + q) * DW_ITEM_BYTES;
                int8_t *slot = g_gemm_b + (size_t)(c * rows_nb + 16 * r + 8 * q) * K;
                memcpy(slot, it, DW_ITEM_BYTES);
                if (16 * r + 8 * q + 7 == rows_nb - 1) // last row = the descriptor row: stash its first bytes
                    memcpy(slot + DWQ_STASH, it + 3584, DWQ_STASH_N);
            }
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        const int v[] = {0xD8, c, geom[0], geom[1], geom[2]};
        for (int i = 0; i < 5; i++)
            d[i] = (uint8_t)v[i];
        for (int i = 0; i < 3; i++) {
            d[5 + 2 * i] = (uint8_t)(geom[3 + i] & 0xff);
            d[6 + 2 * i] = (uint8_t)(geom[3 + i] >> 8);
        }
        d[11] = 2;
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++)
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 16);
    const int8_t *cb = (const int8_t *)g_gemm_c; // core (r, c) tile: rows r*32.., bytes c*128.. of 512-byte C rows
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++)
            for (int q = 0; q < 3; q++) {
                const int item = q == 0 ? r * HW_COLS + c : DW_ITEMS + 2 * (r * HW_COLS + c) + q - 1;
                int8_t *o8 = out + (size_t)item * DW_OUT;
                for (int i = 0; i < DW_OUT; i++) {
                    const int b = q * 1024 + i;
                    o8[i] = cb[(size_t)(r * 32 + b / 128) * GEMM_N * 4 + c * 128 + b % 128];
                }
            }
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(dwq_blk_s8s8) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return dwq_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int32_t), AIEK_PTR(&args[2], int8_t));
}

// clang-format off
AIEK_SPEC(dwq_blk_s8s8,
    "kernel    dwq_s8s8\n"
    "op        conv2d\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     DW DWQ\n"
    "mesh      src[0]=3*cores dst[0]=3*cores\n"
    "in    int8  [48,4096]  src\n"
    "param int32 [8]        weight\n"
    "out   int8  [48,1024]  dst\n");
// clang-format on

/* Pool overlay: same DW_ITEMS items and launch as dw_blk_run, opcode AIE_MODE_OP_POOL, descriptor =
 * geom {op, signed, s, KH, KW, Wp, Rout, Wo} (aie_modes_pool.h). Output: exact int32 window
 * max / sum per item; the host finishes (aiednn_rt_pool.h). */
static int pool_blk_run(const int8_t *items, const int32_t *geom, int32_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (!g_gemm_a) {
        g_gemm_a = (int8_t *)device.alloc(M * K * 4);
        g_gemm_b = (int8_t *)device.alloc(N * K * 4);
        g_gemm_c = (int8_t *)device.alloc(M * N * 4);
        if (!g_gemm_a || !g_gemm_b || !g_gemm_c)
            return AIEK_ERR_ALLOC;
        memset(g_gemm_b, 0, N * K);
    }
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES);
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_POOL;
        d[1] = (uint8_t)c;
        AIE_MODE_POOL_DESC(d + 2, geom);
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++)
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 16);
    const int32_t *cm = (const int32_t *)g_gemm_c;
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++)
            for (int i = 0; i < DW_OUT; i++)
                out[(r * HW_COLS + c) * DW_OUT + i] = cm[(r * 32 + i / 32) * GEMM_N + c * 32 + i % 32];
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(pool_blk_s8s32) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return pool_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int32_t),
                        AIEK_PTR(&args[2], int32_t));
}

// clang-format off
AIEK_SPEC(pool_blk_s8s32,
    "kernel    pool_s8s32\n"
    "op        pool\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     POOL\n"
    "mesh      src[0]=cores dst[0]=cores\n"
    "in    int8  [16,4096]  src\n"
    "param int32 [8]        weight\n"
    "out   int32 [16,1024]  dst\n");
// clang-format on

/* f32 pooling (aie_modes_fpool.h, opcode AIE_MODE_OP_FPOOL): DW_ITEMS items as pool_blk_run, each 1024
 * floats of padded input slices; geom {op, kd, kh, kw, sh | sw << 8, wo, wp, rout}; out per item 1024 f32
 * (max, or the sum the host divides). */
static int fpool_blk_run(const int8_t *items, const int32_t *geom, float *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES);
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_FPOOL;
        d[1] = (uint8_t)c;
        AIE_FPOOL_DESC(d + 2, geom);
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++)
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 16);
    const float *cm = (const float *)g_gemm_c;
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++)
            for (int i = 0; i < DW_OUT; i++)
                out[(r * HW_COLS + c) * DW_OUT + i] = cm[(r * 32 + i / 32) * GEMM_N + c * 32 + i % 32];
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(fpool_blk_f32) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return fpool_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int32_t),
                         AIEK_PTR(&args[2], float));
}

// clang-format off
AIEK_SPEC(fpool_blk_f32,
    "kernel    fpool_f32\n"
    "op        pool\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     FPOOL\n"
    "mesh      src[0]=cores dst[0]=cores\n"
    "in    int8    [16,4096]  src\n"
    "param int32   [8]        weight\n"
    "out   float32 [16,1024]  dst\n");
// clang-format on

/* SDPA overlay (aie_modes_sdpa.h, opcode AIE_MODE_OP_SDPA): DW_ITEMS items as pool_blk_run, one
 * (batch, head, query block) of int8 Q / K / V each; descriptor = geom {Sq, Skv, D, Dv, causal, M, SH, 0}.
 * Output: per item the softmax-weighted V in Q16, int32 [DW_OUT]; the host finishes (aiednn_rt_sdpa.h). */
static int sdpa_blk_run(const int8_t *items, const int32_t *geom, int32_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES);
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_SDPA;
        d[1] = (uint8_t)c;
        AIE_MODE_SDPA_DESC(d + 2, geom);
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++)
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 16);
    const int32_t *cm = (const int32_t *)g_gemm_c;
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++)
            for (int i = 0; i < DW_OUT; i++)
                out[(r * HW_COLS + c) * DW_OUT + i] = cm[(r * 32 + i / 32) * GEMM_N + c * 32 + i % 32];
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(sdpa_blk_s8s32) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return sdpa_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int32_t),
                        AIEK_PTR(&args[2], int32_t));
}

// clang-format off
AIEK_SPEC(sdpa_blk_s8s32,
    "kernel    sdpa_s8s32\n"
    "op        sdpa\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     SDPA\n"
    "mesh      src[0]=cores dst[0]=cores\n"
    "in    int8  [16,4096]  src\n"
    "param int32 [8]        weight\n"
    "out   int32 [16,1024]  dst\n");
// clang-format on

/* Eltwise overlay (aie_modes_pool.h, opcode AIE_MODE_OP_ELTWISE): DW_ITEMS items as dw_blk_run, a at
 * item bytes [0, 1024), b at [2048, 3072), geom {op, signed bits, n}; out int32 [16][1024] = a op b. */
static int elt_blk_run(const int8_t *items, const int32_t *geom, int32_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES);
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_ELTWISE;
        d[1] = (uint8_t)c;
        AIE_MODE_ELTWISE_DESC(d + 2, geom);
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++)
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 16);
    const int32_t *cm = (const int32_t *)g_gemm_c;
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++)
            for (int i = 0; i < 32; i++) // core (r, c): 32 rows of 32 int32 at rows r*32.., cols c*32..
                memcpy(out + (r * HW_COLS + c) * DW_OUT + i * 32, cm + (size_t)(r * 32 + i) * GEMM_N + c * 32, 128);
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(elt_blk_s8s32) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return elt_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int32_t),
                       AIEK_PTR(&args[2], int32_t));
}

// clang-format off
AIEK_SPEC(elt_blk_s8s32,
    "kernel    elt_s8s32\n"
    "op        eltwise\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     ELTWISE\n"
    "mesh      src[0]=cores dst[0]=cores\n"
    "in    int8  [16,4096]  src\n"
    "param int32 [3]        geom\n"
    "out   int32 [16,1024]  dst\n");
// clang-format on

/* Several megakernels (_binary_kernel_gemm_s8s32__mk<k>_*): aiednn_megakernel_select(k) makes the next
 * launch load + start ELF k; host program, routing and windows are shared. No-op in a single-ELF build. */
#if defined(AIEHLC_ASYNC_LAUNCH)
#define AIEDNN_MK_MAX 8
#define AIEDNN_MK_DECL(k) extern "C" unsigned char _binary_kernel_gemm_s8s32__mk##k##_start[] __attribute__((weak));
AIEDNN_MK_DECL(0)
AIEDNN_MK_DECL(1) AIEDNN_MK_DECL(2) AIEDNN_MK_DECL(3) AIEDNN_MK_DECL(4) AIEDNN_MK_DECL(5) AIEDNN_MK_DECL(6)
    AIEDNN_MK_DECL(7)
    /* _binary_kernel_gemm_s8s32_start: the default core ELF, declared by the generated host program */
    /* ELF k; a library build (dnn_build modes="library") has no __mk0: ELF 0 is the default kernel.o */
    static unsigned char *aiednn_mk_elf(int k) {
    unsigned char *const elf[AIEDNN_MK_MAX] = {
        _binary_kernel_gemm_s8s32__mk0_start, _binary_kernel_gemm_s8s32__mk1_start,
        _binary_kernel_gemm_s8s32__mk2_start, _binary_kernel_gemm_s8s32__mk3_start,
        _binary_kernel_gemm_s8s32__mk4_start, _binary_kernel_gemm_s8s32__mk5_start,
        _binary_kernel_gemm_s8s32__mk6_start, _binary_kernel_gemm_s8s32__mk7_start};
    if (k < 0 || k >= AIEDNN_MK_MAX)
        return nullptr;
    return elf[k] ? elf[k] : k == 0 ? _binary_kernel_gemm_s8s32_start : nullptr;
}
static unsigned char *aiednn_mk_resident(void) {
    return s_persist_elf ? s_persist_elf : _binary_kernel_gemm_s8s32_start;
}
extern "C" long aiednn_megakernel_switches;
extern "C" void aiednn_megakernel_select(int k) {
    unsigned char *elf = aiednn_mk_elf(k);
    if (!elf || elf == aiednn_mk_resident())
        return;
    aiednn_megakernel_switches++; /* a per-op segment plan's switch counts too (dnn_build modes="auto") */
    __aie_persist_select_elf(elf);
}
/* library megakernels: the calling entry's modes are in the ELFs in *mask* (bit k = ELF k). Keep the
 * resident one if it is among them, else load the lowest. */
extern "C" long aiednn_megakernel_switches = 0;
extern "C" void aiednn_megakernel_need(unsigned mask) {
    unsigned char *cur = aiednn_mk_resident();
    for (int k = 0; k < AIEDNN_MK_MAX; k++)
        if ((mask >> k & 1u) && aiednn_mk_elf(k) == cur)
            return;
    for (int k = 0; k < AIEDNN_MK_MAX; k++)
        if ((mask >> k & 1u) && aiednn_mk_elf(k)) {
            aiednn_megakernel_switches++;
            __aie_persist_select_elf(aiednn_mk_elf(k));
            return;
        }
}
#endif

/* is the MMT mode compiled into the core (aie_modes_cfg.h; opt-in, or AIEK_LIB_MODE_MMT in a
 * library build)? The entry links either way. */
#ifdef AIEK_LIB_MODE_MMT
#define AIEK_HOST_MODE_MMT AIEK_LIB_MODE_MMT
#else
#define AIEK_HOST_MODE_MMT AIEK_MODE_MMT
#endif
extern "C" int aiednn_gemm_mmt_available(void) { return AIEK_HOST_MODE_MMT; }

/* MMT (aie_modes_mmt.h, opcode AIE_MODE_OP_MMT): short-K int8 GEMM, 128 rows x 32 channels per core.
 * a int8 [HW_ROWS][16384]: mesh row r's A as aie_mmt_pack_a lays it (round kr at kr * 2048);
 * b int8 [HW_COLS][AIE_MMT_RES]: column c's pre-tiled B (aie_mmt_pack_b) + its requant params at
 * AIE_MMT_QOFF; geom int32 [1] = 2 (the K-128 tile; K <= 128 zero-padded);
 * out int8 [cores][4096]: core (r, c)'s rows, row m channel j at m * 32 + j. */
static int mmt_blk_run(const int8_t *a, const int8_t *b, const int32_t *geom, int8_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (!AIEK_HOST_MODE_MMT || geom[0] != 2 || gemm_bufs(device) != AIEK_OK) /* the core: K tile 128 only */
        return AIEK_ERR_ARGS; /* compiled out: the core would run a GEMM on these bytes */
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    for (int r = 0; r < HW_ROWS; r++) /* round kr's window = bytes [kr * 64, +64) of the 32 rows */
        aie_mmt_a_to_launch(a + (size_t)r * 16384, r, g_gemm_a);
    for (int c = 0; c < HW_COLS; c++) {
        int8_t *col = g_gemm_b + (size_t)c * rows_nb * K;
        for (int i = 0; i < AIE_MMT_RES / 64; i++) /* k-round 0's window: rows 0.. of the column block */
            memcpy(col + (size_t)i * K, b + (size_t)c * AIE_MMT_RES + i * 64, 64);
        uint8_t *d = (uint8_t *)col + (size_t)(rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_MMT, d[1] = (uint8_t)c, d[2] = (uint8_t)geom[0];
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++) /* a GEMM must see a zero descriptor row again */
        memset(g_gemm_b + (size_t)(c * rows_nb + rows_nb - 1) * K, 0, 64);
    const int8_t *cb = (const int8_t *)g_gemm_c;
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++)
            for (int row = 0; row < 32; row++)
                memcpy(out + (size_t)(r * HW_COLS + c) * 4096 + row * 128,
                       cb + (size_t)(r * 32 + row) * GEMM_N * 4 + c * 128, 128);
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(mmt_blk_s8s8) {
    if (num_args != 4)
        return AIEK_ERR_ARGS;
    return mmt_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                       AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], int8_t));
}

// "match conv2d" as the other overlays; the runtime binds the block by signature (is_mmt_block)
// clang-format off
/* target vek280 only: MMT runs mlir-aie mm_aie2.h, an AIE-ML (__AIE_ARCH__ 20) micro-kernel */
AIEK_SPEC(mmt_blk_s8s8,
    "kernel    mmt_s8s8\n"
    "op        gemm_short_k\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280\n"
    "modes     MMT\n"
    "mesh      src[0]=rows weight[0]=cols dst[0]=cores\n"
    "in    int8  [4,16384]  src\n"
    "param int8  [4,5120]   weight\n"
    "param int32 [1]        geom\n"
    "out   int8  [16,4096]  dst\n");
// clang-format on

/* LUT / STEPS overlay (aie_modes_lut.h, opcodes AIE_MODE_OP_LUT / _STEPS): item k = r*HW_COLS+c on
 * core (r, c) as in dw_blk_run, its AIE_LUT_TBYTES of tables in rows AIE_LUT_ROWS*r .. of column c's
 * B block, geom int32 [k][4] = {n, run0, inner, nt | es << 8 | stride/8 << 16 | kind << 26 |
 * steps << 27} (one es / stride / kind / mode per launch); out = the core's 4 KB C tile per item. */
static int lut_blk_run(const int8_t *items, const int8_t *tables, const int32_t *geom, int8_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES);
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        const uint32_t g3 = (uint32_t)geom[3]; // ES, stride, kind, mode: one per launch
        d[0] = (g3 >> 27) & 1 ? AIE_MODE_OP_STEPS : AIE_MODE_OP_LUT;
        d[1] = (uint8_t)c;
        d[2] = (uint8_t)(g3 >> 8);
        d[3] = (uint8_t)((g3 >> 26) & 1);
        d[4] = (uint8_t)(((g3 >> 16) & 0x3ff) * 8), d[5] = (uint8_t)((((g3 >> 16) & 0x3ff) * 8) >> 8);
        for (int r = 0; r < HW_ROWS; r++) {
            const int k = r * HW_COLS + c;
            const int32_t *g = geom + 4 * k;
            aie_lut_entry(d + AIE_LUT_ENTRY(r), g[0], g[1], g[2], g[3] & 0xff);
            memcpy(g_gemm_b + (size_t)(c * rows_nb + AIE_LUT_ROWS * r) * K, tables + (size_t)k * AIE_LUT_TBYTES,
                   AIE_LUT_TBYTES);
        }
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++) // a GEMM must see a zero descriptor row again
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 64);
    const int8_t *cb = (const int8_t *)g_gemm_c; // core (r, c) tile: rows r*32.., bytes c*128.. of 512-byte C rows
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++) {
            const int k = r * HW_COLS + c;
            const int nb = geom[4 * k] * ((geom[4 * k + 3] >> 8) & 0xff);
            for (int row = 0; row * 128 < nb; row++)
                memcpy(out + (size_t)k * AIE_LUT_ITEM + row * 128, cb + (size_t)(r * 32 + row) * GEMM_N * 4 + c * 128,
                       nb - row * 128 < 128 ? nb - row * 128 : 128);
        }
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(lut_blk_s8) {
    if (num_args != 4)
        return AIEK_ERR_ARGS;
    return lut_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                       AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], int8_t));
}

// "match conv2d" as the pool overlay: the library indexes conv2d / dense families only; the
// runtime binds the block by signature (is_lut_block) to eltwise / prelu / bnorm / binary / reorder
// clang-format off
AIEK_SPEC(lut_blk_s8,
    "kernel    lut8\n"
    "op        lut\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     LUT STEPS\n"
    "mesh      src[0]=cores table[0]=cores geom[0]=cores dst[0]=cores\n"
    "in    int8  [16,4096]  src\n"
    "param int8  [16,7680]  table\n"
    "param int32 [16,4]     geom\n"
    "out   int8  [16,4096]  dst\n");
// clang-format on

/* FVEC overlay (aie_modes_fvec.h, opcode AIE_MODE_OP_FVEC): item k = r*HW_COLS+c on core (r, c) as
 * lut_blk_run, its AIE_FVEC_AUX floats of AUX in rows AIE_LUT_ROWS*r .. of column c's B block, geom int32
 * [k][4] = {n, rowlen, op | flags << 8, 0} (op / flags one per launch), ab = {alpha, beta}; out = the
 * core's 4 KB C tile per item. */
static int fvec_blk_run(const float *items, const float *aux, const int32_t *geom, const float *ab, float *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ALLOC;
    const int rows_nb = N / HW_COLS;
    b_not_cached();
    const uint64_t t0 = gemm_ticks();
    memcpy(g_gemm_a, items, DW_ITEMS * DW_ITEM_BYTES);
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_gemm_b + (c * rows_nb + rows_nb - 1) * K;
        d[0] = AIE_MODE_OP_FVEC, d[1] = (uint8_t)c, d[2] = (uint8_t)(geom[2] & 0xff), d[3] = (uint8_t)(geom[2] >> 8);
        memcpy(d + 4, ab, 2 * sizeof(float));
        for (int r = 0; r < HW_ROWS; r++) {
            const int k = r * HW_COLS + c;
            aie_fvec_entry(d + AIE_FVEC_ENTRY(r), geom[4 * k], geom[4 * k + 1]);
            memcpy(g_gemm_b + (size_t)(c * rows_nb + AIE_LUT_ROWS * r) * K, aux + (size_t)k * AIE_FVEC_AUX,
                   AIE_FVEC_AUX * sizeof(float));
        }
    }
    const uint64_t t1 = gemm_ticks();
    gemm_s8s32<<<mesh>>>(g_gemm_a, g_gemm_b, g_gemm_c, M, N, K);
    const uint64_t t2 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++) // a GEMM must see a zero descriptor row again
        memset(g_gemm_b + (c * rows_nb + rows_nb - 1) * K, 0, 64);
    const int8_t *cb = (const int8_t *)g_gemm_c; // core (r, c) tile: rows r*32.., bytes c*128.. of 512-byte C rows
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++) {
            const int k = r * HW_COLS + c, nb = geom[4 * k] * 4;
            for (int row = 0; row * 128 < nb; row++)
                memcpy((int8_t *)out + (size_t)k * AIE_FVEC_N * 4 + row * 128,
                       cb + (size_t)(r * 32 + row) * GEMM_N * 4 + c * 128, nb - row * 128 < 128 ? nb - row * 128 : 128);
        }
    g_gemm_t[0] += t1 - t0, g_gemm_t[1] += t2 - t1, g_gemm_t[2] += gemm_ticks() - t2, g_gemm_calls++;
    return AIEK_OK;
}

AIEK_ENTRY(fvec_blk_f32) {
    if (num_args != 5)
        return AIEK_ERR_ARGS;
    return fvec_blk_run(AIEK_PTR(&args[0], const float), AIEK_PTR(&args[1], const float),
                        AIEK_PTR(&args[2], const int32_t), AIEK_PTR(&args[3], const float), AIEK_PTR(&args[4], float));
}

// the runtime binds the block by signature (is_fvec_block) to eltwise / binary / softmax / layer norm
// clang-format off
AIEK_SPEC(fvec_blk_f32,
    "kernel    fvec_f32\n"
    "op        fvec\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     FVEC\n"
    "mesh      src[0]=cores aux[0]=cores geom[0]=cores dst[0]=cores\n"
    "in    float32 [16,1024]  src\n"
    "param float32 [16,1920]  aux\n"
    "param int32   [16,4]     geom\n"
    "param float32 [1,2]      ab\n"
    "out   float32 [16,1024]  dst\n");
// clang-format on

AIEK_ENTRY(dw_blk_s8s32) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return dw_blk_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int32_t), AIEK_PTR(&args[2], int32_t));
}

// clang-format off
AIEK_SPEC(dw_blk_s8s32,
    "kernel    dw_s8s32\n"
    "op        conv2d\n"
    "match     conv2d\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "modes     DW\n"
    "mesh      src[0]=cores dst[0]=cores\n"
    "in    int8  [16,4096]  src\n"
    "param int32 [8]        weight\n"
    "out   int32 [16,1024]  dst\n");
// clang-format on

// clang-format off
AIEK_SPEC(gemm_blk_s8s32,
    "kernel    gemm_s8s32\n"
    "op        gemm_s8s32\n"
    "match     dense\n"
    "reference gemm_uk_ref.c\n"
    "target    vek280 vek385\n"
    "mesh      src[0]=32*rows dst[0]=32*rows\n"
    "in    int8  [128,512]  src\n"
    "param int8  [128,512]  weight\n"
    "out   int32 [128,128]  dst\n");
// clang-format on

/* n GEMM tiles per launch (persistent build only): A tiles go into aiednn_gemm_ntile_stage_a() (m blocks
 * outer, k blocks inner), aiednn_gemm_ntile_start() streams nm x nkb tiles and aiednn_gemm_ntile_c(mb)
 * is m block mb's result. nkb == 1: one shared B block re-emitted from the mem tile; nkb > 1: B block kb
 * per tile, K chained on the core (0xA0), ending in int32 C (which 0) or int8 requant (which 1). */
#if defined(AIEHLC_NTILE_LAUNCH)
#define NT_MAX 32 /* tiles per launch: the mem tile start queue's repeat count (8 windows each) */
static int8_t *g_nt_a, *g_nt_b, *g_nt_c;
static int64_t g_nt_a_tiles, g_nt_b_blocks;
static int g_nt_n, g_nt_nkb;
static const int8_t *g_nt_arg_a;
static uint64_t g_nt_t1;

extern "C" int aiednn_gemm_ntile_max(void) { return NT_MAX; }

/* the next aiednn_gemm_ntile_start's A tiles are direct-conv regions (aie_modes_conv.h: mesh row r's
 * input rows, its geometry in the slice's last 64 bytes, written by the runtime): every B block then
 * carries CONV_MAGIC in word 0 of row CONV_GEOM_ROW0, which the core reads per tile */
static int g_nt_conv;
extern "C" void aiednn_gemm_ntile_conv(int on) { g_nt_conv = AIEK_MODE_CONV && on; }

/* device memory for `tiles` A tiles of [GEMM_M][K] (grow-only; contents kept on growth) */
extern "C" int8_t *aiednn_gemm_ntile_stage_a(int64_t tiles, int64_t tile_bytes) {
    if (tile_bytes != (int64_t)M * K)
        return nullptr;
    if (tiles > g_nt_a_tiles) {
        aieArray device;
        aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS); /* alloc needs the partition */
        (void)mesh;
        int8_t *grown = (int8_t *)device.alloc((size_t)tiles * M * K);
        if (!grown)
            return nullptr;
        if (g_nt_a)
            memcpy(grown, g_nt_a, (size_t)g_nt_a_tiles * M * K);
        g_nt_a = grown, g_nt_a_tiles = tiles; /* the old block stays allocated: a launch may be reading it */
    }
    return g_nt_a;
}

/* B block kb (logical [GEMM_N][K]) into launch block kb: each column's real rows, then its
 * descriptor row */
static void nt_b_block(int8_t *blk, const int8_t *b, int kb, int nkb, int which, const int32_t *qp) {
    const int real = GEMM_N / HW_COLS, rows_nb = N / HW_COLS;
    const int last = kb == nkb - 1;
    for (int j = 0; j < HW_COLS; j++) {
        int8_t *col = blk + (size_t)j * rows_nb * K;
        memcpy(col, b + (size_t)j * real * K, (size_t)real * K);
        const int32_t magic = g_nt_conv ? CONV_MAGIC : 0; // set or cleared: the block buffer persists
        memcpy(col + (size_t)CONV_GEOM_ROW0 * K, &magic, 4);
        uint8_t *d = (uint8_t *)col + (size_t)(rows_nb - 1) * K;
        memset(d, 0, K);
        d[1] = (uint8_t)j;
        d[2] = (uint8_t)(kb > 0); /* continue from the accumulators */
        if (!last) {
            d[0] = 0xA0; /* accumulate only */
        } else if (which) {
            d[0] = 0xF1;
            int32_t *w = (int32_t *)(d + 4);
            w[0] = qp[0], w[1] = qp[1], w[2] = qp[2];
            int64_t *bb = (int64_t *)(d + 16);
            int32_t *m = (int32_t *)(d + 272);
            int8_t *sh = (int8_t *)(d + 400);
            for (int i = 0; i < real; i++) {
                memcpy(&bb[i], qp + 4 + 2 * (j * real + i), 8);
                m[i] = qp[4 + 2 * GEMM_N + j * real + i];
                sh[i] = (int8_t)qp[4 + 3 * GEMM_N + j * real + i];
            }
        }
    }
}

extern "C" int aiednn_gemm_ntile_start(int which, const int8_t *a, int nm, int nkb, const int8_t *b, const int32_t *qp,
                                       uint64_t key, int64_t bm, int64_t bn, int64_t bk) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const int n = nm * nkb;
    const size_t blk = (size_t)N * K;
    if (n < 1 || n > NT_MAX || (which && !qp) || bm != GEMM_M || bn != GEMM_N || bk != K ||
        gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ARGS;
    if (!s_persist_started) { /* the first launch of the process configures the array: one plain GEMM */
        memset(g_gemm_a, 0, (size_t)M * K);
        b_not_cached();
        gemm_s8s32_run(g_gemm_a, g_gemm_b, (int32_t *)g_gemm_c);
    }
    if (nkb > g_nt_b_blocks) {
        g_nt_b = (int8_t *)device.alloc((size_t)nkb * blk);
        if (!g_nt_b)
            return AIEK_ERR_ALLOC;
        memset(g_nt_b, 0, (size_t)nkb * blk); /* the padding rows stay zero */
        g_nt_b_blocks = nkb;
    }
    if (!g_nt_c && !(g_nt_c = (int8_t *)device.alloc((size_t)NT_MAX * M * N)))
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    for (int kb = 0; kb < nkb; kb++)
        nt_b_block(g_nt_b + kb * blk, b + (size_t)kb * GEMM_N * K, kb, nkb, which, qp);
    __Runtime_memtile_weights_key(nkb == 1 ? key : 0, 32); /* a shared block may stay in the mem tiles */
    __Runtime_memtile_weights_head(GEMM_N / HW_COLS);      /* its padding rows stay in DDR (compact miss) */
    g_wkey_next = 0, g_wkey_last = 0;
    const uint64_t t1 = gemm_ticks();
    __aie_launch_n_start("gemm_s8s32", mesh, (void *)a, (size_t)n * M * K, g_nt_b, (size_t)nkb * blk, g_nt_c,
                         (size_t)n * M * N, n, (int64_t)M * K, nkb > 1 ? (int64_t)blk : 0, nkb, (int64_t)M * N);
    g_nt_n = n, g_nt_nkb = nkb, g_nt_arg_a = a, g_nt_t1 = t1;
    g_gemm_t[0] += t1 - t0;
    return AIEK_OK;
}

extern "C" int aiednn_gemm_ntile_finish(void) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    if (!g_nt_n)
        return AIEK_ERR_ARGS;
    __aie_launch_finish("gemm_s8s32", mesh, (void *)g_nt_arg_a, (size_t)g_nt_n * M * K, g_nt_b,
                        (size_t)g_nt_nkb * N * K, g_nt_c, (size_t)g_nt_n * M * N);
    g_gemm_t[1] += gemm_ticks() - g_nt_t1, g_gemm_calls++;
    g_nt_n = 0;
    return AIEK_OK;
}

/* m block mb's result of the last n-tile launch (its last k tile): int32 [GEMM_M][GEMM_N] (which 0),
 * or int8 C with row r's core column j at bytes [r * 4 * GEMM_N + j * 128, +GEMM_N / HW_COLS) (which 1) */
/* n MMT tiles in one launch (as mmt_blk_s8s8, aie_modes_mmt.h): a = n launch A tiles in the device
 * buffer of aiednn_gemm_ntile_stage_a (tile t: [M][K], mesh row r's packed A put by
 * aie_mmt_a_to_launch), b = every column's AIE_MMT_RES bytes, shared by the n tiles (re-emitted from
 * the mem tile); tile t's result is aiednn_gemm_ntile_c(t), core (r, c)'s 4 KB at rows r * 32.., bytes
 * c * 128.. of its 512-byte rows. Finished by aiednn_gemm_ntile_finish. */
extern "C" int aiednn_gemm_mmt_n_start(const int8_t *a, int n, const int8_t *b, int kcode) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const size_t blk = (size_t)N * K;
    const int rows_nb = N / HW_COLS;
    if (!AIEK_HOST_MODE_MMT || kcode != 2 || n < 1 || n > NT_MAX || gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ARGS;
    if (!s_persist_started) { /* the first launch of the process configures the array: one plain GEMM */
        memset(g_gemm_a, 0, (size_t)M * K);
        b_not_cached();
        gemm_s8s32_run(g_gemm_a, g_gemm_b, (int32_t *)g_gemm_c);
    }
    if (g_nt_b_blocks < 1) {
        g_nt_b = (int8_t *)device.alloc(blk);
        if (!g_nt_b)
            return AIEK_ERR_ALLOC;
        memset(g_nt_b, 0, blk);
        g_nt_b_blocks = 1;
    }
    if (!g_nt_c && !(g_nt_c = (int8_t *)device.alloc((size_t)NT_MAX * M * N)))
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    for (int c = 0; c < HW_COLS; c++) {
        int8_t *col = g_nt_b + (size_t)c * rows_nb * K;
        for (int i = 0; i < AIE_MMT_RES / 64; i++)
            memcpy(col + (size_t)i * K, b + (size_t)c * AIE_MMT_RES + i * 64, 64);
        uint8_t *d = (uint8_t *)col + (size_t)(rows_nb - 1) * K;
        memset(d, 0, 64);
        d[0] = AIE_MODE_OP_MMT, d[1] = (uint8_t)c, d[2] = (uint8_t)kcode;
    }
    __Runtime_memtile_weights_key(0, 0);
    g_wkey_next = 0, g_wkey_last = 0;
    const uint64_t t1 = gemm_ticks();
    __aie_launch_n_start("gemm_s8s32", mesh, (void *)a, (size_t)n * M * K, g_nt_b, blk, g_nt_c, (size_t)n * M * N, n,
                         (int64_t)M * K, 0, 1, (int64_t)M * N);
    g_nt_n = n, g_nt_nkb = 1, g_nt_arg_a = a, g_nt_t1 = t1;
    g_gemm_t[0] += t1 - t0;
    return AIEK_OK;
}

extern "C" const int8_t *aiednn_gemm_ntile_c(int mb) { return g_nt_c + (size_t)(mb * g_nt_nkb + g_nt_nkb - 1) * M * N; }

/* n depthwise + requant launches in one (dwq_blk_s8s8's 3 * DW_ITEMS items per tile, 0xD8): items =
 * n tiles of dwq_blk_run's input, geom as there. Tile t's A is its first DW_ITEMS items, its own B
 * block (streamed through the mem tile, b_step one block) the other 2 * DW_ITEMS + the descriptor.
 * B blocks live in their own device buffer: the items land in rows a GEMM tile keeps as zero padding.
 * aiednn_gemm_dwq_n_finish(out) waits and writes int8 [n * 3 * DW_ITEMS][DW_OUT] in item order. */
static int8_t *g_dwn_b;
static int64_t g_dwn_blocks;
static int g_dwn_n;

extern "C" int aiednn_gemm_dwq_n_start(const int8_t *items, int n, const int32_t *geom) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const size_t blk = (size_t)N * K, per = (size_t)3 * DW_ITEMS * DW_ITEM_BYTES;
    const int rows_nb = N / HW_COLS;
    if (!items || n < 1 || n > NT_MAX || gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ARGS;
    if (!s_persist_started) { /* the first launch of the process configures the array: one plain GEMM */
        memset(g_gemm_a, 0, (size_t)M * K);
        b_not_cached();
        gemm_s8s32_run(g_gemm_a, g_gemm_b, (int32_t *)g_gemm_c);
    }
    int8_t *a = aiednn_gemm_ntile_stage_a(n, (int64_t)M * K);
    if (!a)
        return AIEK_ERR_ALLOC;
    if (n > g_dwn_blocks) {
        g_dwn_b = (int8_t *)device.alloc((size_t)n * blk);
        if (!g_dwn_b)
            return AIEK_ERR_ALLOC;
        memset(g_dwn_b, 0, (size_t)n * blk);
        g_dwn_blocks = n;
    }
    if (!g_nt_c && !(g_nt_c = (int8_t *)device.alloc((size_t)NT_MAX * M * N)))
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    for (int t = 0; t < n; t++) {
        const int8_t *its = items + (size_t)t * per;
        int8_t *bt = g_dwn_b + (size_t)t * blk;
        memcpy(a + (size_t)t * M * K, its, (size_t)DW_ITEMS * DW_ITEM_BYTES);
        for (int c = 0; c < HW_COLS; c++) { /* as dwq_blk_run */
            for (int r = 0; r < HW_ROWS; r++)
                for (int q = 0; q < 2; q++) {
                    const int8_t *it = its + (size_t)(DW_ITEMS + 2 * (r * HW_COLS + c) + q) * DW_ITEM_BYTES;
                    int8_t *slot = bt + (size_t)(c * rows_nb + 16 * r + 8 * q) * K;
                    memcpy(slot, it, DW_ITEM_BYTES);
                    if (16 * r + 8 * q + 7 == rows_nb - 1)
                        memcpy(slot + DWQ_STASH, it + 3584, DWQ_STASH_N);
                }
            uint8_t *d = (uint8_t *)bt + (size_t)(c * rows_nb + rows_nb - 1) * K;
            const int v[] = {0xD8, c, geom[0], geom[1], geom[2]};
            for (int i = 0; i < 5; i++)
                d[i] = (uint8_t)v[i];
            for (int i = 0; i < 3; i++) {
                d[5 + 2 * i] = (uint8_t)(geom[3 + i] & 0xff);
                d[6 + 2 * i] = (uint8_t)(geom[3 + i] >> 8);
            }
            d[11] = 2;
        }
    }
    __Runtime_memtile_weights_key(0, 0);
    g_wkey_next = 0, g_wkey_last = 0;
    const uint64_t t1 = gemm_ticks();
    __aie_launch_n_start("gemm_s8s32", mesh, (void *)a, (size_t)n * M * K, g_dwn_b, (size_t)n * blk, g_nt_c,
                         (size_t)n * M * N, n, (int64_t)M * K, n > 1 ? (int64_t)blk : 0, n, (int64_t)M * N);
    g_dwn_n = n, g_nt_t1 = t1, g_nt_arg_a = a;
    g_gemm_t[0] += t1 - t0;
    return AIEK_OK;
}

/* n tiles of DW_ITEMS A items (one 4 KB item per core: pool, eltwise, 0xD7 depthwise) in one launch,
 * every tile with the same B -- only its descriptor rows {opcode, column, desc[0..desc_n)} -- which the
 * mem tile re-emits n times (b_step 0). aiednn_gemm_items_n_finish(out) waits and writes int32
 * [n * DW_ITEMS][DW_OUT] in item order. */
static int8_t *g_itn_b;
static int g_itn_n;

extern "C" int aiednn_gemm_items_n_start(const int8_t *items, int n, int opcode, const uint8_t *desc, int desc_n) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const size_t blk = (size_t)N * K;
    const int rows_nb = N / HW_COLS;
    if (!items || n < 1 || n > NT_MAX || desc_n < 0 || desc_n > 30 || gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ARGS;
    if (!s_persist_started) { /* the first launch of the process configures the array: one plain GEMM */
        memset(g_gemm_a, 0, (size_t)M * K);
        b_not_cached();
        gemm_s8s32_run(g_gemm_a, g_gemm_b, (int32_t *)g_gemm_c);
    }
    int8_t *a = aiednn_gemm_ntile_stage_a(n, (int64_t)M * K);
    if (!a)
        return AIEK_ERR_ALLOC;
    if (!g_itn_b) {
        if (!(g_itn_b = (int8_t *)device.alloc(blk)))
            return AIEK_ERR_ALLOC;
        memset(g_itn_b, 0, blk);
    }
    if (!g_nt_c && !(g_nt_c = (int8_t *)device.alloc((size_t)NT_MAX * M * N)))
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    memcpy(a, items, (size_t)n * DW_ITEMS * DW_ITEM_BYTES); /* tile t's items are A tile t */
    for (int c = 0; c < HW_COLS; c++) {
        uint8_t *d = (uint8_t *)g_itn_b + (size_t)(c * rows_nb + rows_nb - 1) * K;
        memset(d, 0, 32);
        d[0] = (uint8_t)opcode, d[1] = (uint8_t)c;
        memcpy(d + 2, desc, (size_t)desc_n);
    }
    __Runtime_memtile_weights_key(0, 0);
    g_wkey_next = 0, g_wkey_last = 0;
    const uint64_t t1 = gemm_ticks();
    __aie_launch_n_start("gemm_s8s32", mesh, (void *)a, (size_t)n * M * K, g_itn_b, blk, g_nt_c, (size_t)n * M * N, n,
                         (int64_t)M * K, 0, 1, (int64_t)M * N);
    g_itn_n = n, g_nt_t1 = t1, g_nt_arg_a = a;
    g_gemm_t[0] += t1 - t0;
    return AIEK_OK;
}

extern "C" int aiednn_gemm_items_n_finish(int32_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const int n = g_itn_n;
    if (!n || !out)
        return AIEK_ERR_ARGS;
    __aie_launch_finish("gemm_s8s32", mesh, (void *)g_nt_arg_a, (size_t)n * M * K, g_itn_b, (size_t)N * K, g_nt_c,
                        (size_t)n * M * N);
    const uint64_t t2 = gemm_ticks();
    g_gemm_t[1] += t2 - g_nt_t1, g_gemm_calls++;
    for (int t = 0; t < n; t++) { /* core (r, c)'s int32 tile: rows r*32.., int32 columns c*32.. of GEMM_N-wide rows */
        const int32_t *cm = (const int32_t *)(g_nt_c + (size_t)t * M * N);
        for (int r = 0; r < HW_ROWS; r++)
            for (int c = 0; c < HW_COLS; c++) {
                int32_t *o = out + ((size_t)t * DW_ITEMS + r * HW_COLS + c) * DW_OUT;
                for (int row = 0; row < DW_OUT / 32; row++)
                    memcpy(o + row * 32, cm + (size_t)(r * 32 + row) * GEMM_N + c * 32, 32 * sizeof(int32_t));
            }
    }
    g_gemm_t[2] += gemm_ticks() - t2;
    g_itn_n = 0;
    return AIEK_OK;
}

extern "C" int aiednn_gemm_dwq_n_finish(int8_t *out) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const int n = g_dwn_n;
    if (!n || !out)
        return AIEK_ERR_ARGS;
    __aie_launch_finish("gemm_s8s32", mesh, (void *)g_nt_arg_a, (size_t)n * M * K, g_dwn_b, (size_t)n * N * K, g_nt_c,
                        (size_t)n * M * N);
    const uint64_t t2 = gemm_ticks();
    g_gemm_t[1] += t2 - g_nt_t1, g_gemm_calls++;
    for (int t = 0; t < n; t++) { /* core (r, c) tile: rows r*32.., bytes c*128.. of 512-byte C rows */
        const int8_t *cb = g_nt_c + (size_t)t * M * N;
        int8_t *ot = out + (size_t)t * 3 * DW_ITEMS * DW_OUT;
        for (int r = 0; r < HW_ROWS; r++)
            for (int c = 0; c < HW_COLS; c++)
                for (int q = 0; q < 3; q++) {
                    const int item = q == 0 ? r * HW_COLS + c : DW_ITEMS + 2 * (r * HW_COLS + c) + q - 1;
                    int8_t *o8 = ot + (size_t)item * DW_OUT;
                    for (int row = 0; row < DW_OUT / 128; row++) /* 128-byte runs of the band */
                        memcpy(o8 + row * 128,
                               cb + (size_t)(r * 32 + (q * 1024 + row * 128) / 128) * GEMM_N * 4 + c * 128, 128);
                }
    }
    g_gemm_t[2] += gemm_ticks() - t2;
    g_dwn_n = 0;
    return AIEK_OK;
}

/* n bf16 GEMM tiles in one launch (bf16_blk_f32's block, AIE_MODE_OP_BF16): a = n A tiles bf16
 * [GEMM_M][AIE_BF16_K], w = n W tiles bf16 [GEMM_N][AIE_BF16_K] (channels x K), flags[t] = tile t's
 * bf16_blk_run flags (bit 0 continue the previous tile's accumulators, bit 1 more K follows). Every tile
 * streams its own B block through the mem tile (b_step one block), whose descriptor row carries the
 * opcode and flags. Tile t's f32 [GEMM_M][GEMM_N] is aiednn_gemm_bf16_n_c(t) after
 * aiednn_gemm_bf16_n_finish (meaningful where flags[t] has no bit 1). */
static int8_t *g_b16n_b;
static int64_t g_b16n_blocks;
static int g_b16n_n;

extern "C" int aiednn_gemm_bf16_n_start(const uint16_t *a, int n, const uint16_t *w, const uint8_t *flags) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const size_t blk = (size_t)N * K;
    const int rows_nb = N / HW_COLS;
    if (!a || !w || !flags || n < 1 || n > NT_MAX || gemm_bufs(device) != AIEK_OK)
        return AIEK_ERR_ARGS;
    if (!s_persist_started) { /* the first launch of the process configures the array: one plain GEMM */
        memset(g_gemm_a, 0, (size_t)M * K);
        b_not_cached();
        gemm_s8s32_run(g_gemm_a, g_gemm_b, (int32_t *)g_gemm_c);
    }
    int8_t *at = aiednn_gemm_ntile_stage_a(n, (int64_t)M * K);
    if (!at)
        return AIEK_ERR_ALLOC;
    if (n > g_b16n_blocks) {
        g_b16n_b = (int8_t *)device.alloc((size_t)n * blk);
        if (!g_b16n_b)
            return AIEK_ERR_ALLOC;
        memset(g_b16n_b, 0, (size_t)n * blk); /* the padding rows stay zero */
        g_b16n_blocks = n;
    }
    if (!g_nt_c && !(g_nt_c = (int8_t *)device.alloc((size_t)NT_MAX * M * N)))
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = gemm_ticks();
    for (int t = 0; t < n; t++) { /* as bf16_blk_run, tile t into A tile t and B block t */
        const uint16_t *ta = a + (size_t)t * GEMM_M * AIE_BF16_K, *tw = w + (size_t)t * GEMM_N * AIE_BF16_K;
        for (int r = 0; r < HW_ROWS; r++)
            aie_bf16_pack_a(ta + (size_t)r * 32 * AIE_BF16_K, 32, AIE_BF16_K, 0, AIE_BF16_K,
                            (uint8_t *)at + (size_t)t * M * K + (size_t)r * 32 * K);
        for (int j = 0; j < HW_COLS; j++) {
            uint8_t *col = (uint8_t *)g_b16n_b + (size_t)t * blk + (size_t)j * rows_nb * K;
            aie_bf16_pack_b(tw + (size_t)j * 32 * AIE_BF16_K, 32, AIE_BF16_K, 0, AIE_BF16_K, col);
            uint8_t *d = col + (size_t)(rows_nb - 1) * K;
            memset(d, 0, 16);
            d[0] = AIE_MODE_OP_BF16;
            d[1] = (uint8_t)j;
            d[2] = (uint8_t)(flags[t] & 3);
        }
    }
    __Runtime_memtile_weights_key(0, 0);
    g_wkey_next = 0, g_wkey_last = 0;
    const uint64_t t1 = gemm_ticks();
    __aie_launch_n_start("gemm_s8s32", mesh, (void *)at, (size_t)n * M * K, g_b16n_b, (size_t)n * blk, g_nt_c,
                         (size_t)n * M * N, n, (int64_t)M * K, n > 1 ? (int64_t)blk : 0, n, (int64_t)M * N);
    g_b16n_n = n, g_nt_t1 = t1, g_nt_arg_a = at;
    g_gemm_t[0] += t1 - t0;
    return AIEK_OK;
}

extern "C" int aiednn_gemm_bf16_n_finish(void) {
    aieArray device;
    aieMesh mesh = device.partition(GEMM_PARTITION, HW_ROWS, HW_COLS);
    const int n = g_b16n_n;
    if (!n)
        return AIEK_ERR_ARGS;
    __aie_launch_finish("gemm_s8s32", mesh, (void *)g_nt_arg_a, (size_t)n * M * K, g_b16n_b, (size_t)n * N * K, g_nt_c,
                        (size_t)n * M * N);
    g_gemm_t[1] += gemm_ticks() - g_nt_t1, g_gemm_calls++;
    g_b16n_n = 0;
    return AIEK_OK;
}

/* tile t's f32 [GEMM_M][GEMM_N] of the last aiednn_gemm_bf16_n_start (core (r, c): rows r*32.., floats c*32..) */
extern "C" const float *aiednn_gemm_bf16_n_c(int t) { return (const float *)(g_nt_c + (size_t)t * M * N); }
#endif

#undef M
#undef N
#undef K
