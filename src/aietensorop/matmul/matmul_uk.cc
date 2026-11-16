/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * matmul_uk -- the VEK280 int8 GEMM as an aiednn kernel (include/aie_user_kernel.h).
 * aie::mmul<8,8,8> core, 64x64 tiles, 4x4 cores, MMIO load/launch.
 *
 * The launch only accepts device.alloc() buffers, so the entry copies the
 * caller's tensors into three process-lifetime device buffers.
 ******************************************************************************/
#include "matmul.h"

#include "aie_user_kernel.h"

#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)

constexpr int kTileMn = 64;
constexpr aie::Bytes kL1Budget{8192};
constexpr aie::GemmSpace RowBA = {.policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = kL1Budget}},
                                  .d1 = {.fullsize = M, .tile_size = kTileMn, .stride = kTileMn},
                                  .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}};
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = kL1Budget}},
                                  .d1 = {.fullsize = N, .tile_size = kTileMn, .stride = kTileMn},
                                  .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}};
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = kL1Budget}},
    .d1 = {.fullsize = M, .tile_size = kTileMn, .stride = kTileMn},
    .d2 = {.fullsize = N, .tile_size = kTileMn, .stride = kTileMn}};
constexpr aie::GlobalPolicy matmul_policy = {.fullconnect_auto = 1};
__global__(matmul_policy) void matmul(aie::port<input_window_int8 *, RowBA> win_a,
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

    const int k_tiles = eff_k / 8;
    const int m_tiles = tile_rows / 8;
    const int n_tiles = tile_cols / 8;
    const int b_strips = cols_per_round / 8;
    alignas(aie::vector_decl_align) int8_t a_pack[tile_rows * eff_k];
    alignas(aie::vector_decl_align) int8_t b_strip[eff_k * 8];
    alignas(aie::vector_decl_align) int32_t acc[tile_rows * tile_cols];
    alignas(aie::vector_decl_align) int8_t c_tile[64];
    alignas(4) int8_t local_out[tile_rows * tile_cols];

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        for (int kr = 0; kr < k_rounds; kr++) {
            const bool first = (kr == 0);
            const bool last = (kr == k_rounds - 1);
            for (int ra = 0; ra < num_a_rounds; ra++) {
                const uint32_t *A_w = (const uint32_t *)acquire_input_window(win_a);
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
                const uint32_t *B_w = (const uint32_t *)acquire_input_window(win_b);
                for (int jt = 0; jt < b_strips; jt++) {
                    for (int kt = 0; kt < k_tiles; kt++) {
                        uint32_t *dst = (uint32_t *)(b_strip + kt * 64);
                        for (int j = 0; j < 8; j++) {
                            const uint32_t *src = B_w + (jt * 8 + j) * (eff_k / 4) + kt * 2;
                            dst[j * 2] = src[0];
                            dst[j * 2 + 1] = src[1];
                        }
                        aie::store_v(b_strip + kt * 64, aie::transpose(aie::load_v<64>(b_strip + kt * 64), 8, 8));
                    }
                    const int ct = rb * b_strips + jt;
                    for (int it = 0; it < m_tiles; it++) {
                        int32_t *acc_t = acc + (it * n_tiles + ct) * 64;
                        const int8_t *a_t = a_pack + it * k_tiles * 64;
                        aie::mmul<8, 8, 8, int8, int8> block;
                        if (first) {
                            block.mul(aie::load_v<64>(a_t), aie::load_v<64>(b_strip));
                        } else {
                            block = aie::mmul<8, 8, 8, int8, int8>(aie::load_v<64>(acc_t));
                            block.mac(aie::load_v<64>(a_t), aie::load_v<64>(b_strip));
                        }
                        for (int kt = 1; kt < k_tiles; kt++)
                            block.mac(aie::load_v<64>(a_t + kt * 64), aie::load_v<64>(b_strip + kt * 64));
                        if (!last) {
                            aie::store_v(acc_t, block.to_vector<int32>());
                            continue;
                        }
                        const auto c32 = block.to_vector<int32>();
                        for (int h = 0; h < 2; h++) {
                            aie::accum<acc32, 32> sat;
                            sat.from_vector(aie::min(aie::max(c32.extract<32>(h), (int32_t)-128), (int32_t)127));
                            aie::store_v(c_tile + h * 32, sat.to_vector<int8>(0));
                        }
                        const uint32_t *ct_w = (const uint32_t *)c_tile;
                        for (int i = 0; i < 8; i++) {
                            uint32_t *o = (uint32_t *)(local_out + (it * 8 + i) * tile_cols + ct * 8);
                            o[0] = ct_w[i * 2];
                            o[1] = ct_w[i * 2 + 1];
                        }
                    }
                }
                release_input_window(win_b);
            }
        }

        for (int rc = 0; rc < num_c_rounds; rc++) {
            uint32_t *out = (uint32_t *)acquire_output_window(win_c);
            const uint32_t *src = (const uint32_t *)(local_out + rc * buf_sz_c);
            for (int w = 0; w < buf_sz_c / 4; w++)
                out[w] = src[w];
            release_output_window(win_c);
        }
    }
}

// The mesh must be a function-local VarDecl from device.partition(...) in the
// same function as the launch (aiehlc recovers it from there); partition init is
// once per meshId.
static int8_t *g_mm_a = nullptr, *g_mm_b = nullptr, *g_mm_c = nullptr;

static int matmul_s8_run(const int8_t *a, const int8_t *b, int8_t *c) {
    aieArray device;
    aieMesh mesh = device.partition(MATMUL_PARTITION, HW_ROWS, HW_COLS);
    if (!g_mm_a) {
        g_mm_a = (int8_t *)device.alloc(M * K * 4);
        g_mm_b = (int8_t *)device.alloc(K * N * 4);
        g_mm_c = (int8_t *)device.alloc(M * N * 4);
        if (!g_mm_a || !g_mm_b || !g_mm_c)
            return AIEK_ERR_ALLOC;
    }
    memcpy(g_mm_a, a, M * K);
    memcpy(g_mm_b, b, K * N);
    matmul<<<mesh>>>(g_mm_a, g_mm_b, g_mm_c, M, N, K);
    memcpy(c, g_mm_c, M * N);
    return AIEK_OK;
}

AIEK_ENTRY(matmul_s8) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    return matmul_s8_run(AIEK_PTR(&args[0], const int8_t), AIEK_PTR(&args[1], const int8_t),
                         AIEK_PTR(&args[2], int8_t));
}

// clang-format off
AIEK_SPEC(matmul_s8,
    "kernel    matmul_s8\n"
    "op        dense_s8_sat\n"
    "match     dense\n"
    "reference matmul_uk_ref.c\n"
    "target    vek280 vek385\n"
    "in    int8 [256,256]  src\n"
    "param int8 [256,256]  weight\n"
    "out   int8 [256,256]  dst\n");
// clang-format on

#undef M
#undef N
#undef K
