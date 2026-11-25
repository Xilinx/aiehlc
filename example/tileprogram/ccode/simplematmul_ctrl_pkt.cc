/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication (control-packet variant)
 *
 * Identical to simplematmul2.cc except lock init, kernel ELF load, and core
 * launch use the reserved row-control fabric.
 */
#include "simplematmul.h"
void __Runtime_ctrl_pmap_enable(int on);
void __Runtime_ctrl_high_throughput_enable(int on);
#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)
#pragma CONTROL_PLAN_GROUP_REG_WRITE
#pragma control_plan_op_control_packet
#define MATMUL_LARGE_MMUL 1
#ifdef MATMUL_LARGE_MMUL
constexpr int kTileMn = 64;
constexpr aie::Bytes kL1Budget{8192};
#else
constexpr int kTileMn = 16;
constexpr aie::Bytes kL1Budget{4096};
#endif
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
    const int eff_k_b = aie::get_arg_per_round_size_in_dim(1, win_b);
    assert(eff_k == eff_k_b);

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

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    int8_t tag = (int8_t)((row & 0x7) | ((col & 0x7) << 3));
    klog("DEBUG", 3);
    klog("PRA0", (int32_t)aie::get_arg_per_round_size_in_dim(0, win_a));
    klog("PRA1", (int32_t)aie::get_arg_per_round_size_in_dim(1, win_a));
    klog("PRB0", (int32_t)aie::get_arg_per_round_size_in_dim(0, win_b));
    klog("PRB1", (int32_t)aie::get_arg_per_round_size_in_dim(1, win_b));
    klog("PRC0", (int32_t)aie::get_arg_per_round_size_in_dim(0, win_c));
    klog("PRC1", (int32_t)aie::get_arg_per_round_size_in_dim(1, win_c));
    klog("TRA0", (int32_t)aie::get_arg_total_rounds_in_dim(0, win_a));
    klog("TRA1", (int32_t)aie::get_arg_total_rounds_in_dim(1, win_a));
#endif

#ifdef MATMUL_LARGE_MMUL
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
#ifndef DEBUG_NOCOMPUTE
                for (int r = 0; r < rows_per_round; r++) {
                    const int row = ra * rows_per_round + r;
                    uint32_t *dst = (uint32_t *)(a_pack + (row >> 3) * k_tiles * 64 + (row & 7) * 8);
                    const uint32_t *src = A_w + r * (eff_k / 4);
                    for (int kt = 0; kt < k_tiles; kt++) {
                        dst[kt * 16] = src[kt * 2];
                        dst[kt * 16 + 1] = src[kt * 2 + 1];
                    }
                }
#endif
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                const uint32_t *B_w = (const uint32_t *)acquire_input_window(win_b);
#ifndef DEBUG_NOCOMPUTE
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
#endif
                release_input_window(win_b);
            }
        }

        for (int rc = 0; rc < num_c_rounds; rc++) {
            uint32_t *out = (uint32_t *)acquire_output_window(win_c);
#ifndef DEBUG_NOCOMPUTE
            const uint32_t *src = (const uint32_t *)(local_out + rc * buf_sz_c);
            for (int w = 0; w < buf_sz_c / 4; w++)
                out[w] = src[w];
#endif
            release_output_window(win_c);
        }
    }
#else
    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * tile_cols];
    int8_t local_out[tile_rows * tile_cols];

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
#ifndef DEBUG_NOCOMPUTE
        klog("MR  ", (int32_t)mr);
        for (int i = 0; i < tile_rows * tile_cols; i++)
            accum[i] = 0;
#endif
        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
#ifndef DEBUG_NOCOMPUTE
                for (int i = 0; i < buf_sz_a; i++) {
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                }
#endif
#if DEBUG_OUTPUT_ORDER
                for (int l = 0; l < (buf_sz_a < 8 ? buf_sz_a : 8); l++) {
                    klog("A   ", (int32_t)A_ptr[l]);
                }
#endif
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
#ifndef DEBUG_NOCOMPUTE
                for (int i = 0; i < tile_rows; i++) {
                    for (int j = 0; j < cols_per_round; j++) {
                        int16_t sum = 0;
                        for (int k = 0; k < eff_k; k++) {
                            sum += (int16_t)all_A[i * eff_k + k] * (int16_t)B_ptr[j * eff_k + k];
                        }
                        accum[i * tile_cols + rb * cols_per_round + j] += sum;
                    }
                }
#endif

#if DEBUG_OUTPUT_ORDER
                klog("B0  ", (int32_t)B_ptr[0]);
#endif

                release_input_window(win_b);
            }
        }

#ifndef DEBUG_NOCOMPUTE
        for (int i = 0; i < tile_rows * tile_cols; i++) {
            int16_t val = accum[i];
            if (val > 127)
                val = 127;
            else if (val < -128)
                val = -128;
            local_out[i] = (int8_t)val;
        }
#endif
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
#ifndef DEBUG_NOCOMPUTE
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++) {
                for (int j = 0; j < tile_cols; j++) {
                    out[i * tile_cols + j] = local_out[rc * buf_sz_c + i * tile_cols + j];
                }
            }
#endif
#if DEBUG_OUTPUT_ORDER
            klog("C0 ", (int32_t)out[0]);
#endif

            release_output_window(win_c);
        }
    }
#endif
}

__global__ void mul2(aie::port<input_window_int8 *, RowBA> win_a, aie::port<input_window_int8 *, ColBB> win_b,
                     aie::port<output_window_int8 *, LtoR_Merge> win_c) {

    const int tile_rows = aie::get_data_row();
    const int tile_cols = aie::get_data_col();
    const int eff_k = aie::get_effective_k();
    const int k_rounds = aie::get_k_rounds();
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);

    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);

    const int rows_per_round = buf_sz_a / eff_k;
    const int cols_per_round = buf_sz_b / eff_k;

    const int data_cols = n_rounds * tile_cols;

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    int8_t tag = (int8_t)((row & 0x7) | ((col & 0x7) << 3));
    klog("DEBUG", 3);
#endif

    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * data_cols];
    int8_t local_out[tile_rows * data_cols];

    for (int mr = 0; mr < m_rounds; mr++) {

        for (int i = 0; i < tile_rows * data_cols; i++)
            accum[i] = 0;

        for (int kr = 0; kr < k_rounds; kr++) {

            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }

#if DEBUG_OUTPUT_ORDER
            if (kr == 0 && mr == 0) {
                for (int i = 0; i < 16; i++) {
                    klog("A0  ", (int32_t)all_A[i]);
                }
            }
#endif

            for (int nr = 0; nr < n_rounds; nr++) {

                for (int rb = 0; rb < num_b_rounds; rb++) {
                    int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);

#if DEBUG_OUTPUT_ORDER
                    if (kr == 0 && mr == 0 && nr == 0 && rb == 0) {
                        for (int i = 0; i < 16; i++) {
                            klog("B0  ", (int32_t)B_ptr[i]);
                        }
                    }
#endif

                    for (int ra = 0; ra < num_a_rounds; ra++) {
                        for (int i = 0; i < rows_per_round; i++) {
                            for (int j = 0; j < cols_per_round; j++) {
                                int16_t sum = 0;
                                for (int k = 0; k < eff_k; k++)
                                    sum += (int16_t)all_A[(ra * rows_per_round + i) * eff_k + k] *
                                           (int16_t)B_ptr[j * eff_k + k];
                                accum[(ra * rows_per_round + i) * data_cols + nr * tile_cols + rb * cols_per_round +
                                      j] += sum;
                            }
                        }
                    }

                    release_input_window(win_b);
                }
            }
        }

        for (int i = 0; i < tile_rows * data_cols; i++) {
            int16_t val = accum[i];
            if (val > 127)
                val = 127;
            else if (val < -128)
                val = -128;
            local_out[i] = (int8_t)val;
        }

        for (int nr = 0; nr < n_rounds; nr++) {
            for (int rc = 0; rc < num_c_rounds; rc++) {
                int8_t *out = (int8_t *)acquire_output_window(win_c);
                const int rows_per_c_round = buf_sz_c / tile_cols;
                for (int i = 0; i < rows_per_c_round; i++) {
                    for (int j = 0; j < tile_cols; j++) {
                        out[i * tile_cols + j] =
                            local_out[(rc * rows_per_c_round + i) * data_cols + nr * tile_cols + j];
                    }
                }
#if DEBUG_OUTPUT_ORDER
                if (rc == 0) {
                    for (int l = 0; l < 8; l++) {
                        klog("C0 ", (int32_t)out[l]);
                    }
                }
#endif
                release_output_window(win_c);
            }
        }
    }
}

#ifndef MATMUL_LAUNCHES
#define MATMUL_LAUNCHES 2
#endif

struct PerfSnap {
    unsigned long long ph[4], wio, kelf, kfill, krst, plan, sync, pmap;
    unsigned int pmapn;
    int aot;
};

static void perf_snap(PerfSnap *s) {
    *s = PerfSnap{};
    __Runtime_phase_cycles(s->ph, NULL);
    __Runtime_wait_io_cycles(&s->wio, NULL);
    __Runtime_kload_split_cycles(&s->kelf, NULL, &s->krst, NULL);
    __Runtime_setup_split_cycles(&s->plan, NULL, &s->sync, NULL);
    __Runtime_pmap_print_cycles(&s->pmap, &s->pmapn);
    __Runtime_kload_fill_cycles(&s->kfill, NULL, &s->aot);
}

static void perf_print(const char *tag, int launch, double ms, const PerfSnap *a, const PerfSnap *b) {
    printf("%s wall_ms=%.3f ctrlpkt=%s sync=%llu plan=%llu kload=%llu elf=%llu elf_fill=%llu rst=%llu "
           "bdcfg=%llu coreen=%llu startio=%llu wait_io=%llu pmap_print=%llu launch=%d\n",
           tag, ms, b->aot ? "aot" : "jit", b->sync - a->sync, b->plan - a->plan, b->ph[0] - a->ph[0],
           b->kelf - a->kelf, b->kfill - a->kfill, b->krst - a->krst, b->ph[1] - a->ph[1], b->ph[2] - a->ph[2],
           b->ph[3] - a->ph[3], b->wio - a->wio, b->pmap - a->pmap, launch);
}

int main() {
    __Runtime_ctrl_high_throughput_enable(1);
    __Runtime_ctrl_pmap_enable(1);
    printf("=== Matrix Multiply CTRL-PKT %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8\n", M, N, M, K, K, N);
    __ps_pmccntr_enable();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, HW_COLS, 0, 6}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *Cs[MATMUL_LAUNCHES];
    for (int l = 0; l < MATMUL_LAUNCHES; l++) {
        Cs[l] = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
        for (int i = 0; i < M * N; i++)
            Cs[l][i] = 0;
    }
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    double wall_ms[MATMUL_LAUNCHES];
    PerfSnap snap[MATMUL_LAUNCHES + 1];
    perf_snap(&snap[0]);
    for (int l = 0; l < MATMUL_LAUNCHES; l++) {
        int8_t *C = Cs[l];
        XTime t_start, t_end;
        XTime_GetTime(&t_start);
        matmul<<<mesh>>>(A, B, C, M, N, K);
        XTime_GetTime(&t_end);
        wall_ms[l] = 1.0 * (t_end - t_start) / COUNTS_PER_SECOND * 1000.0;
        perf_snap(&snap[l + 1]);
        printf("aie matmul time: %.3f ms launch=%d\n", wall_ms[l], l + 1);
        perf_print("[PERF] variant=ctrl_pkt", l + 1, wall_ms[l], &snap[l], &snap[l + 1]);
    }
    for (int l = 0; l < MATMUL_LAUNCHES; l++)
        perf_print("[FINAL_PERF]", l + 1, wall_ms[l], &snap[l], &snap[l + 1]);
    int result = 0;
    for (int l = 1; l < MATMUL_LAUNCHES; l++) {
        int diff = 0;
        for (int i = 0; i < M * N; i++) {
            if (Cs[l][i] == Cs[0][i])
                continue;
            if (diff < 8)
                printf("[RELAUNCH] launch=%d C[%d][%d] = %d, launch 1 = %d\n", l + 1, i / N, i % N, (int)Cs[l][i],
                       (int)Cs[0][i]);
            diff++;
        }
        printf("[RELAUNCH] launch=%d differs_from_launch1=%d of %d\n", l + 1, diff, M * N);
        result |= diff != 0;
    }
    if (result)
        printf("FAIL: a relaunch output differs from launch 1\n");
#ifndef DEBUG_NOCOMPUTE
    result |= verify_matmul(A, B, Cs[0]);
#else
    printf("test end=----------------------------------%.3f ms\n", wall_ms[0]);
#endif
    device.free(A);
    device.free(B);
    for (int l = 0; l < MATMUL_LAUNCHES; l++)
        device.free(Cs[l]);
    return result;
}
