/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#define M 256
#define K 256
#define N 256
#define HW_ROWS 4
#define HW_COLS 4
#define TILE_M 64
#define TILE_N 64
#define KCHUNK 64
#define NUM_MATRICES 1
#define PASSTHROUGH_KERNEL 0
#define PASSTHROUGH_VECTORIZED 1
#include "simplematmul.h"
#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER |           \
                        AIE_DEBUG_FLAG_CORE_PERF_COUNTER)

extern void __Runtime_core_perf_read_probe(uint32_t *active, uint32_t *vec_instr, uint32_t *stream_stall,
                                           uint32_t *lock_stall);
extern void __Runtime_perfcnt_read_mm2s_probe(uint32_t *ch0, uint32_t *ch1);
extern int __Runtime_core_perf_probe_valid(void);
extern void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
extern void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
extern void __Runtime_bd_subphase_cycles(unsigned long long *init_cyc, unsigned int *init_n,
                                         unsigned long long *write_cyc, unsigned int *write_n);
extern void __Runtime_bd_midtail_cycles(unsigned long long *mid_cyc, unsigned int *mid_n, unsigned long long *tail_cyc,
                                        unsigned int *tail_n);
extern void __Runtime_bd_mid3_cycles(unsigned long long *gtt_cyc, unsigned int *gtt_n, unsigned long long *saddr_cyc,
                                     unsigned int *saddr_n, unsigned long long *en_cyc, unsigned int *en_n);
extern void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc,
                                         unsigned int *rst_n);
extern void __Runtime_wait_io_iters(unsigned long long *iters);

constexpr aie::GemmSpace RowBA = {.policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{8192}}},
                                  .d1 = {.fullsize = M, .tile_size = TILE_M, .stride = TILE_M},
                                  .d2 = {.fullsize = K, .tile_size = KCHUNK, .stride = KCHUNK}};
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{8192}}},
                                  .d1 = {.fullsize = N, .tile_size = TILE_N, .stride = TILE_N},
                                  .d2 = {.fullsize = K, .tile_size = KCHUNK, .stride = KCHUNK}};
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{8192}}},
    .d1 = {.fullsize = M, .tile_size = TILE_M, .stride = TILE_M},
    .d2 = {.fullsize = N, .tile_size = TILE_N, .stride = TILE_N}};

__global__ void matmul(aie::port<input_window_int8 *, RowBA> win_a, aie::port<input_window_int8 *, ColBB> win_b,
                       aie::port<output_window_int8 *, LtoR_Merge> win_c) {
    const int tile_rows = aie::get_tile_rows();
    const int tile_cols = aie::get_tile_cols();
    const int eff_k = aie::get_effective_k();
    const int k_rounds = aie::get_k_rounds();
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_c = aie::get_buffer_size(win_c);
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);
    const int cols_per_round = aie::get_buffer_size(win_b) / eff_k;

    alignas(aie::vector_decl_align) int8_t all_A[tile_rows * eff_k];
#if PASSTHROUGH_KERNEL
    const int buf_sz_b = aie::get_buffer_size(win_b);
#if PASSTHROUGH_VECTORIZED
    constexpr int VW = 32;
    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        aie::vector<int8, VW> vbsum = aie::zeros<int8, VW>();
        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i += VW)
                    aie::store_v(all_A + ra * buf_sz_a + i, aie::load_v<VW>(A_ptr + i));
                release_input_window(win_a);
            }
            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                for (int i = 0; i < buf_sz_b; i += VW)
                    vbsum = aie::add(vbsum, aie::load_v<VW>(B_ptr + i));
                release_input_window(win_b);
            }
        }
        alignas(aie::vector_decl_align) int8_t local_out[tile_rows * tile_cols];
        for (int idx = 0; idx < tile_rows * tile_cols; idx += VW)
            aie::store_v(local_out + idx, aie::add(aie::load_v<VW>(all_A + idx), vbsum));
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            for (int i = 0; i < buf_sz_c; i += VW)
                aie::store_v(out + i, aie::load_v<VW>(local_out + rc * buf_sz_c + i));
            release_output_window(win_c);
        }
    }
#else
    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        int8_t bsum = 0;
        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }
            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                for (int i = 0; i < buf_sz_b; i++)
                    bsum = (int8_t)(bsum + B_ptr[i]);
                release_input_window(win_b);
            }
        }
        int8_t local_out[tile_rows * tile_cols];
        for (int idx = 0; idx < tile_rows * tile_cols; idx++)
            local_out[idx] = (int8_t)(all_A[idx] + bsum);
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++)
                for (int j = 0; j < tile_cols; j++)
                    out[i * tile_cols + j] = local_out[rc * rows_per_c_round * tile_cols + i * tile_cols + j];
            release_output_window(win_c);
        }
    }
#endif
#else
    int32_t acc_buf[tile_rows * tile_cols];

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        for (int idx = 0; idx < tile_rows * tile_cols; idx++)
            acc_buf[idx] = 0;

        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);

                for (int i = 0; i < tile_rows; i++) {
                    const int8_t *Arow = all_A + i * eff_k;
                    aie::vector<int8, 16> a0 = aie::load_v<16>(Arow);
                    aie::vector<int8, 16> a1 = aie::load_v<16>(Arow + 16);
                    aie::vector<int8, 16> a2 = aie::load_v<16>(Arow + 32);
                    aie::vector<int8, 16> a3 = aie::load_v<16>(Arow + 48);

                    for (int j = 0; j < cols_per_round; j++) {
                        const int8_t *Bcol = B_ptr + j * eff_k;
                        aie::accum<acc32, 16> acc0 = aie::mul(a0, aie::load_v<16>(Bcol));
                        aie::accum<acc32, 16> acc1 = aie::mul(a1, aie::load_v<16>(Bcol + 16));
                        aie::accum<acc32, 16> acc2 = aie::mul(a2, aie::load_v<16>(Bcol + 32));
                        aie::accum<acc32, 16> acc3 = aie::mul(a3, aie::load_v<16>(Bcol + 48));
                        aie::accum<acc32, 16> acc01 = aie::add(acc0, acc1);
                        aie::accum<acc32, 16> acc23 = aie::add(acc2, acc3);
                        aie::accum<acc32, 16> acc_all = aie::add(acc01, acc23);
                        acc_buf[i * tile_cols + j] += aie::reduce_add(acc_all.to_vector<int32>());
                    }
                }
                release_input_window(win_b);
            }
        }

        int8_t local_out[tile_rows * tile_cols];
        for (int idx = 0; idx < tile_rows * tile_cols; idx++) {
            int32_t v = acc_buf[idx];
            local_out[idx] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
        }

        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++)
                for (int j = 0; j < tile_cols; j++)
                    out[i * tile_cols + j] = local_out[rc * rows_per_c_round * tile_cols + i * tile_cols + j];
            release_output_window(win_c);
        }
    }
#endif
}

#define SPOT_STRIDE 4093
#define MAX_SPOTS ((M * N) / SPOT_STRIDE + 2)
static int g_spot_idx[MAX_SPOTS];
static int8_t g_spot_gold[MAX_SPOTS];
static int g_num_spots = 0;

static void build_spots(const int8_t *A, const int8_t *B) {
    g_num_spots = 0;
    for (int idx = 0; idx < M * N; idx += SPOT_STRIDE)
        g_spot_idx[g_num_spots++] = idx;
    if (g_num_spots == 0 || g_spot_idx[g_num_spots - 1] != M * N - 1)
        g_spot_idx[g_num_spots++] = M * N - 1;
    for (int s = 0; s < g_num_spots; s++) {
        int idx = g_spot_idx[s];
        int i = idx / N, j = idx % N;
        int16_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += (int16_t)A[i * K + k] * (int16_t)B[j * K + k];
        if (acc > 127)
            acc = 127;
        else if (acc < -128)
            acc = -128;
        g_spot_gold[s] = (int8_t)acc;
    }
}

static int prof_verify(const int8_t *C) {
    int mismatches = 0;
    for (int s = 0; s < g_num_spots; s++) {
        int idx = g_spot_idx[s];
        if (C[idx] != g_spot_gold[s]) {
            if (mismatches < 8)
                printf("  mismatch C[%d,%d] got %d exp %d\n", idx / N, idx % N, (int)C[idx], (int)g_spot_gold[s]);
            mismatches++;
        }
    }
    if (mismatches == 0)
        printf("RESULT: PASS (spot-check: all %d sampled of %d elements match)\n", g_num_spots, M * N);
    else
        printf("RESULT: FAIL (%d / %d spot mismatches)\n", mismatches, g_num_spots);
    return mismatches;
}

int main() {
    printf("\n=== aiehlc GEMM profiling ===\n");
    printf("  C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8, %dx%d mesh (%d tiles)\n", M, N, M, K, N, K, HW_ROWS, HW_COLS,
           HW_ROWS * HW_COLS);

    __ps_pmccntr_enable();
    unsigned long long pc_init0 = __ps_pmccntr();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, 3, 0, 5}, HW_ROWS, HW_COLS);
    unsigned long long pc_init1 = __ps_pmccntr();

    unsigned long long pc_setup0 = __ps_pmccntr();
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    extern void __Runtime_sync_for_dev(XAie_DevInst * dev, void *ptr, __SIZE_TYPE__ size);
    for (int i = 0; i < M * N; i++)
        C[i] = (int8_t)0x5A;
    __Runtime_sync_for_dev(device._dev, C, M * N * sizeof(int8_t) * 4);
    printf("[exp07] poisoned device C with 0x5A and flushed to DDR\n");

    build_spots(A, B);
    printf("[spot] verifying %d sampled outputs (stride %d) of %d total\n", g_num_spots, SPOT_STRIDE, M * N);

    unsigned long long pc_setup1 = __ps_pmccntr();

    const uint64_t MAX_POLL = 500000000ULL;
    XTime t0, t1;
    unsigned long long cv0 = __ps_cntvct();
    unsigned long long pc0 = __ps_pmccntr();
    XTime_GetTime(&t0);
    matmul<<<mesh>>>(A, B, C, M, N, K);
    unsigned long long pc_mid = __ps_pmccntr();
    uint64_t polls = 0;
    int complete = 0;
    unsigned long long poll_sync_cyc = 0ULL, poll_cmp_cyc = 0ULL;
    do {
        unsigned long long __ps0 = __ps_pmccntr();
        device.synchronizecpu(C, M * N * sizeof(int8_t) * 4);
        poll_sync_cyc += (__ps_pmccntr() - __ps0);
        unsigned long long __pc0 = __ps_pmccntr();
        complete = 1;
#if !PASSTHROUGH_KERNEL
        for (int s = 0; s < g_num_spots; s++) {
            if (C[g_spot_idx[s]] != g_spot_gold[s]) {
                complete = 0;
                break;
            }
        }
#endif
        poll_cmp_cyc += (__ps_pmccntr() - __pc0);
        polls++;
    } while (!complete && polls < MAX_POLL);
    XTime_GetTime(&t1);
    unsigned long long cv1 = __ps_cntvct();
    unsigned long long pc1 = __ps_pmccntr();
    if (!complete)
        printf("  WARNING: completion barrier hit MAX_POLL=%llu without full result\n", (unsigned long long)MAX_POLL);

    uint64_t raw_counts = (uint64_t)(t1 - t0);
    uint64_t timer_hz = (uint64_t)COUNTS_PER_SECOND;
    double wall_ms = 1000.0 * (double)raw_counts / (double)timer_hz;
    double wall_us = 1.0e6 * (double)raw_counts / (double)timer_hz;
    double tick_ns = 1.0e9 / (double)timer_hz;
    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_wall = (wall_ms > 0.0) ? total_flops / (wall_ms * 1e-3) / 1e9 : 0.0;

    uint32_t active = 0, vec = 0, sstall = 0, lstall = 0, mm0 = 0, mm1 = 0;
    int have_core = __Runtime_core_perf_probe_valid();
    __Runtime_core_perf_read_probe(&active, &vec, &sstall, &lstall);
    __Runtime_perfcnt_read_mm2s_probe(&mm0, &mm1);

    double tile_macs = (double)(M / HW_ROWS) * (double)(N / HW_COLS) * (double)K;
    double tile_flops = 2.0 * tile_macs;
    double total_budget = (double)active + (double)sstall + (double)lstall;
    double compute_pct = total_budget ? 100.0 * (double)active / total_budget : 0.0;
    double stream_pct = total_budget ? 100.0 * (double)sstall / total_budget : 0.0;
    double lock_pct = total_budget ? 100.0 * (double)lstall / total_budget : 0.0;
    double macs_per_vec = vec ? tile_macs / (double)vec : 0.0;
    (void)tile_flops;

    const double DEVICE_INT8_TOPS = 184.0;
    const int DEVICE_TILES = 144;
    int array_tiles = HW_ROWS * HW_COLS;
    double array_peak_gops = DEVICE_INT8_TOPS * 1000.0 * (double)array_tiles / (double)DEVICE_TILES;
    double util_pct = array_peak_gops ? 100.0 * gflops_wall / array_peak_gops : 0.0;

    printf("\n--- Layer 0: pre-launch setup (outside timed window) ---\n");
    printf("  [pmccntr] device_init: %llu cyc  (aieSetDevice + partition)\n",
           (unsigned long long)(pc_init1 - pc_init0));
    printf("  [pmccntr] data_setup:  %llu cyc  (alloc + A/B init + poison-C + golden compute)\n",
           (unsigned long long)(pc_setup1 - pc_setup0));

    printf("\n--- Layer 1: PS wall-clock (end-to-end launch) ---\n");
    printf("  raw counts:        %llu  (t1 - t0)\n", (unsigned long long)raw_counts);
    printf("  timer freq:        %llu Hz  (COUNTS_PER_SECOND; 1 tick = %.3f ns)\n", (unsigned long long)timer_hz,
           tick_ns);
    printf("  total time:        %.6f ms  (%.3f us)\n", wall_ms, wall_us);
    printf("  completion polls:  %llu  (DDR read-back until full result present)\n", (unsigned long long)polls);
    printf("  wall GFLOPS:       %.3f GOPS  (2*M*N*K / total_ms)\n", gflops_wall);
    printf("  note: launch -> full result in DDR (async launch + poll-to-result barrier)\n");
    {
        unsigned long long cv_raw = (cv1 >= cv0) ? (cv1 - cv0) : 0ULL;
        unsigned long long cv_hz = __ps_cntfrq();
        double cv_ms = cv_hz ? 1000.0 * (double)cv_raw / (double)cv_hz : 0.0;
        printf("  [cntvct] raw:      %llu counts  freq: %llu Hz  wall: %.6f ms\n", cv_raw, cv_hz, cv_ms);
    }
    {
        unsigned long long pc_raw = (pc1 >= pc0) ? (pc1 - pc0) : 0ULL;
        unsigned long long pmcr = __ps_pmcr();
        unsigned int d_bit = (unsigned int)((pmcr >> 3) & 1ULL);
        printf("  [pmccntr] raw:     %llu cycles  pmcr:0x%llx (D=%u, %s)\n", pc_raw, pmcr, d_bit,
               d_bit ? "counts=CPUcyc/64" : "counts=CPUcyc");
        unsigned long long pc_launch = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        unsigned long long pc_poll = (pc1 >= pc_mid) ? (pc1 - pc_mid) : 0ULL;
        printf("  [pmccntr] launch:  %llu cycles  poll: %llu cycles\n", pc_launch, pc_poll);
        unsigned long long wio_cyc = 0ULL;
        unsigned int wio_calls = 0U;
        __Runtime_wait_io_cycles(&wio_cyc, &wio_calls);
        double wio_pct = (pc_launch > 0) ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0;
        printf("  [phase] wait_io:   %llu cycles over %u calls  (=%.1f%% of launch)\n", wio_cyc, wio_calls, wio_pct);
        unsigned long long wio_iters = 0ULL;
        __Runtime_wait_io_iters(&wio_iters);
        double wio_cyc_per_iter = (wio_iters > 0) ? (double)wio_cyc / (double)wio_iters : 0.0;
        double wio_cyc_per_call = (wio_calls > 0) ? (double)wio_cyc / (double)wio_calls : 0.0;
        printf("  [wait_io] poll iters: %llu total  (%.1f cyc/iter avg, %.0f avg cyc/call)\n", wio_iters,
               wio_cyc_per_iter, wio_cyc_per_call);
        double sync_pct = (pc_poll > 0) ? 100.0 * (double)poll_sync_cyc / (double)pc_poll : 0.0;
        double cmp_pct = (pc_poll > 0) ? 100.0 * (double)poll_cmp_cyc / (double)pc_poll : 0.0;
        printf("  [poll] synchronizecpu: %llu cyc over %llu calls  (=%.1f%% of poll)\n", poll_sync_cyc,
               (unsigned long long)polls, sync_pct);
        printf("  [poll] compare (C vs golden): %llu cyc over %llu calls  (=%.1f%% of poll)\n", poll_cmp_cyc,
               (unsigned long long)polls, cmp_pct);
        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        const char *phn[4] = {"kload  ", "bdcfg  ", "coreen ", "startio"};
        for (int i = 0; i < 4; i++) {
            double p = (pc_launch > 0) ? 100.0 * (double)ph[i] / (double)pc_launch : 0.0;
            printf("  [phase] %s: %llu cycles over %u calls  (=%.1f%% of launch)\n", phn[i], ph[i], phc[i], p);
        }
        unsigned long long bi = 0ULL, bw = 0ULL;
        unsigned int bin = 0U, bwn = 0U;
        __Runtime_bd_subphase_cycles(&bi, &bin, &bw, &bwn);
        double bip = (ph[1] > 0) ? 100.0 * (double)bi / (double)ph[1] : 0.0;
        double bwp = (ph[1] > 0) ? 100.0 * (double)bw / (double)ph[1] : 0.0;
        printf("  [bdcfg] descinit: %llu cyc over %u  (=%.1f%% of bdcfg)\n", bi, bin, bip);
        printf("  [bdcfg] writebd:  %llu cyc over %u  (=%.1f%% of bdcfg)\n", bw, bwn, bwp);
        unsigned long long bmid = 0ULL, btail = 0ULL;
        unsigned int bmidn = 0U, btailn = 0U;
        __Runtime_bd_midtail_cycles(&bmid, &bmidn, &btail, &btailn);
        double bmp = (ph[1] > 0) ? 100.0 * (double)bmid / (double)ph[1] : 0.0;
        double btp = (ph[1] > 0) ? 100.0 * (double)btail / (double)ph[1] : 0.0;
        unsigned long long bacct = bi + bw + bmid + btail;
        double bap = (ph[1] > 0) ? 100.0 * (double)bacct / (double)ph[1] : 0.0;
        printf("  [bdcfg] mid:      %llu cyc over %u  (=%.1f%% of bdcfg)\n", bmid, bmidn, bmp);
        printf("  [bdcfg] tail:     %llu cyc over %u  (=%.1f%% of bdcfg)\n", btail, btailn, btp);
        printf("  [bdcfg] accounted (init+write+mid+tail): %llu cyc  (=%.1f%% of bdcfg)\n", bacct, bap);
        unsigned long long bgtt = 0ULL, bsa = 0ULL, ben = 0ULL;
        unsigned int bgttn = 0U, bsan = 0U, benn = 0U;
        __Runtime_bd_mid3_cycles(&bgtt, &bgttn, &bsa, &bsan, &ben, &benn);
        double gttp = (bmid > 0) ? 100.0 * (double)bgtt / (double)bmid : 0.0;
        double sap = (bmid > 0) ? 100.0 * (double)bsa / (double)bmid : 0.0;
        double enp = (bmid > 0) ? 100.0 * (double)ben / (double)bmid : 0.0;
        unsigned long long bres = (bmid > bgtt + bsa + ben) ? (bmid - bgtt - bsa - ben) : 0ULL;
        double resp = (bmid > 0) ? 100.0 * (double)bres / (double)bmid : 0.0;
        printf("  [mid] gettiletype: %llu cyc over %u  (=%.1f%% of mid)\n", bgtt, bgttn, gttp);
        printf("  [mid] setaddr:     %llu cyc over %u  (=%.1f%% of mid)\n", bsa, bsan, sap);
        printf("  [mid] enablebd:    %llu cyc over %u  (=%.1f%% of mid)\n", ben, benn, enp);
        printf("  [mid] residual (setlock/nextbd/pkt/ooo+printf): %llu cyc  (=%.1f%% of mid)\n", bres, resp);
        unsigned long long kelf = 0ULL, krst = 0ULL;
        unsigned int kelfn = 0U, krstn = 0U;
        __Runtime_kload_split_cycles(&kelf, &kelfn, &krst, &krstn);
        double kep = (ph[0] > 0) ? 100.0 * (double)kelf / (double)ph[0] : 0.0;
        double krp = (ph[0] > 0) ? 100.0 * (double)krst / (double)ph[0] : 0.0;
        printf("  [kload] loadelf:  %llu cyc over %u  (=%.1f%% of kload)\n", kelf, kelfn, kep);
        printf("  [kload] corerst:  %llu cyc over %u  (=%.1f%% of kload)\n", krst, krstn, krp);
        unsigned long long ph_total = ph[0] + ph[1] + ph[2] + ph[3] + wio_cyc;
        unsigned long long unacct = (pc_launch > ph_total) ? (pc_launch - ph_total) : 0ULL;
        double unacct_p = (pc_launch > 0) ? 100.0 * (double)unacct / (double)pc_launch : 0.0;
        printf("  [launch] unaccounted (lock_init+glue): %llu cyc  (=%.1f%% of launch)\n", unacct, unacct_p);
        printf("  [launch] BUDGET SUMMARY (all in cycles):\n");
        printf("    kload    %10llu  (%.1f%%)\n", ph[0],
               (pc_launch > 0) ? 100.0 * (double)ph[0] / (double)pc_launch : 0.0);
        printf("    bdcfg    %10llu  (%.1f%%)\n", ph[1],
               (pc_launch > 0) ? 100.0 * (double)ph[1] / (double)pc_launch : 0.0);
        printf("    lockinit %10llu  (%.1f%%) [unaccounted proxy]\n", unacct, unacct_p);
        printf("    startio  %10llu  (%.1f%%)\n", ph[3],
               (pc_launch > 0) ? 100.0 * (double)ph[3] / (double)pc_launch : 0.0);
        printf("    coreen   %10llu  (%.1f%%)\n", ph[2],
               (pc_launch > 0) ? 100.0 * (double)ph[2] / (double)pc_launch : 0.0);
        printf("    wait_io  %10llu  (%.1f%%)\n", wio_cyc,
               (pc_launch > 0) ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0);
        printf("    TOTAL    %10llu  (launch=%llu)\n", ph_total + unacct, pc_launch);
    }

    printf("\n--- Layer 2: DMA stream (probe tile MM2S BD finished) ---\n");
    printf("  MM2S ch0 BDs done: %u\n", mm0);
    printf("  MM2S ch1 BDs done: %u\n", mm1);

    printf("\n--- Layer 3: AIE core tile cycle budget (probe = first compute tile) ---\n");
    if (!have_core)
        printf("  [no probe tile armed]\n");
    printf("  core-state split (sampled window; ratios valid, absolute cycles are a sub-window):\n");
    printf("    compute:      %.2f%%  (%u cyc active/executing)\n", compute_pct, active);
    printf("    stream stall: %.2f%%  (%u cyc)  [waiting for window data]\n", stream_pct, sstall);
    printf("    lock stall:   %.2f%%  (%u cyc)  [waiting for buffer lock/DMA]\n", lock_pct, lstall);

    printf("\n--- Vector density (full-run, window-independent) ---\n");
    printf("  vector instrs:     %u  (INSTR_VECTOR over whole run)\n", vec);
    printf("  MACs/vector-instr: %.2f  (tile MACs %.3g / vec instrs; higher = denser vectorization)\n", macs_per_vec,
           tile_macs);
    printf("  note: an aie::mmul<4,16,8> retires ~64 MAC/op; a mul+reduce_add dot-product is much lower,\n");
    printf("        so a low value here flags the microkernel (not feed/locks) as the compute-side lever.\n");

    printf("\n--- Hardware utilization (INT8, same yardstick as AEG) ---\n");
    printf("  array INT8 peak:   %.1f GOPS  (%d/%d tiles of %.0f TOPS device)\n", array_peak_gops, array_tiles,
           DEVICE_TILES, DEVICE_INT8_TOPS);
    printf("  measured (wall):   %.3f GOPS  ->  %.4f %% of array peak\n", gflops_wall, util_pct);

    printf("\n--- Correctness ---\n");
    unsigned long long pc_verify0 = __ps_pmccntr();
#if PASSTHROUGH_KERNEL
    int result = 0;
    printf("RESULT: SKIP (passthrough kernel — feed/DMA floor measurement, no compute)\n");
    (void)prof_verify;
#else
    int result = prof_verify(C);
#endif
    unsigned long long pc_verify1 = __ps_pmccntr();

    unsigned long long pc_free0 = __ps_pmccntr();
    device.free(A);
    device.free(B);
    device.free(C);
    unsigned long long pc_free1 = __ps_pmccntr();

    printf("\n--- Layer 0 (post-launch): out-of-timed-window teardown ---\n");
    printf("  [pmccntr] verify:      %llu cyc  (prof_verify spot-check compare)\n",
           (unsigned long long)(pc_verify1 - pc_verify0));
    printf("  [pmccntr] device.free: %llu cyc  (free A+B+C)\n", (unsigned long long)(pc_free1 - pc_free0));

    {
        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        unsigned long long wio_cyc2 = 0ULL;
        unsigned int wio_calls2 = 0U;
        __Runtime_wait_io_cycles(&wio_cyc2, &wio_calls2);
        unsigned long long pc_launch2 = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        double kp = pc_launch2 ? 100.0 * (double)ph[0] / (double)pc_launch2 : 0.0;
        double bp = pc_launch2 ? 100.0 * (double)ph[1] / (double)pc_launch2 : 0.0;
        double wp = pc_launch2 ? 100.0 * (double)wio_cyc2 / (double)pc_launch2 : 0.0;
        double mpv = vec ? ((double)(M / HW_ROWS) * (double)(N / HW_COLS) * (double)K) / (double)vec : 0.0;
        double tb2 = (double)active + (double)sstall + (double)lstall;
        double lsp = tb2 ? 100.0 * (double)lstall / tb2 : 0.0;
        double ssp = tb2 ? 100.0 * (double)sstall / tb2 : 0.0;
        unsigned long long ph_tot = ph[0] + ph[1] + ph[2] + ph[3] + wio_cyc2;
        unsigned long long unacct2 = pc_launch2 > ph_tot ? pc_launch2 - ph_tot : 0ULL;
        printf("\n[PERF] launch_cyc=%llu\n", pc_launch2);
        printf("[PERF] kload_cyc=%llu kload_pct=%.1f\n", ph[0], kp);
        printf("[PERF] bdcfg_cyc=%llu bdcfg_pct=%.1f\n", ph[1], bp);
        printf("[PERF] lockinit_cyc=%llu lockinit_pct=%.1f\n", unacct2,
               pc_launch2 ? 100.0 * (double)unacct2 / (double)pc_launch2 : 0.0);
        printf("[PERF] coreen_cyc=%llu coreen_pct=%.1f\n", ph[2],
               pc_launch2 ? 100.0 * (double)ph[2] / (double)pc_launch2 : 0.0);
        printf("[PERF] startio_cyc=%llu startio_pct=%.1f\n", ph[3],
               pc_launch2 ? 100.0 * (double)ph[3] / (double)pc_launch2 : 0.0);
        printf("[PERF] wait_io_cyc=%llu wait_io_pct=%.1f\n", wio_cyc2, wp);
        printf("[PERF] core_active=%u core_sstall=%u core_lstall=%u\n", active, sstall, lstall);
        printf("[PERF] vec_instr=%u macs_per_vec=%.2f\n", vec, mpv);
        printf("[PERF] lock_stall_pct=%.1f stream_stall_pct=%.1f\n", lsp, ssp);
        printf("[PERF] result=%s\n", result == 0 ? "PASS" : "FAIL");
    }

    printf("\n[prof] device_teardown done\n");
    return result;
}
