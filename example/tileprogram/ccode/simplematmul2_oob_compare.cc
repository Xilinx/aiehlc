/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE OOB Comparison Benchmark — uint32 GEMM, 8 matrices.
 *
 * Matches the OOB example_oob_4x4 workload parameters exactly:
 *   - Data type:    uint32 (same as OOB)
 *   - Matrix size:  256x256x256 per matrix  (OOB PANEL_SIZE=256, NUM_PANELS=1)
 *   - Tile array:   4x4 = 16 compute tiles  (OOB NUM_HW_ROWS=4, NUM_HW_COLS=4)
 *   - Matrices:     8 back-to-back GEMMs    (OOB NUM_MATRICES=8)
 *   - GFLOPS:       8 * 2 * 256^3 / wall_s / 1e9 (exact OOB formula)
 *
 * exp60 — Direct port of OOB mmul_gemm.h to aiehlc DMA window model.
 *
 * Kernel: aie::mmul<4,4,4,uint32,uint32> with in-kernel B staging (OOB pattern).
 *   - B delivered col-major via ColBB GemmSpace (same as aiehlc baseline)
 *   - In-kernel B staging: col-major → K-major bbuf[k*TILE_N+n] (OOB one_input.cc)
 *   - A cached locally per K-round (OOB caches full A stripe; here per KCHUNK window)
 *   - No host-side B pre-transpose (fixes exp59 DMA hang from KmajBB)
 *   - mmul<4,4,4> shape: MM_M=4, MM_K=4, MM_N=4
 *     A tile: 4×4 uint32 (16 elem) = concat(r0,r1,r2,r3) each load_v<4>
 *     B tile: 4×4 uint32 (16 elem) = bbuf K-major contiguous load_v<16>
 *     C tile: 4×4 acc64 → to_vector<uint64_t>
 *   - Outer loops: MM_M_SUB=TILE_M/4 (per-tile M groups), MM_N_SUB=TILE_N/4 (N groups)
 *     MM_K_STEPS=KCHUNK/4 (K-steps per window)
 *
 * Primary metric: wall GFLOPS (same as OOB) + PMCCNTR cycles per matrix.
 * [PERF] summary block matches simplematmul2_prof.cc format for easy grep.
 ******************************************************************************/
#define M 256
#define K 256
#define N 256
#define HW_ROWS 4
#define HW_COLS 4
#define NUM_MATRICES 8
#define TILE_M 16
#define TILE_N 16
#define KCHUNK 64
#define MM_M 4
#define MM_N 4
#define MM_K 4
#define MM_M_SUB (TILE_M / MM_M)
#define MM_N_SUB (TILE_N / MM_N)
#define MM_K_STEPS (KCHUNK / MM_K)
#include "simplematmul.h"
#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER |           \
                        AIE_DEBUG_FLAG_CORE_PERF_COUNTER)

extern void __Runtime_core_perf_read_probe(uint32_t *active, uint32_t *vec_instr, uint32_t *stream_stall,
                                           uint32_t *lock_stall);
extern void __Runtime_perfcnt_read_mm2s_probe(uint32_t *ch0, uint32_t *ch1);
extern int __Runtime_core_perf_probe_valid(void);
extern void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
extern void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
extern void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc,
                                         unsigned int *rst_n);
extern void __Runtime_wait_io_iters(unsigned long long *iters);

constexpr aie::GemmSpace RowBA = {.policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                  .d1 = {.fullsize = M, .tile_size = TILE_M, .stride = TILE_M},
                                  .d2 = {.fullsize = K, .tile_size = KCHUNK, .stride = KCHUNK}};
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                  .d1 = {.fullsize = N, .tile_size = TILE_N, .stride = TILE_N},
                                  .d2 = {.fullsize = K, .tile_size = KCHUNK, .stride = KCHUNK}};
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = TILE_M, .stride = TILE_M},
    .d2 = {.fullsize = N, .tile_size = TILE_N, .stride = TILE_N}};

__global__ void matmul(aie::port<input_window_int32 *, RowBA> win_a, aie::port<input_window_int32 *, ColBB> win_b,
                       aie::port<output_window_int32 *, LtoR_Merge> win_c) {
    const int k_rounds = aie::get_k_rounds();
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);

    alignas(32) static uint32_t all_A[TILE_M * KCHUNK];

    alignas(64) static uint32_t bbuf[KCHUNK * TILE_N];

    static uint64_t acc_buf[TILE_M * TILE_N];

    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);
    klog("krnds", k_rounds);
    klog("mrnds", m_rounds);
    klog("nrnds", n_rounds);
    klog("narnds", num_a_rounds);
    klog("nbrnds", num_b_rounds);
    klog("ncrnds", num_c_rounds);
    klog("bsza ", buf_sz_a);
    klog("bszb ", buf_sz_b);
    klog("bszc ", buf_sz_c);

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        for (int idx = 0; idx < TILE_M * TILE_N; idx++)
            acc_buf[idx] = 0;

        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                uint32_t *A_ptr = (uint32_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                uint32_t *B_ptr = (uint32_t *)acquire_input_window(win_b);
                for (int n = 0; n < TILE_N; n++) {
                    int ns = n / MM_N, nl = n % MM_N;
                    for (int ki = 0; ki < KCHUNK; ki++) {
                        int ks = ki / MM_K, kl = ki % MM_K;
                        bbuf[((ns * MM_K_STEPS + ks) * MM_K + kl) * MM_N + nl] = B_ptr[n * KCHUNK + ki];
                    }
                }
                release_input_window(win_b);
            }

            using MMUL = aie::mmul<MM_M, MM_K, MM_N, uint32_t, uint32_t>;
            for (int ms = 0; ms < MM_M_SUB; ms++) {
                for (int ns = 0; ns < MM_N_SUB; ns++) {
                    MMUL C;
                    for (int ks = 0; ks < MM_K_STEPS; ks++)
                        chess_prepare_for_pipelining {
                            aie::vector<uint32_t, MM_K> ar0 =
                                aie::load_v<MM_K>(all_A + (ms * MM_M + 0) * KCHUNK + ks * MM_K);
                            aie::vector<uint32_t, MM_K> ar1 =
                                aie::load_v<MM_K>(all_A + (ms * MM_M + 1) * KCHUNK + ks * MM_K);
                            aie::vector<uint32_t, MM_K> ar2 =
                                aie::load_v<MM_K>(all_A + (ms * MM_M + 2) * KCHUNK + ks * MM_K);
                            aie::vector<uint32_t, MM_K> ar3 =
                                aie::load_v<MM_K>(all_A + (ms * MM_M + 3) * KCHUNK + ks * MM_K);
                            aie::vector<uint32_t, MM_M * MM_K> av = aie::concat(ar0, ar1, ar2, ar3);
                            aie::vector<uint32_t, MM_K * MM_N> bv =
                                aie::load_v<MM_K * MM_N>(bbuf + (ns * MM_K_STEPS + ks) * (MM_K * MM_N));
                            if (ks == 0)
                                C.mul(av, bv);
                            else
                                C.mac(av, bv);
                        }
                    aie::vector<uint32_t, MM_M * MM_N> cv = C.template to_vector<uint32_t>();
                    for (int mrow = 0; mrow < MM_M; mrow++)
                        for (int nn = 0; nn < MM_N; nn++)
                            acc_buf[(ms * MM_M + mrow) * TILE_N + (ns * MM_N + nn)] +=
                                (uint64_t)cv.get(mrow * MM_N + nn);
                }
            }
        }

        for (int rc = 0; rc < num_c_rounds; rc++) {
            uint32_t *out = (uint32_t *)acquire_output_window(win_c);
            for (int idx = 0; idx < TILE_M * TILE_N; idx++)
                out[idx] = (uint32_t)acc_buf[idx];
            release_output_window(win_c);
        }
    }
}

static int oob_verify(const uint32_t *A, const uint32_t *B, const uint32_t *C) {
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            uint64_t s = 0;
            for (int k = 0; k < K; k++)
                s += (uint64_t)A[i * K + k] * (uint64_t)B[j * K + k];
            uint32_t expected = (uint32_t)s;
            if (C[i * N + j] != expected) {
                if (mismatches < 8)
                    printf("  mismatch C[%d,%d] got %u exp %u\n", i, j, C[i * N + j], expected);
                mismatches++;
            }
        }
    }
    if (mismatches == 0)
        printf("RESULT: PASS (all %d elements match)\n", M * N);
    else
        printf("RESULT: FAIL (%d / %d mismatches)\n", mismatches, M * N);
    return mismatches;
}

int main() {
    printf("\n=== aiehlc OOB Comparison Benchmark exp61 (uint32, 8 matrices, mmul<4,4,4> MATRIX-ENGINE, col-major B "
           "in-kernel staged) ===\n");
    printf("  C[%dx%d] = A[%dx%d] * B^T[%dx%d], uint32, %dx%d mesh (%d tiles), %d matrices\n", M, N, M, K, N, K,
           HW_ROWS, HW_COLS, HW_ROWS * HW_COLS, NUM_MATRICES);
    printf("  OOB reference: example_oob_4x4 (PANEL_SIZE=256, NUM_MATRICES=8, aie::mmul<4,4,4,uint32,uint32>)\n");
    printf("  Kernel: aie::mmul<4,4,4,uint32,uint32> (B staged K-major in-kernel, OOB one_input.cc pattern)\n");

    __ps_pmccntr_enable();
    unsigned long long pc_init0 = __ps_pmccntr();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, 3, 0, 6}, HW_ROWS, HW_COLS);
    unsigned long long pc_init1 = __ps_pmccntr();

    unsigned long long pc_setup0 = __ps_pmccntr();
    uint32_t *A = (uint32_t *)device.alloc(M * K * sizeof(uint32_t));
    uint32_t *B = (uint32_t *)device.alloc(K * N * sizeof(uint32_t));
    uint32_t *C = (uint32_t *)device.alloc(M * N * sizeof(uint32_t));

    for (int i = 0; i < M * K; i++)
        A[i] = (uint32_t)(i % 7);

    for (int i = 0; i < K * N; i++)
        B[i] = (uint32_t)(i % 5);

    extern void __Runtime_sync_for_dev(XAie_DevInst * dev, void *ptr, __SIZE_TYPE__ size);
    for (int i = 0; i < M * N; i++)
        C[i] = (uint32_t)0x5A5A5A5AU;
    __Runtime_sync_for_dev(device._dev, C, M * N * sizeof(uint32_t));
    printf("[oob] poisoned C and flushed to DDR\n");

    static uint32_t golden[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            uint64_t s = 0;
            for (int k = 0; k < K; k++)
                s += (uint64_t)A[i * K + k] * (uint64_t)B[j * K + k];
            golden[i * N + j] = (uint32_t)s;
        }
    }
    unsigned long long pc_setup1 = __ps_pmccntr();

    const uint64_t MAX_POLL = 500000000ULL;
    XTime t0, t1;
    unsigned long long cv0 = __ps_cntvct();
    unsigned long long pc0 = __ps_pmccntr();
    XTime_GetTime(&t0);

    for (int mat = 0; mat < NUM_MATRICES; mat++) {
        matmul<<<mesh>>>(A, B, C, M, N, K);
    }

    unsigned long long pc_mid = __ps_pmccntr();
    uint64_t polls = 0;
    int complete = 0;
    unsigned long long poll_sync_cyc = 0ULL, poll_cmp_cyc = 0ULL;
    do {
        unsigned long long __ps0 = __ps_pmccntr();
        device.synchronizecpu(C, M * N * sizeof(uint32_t));
        poll_sync_cyc += (__ps_pmccntr() - __ps0);
        unsigned long long __pc0 = __ps_pmccntr();
        complete = 1;
        for (int idx = 0; idx < M * N; idx++) {
            if (C[idx] != golden[idx]) {
                complete = 0;
                break;
            }
        }
        poll_cmp_cyc += (__ps_pmccntr() - __pc0);
        polls++;
    } while (!complete && polls < MAX_POLL);
    XTime_GetTime(&t1);
    unsigned long long cv1 = __ps_cntvct();
    unsigned long long pc1 = __ps_pmccntr();

    if (!complete)
        printf("  WARNING: completion barrier hit MAX_POLL without full result\n");

    uint64_t raw_counts = (uint64_t)(t1 - t0);
    uint64_t timer_hz = (uint64_t)COUNTS_PER_SECOND;
    double wall_ms = 1000.0 * (double)raw_counts / (double)timer_hz;
    double wall_us = 1.0e6 * (double)raw_counts / (double)timer_hz;
    double tick_ns = 1.0e9 / (double)timer_hz;
    double total_flops = (double)NUM_MATRICES * 2.0 * (double)M * (double)N * (double)K;
    double gflops_wall = (wall_ms > 0.0) ? total_flops / (wall_ms * 1e-3) / 1e9 : 0.0;
    double gflops_per_mat = gflops_wall / NUM_MATRICES;

    uint32_t active = 0, vec = 0, sstall = 0, lstall = 0, mm0 = 0, mm1 = 0;
    int have_core = __Runtime_core_perf_probe_valid();
    __Runtime_core_perf_read_probe(&active, &vec, &sstall, &lstall);
    __Runtime_perfcnt_read_mm2s_probe(&mm0, &mm1);

    double total_budget = (double)active + (double)sstall + (double)lstall;
    double compute_pct = total_budget ? 100.0 * (double)active / total_budget : 0.0;
    double stream_pct = total_budget ? 100.0 * (double)sstall / total_budget : 0.0;
    double lock_pct = total_budget ? 100.0 * (double)lstall / total_budget : 0.0;
    double vec_util = active ? 100.0 * (double)vec / (double)active : 0.0;

    const double DEVICE_INT8_TOPS = 184.0;
    const int DEVICE_TILES = 144;
    int array_tiles = HW_ROWS * HW_COLS;
    double array_peak_int32_gops = (DEVICE_INT8_TOPS / 4.0) * 1000.0 * (double)array_tiles / (double)DEVICE_TILES;
    double util_pct = array_peak_int32_gops ? 100.0 * gflops_wall / array_peak_int32_gops : 0.0;

    printf("\n--- Layer 0: pre-launch setup (outside timed window) ---\n");
    printf("  [pmccntr] device_init: %llu cyc\n", (unsigned long long)(pc_init1 - pc_init0));
    printf("  [pmccntr] data_setup:  %llu cyc  (alloc + A/B init + B-transpose + poison-C + golden)\n",
           (unsigned long long)(pc_setup1 - pc_setup0));

    printf("\n--- Layer 1: PS wall-clock (%d matrices, OOB-formula GFLOPS) ---\n", NUM_MATRICES);
    printf("  raw counts:        %llu\n", (unsigned long long)raw_counts);
    printf("  timer freq:        %llu Hz  (1 tick = %.3f ns)\n", (unsigned long long)timer_hz, tick_ns);
    printf("  total time:        %.6f ms  (%.3f us)\n", wall_ms, wall_us);
    printf("  per-matrix time:   %.6f ms\n", wall_ms / NUM_MATRICES);
    printf("  completion polls:  %llu\n", (unsigned long long)polls);
    printf("  wall GFLOPS:       %.3f GOPS  (%d matrices, OOB formula: N*2*M*N*K/wall_s/1e9)\n", gflops_wall,
           NUM_MATRICES);
    printf("  per-matrix GFLOPS: %.3f GOPS  (= wall GFLOPS / %d)\n", gflops_per_mat, NUM_MATRICES);
    {
        unsigned long long cv_raw = (cv1 >= cv0) ? (cv1 - cv0) : 0ULL;
        unsigned long long cv_hz = __ps_cntfrq();
        double cv_ms = cv_hz ? 1000.0 * (double)cv_raw / (double)cv_hz : 0.0;
        printf("  [cntvct] raw: %llu counts  freq: %llu Hz  wall: %.6f ms\n", cv_raw, cv_hz, cv_ms);
    }
    {
        unsigned long long pc_raw = (pc1 >= pc0) ? (pc1 - pc0) : 0ULL;
        unsigned long long pmcr = __ps_pmcr();
        unsigned int d_bit = (unsigned int)((pmcr >> 3) & 1ULL);
        printf("  [pmccntr] raw: %llu cycles  pmcr:0x%llx (D=%u, %s)\n", pc_raw, pmcr, d_bit,
               d_bit ? "counts=CPUcyc/64" : "counts=CPUcyc");
        printf("  [pmccntr] per-matrix: %llu cycles\n", pc_raw / NUM_MATRICES);

        unsigned long long pc_launch = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        unsigned long long pc_poll = (pc1 >= pc_mid) ? (pc1 - pc_mid) : 0ULL;
        printf("  [pmccntr] launch: %llu cycles  poll: %llu cycles\n", pc_launch, pc_poll);

        unsigned long long wio_cyc = 0ULL;
        unsigned int wio_calls = 0U;
        __Runtime_wait_io_cycles(&wio_cyc, &wio_calls);
        double wio_pct = (pc_launch > 0) ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0;
        printf("  [phase] wait_io: %llu cycles over %u calls  (=%.1f%% of launch)\n", wio_cyc, wio_calls, wio_pct);

        unsigned long long wio_iters = 0ULL;
        __Runtime_wait_io_iters(&wio_iters);
        printf("  [wait_io] poll iters: %llu  (%.1f cyc/iter)\n", wio_iters,
               wio_iters ? (double)wio_cyc / (double)wio_iters : 0.0);

        double sync_pct = (pc_poll > 0) ? 100.0 * (double)poll_sync_cyc / (double)pc_poll : 0.0;
        printf("  [poll] synchronizecpu: %llu cyc over %llu calls  (=%.1f%% of poll)\n", poll_sync_cyc,
               (unsigned long long)polls, sync_pct);

        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        const char *phn[4] = {"kload  ", "bdcfg  ", "coreen ", "startio"};
        for (int i = 0; i < 4; i++) {
            double p = (pc_launch > 0) ? 100.0 * (double)ph[i] / (double)pc_launch : 0.0;
            printf("  [phase] %s: %llu cycles over %u calls  (=%.1f%% of launch)\n", phn[i], ph[i], phc[i], p);
        }

        unsigned long long kelf = 0ULL, krst = 0ULL;
        unsigned int kelfn = 0U, krstn = 0U;
        __Runtime_kload_split_cycles(&kelf, &kelfn, &krst, &krstn);
        double kep = (ph[0] > 0) ? 100.0 * (double)kelf / (double)ph[0] : 0.0;
        double krp = (ph[0] > 0) ? 100.0 * (double)krst / (double)ph[0] : 0.0;
        printf("  [kload] loadelf: %llu cyc  (=%.1f%% of kload)\n", kelf, kep);
        printf("  [kload] corerst: %llu cyc  (=%.1f%% of kload)\n", krst, krp);

        unsigned long long ph_total = ph[0] + ph[1] + ph[2] + ph[3] + wio_cyc;
        unsigned long long unacct = (pc_launch > ph_total) ? (pc_launch - ph_total) : 0ULL;
        printf("  [launch] unaccounted (lock_init+glue): %llu cyc  (=%.1f%%)\n", unacct,
               pc_launch ? 100.0 * (double)unacct / (double)pc_launch : 0.0);
        printf("  [launch] BUDGET (%d matrices):\n", NUM_MATRICES);
        printf("    kload    %10llu  (%.1f%%)\n", ph[0], pc_launch ? 100.0 * (double)ph[0] / (double)pc_launch : 0.0);
        printf("    bdcfg    %10llu  (%.1f%%)\n", ph[1], pc_launch ? 100.0 * (double)ph[1] / (double)pc_launch : 0.0);
        printf("    lockinit %10llu  (%.1f%%)\n", unacct, pc_launch ? 100.0 * (double)unacct / (double)pc_launch : 0.0);
        printf("    startio  %10llu  (%.1f%%)\n", ph[3], pc_launch ? 100.0 * (double)ph[3] / (double)pc_launch : 0.0);
        printf("    coreen   %10llu  (%.1f%%)\n", ph[2], pc_launch ? 100.0 * (double)ph[2] / (double)pc_launch : 0.0);
        printf("    wait_io  %10llu  (%.1f%%)\n", wio_cyc,
               pc_launch ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0);
        printf("    TOTAL    %10llu  (launch=%llu)\n", ph_total + unacct, pc_launch);
    }

    printf("\n--- Layer 2: DMA stream ---\n");
    printf("  MM2S ch0 BDs done: %u\n", mm0);
    printf("  MM2S ch1 BDs done: %u\n", mm1);

    printf("\n--- Layer 3: AIE core tile cycle budget ---\n");
    if (!have_core)
        printf("  [no probe tile armed]\n");
    printf("  total budget:    %.0f cycles\n", total_budget);
    printf("  active:          %u  (%.2f%%)\n", active, compute_pct);
    printf("  stream stall:    %u  (%.2f%%)\n", sstall, stream_pct);
    printf("  lock stall:      %u  (%.2f%%)\n", lstall, lock_pct);
    printf("  vector instrs:   %u\n", vec);
    printf("  vec utilization: %.1f%%\n", vec_util);

    printf("\n--- Hardware utilization (INT32, OOB yardstick) ---\n");
    printf("  array INT32 peak: %.1f GOPS  (INT8_peak/4 * %d/%d tiles)\n", array_peak_int32_gops, array_tiles,
           DEVICE_TILES);
    printf("  measured (wall):  %.3f GOPS  ->  %.4f%% of array INT32 peak\n", gflops_wall, util_pct);
    printf("  OOB-equivalent:   %.3f GOPS (8 matrices, OOB formula)\n", gflops_wall);

    printf("\n--- Correctness ---\n");
    unsigned long long pc_verify0 = __ps_pmccntr();
    int result = oob_verify(A, B, C);
    unsigned long long pc_verify1 = __ps_pmccntr();

    device.free(A);
    device.free(B);
    device.free(C);

    printf("  [pmccntr] verify: %llu cyc\n", (unsigned long long)(pc_verify1 - pc_verify0));

    {
        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        unsigned long long wio_cyc2 = 0ULL;
        unsigned int wio_calls2 = 0U;
        __Runtime_wait_io_cycles(&wio_cyc2, &wio_calls2);
        unsigned long long pc_launch2 = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        unsigned long long pc_raw2 = (pc1 >= pc0) ? (pc1 - pc0) : 0ULL;
        double kp = pc_launch2 ? 100.0 * (double)ph[0] / (double)pc_launch2 : 0.0;
        double bp = pc_launch2 ? 100.0 * (double)ph[1] / (double)pc_launch2 : 0.0;
        double wp = pc_launch2 ? 100.0 * (double)wio_cyc2 / (double)pc_launch2 : 0.0;
        double vp = active ? 100.0 * (double)vec / (double)active : 0.0;
        double tb2 = (double)active + (double)sstall + (double)lstall;
        double lsp = tb2 ? 100.0 * (double)lstall / tb2 : 0.0;
        double ssp = tb2 ? 100.0 * (double)sstall / tb2 : 0.0;
        unsigned long long ph_tot = ph[0] + ph[1] + ph[2] + ph[3] + wio_cyc2;
        unsigned long long unacct2 = pc_launch2 > ph_tot ? pc_launch2 - ph_tot : 0ULL;
        printf("\n[PERF] exp=oob_mmul61 dtype=uint32 matrices=%d kernel=mmul<4,4,4,uint32,uint32> B=col-major-staged\n",
               NUM_MATRICES);
        printf("[PERF] launch_cyc=%llu\n", pc_launch2);
        printf("[PERF] launch_cyc_per_mat=%llu\n", pc_launch2 / NUM_MATRICES);
        printf("[PERF] total_cyc=%llu\n", pc_raw2);
        printf("[PERF] total_cyc_per_mat=%llu\n", pc_raw2 / NUM_MATRICES);
        printf("[PERF] wall_ms=%.6f\n", wall_ms);
        printf("[PERF] gflops_wall=%.3f\n", gflops_wall);
        printf("[PERF] gflops_per_mat=%.3f\n", gflops_per_mat);
        printf("[PERF] util_pct_int32=%.4f\n", util_pct);
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
        printf("[PERF] vec_instr=%u vec_util_pct=%.1f\n", vec, vp);
        printf("[PERF] lock_stall_pct=%.1f stream_stall_pct=%.1f\n", lsp, ssp);
        printf("[PERF] result=%s\n", result == 0 ? "PASS" : "FAIL");
    }

    printf("\n[prof] device_teardown done\n");
    return result;
}
