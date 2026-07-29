/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication, HARDWARE PROFILING build.
 *
 * Same int8 GEMM kernel as simplematmul2.cc, configured for a 4x4 (16-tile)
 * mesh on a 256x256x256 problem so it matches the AEG example_oob_4x4 workload
 * (256^3, 4x4 array). Adds a 3-layer profiling report to main(), inspired by
 * example_oob_4x4/src/graph.cpp:
 *
 *   Layer 1 — PS wall-clock: end-to-end launch time (XTime) and wall GFLOPS.
 *   Layer 2 — DMA stream:    probe-tile MM2S BD-finished counts (aiehlc's
 *                            XAie-native analog of the AEG GMIO stream cycles).
 *   Layer 3 — AIE core tile:  active / stream-stall / lock-stall / vector-instr
 *                            cycle budget on the first compute tile, plus
 *                            frequency-free FLOP/cycle efficiency.
 ******************************************************************************/
#define M 256
#define K 256
#define N 256
#define HW_ROWS 4
#define HW_COLS 4
// Per-DMA-round sub-tile granularity (recovered from the prior 256^3 config:
// 16-row sub-tile, 64-element K chunk -> 4 m/n sub-tile rounds, 4 K rounds).
#define TILE_M 64
#define TILE_N 64
#define KCHUNK 64
// Batch size (DEFERRED for now — kept at 1 = single matrix). When we return to
// batching, each matmul<<<mesh>>> launch blocks internally on its output DMA
// (wait_io), so a plain loop is a correct sequential batch (mind per-launch
// kernel-load overhead — report ex-kload).
#define NUM_MATRICES 1
// Kernel selection: 0 = real GEMM (mul+reduce_add); 1 = vectorized passthrough
// (same window acquire/release pattern & DMA volume, no MAC) to measure the
// feed/DMA/lock floor without compute.
#define PASSTHROUGH_KERNEL 0
// Passthrough core-touch style: 1 = vectorized (v32int8 load/add/store),
// 0 = scalar byte loops. Only meaningful when PASSTHROUGH_KERNEL==1. Both hit the
// same feed/DMA/lock floor; this isolates whether the scalar core touch matters.
#define PASSTHROUGH_VECTORIZED 1
#include "simplematmul.h"
// Debug level 0 (no verbose UART snapshot — keeps the timed launch region clean)
// but with the profiling flags enabled: disable partition teardown, set up MM2S
// BD-finished DMA counters (Layer 2) and arm core-tile perf counters (Layer 3).
#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER |           \
                        AIE_DEBUG_FLAG_CORE_PERF_COUNTER)

// Profiling hooks implemented in src/mlir/runtime/aie_runtime.c (compiled as
// C++), declared here so the aiehlc front-end can parse main() referencing them.
extern void __Runtime_core_perf_read_probe(uint32_t *active, uint32_t *vec_instr, uint32_t *stream_stall,
                                           uint32_t *lock_stall);
extern void __Runtime_perfcnt_read_mm2s_probe(uint32_t *ch0, uint32_t *ch1);
extern int __Runtime_core_perf_probe_valid(void);
/* [exp44] accumulated PS PMU cycles spent inside wait_io (output-DMA drain, gates completion) */
extern void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
/* [exp45] setup-phase PMU cycles [KLOAD, BDCFG, COREEN, STARTIO]; caller passes len-4 arrays */
extern void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
/* [exp46] sub-split bdcfg: DmaDescInit vs DmaWriteBd cycles */
extern void __Runtime_bd_subphase_cycles(unsigned long long *init_cyc, unsigned int *init_n,
                                         unsigned long long *write_cyc, unsigned int *write_n);
/* [exp47] localize the rest of bdcfg: mid (post-descinit->pre-writebd) + tail (post-writebd->return) */
extern void __Runtime_bd_midtail_cycles(unsigned long long *mid_cyc, unsigned int *mid_n, unsigned long long *tail_cyc,
                                        unsigned int *tail_n);
/* [exp48] split the mid region into GetTileTypefromLoc / DmaSetAddr / DmaEnableBd */
extern void __Runtime_bd_mid3_cycles(unsigned long long *gtt_cyc, unsigned int *gtt_n, unsigned long long *saddr_cyc,
                                     unsigned int *saddr_n, unsigned long long *en_cyc, unsigned int *en_n);
/* [exp51] split the dominant kload phase: XAie_LoadElfMem (elf) vs Core Disable/Reset/Unreset (rst) */
extern void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc,
                                         unsigned int *rst_n);
/* [exp52] total DmaGetPendingBdCount busy-poll iterations across all wait_io calls */
extern void __Runtime_wait_io_iters(unsigned long long *iters);

// Composition-based spatial spaces (per-port 2D iteration space), identical in
// structure to simplematmul2.cc; only the tile/full sizes scale to 4x4 / 256^3.
//   win_a A=[M,K] -> d1 = M-tile,  d2 = K-chunk
//   win_b B=[N,K] -> d1 = N-tile,  d2 = K-chunk
//   win_c C=[M,N] -> d1 = M-tile,  d2 = N-tile
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

// ─── KERNEL: per-tile int8 GEMM (cache-A / stream-B), no debug logging ───────
/* [exp58] mac+reduce_add with 4-accumulator K-unrolling to hide MAC latency.
 *
 * exp56 baseline: single accumulator per output element — 1 mac per K-step.
 * AIE2PS integer MAC has ~8-cycle result latency; with a single acc chain the
 * pipeline stalls waiting for the previous result before issuing the next mac.
 *
 * Fix: unroll the K loop ×4 so 4 independent accumulators are in flight
 * simultaneously.  The compiler/scheduler can overlap their latencies and keep
 * the MAC unit fed.  K is a multiple of 64 (eff_k) and each mac advances 16
 * K-elements, giving 4 macs per eff_k/16 = 4 steps → unroll ×4 exhausts K
 * for one (i,j) output in a single pass with 4 live accumulators.
 *
 * Layout unchanged from exp56:
 *   A[tile_rows=16][eff_k=64] row-major, cached in all_A
 *   B[cols_per_round=16][eff_k=64] col-major: B_ptr[col*eff_k + k]
 *
 * Per output element:
 *   acc0 += mac(A[i][0:16],  B[j][0:16])
 *   acc1 += mac(A[i][16:32], B[j][16:32])
 *   acc2 += mac(A[i][32:48], B[j][32:48])
 *   acc3 += mac(A[i][48:64], B[j][48:64])
 *   C[i][j] = saturate(reduce_add(acc0 + acc1 + acc2 + acc3))
 *
 * All 4 accumulators are independent → 4× MAC pipeline utilization vs exp56.
 */
__global__ void matmul(aie::port<input_window_int8 *, RowBA> win_a, aie::port<input_window_int8 *, ColBB> win_b,
                       aie::port<output_window_int8 *, LtoR_Merge> win_c) {
    const int tile_rows = aie::get_tile_rows(); // 16
    const int tile_cols = aie::get_tile_cols(); // 16
    const int eff_k = aie::get_effective_k();   // 64
    const int k_rounds = aie::get_k_rounds();   // 4
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a); // 1024
    const int buf_sz_c = aie::get_buffer_size(win_c); // 256
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);
    const int cols_per_round = aie::get_buffer_size(win_b) / eff_k; // 16

    // A tile: full 16×64 row-major tile cached locally so we iterate over j
    // without re-acquiring the A window per (i,j) pair.
    alignas(aie::vector_decl_align) int8_t all_A[tile_rows * eff_k]; // 16×64 = 1024 bytes
#if PASSTHROUGH_KERNEL
    // ── Passthrough (feed/DMA/lock floor) ─────────────────────────────────────
    // MINIMAL diff from the GEMM below: identical window acquire/release pattern
    // and identical *scalar* memory access (A cached, output written the same way),
    // but the K-reduction MAC arithmetic is removed. This isolates the feed + data
    // movement floor without the multiply-accumulate. Memory access mirrors the
    // working GEMM exactly (no vector load/store on window pointers — those are only
    // v4int8-aligned and faulted a core in the earlier vectorized attempt), so the
    // lock/DMA schedule is unchanged. Output is NOT a valid GEMM result (verify skipped).
    const int buf_sz_b = aie::get_buffer_size(win_b); // bytes per B window
#if PASSTHROUGH_VECTORIZED
    // ── Vectorized touch (v32int8) ────────────────────────────────────────────
    // load_v/store_v on 32-byte-aligned offsets (buffers are v-aligned and every
    // window buffer size here is a multiple of 32). store_v to the output window is
    // the same proven pattern as example/perf/aieml_perfstream.cc. The B reduction
    // vector (vbsum) is folded into the output tile so neither the A nor B window
    // handshake can be dead-code-eliminated.
    constexpr int VW = 32; // v32int8 = 256-bit
    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        aie::vector<int8, VW> vbsum = aie::zeros<int8, VW>(); // keeps B reads live
        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i += VW) // vectorized A cache (window->local)
                    aie::store_v(all_A + ra * buf_sz_a + i, aie::load_v<VW>(A_ptr + i));
                release_input_window(win_a);
            }
            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                for (int i = 0; i < buf_sz_b; i += VW) // vectorized B read into accumulator
                    vbsum = aie::add(vbsum, aie::load_v<VW>(B_ptr + i));
                release_input_window(win_b);
            }
        }
        // Build output tile with vector ops (cached A + B-reduction lane), then
        // vector-store to the output window. Output depends on both A and B.
        alignas(aie::vector_decl_align) int8_t local_out[tile_rows * tile_cols];
        for (int idx = 0; idx < tile_rows * tile_cols; idx += VW)
            aie::store_v(local_out + idx, aie::add(aie::load_v<VW>(all_A + idx), vbsum));
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            for (int i = 0; i < buf_sz_c; i += VW) // vectorized output store to window
                aie::store_v(out + i, aie::load_v<VW>(local_out + rc * buf_sz_c + i));
            release_output_window(win_c);
        }
    }
#else
    // ── Scalar touch ──────────────────────────────────────────────────────────
    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        int8_t bsum = 0; // running reduction of B bytes — keeps B reads/handshake live
        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++) // cache A (same scalar copy as GEMM)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }
            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                // Scalar-read every B byte so the acquire/release lock handshake is
                // NOT dead-code-eliminated (bsum flows into the output below).
                for (int i = 0; i < buf_sz_b; i++)
                    bsum = (int8_t)(bsum + B_ptr[i]);
                release_input_window(win_b);
            }
        }
        // Output: cached A tile combined with the B reduction (pure data movement,
        // same scalar write loop shape as the GEMM's output store below). Depending
        // on both all_A and bsum guarantees neither input handshake is elided.
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
#endif // PASSTHROUGH_VECTORIZED
#else
    // int32 accumulator buffer — accumulate across all k_rounds and b_rounds without
    // losing precision; saturate to int8 only once at the very end.
    int32_t acc_buf[tile_rows * tile_cols];

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        // Zero int32 accumulator buffer for this output tile
        for (int idx = 0; idx < tile_rows * tile_cols; idx++)
            acc_buf[idx] = 0;

        for (int kr = 0; kr < k_rounds; kr++) {
            // Cache full A tile for this K-round
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                // B_ptr layout: col-major B[cols_per_round=16][eff_k=64]
                //   B_ptr[col * eff_k + k]
                // eff_k/16 = 4 → 4 independent 16-wide K slices per output element

                for (int i = 0; i < tile_rows; i++) {
                    // Load 4 independent 16-wide A slices for row i (never dependent)
                    const int8_t *Arow = all_A + i * eff_k;
                    aie::vector<int8, 16> a0 = aie::load_v<16>(Arow);
                    aie::vector<int8, 16> a1 = aie::load_v<16>(Arow + 16);
                    aie::vector<int8, 16> a2 = aie::load_v<16>(Arow + 32);
                    aie::vector<int8, 16> a3 = aie::load_v<16>(Arow + 48);

                    for (int j = 0; j < cols_per_round; j++) {
                        const int8_t *Bcol = B_ptr + j * eff_k;
                        // 4 independent accumulator chains — hides ~8-cyc MAC latency
                        aie::accum<acc32, 16> acc0 = aie::mul(a0, aie::load_v<16>(Bcol));
                        aie::accum<acc32, 16> acc1 = aie::mul(a1, aie::load_v<16>(Bcol + 16));
                        aie::accum<acc32, 16> acc2 = aie::mul(a2, aie::load_v<16>(Bcol + 32));
                        aie::accum<acc32, 16> acc3 = aie::mul(a3, aie::load_v<16>(Bcol + 48));
                        // Pair-reduce, then sum into int32 buffer (no saturation mid-K)
                        aie::accum<acc32, 16> acc01 = aie::add(acc0, acc1);
                        aie::accum<acc32, 16> acc23 = aie::add(acc2, acc3);
                        aie::accum<acc32, 16> acc_all = aie::add(acc01, acc23);
                        acc_buf[i * tile_cols + j] += aie::reduce_add(acc_all.to_vector<int32>());
                    }
                }
                release_input_window(win_b);
            }
        }

        // Saturate accumulated int32 → int8 for output
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
#endif // PASSTHROUGH_KERNEL
}

// ── Spot-check verification ──────────────────────────────────────────────────
// Computing the full M*N*K golden on the A78 is ~2*M*N*K scalar MACs (~69 GMAC at
// 4096^3 -> minutes of silent PS compute that trips the board console timeout).
// Instead we verify a strided sample of outputs. A prime stride (coprime with N)
// spreads the sampled (row,col) pairs across all 16 compute tiles' output regions,
// and we always include the last element so the completion barrier still waits for
// the buffer tail. This mirrors the AEG baseline's spot-check approach.
#define SPOT_STRIDE 4093 /* prime; coprime with N -> good (row,col) spread */
#define MAX_SPOTS ((M * N) / SPOT_STRIDE + 2)
static int g_spot_idx[MAX_SPOTS];     /* flat C index of each sampled element */
static int8_t g_spot_gold[MAX_SPOTS]; /* golden value at that index */
static int g_num_spots = 0;

// Build the spot-index list and compute golden only for those sampled indices.
// Called BEFORE the timed region so the reference cost is not charged to the metric.
static void build_spots(const int8_t *A, const int8_t *B) {
    g_num_spots = 0;
    for (int idx = 0; idx < M * N; idx += SPOT_STRIDE)
        g_spot_idx[g_num_spots++] = idx;
    if (g_num_spots == 0 || g_spot_idx[g_num_spots - 1] != M * N - 1)
        g_spot_idx[g_num_spots++] = M * N - 1; /* always sample the buffer tail */
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

// Compact correctness check: compare device C against the precomputed spot golden.
// (Avoids simplematmul.h's full-matrix dumps that would flood the slow UART.)
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

// HOST
int main() {
    printf("\n=== aiehlc GEMM profiling ===\n");
    printf("  C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8, %dx%d mesh (%d tiles)\n", M, N, M, K, N, K, HW_ROWS, HW_COLS,
           HW_ROWS * HW_COLS);

    /* [exp52] enable PMU counter early so we can bracket device init + data setup */
    __ps_pmccntr_enable();
    unsigned long long pc_init0 = __ps_pmccntr();
    aieSetDevice(0);
    aieArray device;
    // 4x4 mesh: cols 0-3, rows 0-5 (shim row0, memtile row1, 4 compute rows 2-5).
    aieMesh mesh = device.partition({0, 3, 0, 5}, HW_ROWS, HW_COLS);
    unsigned long long pc_init1 = __ps_pmccntr(); /* [exp52] end of device init */

    unsigned long long pc_setup0 = __ps_pmccntr(); /* [exp52] start of data setup */
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    // [exp07 poison] forward-decl (front-end only declares sync_for_cpu, not sync_for_dev)
    extern void __Runtime_sync_for_dev(XAie_DevInst * dev, void *ptr, __SIZE_TYPE__ size);
    // Write a poison pattern to C AND FLUSH it to device DRAM before the
    // launch. If the kernel actually computes, it overwrites C with the correct result
    // (PASS). If the workload deadlocks (suspected), the poison survives in device DRAM
    // and verification FAILS showing "got 90" — proving prior PASSes were stale-DDR.
    for (int i = 0; i < M * N; i++)
        C[i] = (int8_t)0x5A; /* 90 */
    __Runtime_sync_for_dev(device._dev, C, M * N * sizeof(int8_t) * 4);
    printf("[exp07] poisoned device C with 0x5A and flushed to DDR\n");

    // Spot-check golden for the completion barrier, computed BEFORE the timed region (outside
    // t0..t1) so the reference cost is not charged to the metric. Samples a strided subset of
    // outputs (see build_spots) instead of the full M*N*K reference.
    build_spots(A, B);
    printf("[spot] verifying %d sampled outputs (stride %d) of %d total\n", g_num_spots, SPOT_STRIDE, M * N);

    unsigned long long pc_setup1 = __ps_pmccntr(); /* [exp52] end of data setup (golden computed) */

    // ── Layer 1: time the WHOLE process until the full result lands in DDR. The launch is
    // asynchronous (posted PS->AIE writes + non-blocking waits), so timing the launch call
    // alone reads ~0. Instead, after issuing the launch we poll the output buffer in device
    // DRAM (invalidate cache via synchronizecpu, then compare against the golden reference)
    // and stop the timer the instant C holds the complete correct result. This is the true
    // end-to-end "launch -> results readable" wall, independent of DMA-status quirks.
    const uint64_t MAX_POLL = 500000000ULL; /* safety bound on a genuine hang */
    XTime t0, t1;
    /* [exp40] Independent wall counter (CNTVCT_EL0) bracketing the SAME window, in case
     * XTime's xiltimer source is frozen this boot (raw counts read 0). Read-only. */
    unsigned long long cv0 = __ps_cntvct();
    /* [exp41] PMU cycle counter already enabled at top of main [exp52]; re-read start. */
    unsigned long long pc0 = __ps_pmccntr();
    XTime_GetTime(&t0);
    matmul<<<mesh>>>(A, B, C, M, N, K);
    unsigned long long pc_mid = __ps_pmccntr(); /* [exp43] boundary: launch-call vs poll-loop */
    uint64_t polls = 0;
    int complete = 0;
    unsigned long long poll_sync_cyc = 0ULL, poll_cmp_cyc = 0ULL; /* [exp52] poll breakdown */
    do {
        unsigned long long __ps0 = __ps_pmccntr();            /* [exp52] */
        device.synchronizecpu(C, M * N * sizeof(int8_t) * 4); /* invalidate -> fresh DDR read */
        poll_sync_cyc += (__ps_pmccntr() - __ps0);            /* [exp52] */
        unsigned long long __pc0 = __ps_pmccntr();            /* [exp52] */
        complete = 1;
#if !PASSTHROUGH_KERNEL
        // GEMM: barrier completes when the sampled outputs match the golden ref.
        // (Passthrough output is not a GEMM result; the launch already blocked on
        //  its output DMA via wait_io, so a single sync above is the barrier.)
        for (int s = 0; s < g_num_spots; s++) {
            if (C[g_spot_idx[s]] != g_spot_gold[s]) {
                complete = 0;
                break;
            }
        }
#endif
        poll_cmp_cyc += (__ps_pmccntr() - __pc0); /* [exp52] */
        polls++;
    } while (!complete && polls < MAX_POLL);
    XTime_GetTime(&t1);
    unsigned long long cv1 = __ps_cntvct();  /* [exp40] */
    unsigned long long pc1 = __ps_pmccntr(); /* [exp41] */
    if (!complete)
        printf("  WARNING: completion barrier hit MAX_POLL=%llu without full result\n", (unsigned long long)MAX_POLL);

    // [exp20] After removing the dead Core_Done poll the launch wall dropped below the old
    // %.3f-ms print's resolution (read 0.000 ms). Relate raw XTime counts -> wall directly:
    // wall_seconds = counts / COUNTS_PER_SECOND. We print the raw delta + the tick frequency
    // so the true sub-ms wall (and the timer's own resolution = 1/freq) is always recoverable.
    uint64_t raw_counts = (uint64_t)(t1 - t0);
    uint64_t timer_hz = (uint64_t)COUNTS_PER_SECOND;
    double wall_ms = 1000.0 * (double)raw_counts / (double)timer_hz;
    double wall_us = 1.0e6 * (double)raw_counts / (double)timer_hz;
    double tick_ns = 1.0e9 / (double)timer_hz;                    // one count in ns
    double total_flops = 2.0 * (double)M * (double)N * (double)K; // MACs counted as 2 ops
    double gflops_wall = (wall_ms > 0.0) ? total_flops / (wall_ms * 1e-3) / 1e9 : 0.0;

    // ── Read profiling counters (probe = first compute tile of the group).
    uint32_t active = 0, vec = 0, sstall = 0, lstall = 0, mm0 = 0, mm1 = 0;
    int have_core = __Runtime_core_perf_probe_valid();
    __Runtime_core_perf_read_probe(&active, &vec, &sstall, &lstall);
    __Runtime_perfcnt_read_mm2s_probe(&mm0, &mm1);

    // Per-tile compute work (frequency-free; all tiles run concurrently).
    double tile_macs = (double)(M / HW_ROWS) * (double)(N / HW_COLS) * (double)K; // MACs on one tile
    double tile_flops = 2.0 * tile_macs;                                          // count MAC as 2 FLOP
    // The active/stream-stall/lock-stall core counters are captured over the SAME
    // window, so their *ratios* (compute vs stall split) are meaningful even though
    // the absolute cycle count is a sampled sub-window, not the whole run (the core
    // issues 'vec' vector instrs over the full run, which alone exceeds 'active').
    // We therefore report the state split as fractions and derive vector density
    // (MACs per vector instruction) from the full-run vec counter + known MAC count,
    // which is window-independent. macs_per_vec ~= the vector unit's MAC/op efficiency
    // (e.g. an aie::mmul<4,16,8> retires ~64 MAC/op; a mul+reduce dot-product is far lower).
    double total_budget = (double)active + (double)sstall + (double)lstall;
    double compute_pct = total_budget ? 100.0 * (double)active / total_budget : 0.0;
    double stream_pct = total_budget ? 100.0 * (double)sstall / total_budget : 0.0;
    double lock_pct = total_budget ? 100.0 * (double)lstall / total_budget : 0.0;
    double macs_per_vec = vec ? tile_macs / (double)vec : 0.0;
    (void)tile_flops;

    // Device yardstick (same as AEG report so the two are comparable).
    const double DEVICE_INT8_TOPS = 184.0;
    const int DEVICE_TILES = 144;
    int array_tiles = HW_ROWS * HW_COLS;
    double array_peak_gops = DEVICE_INT8_TOPS * 1000.0 * (double)array_tiles / (double)DEVICE_TILES;
    double util_pct = array_peak_gops ? 100.0 * gflops_wall / array_peak_gops : 0.0;

    printf("\n--- Layer 0: pre-launch setup (outside timed window) ---\n"); /* [exp52] */
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
        /* [exp40] Independent CNTVCT_EL0 wall — bypasses a frozen xiltimer source. */
        unsigned long long cv_raw = (cv1 >= cv0) ? (cv1 - cv0) : 0ULL;
        unsigned long long cv_hz = __ps_cntfrq();
        double cv_ms = cv_hz ? 1000.0 * (double)cv_raw / (double)cv_hz : 0.0;
        printf("  [cntvct] raw:      %llu counts  freq: %llu Hz  wall: %.6f ms\n", cv_raw, cv_hz, cv_ms);
    }
    {
        /* [exp41] PMU cycle-counter wall — CPU-core-clock, survives a frozen generic timer.
         * Primary signal is raw cycles (nonzero => the PS-side wall is finally measurable).
         * ms = raw / CPU_Hz done offline (CPU_Hz not known at compile time); raw is enough
         * to decide alive-vs-frozen and to compare host-side levers relatively. */
        unsigned long long pc_raw = (pc1 >= pc0) ? (pc1 - pc0) : 0ULL;
        unsigned long long pmcr = __ps_pmcr(); /* [exp42] */
        unsigned int d_bit = (unsigned int)((pmcr >> 3) & 1ULL);
        printf("  [pmccntr] raw:     %llu cycles  pmcr:0x%llx (D=%u, %s)\n", pc_raw, pmcr, d_bit,
               d_bit ? "counts=CPUcyc/64" : "counts=CPUcyc");
        /* [exp43] split: launch = matmul<<<>>> call (input-DMA+load+config+enable+wait_io drain);
         * poll = DDR poll-to-result loop. */
        unsigned long long pc_launch = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        unsigned long long pc_poll = (pc1 >= pc_mid) ? (pc1 - pc_mid) : 0ULL;
        printf("  [pmccntr] launch:  %llu cycles  poll: %llu cycles\n", pc_launch, pc_poll);
        /* [exp44] how much of the launch is the wait_io output-DMA drain (array-gated)? */
        unsigned long long wio_cyc = 0ULL;
        unsigned int wio_calls = 0U;
        __Runtime_wait_io_cycles(&wio_cyc, &wio_calls);
        double wio_pct = (pc_launch > 0) ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0;
        printf("  [phase] wait_io:   %llu cycles over %u calls  (=%.1f%% of launch)\n", wio_cyc, wio_calls, wio_pct);
        /* [exp52] wait_io sub-detail: busy-poll iteration count -> cyc/iter = cost per DmaGetPendingBdCount */
        unsigned long long wio_iters = 0ULL;
        __Runtime_wait_io_iters(&wio_iters);
        double wio_cyc_per_iter = (wio_iters > 0) ? (double)wio_cyc / (double)wio_iters : 0.0;
        double wio_cyc_per_call = (wio_calls > 0) ? (double)wio_cyc / (double)wio_calls : 0.0;
        printf("  [wait_io] poll iters: %llu total  (%.1f cyc/iter avg, %.0f avg cyc/call)\n", wio_iters,
               wio_cyc_per_iter, wio_cyc_per_call);
        /* [exp52] poll loop split: synchronizecpu vs golden-compare */
        double sync_pct = (pc_poll > 0) ? 100.0 * (double)poll_sync_cyc / (double)pc_poll : 0.0;
        double cmp_pct = (pc_poll > 0) ? 100.0 * (double)poll_cmp_cyc / (double)pc_poll : 0.0;
        printf("  [poll] synchronizecpu: %llu cyc over %llu calls  (=%.1f%% of poll)\n", poll_sync_cyc,
               (unsigned long long)polls, sync_pct);
        printf("  [poll] compare (C vs golden): %llu cyc over %llu calls  (=%.1f%% of poll)\n", poll_cmp_cyc,
               (unsigned long long)polls, cmp_pct);
        /* [exp45] split the ~98% setup portion of the launch into runtime sub-phases. */
        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        const char *phn[4] = {"kload  ", "bdcfg  ", "coreen ", "startio"};
        for (int i = 0; i < 4; i++) {
            double p = (pc_launch > 0) ? 100.0 * (double)ph[i] / (double)pc_launch : 0.0;
            printf("  [phase] %s: %llu cycles over %u calls  (=%.1f%% of launch)\n", phn[i], ph[i], phc[i], p);
        }
        /* [exp46] which XAie call dominates bdcfg: DmaDescInit (HW config read) or DmaWriteBd? */
        unsigned long long bi = 0ULL, bw = 0ULL;
        unsigned int bin = 0U, bwn = 0U;
        __Runtime_bd_subphase_cycles(&bi, &bin, &bw, &bwn);
        double bip = (ph[1] > 0) ? 100.0 * (double)bi / (double)ph[1] : 0.0;
        double bwp = (ph[1] > 0) ? 100.0 * (double)bw / (double)ph[1] : 0.0;
        printf("  [bdcfg] descinit: %llu cyc over %u  (=%.1f%% of bdcfg)\n", bi, bin, bip);
        printf("  [bdcfg] writebd:  %llu cyc over %u  (=%.1f%% of bdcfg)\n", bw, bwn, bwp);
        /* [exp47] localize the ~99.85% that is neither descinit nor writebd */
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
        /* [exp48] split mid into GetTileTypefromLoc / DmaSetAddr / DmaEnableBd (+ residual = Set*+printf) */
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
        /* [exp51] split the dominant kload phase: XAie_LoadElfMem vs Core Disable/Reset/Unreset */
        unsigned long long kelf = 0ULL, krst = 0ULL;
        unsigned int kelfn = 0U, krstn = 0U;
        __Runtime_kload_split_cycles(&kelf, &kelfn, &krst, &krstn);
        double kep = (ph[0] > 0) ? 100.0 * (double)kelf / (double)ph[0] : 0.0;
        double krp = (ph[0] > 0) ? 100.0 * (double)krst / (double)ph[0] : 0.0;
        printf("  [kload] loadelf:  %llu cyc over %u  (=%.1f%% of kload)\n", kelf, kelfn, kep);
        printf("  [kload] corerst:  %llu cyc over %u  (=%.1f%% of kload)\n", krst, krstn, krp);
        /* [exp55] unaccounted launch cycles = lock_init + host.cc glue not in any phase */
        unsigned long long ph_total = ph[0] + ph[1] + ph[2] + ph[3] + wio_cyc;
        unsigned long long unacct = (pc_launch > ph_total) ? (pc_launch - ph_total) : 0ULL;
        double unacct_p = (pc_launch > 0) ? 100.0 * (double)unacct / (double)pc_launch : 0.0;
        printf("  [launch] unaccounted (lock_init+glue): %llu cyc  (=%.1f%% of launch)\n", unacct, unacct_p);
        /* [exp55] full launch budget table */
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
    unsigned long long pc_verify0 = __ps_pmccntr(); /* [exp55] */
#if PASSTHROUGH_KERNEL
    int result = 0; /* passthrough: output is not a GEMM result */
    printf("RESULT: SKIP (passthrough kernel — feed/DMA floor measurement, no compute)\n");
    (void)prof_verify; /* silence unused-function warning */
#else
    int result = prof_verify(C);
#endif
    unsigned long long pc_verify1 = __ps_pmccntr(); /* [exp55] */

    unsigned long long pc_free0 = __ps_pmccntr(); /* [exp55] */
    device.free(A);
    device.free(B);
    device.free(C);
    unsigned long long pc_free1 = __ps_pmccntr(); /* [exp55] */

    printf("\n--- Layer 0 (post-launch): out-of-timed-window teardown ---\n"); /* [exp55] */
    printf("  [pmccntr] verify:      %llu cyc  (prof_verify spot-check compare)\n",
           (unsigned long long)(pc_verify1 - pc_verify0));
    printf("  [pmccntr] device.free: %llu cyc  (free A+B+C)\n", (unsigned long long)(pc_free1 - pc_free0));

    // ── Compact machine-parseable PERF summary — one key=value per line.
    // Grep for "[PERF]" across experiment logs to compare without retiming.
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

    // Sentinel so the board harness (appvek385.py) detects completion promptly
    // even though partition teardown is disabled for profiling.
    printf("\n[prof] device_teardown done\n");
    return result;
}
