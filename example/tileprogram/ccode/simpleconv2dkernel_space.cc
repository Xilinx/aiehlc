#include "simpleconv2d_new_load.h"
#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DMA_ISSUE_COUNT)
// Hand-written GemmSpaces + Triton-style block loads.
//
// The same conv2d as simpleconv2dkernel_port.cc, with the same kernel body,
// but the DMA is written by hand as one GemmSpace per port (the explicit form
// of simpleconv2d.cc) instead of being derived from the kernel. aiehlc traces
// the body's aie::load_buffer / store_buffer blocks and requires every
// GemmSpace below to give exactly that DMA (convautotile verifySpaceKernels):
// change a number here without changing the kernel's loops -- or the other
// way round -- and the build stops naming the field, e.g.
//   win_a (GemmSpace RowBC_halo) .d1.slice_tiling.stride = 13, the kernel's blocks give 14
// pp_depth / l1_budget / pad are scheduling choices and are not compared.

// IFM [230, 230, 4] (zero-padded, channel-aligned): one 61-row band per mesh
// row (step 56), walked on-core in 4 halo slabs of 19 rows (step 14) x 4
// column chunks of 61 (step 56). The same band for every core of a mesh row.
constexpr aie::GemmSpace RowBC_halo = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = 230,
           .tile_round = 4, // mesh rows
           .tile_size = 61,
           .stride = 56,
           .slice_tiling = {.tile_size = 19, .stride = 14, .rounds = 4}},    // outer loop i
    .d2 = {.fullsize = 230, .tile_round = 4, .tile_size = 61, .stride = 56}, // inner loop j
    .d3 = {.fullsize = 4, .tile_size = 4}};                                  // channels, packed (no stride)

// WEIGHTS [64, 7, 7, 4]: 16 output channels per mesh column, sent once; the
// same block for every core of a mesh column.
constexpr aie::GemmSpace ColBC = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                  .d1 = {.fullsize = 64, .tile_round = 4, .tile_size = 16, .stride = 16},
                                  .d2 = {.fullsize = 7, .tile_size = 7, .stride = 7},
                                  .d3 = {.fullsize = 7, .tile_size = 7, .stride = 7},
                                  .d4 = {.fullsize = 4, .tile_size = 4, .stride = 4}};

// OFM [112, 112, 64]: each core writes 28 rows (mesh row) x 16 channels (mesh
// col) as 4 x 4 blocks of 7 x 28, gathered left to right.
constexpr aie::GemmSpace LtoR_Merge = {.policy = {.map = {.layout = aie::Layout::Row,
                                                          .merge_order = aie::Flow::LeftToRight,
                                                          .mesh_tiling_group1_dim = 1,  // d1 split across mesh rows
                                                          .mesh_tiling_group2_dim = 3}, // d3 split across mesh cols
                                                  .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                                  .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                       .d1 = {.fullsize = 112,
                                              .tile_round = 4,
                                              .tile_size = 28,
                                              .stride = 28,
                                              .slice_tiling = {.tile_size = 7, .stride = 7, .rounds = 4}},
                                       .d2 = {.fullsize = 112, .tile_round = 4, .tile_size = 28, .stride = 28},
                                       .d3 = {.fullsize = 64, .tile_round = 4, .tile_size = 16}};

constexpr aie::GlobalPolicy conv_policy = {.fullconnect_auto = 0};
__global__(conv_policy) void conv2d_spatial(aie::port<input_window_int8 *, RowBC_halo> win_a,
                                            aie::port<input_window_int8 *, ColBC> win_b,
                                            aie::port<output_window_int8 *, LtoR_Merge> win_c) {
    // The operator and the tiling live in the kernel.
    const int KH = 7, KW = 7, S = 2, C = 4;         // 7x7 stride-2 conv, 4 (aligned) channels
    const int oh_t = 7, ow_t = 28, oc_t = 16;       // output block computed per round
    const int ih_t = (oh_t - 1) * S + KH;           // 19: input footprint of one output block
    const int iw_t = (ow_t - 1) * S + KW;           // 61
    const int oh_core = 112 / aie::num_programs(0); // 28 output rows per mesh row
    const int TAPS = KH * KW * C;                   // 196
    const int ROW = iw_t * C;                       // 244 bytes per slab row
    const int pid_h = aie::program_id(0);           // mesh row: which OH band
    const int pid_c = aie::program_id(1);           // mesh col: which OC group
    using MM = aie::mmul<8, 8, 8, int8_t, int8_t>;
    aie::set_saturation(aie::saturation_mode::saturate); // int8 results clamp like the scalar reference
    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C});
    // const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t*10, 0, 0, 0}, {oc_t, KH, KW, C});//for bug test
    alignas(64) int8_t Bp[7 * 4 * 2 * 64];
    for (int kh = 0; kh < KH; kh++)
        for (int kb = 0; kb < 4; kb++)
            for (int nb = 0; nb < 2; nb++)
                for (int k = 0; k < 8; k++)
                    for (int n = 0; n < 8; n++) {
                        const int byte = kb < 3 ? kb * 8 + k : 20 + k; // tap byte within kernel row kh
                        const bool live = kb < 3 || k >= 4;
                        Bp[(((kh * 4 + kb) * 2 + nb) * 8 + k) * 8 + n] =
                            live ? w[(nb * 8 + n) * TAPS + kh * KW * C + byte] : (int8_t)0;
                    }
    aie::release(win_b);
    const aie::accum<acc32, 64> zero = aie::zeros<acc32, 64>();

    for (int i = 0; i < oh_core / oh_t; i++) {          // 4 row blocks
        for (int j = 0; j < 112 / ow_t; j++) {          // 4 column blocks
            const int oh0 = pid_h * oh_core + i * oh_t; // output block origin
            const int ow0 = j * ow_t;
            // Its input footprint (the halo overlaps the neighbouring blocks).
            const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});
            int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t});

            for (int oh = 0; oh < oh_t; oh++) {
                for (int pass = 0; pass < 2; pass++) {
                    const int owA = pass * 16, owB = pass * 16 + (pass ? 4 : 8); // M-block first columns
                    MM cA0(zero), cA1(zero), cB0(zero), cB1(zero);
                    for (int kh = 0; kh < KH; kh++)
                        chess_prepare_for_pipelining chess_loop_range(7, 7) {
                            const int8_t *r = x + (oh * S + kh) * ROW;
                            const int8_t *bk = Bp + kh * 4 * 2 * 64;
                            for (int kb = 0; kb < 4; kb++)
                                chess_unroll_loop() {
                                    const int off = kb < 3 ? kb * 8 : 20;
                                    const aie::vector<int8_t, 64> aA = aie::load_unaligned_v<64>(r + owA * 8 + off, 4);
                                    const aie::vector<int8_t, 64> aB = aie::load_unaligned_v<64>(r + owB * 8 + off, 4);
                                    const aie::vector<int8_t, 64> b0 = aie::load_v<64>(bk + (kb * 2 + 0) * 64);
                                    const aie::vector<int8_t, 64> b1 = aie::load_v<64>(bk + (kb * 2 + 1) * 64);
                                    cA0.mac(aA, b0);
                                    cA1.mac(aA, b1);
                                    cB0.mac(aB, b0);
                                    cB1.mac(aB, b1);
                                }
                        }
                    // 8 columns x 16 channels = 128 contiguous output bytes:
                    // interleave the two N-blocks 8 channels at a time.
                    const int owc[2] = {owA, owB};
                    const MM *acc[2][2] = {{&cA0, &cA1}, {&cB0, &cB1}};
                    for (int m = 0; m < 2; m++) {
                        const auto z =
                            aie::interleave_zip(acc[m][0]->to_vector<int8_t>(0), acc[m][1]->to_vector<int8_t>(0), 8);
                        int8_t *o = y + (oh * ow_t + owc[m]) * oc_t; // 32-byte aligned (window base is)
                        aie::store_v(o, z.first.extract<32>(0));
                        aie::store_v(o + 32, z.first.extract<32>(1));
                        aie::store_v(o + 64, z.second.extract<32>(0));
                        aie::store_v(o + 96, z.second.extract<32>(1));
                    }
                }
            }
            aie::release(win_a);
            aie::commit(win_c);
        }
    }
}
// ═══════════════════════════════════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("=== Conv2d via Im2col on AIE %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    // --- Device + mesh ---
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    // --- Allocate host memory (DDR pre-pad: spatial border + channel layout) ---
    int8_t *input = (int8_t *)__Runtime_Alloc(INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN * sizeof(int8_t));
    int8_t *filter = (int8_t *)__Runtime_Alloc(KERNEL_H * KERNEL_W * INPUT_C_ALIGN * NUM_FILTERS * sizeof(int8_t));
    int8_t *output = (int8_t *)__Runtime_Alloc(OUTPUT_H * OUTPUT_W * NUM_FILTERS * sizeof(int8_t));
    // --- Initialize test data ---
    memset(input, 0, INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN * sizeof(int8_t));
    for (int h = 0; h < INPUT_H; h++) {
        for (int w = 0; w < INPUT_W; w++) {
            for (int c = 0; c < INPUT_C; c++) {
                int real_idx = (h * INPUT_W + w) * INPUT_C + c;
                int val = (int)(mix32((uint32_t)real_idx + INPUT_SEED) % 9u) - 4; // [-4, 4]
                input[((h + PAD) * INPUT_W_PAD + (w + PAD)) * INPUT_C_ALIGN + c] = (int8_t)val;
            }
            // padding channels [INPUT_C, INPUT_C_ALIGN) and spatial border left at 0
        }
    }
    // Filter: zero-mean samples in {-1, 0, 1} for the real channels, 0 for the
    memset(filter, 0, KERNEL_H * KERNEL_W * INPUT_C_ALIGN * NUM_FILTERS * sizeof(int8_t));
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                for (int c = 0; c < INPUT_C; c++) {
                    int kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c;
                    int filt_key = ((f * KERNEL_H + kh) * KERNEL_W + kw) * INPUT_C + c;
                    int val = (int)(mix32((uint32_t)filt_key + FILTER_SEED) % 3u) - 1; // {-1,0,1}
                    filter[f * K + kk] = (int8_t)val;
                }
                // padding channel c == INPUT_C..INPUT_C_ALIGN-1 left at 0
            }
        }
    }
    memset(output, 0, OUTPUT_H * OUTPUT_W * NUM_FILTERS * sizeof(int8_t));
    printf("\n--- Launching conv2d_spatial (spatial-halo) on AIE ---\n");
    XTime t_aie_start, t_aie_end;
    XTime_GetTime(&t_aie_start);
    conv2d_spatial<<<mesh>>>(input, filter, output, M, N, K);
    XTime_GetTime(&t_aie_end);
    double aie_wall_ms = 1.0 * (t_aie_end - t_aie_start) / COUNTS_PER_SECOND * 1000.0;
    printf("[BENCH] aie_wall_ms=%.3f\n", aie_wall_ms);
    // Bit-exact against the CPU reference: the hand-written GemmSpaces deliver
    // the blocks the kernel computes on.
    const int mismatches = verify_conv2d(input, filter, output);
    free(input);
    free(filter);
    free(output);
    return mismatches ? 1 : 0;
}
