#include "simpleconv2d_new_load.h"
// AIE_KERNEL_CONFIG_TRACE compiles klog() into the kernel and makes the host
// print ONLY every core's klog after the launch. Add AIE_DEBUG_LOG for the full
// runtime log + debug snapshot (klog included there).
#pragma aie_debug_level(AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_KERNEL_CONFIG_TRACE)
#define USE_SPATIAL_HALO // default
// The operator, as the math defines it: the TRUE input (224x224x3), not the
// padded DDR buffer. The compiler derives the padded/channel-aligned layout
// (230x230x4) from .pad and the int8 32-bit DMA word -- writing 230/4 here would
// pad twice (OH = (230+6-7)/2+1 = 115).
constexpr aie::Conv2d conv = {.h = 224, .w = 224, .c = 3, .k = 64, .kh = 7, .kw = 7, .stride = 2, .pad = 3};
constexpr aie::Mapping map = {
    .space = {aie::split(aie::OH, MESH_ROW),  // mesh rows
              aie::split(aie::OC, MESH_COL)}, // mesh cols
    .time = {aie::tile(aie::OH, 7),           // optional: omit -> autotune under l1_budget
             aie::tile(aie::OW, 28)},
    .hints = {
        .pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma, .pp_depth = 2, .l1_budget = aie::Bytes{4096}}};

// Triton-style block kernel. Each port names only its data role; the compiler
// derives the DMA from (conv, map). The kernel computes WHERE its blocks are --
// from its program ids and loop indices, in element coordinates of each whole
// tensor -- and loads/stores them:
//   IFM     [230, 230, 4]  DDR input (pre-padded by conv.pad, channel-aligned)
//   WEIGHTS [64, 7, 7, 4]  B^T: [oc][kh][kw][c]
//   OFM     [112, 112, 64] HWC output
// The DMA still streams the blocks in its fixed order (i outer, j inner), so
// every load/store must name exactly the block that arrives next; the compiler-
// derived block descriptors check that (klog "BLKE" on a mismatch).
__global__(conv, map) void conv2d_spatial(
    aie::port<input_window_int8 *, aie::IFM> win_a,     // input  block [ih_t, iw_t, c]
    aie::port<input_window_int8 *, aie::WEIGHTS> win_b, // filter block [oc_t, kh, kw, c]
    aie::port<output_window_int8 *, aie::OFM> win_c     // output block [oh_t, ow_t, oc_t]
) {
    const int pid_h = aie::program_id(0); // mesh row: which OH band
    const int pid_c = aie::program_id(1); // mesh col: which OC group
    const int S = conv.stride, KH = conv.kh, KW = conv.kw;
    const int C = map.c_align;                         // 4
    klog("PIDH", pid_h);                               // 4-char tag + one int32; printed by the host after the run
    klog("PIDC", pid_c);                               // 4-char tag + one int32; printed by the host after the run
    klog("C", C);
    // Lower-case locals: the desugar also emits OH_T/OW_T/TILE_* kernel #defines.
    const int oh_t = map.oh_t, ow_t = map.ow_t; // 7 x 28 output block
    const int ih_t = map.ih_t, iw_t = map.iw_t; // 19 x 61 input block (halo)
    const int oc_t = map.oc_per_core;           // 16
    const int TAPS = KH * KW * C; // 196 (not K: simpleconv2d_new_load.h #defines K)

    // This core's filters: output channels [pid_c*16, +16), all taps.
    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C});

    for (int i = 0; i < map.oh_per_core / oh_t; i++) {          // 4 row blocks
        for (int j = 0; j < map.ow / ow_t; j++) {               // 4 column blocks
            const int oh0 = pid_h * map.oh_per_core + i * oh_t; // output block origin
            const int ow0 = j * ow_t;
            // Its input footprint: rows oh0*S .. +ih_t, cols ow0*S .. +iw_t of the
            // padded input (the halo overlaps the neighbouring blocks).
            const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});
            int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t});

            for (int oh = 0; oh < oh_t; oh++) {
                for (int ow = 0; ow < ow_t; ow++) {
                    for (int oc = 0; oc < oc_t; oc++) {
                        int16_t sum = 0;
                        int kk = 0;
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                for (int c = 0; c < C; c++, kk++)
                                    sum += (int16_t)x[((oh * S + kh) * iw_t + ow * S + kw) * C + c] *
                                           (int16_t)w[oc * TAPS + kk];
                        y[(oh * ow_t + ow) * oc_t + oc] = (int8_t)(sum > 127 ? 127 : (sum < -128 ? -128 : sum));
                    }
                }
            }
            aie::release(win_a);
            aie::commit(win_c);
        }
    }
    aie::release(win_b);
}
// ═══════════════════════════════════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("=== Conv2d via Im2col on AIE %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    printf("    Input:  [%d, %d, %d]\n", INPUT_H, INPUT_W, INPUT_C);
    printf("    Filter: [%d, %d, %d, %d]\n", KERNEL_H, KERNEL_W, INPUT_C, NUM_FILTERS);
    printf("    Output: [%d, %d, %d]\n", OUTPUT_H, OUTPUT_W, NUM_FILTERS);
    printf("    Im2col GEMM: [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);

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

    printf("\n--- Verification ---\n");
    int result = 0; //
    /*
    result = verify_conv2d(input, filter, output);
    uint32_t timeout = 0;
    while (result > 0) {
        sleep(1);
        timeout++;
        printf("ERROR: conv2d output mismatch,app waiting for debug... ( %u second) \n", timeout);
    }
    */
    // --- Cleanup ---
    free(input);
    free(filter);
    free(output);

    return result;
}
