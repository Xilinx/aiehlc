/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * An AIEDNN template kernel (tutorial/aiednn/howto/templatekernel.md):
 *   C[M][N] = epilogue(A[M][K] . W[N][K]^T), bf16 in / out, fp32 accumulation,
 *   ONE launch for the whole GEMM on the 4x4 mesh.
 *
 * The decisions, one per tutorial section:
 *   tiling      each core owns a TM x TN output tile at a time (16 x 16: 2 x 2
 *               aie::mmul<8,8,8> accumulators -- the register blocking)
 *   scheduling  the tile loop runs IN the core: core (pm, pn) walks M/(4*TM)
 *               tiles of its mesh row; the host launches once
 *   K           K streamed in chunks of TK, accumulated in fp32 on the core,
 *               rounded to bf16 once (output-stationary)
 *   fusion      the epilogue (ReLU when AD_RELU) runs on the accumulator before
 *               the single write-back
 *   layout      the host pre-tiles A / W chunks in mmul order (gemm_kacc_bf16.h)
 *
 * Template parameters: the block below. One set = one compiled kernel;
 * gen_variants.py writes one copy of this file per set.
 ******************************************************************************/
#include "gemm_kacc_bf16.h"

// ---- template parameters (gen_variants.py rewrites this block) ----
// AD_M rows of A / C (output pixels), a multiple of 4 * AD_TM; AD_N columns of C
// (output channels) = 4 mesh cols x AD_TN; AD_K the reduction (Cin * kh * kw), a
// multiple of AD_TK; AD_TK the K chunk in bf16 elements, a multiple of 8 (the
// mmul K); AD_RELU the fused epilogue (1 = ReLU, 0 = none). Comments stay off the
// #define lines: the static bounds analysis reads the values from them.
#define AD_M 256
#define AD_N 64
#define AD_K 2304
#define AD_TK 64
#define AD_RELU 1
// ---- end of template parameters ----
// The output tile, fixed by the 2 x 2 accumulator blocking below.
#define AD_TM 16
#define AD_TN 16

__global__ void gemm_kacc(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<AD_M, AD_K>> a,
                          aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<AD_N, AD_K>> w,
                          aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<AD_M, AD_N>> c) {
    const int ntile = AD_M / (4 * AD_TM); // output tiles per core, walked on the core
    const int nkc = AD_K / AD_TK;         // K chunks per tile
    const int kt = AD_TK / 8;             // 8x8 tiles per chunk row
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * AD_TM) + t * AD_TM;
#if defined(__chess__)
        using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
        MM c00(aie::zeros<accfloat, 64>()), c01(aie::zeros<accfloat, 64>());
        MM c10(aie::zeros<accfloat, 64>()), c11(aie::zeros<accfloat, 64>());
#else
        float acc[AD_TM * AD_TN] = {}; // x86 trace / emulation: the same math, scalar
#endif
        for (int j = 0; j < nkc; j++) {
            const bfloat16 *x = aie::load_buffer(a, {m0, j * AD_TK}, {AD_TM, AD_TK});
            const bfloat16 *v = aie::load_buffer(w, {pn * AD_TN, j * AD_TK}, {AD_TN, AD_TK});
#if defined(__chess__)
            for (int kb = 0; kb < kt; kb++) {
                const auto a0 = aie::load_v<64>(x + kb * 64), a1 = aie::load_v<64>(x + (kt + kb) * 64);
                const auto b0 = aie::load_v<64>(v + kb * 64), b1 = aie::load_v<64>(v + (kt + kb) * 64);
                c00.mac(a0, b0);
                c01.mac(a0, b1);
                c10.mac(a1, b0);
                c11.mac(a1, b1);
            }
#else
            for (int m = 0; m < AD_TM; m++)
                for (int n = 0; n < AD_TN; n++)
                    for (int k = 0; k < AD_TK; k++)
                        acc[m * AD_TN + n] += (float)x[((m / 8) * kt + k / 8) * 64 + (m % 8) * 8 + k % 8] *
                                              (float)v[((n / 8) * kt + k / 8) * 64 + (k % 8) * 8 + n % 8];
#endif
            aie::release(a);
            aie::release(w);
        }
        // The tile is done: epilogue on the fp32 accumulator, one rounding, one write-back.
        bfloat16 *y = aie::store_buffer(c, {m0, pn * AD_TN}, {AD_TM, AD_TN});
#if defined(__chess__)
        alignas(64) float acc[AD_TM * AD_TN];
        const MM *blk[4] = {&c00, &c01, &c10, &c11};
        for (int b = 0; b < 4; b++) {
            alignas(64) float t8[64];
            aie::store_v(t8, blk[b]->to_vector<float>());
            for (int i = 0; i < 64; i++)
                acc[((b / 2) * 8 + i / 8) * AD_TN + (b % 2) * 8 + i % 8] = t8[i];
        }
#endif
        for (int i = 0; i < AD_TM * AD_TN; i++) {
            float o = acc[i];
            if (AD_RELU && o < 0.0f)
                o = 0.0f;
            y[i] = (bfloat16)o;
        }
        aie::commit(c);
    }
}

// One launch; the mesh must be declared in the function that launches.
static void gemm_kacc_run(uint16_t *A, uint16_t *W, uint16_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, AD_HW_ROWS, AD_HW_COLS);
    gemm_kacc<<<mesh>>>(A, W, C);
}

int main() {
    printf("=== AIEDNN gemm_kacc: C[%d][%d] = A[%d][%d] . W[%d][%d]^T, bf16, TK %d, relu %d, one launch ===\n", AD_M,
           AD_N, AD_M, AD_K, AD_N, AD_K, AD_TK, AD_RELU);
    const long na = (long)AD_M * AD_K, nw = (long)AD_N * AD_K, nc = (long)AD_M * AD_N;
    uint16_t *A = (uint16_t *)malloc(na * 2), *W = (uint16_t *)malloc(nw * 2), *ref = (uint16_t *)malloc(nc * 2);
    uint16_t *Ad = (uint16_t *)__Runtime_Alloc(na * 2), *Wd = (uint16_t *)__Runtime_Alloc(nw * 2);
    uint16_t *Cd = (uint16_t *)__Runtime_Alloc(nc * 2);
    ad_init(A, W, AD_M, AD_N, AD_K);
    ad_pretile(A, Ad, AD_M, AD_K, AD_TK, 0); // mmul A operand order
    ad_pretile(W, Wd, AD_N, AD_K, AD_TK, 1); // mmul B operand order
    memset(Cd, 0, nc * 2);
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm_kacc_run(Ad, Wd, Cd);
    XTime_GetTime(&t1);
    // The launch invalidates C only before the DMA; drop lines refetched while the cores wrote it.
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc * 2);
    const double ms = ad_ms(t0, t1);
    printf("[gemm_kacc] PERF SUMMARY launch %.3f ms, %.2f GOPS\n", ms, 2.0 * AD_M * AD_N * AD_K / (ms * 1e-3) / 1e9);
    ad_report_phases();
    ad_reference(A, W, ref, AD_M, AD_N, AD_K, AD_RELU);
    return ad_check(Cd, ref, nc);
}
