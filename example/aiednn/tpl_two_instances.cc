/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two instances of ONE template kernel in one source file (aiehlc
 * regression for multi-instance templates, sprint aiednn-tpl P1).
 *
 * tpl2_gemm is the gemm_kacc_bf16.cc kernel (C = epilogue(A . B^T), bf16,
 * fp32 accumulation) with ports a / b / c. It is launched with two different
 * argument sets, so aiehlc compiles two kernels from it:
 *   tpl2_gemm<256, 64, 2304, 64, true>   -> tpl2_gemm__256_64_2304_64_1
 *   tpl2_gemm<128, 64, 2304, 64, false>  -> tpl2_gemm__128_64_2304_64_0
 * Both instances have the same port names, and port a's shape <M, K> (and c's
 * <M, N>) depends on the template parameter M. Every kernel TU sees the block
 * descriptors of all kernels of the program, so aiehlc names them per instance:
 * aiehlc_desc_<instance>_<port> / aiehlc_cur_<instance>_<port>
 * (src/llvm/convautotile/block_io_names.h). A template with a single instance
 * keeps aiehlc_desc_<port> (gemm_kacc_bf16.cc).
 *
 *   source script/aiehlc.sh --aie-version 5 --runtime-source-file example/aiednn/tpl_two_instances.cc
 ******************************************************************************/
#include "gemm_kacc_bf16.h"

#define M0 256
#define M1 128
#define GN 64
#define GK 2304
#define GTK 64

template <int M, int N, int K, int TK = 64, bool RELU = true>
__global__ void tpl2_gemm(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<M, K>> a,
                          aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<N, K>> b,
                          aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<M, N>> c) {
    const int TM = 16, TN = 16;     // the output tile: fixed by the 2 x 2 accumulator blocking below
    const int ntile = M / (4 * TM); // output tiles per core, walked on the core
    const int nkc = K / TK;         // K chunks per tile
    const int kt = TK / 8;          // 8x8 tiles per chunk row
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
#if defined(__chess__)
        using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
        MM c00(aie::zeros<accfloat, 64>()), c01(aie::zeros<accfloat, 64>());
        MM c10(aie::zeros<accfloat, 64>()), c11(aie::zeros<accfloat, 64>());
#else
        float acc[TM * TN] = {}; // x86 trace / emulation: the same math, scalar
#endif
        for (int j = 0; j < nkc; j++) {
            const bfloat16 *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const bfloat16 *v = aie::load_buffer(b, {pn * TN, j * TK}, {TN, TK});
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
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    for (int k = 0; k < TK; k++)
                        acc[m * TN + n] += (float)x[((m / 8) * kt + k / 8) * 64 + (m % 8) * 8 + k % 8] *
                                           (float)v[((n / 8) * kt + k / 8) * 64 + (k % 8) * 8 + n % 8];
#endif
            aie::release(a);
            aie::release(b);
        }
        // The tile is done: epilogue on the fp32 accumulator, one rounding, one write-back.
        bfloat16 *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
#if defined(__chess__)
        alignas(64) float acc[TM * TN];
        const MM *blk[4] = {&c00, &c01, &c10, &c11};
        for (int b = 0; b < 4; b++) {
            alignas(64) float t8[64];
            aie::store_v(t8, blk[b]->to_vector<float>());
            for (int i = 0; i < 64; i++)
                acc[((b / 2) * 8 + i / 8) * TN + (b % 2) * 8 + i % 8] = t8[i];
        }
#endif
        for (int i = 0; i < TM * TN; i++) {
            float o = acc[i];
            if (RELU && o < 0.0f)
                o = 0.0f;
            y[i] = (bfloat16)o;
        }
        aie::commit(c);
    }
}

// Both launches in the function that declares the mesh.
static void tpl2_run(int which, uint16_t *A, uint16_t *W, uint16_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, AD_HW_ROWS, AD_HW_COLS);
    if (which == 0)
        tpl2_gemm<M0, GN, GK, GTK, true><<<mesh>>>(A, W, C);
    else
        tpl2_gemm<M1, GN, GK, GTK, false><<<mesh>>>(A, W, C);
}

static int tpl2_check(int which, int m, int relu) {
    const long na = (long)m * GK, nw = (long)GN * GK, nc = (long)m * GN;
    uint16_t *A = (uint16_t *)malloc(na * 2), *W = (uint16_t *)malloc(nw * 2), *ref = (uint16_t *)malloc(nc * 2);
    uint16_t *Ad = (uint16_t *)__Runtime_Alloc(na * 2), *Wd = (uint16_t *)__Runtime_Alloc(nw * 2);
    uint16_t *Cd = (uint16_t *)__Runtime_Alloc(nc * 2);
    ad_init(A, W, m, GN, GK);
    ad_pretile(A, Ad, m, GK, GTK, 0);
    ad_pretile(W, Wd, GN, GK, GTK, 1);
    memset(Cd, 0, nc * 2);
    XTime t0, t1;
    XTime_GetTime(&t0);
    tpl2_run(which, Ad, Wd, Cd);
    XTime_GetTime(&t1);
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc * 2);
    printf("[tpl2_gemm<%d, %d, %d, %d, %d>] launch %.3f ms\n", m, GN, GK, GTK, relu, ad_ms(t0, t1));
    ad_reference(A, W, ref, m, GN, GK, relu);
    return ad_check(Cd, ref, nc);
}

int main() {
    printf("=== tpl2_gemm: two instances of one template kernel ===\n");
    int rc = tpl2_check(0, M0, 1);
    rc |= tpl2_check(1, M1, 0);
    return rc;
}
