/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#pragma control_packet_mode(aot)
#pragma control_plan_op_control_packet
#pragma CONTROL_PLAN_GROUP_REG_WRITE
#pragma control_plan_dedicated_shim
#pragma control_plan_shim_bd_ctrl
#pragma KERNELCONFIGOFFLOAD
// kco2.cc -- GAP-125ece regression fixture (KCO-BISECT): two gemm_kacc kernels whose core output windows differ
// (gemm_kacc_320x320x1024: 5120 B, gemm_kacc_960x320x512: 8192 B) in one program under #pragma KERNELCONFIGOFFLOAD.
// Kernel texts = unet_aie.kacc_kernel_text (as in the UNet KCO build); host main reduced to the launches (no data).
#define AIEDNN_KACC_KERNEL 1
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__ void gemm_kacc_320x320x1024(aie::port<input_window_int32 *, aie::IFM, aie::shape<1024, 160>> a_320x320x1024,
                          aie::port<input_window_int32 *, aie::WEIGHTS, aie::shape<320, 160>> b_320x320x1024,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<1024, 320>> c_320x320x1024) {
    // ports named per instance: aiehlc's block descriptors (aiehlc_desc_<port>) are file-scope statics
    // P4bg: gemm_kacc (bf16_gemm_kacc.cc, Maxwell P4bk) for ch=320 K=320 px=1024 -- see unet_aie.kacc_geom
    const int tm = 16, tn = 16, tkw = 8, kch = 16; // tile; K chunk = tkw words = kch bf16
    const int nkc = 20;   // K chunks per tile
    const int ntile = 16;  // a-port 16-row tiles per core (channels; pixels when transposed)
    const int ns = 5;      // 16-row sub-tiles of the core's b block (pixels; channels when transposed)
    // transposed=1: the mmul's left operand is always W (pre-tiled [mi][kb][8x8 ch x k]), right is X
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * tm) + t * tm;
        // P4bx: the c window {16, ns*16} fp32 is the accumulator: ns 16x16 tiles in [s][mi][ni][8x8] order
        int32_t *y = (int32_t *)aie::store_buffer(c_320x320x1024, {m0, pn * (ns * tn)}, {tm, ns * tn});
        float *const kacc = (float *)y;
        for (int i = 0; i < ns * 256; i++)
            y[i] = 0;
        for (int j = 0; j < nkc; j++) {
            const int32_t *x = (const int32_t *)aie::load_buffer(a_320x320x1024, {m0, j * tkw}, {tm, tkw});
            const int32_t *w = (const int32_t *)aie::load_buffer(b_320x320x1024, {pn * (ns * tn), j * tkw}, {ns * tn, tkw});
            for (int s = 0; s < ns; s++) {
                float *ac = kacc + s * 256;
#if defined(__chess__)
                using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
                aie::accum<accfloat, 64> q00, q01, q10, q11;
                q00.from_vector(aie::load_v<64>(ac));
                q01.from_vector(aie::load_v<64>(ac + 64));
                q10.from_vector(aie::load_v<64>(ac + 128));
                q11.from_vector(aie::load_v<64>(ac + 192));
                MM c00(q00), c01(q01), c10(q10), c11(q11);
                const bfloat16 *a0 = (const bfloat16 *)((const uint16_t *)w + s * 16 * kch), *a1 = a0 + 8 * kch;
                const bfloat16 *b0 = (const bfloat16 *)(x), *b1 = b0 + 8 * kch;
                for (int kb = 0; kb < kch / 8; kb++) {
                    auto va0 = aie::load_v<64>(a0 + kb * 64);
                    auto va1 = aie::load_v<64>(a1 + kb * 64);
                    auto vb0 = aie::load_v<64>(b0 + kb * 64);
                    auto vb1 = aie::load_v<64>(b1 + kb * 64);
                    c00.mac(va0, vb0);
                    c01.mac(va0, vb1);
                    c10.mac(va1, vb0);
                    c11.mac(va1, vb1);
                }
                aie::store_v(ac, c00.to_vector<float>());
                aie::store_v(ac + 64, c01.to_vector<float>());
                aie::store_v(ac + 128, c10.to_vector<float>());
                aie::store_v(ac + 192, c11.to_vector<float>());
#else // scalar, same pre-tiled layout (chunk = [mi 2][kb kch/8][8x8])
                const uint16_t *xs = (const uint16_t *)((const uint16_t *)w + s * 16 * kch), *ws = (const uint16_t *)(x);
                for (int m = 0; m < tm; m++)
                    for (int n = 0; n < tn; n++) {
                        float sum = 0.0f;
                        for (int k = 0; k < kch; k++) {
                            union {
                                uint32_t u;
                                float f;
                            } xa, wb;
                            xa.u = (uint32_t)xs[((m >> 3) * (kch / 8) + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7)] << 16;
                            wb.u = (uint32_t)ws[((n >> 3) * (kch / 8) + (k >> 3)) * 64 + (k & 7) * 8 + (n & 7)] << 16;
                            sum += xa.f * wb.f;
                        }
                        ac[((m >> 3) * 2 + (n >> 3)) * 64 + (m & 7) * 8 + (n & 7)] += sum;
                    }
#endif
            }
            aie::release(a_320x320x1024);
            aie::release(b_320x320x1024);
        }
        aie::commit(c_320x320x1024); // one fp32 write-back of the core's 16 x px/4 block after the last chunk
    }
}

__global__ void gemm_kacc_960x320x512(aie::port<input_window_int32 *, aie::IFM, aie::shape<960, 160>> a_960x320x512,
                          aie::port<input_window_int32 *, aie::WEIGHTS, aie::shape<512, 160>> b_960x320x512,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<960, 512>> c_960x320x512) {
    // ports named per instance: aiehlc's block descriptors (aiehlc_desc_<port>) are file-scope statics
    // P4bg: gemm_kacc (bf16_gemm_kacc.cc, Maxwell P4bk) for ch=960 K=320 px=512 -- see unet_aie.kacc_geom
    const int tm = 16, tn = 16, tkw = 8, kch = 16; // tile; K chunk = tkw words = kch bf16
    const int nkc = 20;   // K chunks per tile
    const int ntile = 15;  // a-port 16-row tiles per core (channels; pixels when transposed)
    const int ns = 8;      // 16-row sub-tiles of the core's b block (pixels; channels when transposed)
    // transposed=0: the mmul's left operand is always W (pre-tiled [mi][kb][8x8 ch x k]), right is X
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * tm) + t * tm;
        // P4bx: the c window {16, ns*16} fp32 is the accumulator: ns 16x16 tiles in [s][mi][ni][8x8] order
        int32_t *y = (int32_t *)aie::store_buffer(c_960x320x512, {m0, pn * (ns * tn)}, {tm, ns * tn});
        float *const kacc = (float *)y;
        for (int i = 0; i < ns * 256; i++)
            y[i] = 0;
        for (int j = 0; j < nkc; j++) {
            const int32_t *x = (const int32_t *)aie::load_buffer(a_960x320x512, {m0, j * tkw}, {tm, tkw});
            const int32_t *w = (const int32_t *)aie::load_buffer(b_960x320x512, {pn * (ns * tn), j * tkw}, {ns * tn, tkw});
            for (int s = 0; s < ns; s++) {
                float *ac = kacc + s * 256;
#if defined(__chess__)
                using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
                aie::accum<accfloat, 64> q00, q01, q10, q11;
                q00.from_vector(aie::load_v<64>(ac));
                q01.from_vector(aie::load_v<64>(ac + 64));
                q10.from_vector(aie::load_v<64>(ac + 128));
                q11.from_vector(aie::load_v<64>(ac + 192));
                MM c00(q00), c01(q01), c10(q10), c11(q11);
                const bfloat16 *a0 = (const bfloat16 *)(x), *a1 = a0 + 8 * kch;
                const bfloat16 *b0 = (const bfloat16 *)((const uint16_t *)w + s * 16 * kch), *b1 = b0 + 8 * kch;
                for (int kb = 0; kb < kch / 8; kb++) {
                    auto va0 = aie::load_v<64>(a0 + kb * 64);
                    auto va1 = aie::load_v<64>(a1 + kb * 64);
                    auto vb0 = aie::load_v<64>(b0 + kb * 64);
                    auto vb1 = aie::load_v<64>(b1 + kb * 64);
                    c00.mac(va0, vb0);
                    c01.mac(va0, vb1);
                    c10.mac(va1, vb0);
                    c11.mac(va1, vb1);
                }
                aie::store_v(ac, c00.to_vector<float>());
                aie::store_v(ac + 64, c01.to_vector<float>());
                aie::store_v(ac + 128, c10.to_vector<float>());
                aie::store_v(ac + 192, c11.to_vector<float>());
#else // scalar, same pre-tiled layout (chunk = [mi 2][kb kch/8][8x8])
                const uint16_t *xs = (const uint16_t *)(x), *ws = (const uint16_t *)((const uint16_t *)w + s * 16 * kch);
                for (int m = 0; m < tm; m++)
                    for (int n = 0; n < tn; n++) {
                        float sum = 0.0f;
                        for (int k = 0; k < kch; k++) {
                            union {
                                uint32_t u;
                                float f;
                            } xa, wb;
                            xa.u = (uint32_t)xs[((m >> 3) * (kch / 8) + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7)] << 16;
                            wb.u = (uint32_t)ws[((n >> 3) * (kch / 8) + (k >> 3)) * 64 + (k & 7) * 8 + (n & 7)] << 16;
                            sum += xa.f * wb.f;
                        }
                        ac[((m >> 3) * 2 + (n >> 3)) * 64 + (m & 7) * 8 + (n & 7)] += sum;
                    }
#endif
            }
            aie::release(a_960x320x512);
            aie::release(b_960x320x512);
        }
        aie::commit(c_960x320x512); // one fp32 write-back of the core's 16 x px/4 block after the last chunk
    }
}


static int k2_launch(int which, int32_t *w, int32_t *x, int32_t *y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, 4, 4);
    if (which == 0) {
        gemm_kacc_320x320x1024<<<mesh>>>(x, w, y);
        return 0;
    }
    if (which == 1) {
        gemm_kacc_960x320x512<<<mesh>>>(w, x, y);
        return 0;
    }
    return -1;
}

int main() {
    int32_t *Wd = (int32_t *)__Runtime_Alloc(960 * 320 * 2);
    int32_t *Xd = (int32_t *)__Runtime_Alloc(1024 * 320 * 2);
    int32_t *Yd = (int32_t *)__Runtime_Alloc(960 * 1024 * 4);
    k2_launch(0, Wd, Xd, Yd);
    k2_launch(1, Wd, Xd, Yd);
    k2_launch(1, Wd, Xd + 512 * 160, Yd + 960 * 512);
    printf("kco2 done\n");
    return 0;
}
