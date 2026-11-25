/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// x86 check of aie_modes_bf16.h (the bf16 GEMM mode's layouts + core round) through the AIE-API emulation:
// pack A / B as the host does, slice each round's windows as the DMA delivers them, run the core rounds,
// compare C with the f32 reference.
#include "aie_api_x86emu.h"
using aie::accfloat;
using aie::bfloat16;
#include <cmath>
#include <vector>
#include <cstdio>
#include <random>
#include "aie_modes_bf16.h"

static uint16_t bf(float f) { return aie::bfloat16(f).bits; }

int main() {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-2.f, 2.f);
    int bad = 0;
    for (int t = 0; t < 40; t++) {
        const int kk = 1 + rng() % 600, rows = 1 + rng() % 32, nch = 1 + rng() % 32;
        std::vector<uint16_t> a(32 * 640, 0), w(32 * 640, 0);
        for (int m = 0; m < rows; m++)
            for (int k = 0; k < kk; k++)
                a[m * 640 + k] = bf(u(rng));
        for (int n = 0; n < nch; n++)
            for (int k = 0; k < kk; k++)
                w[n * 640 + k] = bf(u(rng));
        alignas(64) float acc[1024], out[1024], ref[1024];
        std::vector<uint8_t> la(32 * 512), lb(32 * 512);
        alignas(64) uint16_t wa[1024], wb[1024];
        const int nkb = (kk + AIE_BF16_K - 1) / AIE_BF16_K;
        for (int kb = 0; kb < nkb; kb++) { /* one launch per K block, the tile continuing */
            aie_bf16_pack_a(a.data(), rows, 640, (int64_t)kb * AIE_BF16_K, kk, la.data());
            aie_bf16_pack_b(w.data(), nch, 640, (int64_t)kb * AIE_BF16_K, kk, lb.data());
            for (int kr = 0; kr < 8; kr++) {
                for (int i = 0; i < 32; i++) { /* round kr's windows: 32 rows x 64 bytes */
                    memcpy((uint8_t *)wa + i * 64, la.data() + i * 512 + kr * 64, 64);
                    memcpy((uint8_t *)wb + i * 64, lb.data() + i * 512 + kr * 64, 64);
                }
                aie_bf16_round((const bfloat16 *)wa, (const bfloat16 *)wb, acc, kb == 0 && kr == 0);
            }
        }
        aie_bf16_out(acc, out);
        aie_bf16_ref_tile(a.data(), 640, w.data(), 640, kk, ref, 0);
        for (int m = 0; m < 32; m++)
            for (int n = 0; n < 32; n++) {
                const float r = (m < rows && n < nch) ? ref[m * 32 + n] : 0.f, g = out[m * 32 + n];
                if (std::fabs(g - r) > 1e-4f * (1.f + std::fabs(r)) * std::sqrt((float)kk)) {
                    if (bad++ < 5)
                        printf("MISMATCH k %d rows %d n %d at (%d,%d): %g vs %g\n", kk, rows, nch, m, n, g, r);
                }
            }
    }
    printf("bf16: %d mismatching\n", bad);
    return bad != 0;
}
