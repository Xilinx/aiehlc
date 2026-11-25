/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// x86 check of aie_modes_wq.h (the grouped weight-only quant mode) through the AIE-API emulation: run
// the core's round structure -- int32 8x8-tiled partial sums per 64-K round, aie_wq_flush at each group's
// last round, the int32 partial and the f32 tile carried across launches -- for group sizes 64 .. 1024
// and K spanning several launches, and compare with aie_wq_ref_tile.
#include "aie_api_x86emu.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#define AIE_WQ_EMU 1
#include "aie_modes_wq.h"

int main() {
    std::mt19937 rng(5);
    int bad = 0;
    for (int gr : {1, 2, 4, 8, 16}) {
        const int nl = 4; /* launches: K = 2048 */
        const int KT = 512 * nl;
        std::vector<int8_t> a(32 * KT), w(32 * KT);
        for (auto &x : a)
            x = (int8_t)(rng() % 255 - 127);
        for (auto &x : w)
            x = (int8_t)(rng() % 255 - 127);
        std::vector<float> sc(KT / 64 * 32); /* per global round: the scales of the group ending there */
        for (auto &x : sc)
            x = (float)(rng() % 2000 + 1) * 1e-5f;
        alignas(64) int32_t acc[1024];
        alignas(64) float f[1024];
        int32_t racc[1024];
        float rf[1024];
        for (int l = 0; l < nl; l++) {
            const int phase = (l * 8) % gr, flags = (l > 0 ? 1 : 0) | (l < nl - 1 ? 2 : 0);
            /* core */
            if (!(flags & 1))
                for (int i = 0; i < 1024; i++)
                    f[i] = 0.f;
            for (int kr = 0; kr < 8; kr++) {
                const bool fresh = (phase + kr) % gr == 0 && !((flags & 1) && kr == 0 && phase != 0);
                for (int it = 0; it < 4; it++)
                    for (int ct = 0; ct < 4; ct++)
                        for (int i = 0; i < 8; i++)
                            for (int j = 0; j < 8; j++) {
                                int32_t s = 0;
                                for (int k = 0; k < 64; k++)
                                    s += (int32_t)a[(it * 8 + i) * KT + l * 512 + kr * 64 + k] *
                                         (int32_t)w[(ct * 8 + j) * KT + l * 512 + kr * 64 + k];
                                int32_t &c = acc[(it * 4 + ct) * 64 + i * 8 + j];
                                c = fresh ? s : c + s;
                            }
                if ((phase + kr) % gr == gr - 1) {
                    const float *s = sc.data() + (l * 8 + kr) * 32;
                    aie_wq_flush(acc, f, s, s + 16, 4, 4);
                }
            }
            /* reference */
            aie_wq_ref_tile(a.data() + l * 512, KT, w.data() + l * 512, KT, sc.data() + l * 8 * 32, gr, phase, flags,
                            racc, rf);
        }
        /* and in double, straight from the definition */
        for (int m = 0; m < 32; m++)
            for (int n = 0; n < 32; n++) {
                double y = 0, mag = 0;
                for (int g = 0; g < KT / (64 * gr); g++) {
                    int64_t p = 0;
                    for (int k = g * 64 * gr; k < (g + 1) * 64 * gr; k++)
                        p += (int64_t)a[m * KT + k] * w[n * KT + k];
                    const double s = sc[((g + 1) * gr - 1) * 32 + n];
                    y += s * p, mag += std::fabs(s * p);
                }
                const double e1 = std::fabs(f[m * 32 + n] - y), e2 = std::fabs(rf[m * 32 + n] - y);
                if (e1 > 1e-6 * (mag + 1e-30) || e2 > 1e-6 * (mag + 1e-30)) {
                    if (bad++ < 5)
                        printf("MISMATCH gr %d (%d,%d): core %g ref %g double %g\n", gr, m, n, f[m * 32 + n],
                               rf[m * 32 + n], y);
                }
            }
    }
    printf("wq: %d mismatching\n", bad);
    return bad != 0;
}
