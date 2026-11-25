/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the depthwise template on the board: two instances of dwconv_q<M, C, TP, TM> in one ELF --
 * 3x3 taps over 64 channels and 5x5 taps over 32 channels, requant on the core. Self-checks every code against
 * eager_dwconv_q_ref (dwconv_q_eager_ref.h).
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "dwconv_q_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

static void dw3_run(int8_t *A, int8_t *W, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    dwconv_q<128, 64, 12, 16><<<mesh>>>(A, W, Y);
}

static void dw5_run(int8_t *A, int8_t *W, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    dwconv_q<64, 32, 28, 16><<<mesh>>>(A, W, Y);
}

static uint32_t dw_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static long dw_case(const char *name, long M, long C, long TP, int taps, void (*run)(int8_t *, int8_t *, int8_t *)) {
    const long TK = 4 * TP, K = C / 4 * TK, na = M * K, nw = C * K, ny = M * C;
    int8_t *A = (int8_t *)calloc(na, 1), *W = (int8_t *)calloc(nw, 1), *got = (int8_t *)calloc(ny, 1);
    int8_t *want = (int8_t *)calloc(ny, 1);
    for (long m = 0; m < M; m++)
        for (long c = 0; c < C; c++)
            for (long t = 0; t < taps; t++) /* taps past KH x KW stay 0; some pixels get padding (0) taps */
                if (!(m % 9 == 4 && t < 3))
                    A[m * K + c / 4 * TK + c % 4 * TP + t] =
                        (int8_t)((int)(dw_mix((uint32_t)(m * K + c * TP + t) + 5u) % 255u) - 127);
    for (long c = 0; c < C; c++) {
        int8_t *w = W + c * K + c / 4 * TK;
        for (long t = 0; t < taps; t++)
            w[t] = (int8_t)((int)(dw_mix((uint32_t)(c * TP + t) + 777u) % 255u) - 127);
        const uint32_t h = dw_mix((uint32_t)c + 31u);
        const int32_t Mq = (int32_t)(0x40000000u + (h & 0x3fffffffu)), sh = 36 + (int32_t)(h % 5u);
        const int64_t B = ((int64_t)((int32_t)(dw_mix(h) % 2001u) - 1000) << 28) + ((int64_t)1 << (sh - 1));
        const int32_t q[EG_Q_WORDS] = {Mq,
                                       (int32_t)(uint32_t)(uint64_t)B,
                                       (int32_t)(B >> 32),
                                       sh,
                                       (int32_t)(c % 7) - 3,
                                       c % 2 ? 0 : -128,
                                       c % 2 ? 255 : 127,
                                       0};
        memcpy(w + TP, q, sizeof q);
    }
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw),
           *Yd = (int8_t *)__Runtime_Alloc(ny);
    memcpy(Ad, A, na);
    memcpy(Wd, W, nw);
    memset(Yd, 0x55, ny);
    Xil_DCacheFlushRange((UINTPTR)Ad, na);
    Xil_DCacheFlushRange((UINTPTR)Wd, nw);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny);
    run(Ad, Wd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny);
    memcpy(got, Yd, ny);
    const int64_t a_shape[2] = {M, K}, w_shape[2] = {C, K}, y_shape[2] = {M, C};
    const aiek_tensor t[3] = {{A, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)na},
                              {W, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)nw},
                              {want, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)ny}};
    eager_dwconv_q_ref(t, 3, M, C, TP);
    long bad = 0, first = -1;
    for (long i = 0; i < ny; i++)
        if (got[i] != want[i] && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld exact", name, ny - bad, ny);
    if (first >= 0)
        printf(" (first at %ld: got %d, want %d)", first, got[first], want[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager dwconv_q<M, C, TP, 16>: depthwise + requant on the core, two instances ===\n");
    long bad = dw_case("dwconv_q<128,64,12> 3x3", 128, 64, 12, 9, dw3_run);
    bad += dw_case("dwconv_q<64,32,28> 5x5", 64, 32, 28, 25, dw5_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
