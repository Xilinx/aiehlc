/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the quant template on the board: two instances of quant<M, N, U8, TM> (f32 -> s8, f32 -> u8)
 * in one ELF. A third of the inputs are exact ties of the scale, some NaN / inf / huge: the self-check against
 * eager_quant_ref (computed here on the A72) checks the core's integer round-half-even of f / scale.
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "quant_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void q_s8_run(int8_t *A, int8_t *W, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    quant<128, 64, 0, 16><<<mesh>>>(A, W, Y);
}

static void q_u8_run(int8_t *A, int8_t *W, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    quant<64, 64, 1, 16><<<mesh>>>(A, W, Y);
}

static uint32_t qt_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static long quant_case(const char *name, long M, long N, int u8, double scale, double zp,
                       void (*run)(int8_t *, int8_t *, int8_t *)) {
    const long na = M * 4 * N, nw = N * 4 * N, ny = M * N, TN = N / 4;
    uint8_t *A = (uint8_t *)calloc(na, 1), *W = (uint8_t *)calloc(nw, 1);
    int8_t *got = (int8_t *)calloc(ny, 1), *want = (int8_t *)calloc(ny, 1);
    for (long i = 0; i < M * N; i++) {
        const uint32_t h = qt_mix((uint32_t)i + 77u);
        float f;
        switch (h % 6) {
        case 0:
        case 1:
            f = (float)(((int)(h >> 8) % 400 - 200 + 0.5) * scale);
            break; /* ties */
        case 2:
            f = (float)(((int)(h >> 8) % 400 - 200) * scale);
            break;
        case 3:
            f = (float)((((h >> 8) & 0xffff) / 65536.0 - 0.5) * 500.0 * scale);
            break;
        case 4:
            f = (h >> 8) % 3 == 0 ? (float)NAN : (h >> 9) % 2 ? (float)INFINITY : -1e30f;
            break;
        default:
            f = (float)((((h >> 8) & 0xffff) / 65536.0 - 0.5) * 2.0 * scale);
            break;
        }
        memcpy(A + 4 * i, &f, 4);
    }
    const float sf = (float)scale;
    uint32_t sz[2];
    memcpy(&sz[0], &sf, 4);
    sz[1] = (uint32_t)(int32_t)zp;
    for (long n = 0; n < N; n++)
        for (long j = 0; j < 4; j++)
            memcpy(W + n * 4 * N + j * N, sz, sizeof sz);
    (void)TN;
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
    const int64_t a_shape[2] = {M, 4 * N}, w_shape[2] = {N, 4 * N}, y_shape[2] = {M, N};
    const aiek_tensor t[3] = {{A, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)na},
                              {W, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)nw},
                              {want, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)ny}};
    eager_quant_ref(t, 3, M, N, u8);
    long bad = 0, first = -1;
    for (long i = 0; i < ny; i++)
        if (got[i] != want[i] && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld exact", name, ny - bad, ny);
    if (first >= 0) {
        float f;
        memcpy(&f, A + 4 * first, 4);
        printf(" (first at %ld: f %.9g got %d, want %d)", first, f, got[first], want[first]);
    }
    printf("\n");
    return bad;
}

int main() {
    printf(
        "=== aiednn eager quant<M, N, U8, 16>: f32 -> 8-bit on the core (exact integer division), two instances ===\n");
    long bad = quant_case("quant<128,64> s8", 128, 64, 0, 0.0371, 3.0, q_s8_run);
    bad += quant_case("quant<64,64> u8", 64, 64, 1, 0.125, 113.0, q_u8_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
