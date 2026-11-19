/* An ordinary app on the PREBUILT aiednn library: plain C++ against aiednn.h, compiled and linked with
 * aarch64-none-elf-g++ (Makefile + <dist>/link.mk) -- no aiehlc, no kernel source. Creating a primitive
 * is the query: dispatch picks the implementation (printed); each result is checked against plain loops. */
#include "aiednn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned s_rng = 12345;
static int rnd(int lo, int hi) {
    s_rng = s_rng * 1103515245u + 12345u;
    return lo + (int)((s_rng >> 8) % (unsigned)(hi - lo + 1));
}

/* per case "ok" / "bad:" -- only the final verdict is PASS: / FAIL: (the board runner stops at the
 * first FAIL: anywhere on the console). The candidate list is the query: every implementation that
 * accepted the problem, in dispatch rank. */
static int check(const char *what, int ok, aiednn_primitive_t p) {
    char cands[512] = "";
    if (p)
        aiednn_primitive_impl_candidates(p, cands, sizeof cands);
    printf("%s %s impl=%s candidates=%s\n", ok ? "ok  " : "bad:", what, p ? aiednn_primitive_impl_info(p) : "-", cands);
    return ok ? 0 : 1;
}

static int int8_matmul(void) { /* s8 [M,K] x s8 [N,K]^T -> s32: the AIE GEMM block */
    enum { M = 100, K = 300, N = 70 };
    static int8_t a[M * K], w[N * K];
    static int32_t c[M * N];
    for (int i = 0; i < M * K; i++)
        a[i] = (int8_t)rnd(-30, 30);
    for (int i = 0; i < N * K; i++)
        w[i] = (int8_t)rnd(-30, 30);
    aiednn_memory_desc_t ma, mw, mc;
    aiednn_memory_desc_init(&ma, 2, (const int64_t[]){M, K}, aiednn_s8);
    aiednn_memory_desc_init(&mw, 2, (const int64_t[]){N, K}, aiednn_s8);
    aiednn_memory_desc_init(&mc, 2, (const int64_t[]){M, N}, aiednn_s32);
    aiednn_primitive_t p;
    if (aiednn_matmul_forward_create(&p, &ma, &mw, nullptr, &mc, 1) != aiednn_success)
        return check("int8 matmul create", 0, nullptr);
    aiednn_primitive_execute(p, a, w, c);
    int ok = 1;
    for (int m = 0; m < M && ok; m++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += a[m * K + k] * w[n * K + k];
            if (s != c[m * N + n]) {
                ok = 0;
                break;
            }
        }
    int rc = check("int8 matmul 100x300x70 (exact)", ok, p);
    aiednn_primitive_destroy(p);
    return rc;
}

static int f32_matmul(void) { /* f32 [M,K] x [K,N] -> f32: bf16x3 on the AIE bf16 mode */
    enum { M = 20, K = 500, N = 40 };
    static float a[M * K], w[K * N], c[M * N];
    for (int i = 0; i < M * K; i++)
        a[i] = rnd(-1000, 1000) / 500.f;
    for (int i = 0; i < K * N; i++)
        w[i] = rnd(-1000, 1000) / 500.f;
    aiednn_memory_desc_t ma, mw, mc;
    aiednn_memory_desc_init(&ma, 2, (const int64_t[]){M, K}, aiednn_f32);
    aiednn_memory_desc_init(&mw, 2, (const int64_t[]){K, N}, aiednn_f32);
    aiednn_memory_desc_init(&mc, 2, (const int64_t[]){M, N}, aiednn_f32);
    aiednn_primitive_t p;
    if (aiednn_matmul_forward_create(&p, &ma, &mw, nullptr, &mc, 0) != aiednn_success)
        return check("f32 matmul create", 0, nullptr);
    aiednn_primitive_execute(p, a, w, c);
    double worst = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double s = 0, sa = 0;
            for (int k = 0; k < K; k++)
                s += (double)a[m * K + k] * w[k * N + n], sa += fabs((double)a[m * K + k] * w[k * N + n]);
            const double e = fabs(c[m * N + n] - s) / (sa + 1e-30);
            worst = e > worst ? e : worst;
        }
    printf("     f32 matmul worst error %.2e of sum|ab|\n", worst);
    int rc = check("f32 matmul 20x500x40 (< 1e-5)", worst < 1e-5, p);
    aiednn_primitive_destroy(p);
    return rc;
}

static int int8_conv(void) { /* s8 3x3 conv, pad 1 -> s32: im2col / direct conv on the AIE GEMM */
    enum { C = 16, H = 14, W = 14, O = 32 };
    static int8_t x[C * H * W], w[O * C * 9];
    static int32_t y[O * H * W];
    for (int i = 0; i < C * H * W; i++)
        x[i] = (int8_t)rnd(-20, 20);
    for (int i = 0; i < O * C * 9; i++)
        w[i] = (int8_t)rnd(-20, 20);
    aiednn_memory_desc_t mx, mw, my;
    aiednn_memory_desc_init(&mx, 4, (const int64_t[]){1, C, H, W}, aiednn_s8);
    aiednn_memory_desc_init(&mw, 4, (const int64_t[]){O, C, 3, 3}, aiednn_s8);
    aiednn_memory_desc_init(&my, 4, (const int64_t[]){1, O, H, W}, aiednn_s32);
    const aiednn_conv_params_t cp = {{1, 1}, {1, 1}, {1, 1}, {0, 0}, 1};
    aiednn_primitive_t p;
    if (aiednn_convolution_forward_create(&p, &mx, &mw, nullptr, &my, &cp) != aiednn_success)
        return check("int8 conv create", 0, nullptr);
    aiednn_primitive_execute(p, x, w, y);
    int ok = 1;
    for (int o = 0; o < O && ok; o++)
        for (int h = 0; h < H; h++)
            for (int v = 0; v < W; v++) {
                int32_t s = 0;
                for (int c = 0; c < C; c++)
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++) {
                            const int ih = h + i - 1, iw = v + j - 1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                s += x[(c * H + ih) * W + iw] * w[((o * C + c) * 3 + i) * 3 + j];
                        }
                if (s != y[(o * H + h) * W + v])
                    ok = 0;
            }
    int rc = check("int8 conv 3x3 16->32 14x14 (exact)", ok, p);
    aiednn_primitive_destroy(p);
    return rc;
}

int main() {
    printf("=== aiednn prebuilt library app ===\n");
    const int bad = int8_matmul() + f32_matmul() + int8_conv();
    if (bad) {
        printf("FAIL: %d check(s)\n", bad);
        return 1;
    }
    printf("PASS: prebuilt aiednn library, every check\n");
    return 0;
}
