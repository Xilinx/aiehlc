
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h:936 */

/* device index of Y[m][n] in launch s's fp32 c buffer (unet_aie uc_conv un-tiling, P4bx [s][mi][ni][8x8]) */
static size_t kacc_yidx(int m, int n) {
    const int s = KACC_SPLIT_PX ? m / KACC_PXS : n / KACC_CHS;
    const int ml = KACC_SPLIT_PX ? m % KACC_PXS : m, nl = KACC_SPLIT_PX ? n : n % KACC_CHS;
    const int Ml = KACC_PXS, Nl = KACC_CHS, r = nl & 15, c = ml & 15;
    size_t base = (size_t)s * Ml * Nl, i;
    if (!KACC_TR) {
        const int q = Ml / 4, cc = ml % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        i = (size_t)((nl & ~15) + wi / q) * Ml + (ml - cc) + wi % q;
    } else {
        const int q = Nl / 4, cc = nl % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        i = (size_t)((ml & ~15) + wi / q) * Nl + (nl - cc) + wi % q;
    }
    return base + i;
}

int main() {
    printf("=== PDK gemm_kacc dense GEMM M=%d (pad %d) K=%d N=%d: %d launch(es) of @NAME@ per pass, 4x4 mesh ===\n",
           KACC_M, KACC_MP, KACC_K, KACC_N, KACC_S);
    __Runtime_ctrl_high_throughput_enable(1);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, 4, 4);
    const size_t wbytes = (size_t)KACC_N * KACC_KP * 2, xbytes = (size_t)KACC_MP * KACC_KP * 2;
    const size_t ybytes = (size_t)KACC_N * KACC_MP * 4;
    int32_t *Wd = (int32_t *)__Runtime_Alloc(wbytes);
    int32_t *Xd = (int32_t *)__Runtime_Alloc(xbytes);
    int32_t *Yd = (int32_t *)__Runtime_Alloc(ybytes);
    if (!Wd || !Xd || !Yd) {
        printf("KCMP %s data=%s y_sha256=%s cos=0.000000 FAIL (alloc)\n", KACC_TAG, KACC_DATA, KACC_Y_SHA);
        return 1;
    }
    memcpy(Wd, kacc_w_bin, wbytes);
    memcpy(Xd, kacc_x_bin, xbytes);
    Xil_DCacheFlushRange((UINTPTR)Wd, wbytes);
    Xil_DCacheFlushRange((UINTPTR)Xd, xbytes);
    double wall[2] = {0, 0};
    int launches = 0;
    for (int pass = 0; pass < 2; pass++) { /* pass 0: warm-up (kernel load), pass 1: timed */
        memset(Yd, 0xff, ybytes);          /* sentinel: an element the timed pass does not write is nonfinite */
        Xil_DCacheFlushRange((UINTPTR)Yd, ybytes);
        XTime t0, t1;
        XTime_GetTime(&t0);
        for (int s = 0; s < KACC_S; s++) {
            int32_t *Ws = Wd + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
            int32_t *Xs = Xd + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
            int32_t *Ys = Yd + (size_t)s * KACC_CHS * KACC_PXS;
            @NAME@<<<mesh>>>(@ARGS@);
            if (pass)
                launches++;
        }
        XTime_GetTime(&t1);
        wall[pass] = 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0;
        Xil_DCacheInvalidateRange((UINTPTR)Yd, ybytes);
    }
    printf("[BENCH] aie_wall_ms=%.3f launches=%d warmup_wall_ms=%.3f warmup_launches=%d gmac=%.4f ms_per_gmac=%.2f\n",
           wall[1], launches, wall[0], KACC_S, KACC_GMAC, wall[1] / KACC_GMAC);
    const float *ref = (const float *)kacc_y_bin; /* [M][N] */
    double dot = 0, na = 0, nr = 0, maxabs = 0;
    int nonfinite = 0;
    for (int m = 0; m < KACC_M; m++)
        for (int n = 0; n < KACC_N; n++) {
            float f;
            memcpy(&f, &Yd[kacc_yidx(m, n)], 4);
            const double o = (double)f, r = ref[(size_t)m * KACC_N + n];
            if (!(o == o) || o > 1e30 || o < -1e30) {
                nonfinite++;
                continue;
            }
            dot += o * r;
            na += o * o;
            nr += r * r;
            if (fabs(o - r) > maxabs)
                maxabs = fabs(o - r);
        }
    const double cs = nonfinite ? 0.0 : dot / (sqrt(na) * sqrt(nr) + 1e-30);
    const int pass = nonfinite == 0 && cs >= 0.9995;
    printf("[PDK] maxabs_vs_ref=%.6g nonfinite=%d M=%d N=%d K=%d\n", maxabs, nonfinite, KACC_M, KACC_N, KACC_K);
    printf("KCMP %s data=%s y_sha256=%s cos=%.6f %s\n", KACC_TAG, KACC_DATA, KACC_Y_SHA, cs, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
