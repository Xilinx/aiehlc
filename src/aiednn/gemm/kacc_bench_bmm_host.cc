
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h:936 */

static double ms(XTime a, XTime b) { return 1.0 * (b - a) / COUNTS_PER_SECOND * 1000.0; }

/* launch s of head h: Y slice -> out (Y [H][M][N] bf16). Instance index: m = px (X rows), n = ch (W rows). */
static void kacc_unpad_launch(uint16_t *Yo, const int32_t *Yd, int h, int s) {
    const size_t yh = (size_t)KACC_CHP * KACC_PXP;
    const int p0 = KACC_SPLIT_PX ? s * KACC_PXS : 0, c0 = KACC_SPLIT_PX ? 0 : s * KACC_CHS;
    const int pr = KACC_PXR - p0 < KACC_PXS ? KACC_PXR - p0 : KACC_PXS;
    const int cr = KACC_CHR - c0 < KACC_CHS ? KACC_CHR - c0 : KACC_CHS;
    if (pr <= 0 || cr <= 0)
        return;
    const size_t off = KACC_SWAP ? (size_t)c0 * KACC_N + p0 : (size_t)p0 * KACC_N + c0;
    akp_unpad(Yo + (size_t)h * KACC_M * KACC_N + off, KACC_N,
              (const float *)(Yd + h * yh + (size_t)s * KACC_CHS * KACC_PXS), KACC_PXS, KACC_CHS, KACC_TR, pr, cr,
              KACC_SWAP);
}

int main() {
    printf("=== PBK gemm_kacc attention bmm H=%d M=%d K=%d N=%d swap=%d (ch %d px %d kp %d): %d launch(es)/head of "
           "@NAME@, runtime host pack, 4x4 mesh ===\n", KACC_H, KACC_M, KACC_K, KACC_N, KACC_SWAP, KACC_CHP, KACC_PXP,
           KACC_KP, KACC_S);
    __Runtime_ctrl_high_throughput_enable(1);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, 4, 4);
    const size_t wh = (size_t)KACC_CHP * KACC_KP, xh = (size_t)KACC_PXP * KACC_KP, yh = (size_t)KACC_CHP * KACC_PXP;
    const size_t wbytes = KACC_H * wh * 2, xbytes = KACC_H * xh * 2, ybytes = KACC_H * yh * 4;
    int32_t *Wd = (int32_t *)__Runtime_Alloc(wbytes);
    int32_t *Xd = (int32_t *)__Runtime_Alloc(xbytes);
    int32_t *Yd = (int32_t *)__Runtime_Alloc(ybytes);
    uint16_t *Yo = (uint16_t *)aligned_alloc(64, ((size_t)KACC_H * KACC_M * KACC_N * 2 + 63) & ~(size_t)63);
    if (!Wd || !Xd || !Yd || !Yo) {
        printf("KCMP %s data=%s y_sha256=%s cos=0.000000 FAIL (alloc)\n", KACC_TAG, KACC_DATA, KACC_Y_SHA);
        return 1;
    }
    const float *A = (const float *)kacc_a_bin, *B = (const float *)kacc_b_bin; /* natural f32 activations */
    /* W side = ch rows, X side = px rows. A [M][K] row-major: rows stride K, k stride 1. B^T rows = B columns:
     * row stride 1, k stride N. */
    const float *Wsrc = KACC_SWAP ? A : B, *Xsrc = KACC_SWAP ? B : A;
    const long wrs = KACC_SWAP ? KACC_K : 1, wks = KACC_SWAP ? 1 : KACC_N;
    const long xrs = KACC_SWAP ? 1 : KACC_K, xks = KACC_SWAP ? KACC_N : 1;
    const size_t aph = (size_t)KACC_M * KACC_K, bph = (size_t)KACC_K * KACC_N; /* per-head A / B elements */
    const size_t wph = KACC_SWAP ? aph : bph, xph = KACC_SWAP ? bph : aph;
    /* the device operand buffers are persistent: their zero pad (K / rows past the real data) is written once
     * here; the per-call pack writes every tile that holds real data (akp_pack zeroed=1 skips pure-pad tiles) */
    memset(Wd, 0, wbytes);
    memset(Xd, 0, xbytes);
    Xil_DCacheFlushRange((UINTPTR)Wd, wbytes);
    Xil_DCacheFlushRange((UINTPTR)Xd, xbytes);
    double aie[2] = {0, 0}, host[2] = {0, 0}, hpack[2] = {0, 0}, hcache[2] = {0, 0};
    int launches = 0;
    for (int pass = 0; pass < 2; pass++) { /* pass 0: warm-up (kernel load), pass 1: timed */
        memset(Yd, 0xff, ybytes);          /* sentinel: an element the AIE does not write is nonfinite */
        Xil_DCacheFlushRange((UINTPTR)Yd, ybytes);
        XTime t0, tf, t1, t2, ti, t3;
        XTime_GetTime(&t0);
        for (int h = 0; h < KACC_H; h++) { /* A / B packed at run time from their natural layout */
            akp_pack((uint16_t *)Wd + h * wh, Wsrc + h * wph, KACC_CHP,
                     KACC_CHR, KACC_KP, KACC_K, KACC_KCH, wrs, wks, 1, 1);
            akp_pack((uint16_t *)Xd + h * xh, Xsrc + h * xph, KACC_PXP,
                     KACC_PXR, KACC_KP, KACC_K, KACC_KCH, xrs, xks, 0, 1);
        }
        XTime_GetTime(&tf);
        Xil_DCacheFlushRange((UINTPTR)Wd, wbytes);
        Xil_DCacheFlushRange((UINTPTR)Xd, xbytes);
        XTime_GetTime(&t1);
        for (int h = 0; h < KACC_H; h++)
            for (int s = 0; s < KACC_S; s++) {
                int32_t *Ws = Wd + h * wh / 2 + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
                int32_t *Xs = Xd + h * xh / 2 + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
                int32_t *Ys = Yd + h * yh + (size_t)s * KACC_CHS * KACC_PXS;
                @NAME@<<<mesh>>>(@ARGS@);
                if (pass)
                    launches++;
            }
        XTime_GetTime(&t2);
        if (ybytes >= ((size_t)8 << 20))
            Xil_DCacheFlush(); /* set/way clean+invalidate of the whole D-cache: 0.36 ms vs 1.4 ms for 0.5M line
                                  ops (c004); Yd has no dirty lines (the CPU wrote it only before its flush) */
        else
            Xil_DCacheInvalidateRange((UINTPTR)Yd, ybytes);
        XTime_GetTime(&ti);
        for (int h = 0; h < KACC_H; h++)
            for (int s = 0; s < KACC_S; s++)
                kacc_unpad_launch(Yo, Yd, h, s);
        XTime_GetTime(&t3);
        aie[pass] = ms(t1, t2);
        hpack[pass] = ms(t0, tf);
        hcache[pass] = ms(tf, t1) + ms(t2, ti);
        host[pass] = ms(t0, t1) + ms(t2, t3);
    }
    /* winlint: ok -- after the timed passes (t3), before the untimed-in-[BENCH] unpad sweep */
    printf("[BENCH] aie_wall_ms=%.3f host_ms=%.3f launches=%d host_pack_ms=%.3f host_cache_ms=%.3f host_unpad_ms=%.3f "
           "warmup_aie_ms=%.3f warmup_host_ms=%.3f heads=%d swap=%d gmac_padded=%.4f ms_per_gmac=%.2f\n", aie[1],
           host[1], launches, hpack[1], hcache[1], host[1] - hpack[1] - hcache[1], aie[0], host[0], KACC_H,
           KACC_SWAP, KACC_GMAC, aie[1] / KACC_GMAC);
#ifdef KACC_UNPAD_SWEEP
    { /* PBK diagnostic (not timed in [BENCH]): unpad variants on the same device result */
        static const int cfgs[] = {0, 1, 2, 3};
        for (unsigned v = 0; v < sizeof(cfgs) / sizeof(cfgs[0]); v++) {
            akp_unpad_cfg = cfgs[v];
            XTime u0, u1;
            Xil_DCacheFlush();
            XTime_GetTime(&u0);
            for (int h = 0; h < KACC_H; h++)
                for (int s = 0; s < KACC_S; s++)
                    kacc_unpad_launch(Yo, Yd, h, s);
            XTime_GetTime(&u1);
            printf("[PBKU] cfg=0x%x unpad_ms=%.3f\n", cfgs[v], ms(u0, u1));
        }
        akp_unpad_cfg = AKP_UNPAD_CFG;
    }
#endif
    const float *ref = (const float *)kacc_y_bin; /* [H][M][N] */
    double dot = 0, na = 0, nr = 0, maxabs = 0;
    int nonfinite = 0;
    for (size_t i = 0; i < (size_t)KACC_H * KACC_M * KACC_N; i++) {
        const uint32_t u = (uint32_t)Yo[i] << 16;
        float f;
        memcpy(&f, &u, 4);
        const double o = (double)f, r = ref[i];
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
    const int pass = nonfinite == 0 && cs >= 0.999;
    for (int h = 0; h < KACC_H; h++) { /* per-head / per-quadrant diagnostics */
        double d4[4] = {0}, a4[4] = {0}, r4[4] = {0};
        for (int m = 0; m < KACC_M; m++)
            for (int n = 0; n < KACC_N; n++) {
                const size_t i = ((size_t)h * KACC_M + m) * KACC_N + n;
                const uint32_t u = (uint32_t)Yo[i] << 16;
                float f;
                memcpy(&f, &u, 4);
                const int qd = (m >= KACC_M / 2) * 2 + (n >= KACC_N / 2);
                d4[qd] += (double)f * ref[i];
                a4[qd] += (double)f * f;
                r4[qd] += (double)ref[i] * ref[i];
            }
        printf("[PBKH] head=%d cos_q=%.4f,%.4f,%.4f,%.4f\n", h, d4[0] / sqrt(a4[0] * r4[0] + 1e-30),
               d4[1] / sqrt(a4[1] * r4[1] + 1e-30), d4[2] / sqrt(a4[2] * r4[2] + 1e-30),
               d4[3] / sqrt(a4[3] * r4[3] + 1e-30));
    }
    printf("[PBK] maxabs_vs_ref=%.6g nonfinite=%d H=%d M=%d N=%d K=%d out=bf16\n", maxabs, nonfinite, KACC_H, KACC_M,
           KACC_N, KACC_K);
    printf("KCMP %s data=%s y_sha256=%s cos=%.6f %s\n", KACC_TAG, KACC_DATA, KACC_Y_SHA, cs, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
