/* script/perf/p0_kacc_host.cc -- cost-model P0 probe host for one gemm_kacc instance (kacc_gemm.py, KACC_HOST_FILE).
 * Same data/numerics as src/aiednn/gemm/kacc_bench_gemm_host.cc, plus:
 *   - "BOARD machine=aarch64 midr=0x%08lx" (MIDR_EL1) first; the wrapper appends " board=<name>";
 *   - AIE core perf counters armed at every launch (--profiling build + g_runtime_debug_level 0x100) and read once per
 *     timed launch AFTER the launch returned (wait_io done, Y invalidated), outside every timed region:
 *     "[PERF] tag=<launch> ..." per core tile + "[PERFSUM] tag=<launch> ...";
 *   - per launch the host regions of kacc_chunk_board.cc uc_launch (flush W/X/Y | launch | invalidate Y, pmccntr),
 *     and the runtime phase / wait_io accumulators (delta over the timed pass) printed in the
 *     src/aiegraph/runtime/main_launch_line.c format ("LAUNCH split" / "LAUNCH split_detail calls").
 * KACC_REPS (default 8) timed passes of the KACC_S launches after one warm-up pass. */
#ifndef KACC_REPS
#define KACC_REPS 8
#endif
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h:936 */
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
void __Runtime_core_perf_report(int tag);
void __Runtime_launch_words(unsigned long long *mmio_w, unsigned long long *cp_w, unsigned long long *ph_w,
                            unsigned int *bd_n, unsigned int *rst_n, unsigned long long *miss);
/* pmu_hz fallback when XTime does not advance (VEK385 A78: CNTPCT frozen unless the SCNTRS is started): the A78
 * PMCCNTR rate measured against XTime on vek385-12 (1.8675 GHz). Palmyra always takes the xtime branch. */
#ifndef P0_A78_FALLBACK_HZ
#define P0_A78_FALLBACK_HZ 1867500000ULL
#endif
int g_runtime_debug_level = 0x100; /* AIE_DEBUG_FLAG_CORE_PERF_COUNTER (aie_runtime.h:69); overrides the weak 0 */

static inline unsigned long long p0_pmc(void) {
    unsigned long long v;
#if defined(__aarch64__)
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v));
#else
    v = 0ULL;
#endif
    return v;
}
static inline unsigned long long p0_rt_cyc(void) {
    unsigned long long ph[4] = {0, 0, 0, 0}, w = 0;
    unsigned int pn[4] = {0, 0, 0, 0}, wn = 0;
    __Runtime_phase_cycles(ph, pn);
    __Runtime_wait_io_cycles(&w, &wn);
    return ph[0] + ph[1] + ph[2] + ph[3] + w;
}

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
#if defined(__aarch64__)
    {
        unsigned long midr;
        __asm__ volatile("mrs %0, midr_el1" : "=r"(midr));
        printf("BOARD machine=aarch64 midr=0x%08lx\n", midr);
    }
#else
    printf("BOARD machine=other midr=0x00000000\n");
#endif
    printf("=== P0 gemm_kacc M=%d (pad %d) K=%d N=%d: %d launch(es) of @NAME@ per pass x %d timed passes, 4x4 mesh, "
           "core perf counters on ===\n",
           KACC_M, KACC_MP, KACC_K, KACC_N, KACC_S, KACC_REPS);
    __Runtime_ctrl_high_throughput_enable(1);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, 4, 4);
    const size_t wbytes = (size_t)KACC_N * KACC_KP * 2, xbytes = (size_t)KACC_MP * KACC_KP * 2;
    const size_t ybytes = (size_t)KACC_N * KACC_MP * 4;
#ifndef P0_BASE_OFF
#define P0_BASE_OFF 0 /* bytes added to the W and X base addresses (DDR address-mapping probe; multiple of 64) */
#endif
    char *Wraw = (char *)__Runtime_Alloc(wbytes + P0_BASE_OFF), *Xraw = (char *)__Runtime_Alloc(xbytes + P0_BASE_OFF);
    int32_t *Wd = Wraw ? (int32_t *)(Wraw + P0_BASE_OFF) : 0;
    int32_t *Xd = Xraw ? (int32_t *)(Xraw + P0_BASE_OFF) : 0;
    int32_t *Yd = (int32_t *)__Runtime_Alloc(ybytes);
    printf("[P0] base W=%p X=%p Y=%p off=%d\n", (void *)Wd, (void *)Xd, (void *)Yd, (int)P0_BASE_OFF);
    if (!Wd || !Xd || !Yd) {
        printf("KCMP %s data=%s y_sha256=%s cos=0.000000 FAIL (alloc)\n", KACC_TAG, KACC_DATA, KACC_Y_SHA);
        return 1;
    }
    memcpy(Wd, kacc_w_bin, wbytes);
    memcpy(Xd, kacc_x_bin, xbytes);
    Xil_DCacheFlushRange((UINTPTR)Wd, wbytes);
    Xil_DCacheFlushRange((UINTPTR)Xd, xbytes);
    /* warm-up pass: kernel load + first-launch config, not counted */
    memset(Yd, 0xff, ybytes);
    Xil_DCacheFlushRange((UINTPTR)Yd, ybytes);
    for (int s = 0; s < KACC_S; s++) {
        int32_t *Ws = Wd + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
        int32_t *Xs = Xd + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
        int32_t *Ys = Yd + (size_t)s * KACC_CHS * KACC_PXS;
        @NAME@<<<mesh>>>(@ARGS@);
    }
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ybytes);
    printf("[P0] warmup done launches=%d\n", KACC_S);

    unsigned long long ph0[4] = {0, 0, 0, 0}, ph1[4] = {0, 0, 0, 0}, wio0 = 0, wio1 = 0;
    unsigned int phn0[4] = {0, 0, 0, 0}, phn1[4] = {0, 0, 0, 0}, wion0 = 0, wion1 = 0;
    unsigned long long sp_pre = 0, sp_call = 0, sp_glue = 0, sp_post = 0, ticks = 0, pmc_wall = 0;
    __Runtime_phase_cycles(ph0, phn0);
    __Runtime_wait_io_cycles(&wio0, &wion0);
    unsigned long long wm0 = 0, wc0 = 0, wp0[4] = {0, 0, 0, 0}, wx0 = 0;
    unsigned int wbd0 = 0, wrs0 = 0;
    __Runtime_launch_words(&wm0, &wc0, wp0, &wbd0, &wrs0, &wx0);
    XTime g0, g1;
    XTime_GetTime(&g0);
    const unsigned long long pmc0 = p0_pmc();
    int launches = 0;
    for (int rep = 0; rep < KACC_REPS; rep++) {
        memset(Yd, 0xff, ybytes); /* sentinel: an element the pass does not write is nonfinite */
        for (int s = 0; s < KACC_S; s++) {
            int32_t *Ws = Wd + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
            int32_t *Xs = Xd + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
            int32_t *Ys = Yd + (size_t)s * KACC_CHS * KACC_PXS;
            const size_t yb = (size_t)KACC_CHS * KACC_PXS * 4;
            XTime lt0, lt1;
            XTime_GetTime(&lt0);
            const unsigned long long a0 = p0_pmc();
            Xil_DCacheFlushRange((UINTPTR)Ws, (size_t)KACC_CHS * KACC_KP * 2);
            Xil_DCacheFlushRange((UINTPTR)Xs, (size_t)KACC_PXS * KACC_KP * 2);
            Xil_DCacheFlushRange((UINTPTR)Ys, yb);
            const unsigned long long r0 = p0_rt_cyc(), a1 = p0_pmc();
            @NAME@<<<mesh>>>(@ARGS@);
            const unsigned long long a2 = p0_pmc(), r1 = p0_rt_cyc();
            Xil_DCacheInvalidateRange((UINTPTR)Ys, yb);
            const unsigned long long a3 = p0_pmc();
            XTime_GetTime(&lt1);
            sp_pre += a1 - a0;
            sp_call += a2 - a1;
            sp_glue += (a2 - a1) > (r1 - r0) ? (a2 - a1) - (r1 - r0) : 0ULL;
            sp_post += a3 - a2;
            ticks += (unsigned long long)(lt1 - lt0);
            pmc_wall += a3 - a0;
            /* outside the timed regions: the counters armed at this launch (core done: wait_io returned) */
            __Runtime_core_perf_report(launches);
            launches++;
        }
    }
    const unsigned long long pmc1 = p0_pmc();
    XTime_GetTime(&g1);
    __Runtime_phase_cycles(ph1, phn1);
    __Runtime_wait_io_cycles(&wio1, &wion1);
    unsigned long long wm1 = 0, wc1 = 0, wp1[4] = {0, 0, 0, 0}, wx1 = 0;
    unsigned int wbd1 = 0, wrs1 = 0;
    __Runtime_launch_words(&wm1, &wc1, wp1, &wbd1, &wrs1, &wx1);
    unsigned long long ph[4];
    unsigned int phn[4];
    for (int i = 0; i < 4; i++) {
        ph[i] = ph1[i] - ph0[i];
        phn[i] = phn1[i] - phn0[i];
    }
    const unsigned long long wio = wio1 - wio0;
    const unsigned int wion = wion1 - wion0;
    const double g_s = (double)(g1 - g0) / (double)COUNTS_PER_SECOND;
    const int pmu_xtime = g_s > 0.0;
    const unsigned long long pmu_hz = pmu_xtime ? (unsigned long long)((double)(pmc1 - pmc0) / g_s + 0.5) : P0_A78_FALLBACK_HZ;
    printf("[P0] pmu_src=%s pmu_hz=%llu\n", pmu_xtime ? "xtime" : "fallback", pmu_hz);
    const double cms = pmu_hz ? 1000.0 / (double)pmu_hz : 0.0;
    const double nl = launches ? (double)launches : 1.0;
    const double aie_ms = pmu_xtime ? 1000.0 * (double)ticks / (double)COUNTS_PER_SECOND
                                    : 1000.0 * (double)pmc_wall / (double)pmu_hz;
    const unsigned long long c_kl = ph[0], c_bd = ph[1] + sp_glue, c_si = ph[3] + sp_pre, c_ce = ph[2],
                             c_wi = wio + sp_post;
    const double rt_ms = (double)(ph[0] + ph[1] + ph[2] + ph[3] + wio) * cms;
    const double sum_ms = (double)(c_kl + c_bd + c_si + c_ce + c_wi) * cms;
    printf("[BENCH] aie_wall_ms=%.3f launches=%d gmac=%.4f\n", aie_ms, launches, KACC_GMAC * KACC_REPS);
    printf("LAUNCH split kload=%.3f bdcfg=%.3f startio=%.3f coreen=%.3f wait_io=%.3f core_reset=%.3f "
           "ms_per_launch=%.3f pmu_hz=%llu sum_vs_aie_wall=%.3f\n",
           c_kl * cms / nl, c_bd * cms / nl, c_si * cms / nl, c_ce * cms / nl, c_wi * cms / nl, 0.0, aie_ms / nl,
           pmu_hz, aie_ms > 0.0 ? sum_ms / aie_ms : 0.0);
    printf("LAUNCH split_detail calls kload=%u bdcfg=%u startio=%u coreen=%u wait_io=%u core_reset=0 "
           "rt_bdcfg=%.4f rt_startio=%.4f rt_wait_io=%.4f "
           "host_glue_bdcfg=%.4f host_flush_startio=%.4f host_inval_wait_io=%.4f gemm_call=%.4f "
           "rt_phase_sum_ms=%.3f phase_sum_ms=%.3f aie_wall_ms=%.3f unattributed_ms_per_launch=%.4f "
           "pmc_delta=%llu xtime_s=%.3f launches=%d\n",
           phn[0], phn[1], phn[3], phn[2], wion, ph[1] * cms / nl, ph[3] * cms / nl, wio * cms / nl,
           sp_glue * cms / nl, sp_pre * cms / nl, sp_post * cms / nl, sp_call * cms / nl, rt_ms, sum_ms, aie_ms,
           (aie_ms - sum_ms) / nl, pmc1 - pmc0, g_s, launches);
#if defined(AIEHLC_PROFILING) && AIEHLC_PROFILING
    /* M1.1 launch words per launch over the timed pass (main_launch_line.c format; PH_KLOAD=0 PH_BDCFG=1
     * PH_COREEN=2 PH_STARTIO=3) */
    printf("LAUNCH split_words mmio_w=%.1f cp_w=%.1f bd_n=%.1f rst_n=%.1f kload_w=%.1f bdcfg_w=%.1f startio_w=%.1f coreen_w=%.1f\n",
           (double)(wm1 - wm0) / nl, (double)(wc1 - wc0) / nl, (double)(wbd1 - wbd0) / nl, (double)(wrs1 - wrs0) / nl,
           (double)(wp1[0] - wp0[0]) / nl, (double)(wp1[1] - wp0[1]) / nl, (double)(wp1[3] - wp0[3]) / nl,
           (double)(wp1[2] - wp0[2]) / nl);
    printf("[P0] split_words_miss=%llu\n", wx1 - wx0);
#endif
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
    fflush(stdout);
    return pass ? 0 : 1;
}
