/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// ═══════════════════════════════════════════════════════════════════════════
// HOST -- after the kernel: nothing that reaches the launch is declared above it.
// ═══════════════════════════════════════════════════════════════════════════
#include "aie_user_kernel.h"
#include "xil_cache.h"
#include "aie_timer.h"

#define UC_WARM @WARM@ /* build_unet.py --launch-opt warm: kernel loaded once (PM warm skip), --skip-bss */
#if UC_WARM
/* aie_runtime.c (AIEHLC_PM_WARM_SKIP): per-tile PM segs skipped / per-tile cold loads since boot */
extern "C" void __Runtime_pm_warm_stats(unsigned *skipped_segs, unsigned *cold_tile_loads);
#endif

#define UC_CTRLPKT @CTRLPKT@ /* build_unet.py --ctrlpkt aot: control-packet plan pragmas at the top of this file */
#if UC_CTRLPKT
/* aie_runtime.c: required by control_plan_shim_bd_ctrl (aiehlc lint L023), before device init */
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h:936 */
static int g_uc_ht = 0;
#endif

#define UC_MAX_ACC @MAXACC@ /* largest M*N of the partition layers (fp32 K-sum on the APU) */
#define UC_KPORT 128        /* staged row: 64 bf16 + 64 bf16 zeros (int32 [64][64] port) */
/*KT+*/#define UC_BR 80            /* port b rows: 4 core groups x (16 weight rows + 4 zero rows) */
#define UC_CW 80            /* port c row: 4 core blocks x (16 results + 4 KTIMER tail words) */
#define UC_GRP(i) ((((i) >> 4) * 20) + ((i) & 15)) /* row/col i of the 64 -> padded core-group layout */
/*KT-*//*NK+*/#define UC_BR 64            /* port b rows */
#define UC_CW 64            /* port c row */
#define UC_GRP(i) (i)
/*NK-*/
static inline uint16_t uc_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float uc_bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
/* the DLTensor / TVMValue the generated driver passes (tvm_graph_types.h) */
typedef struct {
    void *data;
    int32_t device_type, device_id;
    int32_t ndim;
    uint8_t code, bits;
    uint16_t lanes;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} uc_dlt;

static float g_uc_acc[UC_MAX_ACC] __attribute__((aligned(64))); // .bss, never malloc
static int32_t *g_uc_a = nullptr; // port a: bf16 [64][128]
static int32_t *g_uc_b = nullptr; // port b: bf16 [UC_BR][128]
static int32_t *g_uc_c = nullptr; // port c: f32  [64][UC_CW]
static int g_aie_launches = 0;    // every gemm_bf16 launch that ran
static unsigned long long g_uc_macs = 0; // MACs of those launches (valid rows x cols x K)
static int g_uc_calls = 0;        // aiednn_aie_conv calls
static int g_uc_rc = 0;           // first nonzero rc
static unsigned long long g_uc_aie_ticks = 0; // A78 XTime ticks spent in flush + launch + invalidate
static int g_uc_kloads = 0;       // launches that (re)wrote the kernel's program memory
/* P1 split: A78 pmccntr per launch region inside aie_wall. pre = B/C flush (host->DMA sync, -> startio),
 * post = C invalidate + pm stats (DMA->host sync, -> wait_io), call = gemm_bf16; glue = call minus the
 * runtime phase + wait_io cycles it accrued (48 lock-init writes, 60 createio, dev sync, sb commit in the
 * generated host code -> bdcfg). */
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls); /* C++ linkage, aie_runtime.h */
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
static inline unsigned long long uc_pmc(void) {
    unsigned long long v;
#if defined(__aarch64__)
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v));
#else
    v = 0ULL;
#endif
    return v;
}
static inline unsigned long long uc_rt_cyc(void) {
    unsigned long long ph[4] = {0, 0, 0, 0}, w = 0;
    unsigned int pn[4] = {0, 0, 0, 0}, wn = 0;
    __Runtime_phase_cycles(ph, pn);
    __Runtime_wait_io_cycles(&w, &wn);
    return ph[0] + ph[1] + ph[2] + ph[3] + w;
}
static unsigned long long g_sp_pre = 0, g_sp_call = 0, g_sp_glue = 0, g_sp_post = 0;
/* GPOOL: per-layer [UCL] split, recorded in the site and printed after graph_run */
#define UC_UCL_MAX 128
typedef struct {
    int M, N, K, fast;
    float pre, aie, post;
} uc_ucl;
static uc_ucl g_uc_ucl[UC_UCL_MAX];
static int g_uc_nucl = 0, g_uc_ucl_done = 0;
extern "C" int aiednn_aie_split_stats(unsigned long long *pre, unsigned long long *call, unsigned long long *glue,
                                    unsigned long long *post) {
    if (!g_uc_ucl_done) {
        g_uc_ucl_done = 1;
        for (int i = 0; i < g_uc_nucl; i++)
            printf("[UCL] l=%d M=%d N=%d K=%d pre_us=%.0f aie_us=%.0f post_us=%.0f fast=%d\n", i, g_uc_ucl[i].M,
                   g_uc_ucl[i].N, g_uc_ucl[i].K, (double)g_uc_ucl[i].pre, (double)g_uc_ucl[i].aie,
                   (double)g_uc_ucl[i].post, g_uc_ucl[i].fast);
    }
    *pre = g_sp_pre;
    *call = g_sp_call;
    *glue = g_sp_glue;
    *post = g_sp_post;
    return 0;
}
static unsigned g_uc_cold_tiles = 0; // per-tile cold loads inside those launches

extern "C" int aiednn_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles) {
    if (aie_ms)
        *aie_ms = 1000.0 * (double)g_uc_aie_ticks / (double)COUNTS_PER_SECOND;
    if (kloads)
        *kloads = g_uc_kloads;
    if (cold_tiles)
        *cold_tiles = g_uc_cold_tiles;
    return 0;
}
/* GAP-RT1: per-site aie_host split (pack / scatter / dma / wait) -> PROF aie_host_split (aie_host_split.h).
 * The site index comes from graph_driver's UP_AIE_BEGIN; wait = the runtime's always-on wait_io ticks. */
#ifndef UC_HS_DEFINED
#define UC_HS_DEFINED 1
#include "aie_host_split.h"
void __Runtime_host_split_wait_ticks(unsigned long long *ticks, unsigned int *calls); /* C++ linkage, aie_runtime.h */
static ahs_table g_ahs;
static inline unsigned long long uc_hs_wait(void) {
    unsigned long long t = 0ULL;
    unsigned int n = 0U;
    __Runtime_host_split_wait_ticks(&t, &n);
    return t;
}
#endif
extern "C" void aiednn_aie_host_split_site(int site) { ahs_set_site(&g_ahs, site); }
extern "C" int aiednn_aie_host_split_line(int site, char *buf, int n) {
    return ahs_format(&g_ahs, site, (double)COUNTS_PER_SECOND, buf, (size_t)n);
}

/*KT+*/static unsigned long long g_kt_max = 0, g_kt_all = 0; // KTIMER: sum over launches of max-core / all-core cycles
static int g_kt_n = 0, g_kt_bad = 0;                    // launches read, core tails without the magic word

extern "C" int aiednn_aie_ktimer_stats(unsigned long long *max_sum, unsigned long long *all_sum, int *n, int *bad) {
    if (max_sum)
        *max_sum = g_kt_max;
    if (all_sum)
        *all_sum = g_kt_all;
    if (n)
        *n = g_kt_n;
    if (bad)
        *bad = g_kt_bad;
    return 0;
}

/*KT-*/extern "C" int aiednn_aie_stats(int *launches, int *run_rc, int *calls, unsigned long long *macs) {
    if (launches)
        *launches = g_aie_launches;
    if (run_rc)
        *run_rc = g_uc_calls ? g_uc_rc : -99;
    if (calls)
        *calls = g_uc_calls;
    if (macs)
        *macs = g_uc_macs;
    return 0;
}

static int uc_alloc(void) {
    if (g_uc_c)
        return 0;
    g_uc_a = (int32_t *)__Runtime_Alloc(64 * UC_KPORT * 2);
    g_uc_b = (int32_t *)__Runtime_Alloc(UC_BR * UC_KPORT * 2);
    g_uc_c = (int32_t *)__Runtime_Alloc(64 * UC_CW * 4);
    if (!g_uc_a || !g_uc_b || !g_uc_c)
        return -1;
    memset(g_uc_a, 0, 64 * UC_KPORT * 2); // cols 64..127 stay zero
    memset(g_uc_b, 0, UC_BR * UC_KPORT * 2); // --ktimer: rows 16..19 of each core group stay zero
    return 0;
}

static long uc_numel(const uc_dlt *t) {
    long n = 1;
    for (int d = 0; d < t->ndim; d++)
        n *= (long)t->shape[d];
    return n;
}

static int uc_conv(void *const *argv, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw) {
    if (nargs < 4)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 5 || w->ndim != 6 || y->ndim != 5 || x->shape[0] != 1 || y->shape[0] != 1)
        return -2;
    const int cb = (int)x->shape[4], C = (int)x->shape[1] * cb, H = (int)x->shape[2], W = (int)x->shape[3];
    const int ib = (int)w->shape[4], ob = (int)w->shape[5], CI = (int)w->shape[1] * ib;
    const int O = (int)w->shape[0] * ob, KH = (int)w->shape[2], KW = (int)w->shape[3];
    const int yb = (int)y->shape[4], OH = (int)y->shape[2], OW = (int)y->shape[3];
    if (CI != C || (int)y->shape[1] * yb != O || OH != (H + 2 * ph - KH) / sh + 1 || OW != (W + 2 * pw - KW) / sw + 1)
        return -3;
    const int M = OH * OW, N = O, K = C * KH * KW, KK = KH * KW;
    if ((long)M * N > UC_MAX_ACC)
        return -3;
    const float *xin = (const float *)x->data;
    float *yout = (float *)y->data;
    if (uc_alloc() != 0)
        return -1;
    memset(g_uc_acc, 0, (size_t)M * N * 4);
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({@PART_C0@, 6, 0, 6}, 4, 4); // one mesh per call; ctrlpkt aot: col 2 = spare control column west (control_plan_dedicated_shim, as conv2dstem.cc)
    uint16_t *A = (uint16_t *)g_uc_a, *B = (uint16_t *)g_uc_b;
    const float *Cb = (const float *)g_uc_c;
    int bad = 0;
    for (int mb = 0; mb * 64 < M; mb++)
        for (int kc = 0; kc * 64 < K; kc++) {
            const int kv = (K - kc * 64) < 64 ? (K - kc * 64) : 64;
            for (int r = 0; r < 64; r++) { // A tile: im2col of the activation, fp32 -> bf16 RNE
                const int m = mb * 64 + r, oh = m / OW, ow = m % OW;
                for (int kk = 0; kk < 64; kk++) {
                    float v = 0.0f;
                    const int k = kc * 64 + kk;
                    if (m < M && k < K) {
                        const int c = k / KK, t = k % KK, ih = oh * sh + t / KW - ph, iw = ow * sw + t % KW - pw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            v = xin[(((size_t)(c / cb) * H + ih) * W + iw) * cb + c % cb];
                    }
                    A[r * UC_KPORT + kk] = uc_f2bf(v);
                }
            }
            Xil_DCacheFlushRange((UINTPTR)g_uc_a, 64 * UC_KPORT * 2);
            for (int nb = 0; nb * 64 < N; nb++) {
                for (int r = 0; r < 64; r++) { // B tile: straight from the compact bf16 blob (OIHW{ib}i{ob}o)
                    const int n = nb * 64 + r;
                    for (int kk = 0; kk < 64; kk++) {
                        const int k = kc * 64 + kk;
                        uint16_t v = 0;
                        if (n < N && k < K) {
                            const int c = k / KK, t = k % KK;
                            v = wbf[((((size_t)(n / ob) * (C / ib) + c / ib) * KK + t) * ib + c % ib) * ob + n % ob];
                        }
                        B[UC_GRP(r) * UC_KPORT + kk] = v;
                    }
                }
                memset(g_uc_c, 0xff, 64 * UC_CW * 4); // sentinel: unwritten -> NaN -> counted
                XTime lt0, lt1;
                XTime_GetTime(&lt0);
                const unsigned long long sp0 = uc_pmc();
#if UC_WARM
                unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
                __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
                Xil_DCacheFlushRange((UINTPTR)g_uc_b, UC_BR * UC_KPORT * 2);
                Xil_DCacheFlushRange((UINTPTR)g_uc_c, 64 * UC_CW * 4);
                const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
                gemm_bf16<<<mesh>>>(g_uc_a, g_uc_b, g_uc_c);
                const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
                Xil_DCacheInvalidateRange((UINTPTR)g_uc_c, 64 * UC_CW * 4);
#if UC_WARM
                __Runtime_pm_warm_stats(&seg1, &cold1);
                if (cold1 != cold0) {
                    g_uc_kloads++;
                    g_uc_cold_tiles += cold1 - cold0;
                }
#endif
                const unsigned long long sp3 = uc_pmc();
                XTime_GetTime(&lt1);
                g_sp_pre += sp1 - sp0;
                g_sp_call += sp2 - sp1;
                g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
                g_sp_post += sp3 - sp2;
                g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
                g_aie_launches++;
/*KT+*/                { // KTIMER: AIE core timer delta each core wrote into its block tail (row 0, words 16..18)
                    const uint32_t *Cw = (const uint32_t *)g_uc_c;
                    unsigned long long mx = 0;
                    for (int q = 0; q < 16; q++) {
                        const uint32_t *kt = Cw + (size_t)(q >> 2) * 16 * UC_CW + (q & 3) * 20 + 16;
                        if (kt[1] != 0x4B54494Du) {
                            g_kt_bad++;
                            continue;
                        }
                        const unsigned long long d = (unsigned long long)kt[0] | ((unsigned long long)kt[2] << 32);
                        g_kt_all += d;
                        if (d > mx)
                            mx = d;
                    }
                    g_kt_max += mx;
                    g_kt_n++;
                }
/*KT-*/                const int mv = (M - mb * 64) < 64 ? (M - mb * 64) : 64, nv = (N - nb * 64) < 64 ? (N - nb * 64) : 64;
                g_uc_macs += (unsigned long long)mv * nv * kv;
                for (int r = 0; r < mv; r++)
                    for (int j = 0; j < nv; j++) {
                        const float v = Cb[r * UC_CW + UC_GRP(j)];
                        if (!(v == v) || v > 1e30f || v < -1e30f)
                            bad++;
                        g_uc_acc[(size_t)(mb * 64 + r) * N + nb * 64 + j] += uc_bf2f(uc_f2bf(v)); // bf16 slice out
                    }
            }
        }
    if (bad) {
        printf("[aiednn_aie_conv] %d nonfinite AIE outputs\n", bad);
        return -4;
    }
    for (int n = 0; n < N; n++)
        for (int m = 0; m < M; m++)
            yout[((size_t)(n / yb) * M + m) * yb + n % yb] = g_uc_acc[(size_t)m * N + n];
