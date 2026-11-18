/* aiednn/conv2d/kacc_chunk_board.cc -- board glue of kacc_chunk_conv.cc (A78 + aiehlc runtime): port buffers,
 * cache sync, the gemm_kacc dispatch of every (Np, Kk, MC) instance, and the host ABI the generated board
 * main / driver read (@K@_aie_stats, @K@_aie_launch_stats, @K@_aie_split_stats, @K@_aie_host_split_*).
 * Parameters: @K@ host ABI prefix, @WARM@ warm launch, @CTRLPKT@ control-packet aot, @PART_C0@ first mesh
 * column, @MAXW@ / @MAXX@ / @MAXY@ port sizes (elements), @DISPATCH@ the instance switch. */
#include <math.h>
#include "aie_user_kernel.h"
#include "xil_cache.h"
#include "aie_timer.h"

#define UC_WARM @WARM@
#if UC_WARM
extern "C" void __Runtime_pm_warm_stats(unsigned *skipped_segs, unsigned *cold_tile_loads);
#endif
#define UC_CTRLPKT @CTRLPKT@
#if UC_CTRLPKT
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h */
static int g_uc_ht = 0;
#endif
#define UC_KW_ELEMS @MAXW@ /* largest Np*Kp (bf16) */
#define UC_KX_ELEMS @MAXX@ /* largest MC*Kp (bf16) */
#define UC_KY_ELEMS @MAXY@ /* largest Np*MC (fp32) */

static inline uint16_t uc_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
/* the DLTensor the generated driver passes (tvm_graph_types.h) */
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
static long uc_numel(const uc_dlt *t) {
    long n = 1;
    for (int d = 0; d < t->ndim; d++)
        n *= (long)t->shape[d];
    return n;
}

static int32_t *g_kw = nullptr, *g_kx = nullptr, *g_ky = nullptr;
static int uc_kalloc(void) {
    if (g_ky)
        return 0;
    g_kw = (int32_t *)__Runtime_Alloc((size_t)UC_KW_ELEMS * 2);
    g_kx = (int32_t *)__Runtime_Alloc((size_t)UC_KX_ELEMS * 2);
    g_ky = (int32_t *)__Runtime_Alloc((size_t)UC_KY_ELEMS * 4);
    return (g_kw && g_kx && g_ky) ? 0 : -1;
}

static int g_aie_launches = 0;            /* every gemm_kacc launch that ran */
static unsigned long long g_uc_macs = 0;  /* conv MACs of those launches (valid pixels x N x K, dw: x KH*KW) */
static int g_uc_calls = 0;                /* @K@_aie_conv calls */
static int g_uc_rc = 0;                   /* first nonzero rc */
static unsigned long long g_uc_aie_ticks = 0; /* XTime ticks in flush + launch + invalidate */
static int g_uc_kloads = 0;
static unsigned g_uc_cold_tiles = 0;
static unsigned long long g_sp_pre = 0, g_sp_call = 0, g_sp_glue = 0, g_sp_post = 0;
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
extern "C" int @K@_aie_split_stats(unsigned long long *pre, unsigned long long *call, unsigned long long *glue,
                                   unsigned long long *post) {
    *pre = g_sp_pre;
    *call = g_sp_call;
    *glue = g_sp_glue;
    *post = g_sp_post;
    return 0;
}
extern "C" int @K@_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles) {
    if (aie_ms)
        *aie_ms = 1000.0 * (double)g_uc_aie_ticks / (double)COUNTS_PER_SECOND;
    if (kloads)
        *kloads = g_uc_kloads;
    if (cold_tiles)
        *cold_tiles = g_uc_cold_tiles;
    return 0;
}
extern "C" void @K@_aie_host_split_site(int site) { (void)site; }
extern "C" int @K@_aie_host_split_line(int site, char *buf, int n) {
    (void)site;
    (void)buf;
    (void)n;
    return 0;
}
extern "C" int @K@_aie_stats(int *launches, int *run_rc, int *calls, unsigned long long *macs) {
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

/* one launch of the gemm_kacc instance (N, K, M): ports W [N][Kp/2] int32, X [M][Kp/2], Y fp32 [N][M] */
static int uc_kacc_launch(int N, int K, int M, int32_t *kw, int32_t *kx, int32_t *ky) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({@PART_C0@, 6, 0, 6}, 4, 4); // aiegraph lint L015: the mesh VarDecl in the launching function
@DISPATCH@
    return -6;
}
/* the instance's K with its zero chunk (aiegraph.cnn.kacc_geom kp: one more chunk when W would be square) */
static inline int uc_kp(int N, int K, int M) {
    const int tkw = 4096 / M < 32 ? 4096 / M : 32;
    return N == K / 2 ? K + 2 * tkw : K;
}
/* sync + launch + stats; *first*: first chunk of a call (W written since the last launch) */
static int uc_launch(int N, int K, int M, int first, unsigned long long macs) {
    const int Kp = uc_kp(N, K, M);
    const size_t wb = (size_t)N * Kp * 2, xb = (size_t)M * Kp * 2, yb = (size_t)N * M * 4;
#if UC_CTRLPKT
    if (!g_uc_ht) {
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    memset(g_ky, 0xff, yb); /* sentinel: unwritten -> NaN -> counted by the scatter */
    XTime lt0, lt1;
    XTime_GetTime(&lt0);
    const unsigned long long sp0 = uc_pmc();
#if UC_WARM
    unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
    __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
    if (first)
        Xil_DCacheFlushRange((UINTPTR)g_kw, wb);
    Xil_DCacheFlushRange((UINTPTR)g_kx, xb);
    Xil_DCacheFlushRange((UINTPTR)g_ky, yb);
    const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
    const int lrc = uc_kacc_launch(N, K, M, g_kw, g_kx, g_ky);
    const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
    Xil_DCacheInvalidateRange((UINTPTR)g_ky, yb);
#if UC_WARM
    __Runtime_pm_warm_stats(&seg1, &cold1);
    if (cold1 != cold0) {
        g_uc_kloads++;
        g_uc_cold_tiles += cold1 - cold0;
    }
#endif
    const unsigned long long sp3 = uc_pmc();
    XTime_GetTime(&lt1);
    if (lrc != 0) {
        printf("[aie_conv] launch N=%d K=%d M=%d rc=%d (no instance)\n", N, K, M, lrc);
        return lrc;
    }
    g_sp_pre += sp1 - sp0;
    g_sp_call += sp2 - sp1;
    g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
    g_sp_post += sp3 - sp2;
    g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
    g_aie_launches++;
    g_uc_macs += macs;
    return 0;
}
