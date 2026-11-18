
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
@DLT@
typedef unsigned long long XTime;
static void XTime_GetTime(XTime *t) { *t = 0; }
#define UC_CTRLPKT 0
#define UC_KW_ELEMS (1 << 22)
#define UC_KX_ELEMS (1 << 22)
#define UC_KY_ELEMS (1 << 21)
static int32_t *g_kw = nullptr, *g_kx = nullptr, *g_ky = nullptr;
static int uc_kalloc(void) {
    if (g_ky)
        return 0;
    g_kw = (int32_t *)calloc(UC_KW_ELEMS / 2, 4);
    g_kx = (int32_t *)calloc(UC_KX_ELEMS / 2, 4);
    g_ky = (int32_t *)calloc(UC_KY_ELEMS, 4);
    return (g_kw && g_kx && g_ky) ? 0 : -1;
}
static unsigned long long g_uc_aie_ticks = 0, g_uc_macs = 0;
static int g_aie_launches = 0, g_uc_calls = 0, g_uc_rc = 0;
static void uc_sync_pre(const void *, size_t, const void *, size_t, const void *, size_t) {}
static void uc_sync_post(const void *, size_t) {}
typedef void (*launch_cb)(int, int, int, long, long, long);
static launch_cb g_cb = 0;
extern "C" void set_cb(launch_cb c) { g_cb = c; }
static int uc_kacc_launch(int N, int K, int M, int32_t *kw, int32_t *kx, int32_t *ky) {
    g_cb(N, K, M, (long)(kw - g_kw), (long)(kx - g_kx), (long)(ky - g_ky));
    return 0;
}
extern "C" void *bufs(int i) { uc_kalloc(); return i == 0 ? (void *)g_kw : i == 1 ? (void *)g_kx : (void *)g_ky; }
extern "C" int launches(void) { return g_aie_launches; }
@ATTN@
extern "C" int site(int cid, void *args, int qk, int swap, int CHP, int PXP, int Kp, int kp, int kch, int splitpx,
                    int S, int chs, int pxs, int tr) {
    return unet_aie_attn(cid, args, 3, qk, swap, CHP, PXP, Kp, kp, kch, splitpx, S, chs, pxs, tr);
}
