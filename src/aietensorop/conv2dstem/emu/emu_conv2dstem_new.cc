// emu_conv2dstem_new.cc -- run the GENERATED kernel of conv2dstem_new.cc on x86
// for every core of the 4x4 mesh, end to end with the library's REAL host code:
// stage_ifm() + pack_wts_tiles() (quant params in the pad channel, mmul order)
// feed windows served block by block exactly as the derived DMA descriptors
// (aiehlc_desc_win_*) say; output blocks are written TILE-MAJOR, as
// #pragma output_tile_major makes the shim DMA do; collect_ofm_hwc() de-tiles;
// the result is compared with the library's own scalar_conv2d_ref() over the
// same staged buffers (what conv2d_stem_verify() does on a board).
//
// Vector code runs on src/mlir/runtime/aie_api_x86emu.h (DMA windows 32-byte
// aligned, not 64, as on the core). Data: random int8 activations / weights and
// per-channel quant params inside pack_wts_tiles()'s accepted range.
//
//   bash src/aietensorop/conv2dstem/emu/emu_conv2dstem_new.sh <aout/worklocal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "aie_api_x86emu.h"

// ── runtime stubs the host code calls ──────────────────────────────────────
static void *emu_alloc(size_t n) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, n ? n : 1))
        abort();
    return p;
}
void *__Runtime_Alloc(size_t n) { return emu_alloc(n); }
void *__Runtime_alloc_uncached(size_t n) { return emu_alloc(n); }
void __Runtime_free_uncached(void *p) { free(p); }
void __Runtime_buffer_device_written(void *, size_t) {}
void __Runtime_buffer_forget(void *) {}

// ── the core's view: windows served by the derived descriptors ─────────────
static int g_blke = 0, g_row = 0, g_col = 0;
static void klog(const char *tag, int) { g_blke += !strcmp(tag, "BLKE"); }
static unsigned get_coreid() { return ((unsigned)(3 + g_col) << 16) | (unsigned)(3 + g_row); }
struct Win {
    int8_t *base;     // the staged tensor (input) / tile-major g_ofm (output)
    int dims[4];      // tensor extents (input)
    const void *desc; // aiehlc_blk_desc
    int round;
    bool tile_major; // output: core-sequential blocks
    int8_t *buf;     // the window, 32-byte but not 64-byte aligned
    void *raw;
};
using input_window_int8 = Win;
using output_window_int8 = Win;
static void serve(Win *w, bool write);
static void *acquire_input_window(Win *w) {
    serve(w, false);
    return w->buf;
}
static void release_input_window(Win *w) { w->round++; }
static void *acquire_output_window(Win *w) { return w->buf; }
static void release_output_window(Win *w) {
    serve(w, true);
    w->round++;
}
#define KERNEL_COMPILE
#include "conv2d_spatial.cc" // the generated kernel + its block descriptors

static void serve(Win *w, bool write) {
    const aiehlc_blk_desc &d = *(const aiehlc_blk_desc *)w->desc;
    const int per = d.rounds_i * d.rounds_j, r = w->round % per, i = r / d.rounds_j, j = r % d.rounds_j;
    int off[4] = {0, 0, 0, 0}, sh[4] = {1, 1, 1, 1}, dm[4] = {1, 1, 1, 1};
    long bytes = 1;
    for (int k = 0; k < d.rank; k++) {
        off[k] = d.base_p0[k] * g_row + d.base_p1[k] * g_col + d.step_i[k] * i + d.step_j[k] * j;
        sh[k] = d.shape[k];
        dm[k] = w->dims[k];
        bytes *= sh[k];
    }
    if (write && w->tile_major) { // [mesh row][mesh col][round][block]
        memcpy(w->base + ((long)(g_row * 4 + g_col) * per + w->round) * bytes, w->buf, bytes);
        return;
    }
    long b = 0;
    for (int a0 = 0; a0 < sh[0]; a0++)
        for (int a1 = 0; a1 < sh[1]; a1++)
            for (int a2 = 0; a2 < sh[2]; a2++)
                for (int a3 = 0; a3 < sh[3]; a3++, b++) {
                    long t = (((long)(off[0] + a0) * dm[1] + off[1] + a1) * dm[2] + off[2] + a2) * dm[3] + off[3] + a3;
                    if (write)
                        w->base[t] = w->buf[b];
                    else
                        w->buf[b] = w->base[t];
                }
}
static Win make_win(int8_t *base, int d0, int d1, int d2, int d3, const void *desc, long bytes, bool tm) {
    Win w{base, {d0, d1, d2, d3}, desc, 0, tm, nullptr, nullptr};
    if (posix_memalign(&w.raw, 64, bytes + 32))
        abort();
    w.buf = (int8_t *)w.raw + 32;
    return w;
}

// ── the library's host code, verbatim (extracted by the .sh) ───────────────
#include "conv2dstem.h"
constexpr int OC_PER_G = NUM_FILTERS / HW_COLS;
#include "host_part.inc"

static uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
// One emulated launch with this core-shift range; returns mismatches.
static long run_case(const char *name, int sh_lo, int sh_span) {
    static int8_t ifm[INPUT_H * INPUT_W * INPUT_C], wts[NUM_FILTERS * KERNEL_H * KERNEL_W * INPUT_C];
    static conv2dstem_qparam qp[NUM_FILTERS];
    for (int i = 0; i < (int)sizeof ifm; i++)
        ifm[i] = (int8_t)(mix(i + 1) & 0xFF);
    for (int i = 0; i < (int)sizeof wts; i++)
        wts[i] = (int8_t)((int)(mix(i + 77777) % 255u) - 127);
    for (int f = 0; f < NUM_FILTERS; f++) {
        qp[f].bias = (int32_t)(mix(f + 5) % 400000u) - 200000;
        qp[f].zero_point = (int32_t)(mix(f + 9) % 2000u) - 1000;
        qp[f].multiplier = (int32_t)(1073741824u + mix(f + 13) % 1000000000u); // [2^30, 2^31 - 2^16)
        qp[f].shift = sh_lo + (int32_t)(mix(f + 17) % (uint32_t)(sh_span + 1));
    }
    if (ensure_staging() != CONV2DSTEM_OK || pack_wts_tiles(wts, 0, qp) != CONV2DSTEM_OK) {
        printf("staging / packing failed\n");
        return -1;
    }
    stage_ifm(ifm);
    Win a = make_win(g_ifm_pad, INPUT_H_PAD, IFM_W_PITCH, INPUT_C_ALIGN, 1, &aiehlc_desc_win_a, 19 * 61 * 4, false);
    Win b = make_win(g_wts_bt, NUM_FILTERS, KERNEL_H, KERNEL_W, INPUT_C_ALIGN, &aiehlc_desc_win_b, 3136, false);
    Win c = make_win((int8_t *)g_ofm, 0, 0, 0, 0, &aiehlc_desc_win_c, 7 * 28 * 16, true);
    for (g_row = 0; g_row < HW_ROWS; g_row++)
        for (g_col = 0; g_col < HW_COLS; g_col++) {
            a.round = b.round = c.round = 0;
            conv2d_spatial(&a, &b, &c);
        }
    static uint8_t ofm[OFM_ELEMS], ref[OFM_ELEMS];
    collect_ofm_hwc(ofm);
    scalar_conv2d_ref(g_ifm_pad, g_wts_bt, ref);
    long bad = 0, zero = 0, top = 0;
    for (int i = 0; i < OFM_ELEMS; i++) {
        zero += ref[i] == 0;
        top += ref[i] == 255;
        if (ofm[i] != ref[i] && bad++ < 8)
            printf("  mismatch flat %d: got %d, want %d\n", i, ofm[i], ref[i]);
    }
    printf("%-28s %ld / %d match (ReLU 0: %ld, clamped 255: %ld, in between: %ld)\n", name, (long)OFM_ELEMS - bad,
           OFM_ELEMS, zero, top, (long)OFM_ELEMS - zero - top);
    return bad;
}

int main() {
    long bad = run_case("shifts 9..13 (mid-range):", 9, 4);
    bad += run_case("shifts 7..27 (wide span):", 7, 20);
    printf("BLKE=%d\n%s\n", g_blke, bad || g_blke ? "FAIL" : "PASS");
    return bad || g_blke;
}
