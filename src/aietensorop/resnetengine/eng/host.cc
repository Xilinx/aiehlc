/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Persistent conv engine host driver: every conv of int8 ResNet-18 (stem +
 * maxpool and 16 engine layers = 20 TVM convs) on the AIE.
 * Per layer it re-arms shim BDs: weights (+header) per mesh column, input rows
 * per mesh row, output packets per core. Each layer's output is checked
 * byte-for-byte against TVM (gen_engine.py's golden, itself proven equal to
 * the TRACE=1 dumps).
 *
 * Two modes, chosen at run time (engine_set_mode):
 *   persistent  the kernel is loaded once and runs every layer, one header
 *               per layer arriving on the weight stream;
 *   eager       the kernel ELF is reloaded (all 16 cores, program memory
 *               included, over the control-packet plane) before EVERY layer.
 * Both run the same kernel and are checked the same way.
 ******************************************************************************/
#include "aie_runtime.h"
#include "aie_timer.h"
#include "engine_fabric.h"
#include "eng_desc.h"
#include "eng_net.h"
#include "imagenet_labels.h" /* deploy_flow's ImageNet class names */
#include <stdio.h>
#include <arm_neon.h>
#include <math.h>
#include <string.h>

// Kernel variants (eng/kernel_variants): the full image-major engine for
// persistent mode, and per-layer-kind ELFs eager mode loads layer by layer.
extern unsigned char _binary_kernel_eng_start[], _binary_kernel_engstem_start[], _binary_kernel_engconv_start[];
enum { KV_FULL, KV_STEM, KV_CONV, KV_N };
static unsigned char *const k_kv_elf[KV_N] = {_binary_kernel_eng_start, _binary_kernel_engstem_start,
                                              _binary_kernel_engconv_start};
static const char *const k_kv_name[KV_N] = {"eng", "engstem", "engconv"};

#define INCBIN(sym, path)                                                                                              \
    __asm__(".section .rodata\n.balign 64\n.global " #sym "\n" #sym ":\n.incbin \"" path "\"\n.previous\n")
INCBIN(eng_blob_src, ENG_BLOB_PATH);
INCBIN(eng_gold_src, ENG_GOLD_PATH);
INCBIN(eng_img_src, ENG_IMG_PATH);
INCBIN(eng_image_src, ENG_IMAGE_PATH);
INCBIN(eng_logits_src, ENG_LOGITS_PATH);
INCBIN(eng_dense_src, ENG_DENSE_PATH);
extern const int8_t eng_blob_src[], eng_gold_src[], eng_img_src[], eng_dense_src[];
extern const float eng_image_src[], eng_logits_src[];

static const int k_ooo[EF_MESH_ROWS][EF_MESH_COLS] = EF_OOO_BD_TABLE;
static const int k_pkt[EF_MESH_ROWS][EF_MESH_COLS] = EF_PKT_ID_TABLE;
static const int k_out_col[EF_MESH_ROWS] = EF_OUT_SHIM_COL_TABLE;
static const int k_out_ch[EF_MESH_ROWS] = EF_OUT_SHIM_CH_TABLE;

static XAie_LocType shim(int rel_col) { return XAie_TileLoc((uint8_t)rel_col, 0); }

static long l8_bytes(int h, int blk, int wst) { return (long)(h + 2) * blk * wst * 8; }

/* Schedules. image-major: one image at a time through all layers; a block's
 * residual stays in core memory (GRAB/DOWN -> COMB). layer-major: every image
 * of the batch through layer l before layer l+1 -- the only order in which one
 * weight fetch can serve the whole batch, and the only one an eager kernel
 * reload per layer can afford -- so the residual goes through DDR: COMB layers
 * read it on the input channel (ENG_F_RESX), DOWN layers emit v_d next to y
 * (ENG_F_DOWNOUT). The header flags in the blob are patched to match. */
static int lm_flags(const eng_layer_t *L) {
#ifdef ENG_EXP_SHARED
    int f = L->flags & ~ENG_F_GRAB;
#else
    int f = (L->flags & ~ENG_F_GRAB) | ENG_F_GO;
#endif
    if (L->flags & ENG_F_COMB)
        f |= ENG_F_RESX;
    if (L->flags & ENG_F_DOWN)
        f |= ENG_F_DOWNOUT;
    return f;
}

// Output: one OOO BD per core; packet = [p][nb][Wo][8] of one output row
// (np passes: P, or 2P when v_d rides along; a sub-layer starts at pass p0).
// outs[r]: mesh row r's output tensor -- one image's for a row=image layer,
// whose mesh rows each write all rows of their own image.
static void arm_output_rows(XAie_DevInst *dev, const eng_layer_t *L, int8_t *const *outs, int np, struct_ioevent *ev) {
    const long row_bytes = (long)L->out_blk * L->wst_out * 8;
    int r0 = 0;
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        XAie_LocType t = shim(k_out_col[r]);
        XAie_DmaDesc d = {};
        for (int c = 0; c < EF_MESH_COLS; c++) {
            int8_t *dst = outs[r] + (r0 + 1) * row_bytes + (long)(2 * c + 8 * L->p0) * L->wst_out * 8 + 8;
            d = __Runtime_dma_bd_config_multidim_ooo(dev, t, dst, k_ooo[r][c], np * 2 * L->Wo * 8, -1, 0, k_pkt[r][c],
                                                     -1, 0, -1, 0, -1, 3, 4, L->Wo * 2, L->wst_out * 8, 2,
                                                     8 * L->wst_out * 8, np, (int)row_bytes, L->rows[r]);
        }
        __Runtime_dma_channel_enable_ooo(dev, t, k_out_ch[r], DMA_S2MM);
        ev[r] = __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, k_out_ch[r], k_ooo[r][0], DMA_S2MM), k_ooo[r][0],
                                  EF_MESH_COLS * L->rows[r]);
        if (!L->ri)
            r0 += L->rows[r];
    }
}

static void arm_output(XAie_DevInst *dev, const eng_layer_t *L, int8_t *out, int np, struct_ioevent *ev) {
    int8_t *const outs[EF_MESH_ROWS] = {out, out, out, out};
    arm_output_rows(dev, L, outs, np, ev);
}

// Weights: header + P*Q chunks, one contiguous run per mesh column.
static void arm_weights(XAie_DevInst *dev, const eng_layer_t *L, const int8_t *blob) {
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType t = shim(EF_FIRST_COL + c);
        // Contiguous, but expressed as 3 dims: a plain BD this long (> 1023
        // words) takes the runtime's fallback path, which leaves the AXI burst
        // at its default and streams ~3x slower than burst 16.
        XAie_DmaDesc d =
            __Runtime_dma_bd_config_multidim_ooo(dev, t, (void *)(blob + L->blob_off[c]), 0, (int)L->blob_len, -1, 0, 0,
                                                 0, 0, 0, 0, -1, 3, 4, 64, 256, 512, 256 * 512, 0, 0, 0);
        __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, 0, 0, DMA_MM2S), 0, 1);
    }
}

// Stem input: per mesh row, image rows 56r-2 .. (mesh row 0 has no halo row
// above it, so it starts at row 0 with two rows fewer).
static void arm_input_stem(XAie_DevInst *dev, const int8_t *img) {
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        const int r0 = r == 0 ? 0 : 56 * r - 2, rows_in = r == 0 ? 61 : 63;
        XAie_LocType t = shim(EF_FIRST_COL + r);
        XAie_DmaDesc d = __Runtime_dma_bd_config_multidim_ooo(
            dev, t, (void *)(img + (long)r0 * ENG_STEM_IN_ROW), 1, rows_in * ENG_STEM_IN_ROW, -1, 0, 0, 0, 0, 0, 0, -1,
            2, 4, ENG_STEM_IN_ROW / 4, ENG_STEM_IN_ROW, rows_in, 0, 0, 0, 0);
        __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, 1, 1, DMA_MM2S), 1, 1);
    }
}

// Rows [row0, row0 + rows) of 64-channel chunk q = 0..nchunk-1 of an L8 tensor
// (blk blocks per row), into mesh row r's input channel: one BD use per chunk,
// repeat uses in total (the iteration walks the chunks).
static void arm_rows(XAie_DevInst *dev, int r, int bd, const int8_t *t8, int blk, int wst, int row0, int rows,
                     int nchunk, int repeat) {
    const long row_bytes = (long)blk * wst * 8;
    const int chunk = ENG_CHUNK_BLK * wst * 8;
    XAie_LocType t = shim(EF_FIRST_COL + r);
    // (the _ooo variant is the one with an iteration dimension; ooo = -1)
    XAie_DmaDesc d = __Runtime_dma_bd_config_multidim_ooo(dev, t, (void *)(t8 + (long)row0 * row_bytes), bd,
                                                          chunk * rows, -1, 0, 0, 0, 0, 0, 0, -1, 3, 4, wst * 2,
                                                          wst * 8, ENG_CHUNK_BLK, (int)row_bytes, rows, chunk, nchunk);
    __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, 1, bd, DMA_MM2S), bd, repeat);
}

// Input: per mesh row, its rows of one 64-channel chunk; iteration walks the
// Q chunks, P*Q repeats re-stream them once per pass (resident: once).
/* Shim BD banks for the input channel (ch1): image b of a layer-major batch
 * uses bank b & 1, so image b+1's input can be queued while b streams. */
static const int k_in_bd[2] = {1, 13}, k_res_bd[2] = {12, 14};

static void arm_input_rows(XAie_DevInst *dev, const eng_layer_t *L, const int8_t *const *ins, int bank);
static void arm_input(XAie_DevInst *dev, const eng_layer_t *L, const int8_t *in) {
    const int8_t *const ins[EF_MESH_ROWS] = {in, in, in, in};
    arm_input_rows(dev, L, ins, 0);
}

// ins[r]: mesh row r's input tensor (row=image: its own image, from row 0, each
// chunk once; spatial: its rows of the shared image, once per pass unless resident).
static void arm_input_rows(XAie_DevInst *dev, const eng_layer_t *L, const int8_t *const *ins, int bank) {
    int r0 = 0;
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        const int rows_in = (L->rows[r] - 1) * L->S + 3;
        arm_rows(dev, r, k_in_bd[bank], ins[r], L->in_blk, L->wst_in, r0 * L->S, rows_in, L->Q,
                 L->ri || L->resident ? L->Q : L->P * L->Q);
        if (!L->ri)
            r0 += L->rows[r];
    }
}

// ENG_F_RESX: the residual's centre rows (the mesh row's output rows, padded
// coordinates r0+1..), the layer's P chunks from pass p0, queued on the input
// channel AHEAD of the input. Same geometry as the conv's input (identity: the
// block input x; downsample: v_d, the second half of conv1's 2C-channel output).
static void arm_residual(XAie_DevInst *dev, const eng_layer_t *L, const int8_t *const *res, int bank) {
    const int blk = eng_layers[L->res_act > 0 ? L->res_act - 1 : 0].out_blk;
    int r0 = 0;
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        const int8_t *base = res[r] + (long)(L->res_blk0 + 8 * L->p0) * L->wst_in * 8;
        arm_rows(dev, r, k_res_bd[bank], base, blk, L->wst_in, r0 + 1, L->rows[r], L->P, L->P);
        if (!L->ri)
            r0 += L->rows[r];
    }
}

/* ---- memtile weight cache (layer-major schedule) ---------------------------
 * A mesh column's weight stream (header + P*Q chunks) is loaded from DDR into
 * the row-2 memtiles ONCE per layer and batch, then re-streamed from there to
 * the column for every image (~7.2 GB/s per column, on chip) -- DDR, ~11.5
 * GB/s for the whole array, carries the weights once per batch instead of once
 * per image. Routing: at memtile (c, 2) the column's circuit SOUTH 0 -> NORTH 0
 * becomes SOUTH 0 -> DMA S2MM 0 (load) and DMA MM2S 0 -> NORTH 0 (stream).
 * A stream <= 512 KB sits in the column's own memtile (DMA address 0x80000).
 * Longer ones (the 512-channel layers, 608 KB) are laid end to end over the
 * five row-2 memtiles of the partition (rel cols 0..4, 2.5 MB), each within
 * its own DMA's reach (west neighbour 0x0, own 0x80000, east 0x100000). */
#define MT_OWN 0x80000u
#define MT_KB 1024u
#define MT_BD_MAX (256u * MT_KB) /* per BD; a longer stream chains BDs (BD length < 512 KB) */
#define MT_S2MM_BD 0             /* even channels share BDs 0..23 */
#define MT_MM2S_BD 8

/* Two banks so layer l+1 loads while layer l streams: bank 0 = the row-2
 * memtiles, bank 1 = row 1. A memtile switch connects SOUTH n only to NORTH n,
 * so loads take lane 2 and streams lane 0. Paths per column c:
 *   load bank 0  shim SOUTH3(ch0)->NORTH2 -> (c,1) SOUTH2->NORTH2 -> (c,2) SOUTH2->DMA S2MM 0
 *   load bank 1  shim SOUTH3(ch0)->NORTH2 -> (c,1) SOUTH2->DMA S2MM 0
 *   stream 0     (c,2) DMA MM2S 0 -> NORTH0 -> cores
 *   stream 1     (c,1) DMA MM2S 0 -> NORTH0 -> (c,2) SOUTH0->NORTH0 -> cores
 * (c,1)'s SOUTH2 and (c,2)'s NORTH0 are re-pointed per load / per layer. The
 * direct (uncached) route is the fabric's: shim SOUTH3->NORTH0, SOUTH0->NORTH0 up. */
static int mt_row(int bank) { return bank ? 1 : 2; }
static XAie_LocType mt_tile(int c, int bank) {
    return XAie_TileLoc((uint8_t)(EF_FIRST_COL + c), (uint8_t)mt_row(bank));
}

static void mt_route(XAie_DevInst *dev, int cached) {
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType sh = XAie_TileLoc((uint8_t)(EF_FIRST_COL + c), 0), m1 = mt_tile(c, 1), m2 = mt_tile(c, 0);
        if (cached) {
            XAie_StrmConnCctDisable(dev, sh, SOUTH, 3, NORTH, 0);
            XAie_StrmConnCctDisable(dev, m1, SOUTH, 0, NORTH, 0);
            XAie_StrmConnCctDisable(dev, m2, SOUTH, 0, NORTH, 0);
            XAie_StrmConnCctEnable(dev, sh, SOUTH, 3, NORTH, 2);
            XAie_StrmConnCctEnable(dev, m1, SOUTH, 2, NORTH, 2); // load bank 0 (mt_route_load)
            XAie_StrmConnCctEnable(dev, m2, SOUTH, 2, DMA, 0);
            XAie_StrmConnCctEnable(dev, m1, DMA, 0, NORTH, 0);
            XAie_StrmConnCctEnable(dev, m2, DMA, 0, NORTH, 0); // stream bank 0 (mt_route_stream)
        } else {
            XAie_StrmConnCctDisable(dev, sh, SOUTH, 3, NORTH, 2);
            XAie_StrmConnCctDisable(dev, m1, SOUTH, 2, NORTH, 2);
            XAie_StrmConnCctDisable(dev, m1, SOUTH, 2, DMA, 0);
            XAie_StrmConnCctDisable(dev, m2, SOUTH, 2, DMA, 0);
            XAie_StrmConnCctDisable(dev, m1, DMA, 0, NORTH, 0);
            XAie_StrmConnCctDisable(dev, m2, DMA, 0, NORTH, 0);
            XAie_StrmConnCctEnable(dev, sh, SOUTH, 3, NORTH, 0);
            XAie_StrmConnCctEnable(dev, m1, SOUTH, 0, NORTH, 0);
            XAie_StrmConnCctEnable(dev, m2, SOUTH, 0, NORTH, 0);
        }
    }
}

// (c,1)'s SOUTH2 -> the bank being loaded.
static void mt_route_load(XAie_DevInst *dev, int bank) {
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType m1 = mt_tile(c, 1);
        XAie_StrmConnCctDisable(dev, m1, SOUTH, 2, bank ? NORTH : DMA, bank ? 2 : 0);
        XAie_StrmConnCctEnable(dev, m1, SOUTH, 2, bank ? DMA : NORTH, bank ? 0 : 2);
    }
}

// (c,2)'s NORTH0 (the column broadcast) <- the bank being streamed.
static void mt_route_stream(XAie_DevInst *dev, int bank) {
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType m2 = mt_tile(c, 0);
        if (bank) {
            XAie_StrmConnCctDisable(dev, m2, DMA, 0, NORTH, 0);
            XAie_StrmConnCctEnable(dev, m2, SOUTH, 0, NORTH, 0);
        } else {
            XAie_StrmConnCctDisable(dev, m2, SOUTH, 0, NORTH, 0);
            XAie_StrmConnCctEnable(dev, m2, DMA, 0, NORTH, 0);
        }
    }
}

// DMA address (in mesh column c's memtile) of the column's copy of a stream.
static uint32_t mt_addr(int c, long len) {
    if (len <= 512 * (long)MT_KB)
        return MT_OWN;
    const long lin = 5 * 512 * (long)MT_KB - (EF_MESH_COLS - c) * len; // end-to-end from the top of rel col 4
    return (uint32_t)(lin - (long)c * 512 * MT_KB);                    // DMA 0x0 = west neighbour (rel col c)
}

// len bytes at DMA address addr as BDs bd0, bd0+1, .. (chained), MT_BD_MAX each.
static void mt_bds(XAie_DevInst *dev, XAie_LocType m, int bd0, uint32_t addr, long len) {
    const int n = (int)((len + MT_BD_MAX - 1) / MT_BD_MAX);
    for (int i = 0; i < n; i++) {
        const long part = len - (long)i * MT_BD_MAX < (long)MT_BD_MAX ? len - (long)i * MT_BD_MAX : (long)MT_BD_MAX;
        XAie_DmaDesc d;
        XAie_DmaDescInit(dev, &d, m);
        XAie_DmaSetAddrLen(&d, addr + (uint32_t)i * MT_BD_MAX, (u32)part);
        if (i + 1 < n)
            XAie_DmaSetNextBd(&d, (u16)(bd0 + i + 1), XAIE_ENABLE);
        XAie_DmaEnableBd(&d);
        XAie_DmaWriteBd(dev, &d, m, (u16)(bd0 + i));
    }
}

static void mt_wait(XAie_DevInst *dev, XAie_LocType m, XAie_DmaDirection dir) {
    u8 n = 1;
    for (long i = 0; n && i < 50000000L; i++)
        XAie_DmaGetPendingBdCount(dev, m, 0, dir, &n);
    if (n)
        printf("ENG memtile (%u,%u) dir %d TIMEOUT\n", (unsigned)m.Col, (unsigned)m.Row, (int)dir);
}

static void arm_weights(XAie_DevInst *dev, const eng_layer_t *L, const int8_t *blob);

// Start loading layer L's four column streams DDR -> memtile bank.
static void mt_load_start(XAie_DevInst *dev, __Runtime_CtrlRowFabric *fab, const eng_layer_t *L, const int8_t *blob,
                          int bank) {
    mt_route_load(dev, bank);
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType m = mt_tile(c, bank);
        mt_bds(dev, m, MT_S2MM_BD, mt_addr(c, L->blob_len), L->blob_len);
        XAie_DmaChannelSetStartQueue(dev, m, 0, DMA_S2MM, MT_S2MM_BD, 1, 0);
        XAie_DmaChannelEnable(dev, m, 0, DMA_S2MM);
    }
    __Runtime_ctrl_shim_bd_begin(fab);
    arm_weights(dev, L, blob);
    __Runtime_ctrl_shim_bd_commit();
}

static void mt_load_wait(XAie_DevInst *dev, int bank) {
    for (int c = 0; c < EF_MESH_COLS; c++)
        mt_wait(dev, mt_tile(c, bank), DMA_S2MM);
}

// Re-stream the cached layer from its bank to every column, once per image.
static void mt_stream(XAie_DevInst *dev, const eng_layer_t *L, int bank, int images) {
    mt_route_stream(dev, bank);
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType m = mt_tile(c, bank);
        mt_bds(dev, m, MT_MM2S_BD, mt_addr(c, L->blob_len), L->blob_len);
        XAie_DmaChannelSetStartQueue(dev, m, 0, DMA_MM2S, MT_MM2S_BD, (u32)images, 0);
        XAie_DmaChannelEnable(dev, m, 0, DMA_MM2S);
    }
}

void __Runtime_ctrl_high_throughput_enable(int on);
void __Runtime_ctrl_pmap_enable(int on);

static double us_since(XTime t0) {
    XTime t1;
    XTime_GetTime(&t1);
    return 1e6 * (double)(t1 - t0) / COUNTS_PER_SECOND;
}

void *__Runtime_ctrl_kernel_pkt_build(__Runtime_CtrlRowFabric *fab, const unsigned char *elf, unsigned int *bytes);
void __Runtime_ctrl_elf_warm_enable(int on);

/* Control-shim BD and MM2S channel the ctrl-plane kernel load streams on (the
 * same pair aiehlc's generated hosts use on this fabric). */
#define ENG_KLOAD_BD 2
#define ENG_KLOAD_CH 0
#define ENG_MAX_BATCH 32

typedef struct {
    XAie_DevInst *dev;
    __Runtime_CtrlRowFabric *fab;
    XAie_LocType tiles[EF_MESH_ROWS * EF_MESH_COLS];
    void *kpkt[KV_N]; /* each variant's control-packet image, encoded once */
    unsigned int kpkt_bytes[KV_N];
    int eager;            /* runtime: reload the kernel before every layer */
    int layer_major;      /* runtime: schedule (see lm_flags) */
    const eng_layer_t *T; /* the schedule's layer table: eng_layers or eng_lm_layers */
    int NT;
    int mt_loaded; /* layer whose weights are loaded/loading into memtile bank mt_bank, or -1 */
    int mt_bank;
    double kload_us; /* kernel (re)load time accumulated since the last reset */
    int8_t *blob;
    int8_t *act[ENG_MAX_BATCH][ENG_NLAYERS + 1];
    long act_bytes[ENG_NLAYERS + 1];
} Engine;

static void engine_go(Engine *e, int value);

// Kernel ELF -> all 16 cores over the control-packet plane (broadcast writes:
// core + core-DMA resets, data segments, program memory), then core enable.
static void engine_load_kernel(Engine *e, int kv) {
    XTime t0;
    XTime_GetTime(&t0);
    __Runtime_set_kernel_elf(k_kv_elf[kv]);
    __Runtime_ctrl_kernel_pkt_set(e->kpkt[kv], e->kpkt_bytes[kv]);
    const XAie_LocType *t = e->tiles;
    struct_kernel_group kg =
        __Runtime_load_kernel_group_16t_ctrl(e->dev, e->fab, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9],
                                             t[10], t[11], t[12], t[13], t[14], t[15], 16, ENG_KLOAD_BD, ENG_KLOAD_CH);
    __Runtime_launch_kernel_group_ctrl(e->dev, e->fab, kg, ENG_KLOAD_BD, ENG_KLOAD_CH);
    e->kload_us += us_since(t0);
}

// The variant layer l runs on in eager mode.
static int layer_kv(const eng_layer_t *L) { return L->stem ? KV_STEM : KV_CONV; }

// Eager: reload before every layer. Reloads are full (warm reload off):
// program memory is resent every time. Core data memory is not part of the
// ELF image, so an on-core residual (image-major schedule) survives a reload.
static void engine_reload(Engine *e, int l) {
    if (e->eager)
        engine_load_kernel(e, layer_kv(&e->T[l]));
}

// Runtime mode switch. Persistent mode loads the full kernel once, here; it
// only runs the image-major schedule (the full ELF has no room for RESX/DOWNOUT).
static void engine_set_mode(Engine *e, int eager, int layer_major) {
    e->eager = eager;
    e->layer_major = layer_major;
    e->T = layer_major ? eng_lm_layers : eng_layers;
    e->NT = layer_major ? ENG_LM_NLAYERS : ENG_NLAYERS;
    for (int l = 0; l < e->NT; l++) {
        const eng_layer_t *L = &e->T[l];
        for (int c = 0; c < EF_MESH_COLS; c++) {
            int32_t *h = (int32_t *)(e->blob + L->blob_off[c]);
            h[EH_FLAGS] = layer_major ? lm_flags(L) : L->flags;
            __Runtime_sync_for_dev(e->dev, h, ENG_HDR_BYTES);
        }
    }
    if (layer_major) // no stale go from an earlier run
        engine_go(e, 0);
    if (e->mt_loaded >= 0) // a prefetch may still be in flight: let it land before re-routing
        mt_load_wait(e->dev, e->mt_bank);
    mt_route(e->dev, layer_major);
    e->mt_loaded = -1;
    __Runtime_ctrl_elf_warm_enable(!eager);
    if (!eager)
        engine_load_kernel(e, KV_FULL);
    e->kload_us = 0;
}

// Partition, control-packet fabric, kernel loaded + started, weights and every
// activation buffer allocated once (borders written once, never again).
static void engine_init(Engine *e) {
    // Shim BDs go over the control-packet plane (~3 ns/word) instead of MMIO
    // (~372 ns/word, doc/performance/register_write_cost.md). AIE2PS needs high
    // throughput mode for that, and it must be set before device init.
    __Runtime_ctrl_high_throughput_enable(1);
    __Runtime_ctrl_pmap_enable(0);
    e->dev = __Runtime_init_mesh_partition(0, EF_PART_START_COL, EF_PART_NUM_COLS);
    // Control spine on partition col 0 (abs 2), dedicated: the data plane is cols 1..4.
    static __Runtime_CtrlRowFabric fab;
    static const __Runtime_CtrlRowChain fab_rows[] = {{3, 1, 4}, {4, 1, 4}, {5, 1, 4}, {6, 1, 4}};
    __Runtime_ctrl_plan_init(&fab, e->dev, 0, 0, 1, fab_rows, 4);
    __Runtime_ctrl_plan_set_exclusive(&fab, 1);
    e->fab = &fab;
    for (int r = 0; r < EF_MESH_ROWS; r++)
        for (int c = 0; c < EF_MESH_COLS; c++)
            e->tiles[r * 4 + c] = XAie_TileLoc((uint8_t)(EF_FIRST_COL + c), (uint8_t)(EF_FIRST_ROW + r));
    // encoded once; every (re)load replays it. A NULL image means the ELF does
    // not fit the core (e.g. > 16 KB of program memory).
    for (int kv = 0; kv < KV_N; kv++) {
        e->kpkt[kv] = __Runtime_ctrl_kernel_pkt_build(&fab, k_kv_elf[kv], &e->kpkt_bytes[kv]);
        printf("ENG kernel %-8s %u B of control packets%s\n", k_kv_name[kv], e->kpkt_bytes[kv],
               e->kpkt[kv] ? "" : "  -- FAIL: does not encode (program memory overflow?)");
    }

    e->blob = (int8_t *)__Runtime_Alloc(ENG_BLOB_BYTES);
    memcpy(e->blob, eng_blob_src, ENG_BLOB_BYTES);
    __Runtime_sync_for_dev(e->dev, e->blob, ENG_BLOB_BYTES);
    for (int i = 0; i <= ENG_NLAYERS; i++) {
        const eng_layer_t *L = &eng_layers[i == ENG_NLAYERS ? i - 1 : i];
        e->act_bytes[i] = i == 0            ? (long)ENG_IMG_BYTES
                          : i < ENG_NLAYERS ? l8_bytes(L->Hin, L->in_blk, L->wst_in)
                                            : l8_bytes(L->Ho, L->out_blk, L->wst_out);
    }
    for (int b = 0; b < ENG_MAX_BATCH; b++)
        for (int i = 0; i <= ENG_NLAYERS; i++) {
            // the image is the only buffer the CPU writes per run: non-cacheable,
            // so a run needs no cache clean (falls back to cached where it cannot)
            e->act[b][i] = (int8_t *)(i == 0 ? __Runtime_alloc_uncached(e->act_bytes[i] + 256)
                                             : __Runtime_Alloc(e->act_bytes[i] + 256));
#ifdef ENG_EXP_SHARED /* experiment: every image shares image 0's buffers (all images are the same photo) */
            if (b > 0) {
                free(e->act[b][i]);
                e->act[b][i] = e->act[0][i];
                continue;
            }
#endif
            memset(e->act[b][i], i == 0 ? ENG_Q_PAD : -128, e->act_bytes[i] + 256);
            __Runtime_sync_for_dev(e->dev, e->act[b][i], e->act_bytes[i] + 256);
        }
    e->mt_loaded = -1;
    engine_set_mode(e, 0, 0);
}

#define ENG_TMAX (ENG_LM_NLAYERS > ENG_NLAYERS ? ENG_LM_NLAYERS : ENG_NLAYERS)
static double g_arm_us[ENG_TMAX];  // host arm time (begin..commit), summed over images
static double g_turn_us[ENG_TMAX]; // unit turnaround: output drained -> go sent, summed

// One engine layer on image b (image-major schedule): arm shim BDs (control
// packets), wait for the output. The caller does the eager reload.
static double engine_layer(Engine *e, int l, int b) {
    const eng_layer_t *L = &e->T[l];
    int8_t *const *act = e->act[b];
    struct_ioevent ev[EF_MESH_ROWS];
    XTime t0;
    XTime_GetTime(&t0);
    __Runtime_ctrl_shim_bd_begin(e->fab);
    arm_output(e->dev, L, act[L->out_act], L->P, ev);
    arm_weights(e->dev, L, e->blob);
    if (L->stem)
        arm_input_stem(e->dev, act[L->in_act]);
    else
        arm_input(e->dev, L, act[L->in_act]);
    __Runtime_ctrl_shim_bd_commit();
    g_arm_us[l] += us_since(t0);
    for (int r = 0; r < EF_MESH_ROWS; r++)
        __Runtime_wait_io(e->dev, ev[r]);
    return us_since(t0);
}

/* Layer-major work unit: an image, or for a row=image layer a group of four
 * (mesh row r takes image 4u + r; a short last group repeats the last image,
 * its rows then compute -- and write -- identical bytes). */
static int units_of(const eng_layer_t *L, int B) { return L->ri ? (B + EF_MESH_ROWS - 1) / EF_MESH_ROWS : B; }
static int unit_image(const eng_layer_t *L, int B, int u, int r) {
    if (!L->ri)
        return u;
    const int b = u * EF_MESH_ROWS + r;
    return b < B ? b : B - 1;
}

static void arm_unit_out(Engine *e, const eng_layer_t *L, int B, int u, struct_ioevent *ev) {
    int8_t *outs[EF_MESH_ROWS];
    for (int r = 0; r < EF_MESH_ROWS; r++)
        outs[r] = e->act[unit_image(L, B, u, r)][L->out_act];
    arm_output_rows(e->dev, L, outs, (L->flags & ENG_F_DOWN) ? 2 * L->P : L->P, ev);
}

// Unit u's input (+ residual) on input bank u & 1, queued on the shim channel
// behind the previous unit's, so it streams as soon as that is done.
static void arm_unit_in(Engine *e, const eng_layer_t *L, int B, int u) {
    if (L->stem) {
        arm_input_stem(e->dev, e->act[u][L->in_act]);
        return;
    }
    const int8_t *ins[EF_MESH_ROWS], *res[EF_MESH_ROWS];
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        const int b = unit_image(L, B, u, r);
        ins[r] = e->act[b][L->in_act];
        res[r] = L->res_act >= 0 ? e->act[b][L->res_act] : NULL;
    }
    if (L->flags & ENG_F_COMB)
        arm_residual(e->dev, L, res, u & 1);
    arm_input_rows(e->dev, L, ins, u & 1);
}

static int cpu_slice(void);

#ifdef ENG_TIMELINE /* debug: one unit's channel timeline (us after its arm) */
static int g_tl_layer = -1, g_tl_unit = 5;
static void timeline(Engine *e, const eng_layer_t *L, struct_ioevent *ev, XTime t0) {
    double tin[EF_MESH_ROWS] = {0}, tout[EF_MESH_ROWS] = {0};
    int left = 2 * EF_MESH_ROWS;
    while (left) {
        for (int r = 0; r < EF_MESH_ROWS; r++) {
            u8 n = 1;
            if (!tin[r]) {
                XAie_DmaGetPendingBdCount(e->dev, XAie_TileLoc((uint8_t)(EF_FIRST_COL + r), 0), 1, DMA_MM2S, &n);
                if (!n) {
                    tin[r] = us_since(t0);
                    left--;
                }
            }
            if (!tout[r]) {
                XAie_DmaGetPendingBdCount(e->dev, ev[r].io.tile_loc, (u8)ev[r].io.channel_id, ev[r].io.direction, &n);
                if (!n) {
                    tout[r] = us_since(t0);
                    left--;
                }
            }
        }
    }
    printf("ENG TIMELINE call %d unit %d (us after go): in done", L->call, g_tl_unit);
    for (int r = 0; r < EF_MESH_ROWS; r++)
        printf(" %.1f", tin[r]);
    printf(" | out done");
    for (int r = 0; r < EF_MESH_ROWS; r++)
        printf(" %.1f", tout[r]);
    printf("\n");
}
#endif

// ENG_F_GO: once the unit's output BDs are live, let all 16 cores release
// their first output row (lock ENG_LK_GO = 1, one broadcast write).
#define ENG_LOCK_VALUE_REG(id) (0x1F000u + (uint32_t)(id) * 0x10u)
static void engine_go(Engine *e, int value) {
    const uint32_t v = (uint32_t)value;
    __Runtime_ctrl_shim_bd_sync();
    __Runtime_ctrl_row_broadcast_write(e->fab, ENG_LOCK_VALUE_REG(ENG_LK_GO), &v, 1, ENG_KLOAD_BD, ENG_KLOAD_CH, 0);
}

// Wait for a layer's four output channels; while the array works, the host
// runs queued CPU work (quantize of the next batch, tail of the previous) in
// slices of a few us.
static void wait_out(Engine *e, struct_ioevent *ev) {
    // Never poll a shim tile over MMIO while control packets are still writing
    // it: the last batch's acks first (lost writes/acks otherwise).
    __Runtime_ctrl_shim_bd_sync();
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        u8 n = 1;
        for (;;) {
            XAie_DmaGetPendingBdCount(e->dev, ev[r].io.tile_loc, (u8)ev[r].io.channel_id, ev[r].io.direction, &n);
            if (!n || !cpu_slice())
                break;
        }
        __Runtime_wait_io(e->dev, ev[r]);
    }
}

// Layer-major: units 0..n-1 through layer l, pipelined. The input of unit u+1
// is queued while u runs; the output BDs (out-of-order, one per core, no spare
// bank) are re-armed for u+1 once u's output has drained, and the cores hold
// u+1's first output row until then (ENG_F_GO).
static void engine_layer_images(Engine *e, int l, int B) {
    const eng_layer_t *L = &e->T[l];
    const int n = units_of(L, B);
    struct_ioevent ev[EF_MESH_ROWS];
    XTime t0;
    XTime_GetTime(&t0);
    __Runtime_ctrl_shim_bd_begin(e->fab);
    arm_unit_out(e, L, B, 0, ev);
    arm_unit_in(e, L, B, 0);
    if (n > 1 && !L->stem) // the stem's input has a single bank
        arm_unit_in(e, L, B, 1);
    __Runtime_ctrl_shim_bd_commit();
    engine_go(e, 1);
    g_arm_us[l] += us_since(t0);
    for (int u = 0; u < n; u++) {
        wait_out(e, ev);
        if (u + 1 == n)
            break;
        XTime_GetTime(&t0);
        __Runtime_ctrl_shim_bd_begin(e->fab);
#ifdef ENG_EXP_SHARED /* experiment: identical output BDs -> just queue the next task, no go */
        for (int r = 0; r < EF_MESH_ROWS; r++)
            __Runtime_startio(e->dev, ev[r].io, ev[r].io.bd_id, EF_MESH_COLS * L->rows[r]);
        __Runtime_ctrl_shim_bd_commit();
#else
        arm_unit_out(e, L, B, u + 1, ev);
        __Runtime_ctrl_shim_bd_commit();
        engine_go(e, 1);
#endif
        g_turn_us[l] += us_since(t0); // output drained -> next unit may emit
#ifdef ENG_TIMELINE
        if (u + 1 == g_tl_unit && !L->ri && !L->stem) { // in-queue of u+1 holds only u+1 then (u+2 not armed yet)
            XTime tg;
            XTime_GetTime(&tg);
            __Runtime_ctrl_shim_bd_sync();
            timeline(e, L, ev, tg);
        }
#endif
        __Runtime_ctrl_shim_bd_begin(e->fab);
        if (L->stem)
            arm_unit_in(e, L, B, u + 1);
        else if (u + 2 < n)
            arm_unit_in(e, L, B, u + 2);
        __Runtime_ctrl_shim_bd_commit();
        g_arm_us[l] += us_since(t0);
    }
}

// Per-layer time split of the last layer-major batch (us): reload, memtile
// load, image runs.
static double g_lt[ENG_TMAX][3];

// Layer l on images 0..B-1 (layer-major): eager reload, weights DDR ->
// memtile once (prefetched while layer l-1 ran), then every unit streams them
// from the memtile.
static void engine_layer_batch(Engine *e, int l, int B) {
    const eng_layer_t *L = &e->T[l];
    XTime t0, t1, t2;
    XTime_GetTime(&t0);
    engine_reload(e, l);
    XTime_GetTime(&t1);
    // banks alternate per layer run, so the wrap-around prefetch never lands
    // in the bank still streaming
    if (e->mt_loaded != l) {
        if (e->mt_loaded >= 0)
            mt_load_wait(e->dev, e->mt_bank);
        e->mt_bank ^= 1;
        mt_load_start(e->dev, e->fab, L, e->blob, e->mt_bank);
        e->mt_loaded = l;
    }
    const int bank = e->mt_bank;
    mt_load_wait(e->dev, bank);
    mt_stream(e->dev, L, bank, units_of(L, B));
    // prefetch the next layer (layer 0 of the next batch after the last)
    const int next = (l + 1) % e->NT;
    e->mt_bank ^= 1;
    mt_load_start(e->dev, e->fab, &e->T[next], e->blob, e->mt_bank);
    e->mt_loaded = next;
    XTime_GetTime(&t2);
    g_arm_us[l] = g_turn_us[l] = 0;
    engine_layer_images(e, l, B);
    g_lt[l][0] = 1e6 * (double)(t1 - t0) / COUNTS_PER_SECOND;
    g_lt[l][1] = 1e6 * (double)(t2 - t1) / COUNTS_PER_SECOND;
    g_lt[l][2] = us_since(t2);
}

static void print_layer_split(const Engine *e, int B) {
    printf("ENG layer split B=%d (us per batch: reload | memtile load | runs -> runs per image)\n", B);
    for (int l = 0; l < e->NT; l++)
        printf("  call %2d%s p0 %d: %6.1f | %7.1f | %8.1f -> %6.1f per image  (arm %5.1f, turnaround %5.1f per unit)\n",
               e->T[l].call, e->T[l].ri ? " RI" : "   ", e->T[l].p0, g_lt[l][0], g_lt[l][1], g_lt[l][2], g_lt[l][2] / B,
               g_arm_us[l] / units_of(&e->T[l], B), g_turn_us[l] / units_of(&e->T[l], B));
}

// Output y of table entry l vs TVM, row by row (a DOWN layer's rows carry v_d
// after y; a row=image sub-layer before the last has no golden: gold_len 0).
static int check_layer(Engine *e, int l, int b) {
    const eng_layer_t *L = &e->T[l];
    const int8_t *out = e->act[b][L->out_act];
    __Runtime_sync_for_cpu(e->dev, (void *)out, e->act_bytes[L->out_act]);
    const int8_t *gold = eng_gold_src + L->gold_off;
    const long grow = (long)(L->Cout / 8) * L->wst_out * 8, orow = (long)L->out_blk * L->wst_out * 8;
    int bad = 0;
    for (long k = 0; k < L->gold_len; k++) {
        const int8_t got = out[(k / grow) * orow + k % grow];
        if (got != gold[k] && bad++ < 4)
            printf("  MISMATCH call %d byte %ld: got %d want %d\n", L->call, k, got, gold[k]);
    }
    return bad;
}

// After a layer-major run: every image of the last batch is the same photo, so
// every layer's output of every image must equal TVM's. Names the wrong ones.
static int check_batch(Engine *e, int B) {
    int nbad = 0;
    for (int l = 0; l < ENG_NLAYERS; l++)
        for (int b = 0; b < B; b++) {
            const eng_layer_t *L = &eng_layers[l];
            const int8_t *out = e->act[b][l + 1];
            __Runtime_sync_for_cpu(e->dev, (void *)out, e->act_bytes[l + 1]);
            const int8_t *gold = eng_gold_src + L->gold_off;
            const long grow = (long)(L->Cout / 8) * L->wst_out * 8, orow = (long)L->out_blk * L->wst_out * 8;
            long bad = 0, first = -1;
            for (long k = 0; k < L->gold_len; k++)
                if (out[(k / grow) * orow + k % grow] != gold[k] && bad++ == 0)
                    first = k;
            if (bad) {
                if (nbad++ < 12) {
                    printf("  BATCHCHECK call %d image %d: %ld bad, first byte %ld (row %ld); bad rows:", L->call, b,
                           bad, first, first / grow);
                    const long nrows = L->gold_len / grow;
                    for (long r = 0; r < nrows; r++) {
                        long rb = 0;
                        for (long k = 0; k < grow; k++)
                            rb += out[r * orow + k] != gold[r * grow + k];
                        if (!rb)
                            continue;
                        long src = -1; // a golden row this row's bytes equal exactly?
                        for (long g = 0; g < nrows && src < 0; g++)
                            if (!memcmp(out + r * orow, gold + g * grow, grow))
                                src = g;
                        printf(" %ld(%ld%s%ld)", r, rb, src >= 0 ? "=gold row " : "", src >= 0 ? src : 0L);
                    }
                    printf("\n");
                }
            }
        }
    return nbad;
}

// TVM's golden input of table entry l (= the golden output of the spatial
// layer that produced its input) into act[b][in_act], row by row (keeps a 2C
// tensor's v_d half).
static void load_golden_input(Engine *e, int l, int b) {
    const int a = e->T[l].in_act;
    const eng_layer_t *P = &eng_layers[a - 1];
    const long grow = (long)(P->Cout / 8) * P->wst_out * 8, orow = (long)P->out_blk * P->wst_out * 8;
    for (long r = 0; r < P->gold_len / grow; r++)
        memcpy(e->act[b][a] + r * orow, eng_gold_src + P->gold_off + r * grow, grow);
    __Runtime_sync_for_dev(e->dev, e->act[b][a], e->act_bytes[a]);
}

// Verification on image 0: pass 0 isolated (each layer fed TVM's golden input,
// so a bad layer cannot hide the ones after it), pass 1 chained.
static int verify_layers(Engine *e) {
    int total = 0;
    for (int pass = 0; pass < 2; pass++) {
        memcpy(e->act[0][0], eng_img_src, ENG_IMG_BYTES);
        for (int l = 0; l < e->NT; l++) {
            if (pass == 0 && e->T[l].in_act > 0)
                load_golden_input(e, l, 0);
            e->kload_us = 0;
            XTime t0;
            XTime_GetTime(&t0);
            if (e->layer_major) {
                engine_layer_batch(e, l, 1);
            } else {
                engine_reload(e, l);
                engine_layer(e, l, 0);
            }
            const double us = us_since(t0);
            const int bad = check_layer(e, l, 0);
            printf("ENG %s call %2d%s: %s (%d bad)  %.1f us", pass ? "chained " : "isolated", e->T[l].call,
                   e->T[l].ri ? (e->T[l].p0 ? " RI p0>0" : " RI p0=0") : "", bad ? "FAIL" : "PASS", bad, us);
            if (e->eager)
                printf("  (kernel reload %.1f us)", e->kload_us);
            printf("\n");
            total += bad;
        }
    }
    return total;
}

// ---- CPU ends ---------------------------------------------------------------

// TVM call 00: float NCHW [3][224][224] -> int8 (round(x*s) + zp, clamp, -128),
// zero-point border (written once at init); emitted straight into the stem's
// [230][232][4] layout. vrndaq = round half away from zero = roundf, and the
// multiply/add are the same IEEE single ops TVM's C does, so it is bit-exact.
__attribute__((optimize("O3"))) static void quantize_rows(const float *x, int8_t *img, int h0, int h1) {
    const float32x4_t s = vdupq_n_f32(ENG_Q_SCALE), zp = vdupq_n_f32(ENG_Q_ZP);
    const float32x4_t hi = vdupq_n_f32(255.0f), lo = vdupq_n_f32(0.0f);
    const int32x4_t off = vdupq_n_s32(128);
    for (int h = h0; h < h1; h++) {
        int8_t *dst = img + ((h + 3) * 232 + 3) * 4;
        for (int w = 0; w < 224; w += 16) {
            int8x16x4_t o;
            o.val[3] = vdupq_n_s8(0);
            for (int c = 0; c < 3; c++) {
                const float *src = x + (c * 224 + h) * 224 + w;
                int16x8_t n[2];
                for (int k = 0; k < 2; k++) {
                    int32x4_t q[2];
                    for (int j = 0; j < 2; j++) {
                        float32x4_t v = vaddq_f32(vrndaq_f32(vmulq_f32(vld1q_f32(src + k * 8 + j * 4), s)), zp);
                        v = vmaxq_f32(vminq_f32(v, hi), lo);
                        q[j] = vsubq_s32(vcvtq_s32_f32(v), off);
                    }
                    n[k] = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]));
                }
                o.val[c] = vcombine_s8(vmovn_s16(n[0]), vmovn_s16(n[1]));
            }
            vst4q_s8(dst + w * 4, o);
        }
    }
}

static void quantize_image(const float *x, int8_t *img) { quantize_rows(x, img, 0, 224); }

// dense weights repacked once from TVM's [N/8][K][8] to [N][K] for SDOT
static int8_t g_dense_nk[ENG_NCLASS * ENG_FEAT] __attribute__((aligned(64)));
static void dense_repack(void) {
    for (int o = 0; o < ENG_NCLASS; o++)
        for (int k = 0; k < ENG_FEAT; k++)
            g_dense_nk[o * ENG_FEAT + k] = eng_dense_src[((long)(o / 8) * ENG_FEAT + k) * 8 + o % 8];
}

// TVM calls 38..40: Q(v << 2) per element (a 256-entry table), 7x7 average,
// clamp to u8, - 128, dense 512 -> 1000 + requant, then to float. In pieces
// (pool row h, feature, dense outputs [o0, o1)) so they can run in the gaps
// while the host waits on the array.
__attribute__((optimize("O3"))) static void tail_pool_row(const int8_t *fin, const eng_layer_t *L, int h,
                                                          int32_t *sum) {
    if (h == 0)
        memset(sum, 0, ENG_FEAT * sizeof(int32_t));
    for (int b = 0; b < ENG_FEAT / 8; b++) {
        const int8_t *row = fin + (((h + 1) * (ENG_FEAT / 8) + b) * L->wst_out + 1) * 8;
        for (int w = 0; w < L->Wo; w++)
            for (int i = 0; i < 8; i++)
                sum[b * 8 + i] += eng_final_lut[(uint8_t)(row[w * 8 + i] ^ 0x80)];
    }
}

static void tail_feat(const eng_layer_t *L, const int32_t *sum, int8_t *feat) {
    for (int c = 0; c < ENG_FEAT; c++) {
        int32_t v = sum[c] / (L->Ho * L->Wo);
        v = v < 255 ? v : 255;
        feat[c] = (int8_t)((v > 0 ? v : 0) - 128);
    }
}

__attribute__((optimize("O3"))) static void tail_dense(const int8_t *feat, float *logits, int o0, int o1) {
    for (int o = o0; o < o1; o++) {
        const int8_t *w = g_dense_nk + o * ENG_FEAT;
        int32x4_t a = vdupq_n_s32(0);
        for (int k = 0; k < ENG_FEAT; k += 16)
            a = vdotq_s32(a, vld1q_s8(feat + k), vld1q_s8(w + k));
        const int32_t acc = vaddvq_s32(a);
        const int64_t sh = (int64_t)eng_dense_shift[o] + 31;
        int32_t v =
            (int32_t)((((int64_t)(acc + eng_dense_bz[o])) * (int64_t)eng_dense_mult[o] + ((int64_t)1 << (sh - 1))) >>
                      sh) +
            ENG_DENSE_C0;
        v = v < 255 ? v : 255;
        logits[o] = (float)((v > 0 ? v : 0) - ENG_DENSE_ZP) * ENG_DENSE_SCALE;
    }
}

static void cpu_tail(const int8_t *fin, const eng_layer_t *L, float *logits) {
    static int32_t sum[ENG_FEAT];
    static int8_t feat[ENG_FEAT] __attribute__((aligned(16)));
    for (int h = 0; h < L->Ho; h++)
        tail_pool_row(fin, L, h, sum);
    tail_feat(L, sum, feat);
    tail_dense(feat, logits, 0, ENG_NCLASS);
}

static void top5(const float *lg, int *idx) {
    for (int k = 0; k < 5; k++) {
        int best = -1;
        for (int i = 0; i < ENG_NCLASS; i++) {
            int used = 0;
            for (int j = 0; j < k; j++)
                used |= idx[j] == i;
            if (!used && (best < 0 || lg[i] > lg[best]))
                best = i;
        }
        idx[k] = best;
    }
}

// One batch end to end: B float images -> B x logits, all timed. t[0]
// quantize, t[1] AIE (incl. kernel loads), t[2] CPU tail, t[3] total; returns
// the number of images whose logits differ from TVM's.
static int run_batch(Engine *e, int B, float (*logits)[ENG_NCLASS], double *t) {
    XTime t0;
    XTime_GetTime(&t0);
    for (int b = 0; b < B; b++) {
        quantize_image(eng_image_src, e->act[b][0]);
        // a dsb for the non-cacheable image buffers, a clean for the rest
        __Runtime_sync_for_dev(e->dev, e->act[b][0], ENG_IMG_BYTES);
    }
    t[0] = us_since(t0);
    XTime t1;
    XTime_GetTime(&t1);
    if (e->layer_major) {
        for (int l = 0; l < e->NT; l++)
            engine_layer_batch(e, l, B);
    } else {
        for (int b = 0; b < B; b++)
            for (int l = 0; l < e->NT; l++) {
                engine_reload(e, l);
                engine_layer(e, l, b);
            }
    }
    t[1] = us_since(t1);
    XTime t2;
    XTime_GetTime(&t2);
    const eng_layer_t *L = &eng_layers[ENG_NLAYERS - 1];
    int diff = 0;
    for (int b = 0; b < B; b++) {
        __Runtime_sync_for_cpu(e->dev, e->act[b][ENG_NLAYERS], e->act_bytes[ENG_NLAYERS]);
        cpu_tail(e->act[b][ENG_NLAYERS], L, logits[b]);
        diff += memcmp(logits[b], eng_logits_src, ENG_NCLASS * sizeof(float)) != 0;
    }
    t[2] = us_since(t2);
    t[3] = us_since(t0);
    return diff;
}

/* ---- CPU work in the array's shadow (layer-major) -------------------------
 * Quantizing batch k+1 and the avgpool + dense tail of batch k-1 run while
 * batch k is on the array, in slices small enough (a few us) not to delay the
 * host's re-arming much: wait_out() runs one per poll of a busy channel. */
enum { JOB_QUANT, JOB_TAIL };
#define QUANT_ROWS 8   /* rows per quantize slice: 224 / 8 = 28 slices */
#define TAIL_DENSE 125 /* dense outputs per tail slice */
#define TAIL_PARTS (7 + 1 + ENG_NCLASS / TAIL_DENSE)
#define JQ_MAX (ENG_MAX_BATCH * 2 * (224 / QUANT_ROWS + TAIL_PARTS))
typedef struct {
    uint8_t kind, b, set;
    int16_t part;
} CpuJob;
static struct {
    Engine *e;
    CpuJob j[JQ_MAX];
    int head, tail, diff, batch; /* batch: the one on the array (tails run one batch behind) */
    float logits[2][ENG_MAX_BATCH][ENG_NCLASS];
    int32_t sum[ENG_MAX_BATCH][ENG_FEAT];
    int8_t feat[ENG_MAX_BATCH][ENG_FEAT] __attribute__((aligned(16)));
} g_cq;

static void cq_push(int kind, int b, int set, int parts) {
    for (int p = 0; p < parts; p++)
        g_cq.j[g_cq.tail++ % JQ_MAX] = (CpuJob){(uint8_t)kind, (uint8_t)b, (uint8_t)set, (int16_t)p};
}

static int cpu_slice(void) {
    if (g_cq.head == g_cq.tail)
        return 0;
    const CpuJob j = g_cq.j[g_cq.head++ % JQ_MAX];
    Engine *e = g_cq.e;
    if (j.kind == JOB_QUANT) {
        quantize_rows(eng_image_src, e->act[j.b][0], j.part * QUANT_ROWS, (j.part + 1) * QUANT_ROWS);
        if ((j.part + 1) * QUANT_ROWS == 224) // a dsb for the non-cacheable image buffers, a clean for the rest
            __Runtime_sync_for_dev(e->dev, e->act[j.b][0], ENG_IMG_BYTES);
        return 1;
    }
    const eng_layer_t *L = &eng_layers[ENG_NLAYERS - 1];
    const int8_t *fin = e->act[j.b][ENG_NLAYERS];
    float *lg = g_cq.logits[j.set][j.b];
    if (j.part == 0)
        __Runtime_sync_for_cpu(e->dev, (void *)fin, e->act_bytes[ENG_NLAYERS]);
    if (j.part < L->Ho) {
        tail_pool_row(fin, L, j.part, g_cq.sum[j.b]);
    } else if (j.part == L->Ho) {
        tail_feat(L, g_cq.sum[j.b], g_cq.feat[j.b]);
    } else {
        const int o0 = (j.part - L->Ho - 1) * TAIL_DENSE;
        tail_dense(g_cq.feat[j.b], lg, o0, o0 + TAIL_DENSE);
        if (o0 + TAIL_DENSE == ENG_NCLASS && memcmp(lg, eng_logits_src, ENG_NCLASS * sizeof(float)) != 0) {
            if (g_cq.diff++ < 8)
                printf("  LOGITS DIFFER: batch %d image %d\n", g_cq.batch - 1, j.b);
        }
    }
    return 1;
}

static void cpu_drain(void) {
    while (cpu_slice())
        ;
}

// nb back-to-back layer-major batches of B images, CPU ends overlapped with
// the array: batch k+1 is quantized once batch k's stem has read its images,
// batch k's tail runs during batch k+1 (it must finish before batch k+1's last
// layer rewrites the final activations). Returns the number of images whose
// logits differ from TVM; *us = wall time of the whole run.
static int check_batch(Engine *e, int B);
static int run_batches(Engine *e, int B, int nb, double *us) {
    g_cq.e = e;
    g_cq.head = g_cq.tail = g_cq.diff = 0;
    XTime t0;
    XTime_GetTime(&t0);
    for (int b = 0; b < B; b++)
        cq_push(JOB_QUANT, b, 0, 224 / QUANT_ROWS);
    cpu_drain();
    for (int k = 0; k < nb; k++) {
        g_cq.batch = k;
        for (int l = 0; l < e->NT; l++) {
            if (l == 1 && k + 1 < nb)
                for (int b = 0; b < B; b++)
                    cq_push(JOB_QUANT, b, 0, 224 / QUANT_ROWS);
            if (l == e->NT - 1)
                cpu_drain();
            engine_layer_batch(e, l, B);
        }
        cpu_drain();
#ifdef ENG_CHECK_EVERY_BATCH /* debug: locate intermittent errors (distorts timing) */
        if (check_batch(e, B))
            printf("  BATCHCHECK batch %d above\n", k);
#endif
        for (int b = 0; b < B; b++)
            cq_push(JOB_TAIL, b, k & 1, TAIL_PARTS);
    }
    g_cq.batch = nb;
    cpu_drain();
    *us = us_since(t0);
    return g_cq.diff;
}

typedef struct {
    const char *name;
    int eager, layer_major, batch;
} EngMode;

/* Runtime configurations main() runs, each verified then timed. eng_modes
 * selects them (bit i = row i); volatile so it can be poked on a loaded ELF
 * (xsdb mwr) without a rebuild. */
static const EngMode k_modes[] = {
    {"persistent", 0, 0, 1},         {"eager", 1, 0, 1},
    {"eager layer-major", 1, 1, 1},  {"eager layer-major", 1, 1, 4},
    {"eager layer-major", 1, 1, 8},  {"eager layer-major", 1, 1, 16},
    {"eager layer-major", 1, 1, 32},
};
#define ENG_NMODES (int)(sizeof(k_modes) / sizeof(k_modes[0]))
#define ENG_NBATCH 6 /* layer-major: batches per timed run */
#ifndef ENG_NRUNS
#define ENG_NRUNS 3
#endif
#ifndef ENG_MODES_DEFAULT /* default: persistent (image-major) only; -DENG_MODES_DEFAULT=<mask> selects others */
#define ENG_MODES_DEFAULT 1
#endif
volatile int eng_modes = ENG_MODES_DEFAULT;

// Verify every layer (image 0), then time 5 batches in the given mode.
// Returns the number of failures (bad bytes + images whose logits differ).
static int run_mode(Engine *e, const EngMode *m) {
    engine_set_mode(e, m->eager, m->layer_major);
    const int bad = m->batch == 1 ? verify_layers(e) : 0;
    if (m->batch == 1)
        printf("ENG [%s B=%d] layers %s (%d bad bytes)\n", m->name, m->batch, bad ? "FAIL" : "PASS", bad);
    static float logits[ENG_MAX_BATCH][ENG_NCLASS];
    int fails = 0;
    double best = 1e30;
    if (m->layer_major) { // throughput: ENG_NBATCH batches back to back, CPU ends overlapped
        for (int it = 0; it < ENG_NRUNS; it++) {
            double us;
            e->kload_us = 0;
            const int diff = run_batches(e, m->batch, ENG_NBATCH, &us);
            fails += diff;
            if (diff)
                check_batch(e, m->batch);
            best = us < best ? us : best;
            const int n = m->batch * ENG_NBATCH;
            printf("ENG [%s B=%d] run %d: %d batches, %d/%d logits == TVM (bit-exact) | %.1f us per image "
                   "(kernel loads %.1f)\n",
                   m->name, m->batch, it, ENG_NBATCH, n - diff, n, us / n, e->kload_us / n);
        }
        memcpy(logits, g_cq.logits[(ENG_NBATCH - 1) & 1], sizeof(logits));
        best /= ENG_NBATCH;
    }
    for (int it = 0; it < (m->layer_major ? 0 : 5); it++) {
        double t[4];
        e->kload_us = 0;
        const int diff = run_batch(e, m->batch, logits, t);
        fails += diff;
        best = t[3] < best ? t[3] : best;
        int k5[5];
        top5(logits[0], k5);
        printf("ENG [%s B=%d] batch %d: %d/%d logits == TVM (bit-exact)  top5 %d %d %d %d %d | per image: quantize "
               "%.1f  aie %.1f (kernel loads %.1f)  tail %.1f  total %.1f us\n",
               m->name, m->batch, it, m->batch - diff, m->batch, k5[0], k5[1], k5[2], k5[3], k5[4], t[0] / m->batch,
               t[1] / m->batch, e->kload_us / m->batch, t[2] / m->batch, t[3] / m->batch);
    }
    if (m->layer_major)
        print_layer_split(e, m->batch);
    int k5[5];
    top5(logits[0], k5);
    printf("resnet18 [%s B=%d]: top1 class=%d (%s) logit=%.6f\n", m->name, m->batch, k5[0], imagenet_labels[k5[0]],
           (double)logits[0][k5[0]]);
    printf("PERF SUMMARY [%s B=%d] best %.3f ms per image (float image -> logits)\n", m->name, m->batch,
           best / m->batch / 1000.0);
    return bad + fails;
}

int main() {
    printf("=== resnet engine: int8 ResNet-18, every conv on the AIE ===\n");
    static Engine e;
    engine_init(&e);
    for (int kv = 0; kv < KV_N; kv++)
        if (!e.kpkt[kv]) {
            printf("ENG FAIL\n");
            return 1;
        }
    dense_repack();
    int fails = 0;
    for (int i = 0; i < ENG_NMODES; i++)
        if (eng_modes & (1 << i))
            fails += run_mode(&e, &k_modes[i]);
    { // AIE clock: core-module timer vs the A78's XTime over ~2 ms
        u64 a0, a1;
        XTime h0, h1;
        XAie_LocType t = XAie_TileLoc(EF_FIRST_COL, EF_FIRST_ROW);
        XTime_GetTime(&h0);
        XAie_ReadTimer(e.dev, t, XAIE_CORE_MOD, &a0);
        do
            XTime_GetTime(&h1);
        while ((double)(h1 - h0) / COUNTS_PER_SECOND < 2e-3);
        XAie_ReadTimer(e.dev, t, XAIE_CORE_MOD, &a1);
        XTime_GetTime(&h1);
        printf("ENG AIE clock %.1f MHz\n", (double)(a1 - a0) / ((double)(h1 - h0) / COUNTS_PER_SECOND) / 1e6);
    }
    const int ok = !fails;
    printf("ENG %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
