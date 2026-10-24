/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * E0 host driver: bring up the fabric once, load + start the persistent kernel
 * once, then per layer re-arm only the shim BDs and wait for the output. Runs
 * the 3-layer table for several "images" and checks every output byte.
 ******************************************************************************/
#include "aie_runtime.h"
#include "aie_timer.h"
#include "engine_fabric.h"
#include "e0_layers.h"
#include <stdio.h>
#include <string.h>

extern unsigned char _binary_kernel_e0_start[];

static const int k_layers[E0_NUM_LAYERS][3] = E0_LAYER_TABLE;
static const int k_ooo[EF_MESH_ROWS][EF_MESH_COLS] = EF_OOO_BD_TABLE;
static const int k_pkt[EF_MESH_ROWS][EF_MESH_COLS] = EF_PKT_ID_TABLE;
static const int k_out_col[EF_MESH_ROWS] = EF_OUT_SHIM_COL_TABLE;
static const int k_out_ch[EF_MESH_ROWS] = EF_OUT_SHIM_CH_TABLE;

#define NIMG 3

static XAie_LocType shim(int rel_col) { return XAie_TileLoc((uint8_t)rel_col, 0); }

// One layer: weights per mesh col, input per mesh row, output per mesh row.
// in:  [row][R][IN]   w: [col][W]   out: [row][col][R][IN] (tile-major)
static int run_layer(XAie_DevInst *dev, int L, int8_t *in, int8_t *w, int8_t *out) {
    const int in_len = k_layers[L][0], w_len = k_layers[L][1], R = k_layers[L][2];
    __Runtime_sync_for_dev(dev, in, (size_t)EF_MESH_ROWS * R * in_len);
    __Runtime_sync_for_dev(dev, w, (size_t)EF_MESH_COLS * w_len);
    __Runtime_sync_for_dev(dev, out, (size_t)EF_MESH_ROWS * EF_MESH_COLS * R * in_len);

    struct_ioevent ev_out[EF_MESH_ROWS];
    // Sinks first: the output channels must be armed before any data can flow.
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        XAie_LocType t = shim(k_out_col[r]);
        XAie_DmaDesc d = {};
        for (int c = 0; c < EF_MESH_COLS; c++) {
            int8_t *dst = out + ((size_t)(r * EF_MESH_COLS + c) * R) * in_len;
            d = __Runtime_dma_bd_config_multidim_ooo(dev, t, dst, k_ooo[r][c], in_len, -1, 0, k_pkt[r][c], -1, 0, -1, 0,
                                                     -1, 2, 4, 16, 64, in_len / 64, 0, 0, in_len, R);
        }
        __Runtime_dma_channel_enable_ooo(dev, t, k_out_ch[r], DMA_S2MM);
        struct_io io = __Runtime_dma_createio_4(t, d, k_out_ch[r], k_ooo[r][0], DMA_S2MM);
        ev_out[r] = __Runtime_startio(dev, io, k_ooo[r][0], EF_MESH_COLS * R);
    }
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType t = shim(EF_FIRST_COL + c);
        XAie_DmaDesc d = __Runtime_dma_bd_config(dev, t, w + (size_t)c * w_len, 0, w_len, -1, 0, 0, 0, 0, 0, 0, -1);
        __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, 0, 0, DMA_MM2S), 0, 1);
    }
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        XAie_LocType t = shim(EF_FIRST_COL + r);
        XAie_DmaDesc d =
            __Runtime_dma_bd_config(dev, t, in + (size_t)r * R * in_len, 1, R * in_len, -1, 0, 0, 0, 0, 0, 0, -1);
        __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, 1, 1, DMA_MM2S), 1, 1);
    }
    for (int r = 0; r < EF_MESH_ROWS; r++)
        __Runtime_wait_io(dev, ev_out[r]);
    __Runtime_sync_for_cpu(dev, out, (size_t)EF_MESH_ROWS * EF_MESH_COLS * R * in_len);

    int bad = 0;
    for (int r = 0; r < EF_MESH_ROWS; r++)
        for (int c = 0; c < EF_MESH_COLS; c++)
            for (int k = 0; k < R; k++)
                for (int j = 0; j < in_len; j++) {
                    const int8_t want =
                        (int8_t)(in[((size_t)r * R + k) * in_len + j] + w[(size_t)c * w_len + j % w_len] + L + c);
                    const int8_t got = out[(((size_t)(r * EF_MESH_COLS + c) * R) + k) * in_len + j];
                    if (got != want && bad++ < 4)
                        printf("  MISMATCH L%d row %d col %d round %d byte %d: got %d want %d\n", L, r, c, k, j, got,
                               want);
                }
    return bad;
}

int main() {
    printf("=== E0: persistent AIE kernel, per-layer DMA re-arm ===\n");
    XAie_DevInst *dev = __Runtime_init_mesh_partition(0, EF_PART_START_COL, EF_PART_NUM_COLS);
    if (!dev) {
        printf("E0 FAIL: partition init\n");
        return 1;
    }
    XAie_LocType tiles[16];
    for (int r = 0; r < EF_MESH_ROWS; r++)
        for (int c = 0; c < EF_MESH_COLS; c++)
            tiles[r * 4 + c] = XAie_TileLoc((uint8_t)(EF_FIRST_COL + c), (uint8_t)(EF_FIRST_ROW + r));
    __Runtime_set_kernel_elf(_binary_kernel_e0_start);
    struct_kernel_group kg = __Runtime_load_kernel_group_nt(dev, tiles, 16);
    __Runtime_launch_kernel_group(dev, kg);
    printf("E0 kernel loaded + started once\n");

    int8_t *in = (int8_t *)__Runtime_Alloc(EF_MESH_ROWS * 6 * E0_IN_MAX);
    int8_t *w = (int8_t *)__Runtime_Alloc(EF_MESH_COLS * E0_W_MAX);
    int8_t *out = (int8_t *)__Runtime_Alloc(EF_MESH_ROWS * EF_MESH_COLS * 6 * E0_IN_MAX);
    int total_bad = 0;
    for (int img = 0; img < NIMG; img++) {
        for (int L = 0; L < E0_NUM_LAYERS; L++) {
            const int R = k_layers[L][2];
            for (int i = 0; i < EF_MESH_ROWS * R * k_layers[L][0]; i++)
                in[i] = (int8_t)(i * 7 + img * 13 + L);
            for (int i = 0; i < EF_MESH_COLS * k_layers[L][1]; i++)
                w[i] = (int8_t)(i * 3 - img);
            memset(out, 0x5A, (size_t)EF_MESH_ROWS * EF_MESH_COLS * R * k_layers[L][0]);
            XTime t0, t1;
            XTime_GetTime(&t0);
            int bad = run_layer(dev, L, in, w, out);
            XTime_GetTime(&t1);
            printf("E0 image %d layer %d: %s (%d bad)  %.1f us incl. check\n", img, L, bad ? "FAIL" : "PASS", bad,
                   1e6 * (double)(t1 - t0) / COUNTS_PER_SECOND);
            total_bad += bad;
        }
    }
    printf("E0 %s\n", total_bad ? "FAIL" : "PASS");
    return total_bad ? 1 : 0;
}
