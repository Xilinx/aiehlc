/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * E1 host: TVM call 04 (3x3 s1, 64->64, 56x56) on the persistent engine,
 * checked byte-for-byte against TVM's own output (layeriohex dump).
 ******************************************************************************/
#include "aie_runtime.h"
#include "aie_timer.h"
#include "engine_fabric.h"
#include "e1_layer.h"
#include "l04_nn_contrib_conv2d_NCHWc_in.h"
#include "l04_nn_contrib_conv2d_NCHWc_out.h"
#include <stdio.h>
#include <string.h>

extern unsigned char _binary_kernel_e1_start[];

#define T_IN l04_nn_contrib_conv2d_NCHWc_in0_data
#define T_W l04_nn_contrib_conv2d_NCHWc_in1_p7_weight_data
#define T_ZPA l04_nn_contrib_conv2d_NCHWc_in2_p8_zero_point_data
#define T_BIAS l04_nn_contrib_conv2d_NCHWc_in3_p9_bias_data
#define T_ZPB l04_nn_contrib_conv2d_NCHWc_in4_p10_zero_point_data
#define T_MULT l04_nn_contrib_conv2d_NCHWc_in5_p11_requant_multiplier_data
#define T_SHIFT l04_nn_contrib_conv2d_NCHWc_in7_p13_requant_shift_data
#define T_OUT l04_nn_contrib_conv2d_NCHWc_out0_data

static const int k_ooo[EF_MESH_ROWS][EF_MESH_COLS] = EF_OOO_BD_TABLE;
static const int k_pkt[EF_MESH_ROWS][EF_MESH_COLS] = EF_PKT_ID_TABLE;
static const int k_out_col[EF_MESH_ROWS] = EF_OUT_SHIM_COL_TABLE;
static const int k_out_ch[EF_MESH_ROWS] = EF_OUT_SHIM_CH_TABLE;

#define L8_ROWS (E1_H + 2)
#define L8_BYTES (L8_ROWS * E1_IN_ROW_BYTES) /* in and out share the geometry */

static int32_t rd_i32(const unsigned char *p, int i) {
    p += 4 * i;
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}
static void wr_i32(int8_t *p, int32_t v) {
    for (int b = 0; b < 4; b++)
        p[b] = (int8_t)((uint32_t)v >> (8 * b));
}

// NCHW4c [16][58][58][4] -> L8 [58][8][64][8]
static void to_l8(const unsigned char *src, int8_t *dst) {
    memset(dst, 0, L8_BYTES);
    for (int h = 0; h < 58; h++)
        for (int c = 0; c < E1_CIN; c++)
            for (int w = 0; w < 58; w++)
                dst[((h * (E1_CIN / 8) + c / 8) * E1_WST + w) * 8 + c % 8] =
                    (int8_t)src[(((c / 4) * 58 + h) * 58 + w) * 4 + c % 4];
}

// Per mesh column: [16 oc x {bias - zp_a - zp_b, mult, shift}] + B tiles.
static int pack_weights(int8_t *blob) {
    memset(blob, 0, (size_t)EF_MESH_COLS * E1_W_BYTES);
    for (int mc = 0; mc < EF_MESH_COLS; mc++) {
        int8_t *b = blob + (size_t)mc * E1_W_BYTES;
        for (int i = 0; i < E1_COUT_CORE; i++) {
            const int oc = mc * E1_COUT_CORE + i;
            const uint32_t bz =
                (uint32_t)rd_i32(T_BIAS, oc) - (uint32_t)rd_i32(T_ZPA, oc) - (uint32_t)rd_i32(T_ZPB, oc);
            wr_i32(b + (i * 3 + 0) * 4, (int32_t)bz);
            wr_i32(b + (i * 3 + 1) * 4, rd_i32(T_MULT, oc));
            wr_i32(b + (i * 3 + 2) * 4, rd_i32(T_SHIFT, oc));
        }
        int8_t *bp = b + E1_PARAM_BYTES;
        for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
                for (int kc = 0; kc < E1_CIN / 8; kc++)
                    for (int nb = 0; nb < 2; nb++)
                        for (int k = 0; k < 8; k++)
                            for (int n = 0; n < 8; n++) {
                                const int oc = mc * E1_COUT_CORE + nb * 8 + n, ic = kc * 8 + k;
                                // TVM OIHW4i4o: [oc/4][ic/4][kh][kw][ic%4][oc%4]
                                const int src =
                                    (((((oc / 4) * 16 + ic / 4) * 3 + kh) * 3 + kw) * 4 + ic % 4) * 4 + oc % 4;
                                bp[((((kh * 3 + kw) * (E1_CIN / 8) + kc) * 2 + nb) * 8 + k) * 8 + n] = (int8_t)T_W[src];
                            }
    }
    return 0;
}

static XAie_LocType shim(int rel_col) { return XAie_TileLoc((uint8_t)rel_col, 0); }

static void run_layer(XAie_DevInst *dev, int8_t *in, int8_t *wblob, int8_t *out) {
    struct_ioevent ev[EF_MESH_ROWS];
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        XAie_LocType t = shim(k_out_col[r]);
        XAie_DmaDesc d = {};
        for (int c = 0; c < EF_MESH_COLS; c++) {
            // padded output row r*14+1, channel blocks 2c and 2c+1, from column 1
            int8_t *dst = out + ((size_t)((r * E1_ROWS_CORE + 1) * (E1_COUT / 8) + 2 * c) * E1_WST + 1) * 8;
            d = __Runtime_dma_bd_config_multidim_ooo(dev, t, dst, k_ooo[r][c], E1_OUT_ROW_BYTES, -1, 0, k_pkt[r][c], -1,
                                                     0, -1, 0, -1, 2, 4, E1_W * 8 / 4, E1_WST * 8, 2, 0, 0,
                                                     E1_IN_ROW_BYTES, E1_ROWS_CORE);
        }
        __Runtime_dma_channel_enable_ooo(dev, t, k_out_ch[r], DMA_S2MM);
        ev[r] = __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, k_out_ch[r], k_ooo[r][0], DMA_S2MM), k_ooo[r][0],
                                  EF_MESH_COLS * E1_ROWS_CORE);
    }
    for (int c = 0; c < EF_MESH_COLS; c++) {
        XAie_LocType t = shim(EF_FIRST_COL + c);
        XAie_DmaDesc d =
            __Runtime_dma_bd_config(dev, t, wblob + (size_t)c * E1_W_BYTES, 0, E1_W_BYTES, -1, 0, 0, 0, 0, 0, 0, -1);
        __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, 0, 0, DMA_MM2S), 0, 1);
    }
    for (int r = 0; r < EF_MESH_ROWS; r++) {
        XAie_LocType t = shim(EF_FIRST_COL + r);
        XAie_DmaDesc d = __Runtime_dma_bd_config(dev, t, in + (size_t)r * E1_ROWS_CORE * E1_IN_ROW_BYTES, 1,
                                                 E1_ROWS_IN_CORE * E1_IN_ROW_BYTES, -1, 0, 0, 0, 0, 0, 0, -1);
        __Runtime_startio(dev, __Runtime_dma_createio_4(t, d, 1, 1, DMA_MM2S), 1, 1);
    }
    for (int r = 0; r < EF_MESH_ROWS; r++)
        __Runtime_wait_io(dev, ev[r]);
}

static int check(const int8_t *out) {
    int bad = 0;
    for (int oc = 0; oc < E1_COUT; oc++)
        for (int h = 0; h < E1_H; h++)
            for (int w = 0; w < E1_W; w++) {
                const int8_t got = out[(((h + 1) * (E1_COUT / 8) + oc / 8) * E1_WST + w + 1) * 8 + oc % 8];
                const int8_t want = (int8_t)T_OUT[(((oc / 4) * E1_H + h) * E1_W + w) * 4 + oc % 4];
                if (got != want && bad++ < 8)
                    printf("  MISMATCH oc %d h %d w %d: got %d want %d\n", oc, h, w, got, want);
            }
    return bad;
}

int main() {
    printf("=== E1: 3x3 s1 conv (TVM call 04) on the persistent engine ===\n");
    XAie_DevInst *dev = __Runtime_init_mesh_partition(0, EF_PART_START_COL, EF_PART_NUM_COLS);
    XAie_LocType tiles[16];
    for (int r = 0; r < EF_MESH_ROWS; r++)
        for (int c = 0; c < EF_MESH_COLS; c++)
            tiles[r * 4 + c] = XAie_TileLoc((uint8_t)(EF_FIRST_COL + c), (uint8_t)(EF_FIRST_ROW + r));
    __Runtime_set_kernel_elf(_binary_kernel_e1_start);
    __Runtime_launch_kernel_group(dev, __Runtime_load_kernel_group_nt(dev, tiles, 16));

    int8_t *in = (int8_t *)__Runtime_Alloc(L8_BYTES);
    int8_t *out = (int8_t *)__Runtime_Alloc(L8_BYTES);
    int8_t *wb = (int8_t *)__Runtime_Alloc((size_t)EF_MESH_COLS * E1_W_BYTES);
    to_l8(T_IN, in);
    pack_weights(wb);
    int total = 0;
    for (int img = 0; img < 3; img++) {
        memset(out, 0x55, L8_BYTES);
        __Runtime_sync_for_dev(dev, in, L8_BYTES);
        __Runtime_sync_for_dev(dev, wb, (size_t)EF_MESH_COLS * E1_W_BYTES);
        __Runtime_sync_for_dev(dev, out, L8_BYTES);
        XTime t0, t1;
        XTime_GetTime(&t0);
        run_layer(dev, in, wb, out);
        XTime_GetTime(&t1);
        __Runtime_sync_for_cpu(dev, out, L8_BYTES);
        const int bad = check(out);
        printf("E1 image %d: %s %d/%d  layer %.1f us\n", img, bad ? "FAIL" : "PASS", E1_COUT * E1_H * E1_W - bad,
               E1_COUT * E1_H * E1_W, 1e6 * (double)(t1 - t0) / COUNTS_PER_SECOND);
        total += bad;
    }
    printf("E1 %s\n", total ? "FAIL" : "PASS");
    return total ? 1 : 0;
}
