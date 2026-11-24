/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef AIE_CTRLPKT_AOT_H
#define AIE_CTRLPKT_AOT_H

#include <stdint.h>

#define RT_AOT_MAGIC 0x43504B32U
#define RT_AOT_VERSION 1U

#define RT_AOT_WR_BCAST 1U
#define RT_AOT_WR_ROW_ACK 2U
typedef struct {
    uint32_t kind, row, tile_addr, nwords;
    const uint32_t *data;
    const uint32_t *pkt;
    uint32_t pkt_words;
} rt_aot_write;

#define RT_AOT_CALL_BD 1U
#define RT_AOT_CALL_BD_OOO 2U
#define RT_AOT_CALL_CH_OOO 3U
#define RT_AOT_CALL_START 4U
#define RT_AOT_NO_PATCH 0xFFFFU
typedef struct {
    uint8_t kind, col, row, ch, bd, dir, seg, repeat;
    uint32_t arg_hash;
    uint16_t patch[3];
} rt_aot_sb_call;

#define RT_AOT_SB_MAX_SEGS 5
typedef struct {
    uint32_t lo, hi, ncalls;
    const rt_aot_sb_call *calls;
    const uint32_t *seg[RT_AOT_SB_MAX_SEGS];
    uint32_t seg_len[RT_AOT_SB_MAX_SEGS];
    uint32_t seg_last[RT_AOT_SB_MAX_SEGS];
    uint32_t prefix_len[RT_AOT_SB_MAX_SEGS];
    uint32_t prefix_last[RT_AOT_SB_MAX_SEGS];
} rt_aot_shim_bd;

#define RT_AOT_PLAN_MAX_ROWS 16
typedef struct {
    uint32_t n;
    const uint32_t *off, *val;
    const uint32_t *pkt;
    uint32_t pkt_words;
    int32_t shim_col, ctrl_id, resp_s2mm_ch;
    uint32_t nrows;
    uint8_t rows[RT_AOT_PLAN_MAX_ROWS][3];
    uint32_t nmmio;
    const uint8_t (*mmio_tile)[2];
    const uint32_t *mmio_off, *mmio_val;
} rt_aot_plan_ret;

typedef struct {
    uint8_t kind, col, row, ch, bd, dir, repeat, bufidx;
    int32_t args[19];
} rt_aot_win_call;

typedef struct {
    uint32_t ncalls, nbufs;
    const rt_aot_win_call *calls;
} rt_aot_window;

typedef struct {
    uint32_t magic, version, fingerprint;
    uint32_t nwrites;
    const rt_aot_write *writes;
    const rt_aot_shim_bd *shim_bd;
    const rt_aot_plan_ret *plan_ret;
    const rt_aot_window *window;
} rt_aot_table;

static inline uint32_t rt_aot_bd_hash(const int32_t *args, uint32_t n) {
    uint32_t h = 2166136261U;
    for (uint32_t i = 0U; i < n; i++) {
        h ^= (uint32_t)args[i];
        h *= 16777619U;
    }
    return h;
}

#endif
