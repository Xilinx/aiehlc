/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * E0 -- persistent-kernel bring-up for the ResNet conv engine.
 *
 * Loaded once, never reset between layers or images: the core walks a layer
 * table forever and, per layer, programs its OWN tile DMAs (runtime lengths,
 * finite ping-pong tasks), consumes R input blocks + one weight block, emits R
 * output packets, and waits for its output channel to drain before touching the
 * BDs again. Checks the three things the real engine depends on: per-layer core
 * DMA reconfiguration, host re-arming of shim BDs per layer, and persistence
 * across images. See doc/design/resnet_aie_engine.md (E0).
 ******************************************************************************/
#include <stdint.h>
#include <adf.h>
#include <aie_api/aie.hpp>
#include "aie_kernel_runtime.h"
#include "engine_fabric.h"
#include "e0_layers.h"

// Placed by aieml.bcf (_symbol); 64-byte aligned, own tile memory.
alignas(64) int8_t e0_in_ping[E0_IN_MAX];
alignas(64) int8_t e0_in_pong[E0_IN_MAX];
alignas(64) int8_t e0_w[E0_W_MAX];
alignas(64) int8_t e0_out_ping[E0_IN_MAX];
alignas(64) int8_t e0_out_pong[E0_IN_MAX];

static const int k_layers[E0_NUM_LAYERS][3] = E0_LAYER_TABLE;
static const int k_pkt[EF_MESH_ROWS][EF_MESH_COLS] = EF_PKT_ID_TABLE;
static const int k_ooo[EF_MESH_ROWS][EF_MESH_COLS] = EF_OOO_BD_TABLE;

#define LK_ACQ(id, v) acquire_greater_equal(EF_LOCK_BASE + (id), (v))
#define LK_REL(id, v) release(EF_LOCK_BASE + (id), (v))

static inline void wr_block(const AieKcReg *kc, int n) { core_reg_write_block(kc, n); }

// Finite ping-pong S2MM task: rounds blocks of len bytes, alternating ping/pong.
// Chain ping -> pong (pong ends the chain), repeated rounds/2 times, so the
// channel goes idle after exactly `rounds` blocks and the BDs may be rewritten.
static void cfg_s2mm(int ch, int bd0, int bd1, int8_t *b0, int8_t *b1, unsigned len, int lk_empty, int lk_full,
                     unsigned rounds) {
    AieKcReg kc[8];
    if (rounds == 1) {
        wr_block(kc, aie_kc_encode_bd(kc, bd0, (uintptr_t)b0, len, -1, lk_empty, -1, lk_full, 1, 0, 0, 0, -1));
        aie_kc_encode_s2mm_start(kc, ch, bd0, 1, 0);
    } else {
        wr_block(kc, aie_kc_encode_bd(kc, bd1, (uintptr_t)b1, len, -1, lk_empty, -1, lk_full, 1, 0, 0, 0, -1));
        wr_block(kc, aie_kc_encode_bd(kc, bd0, (uintptr_t)b0, len, bd1, lk_empty, -1, lk_full, 1, 0, 0, 0, -1));
        aie_kc_encode_s2mm_start(kc, ch, bd0, rounds / 2, 0);
    }
    core_reg_write(kc[0].off, kc[0].val);
}

// Same for the packet-mode output MM2S (every block one packet to the shim).
static void cfg_mm2s_out(unsigned len, unsigned rounds, int pkt, int ooo) {
    AieKcReg kc[8];
    wr_block(kc, aie_kc_encode_bd(kc, EF_BD_OUT_PONG, (uintptr_t)e0_out_pong, len, -1, EF_LK_OUT_FULL, -1,
                                  EF_LK_OUT_EMPTY, 1, 1, pkt, 0, ooo));
    wr_block(kc, aie_kc_encode_bd(kc, EF_BD_OUT_PING, (uintptr_t)e0_out_ping, len, EF_BD_OUT_PONG, EF_LK_OUT_FULL, -1,
                                  EF_LK_OUT_EMPTY, 1, 1, pkt, 0, ooo));
    aie_kc_encode_mm2s_start(kc, 0, EF_BD_OUT_PING, rounds / 2, 0);
    core_reg_write(kc[0].off, kc[0].val);
}

// kernel_decl e0
int32_t main() {
    AieKcReg kc[2];
    const int init[6][2] = {{EF_LK_IN_EMPTY, 2}, {EF_LK_IN_FULL, 0},   {EF_LK_W_EMPTY, 1},
                            {EF_LK_W_FULL, 0},   {EF_LK_OUT_EMPTY, 2}, {EF_LK_OUT_FULL, 0}};
    for (int i = 0; i < 6; i++) {
        aie_kc_encode_lock(kc, init[i][0], init[i][1]);
        core_reg_write(kc[0].off, kc[0].val);
    }
    const unsigned cid = get_coreid();
    const int mc = (int)(cid >> 16) - EF_PART_START_COL - EF_FIRST_COL;
    const int mr = (int)(cid & 0x1F) - EF_FIRST_ROW;
    const int pkt = k_pkt[mr][mc], ooo = k_ooo[mr][mc];

    for (;;) {
        for (int L = 0; L < E0_NUM_LAYERS; L++) {
            const unsigned in_len = k_layers[L][0], w_len = k_layers[L][1], rounds = k_layers[L][2];
            cfg_s2mm(1, EF_BD_W_PING, EF_BD_W_PONG, e0_w, e0_w, w_len, EF_LK_W_EMPTY, EF_LK_W_FULL, 1);
            cfg_s2mm(0, EF_BD_IN_PING, EF_BD_IN_PONG, e0_in_ping, e0_in_pong, in_len, EF_LK_IN_EMPTY, EF_LK_IN_FULL,
                     rounds);
            cfg_mm2s_out(in_len, rounds, pkt, ooo);

            LK_ACQ(EF_LK_W_FULL, 1);
            for (unsigned r = 0; r < rounds; r++) {
                const int8_t *in = (r & 1) ? e0_in_pong : e0_in_ping;
                int8_t *out = (r & 1) ? e0_out_pong : e0_out_ping;
                LK_ACQ(EF_LK_IN_FULL, 1);
                LK_ACQ(EF_LK_OUT_EMPTY, 1);
                for (unsigned j = 0; j < in_len; j++)
                    out[j] = (int8_t)(in[j] + e0_w[j % w_len] + L + mc);
                chess_memory_fence();
                LK_REL(EF_LK_IN_EMPTY, 1);
                LK_REL(EF_LK_OUT_FULL, 1);
            }
            LK_REL(EF_LK_W_EMPTY, 1);
            // Both output buffers back = the MM2S task has finished reading them;
            // only then may the next layer rewrite BDs 4/5.
            LK_ACQ(EF_LK_OUT_EMPTY, 2);
            LK_REL(EF_LK_OUT_EMPTY, 2);
        }
    }
    return 0;
}
