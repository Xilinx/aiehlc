/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * E1 -- persistent conv engine core program, first real layer: 3x3 stride-1
 * int8 conv with TVM's per-channel requant epilogue (int8 out), L8 layout in
 * and out. Loaded once; loops over images forever.
 *
 * Per core (mesh row r, mesh col c): output rows r*14..r*14+13, output
 * channels c*16..c*16+15.
 *   weights  S2MM ch1, once per layer: [16 x {bz, mult, shift}] + B tiles
 *            [tap 9][kc CIN/8][nb 2][k 8][n 8]
 *   input    S2MM ch0, the core's 16 padded input rows streamed through a
 *            4-slot ring (BDs 0,1,6,7 chained, lock-counted)
 *   output   MM2S ch0, one packet per output row: [nb 2][px 56][8] =
 *            exactly the two L8 channel blocks this core owns
 * The 3 kw taps of one (row, channel block) come from three ALIGNED 64-byte
 * loads + shuffle_down_fill; two output M-blocks share each B load.
 * Requant: the exact 2x32-bit-split scheme of conv2dstem.cc (kernel header
 * there), then uint8 - 128 = TVM's int8 result.
 ******************************************************************************/
#include <stdint.h>
#include <adf.h>
#include <aie_api/aie.hpp>
#include "aie_kernel_runtime.h"
#include "engine_fabric.h"
#include "e1_layer.h"

alignas(64) int8_t e1_ring[E1_RING][E1_IN_ROW_BYTES];
alignas(64) int8_t e1_w[E1_W_BYTES];
alignas(64) int8_t e1_out_ping[E1_OUT_ROW_BYTES];
alignas(64) int8_t e1_out_pong[E1_OUT_ROW_BYTES];

static const int k_pkt[EF_MESH_ROWS][EF_MESH_COLS] = EF_PKT_ID_TABLE;
static const int k_ooo[EF_MESH_ROWS][EF_MESH_COLS] = EF_OOO_BD_TABLE;

#define LK_ACQ(id, v) acquire_greater_equal(EF_LOCK_BASE + (id), (v))
#define LK_REL(id, v) release(EF_LOCK_BASE + (id), (v))
#define BD_IN_2 6 /* ring slots 2,3 (slots 0,1 use EF_BD_IN_PING/PONG) */
#define BD_IN_3 7

// Requant parameters, per N-block (8 output channels), per lane of a 32-lane
// half of the 8x8 C tile (lane = m*8 + n).
alignas(64) static int32_t ep_bz[2][32];
alignas(64) static int16_t ep_mh[2][32], ep_ml[2][32], ep_p1[2][32], ep_p2[2][32];
alignas(64) static int32_t ep_r1[2][32], ep_r2[2][32];
static int ep_a[2], ep_B1[2], ep_B2[2];

static inline void wr(const AieKcReg *kc, int n) { core_reg_write_block(kc, n); }

static void cfg_dma(int pkt, int ooo) {
    AieKcReg kc[8];
    // weights: one block
    wr(kc, aie_kc_encode_bd(kc, EF_BD_W_PING, (uintptr_t)e1_w, E1_W_BYTES, -1, EF_LK_W_EMPTY, -1, EF_LK_W_FULL, 1, 0, 0,
                            0, -1));
    aie_kc_encode_s2mm_start(kc, 1, EF_BD_W_PING, 1, 0);
    core_reg_write(kc[0].off, kc[0].val);
    // input ring: 0 -> 1 -> 6 -> 7, repeated ROWS_IN/4 times
    const int ids[4] = {EF_BD_IN_PING, EF_BD_IN_PONG, BD_IN_2, BD_IN_3};
    for (int s = 3; s >= 0; s--)
        wr(kc, aie_kc_encode_bd(kc, ids[s], (uintptr_t)e1_ring[s], E1_IN_ROW_BYTES, s < 3 ? ids[s + 1] : -1,
                                EF_LK_IN_EMPTY, -1, EF_LK_IN_FULL, 1, 0, 0, 0, -1));
    aie_kc_encode_s2mm_start(kc, 0, EF_BD_IN_PING, E1_ROWS_IN_CORE / E1_RING, 0);
    core_reg_write(kc[0].off, kc[0].val);
    // output: ping -> pong packets, ROWS/2 times
    wr(kc, aie_kc_encode_bd(kc, EF_BD_OUT_PONG, (uintptr_t)e1_out_pong, E1_OUT_ROW_BYTES, -1, EF_LK_OUT_FULL, -1,
                            EF_LK_OUT_EMPTY, 1, 1, pkt, 0, ooo));
    wr(kc, aie_kc_encode_bd(kc, EF_BD_OUT_PING, (uintptr_t)e1_out_ping, E1_OUT_ROW_BYTES, EF_BD_OUT_PONG,
                            EF_LK_OUT_FULL, -1, EF_LK_OUT_EMPTY, 1, 1, pkt, 0, ooo));
    aie_kc_encode_mm2s_start(kc, 0, EF_BD_OUT_PING, E1_ROWS_CORE / 2, 0);
    core_reg_write(kc[0].off, kc[0].val);
}

static inline int32_t rd_i32(const int8_t *p) {
    return (int32_t)((uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) | ((uint32_t)(uint8_t)p[2] << 16) |
                     ((uint32_t)(uint8_t)p[3] << 24));
}

// Decode the 16 channels' {bz, mult, shift} into the per-lane epilogue vectors.
static void decode_params(void) {
    for (int nb = 0; nb < 2; nb++) {
        int S[8], smin = 1 << 30;
        for (int n = 0; n < 8; n++) {
            S[n] = rd_i32(&e1_w[((nb * 8 + n) * 3 + 2) * 4]) + 31 - 16;
            smin = S[n] < smin ? S[n] : smin;
        }
        const int a = smin - 1;
        int B1 = 1, B2 = 0;
        for (int n = 0; n < 8; n++) {
            const int b = S[n] - a, b1 = b < 14 ? b : 14;
            B1 = b1 > B1 ? b1 : B1;
            B2 = b - b1 > B2 ? b - b1 : B2;
        }
        ep_a[nb] = a;
        ep_B1[nb] = B1;
        ep_B2[nb] = B2;
        for (int n = 0; n < 8; n++) {
            const int b = S[n] - a, b1 = b < 14 ? b : 14, b2 = b - b1;
            const int32_t bz = rd_i32(&e1_w[((nb * 8 + n) * 3 + 0) * 4]);
            const int32_t mult = rd_i32(&e1_w[((nb * 8 + n) * 3 + 1) * 4]);
            const int32_t ml = (int32_t)(int16_t)(uint16_t)((uint32_t)mult & 0xFFFFu);
            const int32_t mh = (mult - ml) >> 16;
            for (int m = 0; m < 4; m++) {
                ep_bz[nb][m * 8 + n] = bz;
                ep_mh[nb][m * 8 + n] = (int16_t)mh;
                ep_ml[nb][m * 8 + n] = (int16_t)ml;
                ep_p1[nb][m * 8 + n] = (int16_t)(1 << (B1 - b1));
                ep_r1[nb][m * 8 + n] = b2 >= 1 ? 0 : (int32_t)1 << (B1 - 1);
                ep_p2[nb][m * 8 + n] = (int16_t)(1 << (B2 - b2));
                ep_r2[nb][m * 8 + n] = b2 >= 1 ? (int32_t)1 << (B2 - 1) : 0;
            }
        }
    }
}

typedef aie::mmul<8, 8, 8, int8_t, int8_t> MM;

// Requantize one 32-lane half (pixel rows 4h..4h+3) of N-block nb -> int8.
static inline aie::vector<int8_t, 32> req_half(const aie::accum<acc32, 64> &c, int nb, int h) {
    const aie::vector<int32_t, 32> x = c.template extract<32>(h).template to_vector<int32_t>(0);
    const aie::vector<int32_t, 32> t = aie::mul(x, aie::load_v<32>(ep_ml[nb])).template to_vector<int32_t>(16);
    aie::accum<acc64, 32> p;
    p.from_vector(t, 0);
    p = aie::mac(p, x, aie::load_v<32>(ep_mh[nb]));
    const aie::vector<int32_t, 32> q = p.template to_vector<int32_t>(ep_a[nb]);
    aie::accum<acc64, 32> r;
    r.from_vector(aie::load_v<32>(ep_r1[nb]), 0);
    r = aie::mac(r, q, aie::load_v<32>(ep_p1[nb]));
    const aie::vector<int32_t, 32> q2 = r.template to_vector<int32_t>(ep_B1[nb]);
    r.from_vector(aie::load_v<32>(ep_r2[nb]), 0);
    r = aie::mac(r, q2, aie::load_v<32>(ep_p2[nb]));
    const aie::vector<uint8_t, 32> u = r.template to_vector<uint8_t>(ep_B2[nb]);
    // TVM: clip(v, 0, 255) - 128 as int8 == flip the top bit of the uint8.
    return aie::bit_xor(u.template cast_to<int8_t>(), aie::broadcast<int8_t, 32>((int8_t)-128));
}

static inline void store_block(int8_t *out, int ow0, const MM &c0, const MM &c1) {
    const aie::accum<acc32, 64> a0 = c0.to_accum(), a1 = c1.to_accum();
    // C tile [px][oc] row-major == one L8 block: nb b at out + b*W*8 + ow0*8.
    aie::store_v(out + ow0 * 8, req_half(a0, 0, 0));
    aie::store_v(out + ow0 * 8 + 32, req_half(a0, 0, 1));
    aie::store_v(out + E1_W * 8 + ow0 * 8, req_half(a1, 1, 0));
    aie::store_v(out + E1_W * 8 + ow0 * 8 + 32, req_half(a1, 1, 1));
}

#define LD64(p) aie::load_v<64>(p)

// One output row from the three ring rows r0,r1,r2 (kh = 0,1,2).
static void conv_row(const int8_t *r0, const int8_t *r1, const int8_t *r2, int8_t *out,
                     const aie::accum<acc32, 64> *bzacc) {
    const int8_t *rows[3] = {r0, r1, r2};
    const int8_t *bp = e1_w + E1_PARAM_BYTES;
    // Pairs of 8-px M-blocks: (0,8) (16,24) (32,40); then 48 alone.
    for (int ow0 = 0; ow0 < 48; ow0 += 16) {
        MM cA0(bzacc[0]), cA1(bzacc[1]), cB0(bzacc[0]), cB1(bzacc[1]);
        for (int kh = 0; kh < 3; kh++) {
            for (int kc = 0; kc < E1_CIN / 8; kc++)
                chess_prepare_for_pipelining {
                    const int8_t *seg = rows[kh] + kc * E1_WST * 8 + ow0 * 8;
                    const aie::vector<int8_t, 64> v0 = LD64(seg), v1 = LD64(seg + 64), v2 = LD64(seg + 128);
                    const int8_t *bk = bp + ((kh * 3) * (E1_CIN / 8) + kc) * 128;
                    for (int kw = 0; kw < 3; kw++)
                        chess_unroll_loop() {
                            const int8_t *b = bk + kw * (E1_CIN / 8) * 128;
                            const aie::vector<int8_t, 64> b0 = LD64(b), b1 = LD64(b + 64);
                            const aie::vector<int8_t, 64> aA = aie::shuffle_down_fill(v0, v1, kw * 8);
                            const aie::vector<int8_t, 64> aB = aie::shuffle_down_fill(v1, v2, kw * 8);
                            cA0.mac(aA, b0);
                            cA1.mac(aA, b1);
                            cB0.mac(aB, b0);
                            cB1.mac(aB, b1);
                        }
                }
        }
        store_block(out, ow0, cA0, cA1);
        store_block(out, ow0 + 8, cB0, cB1);
    }
    {
        const int ow0 = 48;
        MM c0(bzacc[0]), c1(bzacc[1]);
        for (int kh = 0; kh < 3; kh++) {
            for (int kc = 0; kc < E1_CIN / 8; kc++)
                chess_prepare_for_pipelining {
                    const int8_t *seg = rows[kh] + kc * E1_WST * 8 + ow0 * 8;
                    const aie::vector<int8_t, 64> v0 = LD64(seg), v1 = LD64(seg + 64);
                    const int8_t *bk = bp + ((kh * 3) * (E1_CIN / 8) + kc) * 128;
                    for (int kw = 0; kw < 3; kw++)
                        chess_unroll_loop() {
                            const int8_t *b = bk + kw * (E1_CIN / 8) * 128;
                            const aie::vector<int8_t, 64> a = aie::shuffle_down_fill(v0, v1, kw * 8);
                            c0.mac(a, LD64(b));
                            c1.mac(a, LD64(b + 64));
                        }
                }
        }
        store_block(out, ow0, c0, c1);
    }
}

// kernel_decl e1
int32_t main() {
    AieKcReg kc[2];
    const int init[6][2] = {{EF_LK_IN_EMPTY, E1_RING}, {EF_LK_IN_FULL, 0},   {EF_LK_W_EMPTY, 1},
                            {EF_LK_W_FULL, 0},         {EF_LK_OUT_EMPTY, 2}, {EF_LK_OUT_FULL, 0}};
    for (int i = 0; i < 6; i++) {
        aie_kc_encode_lock(kc, init[i][0], init[i][1]);
        core_reg_write(kc[0].off, kc[0].val);
    }
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::saturate);
    const unsigned cid = get_coreid();
    const int mc = (int)(cid >> 16) - EF_PART_START_COL - EF_FIRST_COL;
    const int mr = (int)(cid & 0x1F) - EF_FIRST_ROW;
    const int pkt = k_pkt[mr][mc], ooo = k_ooo[mr][mc];

    for (;;) {
        cfg_dma(pkt, ooo);
        LK_ACQ(EF_LK_W_FULL, 1);
        decode_params();
        aie::accum<acc32, 64> bzacc[2];
        for (int nb = 0; nb < 2; nb++) {
            aie::accum<acc32, 32> half;
            half.from_vector(aie::load_v<32>(ep_bz[nb]), 0);
            bzacc[nb].insert(0, half);
            bzacc[nb].insert(1, half);
        }
        LK_ACQ(EF_LK_IN_FULL, 2); // rows 0,1; each output row then adds one
        for (int oh = 0; oh < E1_ROWS_CORE; oh++) {
            LK_ACQ(EF_LK_IN_FULL, 1);
            LK_ACQ(EF_LK_OUT_EMPTY, 1);
            int8_t *out = (oh & 1) ? e1_out_pong : e1_out_ping;
            conv_row(e1_ring[oh % E1_RING], e1_ring[(oh + 1) % E1_RING], e1_ring[(oh + 2) % E1_RING], out, bzacc);
            chess_memory_fence();
            LK_REL(EF_LK_OUT_FULL, 1);
            LK_REL(EF_LK_IN_EMPTY, 1); // row oh is no longer needed
        }
        LK_REL(EF_LK_IN_EMPTY, 2); // rows ROWS_CORE, ROWS_CORE+1
        LK_REL(EF_LK_W_EMPTY, 1);
        LK_ACQ(EF_LK_OUT_EMPTY, 2); // output drained -> BDs may be rewritten
        LK_REL(EF_LK_OUT_EMPTY, 2);
    }
    return 0;
}
