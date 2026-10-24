/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Persistent conv engine -- the AIE core program. Loaded once; runs every
 * conv layer of int8 ResNet-18 (minus the stem) forever, one layer per header
 * that arrives on the weight channel. Contract: eng_desc.h.
 *
 * Per layer, per core (mesh row r, mesh col c):
 *   S2MM ch1  header, then P*Q weight chunks through a 2-slot ring
 *   S2MM ch0  the core's input rows, one 64-channel chunk at a time, through
 *             a ring of 4 (s1) / 5 (s2) row slots -- P*Q passes over the rows
 *   MM2S ch0  one packet per output row: [p][nb][Wo][8] (OOO shim BD)
 * The core programs all three channels itself (one BD each, iteration
 * addressing), so the host only re-arms shim BDs between layers.
 *
 * Math per output element (exact, = TVM's fused tails):
 *   v = clamp(requant(acc) + c0, 0, 255)
 *   COMB: v = clamp(Q(v - szp, sl, sM, ss) + Q(res - rzr, rl, rM, rs), 0, 255)
 *   stored as int8 v - 128
 * Q(y, l, M, s) = ((y << l) * M + 2^(s-1)) >> s; requant = conv2dstem's exact
 * two-stage scheme. Residuals never leave the core: conv1 of an identity
 * block keeps its slice of x (GRAB), conv1 of a downsample block computes the
 * 1x1 s2 downsample as a 10th tap (DOWN); conv2 reads RES (COMB).
 ******************************************************************************/
/* Variants -- build.sh compiles one ELF per line of kernel_variants, since one
 * ELF with every feature overflows AIE2PS's 16 KB program memory (the chess
 * linker does not stop it; the control-plane loader rejects the segment):
 *   eng      (no define)    stem + conv, image-major schedule: persistent mode
 *   engstem  ENG_KV_STEM    the stem only
 *   engconv  ENG_KV_CONV    conv layers, both schedules (RESX / DOWNOUT)
 * Eager mode reloads per layer anyway, so it loads the variant the layer needs. */
#if defined(ENG_KV_STEM)
#define ENG_HAS_STEM 1
#define ENG_HAS_CONV 0
#define ENG_HAS_LM 0
#elif defined(ENG_KV_CONV)
#define ENG_HAS_STEM 0
#define ENG_HAS_CONV 1
#define ENG_HAS_LM 1
#else
#define ENG_HAS_STEM 1
#define ENG_HAS_CONV 1
#define ENG_HAS_LM 0
#endif

#include <stdint.h>
#include <adf.h>
#include <aie_api/aie.hpp>
#include "aie_kernel_runtime.h"
#include "engine_fabric.h"
#include "eng_desc.h"

#define ENG_W_BASE 0x71C00 /* weight chunk ring (core view), see aieml.bcf */
#define ENG_ARENA_BYTES (0x80000 - ENG_W_BASE)
alignas(64) int8_t eng_arena[ENG_ARENA_BYTES];
alignas(64) int8_t eng_hdr[ENG_HDR_BYTES];

static const int k_pkt[EF_MESH_ROWS][EF_MESH_COLS] = EF_PKT_ID_TABLE;
static const int k_ooo[EF_MESH_ROWS][EF_MESH_COLS] = EF_OOO_BD_TABLE;

#define LK_ACQ(id, v) acquire_greater_equal(EF_LOCK_BASE + (id), (v))
#define LK_REL(id, v) release(EF_LOCK_BASE + (id), (v))
#define LK_HDR_EMPTY 6
#define LK_HDR_FULL 7
#define BD_RING 0
#define BD_RESX 1 /* resident + RESX: residual rows into the staging slots */
#define BD_W 2
#define BD_HDR 3
#define BD_OUT 4
#define ARENA(addr) (eng_arena + ((addr) - ENG_W_BASE))
#ifdef ENG_NO_MAC /* experiment: no MAC loops -- isolates data movement time */
#define ENG_KH 0
#else
#define ENG_KH 3
#endif

typedef aie::mmul<8, 8, 8, int8_t, int8_t> MM;
typedef aie::accum<acc32, 64> Acc;

// Per-lane requant constants of one 16-channel pass (two N-blocks of 8).
struct EpSet {
    alignas(64) int32_t bz[2][32];
    alignas(64) int16_t mh[2][32], ml[2][32], p1[2][32], p2[2][32];
    alignas(64) int32_t r1[2][32], r2[2][32];
    int a[2], B1[2], B2[2];
};
static EpSet ep_main;
static EpSet *ep_down; // in the arena, only for ENG_F_DOWN layers (saves fixed memory)

// Decoded header.
static int g_s, g_q, g_p, g_wst, g_wo, g_mb, g_wr, g_flags, g_rows, g_rows_in, g_slots, g_oslots, g_chunk, g_ntap;
static int g_szp, g_sl, g_ss, g_rzr, g_rl, g_rs, g_c0, g_c0d;
static int32_t g_sM, g_rM;
static int8_t *g_ring, *g_acc, *g_accd, *g_out, *g_res;
static int g_slot_bytes, g_oslot_bytes, g_ostride, g_res_row, g_resident, g_stem, g_go, g_ri;
static int8_t *g_srow;

// ENG_F_GO: the image's first OUT_FULL waits for the host's go (once per image)
static inline void out_release() {
    if (g_go) {
        LK_ACQ(ENG_LK_GO, 1);
        g_go = 0;
    }
    LK_REL(EF_LK_OUT_FULL, 1);
}

static inline int32_t rd_i32(const int8_t *p) {
    return (int32_t)((uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) | ((uint32_t)(uint8_t)p[2] << 16) |
                     ((uint32_t)(uint8_t)p[3] << 24));
}

static void lock_set(int id, int v) {
    AieKcReg kc[1];
    aie_kc_encode_lock(kc, (uint8_t)id, v);
    core_reg_write(kc[0].off, kc[0].val);
}

static void read_header(int mr) {
    const int32_t *h = (const int32_t *)eng_hdr;
    g_s = h[EH_S];
    g_q = h[EH_Q];
    g_p = h[EH_P];
    g_wst = h[EH_WST];
    g_wo = h[EH_WO];
    g_mb = h[EH_MB];
    g_wr = g_mb * 8;
    g_flags = h[EH_FLAGS];
    g_szp = h[EH_SZP];
    g_sl = h[EH_SL];
    g_sM = h[EH_SM];
    g_ss = h[EH_SS];
    g_rzr = h[EH_RZR];
    g_rl = h[EH_RL];
    g_rM = h[EH_RM];
    g_rs = h[EH_RS];
    g_ntap = h[EH_NTAP];
    g_chunk = h[EH_CHUNK];
    g_slots = h[EH_RING_SLOTS];
    g_rows = h[EH_ROWS + mr];
    // P == 1: rows stream through 2 slots; P > 1: a row is complete only in
    // the last pass, so every row has its own slot.
    g_oslots = g_p == 1 ? 2 : g_rows;
    g_rows_in = (g_rows - 1) * g_s + 3;
    g_ring = ARENA(h[EH_A_RING]);
    g_acc = ARENA(h[EH_A_ACC]);
    g_accd = ARENA(h[EH_A_ACCD]);
    g_out = ARENA(h[EH_A_OUT]);
    g_res = ARENA(h[EH_A_RES]);
    ep_down = (EpSet *)ARENA(h[EH_A_EPD]);
    g_resident = (g_flags & ENG_F_RES_IN) != 0;
    g_go = (g_flags & ENG_F_GO) != 0;
    g_ri = ENG_HAS_LM && (g_flags & ENG_F_RI) != 0;
    g_stem = (g_flags & ENG_F_STEM) != 0;
    g_srow = ARENA(h[EH_A_SROW]);
    g_slot_bytes = ENG_CHUNK_BLK * g_wst * 8;
    g_oslot_bytes = g_p * 2 * g_wr * 8;
    g_ostride = g_oslot_bytes;
    g_res_row = g_p * 2 * g_wr * 8;
    if (ENG_HAS_LM && (g_flags & ENG_F_DOWNOUT)) { // v_d rides in the second half of each OUT row slot
        g_ostride = 2 * g_oslot_bytes;
        g_res = g_out + g_oslot_bytes;
        g_res_row = g_ostride;
    }
    if (g_stem) { // see eng_desc.h: mesh row 0 has no halo stem row above it
        g_slot_bytes = ENG_STEM_SLOT;
        g_rows_in = mr == 0 ? 5 + 2 * 28 : 5 + 2 * 29;
    }
    g_c0 = h[EH_C0]; // eng_hdr may be refilled by the next header mid-layer
    g_c0d = h[EH_C0D];
}

// Arm this layer's three channels, then queue the next layer's header.
static void arm_layer(int pkt, int ooo) {
    AieKcReg kc[8];
    AieKcDims d = {0, 0, 0, 0, 0, 0, 0};
    const int resx = (g_flags & ENG_F_RESX) != 0;
    // resident + RESX: the residual rows use a 2-slot staging ring (in OUT,
    // free until compute), the only user of IN_EMPTY in resident mode
    lock_set(EF_LK_IN_EMPTY, g_resident && resx ? 2 : g_slots);
    lock_set(EF_LK_IN_FULL, 0);
    lock_set(EF_LK_OUT_EMPTY, g_p == 1 ? g_oslots : 0); // P > 1: core never waits
    lock_set(EF_LK_OUT_FULL, 0);
    // weights: chunk ring of 2
    d.iter_step = (uint32_t)g_chunk / 4;
    d.iter_wrap = 2;
    core_reg_write_block(kc, aie_kc_encode_bd_nd(kc, BD_W, (uintptr_t)ARENA(ENG_W_BASE), g_chunk, -1, EF_LK_W_EMPTY, -1,
                                                 EF_LK_W_FULL, 1, 0, 0, 0, -1, &d));
    aie_kc_encode_s2mm_start(kc, 1, BD_W, g_p * g_q, 0);
    core_reg_write(kc[0].off, kc[0].val);
    aie_kc_encode_s2mm_start(kc, 1, BD_HDR, 1, 0); // next layer's header
    core_reg_write(kc[0].off, kc[0].val);
    if (ENG_HAS_STEM && g_stem) {
        // stem: 928 B image rows into 960 B slots
        d.iter_step = ENG_STEM_SLOT / 4;
        d.iter_wrap = (uint32_t)g_slots;
        core_reg_write_block(kc, aie_kc_encode_bd_nd(kc, BD_RING, (uintptr_t)g_ring, ENG_STEM_IN_ROW, -1,
                                                     EF_LK_IN_EMPTY, -1, EF_LK_IN_FULL, 1, 0, 0, 0, -1, &d));
        aie_kc_encode_s2mm_start(kc, 0, BD_RING, g_rows_in, 0);
    } else if (g_resident) {
        if (ENG_HAS_LM && resx) { // residual rows first, through 2 staging slots in OUT
            d.iter_step = (uint32_t)g_slot_bytes / 4;
            d.iter_wrap = 2;
            core_reg_write_block(kc, aie_kc_encode_bd_nd(kc, BD_RESX, (uintptr_t)g_out, g_slot_bytes, -1,
                                                         EF_LK_IN_EMPTY, -1, EF_LK_IN_FULL, 1, 0, 0, 0, -1, &d));
            aie_kc_encode_s2mm_start(kc, 0, BD_RESX, g_p * g_rows, 0);
            core_reg_write(kc[0].off, kc[0].val);
        }
        // input resident: chunk q's rows land in slots q*rows_in.., streamed
        // once; one BD use per chunk, no EMPTY gating (slots never reused)
        d.iter_step = (uint32_t)(g_rows_in * g_slot_bytes) / 4;
        d.iter_wrap = (uint32_t)g_q;
        core_reg_write_block(kc, aie_kc_encode_bd_nd(kc, BD_RING, (uintptr_t)g_ring, g_rows_in * g_slot_bytes, -1, -1,
                                                     0, EF_LK_IN_FULL, 1, 0, 0, 0, -1, &d));
        aie_kc_encode_s2mm_start(kc, 0, BD_RING, g_q, 0);
    } else {
        // input: row ring, P*Q passes over the core's rows
        d.iter_step = (uint32_t)g_slot_bytes / 4;
        d.iter_wrap = (uint32_t)g_slots;
        core_reg_write_block(kc, aie_kc_encode_bd_nd(kc, BD_RING, (uintptr_t)g_ring, g_slot_bytes, -1, EF_LK_IN_EMPTY,
                                                     -1, EF_LK_IN_FULL, 1, 0, 0, 0, -1, &d));
        // RESX: the residual rows come through the same ring first
        // (row=image: each chunk's rows once -- every pass reads them from the ring)
        aie_kc_encode_s2mm_start(kc, 0, BD_RING,
                                 (g_ri ? 1 : g_p) * g_q * g_rows_in + (ENG_HAS_LM && resx ? g_p * g_rows : 0), 0);
    }
    core_reg_write(kc[0].off, kc[0].val);
    // output: one packet per row = slot's [p][nb] blocks, Wo px each
    // (DOWNOUT: 2P passes, the v_d half right after the main half)
    const int np = ENG_HAS_LM && (g_flags & ENG_F_DOWNOUT) ? 2 * g_p : g_p;
    AieKcDims o = {1, (uint32_t)g_wo * 2,      (uint32_t)g_wr * 2, (uint32_t)np * 2,
                   0, (uint32_t)g_ostride / 4, (uint32_t)g_oslots};
    core_reg_write_block(kc, aie_kc_encode_bd_nd(kc, BD_OUT, (uintptr_t)g_out, np * 2 * g_wo * 8, -1, EF_LK_OUT_FULL,
                                                 -1, EF_LK_OUT_EMPTY, 1, 1, pkt, 0, ooo, &o));
    aie_kc_encode_mm2s_start(kc, 0, BD_OUT, g_rows, 0);
    core_reg_write(kc[0].off, kc[0].val);
}

// {bz, mult, shift} x 16 -> per-lane two-stage requant constants; c0 is
// folded into the stage-2 rounding constant (exact, see kernel header).
static void decode_params(const int8_t *prm, EpSet &e, int c0) {
    for (int nb = 0; nb < 2; nb++) {
        int S[8], smin = 1 << 30;
        for (int n = 0; n < 8; n++) {
            S[n] = rd_i32(prm + ((nb * 8 + n) * 3 + 2) * 4) + 31 - 16;
            smin = S[n] < smin ? S[n] : smin;
        }
        const int a = smin - 1;
        int B1 = 1, B2 = 0;
        for (int n = 0; n < 8; n++) {
            const int b = S[n] - a, b1 = b < 14 ? b : 14;
            B1 = b1 > B1 ? b1 : B1;
            B2 = b - b1 > B2 ? b - b1 : B2;
        }
        e.a[nb] = a;
        e.B1[nb] = B1;
        e.B2[nb] = B2;
        for (int n = 0; n < 8; n++) {
            const int b = S[n] - a, b1 = b < 14 ? b : 14, b2 = b - b1;
            const int32_t bz = rd_i32(prm + ((nb * 8 + n) * 3 + 0) * 4);
            const int32_t mult = rd_i32(prm + ((nb * 8 + n) * 3 + 1) * 4);
            const int32_t ml = (int32_t)(int16_t)(uint16_t)((uint32_t)mult & 0xFFFFu);
            const int32_t mh = (mult - ml) >> 16;
            const int32_t r2 = (b2 >= 1 ? (int32_t)1 << (B2 - 1) : 0) + ((int32_t)c0 << B2);
            for (int m = 0; m < 4; m++) {
                e.bz[nb][m * 8 + n] = bz;
                e.mh[nb][m * 8 + n] = (int16_t)mh;
                e.ml[nb][m * 8 + n] = (int16_t)ml;
                e.p1[nb][m * 8 + n] = (int16_t)(1 << (B1 - b1));
                e.r1[nb][m * 8 + n] = b2 >= 1 ? 0 : (int32_t)1 << (B1 - 1);
                e.p2[nb][m * 8 + n] = (int16_t)(1 << (B2 - b2));
                e.r2[nb][m * 8 + n] = r2;
            }
        }
    }
}

// Requantize one 32-lane half (pixel rows 4h..4h+3) of N-block nb, up to the
// final shift by e.B2[nb] (caller picks the result type).
static inline aie::accum<acc64, 32> req_acc(const Acc &c, const EpSet &e, int nb, int h) {
    const aie::vector<int32_t, 32> x = c.template extract<32>(h).template to_vector<int32_t>(0);
    const aie::vector<int32_t, 32> t = aie::mul(x, aie::load_v<32>(e.ml[nb])).template to_vector<int32_t>(16);
    aie::accum<acc64, 32> p;
    p.from_vector(t, 0);
    p = aie::mac(p, x, aie::load_v<32>(e.mh[nb]));
    const aie::vector<int32_t, 32> q = p.template to_vector<int32_t>(e.a[nb]);
    aie::accum<acc64, 32> r;
    r.from_vector(aie::load_v<32>(e.r1[nb]), 0);
    r = aie::mac(r, q, aie::load_v<32>(e.p1[nb]));
    const aie::vector<int32_t, 32> q2 = r.template to_vector<int32_t>(e.B1[nb]);
    r.from_vector(aie::load_v<32>(e.r2[nb]), 0);
    return aie::mac(r, q2, aie::load_v<32>(e.p2[nb]));
}

static inline aie::vector<int8_t, 32> u8_to_i8(const aie::vector<uint8_t, 32> &u) {
    return aie::bit_xor(u.template cast_to<int8_t>(), aie::broadcast<int8_t, 32>((int8_t)-128));
}

// Exact Q(x, M, s) = (x * M + 2^(s-1)) >> s for small x, M split mh*2^16 + ml.
static inline aie::vector<int32_t, 32> qmul(const aie::vector<int32_t, 32> &x, int32_t M, int s) {
    const int32_t ml = (int32_t)(int16_t)(uint16_t)((uint32_t)M & 0xFFFFu), mh = (M - ml) >> 16;
    const aie::vector<int32_t, 32> t =
        aie::mul(x, aie::broadcast<int16_t, 32>((int16_t)ml)).template to_vector<int32_t>(16);
    aie::accum<acc64, 32> p;
    p.from_vector(aie::add(t, aie::broadcast<int32_t, 32>((int32_t)1 << (s - 17))), 0);
    p = aie::mac(p, x, aie::broadcast<int16_t, 32>((int16_t)mh));
    return p.template to_vector<int32_t>(s - 16);
}

// (x << l) as int32 lanes.
static inline aie::vector<int32_t, 32> shl32(const aie::vector<int16_t, 32> &x, int l) {
    return aie::mul(x, aie::broadcast<int16_t, 32>((int16_t)(1 << l))).template to_vector<int32_t>(0);
}

// COMB epilogue for one 32-lane half: v (requantized, pre-clamp) + residual.
static inline aie::vector<int8_t, 32> combine(const aie::accum<acc64, 32> &r, int B2, const int8_t *res) {
    aie::vector<int16_t, 32> v = r.template to_vector<int16_t>(B2);
    v = aie::max(aie::min(v, aie::broadcast<int16_t, 32>(255)), aie::broadcast<int16_t, 32>(0));
    const aie::vector<int16_t, 32> xs = aie::sub(v, aie::broadcast<int16_t, 32>((int16_t)g_szp));
    const aie::vector<int32_t, 32> ts = qmul(shl32(xs, g_sl), g_sM, g_ss);
    // stored residual is int8 (res - 128)
    const aie::vector<int16_t, 32> xr =
        aie::add(aie::unpack(aie::load_v<32>(res)), aie::broadcast<int16_t, 32>((int16_t)(128 - g_rzr)));
    const aie::vector<int32_t, 32> tr = qmul(shl32(xr, g_rl), g_rM, g_rs);
    aie::accum<acc32, 32> t;
    t.from_vector(aie::sub(aie::add(ts, tr), aie::broadcast<int32_t, 32>(128)), 0);
    return t.template to_vector<int8_t>(0); // saturates: clamp(0,255) - 128
}

#define LD64(p) aie::load_v<64>(p)

// ACC row of output row oh: Q > 1 keeps every row's partial sums; Q == 1 only
// the row being finished (its epilogue runs right after the row's MACs).
static inline int acc_row(int oh) { return g_q > 1 ? oh : 0; }

static inline Acc acc_init(const int8_t *buf, const Acc &bz, int oh, int j, int nb, int q) {
    if (q == 0)
        return bz;
    Acc a;
    a.from_vector(aie::load_v<64>((const int32_t *)(buf + ((acc_row(oh) * g_mb + j) * 2 + nb) * 256)), 0);
    return a;
}

static inline void acc_save(int8_t *buf, const MM &c, int oh, int j, int nb) {
    aie::store_v((int32_t *)(buf + ((acc_row(oh) * g_mb + j) * 2 + nb) * 256),
                 c.to_accum().template to_vector<int32_t>(0));
}

// An M-block's raw sums (bias included) -> ACC, also on the last chunk: the
// epilogue then runs once per output row (epilogue_row), not per M-block.
static inline void finish(const MM &c0, const MM &c1, int oh, int j) {
    acc_save(g_acc, c0, oh, j, 0);
    acc_save(g_acc, c1, oh, j, 1);
}

// Requant of 32 lanes with one N-block's constants held in registers (the
// arithmetic of req_acc, exact; it only hoists the loads out of the row loop).
struct EpRegs {
    aie::vector<int16_t, 32> ml, mh, p1, p2;
    aie::vector<int32_t, 32> r1, r2;
    int a, B1;
};

static inline EpRegs ep_regs(const EpSet &e, int nb) {
    EpRegs k;
    k.ml = aie::load_v<32>(e.ml[nb]);
    k.mh = aie::load_v<32>(e.mh[nb]);
    k.p1 = aie::load_v<32>(e.p1[nb]);
    k.p2 = aie::load_v<32>(e.p2[nb]);
    k.r1 = aie::load_v<32>(e.r1[nb]);
    k.r2 = aie::load_v<32>(e.r2[nb]);
    k.a = e.a[nb];
    k.B1 = e.B1[nb];
    return k;
}

static inline aie::accum<acc64, 32> req32(const aie::vector<int32_t, 32> &x, const EpRegs &k) {
    const aie::vector<int32_t, 32> t = aie::mul(x, k.ml).template to_vector<int32_t>(16);
    aie::accum<acc64, 32> p;
    p.from_vector(t, 0);
    p = aie::mac(p, x, k.mh);
    const aie::vector<int32_t, 32> q = p.template to_vector<int32_t>(k.a);
    aie::accum<acc64, 32> r;
    r.from_vector(k.r1, 0);
    r = aie::mac(r, q, k.p1);
    const aie::vector<int32_t, 32> q2 = r.template to_vector<int32_t>(k.B1);
    r.from_vector(k.r2, 0);
    return aie::mac(r, q2, k.p2);
}

#ifdef ENG_EPI_NOPIPE /* experiment: epilogue loops not software-pipelined */
#define ENG_EPI_PIPE
#else
#define ENG_EPI_PIPE chess_prepare_for_pipelining chess_loop_range(2, )
#endif

// Plain requant of one row's N-block nb: acc [j][nb][64] int32 -> dst [Wr][8]
// int8 (u8 - 128), 2*nmb half-blocks of 32 lanes.
static inline void req_row_nb(const int8_t *acc, int8_t *dst, const EpSet &e, int nb, int nmb) {
    const EpRegs k = ep_regs(e, nb);
    const int B2 = e.B2[nb];
    for (int i = 0; i < 2 * nmb; i++)
        ENG_EPI_PIPE {
            const aie::vector<int32_t, 32> x =
                aie::load_v<32>((const int32_t *)(acc + ((i >> 1) * 2 + nb) * 256 + (i & 1) * 128));
            aie::store_v(dst + i * 32, u8_to_i8(req32(x, k).template to_vector<uint8_t>(B2)));
        }
}

// Epilogue of output row oh, pass p, from ACC row ar: -> OUT (COMB: + residual).
__attribute__((noinline)) static void epilogue_row_at(int oh, int p, int ar) {
    const int8_t *acc = g_acc + ar * g_mb * 512;
    int8_t *out = g_out + (g_p == 1 ? (oh & 1) : oh) * g_ostride;
    for (int nb = 0; nb < 2; nb++) {
        const int off = (p * 2 + nb) * g_wr * 8;
        if (!(g_flags & ENG_F_COMB)) {
            req_row_nb(acc, out + off, ep_main, nb, g_mb);
            continue;
        }
        const EpRegs k = ep_regs(ep_main, nb);
        const int B2 = ep_main.B2[nb];
        const int8_t *res = g_res + oh * g_res_row + off;
        for (int i = 0; i < 2 * g_mb; i++)
            ENG_EPI_PIPE {
                const aie::vector<int32_t, 32> x =
                    aie::load_v<32>((const int32_t *)(acc + ((i >> 1) * 2 + nb) * 256 + (i & 1) * 128));
                aie::store_v(out + off + i * 32, combine(req32(x, k), B2, res + i * 32));
            }
    }
}

static inline void epilogue_row(int oh, int p) { epilogue_row_at(oh, p, acc_row(oh)); }

// DOWN: the fused 1x1 s2 tap's row (ACCD) -> RES (or the v_d half of OUT).
__attribute__((noinline)) static void epilogue_down_row(int oh, int p) {
    const int8_t *acc = g_accd + acc_row(oh) * g_mb * 512;
    for (int nb = 0; nb < 2; nb++)
        req_row_nb(acc, g_res + oh * g_res_row + (p * 2 + nb) * g_wr * 8, *ep_down, nb, g_mb);
}

// Stride 1: two M-blocks (16 px) at ow0 share every B load.
static void s1_pair(const int8_t *const *rows, const int8_t *bp, int ow0, int oh, int p, int q, const Acc *bz) {
    const int j = ow0 / 8;
    MM cA0(acc_init(g_acc, bz[0], oh, j, 0, q)), cA1(acc_init(g_acc, bz[1], oh, j, 1, q));
    MM cB0(acc_init(g_acc, bz[0], oh, j + 1, 0, q)), cB1(acc_init(g_acc, bz[1], oh, j + 1, 1, q));
    for (int kh = 0; kh < ENG_KH; kh++) {
        for (int kc = 0; kc < ENG_CHUNK_BLK; kc++)
            chess_prepare_for_pipelining {
                const int8_t *seg = rows[kh] + kc * g_wst * 8 + ow0 * 8;
                const aie::vector<int8_t, 64> v0 = LD64(seg), v1 = LD64(seg + 64), v2 = LD64(seg + 128);
                const int8_t *bk = bp + (kh * 3 * ENG_CHUNK_BLK + kc) * ENG_TILE_BYTES;
                for (int kw = 0; kw < 3; kw++)
                    chess_unroll_loop() {
                        const int8_t *b = bk + kw * ENG_CHUNK_BLK * ENG_TILE_BYTES;
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
    finish(cA0, cA1, oh, j);
    finish(cB0, cB1, oh, j + 1);
}

static void s1_single(const int8_t *const *rows, const int8_t *bp, int ow0, int oh, int p, int q, const Acc *bz) {
    const int j = ow0 / 8;
    MM c0(acc_init(g_acc, bz[0], oh, j, 0, q)), c1(acc_init(g_acc, bz[1], oh, j, 1, q));
    for (int kh = 0; kh < ENG_KH; kh++) {
        for (int kc = 0; kc < ENG_CHUNK_BLK; kc++)
            chess_prepare_for_pipelining {
                const int8_t *seg = rows[kh] + kc * g_wst * 8 + ow0 * 8;
                const aie::vector<int8_t, 64> v0 = LD64(seg), v1 = LD64(seg + 64);
                const int8_t *bk = bp + (kh * 3 * ENG_CHUNK_BLK + kc) * ENG_TILE_BYTES;
                for (int kw = 0; kw < 3; kw++)
                    chess_unroll_loop() {
                        const int8_t *b = bk + kw * ENG_CHUNK_BLK * ENG_TILE_BYTES;
                        const aie::vector<int8_t, 64> a = aie::shuffle_down_fill(v0, v1, kw * 8);
                        c0.mac(a, LD64(b));
                        c1.mac(a, LD64(b + 64));
                    }
            }
    }
    finish(c0, c1, oh, j);
}

// Stride 2, one M-block (8 output px <- 17 input px), plus the fused 1x1 s2
// downsample tap (= the centre tap's A tile, its own B and accumulators).
static void s2_single(const int8_t *const *rows, const int8_t *bp, int ow0, int oh, int p, int q, const Acc *bz,
                      const Acc *bzd) {
    const int j = ow0 / 8;
    const bool down = (g_flags & ENG_F_DOWN) != 0;
    MM c0(acc_init(g_acc, bz[0], oh, j, 0, q)), c1(acc_init(g_acc, bz[1], oh, j, 1, q));
    MM d0(acc_init(g_accd, bzd[0], oh, j, 0, q)), d1(acc_init(g_accd, bzd[1], oh, j, 1, q));
    for (int kh = 0; kh < ENG_KH; kh++) {
        for (int kc = 0; kc < ENG_CHUNK_BLK; kc++)
            chess_prepare_for_pipelining {
                const int8_t *seg = rows[kh] + kc * g_wst * 8 + ow0 * 16;
                const aie::vector<int8_t, 64> v0 = LD64(seg), v1 = LD64(seg + 64), v2 = LD64(seg + 128);
                const aie::vector<int8_t, 128> w01 = aie::concat(v0, v1);
                const aie::vector<int8_t, 128> w12 =
                    aie::concat(aie::shuffle_down_fill(v0, v1, 16), aie::shuffle_down_fill(v1, v2, 16));
                const aie::vector<int8_t, 64> a[3] = {aie::filter_even(w01, 8), aie::filter_odd(w01, 8),
                                                      aie::filter_even(w12, 8)};
                const int8_t *bk = bp + (kh * 3 * ENG_CHUNK_BLK + kc) * ENG_TILE_BYTES;
                for (int kw = 0; kw < 3; kw++)
                    chess_unroll_loop() {
                        const int8_t *b = bk + kw * ENG_CHUNK_BLK * ENG_TILE_BYTES;
                        c0.mac(a[kw], LD64(b));
                        c1.mac(a[kw], LD64(b + 64));
                    }
                if (down && kh == 1) {
                    const int8_t *b = bp + (9 * ENG_CHUNK_BLK + kc) * ENG_TILE_BYTES;
                    d0.mac(a[1], LD64(b));
                    d1.mac(a[1], LD64(b + 64));
                }
            }
    }
    finish(c0, c1, oh, j);
    if (down) {
        acc_save(g_accd, d0, oh, j, 0);
        acc_save(g_accd, d1, oh, j, 1);
    }
}

// GRAB: keep this core's channel slice of input row (oh + 1), px 1..Wo, as
// the residual of pass q of the next layer.
static void grab_row(const int8_t *row, int oh, int q, int mc) {
    for (int nb = 0; nb < 2; nb++) {
        const int8_t *src = row + (2 * mc + nb) * g_wst * 8;
        int8_t *dst = g_res + oh * g_res_row + (q * 2 + nb) * g_wr * 8;
        for (int j = 0; j < g_mb; j++)
            aie::store_v(dst + j * 64, aie::shuffle_down_fill(LD64(src + j * 64), LD64(src + j * 64 + 64), 8));
    }
}

static void bias_acc(const EpSet &e, Acc *bz) {
    for (int nb = 0; nb < 2; nb++) {
        aie::accum<acc32, 32> half;
        half.from_vector(aie::load_v<32>(e.bz[nb]), 0);
        bz[nb].insert(0, half);
        bz[nb].insert(1, half);
    }
}

// One (pass p, chunk q): stream the core's rows through the ring once.
static int run_chunk(int p, int q, int g, int mc, const int8_t *wb) {
    Acc bz[2], bzd[2];
    bias_acc(ep_main, bz);
    if (g_flags & ENG_F_DOWN)
        bias_acc(*ep_down, bzd);
    const int8_t *bp = wb + ENG_PRM_BYTES;
    const bool last = q == g_q - 1;
    if (g_resident) {
        if (p == 0)
            LK_ACQ(EF_LK_IN_FULL, 1); // chunk q's rows have all landed
    } else {
        LK_ACQ(EF_LK_IN_FULL, 3 - g_s);
    }
    int cur = g_resident ? q * g_rows_in : g % g_slots; // slot of input row oh * s
    for (int oh = 0; oh < g_rows; oh++) {
        if (!g_resident)
            LK_ACQ(EF_LK_IN_FULL, g_s);
        const int8_t *rows[3];
        for (int kh = 0, sl = cur; kh < 3; kh++, sl = sl + 1 == g_slots ? 0 : sl + 1)
            rows[kh] = g_ring + sl * g_slot_bytes;
        for (int k = 0; k < g_s; k++)
            cur = cur + 1 == g_slots ? 0 : cur + 1;
        if (last && g_p == 1)
            LK_ACQ(EF_LK_OUT_EMPTY, 1);
        if (g_s == 1) {
            int ow0 = 0;
            for (; ow0 + 16 <= g_wr; ow0 += 16)
                s1_pair(rows, bp, ow0, oh, p, q, bz);
            if (ow0 < g_wr)
                s1_single(rows, bp, ow0, oh, p, q, bz);
        } else {
            for (int ow0 = 0; ow0 < g_wr; ow0 += 8)
                s2_single(rows, bp, ow0, oh, p, q, bz, bzd);
        }
        if (last) {
            epilogue_row(oh, p);
            if (g_flags & ENG_F_DOWN)
                epilogue_down_row(oh, p);
        }
        if ((g_flags & ENG_F_GRAB) && p == 0)
            grab_row(rows[1], oh, q, mc);
        chess_memory_fence();
        if (last && p == g_p - 1)
            out_release();
        if (!g_resident)
            LK_REL(EF_LK_IN_EMPTY, g_s);
    }
    if (!g_resident)
        LK_REL(EF_LK_IN_EMPTY, 3 - g_s);
    return g + g_rows_in;
}

#if ENG_HAS_STEM
// ---- stem: 7x7 s2 conv over [230][232][4] + fused 3x3 s2 maxpool ----------
// A tile of (kh, kw-pair kp) for 8 stem px at ow0 is the 64 contiguous bytes
// at input byte 8*(ow0 + kp): stride 2 px x 4 B = 8 B per output pixel.

static inline void stem_store(int8_t *srow, int ow0, const MM &c0, const MM &c1) {
    const Acc acc[2] = {c0.to_accum(), c1.to_accum()};
    for (int nb = 0; nb < 2; nb++)
        for (int h = 0; h < 2; h++)
            aie::store_v(srow + nb * ENG_STEM_W * 8 + ow0 * 8 + h * 32,
                         u8_to_i8(req_acc(acc[nb], ep_main, nb, h).template to_vector<uint8_t>(ep_main.B2[nb])));
}

static void stem_pair(const int8_t *const *rows, const int8_t *bp, int ow0, int8_t *srow, const Acc *bz) {
    MM cA0(bz[0]), cA1(bz[1]), cB0(bz[0]), cB1(bz[1]);
    for (int kh = 0; kh < 7; kh++)
        chess_prepare_for_pipelining {
            const int8_t *seg = rows[kh] + ow0 * 8;
            const aie::vector<int8_t, 64> v0 = LD64(seg), v1 = LD64(seg + 64), v2 = LD64(seg + 128);
            const int8_t *bk = bp + kh * 4 * ENG_TILE_BYTES;
            for (int kp = 0; kp < 4; kp++)
                chess_unroll_loop() {
                    const int8_t *b = bk + kp * ENG_TILE_BYTES;
                    const aie::vector<int8_t, 64> b0 = LD64(b), b1 = LD64(b + 64);
                    const aie::vector<int8_t, 64> aA = aie::shuffle_down_fill(v0, v1, kp * 8);
                    const aie::vector<int8_t, 64> aB = aie::shuffle_down_fill(v1, v2, kp * 8);
                    cA0.mac(aA, b0);
                    cA1.mac(aA, b1);
                    cB0.mac(aB, b0);
                    cB1.mac(aB, b1);
                }
        }
    stem_store(srow, ow0, cA0, cA1);
    stem_store(srow, ow0 + 8, cB0, cB1);
}

// One maxpool output row (56 px) from stem rows a, b, c (int8 = u8 - 128, so
// signed max == unsigned max). Out px j = max over stem px 2j-1 .. 2j+1; px -1
// is the pool's zero padding (u8 0 = int8 -128).
static void maxpool_row(const int8_t *a, const int8_t *b, const int8_t *c, int8_t *out) {
    for (int nb = 0; nb < 2; nb++) {
        const int off = nb * ENG_STEM_W * 8;
        aie::vector<int8_t, 64> prev_odd = aie::broadcast<int8_t, 64>((int8_t)-128);
        for (int j0 = 0; j0 < 7; j0++) {
            const int o = off + j0 * 128;
            const aie::vector<int8_t, 64> v0 = aie::max(aie::max(LD64(a + o), LD64(b + o)), LD64(c + o));
            const aie::vector<int8_t, 64> v1 = aie::max(aie::max(LD64(a + o + 64), LD64(b + o + 64)), LD64(c + o + 64));
            const aie::vector<int8_t, 128> w = aie::concat(v0, v1);
            const aie::vector<int8_t, 64> ev = aie::filter_even(w, 8), od = aie::filter_odd(w, 8);
            const aie::vector<int8_t, 64> om1 = aie::shuffle_up_fill(od, prev_odd, 8);
            aie::store_v(out + nb * 56 * 8 + j0 * 64, aie::max(aie::max(ev, od), om1));
            prev_odd = od;
        }
    }
}

// noinline: a run_* inlined into main stacks its frames onto main's, and the
// overflow lands on ep_main just above the stack (bit-exact +1 errors).
__attribute__((noinline)) static void run_stem(int mr) {
    LK_ACQ(EF_LK_W_FULL, 1);
    const int8_t *wb = ARENA(ENG_W_BASE);
    decode_params(wb, ep_main, g_c0);
    Acc bz[2];
    bias_acc(ep_main, bz);
    const int8_t *bp = wb + ENG_PRM_BYTES;
    const int first = mr == 0 ? 1 : 0; // mesh row 0: local stem row 0 is pool padding
    const int sbytes = 2 * ENG_STEM_W * 8;
    if (first)
        for (int i = 0; i < sbytes; i += 64)
            aie::store_v(g_srow + i, aie::broadcast<int8_t, 64>((int8_t)-128));
    LK_ACQ(EF_LK_IN_FULL, 5);
    int cur = 0;
    for (int s = first; s <= 28; s++) {
        LK_ACQ(EF_LK_IN_FULL, 2);
        const int8_t *rows[7];
        for (int kh = 0, sl = cur; kh < 7; kh++, sl = sl + 1 == g_slots ? 0 : sl + 1)
            rows[kh] = g_ring + sl * ENG_STEM_SLOT;
        int8_t *srow = g_srow + (s % 3) * sbytes;
        for (int ow0 = 0; ow0 < ENG_STEM_W; ow0 += 16)
            stem_pair(rows, bp, ow0, srow, bz);
        for (int k = 0; k < 2; k++)
            cur = cur + 1 == g_slots ? 0 : cur + 1;
        chess_memory_fence();
        LK_REL(EF_LK_IN_EMPTY, 2);
        if (s >= 2 && (s & 1) == 0) {
            const int m = s / 2 - 1;
            LK_ACQ(EF_LK_OUT_EMPTY, 1);
            maxpool_row(g_srow + ((s - 2) % 3) * sbytes, g_srow + ((s - 1) % 3) * sbytes, srow,
                        g_out + (m & 1) * g_oslot_bytes);
            chess_memory_fence();
            out_release();
        }
    }
    LK_REL(EF_LK_IN_EMPTY, 5);
    LK_REL(EF_LK_W_EMPTY, 1);
}

#endif // ENG_HAS_STEM

#if ENG_HAS_LM
// RESX: the residual tensor's rows (own output rows, chunk by chunk) arrive
// before the conv's input; keep this core's channel slice of each -> RES.
// Returns the ring cursor the conv's input starts at.
__attribute__((noinline)) static int grab_residual(int mc) {
    int g = 0;
    for (int qr = 0; qr < g_p; qr++)
        for (int oh = 0; oh < g_rows; oh++, g++) {
            LK_ACQ(EF_LK_IN_FULL, 1);
            const int8_t *row = g_resident ? g_out + (g & 1) * g_slot_bytes : g_ring + (g % g_slots) * g_slot_bytes;
            // the residual row is the centre row: grab_row's rows[1]
            grab_row(row, oh, qr, mc);
            chess_memory_fence();
            LK_REL(EF_LK_IN_EMPTY, 1);
        }
    return g_resident ? 0 : g;
}

#endif // ENG_HAS_LM

#if ENG_HAS_LM
// ENG_F_RI (stride 1, one M-block per row): q-outer. Input chunk q's rows_in
// rows land once and serve all P passes; pass p's partial sums live in ACC row
// p * rows + oh; the last chunk runs each row's epilogue. A row's packet holds
// all P passes, so the rows go out at the end.
__attribute__((noinline)) static void run_layer_ri(int g) {
    int w = 0;
    Acc bz[2];
    for (int q = 0; q < g_q; q++) {
        LK_ACQ(EF_LK_IN_FULL, g_rows_in);
        const int base = g % g_slots;
        for (int p = 0; p < g_p; p++, w++) {
            LK_ACQ(EF_LK_W_FULL, 1);
            const int8_t *wb = ARENA(ENG_W_BASE) + (w & 1) * g_chunk;
            if (q == 0 || q == g_q - 1) { // bias (first chunk), requant constants (last)
                decode_params(wb, ep_main, g_c0);
                bias_acc(ep_main, bz);
            }
            const int8_t *bp = wb + ENG_PRM_BYTES;
            for (int oh = 0; oh < g_rows; oh++) {
                const int8_t *rows[3];
                for (int kh = 0; kh < 3; kh++)
                    rows[kh] = g_ring + ((base + oh + kh) % g_slots) * g_slot_bytes;
                s1_single(rows, bp, 0, p * g_rows + oh, p, q, bz);
                if (q == g_q - 1)
                    epilogue_row_at(oh, p, p * g_rows + oh);
            }
            chess_memory_fence();
            LK_REL(EF_LK_W_EMPTY, 1);
        }
        LK_REL(EF_LK_IN_EMPTY, g_rows_in);
        g += g_rows_in;
    }
    for (int oh = 0; oh < g_rows; oh++)
        out_release();
}
#endif

__attribute__((noinline)) static void run_layer(int mc, int g) {
    int w = 0;
    for (int p = 0; p < g_p; p++)
        for (int q = 0; q < g_q; q++, w++) {
            LK_ACQ(EF_LK_W_FULL, 1);
            const int8_t *wb = ARENA(ENG_W_BASE) + (w & 1) * g_chunk;
            if (q == 0) {
                decode_params(wb, ep_main, g_c0);
                if (g_flags & ENG_F_DOWN)
                    decode_params(wb + ENG_PRM_DOWN, *ep_down, g_c0d);
            }
            g = run_chunk(p, q, g, mc, wb);
            chess_memory_fence();
            LK_REL(EF_LK_W_EMPTY, 1);
        }
}

// kernel_decl eng
int32_t main() {
    AieKcReg kc[8];
    lock_set(LK_HDR_EMPTY, 1);
    lock_set(LK_HDR_FULL, 0);
    lock_set(EF_LK_W_EMPTY, 2);
    lock_set(EF_LK_W_FULL, 0);
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::saturate);
    const unsigned cid = get_coreid();
    const int mc = (int)(cid >> 16) - EF_PART_START_COL - EF_FIRST_COL;
    const int mr = (int)(cid & 0x1F) - EF_FIRST_ROW;
    const int pkt = k_pkt[mr][mc], ooo = k_ooo[mr][mc];
    core_reg_write_block(kc, aie_kc_encode_bd(kc, BD_HDR, (uintptr_t)eng_hdr, ENG_HDR_BYTES, -1, LK_HDR_EMPTY, -1,
                                              LK_HDR_FULL, 1, 0, 0, 0, -1));
    aie_kc_encode_s2mm_start(kc, 1, BD_HDR, 1, 0);
    core_reg_write(kc[0].off, kc[0].val);
    for (;;) {
        LK_ACQ(LK_HDR_FULL, 1);
        read_header(mr);
        arm_layer(pkt, ooo);
        LK_REL(LK_HDR_EMPTY, 1);
        if (g_stem) {
#if ENG_HAS_STEM
            run_stem(mr);
#endif
        } else {
#if ENG_HAS_CONV
#if ENG_HAS_LM
            const int g0 = (g_flags & ENG_F_RESX) ? grab_residual(mc) : 0;
            if (g_ri)
                run_layer_ri(g0);
            else
                run_layer(mc, g0);
#else
            run_layer(mc, 0);
#endif
#endif
        }
        LK_ACQ(EF_LK_OUT_EMPTY, g_oslots); // every packet sent: BDs/locks may be reused
    }
    return 0;
}
