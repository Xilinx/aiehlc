/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * conv2dstem_new — the ResNet-18 stem convolution (conv1) as an AIE static
 * library, written in the KERNEL-ONLY block-load model.
 *
 *   ifm [224,224,3] (*) wts [64,7,7,3]  stride 2, pad 3  ->  ofm [112,112,64]
 *
 * Same library as conv2dstem.cc -- same public API (conv2dstem.h), same host
 * staging / weight packing / tile-major output collection, same bit-exact
 * vector compute (aie::mmul<8,8,8,int8,int8> + the two-stage vector
 * requantize) and the same control-plane / KCO / output_tile_major pragmas.
 * What changed is how the data movement is described:
 *
 *   conv2dstem.cc      three hand-written GemmSpace descriptors (RowBC_spatial,
 *                      ColBC, LtoR_Merge) + ~25 mirrored #defines (SP_*, OH_T,
 *                      TILE_*, ...); the core walks windows with
 *                      acquire_input_window / aie::get_num_rounds.
 *   conv2dstem_new.cc  no descriptors, no mirrored tiling macros. The kernel's
 *                      ports declare their tensors (aie::shape), and the kernel
 *                      says which BLOCK of each it reads / writes per round:
 *                        aie::load_buffer(win_a, {oh0*2, ow0*2, 0}, {19, 61, 4})
 *                        aie::store_buffer(win_c, {oh0, ow0, pid_c*16}, {7, 28, 16})
 *                      with aie::program_id / release / commit. aiehlc traces
 *                      that block schedule on x86 for every core (kernel
 *                      analysis), recognizes the spatial-halo conv, derives the
 *                      DMA with convautotile and checks every block against it
 *                      (doc/design/op_mapping_programming_model.md sections 5, 7).
 *
 * One consequence: the input window is exactly the conv footprint, 61 px
 * (244-byte rows), instead of conv2dstem.cc's 64-px / 256-byte window. A-tiles
 * are therefore unaligned 64-byte loads, and the last K-block of each kernel
 * row starts 4 bytes early (its first 4 B rows are zero) so no load runs past
 * a slab row or the block. The staging rows are 230 px (the tensor), not 232.
 *
 * Build:   source script/aiehlc.sh --aie-version 5 --runtime-source-file \
 *              src/aietensorop/conv2dstem/conv2dstem_uk_new.cc
 * Check:   build/aiehlc-convtile trace src/aietensorop/conv2dstem/conv2dstem_new.cc
 ******************************************************************************/
// NOTE: conv2dstem.h is deliberately NOT included here — it is included BELOW,
// after the kernel definition (see the "Header include placement" comment).
// The aiehlc source rewriter scans this file as TEXT for the kernel annotation
// keyword and the launch token; neither is spelled in any comment here.
#include "unistd.h"
#include "xil_cache.h"
#include "xiltimer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DMA_ISSUE_COUNT)

// ═══════════════════════════════════════════════════════════════════════════
// Control plane — move host->array configuration off the config bus
//
// Four opt-ins, each a bare marker pragma that aiehlc publishes as a module attr
// gating one MLIR pipeline stage. They are independent features but they compose
// into one story: a 32-bit XAie_Write32 is a non-posted Device-nGnRnE NoC round
// trip at ~372 ns (doc/performance/register_write_cost.md), and an N-word BD is
// N serial writes, so on this layer the host's register traffic is a first-class
// cost term. Every pragma below replaces some of it.
//
//   control_plan_op_control_packet   THE plan. Reserves control-plane
//       stream-switch resources (slots, arbiters, msel, pkt-ids) so the fabric
//       can share the array with the data plane, and lowers KERNEL ELF LOADING
//       plus core launch onto it -- load_kernel_group_16t_ctrl streams the ELF's
//       PT_LOAD bytes as 4-word broadcast writes and launch_kernel_group_ctrl
//       enables the cores through the same path, instead of per-tile
//       XAie_LoadElfMem / XAie_CoreEnable. Every other pragma here requires it.
//   CONTROL_PLAN_GROUP_REG_WRITE     Runs GroupRegWritePass, which folds the
//       per-tile lock inits -- identical across all 16 tiles of this mesh --
//       into broadcast/row group writes (__Runtime_ctrl_row_write_ack) instead
//       of one XAie_LockSetValue per tile per lock.
//   KERNELCONFIGOFFLOAD              Core self-configures ALL of its own
//       core-tile DMA (BD chains, lock init/config, channel starts, both
//       directions) from kernel.cc over the processor bus, so the host stops
//       programming it over the config bus entirely. Gen5 only -- this app is.
//   control_plan_shim_bd_ctrl        The remaining host-side writes are the SHIM
//       BDs; this sends those, the out-of-order channel enable and the
//       start-queue writes along row 0 as control packets instead of MMIO.
//
// control_plan_dedicated_shim is a FORCE flag, not a feature: shim_bd_ctrl needs
// a spare column west of the mesh for the control spine, and on its own it only
// WARNS and silently falls back to MMIO when there isn't one. With this pragma
// the build fails instead. That matters here because the spare column is exactly
// what the partition change below buys -- see the launch sites.
//
// Verify in the build log, not by assumption:
//   [aiehlc] Detected #pragma ...                      (x5, all parsed)
//   [TilingLinalg] control plane: dedicated shim col 0
//       (MM2S ch0 / S2MM ch0; shim-row BD control on MM2S ch1 / S2MM ch1);
//       data plane uses cols [1,4]                     (spare column found)
// A "WARNING: #pragma control_plan_shim_bd_ctrl needs ..." line means the shim
// BDs stayed on MMIO. Skills: shim-row-ctrl-bd-config, ctrlpkt-aot-elf.
// ═══════════════════════════════════════════════════════════════════════════
#pragma control_plan_op_control_packet
#pragma CONTROL_PLAN_GROUP_REG_WRITE
#pragma control_plan_dedicated_shim
#pragma control_plan_shim_bd_ctrl
// Precompute every control-packet payload at BUILD time (skill ctrlpkt-aot-elf):
// the kernel-ELF body, the plan's return routes, core enable, lock-init group
// writes and the whole shim-BD window, so a launch memcpy's + patches addresses
// instead of encoding through aie-rt. The ELF encode and the shim-BD window were
// the host time the cores spent waiting for their first filter bytes.
#pragma control_packet_mode(aot)
// KERNELCONFIGOFFLOAD on {2,6} needed two compiler fixes, or the mesh deadlocks
// (wait_io TIMEOUT; doc/debug/troubleshoot_aie_offload_timeout.md): the get_coreid()
// dispatch rebased by the partition start column, and MM2S BD length taken from the
// host plan (3136 B), not sizeof(buf_out_ping_0) (784 B). Skill: kcocoreidpartition.
#pragma KERNELCONFIGOFFLOAD

// Output tile-major. The array writes g_ofm as [mesh row][mesh col][round]
// [7 x 28 x 16] -- each core's 16 rounds back to back -- instead of scattered
// into the [112,112,64] image. That makes each core's output one linear run, so
// the shim S2MM needs ONE BD per core for the whole launch. The dense layout
// needed a 2-D round walk the BD cannot express, so the compiler emitted a host
// loop re-programming the BDs between height slabs; once the kernel got fast
// enough, cores emitted the next slab into BDs still being re-programmed and
// whole tiles landed at stale addresses (GOLDEN FAIL, timing-dependent). The
// collect_ofm_* functions below own the layout conversion; it is a gather they
// were already doing. Index map: tile_major_off().
#pragma output_tile_major

// Runtime-side opt-ins for the above. Declared, not included: aie_runtime.h
// pulls in the XAie driver headers, which this file does not otherwise need and
// which must not reach the kernel translation unit. Same C++-linkage rule as the
// rest of the runtime -- plain declarations, NOT extern "C".
//
// high_throughput_enable(1) is REQUIRED, not a tuning knob: on AIE2PS it
// disables CTRL TLAST errors at partition init and packs all self-delimiting
// accesses into one shim MM2S BD with a single final TLAST. control_plan_shim_bd_ctrl
// does not work without it, and the ELF load falls back to one BD and one TLAST
// per access. It must run BEFORE device init, so both launch sites call it ahead
// of their aieSetDevice/partition pair; it only sets a flag, so repeating is free.
//
// pmap_enable(0) pins the control-packet provenance map OFF. It defaults to
// "resolve from the AIE_CTRL_PMAP env var", so without this an exported
// AIE_CTRL_PMAP in the operator's shell would start printing a line per packet
// and quietly wreck test_conv2d.cc's PERF numbers.
void __Runtime_ctrl_high_throughput_enable(int on);
void __Runtime_ctrl_pmap_enable(int on);
// g_ofm is written only by the array and only READ by the CPU after a launch
// (collect_ofm_*), so its per-launch cache maintenance can be invalidate-only and
// overlap the launch instead of a full clean+invalidate of 802 KB in front of it.
// That is a contract, so it is declared, not inferred. Same linkage rule as above.
void __Runtime_buffer_device_written(void *ptr, size_t size);
void __Runtime_buffer_forget(void *ptr);
// g_ifm_pad is the opposite case: the CPU rewrites it before every launch. Mapped
// non-cacheable, those stores reach DDR as they are made, so the launch does not
// spend ~25 us cleaning 213 KB of dirty lines while the cores wait for data.
void *__Runtime_alloc_uncached(size_t bytes);
void __Runtime_free_uncached(void *ptr);

// Turn the control-plane runtime opt-ins on. Called at the top of every launch
// site, before the device exists.
static inline void conv2dstem_control_plane_init(void) {
    __Runtime_ctrl_high_throughput_enable(1);
    __Runtime_ctrl_pmap_enable(0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometry (host + kernel). Bare literals, cross-checked against conv2dstem.h
// by static_assert after the include site. NOT defined here on purpose: the
// spatial-halo tiling macros (SP_*, OH_T, OW_T, TILE_*, PAD_H/W, OUTPUT_FULL_*)
// -- aiehlc derives them from the kernel and emits them itself.
// ═══════════════════════════════════════════════════════════════════════════
#define INPUT_H 224
#define INPUT_W 224
#define INPUT_C 3       // real channels
#define INPUT_C_ALIGN 4 // channel-layout stride (3 + 1 pad channel carrying the quant params in the filter)
#define KERNEL_H 7
#define KERNEL_W 7
#define NUM_FILTERS 64
#define STRIDE 2
#define PAD 3

#define OUTPUT_H 112 // (224 + 2*3 - 7)/2 + 1
#define OUTPUT_W 112

// Spatially pre-padded DDR extents: the IFM tensor the kernel's port declares.
#define INPUT_H_PAD (INPUT_H + 2 * PAD) // 230
#define INPUT_W_PAD (INPUT_W + 2 * PAD) // 230
// Row pitch of the staging buffer g_ifm_pad, in pixels: the tensor's own width.
// The derived DMA reads 61-px windows of the [230,230,4] tensor (conv2dstem.cc
// used 232 to give its 64-px windows room).
#define IFM_W_PITCH INPUT_W_PAD

// im2col -> GEMM dims (launch scalars). K uses the ALIGNED channel count.
#define M (OUTPUT_H * OUTPUT_W)                 // 12544
#define K (KERNEL_H * KERNEL_W * INPUT_C_ALIGN) // 196
#define N NUM_FILTERS                           // 64

// HW mesh
#define HW_ROWS 4
#define HW_COLS 4

// Host-side view of the tiling the kernel uses (the output tile per round and
// the channels per mesh column) -- for the tile-major output walk. The kernel
// states the same numbers as its own constants; static_asserts below pin them.
#define CS_OH_T 7                               // output rows per round
#define CS_OW_T 28                              // output cols per round
constexpr int OC_PER_G = NUM_FILTERS / HW_COLS; // 16 output channels per mesh col

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: conv2d_spatial — VECTOR version (aie::mmul<8,8,8,int8,int8>),
// kernel-only block-load form
//
// One round = one slab -> one 7x28x16 output tile. GEMM view:
//   M = 7 oh x 28 ow  -> per oh row: M-blocks of 8 consecutive ow starting at
//                        ow0 = 0, 8, 16, 20. The last block overlaps the third
//                        by 4 columns and rewrites them with identical values,
//                        so 28 columns cost 32 instead of the 56 a 4-row block
//                        would (mmul<4,8,8> lowers to the SAME 8x8x8 instruction
//                        on AIE2P/AIE2PS with A grown to 8 rows -- half wasted).
//   N = 16 channels   -> 2 N-blocks of 8
//   K = 7 kh x 32     -> per kh: the 7 taps x 4 ch of one output are 28
//                        CONTIGUOUS slab bytes, padded to 32 = 4 K-blocks of 8
// The A-tile (8 rows x 8 K) for M-block ow0, kernel row kh, K-block kb is
//     row m, col k  =  slab[(oh*2+kh)*244 + (ow0+m)*8 + kb*8 + k]
// i.e. 64 CONTIGUOUS bytes starting at (oh*2+kh)*PITCH + (ow0+kb)*8: stride 2 x
// 4 channels = 8 bytes per output column = exactly one K-block. So every A-tile
// is a single (unaligned: 244-byte rows) vector load, and im2col costs nothing.
//
// K-block 3 starts 4 bytes EARLY (bytes 20..27 of the column's 28) so its last
// load ends exactly at the 244-byte row end -- no load leaves its row or the
// block, which the kernel analysis checks under ASan. Its B tile holds zeros in
// rows 0..3 (taps already counted by block 2) and the half tile in rows 4..7.
// The pad channel (k-index % 4 == 3) carries the quant params in the window and
// is masked to zero in Bp, so the MAC loop needs no c < 3 test.
//
// ── Filter window layout (packed by the HOST, pack_wts_tiles) ──────────────
// The window is the same 3136 bytes per mesh column as before, but already in
// mmul order, so the core only copies + masks (it used to transpose 3584 bytes
// with scalar loads: 67k cycles, 17% of the kernel). Per kh, 448 bytes:
//     kb 0..2, nb 0..1 : 8(K) x 8(N) tile at kh*448 + (kb*2+nb)*64
//     kb 3,    nb 0..1 : 4(K) x 8(N) half tile at kh*448 + 384 + nb*32
// Quant params ride in pad-channel rows (k % 4 == 3), see VK_PRM_OFF.
//
// ── Bit-exact vector requantize with NATIVE 32x16 multiplies ───────────────
// Reference (per channel):  v = (x*mult + 2^(s-1)) >> s,  s = shift+31, 64-bit.
// mult is split mult = mh*2^16 + ml with BOTH halves in int16 (ml in
// [-2^15, 2^15), the host refuses an mh that would not fit), so no product
// needs the emulated 32x32 multiply the old epilogue spent ~2 cycles/output on:
//   t  = floor(x*ml / 2^16)                        (SRS 16, fits int32)
//   v  = floor((x*mh + t + 2^(S-1)) / 2^S),   S = s - 16
// exact because floor((n*2^16 + y)/2^s) == floor((n + floor(y/2^16))/2^(s-16))
// for integer n, and 2^(s-1) is a multiple of 2^16 (s >= 17). S still differs
// per channel and an SRS shift is one scalar per vector, so the last division
// is split, exactly, into floor shifts per N-block:
//   a  = min_n(S_n) - 1                   (every lane has b_n = S_n - a >= 1)
//   q  = floor((x*mh + t) / 2^a)          (SRS, floor rounding, fits int32)
// then the per-lane rest b_n = b1_n + b2_n, b1_n = min(b_n, 14), in two stages
// that each scale by an int16 power of two and share one SRS:
//   q2 = floor((q *2^(B1-b1_n) + R1_n) / 2^B1)       B1 = max_n b1_n
//   v  = floor((q2*2^(B2-b2_n) + R2_n) / 2^B2)       B2 = max_n b2_n
// with the rounding 2^(b_n-1) added in whichever stage finishes the division:
// R2_n = 2^(B2-1) when b2_n >= 1 (R1_n = 0), else R1_n = 2^(B1-1) (R2_n = 0).
// floor((floor(y)+k)/m) == floor((y+k)/m) for integer k, m>0, and scaling the
// numerator and denominator by a power of two changes nothing. One stage would
// need 2^(B-b_n) up to 2^19 on the real layer (shifts 8..27 in one N-block),
// which does not fit int16. Then clamp [0,255] == ReLU + uint8.
// ═══════════════════════════════════════════════════════════════════════════
// CONV2DSTEM_KCYC 1: every core times itself with its cycle counter and writes
// {0xC0DE0001, total, compute, out-wait} {0xC0DE0002, B-prep, in-wait, copy}
// over the first 32 bytes of its LAST output tile; test_conv2d.cc scans g_ofm
// for the magic and prints one KCYC line per core. The golden check then fails
// on exactly those bytes, so this is a measurement build, never a shipping one.
// Edit the literal: the kernel extractor copies only UNCONDITIONAL #defines into
// the kernel TU (one inside #ifndef/#if is silently dropped), so this cannot be
// an overridable default, and the kernel tests it with `if`, not `#if`.
#define CONV2DSTEM_KCYC 0
#define VK_TIC() (CONV2DSTEM_KCYC ? (uint64_t)aie::tile::current().cycles() : (uint64_t)0)
#define VK_PITCH 244    // input slab row pitch: 61 px * 4 B (the derived window)
#define VK_KB 4         // K-blocks of 8 per kernel row (28 bytes -> 32)
#define VK_MB 8         // mmul M (output columns per M-block)
#define VK_NB 2         // N-blocks of 8 channels (16 channels per core)
#define VK_KH_BYTES 448 // packed filter bytes per kernel row (3 full + 1 half tile per nb)
#define VK_MMUL aie::mmul<VK_MB, 8, 8, int8_t, int8_t>
// An A-tile: 64 bytes from a slab row at a 4-byte aligned offset (244-byte rows,
// and DMA windows are only 32-byte aligned).
#define VK_LDA(p) aie::load_unaligned_v<64>((p), 4)
// Requantize one 32-lane half (rows 4h..4h+3 of an 8x8 C tile) of N-block nb.
// Straight-line on purpose -- no `if (B2 > 0)`: with B2 == 0 the second scaling
// stage is an exact identity (p2 = 1, R2 = 0, shift 0), and a branch here split
// the epilogue's basic block so the scheduler could not interleave the eight
// independent chains of a pair. See the kernel header for the math.
#define VK_REQ_HALF(cacc, nb, h)                                                                                       \
    ([&]() {                                                                                                           \
        const aie::vector<int32_t, 32> x_ = (cacc).template extract<32>(h).template to_vector<int32_t>(0);             \
        const aie::vector<int32_t, 32> t_ = aie::mul(x_, aie::load_v<32>(ep_ml[nb])).template to_vector<int32_t>(16);  \
        aie::accum<acc64, 32> p_;                                                                                      \
        p_.from_vector(t_, 0);                                                                                         \
        p_ = aie::mac(p_, x_, aie::load_v<32>(ep_mh[nb]));                                                             \
        const aie::vector<int32_t, 32> q_ = p_.template to_vector<int32_t>(ep_a[nb]);                                  \
        aie::accum<acc64, 32> r_;                                                                                      \
        r_.from_vector(aie::load_v<32>(ep_r1[nb]), 0);                                                                 \
        r_ = aie::mac(r_, q_, aie::load_v<32>(ep_p1[nb]));                                                             \
        const aie::vector<int32_t, 32> q2_ = r_.template to_vector<int32_t>(ep_B1[nb]);                                \
        r_.from_vector(aie::load_v<32>(ep_r2[nb]), 0);                                                                 \
        r_ = aie::mac(r_, q2_, aie::load_v<32>(ep_p2[nb]));                                                            \
        return r_.template to_vector<uint8_t>(ep_B2[nb]);                                                              \
    }())
// Epilogue of one M-block pair: blocks A (cA0/cA1, columns owA..) and B
// (cB0/cB1, columns owB..), each 8 columns x [ch 0..7 | ch 8..15] -> 128
// contiguous HWC bytes in the output window.
#define VK_EPILOGUE(cA0, cA1, cB0, cB1, oh, owA, owB)                                                                  \
    do {                                                                                                               \
        const aie::accum<acc32, 64> aA0_ = (cA0).to_accum(), aA1_ = (cA1).to_accum();                                  \
        const aie::accum<acc32, 64> aB0_ = (cB0).to_accum(), aB1_ = (cB1).to_accum();                                  \
        const aie::vector<uint8_t, 32> oA00_ = VK_REQ_HALF(aA0_, 0, 0), oA01_ = VK_REQ_HALF(aA0_, 0, 1);               \
        const aie::vector<uint8_t, 32> oA10_ = VK_REQ_HALF(aA1_, 1, 0), oA11_ = VK_REQ_HALF(aA1_, 1, 1);               \
        const aie::vector<uint8_t, 32> oB00_ = VK_REQ_HALF(aB0_, 0, 0), oB01_ = VK_REQ_HALF(aB0_, 0, 1);               \
        const aie::vector<uint8_t, 32> oB10_ = VK_REQ_HALF(aB1_, 1, 0), oB11_ = VK_REQ_HALF(aB1_, 1, 1);               \
        uint8_t *dA_ = (uint8_t *)&out[((oh) * ow_dim + (owA)) * out_c_num];                                           \
        uint8_t *dB_ = (uint8_t *)&out[((oh) * ow_dim + (owB)) * out_c_num];                                           \
        const auto zA0_ = aie::interleave_zip(oA00_, oA10_, 8), zA1_ = aie::interleave_zip(oA01_, oA11_, 8);           \
        const auto zB0_ = aie::interleave_zip(oB00_, oB10_, 8), zB1_ = aie::interleave_zip(oB01_, oB11_, 8);           \
        aie::store_unaligned_v(dA_, zA0_.first, 16);                                                                   \
        aie::store_unaligned_v(dA_ + 32, zA0_.second, 16);                                                             \
        aie::store_unaligned_v(dA_ + 64, zA1_.first, 16);                                                              \
        aie::store_unaligned_v(dA_ + 96, zA1_.second, 16);                                                             \
        aie::store_unaligned_v(dB_, zB0_.first, 16);                                                                   \
        aie::store_unaligned_v(dB_ + 32, zB0_.second, 16);                                                             \
        aie::store_unaligned_v(dB_ + 64, zB1_.first, 16);                                                              \
        aie::store_unaligned_v(dB_ + 96, zB1_.second, 16);                                                             \
    } while (0)
// Byte offset, inside one mesh column's filter window, of quant-param byte row
// j (0..11; param p = j/4 in {bias-zp, mult, shift}, little-endian byte j%4)
// for channel n of N-block nb. Row j lives in kernel row j/7 at pad-channel
// row k = 3 or 7 of tile kb (slot j%7 < 6), or row 3 of the half tile (slot 6).
#define VK_PRM_OFF(j, nb, n)                                                                                           \
    ((((j) % 7) < 6)                                                                                                   \
         ? ((j) / 7) * VK_KH_BYTES + ((((j) % 7) / 2) * 2 + (nb)) * 64 + ((((j) % 7) & 1) ? 7 : 3) * 8 + (n)           \
         : ((j) / 7) * VK_KH_BYTES + 384 + (nb) * 32 + 3 * 8 + (n))

__global__ void conv2d_spatial(
    aie::port<input_window_int8 *, aie::IFM, aie::shape<INPUT_H_PAD, INPUT_W_PAD, INPUT_C_ALIGN>> win_a, // [230,230,4]
    aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<NUM_FILTERS, KERNEL_H, KERNEL_W, INPUT_C_ALIGN>>
        win_b, // [64,7,7,4]: per mesh col one 3136-byte block, packed in mmul order by the host
    aie::port<output_window_int8 *, aie::OFM, aie::shape<OUTPUT_H, OUTPUT_W, NUM_FILTERS>> win_c // [112,112,64]
) {
    // The operator and the tiling live in the kernel.
    const int KH = 7, KW = 7, S = 2, C = 4;                           // 7x7 stride-2 conv, 4 (aligned) channels
    const int oh_t = 7, ow_t = 28, oc_t = 16;                         // output tile per round
    const int ih_t = (oh_t - 1) * S + KH, iw_t = (ow_t - 1) * S + KW; // 19 x 61 input slab (halo)
    const int oh_core = OUTPUT_H / aie::num_programs(0);              // 28 output rows per mesh row
    const int pid_h = aie::program_id(0);                             // mesh row: which OH band
    const int pid_c = aie::program_id(1);                             // mesh col: which OC group
    const int ow_dim = ow_t, out_c_num = oc_t;                        // VK_EPILOGUE's names

    // floor rounding for the exact two-stage requantize; saturation makes the
    // final narrowing a clamp rather than a wrap.
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::saturate);

    // Packed B: [kh][kb][nb] tiles of 8(K) x 8(N), row-major, zero where the
    // K-lane is already counted / past tap 6 or is the pad/param channel.
    alignas(64) int8_t Bp[KH * VK_KB * VK_NB * 64];
    // Per-lane epilogue vectors for one 32-lane half of the 8x8 C tile
    // (lane = m*8 + n, m = 0..3): they depend on n only.
    alignas(64) int32_t ep_bz[VK_NB][32]; // bias - zero_point
    alignas(64) int16_t ep_mh[VK_NB][32]; // mult high half (signed)
    alignas(64) int16_t ep_ml[VK_NB][32]; // mult low half (signed)
    alignas(64) int16_t ep_p1[VK_NB][32]; // 2^(B1 - b1_n)
    alignas(64) int32_t ep_r1[VK_NB][32]; // R1_n
    alignas(64) int16_t ep_p2[VK_NB][32]; // 2^(B2 - b2_n)
    alignas(64) int32_t ep_r2[VK_NB][32]; // R2_n
    int ep_a[VK_NB];
    int ep_B1[VK_NB];
    int ep_B2[VK_NB];

    const uint64_t kc0 = VK_TIC();
    uint64_t kc_t = 0;
    uint32_t kc_bprep = 0, kc_acqa = 0, kc_copy = 0, kc_acqc = 0, kc_comp = 0;
    (void)kc_t;

    // ===== B (filter + params): this core's 16 filters, once per launch =====
    {
        kc_t = VK_TIC();
        const int8_t *Bw = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C});
        // Zero the pad-channel rows (k = 3, 7) -- they carry the params.
        alignas(64) int8_t mask_b[64];
        for (int i = 0; i < 64; i++)
            mask_b[i] = ((i / 8) % 4 == 3) ? (int8_t)0 : (int8_t)-1;
        const aie::vector<int8_t, 64> mask = aie::load_v<64>(mask_b);
        const aie::vector<int8_t, 32> zero32 = aie::zeros<int8_t, 32>();
        for (int kh = 0; kh < KH; kh++)
            chess_prepare_for_pipelining {
                const int8_t *src = Bw + kh * VK_KH_BYTES;
                int8_t *dst = &Bp[kh * VK_KB * VK_NB * 64];
                for (int t = 0; t < 6; t++)
                    aie::store_v(dst + t * 64, aie::bit_and(aie::load_unaligned_v<64>(src + t * 64, 32), mask));
                // K-block 3 is loaded 4 bytes early: the half tile (taps 6) goes to rows 4..7.
                for (int nb = 0; nb < VK_NB; nb++) {
                    const aie::vector<int8_t, 32> lo =
                        aie::bit_and(aie::load_v<32>(src + 384 + nb * 32), mask.template extract<32>(0));
                    aie::store_v(dst + (6 + nb) * 64, aie::concat(zero32, lo));
                }
            }
        for (int nb = 0; nb < VK_NB; nb++) {
            // Gather the 12 param byte rows (VK_PRM_OFF, walked without the
            // division/modulo the macro spells -- the core has no divider, and
            // the macro form cost ~48k cycles here) into prm[n][0..2].
            int32_t prm[8][3];
            for (int n = 0; n < 8; n++)
                prm[n][0] = prm[n][1] = prm[n][2] = 0;
            for (int j = 0; j < 12; j++) {
                const int kh = j < 7 ? 0 : 1, slot = j - kh * 7;
                const int off = kh * VK_KH_BYTES + (slot < 6 ? ((slot >> 1) * 2 + nb) * 64 + ((slot & 1) ? 7 : 3) * 8
                                                             : 384 + nb * 32 + 3 * 8);
                for (int n = 0; n < 8; n++)
                    prm[n][j >> 2] |= (int32_t)((uint32_t)(uint8_t)Bw[off + n] << ((j & 3) * 8));
            }
            int S[8];
            int smin = 1 << 30;
            for (int n = 0; n < 8; n++) {
                S[n] = prm[n][2] + 31 - 16;
                if (S[n] < smin)
                    smin = S[n];
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
                const int32_t bz = prm[n][0];
                const int32_t mult = prm[n][1];
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
        aie::release(win_b);
    }
    kc_bprep = (uint32_t)(VK_TIC() - kc0);

    // Accumulator seeds: bias - zero_point per lane of the 8x8 C tile (lane =
    // m*8 + n, so the 32-lane ep_bz row repeats for rows 0-3 and 4-7). Seeding
    // the MAC with it saves an add per output in the epilogue.
    aie::accum<acc32, 64> bzacc[VK_NB];
    for (int nb = 0; nb < VK_NB; nb++) {
        aie::accum<acc32, 32> half;
        half.from_vector(aie::load_v<32>(ep_bz[nb]), 0);
        bzacc[nb].insert(0, half);
        bzacc[nb].insert(1, half);
    }

    for (int i = 0; i < oh_core / oh_t; i++) {      // 4 row blocks
        for (int j = 0; j < OUTPUT_W / ow_t; j++) { // 4 column blocks
            const int oh0 = pid_h * oh_core + i * oh_t, ow0 = j * ow_t;
            kc_t = VK_TIC();
            // Its input footprint (the halo overlaps the neighbouring blocks).
            const int8_t *slab = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});
            kc_acqa += (uint32_t)(VK_TIC() - kc_t);
            kc_t = VK_TIC();
            int8_t *out = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t});
            kc_acqc += (uint32_t)(VK_TIC() - kc_t);
            kc_t = VK_TIC();
            for (int oh = 0; oh < oh_t; oh++) {
                // Two M-blocks per pass so every B tile load feeds two MACs:
                // pass 0 = ow0 {0, 8}, pass 1 = ow0 {16, 20} (20..23 written
                // twice, identically). Written out separately on purpose: a
                // select inside the kh loop kept it from pipelining.
                const int8_t *row0 = &slab[(oh * S) * VK_PITCH];
                {
                    VK_MMUL cA0(bzacc[0]), cA1(bzacc[1]), cB0(bzacc[0]), cB1(bzacc[1]);
                    for (int kh = 0; kh < KH; kh++)
                        chess_prepare_for_pipelining chess_loop_range(7, 7) {
                            const int8_t *r = row0 + kh * VK_PITCH;
                            const int8_t *bk = &Bp[kh * VK_KB * VK_NB * 64];
                            for (int kb = 0; kb < VK_KB; kb++)
                                chess_unroll_loop() {
                                    const int off = kb < 3 ? kb * 8 : 20;
                                    const aie::vector<int8_t, 64> b0 = aie::load_v<64>(bk + (kb * VK_NB + 0) * 64);
                                    const aie::vector<int8_t, 64> b1 = aie::load_v<64>(bk + (kb * VK_NB + 1) * 64);
                                    const aie::vector<int8_t, 64> aA = VK_LDA(r + 0 * 8 + off);
                                    const aie::vector<int8_t, 64> aB = VK_LDA(r + 8 * 8 + off);
                                    cA0.mac(aA, b0);
                                    cA1.mac(aA, b1);
                                    cB0.mac(aB, b0);
                                    cB1.mac(aB, b1);
                                }
                        }
                    VK_EPILOGUE(cA0, cA1, cB0, cB1, oh, 0, 8);
                }
                {
                    VK_MMUL cA0(bzacc[0]), cA1(bzacc[1]), cB0(bzacc[0]), cB1(bzacc[1]);
                    for (int kh = 0; kh < KH; kh++)
                        chess_prepare_for_pipelining chess_loop_range(7, 7) {
                            const int8_t *r = row0 + kh * VK_PITCH;
                            const int8_t *bk = &Bp[kh * VK_KB * VK_NB * 64];
                            for (int kb = 0; kb < VK_KB; kb++)
                                chess_unroll_loop() {
                                    const int off = kb < 3 ? kb * 8 : 20;
                                    const aie::vector<int8_t, 64> b0 = aie::load_v<64>(bk + (kb * VK_NB + 0) * 64);
                                    const aie::vector<int8_t, 64> b1 = aie::load_v<64>(bk + (kb * VK_NB + 1) * 64);
                                    const aie::vector<int8_t, 64> aA = VK_LDA(r + 16 * 8 + off);
                                    const aie::vector<int8_t, 64> aB = VK_LDA(r + 20 * 8 + off);
                                    cA0.mac(aA, b0);
                                    cA1.mac(aA, b1);
                                    cB0.mac(aB, b0);
                                    cB1.mac(aB, b1);
                                }
                        }
                    VK_EPILOGUE(cA0, cA1, cB0, cB1, oh, 16, 20);
                }
            }
            kc_comp += (uint32_t)(VK_TIC() - kc_t);
            if (CONV2DSTEM_KCYC && i == oh_core / oh_t - 1 && j == OUTPUT_W / ow_t - 1) {
                uint32_t *o32 = (uint32_t *)out;
                o32[0] = 0xC0DE0001u;
                o32[1] = (uint32_t)(VK_TIC() - kc0);
                o32[2] = kc_comp;
                o32[3] = kc_acqc;
                o32[4] = 0xC0DE0002u;
                o32[5] = kc_bprep;
                o32[6] = kc_acqa;
                o32[7] = kc_copy;
            }
            aie::release(win_a);
            aie::commit(win_c);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Header include placement — DO NOT move this to the top of the file.
//
// conv2dstem.h declares conv2d_stem(), whose call chain reaches the mesh launch.
// The aiehlc Clang frontend's VisitFunctionDecl registers an annotated kernel
// into globalKernelFuncs when it visits the kernel DEFINITION, and resolves the
// launch's tensor parameters by looking the kernel up in that map.
//
// Clang's FunctionDecl::hasBody()/getBody() resolve across the whole
// redeclaration chain: visiting a mere PROTOTYPE of conv2d_stem yields the real
// body, so the visitor traverses the launch at the prototype's source position.
// With the header included at the top, that position is ABOVE the kernel
// definition — the launch is then processed while globalKernelFuncs is still
// empty, no tensor params are parsed, and the pipeline emits an empty module.
//
// The failure is SILENT in the worst way: every MLIR pass still "succeeds" and
// host.cc/kernel.cc are still written, but with zero routing connections. The
// only surface symptom is a downstream "Couldn't open aie2ps.prx" from xchessmk,
// because BCF/PRX emission is skipped when the allocator has no allocations.
//
// Including the header AFTER the kernel definition keeps the first-seen
// declaration of conv2d_stem below the kernel, so the kernel registers first.
// Verified by bisection: adding a single forward declaration above the kernel
// to an otherwise-working source reproduces the empty-module build exactly.
// ═══════════════════════════════════════════════════════════════════════════
#include "conv2dstem.h"

// The geometry literals above are duplicated from the header (the header is not
// in scope where they are defined). Pin them together so a change to one side
// breaks the build instead of silently miscompiling the data path.
static_assert(INPUT_H == CONV2DSTEM_INPUT_H, "INPUT_H out of sync with conv2dstem.h");
static_assert(INPUT_W == CONV2DSTEM_INPUT_W, "INPUT_W out of sync with conv2dstem.h");
static_assert(INPUT_C == CONV2DSTEM_INPUT_C, "INPUT_C out of sync with conv2dstem.h");
static_assert(KERNEL_H == CONV2DSTEM_KERNEL_H, "KERNEL_H out of sync with conv2dstem.h");
static_assert(KERNEL_W == CONV2DSTEM_KERNEL_W, "KERNEL_W out of sync with conv2dstem.h");
static_assert(NUM_FILTERS == CONV2DSTEM_NUM_FILTERS, "NUM_FILTERS out of sync with conv2dstem.h");
static_assert(STRIDE == CONV2DSTEM_STRIDE, "STRIDE out of sync with conv2dstem.h");
static_assert(PAD == CONV2DSTEM_PAD, "PAD out of sync with conv2dstem.h");
static_assert(OUTPUT_H == CONV2DSTEM_OUTPUT_H, "OUTPUT_H out of sync with conv2dstem.h");
static_assert(OUTPUT_W == CONV2DSTEM_OUTPUT_W, "OUTPUT_W out of sync with conv2dstem.h");
// The host's view of the kernel's tiling (tile-major collect) must match the
// kernel's own constants: oh_t = 7, ow_t = 28, oc_t = 16 on a 4x4 mesh, 4 x 4
// rounds per core. The kernel analysis checks the kernel side against the DMA.
static_assert(CS_OH_T * HW_ROWS * 4 == OUTPUT_H, "height tiling does not cover OUTPUT_H");
static_assert(CS_OW_T * 4 == OUTPUT_W, "width rounds do not cover OUTPUT_W");
static_assert(OC_PER_G == 16, "the kernel computes 16 output channels per mesh column");

// ═══════════════════════════════════════════════════════════════════════════
// HOST — staging buffer cache
//
// The mesh itself CANNOT be hoisted into a static: the aiehlc Clang frontend
// extracts the mesh dims by pattern-matching a function-local VarDecl
// initialized via device.partition(...) in the SAME function as the launch
// (src/llvm/aiehlc.cc). So aieArray/aieMesh are re-declared per call; only the
// DDR buffers are cached. That is where the real cost is anyway —
// __Runtime_Alloc is a plain 64-byte-aligned aligned_alloc with no device
// handle, so the buffers are independent of any mesh lifetime.
// ═══════════════════════════════════════════════════════════════════════════
#define IFM_PAD_ELEMS (INPUT_H_PAD * IFM_W_PITCH * INPUT_C_ALIGN) // 230*232*4 = 213440

// OC_PER_G*K bytes per mesh column: filter taps in mmul-tile order + quant params.
#define WTS_BT_ELEMS (NUM_FILTERS * K)                // 64*196    = 12544
#define OFM_ELEMS (OUTPUT_H * OUTPUT_W * NUM_FILTERS) // 112*112*64 = 802816

static int8_t *g_ifm_pad = nullptr;       // [230,230,4] spatially + channel padded
static int8_t *g_wts_bt = nullptr;        // B^T [64,196]
static uint8_t *g_ofm = nullptr;          // AIE destination, TILE-MAJOR (see tile_major_off)
static const int8_t *g_wts_src = nullptr; // provenance of the packed weights
// Set once conv2d_stem_prepadded() has written the spatial border with the
// caller's pad value; stage_ifm() then re-zeroes it before trusting the zero
// border again.
static int g_border_dirty = 0;

// Allocate the staging buffers on first use. The zero-pad regions are written
// once here and never touched again: stage_ifm() only ever overwrites the
// interior real-pixel positions, so the borders stay zero for the process
// lifetime. Same for the padding channel in the weights.
static int ensure_staging(void) {
    if (g_ifm_pad && g_wts_bt && g_ofm)
        return CONV2DSTEM_OK;

    if (!g_ifm_pad)
        g_ifm_pad = (int8_t *)__Runtime_alloc_uncached(IFM_PAD_ELEMS * sizeof(int8_t));
    if (!g_wts_bt)
        g_wts_bt = (int8_t *)__Runtime_Alloc(WTS_BT_ELEMS * sizeof(int8_t));
    if (!g_ofm)
        g_ofm = (uint8_t *)__Runtime_Alloc(OFM_ELEMS * sizeof(uint8_t));

    if (!g_ifm_pad || !g_wts_bt || !g_ofm) {
        conv2d_stem_release();
        return CONV2DSTEM_ERR_ALLOC;
    }

    // Materialize both pads once. memset covers the spatial border, the padding
    // channel, and the filter's padding channel in one go.
    memset(g_ifm_pad, 0, IFM_PAD_ELEMS * sizeof(int8_t));
    memset(g_wts_bt, 0, WTS_BT_ELEMS * sizeof(int8_t));
    // The CPU never writes g_ofm again (collect_ofm_* only read it). Registering
    // flushes it once, which also retires any line the allocator left dirty.
    __Runtime_buffer_device_written(g_ofm, OFM_ELEMS * sizeof(uint8_t));
    return CONV2DSTEM_OK;
}

// One staged pixel: 3 channel bytes + the zero pad channel as one aligned 32-bit
// little-endian store. g_ifm_pad is non-cacheable (see ensure_staging), where a
// word store merges in the write buffer and a byte store would be its own
// transaction; it is also cheaper than byte stores on a cached buffer.
#define STAGE_PIXEL(p) ((uint32_t)(p)[0] | ((uint32_t)(p)[1] << 8) | ((uint32_t)(p)[2] << 16))

// Scatter the caller's raw [224,224,3] IFM into the padded, channel-aligned
// staging buffer. Real pixel (h,w,c) -> ((h+PAD)*IFM_W_PITCH + (w+PAD))*4 + c
// (IFM_W_PITCH = 230 here: the derived DMA walks the [230,230,4] tensor).
// Channel 3 and the spatial border are left at the zero written by
// ensure_staging() — do NOT memset here, it would re-zero 211600 bytes every
// call to rewrite regions that are already zero.
static void stage_ifm(const int8_t *ifm) {
    if (g_border_dirty) {
        memset(g_ifm_pad, 0, IFM_PAD_ELEMS * sizeof(int8_t));
        g_border_dirty = 0;
    }
    for (int h = 0; h < INPUT_H; h++) {
        uint32_t *dst = (uint32_t *)&g_ifm_pad[((h + PAD) * IFM_W_PITCH + PAD) * INPUT_C_ALIGN];
        const uint8_t *src = (const uint8_t *)&ifm[h * INPUT_W * INPUT_C];
        for (int w = 0; w < INPUT_W; w++, src += INPUT_C)
            dst[w] = STAGE_PIXEL(src);
    }
}

// Copy a caller-padded [230,230,3] IFM -- border included -- into the
// channel-aligned staging buffer. The border carries whatever the caller padded
// with (a TVM uint8 graph pads with the input zero-point, NOT zero), so it is
// rewritten on every call and the zero-border invariant stage_ifm() relies on is
// flagged as broken. Channel 3 is never written and stays zero.
static void stage_ifm_prepadded(const int8_t *ifm_pad) {
    for (int h = 0; h < INPUT_H_PAD; h++) {
        uint32_t *dst = (uint32_t *)&g_ifm_pad[h * IFM_W_PITCH * INPUT_C_ALIGN];
        const uint8_t *src = (const uint8_t *)&ifm_pad[h * INPUT_W_PAD * INPUT_C];
        for (int w = 0; w < INPUT_W_PAD; w++, src += INPUT_C)
            dst[w] = STAGE_PIXEL(src);
    }
    g_border_dirty = 1;
}

// Adopt a caller-supplied [230,230,4] IFM -- already spatially padded AND
// channel-aligned, so each row is one straight copy into the IFM_W_PITCH-pitch
// staging rows (no per-pixel scatter).
//
// This is what the generated fixture (data/make_image_header.py ->
// conv2dstem_image.h) emits, precisely so the board does no layout work on an
// image the host already knows the final shape of. Like
// stage_ifm_prepadded(), the border carries the caller's pad value -- the
// quantized ResNet graph pads with the input zero-point, not zero -- so the
// zero-border invariant stage_ifm() relies on is flagged broken.
//
// The 4th channel is whatever the caller put there (the fixture writes 0);
// nothing reads it: the kernel masks the pad-channel lanes of its filter.
static void stage_ifm_pad4(const int8_t *ifm_pad4) {
    for (int h = 0; h < INPUT_H_PAD; h++)
        memcpy(&g_ifm_pad[h * IFM_W_PITCH * INPUT_C_ALIGN], &ifm_pad4[h * INPUT_W_PAD * INPUT_C_ALIGN],
               INPUT_W_PAD * INPUT_C_ALIGN * sizeof(int8_t));
    g_border_dirty = 1;
}

// Byte offset in g_wts_bt of weight (f, kh, kw, c), c < INPUT_C, in the
// mmul-tile layout the kernel consumes (see "Filter window layout" above the
// kernel): one 3136-byte block per mesh column of OC_PER_G filters, and inside
// it per kh 448 bytes of 8(K) x 8(N) tiles, K-lane ki = kw*4 + c.
static inline int wts_tile_off(int f, int kh, int kw, int c) {
    const int col = f / OC_PER_G, nb = (f % OC_PER_G) / 8, n = f % 8;
    const int ki = kw * INPUT_C_ALIGN + c, kb = ki / 8, k = ki % 8;
    const int base = col * (OC_PER_G * K) + kh * VK_KH_BYTES;
    return kb < 3 ? base + (kb * 2 + nb) * 64 + k * 8 + n : base + 384 + nb * 32 + k * 8 + n;
}

// Byte offset in g_wts_bt of quant-param byte row j (VK_PRM_OFF) for filter f.
static inline int wts_param_off(int f, int j) {
    return (f / OC_PER_G) * (OC_PER_G * K) + VK_PRM_OFF(j, (f % OC_PER_G) / 8, f % 8);
}

// Read quant param p (0 = bias - zero_point, 1 = multiplier, 2 = shift) of
// filter f back out of a packed buffer -- the inverse of pack_wts_tiles().
static int32_t wts_param(const int8_t *bt, int f, int p) {
    uint32_t u = 0;
    for (int b = 0; b < 4; b++)
        u |= (uint32_t)(uint8_t)bt[wts_param_off(f, p * 4 + b)] << (8 * b);
    return (int32_t)u;
}

// Pack the caller's filter + quant params into g_wts_bt in mmul-tile order.
//
// `nchwc` selects the SOURCE layout; the destination is the same either way:
//   0  plain OHWI [64][7][7][3]
//   1  TVM's x86 NCHWc pack OIHW3i4o [16][7][7][3][4]: `oc` is the OUTER product
//      oc_chunk*4 + oc_block, so one filter's taps are strided 4 apart. Strides
//      are TVM's own: oc_chunk 588 = KH*KW*INPUT_C*OC_BLOCK, kh 84, kw 12, ic 4.
//
// The kernel's exact requantize runs on NATIVE 32x16 multiplies (see the kernel
// header), which is valid only while, per 8-channel N-block:
//   * the multiplier splits into two int16 halves (mult < 2^31 - 2^15),
//   * shift >= -14 (2^(s-1) a multiple of 2^16), and
//   * max(shift) - min(shift) <= 27 (each of the two scaling stages stays an
//     int16 power of two).
// Those are checked here rather than assumed: a violating layer gets
// CONV2DSTEM_ERR_BAD_QPARAM, never quietly wrong pixels. On the real layer 01
// one N-block's shifts span 8..27 (19).
static int pack_wts_tiles(const int8_t *wts, int nchwc, const conv2dstem_qparam *qp) {
    for (int g = 0; g < NUM_FILTERS; g += 8) {
        int smin = qp[g].shift, smax = qp[g].shift;
        for (int f = g; f < g + 8; f++) {
            const int32_t mult = qp[f].multiplier;
            const int32_t ml = (int32_t)(int16_t)(uint16_t)((uint32_t)mult & 0xFFFFu);
            const int32_t mh = (int32_t)(((int64_t)mult - ml) >> 16);
            if (mh > 32767 || mh < -32768 || qp[f].shift < -14) {
                printf("[conv2dstem] ERROR: channel %d multiplier %d / shift %d outside the vector "
                       "requantize's range\n",
                       f, (int)mult, (int)qp[f].shift);
                return CONV2DSTEM_ERR_BAD_QPARAM;
            }
            smin = qp[f].shift < smin ? qp[f].shift : smin;
            smax = qp[f].shift > smax ? qp[f].shift : smax;
        }
        if (smax - smin > 27) {
            printf("[conv2dstem] ERROR: channels %d..%d shifts span %d..%d (> 27)\n", g, g + 7, smin, smax);
            return CONV2DSTEM_ERR_BAD_QPARAM;
        }
    }

    // Zero first: the K-lanes past tap 6 and the unused pad-channel rows must
    // be zero (the kernel masks the pad rows, but not the kb=3 tail lanes it
    // never loads -- keep the whole buffer deterministic anyway).
    memset(g_wts_bt, 0, WTS_BT_ELEMS * sizeof(int8_t));
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int kh = 0; kh < KERNEL_H; kh++)
            for (int kw = 0; kw < KERNEL_W; kw++)
                for (int c = 0; c < INPUT_C; c++) {
                    const int src =
                        nchwc ? (f / CONV2DSTEM_OC_BLOCK) * (KERNEL_H * KERNEL_W * INPUT_C * CONV2DSTEM_OC_BLOCK) +
                                    kh * (KERNEL_W * INPUT_C * CONV2DSTEM_OC_BLOCK) +
                                    kw * (INPUT_C * CONV2DSTEM_OC_BLOCK) + c * CONV2DSTEM_OC_BLOCK +
                                    f % CONV2DSTEM_OC_BLOCK
                              : ((f * KERNEL_H + kh) * KERNEL_W + kw) * INPUT_C + c;
                    g_wts_bt[wts_tile_off(f, kh, kw, c)] = wts[src];
                }
        // bias - zero_point is folded here (uint32 arithmetic: wraps exactly
        // like the int32 accumulator the reference adds both into).
        const uint32_t fields[3] = {(uint32_t)qp[f].bias - (uint32_t)qp[f].zero_point, (uint32_t)qp[f].multiplier,
                                    (uint32_t)qp[f].shift};
        for (int j = 0; j < 12; j++)
            g_wts_bt[wts_param_off(f, j)] = (int8_t)((fields[j / 4] >> (8 * (j % 4))) & 0xFF);
    }
    return CONV2DSTEM_OK;
}

// g_ofm is TILE-MAJOR (#pragma output_tile_major):
//   [mesh row 4][mesh col 4][round 16][oh 7][ow 28][c 16]
// Mesh row r owns image rows r*28.., mesh col owns channels col*16..; a core's
// round = h*4 + w walks its 4 height slabs of CS_OH_T rows, each in 4 width slabs
// of CS_OW_T columns (the input BD's order: width inside one activation, height
// across activations). Byte offset of output pixel (oh, ow, channel oc):
static inline int tile_major_off(int oh, int ow, int oc) {
    const int r = oh / (CS_OH_T * 4), h = (oh % (CS_OH_T * 4)) / CS_OH_T, w = ow / CS_OW_T;
    const int core = r * HW_COLS + oc / OC_PER_G, round = h * 4 + w;
    return ((core * 16 + round) * CS_OH_T + oh % CS_OH_T) * (CS_OW_T * OC_PER_G) + (ow % CS_OW_T) * OC_PER_G +
           oc % OC_PER_G;
}

// Walk g_ofm sequentially, one OC_PER_G-channel pixel run (16 bytes) at a time,
// handing each to put(oh, ow, col, src). Reading in storage order keeps the
// freshly-invalidated source streaming; the scattered side is the write.
template <typename Put> static void walk_tile_major(Put put) {
    const uint8_t *src = g_ofm;
    for (int r = 0; r < HW_ROWS; r++)
        for (int col = 0; col < HW_COLS; col++)
            for (int h = 0; h < 4; h++)
                for (int w = 0; w < 4; w++)
                    for (int y = 0; y < CS_OH_T; y++)
                        for (int x = 0; x < CS_OW_T; x++, src += OC_PER_G)
                            put(r * CS_OH_T * 4 + h * CS_OH_T + y, w * CS_OW_T + x, col, src);
}

// AIE result -> TVM's NCHW4c [16][112][112][4]. Each 16-channel pixel run
// splits into four 4-channel blocks, one per oc_chunk plane.
static void collect_ofm_nchwc(uint8_t *ofm) {
    walk_tile_major([ofm](int oh, int ow, int col, const uint8_t *src) {
        for (int q = 0; q < OC_PER_G / CONV2DSTEM_OC_BLOCK; q++) {
            const int occ = col * (OC_PER_G / CONV2DSTEM_OC_BLOCK) + q;
            memcpy(&ofm[((occ * OUTPUT_H + oh) * OUTPUT_W + ow) * CONV2DSTEM_OC_BLOCK], src + q * CONV2DSTEM_OC_BLOCK,
                   CONV2DSTEM_OC_BLOCK);
        }
    });
}

// AIE result -> dense HWC [112][112][64] (the conv2d_stem* contract).
static void collect_ofm_hwc(uint8_t *ofm) {
    walk_tile_major([ofm](int oh, int ow, int col, const uint8_t *src) {
        memcpy(&ofm[(oh * OUTPUT_W + ow) * NUM_FILTERS + col * OC_PER_G], src, OC_PER_G);
    });
}

void conv2d_stem_invalidate_weights(void) { g_wts_src = nullptr; }

void conv2d_stem_release(void) {
    if (g_ifm_pad) {
        __Runtime_free_uncached(g_ifm_pad);
        g_ifm_pad = nullptr;
    }
    if (g_wts_bt) {
        free(g_wts_bt);
        g_wts_bt = nullptr;
    }
    if (g_ofm) {
        __Runtime_buffer_forget(g_ofm);
        free(g_ofm);
        g_ofm = nullptr;
    }
    g_wts_src = nullptr;
    g_border_dirty = 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST — library entry
// ═══════════════════════════════════════════════════════════════════════════

// Pack weights (if they changed), run conv2d_spatial on the mesh, copy out.
// The ifm staging buffer must already be filled. This is the ONE launch site:
// the aiehlc Clang frontend recovers the mesh from the function-local
// device.partition(...) VarDecl, which must sit in the same function as the
// kernel launch -- so both public entries funnel through here rather than
// each carrying its own launch. (Never write the triple-angle launch syntax in
// a comment near here: aiehlc's source rewriter matches it textually and
// splices __aie_launch over the surrounding code -- skill aiesourcetextrewrite.)
static int stem_run(const int8_t *wts, const conv2dstem_qparam *qp, uint8_t *ofm) {
    // Re-pack only when the weight buffer changed identity. Callers mutating
    // weights in place, or swapping qp while reusing the same wts pointer,
    // must call conv2d_stem_invalidate_weights().
    if (g_wts_src != wts) {
        const int prc = pack_wts_tiles(wts, 0, qp);
        if (prc != CONV2DSTEM_OK)
            return prc;
        g_wts_src = wts;
    }

    // --- Device + mesh ---
    // Must stay function-local and constant-foldable: the Clang frontend
    // pattern-matches this VarDecl to recover the mesh dims and partition.
    //
    // Partition is {2,6}, FIVE columns for a 4-column mesh. The spare column is
    // not slack: planControlColumn only grants a dedicated control spine when
    // partition cols >= mesh cols + 1, and the spine is what
    // control_plan_shim_bd_ctrl runs its row-0 packets along. The data plane is
    // unmoved by it -- the planner sets dataStartCol = relStartCol + 1, so the
    // mesh still occupies absolute columns 3..6 exactly as it did when the
    // partition was {3,6}. Narrow this back and the shim BDs silently return to
    // MMIO (control_plan_dedicated_shim turns that silence into a build error).
    // Written as a bare literal, per this file's rule for geometry: the frontend
    // recovers the mesh from this VarDecl.
    conv2dstem_control_plane_init();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, HW_ROWS, HW_COLS);

    conv2d_spatial<<<mesh>>>(g_ifm_pad, g_wts_bt, g_ofm, M, N, K);

    // Copy the AIE result out to the caller's buffer, de-tiling it. The AIE
    // destination is a cached staging buffer rather than `ofm` directly because
    // the caller's pointer carries no alignment guarantee (the shim S2MM needs
    // the 64-byte-aligned DMA-capable allocation from __Runtime_Alloc), and the
    // array writes it tile-major (#pragma output_tile_major).
    collect_ofm_hwc(ofm);
    return CONV2DSTEM_OK;
}

// Stage a raw caller input, then run. The 4-argument spelling of stem_run().
//
// Exists so a driver that INCLUDES this file (src/aietensorop/conv2dstem/main.cc)
// can say stem_run(ifm, wts, ofm, qp) and get staging + launch in one call,
// without reaching into ensure_staging()/stage_ifm() itself. An overload rather
// than a change to the 3-argument form, because the two public entries stage the
// input DIFFERENTLY -- conv2d_stem_prepadded() uses stage_ifm_prepadded() -- so
// the input staging cannot be folded into the shared launch site.
static int stem_run(const int8_t *ifm, const int8_t *wts, uint8_t *ofm, const conv2dstem_qparam *qp) {
    if (!ifm || !wts || !qp || !ofm)
        return CONV2DSTEM_ERR_NULL_ARG;

    const int rc = ensure_staging();
    if (rc != CONV2DSTEM_OK)
        return rc;

    stage_ifm(ifm);
    return stem_run(wts, qp, ofm);
}

int conv2d_stem(const int8_t *ifm, const int8_t *wts, const conv2dstem_qparam *qp, uint8_t *ofm) {
    return stem_run(ifm, wts, ofm, qp);
}

// Fused conv on an input the caller already padded -- the TVM BYOC entry.
//
// TVM hoists the stem's spatial padding into its own nn.pad and fills it with
// the INPUT ZERO-POINT (113 for this model), not zero, so the library must not
// re-pad: it takes the [230,230,3] tensor border and all. The caller also
// shifts the uint8 activations into int8 (x - 128) and folds the matching
// +128*sum(w) into qp[].bias, which keeps the int8 x int8 kernel exact.
int conv2d_stem_prepadded(const int8_t *ifm_pad, const int8_t *wts, const conv2dstem_qparam *qp, uint8_t *ofm) {
    if (!ifm_pad || !wts || !qp || !ofm)
        return CONV2DSTEM_ERR_NULL_ARG;

    const int rc = ensure_staging();
    if (rc != CONV2DSTEM_OK)
        return rc;

    stage_ifm_prepadded(ifm_pad);
    return stem_run(wts, qp, ofm);
}

// Fused conv in TVM's PACKED layouts -- the exact operator the deployed graph
// calls (see conv2dstem.h for the four shapes).
//
// Deliberately a separate entry rather than a change to
// conv2d_stem_prepadded(): that one is the BYOC wrapper's contract, and the
// two differ only in layout, so folding them would silently reinterpret every
// weight and output byte for the offload path.
//
// The input needs no conversion at all -- TVM's [230,230,3] zero-point-padded
// tensor is already exactly what stage_ifm_prepadded() consumes.
// Generic-timer stamps of the LAST conv2d_stem_nchwc() call: entry, just before
// the launch, just after it, and after the output repack. Stored rather than
// printed -- a UART printf inside the call would inflate the very latency it
// reports. test_conv2d.cc's run_perf() turns them into the stage/launch/collect split.
XTime g_conv2dstem_stamp[4];

int conv2d_stem_nchwc(const int8_t *ifm_pad, const int8_t *wts, const conv2dstem_qparam *qp, uint8_t *ofm) {
    XTime_GetTime(&g_conv2dstem_stamp[0]);
    if (!ifm_pad || !wts || !qp || !ofm)
        return CONV2DSTEM_ERR_NULL_ARG;

    const int rc = ensure_staging();
    if (rc != CONV2DSTEM_OK)
        return rc;

    stage_ifm_prepadded(ifm_pad);

    // Same re-pack-on-pointer-change rule as stem_run(); spelled out here
    // because the packing function differs, so this cannot call stem_run().
    if (g_wts_src != wts) {
        const int prc = pack_wts_tiles(wts, 1, qp);
        if (prc != CONV2DSTEM_OK)
            return prc;
        g_wts_src = wts;
    }

    // Same five-column partition and the same control-plane opt-ins as
    // stem_run(); see the comment there for why the spare column exists. The two
    // launch sites must agree -- they describe the same physical mesh.
    conv2dstem_control_plane_init();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, HW_ROWS, HW_COLS);

    XTime_GetTime(&g_conv2dstem_stamp[1]);
    conv2d_spatial<<<mesh>>>(g_ifm_pad, g_wts_bt, g_ofm, M, N, K);
    XTime_GetTime(&g_conv2dstem_stamp[2]);

    collect_ofm_nchwc(ofm);
    XTime_GetTime(&g_conv2dstem_stamp[3]);
    return CONV2DSTEM_OK;
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST — bring-up verification (opt-in, NOT on the conv2d_stem() fast path)
// ═══════════════════════════════════════════════════════════════════════════

// Scalar reference conv over the SAME padded staging buffer the AIE consumed,
// so a divergence points at the data path and not at a staging mismatch.
// Iterating the padded extents gives true padded conv with no OOB reads.
// Reads the SAME packed window the core reads, params included, so this stays
// in lock-step with the kernel by construction. int32 accumulator and the full
// fused epilogue -- if this drifts from the kernel the verify is worthless.
static void scalar_conv2d_ref(const int8_t *ifm_pad, const int8_t *wts_bt, uint8_t *ref) {
    for (int oh = 0; oh < OUTPUT_H; oh++) {
        for (int ow = 0; ow < OUTPUT_W; ow++) {
            for (int f = 0; f < NUM_FILTERS; f++) {
                int32_t acc = 0;
                for (int kh = 0; kh < KERNEL_H; kh++) {
                    for (int kw = 0; kw < KERNEL_W; kw++) {
                        // c < INPUT_C, not INPUT_C_ALIGN: the pad channel holds
                        // the quant params, exactly as in the kernel.
                        for (int c = 0; c < INPUT_C; c++) {
                            const int ih = oh * STRIDE + kh;
                            const int iw = ow * STRIDE + kw;
                            const int8_t iv = ifm_pad[(ih * IFM_W_PITCH + iw) * INPUT_C_ALIGN + c];
                            const int8_t fv = wts_bt[wts_tile_off(f, kh, kw, c)];
                            acc += (int32_t)iv * (int32_t)fv;
                        }
                    }
                }
                acc += wts_param(wts_bt, f, 0); // bias - zero-point
                const int32_t mult = wts_param(wts_bt, f, 1);
                const int sh = (int)wts_param(wts_bt, f, 2);
                int64_t v = ((int64_t)acc * (int64_t)mult + ((int64_t)1 << (sh + 30))) >> (sh + 31);
                if (v > 255)
                    v = 255;
                else if (v < 0)
                    v = 0;
                ref[(oh * OUTPUT_W + ow) * NUM_FILTERS + f] = (uint8_t)v;
            }
        }
    }
}

#define CONV2DSTEM_MAX_REPORTED_MISMATCHES 32

int conv2d_stem_verify(const int8_t *ifm, const int8_t *wts, const conv2dstem_qparam *qp, uint8_t *ofm) {
    const int rc = conv2d_stem(ifm, wts, qp, ofm);
    if (rc != CONV2DSTEM_OK)
        return rc;

    // 802816 bytes — too large for the stack, so allocate it.
    uint8_t *ref = (uint8_t *)malloc(OFM_ELEMS * sizeof(uint8_t));
    if (!ref)
        return CONV2DSTEM_ERR_ALLOC;

    printf("[conv2dstem] computing CPU reference (157M MACs, this is slow)...\n");
    XTime t_start, t_end;
    XTime_GetTime(&t_start);
    scalar_conv2d_ref(g_ifm_pad, g_wts_bt, ref);
    XTime_GetTime(&t_end);
    printf("[conv2dstem] CPU reference time: %.3f ms\n", 1.0 * (t_end - t_start) / COUNTS_PER_SECOND * 1000.0);

    int mismatches = 0;
    for (int i = 0; i < OFM_ELEMS; i++) {
        if (ofm[i] != ref[i]) {
            if (mismatches < CONV2DSTEM_MAX_REPORTED_MISMATCHES) {
                const int oh = i / (OUTPUT_W * NUM_FILTERS);
                const int ow = (i / NUM_FILTERS) % OUTPUT_W;
                const int f = i % NUM_FILTERS;
                printf("[conv2dstem] MISMATCH ofm[oh=%d,ow=%d,f=%d] (flat %d): got %d, expected %d\n", oh, ow, f, i,
                       ofm[i], ref[i]);
            }
            mismatches++;
        }
    }
    free(ref);

    if (mismatches == 0)
        printf("[conv2dstem] PASS: all %d output elements match.\n", OFM_ELEMS);
    else
        printf("[conv2dstem] FAIL: %d mismatches out of %d.\n", mismatches, OFM_ELEMS);
    return mismatches;
}
