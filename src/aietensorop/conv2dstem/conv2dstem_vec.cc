/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * conv2dstem_vec — VECTORIZED variant of conv2dstem.cc (ResNet-18 stem conv1).
 *
 * Identical host side, spatial spaces, windows, DMA and control-plane pragmas
 * to conv2dstem.cc -- byte-for-byte outside the kernel section -- so the two
 * are interchangeable behind conv2dstem.h. Only the core's arithmetic differs:
 * aie::mmul<4,8,8,int8,int8> + a bit-exact vector requantize replace the scalar
 * MAC loop that profiled at 0.09 % of the 4x4 mesh's int8 peak
 * (doc/performance/conv2dstem_perf_review.md). Build/run it with
 * `./buildtest.sh --vec` (entry test_conv2d_vec.cc, generated from
 * test_conv2d.cc). conv2dstem.cc stays the --aie-offload source.
 *
 * Original description of the operator follows.
 *
 *   ifm [224,224,3] (*) wts [64,7,7,3]  stride 2, pad 3  ->  ofm [112,112,64]
 *
 * Derived from example/tileprogram/ccode/simpleconv2d.cc, which already targets
 * exactly this geometry. Two things change:
 *
 *   1. No main(). The host entry is conv2d_stem(ifm, wts, ofm) and the input
 *      data comes from the CALLER. hostcompile.sh classifies the generated
 *      host.cc as HOST_ENTRY_KIND=library and archives libconv2dstem.a instead
 *      of linking an ELF (see skill: hostlibrarymode).
 *   2. The DDR pre-pad / channel-align / filter-transpose that simpleconv2d.cc
 *      did inline while synthesizing test data is now a staging step that
 *      scatters CALLER buffers into AIE-layout buffers.
 *
 * ── Why staging buffers exist ───────────────────────────────────────────────
 * The AIE DMA can only RELOCATE bytes; it cannot synthesize zeros. So every
 * zero the convolution reads must physically exist in DDR. Two independent pads
 * are materialized here:
 *   * spatial: PAD=3 zero pixels on each H/W border -> [230, 230, ...]
 *   * channel: cin 3 -> INPUT_C_ALIGN 4 via a zero channel. The padded channel
 *     contributes 0 to every dot product, so the result is bit-identical to an
 *     unaligned cin=3 run while keeping the K dim a multiple of 4.
 * Real pixel (h,w,c) therefore lands at ((h+PAD)*INPUT_W_PAD + (w+PAD))*4 + c.
 *
 * The staging buffers are cached across calls (allocation + the 9408-element
 * weight transpose are not free per inference). See conv2d_stem_release().
 ******************************************************************************/
// NOTE: conv2dstem.h is deliberately NOT included here — it is included BELOW,
// after the kernel definition. See the "Header include placement" comment above
// the include site. Including it here silently produces a library that builds
// but does nothing.
//
// Also note: the aiehlc source rewriter scans this file as TEXT for the kernel
// annotation keyword and wraps "the next { it finds" in an #ifdef guard. Writing
// that keyword in a comment ABOVE the spatial-space descriptors makes it wrap the
// first descriptor's initializer instead of the kernel body, which silently
// default-constructs the descriptor. Do not spell that keyword in prose here;
// say "the kernel" instead.
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
// KERNELCONFIGOFFLOAD on {2,6} needed two compiler fixes, or the mesh deadlocks
// (wait_io TIMEOUT; doc/debug/troubleshoot_aie_offload_timeout.md): the get_coreid()
// dispatch rebased by the partition start column, and MM2S BD length taken from the
// host plan (3136 B), not sizeof(buf_out_ping_0) (784 B). Skill: kcocoreidpartition.
#pragma KERNELCONFIGOFFLOAD

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

// Turn the control-plane runtime opt-ins on. Called at the top of every launch
// site, before the device exists.
static inline void conv2dstem_control_plane_init(void) {
    __Runtime_ctrl_high_throughput_enable(1);
    __Runtime_ctrl_pmap_enable(0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Geometry
//
// Written as bare literals, NOT as aliases of the CONV2DSTEM_* macros in
// conv2dstem.h, because that header is included only AFTER the kernel (see the
// include site below) and so is not in scope here.
//
// The kernel extractor carries only #defines written DIRECTLY in this .cc into
// the kernel translation unit; the kernel TU does NOT include conv2dstem.h and
// must not — a public header pulling in xil_cache.h/xiltimer.h would not build
// under the AIE chess compiler.
//
// These are cross-checked against the header's CONV2DSTEM_* values by
// static_assert at the include site below, so the duplication cannot drift.
// ═══════════════════════════════════════════════════════════════════════════
#define INPUT_H 224
#define INPUT_W 224
#define INPUT_C 3       // real channels
#define INPUT_C_ALIGN 4 // channel-layout stride (3 + 1 zero pad)
#define KERNEL_H 7
#define KERNEL_W 7
#define NUM_FILTERS 64
#define STRIDE 2
#define PAD 3

#define OUTPUT_H 112 // (224 + 2*3 - 7)/2 + 1
#define OUTPUT_W 112

// Spatially pre-padded DDR extents (what the shim DMA actually walks).
#define INPUT_H_PAD (INPUT_H + 2 * PAD) // 230
#define INPUT_W_PAD (INPUT_W + 2 * PAD) // 230

// im2col -> GEMM dims. K uses the ALIGNED channel count.
#define M (OUTPUT_H * OUTPUT_W)                 // 12544
#define K (KERNEL_H * KERNEL_W * INPUT_C_ALIGN) // 196
#define N NUM_FILTERS                           // 64

// HW mesh
#define HW_ROWS 4
#define HW_COLS 4

// ═══════════════════════════════════════════════════════════════════════════
// Kernel-visible conv geometry (spatial-halo path)
//
// These SP_* literals are what the conv2d_spatial kernel below compiles
// against. They MUST agree with the geometry above; the kernel TU cannot see
// the constexpr descriptors, only these #defines.
// ═══════════════════════════════════════════════════════════════════════════
#define SP_KH 7   // KERNEL_H
#define SP_KW 7   // KERNEL_W
#define SP_C 4    // INPUT_C_ALIGN (layout stride; real cin=3, +1 zero pad)
#define SP_S 2    // STRIDE
#define SP_OW 112 // OUTPUT_W
#define SP_K 196  // KH*KW*INPUT_C_ALIGN

// Per-output-channel quant params (bias, zero_point, multiplier, shift -- four
// int32 LE = 16 bytes) ride INSIDE the existing filter window, in the padding
// channel, costing zero extra bytes and zero DMA changes.
//
// Why here and not on a fourth port or in an appended row:
//   * A fourth aie::port is unproven -- every kernel in this repo uses exactly
//     three, so the routing/DMA path has never been exercised with four.
//   * Appending 16B per channel (a 212-byte row) breaks ColBC: that descriptor
//     factorizes the window as EXACTLY d1(N) x d2(KW) x d3(KH) x d4(C_ALIGN)
//     and has no slack dimension to absorb a tail, so the DMA would move the
//     wrong bytes while still "working".
//
// The input has 3 real channels padded to INPUT_C_ALIGN=4, so byte c==3 of
// every one of the 49 taps is a hard zero the convolution never reads (the
// matching ifm channel is zero, so the product is zero regardless). That is
// 49 free bytes per output channel; the params need 16.
//
// Params live in taps 0..3, at tap-local offset SP_C-1. The kernel skips them
// by zeroing the pad channel in its own accumulation -- see the c < INPUT_C
// guard in the MAC loop.
#define SP_REAL_C 3 // INPUT_C: channels the MAC loop may read
#define SP_PARAM_BYTES 16
#define SP_PARAM_TAP_OFF (SP_C - 1) // byte 3 of each 4-byte tap: the pad channel

// Read quant param `idx` (0..3) out of one output channel's filter row `brow`.
//
// A MACRO, not a function: the kernel extractor copies only #defines written
// directly in this file into the kernel translation unit, so a `static inline`
// helper compiles here but leaves the kernel TU with "use of undeclared
// identifier". Keep every kernel-visible helper in this form.
//
// Byte b of param idx lives at tap (idx*4 + b), offset SP_PARAM_TAP_OFF within
// that tap -- little-endian, strided SP_C apart. Assembled byte-wise because
// the bytes are non-contiguous and the row has no alignment guarantee.
#define RD_PARAM_I32(brow, idx)                                                                                        \
    ((int32_t)(((uint32_t)(uint8_t)(brow)[((idx) * 4 + 0) * SP_C + SP_PARAM_TAP_OFF]) |                                \
               ((uint32_t)(uint8_t)(brow)[((idx) * 4 + 1) * SP_C + SP_PARAM_TAP_OFF] << 8) |                           \
               ((uint32_t)(uint8_t)(brow)[((idx) * 4 + 2) * SP_C + SP_PARAM_TAP_OFF] << 16) |                          \
               ((uint32_t)(uint8_t)(brow)[((idx) * 4 + 3) * SP_C + SP_PARAM_TAP_OFF] << 24)))

#define PAD_W 3
#define PAD_H 3

// Output tile geometry. 112 / (HW_ROWS*OH_T) = 112/(4*7) = 4 slabs per core —
// a clean integer tiling with no remainder slab.
#define OH_T 7  // output tile height
#define OW_T 28 // output tile width

// Spatial-halo HEIGHT descriptor. Drives both RowBC_spatial.d1 and SP_OHR, so
// the per-tile slab supply == the kernel's per-tile output-row demand.
#define PAD_H_LO 3                              // top pad (first tile only)
#define PAD_H_HI 3                              // bottom pad (last tile only)
#define TILE_H ((OH_T - 1) * SP_S + SP_KH)      // 19 input rows per on-core round
#define TILE_STRIDE_H (TILE_H - (SP_KH - SP_S)) // 14 halo step (overlap = KH-S = 5)
// Output rows produced from a TILE_H-row slab. DDR is PRE-padded, so the slab is
// a window of the already-padded buffer (top pad materialized) — do NOT re-add
// PAD_H_LO here.
#define SP_OHR ((TILE_H - SP_KH) / SP_S + 1) // = 7 (== OH_T)

// Spatial-halo WIDTH descriptor — the width dim is split across mesh COLS the
// same way the height dim splits across mesh ROWS.
#define PAD_W_LO 3
#define PAD_W_HI 3
#define TILE_W ((OW_T - 1) * SP_S + SP_KW)      // 61 per-tile input cols
#define TILE_STRIDE_W (TILE_W - (SP_KW - SP_S)) // 56 (overlap = KW-S = 5)
// Mirrors SP_OHR. Unused by the kernel (it uses ow_dim = OW_T directly), kept
// for symmetry with the height descriptor.
#define SP_OWR ((TILE_W - SP_KW) / SP_S + 1)

constexpr int OC_PER_G = NUM_FILTERS / HW_COLS; // 16 output channels per mesh col

// Full output extents, as kernel-visible literals.
#define OUTPUT_FULL_H 112
#define OUTPUT_FULL_W 112
#define OUTPUT_FULL_C 64

// ═══════════════════════════════════════════════════════════════════════════
// Composition-based spatial spaces
//
// Distribution strategy (same as GEMM):
//   A (input)  : Broadcast per Row — all tiles in a mesh row see the same slab
//   B (filter) : Broadcast per Col — all tiles in a mesh col see the same filter
//   C (output) : Gather per Row    — tile outputs merged left-to-right
// ═══════════════════════════════════════════════════════════════════════════

// A-input: spatial-halo model. RowBC_spatial is a lean GemmSpace whose d1 carries
// the HEIGHT halo split AND the conv kernel window (win) + stride (win_stride);
// the compiler derives the conv tiling from d1/d2 — no Conv2dSpace needed.
//   d1 = HEIGHT halo across mesh ROWS: 61-row outer slices stepping 56 over the
//        padded height 230 -> 4 mesh rows. Each slab is chunked on-core into 4
//        L2 rounds of TILE_H=19 stepping TILE_STRIDE_H=14. Coverage:
//        (4-1)*14 + 19 = 61. Derived oh_per_row = (19-7)/2+1 = 7 == OH_T.
//   d2 = WIDTH K-accum split across mesh COLS: 61-col chunks stepping 56 over
//        the padded width 230; with the d3 channel stride this gives the routing
//        slice 244 / step 224 and a padded row pitch of 230*4 = 920.
//   d3 = CHANNEL: real cin=3 padded to 4. fullsize is the layout stride;
//        padsize records the single zero channel the host materializes.
constexpr aie::GemmSpace RowBC_spatial = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = INPUT_H + PAD_H_LO + PAD_H_HI, // 230 padded H
           .tile_round = 4,
           .tile_size = 61,                          // outer height slice (rows per mesh row)
           .stride = 56,                             // outer height step
           .slice_tiling = {.tile_size = TILE_H,     // 19 rows per on-core round
                            .stride = TILE_STRIDE_H, // 14 row step between rounds
                            .rounds = 4}},
    .d2 = {.fullsize = INPUT_W + PAD_W_LO + PAD_W_HI, // 230 padded W
           .tile_round = 4,
           .tile_size = TILE_W,       // 61 per-chunk input cols
           .stride = TILE_STRIDE_W},  // 56 width halo step
    .d3 = {.fullsize = INPUT_C_ALIGN, // 4 (layout stride; not split)
           .tile_size = INPUT_C_ALIGN,
           .padsize = INPUT_C_ALIGN - INPUT_C}}; // 1 zero pad channel (3 -> 4)

// Filter (B), described per-port as B^T[N, K] decomposed over its K factors:
//   d1 = N-tile (output channels per mesh col), d2/d3 = KW/KH, d4 = channel.
constexpr aie::GemmSpace ColBC = {
    .policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = OC_PER_G * HW_COLS, .tile_round = HW_COLS, .tile_size = OC_PER_G, .stride = OC_PER_G},
    .d2 = {.fullsize = SP_KW, .tile_size = SP_KH, .stride = SP_KW},
    .d3 = {.fullsize = SP_KH, .tile_size = SP_KH, .stride = SP_KH},
    .d4 = {.fullsize = INPUT_C_ALIGN,
           .tile_size = INPUT_C_ALIGN,
           .stride = INPUT_C_ALIGN,
           .padsize = INPUT_C_ALIGN - INPUT_C}};

// Output (C): gathered left-to-right. Must be described with the FULL output
// tile size, not the per-tile sub-split — the compiler needs total coverage.
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row,
                       .merge_order = aie::Flow::LeftToRight,
                       .mesh_tiling_group1_dim = 1 /* d1 = H, split across mesh rows */,
                       .mesh_tiling_group2_dim = 3 /* d3 = channel, split across mesh cols */},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = OUTPUT_FULL_H,
           .tile_round = 4,
           .tile_size = 28,
           .stride = 28,
           .slice_tiling = {.tile_size = 7, .stride = 7, .rounds = 4}},
    .d2 = {.fullsize = OUTPUT_FULL_W, .tile_round = 4, .tile_size = 28, .stride = 28},
    .d3 = {.fullsize = OUTPUT_FULL_C, .tile_round = 4, .tile_size = OC_PER_G},
};

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: conv2d_spatial — VECTOR version (aie::mmul<4,8,8,int8,int8>)
//
// Same three windows, same sizes, same DMA as conv2dstem.cc; only the core's
// arithmetic changes. Follows doc/performance/conv2dstem_perf_review.md §4.0:
//
//   K1  quant params decoded ONCE per launch (not 16 byte loads per output)
//   K2  each input A-tile is loaded once and feeds BOTH 8-channel N-blocks
//   K3  the MAC is aie::mmul<4,8,8> (256 int8 MACs per call) instead of the
//       scalar `MAC r` (1 MAC per 3.2 cycles)
//   K4  bias/zp/requant/ReLU/uint8 on 32-lane vectors, bit-exact (see below)
//   K5  windows copied with 32-byte vector moves into 64-byte-aligned locals;
//       output written straight into the output window (no local_out copy)
//
// ── GEMM view of one round (one slab -> one 7x28x16 output tile) ────────────
//   M = 7 oh x 28 ow  -> per oh row: 7 M-blocks of 4 consecutive ow
//   N = 16 channels   -> 2 N-blocks of 8
//   K = 7 kh x 32     -> per kh: the 7 taps x 4 ch of one output are 28
//                        CONTIGUOUS slab bytes, padded to 32 = 4 K-blocks of 8
// The A-tile (4 rows x 8 K) for M-block ow0, kernel row kh, K-block kb is
//     row m, col k  =  slab[(oh*2+kh)*PITCH + (ow0+m)*8 + kb*8 + k]
// i.e. 32 CONTIGUOUS bytes starting at (oh*2+kh)*PITCH + (ow0+kb)*8: stride 2 x
// 4 channels = 8 bytes per output column = exactly one K-block. So every A-tile
// is a single vector load, and im2col costs nothing.
//
// The 4 K-lanes beyond tap 6 (k-index 28..31) read the NEXT pixel / row pad;
// their weights are packed as zero, so they contribute nothing. Likewise the
// pad channel (k-index % 4 == 3) — which carries the quant params in the window
// — is packed as a zero weight, so no c < 3 mask is needed in the MAC loop.
//
// ── Bit-exact vector requantize ─────────────────────────────────────────────
// Reference (per channel):  v = (x*mult + 2^(s-1)) >> s,  s = shift+31, 64-bit.
// Shifts differ per channel (8..22 on the real layer) and an SRS shift is one
// scalar per vector, so it is split, exactly, into two floor shifts per N-block:
//   a  = min_n(s_n) - 1                  (scalar, every lane has b_n = s_n - a >= 1)
//   q  = floor(x*mult / 2^a)             (SRS, floor rounding, fits int32)
//   v  = floor((q*2^(B-b_n) + 2^(B-1)) / 2^B),   B = max_n b_n   (SRS again)
// floor((floor(y)+k)/m) == floor((y+k)/m) for integer k, m>0, and scaling the
// numerator and denominator by 2^(B-b_n) changes nothing, so v equals the
// reference for every input. Then clamp [0,255] == ReLU + uint8.
// ═══════════════════════════════════════════════════════════════════════════
#define VK_PITCH 256 // local slab row pitch: 61 px * 4 B = 244 rounded up to 256
#define VK_KB 4      // K-blocks of 8 per kernel row (28 bytes -> 32)
#define VK_MB 4      // mmul M (output columns per M-block)
#define VK_NB 2      // N-blocks of 8 channels (16 channels per core)

constexpr aie::GlobalPolicy conv_policy = {.fullconnect_auto = 0};
__global__(conv_policy) void conv2d_spatial(
    aie::port<input_window_int8 *, RowBC_spatial> win_a, // raw slab [TILE_H, TILE_W*C]
    aie::port<input_window_int8 *, ColBC> win_b,         // filter [K, tile_N]
    aie::port<output_window_int8 *, LtoR_Merge> win_c    // output [oh_per_row*OW_T, tile_N]
) {
    const int ow_dim = OW_T;                                            // 28
    const int oh_per_row = SP_OHR;                                      // 7
    const int out_c_num = aie::get_arg_per_round_size_in_dim(2, win_c); // 16

    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int raw_wc = TILE_W * SP_C; // 244 bytes per slab row in the window

    // floor rounding for the exact two-stage requantize; saturation is
    // harmless (every intermediate is proven in range) but makes the final
    // int32 -> uint8 narrowing a clamp rather than a wrap.
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::saturate);

    // Slab rows at a 256-byte pitch: every A-tile start is then 8-byte aligned
    // and the over-read of the last K-block stays inside the row's pad.
    alignas(64) int8_t slab[TILE_H * VK_PITCH];
    // Packed B: [kh][kb][nb] tiles of 8(K) x 8(N), row-major, zero where the
    // K-lane is past tap 6 or is the pad/param channel.
    alignas(64) int8_t Bp[SP_KH * VK_KB * VK_NB * 64];
    // Per-lane epilogue vectors for the 4x8 C tile (lane = m*8 + n).
    alignas(64) int32_t ep_bz[VK_NB][32];   // bias - zero_point
    alignas(64) int32_t ep_mult[VK_NB][32]; // fixed-point multiplier
    alignas(64) int32_t ep_p2[VK_NB][32];   // 2^(B - b_n)
    alignas(64) int32_t ep_rnd[VK_NB][32];  // 2^(B - 1)
    int ep_a[VK_NB];
    int ep_B[VK_NB];

    // ===== B (filter + params): once per launch =====
    for (int rb = 0; rb < num_b_rounds; rb++) {
        const int8_t *Bw = (const int8_t *)acquire_input_window(win_b);
        for (int kh = 0; kh < SP_KH; kh++)
            for (int kb = 0; kb < VK_KB; kb++)
                for (int nb = 0; nb < VK_NB; nb++) {
                    int8_t *t = &Bp[((kh * VK_KB + kb) * VK_NB + nb) * 64];
                    for (int k = 0; k < 8; k++) {
                        const int ki = kb * 8 + k; // byte within the kernel row (kw*4 + c)
                        const bool live = ki < SP_KW * SP_C && (ki % SP_C) < SP_REAL_C;
                        for (int n = 0; n < 8; n++)
                            t[k * 8 + n] = live ? Bw[(nb * 8 + n) * SP_K + kh * SP_KW * SP_C + ki] : (int8_t)0;
                    }
                }
        for (int nb = 0; nb < VK_NB; nb++) {
            int s[8];
            int smin = 1 << 30;
            for (int n = 0; n < 8; n++) {
                s[n] = (int)RD_PARAM_I32(&Bw[(nb * 8 + n) * SP_K], 3) + 31;
                if (s[n] < smin)
                    smin = s[n];
            }
            const int a = smin - 1;
            int B = 1;
            for (int n = 0; n < 8; n++)
                if (s[n] - a > B)
                    B = s[n] - a;
            ep_a[nb] = a;
            ep_B[nb] = B;
            for (int n = 0; n < 8; n++) {
                const int8_t *brow = &Bw[(nb * 8 + n) * SP_K];
                const int32_t bz = (int32_t)((uint32_t)RD_PARAM_I32(brow, 0) - (uint32_t)RD_PARAM_I32(brow, 1));
                const int32_t mult = RD_PARAM_I32(brow, 2);
                for (int m = 0; m < VK_MB; m++) {
                    ep_bz[nb][m * 8 + n] = bz;
                    ep_mult[nb][m * 8 + n] = mult;
                    ep_p2[nb][m * 8 + n] = (int32_t)1 << (B - (s[n] - a));
                    ep_rnd[nb][m * 8 + n] = (int32_t)1 << (B - 1);
                }
            }
        }
        release_input_window(win_b);
    }

    for (int mr = 0; mr < m_rounds; mr++) {
        // ===== A: window -> pitched local slab (32-byte vector moves) =====
        // Rows 0..TILE_H-2 copy 8 vectors (256 B: the row + 12 B of the next
        // row, landing in the pad); the LAST row copies 7 vectors + a 20-byte
        // tail so nothing is read past the window.
        for (int ra = 0; ra < num_a_rounds; ra++) {
            const int8_t *Aw = (const int8_t *)acquire_input_window(win_a);
            for (int r = 0; r < TILE_H - 1; r++)
                for (int c = 0; c < VK_PITCH / 32; c++)
                    aie::store_v(&slab[r * VK_PITCH + c * 32], aie::load_unaligned_v<32>(&Aw[r * raw_wc + c * 32], 4));
            {
                const int r = TILE_H - 1;
                for (int c = 0; c < 7; c++)
                    aie::store_v(&slab[r * VK_PITCH + c * 32], aie::load_unaligned_v<32>(&Aw[r * raw_wc + c * 32], 4));
                for (int i = 7 * 32; i < raw_wc; i++)
                    slab[r * VK_PITCH + i] = Aw[r * raw_wc + i];
            }
            release_input_window(win_a);
        }

        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            for (int oh = 0; oh < oh_per_row; oh++) {
                for (int ow0 = 0; ow0 < ow_dim; ow0 += VK_MB) {
                    const int8_t *ap = &slab[(oh * SP_S) * VK_PITCH + ow0 * 8];
                    aie::mmul<VK_MB, 8, 8, int8_t, int8_t> c0, c1;
                    {
                        const aie::vector<int8_t, 32> a = aie::load_unaligned_v<32>(ap, 8);
                        c0.mul(a, aie::load_v<64>(&Bp[0]));
                        c1.mul(a, aie::load_v<64>(&Bp[64]));
                    }
                    for (int kb = 1; kb < VK_KB; kb++)
                        chess_prepare_for_pipelining {
                            const aie::vector<int8_t, 32> a = aie::load_unaligned_v<32>(ap + kb * 8, 8);
                            c0.mac(a, aie::load_v<64>(&Bp[(kb * VK_NB + 0) * 64]));
                            c1.mac(a, aie::load_v<64>(&Bp[(kb * VK_NB + 1) * 64]));
                        }
                    for (int kh = 1; kh < SP_KH; kh++)
                        chess_prepare_for_pipelining {
                            for (int kb = 0; kb < VK_KB; kb++) {
                                const aie::vector<int8_t, 32> a =
                                    aie::load_unaligned_v<32>(ap + kh * VK_PITCH + kb * 8, 8);
                                c0.mac(a, aie::load_v<64>(&Bp[((kh * VK_KB + kb) * VK_NB + 0) * 64]));
                                c1.mac(a, aie::load_v<64>(&Bp[((kh * VK_KB + kb) * VK_NB + 1) * 64]));
                            }
                        }

                    // ===== fused epilogue, both N-blocks =====
                    aie::vector<uint8_t, 32> o[VK_NB];
                    for (int nb = 0; nb < VK_NB; nb++) {
                        aie::vector<int32_t, 32> x = (nb == 0 ? c0 : c1).template to_vector<int32_t>();
                        x = aie::add(x, aie::load_v<32>(ep_bz[nb]));
                        const aie::accum<acc64, 32> p = aie::mul(x, aie::load_v<32>(ep_mult[nb]));
                        const aie::vector<int32_t, 32> q = p.template to_vector<int32_t>(ep_a[nb]);
                        aie::accum<acc64, 32> r;
                        r.from_vector(aie::load_v<32>(ep_rnd[nb]), 0);
                        r = aie::mac(r, q, aie::load_v<32>(ep_p2[nb]));
                        aie::vector<int32_t, 32> v = r.template to_vector<int32_t>(ep_B[nb]);
                        v = aie::min(aie::max(v, (int32_t)0), (int32_t)255);
                        aie::accum<acc32, 32> n8;
                        n8.from_vector(v, 0);
                        o[nb] = n8.template to_vector<uint8_t>(0);
                    }
                    // Rows m=0..3 x [ch 0..7 | ch 8..15] -> 64 contiguous HWC bytes.
                    const auto z = aie::interleave_zip(o[0], o[1], 8);
                    uint8_t *dst = (uint8_t *)&out[(oh * ow_dim + ow0) * out_c_num];
                    aie::store_unaligned_v(dst, z.first, 4);
                    aie::store_unaligned_v(dst + 32, z.second, 4);
                }
            }
            release_output_window(win_c);
        }
    } // end m_rounds
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
// The SP_* kernel-visible literals must agree with the host geometry too — the
// kernel TU sees only those, so a mismatch here is a silent wrong-result bug.
static_assert(SP_KH == KERNEL_H && SP_KW == KERNEL_W, "SP_K* out of sync with KERNEL_*");
static_assert(SP_C == INPUT_C_ALIGN, "SP_C out of sync with INPUT_C_ALIGN");
static_assert(SP_S == STRIDE, "SP_S out of sync with STRIDE");
static_assert(SP_K == KERNEL_H * KERNEL_W * INPUT_C_ALIGN, "SP_K out of sync with KH*KW*C");
static_assert(SP_OHR == OH_T, "per-tile slab supply != kernel output-row demand");
static_assert(OH_T * HW_ROWS * 4 == OUTPUT_H, "height tiling does not cover OUTPUT_H");
static_assert(OW_T * HW_COLS == OUTPUT_W, "width tiling does not cover OUTPUT_W");

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
#define IFM_PAD_ELEMS (INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN) // 230*230*4 = 211600
// One row per output channel: K filter taps + SP_PARAM_BYTES of quant params.
#define WTS_BT_ELEMS (NUM_FILTERS * K)                // 64*196    = 12544
#define OFM_ELEMS (OUTPUT_H * OUTPUT_W * NUM_FILTERS) // 112*112*64 = 802816

static int8_t *g_ifm_pad = nullptr;       // [230,230,4] spatially + channel padded
static int8_t *g_wts_bt = nullptr;        // B^T [64,196]
static uint8_t *g_ofm = nullptr;          // [112,112,64] AIE destination (uint8 after requantize)
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
        g_ifm_pad = (int8_t *)__Runtime_Alloc(IFM_PAD_ELEMS * sizeof(int8_t));
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
    return CONV2DSTEM_OK;
}

// Scatter the caller's raw [224,224,3] IFM into the padded, channel-aligned
// staging buffer. Real pixel (h,w,c) -> ((h+PAD)*INPUT_W_PAD + (w+PAD))*4 + c.
// Channel 3 and the spatial border are left at the zero written by
// ensure_staging() — do NOT memset here, it would re-zero 211600 bytes every
// call to rewrite regions that are already zero.
static void stage_ifm(const int8_t *ifm) {
    if (g_border_dirty) {
        memset(g_ifm_pad, 0, IFM_PAD_ELEMS * sizeof(int8_t));
        g_border_dirty = 0;
    }
    for (int h = 0; h < INPUT_H; h++) {
        for (int w = 0; w < INPUT_W; w++) {
            const int src = (h * INPUT_W + w) * INPUT_C;
            const int dst = ((h + PAD) * INPUT_W_PAD + (w + PAD)) * INPUT_C_ALIGN;
            for (int c = 0; c < INPUT_C; c++)
                g_ifm_pad[dst + c] = ifm[src + c];
        }
    }
}

// Copy a caller-padded [230,230,3] IFM -- border included -- into the
// channel-aligned staging buffer. The border carries whatever the caller padded
// with (a TVM uint8 graph pads with the input zero-point, NOT zero), so it is
// rewritten on every call and the zero-border invariant stage_ifm() relies on is
// flagged as broken. Channel 3 is never written and stays zero.
static void stage_ifm_prepadded(const int8_t *ifm_pad) {
    for (int h = 0; h < INPUT_H_PAD; h++) {
        for (int w = 0; w < INPUT_W_PAD; w++) {
            const int src = (h * INPUT_W_PAD + w) * INPUT_C;
            const int dst = (h * INPUT_W_PAD + w) * INPUT_C_ALIGN;
            for (int c = 0; c < INPUT_C; c++)
                g_ifm_pad[dst + c] = ifm_pad[src + c];
        }
    }
    g_border_dirty = 1;
}

// Adopt a caller-supplied [230,230,4] IFM verbatim -- already spatially padded
// AND channel-aligned, i.e. exactly this buffer's own layout, so it is a
// straight copy with no scatter.
//
// This is what the generated fixture (data/make_image_header.py ->
// conv2dstem_image.h) emits, precisely so the board does no layout work on an
// image the host already knows the final shape of. Like
// stage_ifm_prepadded(), the border carries the caller's pad value -- the
// quantized ResNet graph pads with the input zero-point, not zero -- so the
// zero-border invariant stage_ifm() relies on is flagged broken.
//
// The 4th channel is whatever the caller put there (the fixture writes 0);
// nothing reads it, because the MAC loop stops at SP_REAL_C.
static void stage_ifm_pad4(const int8_t *ifm_pad4) {
    memcpy(g_ifm_pad, ifm_pad4, IFM_PAD_ELEMS * sizeof(int8_t));
    g_border_dirty = 1;
}

// Pack the caller's raw [64,7,7,3] filter into the B^T[N,K] layout the kernel
// indexes as B_ptr[f*K + (kh*KW + kw)*INPUT_C_ALIGN + c], and hide that
// channel's four quant params in the padding channel of taps 0..15.
//
// Mirror image of RD_PARAM_I32(): byte b of param idx goes to tap (idx*4 + b)
// at offset SP_PARAM_TAP_OFF, little-endian. Taps 16..48 keep the zero from
// ensure_staging(). Nothing reads the pad channel arithmetically -- the MAC
// loop stops at SP_REAL_C -- so this is storage the convolution cannot see.
static void stage_weights(const int8_t *wts, const conv2dstem_qparam *qp) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                const int src = ((f * KERNEL_H + kh) * KERNEL_W + kw) * INPUT_C;
                const int dst = f * K + (kh * KERNEL_W + kw) * INPUT_C_ALIGN;
                for (int c = 0; c < INPUT_C; c++)
                    g_wts_bt[dst + c] = wts[src + c];
            }
        }
        int8_t *brow = &g_wts_bt[f * K];
        const int32_t fields[4] = {qp[f].bias, qp[f].zero_point, qp[f].multiplier, qp[f].shift};
        for (int i = 0; i < 4; i++) {
            const uint32_t u = (uint32_t)fields[i];
            for (int b = 0; b < 4; b++)
                brow[(i * 4 + b) * INPUT_C_ALIGN + SP_PARAM_TAP_OFF] = (int8_t)((u >> (8 * b)) & 0xFF);
        }
    }
}

// Same destination as stage_weights(), from TVM's PACKED filter layout.
//
// TVM's x86 NCHWc schedule splits the output channels by oc_bn=4 and stores
// the filter as [oc_chunk][kh][kw][ic][oc_block] -- `oc` is the OUTER product
// oc_chunk*4 + oc_block, so a filter's 147 taps are strided 4 apart rather
// than contiguous. This library's layout is the plain [f][kh][kw][ic].
// Only the SOURCE index differs; the destination and the quant-param tail are
// byte-for-byte what stage_weights() writes, which is what lets the two be
// compared directly (the layout round-trip check in the commit that added
// this: both produce an identical g_wts_bt).
//
// The stride constants are TVM's own, read off the generated kernel:
//   oc_chunk 588 = KH*KW*INPUT_C*OC_BLOCK,  kh 84 = KW*INPUT_C*OC_BLOCK,
//   kw 12 = INPUT_C*OC_BLOCK,               ic 4 = OC_BLOCK
static void stage_weights_nchwc(const int8_t *wts, const conv2dstem_qparam *qp) {
    for (int occ = 0; occ < CONV2DSTEM_OC_CHUNK; occ++) {
        for (int ocb = 0; ocb < CONV2DSTEM_OC_BLOCK; ocb++) {
            const int f = occ * CONV2DSTEM_OC_BLOCK + ocb;
            for (int kh = 0; kh < KERNEL_H; kh++) {
                for (int kw = 0; kw < KERNEL_W; kw++) {
                    const int src = occ * (KERNEL_H * KERNEL_W * INPUT_C * CONV2DSTEM_OC_BLOCK) +
                                    kh * (KERNEL_W * INPUT_C * CONV2DSTEM_OC_BLOCK) +
                                    kw * (INPUT_C * CONV2DSTEM_OC_BLOCK) + ocb;
                    const int dst = f * K + (kh * KERNEL_W + kw) * INPUT_C_ALIGN;
                    for (int c = 0; c < INPUT_C; c++)
                        g_wts_bt[dst + c] = wts[src + c * CONV2DSTEM_OC_BLOCK];
                }
            }
            // Identical to stage_weights()'s tail: the four quant params hide
            // in the pad channel of taps 0..15, little-endian, where the MAC
            // loop (which stops at SP_REAL_C) cannot see them.
            int8_t *brow = &g_wts_bt[f * K];
            const int32_t fields[4] = {qp[f].bias, qp[f].zero_point, qp[f].multiplier, qp[f].shift};
            for (int i = 0; i < 4; i++) {
                const uint32_t u = (uint32_t)fields[i];
                for (int b = 0; b < 4; b++)
                    brow[(i * 4 + b) * INPUT_C_ALIGN + SP_PARAM_TAP_OFF] = (int8_t)((u >> (8 * b)) & 0xFF);
            }
        }
    }
}

// Scatter the AIE result -- HWC [112][112][64] -- into TVM's NCHW4c
// [16][112][112][4]. Pure index shuffle; every byte is copied exactly once.
static void collect_ofm_nchwc(uint8_t *ofm) {
    for (int occ = 0; occ < CONV2DSTEM_OC_CHUNK; occ++) {
        for (int oh = 0; oh < OUTPUT_H; oh++) {
            for (int ow = 0; ow < OUTPUT_W; ow++) {
                const int dst = ((occ * OUTPUT_H + oh) * OUTPUT_W + ow) * CONV2DSTEM_OC_BLOCK;
                const int src = (oh * OUTPUT_W + ow) * NUM_FILTERS + occ * CONV2DSTEM_OC_BLOCK;
                for (int ocb = 0; ocb < CONV2DSTEM_OC_BLOCK; ocb++)
                    ofm[dst + ocb] = g_ofm[src + ocb];
            }
        }
    }
}

void conv2d_stem_invalidate_weights(void) { g_wts_src = nullptr; }

void conv2d_stem_release(void) {
    if (g_ifm_pad) {
        free(g_ifm_pad);
        g_ifm_pad = nullptr;
    }
    if (g_wts_bt) {
        free(g_wts_bt);
        g_wts_bt = nullptr;
    }
    if (g_ofm) {
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
        stage_weights(wts, qp);
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

    // Copy the AIE result out to the caller's buffer. The AIE destination is a
    // cached staging buffer rather than `ofm` directly because the caller's
    // pointer carries no alignment guarantee, while the shim S2MM gather needs
    // the 64-byte-aligned DMA-capable allocation from __Runtime_Alloc.
    memcpy(ofm, g_ofm, OFM_ELEMS * sizeof(uint8_t));
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
int conv2d_stem_nchwc(const int8_t *ifm_pad, const int8_t *wts, const conv2dstem_qparam *qp, uint8_t *ofm) {
    if (!ifm_pad || !wts || !qp || !ofm)
        return CONV2DSTEM_ERR_NULL_ARG;

    const int rc = ensure_staging();
    if (rc != CONV2DSTEM_OK)
        return rc;

    stage_ifm_prepadded(ifm_pad);

    // Same re-pack-on-pointer-change rule as stem_run(); spelled out here
    // because the packing function differs, so this cannot call stem_run().
    if (g_wts_src != wts) {
        stage_weights_nchwc(wts, qp);
        g_wts_src = wts;
    }

    // Same five-column partition and the same control-plane opt-ins as
    // stem_run(); see the comment there for why the spare column exists. The two
    // launch sites must agree -- they describe the same physical mesh.
    conv2dstem_control_plane_init();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, HW_ROWS, HW_COLS);

    conv2d_spatial<<<mesh>>>(g_ifm_pad, g_wts_bt, g_ofm, M, N, K);

    collect_ofm_nchwc(ofm);
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
                            const int kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c;
                            const int8_t iv = ifm_pad[(ih * INPUT_W_PAD + iw) * INPUT_C_ALIGN + c];
                            const int8_t fv = wts_bt[f * K + kk];
                            acc += (int32_t)iv * (int32_t)fv;
                        }
                    }
                }
                const int8_t *brow = &wts_bt[f * K];
                acc += RD_PARAM_I32(brow, 0); // bias
                acc -= RD_PARAM_I32(brow, 1); // zero-point
                const int32_t mult = RD_PARAM_I32(brow, 2);
                const int sh = (int)RD_PARAM_I32(brow, 3);
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
