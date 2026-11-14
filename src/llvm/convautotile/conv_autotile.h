/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Conv2d auto-tiling model.
 *
 * Derives the per-port GemmSpace descriptors (d1/d2/d3/d4, halo slice/step,
 * on-core slice_tiling) and the kernel-visible tile constants of a spatial-halo
 * conv2d from a declarative description:
 *
 *   aie::Conv2d  conv = {.h, .w, .c, .k, .kh, .kw, .stride, .pad}
 *   aie::Mapping map  = {.space = {split(OH, R), split(OC, C)},     // mesh
 *                        .time  = {tile(OH, th), tile(OW, tw)},      // on-core
 *                        .hints = {.pad, .im2col, .pp_depth, .l1_budget}}
 *
 * Every halo quantity is the footprint of an output tile under the conv
 * access function ih = oh*S + kh (iw likewise):
 *     slice = (T - 1)*S + (K - 1)*D + 1      step = T*S      overlap = slice - step
 * applied twice: once for the mesh split (GemmSpace d*.tile_size/stride/
 * tile_round) and once for the on-core time tile (d*.slice_tiling).
 *
 * The model is dependency-free (std only) so it is shared by aiehlc, the
 * aiehlc-convtile CLI (tuner backend) and the unit test.
 ******************************************************************************/
#pragma once

#include <map>
#include <string>
#include <vector>

namespace convautotile {

enum class Axis { None = -1, OH = 0, OW = 1, OC = 2, IC = 3 };

const char *axisName(Axis a);
Axis axisFromName(const std::string &n); // "OH", "aie::OH", ... -> Axis (None if unknown)

struct Conv2dSpec {
    int h = 0, w = 0, c = 0; // input H, W, C (real channels)
    int k = 0;               // output channels
    int kh = 0, kw = 0;
    int stride = 1;
    int pad = 0;
    int dilation = 1;
    int c_align = 0; // channel layout stride; 0 => target rule (round up to 4)
};

struct AxisSplit {
    Axis axis = Axis::None;
    int factor = 0; // space: mesh parts; time: on-core tile size (output elements)
};

struct Hints {
    int pad = 0;          // aie::PadMaterialize (0 = DDR, 1 = Memtile)
    int im2col = 1;       // aie::Im2col (0 = None, 1 = Dma)
    int pp_depth = 2;     // ping-pong depth
    int l1_budget = 4096; // per-buffer hint carried into the policy (maxBufferBytes)
};

struct MappingSpec {
    AxisSplit space_rows;        // mesh rows split
    AxisSplit space_cols;        // mesh cols split
    std::vector<AxisSplit> time; // empty => autotune
    Hints hints;
};

// Target facts the legality check needs. Defaults match the AIE-ML core tile the
// tilinglinalg pipeline targets ("Buffer memory check ... / 49152 bytes").
struct Target {
    int data_mem_budget = 49152; // bytes of core data memory usable for port buffers
    // Measured from the generated bcf: data memory = 32 KB stack (0x70000) + a
    // 32 KB region for the port buffers (0x78000..0x80000), whose last 2 KB are
    // the klog ring; kernel statics (block descriptors, cursors) share it. A
    // buffer set past it fails at link: "could not find free space for
    // SpaceSymbol buf_in_ping_0 in memory DMb". kernel.cc declares ping AND pong
    // for every window, so this holds for 2 x (ifm + wgt + ofm) at any pp_depth.
    int buffer_region = 32768;
    int buffer_reserve = 3072;   // klog 2 KB + statics
    int channel_align = 4;       // int8 channel layout granularity (C=3 -> 4)
    int elem_bytes = 1;          // int8 activations / weights / outputs
    // Core stack (bcf `_stack DM_stack 0x70000 0x8000`). The reference kernels
    // keep a local copy of each per-round port buffer on the stack, so their
    // frame is ifm + wgt + ofm + kernel_stack_fixed (11200 B measured in
    // kernel.calltree for tile(OH,7) x tile(OW,28)). 0 disables the check.
    // Capabilities of the downstream tilinglinalg spatial-halo path (measured):
    //  * w_rounds == 1 falls back to the legacy height-only path and aborts with
    //    "conv spatial-halo M-tile mismatch";
    //  * non-halo ports (OFM, WEIGHTS) are clamped to hints.l1_budget
    //    (maxPingPongBytes) and chunked into ceil(bytes/l1_budget) rounds of
    //    l1_budget bytes, which overruns unless the buffer fits.
    bool require_width_split = true;
    bool non_halo_ports_fit_l1_budget = true;
    int stack_budget = 32768;
    int stack_headroom = 1024; // skill aiecorestackoverflow: keep >= ~1 KB spare
    int kernel_stack_fixed = 320;
    // Analytical cost model constants, calibrated on the scalar reference conv
    // kernel (doc/performance/conv2dstem_perf_review.md: 68 cycles / 21 MACs,
    // 7 cycles per copied byte; AIE clock 998 MHz). Only used to rank candidates
    // and as the tuner's "analytic" runner; the board runner measures instead.
    double macs_per_cycle = 21.0 / 68.0;
    double copy_cycles_per_byte = 7.0;    // core-side window copies (0 for zero-copy kernels)
    double dma_bytes_per_cycle = 2.3;     // ~2.3 GB/s per shim MM2S channel at ~1 GHz
    double round_overhead_cycles = 600.0; // lock acquire/release + BD/round bookkeeping
    double clock_mhz = 998.0;
};

// One GemmSpace tile_dim, mirroring the aiehlc stub (field order contractual).
struct TileDimOut {
    int fullsize = 0, tile_round = 0, tile_size = 0, stride = 0, padsize = 0;
    int l2_size = 0, l2_stride = 0, l2_rounds = 0; // slice_tiling (0 => none)
    bool has_stride = true;                        // emit .stride
    bool used = false;
};

struct PortSpace {
    std::string role; // IFM / WEIGHTS / OFM
    // SpatialPolicy
    std::string act, wgt, layout, merge_order;
    int group1_dim = -1, group2_dim = -1;
    int pad = 0, im2col = 1, pp_depth = 2, l1_budget = 4096;
    TileDimOut d[4];
};

struct Derived {
    bool ok = false;
    std::vector<std::string> errors;
    std::vector<std::string> notes;

    // Geometry
    int oh = 0, ow = 0, c_align = 0, h_pad = 0, w_pad = 0, k_red = 0;
    int mesh_rows = 0, mesh_cols = 0;
    int oh_mesh = 0, oc_mesh = 0; // output rows / channels per core
    int oh_t = 0, ow_t = 0;       // on-core time tile
    int tile_h = 0, tile_w = 0;   // input slab per round (halo slice)
    int tile_step_h = 0, tile_step_w = 0;
    int mesh_slice_h = 0, mesh_step_h = 0; // input rows per mesh row
    int h_rounds = 0, w_rounds = 0, rounds = 0;
    bool time_autotuned = false;

    PortSpace ifm, weights, ofm;

    // Footprint / cost features (per core)
    long ifm_round_bytes = 0, wgt_bytes = 0, ofm_round_bytes = 0;
    long l1_bytes = 0;    // pp_depth * (ifm + wgt + ofm), as the pipeline checks it
    long stack_bytes = 0; // kernel frame estimate (local window copies)
    long dma_bytes = 0;
    long macs = 0;
    double halo_factor = 0; // IFM bytes moved / IFM bytes strictly needed
    double est_cycles = 0;
    double est_ms = 0;

    // Kernel-visible constants (#define NAME value), in emission order.
    std::vector<std::pair<std::string, long>> kernel_defines;
};

// Overrides applied on top of the user's Mapping (tuner knobs).
struct Overrides {
    int time_oh = 0, time_ow = 0, pp_depth = 0, c_align = 0, l1_budget = 0;
    bool any() const { return time_oh || time_ow || pp_depth || c_align || l1_budget; }
};

// Parse a flat JSON object of integer knobs, e.g. {"time_oh": 14, "time_ow": 28}.
// Keys: time_oh, time_ow, pp_depth, l1_budget, c_align.
// Unknown keys are reported in *err (non-fatal); returns false on syntax error.
bool parseOverridesJson(const std::string &text, Overrides &ov, std::string *err);

// Derive everything for one (conv, mapping). When mapping.time is empty the
// best legal time tile under the cost model is chosen (autotune).
Derived derive(const Conv2dSpec &conv, MappingSpec mapping, const Target &tgt = Target(),
               const Overrides &ov = Overrides());

// All legal candidates (time_oh x time_ow x pp_depth) for the given space split,
// ranked by the analytical cost model (best first).
std::vector<Derived> enumerateCandidates(const Conv2dSpec &conv, const MappingSpec &mapping,
                                         const Target &tgt = Target());

// Divisors of n (ascending).
std::vector<int> divisors(int n);

// Emission
std::string emitGemmSpace(const PortSpace &p, const std::string &varName);
std::string emitKernelDefines(const Derived &d, const std::vector<std::string> &skipNames);
std::string toJson(const Derived &d, const std::string &indent = "");
std::string toJson(const Conv2dSpec &c);

} // namespace convautotile
