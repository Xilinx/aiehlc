/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernel trace -> routing: map the traced block sequences of a kernel straight
 * onto the routing IR tiling carriers, with no operator recognition.
 *
 * kernel_trace.cpp fits every port's blocks to
 *     offset = base_p0*p0 + base_p1*p1 + step_i*i + step_j*j,  one shape
 * (PortFit). Read per tensor dimension, that fit already IS a routing
 * description:
 *
 *   - which mesh axis splits the dim (base_p0 / base_p1 != 0) and its step,
 *   - which on-core loop walks it (step_i / step_j != 0), its step and rounds,
 *   - the block extent, and so the halo (block > step: overlapping blocks),
 *   - per port: broadcast along a mesh row (blocks independent of p1),
 *     along a mesh col (independent of p0), or a gather (output).
 *
 * planRouting() builds that per-dim RoutingPlan and checks it is expressible
 * as a DMA (one mesh axis and one loop per dim, outer loop on an outer dim,
 * ...); routingSpaces() renders it as the per-port GemmSpace tile_dims
 * (d1..d4, slice_tiling) + SpatialPolicy that the routing IR consumes
 * (#routing.tiling / #routing.halo / #routing.kernel_tiling,
 * doc/design/routing_ir_tiling_carriers.md). Nothing here knows "conv" or
 * "gemm": a conv2d (halo slabs, OC split), a GEMM (M rows / N cols broadcast,
 * K chunks) or an eltwise op are just different fits.
 *
 * Design: doc/design/trace_to_routing.md.
 ******************************************************************************/
#pragma once

#include "conv_autotile.h"
#include "kernel_trace.h"

#include <string>
#include <vector>

namespace convautotile {

// One tensor dimension of one port, as the trace moves it.
struct RoutingDim {
    int fullsize = 0;   // tensor extent
    int block = 0;      // block extent (the DMA window in this dim)
    int mesh_axis = -1; // -1 none, 0 = program_id(0) (mesh rows), 1 = program_id(1) (mesh cols)
    int mesh_parts = 0; // mesh rows / cols along mesh_axis
    int mesh_step = 0;  // offset between neighbouring cores
    int loop = -1;      // -1 none, 0 = outer loop (i), 1 = inner loop (j)
    int rounds = 1;     // rounds of that loop
    int step = 0;       // offset between consecutive rounds
    // Extent one core covers: (rounds - 1) * step + block.
    int coreSlice() const { return loop >= 0 ? (rounds - 1) * step + block : block; }
    // Consecutive blocks overlap (a halo): the DMA re-reads block - step.
    bool halo() const { return loop >= 0 && rounds > 1 && block > step; }
    bool meshHalo() const { return mesh_axis >= 0 && coreSlice() > mesh_step; }
};

struct RoutingPort {
    std::string var, role;
    bool output = false;
    int elem_bytes = 1;
    std::vector<RoutingDim> dims;
    int rounds_i = 1, rounds_j = 1; // blocks per core = rounds_i * rounds_j
    // Which mesh axes the port's blocks depend on (any dim).
    bool dependsOn(int axis) const;
    long blockBytes() const;
};

struct RoutingPlan {
    bool ok = true;
    std::vector<std::string> errors;
    int rows = 0, cols = 0;
    bool halo = false; // some port has overlapping blocks (spatial halo path)
    std::vector<RoutingPort> ports;
};

// Per-dim routing plan of a traced kernel. `ports` / `fits` as kernel_trace
// produced them (fits by port variable).
RoutingPlan planRouting(const std::vector<TracePort> &ports, const std::map<std::string, PortFit> &fits, int rows,
                        int cols);

// Scheduling hints for the carriers (defaults = the GemmSpace defaults the
// conv derivation uses).
struct RoutingHints {
    int pp_depth = 2;
    int l1_budget = 4096;
    int pad = 0; // aie::PadMaterialize::DDR
};

// The plan as per-port routing carriers (GemmSpace tile_dims + SpatialPolicy),
// in port order.
std::vector<PortSpace> routingSpaces(const RoutingPlan &plan, const RoutingHints &hints = RoutingHints());

// One line per port: how each dim is routed.
std::string planText(const RoutingPlan &plan);

} // namespace convautotile
