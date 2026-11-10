/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernel trace -> routing carriers. See trace_routing.h.
 ******************************************************************************/
#include "trace_routing.h"

#include <algorithm>
#include <sstream>

namespace convautotile {

namespace {

// AIE-ML core data memory every port buffer shares (bytes).
constexpr long kCoreDataBytes = 49152;

std::string vec(const std::vector<int> &v) {
    std::string s = "(";
    for (size_t i = 0; i < v.size(); i++)
        s += (i ? "," : "") + std::to_string(v[i]);
    return s + ")";
}

int at(const std::vector<int> &v, int d) { return d < (int)v.size() ? v[d] : 0; }

// Fill one port's dims from its fit; append what cannot be routed to `errs`.
void planPort(const TracePort &p, const PortFit &f, int rows, int cols, RoutingPort &rp,
              std::vector<std::string> &errs) {
    const std::string who = p.var + " (" + p.role + ")";
    rp.var = p.var;
    rp.role = p.role;
    rp.output = p.output;
    rp.elem_bytes = p.elem_bytes;
    rp.rounds_i = f.rounds_i;
    rp.rounds_j = f.rounds_j;
    if (f.rank != (int)p.shape.size()) {
        errs.push_back(who + ": blocks have " + std::to_string(f.rank) + " dims, the tensor has " +
                       std::to_string(p.shape.size()));
        return;
    }
    if (f.rank < 1 || f.rank > 4) {
        errs.push_back(who + ": a routed tensor has 1..4 dims (GemmSpace d1..d4), this one has " +
                       std::to_string(f.rank));
        return;
    }
    int loopDim[2] = {-1, -1};
    for (int d = 0; d < f.rank; d++) {
        RoutingDim r;
        r.fullsize = p.shape[d];
        if (d < (int)p.extent.size() && p.extent[d] > 0 && p.extent[d] < p.shape[d])
            r.padsize = p.shape[d] - p.extent[d];
        r.block = f.shape[d];
        const int b0 = rows > 1 ? at(f.base_p0, d) : 0, b1 = cols > 1 ? at(f.base_p1, d) : 0;
        if (b0 && b1) {
            errs.push_back(who + ": dim " + std::to_string(d) + " moves with both program_id(0) and program_id(1) " +
                           "(offsets " + vec(f.base_p0) + " / " + vec(f.base_p1) +
                           "); a mesh axis can split one tensor dim only");
            continue;
        }
        if (b0 || b1) {
            r.mesh_axis = b0 ? 0 : 1;
            r.mesh_parts = b0 ? rows : cols;
            r.mesh_step = b0 ? b0 : b1;
            if (r.mesh_step < 0)
                errs.push_back(who + ": dim " + std::to_string(d) + " walks backwards across the mesh (step " +
                               std::to_string(r.mesh_step) + ")");
        }
        const int si = f.rounds_i > 1 ? at(f.step_i, d) : 0, sj = f.rounds_j > 1 ? at(f.step_j, d) : 0;
        if (si && sj) {
            errs.push_back(who + ": dim " + std::to_string(d) + " moves in both the outer and the inner loop " +
                           "(steps " + vec(f.step_i) + " / " + vec(f.step_j) +
                           "); each loop must walk its own tensor dim");
            continue;
        }
        if (si || sj) {
            r.loop = si ? 0 : 1;
            r.rounds = si ? f.rounds_i : f.rounds_j;
            r.step = si ? si : sj;
            if (r.step < 0)
                errs.push_back(who + ": dim " + std::to_string(d) + " walks backwards in a loop (step " +
                               std::to_string(r.step) + ")");
            if (loopDim[r.loop] >= 0)
                errs.push_back(who + ": the " + (r.loop ? "inner" : "outer") + " loop moves dims " +
                               std::to_string(loopDim[r.loop]) + " and " + std::to_string(d) +
                               " together (a diagonal walk); a DMA dimension walks one tensor dim");
            else
                loopDim[r.loop] = d;
        }
        rp.dims.push_back(r);
    }
    // The DMA streams the outer loop on the outer (lower-index) tensor dim.
    if (loopDim[0] >= 0 && loopDim[1] >= 0 && loopDim[0] > loopDim[1])
        errs.push_back(who + ": blocks are visited with the outer loop on dim " + std::to_string(loopDim[0]) +
                       " and the inner loop on dim " + std::to_string(loopDim[1]) +
                       "; the DMA streams the outer tensor dim in the outer loop -- swap the two loops");
}

} // namespace

bool RoutingPort::dependsOn(int axis) const {
    for (auto &d : dims)
        if (d.mesh_axis == axis)
            return true;
    return false;
}

long RoutingPort::blockBytes() const {
    long b = elem_bytes;
    for (auto &d : dims)
        b *= d.block;
    return b;
}

RoutingPlan planRouting(const std::vector<TracePort> &ports, const std::map<std::string, PortFit> &fits, int rows,
                        int cols) {
    RoutingPlan plan;
    plan.rows = rows;
    plan.cols = cols;
    for (auto &p : ports) {
        auto it = fits.find(p.var);
        if (it == fits.end()) {
            plan.errors.push_back(p.var + " (" + p.role + "): no traced blocks to route");
            continue;
        }
        RoutingPort rp;
        planPort(p, it->second, rows, cols, rp, plan.errors);
        for (auto &d : rp.dims)
            plan.halo = plan.halo || d.halo() || d.meshHalo();
        if (!rp.output && rp.dependsOn(0) && rp.dependsOn(1))
            plan.errors.push_back(p.var + " (" + p.role + "): an input block that depends on both program ids is " +
                                  "unique per core; routing supports inputs broadcast along a mesh row " +
                                  "(depends on program_id(0) only) or a mesh col (program_id(1) only)");
        plan.ports.push_back(rp);
    }
    plan.ok = plan.errors.empty();
    return plan;
}

std::vector<PortSpace> routingSpaces(const RoutingPlan &plan, const RoutingHints &hints) {
    std::vector<PortSpace> out;
    for (auto &rp : plan.ports) {
        PortSpace s;
        s.role = rp.role;
        s.pad = hints.pad;
        s.pp_depth = hints.pp_depth;
        s.l1_budget = hints.l1_budget;
        // Halo plans read the activation through the DMA im2col path.
        s.im2col = (!rp.output && plan.halo) ? 1 : 0;
        if (rp.output) {
            // Gather: every core writes its own block, merged left to right.
            s.layout = "Row";
            s.merge_order = "LeftToRight";
            for (size_t d = 0; d < rp.dims.size(); d++) {
                if (rp.dims[d].mesh_axis == 0 && s.group1_dim < 0)
                    s.group1_dim = (int)d + 1;
                if (rp.dims[d].mesh_axis == 1 && s.group2_dim < 0)
                    s.group2_dim = (int)d + 1;
            }
        } else if (rp.dependsOn(1) && !rp.dependsOn(0)) {
            s.wgt = "Broadcast"; // same block for every core of a mesh col
            s.layout = "Col";
        } else {
            s.act = "Broadcast"; // same block for every core of a mesh row
            s.layout = "Row";
        }
        for (size_t d = 0; d < rp.dims.size() && d < 4; d++) {
            const RoutingDim &r = rp.dims[d];
            TileDimOut &t = s.d[d];
            t.used = true;
            t.fullsize = r.fullsize;
            t.padsize = r.padsize;
            if (r.mesh_axis >= 0) {
                // Mesh split; an on-core loop on the same dim tiles the core's slice.
                t.tile_round = r.mesh_parts;
                t.tile_size = r.coreSlice();
                t.stride = r.mesh_step;
                if (r.loop >= 0 && r.rounds > 1) {
                    t.l2_size = r.block;
                    t.l2_stride = r.step;
                    t.l2_rounds = r.rounds;
                }
                // Gathered output: the merge places the column groups (no stride).
                t.has_stride = !(rp.output && r.mesh_axis == 1);
            } else if (r.loop >= 0) {
                t.tile_round = r.rounds;
                t.tile_size = r.block;
                t.stride = r.step;
            } else {
                // Whole (or fixed) window, not split. A halo slab's innermost
                // dim is the packed layout dim: it carries no stride.
                t.tile_size = r.block;
                t.stride = r.block;
                t.has_stride = !(plan.halo && !rp.output && rp.dependsOn(0) && d + 1 == rp.dims.size());
            }
        }
        out.push_back(s);
    }
    return out;
}

std::string planText(const RoutingPlan &plan) {
    std::ostringstream os;
    os << "  routing plan on the " << plan.rows << "x" << plan.cols << " mesh" << (plan.halo ? " (halo)" : "") << ":\n";
    for (auto &rp : plan.ports) {
        os << "    " << rp.var << " (" << rp.role << ", "
           << (rp.output ? "gather" : rp.dependsOn(1) && !rp.dependsOn(0) ? "col broadcast" : "row broadcast")
           << ", " << rp.rounds_i << "x" << rp.rounds_j << " blocks/core):";
        for (size_t d = 0; d < rp.dims.size(); d++) {
            const RoutingDim &r = rp.dims[d];
            os << " d" << d + 1 << "[" << r.fullsize << "]=";
            if (r.mesh_axis >= 0)
                os << (r.mesh_axis ? "cols" : "rows") << "/" << r.mesh_parts << "@" << r.mesh_step;
            if (r.loop >= 0)
                os << (r.mesh_axis >= 0 ? "+" : "") << (r.loop ? "j" : "i") << r.rounds << "x" << r.block << "@"
                   << r.step;
            if (r.mesh_axis < 0 && r.loop < 0)
                os << "whole" << (r.block != r.fullsize ? "(" + std::to_string(r.block) + ")" : "");
        }
        os << "\n";
    }
    return os.str();
}

RoutingHints routingHints(const RoutingPlan &plan) {
    RoutingHints h;
    long need = 0;
    for (auto &rp : plan.ports) {
        bool slab = false; // halo slabs are sized by the halo path, not l1_budget
        for (auto &d : rp.dims)
            slab = slab || d.halo() || d.meshHalo();
        if (!slab && !(plan.halo && !rp.output && rp.dependsOn(0)))
            need = std::max(need, rp.blockBytes());
    }
    if (need > h.l1_budget)
        h.l1_budget = (int)((need + 255) / 256 * 256);
    const long n = (long)plan.ports.size();
    if ((long)h.pp_depth * h.l1_budget * n > kCoreDataBytes && (long)h.l1_budget * n <= kCoreDataBytes)
        h.pp_depth = 1;
    return h;
}

double routingEstMs(const RoutingPlan &plan, const RoutingHints &hints, const Target &tgt) {
    double bytes = 0;
    long rounds = 1;
    for (auto &rp : plan.ports) {
        bytes += (double)rp.blockBytes() * rp.rounds_i * rp.rounds_j;
        rounds = std::max(rounds, (long)rp.rounds_i * rp.rounds_j);
    }
    const double copy = bytes * tgt.copy_cycles_per_byte, dma = bytes / tgt.dma_bytes_per_cycle;
    const double cycles =
        (hints.pp_depth >= 2 ? std::max(copy, dma) : copy + dma) + (double)rounds * tgt.round_overhead_cycles;
    return cycles / (tgt.clock_mhz * 1000.0);
}

std::string routedSpaceVar(const RoutingPlan &plan, const std::string &kernel, const RoutingPort &rp) {
    int same = 0;
    for (auto &o : plan.ports)
        same += o.role == rp.role;
    return "__aie_" + kernel + "_" + (same == 1 ? rp.role : rp.var);
}

std::string routedDeclsText(const std::string &kernel, const RoutingPlan &plan, const std::vector<PortSpace> &spaces) {
    std::ostringstream os;
    os << "// ==== aiehlc trace -> routing: " << kernel << "'s ports routed from its traced load_buffer / store_buffer"
       << " blocks ====\n";
    std::istringstream text(planText(plan));
    for (std::string line; std::getline(text, line);)
        os << "//" << line << "\n";
    for (size_t i = 0; i < plan.ports.size() && i < spaces.size(); i++)
        os << emitGemmSpace(spaces[i], routedSpaceVar(plan, kernel, plan.ports[i]));
    os << "constexpr aie::GlobalPolicy __aie_" << kernel << "_policy = {.fullconnect_auto = 0};\n";
    os << "// ==== end aiehlc trace -> routing ====\n";
    return os.str();
}

std::vector<std::string> routedTUDecls(const RoutingPlan &plan, const std::map<std::string, PortFit> &fits) {
    auto list4 = [](const std::vector<int> &v) {
        std::string s = "{";
        for (int i = 0; i < 4; i++)
            s += (i ? ", " : "") + std::to_string(at(v, i));
        return s + "}";
    };
    std::vector<std::string> out{"#include \"aie_block_io.h\""};
    for (auto &rp : plan.ports) {
        auto it = fits.find(rp.var);
        if (it == fits.end())
            continue;
        const PortFit &f = it->second;
        out.push_back("static const aiehlc_blk_desc aiehlc_desc_" + rp.var + " = {" + std::to_string(f.rank) + ", " +
                      list4(f.shape) + ", " + list4(f.base_p0) + ", " + list4(f.base_p1) + ", " + list4(f.step_i) +
                      ", " + list4(f.step_j) + ", " + std::to_string(f.rounds_i) + ", " + std::to_string(f.rounds_j) +
                      "};");
        out.push_back("static aiehlc_blk_cursor aiehlc_cur_" + rp.var + ";");
    }
    return out;
}

} // namespace convautotile
