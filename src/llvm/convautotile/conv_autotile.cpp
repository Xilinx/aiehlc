/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Conv2d auto-tiling model — see conv_autotile.h.
 ******************************************************************************/
#include "conv_autotile.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace convautotile {

const char *axisName(Axis a) {
    switch (a) {
    case Axis::OH:
        return "OH";
    case Axis::OW:
        return "OW";
    case Axis::OC:
        return "OC";
    case Axis::IC:
        return "IC";
    default:
        return "None";
    }
}

Axis axisFromName(const std::string &n) {
    std::string s = n;
    size_t p = s.rfind("::");
    if (p != std::string::npos)
        s = s.substr(p + 2);
    if (s == "OH")
        return Axis::OH;
    if (s == "OW")
        return Axis::OW;
    if (s == "OC")
        return Axis::OC;
    if (s == "IC")
        return Axis::IC;
    return Axis::None;
}

std::vector<int> divisors(int n) {
    std::vector<int> r;
    for (int i = 1; i <= n; i++)
        if (n % i == 0)
            r.push_back(i);
    return r;
}

static int roundUp(int v, int a) { return a <= 1 ? v : (v + a - 1) / a * a; }

// Footprint of T consecutive outputs along one axis: (T-1)*S + (K-1)*D + 1.
static int footprint(int t, int s, int k, int d) { return (t - 1) * s + (k - 1) * d + 1; }

static void fillCost(Derived &d, const Hints &h, const Target &tgt) {
    const long eb = tgt.elem_bytes;
    d.ifm_round_bytes = (long)d.tile_h * d.tile_w * d.c_align * eb;
    d.wgt_bytes = (long)d.k_red * d.oc_mesh * eb;
    d.ofm_round_bytes = (long)d.oh_t * d.ow_t * d.oc_mesh * eb;
    d.l1_bytes = (long)h.pp_depth * (d.ifm_round_bytes + d.wgt_bytes + d.ofm_round_bytes);
    d.dma_bytes = (long)d.rounds * (d.ifm_round_bytes + d.ofm_round_bytes) + d.wgt_bytes;
    d.macs = (long)d.oh_mesh * d.ow * d.oc_mesh * d.k_red;
    // 1.0 == every byte of this core's input footprint is fetched exactly once.
    long footprintBytes = (long)d.mesh_slice_h * ((d.w_rounds - 1) * (long)d.tile_step_w + d.tile_w) * d.c_align * eb;
    d.halo_factor = footprintBytes > 0 ? (double)d.rounds * d.ifm_round_bytes / footprintBytes : 0.0;
    d.stack_bytes = d.ifm_round_bytes + d.wgt_bytes + d.ofm_round_bytes + tgt.kernel_stack_fixed;
    // Core time = MACs + its own window copies + per-round bookkeeping; DMA runs
    // in parallel when double-buffered, otherwise it is exposed.
    double compute = d.macs / tgt.macs_per_cycle + d.dma_bytes * tgt.copy_cycles_per_byte;
    double dma = d.dma_bytes / tgt.dma_bytes_per_cycle;
    double overhead = d.rounds * tgt.round_overhead_cycles;
    d.est_cycles = (h.pp_depth >= 2 ? std::max(compute, dma) : compute + dma) + overhead;
    d.est_ms = d.est_cycles / (tgt.clock_mhz * 1000.0);
}

static void buildPorts(Derived &d, const Conv2dSpec &c, const Hints &h) {
    auto policy = [&](PortSpace &p) {
        p.pad = h.pad;
        p.pp_depth = h.pp_depth;
        p.l1_budget = h.l1_budget;
    };
    // IFM: raw overlapping slabs, broadcast along each mesh row.
    PortSpace &a = d.ifm;
    a.role = "IFM";
    policy(a);
    a.act = "Broadcast";
    a.layout = "Row";
    a.im2col = h.im2col;
    TileDimOut &a1 = a.d[0]; // HEIGHT halo across mesh rows (+ on-core rounds)
    a1.used = true;
    a1.fullsize = d.h_pad;
    a1.tile_round = d.mesh_rows;
    a1.tile_size = d.mesh_slice_h;
    a1.stride = d.mesh_step_h;
    if (d.h_rounds > 1) {
        a1.l2_size = d.tile_h;
        a1.l2_stride = d.tile_step_h;
        a1.l2_rounds = d.h_rounds;
    }
    TileDimOut &a2 = a.d[1]; // WIDTH halo, on-core chunks
    a2.used = true;
    a2.fullsize = d.w_pad;
    a2.tile_round = d.w_rounds;
    a2.tile_size = d.tile_w;
    a2.stride = d.tile_step_w;
    TileDimOut &a3 = a.d[2]; // CHANNEL (not split; padded layout stride)
    a3.used = true;
    a3.fullsize = d.c_align;
    a3.tile_size = d.c_align;
    a3.padsize = d.c_align - c.c;
    a3.has_stride = false;

    // WEIGHTS: B^T [OC, KH, KW, C], OC split across mesh cols, broadcast per col.
    PortSpace &b = d.weights;
    b.role = "WEIGHTS";
    policy(b);
    b.wgt = "Broadcast";
    b.layout = "Col";
    b.im2col = h.im2col;
    TileDimOut &b1 = b.d[0];
    b1.used = true;
    b1.fullsize = c.k;
    b1.tile_round = d.mesh_cols;
    b1.tile_size = d.oc_mesh;
    b1.stride = d.oc_mesh;
    TileDimOut &b2 = b.d[1]; // KW
    b2.used = true;
    b2.fullsize = b2.tile_size = b2.stride = c.kw;
    TileDimOut &b3 = b.d[2]; // KH
    b3.used = true;
    b3.fullsize = b3.tile_size = b3.stride = c.kh;
    TileDimOut &b4 = b.d[3]; // C (padded)
    b4.used = true;
    b4.fullsize = b4.tile_size = b4.stride = d.c_align;
    b4.padsize = d.c_align - c.c;

    // OFM: [OH, OW, OC] gathered per mesh row, merged left to right.
    PortSpace &o = d.ofm;
    o.role = "OFM";
    policy(o);
    o.layout = "Row";
    o.merge_order = "LeftToRight";
    o.group1_dim = 1; // d1 = OH split across mesh rows
    o.group2_dim = 3; // d3 = OC split across mesh cols
    o.im2col = 0;
    TileDimOut &o1 = o.d[0];
    o1.used = true;
    o1.fullsize = d.oh;
    o1.tile_round = d.mesh_rows;
    o1.tile_size = d.oh_mesh;
    o1.stride = d.oh_mesh;
    if (d.h_rounds > 1) {
        o1.l2_size = d.oh_t;
        o1.l2_stride = d.oh_t;
        o1.l2_rounds = d.h_rounds;
    }
    TileDimOut &o2 = o.d[1];
    o2.used = true;
    o2.fullsize = d.ow;
    o2.tile_round = d.w_rounds;
    o2.tile_size = d.ow_t;
    o2.stride = d.ow_t;
    TileDimOut &o3 = o.d[2];
    o3.used = true;
    o3.fullsize = c.k;
    o3.tile_round = d.mesh_cols;
    o3.tile_size = d.oc_mesh;
    o3.has_stride = false;
}

static void buildKernelDefines(Derived &d, const Conv2dSpec &c) {
    auto &k = d.kernel_defines;
    k.clear();
    k.push_back({"SP_KH", c.kh});
    k.push_back({"SP_KW", c.kw});
    k.push_back({"SP_C", d.c_align});
    k.push_back({"SP_S", c.stride});
    k.push_back({"SP_K", d.k_red});
    k.push_back({"SP_OW", d.ow});
    k.push_back({"PAD_H", c.pad});
    k.push_back({"PAD_W", c.pad});
    k.push_back({"OH_T", d.oh_t});
    k.push_back({"OW_T", d.ow_t});
    k.push_back({"SP_OHR", d.oh_t});
    k.push_back({"SP_OWR", d.ow_t});
    k.push_back({"TILE_H", d.tile_h});
    k.push_back({"TILE_W", d.tile_w});
    k.push_back({"TILE_STRIDE_H", d.tile_step_h});
    k.push_back({"TILE_STRIDE_W", d.tile_step_w});
    k.push_back({"OUTPUT_FULL_H", d.oh});
    k.push_back({"OUTPUT_FULL_W", d.ow});
    k.push_back({"OUTPUT_FULL_C", c.k});
}

// Derivation for a fully specified time tile (oh_t/ow_t already resolved).
static Derived deriveFixed(const Conv2dSpec &c, const MappingSpec &m, const Target &tgt, int c_align, int oh_t,
                           int ow_t) {
    Derived d;
    auto err = [&](const std::string &s) { d.errors.push_back(s); };
    const int D = c.dilation > 0 ? c.dilation : 1;
    const int ekh = (c.kh - 1) * D + 1, ekw = (c.kw - 1) * D + 1;
    d.c_align = c_align;
    d.h_pad = c.h + 2 * c.pad;
    d.w_pad = c.w + 2 * c.pad;
    d.k_red = c.kh * c.kw * c_align;
    d.oh = (d.h_pad - ekh) / c.stride + 1;
    d.ow = (d.w_pad - ekw) / c.stride + 1;
    d.mesh_rows = m.space_rows.factor;
    d.mesh_cols = m.space_cols.factor;
    if (d.mesh_rows <= 0 || d.oh % d.mesh_rows) {
        err("space: OH=" + std::to_string(d.oh) + " is not divisible by the mesh-row split " +
            std::to_string(d.mesh_rows) + " (tails are not supported by the spatial-halo path)");
        return d;
    }
    if (d.mesh_cols <= 0 || c.k % d.mesh_cols) {
        err("space: OC=" + std::to_string(c.k) + " is not divisible by the mesh-col split " +
            std::to_string(d.mesh_cols));
        return d;
    }
    d.oh_mesh = d.oh / d.mesh_rows;
    d.oc_mesh = c.k / d.mesh_cols;
    d.oh_t = oh_t;
    d.ow_t = ow_t;
    if (oh_t <= 0 || d.oh_mesh % oh_t) {
        err("time: tile(OH," + std::to_string(oh_t) + ") must divide the per-core output rows " +
            std::to_string(d.oh_mesh));
        return d;
    }
    if (ow_t <= 0 || d.ow % ow_t) {
        err("time: tile(OW," + std::to_string(ow_t) + ") must divide OW=" + std::to_string(d.ow));
        return d;
    }
    d.h_rounds = d.oh_mesh / oh_t;
    d.w_rounds = d.ow / ow_t;
    d.rounds = d.h_rounds * d.w_rounds;
    d.tile_h = footprint(oh_t, c.stride, c.kh, D);
    d.tile_step_h = oh_t * c.stride;
    d.tile_w = footprint(ow_t, c.stride, c.kw, D);
    d.tile_step_w = ow_t * c.stride;
    d.mesh_slice_h = footprint(d.oh_mesh, c.stride, c.kh, D);
    d.mesh_step_h = d.oh_mesh * c.stride;
    // Every read must stay inside the (DDR pre-padded) input.
    int lastRow = (d.mesh_rows - 1) * d.mesh_step_h + d.mesh_slice_h;
    if (lastRow > d.h_pad)
        err("halo: last mesh row reads input row " + std::to_string(lastRow) + " > padded H " +
            std::to_string(d.h_pad));
    int lastCol = (d.w_rounds - 1) * d.tile_step_w + d.tile_w;
    if (lastCol > d.w_pad)
        err("halo: last width chunk reads input col " + std::to_string(lastCol) + " > padded W " +
            std::to_string(d.w_pad));
    if ((d.h_rounds - 1) * d.tile_step_h + d.tile_h != d.mesh_slice_h)
        err("halo: on-core rounds do not tile the mesh slice exactly");
    Hints h = m.hints;
    buildPorts(d, c, h);
    buildKernelDefines(d, c);
    fillCost(d, h, tgt);
    if (d.l1_bytes > tgt.data_mem_budget)
        err("L1: pp_depth*(ifm+wgt+ofm) = " + std::to_string(d.l1_bytes) + " B exceeds the core data budget " +
            std::to_string(tgt.data_mem_budget) + " B");
    if (tgt.stack_budget > 0 && d.stack_bytes > tgt.stack_budget - tgt.stack_headroom)
        err("stack: kernel frame ~" + std::to_string(d.stack_bytes) + " B (local window copies) exceeds the " +
            std::to_string(tgt.stack_budget) + " B core stack minus " + std::to_string(tgt.stack_headroom) +
            " B headroom");
    if (tgt.require_width_split && d.w_rounds < 2)
        err("pipeline: the spatial-halo conv path needs tile(OW, n) with n < OW=" + std::to_string(d.ow) +
            " (>= 2 on-core width rounds); one width round hits 'conv spatial-halo M-tile mismatch'");
    // The generated kernel.cc declares a ping AND a pong array for every window
    // whatever pp_depth says, so the link needs both.
    const long linkBytes = 2L * (d.ifm_round_bytes + d.wgt_bytes + d.ofm_round_bytes);
    if (linkBytes + tgt.buffer_reserve > tgt.buffer_region)
        err("link: 2 x (ifm+wgt+ofm) = " + std::to_string(linkBytes) + " B (ping + pong are always declared) + " +
            std::to_string(tgt.buffer_reserve) + " B (klog, statics) exceeds the " + std::to_string(tgt.buffer_region) +
            " B port-buffer region of the core (the rest is stack)");
    // The pipeline's own check (TilingLinalg "Memory budget exceeded"): every
    // tensor's ping-pong buffers are sized at hints.l1_budget.
    const long pipeBytes = (long)h.pp_depth * h.l1_budget * 3;
    if (pipeBytes > tgt.data_mem_budget)
        err("pipeline: pp_depth x l1_budget x 3 tensors = " + std::to_string(pipeBytes) +
            " B exceeds the core data budget " + std::to_string(tgt.data_mem_budget) +
            " B (the pipeline sizes every buffer at l1_budget) -- smaller blocks or more mesh columns");
    if (tgt.non_halo_ports_fit_l1_budget) {
        if (d.ofm_round_bytes > h.l1_budget)
            err("pipeline: OFM tile " + std::to_string(d.ofm_round_bytes) +
                " B (oh_t*ow_t*oc) exceeds hints.l1_budget " + std::to_string(h.l1_budget) +
                " B; non-halo port buffers are clamped to l1_budget and chunked "
                "unevenly -- raise .l1_budget or shrink the time tile");
        if (d.wgt_bytes > h.l1_budget)
            err("pipeline: WEIGHTS tile " + std::to_string(d.wgt_bytes) + " B exceeds hints.l1_budget " +
                std::to_string(h.l1_budget) + " B");
    }
    if (d.ifm_round_bytes > h.l1_budget)
        d.notes.push_back("IFM halo slab " + std::to_string(d.ifm_round_bytes) + " B is above hints.l1_budget " +
                          std::to_string(h.l1_budget) + " B (fine: the halo port is shipped whole, exempt from it)");
    d.ok = d.errors.empty();
    return d;
}

static bool validateSpec(const Conv2dSpec &c, const MappingSpec &m, std::vector<std::string> &errs) {
    if (c.h <= 0 || c.w <= 0 || c.c <= 0 || c.k <= 0 || c.kh <= 0 || c.kw <= 0 || c.stride <= 0 || c.pad < 0)
        errs.push_back("conv: h, w, c, k, kh, kw, stride must be > 0 and pad >= 0");
    if (m.space_rows.axis != Axis::OH)
        errs.push_back(std::string("space: mesh rows must split OH (got ") + axisName(m.space_rows.axis) +
                       "); the spatial-halo path carries the halo on the row axis");
    if (m.space_cols.axis != Axis::OC)
        errs.push_back(std::string("space: mesh cols must split OC (got ") + axisName(m.space_cols.axis) +
                       "); IFM is broadcast along a mesh row and filters are split by column");
    for (auto &t : m.time)
        if (t.axis != Axis::OH && t.axis != Axis::OW)
            errs.push_back(std::string("time: only tile(OH, n) and tile(OW, n) are supported (got ") +
                           axisName(t.axis) + ")");
    if (m.hints.pad != 0)
        errs.push_back("hints: only PadMaterialize::DDR is supported by the spatial-halo path");
    if (m.hints.pp_depth < 1)
        errs.push_back("hints: pp_depth must be >= 1");
    return errs.empty();
}

static int resolveCAlign(const Conv2dSpec &c, const Target &tgt, const Overrides &ov) {
    if (ov.c_align > 0)
        return ov.c_align;
    if (c.c_align > 0)
        return c.c_align;
    return roundUp(c.c, tgt.channel_align);
}

std::vector<Derived> enumerateCandidates(const Conv2dSpec &conv, const MappingSpec &mapping, const Target &tgt) {
    std::vector<Derived> out;
    std::vector<std::string> errs;
    if (!validateSpec(conv, mapping, errs))
        return out;
    Derived probe = deriveFixed(conv, mapping, tgt, resolveCAlign(conv, tgt, Overrides()), 1, 1);
    if (probe.oh_mesh <= 0)
        return out;
    int ca = resolveCAlign(conv, tgt, Overrides());
    for (int th : divisors(probe.oh_mesh))
        for (int tw : divisors(probe.ow)) {
            Derived d = deriveFixed(conv, mapping, tgt, ca, th, tw);
            if (d.ok)
                out.push_back(std::move(d));
        }
    std::stable_sort(out.begin(), out.end(), [](const Derived &a, const Derived &b) {
        if (a.est_cycles != b.est_cycles)
            return a.est_cycles < b.est_cycles;
        return a.l1_bytes < b.l1_bytes;
    });
    return out;
}

Derived derive(const Conv2dSpec &conv, MappingSpec m, const Target &tgt, const Overrides &ov) {
    Derived bad;
    if (!validateSpec(conv, m, bad.errors))
        return bad;
    if (ov.pp_depth > 0)
        m.hints.pp_depth = ov.pp_depth;
    if (ov.l1_budget > 0)
        m.hints.l1_budget = ov.l1_budget;
    int ca = resolveCAlign(conv, tgt, ov);
    if (ca < conv.c) {
        bad.errors.push_back("c_align " + std::to_string(ca) + " < C " + std::to_string(conv.c));
        return bad;
    }
    int th = 0, tw = 0;
    for (auto &t : m.time) {
        if (t.axis == Axis::OH)
            th = t.factor;
        if (t.axis == Axis::OW)
            tw = t.factor;
    }
    if (ov.time_oh > 0)
        th = ov.time_oh;
    if (ov.time_ow > 0)
        tw = ov.time_ow;
    bool autotune = m.time.empty() && !ov.time_oh && !ov.time_ow;
    if (autotune) {
        Conv2dSpec c2 = conv;
        c2.c_align = ca;
        auto cands = enumerateCandidates(c2, m, tgt);
        if (cands.empty()) {
            Derived d = deriveFixed(conv, m, tgt, ca, 1, 1);
            d.ok = false;
            d.errors.push_back("autotune: no legal time tile fits the core data budget");
            return d;
        }
        Derived d = cands.front();
        d.time_autotuned = true;
        d.notes.push_back("time tile autotuned by the analytical model: tile(OH," + std::to_string(d.oh_t) +
                          "), tile(OW," + std::to_string(d.ow_t) + ") out of " + std::to_string(cands.size()) +
                          " legal candidates");
        return d;
    }
    // Partially specified time: an omitted axis is not split on-core.
    Derived probe = deriveFixed(conv, m, tgt, ca, 1, 1);
    if (probe.oh_mesh <= 0)
        return probe;
    if (th <= 0)
        th = probe.oh_mesh;
    if (tw <= 0)
        tw = probe.ow;
    return deriveFixed(conv, m, tgt, ca, th, tw);
}

// ---------------------------------------------------------------------------
// Overrides JSON (flat object of integers)
// ---------------------------------------------------------------------------
bool parseOverridesJson(const std::string &text, Overrides &ov, std::string *err) {
    size_t i = 0, n = text.size();
    auto ws = [&]() {
        while (i < n && std::isspace((unsigned char)text[i]))
            i++;
    };
    auto fail = [&](const std::string &m) {
        if (err)
            *err = m + " at offset " + std::to_string(i);
        return false;
    };
    ws();
    if (i >= n || text[i] != '{')
        return fail("expected '{'");
    i++;
    ws();
    if (i < n && text[i] == '}')
        return true;
    while (i < n) {
        ws();
        if (text[i] != '"')
            return fail("expected key string");
        size_t e = text.find('"', i + 1);
        if (e == std::string::npos)
            return fail("unterminated key");
        std::string key = text.substr(i + 1, e - i - 1);
        i = e + 1;
        ws();
        if (i >= n || text[i] != ':')
            return fail("expected ':'");
        i++;
        ws();
        size_t s = i;
        if (i < n && (text[i] == '-' || text[i] == '+'))
            i++;
        while (i < n && std::isdigit((unsigned char)text[i]))
            i++;
        if (s == i)
            return fail("expected integer value for '" + key + "'");
        int v = (int)std::strtol(text.substr(s, i - s).c_str(), nullptr, 10);
        if (key == "time_oh")
            ov.time_oh = v;
        else if (key == "time_ow")
            ov.time_ow = v;
        else if (key == "pp_depth")
            ov.pp_depth = v;
        else if (key == "c_align")
            ov.c_align = v;
        else if (key == "l1_budget")
            ov.l1_budget = v;
        else if (err)
            *err += (err->empty() ? "" : "; ") + std::string("ignored unknown knob '") + key + "'";
        ws();
        if (i < n && text[i] == ',') {
            i++;
            continue;
        }
        if (i < n && text[i] == '}')
            return true;
        return fail("expected ',' or '}'");
    }
    return fail("unterminated object");
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------
static std::string emitTileDim(const TileDimOut &t) {
    std::ostringstream os;
    os << "{.fullsize = " << t.fullsize;
    if (t.tile_round)
        os << ", .tile_round = " << t.tile_round;
    os << ", .tile_size = " << t.tile_size;
    if (t.has_stride)
        os << ", .stride = " << t.stride;
    if (t.padsize)
        os << ", .padsize = " << t.padsize;
    if (t.l2_rounds > 1)
        os << ", .slice_tiling = {.tile_size = " << t.l2_size << ", .stride = " << t.l2_stride
           << ", .rounds = " << t.l2_rounds << "}";
    os << "}";
    return os.str();
}

std::string emitGemmSpace(const PortSpace &p, const std::string &varName) {
    std::ostringstream os;
    os << "constexpr aie::GemmSpace " << varName << " = {\n";
    os << "    .policy = {.map = {";
    std::vector<std::string> mf;
    if (!p.act.empty())
        mf.push_back(".act = aie::Pattern::" + p.act);
    if (!p.wgt.empty())
        mf.push_back(".wgt = aie::Pattern::" + p.wgt);
    if (!p.layout.empty())
        mf.push_back(".layout = aie::Layout::" + p.layout);
    if (!p.merge_order.empty())
        mf.push_back(".merge_order = aie::Flow::" + p.merge_order);
    if (p.group1_dim >= 0)
        mf.push_back(".mesh_tiling_group1_dim = " + std::to_string(p.group1_dim));
    if (p.group2_dim >= 0)
        mf.push_back(".mesh_tiling_group2_dim = " + std::to_string(p.group2_dim));
    for (size_t i = 0; i < mf.size(); i++)
        os << (i ? ", " : "") << mf[i];
    os << "},\n";
    os << "               .mat = {.pad = aie::PadMaterialize::" << (p.pad ? "Memtile" : "DDR")
       << ", .im2col = aie::Im2col::" << (p.im2col ? "Dma" : "None") << "},\n";
    os << "               .sched = {.pp_depth = " << p.pp_depth << ", .l1_budget = aie::Bytes{" << p.l1_budget << "}}}";
    static const char *names[4] = {"d1", "d2", "d3", "d4"};
    for (int i = 0; i < 4; i++)
        if (p.d[i].used)
            os << ",\n    ." << names[i] << " = " << emitTileDim(p.d[i]);
    os << "};\n";
    return os.str();
}

std::string emitKernelDefines(const Derived &d, const std::vector<std::string> &skipNames) {
    std::ostringstream os;
    for (auto &kv : d.kernel_defines) {
        if (std::find(skipNames.begin(), skipNames.end(), kv.first) != skipNames.end())
            continue;
        os << "#define " << kv.first << " " << kv.second << "\n";
    }
    return os.str();
}

static std::string jsonStr(const std::string &s) {
    std::string r = "\"";
    for (char ch : s) {
        if (ch == '\n')
            r += "\\n";
        else if (ch == '\t')
            r += "\\t";
        else if ((unsigned char)ch < 0x20)
            r += ' ';
        else {
            if (ch == '"' || ch == '\\')
                r += '\\';
            r += ch;
        }
    }
    return r + "\"";
}

static std::string jsonTileDim(const TileDimOut &t) {
    std::ostringstream os;
    os << "{\"fullsize\": " << t.fullsize << ", \"tile_round\": " << t.tile_round << ", \"tile_size\": " << t.tile_size
       << ", \"stride\": " << (t.has_stride ? t.stride : 0) << ", \"padsize\": " << t.padsize
       << ", \"slice_tiling\": {\"tile_size\": " << t.l2_size << ", \"stride\": " << t.l2_stride
       << ", \"rounds\": " << t.l2_rounds << "}}";
    return os.str();
}

static std::string jsonPort(const PortSpace &p, const std::string &ind) {
    std::ostringstream os;
    os << "{\"role\": " << jsonStr(p.role);
    static const char *names[4] = {"d1", "d2", "d3", "d4"};
    for (int i = 0; i < 4; i++)
        if (p.d[i].used)
            os << ",\n" << ind << "  " << jsonStr(names[i]) << ": " << jsonTileDim(p.d[i]);
    os << "}";
    return os.str();
}

std::string toJson(const Conv2dSpec &c) {
    std::ostringstream os;
    os << "{\"h\": " << c.h << ", \"w\": " << c.w << ", \"c\": " << c.c << ", \"k\": " << c.k << ", \"kh\": " << c.kh
       << ", \"kw\": " << c.kw << ", \"stride\": " << c.stride << ", \"pad\": " << c.pad
       << ", \"dilation\": " << c.dilation << ", \"c_align\": " << c.c_align << "}";
    return os.str();
}

std::string toJson(const Derived &d, const std::string &ind) {
    std::ostringstream os;
    const std::string i1 = ind + "  ";
    os << "{\n";
    os << i1 << "\"ok\": " << (d.ok ? "true" : "false") << ",\n";
    os << i1 << "\"errors\": [";
    for (size_t i = 0; i < d.errors.size(); i++)
        os << (i ? ", " : "") << jsonStr(d.errors[i]);
    os << "],\n" << i1 << "\"notes\": [";
    for (size_t i = 0; i < d.notes.size(); i++)
        os << (i ? ", " : "") << jsonStr(d.notes[i]);
    os << "],\n";
    os << i1 << "\"knobs\": {\"time_oh\": " << d.oh_t << ", \"time_ow\": " << d.ow_t
       << ", \"pp_depth\": " << d.ifm.pp_depth << ", \"l1_budget\": " << d.ifm.l1_budget
       << ", \"c_align\": " << d.c_align << ", \"autotuned\": " << (d.time_autotuned ? "true" : "false") << "},\n";
    os << i1 << "\"geometry\": {\"oh\": " << d.oh << ", \"ow\": " << d.ow << ", \"h_pad\": " << d.h_pad
       << ", \"w_pad\": " << d.w_pad << ", \"k_red\": " << d.k_red << ", \"mesh_rows\": " << d.mesh_rows
       << ", \"mesh_cols\": " << d.mesh_cols << ", \"oh_mesh\": " << d.oh_mesh << ", \"oc_mesh\": " << d.oc_mesh
       << ", \"mesh_slice_h\": " << d.mesh_slice_h << ", \"mesh_step_h\": " << d.mesh_step_h
       << ", \"tile_h\": " << d.tile_h << ", \"tile_step_h\": " << d.tile_step_h << ", \"tile_w\": " << d.tile_w
       << ", \"tile_step_w\": " << d.tile_step_w << ", \"h_rounds\": " << d.h_rounds << ", \"w_rounds\": " << d.w_rounds
       << ", \"rounds\": " << d.rounds << "},\n";
    os << i1 << "\"features\": {\"ifm_round_bytes\": " << d.ifm_round_bytes << ", \"wgt_bytes\": " << d.wgt_bytes
       << ", \"ofm_round_bytes\": " << d.ofm_round_bytes << ", \"l1_bytes\": " << d.l1_bytes
       << ", \"dma_bytes\": " << d.dma_bytes << ", \"macs\": " << d.macs << ", \"halo_factor\": " << d.halo_factor
       << ", \"stack_bytes\": " << d.stack_bytes << ", \"est_cycles\": " << d.est_cycles << ", \"est_ms\": " << d.est_ms
       << "},\n";
    os << i1 << "\"ports\": {\n";
    os << i1 << "  \"IFM\": " << jsonPort(d.ifm, i1 + "  ") << ",\n";
    os << i1 << "  \"WEIGHTS\": " << jsonPort(d.weights, i1 + "  ") << ",\n";
    os << i1 << "  \"OFM\": " << jsonPort(d.ofm, i1 + "  ") << "},\n";
    os << i1 << "\"kernel_defines\": {";
    for (size_t i = 0; i < d.kernel_defines.size(); i++)
        os << (i ? ", " : "") << jsonStr(d.kernel_defines[i].first) << ": " << d.kernel_defines[i].second;
    os << "}\n" << ind << "}";
    return os.str();
}

} // namespace convautotile
