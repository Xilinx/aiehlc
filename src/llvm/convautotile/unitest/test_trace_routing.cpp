/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Trace -> routing carriers (trace_routing.h): the operator-agnostic mapping
 * must reproduce the conv derivation's GemmSpace ports exactly, and route a
 * GEMM (and an eltwise op) from the same rules.
 *
 * Build + run:  src/llvm/convautotile/unitest/run.sh
 ******************************************************************************/
#include "../conv_desugar.h"
#include "../kernel_trace.h"
#include "../trace_routing.h"

#include <cstdio>
#include <fstream>
#include <sstream>

using namespace convautotile;

static int g_fail = 0, g_pass = 0;
#define CHECK_EQ(a, b)                                                                                                 \
    do {                                                                                                               \
        long _a = (long)(a), _b = (long)(b);                                                                           \
        if (_a != _b) {                                                                                                \
            std::printf("FAIL %s:%d  %s = %ld, expected %ld\n", __FILE__, __LINE__, #a, _a, _b);                       \
            g_fail++;                                                                                                  \
        } else                                                                                                         \
            g_pass++;                                                                                                  \
    } while (0)
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c);                                                   \
            g_fail++;                                                                                                  \
        } else                                                                                                         \
            g_pass++;                                                                                                  \
    } while (0)

static TracePort port(const std::string &var, const std::string &role, std::vector<int> shape, bool out) {
    TracePort p;
    p.var = var;
    p.role = role;
    p.shape = shape;
    p.output = out;
    return p;
}

static PortFit fit(std::vector<int> shape, std::vector<int> p0, std::vector<int> p1, int ri, std::vector<int> si,
                   int rj, std::vector<int> sj) {
    PortFit f;
    f.rank = (int)shape.size();
    f.shape = shape;
    f.base_p0 = p0;
    f.base_p1 = p1;
    f.rounds_i = ri;
    f.step_i = si;
    f.rounds_j = rj;
    f.step_j = sj;
    return f;
}

// Every carrier field except padsize (the trace sees the padded tensor only;
// the kernel-only conv has c == c_align, so the derivation's padsize is 0 too).
static void checkSameSpace(const PortSpace &got, const PortSpace &want) {
    CHECK(got.act == want.act);
    CHECK(got.wgt == want.wgt);
    CHECK(got.layout == want.layout);
    CHECK(got.merge_order == want.merge_order);
    CHECK_EQ(got.group1_dim, want.group1_dim);
    CHECK_EQ(got.group2_dim, want.group2_dim);
    CHECK_EQ(got.im2col, want.im2col);
    for (int i = 0; i < 4; i++) {
        const TileDimOut &g = got.d[i], &w = want.d[i];
        CHECK_EQ(g.used, w.used);
        CHECK_EQ(g.fullsize, w.fullsize);
        CHECK_EQ(g.tile_round, w.tile_round);
        CHECK_EQ(g.tile_size, w.tile_size);
        CHECK_EQ(g.has_stride, w.has_stride);
        if (w.has_stride)
            CHECK_EQ(g.stride, w.stride);
        CHECK_EQ(g.l2_size, w.l2_size);
        CHECK_EQ(g.l2_stride, w.l2_stride);
        CHECK_EQ(g.l2_rounds, w.l2_rounds);
    }
}

// The conv derivation's DMA, read back as fits (what a correct kernel traces
// to), must route to the very same carriers.
// `ch` < 4: the channels are padded to c_align = 4 (simpleconv2d_new_load.cc
// has C = 3); the ports then carry the data extent and the carriers .padsize.
// Round trip of the hand-written path: every rendered carrier, parsed back as
// a hand-written GemmSpace, must give no difference (sourceGemmSpace and
// spaceDiff agree with emitGemmSpace on every field and default).
static void checkRoundTrip(const std::vector<PortSpace> &sp) {
    for (auto &s : sp) {
        PortSpace back;
        std::string err;
        CHECK(sourceGemmSpace(emitGemmSpace(s, "S"), "", "S", back, err));
        if (!err.empty())
            std::printf("  parse error: %s\n", err.c_str());
        std::vector<std::string> diff = spaceDiff(back, s);
        CHECK(diff.empty());
        for (auto &d : diff)
            std::printf("  round-trip diff: %s\n", d.c_str());
    }
}

static void testConvEquivalence(int th, int tw, int ch = 4) {
    Conv2dSpec c;
    c.h = c.w = 230;
    c.c = ch;
    c.c_align = 4;
    c.k = 64;
    c.kh = c.kw = 7;
    c.stride = 2;
    MappingSpec m;
    m.space_rows = {Axis::OH, 4};
    m.space_cols = {Axis::OC, 4};
    m.time = {{Axis::OH, th}, {Axis::OW, tw}};
    Derived d = derive(c, m);
    CHECK(d.ok);
    if (!d.ok)
        return;
    std::vector<TracePort> ports = {port("win_a", "IFM", {d.h_pad, d.w_pad, d.c_align}, false),
                                    port("win_b", "WEIGHTS", {c.k, c.kh, c.kw, d.c_align}, false),
                                    port("win_c", "OFM", {d.oh, d.ow, c.k}, true)};
    if (c.c < d.c_align) {
        ports[0].extent = {d.h_pad, d.w_pad, c.c};
        ports[1].extent = {c.k, c.kh, c.kw, c.c};
    }
    std::map<std::string, PortFit> fits;
    fits["win_a"] = fit({d.tile_h, d.tile_w, d.c_align}, {d.mesh_step_h, 0, 0}, {0, 0, 0}, d.h_rounds,
                        {d.tile_step_h, 0, 0}, d.w_rounds, {0, d.tile_step_w, 0});
    fits["win_b"] = fit({d.oc_mesh, c.kh, c.kw, d.c_align}, {0, 0, 0, 0}, {d.oc_mesh, 0, 0, 0}, 1, {0, 0, 0, 0}, 1,
                        {0, 0, 0, 0});
    fits["win_c"] = fit({d.oh_t, d.ow_t, d.oc_mesh}, {d.oh_mesh, 0, 0}, {0, 0, d.oc_mesh}, d.h_rounds,
                        {d.oh_t, 0, 0}, d.w_rounds, {0, d.ow_t, 0});
    RoutingPlan plan = planRouting(ports, fits, 4, 4);
    CHECK(plan.ok);
    CHECK(plan.halo);
    for (auto &e : plan.errors)
        std::printf("  plan error: %s\n", e.c_str());
    std::vector<PortSpace> sp = routingSpaces(plan);
    CHECK_EQ(sp.size(), 3);
    if (sp.size() != 3)
        return;
    checkSameSpace(sp[0], d.ifm);
    checkSameSpace(sp[1], d.weights);
    checkSameSpace(sp[2], d.ofm);
    CHECK(routedDeclsText("k", plan, sp).find("fullconnect_auto = 0") != std::string::npos); // halo plan
    CHECK_EQ(sp[0].d[2].padsize, d.ifm.d[2].padsize);
    CHECK_EQ(sp[1].d[3].padsize, d.weights.d[3].padsize);
    CHECK_EQ(d.ifm.d[2].padsize, d.c_align - c.c);
    // Rendered text equal too (padsize included).
    CHECK(emitGemmSpace(sp[0], "X") == emitGemmSpace(d.ifm, "X"));
    CHECK(emitGemmSpace(sp[1], "X") == emitGemmSpace(d.weights, "X"));
    CHECK(emitGemmSpace(sp[2], "X") == emitGemmSpace(d.ofm, "X"));
    checkRoundTrip(sp);
}

// GEMM C[M,N] = A[M,K] * B[N,K]^T on a 4x4 mesh: A rows over mesh rows, B rows
// over mesh cols, K in 4 chunks (inner loop), C block per core.
static void testGemm() {
    const int M = 64, N = 64, K = 256;
    std::vector<TracePort> ports = {port("win_a", "IFM", {M, K}, false), port("win_b", "WEIGHTS", {N, K}, false),
                                    port("win_c", "OFM", {M, N}, true)};
    std::map<std::string, PortFit> fits;
    fits["win_a"] = fit({16, 64}, {16, 0}, {0, 0}, 1, {0, 0}, 4, {0, 64});
    fits["win_b"] = fit({16, 64}, {0, 0}, {16, 0}, 1, {0, 0}, 4, {0, 64});
    fits["win_c"] = fit({16, 16}, {16, 0}, {0, 16}, 1, {0, 0}, 1, {0, 0});
    RoutingPlan plan = planRouting(ports, fits, 4, 4);
    CHECK(plan.ok);
    CHECK(!plan.halo);
    std::vector<PortSpace> sp = routingSpaces(plan);
    CHECK_EQ(sp.size(), 3);
    if (sp.size() != 3)
        return;
    // A: RowBA (act Broadcast, Row); d1 M split, d2 K chunks.
    CHECK(sp[0].act == "Broadcast" && sp[0].layout == "Row" && sp[0].wgt.empty());
    CHECK_EQ(sp[0].im2col, 0);
    CHECK_EQ(sp[0].d[0].fullsize, M);
    CHECK_EQ(sp[0].d[0].tile_round, 4);
    CHECK_EQ(sp[0].d[0].tile_size, 16);
    CHECK_EQ(sp[0].d[0].stride, 16);
    CHECK_EQ(sp[0].d[1].fullsize, K);
    CHECK_EQ(sp[0].d[1].tile_round, 4);
    CHECK_EQ(sp[0].d[1].tile_size, 64);
    CHECK_EQ(sp[0].d[1].stride, 64);
    CHECK(!sp[0].d[2].used);
    // B: ColBB (wgt Broadcast, Col); d1 N split, d2 K chunks.
    CHECK(sp[1].wgt == "Broadcast" && sp[1].layout == "Col" && sp[1].act.empty());
    CHECK_EQ(sp[1].d[0].tile_round, 4);
    CHECK_EQ(sp[1].d[0].tile_size, 16);
    CHECK_EQ(sp[1].d[1].tile_size, 64);
    // C: gather, M over rows (group1 = d1), N over cols (group2 = d2).
    CHECK(sp[2].merge_order == "LeftToRight" && sp[2].layout == "Row");
    CHECK_EQ(sp[2].group1_dim, 1);
    CHECK_EQ(sp[2].group2_dim, 2);
    CHECK_EQ(sp[2].d[0].tile_size, 16);
    CHECK_EQ(sp[2].d[1].tile_size, 16);
    checkRoundTrip(sp);
    // No halo: the GEMM fullconnect schedule (with 0 the kernel-only GEMM hung on the board).
    CHECK(routedDeclsText("k", plan, sp).find("fullconnect_auto = 1") != std::string::npos);
}

// Eltwise on a 4x1 mesh: rows split, 4 column chunks per core, no halo.
static void testEltwise() {
    std::vector<TracePort> ports = {port("x", "IFM", {64, 128}, false), port("y", "OFM", {64, 128}, true)};
    std::map<std::string, PortFit> fits;
    fits["x"] = fit({16, 32}, {16, 0}, {0, 0}, 1, {0, 0}, 4, {0, 32});
    fits["y"] = fit({16, 32}, {16, 0}, {0, 0}, 1, {0, 0}, 4, {0, 32});
    RoutingPlan plan = planRouting(ports, fits, 4, 1);
    CHECK(plan.ok);
    std::vector<PortSpace> sp = routingSpaces(plan);
    CHECK_EQ(sp.size(), 2);
    if (sp.size() == 2) {
        CHECK_EQ(sp[1].group1_dim, 1);
        CHECK_EQ(sp[1].group2_dim, -1);
        CHECK_EQ(sp[1].d[1].tile_round, 4);
    }
    checkRoundTrip(sp);
}

// What the routing cannot express is named.
static void testRejects() {
    std::vector<TracePort> ports = {port("x", "IFM", {64, 64}, false), port("y", "OFM", {64, 64}, true)};
    std::map<std::string, PortFit> fits;
    // Input unique per core (depends on both program ids).
    fits["x"] = fit({16, 16}, {16, 0}, {0, 16}, 1, {0, 0}, 1, {0, 0});
    fits["y"] = fit({16, 16}, {16, 0}, {0, 16}, 1, {0, 0}, 1, {0, 0});
    RoutingPlan p1 = planRouting(ports, fits, 4, 4);
    CHECK(!p1.ok);
    // Loops swapped: outer loop on dim 1, inner on dim 0.
    fits["x"] = fit({16, 16}, {0, 0}, {0, 0}, 4, {0, 16}, 4, {16, 0});
    fits["y"] = fit({16, 16}, {0, 0}, {0, 0}, 4, {0, 16}, 4, {16, 0});
    RoutingPlan p2 = planRouting(ports, fits, 1, 1);
    CHECK(!p2.ok);
    CHECK(!p2.errors.empty() && p2.errors[0].find("swap the two loops") != std::string::npos);
    // Diagonal walk.
    fits["x"] = fit({16, 16}, {0, 0}, {0, 0}, 4, {16, 16}, 1, {0, 0});
    RoutingPlan p3 = planRouting(ports, fits, 1, 1);
    CHECK(!p3.ok);
}

// End to end: trace the kernel-only conv example on x86 and route its fits.
static void testTraceExample(const std::string &repo) {
    const std::string dir = repo + "/example/tileprogram/ccode";
    std::ifstream f(dir + "/simpleconv2dkernel_port.cc");
    std::stringstream ss;
    ss << f.rdbuf();
    std::string src = ss.str();
    CHECK(!src.empty());
    std::string err;
    std::vector<TraceKernel> ks = findPlainKernels(src, dir, err);
    CHECK(err.empty() && ks.size() == 1);
    if (ks.size() != 1)
        return;
    auto mesh = sourceLaunchMesh(src, dir);
    KernelTraceResult tr = traceKernel(src, ks[0], mesh.first, mesh.second, traceDefines(src), "port.cc");
    CHECK(tr.errors.empty());
    for (auto &e : tr.errors)
        std::printf("  trace error: %s\n", e.c_str());
    RoutingPlan plan = planRouting(ks[0].ports, tr.fits, mesh.first, mesh.second);
    CHECK(plan.ok);
    std::printf("%s", planText(plan).c_str());
    std::vector<PortSpace> sp = routingSpaces(plan);
    CHECK_EQ(sp.size(), 3);
    if (sp.size() == 3) {
        CHECK_EQ(sp[0].d[0].tile_size, 61);
        CHECK_EQ(sp[0].d[0].l2_size, 19);
        CHECK_EQ(sp[2].d[1].tile_round, 4);
    }
}

// Hand-written GemmSpace parsing: defaults, #define / constexpr int
// arithmetic, and fail-closed errors.
static void testSpaceParse() {
    const std::string src = "#define H 112\n"
                            "constexpr int PARTS = 4;\n"
                            "constexpr aie::GemmSpace A = {\n"
                            "    .policy = {.map = {.layout = aie::Layout::Col}},\n"
                            "    .d1 = {.fullsize = H * 2, .tile_round = PARTS, .tile_size = H / 2, .stride = 56,\n"
                            "           .slice_tiling = {.tile_size = 19, .stride = 14, .rounds = PARTS}},\n"
                            "    .d2 = {.fullsize = 4, .tile_size = 4, .padsize = 1}};\n"
                            "constexpr aie::GemmSpace M = {.m = {.fullsize = 64}};\n"
                            "constexpr aie::GemmSpace U = {.d1 = {.fullsize = 8, .win = 7}};\n"
                            "constexpr aie::GemmSpace V = {.d1 = {.fullsize = runtime_value}};\n";
    PortSpace a;
    std::string err;
    CHECK(sourceGemmSpace(src, "", "A", a, err));
    CHECK(a.act == "Broadcast" && a.wgt == "Broadcast" && a.layout == "Col" && a.merge_order == "Default");
    CHECK_EQ(a.im2col, 0);
    CHECK_EQ(a.pp_depth, 2);
    CHECK_EQ(a.l1_budget, 32 * 1024);
    CHECK_EQ(a.d[0].fullsize, 224);
    CHECK_EQ(a.d[0].tile_round, 4);
    CHECK_EQ(a.d[0].tile_size, 56);
    CHECK_EQ(a.d[0].l2_rounds, 4);
    CHECK(a.d[0].has_stride && !a.d[1].has_stride);
    CHECK_EQ(a.d[1].padsize, 1);
    CHECK(a.d[1].used && !a.d[2].used);
    PortSpace x;
    err.clear();
    CHECK(!sourceGemmSpace(src, "", "M", x, err) && err.find("uses .m (the GEMM form)") != std::string::npos);
    err.clear();
    CHECK(!sourceGemmSpace(src, "", "U", x, err) && err.find("unknown field .d1.win") != std::string::npos);
    err.clear();
    CHECK(!sourceGemmSpace(src, "", "V", x, err) && err.find("not an integer constant") != std::string::npos);
    err.clear();
    CHECK(!sourceGemmSpace(src, "", "Nope", x, err) &&
          err.find("no `constexpr aie::GemmSpace Nope") != std::string::npos);
}

// spaceDiff names each DMA-shaping field and ignores the scheduling ones.
static void testSpaceDiff() {
    PortSpace t;
    t.act = "Broadcast";
    t.layout = "Row";
    t.im2col = 1;
    t.d[0].used = true;
    t.d[0].fullsize = 230;
    t.d[0].tile_round = 4;
    t.d[0].tile_size = 61;
    t.d[0].stride = 56;
    t.d[0].l2_size = 19;
    t.d[0].l2_stride = 14;
    t.d[0].l2_rounds = 4;
    PortSpace h = t;
    h.wgt = "Broadcast"; // a default: no difference
    h.pp_depth = 1;      // scheduling: not compared
    h.l1_budget = 8192;
    h.pad = 1;
    CHECK(spaceDiff(h, t).empty());
    h.d[0].l2_stride = 13;
    std::vector<std::string> d = spaceDiff(h, t);
    CHECK_EQ(d.size(), 1);
    CHECK(!d.empty() && d[0] == ".d1.slice_tiling.stride = 13, the kernel's blocks give 14");
    h = t;
    h.layout = "Col";
    h.d[1].used = true;
    h.d[1].fullsize = 4;
    d = spaceDiff(h, t);
    CHECK_EQ(d.size(), 2);
    CHECK(d.size() == 2 && d[0] == ".policy.map.layout = Col, the kernel's blocks give Row");
    CHECK(d.size() == 2 && d[1] == ".d2 is set, but the kernel's blocks have 1 dims");
    h = t;
    h.d[0].has_stride = false; // omitted .stride is 0, not "whatever the trace says"
    d = spaceDiff(h, t);
    CHECK(d.size() == 1 && d[0] == ".d1.stride = 0, the kernel's blocks give 56");
}

// The hand-written example: verified, and its block descriptors are exactly the
// kernel-only example's (same kernel body, same DMA).
static void testSpaceKernel(const std::string &repo) {
    const std::string dir = repo + "/example/tileprogram/ccode";
    auto read = [&dir](const std::string &f) {
        std::ifstream in(dir + "/" + f);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    };
    std::string hand = read("simpleconv2dkernel_space.cc"), port = read("simpleconv2dkernel_port.cc");
    CHECK(!hand.empty() && !port.empty());
    KernelOnlyResult hs = verifySpaceKernels(hand, dir, "space.cc");
    CHECK(hs.found && hs.ok);
    for (auto &e : hs.errors)
        std::printf("  space error: %s\n", e.c_str());
    KernelOnlyResult ko = inferKernelOnly(port, dir, "port.cc");
    CHECK(ko.found && ko.ok);
    CHECK(hs.tu_decls.count("conv2d_spatial") && ko.tu_decls.count("conv2d_spatial"));
    CHECK(hs.tu_decls["conv2d_spatial"] == ko.tu_decls["conv2d_spatial"]);
    CHECK(hs.derived.count("conv2d_spatial") == 1);
    CHECK(hand.find("aie::num_programs") == std::string::npos); // resolved in place
    // Not applicable: the kernel-only source has no GemmSpace ports.
    std::string again = read("simpleconv2dkernel_port.cc");
    CHECK(!verifySpaceKernels(again, dir, "port.cc").found);
    // inferKernelOnly rewrote `port` into the GemmSpace form: checked as if
    // hand-written it must pass (a round trip on the real path), and aiehlc
    // skips it as already routed.
    std::string rewritten = port;
    KernelOnlyResult rt = verifySpaceKernels(rewritten, dir, "port.cc");
    CHECK(rt.found && rt.ok);
    CHECK(rt.tu_decls["conv2d_spatial"] == ko.tu_decls["conv2d_spatial"]);
    CHECK(!verifySpaceKernels(port, dir, "port.cc", {"conv2d_spatial"}).found);
}

int main(int argc, char **argv) {
    std::string repo = argc > 1 ? argv[1] : AIEHLC_SOURCE_DIR;
    testConvEquivalence(7, 28);
    testConvEquivalence(14, 16);
    testConvEquivalence(28, 8);
    testConvEquivalence(7, 28, 3);
    testConvEquivalence(14, 16, 3);
    testGemm();
    testEltwise();
    testRejects();
    testTraceExample(repo);
    testSpaceParse();
    testSpaceDiff();
    testSpaceKernel(repo);
    std::printf("trace_routing: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
