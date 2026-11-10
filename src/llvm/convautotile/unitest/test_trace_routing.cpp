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
static void testConvEquivalence(int th, int tw) {
    Conv2dSpec c;
    c.h = c.w = 230;
    c.c = c.c_align = 4;
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
    // Rendered text equal too (padsize 0 on both sides here).
    CHECK(emitGemmSpace(sp[0], "X") == emitGemmSpace(d.ifm, "X"));
    CHECK(emitGemmSpace(sp[1], "X") == emitGemmSpace(d.weights, "X"));
    CHECK(emitGemmSpace(sp[2], "X") == emitGemmSpace(d.ofm, "X"));
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

int main(int argc, char **argv) {
    std::string repo = argc > 1 ? argv[1] : AIEHLC_SOURCE_DIR;
    testConvEquivalence(7, 28);
    testConvEquivalence(14, 16);
    testConvEquivalence(28, 8);
    testGemm();
    testEltwise();
    testRejects();
    testTraceExample(repo);
    std::printf("trace_routing: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
