#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Kernel-analysis regression: each case mutates a conv example into one logic
bug and checks `aiehlc-convtile trace` rejects it with the expected diagnostic.
CASES use the kernel-only port form (simpleconv2dkernel_port.cc), TENSOR_CASES
the tensor form (simpleconv2dkernel.cc: aie::in / aie::out, .pad, .reuse,
block views, aie::autotune), DATAFLOW_CASES the {.depth / .resident / .tile}
options (simplegemm_dataflow.cc), SPACE_CASES hand-written GemmSpaces + block
loads (simpleconv2dkernel_space.cc: a GemmSpace and the kernel's blocks must agree).

    python3 src/llvm/convautotile/unitest/kernel_trace_cases.py [build/aiehlc-convtile]
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[4]
CCODE = REPO / "example" / "tileprogram" / "ccode"
BASE = CCODE / "simpleconv2dkernel_port.cc"
TENSOR_BASE = CCODE / "simpleconv2dkernel.cc"
GEMM_BASE = pathlib.Path(__file__).resolve().parent / "gemm_kernel_port.cc"
SPACE_BASE = CCODE / "simpleconv2dkernel_space.cc"
LOAD_T = "aie::load_buffer(x_t, {oh0 * S - P, ow0 * S - P, 0}, {IH_T, IW_T, C}"
LOAD_X = "const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});"

# (name, [(old, new), ...], expected substring in the output; None = must pass)
CASES = [
    ("baseline", [], None),
    # A slab one row taller than the conv footprint is still an affine block
    # walk inside the tensor: the routing follows the kernel (20-row halo slabs).
    ("load bigger than the footprint (routed as is)",
     [(LOAD_X, LOAD_X.replace("{ih_t, iw_t, C}", "{ih_t + 1, iw_t, C}"))],
     None),
    # A slab narrower than the conv footprint makes the compute read past it:
    # caught first as data used without being loaded (at the exact line).
    ("load too small",
     [(LOAD_X, LOAD_X.replace("{ih_t, iw_t, C}", "{ih_t, iw_t - 2, C}"))],
     "data used without being loaded"),
    ("weights offset out of the tensor (pid_c * oc_t * 20)",
     [("{pid_c * oc_t, 0, 0, 0}", "{pid_c * oc_t * 20, 0, 0, 0}")],
     "outside the tensor extent 64"),
    ("data used without load (reads past the block)",
     [("const int8_t *r = x + (oh * S + kh) * ROW;", "const int8_t *r = x + (oh * S + kh + 1) * ROW;")],
     "data used without being loaded"),
    ("data used after release",
     [("            int8_t *y = aie::store_buffer(", "            aie::release(win_a);\n            int8_t *y = aie::store_buffer("),
      ("            aie::release(win_a);\n            aie::commit(win_c);", "            aie::commit(win_c);")],
     "uses a block after aie::release"),
    ("output never committed",
     [("            aie::commit(win_c);\n", "")],
     "never committed"),
    ("loop order swapped (j outer)",
     [("for (int i = 0; i < oh_core / oh_t; i++) {", "@OUTER@"),
      ("for (int j = 0; j < 112 / ow_t; j++) {", "for (int i = 0; i < oh_core / oh_t; i++) {"),
      ("@OUTER@", "for (int j = 0; j < 112 / ow_t; j++) {")],
     "swap the two loops"),
    ("schedule depends on data",
     [("            aie::release(win_a);\n", "            if (x[0] > 0) { aie::release(win_a); x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C}); }\n            aie::release(win_a);\n")],
     "depends on the data"),
    ("output not fully written",
     [("for (int j = 0; j < 112 / ow_t; j++) {", "for (int j = 0; j < 112 / ow_t - 1; j++) {")],
     "never stored"),
    ("weights never loaded",
     [("    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C});",
       "    static int8_t w[16 * 196];"),
      ("    aie::release(win_b);\n", "")],
     "is never loaded"),
    ("store on an input port",
     [("int8_t *y = aie::store_buffer(win_c,", "int8_t *y = aie::store_buffer(win_a,")],
     "aie::store_buffer on an input port"),
    # Vector kernel (aie::mmul, emulated by aie_api_x86emu.h in the trace).
    ("vector: last K-block not shifted back (64-byte load runs past the row)",
     [("const int off = kb < 3 ? kb * 8 : 20;", "const int off = kb * 8;")],
     "data used without being loaded"),
    ("vector: aligned 64-byte load on a 32-byte aligned window",
     [("aie::load_unaligned_v<64>(r + owA * 8 + off, 4)", "aie::load_v<64>(r + owA * 8 + off)")],
     "needs a 64-byte aligned address (DMA windows are only 32-byte aligned"),
    # Static bounds analysis (kernel_static.cpp): the block is sized by oc_t
    # but the compute walks 16 output channels whatever oc_t is -- the trace
    # names the line, the static analysis the constant and the fix, and the
    # OFM store the trace never reached (it stops at the first fault).
    ("weights block narrower than the compute (oc_t = 15): the cause",
     [("oc_t = 16;", "oc_t = 15;")],
     "cause (static analysis): `oc_t = 15` (case.cc:8) sizes the block of the access at line 29, whose index never "
     "reads oc_t -- with oc_t = 16 every analysed block access fits"),
    ("weights block narrower than the compute (oc_t = 15): the store the trace did not reach",
     [("oc_t = 16;", "oc_t = 15;")],
     "out-of-block access (static analysis) at case.cc:74: `aie::store_v(o + 96, z.second.extract<32>(1))`"),
    # Cross-check: the trace faults on a line the static analysis cannot explain.
    ("cross-check: the index reads data (static analysis cannot explain it)",
     [("live ? w[(nb * 8 + n) * TAPS + kh * KW * C + byte] : (int8_t)0;",
       "live ? w[(nb * 8 + n) * TAPS + kh * KW * C + byte + Bp[0] * 0 + 3136] : (int8_t)0;")],
     "static analysis: cannot explain line 29: `w[(nb * 8 + n) * TAPS + kh * KW * C + byte + Bp[0] * 0 + 3136]`: "
     "`Bp[...]` is not a const array with an initializer"),
    # ... and on a line it evaluated as in bounds: `w` moved through an alias
    # the model does not follow -- it must say so, not stay silent.
    ("cross-check: an alias moves the block pointer (static analysis disagrees)",
     [("    alignas(64) int8_t Bp[7 * 4 * 2 * 64];",
       "    const int8_t **pw = &w;\n    *pw += 2940;\n    alignas(64) int8_t Bp[7 * 4 * 2 * 64];")],
     "static analysis DISAGREES: it evaluated `w[(nb * 8 + n) * TAPS + kh * KW * C + byte]` at line 31"),
    ("block larger than core memory",
     [(LOAD_X, LOAD_X.replace("{oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C}", "{0, 0, 0}, {230, 230, 4}"))],
     "larger than the core data memory"),
]

# The tensor form; with aie::autotune every candidate must be rejected, and the
# default's diagnostics are reported.
TENSOR_CASES = [
    ("tensor: baseline (autotuned)", [], None),
    ("tensor: conv border loaded without .pad",
     [(", {.pad = aie::pad_value::zero});", ");")],
     "outside the tensor extent 224 (a conv border? load it with {.pad = aie::pad_value::zero})"),
    ("tensor: wrong .reuse",
     [("{.reuse = OH_CORE / OH_T * Ofm::dim(1) / OW_T}", "{.reuse = OH_CORE / OH_T}")],
     "load_buffer says {.reuse = 4}, but 16 output block(s) are stored while it is held"),
    ("tensor: load bigger than the footprint (routed as is)",
     [(LOAD_T, LOAD_T.replace("{IH_T, IW_T, C}", "{IH_T + 1, IW_T, C}"))],
     None),
    ("tensor: data used without load (reads past the block)",
     [("x(oh * S + kh, ow * S + kw, c)", "x(oh * S + kh + 1, ow * S + kw, c)")],
     "data used without being loaded"),
    ("tensor: data used after release",
     [("auto y = aie::store_buffer(", "aie::release(x_t);\n            auto y = aie::store_buffer(")],
     "uses a block after aie::release"),
    # A block of 2 of the 3 channels (compute uses only those): the staged
    # layout aligns channels to 4, so blocks must take them all.
    ("tensor: part of the channels",
     [(LOAD_T, LOAD_T.replace("{IH_T, IW_T, C}", "{IH_T, IW_T, C - 1}")),
      ("for (int c = 0; c < C; c++)", "for (int c = 0; c < C - 1; c++)")],
     "blocks take part of the channel dim (C = 3)"),
    ("tensor: weights outside the tensor",
     [("{pc * OC_T, 0, 0, 0}", "{pc * OC_T + 1, 0, 0, 0}")],
     "outside the tensor extent 64"),
    ("tensor: output not fully written",
     [("for (int j = 0; j < Ofm::dim(1) / OW_T; j++)", "for (int j = 1; j < Ofm::dim(1) / OW_T; j++)"),
      ("Ofm::dim(1) / OW_T})", "(Ofm::dim(1) / OW_T - 1)})")],
     "never stored"),
    ("tensor: loop order swapped",
     [("for (int i = 0; i < OH_CORE / OH_T; i++)", "@OUTER@"),
      ("for (int j = 0; j < Ofm::dim(1) / OW_T; j++)", "for (int i = 0; i < OH_CORE / OH_T; i++)"),
      ("@OUTER@", "for (int j = 0; j < Ofm::dim(1) / OW_T; j++)"),
      # pinned: with one row round (OH_T=28) the swapped order is harmless
      ("conv2d<<<mesh>>>", "conv2d<7, 28, 16><<<mesh>>>")],
     "swap the two loops"),
]

# Kernel-only GEMM: routed with no operator recognition (trace_routing.h).
LOAD_B = "aie::load_buffer(b, {pn * 16, j * 64}, {16, 64})"
GEMM_CASES = [
    ("gemm: baseline (routed)", [], None),
    ("gemm: role tags omitted (aie::port<T, aie::shape<...>>)",
     [("aie::IFM, ", ""), ("aie::WEIGHTS, ", ""), ("aie::OFM, ", "")],
     None),
    ("gemm: input block unique per core",
     [(LOAD_B, "aie::load_buffer(b, {pn * 16, pm * 64}, {16, 64})")],
     "unique per core"),
    # example/other/aiehlcpm_gemm1.cc: the mesh macros come from an <angle>
    # include, which used to be skipped -> "no device.partition(...) found".
    ("gemm: mesh ROWS/COLS from an #include <header> macro",
     [('#include "simpleconv2d_new_load.h"', "#include <simplegemm_load.h>"),
      ("device.partition({3, 6, 0, 6}, 4, 4)", "device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS)")],
     None),
    ("gemm: mesh ROWS is an undefined macro",
     [("device.partition({3, 6, 0, 6}, 4, 4)", "device.partition({3, 6, 0, 6}, MY_ROWS, 4)")],
     "ROWS `MY_ROWS` is not an integer constant expression -- no `#define MY_ROWS`"),
    ("gemm: window passed by value",
     [("aie::port<input_window_int8 *, aie::IFM", "aie::port<input_window_int8, aie::IFM")],
     "port a: the window type must be a pointer"),
]


# Dataflow options on a kernel-only GEMM (simplegemm_dataflow.cc): each one
# must be something the DMA can do for that port's blocks.
DATAFLOW_BASE = CCODE / "simplegemm_dataflow.cc"
OPT_A = "{.depth = 2, .tile = {8, 8}}"
DATAFLOW_CASES = [
    ("dataflow: baseline (.resident / .depth / .tile)", [], None),
    ("dataflow: no options (still a legal DMA; the compute would then read row-major A)",
     [(", " + OPT_A, ""), ("{cn, GD_K}, {.resident = true}", "{cn, GD_K}")],
     None),
    ("dataflow: .depth = 1 on a 32-strip input (single buffer, re-armed)",
     [(OPT_A, "{.depth = 1, .tile = {8, 8}}")],
     None),
    ("dataflow: .resident on a port loaded every round",
     [(OPT_A, "{.resident = true, .tile = {8, 8}}")],
     "a resident block is loaded once"),
    ("dataflow: .resident with .depth = 2",
     [("{cn, GD_K}, {.resident = true}", "{cn, GD_K}, {.depth = 2, .resident = true}")],
     "contradicts it"),
    ("dataflow: .depth out of range", [(OPT_A, "{.depth = 3, .tile = {8, 8}}")], "single (1) or ping-pong (2)"),
    ("dataflow: .tile that does not divide the block", [(OPT_A, "{.depth = 2, .tile = {8, 7}}")],
     "does not divide the block (16,256)"),
    ("dataflow: .tile row below a 32-bit word", [(OPT_A, "{.depth = 2, .tile = {8, 2}}")],
     "must be multiples of 4 bytes"),
    ("dataflow: .tile on a store",
     [("{GD_TS, cn});", "{GD_TS, cn}, {.tile = {8, 8}});")],
     "aie::store_buffer takes only {.depth}"),
]

# Hand-written GemmSpaces + block loads (verifySpaceKernels): each side may be
# wrong, and the conflict names the GemmSpace field.
SLICE = ".slice_tiling = {.tile_size = 19, .stride = 14, .rounds = 4}"
OFM_D3 = "    .d3 = {.fullsize = 64, .tile_round = 4, .tile_size = 16}};"
SPACE_CASES = [
    ("space: baseline (GemmSpaces match the blocks)", [], None),
    ("space: GemmSpace slab step wrong",
     [(SLICE, SLICE.replace(".stride = 14", ".stride = 13"))],
     "win_a (GemmSpace RowBC_halo) .d1.slice_tiling.stride = 13, the kernel's blocks give 14"),
    ("space: GemmSpace splits the weights over 2 mesh cols",
     [(".d1 = {.fullsize = 64, .tile_round = 4, .tile_size = 16, .stride = 16},",
       ".d1 = {.fullsize = 64, .tile_round = 2, .tile_size = 32, .stride = 32},")],
     "win_b (GemmSpace ColBC) .d1.tile_round = 2, the kernel's blocks give 4"),
    ("space: kernel changed (14-row output blocks), GemmSpace not",
     [("const int oh_t = 7,", "const int oh_t = 14,")],
     "win_c (GemmSpace LtoR_Merge) .d1.slice_tiling.tile_size = 7, the kernel's blocks give 14"),
    ("space: input broadcast along the wrong mesh axis",
     [(".map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row}",
       ".map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Col}")],
     ".policy.map.layout = Col, the kernel's blocks give Row"),
    ("space: GemmSpace missing a dim the kernel blocks have",
     [(OFM_D3, "};")],
     "offset has 3 and shape 3 dims, but the tensor has 2"),
    ("space: scheduling fields are the user's (pp_depth 1, l1_budget 8 KB)",
     [(".sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}}",  # the first: RowBC_halo
       ".sched = {.pp_depth = 1, .l1_budget = aie::Bytes{8192}}}")],
     None),
    ("space: GEMM-form field", [(OFM_D3, OFM_D3[:-2] + ",\n    .m = {.fullsize = 112}};")],
     "uses .m (the GEMM form)"),
    ("space: unknown field", [(SLICE, SLICE[:-1] + ", .win = 7}")], "unknown field .d1.slice_tiling.win"),
    ("space: undefined GemmSpace",
     [("aie::port<input_window_int8 *, ColBC> win_b", "aie::port<input_window_int8 *, ColBX> win_b")],
     "no `constexpr aie::GemmSpace ColBX"),
]


def run_case(cli, name, edits, expect, base=BASE):
    src = base.read_text()
    for old, new in edits:
        if old not in src:
            return False, f"mutation anchor not found: {old[:60]!r}"
        src = src.replace(old, new, 1)
    with tempfile.TemporaryDirectory() as tmp:
        for h in CCODE.glob("*.h"):
            shutil.copy(h, tmp)
        path = pathlib.Path(tmp) / "case.cc"
        path.write_text(src)
        p = subprocess.run([cli, "trace", str(path)], capture_output=True, text=True, timeout=600)
        out = p.stdout + p.stderr
    if expect is None:
        return p.returncode == 0, out
    return p.returncode == 2 and expect in out, out


def main():
    cli = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "build" / "aiehlc-convtile")
    failed = 0
    cases = [(n, e, x, BASE) for n, e, x in CASES] + [(n, e, x, TENSOR_BASE) for n, e, x in TENSOR_CASES]
    cases += [(n, e, x, GEMM_BASE) for n, e, x in GEMM_CASES]
    cases += [(n, e, x, DATAFLOW_BASE) for n, e, x in DATAFLOW_CASES]
    cases += [(n, e, x, SPACE_BASE) for n, e, x in SPACE_CASES]
    if len(sys.argv) > 2:  # a name filter
        cases = [c for c in cases if sys.argv[2] in c[0]]
    for name, edits, expect, base in cases:
        ok, out = run_case(cli, name, edits, expect, base)
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failed += 1
            print("     expected:", expect)
            print("     " + "\n     ".join(out.strip().splitlines()[-8:]))
        elif expect:
            line = next((l for l in out.splitlines() if expect in l), "")
            print("     " + line.strip()[:200])
    print(f"kernel_trace_cases: {len(cases) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
