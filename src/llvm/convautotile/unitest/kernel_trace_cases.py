#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Kernel-analysis regression: each case mutates a conv example into one logic
bug and checks `aiehlc-convtile trace` rejects it with the expected diagnostic.
CASES use the kernel-only port form (simpleconv2dkernel_port.cc), TENSOR_CASES
the tensor form (simpleconv2dkernel.cc: aie::in / aie::out, .pad, .reuse,
block views, aie::autotune).

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
    ("gemm: input block unique per core",
     [(LOAD_B, "aie::load_buffer(b, {pn * 16, pm * 64}, {16, 64})")],
     "unique per core"),
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
