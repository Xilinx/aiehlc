#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Custom PyTorch operators through torch.export -> aiednn eager mode.

Three cases, each using the same model skeleton (Linear -> custom op -> Linear):

  Case A -- inline, the op's Python body found automatically:
    A torch.library.custom_op op keeps its Python body; export_inlined() finds it
    and re-exports with the op inlined as ATen IR (older torch without custom_op:
    a CompositeImplicitAutograd registration, which plain export traces through).

  Case B -- inline via explicit body passed to export_inlined():
    The op is registered against the CPU backend only (opaque to the exporter).
    We pass the body to export_inlined(bodies=...), which re-exports with the
    op inlined as ATen IR.  Useful when you can't change the op's registration.

  Case C -- opaque, C implementation on the A72 host:
    No Python body is available.  The op stays as an aiednn_custom primitive and
    runs on the A72 via softclip_c.c.  The surrounding GEMMs run on the AIE.

Usage:
    # workstation emulation (all three cases)
    PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py

    # board run (VEK280)
    PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py \\
        --target vek280 --board galerina13 \\
        --vitis-settings /proj/xbuilds/.../Vitis/settings64.sh

    # single case
    PYTHONPATH=src python example/aiednn/custom_pytorch_op/run.py --case a
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

import torch                                    # noqa: E402
import torch.nn as nn                           # noqa: E402

from aiegraph import dnn_export                 # noqa: E402

OUT = REPO / "worklocal" / "aiednn" / "custom_pytorch_op"


# ── shared model skeleton ─────────────────────────────────────────────────────

class _CustomModel(nn.Module):
    def __init__(self, op_fn):
        super().__init__()
        self.fc1 = nn.Linear(256, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self._op = op_fn

    def forward(self, x):
        return self.fc2(self._op(self.fc1(x)))


def _make_input():
    torch.manual_seed(0)
    return torch.randn(32, 256)


def _run_case(label, ep, x, out_dir, target, board, vitis_settings, custom_sources=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    # show how many custom nodes remain after export
    custom_nodes = [n for n in ep.graph.nodes
                    if n.op == "call_function" and dnn_export.is_custom_op(n.target)]
    if custom_nodes:
        print(f"  export graph: {len(custom_nodes)} opaque node(s): "
              + ", ".join(n.target.name() for n in custom_nodes))
    else:
        print("  export graph: 0 opaque nodes (all custom ops were inlined)")

    # compile_exported: flow 4, f32 to keep the example simple (no int8 calibration)
    ce = getattr(dnn_export, "compile_exported")
    rep, g, _ = ce(ep, x, mode="f32", name=label, target=target, out=out_dir,
                   eager="library", custom_sources=custom_sources,
                   vitis_settings=vitis_settings)
    print(g.summary())
    if rep.ok and (target == "x86" or board):
        dnn_export.run(rep, board=board)
    log = rep.run_log.read_text(errors="replace") if rep.run_log and rep.run_log.exists() else ""
    summary_lines = [ln for ln in log.splitlines()
                     if ln.startswith(("MODEL ", "PASS", "FAIL"))]
    if summary_lines:
        print("\n".join(summary_lines))
    ran = target == "x86" or bool(board)
    ok = rep.ok and (rep.run_ok if ran else True)
    print(f"  {'PASS' if ok else 'FAIL'}: {label} -> {target}" +
          ("" if ran else " (built, not run)"))
    return 0 if ok else 1


# ── Case A: a Python-bodied custom op, inlined by export_inlined() ───────────

def case_a(x, target, board, vitis_settings) -> int:
    """The op has a Python body written in torch ops.  With torch.library.custom_op (torch >= 2.4 release)
    export_inlined() finds the body itself and re-exports with the op inlined as ATen IR.  Older torch
    (no custom_op): the body is registered as the op's CompositeImplicitAutograd kernel, which a plain
    torch.export traces through -- the same graph either way."""
    body = lambda t: torch.tanh(t) * 2.0                       # noqa: E731
    if hasattr(torch.library, "custom_op"):
        @torch.library.custom_op("custom_op_demo_a::softclip_a", mutates_args=())
        def softclip_a(t: torch.Tensor) -> torch.Tensor:
            return body(t)

        @softclip_a.register_fake
        def _(t):
            return torch.empty_like(t)

        model = _CustomModel(softclip_a).eval()
        ep = dnn_export.export_inlined(model, [x])             # the body is found, no bodies= needed
        how = "torch.library.custom_op, export_inlined() found the body"
    else:
        _lib_a = torch.library.Library("custom_op_demo_a", "DEF")
        _lib_a.define("softclip_a(Tensor t) -> Tensor")
        _lib_a.impl("softclip_a", body, "CompositeImplicitAutograd")
        model = _CustomModel(torch.ops.custom_op_demo_a.softclip_a).eval()  # type: ignore[attr-defined]
        ep = torch.export.export(model, (x,))                  # traced through the composite kernel
        how = "no torch.library.custom_op in this torch: CompositeImplicitAutograd registration"
    print(f"\n--- Case A: Python body, inlined ({how}) ---")
    return _run_case("case_a_inlined", ep, x, OUT / "case_a", target, board, vitis_settings)


# ── Case B: CPU-only op, body passed to export_inlined() ─────────────────────

def case_b(x, target, board, vitis_settings) -> int:
    """The op is registered against the CPU backend key only.  The exporter sees
    it as a black box.  We pass the body to export_inlined(bodies=...) which
    re-exports with the op inlined as AtEn IR for that one export call only."""

    _lib_b = torch.library.Library("custom_op_demo_b", "DEF")
    _lib_b.define("softclip_b(Tensor t) -> Tensor")
    _lib_b.impl("softclip_b", lambda t: torch.tanh(t) * 2.0, "CPU")
    # abstract (shape-only) impl: needed so torch.export can trace the op
    torch.library.impl_abstract("custom_op_demo_b::softclip_b",  # type: ignore[attr-defined]
                                 lambda t: torch.empty_like(t))

    model = _CustomModel(torch.ops.custom_op_demo_b.softclip_b)  # type: ignore[attr-defined]
    model.eval()

    # export_inlined: re-exports with the body registered as CompositeImplicitAutograd
    # for the duration; original CPU-only registration is restored afterwards
    ep = dnn_export.export_inlined(model, [x],
                                   bodies={"custom_op_demo_b::softclip_b":
                                           lambda t: torch.tanh(t) * 2.0})
    print(f"\n--- Case B: CPU-only op, body passed to export_inlined() ---")
    return _run_case("case_b_explicit_body", ep, x,
                     OUT / "case_b", target, board, vitis_settings)


# ── Case C: opaque op, C implementation on the A72 ───────────────────────────

def case_c(x, target, board, vitis_settings) -> int:
    """No Python body available (or deliberately kept opaque).  The op stays as
    an aiednn_custom primitive and runs on the host A72 via softclip_c.c.
    The surrounding GEMMs run on the AIE as usual."""

    _lib_c = torch.library.Library("custom_op_demo_c", "DEF")
    _lib_c.define("softclip_c(Tensor t) -> Tensor")
    _lib_c.impl("softclip_c", lambda t: torch.clamp(t, -1.0, 1.0), "CPU")
    torch.library.impl_abstract("custom_op_demo_c::softclip_c",  # type: ignore[attr-defined]
                                 lambda t: torch.empty_like(t))

    model = _CustomModel(torch.ops.custom_op_demo_c.softclip_c)  # type: ignore[attr-defined]
    model.eval()

    # plain export -- the CPU-only op stays opaque (no CompositeImplicitAutograd)
    ep = torch.export.export(model, (x,))
    print(f"\n--- Case C: opaque op, C function on the A72 (softclip_c.c) ---")
    c_src = Path(__file__).parent / "softclip_c.c"
    return _run_case("case_c_opaque_c", ep, x,
                     OUT / "case_c", target, board, vitis_settings,
                     custom_sources=[str(c_src)])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="x86", choices=["x86", "vek280", "vek385"])
    ap.add_argument("--board", default=None)
    ap.add_argument("--vitis-settings", default=None)
    ap.add_argument("--case", default="all", choices=["all", "a", "b", "c"],
                    help="which case to run (default: all three)")
    a = ap.parse_args()

    x = _make_input()
    rc = 0
    if a.case in ("all", "a"):
        rc |= case_a(x, a.target, a.board, a.vitis_settings)
    if a.case in ("all", "b"):
        rc |= case_b(x, a.target, a.board, a.vitis_settings)
    if a.case in ("all", "c"):
        rc |= case_c(x, a.target, a.board, a.vitis_settings)
    return rc


if __name__ == "__main__":
    sys.exit(main())
