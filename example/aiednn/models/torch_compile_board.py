#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Board numbers for the torch.compile path: the backend's graph emitted as a graph-API board app.

    PYTHONPATH=src python3 example/aiednn/models/torch_compile_board.py --models resnet18,mobilenet_v2 \\
        --mode pt2e --board <host> --vitis-settings <.../settings64.sh>
    ... --mode f32 --models tiny_vit ... ; --target x86 runs the same app in workstation emulation
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(Path(__file__).resolve().parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zoo  # noqa: E402

import aiegraph as ag  # noqa: E402
from aiegraph.dnn_graph import build_graph_lib  # noqa: E402
from aiegraph.dnn_graph_app import emit_graph_app  # noqa: E402
from aiegraph.dnn_torch_backend import _sink_permute  # noqa: E402
from aiegraph.dnn_torch_graph import FxGraph  # noqa: E402
from aiegraph import dnn_torch as dt  # noqa: E402

OUT = REPO / "worklocal" / "aiednn" / "torch_compile_board"


def pt2e(m, x):
    from torch._export import capture_pre_autograd_graph
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
    gm = capture_pre_autograd_graph(m, (x,))
    gm = prepare_pt2e(gm, XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True)))
    gm(x)
    return convert_pt2e(gm)


def graph_of(name: str, mode: str):
    """(fx GraphModule, input, eager output, float top-1, top-1 window or None)."""
    from torch.fx.experimental.optimization import fuse
    m, x = zoo.build(name)
    with torch.no_grad():
        yf = m(x)
    gm = pt2e(m, x) if mode == "pt2e" else dt.prepare(m, x)
    gm = fuse(gm, no_trace=True)
    _sink_permute(gm)
    with torch.no_grad():
        golden = gm(x)
    rng = getattr(m, "top1_range", None)
    return gm, x, golden.numpy().astype(np.float32), dt._top1(yf.numpy(), rng)[0], rng


def _output_step(gm) -> float:
    """Scale of the dequantize producing the graph output; 0 for f32 graphs."""
    out = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    n = out[0] if isinstance(out, (tuple, list)) else out
    for _ in range(8):
        if n.op == "call_function" and "dequantize" in str(n.target):
            return float(n.args[1])
        ins = [a for a in n.args if hasattr(a, "op")]
        if len(ins) != 1:
            return 0.0
        n = ins[0]
    return 0.0


def run_one(name, mode, args, lib):
    gm, x, golden, float_top1, rng = graph_of(name, mode)
    # the emulation golden must dispatch with the board app's preference
    lib.so.aiednn_set_preference(1 if args.prefer == "fastest" else 0)
    try:
        return _run_one(name, mode, args, lib, gm, x, golden, float_top1, rng)
    finally:
        lib.so.aiednn_set_preference(0)


def _run_one(name, mode, args, lib, gm, x, golden, float_top1, rng):
    fx = FxGraph(lib, gm, [x], fuse=args.fuse)
    head = fx.report().splitlines()[0]
    out = [n for n in gm.graph.nodes if n.op == "output"][0].args[0]
    out = (out[0] if isinstance(out, (list, tuple)) else out).name
    feeds = {[n.name for n in gm.graph.nodes if n.op == "placeholder"][0]: x.numpy()}
    d = OUT / (f"{name}_{mode}" + (f"_fuse{args.fuse}" if args.fuse != 1 else "") +
               ("_fastest" if args.prefer == "fastest" else "") + ("_dense" if args.dense else ""))
    # golden: the same partitions run by aiednn on the workstation; PT2E eager is reported as REF2
    env = fx.graph.run({**fx.consts, **feeds})
    emu = env[fx.alias.get(out, out)].astype(np.float32).reshape(golden.shape)
    # part_ref: per-partition emulation outputs (PARTCHK lines on the board)
    app = emit_graph_app(fx.graph, fx.consts, feeds, out, emu, d, f"{name}_{mode}",
                         layouts=False if args.dense else None, top1_range=rng, step=_output_step(gm),
                         ref2=golden, ref2_name="pt2e", part_ref=env)
    hooks = ["aiednn_set_preference(aiednn_prefer_fastest);"] if args.prefer == "fastest" else []
    if hooks:
        src = app.read_text()
        key = "CK(aiednn_graph_create(&g));"
        assert key in src, "emit_graph_app changed: no graph_create line to hook"
        app.write_text(src.replace(key, key + "".join("\n    " + h for h in hooks), 1))
    mesh = tuple(int(v) for v in args.mesh.split("x"))
    t = time.time()
    elf = d / args.target / "aout" / "main.elf"
    if args.run_only and args.target != "x86" and elf.is_file():
        from aiegraph.dnn import DnnBuild, run_app
        rep = DnnBuild(app, args.target, d / args.target, [], ok=True, elf=elf)
        run_app(rep, board=args.board, timeout=args.timeout)
    else:
        rep = ag.dnn_build(app, target=args.target, out=d / args.target, run=True, board=args.board,
                           timeout=args.timeout, mesh=mesh if args.target != "x86" else None, modes=args.modes,
                           vitis_settings=args.vitis_settings)
    log = Path(rep.run_log).read_text(errors="replace") if getattr(rep, "run_log", None) else ""
    ms = re.findall(r"(?:time_ms|ms_per_image|graph_ms)[=: ]+([0-9.]+)", log)
    print(f"| {name} | {mode} fuse {args.fuse}{' fastest' if args.prefer == 'fastest' else ''}{' dense' if args.dense else ''} | {head} | {'PASS' if getattr(rep, 'run_ok', False) else 'FAIL ' + (rep.reason or '')} | "
          f"{ms[-1] if ms else '?'} ms | emulation top-1 {dt._top1(emu, rng)[0]}, pt2e {dt._top1(golden, rng)[0]}, float {float_top1} | {time.time() - t:.0f}s |",
          flush=True)
    for ln in log.splitlines():
        if re.match(r"(MODEL|PROF|GRAPH_BUILD|FAIL)", ln.strip()):
            print("    " + ln.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="resnet18")
    ap.add_argument("--mode", choices=("f32", "pt2e"), default="pt2e")
    ap.add_argument("--target", default="vek280")
    ap.add_argument("--board", default=None)
    ap.add_argument("--mesh", default="8x4")
    ap.add_argument("--modes", default=os.environ.get("MODES", "library"))
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--prefer", choices=("aie", "fastest"), default="aie",
                    help="aiednn_set_preference: aie (default, AIE-first) or fastest (cost refusals, ref: when faster)")
    ap.add_argument("--run-only", action="store_true", help="board: reuse an already built main.elf, only run it")
    ap.add_argument("--dense", action="store_true",
                    help="layout propagation off (emit_graph_app layouts=False): every edge dense NCHW, the A/B baseline")
    ap.add_argument("--fuse", type=int, default=1, help="graph API fusion level: 1 exact (default), 2 also inexact")
    ap.add_argument("--vitis-settings", default=os.environ.get("AIEDNN_VITIS_SETTINGS"))
    args = ap.parse_args()
    lib = build_graph_lib(OUT / "lib")
    for name in args.models.split(","):
        run_one(name, args.mode, args, lib)


if __name__ == "__main__":
    main()
