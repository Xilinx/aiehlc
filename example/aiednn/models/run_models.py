#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Model tests for aiednn: PyTorch models lowered to aiednn primitives and checked.

    PYTHONPATH=src python3 example/aiednn/models/run_models.py                       # every model, x86
    PYTHONPATH=src python3 example/aiednn/models/run_models.py --models mobilenet_v2,tiny_vit --modes int8
    PYTHONPATH=src python3 example/aiednn/models/run_models.py --models tiny_unet --app x86
    PYTHONPATH=src python3 example/aiednn/models/run_models.py --models mobilenet_v2 --modes int8 --html
    PYTHONPATH=src python3 example/aiednn/models/run_models.py --mlir ...        # through the aiednn MLIR pipeline
    PYTHONPATH=src python3 example/aiednn/models/run_models.py --models mobilenet_v2 --app vek280 \\
        --board galerina13 --vitis-settings <.../Vitis/settings64.sh> [--run-only]

x86 (default): ``dnn_torch.model_test`` through the ctypes runtime -- f32 vs the float
model, int8 vs the same model with the same fake quantization, every integer GEMM
layer recomputed exactly from its own inputs. ``--app``: the generated C app
(weights INCBIN'd, golden output embedded, every AIE layer re-run on ref: and
compared on the device) built with ``ag.dnn_build`` and run on x86 or the board.
Writes worklocal/aiednn/models/report.md. doc/design/aiednn.md section 9.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import zoo  # noqa: E402

from aiegraph import dnn_torch as dt  # noqa: E402
from aiegraph.dnn_ctypes import AiednnLib  # noqa: E402

OUT = REPO / "worklocal" / "aiednn" / "models"
_MODEL_LINE = re.compile(r"MODEL (\S+) mode=(\S+) ops=(\d+) time_ms=(\S+) relerr=(\S+) top1=(-?\d+) "
                         r"ref_top1=(-?\d+) float_top1=(-?\d+) gemm_checked=(\d+) gemm_bad=(\d+)"
                         r"(?: batch=(\d+) top1_miss=(\d+))?")


def _input(name, args):
    """the model and its input; --batch B stacks B images (the real one, flipped / shifted copies)"""
    m, x = zoo.build(name, pretrained=not args.random)
    if args.batch > 1:
        import torch
        views = [x, x.flip(3), x.flip(2), x.roll(37, 3), x.roll(-29, 2), x.flip(3).roll(53, 2), x.roll(71, 3),
                 x.flip(2).flip(3)]
        x = torch.cat([views[i % len(views)] for i in range(args.batch)])
    return m, x


def _stem(name, mode, args, prof=False) -> str:
    """the build directory / page name of one (model, mode, options): run_app and --html must agree"""
    return (f"{name}_{mode}" + ("_mlir" if args.mlir else "") + (f"_{args.mesh}" if args.mesh else "")
            + ("_prof" if prof else "") + (f"_b{args.batch}" if args.batch > 1 else "")
            + ("_fastest" if getattr(args, "prefer", "aie") == "fastest" else "")
            + (f"_{args.tag}" if getattr(args, "tag", None) else "")
            + ("_eager" if getattr(args, "eager", False) else "")
            + ("lib" if getattr(args, "eager_library", False) else ""))


def run_app(name, mode, target, args) -> dict:
    import aiegraph as ag
    m, x = _input(name, args)
    d = OUT / _stem(name, mode, args, prof=args.profiling)
    g, ref, yf = dt.references(m, x, mode, name, mlir=args.mlir, keep_ir=d / "ir" if args.mlir else None)
    # AIEDNN_TUNE lines from an --autotune run of this build, per target: workstation emulation times the C
    # references, so its choices say nothing about the board
    table = d / target / "autotune.txt"
    tune = "tune" if args.autotune else table.read_text() if args.autotune_load and table.is_file() else None
    if args.autotune_load and not table.is_file():
        print(f"[run_models] {table} missing: run --autotune first; dispatch decides", flush=True)
    app = dt.emit_app(g, d, ref, dt.TOL[mode], dt._top1(yf, g.top1_range)[0], verify_gemm=not args.no_verify, autotune=tune,
                      prefer=args.prefer)
    elf = d / target / "aout" / "main.elf"
    if args.run_only and target != "x86" and elf.is_file():      # reuse the built ELF: stage + run only
        from aiegraph.dnn import DnnBuild, run_app as dnn_run
        rep = DnnBuild(app, target, d / target, [], ok=True, elf=elf)
        dnn_run(rep, board=args.board, timeout=args.timeout)
    else:
        kw = dict(target=target, out=d / target, run=True, board=args.board, timeout=args.timeout,
                  vitis_settings=args.vitis_settings, mesh=_mesh(args.mesh), modes="all" if args.all_modes else "auto",
                  aiehlc_args=("--profiling",) if args.profiling else ())
        if getattr(args, "eager", False):   # one kernel instance per layer shape, loaded before its layer
            from aiegraph import dnn_eager
            plan = dnn_eager.library_plan(name) if args.eager_library else dnn_eager.plan(g, name)
            print(plan.summary(), flush=True)
            kw.update(mode="eager", eager=plan)
        try:
            rep = ag.dnn_build(app, **kw)
        except ag.AiegraphError as exc:   # a single-layer kernel (VEK385's conv2dstem) ties the GEMM library
            if "serve the app equally" not in str(exc):
                raise
            lib = ag.kernel_lib(REPO / "src" / "aietensorop")
            rep = ag.dnn_build(app, kernels=[e.name for e in lib if e.contract.source.name == "gemm_uk.cc"], **kw)
    text = rep.run_log.read_text(errors="replace") if rep.run_log and rep.run_log.exists() else ""
    if args.autotune:                # keep the board's choices for --autotune-load
        tuned = [ln.strip() for ln in text.replace("\r", "").splitlines() if ln.startswith("AIEDNN_TUNE")]
        if tuned:
            table.parent.mkdir(parents=True, exist_ok=True)
            table.write_text("\n".join(tuned) + "\n")
            print(f"[run_models] {len(tuned)} autotune choices -> {table}", flush=True)
    m_ = _MODEL_LINE.search(text.replace("\r", ""))
    row = dict(model=name, mode=mode, where=target + (" mlir" if args.mlir else "") + (f" {args.mesh}" if args.mesh else "")
               + (f" b{args.batch}" if args.batch > 1 else ""),
               ok=bool(rep.ok and rep.run_ok),
               ops=len(g.ops))
    if m_:
        row.update(time_ms=float(m_.group(4)), rel=float(m_.group(5)), top1=int(m_.group(6)),
                   ref_top1=int(m_.group(7)), float_top1=int(m_.group(8)),
                   gemm=f"{int(m_.group(9)) - int(m_.group(10))}/{m_.group(9)}", batch=int(m_.group(11) or 1))
    else:
        row["note"] = rep.reason or "no MODEL line (see " + str(rep.run_log) + ")"
    return row


def run_x86(name, mode, lib, args) -> dict:
    m, x = _input(name, args)
    t = time.time()
    r = dt.model_test(m, x, mode, lib, name, mlir=args.mlir)
    ms = 1e3 * (time.time() - t)
    if args.html:                                  # model-flow page (aiegraph.dnn_viz), board log if one exists
        from aiegraph import dnn_viz
        stem = _stem(name, mode, args)
        # the board build of the same lowering and batch, if any: the newest of the boards' run.logs
        logs = [p for p in (OUT / stem / b / "run.log" for b in ("vek385", "vek280")) if p.is_file()]
        log = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
        page = dnn_viz.write_model_html(r.graph, lib, x.numpy(), OUT / f"{stem}.html", board_log=log)
        print(f"html: {page}", flush=True)
    return dict(model=name, mode=mode, where="workstation emulation" + (" mlir" if args.mlir else "") + (f" b{args.batch}" if args.batch > 1 else ""),
                ok=r.ok, ops=r.ops, rel=r.relerr, top1=r.top1, batch=r.batch,
                ref_top1=r.ref_top1, float_top1=r.float_top1, gemm=f"{r.gemm_checked - len(r.gemm_bad)}/{r.gemm_checked}",
                on_aie=r.on_aie, time_ms=ms)


def report(rows) -> str:
    head = "| model | mode | where | primitives | on AIE | int GEMM exact | rel err | top-1 (ref / float) | time | result |"
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        lines.append(f"| {r['model']} | {r['mode']} | {r['where']} | {r.get('ops', '')} | {r.get('on_aie', '')} | "
                     f"{r.get('gemm', '')} | {r['rel']:.1e} | {r.get('top1')} ({r.get('ref_top1')} / {r.get('float_top1')}) | "
                     f"{_time(r)} | {'PASS' if r['ok'] else 'FAIL ' + r.get('note', '')} |"
                     if "rel" in r else f"| {r['model']} | {r['mode']} | {r['where']} | | | | | | | FAIL {r.get('note', '')} |")
    return "\n".join(lines)


def _time(r) -> str:
    """time column: per inference, and per image for a batch"""
    t, b = r.get("time_ms", 0), r.get("batch", 1)
    return f"{t:.0f} ms" if b <= 1 else f"{t:.0f} ms / {b} = {t / b:.1f} ms per image"


def _mesh(text):
    return tuple(int(v) for v in text.split("x")) if text else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(zoo.TORCHVISION + list(zoo.CUSTOM)))
    ap.add_argument("--modes", default="f32,int8")
    ap.add_argument("--app", choices=["x86", "vek280", "vek385"], default=None, help="run the generated C app instead")
    ap.add_argument("--board", default=None)
    ap.add_argument("--vitis-settings", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--random", action="store_true", help="random weights / input (no download)")
    ap.add_argument("--no-verify", action="store_true", help="app: skip the on-device AIE-vs-ref layer check")
    ap.add_argument("--run-only", action="store_true", help="board: reuse an already built main.elf, only run it")
    ap.add_argument("--all-modes", action="store_true",
                    help="board: compile every megakernel mode (default: only those the model uses, dnn_build modes=auto)")
    ap.add_argument("--eager", action="store_true",
                    help="eager mode: a templated kernel instance per GEMM layer shape, loaded right before the "
                         "layer runs (aiegraph.dnn_eager; default: the fixed-megakernel library)")
    ap.add_argument("--eager-library", action="store_true",
                    help="with --eager: the premade template library (aiegraph.dnn_eager.LIBRARY) instead of "
                         "instances generated for the model's layer shapes")
    ap.add_argument("--html", action="store_true", help="x86: also write worklocal/aiednn/models/<model>_<mode>.html")
    ap.add_argument("--mlir", action="store_true",
                    help="lower unfused and optimize with the aiednn MLIR pipeline (aiegraph.dnn_mlir.PIPELINE)")
    ap.add_argument("--profiling", action="store_true",
                    help="board: build the runtime's PMU profiling layer (aiehlc.sh --profiling); aiednn_kernel_stats "
                         "then splits launch time into kload / bdcfg / coreen / startio / wait_io (own _prof dir)")
    ap.add_argument("--autotune", action="store_true",
                    help="app: time every implementation that accepts each problem on the warm-up run, keep the "
                         "fastest (aiednn_set_autotune), save the board's choices to <build dir>/autotune.txt")
    ap.add_argument("--autotune-load", action="store_true",
                    help="app: preload <build dir>/autotune.txt (aiednn_autotune_load): tuned choices, no tuning cost")
    ap.add_argument("--prefer", choices=("aie", "fastest"), default="aie",
                    help="the app's aiednn_set_preference: fastest keeps small ops (quantize reorders) on ref: "
                         "when faster -- and off the megakernel that would force an ELF switch")
    ap.add_argument("--tag", default=None, help="suffix for the build directory (an A/B variant, e.g. b16n)")
    ap.add_argument("--batch", type=int, default=1,
                    help="images per inference (stacked along N; the convolutions fold them into the GEMM's M)")
    ap.add_argument("--mesh", default="8x4",
                    help="ROWSxCOLS AIE cores for the GEMM ELF: default 8x4, all VEK280 core rows (the fastest "
                         "measured, doc/design/aiednn.md §18); 4x4 / 6x4 also build")
    a = ap.parse_args(argv)
    if a.mesh == "4x4":  # the library's own block: no mesh override, no dir suffix
        a.mesh = None
    OUT.mkdir(parents=True, exist_ok=True)
    lib = None if a.app else AiednnLib.build(OUT / ("lib" + (f"_{a.mesh}" if a.mesh else "")), mesh=_mesh(a.mesh))
    rows = []
    for name in a.models.split(","):
        for mode in a.modes.split(","):
            try:
                row = run_app(name, mode, a.app, a) if a.app else run_x86(name, mode, lib, a)
            except Exception as e:  # noqa: BLE001 -- one model's failure must not hide the others
                row = dict(model=name, mode=mode, where=a.app or "workstation emulation", ok=False, note=f"{type(e).__name__}: {e}")
            rows.append(row)
            print(report([row]).splitlines()[-1], flush=True)
    text = report(rows)
    (OUT / f"report_{a.app or 'ctypes'}{'_mlir' if a.mlir else ''}.md").write_text(text + "\n")
    print(text)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
