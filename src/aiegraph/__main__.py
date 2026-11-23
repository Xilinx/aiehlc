###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``aiegraph`` command line -- the Python API, one subcommand per function.

    aiegraph doctor                               what works here, and the fixes
    aiegraph setup                                attach (or build) TVM 0.16, then doctor
    aiegraph convert [MODEL] [--calib a.jpg ...] [--workdir D]
    aiegraph layers   [--workdir D]               the layer table
    aiegraph prototype                            the kernel-file format (no model needed)
    aiegraph prototype layer1 [--region R] [--entry NAME] [--write DIR] [--no-test] [--workdir D]
                                                  one layer: runnable kernel .cc + reference .c, AIEK_SPEC
    aiegraph aiprototype layer1 [--entry NAME] [--rounds N] [--no-build] [--out DIR] [--model M] [--workdir D]
                                                  backbone, then claude fills the TODOs until prototype_check passes
    aiegraph protocheck  layer1 DIR [--entry NAME] [--build] [--workdir D]
                                                  is DIR's prototype a correct kernel? (todo, kemu, trace, check[, build])
    aiegraph detect   KERNEL.cc [--workdir D]
    aiegraph check    "layer1=KERNEL.cc" [--workdir D]
    aiegraph offload  "layer1=KERNEL.cc" [--workdir D] [--make]
    aiegraph build    [--workdir D] [--local]
    aiegraph run      [palmyra|vek385] [HOST] [--profile] [--workdir D] [--no-netload] [--reboot|--no-reboot]
                                                  main.elf on a lab board (builds it if stale)
    aiegraph profile  [LAYER ...] [--by-layer] [--top N] [--workdir D]
                                                  per-call / per-layer time of the last board run
    aiegraph compile  MODEL TARGET ["layer1=KERNEL.cc"]
    aiegraph autopart --onnx M.onnx --mode conv|l37|all|attn --out partition.json [--base OLD.json]
                      [--ms-per-launch MS] [--pad-kn] [--budget-ms MS]
                                                  generate the AIE/CPU GEMM partition (GAP-A)

``--workdir`` defaults to ``worklocal/aiegraph/<ag.DEFAULT_MODEL>``. Exit 0 = success /
READY, 1 = not ready / failed, 2 = usage or environment error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _emod(args):
    import aiegraph as ag
    wd = Path(args.workdir or Path.cwd() / "worklocal" / "aiegraph" / ag.DEFAULT_MODEL)
    return ag.open(wd)


def main(argv=None) -> int:
    import aiegraph as ag

    ap = argparse.ArgumentParser(prog="aiegraph", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("setup")
    sub.add_parser("guide")
    p = sub.add_parser("convert")
    p.add_argument("model", nargs="?", default=ag.DEFAULT_MODEL)
    p.add_argument("--target", default="a78")
    p.add_argument("--calib", nargs="*", default=None, help="calibration images")
    p.add_argument("--image", default=None, help="the image the ELF classifies")
    for name in ("layers", "detect", "check", "offload", "build", "convert"):
        q = sub.choices[name] if name in sub.choices else sub.add_parser(name)
        q.add_argument("--workdir", default=None)
        q.add_argument("-v", "--verbose", action="store_true")
        if name in ("detect", "check", "offload"):
            q.add_argument("spec", help='"layer1=k.cc", "k.cc@1,4" style or a kernel path')
    sub.choices["offload"].add_argument("--make", action="store_true",
                                        help="also link main.elf")
    sub.choices["offload"].add_argument("--no-build", action="store_true")
    sub.choices["build"].add_argument("--local", action="store_true",
                                      help="also build main_local.elf (x86)")
    p = sub.add_parser("run")
    p.add_argument("board", nargs="?", default="palmyra", help="palmyra (default) or vek385")
    p.add_argument("host", nargs="?", default=None, help="lab host / board name (default: env)")
    p.add_argument("--workdir", default=None)
    p.add_argument("--elf", default=None, help="run this ELF instead of the model's main.elf")
    p.add_argument("--no-build", action="store_true", help="do not (re)build a stale main.elf")
    p.add_argument("--no-netload", dest="netload", action="store_false", default=None,
                   help="palmyra: JTAG dow instead of the netboot supervisor")
    p.add_argument("--reboot", dest="reboot", action="store_true", default=None)
    p.add_argument("--no-reboot", dest="reboot", action="store_false")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("-q", "--quiet", action="store_true", help="do not echo the console")
    p.add_argument("--profile", action="store_true",
                   help="main.elf with PROFILE=1 per-call timing (then: aiegraph profile)")
    p = sub.add_parser("profile")
    p.add_argument("layers", nargs="*", help="layers (layer1 4 ...); default all")
    p.add_argument("--by-layer", action="store_true", help="one row per layer, not per call")
    p.add_argument("--top", type=int, default=None, help="only the N slowest rows")
    p.add_argument("--workdir", default=None)
    p = sub.add_parser("prototype")
    p.add_argument("layer", nargs="?", default=None, help="TVM layer (layer1, 4, ...); omit for the format")
    p.add_argument("--region", default=None, help="an ONNX region name instead of a layer")
    p.add_argument("--entry", default=None, help="entry / kernel name (default aiek_layerNN)")
    p.add_argument("--write", default=None, metavar="DIR", help="write <entry>.cc (kernel + host + entry), <entry>_ref.c and test_<entry>.cc (main())")
    p.add_argument("--no-test", action="store_true", help="with --write: skip test_<entry>.cc")
    p.add_argument("--workdir", default=None)
    p = sub.add_parser("aiprototype")
    p.add_argument("layer")
    p.add_argument("--entry", default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--no-build", action="store_true", help="skip the final aiehlc build of test_<entry>.cc")
    p.add_argument("--out", default=None, metavar="DIR")
    p.add_argument("--model", default=None)
    p.add_argument("--workdir", default=None)
    p = sub.add_parser("protocheck")
    p.add_argument("layer")
    p.add_argument("dir")
    p.add_argument("--entry", default=None)
    p.add_argument("--build", action="store_true", help="also build test_<entry>.cc with aiehlc")
    p.add_argument("--workdir", default=None)
    p = sub.add_parser("compile")
    p.add_argument("model")
    p.add_argument("target", choices=["cpu", "a78", "x86", "aie"])
    p.add_argument("offload", nargs="?", default=None)
    p.add_argument("--workdir", default=None)
    p.add_argument("--no-build", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("autopart")
    p.add_argument("--onnx", required=True)
    p.add_argument("--mode", default="conv", choices=["conv", "l37", "all", "attn"])
    p.add_argument("--out", required=True)
    p.add_argument("--base", default=None, help="older partition.json, only compared against")
    p.add_argument("--ms-per-launch", type=float, default=7.471)
    p.add_argument("--pad-kn", action="store_true", help="also pad K/N to 64 (needs GKP weight repack)")
    p.add_argument("--budget-ms", type=float, default=None)
    p = sub.add_parser("dnn-build", help="build (and run) an aiednn app with kernels from the library")
    p.add_argument("app")
    p.add_argument("--target", default="vek280", choices=["vek280", "vek385", "x86"])
    p.add_argument("--root", default=None, help="kernel library root (default src/aietensorop)")
    p.add_argument("--kernel", action="append", default=None, help="restrict to this kernel (repeatable)")
    p.add_argument("--out", default=None)
    p.add_argument("--vitis-settings", default=None, help=".../Vitis/settings64.sh")
    p.add_argument("--headers", action="append", default=[], help="dir of generated headers the app includes")
    p.add_argument("--run", action="store_true")
    p.add_argument("--board", default=None, help="e.g. galerina13 (vek280), portobello13 (vek385)")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--modes", default="auto", choices=["auto", "all", "library"],
                   help="megakernel modes: the app's (auto), every mode, or the library's fixed ELF set")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("dnn-lib", help="build the prebuilt aiednn library (runtime + kernel ELF set) to link apps against")
    p.add_argument("--target", default="vek280", choices=["vek280"])
    p.add_argument("--dist", default=None, help="output dir (default worklocal/aiednn/lib_<target>)")
    p.add_argument("--mesh", default=None, help="ROWSxCOLS AIE cores (default 4x4)")
    p.add_argument("--eager", action="store_true",
                   help="eager mode: the premade template library (every eager template, one kernel ELF per "
                        "instance) instead of the fixed megakernel set")
    p.add_argument("--eager-mesh", default="4x4", help="with --eager: the instances' mesh, RxC (rows 4 / 6 / 8, 4 columns)")
    p.add_argument("--vitis-settings", default=None, help=".../Vitis/settings64.sh")
    p.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "doctor":
            d = ag.doctor()
            print(d)
            return 0 if d.ok_for("convert") else 1
        if args.cmd == "setup":
            return 0 if ag.setup().ok_for("convert") else 1
        if args.cmd == "guide":
            ag.guide()
            return 0
        if args.cmd == "convert":
            emod = ag.embedded_convert(args.model, target=args.target, calib=args.calib,
                                       image=args.image, workdir=args.workdir,
                                       verbose=args.verbose)
            print(emod)
            return 0
        if args.cmd == "compile":
            emod = ag.compile(args.model, args.target, args.offload, workdir=args.workdir,
                              build=not args.no_build, verbose=args.verbose)
            print(emod)
            return 0
        if args.cmd == "autopart":
            from . import autopart as apm
            doc = apm.autopartition(args.onnx, args.mode, base=args.base,
                                    ms_per_launch=args.ms_per_launch, pad_kn=args.pad_kn, budget_ms=args.budget_ms)
            print(f"AP_OUT {apm.write_partition(doc, args.out)}")
            return 0
        if args.cmd == "prototype" and args.layer is None and args.region is None:
            print(ag.offload_prototype())
            return 0
        if args.cmd == "dnn-build":
            rep = ag.dnn_build(args.app, target=args.target, out=args.out, root=args.root,
                               kernels=args.kernel, vitis_settings=args.vitis_settings, run=args.run,
                               board=args.board, timeout=args.timeout, verbose=args.verbose,
                               headers=args.headers, modes=args.modes)
            print(rep)
            return 0 if rep.ok and rep.run_ok is not False else 1
        if args.cmd == "dnn-lib":
            from .dnn_lib import build_library
            mesh = tuple(int(v) for v in args.mesh.split("x")) if args.mesh else None
            lb = build_library(args.target, dist=args.dist, mesh=mesh, vitis_settings=args.vitis_settings,
                               verbose=args.verbose, mode="eager" if args.eager else "fixed-megakernel",
                               eager_rows=int(args.eager_mesh.split("x")[0]))
            print(f"dnn-lib {args.target}: {'OK' if lb.ok else 'FAILED: ' + lb.reason} -> {lb.dist}")
            for f in lb.files:
                print(f"  {f}")
            return 0 if lb.ok else 1
        emod = _emod(args)
        if args.cmd == "aiprototype":
            return 0 if emod.ai_offload_prototype(args.layer, entry=args.entry, rounds=args.rounds,
                                                  build=not args.no_build, out_dir=args.out,
                                                  model=args.model).ok else 1
        if args.cmd == "protocheck":
            v = emod.prototype_check(args.layer, args.dir, entry=args.entry, build=args.build)
            print(v)
            return 0 if v.ok else 1
        if args.cmd == "prototype":
            itf = emod.offload_prototype(args.layer, region=args.region, entry=args.entry)
            print(itf)
            if args.write:
                for one in (itf if isinstance(itf, ag.Prototypes) else [itf]):
                    one.write(args.write, test=not args.no_test)
            return 0
        if args.cmd == "run":
            rep = emod.run(args.board, args.host, elf=args.elf, build=not args.no_build,
                           netload=args.netload, reboot=args.reboot, timeout=args.timeout,
                           profile=args.profile, verbose=not args.quiet)
            print(rep)
            return 0 if rep.ok else 1
        if args.cmd == "profile":
            prof = emod.profile(args.layers or ag.ALL, by="layer" if args.by_layer else "call",
                                top=args.top)
            if prof is not None:
                print(prof)
            return 0 if prof is not None else 1
        if args.cmd == "layers":
            print(emod)
            return 0
        if args.cmd == "detect":
            rep = emod.detect(args.spec)
            print(rep)
            return 0 if rep.matches else 1
        spec = args.spec
        if "@" in spec and "=" not in spec:          # aiekernel's k.cc@1,4 spelling
            src, sel = spec.rsplit("@", 1)
            spec = ";".join(f"layer{s.strip()}={src}" for s in sel.split(",") if s.strip())
        if args.cmd == "check":
            rep = emod.offload_check(spec, verbose=args.verbose)
        elif args.cmd == "offload":
            rep = emod.offload(spec, build=not args.no_build, make=args.make,
                               verbose=args.verbose)
        else:                                          # build
            rep = emod.build(local=args.local, verbose=args.verbose)
        print(rep)
        return 0 if rep.ok else 1
    except ag.AiegraphError as exc:
        print(f"aiegraph: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
