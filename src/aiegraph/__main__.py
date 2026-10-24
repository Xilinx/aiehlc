###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``aiegraph`` command line -- the Python API, one subcommand per function.

    aiegraph doctor                               what works here, and the fixes
    aiegraph setup                                attach (or build) TVM 0.16, then doctor
    aiegraph convert resnet18 [--calib a.jpg ...] [--workdir D]
    aiegraph layers   [--workdir D]               the layer table
    aiegraph detect   KERNEL.cc [--workdir D]
    aiegraph check    "layer1=KERNEL.cc" [--workdir D]
    aiegraph offload  "layer1=KERNEL.cc" [--workdir D] [--make]
    aiegraph build    [--workdir D] [--local]
    aiegraph compile  MODEL TARGET ["layer1=KERNEL.cc"]

``--workdir`` defaults to ``worklocal/aiegraph/resnet18``. Exit 0 = success /
READY, 1 = not ready / failed, 2 = usage or environment error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _emod(args):
    import aiegraph as ag
    wd = Path(args.workdir or Path.cwd() / "worklocal" / "aiegraph" / "resnet18")
    return ag.open(wd)


def main(argv=None) -> int:
    import aiegraph as ag

    ap = argparse.ArgumentParser(prog="aiegraph", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("setup")
    sub.add_parser("guide")
    p = sub.add_parser("convert")
    p.add_argument("model", nargs="?", default="resnet18")
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
    p = sub.add_parser("compile")
    p.add_argument("model")
    p.add_argument("target", choices=["cpu", "a78", "x86", "aie"])
    p.add_argument("offload", nargs="?", default=None)
    p.add_argument("--workdir", default=None)
    p.add_argument("--no-build", action="store_true")
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
        emod = _emod(args)
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
