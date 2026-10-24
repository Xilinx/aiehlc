###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``python -m frontend.tvmrelay.aiekernel`` -- user-kernel CLI (run with PYTHONPATH=src).

    lint    SOURCE...                         static checks only, no network needed
    detect  SOURCE [--out-dir D] [--all]      which layers the spec fits, and why not
    check   SOURCE[@L,..]... [--out-dir D]    lint + detect + calibrate + emulate
    offload SOURCE[@L,..]... [--out-dir D]    check + aiehlc build + partition.json

No subcommand touches a board. ``offload --make`` additionally links main.elf
(cross toolchain), it does not run it. Exit status: 0 when every kernel has at
least one layer through every gate that ran, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

_DEFAULT_OUT = "worklocal/tvmrelay_deploy"


def _layers(text):
    if text is None or text in ("auto", "all"):
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def _cmd_lint(args) -> int:
    from .lint import lint_source

    rc = 0
    for src in args.sources:
        rep = lint_source(src)
        print(f"{src}: {'OK' if rep.ok else 'FAIL'} "
              f"({'AIE kernel ' + ', '.join(rep.kernels) if rep.kernels else 'reference-only'})")
        print(rep.text())
        for spec in rep.specs:
            print("  " + spec.describe().replace("\n", "\n  "))
        rc |= 0 if rep.ok else 1
    return rc


def _cmd_detect(args) -> int:
    from . import detect

    verdicts = detect(args.out_dir, args.sources[0])
    hits = [v for v in verdicts if v.ok]
    for v in verdicts:
        if v.ok or args.all or any(m in v.symbol for m in ("conv", "dense", "matmul")):
            calls = ", ".join(f"[{c.call.call:02d}]" for c in v.calls)
            print(f"  layer {v.layer:02d} {'MATCH' if v.ok else '-----'} calls {calls}: "
                  f"{v.reason}")
    print(f"{len(hits)} layer(s) match: {', '.join(f'{v.layer:02d}' for v in hits) or 'none'}"
          + ("" if args.all else "  (--all lists non-conv layers too)"))
    return 0 if hits else 1


def _ready(report) -> int:
    return 0 if all(k["selected"] for k in report["kernels"]) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m frontend.tvmrelay.aiekernel",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lint", help="static checks on the kernel source")
    p.add_argument("sources", nargs="+")

    p = sub.add_parser("detect", help="which layers of the deployed graph fit the spec")
    p.add_argument("sources", nargs=1)
    p.add_argument("--out-dir", default=_DEFAULT_OUT)
    p.add_argument("--all", action="store_true", help="list every layer, not just convs")

    for name, helptext in (("check", "lint + detect + calibrate + emulate (no writes)"),
                           ("offload", "check + build + partition.json + arm_build")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("sources", nargs="+", help="kernel source, optionally @1,4 for layers")
        p.add_argument("--out-dir", default=_DEFAULT_OUT)
        p.add_argument("--layers", default=None,
                       help="layers for sources given without @ (default: every match)")
        if name == "check":
            p.add_argument("--no-calibrate", action="store_true")
            p.add_argument("--no-emulate", action="store_true")
        else:
            p.add_argument("--no-build", action="store_true",
                           help="record the decision only; no aiehlc run")
            p.add_argument("--make", action="store_true",
                           help="also link main.elf (cross toolchain; never runs it)")
            p.add_argument("--force", action="store_true",
                           help="offload even if calibrate/emulate disagree (reported)")
            p.add_argument("--aie-version", type=int, default=5)

    args = ap.parse_args(argv)
    if args.cmd == "lint":
        return _cmd_lint(args)
    if args.cmd == "detect":
        return _cmd_detect(args)
    from .flow import check, offload

    if args.cmd == "check":
        return _ready(check(args.out_dir, args.sources, _layers(args.layers),
                            calibrate=not args.no_calibrate,
                            emulate=not (args.no_emulate or args.no_calibrate)))
    rep = offload(args.out_dir, args.sources, _layers(args.layers), build=not args.no_build,
                  make=args.make, force=args.force, aie_version=args.aie_version)
    return _ready(rep)


if __name__ == "__main__":
    sys.exit(main())
