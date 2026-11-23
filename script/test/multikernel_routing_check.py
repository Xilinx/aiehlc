#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-MKROUTE guard: aiehlc emits ONE routing() (routing.cc, applied at mesh-partition init) for a multi-kernel
program. A kernel whose port topology differs from the others' (e.g. two row-broadcast inputs next to gemm_kacc's
row + column broadcast) silently runs on the other kernels' stream routes and hangs (wait_io TIMEOUT on its first
launch; sprint unet_m3-s1 P2, UNet + flash attention).

Reads the per-kernel "aiehlc trace -> routing" plan comments aiehlc writes into host.cc and fails (rc 1) when the
kernels' (role, distribution) topologies differ.  Usage: multikernel_routing_check.py <host.cc | worklocal dir>"""
import re
import sys
from pathlib import Path

KRE = re.compile(r"^// ==== aiehlc trace -> routing: (\S+?)'s ports")
PRE = re.compile(r"^//    (\S+) \((IFM|WEIGHTS|OFM), ([^,]+),")


def topologies(text: str) -> dict:
    """kernel -> tuple of (role, distribution) in port order"""
    out, cur = {}, None
    for line in text.splitlines():
        m = KRE.match(line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = PRE.match(line)
        if m and cur:
            out[cur].append((m.group(2), m.group(3).strip()))
    return {k: tuple(v) for k, v in out.items()}


def check(text: str) -> tuple:
    topo = topologies(text)
    groups = {}
    for k, t in topo.items():
        groups.setdefault(t, []).append(k)
    return len(groups) <= 1, topo, groups


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = Path(argv[0])
    if p.is_dir():
        p = p / "host.cc"
    ok, topo, groups = check(p.read_text(errors="replace"))
    for t, ks in groups.items():
        print("MKROUTE topology %s: %d kernel(s) %s" % (" ".join("%s=%s" % x for x in t), len(ks), ",".join(ks[:4])))
    print("MKROUTE kernels=%d topologies=%d %s" % (len(topo), len(groups), "OK" if ok else
          "FAIL (one routing() for all kernels: give every kernel the same port topology)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
