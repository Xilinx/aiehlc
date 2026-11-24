#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""SCMB: summarize the "SCMBRAW ..." lines of example/perf/scatter_microbench.cc (costmodel, host scatter vs N).

usage: scmb_fit.py <console log>
Reads only the run log and prints; writes nothing (no params/ file). Results stay in the run's --out directory.
Per point:  SCMB N= M= align=64|80 cache=cold|warm us_per_kb=%.3f board=<board>
Summary:    SCMB ALL board=<board> n=<points parsed> ratio_N= ratio_align= ratio_cache= cause=N|align|cache|none
  ratio_N     = us_per_kb(N=10240) / us_per_kb(N=320), align 64, cold
  ratio_align = us_per_kb(align 80) / us_per_kb(align 64), N=640, cold
  ratio_cache = us_per_kb(cold) / us_per_kb(warm), N=640, align 64
  Each ratio is folded to >= 1 (max(r, 1/r)) for the cause decision; cause = the largest, none if all < 1.2.
Exit 1 if the BOARD marker is missing, no point parsed, a point has chk=BAD, or SCMB END is not PASS.
"""
import re
import sys


def kv(line):
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))


def main():
    log = sys.argv[1]
    lines = open(log, errors="replace").read().splitlines()
    board = None
    for l in lines:
        m = re.match(r"^BOARD machine=\S+ \S+ board=(\S+)$", l)
        if m:
            board = m.group(1)
    errs = []
    if not board:
        errs.append("missing BOARD marker")
        board = "?"
    pts = {}
    for l in lines:
        if not l.startswith("SCMBRAW "):
            continue
        c = kv(l)
        key = (int(c["N"]), int(c["align"]), c["cache"])
        sec = int(c["med_cntpct"]) / int(c["cntfrq"])
        upk = sec * 1e6 / (int(c["bytes"]) / 1024.0)
        pts[key] = upk
        if c.get("chk") != "OK" or c.get("bad") != "0":
            errs.append("point %s chk=%s bad=%s" % (key, c.get("chk"), c.get("bad")))
        print("SCMB N=%d M=%s align=%d cache=%s us_per_kb=%.3f board=%s" % (key[0], c["M"], key[1], key[2], upk, board))
    if not any(re.match(r"^SCMB END rc=0 PASS", l) for l in lines):
        errs.append("SCMB END not PASS")

    def ratio(a, b):
        return pts[a] / pts[b] if a in pts and b in pts and pts[b] > 0 else float("nan")

    rN = ratio((10240, 64, "cold"), (320, 64, "cold"))
    rA = ratio((640, 80, "cold"), (640, 64, "cold"))
    rC = ratio((640, 64, "cold"), (640, 64, "warm"))
    fold = {k: (max(r, 1 / r) if r == r and r > 0 else 0.0) for k, r in (("N", rN), ("align", rA), ("cache", rC))}
    best = max(fold, key=fold.get)
    cause = best if fold[best] >= 1.2 else "none"
    print("SCMB ALL board=%s n=%d ratio_N=%.3f ratio_align=%.3f ratio_cache=%.3f cause=%s"
          % (board, len(pts), rN, rA, rC, cause))
    for e in errs:
        print("SCMB ERROR %s" % e)
    return 1 if errs or not pts else 0


if __name__ == "__main__":
    sys.exit(main())
