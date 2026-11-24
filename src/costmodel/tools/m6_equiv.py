#!/usr/bin/env python3
"""M6 P1: costmodel.estimate() net mode == the frozen board-validated K+L+N golden T_hat of all 3 nets.

  m6_equiv.py --golden-r18 R18_KNET.out --golden-yolo YOLO_KNET.out --golden-unet GOLDEN.tsv
              --build-root ROOT (resnet18_/yolo26n_<cfg>) --unet-build ROOT (unet_<cfg>) [--tol 5e-5]

Golden key = (net, cfg, part|site):
  resnet18 / yolo26n: KSITE lines of k_net --params (frozen m41 / m42) -> T_hat (%.4f); the same key repeats once per
                      board run, T_hat must be identical across runs (it does not depend on the run).
  unet:               m61_golden_that.tsv (cfg site launches T_hat), kln_frozen.sh KLN_K_JSON=m3122.
Calls estimate({"net": n, "build": root, "cfg": cfg}) once per (net, cfg) of the golden and compares every row.
Output:
  M6EQ PARAMS net=<n> file=<relpath> md5=<md5>
  M6EQ net=<n> rows=<r> cfgs=<c> maxabs=<x> launches_ok=<0|1> extra=<n> missing=<n> PASS|FAIL
  M6EQ ALL nets=<n> rows=<r> maxabs=<x> tol=<tol> PASS|FAIL
rc 0 iff ALL PASS.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))

_KS = re.compile(r"^KSITE net=(\S+) cfg=(\S+) run=\S+ part=(\d+) launches=(\d+) .*?\bT_hat=([0-9.eE+-]+)")


def read_knet(path, net):
    """{(cfg, 'part<N>'): (launches, T_hat)}; raises if a key has two different T_hat / launches"""
    out = {}
    for ln in open(path):
        m = _KS.match(ln)
        if not m or m.group(1) != net:
            continue
        k, v = (m.group(2), "part%s" % m.group(3)), (int(m.group(4)), float(m.group(5)))
        if out.setdefault(k, v) != v:
            raise ValueError("%s: key %s has two values %s / %s" % (path, k, out[k], v))
    return out


def read_unet(path):
    with open(path) as f:
        return {(r["cfg"], r["site"]): (int(r["launches"]), float(r["T_hat"]))
                for r in csv.DictReader(f, delimiter="\t")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden-r18", required=True)
    ap.add_argument("--golden-yolo", required=True)
    ap.add_argument("--golden-unet", required=True)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--unet-build", required=True)
    ap.add_argument("--tol", type=float, default=5e-5)
    ap.add_argument("--board", default="palmyra213")
    a = ap.parse_args(argv)
    from costmodel import estimate
    nets = [("unet", read_unet(a.golden_unet), a.unet_build),
            ("resnet18", read_knet(a.golden_r18, "resnet18"), a.build_root),
            ("yolo26n", read_knet(a.golden_yolo, "yolo26n"), a.build_root)]
    ok_all, rows_all, mx_all = True, 0, 0.0
    for net, gold, root in nets:
        cfgs = list(dict.fromkeys(c for c, _s in gold))
        mx, lok, extra, missing, params = 0.0, True, 0, 0, {}
        for c in cfgs:
            r = estimate({"net": net, "build": root, "cfg": c}, board=a.board)
            params.update(r["params"])
            got = {s["site"]: s for s in r["sites"]}
            want = {s: v for (cc, s), v in gold.items() if cc == c}
            extra += len(set(got) - set(want))
            for s, (n, t) in want.items():
                g = got.get(s)
                if g is None:
                    missing += 1
                    continue
                lok &= int(g["launches"]) == n
                d = abs(float(g["T_hat"]) - t)
                mx = max(mx, d) if math.isfinite(d) else float("inf")
            ok_c = not r["unpriced"]
            lok &= ok_c
        for f, m in sorted(params.items()):
            print("M6EQ PARAMS net=%s file=%s md5=%s" % (net, f, m))
        ok = bool(gold) and lok and not extra and not missing and math.isfinite(mx) and mx <= a.tol
        print("M6EQ net=%s rows=%d cfgs=%d maxabs=%.3g launches_ok=%d extra=%d missing=%d %s"
              % (net, len(gold), len(cfgs), mx, int(lok), extra, missing, "PASS" if ok else "FAIL"))
        ok_all &= ok
        rows_all += len(gold)
        mx_all = max(mx_all, mx)
    print("M6EQ ALL nets=%d rows=%d maxabs=%.3g tol=%g %s"
          % (len(nets), rows_all, mx_all, a.tol, "PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
