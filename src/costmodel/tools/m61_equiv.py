#!/usr/bin/env python3
"""M6.1 P1: costmodel.estimate() site mode == the frozen board-validated K+L+N model (M34 PZ).

  python3 -I src/costmodel/tools/m61_equiv.py --golden GOLDEN.tsv --build-root DIR [--tol 1e-6] [--params m3122]

GOLDEN.tsv: cfg site launches T_hat (kln_frozen.sh with KLN_K_JSON=k m3122, T_hat ms per launch; 2 cfgs x 138).
Imports costmodel from THIS worktree (python -I ignores PYTHONPATH, so <worktree>/src is put first on
sys.path) and calls estimate({"unet_build": DIR, "cfg": cfg, "sites": None}, board="palmyra213", params=...) per cfg.
builtins.open / io.open are wrapped for the whole run: every *.log opened is counted.  A board log (anything but
the build's own compiler log <DIR>/unet_<cfg>.build.log or <DIR>/unet_<cfg>/build.log) counts in logs_read;
compiler logs are counted separately as build_logs (build-dir input: GFA classes, -D defs).
Output:
  M61EQ PARAMS file=<relpath> md5=<md5>      one per frozen file estimate() loaded (its r["params"])
  M61EQ AUX file=<relpath> md5=<md5>         fixed side inputs it also read (r["aux"])
  M61EQ cfg=<cfg> sites=<n> golden=<n> maxabs=<x> launches_ok=<0|1> total_ms=<x> PASS|FAIL
  M61EQ ALL sites=<n> maxabs=<x> tol=<tol> logs_read=<n> mod=<worktree|other> PASS|FAIL
  M61EQ INPUTS build_logs=<n> board_logs=<paths|->
rc 0 iff ALL PASS (every golden pair matched within tol, same launches, no extra/missing site, logs_read=0,
module imported from this worktree, 5 params files).
"""
import argparse
import builtins
import csv
import io
import math
import os
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))

_OPENED = []
_real_open = builtins.open


def _counting_open(file, *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)):
        p = os.fsdecode(file)
        if p.endswith(".log"):
            _OPENED.append(os.path.abspath(p))
    return _real_open(file, *a, **k)


def classify(paths, build_root):
    br = os.path.abspath(build_root)
    build_logs = {os.path.join(br, "unet_%s.build.log" % c) for c in ("off", "aot-kco")}
    build_logs |= {os.path.join(br, "unet_%s" % c, "build.log") for c in ("off", "aot-kco")}
    b = [p for p in paths if p in build_logs]
    return b, [p for p in paths if p not in build_logs]


def read_golden(path):
    out = {}
    with _real_open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["cfg"], r["site"])] = (int(r["launches"]), float(r["T_hat"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--params", default="m3122")
    ap.add_argument("--board", default="palmyra213")
    a = ap.parse_args(argv)
    gold = read_golden(a.golden)
    builtins.open = io.open = _counting_open
    try:
        import costmodel as cm
        from costmodel import estimate
        mod = "worktree" if Path(cm.__file__).resolve().is_relative_to(WT) else "other"
        cfgs = list(dict.fromkeys(c for c, _s in gold))
        res = {c: estimate({"unet_build": a.build_root, "cfg": c, "sites": None}, board=a.board, params=a.params)
               for c in cfgs}
    finally:
        builtins.open = io.open = _real_open
    params, aux = {}, {}
    for r in res.values():
        params.update(r["params"])
        aux.update(r.get("aux", {}))
    for f, m in sorted(params.items()):
        print("M61EQ PARAMS file=%s md5=%s" % (f, m))
    for f, m in sorted(aux.items()):
        print("M61EQ AUX file=%s md5=%s" % (f, m))
    ok_all, n_all, mx_all = True, 0, 0.0
    for c in cfgs:
        g = {s: v for (cc, s), v in gold.items() if cc == c}
        got = {r["site"]: r for r in res[c]["sites"]}
        mx, lok = 0.0, True
        for s, (n, t) in g.items():
            r = got.get(s)
            if r is None:
                mx = float("inf")
                continue
            lok &= int(r["launches"]) == n
            d = abs(float(r["T_hat"]) - t)
            mx = d if not math.isfinite(d) else max(mx, d)
        extra = set(got) - set(g)
        ok = (not extra and len(got) == len(g) > 0 and lok and math.isfinite(mx) and mx <= a.tol
              and not res[c]["unpriced"])
        print("M61EQ cfg=%s sites=%d golden=%d maxabs=%.3g launches_ok=%d total_ms=%.4f %s"
              % (c, len(got), len(g), mx, int(lok), res[c]["total_ms"], "PASS" if ok else "FAIL"))
        ok_all &= ok
        n_all += len(got)
        mx_all = max(mx_all, mx)
    bl, logs = classify(_OPENED, a.build_root)
    ok_all &= not logs and mod == "worktree" and len(params) == 5 and len(cfgs) > 0
    print("M61EQ ALL sites=%d maxabs=%.3g tol=%g logs_read=%d mod=%s %s"
          % (n_all, mx_all, a.tol, len(logs), mod, "PASS" if ok_all else "FAIL"))
    print("M61EQ INPUTS build_logs=%d board_logs=%s" % (len(bl), ",".join(sorted(set(logs))) or "-"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
