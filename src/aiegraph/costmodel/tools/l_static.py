#!/usr/bin/env python3
"""Static L-segment features per launch from a BUILD directory only (M1.2.2 P4).

The six features the L model uses (launch_l.feature_row) come from the build products, never from a board counter:

  bdcfg_calls, startio_calls, bdcfg_w, cp_w   static_feat.static_feat (kernel build) / unet_static_feat (UNet)
  kload_w                                     kload_static.kload_static
  rst (core_reset per launch)                 rst_static.rst_per_launch

static_calls_words(build) -> (calls, words) in the dict shape launch_l.feature_row / predict_l take.

CLI (the comparison reads the board logs, the prediction does not):
  l_static.py --build-root R --kernel-logs DIR
      points DIR/<cfg>/<kernel>.log for cfg in off, aot, aot-kco; build R/k_<cfg>/<kernel>
      measured = launch_l.point(log): split_detail calls / launches, split_words, core_reset / launches.
      rel = |pred-meas|/meas, 0/0 -> 0; word features with meas <= 16 use the absolute tolerance 16 words.
  Prints  LSTATIC cfg=<cfg> k=<kernel> <feat>=<pred>/<meas> ... maxrel=%.3f PASS|FAIL
  and     LSTATIC ALL n=<n> src=static maxrel=%.3f PASS|FAIL   (needs n=27 and maxrel <= 0.050)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aiegraph.costmodel import static_feat as sf  # noqa: E402
from aiegraph.costmodel.kload_static import kload_static  # noqa: E402
from aiegraph.costmodel.launch_l import point  # noqa: E402
from aiegraph.costmodel.rst_static import rst_per_launch  # noqa: E402

KCFGS = ("off", "aot", "aot-kco")
FEATS = ("bdcfg_calls", "startio_calls", "bdcfg_w", "cp_w", "kload_w", "rst")
REL_GATE = 0.050
N_EXPECT = 27


def static_calls_words(build):
    """(calls, words) per launch of a kernel build (aout/worklocal/host.cc) or a UNet aiegraph build."""
    b = str(build)
    if os.path.isfile(os.path.join(b, "aout", "worklocal", "host.cc")):
        f = sf.static_feat(b)
    else:
        f, _info = sf.unet_static_feat(b)
    calls = {"bdcfg": float(f["bdcfg_calls"]), "startio": float(f["startio_calls"]), "kload": 1.0,
             "core_reset": float(rst_per_launch(b))}
    words = {"bdcfg_w": float(f["bdcfg_w"]), "cp_w": float(f["cp_w"]),
             "kload_w": float(kload_static(b)["kload_w"])}
    return calls, words


def flat(calls, words):
    return {"bdcfg_calls": calls["bdcfg"], "startio_calls": calls["startio"], "bdcfg_w": words["bdcfg_w"],
            "cp_w": words["cp_w"], "kload_w": words["kload_w"], "rst": calls["core_reset"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--kernel-logs", required=True, help="DIR/<cfg>/<kernel>.log for cfg in off, aot, aot-kco")
    a = ap.parse_args(argv)
    n = fails = 0
    worst = 0.0
    for cfg in KCFGS:
        for log in sorted((Path(a.kernel_logs) / cfg).glob("*.log")):
            k = log.stem
            b = Path(a.build_root) / f"k_{cfg}" / k
            try:
                p = flat(*static_calls_words(b))
                mp = point(str(log))
                m = flat(mp["calls"], mp["words"])
            except Exception as e:  # noqa: BLE001 -- a point that cannot be computed is a FAIL, never skipped
                print(f"LSTATIC cfg={cfg} k={k} FAIL error={type(e).__name__}: {e}")
                fails += 1
                continue
            rels = {x: sf.rel_of(p[x], m[x], x.endswith("_w")) for x in FEATS}
            r = max(rels.values())
            ok = r <= REL_GATE
            n += 1
            fails += not ok
            worst = max(worst, r)
            cols = " ".join(f"{x}={p[x]:.1f}/{m[x]:.1f}" for x in FEATS)
            print(f"LSTATIC cfg={cfg} k={k} {cols} maxrel={r:.3f} {'PASS' if ok else 'FAIL'}")
    ok = fails == 0 and n == N_EXPECT and worst <= REL_GATE
    print(f"LSTATIC ALL n={n} src=static maxrel={worst:.3f} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
