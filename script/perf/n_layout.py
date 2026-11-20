#!/usr/bin/env python3
"""Per-site host layout features of the UNet AIE sites (costmodel M2, N segment), aligned with a board log.

  n_layout.py --build B [--check-log CFG=LOG ...] [--json out.json]

Features (aiegraph.costmodel.n_layout.layout_sites, from B/ag_work/arm_build/graph_driver.c + aiek_src only):
in_pack_bytes, out_scatter_bytes, relayout, up/down = arm|aie. --check-log aligns the sites in order with the
log's 'PROF part=<i> op=aie_host_<kind><cid>' lines (kind and cid must match, as n_feat.check) and prints the
board pack/scatter ms of 'PROF aie_host_split' next to the features (diagnostic only, not used by the features).

  NLAYSITE cfg=.. i=.. part=.. kind=.. cid=.. in_pack=.. out_scatter=.. relayout=.. up=.. down=.. pack_ms=.. scatter_ms=..
  NLAYOUT cfg=.. sites=.. aligned=a/n relayout=<n> arm_up=<n> arm_down=<n>
  NLAYPAIR cfg=.. pair=conv2/conv155 in_pack=a/b out_scatter=a/b relayout=a/b up=a/b down=a/b [pack_ms=a/b scatter_ms=a/b]
rc 0 iff every --check-log aligns all sites.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aiegraph.costmodel.n_layout import layout_sites  # noqa: E402
from aiegraph.costmodel.unet_seq import read  # noqa: E402

_PROF = re.compile(r"^PROF part=(\d+) op=aie_host_(conv|linear|attn)(\d+) ", re.M)
_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=\w+ pack_ms=([\d.]+) scatter_ms=([\d.]+)", re.M)
PAIR = (("conv", 2), ("conv", 155))


def check(st, log):
    txt = read(log)
    prof = _PROF.findall(txt)
    split = {int(p): (float(a), float(b)) for p, a, b in _SPLIT.findall(txt)}
    ok = 0
    for i, s in enumerate(st):
        good = i < len(prof) and prof[i][1] == s["kind"] and int(prof[i][2]) == s["cid"] and \
            int(prof[i][0]) == s["part"]
        if not good:
            print("NLAY_MISALIGN site=%d %s%d part=%d prof=%s" % (i, s["kind"], s["cid"], s["part"],
                                                                   prof[i] if i < len(prof) else None))
        ok += good
        if s["part"] in split:
            s["pack_ms"], s["scatter_ms"] = split[s["part"]]
    if len(prof) != len(st):
        print("NLAY_MISALIGN prof_sites=%d build_sites=%d" % (len(prof), len(st)))
    return ok, len(prof)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--check-log", action="append", default=[], metavar="CFG=LOG")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    build = a.build.rstrip("/")
    checks = [c.split("=", 1) for c in a.check_log] or [[os.path.basename(build), None]]
    rc = 0
    for cfg, log in checks:
        st = layout_sites(build)
        aligned = "-"
        if log:
            ok, nprof = check(st, log)
            aligned = "%d/%d" % (ok, len(st))
            if ok != len(st) or nprof != len(st):
                rc = 1
        for i, s in enumerate(st):
            print("NLAYSITE cfg=%s i=%d part=%d kind=%s cid=%d in_pack=%d out_scatter=%d relayout=%d up=%s down=%s "
                  "pack_ms=%s scatter_ms=%s" % (cfg, i, s["part"], s["kind"], s["cid"], s["in_pack_bytes"],
                                                s["out_scatter_bytes"], s["relayout"], s["up"], s["down"],
                                                s.get("pack_ms", "-"), s.get("scatter_ms", "-")))
        print("NLAYOUT cfg=%s sites=%d aligned=%s relayout=%d arm_up=%d arm_down=%d" %
              (cfg, len(st), aligned, sum(1 for s in st if s["relayout"]), sum(1 for s in st if s["up"] == "arm"),
               sum(1 for s in st if s["down"] == "arm")))
        pa = [next((s for s in st if (s["kind"], s["cid"]) == k), None) for k in PAIR]
        if None in pa:
            print("NLAYPAIR cfg=%s pair=conv2/conv155 missing" % cfg)
            rc = 1
        else:
            def ab(f):
                return "%s/%s" % (pa[0].get(f, "-"), pa[1].get(f, "-"))
            print("NLAYPAIR cfg=%s pair=conv2/conv155 in_pack=%s out_scatter=%s relayout=%s up=%s down=%s "
                  "pack_ms=%s scatter_ms=%s" % (cfg, ab("in_pack_bytes"), ab("out_scatter_bytes"), ab("relayout"),
                                                ab("up"), ab("down"), ab("pack_ms"), ab("scatter_ms")))
        if a.json:
            with open(a.json if len(checks) == 1 else "%s.%s" % (a.json, cfg), "w") as f:
                json.dump(st, f, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
