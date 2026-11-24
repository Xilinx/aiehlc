#!/usr/bin/env python3
"""N-segment pair diagnosis: for site pairs with identical static features (kind, MK, MN, launches),
find which host-split component (pack/scatter/glue/flush/inval_ms) carries the N difference, and
check that it reproduces across two board runs (a, b) of the same config.

  n_diag.py --build-root R --pairs conv16/conv70,... --logs cfg=a.log,b.log [cfg2=...]

Build dir per cfg is R/unet_<cfg> (same layout as n_fit.py).  dom = component with max |delta|;
det=1 iff dom(a)==dom(b) and |d_a-d_b|/max(|d_a|,|d_b|) <= 0.10.

Noise floor: a component's delta counts only if |d| > NOISE[cfg][comp] * max(site1, site2) for that component,
where NOISE is the measured same-ELF a/b per-site max relative diff (shuhua 2026-11-20_M2/postmortem_noise.md;
flush/inval have no reference and use the cfg's largest measured rel noise).  If no component clears its floor in a run, dom=none for that run
(the pair has no measurable difference); dom=none in both runs is reproducible, so det=1.
"""
import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_spec = importlib.util.spec_from_file_location("n_fit", str(Path(__file__).resolve().parent / "n_fit.py"))
n_fit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n_fit)

COMPS = ("pack", "scatter", "glue", "flush", "inval")
AB_TOL = 0.10
NOISE = {"aot-kco": {"glue": 0.077, "pack": 0.118, "scatter": 0.117},
         "off": {"glue": 0.050, "pack": 0.159, "scatter": 0.127}}
_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) (.*)$", re.M)
_PAIR = re.compile(r"^([a-z]+)(\d+)$")


def parse_pair(s):
    out = []
    for x in s.split("/"):
        m = _PAIR.match(x)
        if not m:
            raise ValueError("bad pair member %r (want kind+cid, e.g. conv16)" % x)
        out.append((m.group(1), int(m.group(2))))
    if len(out) != 2:
        raise ValueError("bad pair %r" % s)
    return s, out


def kname(r):
    k = r.get("kernels")
    if not k:
        return "-"
    return "+".join(k) if isinstance(k, (list, tuple)) else str(k)


def load(build, log):
    rows, txt = n_fit.measure(build, log)
    split = {}
    for p, kind, rest in _SPLIT.findall(txt):
        split[int(p)] = (kind, {k: float(v) for k, v in re.findall(r"(\w+)_ms=([\d.]+)", rest)})
    site, prev, last = {}, {}, "-"
    for r in rows:
        key = (r["kind"], r["cid"])
        prev[key] = last
        if r["part"] in split and split[r["part"]][0] == r["kind"]:
            site[key] = dict(r, comp=split[r["part"]][1])
        if r["launches"] > 0:
            last = kname(r)
    return site, prev, txt


def get(site, key, log):
    if key not in site:
        raise KeyError("%s: no aie_host_split line for %s%d" % (log, key[0], key[1]))
    return site[key]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--logs", nargs="+", required=True, help="cfg=a.log,b.log")
    a = ap.parse_args(argv)
    pairs = [parse_pair(p) for p in a.pairs.split(",")]
    n_pairs = n_det = 0
    doms = Counter()
    ok = True
    try:
        for spec in a.logs:
            cfg, logs = spec.split("=", 1)
            logs = logs.split(",")
            if len(logs) != 2:
                raise ValueError("%s: need exactly two logs a,b" % cfg)
            build = str(Path(a.build_root) / ("unet_" + cfg))
            runs = []
            for log in logs:
                site, prev, txt = load(build, log)
                ok &= n_fit.src_line(cfg, log, txt)
                runs.append((site, prev, log))
            for name, (k1, k2) in pairs:
                ds = []
                for site, prev, log in runs:
                    s1, s2 = get(site, k1, log), get(site, k2, log)
                    delta = {c: s2["comp"].get(c, 0.0) - s1["comp"].get(c, 0.0) for c in COMPS}
                    ds.append((delta, s1, s2, prev))
                (da, s1, s2, prev), (db, _, _, _) = ds
                nz = NOISE.get(cfg, {})
                nz = {c: nz.get(c, max(nz.values()) if nz else 0.0) for c in COMPS}

                def dom_of(d, r1, r2):
                    sig = [c for c in COMPS
                           if abs(d[c]) > nz.get(c, 0.0) * max(r1["comp"].get(c, 0.0), r2["comp"].get(c, 0.0))]
                    return max(sig, key=lambda c: abs(d[c])) if sig else "none"
                dom_a = dom_of(da, ds[0][1], ds[0][2])
                dom_b = dom_of(db, ds[1][1], ds[1][2])
                kc = dom_a if dom_a != "none" else max(COMPS, key=lambda c: abs(da[c]))
                x, y = da[kc], db[kc]
                m = max(abs(x), abs(y))
                ab_rel = abs(x - y) / m if m > 0 else 0.0
                det = int(dom_a == dom_b and (dom_a == "none" or ab_rel <= AB_TOL))
                same = int(kname(s1) == kname(s2))
                print("NDIAG cfg=%s pair=%s kernel_same=%d dom=%s d_a=%.3f d_b=%.3f ab_rel=%.3f det=%d "
                      "dom_b=%s prev_kernel=%s/%s N_a=%.3f/%.3f | %s"
                      % (cfg, name, same, dom_a, x, y, ab_rel, det, dom_b, prev[k1], prev[k2],
                         s1["N"], s2["N"], " ".join("d%s=%.3f/%.3f" % (c, da[c], db[c]) for c in COMPS)))
                n_pairs += 1
                n_det += det
                doms[dom_a] += 1
    except (KeyError, ValueError) as e:
        print("NDIAG ERROR %s" % e.args[0])
        return 1
    print("NDIAG ALL pairs=%d det=%d dom=%s" % (n_pairs, n_det,
                                                ",".join("%s:%d" % kv for kv in sorted(doms.items()))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
