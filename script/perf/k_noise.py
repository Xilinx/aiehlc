#!/usr/bin/env python3
"""UNet per-site K a/b noise floor on board logs (costmodel M3.1.2 P0).

  k_noise.py [--require-reach] CFG=L1,L2[,L3...] [CFG=L1,L2[,L3...] ...]

Question: is a per-site KFLOOR maxrel <= 0.15 reachable at all on UNet board data?  Two runs of the same build
(a, b; no counters, no perturbation) bound the best any model can do: if K itself moves by > 10 % between a and
b on some site, a 0.15 per-site gate is mostly noise there.

Per site (launches>0 part of 'PROF aie_host_split', same definition as k_diag.py):
  K_meas = wait_ms - inval_ms
N >= 2 runs per cfg (K_a = first log, K_b = second log).  Per site rel = (max K - min K) / K_a over all runs
(n=2: |K_a-K_b|/K_a).  Gated site set = sites whose K is >= 0.100 ms in EVERY run (sub-0.1 ms sites are timer
noise, not model error).
Output:
  KNSITE  cfg part kind launches K_a K_b rel        (worst 5 per cfg, sorted by rel; rel over all runs)
  KNOISE GATED cfg=<c> runs=<n> kmin_gate=0.100 gated=<n> excluded=<n> maxrel=<x> worst_part=<p>
  KNOISE cfg=<c> src=<board|other> launches=<CMP aie_launches> sites=<n> small=<K_a<0.05 ms> kmin_ms=<x>
         ab_maxrel=<max rel over all sites> worst_part=<p>
  KNOISE ALL ab_maxrel=<x> reach=<1 if ab_maxrel <= 0.10>          (ungated, informational)
  KNOISE GATED ALL maxrel=<x> reach=<1 if gated maxrel <= 0.10>    (always the last line)
src=board needs every log's CMP line on a palmyra board with run_rc=0 and PASS, and equal aie_launches.
rc=1 on unparsable input, a duplicated log path or a site-set mismatch.  Without --require-reach reach=0 is NOT
an error (it is the finding); with --require-reach rc=3 iff GATED ALL reach=0 (KNOISE ALL reach is ignored).
"""
import re
import sys
from pathlib import Path

_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) .*wait_ms=([\d.]+) .*inval_ms=([\d.]+) "
                    r"launches=(\d+)$", re.M)
_CMP = re.compile(r"^CMP unet board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+).* (PASS|FAIL)\s*$", re.M)
SMALL_MS = 0.05
REACH_MAX = 0.10
NWORST = 5


def sites(txt):
    """{part: (kind, launches, K_ms)} for launches>0 parts"""
    out = {}
    for p, k, w, i, n in _SPLIT.findall(txt):
        if int(n) > 0:
            if int(p) in out:
                raise ValueError("duplicate host_split part=%s" % p)
            out[int(p)] = (k, int(n), float(w) - float(i))
    return out


def cmp_line(txt):
    m = _CMP.findall(txt)
    return m[-1] if m else ("-", "-1", "-1", "FAIL")


def rel(a, b):
    return abs(a - b) / a if a > 0 else (0.0 if b == a else float("inf"))


KMIN_GATE = 0.100


def one(cfg, logs):
    txts = [Path(l).read_text() for l in logs]
    ss = [sites(t) for t in txts]
    sa = ss[0]
    for i, sb in enumerate(ss[1:], 2):
        if not sa or set(sa) != set(sb):
            raise ValueError("cfg=%s: run 1/%d launches>0 site sets differ or empty (%d vs %d)"
                             % (cfg, i, len(sa), len(sb)))
        for p in sa:
            if sa[p][:2] != sb[p][:2]:
                raise ValueError("cfg=%s part=%d: kind/launches differ run 1/%d" % (cfg, p, i))
    cs = [cmp_line(t) for t in txts]
    board = (all(c[0].startswith("palmyra") and c[2] == "0" and c[3] == "PASS" for c in cs)
             and len({c[1] for c in cs}) == 1)
    ks = {p: [s[p][2] for s in ss] for p in sa}

    def srel(p):
        v = ks[p]
        return rel(v[0], v[0] + (max(v) - min(v)))

    rows = sorted(((srel(p), p) for p in sa), reverse=True)
    for r, p in rows[:NWORST]:
        k, n, ka = sa[p]
        print("KNSITE cfg=%s part=%d kind=%s launches=%d K_a=%.3f K_b=%.3f rel=%.3f" % (cfg, p, k, n, ka, ss[1][p][2], r))
    grows = [(r, p) for r, p in rows if min(ks[p]) >= KMIN_GATE]
    gworst, gwp = grows[0] if grows else (0.0, -1)
    print("KNOISE GATED cfg=%s runs=%d kmin_gate=%.3f gated=%d excluded=%d maxrel=%.3f worst_part=%d"
          % (cfg, len(logs), KMIN_GATE, len(grows), len(rows) - len(grows), gworst, gwp))
    ka = [v[2] for v in sa.values()]
    worst, wp = rows[0]
    print("KNOISE cfg=%s src=%s launches=%s sites=%d small=%d kmin_ms=%.3f ab_maxrel=%.3f worst_part=%d"
          % (cfg, "board" if board else "other", cs[0][1], len(sa), sum(k < SMALL_MS for k in ka), min(ka), worst, wp))
    return worst, gworst


def main(argv):
    req = "--require-reach" in argv
    argv = [a for a in argv if a != "--require-reach"]
    if not argv or any("=" not in a or "," not in a.split("=", 1)[1] for a in argv):
        print(__doc__, file=sys.stderr)
        return 2
    jobs = []
    for a in argv:
        cfg, logs = a.split("=", 1)
        logs = logs.split(",")
        rp = [str(Path(l).resolve()) for l in logs]
        if any(not l for l in logs) or len(set(rp)) != len(rp):
            raise ValueError("cfg=%s: empty or repeated log path in %s" % (cfg, logs))
        jobs.append((cfg, logs))
    worst = gworst = 0.0
    for cfg, logs in jobs:
        w, g = one(cfg, logs)
        worst, gworst = max(worst, w), max(gworst, g)
    print("KNOISE ALL ab_maxrel=%.3f reach=%d" % (worst, worst <= REACH_MAX))
    greach = gworst <= REACH_MAX
    print("KNOISE GATED ALL maxrel=%.3f reach=%d" % (gworst, greach))
    return 3 if req and not greach else 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError) as e:
        print("KNOISE ERROR %s" % e, file=sys.stderr)
        sys.exit(1)
