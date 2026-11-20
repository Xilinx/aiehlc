#!/usr/bin/env python3
"""UNet per-site K a/b noise floor on board logs (costmodel M3.1.2 P0).

  k_noise.py CFG=LOG_A,LOG_B [CFG=LOG_A,LOG_B ...]

Question: is a per-site KFLOOR maxrel <= 0.15 reachable at all on UNet board data?  Two runs of the same build
(a, b; no counters, no perturbation) bound the best any model can do: if K itself moves by > 10 % between a and
b on some site, a 0.15 per-site gate is mostly noise there.

Per site (launches>0 part of 'PROF aie_host_split', same definition as k_diag.py):
  K_meas = wait_ms - inval_ms
Output:
  KNSITE  cfg part kind launches K_a K_b rel        (worst 5 per cfg, sorted by rel)
  KNOISE cfg=<c> src=<board|other> launches=<CMP aie_launches> sites=<n> small=<K_a<0.05 ms> kmin_ms=<x>
         ab_maxrel=<max |K_a-K_b|/K_a> worst_part=<p>
  KNOISE ALL ab_maxrel=<x> reach=<1 if ab_maxrel <= 0.10>          (always the last line)
src=board needs both logs' CMP line on a palmyra board with run_rc=0 and PASS, and equal aie_launches.
rc=1 on unparsable input or a/b site-set mismatch; reach=0 is NOT an error (it is the finding).
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


def one(cfg, la, lb):
    ta, tb = Path(la).read_text(), Path(lb).read_text()
    sa, sb = sites(ta), sites(tb)
    if not sa or set(sa) != set(sb):
        raise ValueError("cfg=%s: a/b launches>0 site sets differ or empty (%d vs %d)" % (cfg, len(sa), len(sb)))
    ca, cb = cmp_line(ta), cmp_line(tb)
    board = all(c[0].startswith("palmyra") and c[2] == "0" and c[3] == "PASS" for c in (ca, cb)) and ca[1] == cb[1]
    for p in sa:
        if sa[p][:2] != sb[p][:2]:
            raise ValueError("cfg=%s part=%d: kind/launches differ a/b" % (cfg, p))
    rows = sorted(((rel(sa[p][2], sb[p][2]), p) for p in sa), reverse=True)
    for r, p in rows[:NWORST]:
        k, n, ka = sa[p]
        print("KNSITE cfg=%s part=%d kind=%s launches=%d K_a=%.3f K_b=%.3f rel=%.3f" % (cfg, p, k, n, ka, sb[p][2], r))
    ka = [v[2] for v in sa.values()]
    worst, wp = rows[0]
    print("KNOISE cfg=%s src=%s launches=%s sites=%d small=%d kmin_ms=%.3f ab_maxrel=%.3f worst_part=%d"
          % (cfg, "board" if board else "other", ca[1], len(sa), sum(k < SMALL_MS for k in ka), min(ka), worst, wp))
    return worst


def main(argv):
    if not argv or any("=" not in a or "," not in a.split("=", 1)[1] for a in argv):
        print(__doc__, file=sys.stderr)
        return 2
    worst = 0.0
    for a in argv:
        cfg, logs = a.split("=", 1)
        la, lb = logs.split(",", 1)
        worst = max(worst, one(cfg, la, lb))
    print("KNOISE ALL ab_maxrel=%.3f reach=%d" % (worst, worst <= REACH_MAX))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (OSError, ValueError) as e:
        print("KNOISE ERROR %s" % e, file=sys.stderr)
        sys.exit(1)
