#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Board run-to-run jitter of per-site T (site_ms) across the a/b/c runs (costmodel M3.1.2.2 PN).

  t_noise.py SET=LPRED.tsv [SET=LPRED.tsv ...] [--site CFG:SITE ...] [--margin X]
  t_noise.py --selftest

Input: an lpred TSV (kln_site output) with header columns cfg, run, part, site, site_ms (others ignored).
The run id is the letter before '.log' (unet_off.a.log -> a); a bare letter run ('a') is accepted too.
Per (set, cfg, site): spread = max(T)/min(T) - 1 over the runs.  A site key is (part, site); the printed name
is the site column.

Output (stdlib only, so it runs under python -I):
  TNOISE cfg=<c> set=<s> sites=<n> runs=<n> keys_equal=<0|1> spread_max=<x> spread_med=<x> worst=<site> n_ge3pct=<n>
  TNOISE SITE cfg=<c> set=<s> site=<site> T=<a>,<b>,<c> spread=<x>      (per --site, per set; T in run order)
  TNOISE MARGIN spread_max=<x over all sets/cfgs> margin=<m> margin_ok=<1 if spread_max <= margin>
margin_ok=0 is NOT an error (it is the finding).  rc=1 on unparsable input, duplicated (cfg, run, part, site)
rows, unequal run sets per site, non-positive T, or a --site that is missing / ambiguous.  rc=2 on bad usage.
"""
import csv
import re
import statistics
import sys
import tempfile
from pathlib import Path

RUN_RE = re.compile(r"(?:^|\.)([a-z])\.log$")
GE = 0.03


class Bad(Exception):
    pass


def run_letter(run):
    if re.fullmatch(r"[a-z]", run):
        return run
    m = RUN_RE.search(run)
    if not m:
        raise Bad("cannot parse run letter from %r" % run)
    return m.group(1)


def load(path):
    """-> {cfg: {(part, site): {run: T}}}"""
    data = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        need = {"cfg", "run", "part", "site", "site_ms"}
        if not rd.fieldnames or not need <= set(rd.fieldnames):
            raise Bad("%s: missing columns %s" % (path, sorted(need - set(rd.fieldnames or []))))
        for row in rd:
            r = run_letter(row["run"])
            try:
                t = float(row["site_ms"])
            except ValueError:
                raise Bad("%s: bad site_ms %r" % (path, row["site_ms"]))
            if not t > 0:
                raise Bad("%s: non-positive site_ms at %s/%s" % (path, row["cfg"], row["site"]))
            d = data.setdefault(row["cfg"], {}).setdefault((row["part"], row["site"]), {})
            if r in d:
                raise Bad("%s: duplicate row cfg=%s run=%s site=%s" % (path, row["cfg"], r, row["site"]))
            d[r] = t
    if not data:
        raise Bad("%s: no rows" % path)
    return data


def spread(ts):
    return max(ts) / min(ts) - 1.0


def analyse(sets, sites, margin, out):
    gmax = 0.0
    for sname, data in sets:
        for cfg in sorted(data, key=lambda c: (c != "off", c)):
            per = data[cfg]
            allruns = sorted({r for v in per.values() for r in v})
            # keys_equal: every run has the same (part, site) key set (a missing / renamed site in one run -> 0)
            keys_equal = int(all(len(v) == len(allruns) for v in per.values()))
            runs = tuple(allruns)
            if len(runs) < 2:
                raise Bad("set=%s cfg=%s: need >= 2 runs, got %d" % (sname, cfg, len(runs)))
            common = {k: v for k, v in per.items() if len(v) == len(runs)}
            sp = {k: spread([v[r] for r in runs]) for k, v in common.items()}
            if sp:
                wk = max(sp, key=lambda k: (sp[k], k))
                smax = sp[wk]
                gmax = max(gmax, smax)
                out.append("TNOISE cfg=%s set=%s sites=%d runs=%d keys_equal=%d spread_max=%.4f spread_med=%.4f worst=%s "
                           "n_ge3pct=%d" % (cfg, sname, len(sp), len(runs), keys_equal, smax,
                                            statistics.median(sp.values()), wk[1], sum(1 for x in sp.values() if x >= GE)))
            if not keys_equal:
                bad = sorted("%s:%s" % (k[0], k[1]) for k, v in per.items() if len(v) != len(runs))
                raise Bad("set=%s cfg=%s: sites have different run sets (keys_equal=0): %s" % (sname, cfg, ",".join(bad)))
    for cfg, site in sites:
        for sname, data in sets:
            hits = [(k, v) for k, v in data.get(cfg, {}).items() if k[1] == site]
            if len(hits) != 1:
                raise Bad("set=%s --site %s:%s matched %d sites" % (sname, cfg, site, len(hits)))
            v = hits[0][1]
            ts = [v[r] for r in sorted(v)]
            out.append("TNOISE SITE cfg=%s set=%s site=%s T=%s spread=%.4f"
                       % (cfg, sname, site, ",".join("%.4f" % t for t in ts), spread(ts)))
    out.append("TNOISE MARGIN spread_max=%.4f margin=%s margin_ok=%d" % (gmax, fmt_margin(margin), int(gmax <= margin)))
    return gmax


def fmt_margin(m):
    s = ("%.4f" % m).rstrip("0")
    return s + "0" if s.endswith(".") else s


def parse_args(argv):
    sets, sites, margin = [], [], 0.05
    it = iter(argv)
    for a in it:
        if a == "--site":
            v = next(it, None)
            if not v or ":" not in v:
                raise SystemExit(2)
            sites.append(tuple(v.split(":", 1)))
        elif a == "--margin":
            try:
                margin = float(next(it, ""))
            except ValueError:
                raise SystemExit(2)
        elif "=" in a and not a.startswith("-"):
            n, p = a.split("=", 1)
            sets.append((n, p))
        else:
            raise SystemExit(2)
    if not sets:
        raise SystemExit(2)
    return sets, sites, margin


def main(argv):
    try:
        sets, sites, margin = parse_args(argv)
    except SystemExit:
        sys.stderr.write(__doc__)
        return 2
    out = []
    try:
        loaded = [(n, load(p)) for n, p in sets]
        analyse(loaded, sites, margin, out)
    except (Bad, OSError) as e:
        print("\n".join(out))
        print("TNOISE ERROR %s" % e)
        return 1
    print("\n".join(out))
    return 0


HDR = "cfg\trun\tpart\tsite\tkind\tlaunches\tsite_ms\tL_meas\tL_hat\n"


def _tsv(d, name, rows):
    p = Path(d) / name
    p.write_text(HDR + "".join("%s\t%s\t%s\t%s\tconv\t1\t%s\t0\t0\n" % r for r in rows))
    return str(p)


def selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        print("  %s %s" % ("ok  " if cond else "FAIL", name))
        ok = ok and cond

    import io
    import contextlib
    with tempfile.TemporaryDirectory() as d:
        rows = []
        for c, base in (("off", 10.0), ("aot-kco", 2.0)):
            for r, f in (("a", 1.00), ("b", 1.02), ("c", 0.99)):
                rows.append((c, "unet_%s.%s.log" % (c, r), "1", "s1", "%.4f" % (base * f)))
                rows.append((c, r, "2", "s2", "%.4f" % (base * 2)))  # bare-letter run, zero spread
        good = _tsv(d, "g.tsv", rows)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["x=" + good, "--site", "aot-kco:s1", "--margin", "0.05"])
        o = buf.getvalue().splitlines()
        chk("rc0", rc == 0)
        exp = 1.02 / 0.99 - 1
        chk("cfg line off", "TNOISE cfg=off set=x sites=2 runs=3 keys_equal=1 spread_max=%.4f spread_med=%.4f worst=s1 n_ge3pct=1"
            % (exp, exp / 2) in o)
        chk("site T order a,b,c", "TNOISE SITE cfg=aot-kco set=x site=s1 T=2.0000,2.0400,1.9800 spread=%.4f" % exp in o)
        chk("margin ok", o[-1] == "TNOISE MARGIN spread_max=%.4f margin=0.05 margin_ok=1" % exp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["x=" + good, "--margin", "0.02"])
        chk("margin fail is finding", buf.getvalue().splitlines()[-1].endswith("margin=0.02 margin_ok=0"))
        bad = {
            "dup": rows + [rows[0]],
            "runset": rows[:-1],
            "zero": rows + [("off", "a", "3", "s3", "0")],
            "badrun": rows + [("off", "unet.log", "3", "s3", "1")],
        }
        for k, rr in bad.items():
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["x=" + _tsv(d, k + ".tsv", rr)])
            chk("rc1 " + k, rc == 1)
        with contextlib.redirect_stdout(io.StringIO()):
            chk("rc1 missing site", main(["x=" + good, "--site", "off:nope"]) == 1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            chk("rc2 usage", main([]) == 2)
    print("TNOISE SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    sys.exit(main(sys.argv[1:]))
