#!/usr/bin/env python3
"""Check of the overlap-model input ovl_cyc on UNet board logs (costmodel M3.1.2.2 PX).

  k_ovl_check.py --build-root DIR CFG=L1,L2[,L3...] [CFG=...]
  k_ovl_check.py --selftest

Question: is the counter-path ovl (host time of a launch's A->B core-counter window minus its waits,
aie_runtime.c rt_core_perf_launch_snap) consistent with the host glue the N segment measures in the same launches?
The core can only overlap the host while the host does glue between startio and wait, so ovl can not exceed that.

Per launches>0 site of m3_site.site_rows (138 per cfg), sums over the site's perf_n launches:
  ovl_ms  = ovl_cyc / (AIE_MHZ * 1e3)           ('PROF aie_site_perf' tail, k_diag.parse_perf)
  glue_ms = AHS glue_ms * perf_n / launches     ('PROF aie_host_split' glue = (sp2-sp1)-(rt1-rt0): the launch call
                                                 minus runtime phases + wait_io -- lock init, createio, commit; the
                                                 n_diag.py field between startio and wait)
  win_ms  = (dma_ms - flush_ms - inval_ms) * perf_n / launches   (all non-wait host time inside the launch windows
                                                 after the flush and before the inval: glue + kload + bdcfg + startio
                                                 + coreen -- the loosest AHS bound of ovl, printed only)
  ratio = ovl_ms / glue_ms;  viol iff ovl_ms > VIOL_TOL * glue_ms  (win_viol likewise against win_ms)
Output:
  KOVL cfg=<c> run=<r> part=<p> kind=<k> ovl_ms=<x> glue_ms=<x> ratio=<x> win_ms=<x> win_ratio=<x>
  KOVL SUM cfg=<c> runs=<n> site_runs=<n> viol=<n> win_viol=<n> ratio_med=<x> ratio_max=<x>
  KOVL ALL logs=<n> site_runs=<n> viol=<n> ratio_med=<x>          (always the last line)
viol > 0 is the diagnostic finding, not an error.  rc=1 on unparsable input (a site without its aie_site_perf ovl
tail or its aie_host_split line, a log that fails to load).
"""
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VIOL_TOL = 1.02
_AHS = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) (.*)$", re.M)
_MS = re.compile(r"(\w+)_ms=([\d.]+)")


def parse_ahs(txt):
    """{part: {pack, scatter, dma, wait, glue, flush, inval: ms}} -- same field set as n_diag.load"""
    return {int(p): {k: float(v) for k, v in _MS.findall(rest)} for p, _k, rest in _AHS.findall(txt)}


def site_check(perf, ahs, aie_mhz):
    """(ovl_ms, glue_ms, win_ms) of one site over its perf_n launches; raises KeyError/ValueError when unusable"""
    if not perf.get("has_ovl"):
        raise ValueError("no compute_cyc/ovl_cyc tail")
    n, pn = perf["launches"], perf["perf_n"]
    if n <= 0 or pn <= 0:
        raise ValueError("launches=%d perf_n=%d" % (n, pn))
    sc = min(pn, n) / float(n)
    ovl = perf["ovl_cyc"] / (aie_mhz * 1e3)
    glue = ahs["glue"] * sc
    win = (ahs["dma"] - ahs["flush"] - ahs["inval"]) * sc
    return ovl, glue, win


def _ratio(o, g):
    return o / g if g > 0 else (0.0 if o <= 0 else float("inf"))


def check_log(cfg, run, txt, rows, aie_mhz):
    """KOVL lines of one log; returns list of (ratio, viol, win_viol)"""
    from k_diag import parse_perf
    pf, sp = parse_perf(txt), parse_ahs(txt)
    out = []
    for r in rows:
        p = r["part"]
        if p not in pf:
            raise KeyError("part %d: no PROF aie_site_perf line" % p)
        if p not in sp:
            raise KeyError("part %d: no PROF aie_host_split line" % p)
        o, g, w = site_check(pf[p], sp[p], aie_mhz)
        ra, rw = _ratio(o, g), _ratio(o, w)
        print("KOVL cfg=%s run=%s part=%d kind=%s ovl_ms=%.4f glue_ms=%.4f ratio=%.3f win_ms=%.4f win_ratio=%.3f"
              % (cfg, run, p, r["kind"], o, g, ra, w, rw))
        out.append((ra, o > VIOL_TOL * g, o > VIOL_TOL * w))
    return out


def run_all(build_root, specs, aie_mhz, rows_of):
    root = build_root.rstrip("/")
    ok, nlogs, allr = True, 0, []
    for spec in specs:
        cfg, logs = spec.split("=", 1)
        cr = []
        for i, lg in enumerate(logs.split(",")):
            run = "abcdefgh"[i] if i < 8 else str(i)
            try:
                rows, txt = rows_of("%s/unet_%s" % (root, cfg), lg)
                cr += check_log(cfg, run, txt, rows, aie_mhz)
                nlogs += 1
            except (ValueError, KeyError, OSError) as e:
                print("KOVL ERROR cfg=%s run=%s log=%s %s: %s" % (cfg, run, lg, type(e).__name__, e))
                ok = False
        if cr:
            rs = [x[0] for x in cr]
            print("KOVL SUM cfg=%s runs=%d site_runs=%d viol=%d win_viol=%d ratio_med=%.3f ratio_max=%.3f"
                  % (cfg, len(logs.split(",")), len(cr), sum(x[1] for x in cr), sum(x[2] for x in cr),
                     statistics.median(rs), max(rs)))
        allr += cr
    med = statistics.median([x[0] for x in allr]) if allr else float("nan")
    print("KOVL ALL logs=%d site_runs=%d viol=%d ratio_med=%.3f" % (nlogs, len(allr), sum(x[1] for x in allr), med))
    return 0 if ok and allr else 1


def selftest():
    import io
    from contextlib import redirect_stdout
    from k_diag import AIE_MHZ
    res = []

    def chk(name, c, info=""):
        res.append(c)
        print("KOVL SELFTEST %s %s %s" % (name, "ok" if c else "FAIL", info))

    def perf(p, n, pn, ovl, tail=True):
        s = ("PROF aie_site_perf part=%d kind=conv launches=%d perf_n=%d io_act_cyc=100 lock_stall_cyc=1 "
             "stream_stall_cyc=2 pre_io_act_cyc=3 snap_gap_cyc=4" % (p, n, pn))
        return s + (" compute_cyc=50 ovl_cyc=%d dact_cyc=120\n" % ovl if tail else "\n")

    def ahs(p, glue, dma=5.0, flush=0.5, inval=0.1, n=2):
        return ("PROF aie_host_split part=%d kind=conv pack_ms=1.000 scatter_ms=0.500 dma_ms=%.3f wait_ms=3.000 "
                "glue_ms=%.3f flush_ms=%.3f inval_ms=%.3f launches=%d\n" % (p, dma, glue, flush, inval, n))
    # 1 parse: field set of n_diag (pack/scatter/dma/wait/glue/flush/inval)
    a = parse_ahs(ahs(7, 1.25))
    chk("parse_ahs", a[7]["glue"] == 1.25 and a[7]["inval"] == 0.1 and len(a[7]) == 7, str(a[7]))
    # 2 site math: ovl 1e6 cyc = 1 ms @1000 MHz; glue / win scaled by perf_n/launches
    from k_diag import parse_perf
    pf = parse_perf(perf(7, 4, 2, 1000000))
    o, g, w = site_check(pf[7], parse_ahs(ahs(7, 2.0))[7], 1000)
    chk("site_math", abs(o - 1.0) < 1e-12 and abs(g - 1.0) < 1e-12 and abs(w - 2.2) < 1e-12, "%g %g %g" % (o, g, w))
    chk("no_tail", _raises(lambda: site_check(parse_perf(perf(7, 2, 2, 5, False))[7], a[7], 1000)))
    chk("ratio_zero_glue", _ratio(0.0, 0.0) == 0.0 and _ratio(1.0, 0.0) == float("inf"))
    # 3 end to end: 3 sites, one viol (ovl 1.5 ms vs glue 1.0), one exactly at 1.02 (not viol), one under
    mhz = AIE_MHZ
    txt = (perf(1, 1, 1, int(1.5e3 * mhz)) + ahs(1, 1.0, n=1) + perf(2, 1, 1, int(1.02e3 * mhz)) + ahs(2, 1.0, n=1)
           + perf(3, 1, 1, int(0.5e3 * mhz)) + ahs(3, 1.0, n=1))
    rows = [dict(part=p, kind="conv") for p in (1, 2, 3)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_all("/x", ["c=L1,L2"], mhz, lambda b, lg: (rows, txt))
    lines = buf.getvalue().splitlines()
    per = [ln for ln in lines if ln.startswith("KOVL cfg=")]
    chk("e2e_lines", rc == 0 and len(per) == 6 and per[0].startswith("KOVL cfg=c run=a part=1 kind=conv ovl_ms=1.5000 "
        "glue_ms=1.0000 ratio=1.500") and per[3].startswith("KOVL cfg=c run=b part=1"), per[0] if per else "")
    chk("e2e_all", lines[-1] == "KOVL ALL logs=2 site_runs=6 viol=2 ratio_med=1.020", lines[-1])
    # 4 missing site -> rc=1, still an ALL line last
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_all("/x", ["c=L1"], mhz, lambda b, lg: (rows + [dict(part=9, kind="fa")], txt))
    out = buf.getvalue().splitlines()
    chk("missing_rc1", rc == 1 and "KOVL ERROR" in buf.getvalue() and out[-1].startswith("KOVL ALL logs=0"), out[-1])
    ok = all(res)
    print("KOVL SELFTEST %s (%d checks)" % ("PASS" if ok else "FAIL", len(res)))
    return ok


def _raises(f):
    try:
        f()
    except (ValueError, KeyError):
        return True
    return False


def main(argv=None):
    import argparse
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        return 0 if selftest() else 1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("runs", nargs="+", metavar="CFG=LOG[,LOG...]")
    a = ap.parse_args(argv)
    from k_diag import AIE_MHZ
    from m3_site import site_rows
    return run_all(a.build_root, a.runs, AIE_MHZ, site_rows)


if __name__ == "__main__":
    sys.exit(main())
