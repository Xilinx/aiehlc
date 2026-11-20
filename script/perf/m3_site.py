#!/usr/bin/env python3
"""Per-site K / L / N measurement of UNet AIE sites on board logs (costmodel M3).

  m3_site.py --probe --build-root DIR CFG=LOG[,LOG...] [...]

Site table: n_fit.measure (n_feat.sites of DIR/unet_<cfg> + its build.log, aligned with the log's
'PROF part=<i> op=aie_host_*' and 'PROF aie_host_split part=<i>' lines). Only sites with launches>0 count
(72 linear + 48 conv + 18 fa = 138 per cfg).
Per site (from 'PROF aie_host_split part=<i> ... dma_ms wait_ms glue_ms flush_ms inval_ms'):
  K_meas = wait_ms
  L_meas = dma_ms - glue_ms - flush_ms - inval_ms
  N_meas = (site_ms - aie_ms) + glue_ms + flush_ms + inval_ms   (= n_fit.py), so K + L + N = site_ms
  The site wait window is the __Runtime_wait_io poll only; the invalidate (inval) lies in the dma part of the
  launch window (aie_host_split.h ahs_add_window). Checked on M2c: plan formulas K=wait-inval, L=dma-glue-flush
  give off K/L sum rel 0.020/0.013, these give 0.001/0.001.
Mapped site = kind in conv|linear|fa, geometry Mw/Kw/Nw > 0, >= 1 kernel name, launches > 0.
Sums vs 'LAUNCH split' / 'LAUNCH split_detail' (per-launch ms x launches):
  K: rt_wait_io x launches; L: (kload + rt_bdcfg + rt_startio + coreen) x launches; tolerance SUM_TOL.
Output:
  M3SITE  cfg=.. log=.. board=.. launches=.. run_rc=.. sites=.. kacc_sites=.. fa_sites=..
  M3SPLIT cfg=.. log=.. K_sum_rel=.. L_sum_rel=..
  M3SUM   cfg=.. log=.. K_site_ms=.. K_split_ms=.. L_site_ms=.. L_split_ms=.. N_site_ms=.. KL_vs_rt_phase_sum_rel=..
          site_launches=.. unmapped=.. PASS|FAIL   (KL_vs_rt_phase_sum_rel printed only)
  M3PROBE ALL logs=.. sites=.. unmapped=.. PASS|FAIL      (rc=1 on FAIL)
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n_fit import measure  # noqa: E402

SITES_PER_CFG = {"linear": 72, "conv": 48, "fa": 18}
SUM_TOL = 0.03

_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) .*dma_ms=([\d.]+) wait_ms=([\d.]+) "
                    r"glue_ms=([\d.]+) flush_ms=([\d.]+) inval_ms=([\d.]+) launches=(\d+)$", re.M)
_CMP = re.compile(r"^CMP unet board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+)", re.M)
_LSPLIT = re.compile(r"^LAUNCH split kload=([\d.]+) .*coreen=([\d.]+) ", re.M)
_DETAIL = re.compile(r"^LAUNCH split_detail .*rt_bdcfg=([\d.]+) rt_startio=([\d.]+) rt_wait_io=([\d.]+) .*"
                     r"rt_phase_sum_ms=([\d.]+) ", re.M)
_LAUNCH = re.compile(r"^LAUNCH (?:warm|cold) .*?\blaunches=(\d+)", re.M)


def site_rows(build, log):
    """launches>0 sites of build with K/L/N measured from log"""
    rows, txt = measure(build, log)
    sp = {int(p): tuple(map(float, (d, w, g, f, i))) for p, _k, d, w, g, f, i, _n in _SPLIT.findall(txt)}
    out = []
    for r in rows:
        if r["launches"] <= 0:
            continue
        d, w, g, f, i = sp[r["part"]]
        mapped = (r["kind"] in SITES_PER_CFG and min(r.get("Mw", 0), r.get("Kw", 0), r.get("Nw", 0)) > 0
                  and len(r.get("kernels") or []) > 0)
        out.append(dict(r, K=w, L=d - g - f - i, mapped=mapped))
    return out, txt


def probe(cfg, log, build):
    rows, txt = site_rows(build, log)
    m = _CMP.findall(txt)
    board, launches, rc = m[-1] if m else ("-", "-1", "-1")
    kind = {k: sum(r["kind"] == k for r in rows) for k in SITES_PER_CFG}
    print("M3SITE cfg=%s log=%s board=%s launches=%s run_rc=%s sites=%d kacc_sites=%d fa_sites=%d"
          % (cfg, log, board, launches, rc, len(rows), kind["conv"] + kind["linear"], kind["fa"]))
    ls, dt, lm = _LSPLIT.findall(txt), _DETAIL.findall(txt), _LAUNCH.findall(txt)
    if ls and dt and lm:
        n = int(lm[-1])
        kload, coreen = map(float, ls[-1])
        bd, sio, wio, rps = map(float, dt[-1])
        k_split, l_split = wio * n, (kload + bd + sio + coreen) * n
    else:
        n, k_split, l_split, rps = 0, float("nan"), float("nan"), float("nan")
    k_site = sum(r["K"] for r in rows)
    l_site = sum(r["L"] for r in rows)
    n_site = sum(r["N"] for r in rows)
    k_rel = abs(k_site - k_split) / k_split if k_split > 0 else float("inf")
    l_rel = abs(l_site - l_split) / l_split if l_split > 0 else float("inf")
    print("M3SPLIT cfg=%s log=%s K_sum_rel=%.3f L_sum_rel=%.3f" % (cfg, log, k_rel, l_rel))
    kl_rel = abs(k_site + l_site - rps) / rps if rps > 0 else float("inf")
    unmapped = sum(not r["mapped"] for r in rows)
    site_launches = sum(r["launches"] for r in rows)
    ok = (rc == "0" and int(launches) == n == site_launches and kind == SITES_PER_CFG and unmapped == 0
          and k_rel <= SUM_TOL and l_rel <= SUM_TOL)
    print("M3SUM cfg=%s log=%s K_site_ms=%.3f K_split_ms=%.3f L_site_ms=%.3f L_split_ms=%.3f N_site_ms=%.3f "
          "KL_vs_rt_phase_sum_rel=%.4f site_launches=%d unmapped=%d %s"
          % (cfg, log, k_site, k_split, l_site, l_split, n_site, kl_rel, site_launches, unmapped, "PASS" if ok else "FAIL"))
    return ok, len(rows), unmapped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", required=True)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("runs", nargs="+", metavar="CFG=LOG[,LOG...]")
    a = ap.parse_args()
    root = a.build_root.rstrip("/")
    ok, nlogs, nsites, nun = True, 0, 0, 0
    for spec in a.runs:
        cfg, logs = spec.split("=", 1)
        for log in logs.split(","):
            try:
                o, s, u = probe(cfg, log, "%s/unet_%s" % (root, cfg))
            except (ValueError, KeyError, OSError) as e:
                print("M3ERR cfg=%s log=%s %s: %s" % (cfg, log, type(e).__name__, e))
                o, s, u = False, 0, 1
            ok &= o
            nlogs += 1
            nsites += s
            nun += u
    print("M3PROBE ALL logs=%d sites=%d unmapped=%d %s" % (nlogs, nsites, nun, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
