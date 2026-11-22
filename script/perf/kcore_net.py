#!/usr/bin/env python3
"""kcore_net -- static per-launch core cycles (K_core) for every AIE site of a gemm_kacc net vs measured compute_cyc
(costmodel M4.1 P5; no simulator).

  kcore_net.py --net resnet18 --build-root DIR --cc-root DIR [--fit-run a] [--params-out JSON] [--max-rel X]
               CFG=LOG_A,LOG_B[,LOG_C] ...

Static source per site (DIR/<net>_<cfg>.build.log 'aie conv|linear <cid> ... inst=NxKxM launches=n' + the
  'GRAPH_PROF_END(<part>, "aie aiednn_<conv|dense> <cid>")' lines of DIR/<net>_<cfg>/ag_work/arm_build/graph_driver.c):
  the instance's kernel.lst from --cc-root/<cfg>/gemm_kacc__N_K_M_*/build/kernel.lst (arthur tools/aie_cc_only.sh,
  compile-only kc.sh chess flow) -> kcyc_lst.analyse: total = core main (incl. the aot-kco register-write prelude)
  + kernel, issued bundles x static trip counts; pre = cycles before the kernel's first input-window ACQ
  (prologue + first-tile accumulator memset). Cross-checked (KCORE_LSTCHK) against the board build's own
  ag_work/aiek/aiek_cnn/worklocal/build/kpar/<inst>/build/kernel.lst: kernel cycles must be equal.
Window: compute_cyc is counted from launch window A (host snapshot, off: before the first startio; aot-kco: before the
  first wait_io's commit), but the core starts at core enable, so the first ~W cycles of the launch (all inside the
  pre-input part: prologue + memset of tile 0) are outside the counter window:
    pred_cyc(site, per launch) = total - min(pre, W_cfg)
  W_cfg (one scalar per cfg, AIE cycles) = argmin over a 10-cycle grid on [0, max pre] of the --fit-run log's max
  relative error over its sites: the only fitted number; the other runs of the cfg are a check. W does not scale
  any core cycle count (it only moves the window start inside the static pre part).
Measured: 'PROF aie_site_perf part=<p> ... launches=<n> perf_n=<n> ... compute_cyc=<sum>' -> meas = compute_cyc/perf_n
  (perf_n must equal launches). rel = |pred - meas| / meas per site and log; maxrel = max over sites and logs.
expect = number of AIE sites counted in the log ('PROF part=<p> op=aie_host_*' lines; equal over the cfg's logs).
Output:
  KCORE_LSTCHK cfg= inst= cc_kernel= board_kernel= total= pre= vmac= frame= spill= eq=0|1
  KCORE_SITE cfg= part= site=<kind><cid> inst= launches= pred_cyc= meas_cyc=<median> rel_max=
  KCORE_RUN cfg= run= sites= maxrel=
  KCORE net=<net> cfg=<cfg> sites=<n> expect=<n> coverage=<0|1> maxrel=<x.xxxx>   (one per cfg)
  KCORE_ALL net= cfgs= spill=<n> lstchk=<ok>/<n> maxrel= PASS|FAIL
PASS iff every cfg has coverage=1 (every expected site has a finite prediction and a measurement), all LSTCHK eq=1,
and maxrel <= --max-rel (default 0.20); SPILL is reported (KCORE_SPILL), not a FAIL. rc 0 PASS, 1 FAIL.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kcyc_lst  # noqa: E402

_BL = re.compile(r"aie (conv|linear)\s+(\d+) .*?\binst=(\d+)x(\d+)x(\d+) launches=(\d+)")
_GD = re.compile(r'GRAPH_PROF_END\((\d+), "aie aiednn_(conv|dense) (\d+)"\)')
_PERF = re.compile(r"^PROF aie_site_perf part=(\d+) kind=(\w+) launches=(\d+) perf_n=(\d+) .*?\bcompute_cyc=(\d+)", re.M)
_SITE = re.compile(r"^PROF part=(\d+) op=aie_host_\w+ ", re.M)
_RUN = re.compile(r"\.([a-z])\.log$")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sites(build_root, net, cfg):
    """[(part, kind, cid, (N,K,M), launches)] in part order"""
    bdir = os.path.join(build_root, "%s_%s" % (net, cfg))
    bl = {}
    for m in _BL.finditer(open(bdir + ".build.log", errors="replace").read()):
        kind = "conv" if m.group(1) == "conv" else "dense"
        bl[(kind, int(m.group(2)))] = ((int(m.group(3)), int(m.group(4)), int(m.group(5))), int(m.group(6)))
    gd = open(os.path.join(bdir, "ag_work", "arm_build", "graph_driver.c"), errors="replace").read()
    out = []
    for m in _GD.finditer(gd):
        part, kind, cid = int(m.group(1)), m.group(2), int(m.group(3))
        inst, ln = bl[(kind, cid)]
        out.append((part, kind, cid, inst, ln))
    return sorted(out)


def inst_dir(root, inst):
    g = sorted(glob.glob(os.path.join(root, "gemm_kacc__%d_%d_%d_*" % inst)))
    g = [x for x in g if os.path.isdir(x)]
    if len(g) != 1:
        raise ValueError("%s: %d dirs for instance %s" % (root, len(g), inst))
    return g[0]


def meas(log):
    txt = open(log, errors="replace").read()
    per = {}
    for m in _PERF.finditer(txt):
        part, launches, pn, cc = int(m.group(1)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        per[part] = dict(launches=launches, perf_n=pn, cyc=cc / pn if pn else float("nan"))
    expect = len(set(int(x) for x in _SITE.findall(txt)))
    return per, expect


def fnum(x):
    return "%.4f" % x if isinstance(x, float) and math.isfinite(x) else "nan"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="resnet18")
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--cc-root", required=True)
    ap.add_argument("--fit-run", default="a")
    ap.add_argument("--params-out")
    ap.add_argument("--max-rel", type=float, default=0.20)
    ap.add_argument("cfgs", nargs="+")
    a = ap.parse_args(argv)
    ok, worst, n_spill, lst_ok, lst_n = True, 0.0, 0, 0, 0
    params = dict(net=a.net, build_root=a.build_root, cc_root=a.cc_root, fit_run=a.fit_run,
                  model="pred_cyc = total - min(pre, W_cfg); total/pre static (kcyc_lst on aie_cc_only kernel.lst)",
                  cfgs={})
    for spec in a.cfgs:
        cfg, logs = spec.split("=", 1)
        logs = logs.split(",")
        st = sites(a.build_root, a.net, cfg)
        kpar = os.path.join(a.build_root, "%s_%s" % (a.net, cfg), "ag_work", "aiek", "aiek_cnn", "worklocal", "build",
                            "kpar")
        stat = {}
        for inst in sorted({s[3] for s in st}):
            d = inst_dir(os.path.join(a.cc_root, cfg), inst)
            lst = os.path.join(d, "build", "kernel.lst")
            r = kcyc_lst.analyse(lst)
            b = kcyc_lst.analyse(os.path.join(inst_dir(kpar, inst), "build", "kernel.lst"))
            eq = int(r["kernel"] == b["kernel"])
            lst_n += 1
            lst_ok += eq
            n_spill += r["spill"]
            name = os.path.basename(d)
            stat[inst] = dict(name=name, total=r["total"], kernel=r["kernel"], main=r["main"], pre=r["pre"],
                              post=r["post"], vmac=r["vmac"], frame=r["frame"], spill=r["spill"], lst=lst,
                              lst_md5=md5(lst), board_kernel=b["kernel"])
            print("KCORE_LSTCHK cfg=%s inst=%s cc_kernel=%d board_kernel=%d total=%d pre=%d vmac=%d frame=%d spill=%d "
                  "eq=%d" % (cfg, name, r["kernel"], b["kernel"], r["total"], r["pre"], r["vmac"], r["frame"],
                              r["spill"], eq))
            if r["spill"]:
                print("KCORE_SPILL cfg=%s inst=%s sp_ldst=%d" % (cfg, name, r["spill"]))
        runs = {}
        for lg in logs:
            m = _RUN.search(lg)
            runs[m.group(1) if m else lg] = (lg,) + meas(lg)
        expects = {r[2] for r in runs.values()}
        expect = expects.pop() if len(expects) == 1 else -1
        # W_cfg from the fit run: minimax relative error over its sites, 1-D grid (10 cyc) on [0, max pre]
        fl = runs.get(a.fit_run)
        pts = []
        for part, kind, cid, inst, ln in st:
            m = fl[1].get(part) if fl else None
            if m and m["cyc"] > 0:
                pts.append((stat[inst]["total"], stat[inst]["pre"], m["cyc"]))
        W, best = float("nan"), float("inf")
        if pts:
            for w in range(0, max(p[1] for p in pts) + 10, 10):
                e = max(abs(t - min(pr, w) - y) / y for t, pr, y in pts)
                if e < best - 1e-12:
                    W, best = float(w), e
        lags = pts
        cov, cmax, sitep = 0, 0.0, []
        rmax = {k: 0.0 for k in runs}
        for part, kind, cid, inst, ln in st:
            s = stat[inst]
            pred = s["total"] - min(s["pre"], W)
            ms, rels, good = [], [], True
            for k, (lg, per, _) in runs.items():
                m = per.get(part)
                if not m or m["perf_n"] != m["launches"] or m["launches"] != ln or not m["cyc"] > 0:
                    good = False
                    continue
                ms.append(m["cyc"])
                rel = abs(pred - m["cyc"]) / m["cyc"]
                rels.append(rel)
                rmax[k] = max(rmax[k], rel)
            good = good and math.isfinite(pred) and len(ms) == len(runs)
            cov += good
            rm = max(rels) if rels else float("nan")
            if good:
                cmax = max(cmax, rm)
            site = "%s%d" % ("conv" if kind == "conv" else "linear", cid)
            print("KCORE_SITE cfg=%s part=%d site=%s inst=%s launches=%d pred_cyc=%.0f meas_cyc=%.0f rel_max=%s" %
                  (cfg, part, site, s["name"], ln, pred, statistics.median(ms) if ms else float("nan"), fnum(rm)))
            sitep.append(dict(part=part, site=site, inst=s["name"], launches=ln, pred_cyc=round(pred, 1)))
        for k, v in sorted(rmax.items()):
            print("KCORE_RUN cfg=%s run=%s sites=%d maxrel=%s%s" % (cfg, k, len(runs[k][1]), fnum(v),
                                                                    " (fit W)" if k == a.fit_run else ""))
        coverage = int(cov == len(st) == expect and expect > 0)
        print("KCORE_W cfg=%s W_cyc=%.0f fit_run=%s fit_sites=%d" % (cfg, W, a.fit_run, len(lags)))
        print("KCORE net=%s cfg=%s sites=%d expect=%d coverage=%d maxrel=%s" % (a.net, cfg, len(st), expect, coverage,
                                                                               fnum(cmax)))
        ok = ok and coverage == 1
        worst = max(worst, cmax)
        params["cfgs"][cfg] = dict(W_cyc=W, fit_sites=len(lags), logs=logs,
                                   instances={v["name"]: {x: v[x] for x in v if x != "name"} for v in stat.values()},
                                   sites=sitep)
    ok = ok and lst_ok == lst_n and worst <= a.max_rel  # SPILL is reported (KCORE_SPILL), not a FAIL: write a GAP
    print("KCORE_ALL net=%s cfgs=%d spill=%d lstchk=%d/%d maxrel=%s %s" % (a.net, len(a.cfgs), n_spill, lst_ok, lst_n,
                                                                         fnum(worst), "PASS" if ok else "FAIL"))
    if a.params_out:
        with open(a.params_out, "w") as f:
            json.dump(params, f, indent=1, sort_keys=True)
            f.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
