#!/usr/bin/env python3
"""k_net -- static K overlap model of a gemm_kacc net (resnet18) + frozen L + N per launch (costmodel M4.1 P6).

  k_net.py --net resnet18 --build-root R --lparams L.json --nparams N.json --kcyc KCYC.json --board BOARD.json
           [--out K.json | --params K.json] [--spread CFG=X,CFG=X] [--gate 0.20] [--r2 0.90] CFG=A.log,B.log,C.log ...

Per launch (ms), every input static (build dir + frozen params), the board log is only the truth T:
  T_hat = L_hat + N_hat + K_hat
  K_hat = max(core + dma, N_ovl) - N_ovl + c      ( = the brief's  L + N_pre + max(core + dma, N_ovl) + N_post + c
                                                    with N_hat = N_pre + N_ovl + N_post already holding N_ovl )
  core  = pred_cyc / f                            pred_cyc = kcore_net (P5, KCYC json sites), f = BOARD aie_clk_mhz
  dma   = s_rd * rd_b / bw_rd + s_wr * wr_b / bw_wr + bd_n * bd_switch_cyc / f / shim_ch_n
          rd_b = ddr_b * rw_ratio, wr_b = ddr_b - rd_b, bd_n, shim_ch_n: net_kdiag (P4) static shim BD walk;
          bw_rd / bw_wr = BOARD ddr_rd_gbps / ddr_wr_gbps (palmyra213 microbench, 4 shim cols)
  N_ovl = o0 + o1 * glue_bd / launches            host glue (BD patch) running while the array runs; glue_bd static
                                                  (ap_sites, P2)
  L_hat = l_net (P1, frozen l_palmyra213.m3121), N_hat = n_fit --ap predict (P2, frozen n_resnet18 m4)
Fitted per cfg on run a only (scipy least_squares on the relative T error, bounds s_*, o* >= 0): s_rd s_wr o0 o1 c.
  LOO (run a): refit without site i, predict site i -> KLOO loo_maxrel; with --params (frozen) LOO = n/a.
Check runs (b, c; never fitted): per launch T = site_ms / launches, r2 and maxrel over the 21 sites;
  KCHK PASS iff r2 >= --r2 and maxrel + spread_med <= --gate (spread_med = P3 t_noise, --spread CFG=X).
Output:
  KSRC net cfg run log board launches sites
  KFIT net cfg run=a sites s_rd s_wr o0 o1 c fit_maxrel
  KLOO net cfg run=a sites loo_maxrel=<x|n/a> PASS|FAIL|n/a
  KSITE net cfg run part launches T T_hat L_hat N_hat K_hat core dma N_ovl rel
  KCHK net cfg run sites r2 maxrel spread_med margin k0_maxrel PASS|FAIL   (k0_maxrel: same L+N, K_hat=0, info)
  KTOPERR net cfg run part T T_hat rel K_src L_src N_src        (worst site of each check run)
  KAUDIT fn= meas=   /   KAUDIT meas_at_predict=<n>
  KALL net cfgs kchk=<n pass>/<n> kloo=<n pass> maxrel r2_min meas_at_predict PASS|FAIL
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import l_net  # noqa: E402
import n_fit  # noqa: E402
import net_kdiag  # noqa: E402
from aiegraph.costmodel import ap_sites, launch_l  # noqa: E402

PARAMS = ("s_rd", "s_wr", "o0", "o1", "c")
MEAS_FIELDS = ("site_ms", "aie_ms", "T", "glue", "compute_cyc", "lock", "stream", "N_meas", "L_meas", "K_meas", "log",
               "txt")
_RUN = re.compile(r"\.([a-z])\.log$")
_CMP = re.compile(r"^CMP \S+ board=(\S+) .*aie_launches=(\d+)", re.M)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------ static rows (build + frozen params only)
def static_rows(build, cfg, lprm, nprm, kcyc, board):
    """[dict] per site in part order: launches + every static per-launch term the predict path needs"""
    srows = ap_sites.sites(build)
    st, seq, _rs = l_net.launch_feats(build)
    lh, i = {}, 0
    for s in st:
        tot = 0.0
        for d in seq[i:i + s["launches"]]:
            tot += launch_l.predict_l({"bdcfg": d["bdcfg"], "startio": d["startio"], "core_reset": 0.0},
                                      {"bdcfg_w": d["bdcfg_w"], "cp_w": d["cp_w"], "kload_w": d["kload_w"]}, cfg, lprm)
        lh[s["part"]] = tot
        i += s["launches"]
    kc = {s["part"]: s["pred_cyc"] for s in kcyc["cfgs"][cfg]["sites"]}
    si = net_kdiag.site_inst(build)
    kb = net_kdiag.kernel_bds(os.path.join(build, net_kdiag.CNN_WL, "host.cc"))
    npc = nprm[cfg]["kinds"]["conv"]
    f = board["aie_clk_mhz"] * 1e3            # cycles per ms
    out = []
    for s in srows:
        p, n = s["part"], s["launches"]
        ks = [k for k in kb if k.startswith(si[p]["inst"])]
        if len(ks) != 1 or si[p]["launches_static"] != n or p not in kc or p not in lh:
            raise ValueError("%s part=%d: static join failed (inst=%s)" % (build, p, ks))
        fe = net_kdiag.feats_of(kb[ks[0]])
        rd = fe["ddr_b"] * fe["rw_ratio"]
        wr = fe["ddr_b"] - rd
        out.append(dict(part=p, op=s["op"], cid=s["cid"], launches=n, inst=ks[0],
                        L_hat=lh[p] / n,
                        N_hat=n_fit.ap_predict(npc, n_fit.ap_view(s, ap_sites.AP_FEATS)) / n,
                        core=kc[p] / f,
                        rd_ms=rd / (board["ddr_rd_gbps"] * 1e6), wr_ms=wr / (board["ddr_wr_gbps"] * 1e6),
                        bd_ms=fe["bd_n"] * board["bd_switch_cyc"] / f / max(1, fe["shim_ch_n"]),
                        glue_bd=s["glue_bd"] / n))
    return out


# ------------------------------------------------------------------ predict path (audited: no measured field)
def k_view(row):
    """predict view of a static row: per-launch static terms only"""
    return {x: row[x] for x in ("L_hat", "N_hat", "core", "rd_ms", "wr_ms", "bd_ms", "glue_bd")}


def k_predict(th, v):
    """(T_hat, K_hat, dma, N_ovl) per launch, ms; th = dict(PARAMS)"""
    dma = th["s_rd"] * v["rd_ms"] + th["s_wr"] * v["wr_ms"] + v["bd_ms"]
    ovl = th["o0"] + th["o1"] * v["glue_bd"]
    k = max(v["core"] + dma, ovl) - ovl + th["c"]
    return v["L_hat"] + v["N_hat"] + k, k, dma, ovl


PREDICT_FNS = (k_view, k_predict)


def audit():
    bad = 0
    for fn in PREDICT_FNS:
        names = set()
        for nd in ast.walk(ast.parse(inspect.getsource(fn))):
            if isinstance(nd, ast.Name):
                names.add(nd.id)
            elif isinstance(nd, ast.Attribute):
                names.add(nd.attr)
            elif isinstance(nd, ast.arg):
                names.add(nd.arg)
            elif isinstance(nd, ast.Constant) and isinstance(nd.value, str):
                names.add(nd.value)
        hit = sorted(names & set(MEAS_FIELDS))
        bad += len(hit)
        print("KAUDIT fn=%s meas=%s" % (fn.__name__, ",".join(hit) or "-"))
    print("KAUDIT meas_at_predict=%d" % bad)
    return bad


# ------------------------------------------------------------------ measurement (truth only)
def measure(srows, log):
    """per site T = site_ms / launches (ms per launch) from the board log, aligned with the static rows"""
    txt = open(log, errors="replace").read()
    rows, _ = n_fit.ap_measure(srows, log)
    m = _CMP.findall(txt)
    board, launches = m[-1] if m else ("-", "-1")
    return [r["site_ms"] / s["launches"] for s, r in zip(srows, rows)], board, int(launches)


# ------------------------------------------------------------------ fit (run a only)
def _snap(v):
    """stored precision: 9 significant digits, |v| < 1e-9 (a bound-active parameter) -> 0"""
    return 0.0 if abs(v) < 1e-9 else float("%.9g" % v)


def _th(x):
    return dict(zip(PARAMS, (float(v) for v in x)))


def fit(rows, T):
    from scipy.optimize import least_squares
    views = [k_view(r) for r in rows]
    T = np.asarray(T)

    def res(x):
        th = _th(x)
        return np.array([k_predict(th, v)[0] for v in views]) / T - 1.0

    lo = [0.0, 0.0, 0.0, 0.0, -np.inf]
    hi = [np.inf] * 5
    best = None
    for s in (0.5, 1.0, 2.0, 4.0):
        for o0 in (0.0, 0.3, 1.0):
            r = least_squares(res, [s, s, o0, 0.0, 0.0], bounds=(lo, hi), method="trf")
            if best is None or r.cost < best.cost - 1e-15:
                best = r
    return {k: _snap(v) for k, v in _th(best.x).items()}


def evaluate(th, rows, T):
    pr = [k_predict(th, k_view(r)) for r in rows]
    Th = np.array([p[0] for p in pr])
    T = np.asarray(T)
    rel = np.abs(Th - T) / T
    r2 = 1.0 - float(np.sum((Th - T) ** 2)) / float(np.sum((T - T.mean()) ** 2))
    return pr, rel, r2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="resnet18")
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--lparams", required=True)
    ap.add_argument("--nparams", required=True)
    ap.add_argument("--kcyc", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--out")
    ap.add_argument("--params")
    ap.add_argument("--spread", default="", help="CFG=X[,CFG=X] P3 spread_med")
    ap.add_argument("--gate", type=float, default=0.20)
    ap.add_argument("--r2", type=float, default=0.90)
    ap.add_argument("--fit-run", default="a")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args(argv)
    if a.audit:
        return 0 if audit() == 0 else 1
    n_fit.NET = a.net
    lprm = launch_l.load_params(a.lparams)
    nprm = json.load(open(a.nparams))
    kcyc = json.load(open(a.kcyc))
    board = json.load(open(a.board))
    frozen = json.load(open(a.params)) if a.params else None
    spread = {k: float(v) for k, v in (s.split("=", 1) for s in a.spread.split(",") if s)}
    srcs = {"K": "%s (%s)" % (a.params or a.out or "-", "frozen" if frozen else "fit run a"),
            "L": "%s md5=%s" % (a.lparams, md5(a.lparams)[:8]), "N": "%s md5=%s" % (a.nparams, md5(a.nparams)[:8]),
            "core": "%s md5=%s" % (a.kcyc, md5(a.kcyc)[:8])}
    out = dict(net=a.net, model=__doc__.split("Fitted")[0].strip().splitlines()[2:], fit_run=a.fit_run,
               inputs={"lparams": [a.lparams, md5(a.lparams)], "nparams": [a.nparams, md5(a.nparams)],
                       "kcyc": [a.kcyc, md5(a.kcyc)], "board": [a.board, md5(a.board)]}, cfgs={})
    nchk = npass = nloo = 0
    rmax, r2min = 0.0, 1.0
    for spec in a.runs:
        cfg, logs = spec.split("=", 1)
        build = os.path.join(a.build_root, "%s_%s" % (a.net, cfg))
        rows = static_rows(build, cfg, lprm, nprm, kcyc, board)
        runs = []
        for lg in logs.split(","):
            T, bname, nl = measure(rows, lg)
            run = _RUN.search(lg).group(1)
            print("KSRC net=%s cfg=%s run=%s log=%s board=%s launches=%d sites=%d"
                  % (a.net, cfg, run, lg, bname, nl, len(T)))
            runs.append((run, T))
        fitT = dict(runs).get(a.fit_run)
        if frozen:
            th = frozen["cfgs"][cfg]["theta"]
            print("KLOO net=%s cfg=%s run=%s sites=%d loo_maxrel=n/a n/a" % (a.net, cfg, a.fit_run, len(rows)))
            nloo += 1   # frozen: LOO n/a is not a FAIL
        else:
            th = fit(rows, fitT)
            _pr, rel, _r2 = evaluate(th, rows, fitT)
            print("KFIT net=%s cfg=%s run=%s sites=%d %s fit_maxrel=%.4f" % (
                a.net, cfg, a.fit_run, len(rows), " ".join("%s=%.6g" % (k, th[k]) for k in PARAMS), float(rel.max())))
            loo = []
            for i in range(len(rows)):
                ti = fit(rows[:i] + rows[i + 1:], fitT[:i] + fitT[i + 1:])
                loo.append(abs(k_predict(ti, k_view(rows[i]))[0] - fitT[i]) / fitT[i])
            lm = max(loo)
            ok = lm <= a.gate
            nloo += ok
            print("KLOO net=%s cfg=%s run=%s sites=%d loo_maxrel=%.4f %s" % (a.net, cfg, a.fit_run, len(rows), lm,
                                                                           "PASS" if ok else "FAIL"))
            out["cfgs"][cfg] = dict(theta={k: th[k] for k in PARAMS}, loo_maxrel=float("%.6g" % lm),
                                    fit_maxrel=float("%.6g" % float(rel.max())), sites=len(rows),
                                    fit_log=logs.split(",")[0])
        sm = spread.get(cfg, float("nan"))
        for run, T in runs:
            pr, rel, r2 = evaluate(th, rows, T)
            for r, t, p, e in zip(rows, T, pr, rel):
                print("KSITE net=%s cfg=%s run=%s part=%d launches=%d T=%.4f T_hat=%.4f L_hat=%.4f N_hat=%.4f "
                      "K_hat=%.4f core=%.4f dma=%.4f N_ovl=%.4f rel=%.4f" % (
                          a.net, cfg, run, r["part"], r["launches"], t, p[0], r["L_hat"], r["N_hat"], p[1],
                          r["core"], p[2], p[3], e))
            if run == a.fit_run and not frozen:
                continue
            mr = float(rel.max())
            mg = mr + sm
            ok = r2 >= a.r2 and mg <= a.gate
            nchk += 1
            npass += ok
            rmax, r2min = max(rmax, mr), min(r2min, r2)
            i = int(np.argmax(rel))
            print("KTOPERR net=%s cfg=%s run=%s part=%d T=%.4f T_hat=%.4f rel=%.4f K_src=%s L_src=%s N_src=%s "
                  "core_src=%s" % (a.net, cfg, run, rows[i]["part"], T[i], pr[i][0], rel[i], srcs["K"], srcs["L"],
                                   srcs["N"], srcs["core"]))
            k0 = max(abs(p[0] - p[1] - t) / t for p, t in zip(pr, T))   # reference: same L + N, K_hat = 0
            print("KCHK net=%s cfg=%s run=%s sites=%d r2=%.4f maxrel=%.4f spread_med=%.4f margin=%.4f k0_maxrel=%.4f %s"
                  % (a.net, cfg, run, len(rows), r2, mr, sm, mg, k0, "PASS" if ok else "FAIL"))
    bad = audit()
    if a.out and not frozen:
        with open(a.out, "w") as fo:
            json.dump(out, fo, indent=1, sort_keys=True)
            fo.write("\n")
        print("KPARAMS path=%s md5=%s" % (a.out, md5(a.out)))
    ok = nchk > 0 and npass == nchk and nloo == len(a.runs) and bad == 0
    print("KALL net=%s cfgs=%d kchk=%d/%d kloo=%d maxrel=%.4f r2_min=%.4f meas_at_predict=%d %s" % (
        a.net, len(a.runs), npass, nchk, nloo, rmax, r2min, bad, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
