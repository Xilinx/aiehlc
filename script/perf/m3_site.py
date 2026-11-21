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
  K: rt_wait_io x launches; L: (kload + rt_bdcfg + rt_startio + coreen) x launches; tolerances SUM_TOL (K), L_TOL (L / rt_phase_sum_ms), KL_TOL (K+L vs rt_phase_sum_ms).
Output:
  M3SITE  cfg=.. log=.. board=.. launches=.. run_rc=.. sites=.. kacc_sites=.. fa_sites=..
  M3SPLIT cfg=.. log=.. K_sum_rel=.. L_sum_rel=.. KL_rel=.. L_raw_rel=..   (L_raw_rel printed only)
  M3SUM   cfg=.. log=.. K_site_ms=.. K_split_ms=.. L_site_ms=.. L_split_ms=.. N_site_ms=.. KL_vs_rt_phase_sum_rel=..
          site_launches=.. unmapped=.. PASS|FAIL
  M3PROBE ALL logs=.. sites=.. unmapped=.. PASS|FAIL      (rc=1 on FAIL)

  m3_site.py --kl --build-root DIR --cycles unet_kernel_cycles.json
             (--fit CFG=LOG ... --out PARAMS.json | --params PARAMS.json) --check CFG=LOG ...
Per-site prediction of K + L (ms, whole site = launches x per-launch) against the measured K + L above.
  K kacc  des.predict_launch(instance, palmyra213.json + palmyra213_kcal.json (ddr_model=bytes, as the K2c fit),
          core_cyc of the --cycles json) wait_ms x launches; ddr_bw_scale is refit per cfg on --fit
  K fa    kfa_scale x core_cyc_site / aie_clk (FA has no ISS anchor: core_cyc is P1's formula_noloo); kfa_scale
          ('effective K' of FA) is fit per cfg on --fit
  L       sum over the site's launches of launch_l.predict_l(static per-launch features, cfg), params
          l_palmyra213.json; features: static_feat.unet_site_feat (bdcfg/startio calls, bdcfg_w/cp_w + rr), per-launch
          kload_w replayed as kload_static.kload_static, core_reset = 0: core_reset is timed inside kload
          (main_launch_line.c:28), the site L (dma - glue - flush - inval) counts kload once, while launch_l's l_def
          adds core_reset a second time (rst term).
Fit: per cfg, minimax relative error of K+L on the --fit sites; kacc sites depend only on ddr_bw_scale and fa sites
only on kfa_scale, so each is a 1-D grid (KL_S_GRID, KL_F_GRID).  LOO = leave-one-site-out refit on the fit log,
max held-out rel.  --check logs are never fitted.  Gate per cfg: every site rel <= KL_MAXREL and R2 >= KL_R2.
  KLSITE  cfg=.. kind cid launches K_meas L_meas K_pred L_pred rel
  KLGATE  cfg=.. log=.. board=.. launches=.. sites=.. ddr_bw_scale=.. kfa_scale=.. r2=.. maxrel=.. loo_maxrel=..
          sum_rel=.. (printed only) PASS|FAIL
  KLGATE  ALL src=board cfgs=.. sites=.. r2_min=.. maxrel=.. loo_maxrel=.. PASS|FAIL   (rc=1 on FAIL)

  m3_site.py --lpred --kfirst site|avg --params l_palmyra213.json --tsv OUT.tsv --build-root DIR CFG=LOG[,LOG...]
Frozen L (no refit) per site, the --kl L path (launch_l.predict_l + static_feat.unet_site_feat + kload_static
replay, core_reset = 0: the site L's kload already contains one core_reset, do not add the rst term again).
kfirst=site: per-launch kload_w (a kernel's first launch on a tile pays the PM reload); avg: whole-net mean.
Tshare = |L_hat - L_meas| / site_ms (site_ms = K + L + N); PASS per cfg iff every log has 138 sites and
Tshare_max <= L_TSHARE.  probe() lines are printed only (M3SPLIT L_sum_rel is site-vs-LAUNCH bookkeeping, not L error).
  LCHK cfg=.. logs=.. sites=.. rst=<param> kfirst=.. L_maxrel=.. Tshare_max=.. PASS|FAIL
  LFA  cfg=.. small_sites=<FA sites, launches <= 4> maxrel_avg=.. maxrel_site=..
  LCHK ALL logs=.. sites=.. kfirst=.. L_maxrel=.. Tshare_max=.. PASS|FAIL   (rc=1 on FAIL)
  TSV: cfg run part site kind launches site_ms L_meas L_hat
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from n_fit import measure  # noqa: E402

SITES_PER_CFG = {"linear": 72, "conv": 48, "fa": 18}
SUM_TOL = 0.03       # K_sum_rel = |K_site-K_split|/K_split
L_TOL = 0.01         # L_sum_rel = |L_site-L_split|/rt_phase_sum_ms
KL_TOL = 0.01        # KL_rel = |K_site+L_site-rt_phase_sum_ms|/rt_phase_sum_ms

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
    inf = float("inf")
    k_rel = abs(k_site - k_split) / k_split if k_split > 0 else inf
    # L normalised by the whole-run phase sum (rps is already the run total, not per launch):
    # the site wait uses the cntpct hs window (aie_runtime.c:8674/8750), ~2.09 ms (aot-kco) shorter
    # than rt_wait_io's pmccntr window, so that time moves K->L; L_raw_rel shows it, printed only.
    l_rel = abs(l_site - l_split) / rps if rps > 0 else inf
    kl_rel = abs(k_site + l_site - rps) / rps if rps > 0 else inf
    l_raw = abs(l_site - l_split) / l_split if l_split > 0 else inf
    print("M3SPLIT cfg=%s log=%s K_sum_rel=%.3f L_sum_rel=%.3f KL_rel=%.3f L_raw_rel=%.3f"
          % (cfg, log, k_rel, l_rel, kl_rel, l_raw))
    unmapped = sum(not r["mapped"] for r in rows)
    site_launches = sum(r["launches"] for r in rows)
    ok = (rc == "0" and int(launches) == n == site_launches and kind == SITES_PER_CFG and unmapped == 0
          and k_rel <= SUM_TOL and l_rel <= L_TOL and kl_rel <= KL_TOL)
    print("M3SUM cfg=%s log=%s K_site_ms=%.3f K_split_ms=%.3f L_site_ms=%.3f L_split_ms=%.3f N_site_ms=%.3f "
          "KL_vs_rt_phase_sum_rel=%.4f site_launches=%d unmapped=%d %s"
          % (cfg, log, k_site, k_split, l_site, l_split, n_site, kl_rel, site_launches, unmapped, "PASS" if ok else "FAIL"))
    return ok, len(rows), unmapped


# ------------------------------------------------------------------ --kl: K + L prediction per site
HERE = Path(__file__).resolve().parent
KL_MAXREL = 0.20
KL_R2 = 0.90
KL_S_GRID = [round(0.50 + 0.005 * i, 3) for i in range(501)]   # ddr_bw_scale 0.50 .. 3.00
KL_F_GRID = [round(0.05 + 0.01 * i, 2) for i in range(496)]    # kfa_scale 0.05 .. 5.00


def _g6(x):
    return float("%.6g" % x)


def k_params():
    prm = json.load(open(HERE / "params" / "palmyra213.json"))
    cal = json.load(open(HERE / "params" / "palmyra213_kcal.json"))
    prm.update({k: v for k, v in cal.items() if isinstance(v, (int, float))})
    prm["ddr_model"] = cal.get("ddr_model", "bytes")
    return prm


def kload_per_launch(build, kfirst="site"):
    """per-launch kload_w of a UNet build: the replay of kload_static.kload_static, one value per launch.
    kfirst=site: each launch its own replayed words (a kernel's first launch on a tile pays the PM reload, n_feat
    kfirst); kfirst=avg: every launch the whole-net mean (kload_static kload_w), the pre-M3 L input"""
    from aiegraph.costmodel import kswitch
    from aiegraph.costmodel import kload_static as ks
    from aiegraph.costmodel.rst_static import AIE_TILE_ROW_START
    b = Path(build)
    defs = ks.build_defs(b)
    sites = ks.load_sites(b)
    warm = bool(defs.get("PM_WARM_SKIP"))
    launches, _ = kswitch.unet_launches(str(b), require_warm=warm)
    if all(c for c, _ in sites):
        out = [float(ks.CTRL_KLOAD_W)] * len(launches)
    else:
        objdir = b / "ag_work/aiek/aiek_unetconv/worklocal/build"
        cache, res, out = {}, {}, []
        for i, ln in enumerate(launches):
            blob = ln["blob"]
            if blob not in cache:
                obj = objdir / (blob[len("_binary_"):-len("_start")] + ".o")
                cache[blob] = ks.tile_words(ks.elf_segments(ks.elf_from_obj(obj, blob)), defs)
            pm, other = cache[blob]
            w = 0
            for t in ln["tiles"]:
                if t[1] < AIE_TILE_ROW_START:
                    continue
                cold = not (warm and res.get(t) == blob)
                res[t] = blob
                w += (ks.DMA_RESET_W if i else 0) + ks.CORE_CTRL_W + other + (pm if cold else 0)
            out.append(float(w))
    ref = ks.kload_static(b)["kload_w"]
    if abs(sum(out) / len(out) - ref) > 1e-6 * max(1.0, ref):
        raise ValueError("per-launch kload_w mean %.3f != kload_static %.3f" % (sum(out) / len(out), ref))
    if kfirst == "avg":
        out = [float(ref)] * len(out)
    return out, [ln["kernel"] for ln in launches]


def l_sites(build, cfg, params=None, kfirst="site"):
    """{(kind, cid): predicted L ms of the site} from build-dir static features only"""
    from aiegraph.costmodel import launch_l
    from aiegraph.costmodel.static_feat import unet_site_feat
    from site_feat import site_feats
    lp = launch_l.load_params(str(params or HERE / "params" / "l_palmyra213.json"))
    seq = unet_site_feat(build)
    kw, kk = kload_per_launch(build, kfirst)
    if kk != [d["kernel"] for d in seq]:
        raise ValueError("kload launch order != unet_site_feat launch order")
    st, _ = site_feats(build, build + ".build.log")
    out, i = {}, 0
    for s in st:
        tot = 0.0
        for j in range(i, i + s["launches"]):
            d = seq[j]
            calls = {"bdcfg": d["bdcfg_calls"], "startio": d["startio_calls"], "core_reset": 0.0}
            words = {"bdcfg_w": d["bdcfg_w"] + d["rr"], "cp_w": d["cp_w"] + d["rr"], "kload_w": kw[j]}
            tot += launch_l.predict_l(calls, words, cfg, lp)
        out[(s["kind"], s["cid"])] = tot
        i += s["launches"]
    return out


class KLCfg:
    """one cfg: static L per site, K of kacc sites on the ddr_bw_scale grid, FA core time"""

    def __init__(self, cfg, build, cyc, prm):
        from aiegraph.costmodel import des
        self.cfg, self.build, self.prm = cfg, build, prm
        self.L = l_sites(build, cfg)
        self.cyc = {(c["kind"], c["cid"]): c for c in cyc["sites"]}
        self.clk_khz = float(prm["aie_clk_mhz"]) * 1e3
        self._des = des
        self._k = {}

    def k_launch(self, key, s):
        c = self.cyc[key]
        ik = (json.dumps(c["instance"], sort_keys=True), c["core_cyc"], s)
        if ik not in self._k:
            p = dict(self.prm, ddr_bw_scale=s)
            self._k[ik] = self._des.predict_launch({"instance": c["instance"]}, p, {"core_cyc": c["core_cyc"]})["wait_ms"]
        return self._k[ik]

    def rows(self, log):
        rows, txt = site_rows(self.build, log)
        out = []
        for r in rows:
            key = (r["kind"], r["cid"])
            c = self.cyc.get(key)
            if c is None or c["kernel"] not in (r.get("kernels") or []) or int(c["launches"]) != r["launches"]:
                raise ValueError("site %s%d: no matching --cycles entry" % key)
            out.append(dict(r, key=key, Lp=self.L[key]))
        return out, txt

    def pred(self, r, s, f):
        if r["kind"] == "fa":
            k = f * float(self.cyc[r["key"]]["core_cyc_site"]) / self.clk_khz
        else:
            k = self.k_launch(r["key"], s) * r["launches"]
        return k, r["Lp"]


def _rel_grid(kc, rows):
    """(kacc idx, fa idx, rel matrix sites x grid) of K+L on the s grid (kacc) / f grid (fa)"""
    import numpy as np
    ka = [i for i, r in enumerate(rows) if r["kind"] != "fa"]
    fa = [i for i, r in enumerate(rows) if r["kind"] == "fa"]
    rel_s = np.array([[abs(sum(kc.pred(rows[i], s, 1.0)) / (rows[i]["K"] + rows[i]["L"]) - 1.0) for s in KL_S_GRID]
                      for i in ka]).reshape(len(ka), len(KL_S_GRID))
    rel_f = np.array([[abs(sum(kc.pred(rows[i], 1.0, f)) / (rows[i]["K"] + rows[i]["L"]) - 1.0) for f in KL_F_GRID]
                      for i in fa]).reshape(len(fa), len(KL_F_GRID))
    return ka, fa, rel_s, rel_f


def _argmin_max(rel, grid, drop=None):
    """grid value minimising the max over rows of rel (row drop excluded; first on ties)"""
    import numpy as np
    if drop is not None:
        rel = np.delete(rel, drop, axis=0)
    if rel.shape[0] == 0:
        return 1.0
    return grid[int(np.argmin(rel.max(axis=0)))]


def kl_fit(kc, rows):
    """(ddr_bw_scale, kfa_scale, fit maxrel, LOO maxrel): minimax rel of K+L per site on rows; separable (kacc
    sites depend on ddr_bw_scale only, fa sites on kfa_scale only).  LOO = leave-one-site-out refit, held-out rel."""
    ka, fa, rel_s, rel_f = _rel_grid(kc, rows)
    s = _argmin_max(rel_s, KL_S_GRID)
    f = _argmin_max(rel_f, KL_F_GRID)
    fit = max(rel_s[:, KL_S_GRID.index(s)].max() if ka else 0.0, rel_f[:, KL_F_GRID.index(f)].max() if fa else 0.0)
    loo = 0.0
    for j in range(len(ka)):
        loo = max(loo, rel_s[j, KL_S_GRID.index(_argmin_max(rel_s, KL_S_GRID, j))])
    for j in range(len(fa)):
        loo = max(loo, rel_f[j, KL_F_GRID.index(_argmin_max(rel_f, KL_F_GRID, j))])
    return s, f, float(fit), float(loo)


def kl_check(kc, log, s, f, loo):
    rows, txt = kc.rows(log)
    m = _CMP.findall(txt)
    board, launches, rc = m[-1] if m else ("-", "-1", "-1")
    y, p, rels = [], [], []
    for r in rows:
        kp, lp = kc.pred(r, s, f)
        meas = r["K"] + r["L"]
        rel = abs(kp + lp - meas) / meas
        y.append(meas)
        p.append(kp + lp)
        rels.append(rel)
        print("KLSITE cfg=%s %s%d launches=%d K_meas=%.3f L_meas=%.3f K_pred=%.3f L_pred=%.3f rel=%.3f"
              % (kc.cfg, r["kind"], r["cid"], r["launches"], r["K"], r["L"], kp, lp, rel))
    mean = sum(y) / len(y)
    sst = sum((v - mean) ** 2 for v in y)
    r2 = 1.0 - sum((a - b) ** 2 for a, b in zip(y, p)) / sst if sst > 0 else float("-inf")
    mx = max(rels)
    sum_rel = abs(sum(p) - sum(y)) / sum(y)
    ok = rc == "0" and len(rows) == sum(SITES_PER_CFG.values()) and mx <= KL_MAXREL and r2 >= KL_R2
    print("KLGATE cfg=%s log=%s board=%s launches=%s sites=%d ddr_bw_scale=%.3f kfa_scale=%.2f r2=%.3f maxrel=%.3f "
          "loo_maxrel=%s sum_rel=%.3f %s" % (kc.cfg, log, board, launches, len(rows), s, f, r2, mx,
                                             "%.3f" % loo if loo is not None else "-", sum_rel,
                                             "PASS" if ok else "FAIL"))
    return ok, len(rows), r2, mx


def kl_main(a):
    root = a.build_root.rstrip("/")
    cyc = json.load(open(a.cycles))
    prm = k_params()
    kcs = {}

    def kc_of(cfg):
        if cfg not in kcs:
            kcs[cfg] = KLCfg(cfg, "%s/unet_%s" % (root, cfg), cyc, prm)
        return kcs[cfg]

    if a.fit:
        out = {"board": "palmyra213", "def": "per site K+L ms; K kacc = DES wait_ms x launches (palmyra213 + kcal, "
               "ddr_bw_scale per cfg), K fa = kfa_scale x core_cyc_site / aie_clk, L = launch_l.predict_l static "
               "(core_reset=0)", "fit": "minimax rel K+L per site on the fit log", "cfg": {}}
        for spec in a.fit:
            cfg, log = spec.split("=", 1)
            kc = kc_of(cfg)
            rows, _ = kc.rows(log)
            s, f, mx, loo = kl_fit(kc, rows)
            print("KLFIT cfg=%s log=%s sites=%d ddr_bw_scale=%.3f kfa_scale=%.2f fit_maxrel=%.3f loo_maxrel=%.3f"
                  % (cfg, log, len(rows), s, f, mx, loo))
            out["cfg"][cfg] = {"ddr_bw_scale": _g6(s), "kfa_scale": _g6(f), "fit_log": Path(log).name,
                               "fit_maxrel": _g6(mx), "loo_maxrel": _g6(loo), "sites": len(rows)}
        with open(a.out, "w") as fh:
            fh.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print("[kl] params -> %s" % a.out)
    else:
        out = json.load(open(a.params))
    ok, nsite, r2s, mxs, loos = True, 0, [], [], []
    for spec in a.check:
        cfg, log = spec.split("=", 1)
        c = out["cfg"].get(cfg)
        if c is None:
            print("KLGATE cfg=%s log=%s no fitted params FAIL" % (cfg, log))
            ok = False
            continue
        o, n, r2, mx = kl_check(kc_of(cfg), log, c["ddr_bw_scale"], c["kfa_scale"], c.get("loo_maxrel"))
        ok &= o
        nsite += n
        r2s.append(r2)
        mxs.append(mx)
        loos.append(c.get("loo_maxrel", float("nan")))
    ok &= len(r2s) == len(a.check) and len(r2s) > 0
    print("KLGATE ALL src=board cfgs=%d sites=%d r2_min=%.3f maxrel=%.3f loo_maxrel=%.3f %s"
          % (len(r2s), nsite, min(r2s) if r2s else float("nan"), max(mxs) if mxs else float("nan"),
             max(loos) if loos else float("nan"), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ------------------------------------------------------------------ --lpred: frozen L per site vs measured L
L_TSHARE = 0.10      # ASSUMED budget: |L_hat - L_meas| / site_ms per site (L and N each <= 0.10 of T)
LFA_SMALL = 4        # FA sites with launches <= LFA_SMALL: kfirst=avg vs kfirst=site
TSV_COLS = ("cfg", "run", "part", "site", "kind", "launches", "site_ms", "L_meas", "L_hat")


def lpred_main(a):
    """frozen L params (no refit) on every site of every log: Tshare = |L_hat - L_meas| / site_ms.
    probe() lines (M3SITE/M3SPLIT/M3SUM) are printed for provenance only and never enter rc."""
    from aiegraph.costmodel import launch_l
    root = a.build_root.rstrip("/")
    prm = launch_l.load_params(a.params)
    tsv = []
    ok_all, all_n, all_rel, all_ts, all_logs = True, 0, 0.0, 0.0, 0
    for spec in a.runs:
        cfg, logs = spec.split("=", 1)
        build = "%s/unet_%s" % (root, cfg)
        try:
            lh = l_sites(build, cfg, a.params, a.kfirst)
            la = l_sites(build, cfg, a.params, "avg")
        except (ValueError, KeyError, OSError) as e:
            print("LCHK cfg=%s %s: %s FAIL" % (cfg, type(e).__name__, e))
            ok_all = False
            continue
        n, mrel, mts, nlog, ok = 0, 0.0, 0.0, 0, True
        fa_avg, fa_site, fa_keys = 0.0, 0.0, set()
        for log in logs.split(","):
            try:
                probe(cfg, log, build)          # provenance only, verdict ignored
                rows, _ = site_rows(build, log)
            except (ValueError, KeyError, OSError) as e:
                print("M3ERR cfg=%s log=%s %s: %s" % (cfg, log, type(e).__name__, e))
                ok = False
                continue
            nlog += 1
            run = Path(log).name
            ok &= len(rows) == sum(SITES_PER_CFG.values())
            for r in rows:
                key = (r["kind"], r["cid"])
                site_ms = r["K"] + r["L"] + r["N"]
                lm, lp = r["L"], lh[key]
                err = abs(lp - lm)
                rel = err / lm if lm > 0 else float("inf")
                ts = err / site_ms if site_ms > 0 else float("inf")
                mrel, mts, n = max(mrel, rel), max(mts, ts), n + 1
                if r["kind"] == "fa" and r["launches"] <= LFA_SMALL:
                    fa_keys.add(key)
                    fa_site = max(fa_site, rel)
                    fa_avg = max(fa_avg, abs(la[key] - lm) / lm if lm > 0 else float("inf"))
                tsv.append((cfg, run, r["part"], "%s%d" % key, r["kind"], r["launches"], site_ms, lm, lp))
        ok &= nlog == len(logs.split(",")) and mts <= L_TSHARE
        print("LCHK cfg=%s logs=%d sites=%d rst=%.6f kfirst=%s L_maxrel=%.4f Tshare_max=%.4f %s"
              % (cfg, nlog, n, float(prm["cfg"][cfg]["coef"].get("rst", 0.0)), a.kfirst, mrel, mts,
                 "PASS" if ok else "FAIL"))
        print("LFA cfg=%s small_sites=%d maxrel_avg=%.4f maxrel_site=%.4f" % (cfg, len(fa_keys), fa_avg, fa_site))
        ok_all &= ok
        all_n, all_logs = all_n + n, all_logs + nlog
        all_rel, all_ts = max(all_rel, mrel), max(all_ts, mts)
    out = Path(a.tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\t".join(TSV_COLS) + "\n")
        for t in tsv:
            fh.write("%s\t%s\t%d\t%s\t%s\t%d\t%.4f\t%.4f\t%.4f\n" % t)
    ok_all &= all_n > 0
    print("LCHK ALL logs=%d sites=%d kfirst=%s L_maxrel=%.4f Tshare_max=%.4f %s"
          % (all_logs, all_n, a.kfirst, all_rel, all_ts, "PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--kl", action="store_true")
    mode.add_argument("--lpred", action="store_true")
    ap.add_argument("--kfirst", choices=("site", "avg"), default="site")
    ap.add_argument("--tsv")
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--cycles")
    ap.add_argument("--fit", nargs="+", metavar="CFG=LOG")
    ap.add_argument("--out")
    ap.add_argument("--params")
    ap.add_argument("--check", nargs="+", metavar="CFG=LOG")
    ap.add_argument("runs", nargs="*", metavar="CFG=LOG[,LOG...]")
    a = ap.parse_args(argv)
    if a.kl:
        if not a.cycles or not a.check or bool(a.fit) == bool(a.params) or (a.fit and not a.out):
            ap.error("--kl needs --cycles, --check and either --fit + --out or --params")
        return kl_main(a)
    if a.lpred:
        if not a.params or not a.tsv or not a.runs:
            ap.error("--lpred needs --params, --tsv and CFG=LOG[,LOG...] runs")
        return lpred_main(a)
    if not a.runs:
        ap.error("--probe needs CFG=LOG runs")
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
