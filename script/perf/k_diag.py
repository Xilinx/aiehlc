#!/usr/bin/env python3
"""UNet K anchor diagnosis per site: measured core counters + static BD shape (costmodel M3.1).

  k_diag.py --build-root DIR CFG=LOG_A[,LOG_B] ... [--ref CFG=M2C_B_LOG ...]
  k_diag.py --build-root DIR --fit CFG=LOG_A ... --check CFG=LOG_B ... [--ref CFG=M2C_B_LOG ...]
  k_diag.py --map-only --build-root DIR CFG=LOG ...
  k_diag.py --selftest

Site table: m3_site.site_rows(DIR/unet_<cfg>, LOG) -- the launches>0 sites (72 linear + 48 conv + 18 fa = 138
per cfg, one host_split part each); NOT a count of host_split parts. Each site's
'PROF aie_site_perf part=<p> ... perf_n io_act_cyc lock_stall_cyc stream_stall_cyc' line is joined by part=
(sums over the perf_n launches that gave a reading; NEVER scaled -- M3.1.1). A part with perf_n < launches
is 'short': kept in KSITE but excluded from KSANITY / KFLOOR. Optional tail ' pre_io_act_cyc=N snap_gap_cyc=G'
(counter window, M3.1.1); without it snap_gap = 0.
Static BD shape: site_feat.bd_shape_rows (= site_feat --bd-shape), joined by (kind, cid).

Per site (ms; AIE clock AIE_MHZ = 1e3 MHz):
  K_meas  = wait_ms - inval_ms                       (PROF aie_host_split)
  core    = io_act_cyc / AIE_MHZ / 1e3
  compute = (io_act - lock_stall - stream_stall) / AIE_MHZ / 1e3
  tail    = K_meas - core
  dma     = ddr_b / BW(bucket(run_b))                ddr_b = ddr_in_b + ddr_out_b of the site's shim BDs
  bound   = core  if compute >= dma and compute >= lock + stream stalls
            sync  elif lock_stall >= stream_stall
            dma   else
K floor (--model ovl, default; M3.1.2 overlap model, needs the ' compute_cyc= ovl_cyc= dact_cyc=' tail):
  per site compute = compute_cyc / 1e6 ms, ovl = ovl_cyc / 1e6 ms, dma = ddr_b / BW_j  (j = run_b bucket)
    sum form: K_hat = max(0, compute + dma - ovl) + t0 * launches
    max form: K_hat = max(0, max(compute, dma) - ovl) + t0 * launches
  the form is chosen per cfg by the LOO max-rel on LOG_A only (KMODEL); a non-short part without the ovl fields
  -> rc=2 (an old log).  --model add (the M3.1 floor, kept for comparison):
    K_hat = max(compute, ddr_b / BW_j) + t0 * launches   (compute = io_act - stalls)
  j = run_b bucket (RUNB_EDGES, <= 3 buckets), params per cfg: BW_j of the buckets present + t0 (<= 4). Fit on LOG_A (minimax-polished NNLS over the
  max() branch assignment), check on LOG_B (never fitted), LOO = leave-one-site-out refit on LOG_A.
Sanity: io_act / (K_meas + snap_gap) <= IOK_MAX on every non-short site (core counters cannot exceed the wait
  window; snap_gap = snap_gap_cyc / AIE_MHZ / 1e3 ms, the counter-snapshot to wait-timer offset).
Classes (KCLS): conv_tkw4 (conv, kacc tkw == 4), conv_big (other conv), fa_d160 (fa, d160), fa16 (other fa),
  linear.
Output:
  KSITE   cfg run site part launches K_meas core compute tail lock stream dma bound K_hat rel
  KCLS    cfg run cls sites K_ms core_ms compute_ms tail_ms dma_ms bound=core:n/dma:n/sync:n
  KFA     cfg fa128/fa134 meas_ratio run fa128_K_ms fa134_K_ms
  KPERF   cfg=<c> run=<r> parts=<parts with launches>0> short=<n> scaled=0 ovl_fields=<parts with the ovl tail>
  KMODEL  cfg=<c> form=<sum|max> loo_sum=<x> loo_max=<x>   (--model ovl: the form with the smaller LOO on LOG_A)
  KSANITY cfg run io_act_over_K_max PASS|FAIL   (max io_act/(K_meas+snap_gap) over non-short sites; FAIL -> rc=1)
  KFIT    cfg log=A bw_<bucket>_GBps.. t0_ms params r2 maxrel loo_maxrel
  KCHK    cfg log=B r2 maxrel
  KDRIFT  cfg=<c> wait_sum_rel_vs_ref=<%.3f>   (--ref: M2c b log, built without counters)
  KDSRC   cfg log board launches run_rc sites perf_sites   (CMP line of each log; src=board needs palmyra, run_rc=0)
  KFLOOR  cfg ... PASS|FAIL;  KFLOOR ALL src cfgs sites perf_sites r2_min maxrel loo_maxrel act_over_k_max PASS|FAIL  (drift: KDRIFT, gates per-cfg KFLOOR)
--map-only (no counter lines needed):
  KDMAP   cfg site part kind launches bd_n num_dims run_b bucket ddr_b ;  KDMAP cfg=.. sites bd_missing PASS|FAIL
  KDMAP ALL logs= sites= bd_missing= PASS|FAIL   (always the last line)
rc=1 on FAIL; rc=2 when --model ovl meets a non-short part without the ovl fields.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

AIE_MHZ = 1e3
RUNB_EDGES = (64, 1024)          # bucket 0: run_b <= 64, 1: <= 1024, 2: > 1024
BUCKETS = ("r64", "r1k", "rbig")
IOK_MAX = 1.02
SITES_PER_CFG = {"linear": 72, "conv": 48, "fa": 18}
GATE = dict(r2=0.9, maxrel=0.15, loo=0.20, drift=0.05)

_PERF = re.compile(r"^PROF aie_site_perf part=(\d+) kind=(\w+) launches=(\d+) perf_n=(\d+) io_act_cyc=(\d+) "
                   r"lock_stall_cyc=(\d+) stream_stall_cyc=(\d+)(?: pre_io_act_cyc=(\d+) snap_gap_cyc=(\d+))?"
                   r"(?: compute_cyc=(\d+) ovl_cyc=(\d+)(?: dact_cyc=(\d+))?)?$", re.M)
_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) .*wait_ms=([\d.]+) .*inval_ms=([\d.]+) "
                    r"launches=(\d+)$", re.M)
_FA_D = re.compile(r"_d(\d+)_")
_CMP = re.compile(r"^CMP unet board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+)", re.M)


# ------------------------------------------------------------------ parsing / per-site split
def parse_perf(txt):
    """{part: dict(kind, launches, perf_n, io_act, lock, stream, pre_io_act, gap, short)} -- raw counter sums over
    the perf_n launches, never scaled; short = perf_n < launches (excluded from KSANITY / KFLOOR);
    M3.1.2 tail compute_cyc / ovl_cyc / dact_cyc (sums, same tile as the launch window) -> has_ovl"""
    out = {}
    for p, k, n, pn, ia, ls, ss, pre, gap, cc, oc, da in _PERF.findall(txt):
        n, pn = int(n), int(pn)
        out[int(p)] = dict(kind=k, launches=n, perf_n=pn, io_act=int(ia), lock=int(ls), stream=int(ss),
                           pre_io_act=int(pre or 0), gap=int(gap or 0), short=pn < n, has_ovl=cc != "",
                           compute_cyc=int(cc or 0), ovl_cyc=int(oc or 0), dact_cyc=int(da or 0))
    return out


def ovl_missing(pf):
    """non-short parts (launches > 0) without the ovl tail -- --model ovl refuses them (rc=2)"""
    return sum(1 for v in pf.values() if v["launches"] > 0 and not v["short"] and not v.get("has_ovl"))


def perf_ok(s):
    """site has a full (non-short) counter reading"""
    return s.get("perf") is not None and not s["perf"].get("short", False)


def kperf_line(cfg, run, pf):
    """KPERF of one log: parts with launches > 0, short parts; returns (parts, short)"""
    parts = [v for v in pf.values() if v["launches"] > 0]
    sh = sum(v["short"] for v in parts)
    print("KPERF cfg=%s run=%s parts=%d short=%d scaled=0 ovl_fields=%d"
          % (cfg, run, len(parts), sh, sum(bool(v.get("has_ovl")) for v in parts)))
    return len(parts), sh


def parse_split(txt):
    """{part: (wait_ms, inval_ms, launches)}"""
    return {int(p): (float(w), float(i), int(n)) for p, _k, w, i, n in _SPLIT.findall(txt)}


def bucket(run_b):
    for j, e in enumerate(RUNB_EDGES):
        if run_b <= e:
            return j
    return len(RUNB_EDGES)


def site_class(kind, kernels):
    k0 = kernels[0] if kernels else ""
    if kind == "fa":
        m = _FA_D.search(k0)
        return "fa_d160" if m and int(m.group(1)) == 160 else "fa16"
    if kind == "conv":
        try:
            from kcyc_unet import kacc_inst
            return "conv_tkw4" if kacc_inst(k0)["tkw"] == 4 else "conv_big"
        except ValueError:
            return "conv_big"
    return kind


def split_site(wait, inval, perf):
    """the K split of one site (ms); perf = parse_perf entry or None"""
    d = dict(K=wait - inval)
    if perf is None:
        return d
    cyc_ms = 1.0 / (AIE_MHZ * 1e3)
    d.update(core=perf["io_act"] * cyc_ms, compute=(perf["io_act"] - perf["lock"] - perf["stream"]) * cyc_ms,
             lock=perf["lock"] * cyc_ms, stream=perf["stream"] * cyc_ms, gap=perf.get("gap", 0) * cyc_ms)
    d["tail"] = d["K"] - d["core"]
    if perf.get("has_ovl"):  # overlap model terms (ms): compute_cyc / 1e6, ovl_cyc / 1e6
        d.update(oc=perf["compute_cyc"] * cyc_ms, ovl=perf["ovl_cyc"] * cyc_ms)
    return d


def bound(compute, dma, lock, stream):
    if compute >= dma and compute >= lock + stream:
        return "core"
    return "sync" if lock >= stream else "dma"


def load(cfg, log, build_root, need_perf=True):
    """site dicts of one cfg/log: m3_site rows + split + perf + BD shape"""
    from m3_site import site_rows
    from site_feat import bd_shape_rows
    build = str(Path(build_root) / ("unet_" + cfg))
    rows, txt = site_rows(build, log)
    bds, _ = bd_shape_rows(build, rows, build + ".build.log")
    bd = {(f["kind"], f["cid"]): f for f in bds}
    sp, pf = parse_split(txt), parse_perf(txt) if need_perf else {}
    out = []
    for r in rows:
        w, i, n = sp[r["part"]]
        f = bd.get((r["kind"], r["cid"]))
        s = dict(cfg=cfg, site="%s%d" % (r["kind"], r["cid"]), part=r["part"], kind=r["kind"],
                 launches=r["launches"], cls=site_class(r["kind"], r.get("kernels") or []),
                 bd_ok=bool(f and f["ok"]), run_b=f["run_b"] if f else 0,
                 ddr_b=(f["ddr_in_b"] + f["ddr_out_b"]) if f else 0, bd_n=f["bd_n"] if f else 0,
                 num_dims=f["num_dims"] if f else 0, perf=pf.get(r["part"]))
        s["bucket"] = bucket(s["run_b"])
        s.update(split_site(w, i, s["perf"]))
        out.append(s)
    return out, txt


# ------------------------------------------------------------------ K floor fit
MODELS = ("ovl", "add")
FORMS = ("sum", "max")


def _ab(s, form):
    """K_hat = max(base, dma + B) + t0 * launches; returns (base, B) of one site for the form
    add: max(compute, dma) | max: max(0, c - ovl, dma - ovl) | sum: max(0, dma + c - ovl)   (c = oc, ovl ms)"""
    if form in (None, "add"):
        return s["compute"], 0.0
    c, o = s["oc"], s["ovl"]
    if form == "max":
        return max(0.0, c - o), -o
    if form == "sum":
        return 0.0, c - o
    raise ValueError("form " + str(form))


def k_hat(p, s):
    """p = dict(inv={bucket: ms per byte}, t0=ms per launch, form=add|sum|max)"""
    base, b = _ab(s, p.get("form"))
    return max(base, s["ddr_b"] * p["inv"].get(s["bucket"], 0.0) + b) + p["t0"] * s["launches"]


def rels(p, sites):
    return [abs(k_hat(p, s) - s["K"]) / s["K"] for s in sites]


def r2(p, sites):
    y = np.array([s["K"] for s in sites])
    e = y - np.array([k_hat(p, s) for s in sites])
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((e ** 2).sum()) / tot if tot > 0 else float("nan")


def _nnls_assign(sites, bks, dma_set, form=None):
    """weighted (1/K) NNLS of (inv_j, t0) with the max() branch fixed: dma_set sites use the dma term"""
    nb = len(bks)
    X = np.zeros((len(sites), nb + 1))
    y = np.zeros(len(sites))
    for i, s in enumerate(sites):
        w = 1.0 / s["K"]
        base, b = _ab(s, form)
        if i in dma_set:
            X[i, bks.index(s["bucket"])] = s["ddr_b"] * w
            y[i] = (s["K"] - b) * w
        else:
            y[i] = (s["K"] - base) * w
        X[i, nb] = s["launches"] * w
    sc = np.maximum(np.abs(X).max(axis=0), 1e-300)
    c, _ = nnls(X / sc, y)
    c = c / sc
    return dict(inv={b: float(c[j]) for j, b in enumerate(bks)}, t0=float(c[nb]), form=form)


def fit(sites, form=None):
    """minimax-relative fit of the K floor (form: add|sum|max); params: inv per present bucket + t0"""
    bks = sorted({s["bucket"] for s in sites})
    best = None
    ab = [_ab(s, form) for s in sites]
    starts = [set(range(len(sites))), set(),
              {i for i, s in enumerate(sites) if ab[i][0] < 0.5 * s["K"]}]
    for d in starts:
        for _ in range(50):
            p = _nnls_assign(sites, bks, d, form)
            nd = {i for i, s in enumerate(sites)
                  if s["ddr_b"] * p["inv"].get(s["bucket"], 0.0) + ab[i][1] > ab[i][0]}
            if nd == d:
                break
            d = nd
        m = max(rels(p, sites))
        if best is None or m < best[0]:
            best = (m, p)
    m0, p0 = best
    if m0 > 1e-9:  # polish the max relative error (smooth in log-params, non-smooth max: Nelder-Mead)
        keys = bks
        x0 = np.log(np.maximum([p0["inv"][b] for b in keys] + [p0["t0"]], 1e-30))

        def unpack(x):
            return dict(inv={b: float(math.exp(x[j])) for j, b in enumerate(keys)}, t0=float(math.exp(x[-1])),
                        form=form)

        def obj(x):
            return max(rels(unpack(x), sites))
        r = minimize(obj, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
        if r.fun < m0:
            p0 = unpack(r.x)
    return p0


def loo(sites, form=None):
    out = 0.0
    for i in range(len(sites)):
        p = fit(sites[:i] + sites[i + 1:], form)
        out = max(out, rels(p, [sites[i]])[0])
    return out


def choose_form(sites):
    """--model ovl: fit sum and max on run a, keep the smaller LOO max-rel; returns (p, loo, {form: loo})"""
    res = {f: (fit(sites, f), loo(sites, f)) for f in FORMS}
    best = min(FORMS, key=lambda f: res[f][1])
    return res[best][0], res[best][1], {f: res[f][1] for f in FORMS}


def nparams(p):
    return len(p["inv"]) + 1


def fmt_p(p):
    bw = " ".join("bw_%s_GBps=%.4g" % (BUCKETS[b], (1.0 / v / 1e6) if v > 0 else float("inf"))
                  for b, v in sorted(p["inv"].items()))
    return "%s t0_ms=%.5g params=%d form=%s" % (bw, p["t0"], nparams(p), p.get("form") or "add")


# ------------------------------------------------------------------ reporting
def diag_lines(cfg, run, sites, p=None):
    """KSITE / KCLS / KFA / KSANITY of one log; returns io_act/K max"""
    for s in sites:
        if s.get("perf") is None:
            continue
        s.setdefault("gap", 0.0)
        s["dma"] = s["ddr_b"] * p["inv"].get(s["bucket"], 0.0) if p else 0.0
        s["bound"] = bound(s["compute"], s["dma"], s["lock"], s["stream"])
        kh = k_hat(p, s) if p else float("nan")
        print("KSITE cfg=%s run=%s %s part=%d launches=%d K_meas=%.3f core=%.3f compute=%.3f tail=%.3f lock=%.3f "
              "stream=%.3f dma=%.3f bound=%s K_hat=%.3f rel=%.3f%s"
              % (cfg, run, s["site"], s["part"], s["launches"], s["K"], s["core"], s["compute"], s["tail"],
                 s["lock"], s["stream"], s["dma"], s["bound"], kh, abs(kh - s["K"]) / s["K"],
                 " oc=%.3f ovl=%.3f" % (s["oc"], s["ovl"]) if "ovl" in s else ""))
    ps = [s for s in sites if perf_ok(s)]
    for c in ("conv_big", "conv_tkw4", "fa16", "fa_d160", "linear"):
        g = [s for s in ps if s["cls"] == c]
        print("KCLS cfg=%s run=%s cls=%s sites=%d K_ms=%.3f core_ms=%.3f compute_ms=%.3f tail_ms=%.3f dma_ms=%.3f "
              "bound=core:%d/dma:%d/sync:%d"
              % (cfg, run, c, len(g), *(sum(s[x] for s in g) for x in ("K", "core", "compute", "tail", "dma")),
                 *(sum(s["bound"] == b for s in g) for b in ("core", "dma", "sync"))))
    by = {s["site"]: s for s in sites}
    if "fa128" in by and "fa134" in by:
        a, b = by["fa128"]["K"], by["fa134"]["K"]
        print("KFA cfg=%s fa128/fa134 meas_ratio=%.3f run=%s fa128_K_ms=%.3f fa134_K_ms=%.3f"
              % (cfg, a / b if b else float("inf"), run, a, b))
    iok = max((s["core"] / (s["K"] + s.get("gap", 0.0)) for s in ps), default=float("inf"))
    print("KSANITY cfg=%s run=%s io_act_over_K_max=%.3f %s" % (cfg, run, iok, "PASS" if iok <= IOK_MAX else "FAIL"))
    return iok


def drift(sites, ref_sites):
    a, b = sum(s["K"] for s in sites), sum(s["K"] for s in ref_sites)
    return abs(a - b) / b if b > 0 else float("inf")


def map_only(a):
    tot = miss = 0
    ok_all = True
    for spec in a.runs:
        cfg, logs = spec.split("=", 1)
        for log in logs.split(","):
            sites, _ = load(cfg, log, a.build_root, need_perf=False)
            for s in sites:
                print("KDMAP cfg=%s %s part=%d kind=%s launches=%d bd_n=%d num_dims=%d run_b=%d bucket=%s ddr_b=%d%s"
                      % (cfg, s["site"], s["part"], s["kind"], s["launches"], s["bd_n"], s["num_dims"], s["run_b"],
                         BUCKETS[s["bucket"]], s["ddr_b"], "" if s["bd_ok"] else " BD_MISSING"))
            m = sum(not s["bd_ok"] for s in sites)
            kinds = {k: sum(s["kind"] == k for s in sites) for k in SITES_PER_CFG}
            uniq = len({s["site"] for s in sites}) == len(sites) and len({s["part"] for s in sites}) == len(sites)
            ok = kinds == SITES_PER_CFG and m == 0 and uniq
            print("KDMAP cfg=%s log=%s sites=%d conv=%d linear=%d fa=%d bd_missing=%d %s"
                  % (cfg, log, len(sites), kinds["conv"], kinds["linear"], kinds["fa"], m, "PASS" if ok else "FAIL"))
            tot += len(sites)
            miss += m
            ok_all &= ok
    nlogs = sum(len(x.split("=", 1)[1].split(",")) for x in a.runs)
    print("KDMAP ALL logs=%d sites=%d bd_missing=%d %s" % (nlogs, tot, miss, "PASS" if ok_all else "FAIL"))
    return ok_all


def full(a):
    """rc: 0 PASS, 1 FAIL, 2 --model ovl on a log without the ovl fields"""
    model = getattr(a, "model", "ovl") or "ovl"
    refs = dict(x.split("=", 1) for x in (a.ref or []))
    ok_all = True
    agg = dict(cfgs=0, sites=0, perf=0, r2=float("inf"), maxrel=0.0, loo=0.0, drift=0.0, iok=0.0)
    src_ok = True
    ksan_ok = True
    for spec in a.runs:
        cfg, logs = spec.split("=", 1)
        la, lb = (logs.split(",") + [None])[:2]
        sa, ta = load(cfg, la, a.build_root)
        sb, tb = load(cfg, lb, a.build_root) if lb else (sa, ta)
        pa = [s for s in sa if perf_ok(s) and s["bd_ok"] and s["K"] > 0]
        pb = [s for s in sb if perf_ok(s) and s["bd_ok"] and s["K"] > 0]
        miss = 0
        for run, lg, tx in (("a", la, ta), ("b", lb, tb)) if lb else (("a", la, ta),):
            pf = parse_perf(tx)
            kperf_line(cfg, run, pf)
            if model == "ovl" and ovl_missing(pf):
                print("KDIAG ERROR cfg=%s run=%s log=%s: %d non-short parts without compute_cyc/ovl_cyc "
                      "(pre-M3.1.2 log; --model ovl needs them, --model add does not)" % (cfg, run, lg, ovl_missing(pf)))
                miss += 1
        if miss:
            return 2
        for lg, ss, ps, tx in ((la, sa, pa, ta), (lb, sb, pb, tb)) if lb else ((la, sa, pa, ta),):
            m = _CMP.findall(tx)
            bd, nl, rc = m[-1] if m else ("?", "0", "-1")
            print("KDSRC cfg=%s log=%s board=%s launches=%s run_rc=%s sites=%d perf_sites=%d"
                  % (cfg, lg, bd, nl, rc, len(ss), len(ps)))
            src_ok = src_ok and bool(m) and bd.startswith("palmyra") and rc == "0"
        if not pa or not pb:
            print("KFLOOR cfg=%s sites=%d perf_sites=%d FAIL (no counter lines)" % (cfg, len(sb), len(pb)))
            ok_all = False
            continue
        if model == "ovl":
            p, lo, lf = choose_form(pa)
            print("KMODEL cfg=%s form=%s loo_sum=%.4f loo_max=%.4f" % (cfg, p["form"], lf["sum"], lf["max"]))
        else:
            p = fit(pa)
            lo = loo(pa)
        r2a, mra = r2(p, pa), max(rels(p, pa))
        print("KFIT cfg=%s log=%s %s r2=%.4f maxrel=%.4f loo_maxrel=%.4f" % (cfg, la, fmt_p(p), r2a, mra, lo))
        ioka = diag_lines(cfg, "a", sa, p)
        iokb = diag_lines(cfg, "b", sb, p) if lb else ioka
        ksan_ok &= max(ioka, iokb) <= IOK_MAX
        r2b, mrb = r2(p, pb), max(rels(p, pb))
        print("KCHK cfg=%s log=%s r2=%.4f maxrel=%.4f" % (cfg, lb or la, r2b, mrb))
        dr = 0.0
        if cfg in refs:
            from m3_site import site_rows
            rrows, rtxt = site_rows(str(Path(a.build_root) / ("unet_" + cfg)), refs[cfg])
            rsp = parse_split(rtxt)
            dr = drift(sb, [dict(K=rsp[r["part"]][0] - rsp[r["part"]][1]) for r in rrows])
            print("KDRIFT cfg=%s wait_sum_rel_vs_ref=%.3f" % (cfg, dr))
        kinds = {k: sum(s["kind"] == k for s in sb) for k in SITES_PER_CFG}
        ok = (kinds == SITES_PER_CFG and len(pb) == len(sb) and nparams(p) <= 4 and min(r2a, r2b) >= GATE["r2"]
              and max(mra, mrb) <= GATE["maxrel"] and lo <= GATE["loo"] and max(ioka, iokb) <= IOK_MAX
              and dr <= GATE["drift"])
        print("KFLOOR cfg=%s sites=%d perf_sites=%d r2_a=%.4f r2_b=%.4f maxrel_a=%.4f maxrel_b=%.4f loo_maxrel=%.4f "
              "io_act_over_K_max=%.3f %s" % (cfg, len(sb), len(pb), r2a, r2b, mra, mrb, lo, max(ioka, iokb),
                                              "PASS" if ok else "FAIL"))
        ok_all &= ok
        agg["cfgs"] += 1
        agg["sites"] += len(sb)
        agg["perf"] += len(pb)
        agg["r2"] = min(agg["r2"], r2a, r2b)
        agg["maxrel"] = max(agg["maxrel"], mra, mrb)
        agg["loo"] = max(agg["loo"], lo)
        agg["drift"] = max(agg["drift"], dr)
        agg["iok"] = max(agg["iok"], ioka, iokb)
    ok_all = ok_all and src_ok and ksan_ok and agg["cfgs"] > 0
    print("KFLOOR ALL src=%s cfgs=%d sites=%d perf_sites=%d r2_min=%.3f maxrel=%.3f loo_maxrel=%.3f "
          "act_over_k_max=%.3f %s"
          % ("board" if src_ok else "nonboard", agg["cfgs"], agg["sites"], agg["perf"], agg["r2"], agg["maxrel"],
             agg["loo"], agg["iok"], "PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


# ------------------------------------------------------------------ selftest (synthetic, known answer)
def _synth(seed, n, inv, t0, noise=0.0, form=None):
    """synthetic sites with known K = max(compute, ddr_b*inv_j) + t0*launches (x (1+noise_i));
    form sum|max: oc / ovl terms, ovl = u * min(oc, dma), K by that form (no clip at 0 reached)"""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        j = i % 3
        run_b = (32, 512, 10240)[j]
        L = int(rng.integers(1, 20))
        ddr_b = float(rng.integers(1, 40)) * 1e6
        dma = ddr_b * inv[j]
        compute = dma * (0.4 if i % 2 else 1.6) * float(rng.uniform(0.9, 1.1))
        ovl = float(rng.uniform(0.2, 0.8)) * min(compute, dma)
        if form == "sum":
            k0 = compute + dma - ovl
        elif form == "max":
            k0 = max(compute, dma) - ovl
        else:
            k0 = max(compute, dma)
        K = (k0 + t0 * L) * (1.0 + noise * math.sin(7.0 * i + seed))
        out.append(dict(site="s%d" % i, part=i, kind="linear", launches=L, run_b=run_b, bucket=bucket(run_b),
                        ddr_b=ddr_b, compute=compute, K=K, core=compute, lock=0.0, stream=0.0, perf={}, cls="linear",
                        oc=compute, ovl=ovl))
    return out


def selftest():
    res = []

    def chk(name, ok, info=""):
        res.append(bool(ok))
        print("KDIAG SELFTEST %-28s %s %s" % (name, "ok" if ok else "FAIL", info))

    # 1 parser: perf + split lines, perf_n < launches NOT scaled (short), launches=0 parts and other lines ignored
    txt = ("PROF aie_host_split part=7 kind=conv pack_ms=1.000 scatter_ms=0.5 dma_ms=2.000 wait_ms=9.500 glue_ms=0.1 "
           "flush_ms=0.2 inval_ms=0.500 launches=4\n"
           "PROF aie_site_perf part=7 kind=conv launches=4 perf_n=3 io_act_cyc=6000000 lock_stall_cyc=300000 "
           "stream_stall_cyc=900000\n"
           "PROF aie_host_split part=8 kind=linear pack_ms=0 scatter_ms=0 dma_ms=0 wait_ms=0 glue_ms=0 flush_ms=0 "
           "inval_ms=0 launches=0\nPROF aie_site_perf part=9 kind=fa launches=2 perf_n=2 io_act_cyc=1 x\n")
    pf, sp = parse_perf(txt), parse_split(txt)
    chk("parse_perf", set(pf) == {7} and pf[7]["io_act"] == 6e6 and pf[7]["lock"] == 3e5
        and pf[7]["stream"] == 9e5 and pf[7]["perf_n"] == 3 and pf[7]["short"] and pf[7]["gap"] == 0, pf)
    chk("parse_split", sp == {7: (9.5, 0.5, 4), 8: (0.0, 0.0, 0)}, sp)
    # 2 split: K=wait-inval=9.0, core=8e6 cyc @1 GHz=8 ms, compute=(8e6-4e5-1.2e6)=6.4 ms, tail=1.0
    d = split_site(9.5, 0.5, dict(io_act=8e6, lock=4e5, stream=1.2e6))
    chk("split", all(abs(d[k] - v) < 1e-12 for k, v in dict(K=9.0, core=8.0, compute=6.4, tail=1.0, lock=0.4,
                                                           stream=1.2, gap=0.0).items()), d)
    chk("split_no_perf", split_site(3.0, 1.0, None) == dict(K=2.0))
    # 3 bound / bucket / class
    chk("bound", (bound(6.4, 2.0, 0.4, 1.2), bound(1.0, 2.0, 0.4, 1.2), bound(1.0, 2.0, 1.5, 0.2),
                  bound(1.0, 0.5, 1.5, 0.2)) == ("core", "dma", "sync", "sync"))
    chk("bucket", [bucket(x) for x in (0, 32, 64, 65, 1024, 1025, 10240)] == [0, 0, 0, 1, 1, 2, 2])
    chk("class", [site_class("fa", ["fa_kernel__ns_fa_h8_n64_m77_d160_8_64_77_160_1_2560_3_20480_30720_10240"]),
                  site_class("fa", ["fa_kernel__ns_fa_h8_n256_m77_d80_8_256_77_80_0_2560_3_10240_15360_10240"]),
                  site_class("conv", ["gemm_kacc__640_5760_1024_2880_1024_640_4_8_720_16_10_1"]),
                  site_class("conv", ["gemm_kacc__640_640_256_320_640_256_16_32_20_10_4_0"]),
                  site_class("linear", ["gemm_kacc__640_640_256_320_640_256_16_32_20_10_4_0"])]
        == ["fa_d160", "fa16", "conv_tkw4", "conv_big", "linear"])
    # 4 fit recovers the known answer (noiseless), check on an unseen set, LOO
    inv, t0 = (1 / 2.0e6, 1 / 6.0e6, 1 / 9.0e6), 0.02   # 2 / 6 / 9 GB/s, 20 us per launch
    A, B = _synth(1, 60, inv, t0), _synth(2, 45, inv, t0)
    p = fit(A)
    perr = max([abs(p["inv"][j] - inv[j]) / inv[j] for j in range(3)] + [abs(p["t0"] - t0) / t0])
    chk("fit_exact_params", perr < 1e-6 and nparams(p) == 4, "param_maxrel=%.2e" % perr)
    chk("fit_exact_maxrel", max(rels(p, A)) < 1e-9 and r2(p, A) > 1 - 1e-12, "maxrel=%.2e" % max(rels(p, A)))
    chk("check_unseen", max(rels(p, B)) < 1e-9, "maxrel=%.2e" % max(rels(p, B)))
    lo = loo(A[:30])
    chk("loo_exact", lo < 1e-6, "loo=%.2e" % lo)
    # 5 noisy (+-4 %): minimax fit stays within the noise, params within 10 %
    An, Bn = _synth(3, 60, inv, t0, 0.04), _synth(4, 45, inv, t0, 0.04)
    pn = fit(An)
    perr = max([abs(pn["inv"][j] - inv[j]) / inv[j] for j in range(3)] + [abs(pn["t0"] - t0) / t0])
    chk("fit_noisy", max(rels(pn, An)) <= 0.045 and max(rels(pn, Bn)) <= 0.10 and perr <= 0.10 and r2(pn, Bn) > 0.99,
        "maxrel_a=%.3f maxrel_b=%.3f param_maxrel=%.3f" % (max(rels(pn, An)), max(rels(pn, Bn)), perr))
    # 6 two buckets only -> 3 params; compute-only data -> t0 exact
    A2 = [s for s in A if s["bucket"] != 1]
    p2 = fit(A2)
    chk("fit_2buckets", nparams(p2) == 3 and max(rels(p2, A2)) < 1e-9)
    # 7 drift and sanity
    chk("drift", abs(drift([dict(K=10.0), dict(K=11.0)], [dict(K=20.0)]) - 0.05) < 1e-12)
    S = [dict(site="fa128", K=4.0, core=4.0, compute=3.0, tail=0.0, lock=0.5, stream=0.5, ddr_b=1e6, bucket=2,
              launches=16, part=1, perf={}, cls="fa16"),
         dict(site="fa134", K=2.0, core=2.1, compute=1.0, tail=-0.1, lock=0.6, stream=0.5, ddr_b=1e6, bucket=2,
              launches=16, part=2, perf={}, cls="fa16")]
    iok = diag_lines("synth", "a", S, dict(inv={2: 1e-6}, t0=0.0))
    chk("sanity_iok", abs(iok - 1.05) < 1e-12 and iok > IOK_MAX)
    chk("bound_labels", [s["bound"] for s in S] == ["core", "sync"])
    # 8 M3.1.1 no scaling: a short part (perf_n < launches) is reported but excluded from KSANITY
    t2 = ("PROF aie_site_perf part=1 kind=fa launches=4 perf_n=4 io_act_cyc=3000000 lock_stall_cyc=0 stream_stall_cyc=0\n"
          "PROF aie_site_perf part=2 kind=fa launches=4 perf_n=1 io_act_cyc=1000000 lock_stall_cyc=0 stream_stall_cyc=0\n"
          "PROF aie_site_perf part=3 kind=fa launches=0 perf_n=0 io_act_cyc=0 lock_stall_cyc=0 stream_stall_cyc=0\n")
    p2f = parse_perf(t2)
    Ss = []
    for prt, K in ((1, 4.0), (2, 0.5)):
        e = dict(site="fa%d" % prt, part=prt, launches=4, cls="fa16", ddr_b=0.0, bucket=2, perf=p2f[prt])
        e.update(split_site(K, 0.0, p2f[prt]))
        Ss.append(e)
    iok2 = diag_lines("synth", "s", Ss)
    chk("short_noscale", p2f[2]["io_act"] == 1e6 and p2f[2]["short"] and not p2f[1]["short"]
        and kperf_line("synth", "s", p2f) == (2, 1) and abs(iok2 - 0.75) < 1e-12 and perf_ok(Ss[0])
        and not perf_ok(Ss[1]), "iok=%.3f (short part 2.0 excluded)" % iok2)
    # 9 snap gap: io_act/(K+gap): 5.15e6 cyc vs K=5.0 ms is 1.03 (> IOK_MAX) w/o gap; + gap 0.15 ms -> 1.000
    t3 = ("PROF aie_site_perf part=5 kind=linear launches=2 perf_n=2 io_act_cyc=5150000 lock_stall_cyc=0 "
          "stream_stall_cyc=0 pre_io_act_cyc=1234 snap_gap_cyc=150000\n")
    p3 = parse_perf(t3)[5]
    e = dict(site="linear5", part=5, launches=2, cls="linear", ddr_b=0.0, bucket=2, perf=p3)
    e.update(split_site(5.0, 0.0, p3))
    iok3 = diag_lines("synth", "g", [e])
    chk("gap_denominator", p3["gap"] == 150000 and p3["pre_io_act"] == 1234 and abs(e["gap"] - 0.15) < 1e-12
        and abs(iok3 - 5.15 / 5.15) < 1e-12 and 5.15 / 5.0 > IOK_MAX, "iok=%.3f" % iok3)
    # 10 M3.1.2 overlap model: parse the compute_cyc / ovl_cyc / dact_cyc tail (old lines -> has_ovl False)
    t4 = ("PROF aie_site_perf part=4 kind=conv launches=8 perf_n=8 io_act_cyc=5466554 lock_stall_cyc=10 "
          "stream_stall_cyc=20 pre_io_act_cyc=4136485 snap_gap_cyc=14171 compute_cyc=1176402 ovl_cyc=2842704 "
          "dact_cyc=5400000\n"
          "PROF aie_site_perf part=6 kind=fa launches=2 perf_n=2 io_act_cyc=1000 lock_stall_cyc=0 stream_stall_cyc=0 "
          "pre_io_act_cyc=1 snap_gap_cyc=2\n")
    p4 = parse_perf(t4)
    e4 = split_site(5.49, 0.0, p4[4])
    chk("ovl_parse", p4[4]["has_ovl"] and p4[4]["compute_cyc"] == 1176402 and p4[4]["ovl_cyc"] == 2842704
        and p4[4]["dact_cyc"] == 5400000 and p4[4]["gap"] == 14171 and p4[4]["io_act"] == 5466554
        and not p4[6]["has_ovl"] and p4[6]["gap"] == 2 and ovl_missing(p4) == 1
        and abs(e4["oc"] - 1.176402) < 1e-12 and abs(e4["ovl"] - 2.842704) < 1e-12
        and "ovl" not in split_site(1.0, 0.0, p4[6]), "oc=%.3f ovl=%.3f" % (e4["oc"], e4["ovl"]))
    # 11 sum / max forms: hand values, exact recovery on synthetic data, KMODEL picks the generating form by LOO
    hs = dict(oc=3.0, ovl=2.0, compute=9.9, ddr_b=1e6, bucket=0, launches=10)
    hp = dict(inv={0: 1e-6}, t0=0.01)    # dma = 1 ms
    kh = {f: k_hat(dict(hp, form=f), hs) for f in ("sum", "max", "add")}
    hz = dict(hs, oc=0.5, ovl=4.0)        # c + dma - ovl < 0 -> clipped to 0
    ok_hand = (abs(kh["sum"] - (3 + 1 - 2 + 0.1)) < 1e-12 and abs(kh["max"] - (3 - 2 + 0.1)) < 1e-12
               and abs(kh["add"] - (9.9 + 0.1)) < 1e-12 and abs(k_hat(dict(hp, form="sum"), hz) - 0.1) < 1e-12
               and abs(k_hat(dict(hp, form="max"), hz) - 0.1) < 1e-12)
    for form in FORMS:
        Af, Bf = _synth(11, 60, inv, t0, form=form), _synth(12, 45, inv, t0, form=form)
        pf_ = fit(Af, form)
        perr = max([abs(pf_["inv"][j] - inv[j]) / inv[j] for j in range(3)] + [abs(pf_["t0"] - t0) / t0])
        pc, lo_c, lf = choose_form(_synth(13, 24, inv, t0, 0.01, form=form))
        other = [f for f in FORMS if f != form][0]
        chk("ovl_model_" + form, ok_hand and perr < 1e-6 and nparams(pf_) == 4 and pf_["form"] == form
            and max(rels(pf_, Af)) < 1e-9 and max(rels(pf_, Bf)) < 1e-9 and pc["form"] == form
            and lo_c < lf[other], "param_maxrel=%.2e loo_%s=%.4f loo_%s=%.4f" % (perr, form, lf[form], other, lf[other]))
    # 12 --model ovl refuses a log whose non-short parts lack the ovl fields: rc=2 (end to end through main);
    #    --model add on the same log does not stop there
    old = ("PROF aie_site_perf part=1 kind=linear launches=2 perf_n=2 io_act_cyc=1000000 lock_stall_cyc=0 "
           "stream_stall_cyc=0 pre_io_act_cyc=1 snap_gap_cyc=2\n"
           "PROF aie_site_perf part=2 kind=linear launches=2 perf_n=1 io_act_cyc=1000000 lock_stall_cyc=0 "
           "stream_stall_cyc=0\n"
           "PROF aie_site_perf part=3 kind=linear launches=3 perf_n=3 io_act_cyc=2500000 lock_stall_cyc=0 "
           "stream_stall_cyc=0\n")
    seen = []

    def _stub_load(cfg, log, build_root, need_perf=True):
        seen.append(log)
        pf0 = parse_perf(old)
        ss = []
        for prt in (1, 2, 3):
            s = dict(cfg=cfg, site="linear%d" % prt, part=prt, kind="linear", launches=pf0[prt]["launches"], cls="linear", bd_ok=True,
                     run_b=32, ddr_b=1e6, bd_n=1, num_dims=1, perf=pf0[prt], bucket=0)
            s.update(split_site(1.0 + prt, 0.0, pf0[prt]))
            ss.append(s)
        return ss, old
    g = globals()
    real_load = g["load"]
    g["load"] = _stub_load
    try:
        rc_ovl = main(["--build-root", "/nonexistent", "synth=a.log,b.log"])
        rc_add = main(["--model", "add", "--build-root", "/nonexistent", "synth=a.log,b.log"])
    finally:
        g["load"] = real_load
    chk("ovl_missing_rc2", rc_ovl == 2 and rc_add == 1 and ovl_missing(parse_perf(old)) == 2
        and seen == ["a.log", "b.log"] * 2,
        "rc_ovl=%d rc_add=%d" % (rc_ovl, rc_add))
    n = sum(res)
    ok = n == len(res)
    print("KDIAG SELFTEST %s (%d/%d)" % ("PASS" if ok else "FAIL", n, len(res)))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--build-root")
    ap.add_argument("--model", choices=MODELS, default="ovl",
                    help="ovl (default): overlap K floor (sum|max form by LOO); add: M3.1 max(compute, dma) floor")
    ap.add_argument("--ref", nargs="+", metavar="CFG=LOG")
    ap.add_argument("--fit", nargs="+", metavar="CFG=LOG_A", help="logs the K floor is fitted on (run a)")
    ap.add_argument("--check", nargs="+", metavar="CFG=LOG_B", help="held-out logs (run b, never fitted)")
    ap.add_argument("runs", nargs="*", metavar="CFG=LOG_A[,LOG_B]")
    a = ap.parse_args(argv)
    if a.fit:  # --fit cfg=A --check cfg=B  ==  positional cfg=A,B
        chk = dict(x.split("=", 1) for x in (a.check or []))
        a.runs = list(a.runs) + [x + ("," + chk[x.split("=", 1)[0]] if x.split("=", 1)[0] in chk else "")
                                 for x in a.fit]
    elif a.check:
        ap.error("--check needs --fit")
    if a.selftest:
        return 0 if selftest() else 1
    if not a.build_root or not a.runs:
        ap.error("--build-root and CFG=LOG runs are required")
    if a.map_only:
        return 0 if map_only(a) else 1
    return full(a)


if __name__ == "__main__":
    sys.exit(main())
