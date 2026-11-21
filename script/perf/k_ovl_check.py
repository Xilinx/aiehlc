#!/usr/bin/env python3
"""Check of the overlap-model input ovl_cyc on UNet board logs (costmodel M3.1.2.2 PX).

  k_ovl_check.py --build-root DIR [--off-corr PARAMS.json] CFG=L1,L2[,L3...] [CFG=...]
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

--off-corr PARAMS (costmodel M3.1.2.2.3 QP, probe; single value, before the CFG=LOG positionals): static correction of
the cfg=off glue bound.  off takes snapshot A before the launch's FIRST __Runtime_startio (aie_runtime.c:402), so every
rt bdcfg / startio call executed after it lies inside the A->B window but is not in AHS glue (AHS removes the runtime
phases).  Per launch, from the build's host_canonicalized_<kernel> (ordered static walk, same loop-trip counting as
static_feat.host_counts / l_static, M1.2.2: 0 error on the call counts):
  win_bdcfg / win_startio = calls executed at or after the first startio (a call before it in a loop shared with that
                            startio counts in every iteration but the first)
  corr_ms   = sum over the site's launches (unet_site_feat order) of win_bdcfg x c_bdcfg + win_startio x c_startio,
              x perf_n / launches;  c_* = PARAMS cfg.off.coef.bdcfg_calls / startio_calls (ms per call, l_palmyra213)
  glue_corr = glue_ms + corr_ms;  viol iff ovl_ms > VIOL_TOL x glue_corr
aot-kco is never corrected (A is taken after its startio/bdcfg, at the first wait_io's commit).
  KOVL SRC cfg=<c> log=<log> board=<CMP board> launches=<CMP aie_launches> off_corr=<PARAMS|none>   (one per log)
  KOVL cfg=off ... (as above) raw_viol=<0|1> corr_ms=<x> glue_corr_ms=<x> ratio_corr=<x>   (off with --off-corr)
  KOVL SUM cfg=off mode=corr runs=.. site_runs=.. viol=<corrected> raw_viol=.. win_viol=.. ratio_med=<corr> ...
  KOVL STILL cfg=off kind=<k> part=<p> runs=<n> ratio_corr_max=<x> ovl_ms=.. glue_corr_ms=..   (site still violating)
  KOVL STILLSUM cfg=off sites=.. by_kind=<k:n,...>
rc=1 also when cfg=aot-kco has viol>0.

--npred NPRED.tsv [--delta-tsv TSV] (costmodel M3.1.1.2c PV; replaces the KOVL output): static ovl_hat of the n_fit
--predict TSV (site total ms over the site's launches; PN 33aa8e35) against the counter ovl of each log, per launch:
  ovl_meas = ovl_cyc / (AIE_MHZ*1e3) / min(perf_n, launches)   (always the counter: ovl_src=counter; the npred
             ovl_meas column is only cross-checked, npred_meas_mismatch = sites where it differs)
  ovl_hat  = npred ovl_hat / launches;  site_ms = npred site_ms / launches
  OVLSRC cfg=<c> log=<log> board=<CMP board> launches=<CMP aie_launches> run_rc=<CMP run_rc> sites=<n>
  OVLHAT cfg=<c> run=<r> sites=<n> ovl_src=counter r2=<site totals, OVLP basis> maxrel=<max|h-m|/m> sum_rel=
         <sum h/sum m - 1> share_max=<max|h-m|/site_ms> r2_launch=<per launch> npred_meas_mismatch=<n> PASS|FAIL
         (PASS iff r2 >= 0.9 and maxrel <= 0.20)
  KDELTA term=ovl rows=<n> tsv=<TSV>
  TSV (tab, header): cfg run part site delta_ms   delta_ms = -(ovl_hat - ovl_meas) per launch (the K increment
  when the measured ovl is replaced by ovl_hat in K = core + wait - ovl).
rc 0 every OVLHAT PASS, 1 a metric missed (TSV complete), 2 unusable input (old TSV removed, none written).
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


BDCFG_APIS = ("__Runtime_dma_bd_config", "__Runtime_dma_bd_config_multidim", "__Runtime_dma_bd_config_multidim_ooo")
STARTIO_API = "__Runtime_startio"
_FOR = re.compile(r"for \(size_t (v\d+) = (\w+); \1 < (\w+); \1 \+= (\w+)\)")
_CONST = re.compile(r"^\s*size_t (v\d+) = (\d+);")
_CALL = re.compile(r"\b(__Runtime_[A-Za-z0-9_]+)\(")
_CMP = re.compile(r"^CMP unet board=(\S+) .*?\baie_launches=(\d+)", re.M)


def _ordered_walk(txt, fn):
    """[(code, executions per launch, (loop ids), (loop trips))] of host function fn in source order (same loop
    handling as static_feat._walk: product of the enclosing `for (size_t v = vA; v < vB; v += vC)` trips)"""
    m = re.search(r"^void %s\(.*?\{\n(.*?)^\}" % re.escape(fn), txt, re.S | re.M)
    if not m:
        raise ValueError("no %s in host.cc" % fn)
    consts, stack, depth, out = {}, [], 0, []
    for li, line in enumerate(m.group(1).split("\n")):
        code = re.sub(r"/\*.*?\*/", "", line)
        cm = _CONST.match(code)
        if cm:
            consts[cm.group(1)] = int(cm.group(2))
        fm = _FOR.search(code)
        if fm:
            def val(x):
                return int(x) if x.isdigit() else consts[x]
            lo, hi, st = val(fm.group(2)), val(fm.group(3)), val(fm.group(4))
            trip = max(0, (hi - lo + st - 1) // st)
            after = code[fm.end():]
            stack.append((depth, trip, li))
            depth += after.count("{") - after.count("}")
            continue
        mult = 1
        for _d, t, _i in stack:
            mult *= t
        out.append((code, mult, tuple(x[2] for x in stack), tuple(x[1] for x in stack)))
        depth += code.count("{") - code.count("}")
        while stack and depth <= stack[-1][0]:
            stack.pop()
    return out


def inwin_calls(txt, fn):
    """per launch of fn: dict(bdcfg, startio = all calls; win_bdcfg, win_startio = calls at/after snapshot A,
    i.e. the first executed __Runtime_startio).  No startio -> no window calls."""
    rows = _ordered_walk(txt, fn)
    f = next((i for i, r in enumerate(rows) if r[1] > 0 and STARTIO_API + "(" in r[0]), None)
    res = dict(bdcfg=0.0, startio=0.0, win_bdcfg=0.0, win_startio=0.0)
    for i, (code, mult, ids, trips) in enumerate(rows):
        for c in _CALL.findall(code):
            key = "bdcfg" if c in BDCFG_APIS else ("startio" if c == STARTIO_API else None)
            if key is None:
                continue
            res[key] += mult
            if f is None:
                continue
            if i >= f:
                res["win_" + key] += mult
            else:
                shared = 1
                for a, b, t in zip(ids, rows[f][2], trips):
                    if a != b:
                        break
                    shared *= t
                res["win_" + key] += mult - (mult / shared if shared else 0)
    return res


def off_corr_sites(build, coef):
    """[(kind, cid, corr_ms per launch-sum, info)] of the launches>0 sites of a UNet build in graph_driver order"""
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from aiegraph.costmodel.n_feat import base_sites
    from aiegraph.costmodel.static_feat import UNET_WL, host_counts, feat_of_counts, unet_site_feat
    host = os.path.join(build, UNET_WL, "host.cc")
    txt = open(host).read()
    seq = unet_site_feat(build)
    per = {}
    for d in seq:
        k = d["kernel"]
        if k not in per:
            w = inwin_calls(txt, "host_canonicalized_" + k)
            # cross-check against l_static's counter (static_feat.feat_of_counts, jit_mmio: no commit/window phases)
            ref = feat_of_counts(host_counts(host, "host_canonicalized_" + k, txt),
                                 os.path.join(build, UNET_WL, "host_ctrlpkt_%s.h" % k))
            if ref["kicks"] == 0 and (abs(ref["bdcfg_calls"] - w["bdcfg"]) > 1e-9
                                      or abs(ref["startio_calls"] - w["startio"]) > 1e-9):
                raise ValueError("kernel %s: walk calls %s vs l_static %s/%s"
                                 % (k, w, ref["bdcfg_calls"], ref["startio_calls"]))
            per[k] = w
    out, i = [], 0
    for s in base_sites(build, build + ".build.log"):
        lk = [d["kernel"] for d in seq[i:i + s["launches"]]]
        i += s["launches"]
        if s["launches"] <= 0:
            continue
        wb = sum(per[k]["win_bdcfg"] for k in lk)
        ws = sum(per[k]["win_startio"] for k in lk)
        out.append((s["kind"], s["cid"], wb * coef["bdcfg_calls"] + ws * coef["startio_calls"],
                    dict(win_bdcfg=wb, win_startio=ws, launches=s["launches"])))
    if i != len(seq):
        raise ValueError("%d site launches vs %d in the launch sequence" % (i, len(seq)))
    return out


def load_off_coef(path):
    import json
    c = json.load(open(path))["cfg"]["off"]["coef"]
    return {"bdcfg_calls": float(c["bdcfg_calls"]), "startio_calls": float(c["startio_calls"])}


def _ratio(o, g):
    return o / g if g > 0 else (0.0 if o <= 0 else float("inf"))


def check_log(cfg, run, txt, rows, aie_mhz, corr=None):
    """KOVL lines of one log; returns list of (ratio, viol, win_viol, raw_viol, kind, part, ovl, glue_eff).
    corr: [(kind, cid, corr_ms, info)] aligned with rows (off_corr_sites) or None (raw: viol = raw_viol)."""
    from k_diag import parse_perf
    pf, sp = parse_perf(txt), parse_ahs(txt)
    if corr is not None and len(corr) != len(rows):
        raise ValueError("off-corr: %d static sites vs %d log sites" % (len(corr), len(rows)))
    out = []
    for j, r in enumerate(rows):
        p = r["part"]
        if p not in pf:
            raise KeyError("part %d: no PROF aie_site_perf line" % p)
        if p not in sp:
            raise KeyError("part %d: no PROF aie_host_split line" % p)
        o, g, w = site_check(pf[p], sp[p], aie_mhz)
        ra, rw = _ratio(o, g), _ratio(o, w)
        line = ("KOVL cfg=%s run=%s part=%d kind=%s ovl_ms=%.4f glue_ms=%.4f ratio=%.3f win_ms=%.4f win_ratio=%.3f"
                % (cfg, run, p, r["kind"], o, g, ra, w, rw))
        raw = o > VIOL_TOL * g
        if corr is None:
            out.append((ra, raw, o > VIOL_TOL * w, raw, r["kind"], p, o, g))
        else:
            ck, cc, cms, _info = corr[j]
            if (ck, cc) != (r["kind"], r["cid"]):
                raise ValueError("off-corr site %s%d vs log site %s%d (part %d)" % (ck, cc, r["kind"], r["cid"], p))
            cm = cms * min(pf[p]["perf_n"], pf[p]["launches"]) / float(pf[p]["launches"])
            gc = g + cm
            rc = _ratio(o, gc)
            line += " raw_viol=%d corr_ms=%.4f glue_corr_ms=%.4f ratio_corr=%.3f" % (raw, cm, gc, rc)
            out.append((rc, o > VIOL_TOL * gc, o > VIOL_TOL * w, raw, r["kind"], p, o, gc))
        print(line)
    return out


def _src(cfg, lg, txt, corr_path):
    m = _CMP.findall(txt)
    board, n = m[-1] if m else ("-", "-1")
    print("KOVL SRC cfg=%s log=%s board=%s launches=%s off_corr=%s" % (cfg, lg, board, n, corr_path or "none"))


def run_all(build_root, specs, aie_mhz, rows_of, off_corr=None, corr_of=None):
    """off_corr: PARAMS path (cfg=off only); corr_of(build, coef) -> off_corr_sites list (default off_corr_sites)"""
    root = build_root.rstrip("/")
    ok, nlogs, allr = True, 0, []
    coef = load_off_coef(off_corr) if off_corr else None
    corr_of = corr_of or off_corr_sites
    for spec in specs:
        cfg, logs = spec.split("=", 1)
        use = coef is not None and cfg == "off"
        corr = None
        if use:
            try:
                corr = corr_of("%s/unet_%s" % (root, cfg), coef)
            except (ValueError, KeyError, OSError) as e:
                print("KOVL ERROR cfg=%s off-corr %s: %s" % (cfg, type(e).__name__, e))
                ok = False
                continue
        cr = []
        for i, lg in enumerate(logs.split(",")):
            run = "abcdefgh"[i] if i < 8 else str(i)
            try:
                rows, txt = rows_of("%s/unet_%s" % (root, cfg), lg)
                _src(cfg, lg, txt, off_corr if use else None)
                cr += check_log(cfg, run, txt, rows, aie_mhz, corr)
                nlogs += 1
            except (ValueError, KeyError, OSError) as e:
                print("KOVL ERROR cfg=%s run=%s log=%s %s: %s" % (cfg, run, lg, type(e).__name__, e))
                ok = False
        if cr:
            rs = [x[0] for x in cr]
            nv = sum(x[1] for x in cr)
            if use:
                still = {}
                for x in cr:
                    if x[1]:
                        d = still.setdefault((x[4], x[5]), [0, 0.0, 0.0, 0.0])
                        if x[0] > d[1]:
                            d[1:] = [x[0], x[6], x[7]]
                        d[0] += 1
                for (k, p), (n, rmax, o, gc) in sorted(still.items(), key=lambda t: -t[1][1]):
                    print("KOVL STILL cfg=%s kind=%s part=%d runs=%d ratio_corr_max=%.3f ovl_ms=%.4f glue_corr_ms=%.4f"
                          % (cfg, k, p, n, rmax, o, gc))
                byk = {}
                for k, _p in still:
                    byk[k] = byk.get(k, 0) + 1
                print("KOVL STILLSUM cfg=%s sites=%d by_kind=%s" % (cfg, len(still),
                      ",".join("%s:%d" % kv for kv in sorted(byk.items())) or "-"))
                print("KOVL SUM cfg=%s mode=corr runs=%d site_runs=%d viol=%d raw_viol=%d win_viol=%d ratio_med=%.3f "
                      "ratio_max=%.3f" % (cfg, len(logs.split(",")), len(cr), nv, sum(x[3] for x in cr),
                                          sum(x[2] for x in cr), statistics.median(rs), max(rs)))
            else:
                print("KOVL SUM cfg=%s runs=%d site_runs=%d viol=%d win_viol=%d ratio_med=%.3f ratio_max=%.3f"
                      % (cfg, len(logs.split(",")), len(cr), nv, sum(x[2] for x in cr),
                         statistics.median(rs), max(rs)))
                if cfg == "aot-kco" and nv > 0:
                    ok = False
        allr += cr
    med = statistics.median([x[0] for x in allr]) if allr else float("nan")
    print("KOVL ALL logs=%d site_runs=%d viol=%d ratio_med=%.3f" % (nlogs, len(allr), sum(x[1] for x in allr), med))
    return 0 if ok and allr else 1


# ---- --npred / --delta-tsv (costmodel M3.1.1.2c PV): static ovl_hat vs the counter ovl, K increment kdelta_ovl ----
NPRED_COLS = ("cfg", "run", "part", "site", "launches", "site_ms", "N_meas", "N_hat", "ovl_meas", "ovl_hat")
_CMP3 = re.compile(r"^CMP unet board=(\S+) .*?\baie_launches=(\d+) .*?\brun_rc=(-?\d+)", re.M)
OVLHAT_R2, OVLHAT_MAXREL = 0.9, 0.20


def load_npred(path):
    """{(cfg, run, part): row} of an n_fit --predict TSV (site totals in ms over the site's launches)"""
    lines = open(path).read().splitlines()
    hdr = lines[0].split("\t")
    if tuple(hdr[:len(NPRED_COLS)]) != NPRED_COLS:
        raise ValueError("npred header %s" % hdr)
    out = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(hdr, ln.split("\t")))
        out[(r["cfg"], r["run"], int(r["part"]))] = r
    return out


def _r2(h, m):
    mu = sum(m) / len(m)
    ss = sum((x - mu) ** 2 for x in m)
    return 1.0 - sum((a - b) ** 2 for a, b in zip(h, m)) / ss if ss > 0 else float("nan")


def run_npred(build_root, specs, aie_mhz, rows_of, npred, delta_tsv):
    """OVLSRC per log, OVLHAT per (cfg, run), KDELTA; writes delta_tsv (per launch ms).
    ovl_meas = counter ovl_cyc of the log (k_diag.parse_perf, per measured launch); ovl_hat = npred ovl_hat / launches.
    rc 0 all PASS, 1 a metric missed (TSV still complete), 2 unusable input (no TSV)."""
    from k_diag import parse_perf
    root = build_root.rstrip("/")
    import os
    if delta_tsv and os.path.exists(delta_tsv):
        os.unlink(delta_tsv)
    np_ = load_npred(npred)
    bad, fail, delta = False, False, []
    for spec in specs:
        cfg, logs = spec.split("=", 1)
        for i, lg in enumerate(logs.split(",")):
            run = "abcdefgh"[i] if i < 8 else str(i)
            try:
                rows, txt = rows_of("%s/unet_%s" % (root, cfg), lg)
                m = _CMP3.findall(txt)
                board, nl, rrc = m[-1] if m else ("-", "-1", "-1")
                print("OVLSRC cfg=%s log=%s board=%s launches=%s run_rc=%s sites=%d"
                      % (cfg, lg, board, nl, rrc, len(rows)))
                pf = parse_perf(txt)
                hs, ms, ht, mt, sh, xm = [], [], [], [], [], 0
                for r in rows:
                    p, n = r["part"], r["launches"]
                    q = np_.get((cfg, run, p))
                    if q is None:
                        raise KeyError("npred has no %s/%s part %d" % (cfg, run, p))
                    site = "%s%d" % (r["kind"], r["cid"])
                    if q["site"] != site or int(q["launches"]) != n:
                        raise ValueError("npred %s/%s part %d: %s x%s vs log %s x%d"
                                         % (cfg, run, p, q["site"], q["launches"], site, n))
                    pp = pf.get(p)
                    if pp is None or not pp.get("has_ovl"):
                        raise KeyError("part %d: no aie_site_perf ovl tail" % p)
                    om = pp["ovl_cyc"] / (aie_mhz * 1e3) / float(min(pp["perf_n"], n))   # counter, per launch
                    oh = float(q["ovl_hat"]) / n
                    sm = float(q["site_ms"]) / n
                    if abs(float(q["ovl_meas"]) / n - om) > 1e-4 * max(1.0, om) + 0.5e-4 / n:
                        xm += 1    # npred ovl_meas is not this counter (reported, the counter is used)
                    hs.append(oh); ms.append(om); ht.append(oh * n); mt.append(om * n)
                    sh.append(abs(oh - om) / sm)
                    delta.append((cfg, run, "%d" % p, site, "%.6f" % -(oh - om)))
                r2 = _r2(ht, mt)
                mr = max(abs(a - b) / b for a, b in zip(hs, ms))
                good = r2 >= OVLHAT_R2 and mr <= OVLHAT_MAXREL
                fail |= not good
                print("OVLHAT cfg=%s run=%s sites=%d ovl_src=counter r2=%.4f maxrel=%.4f sum_rel=%.4f share_max=%.4f "
                      "r2_launch=%.4f npred_meas_mismatch=%d %s"
                      % (cfg, run, len(rows), r2, mr, sum(ht) / sum(mt) - 1.0, max(sh), _r2(hs, ms), xm,
                         "PASS" if good else "FAIL"))
            except (ValueError, KeyError, OSError, ZeroDivisionError) as e:
                print("OVLHAT ERROR cfg=%s run=%s log=%s %s: %s" % (cfg, run, lg, type(e).__name__, e))
                bad = True
    if bad or not delta:
        return 2
    if delta_tsv:
        with open(delta_tsv, "w") as f:
            f.write("cfg\trun\tpart\tsite\tdelta_ms\n")
            for d in delta:
                f.write("\t".join(d) + "\n")
    print("KDELTA term=ovl rows=%d tsv=%s" % (len(delta), delta_tsv or "-"))
    return 1 if fail else 0


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
    # 5 ordered static walk: calls at/after the first startio (A of cfg=off)
    host = ("void host_canonicalized_k(int a) {\n  size_t v1 = 0;\n  size_t v2 = 4;\n  size_t v3 = 1;\n"
            "  XAie_DmaDesc v5 = __Runtime_dma_bd_config(v0, v4, 1); /* before A */\n"
            "  for (size_t v6 = v1; v6 < v2; v6 += v3) {\n"
            "    XAie_DmaDesc v7 = __Runtime_dma_bd_config(v0, v4, 2);\n"
            "    ioevent v8 = __Runtime_startio(v0, v7, 0, 1);\n  }\n"
            "  XAie_DmaDesc v9 = __Runtime_dma_bd_config_multidim(v0, v4, 3);\n"
            "  ioevent v10 = __Runtime_startio(v0, v9, 0, 1);\n  __Runtime_wait_io(v0, v10);\n}\n")
    w = inwin_calls(host, "host_canonicalized_k")
    chk("inwin_walk", w == dict(bdcfg=6.0, startio=5.0, win_bdcfg=4.0, win_startio=5.0), str(w))
    w0 = inwin_calls(host.replace("__Runtime_startio", "__Runtime_xstart"), "host_canonicalized_k")
    chk("inwin_no_startio", w0["win_bdcfg"] == 0 and w0["bdcfg"] == 6.0, str(w0))
    # 6 corrected off run: site 1 (ovl 1.5 / glue 1.0) + corr 0.5 -> 1.5 = not viol; site 3 corr 0; 2 stays at 1.02
    rows_c = [dict(part=p, kind="conv", cid=10 + p) for p in (1, 2, 3)]
    cmp_ln = "CMP unet board=palmyra213 data=x cos=1.0 aie_launches=258 coreen_calls=258 run_rc=0 PASS\n"
    txt_c = txt + cmp_ln
    txt_v = (perf(1, 1, 1, int(1.5e3 * mhz)) + ahs(1, 1.0, n=1) + cmp_ln)
    rows_v = [dict(part=1, kind="conv", cid=11)]
    corr = [("conv", 11, 0.5, {}), ("conv", 12, 0.0, {}), ("conv", 13, 0.0, {})]
    # load_off_coef reads a PARAMS file
    import json
    import os
    import tempfile
    fd, pj = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps({"cfg": {"off": {"coef": {"bdcfg_calls": 0.002, "startio_calls": 0.001}}}}).encode())
    os.close(fd)
    chk("load_coef", load_off_coef(pj) == {"bdcfg_calls": 0.002, "startio_calls": 0.001})
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = run_all("/x", ["off=L1,L2", "aot-kco=L3"], mhz,
                     lambda b, lg: (rows_c, txt_c) if b.endswith("unet_off") else (rows_c[2:], txt_c),
                     off_corr=pj, corr_of=lambda b, c: corr)
    o = buf.getvalue().splitlines()
    sums = [ln for ln in o if ln.startswith("KOVL SUM")]
    srcs = [ln for ln in o if ln.startswith("KOVL SRC")]
    chk("corr_sum", rc == 0 and sums[0] == "KOVL SUM cfg=off mode=corr runs=2 site_runs=6 viol=0 raw_viol=2 "
        "win_viol=0 ratio_med=1.000 ratio_max=1.020" and sums[1].startswith("KOVL SUM cfg=aot-kco runs=1 site_runs=1 "
        "viol=0 win_viol=0"), " | ".join(sums))
    chk("corr_src", srcs == ["KOVL SRC cfg=off log=L1 board=palmyra213 launches=258 off_corr=%s" % pj,
                             "KOVL SRC cfg=off log=L2 board=palmyra213 launches=258 off_corr=%s" % pj,
                             "KOVL SRC cfg=aot-kco log=L3 board=palmyra213 launches=258 off_corr=none"], str(srcs))
    chk("corr_line", any(ln.startswith("KOVL cfg=off run=a part=1 kind=conv ovl_ms=1.5000 glue_ms=1.0000 ratio=1.500 ")
                         and ln.endswith("raw_viol=1 corr_ms=0.5000 glue_corr_ms=1.5000 ratio_corr=1.000") for ln in o))
    # 7 a still-violating off site is listed; aot-kco viol > 0 -> rc=1
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc1 = run_all("/x", ["off=L1"], mhz, lambda b, lg: (rows_v, txt_v), off_corr=pj,
                      corr_of=lambda b, c: [("conv", 11, 0.1, {})])
        rc2 = run_all("/x", ["aot-kco=L1"], mhz, lambda b, lg: (rows_v, txt_v))
    o = buf.getvalue()
    chk("still_listed", rc1 == 0 and "KOVL STILL cfg=off kind=conv part=1 runs=1 ratio_corr_max=1.364" in o
        and "KOVL STILLSUM cfg=off sites=1 by_kind=conv:1" in o and "mode=corr runs=1 site_runs=1 viol=1 raw_viol=1" in o)
    chk("aot_viol_rc1", rc2 == 1)
    # 8 misaligned static site -> rc=1
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc3 = run_all("/x", ["off=L1"], mhz, lambda b, lg: (rows_v, txt_v), off_corr=pj,
                      corr_of=lambda b, c: [("linear", 11, 0.1, {})])
    chk("corr_misaligned_rc1", rc3 == 1 and "KOVL ERROR" in buf.getvalue())
    os.unlink(pj)
    # 9 --npred: per-launch counter ovl vs npred ovl_hat/launches, OVLSRC from CMP, delta = -(hat - meas)
    rows_n = [dict(part=1, kind="conv", cid=0, launches=1), dict(part=2, kind="fa", cid=5, launches=4)]
    txt_n = (perf(1, 1, 1, int(2.0e3 * mhz)) + perf(2, 4, 2, int(2.0e3 * mhz))
             + cmp_ln.replace("PASS", "aie_flop_frac=1.000 PASS"))
    fd, npj = tempfile.mkstemp(suffix=".tsv")
    os.write(fd, ("\t".join(NPRED_COLS) + "\n" + "c\ta\t1\tconv0\t1\t10.0\t5\t5\t2.0000\t1.5000\n"
                  "c\ta\t2\tfa5\t4\t20.0\t5\t5\t4.0000\t4.4000\n").encode())
    os.close(fd)
    fd, dtj = tempfile.mkstemp(suffix=".tsv")
    os.close(fd)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rcn = run_npred("/x", ["c=L1"], mhz, lambda b, lg: (rows_n, txt_n), npj, dtj)
    o = buf.getvalue().splitlines()
    dt = open(dtj).read().splitlines()
    chk("npred_ovlhat", rcn == 1 and o[0] == "OVLSRC cfg=c log=L1 board=palmyra213 launches=258 run_rc=0 sites=2"
        and o[1].startswith("OVLHAT cfg=c run=a sites=2 ovl_src=counter r2=") and "maxrel=0.2500 " in o[1]
        and "sum_rel=-0.0167 share_max=0.0500 " in o[1] and "npred_meas_mismatch=0 FAIL" in o[1]
        and o[-1] == "KDELTA term=ovl rows=2 tsv=%s" % dtj
        and dt == ["cfg\trun\tpart\tsite\tdelta_ms", "c\ta\t1\tconv0\t0.500000", "c\ta\t2\tfa5\t-0.100000"],
        " | ".join(o + dt))
    with redirect_stdout(io.StringIO()):
        rce = run_npred("/x", ["c=L1"], mhz, lambda b, lg: (rows_n + [dict(part=3, kind="fa", cid=6, launches=1)],
                                                             txt_n), npj, dtj)
    chk("npred_missing_rc2", rce == 2 and not os.path.exists(dtj))
    os.unlink(npj)
    ok = all(res)
    print("KOVL SELFTEST %s (%d/%d)" % ("PASS" if ok else "FAIL", sum(1 for r in res if r), len(res)))
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
    ap.add_argument("--off-corr", default=None, metavar="PARAMS.json",
                    help="l_palmyra213.json: static in-window rt bdcfg/startio correction of the cfg=off glue bound")
    ap.add_argument("--npred", default=None, metavar="NPRED.tsv",
                    help="n_fit --predict TSV: OVLSRC/OVLHAT (static ovl_hat vs counter ovl) + KDELTA instead of KOVL")
    ap.add_argument("--delta-tsv", default=None, metavar="TSV", help="--npred: per-launch K increment -(ovl_hat-ovl)")
    ap.add_argument("runs", nargs="+", metavar="CFG=LOG[,LOG...]")
    a = ap.parse_args(argv)
    from k_diag import AIE_MHZ
    from m3_site import site_rows
    if a.npred:
        return run_npred(a.build_root, a.runs, AIE_MHZ, site_rows, a.npred, a.delta_tsv)
    return run_all(a.build_root, a.runs, AIE_MHZ, site_rows, off_corr=a.off_corr)


if __name__ == "__main__":
    sys.exit(main())
