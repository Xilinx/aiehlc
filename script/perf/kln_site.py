#!/usr/bin/env python3
"""Per-launch K + L + N synthesis of UNet AIE sites (costmodel M3.1.1.2 PC; first cut of M6 estimate()).

  kln_site.py --selftest
  kln_site.py --audit
  kln_site.py --oracle-k --site-src log|lpred [--r2 0.90] [--max-rel 0.20] ARGS
  kln_site.py --insample --khat KHAT.tsv --lpred LPRED.tsv --npred NPRED.tsv [--khat-src SRC] [--fit-run a]
      (KLNPROBE, probe only: rc=0 when it ran, 1 if a cfg has no usable site, 2 if KHAT is missing or empty;
       without --fit/--check the rows are the --fit-run rows of the LPRED/NPRED join, site_ms = LPRED;
       KHAT = k_diag --khat-tsv (8 cols, K_hat_ms per launch, no run column = run a); SRC is not in the
       TSV, default koracle_dma7 = the PKP k_diag --dmafeat run)
  ARGS = --lpred LPRED.tsv --npred NPRED.tsv --build-root DIR --fit CFG=LOG ... --check CFG=LOG[,LOG] ...

Inputs: LPRED = m3_site --lpred TSV (cfg run part site kind launches site_ms L_meas L_hat; run = 'unet_off.a.log'),
NPRED = n_fit --predict TSV (cfg run part site launches site_ms N_meas N_hat ovl_meas ovl_hat; run = 'a').
Run names are unified to a/b/c (norm_run) before the join; the key is (cfg, run, part).
Truth = the board log: site_ms of 'PROF part=<i> op=aie_host_* ... site_ms=' (n_fit._PROF), launches from
'PROF aie_host_split ... launches=' (k_diag.parse_split), sites with launches > 0 only (138 per log).
  K_meas = site_ms - L_meas - N_meas          (oracle K; the measured remainder)
  T      = site_ms / launches                 (ms per launch)
  T_hat  = (K + L_hat + N_hat) / launches     (predict_launch: reads only K, L_hat, N_hat, launches)
Per log (138 sites): r2 = 1 - SS_res / SS_tot of T_hat vs T, maxrel = max |T_hat - T| / T,
sum_rel = sum(T_hat * launches - site_ms) / sum(site_ms).  ALL = the --check logs only (b/c, 4 logs, 552 sites);
PASS iff every check log has r2 >= --r2 and maxrel <= --max-rel.
--site-src lpred uses LPRED's site_ms instead of the log's (reproduces the lead dry run dryrun.md).
Output:
  KLNSRC cfg= log= board= launches= run_rc= sites=        (one per log, from the 'CMP unet' line)
  KLNJOIN logs= keys= lpred= npred= lpred_vs_log_max= npred_vs_log_max= ok=   (*_vs_log_max = max |site_ms - log|,
          info only; ok=1 iff every log key is found in both TSVs and no TSV key is left over)
  KLNORACLE cfg= run= k=meas r2= maxrel= sum_rel= PASS|FAIL [fit]      (fit = a --fit log, not in ALL)
  KLNORACLE ALL k=meas logs= sites= r2_min= maxrel= PASS|FAIL
  KLN121 cfg= part= K_meas_ms= lock_ms= stream_ms= launches= runs=b,c   (conv121, mean over --check logs; lock /
          stream = k_diag.split_site of 'PROF aie_site_perf', report only)
  KTOLNEED cfg= sites= tol_lt015= tol_015_030= tol_030_050= min_tol= at=SITE:RUN   (all logs of the cfg;
          tol = (max_rel - |L_hat - L_meas + N_hat - N_meas| / site_ms) / (K_meas / site_ms) = the K relative error
          a K model may have on that site)
--oracle-k --dump-kmeas PATH: also writes the per-launch K of every joined row (fit + check logs) as TSV
  cfg run part site cls launches K_meas_ms lock_ms stream_ms   (cls = LPRED kind; all *_ms per launch:
  K_meas_ms = K_meas / launches, lock/stream = k_diag.split_site counter sums / perf_n; nan without a perf line)
--k-tsv PATH: external K (TSV cfg run part site K_hat_ms [K_loo_ms] [dw_hat_ms], *_ms per launch); K = K_hat_ms x
  launches goes into predict_launch.  --kswap D1[,D2..]: diag, K = max(0, K_meas + launches x sum delta_ms) with
  delta TSVs cfg run part site delta_ms (missing key = 0).  Both print (--k-label L, default hat|swap):
  KLNHAT cfg= run= k= sites= r2= maxrel= sum_rel= PASS|FAIL[ fit]       (fit = run a, not in ALL)
  KLNHAT LOO cfg= run=a k= sites= loo_maxrel= PASS|FAIL                 (--k-tsv: K_loo_ms of the run-a rows,
          nan when the column is missing or all nan; --kswap: the swapped K itself, nothing is fitted)
  KLNHAT121 cfg= k= part= runs=b,c K_hat_ms= K_meas_ms= lock_ms= dw_hat_ms= rel_T=   (conv121 mean over b/c,
          per launch; rel_T = |mean (T_hat - T) / T| (magnitude, like maxrel; sign = K_hat_ms vs K_meas_ms); dw_hat_ms from the K TSV, nan without the column)
  KLNHAT ALL k= logs= sites= r2_min= maxrel= loo_max= PASS|FAIL  (b/c only; loo_max = max of the LOO lines, with
          --loo X PASS also needs loo_max <= X, nan = FAIL); rc=1 when ALL FAILs.  Uncomputable metrics print nan.
--audit: AST check of the predict path (PREDICT_FNS): no measured field name may appear in it (MEAS_FIELDS);
  prints 'KLN AUDIT fn=.. names=.. meas=..' per function and 'KLN AUDIT meas_at_predict=<n>' last (rc=1 if n>0).
"""
import argparse
import ast
import csv
import inspect
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

KLN_SITE = "conv121"
MEAS_FIELDS = ("K_meas", "L_meas", "N_meas", "ovl_meas", "compute_cyc", "lock", "stream", "wait", "site_ms")
PRED_KEYS = ("L_hat", "N_hat", "launches")   # the only TSV fields handed to the predict path
TOL_BINS = (0.15, 0.30, 0.50)
_RUN = re.compile(r"^(?:.*/)?(?:unet_[\w-]+\.)?([a-z])(?:\.log)?$")


def _prof_re():
    from n_fit import _PROF
    return _PROF


def _cmp_re():
    from n_fit import _CMP
    return _CMP


def norm_run(name):
    """'unet_off.a.log' | '/x/unet_aot-kco.b.log' | 'a' -> 'a'"""
    m = _RUN.match(str(name).strip())
    if not m:
        raise ValueError("cannot unify run name %r" % name)
    return m.group(1)


# ------------------------------------------------------------------ predict path (audited: no measured fields)
def pred_view(row):
    """the predict-path view of a joined TSV row: only PRED_KEYS"""
    return {k: row[k] for k in PRED_KEYS}


def predict_launch(k_ms, pv):
    """T_hat (ms per launch) of one site: K (given) + frozen L_hat + N_hat, over launches"""
    return (k_ms + pv["L_hat"] + pv["N_hat"]) / pv["launches"]


def k_tsv_launch(k_ms_launch, pv):
    """--k-tsv: the external K is per launch; predict_launch takes the site total"""
    return k_ms_launch * pv["launches"]


def k_swap(K_meas, deltas_ms, pv):
    """--kswap (diag, reads the measured K): K = max(0, K_meas + launches x sum of per-launch deltas)"""
    return max(0.0, K_meas + pv["launches"] * sum(deltas_ms))


PREDICT_FNS = (pred_view, predict_launch, k_tsv_launch)
DIAG_FNS = (k_swap,)   # audited and printed, flagged diag, not counted in meas_at_predict


def audit():
    bad = 0
    for fn in PREDICT_FNS + DIAG_FNS:
        tree = ast.parse(inspect.getsource(fn))
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                names.add(n.value)
        if fn is pred_view:
            names.update(PRED_KEYS)
        meas = sorted(x for x in names if any(f.lower() in x.lower() for f in MEAS_FIELDS))
        diag = fn in DIAG_FNS
        if not diag:
            bad += len(meas)
        print("KLN AUDIT fn=%s names=%s meas=%s%s" % (fn.__name__, ",".join(sorted(names)), ",".join(meas) or "-",
                                                     " diag" if diag else ""))
    print("KLN AUDIT meas_at_predict=%d" % bad)
    return bad == 0


# ------------------------------------------------------------------ inputs
def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tsv_map(rows, cols):
    """{(cfg, run, part): {col: float}} with run unified; raises on duplicates"""
    out = {}
    for r in rows:
        k = (r["cfg"], norm_run(r["run"]), int(r["part"]))
        if k in out:
            raise ValueError("duplicate key %r" % (k,))
        d = {c: float(r[c]) for c in cols}
        d["site"] = r["site"]
        if "kind" in r:
            d["kind"] = r["kind"]
        out[k] = d
    return out


def log_sites(txt):
    """{part: (site, launches, site_ms)} of the sites with launches > 0"""
    from k_diag import parse_split
    sp = parse_split(txt)
    out = {}
    for part, op, cid, _ms, site_ms, _aie in _prof_re().findall(txt):
        part = int(part)
        n = sp.get(part, (0, 0, 0))[2]
        if n > 0:
            out[part] = ("%s%s" % ("fa" if op == "attn" else op, cid), n, float(site_ms))
    return out


def src_line(cfg, log, txt, nsites):
    m = _cmp_re().findall(txt)
    board, launches, rc = m[-1] if m else ("-", "-1", "-1")
    print("KLNSRC cfg=%s log=%s board=%s launches=%s run_rc=%s sites=%d" % (cfg, log, board, launches, rc, nsites))
    return rc == "0" and int(launches) > 0


def parse_logs(specs):
    """['off=A,B', ...] -> [(cfg, path)]"""
    out = []
    for s in specs or []:
        cfg, _, paths = s.partition("=")
        out += [(cfg, p) for p in paths.split(",") if p]
    return out


def join(lp, np_, logs):
    """rows of (cfg, run, part) with site_ms from the log, launches from the log; KLNJOIN line"""
    rows, keys, dl, dn, ok_src = [], set(), 0.0, 0.0, True
    for cfg, path, _fit in logs:
        txt = Path(path).read_text(errors="replace")
        ls = log_sites(txt)
        ok_src &= src_line(cfg, path, txt, len(ls))
        run = norm_run(Path(path).name)
        for part, (site, n, site_ms) in sorted(ls.items()):
            k = (cfg, run, part)
            keys.add(k)
            lr, nr = lp.get(k), np_.get(k)
            if lr is None or nr is None:
                continue
            if lr["site"] != site or nr["site"] != site or int(lr["launches"]) != n or int(nr["launches"]) != n:
                raise ValueError("misaligned %r: log %s/%d lpred %s/%d npred %s/%d"
                                 % (k, site, n, lr["site"], lr["launches"], nr["site"], nr["launches"]))
            dl, dn = max(dl, abs(lr["site_ms"] - site_ms)), max(dn, abs(nr["site_ms"] - site_ms))
            rows.append(dict(cfg=cfg, run=run, part=part, site=site, cls=lr.get("kind", "-"), launches=n,
                             site_ms_log=site_ms,
                             site_ms_lpred=lr["site_ms"], L_meas=lr["L_meas"], L_hat=lr["L_hat"],
                             N_meas=nr["N_meas"], N_hat=nr["N_hat"], fit=_fit, log=path))
    nl = sum(k in lp for k in keys)
    nn = sum(k in np_ for k in keys)
    runs = {(c, norm_run(Path(p).name)) for c, p, _ in logs}
    extra = sum((k[0], k[1]) in runs and k not in keys for k in list(lp) + list(np_))
    ok = ok_src and len(keys) > 0 and nl == nn == len(keys) == len(rows) and extra == 0
    print("KLNJOIN logs=%d keys=%d lpred=%d npred=%d lpred_vs_log_max=%.4f npred_vs_log_max=%.4f ok=%d"
          % (len(logs), len(keys), nl, nn, dl, dn, ok))
    return rows, ok


# ------------------------------------------------------------------ metrics
def metrics(pairs):
    """pairs = [(T, T_hat, launches, site)] -> r2, maxrel, worst, sum_rel"""
    t = [p[0] for p in pairs]
    mu = sum(t) / len(t)
    ss_tot = sum((x - mu) ** 2 for x in t)
    ss_res = sum((p[1] - p[0]) ** 2 for p in pairs)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel = [abs(p[1] - p[0]) / p[0] for p in pairs]
    i = max(range(len(rel)), key=rel.__getitem__)
    sum_rel = sum((p[1] - p[0]) * p[2] for p in pairs) / sum(p[0] * p[2] for p in pairs)
    return r2, rel[i], pairs[i][3], sum_rel


def k_meas(r, src):
    return r["site_ms_" + src] - r["L_meas"] - r["N_meas"]


def oracle(rows, src, r2_gate, rel_gate):
    groups = {}
    for r in rows:
        groups.setdefault((r["cfg"], r["run"]), []).append(r)
    ok_all, r2_min, rel_max, nlog, nsite = True, float("inf"), 0.0, 0, 0
    for (cfg, run), rs in groups.items():
        pairs = [(r["site_ms_" + src] / r["launches"], predict_launch(k_meas(r, src), pred_view(r)), r["launches"],
                  r["site"]) for r in rs]
        r2, mr, worst, sr = metrics(pairs)
        ok = r2 >= r2_gate and mr <= rel_gate
        fit = rs[0]["fit"]
        print("KLNORACLE cfg=%s run=%s k=meas r2=%.4f maxrel=%.4f sum_rel=%.4f %s%s"
              % (cfg, run, r2, mr, sr, "PASS" if ok else "FAIL", " fit" if fit else ""))
        print("KLNWORST cfg=%s run=%s n=%d site=%s" % (cfg, run, len(rs), worst))
        if not fit:
            ok_all &= ok
            r2_min, rel_max = min(r2_min, r2), max(rel_max, mr)
            nlog, nsite = nlog + 1, nsite + len(rs)
    ok_all &= nlog > 0
    print("KLNORACLE ALL k=meas logs=%d sites=%d r2_min=%.4f maxrel=%.4f %s"
          % (nlog, nsite, r2_min, rel_max, "PASS" if ok_all else "FAIL"))
    return ok_all


def kln121(rows, src, site=KLN_SITE):
    from k_diag import parse_perf, parse_split, split_site
    for cfg in dict.fromkeys(r["cfg"] for r in rows):
        rs = [r for r in rows if r["cfg"] == cfg and r["site"] == site and not r["fit"]]
        if not rs:
            continue
        km, lk, st = [], [], []
        for r in rs:
            txt = Path(r["log"]).read_text(errors="replace")
            w, i, _n = parse_split(txt)[r["part"]]
            d = split_site(w, i, parse_perf(txt).get(r["part"]))
            km.append(k_meas(r, src))
            lk.append(d.get("lock", float("nan")))
            st.append(d.get("stream", float("nan")))
        print("KLN121 cfg=%s part=%d K_meas_ms=%.4f lock_ms=%.4f stream_ms=%.4f launches=%d runs=%s"
              % (cfg, rs[0]["part"], sum(km) / len(km), sum(lk) / len(lk), sum(st) / len(st), rs[0]["launches"],
                 ",".join(r["run"] for r in rs)))


def _f(x):
    return "%.4f" % x   # nan -> 'nan'


def lock_stream(rows):
    """{(cfg, run, part): (lock_ms, stream_ms) per launch} from 'PROF aie_site_perf' (sums / perf_n)"""
    from k_diag import parse_perf, parse_split, split_site
    out, cache = {}, {}
    for r in rows:
        if r["log"] not in cache:
            txt = Path(r["log"]).read_text(errors="replace")
            cache[r["log"]] = (parse_split(txt), parse_perf(txt))
        sp, pf = cache[r["log"]]
        perf = pf.get(r["part"])
        w, i, _n = sp[r["part"]]
        d = split_site(w, i, perf)
        pn = max(1, perf["perf_n"]) if perf else 1
        out[(r["cfg"], r["run"], r["part"])] = (d.get("lock", float("nan")) / pn, d.get("stream", float("nan")) / pn)
    return out


KMEAS_COLS = ("cfg", "run", "part", "site", "cls", "launches", "K_meas_ms", "lock_ms", "stream_ms")


def dump_kmeas(rows, src, path):
    """--dump-kmeas: per-launch K_meas / lock / stream of every joined row (repr floats, no rounding)"""
    ls = lock_stream(rows)
    with open(path, "w") as f:
        f.write("\t".join(KMEAS_COLS) + "\n")
        for r in rows:
            lk, st = ls[(r["cfg"], r["run"], r["part"])]
            n = r["launches"]
            f.write("\t".join([r["cfg"], r["run"], str(r["part"]), r["site"], r.get("cls", "-"), str(n),
                               repr(k_meas(r, src) / n), repr(lk), repr(st)]) + "\n")
    return len(rows)


def _fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_ktsv(path):
    """--k-tsv: {(cfg, run, part): dict(site, K_hat_ms, K_loo_ms, dw_hat_ms)} (per launch; absent column = nan)"""
    out = {}
    for r in read_tsv(path):
        k = (r["cfg"], norm_run(r["run"]), int(r["part"]))
        if k in out:
            raise ValueError("duplicate key %r in %s" % (k, path))
        out[k] = dict(site=r["site"], K_hat_ms=_fl(r.get("K_hat_ms")), K_loo_ms=_fl(r.get("K_loo_ms")),
                      dw_hat_ms=_fl(r.get("dw_hat_ms")))
    return out


def read_kswap(paths):
    """--kswap D1[,D2..]: {(cfg, run, part): [delta_ms per launch, one per file]}; site checked across files"""
    out, sites = {}, {}
    for p in [x for x in paths.split(",") if x]:
        for r in read_tsv(p):
            k = (r["cfg"], norm_run(r["run"]), int(r["part"]))
            if sites.setdefault(k, r["site"]) != r["site"]:
                raise ValueError("site mismatch %r in %s" % (k, p))
            out.setdefault(k, []).append(float(r["delta_ms"]))
    return out, sites


def khat_eval(rows, src, kmap, label, r2_gate, rel_gate, loo_gate, ls=None):
    """KLNHAT / LOO / 121 / ALL; kmap[(cfg, run, part)] = dict(K=site-total K or None, K_loo=site total or nan,
    dw=per-launch or nan).  True iff ALL PASS."""
    nan = float("nan")
    groups = {}
    for r in rows:
        groups.setdefault((r["cfg"], r["run"]), []).append(r)
    ok_all, r2s, rels, loos, nlog, nsite = True, [], [], [], 0, 0
    for (cfg, run), rs in groups.items():
        fit = rs[0]["fit"]
        pairs, miss = [], 0
        for r in rs:
            km = kmap.get((cfg, run, r["part"]))
            if km is None or km["K"] is None or math.isnan(km["K"]):
                miss += 1
                continue
            pairs.append((r["site_ms_" + src] / r["launches"], predict_launch(km["K"], pred_view(r)),
                          r["launches"], r["site"]))
        if miss or len(pairs) < 2:
            r2, mr, sr = nan, nan, nan
        else:
            r2, mr, _w, sr = metrics(pairs)
        ok = (not miss) and r2 >= r2_gate and mr <= rel_gate
        print("KLNHAT cfg=%s run=%s k=%s sites=%d r2=%s maxrel=%s sum_rel=%s %s%s"
              % (cfg, run, label, len(rs), _f(r2), _f(mr), _f(sr), "PASS" if ok else "FAIL", " fit" if fit else ""))
        if run == "a":
            lp = []
            for r in rs:
                km = kmap.get((cfg, run, r["part"]))
                kl = km["K_loo"] if km else nan
                if math.isnan(kl):
                    lp = None
                    break
                lp.append((r["site_ms_" + src] / r["launches"], predict_launch(kl, pred_view(r)), r["launches"],
                           r["site"]))
            lmr = metrics(lp)[1] if lp and len(lp) >= 2 else nan
            lok = not math.isnan(lmr) and (loo_gate is None or lmr <= loo_gate)
            print("KLNHAT LOO cfg=%s run=a k=%s sites=%d loo_maxrel=%s %s"
                  % (cfg, label, len(rs), _f(lmr), "PASS" if lok else "FAIL"))
            loos.append(lmr)
        if not fit:
            ok_all &= ok
            r2s.append(r2)
            rels.append(mr)
            nlog, nsite = nlog + 1, nsite + len(rs)
    for cfg in dict.fromkeys(r["cfg"] for r in rows):
        rs = [r for r in rows if r["cfg"] == cfg and r["site"] == KLN_SITE and not r["fit"]]
        if not rs:
            continue
        kh, kmv, lk, dw, rt = [], [], [], [], []
        for r in rs:
            key = (cfg, r["run"], r["part"])
            km = kmap.get(key) or dict(K=nan, dw=nan)
            k = nan if km["K"] is None else km["K"]
            n = r["launches"]
            kh.append(k / n)
            kmv.append(k_meas(r, src) / n)
            lk.append(ls[key][0] if ls and key in ls else nan)
            dw.append(km.get("dw", nan))
            t = r["site_ms_" + src] / n
            rt.append((predict_launch(k, pred_view(r)) - t) / t)
        m = lambda v: sum(v) / len(v)
        print("KLNHAT121 cfg=%s k=%s part=%d runs=%s K_hat_ms=%s K_meas_ms=%s lock_ms=%s dw_hat_ms=%s rel_T=%s"
              % (cfg, label, rs[0]["part"], ",".join(r["run"] for r in rs), _f(m(kh)), _f(m(kmv)), _f(m(lk)),
                 _f(m(dw)), _f(abs(m(rt)))))
    any_nan = lambda v: (not v) or any(math.isnan(x) for x in v)
    loo_max = nan if any_nan(loos) else max(loos)
    r2_min = nan if any_nan(r2s) else min(r2s)
    rel_max = nan if any_nan(rels) else max(rels)
    ok_all &= nlog > 0 and not math.isnan(r2_min) and not math.isnan(rel_max)
    if loo_gate is not None:
        ok_all &= not math.isnan(loo_max) and loo_max <= loo_gate
    print("KLNHAT ALL k=%s logs=%d sites=%d r2_min=%s maxrel=%s loo_max=%s %s"
          % (label, nlog, nsite, _f(r2_min), _f(rel_max), _f(loo_max), "PASS" if ok_all else "FAIL"))
    return ok_all


def kmap_tsv(rows, kt):
    """--k-tsv rows -> kmap (K x launches through k_tsv_launch); site mismatch raises"""
    out = {}
    for r in rows:
        key = (r["cfg"], r["run"], r["part"])
        e = kt.get(key)
        if e is None:
            continue
        if e["site"] != r["site"]:
            raise ValueError("k-tsv site mismatch %r: %s vs log %s" % (key, e["site"], r["site"]))
        pv = pred_view(r)
        out[key] = dict(K=k_tsv_launch(e["K_hat_ms"], pv), K_loo=k_tsv_launch(e["K_loo_ms"], pv),
                        dw=e["dw_hat_ms"])
    return out


def kmap_swap(rows, src, deltas, sites):
    """--kswap (diag): K = max(0, K_meas + launches x sum delta); LOO = the same K on run a (nothing fitted)"""
    out = {}
    for r in rows:
        key = (r["cfg"], r["run"], r["part"])
        if key in sites and sites[key] != r["site"]:
            raise ValueError("kswap site mismatch %r: %s vs log %s" % (key, sites[key], r["site"]))
        k = k_swap(k_meas(r, src), deltas.get(key, []), pred_view(r))
        out[key] = dict(K=k, K_loo=k, dw=float("nan"))
    return out


def ktolneed(rows, src, rel_gate):
    for cfg in dict.fromkeys(r["cfg"] for r in rows):
        rs = [r for r in rows if r["cfg"] == cfg]
        cnt, best = [0, 0, 0], (float("inf"), "-")
        for r in rs:
            s = r["site_ms_" + src]
            k = k_meas(r, src)
            e = abs(r["L_hat"] - r["L_meas"] + r["N_hat"] - r["N_meas"]) / s
            tol = (rel_gate - e) / (k / s) if k > 0 else float("inf")
            for j, hi in enumerate(TOL_BINS):
                if tol < hi and (j == 0 or tol >= TOL_BINS[j - 1]):
                    cnt[j] += 1
            if tol < best[0]:
                best = (tol, "%s:%s" % (r["site"], r["run"]))
        print("KTOLNEED cfg=%s sites=%d tol_lt015=%d tol_015_030=%d tol_030_050=%d min_tol=%.4f at=%s"
              % (cfg, len(rs), cnt[0], cnt[1], cnt[2], best[0], best[1]))


def read_khat(path, run="a"):
    """k_diag --khat-tsv file ('# cfg part site cls launches K_hat_ms compute_src ovl_src', space separated,
    K_hat_ms per launch, fitted on run a) -> {(cfg, run, part): (site, launches, K_hat_ms per launch)};
    None if missing or empty"""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    out = {}
    for ln in p.read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split()
        out[(f[0], run, int(f[1]))] = (f[2], int(f[4]), float(f[5]))
    return out or None


def tsv_rows(lp, np_, run):
    """in-sample rows from the LPRED/NPRED join only (no log): keys of run in both TSVs, site_ms = LPRED"""
    rows = []
    for k in sorted(k for k in lp if k[1] == run and k in np_):
        lr, nr = lp[k], np_[k]
        if lr["site"] != nr["site"] or lr["launches"] != nr["launches"]:
            raise ValueError("misaligned %r: lpred %s/%d npred %s/%d" % (k, lr["site"], lr["launches"],
                                                                          nr["site"], nr["launches"]))
        if lr["launches"] <= 0:
            continue
        rows.append(dict(cfg=k[0], run=run, part=k[2], site=lr["site"], launches=int(lr["launches"]),
                         site_ms_lpred=lr["site_ms"], L_meas=lr["L_meas"], L_hat=lr["L_hat"],
                         N_meas=nr["N_meas"], N_hat=nr["N_hat"], fit=True, log=None))
    return rows


def insample(rows, khat, src, khat_src):
    """KLNPROBE (probe, no gate): T_hat = K_hat (per launch, khat) + (L_hat + N_hat) / launches vs T; sites with
    a nan K_hat or a site/launches mismatch are counted, not used.  True iff every cfg has >= 2 usable sites."""
    ok_all = True
    for cfg in dict.fromkeys(r["cfg"] for r in rows):
        pairs, nan, mis = [], 0, 0
        for r in (r for r in rows if r["cfg"] == cfg):
            kh = khat.get((r["cfg"], r["run"], r["part"]))
            if kh is None:
                continue
            if kh[0] != r["site"] or kh[1] != r["launches"]:
                mis += 1
                continue
            if math.isnan(kh[2]):
                nan += 1
                continue
            pairs.append((r["site_ms_" + src] / r["launches"],
                          predict_launch(kh[2] * r["launches"], pred_view(r)), r["launches"], r["site"]))
        if len(pairs) < 2:
            print("KLNPROBE cfg=%s src=%s sites=%d nan=%d mismatch=%d FAIL" % (cfg, khat_src, len(pairs), nan, mis))
            ok_all = False
            continue
        r2, mr, worst, sr = metrics(pairs)
        print("KLNPROBE cfg=%s src=%s run=%s site_src=%s sites=%d nan=%d mismatch=%d k=hat r2=%.4f maxrel=%.4f "
              "sum_rel=%.4f worst=%s probe" % (cfg, khat_src, rows[0]["run"], src, len(pairs), nan, mis, r2, mr, sr,
                                               worst))
    return ok_all


# ------------------------------------------------------------------ selftest
def selftest():
    import io
    import contextlib
    res = []

    def chk(name, c):
        res.append((name, bool(c)))

    chk("norm_lpred", norm_run("unet_off.a.log") == "a")
    chk("norm_npred", norm_run("b") == "b")
    chk("norm_path", norm_run("/x/2026-11-20_M3121/unet_aot-kco.c.log") == "c")
    try:
        norm_run("unet_off.log")
        chk("norm_bad_raises", False)
    except ValueError:
        chk("norm_bad_raises", True)
    keys = tsv_map([dict(cfg="off", run="unet_off.a.log", part="14", site="conv0", site_ms="1", L_meas="1")],
                   ("site_ms",)).keys() & tsv_map([dict(cfg="off", run="a", part="14", site="conv0", site_ms="1")],
                                                  ("site_ms",)).keys()
    chk("join_unified", len(keys) == 1)
    pv = pred_view(dict(L_hat=1.0, N_hat=2.0, launches=2, L_meas=9.0, N_meas=9.0, site_ms=99.0))
    chk("pred_view_keys", sorted(pv) == sorted(PRED_KEYS))
    chk("predict_launch", abs(predict_launch(3.0, pv) - 3.0) < 1e-12)
    r = dict(site_ms_log=10.0, L_meas=2.0, N_meas=3.0)
    chk("k_meas", abs(k_meas(r, "log") - 5.0) < 1e-12)
    r2, mr, w, sr = metrics([(1.0, 1.1, 1, "x"), (2.0, 2.0, 2, "y"), (3.0, 3.0, 1, "z")])
    chk("metrics", abs(mr - 0.1) < 1e-12 and w == "x" and abs(sr - 0.1 / 8) < 1e-12 and abs(r2 - 0.995) < 1e-12)
    txt = ("PROF part=14 op=aie_host_conv0 ms=3.331 bf16_expand_ms=0.001 site_ms=16.183 aie_ms=12.853\n"
           "PROF part=31 op=aie_host_attn4 ms=0.430 bf16_expand_ms=0.000 site_ms=0.430 aie_ms=0.000\n"
           "PROF part=36 op=aie_host_attn5 ms=1.0 bf16_expand_ms=0.000 site_ms=26.1 aie_ms=20.0\n"
           "PROF aie_host_split part=14 kind=conv pack_ms=1.843 scatter_ms=1.466 dma_ms=6.033 wait_ms=6.820 "
           "glue_ms=2.718 flush_ms=0.460 inval_ms=0.054 launches=1\n"
           "PROF aie_host_split part=31 kind=fa pack_ms=0.427 scatter_ms=0.000 dma_ms=0.000 wait_ms=0.000 "
           "glue_ms=0.000 flush_ms=0.000 inval_ms=0.000 launches=0\n"
           "PROF aie_host_split part=36 kind=fa pack_ms=0.0 scatter_ms=0.3 dma_ms=24.7 wait_ms=0.6 "
           "glue_ms=9.7 flush_ms=0.1 inval_ms=0.02 launches=16\n")
    ls = log_sites(txt)
    chk("log_sites", ls == {14: ("conv0", 1, 16.183), 36: ("fa5", 16, 26.1)})
    pv4 = pred_view(dict(L_hat=1.0, N_hat=3.0, launches=4))
    k4 = k_tsv_launch(2.5, pv4)   # 2.5 ms per launch x 4 launches = 10 ms site total
    chk("ktsv_units_launches4", abs(k4 - 10.0) < 1e-12 and abs(predict_launch(k4, pv4) - 3.5) < 1e-12)
    chk("kswap_units_launches4", abs(k_swap(10.0, [0.25, -0.5], pv4) - 9.0) < 1e-12
        and k_swap(1.0, [-1.0], pv4) == 0.0)
    chk("nan_literal", _f(float("nan")) == "nan")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        a_ok = audit()
    chk("audit_clean", a_ok and buf.getvalue().splitlines()[-1] == "KLN AUDIT meas_at_predict=0")
    n = sum(ok for _, ok in res)
    for name, ok in res:
        if not ok:
            print("KLN SELFTEST FAIL %s" % name)
    print("KLN SELFTEST %s (%d/%d)" % ("PASS" if n == len(res) else "FAIL", n, len(res)))
    return n == len(res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    md = ap.add_mutually_exclusive_group(required=True)
    md.add_argument("--selftest", action="store_true")
    md.add_argument("--audit", action="store_true")
    md.add_argument("--oracle-k", action="store_true")
    md.add_argument("--insample", action="store_true")
    md.add_argument("--k-tsv", metavar="PATH")
    md.add_argument("--kswap", metavar="D1[,D2..]")
    ap.add_argument("--k-label")
    ap.add_argument("--loo", type=float)
    ap.add_argument("--dump-kmeas", metavar="PATH")
    ap.add_argument("--site-src", choices=("log", "lpred"), default="log")
    ap.add_argument("--khat")
    ap.add_argument("--khat-src", default="koracle_dma7")
    ap.add_argument("--fit-run", default="a")
    ap.add_argument("--r2", type=float, default=0.90)
    ap.add_argument("--max-rel", type=float, default=0.20)
    ap.add_argument("--lpred")
    ap.add_argument("--npred")
    ap.add_argument("--build-root")
    ap.add_argument("--fit", nargs="*", default=[])
    ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.dump_kmeas and not a.oracle_k:
        ap.error("--dump-kmeas needs --oracle-k")
    if a.selftest:
        return 0 if selftest() else 1
    if a.audit:
        return 0 if audit() else 1
    if not (a.lpred and a.npred and (a.fit or a.check or a.insample)):
        ap.error("--lpred, --npred and --fit/--check are required")
    if a.build_root and not Path(a.build_root).is_dir():
        ap.error("--build-root %s is not a directory" % a.build_root)
    khat = None
    if a.insample:
        khat = read_khat(a.khat, a.fit_run) if a.khat else None
        if khat is None:
            print("KLNPROBE khat=%s missing or empty" % a.khat)
            return 2
    lp = tsv_map(read_tsv(a.lpred), ("launches", "site_ms", "L_meas", "L_hat"))
    np_ = tsv_map(read_tsv(a.npred), ("launches", "site_ms", "N_meas", "N_hat"))
    if a.insample and not (a.fit or a.check):
        return 0 if insample(tsv_rows(lp, np_, a.fit_run), khat, "lpred", a.khat_src) else 1
    logs = [(c, p, True) for c, p in parse_logs(a.fit)] + [(c, p, False) for c, p in parse_logs(a.check)]
    rows, ok = join(lp, np_, logs)
    if not ok:
        return 1
    if a.insample:
        rows = [dict(r, run=a.fit_run) for r in rows if r["fit"]]
        return 0 if insample(rows, khat, a.site_src, a.khat_src) else 1
    if a.k_tsv or a.kswap:
        if a.k_tsv:
            kmap, label = kmap_tsv(rows, read_ktsv(a.k_tsv)), a.k_label or "hat"
        else:
            kmap, label = kmap_swap(rows, a.site_src, *read_kswap(a.kswap)), a.k_label or "swap"
        ok = khat_eval(rows, a.site_src, kmap, label, a.r2, a.max_rel, a.loo, lock_stream(rows))
        return 0 if ok else 1
    ok = oracle(rows, a.site_src, a.r2, a.max_rel)
    if a.dump_kmeas:
        print("KLNDUMP kmeas=%s rows=%d" % (a.dump_kmeas, dump_kmeas(rows, a.site_src, a.dump_kmeas)))
    kln121(rows, a.site_src)
    ktolneed(rows, a.site_src, a.max_rel)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
