#!/usr/bin/env python3
"""Per-launch K + L + N synthesis of UNet AIE sites (costmodel M3.1.1.2 PC; first cut of M6 estimate()).

  kln_site.py --selftest
  kln_site.py --audit
  kln_site.py --oracle-k --site-src log|lpred [--r2 0.90] [--max-rel 0.20] ARGS
  kln_site.py --insample --khat KHAT.tsv ARGS                    (KLNPROBE; rc=2 if KHAT is missing or empty)
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


PREDICT_FNS = (pred_view, predict_launch)


def audit():
    bad = 0
    for fn in PREDICT_FNS:
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
        bad += len(meas)
        print("KLN AUDIT fn=%s names=%s meas=%s" % (fn.__name__, ",".join(sorted(names)), ",".join(meas) or "-"))
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
            rows.append(dict(cfg=cfg, run=run, part=part, site=site, launches=n, site_ms_log=site_ms,
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


def read_khat(path):
    """{(cfg, run, part): K_hat ms (site)}; None if missing or empty"""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    rows = read_tsv(p)
    return {(r["cfg"], norm_run(r["run"]), int(r["part"])): float(r["K_hat"]) for r in rows} or None


def insample(rows, khat, src, r2_gate, rel_gate):
    ok_all = True
    for cfg in dict.fromkeys(r["cfg"] for r in rows):
        rs = [r for r in rows if r["cfg"] == cfg and (r["cfg"], r["run"], r["part"]) in khat]
        if not rs:
            print("KLNPROBE cfg=%s sites=0 FAIL" % cfg)
            ok_all = False
            continue
        pairs = [(r["site_ms_" + src] / r["launches"],
                  predict_launch(khat[(r["cfg"], r["run"], r["part"])], pred_view(r)), r["launches"], r["site"])
                 for r in rs]
        r2, mr, worst, sr = metrics(pairs)
        ok = r2 >= r2_gate and mr <= rel_gate
        ok_all &= ok
        print("KLNPROBE cfg=%s sites=%d k=hat r2=%.4f maxrel=%.4f sum_rel=%.4f worst=%s %s"
              % (cfg, len(rs), r2, mr, sr, worst, "PASS" if ok else "FAIL"))
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
    ap.add_argument("--site-src", choices=("log", "lpred"), default="log")
    ap.add_argument("--khat")
    ap.add_argument("--r2", type=float, default=0.90)
    ap.add_argument("--max-rel", type=float, default=0.20)
    ap.add_argument("--lpred")
    ap.add_argument("--npred")
    ap.add_argument("--build-root")
    ap.add_argument("--fit", nargs="*", default=[])
    ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    if a.audit:
        return 0 if audit() else 1
    if not (a.lpred and a.npred and (a.fit or a.check)):
        ap.error("--lpred, --npred and --fit/--check are required")
    if a.build_root and not Path(a.build_root).is_dir():
        ap.error("--build-root %s is not a directory" % a.build_root)
    khat = None
    if a.insample:
        khat = read_khat(a.khat) if a.khat else None
        if khat is None:
            print("KLNPROBE khat=%s missing or empty" % a.khat)
            return 2
    lp = tsv_map(read_tsv(a.lpred), ("launches", "site_ms", "L_meas", "L_hat"))
    np_ = tsv_map(read_tsv(a.npred), ("launches", "site_ms", "N_meas", "N_hat"))
    logs = [(c, p, True) for c, p in parse_logs(a.fit)] + [(c, p, False) for c, p in parse_logs(a.check)]
    rows, ok = join(lp, np_, logs)
    if not ok:
        return 1
    if a.insample:
        return 0 if insample(rows, khat, a.site_src, a.r2, a.max_rel) else 1
    ok = oracle(rows, a.site_src, a.r2, a.max_rel)
    kln121(rows, a.site_src)
    ktolneed(rows, a.site_src, a.max_rel)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
