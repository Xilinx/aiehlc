#!/usr/bin/env python3
"""params_table -- the report's parameter table straight from the frozen parameter files (costmodel M7 P5).

  params_table.py --md5 script/perf/params/m41_frozen.md5 script/perf/params/m42_frozen.md5 [--unet m3122] --out T.md

Inputs (read only, values printed with repr(), never rounded or edited):
  * every '<md5>  <relpath>' line of each --md5 file (resnet18 = m41, yolo26n = m42): the md5 is checked;
  * the UNet frozen set aiegraph.costmodel.estimate.KLN_PARAMS[palmyra213][--unet] (M34 PZ), each file must be
    unchanged against git HEAD.
Table: K (per-launch kernel body), L (per-launch launch cost), N (per-site non-kernel time, one block per net:
unet, resnet18, yolo26n).  Source column = the json's own log / fit_log / fit.log / refit.log / board fields; a
parameter whose json carries none is counted in missing_src and then filled from SRC_FILL (the fit command of the
commit that froze the file); every source path must exist on disk.  Last line, also on stdout:
  M7PARAMS files=<n> keys=<n> missing_src=<unresolved> PASS|FAIL
plus M7SRC json=<n> filled=<n> (the gaps that were filled) and the remaining-error draft section.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRM = ROOT / "script" / "perf" / "params"
SH = "/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks"
AR = "/scratch/staff/huaj/aiengineers/arthur/clones/cost/tasks"
NETS = ("unet", "resnet18", "yolo26n")
NET_CFGS = {"unet": ("off", "aot-kco"), "resnet18": ("off", "aot-kco"), "yolo26n": ("off", "aot")}

# fill for the frozen files whose json has no log field (key: (file name, cfg)) -- from the freezing commits:
#   k_palmyra213.m3122.json  9c917c86 "k_palmyra213.m3122.json fit on M3121 run a"
#   kdw_palmyra213.m3112c.json 7e9b2e91 "from the PW command on the 6 M3121 logs (fit run a)"
#   l_palmyra213.m3121.json  4dda856b aot-kco refit offset/kswitch "on run a" of the UNet M3121 logs; the other
#     coefficients are l_palmyra213.json's (44a9ac3e/c7a9b0a4/0afba0f5: l_fit.py --kernels on the microbench logs
#     named in cfg.<cfg>.kernels, resolved below by L_KDIRS)
SRC_FILL = {
    ("k_palmyra213.m3122.json", "off"): SH + "/2026-11-20_M3121/unet_off.a.log",
    ("k_palmyra213.m3122.json", "aot-kco"): SH + "/2026-11-20_M3121/unet_aot-kco.a.log",
    ("kdw_palmyra213.m3112c.json", "off"): SH + "/2026-11-20_M3121/unet_off.a.log",
    ("kdw_palmyra213.m3112c.json", "aot-kco"): SH + "/2026-11-20_M3121/unet_aot-kco.a.log",
    ("l_palmyra213.m3121.json", "aot-kco/refit"): SH + "/2026-11-20_M3121/unet_aot-kco.a.log",
}
L_KDIRS = (SH + "/2026-11-20_M1-L/kernels", SH + "/2026-11-20_M1-2/kernels", SH + "/2026-11-20_M1-2-2/kernels",
           "/scratch/staff/huaj/aiengineers/amit/clones/cost/tasks/2026-11-20_M3_alt")   # 0afba0f5 off-alt pairs


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def git_clean(p):
    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", str(p)])
    t = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(p)], capture_output=True)
    return r.returncode == 0 and t.returncode == 0


class Table:
    def __init__(self):
        self.rows = []          # (sec, net, file, cfg, group, key, value, src)
        self.missing = []       # (file, cfg, why) before fill
        self.unresolved = []    # (file, cfg, why) after fill / not on disk
        self.filled = 0
        self.from_json = 0
        self.inputs = []        # (sec, net, file, what, n, src)

    def src(self, fname, cfg, js_src, fill_key=None):
        """source of one parameter group: the json field if present, else SRC_FILL; must exist on disk"""
        if js_src:
            s, how = js_src, "json"
        else:
            self.missing.append((fname, cfg, "no log/board field in json"))
            s, how = SRC_FILL.get((fname, fill_key or cfg)), "fill"
        if not s:
            self.unresolved.append((fname, cfg, "no fill"))
            return None, how
        paths = s if isinstance(s, list) else [s]
        bad = [p for p in paths if not os.path.exists(p)]
        if bad:
            self.unresolved.append((fname, cfg, "not on disk: %s" % bad[0]))
            return None, how
        return s, how

    def add(self, sec, net, fname, cfg, group, params, src, how):
        for k in sorted(params):
            v = params[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            self.rows.append((sec, net, fname, cfg, group, k, v, src, how))
            if src is not None:
                if how == "json":
                    self.from_json += 1
                else:
                    self.filled += 1


def src_txt(s):
    if s is None:
        return "**MISSING**"
    if isinstance(s, list):
        d = os.path.commonpath(s)
        return "`%s/` %d logs: %s" % (d, len(s), ", ".join(os.path.relpath(p, d) for p in s))
    return "`%s`" % s


def load_md5(mf, net, files):
    """{role: path} for one net from a 'md5  relpath' file, md5 checked"""
    out = {}
    for ln in Path(mf).read_text().splitlines():
        if not ln.strip():
            continue
        m, rp = ln.split(None, 1)
        p = ROOT / rp.strip()
        got = md5(p)
        if got != m:
            raise SystemExit("M7PARAMS md5 mismatch %s %s != %s (%s)" % (rp, got, m, mf))
        b = p.name
        role = ("l" if b.startswith("l_") else "n" if b.startswith("n_") else "kcyc" if b.startswith("kcyc_")
                else "k" if b.startswith("k_") else "board" if b == "palmyra213.json" else None)
        if role is None:
            raise SystemExit("M7PARAMS unknown role %s in %s" % (b, mf))
        out[role] = p
        files.setdefault(p, (m, "md5 file " + os.path.relpath(mf, ROOT)))
    return out


def net_of_md5(mf):
    for ln in Path(mf).read_text().splitlines():
        b = ln.split()[-1] if ln.strip() else ""
        for n in NETS:
            if n in Path(b).name and Path(b).name.startswith("n_"):
                return n
    raise SystemExit("M7PARAMS cannot tell the net of %s (no n_<net>_ file)" % mf)


def board_rows(T, p, net="all"):
    d = json.load(open(p))
    grp = {"ddr": {}, "other": {}}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            grp["ddr" if k.startswith("ddr_") else "other"][k] = v
    s, how = T.src(p.name, "-", d.get("ddr_log"))
    T.add("K", net, p.name, "-", "board (DDR microbench)", grp["ddr"], s, how)
    s, how = T.src(p.name, "-", d.get("log"))
    T.add("K", net, p.name, "-", "board (DMA/lock/clock microbench)", grp["other"], s, how)


def knet_rows(T, net, p):
    d = json.load(open(p))
    for cfg, c in sorted(d["cfgs"].items()):
        s, how = T.src(p.name, cfg, c.get("fit_log"))
        T.add("K", net, p.name, cfg, "theta (k_net, fit run %s)" % d.get("fit_run", "?"), c["theta"], s, how)


def kcyc_rows(T, net, p):
    d = json.load(open(p))
    for cfg, c in sorted(d["cfgs"].items()):
        a = [x for x in c["logs"] if x.endswith(".%s.log" % d.get("fit_run", "a"))]
        s, how = T.src(p.name, cfg, a[0] if a else None)
        T.add("K", net, p.name, cfg, "kcore W (pred_cyc = total - min(pre, W))", {"W_cyc": c["W_cyc"]}, s, how)
        lst = [i["lst"] for i in c["instances"].values()]
        ok = all(os.path.exists(x) for x in lst)
        if not ok:
            T.unresolved.append((p.name, cfg, "kernel.lst not on disk"))
        T.inputs.append(("K", net, p.name, cfg + ": static core cycles (aie_cc kernel.lst)", len(lst),
                         "`%s` %d kernel.lst%s" % (d.get("cc_root"), len(lst), "" if ok else " **MISSING**")))


def kunet_rows(T, p_k, p_kdw, p_kcyc):
    d = json.load(open(p_k))
    for cfg, c in sorted(d["cfg"].items()):
        for cls, x in sorted(c["classes"].items()):
            s, how = T.src(p_k.name, cfg, None)
            T.add("K", "unet", p_k.name, cfg, "K_hat class %s (form %s)" % (cls, x.get("form")), x, s, how)
    d = json.load(open(p_kdw))
    for cfg, c in sorted(d["params"].items()):
        for cls, x in sorted(c.items()):
            prm = {"w[%s]" % f: w for f, w in zip(x["feats"], x["w"])}
            prm["b"] = x["b"]
            s, how = T.src(p_kdw.name, cfg, None)
            T.add("K", "unet", p_kdw.name, cfg, "dw_hat %s" % cls, prm, s, how)
    d = json.load(open(p_kcyc))
    logs = [k.get("log") for k in d["kernels"].values()]
    ok = all(x and os.path.exists(x) for x in logs)
    if not ok:
        T.unresolved.append((p_kcyc.name, "-", "iss.log not on disk"))
    T.inputs.append(("K", "unet", p_kcyc.name, "core cycles (ISS, one launch)", len(logs),
                     "%d iss.log (maxwell M3_kcyc / maxwell_faiss)%s" % (len(logs), "" if ok else " **MISSING**")))


def l_kernel_logs(names):
    out = []
    for n in names:
        sub, k = n.split("/", 1)
        hit = [os.path.join(d, sub, k + ".log") for d in L_KDIRS if os.path.exists(os.path.join(d, sub, k + ".log"))]
        out.append(hit[0] if hit else os.path.join(L_KDIRS[0], sub, k + ".log"))
    return out


def l_rows(T, p, used):
    d = json.load(open(p))
    for cfg, c in sorted(d["cfg"].items()):
        net = ",".join(n for n in NETS if cfg in used.get(n, ())) or "none (kept in file)"
        coef = dict(c["coef"])
        refit = {k: coef.pop(k) for k in ("offset", "kswitch") if k in coef}
        logs = l_kernel_logs(c.get("kernels", []))
        s, how = T.src(p.name, cfg, None)          # the L json carries only board=, no log
        if s is None and logs:                    # fill: the microbench logs of cfg.kernels
            T.unresolved.pop()
            bad = [x for x in logs if not os.path.exists(x)]
            if bad:
                T.unresolved.append((p.name, cfg, "not on disk: %s" % bad[0]))
            else:
                s = logs
        T.add("L", net, p.name, cfg, "L per launch (l_fit, n=%s)" % c.get("n"), coef, s, how)
        if refit:
            s, how = T.src(p.name, cfg, None, fill_key=cfg + "/refit")
            T.add("L", net, p.name, cfg, "L refit (fit_runs %s)" % ",".join(c.get("fit_runs", [])), refit, s, how)


def n_rows(T, net, p, cfgs):
    d = json.load(open(p))
    for cfg in cfgs:
        if cfg not in d:
            continue
        c = d[cfg]
        js = (c.get("fit") or c.get("refit") or {}).get("log")
        for kind, x in sorted(c["kinds"].items()):
            prm = dict(x["coef"]) if "coef" in x else {k: x[k] for k in ("a", "b", "c") if k in x}
            prm["d"] = x["d"]
            s, how = T.src(p.name, cfg, js)
            T.add("N", net, p.name, cfg, "N %s (n=%s)" % (kind, x.get("n")), prm, s, how)
        top = {k: c[k] for k in ("attn_host_ms",) if k in c}
        if top:
            s, how = T.src(p.name, cfg, js)
            T.add("N", net, p.name, cfg, "N host attention", top, s, how)
        if "ovl" in c:
            for kind, x in sorted(c["ovl"]["kinds"].items()):
                prm = dict(x["coef"])
                prm["d"] = x["d"]
                s, how = T.src(p.name, cfg, js)
                T.add("N", net, p.name, cfg, "ovl glue %s (n=%s)" % (kind, x.get("n")), prm, s, how)
            s, how = T.src(p.name, cfg, js)
            T.add("N", net, p.name, cfg, "ovl", {"ovl_k": c["ovl"]["ovl_k"]}, s, how)


def model_line(p, cfg=None):
    d = json.load(open(p))
    m = d.get("model") if cfg is None else d.get(cfg, {}).get("model")
    if isinstance(m, list):
        m = " ".join(x.strip() for x in m if x.strip().startswith(("T_hat", "K_hat")))
    return (m or "").replace("|", "/")


ERR_DRAFT = """## Remaining error sources (draft)

1. **conv DMA term (K).** The largest share of the remaining K error on conv sites is the DMA term: k_diag
   KRESID gives `top_term=dma` for conv (cfg=off conv_big: abs_dma=421.908 ms of sum_abs_resid 327.297 ms over 141
   site-runs; conv_tkw4 abs_dma=178.099 / 180.792 ms). The model prices DMA as bytes / measured shim bandwidth
   (s_rd, s_wr, bd switch). It has no term for overlap between shim channels or for memtile contention on the
   conv access pattern. Source: `%(ar)s/2026-11-21_M3122/kdiag.out`.
2. **aot-kco conv2 board jitter.** The PN TNOISE spread between board runs is spread_max=0.035 (0.0352, aot-kco,
   m31223). In M34 the conv2 margin to the 0.20 limit is about 2.2 %%. One run about 2 %% slow on conv2
   (T <= 2.77 ms) would cross 0.20. This is board noise, not model error. A median over runs would remove it; that
   is huaj's decision. Source: `%(ar)s/2026-11-22_M34/REPORT.md`.
3. **L kfirst.** L is an average per launch (frozen l_palmyra213.m3121, l_fit on the microbench kernels). The first
   launch of a site costs more (kernel load, core reset), and L does not model that. The extra cost is absorbed by
   the kfirst terms that were fitted elsewhere: K kfirst in the UNet classes (aot-kco fa16 0.0401, linear 0.0034)
   and N kfirst in resnet18 aot-kco conv (0.1233). So the cost is attributed to the wrong term, and on a net with
   a different number of launches per site it will not carry over.
""" % {"ar": AR}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md5", nargs="+", required=True, help="frozen md5 files (m41_frozen.md5 m42_frozen.md5)")
    ap.add_argument("--unet", default="m3122", help="UNet frozen set name in estimate.KLN_PARAMS (default m3122)")
    ap.add_argument("--board", default="palmyra213")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    from aiegraph.costmodel.estimate import KLN_PARAMS
    files = {}                                   # path -> (md5, how frozen)
    sets = {}
    for mf in a.md5:
        sets[net_of_md5(mf)] = load_md5(mf, net_of_md5(mf), files)
    us = {r: PRM / f for r, f in KLN_PARAMS[a.board][a.unet].items()}
    for r, p in us.items():
        if not git_clean(p):
            raise SystemExit("M7PARAMS unet frozen file %s differs from git HEAD" % p)
        files.setdefault(p, (md5(p), "estimate.KLN_PARAMS[%s][%s] (git HEAD)" % (a.board, a.unet)))
    sets["unet"] = us

    T = Table()
    used = {n: NET_CFGS[n] for n in sets}
    boards = sorted({str(s["board"]) for s in sets.values() if "board" in s})
    for b in boards:
        board_rows(T, Path(b), net=",".join(n for n in NETS if n in sets and str(sets[n].get("board")) == b))
    kunet_rows(T, us["k"], us["kdw"], us["kcyc"])
    for n in ("resnet18", "yolo26n"):
        if n in sets:
            knet_rows(T, n, sets[n]["k"])
            kcyc_rows(T, n, sets[n]["kcyc"])
    lfiles = sorted({str(s["l"]) for s in sets.values()})
    for lf in lfiles:
        l_rows(T, Path(lf), {n: NET_CFGS[n] for n in sets if str(sets[n]["l"]) == lf})
    for n in NETS:
        if n in sets:
            n_rows(T, n, sets[n]["n"], NET_CFGS[n])

    keys = len(T.rows)
    miss = len(T.unresolved)
    ok = miss == 0 and keys > 0 and len(files) > 0
    res = "M7PARAMS files=%d keys=%d missing_src=%d %s" % (len(files), keys, miss, "PASS" if ok else "FAIL")
    srcl = "M7SRC json=%d filled=%d gaps_before_fill=%d" % (T.from_json, T.filled, len(T.missing))

    L = ["# Cost model parameter table (frozen, %s)" % a.board, "",
         "Generated by `script/perf/params_table.py --md5 %s --out %s` from the frozen files below. Values are"
         " the json values printed with repr(); none was edited." % (" ".join(a.md5), a.out), "",
         "## Frozen files", "", "| file | md5 | frozen by |", "|---|---|---|"]
    for p, (m, how) in sorted(files.items(), key=lambda x: str(x[0])):
        L.append("| `%s` | `%s` | %s |" % (os.path.relpath(p, ROOT), m, how))
    L += ["", "## Model structure", "",
          "Per site: T_hat = K_hat x launches + L_hat x launches + N_hat (ms).", "",
          "* K (unet, %s): %s" % (us["k"].name, model_line(us["k"])),
          "* K (resnet18 / yolo26n, k_net): %s" % model_line(sets.get("resnet18", sets.get("yolo26n", us))["k"])
          if "resnet18" in sets or "yolo26n" in sets else "",
          "* L (%s): L_hat per launch = %s, features %s" % (
              Path(lfiles[0]).name, json.load(open(lfiles[0])).get("l_def"),
              ", ".join(json.load(open(lfiles[0])).get("features", []))),
          "* N: per net, per kind; %s" % model_line(us["n"], "off"), ""]
    for sec, title in (("K", "K -- kernel body per launch"), ("L", "L -- launch cost per launch"),
                       ("N", "N -- non-kernel time per site")):
        L += ["## %s" % title, ""]
        nets = ("all",) + NETS if sec != "L" else (None,)
        for net in nets:
            rows = [r for r in T.rows if r[0] == sec and (net is None or r[1] == net or
                                                           (net == "all" and "," in r[1]))]
            if net is not None and net != "all":
                rows = [r for r in rows if "," not in r[1]]
            if not rows:
                continue
            if net is not None:
                L += ["### %s" % ("board (shared by %s)" % rows[0][1] if net == "all" else net), ""]
            L += ["| net | file | cfg | group | param | value | source |", "|---|---|---|---|---|---|---|"]
            last = None
            for _s, rn, f, cfg, g, k, v, s, how in rows:
                sc = src_txt(s) + (" (filled)" if how == "fill" and s else "")
                key = (f, cfg, g)
                L.append("| %s | %s | %s | %s | %s | %r | %s |" % (
                    rn, f, cfg, g, k, v, sc if key != last else "″"))
                last = key
            L.append("")
            inp = [x for x in T.inputs if x[0] == sec and (x[1] == net)]
            if inp:
                L += ["Static inputs (not fitted):", ""]
                L += ["* %s `%s` %s: %d entries, source %s" % (x[1], x[2], x[3], x[4], x[5]) for x in inp]
                L.append("")
    L += ["## Source check", "",
          "Parameters whose json had no log field (counted as gaps, then filled from the freezing commit):", ""]
    seen = set()
    for f, cfg, why in T.missing:
        if (f, cfg) not in seen:
            seen.add((f, cfg))
            L.append("* `%s` cfg=%s: %s" % (f, cfg, why))
    L += ["", "Unresolved after fill: %s" % (", ".join("%s/%s (%s)" % u for u in T.unresolved) or "none"), "",
          srcl, "", ERR_DRAFT, res, ""]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Path(a.out).write_text("\n".join(L))
    print(srcl)
    for u in T.unresolved:
        print("M7UNRESOLVED %s cfg=%s %s" % u)
    print(res)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
