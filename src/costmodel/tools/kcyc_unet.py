#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Per-launch AIE core cycles (core_cyc) for every UNet AIE site (costmodel M3 P1, K = DES(core_cyc)).

  kcyc_unet.py --build-root <unet build dir> --iss <aie2ps_kernel_cycles.json> --out <json>
               [--build-log <dir>.build.log] [--iss-unet src/costmodel/params/kcyc_unet_iss.json]

Sites: n_feat.sites(build, build.log) (graph_driver.c order). Counts are cross-checked with the build log line
  '[build_unet] linked driver: <c> call sites -> aiednn_aie_conv + <l> -> aiednn_aie_linear (AIE) ...; <a> -> aiednn_aie_attn'
  kacc sites = conv + linear sites (gemm_kacc__* kernels, all with launches > 0) must equal c + l;
  attn call sites (distinct attn cids) must equal a; fa sites = attn sites with launches > 0.
core_cyc = ISS cycles of ONE core for ONE launch (multiply by launches per site). Source per kacc site, in order:
  iss       the site's gemm_kacc instance is one of the --iss json's u_* kernels (K2m board-TU ISS, iss_board.sh);
            matched on the instance (ch, k, px, tkw, kch, nkc, ntile, S, tr) decoded from the kernel name
            gemm_kacc__<N>_<K>_<M>_<K/2>_<f4>_<f5>_<tkw>_<kch>_<nkc>_<ntile>_<S>_<tr> (tr=0: ch=f4 px=f5; tr=1: px=f4 ch=f5)
  iss_unet  ISS of the UNet build's own per-kernel TU (--iss-unet json; same iss_board.sh harness, staged from
            <build>/ag_work/aiek/aiek_unetconv/worklocal by maxwell's iss_unet_kernel.sh); used only when its
            recorded source md5 matches this build's TU
  formula   core_cyc = a*MACs + b*chunks + c*rounds + d*chunks*S + e*rounds*S  (per core, per launch;
            MACs = ntile*nkc*S*256*kch, chunks = ntile*nkc (input-window rounds), rounds = ntile (output rounds))
            NNLS on relative error over the --iss json's u_* points; leave-one-out max relative error is printed.
            (a*MACs + b*chunks + c*rounds alone has LOO 0.66 on the 14 points: the TKW=8 vs 32 instances differ
            in per-sub-tile cost, hence the two *S terms.)
fa sites, per site:
  iss            the --iss-unet json has an entry (or one of its "variants") for the site's fa_kernel__<K> (ISS of a
                 UNet build's own FA TU, iss_unet_fa.sh) whose recorded kernel_<K>.cc / <K>.cc / aieml_<K>.bcf md5
                 equal this build's (unet_off and unet_aot-kco FA TUs differ: KCO code; each has its own ISS run)
  formula_noloo  else: core_cyc = (2*h*n*m*d MACs / launches / 16 cores) / median u_* ISS MAC/cyc (ASSUMED)
  fa_src = iss when every FA site is iss, formula_noloo when none, else mixed.
Prints one KCYC_SITE line per AIE site, KCYC_FCHK lines (formula vs iss_unet points, if any), then
  KCYC unet kacc=<covered>/<n> fa=<n_fa_cyc>/<n_fa> iss=<i> formula=<f> fa_src=<iss|mixed|formula_noloo> formula_loo_maxrel=<x>
  KCYC_SUM kacc_sites=<n> covered=<covered>
rc=1 when iss+formula != kacc sites, or any count disagrees with the build log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from costmodel.n_feat import gfa_classes, sites as n_sites  # noqa: E402
from costmodel.unet_seq import read  # noqa: E402

HERE = Path(__file__).resolve().parent
_LINKED = re.compile(r"linked driver: (\d+) call sites -> aiednn_aie_conv \+ (\d+) -> aiednn_aie_linear .*?; (\d+) -> "
                     r"aiednn_aie_attn")
_KACC = re.compile(r"^gemm_kacc__" + "_".join([r"(\d+)"] * 12) + "$")
FEATS = ("macs", "chunks", "rounds", "chunks_S", "rounds_S")
NCORE = 16


def kacc_inst(kernel):
    m = _KACC.match(kernel)
    if not m:
        raise ValueError("not a gemm_kacc kernel name: " + kernel)
    N, K, M, _kh, f4, f5, tkw, kch, nkc, ntile, S, tr = (int(x) for x in m.groups())
    ch, px = (f5, f4) if tr else (f4, f5)
    return dict(ch=ch, k=K, px=px, ntile=ntile, S=S, tkw=tkw, kch=kch, nkc=nkc, tr=tr)


def inst_key(i):
    return tuple(i[x] for x in ("ch", "k", "px", "tkw", "kch", "nkc", "ntile", "S", "tr"))


def feats(i):
    chunks, rounds = i["ntile"] * i["nkc"], i["ntile"]
    return [i["ntile"] * i["nkc"] * i["S"] * 256 * i["kch"], chunks, rounds, chunks * i["S"], rounds * i["S"]]


def fit(X, y):
    w = 1.0 / y
    c, _ = nnls(X * w[:, None], y * w)
    return c


def loo(X, y):
    errs = []
    for i in range(len(y)):
        idx = [j for j in range(len(y)) if j != i]
        c = fit(X[idx], y[idx])
        errs.append(abs(float(X[i] @ c) - y[i]) / y[i])
    return errs


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def fa_iss_pick(wl, kern, u):
    """the --iss-unet FA entry (or one of its 'variants', e.g. the aot-kco TU) whose recorded TU md5
    (kernel_<K>.cc, <K>.cc, aieml_<K>.bcf) equals this build's; None when no ISS run matches this TU"""
    try:
        have = (md5(os.path.join(wl, "kernel_%s.cc" % kern)), md5(os.path.join(wl, "%s.cc" % kern)),
                md5(os.path.join(wl, "aieml_%s.bcf" % kern)))
    except OSError:
        return None
    for v in [u] + list(u.get("variants", [])):
        m = v.get("src_md5") or {}
        if (m.get("kernel"), m.get("fa"), m.get("bcf")) == have and v.get("core_cyc", 0) > 0:
            return v
    return None


def fa_src_of(n_iss, n_fa):
    """iss when every FA site has an ISS value, formula_noloo when none, else mixed"""
    if n_fa and n_iss == n_fa:
        return "iss"
    return "mixed" if n_iss else "formula_noloo"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--build-log")
    ap.add_argument("--iss", required=True)
    ap.add_argument("--iss-unet", default=str(HERE.parent / "params" / "kcyc_unet_iss.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    build = a.build_root.rstrip("/")
    blog = a.build_log or build + ".build.log"
    rc = 0

    m = _LINKED.search(read(blog))
    if not m:
        print("KCYC_ERR no 'linked driver' line in " + blog)
        return 1
    n_conv, n_lin, n_attn = (int(x) for x in m.groups())
    st = n_sites(build, blog)
    gfa = gfa_classes(blog)

    # formula on the ISS json's u_* points
    iss = [k for k in json.load(open(a.iss))["kernels"] if k["name"].startswith("u_")]
    iss_by_key = {inst_key(k["instance"]): k for k in iss}
    X = np.array([feats(k["instance"]) for k in iss], float)
    y = np.array([k["core_cyc"] for k in iss], float)
    coef = fit(X, y)
    loo_err = loo(X, y)
    loo_max = max(loo_err)
    ins_max = float(np.max(np.abs(X @ coef - y) / y))
    mac_rate = statistics.median(k["core_macs"] / k["core_cyc"] for k in iss)

    iss_unet = {}
    if a.iss_unet and os.path.exists(a.iss_unet):
        iss_unet = json.load(open(a.iss_unet)).get("kernels", {})
    wl = os.path.join(build, "ag_work", "aiek", "aiek_unetconv", "worklocal")

    out_sites, fchk = [], []
    cnt = dict(iss=0, iss_unet=0, formula=0, fa=0, fa_iss=0)
    n_kacc = n_fa = 0
    for i, s in enumerate(st):
        if s["kind"] in ("conv", "linear"):
            n_kacc += 1
            if s["launches"] <= 0 or len(s["kernels"]) != 1:
                print("KCYC_ERR site=%d %s%d launches=%d kernels=%s" % (i, s["kind"], s["cid"], s["launches"],
                                                                         s["kernels"]))
                rc = 1
                continue
            kern = s["kernels"][0]
            inst = kacc_inst(kern)
            f = feats(inst)
            pf = float(np.array(f) @ coef)
            rec = dict(i=i, kind=s["kind"], cid=s["cid"], kernel=kern, launches=s["launches"], instance=inst,
                       formula_cyc=round(pf, 1))
            hit = iss_by_key.get(inst_key(inst))
            u = iss_unet.get(kern)
            u_ok = False
            if u is not None and not hit:
                try:
                    u_ok = (md5(os.path.join(wl, "kernel_%s.cc" % kern)) == u["src_md5"]["kernel"] and
                            md5(os.path.join(wl, "%s.cc" % kern)) == u["src_md5"]["gemm"])
                except OSError:
                    u_ok = False
            if hit:
                rec.update(core_cyc=hit["core_cyc"], src="iss", iss_name=hit["name"], iss_log=hit.get("log"))
                cnt["iss"] += 1
            elif u_ok:
                rec.update(core_cyc=u["core_cyc"], src="iss_unet", iss_log=u.get("log"))
                cnt["iss_unet"] += 1
                fchk.append((kern, u["core_cyc"], pf))
            else:
                rec.update(core_cyc=round(pf, 1), src="formula")
                cnt["formula"] += 1
            rec["core_cyc_site"] = rec["core_cyc"] * s["launches"]
            out_sites.append(rec)
        elif s["kind"] == "fa":
            n_fa += 1
            g = gfa[s["cls"]]
            macs = 2 * g["h"] * g["n"] * g["m"] * g["d"]
            kern = s["kernels"][0]
            cyc = macs / s["launches"] / NCORE / mac_rate
            rec = dict(i=i, kind="fa", cid=s["cid"], kernel=kern, launches=s["launches"], cls=s["cls"], fa=g,
                       macs_site=macs, formula_cyc=round(cyc, 1))
            u = iss_unet.get(kern)
            v = fa_iss_pick(wl, kern, u) if u is not None else None
            if v is not None:
                rec.update(core_cyc=v["core_cyc"], src="iss", iss_log=v.get("log"))
                cnt["fa_iss"] += 1
            else:
                rec.update(core_cyc=round(cyc, 1), src="formula_noloo")
            rec["core_cyc_site"] = rec["core_cyc"] * s["launches"]
            out_sites.append(rec)
            cnt["fa"] += 1

    seen = set()
    for r in out_sites:
        print("KCYC_SITE i=%d kind=%s cid=%d launches=%d core_cyc=%.1f src=%s kernel=%s" %
              (r["i"], r["kind"], r["cid"], r["launches"], r["core_cyc"], r["src"], r["kernel"]))
    for kern, cy, pf in fchk:
        if kern not in seen:
            seen.add(kern)
            print("KCYC_FCHK kernel=%s iss=%d formula=%.0f rel=%.3f" % (kern, cy, pf, abs(pf - cy) / cy))

    n_kacc_bl = n_conv + n_lin
    n_conv_st = sum(1 for s in st if s["kind"] == "conv")
    n_lin_st = sum(1 for s in st if s["kind"] == "linear")
    n_attn_st = len({s["cid"] for s in st if s["kind"] in ("fa", "attn")})
    n_fa_bl = sum(1 for s in st if s["kind"] == "fa" and s["launches"] > 0)
    if (n_conv_st, n_lin_st, n_attn_st) != (n_conv, n_lin, n_attn):
        print("KCYC_ERR build.log conv/linear/attn=%d/%d/%d vs n_feat sites %d/%d/%d" %
              (n_conv, n_lin, n_attn, n_conv_st, n_lin_st, n_attn_st))
        rc = 1
    n_iss = cnt["iss"] + cnt["iss_unet"]
    covered = n_iss + cnt["formula"]
    if covered != n_kacc_bl or n_kacc != n_kacc_bl:
        print("KCYC_ERR iss+formula=%d kacc_sites=%d build.log=%d" % (covered, n_kacc, n_kacc_bl))
        rc = 1
    fa_src = fa_src_of(cnt["fa_iss"], cnt["fa"])
    doc = dict(build=build, build_log=blog, iss_json=a.iss, iss_unet_json=a.iss_unet if iss_unet else None,
               core_cyc_unit="ISS cycles, one core, one launch (multiply by launches)",
               counts=dict(build_log=dict(conv=n_conv, linear=n_lin, attn=n_attn), kacc_sites=n_kacc, fa_sites=n_fa,
                           iss=cnt["iss"], iss_unet=cnt["iss_unet"], formula=cnt["formula"]),
               formula=dict(feats=FEATS, coef=[float(x) for x in coef], fitted_on=[k["name"] for k in iss],
                            loo_maxrel=loo_max, in_maxrel=ins_max,
                            loo=dict(zip([k["name"] for k in iss], [round(e, 4) for e in loo_err])),
                            chk_iss_unet=[dict(kernel=k, iss=c, formula=p) for k, c, p in fchk]),
               fa=dict(src=fa_src, n_iss=cnt["fa_iss"], n_fa=cnt["fa"], mac_per_cyc_assumed=mac_rate,
                       note="src=iss: core_cyc = ISS of the site's own FA TU (--iss-unet entry, md5 of kernel/fa/bcf "
                            "checked against this build); fallback formula_noloo = 2*h*n*m*d / launches / 16 / median "
                            "u_* ISS MAC/cyc. No coefficient is fitted to board compute_cyc."),
               sites=out_sites)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print("KCYC unet kacc=%d/%d fa=%d/%d iss=%d formula=%d fa_src=%s formula_loo_maxrel=%.3f" %
          (covered, n_kacc_bl, cnt["fa"], n_fa_bl, n_iss, cnt["formula"], fa_src, loo_max))
    print("KCYC_SUM kacc_sites=%d covered=%d" % (n_kacc, covered))
    return rc


if __name__ == "__main__":
    sys.exit(main())
