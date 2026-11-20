#!/usr/bin/env python3
"""N-segment (host glue) features per UNet AIE site from a build directory only (costmodel N fit, P2).

  n_feat.py --build <unet build dir> [--build-log <build>.build.log] [--check-log <board log>] [--json out.json]

Per site (one aiednn_aie_* call in graph_run, graph_driver.c order; parse via unet_seq.parse_driver/launches_of):
  conv    M=OH*OW, N=O, K=C*KH*KW                     MK=M*K  MN=M*N  launches=1
  linear  M,K from x, N from y; S launches (split)     MK=M*K  MN=M*N  launches=S
  attn_kl role 6 (fa_run_o) of GFA class k:            MK=h*n*d  MN=h*n*m  (ASSUMED definition)
          launches = g_fa_cls[k].launches (aiek src), cross-checked with the build log
          "GFA class k fa_h<h>_n<n>_m<m>_d<d> ... launches/site=<L>"; other attn roles: 0 launches, MK=MN=0.
No board number enters the features. --check-log aligns the sites with the board log 'PROF part=<i> op=aie_host_*'
lines in order: kind and cid (op suffix) must match, sites with launches>0 must be exactly those with aie_ms>0,
and sum(launches) must equal 'LAUNCH warm|cold ... launches=<n>'.

  NFEAT build=<name> conv=<l> linear=<l> fa=<l> kernels=<k> launches=<n> log_launches=<n> aligned=<a>/<sites> PASS|FAIL
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # python -I: no implicit src on sys.path

from aiegraph.costmodel.n_layout import layout_sites  # noqa: E402
from aiegraph.costmodel.static_feat import unet_site_feat  # noqa: E402
from aiegraph.costmodel.unet_seq import launches_of, parse_aiek, parse_driver, read  # noqa: E402

LAUNCH_FEATS = ("cp_w", "bdcfg_w", "shim_bd", "tile_bd", "wbytes", "kswitch", "kfirst", "rr", "rr_pair", "ucs_b",
                "ucs_b2b")
LAYOUT_FEATS = ("in_pack_bytes", "out_scatter_bytes", "relayout", "up", "down")
_GEMM_W = re.compile(r"^gemm_kacc__(\d+)_(\d+)_(\d+)_")

_GFA = re.compile(r"GFA class (\d+) fa_h(\d+)_n(\d+)_m(\d+)_d(\d+) .*launches/site=(\d+)")
_PROF = re.compile(r"^PROF part=(\d+) op=aie_host_(conv|linear|attn)(\d+) ms=([\d.]+) .*site_ms=([\d.]+) aie_ms=([\d.]+)",
                   re.M)
_LAUNCH = re.compile(r"^LAUNCH (?:warm|cold) .*?\blaunches=(\d+)", re.M)


def gfa_classes(build_log):
    cls = {}
    for m in _GFA.finditer(read(build_log)):
        k, h, n, mm, d, L = (int(x) for x in m.groups())
        cls[k] = dict(h=h, n=n, m=mm, d=d, launches=L)
    if not cls:
        raise ValueError("no 'GFA class' lines in " + build_log)
    return cls


def base_sites(build, build_log):
    """kind/cid/launches/kernels/MK/MN per site (graph_driver.c order), no launch or layout features"""
    gemm, fa, fa_launches, _ = parse_aiek(read(os.path.join(build, "aiek_src", "aiek_unetconv.cc")))
    seq = parse_driver(read(os.path.join(build, "ag_work", "arm_build", "graph_driver.c")))
    gfa = gfa_classes(build_log)
    for k, L in enumerate(fa_launches):
        if k not in gfa or gfa[k]["launches"] != L:
            raise ValueError("GFA class %d: build log %s vs g_fa_cls launches=%d" % (k, gfa.get(k), L))
    out = []
    for c in seq:
        ls = launches_of(c, gemm, fa, fa_launches)
        kind = c[0]
        if kind == "conv":
            m = re.search(r"M=(\d+) N=(\d+) K=(\d+)", ls[0][2])
            M, N, K = (int(x) for x in m.groups())
            s = dict(kind="conv", MK=M * K, MN=M * N, Mw=M, Kw=K, Nw=N)
        elif kind == "linear":
            x, _w, y = c[2]
            M, K, N = x[0], x[1], y[1]
            s = dict(kind="linear", MK=M * K, MN=M * N, Mw=M, Kw=K, Nw=N)
        else:
            if ls:
                g = gfa[c[3]]
                s = dict(kind="fa", MK=g["h"] * g["n"] * g["d"], MN=g["h"] * g["n"] * g["m"], cls=c[3],
                         Mw=g["h"] * g["n"], Kw=g["d"], Nw=g["m"])
            else:
                s = dict(kind="attn", MK=0, MN=0, cls=c[3], Mw=0, Kw=0, Nw=0)
        s.update(cid=c[1], launches=len(ls), kernels=sorted({k for k, _s, _d in ls}))
        out.append(s)
    return out


def wbytes(kernel):
    """bf16 weight bytes K*N*2 of one gemm_kacc__N_K_M_... launch, 0 for other kernels"""
    m = _GEMM_W.match(kernel)
    return int(m.group(2)) * int(m.group(1)) * 2 if m else 0


def ucs_line(i, d):
    """console line uc_conv / uc_linear printf+fflush between sp1 and sp2 of launch i (aiek_unetconv.cc
    '[UCS] l=%d M=%d N=%d K=%d tkw=%d' / '[UCS] l=%d lin M=.. N=.. K=.. px=.. s=../.. tkw=%d'), '' for fa launches.
    l = graph_run launch index, tkw = gemm_kacc__ name field 7. Reproduces the board log [UCS] lines exactly."""
    if not _GEMM_W.match(d["kernel"]):
        return ""
    tkw = int(d["kernel"].split("__", 1)[1].split("_")[6])
    desc = re.sub(r"^(conv|lin) cid=\d+ ", lambda m: "" if m.group(1) == "conv" else "lin ", d["desc"])
    return "[UCS] l=%d %s tkw=%d\n" % (i, desc, tkw)


def add_launch_feats(st, seq):
    """sum the per-launch static_feat.unet_site_feat records over each site's launches (in order):
    cp_w/bdcfg_w (+ rr on a kernel-switch launch), shim_bd, tile_bd, wbytes, kswitch, kfirst (first launch of
    that kernel in graph_run), rr, rr_pair (cur's return-route words on shim tiles prev's window does not route),
    ucs_b (console bytes printed inside the launch windows, ucs_line), ucs_b2b (bytes of the previous launch's console
    line when that launch also printed: UART backlog the print may wait on)"""
    i = 0
    tiles, seen = {}, set()
    for d in seq:
        tiles.setdefault(d["kernel"], d["rr_tiles"])
    for s in st:
        f = dict.fromkeys(LAUNCH_FEATS, 0)
        ks = set()
        n = 0
        for d in seq[i:i + s["launches"]]:
            ks.add(d["kernel"])
            f["cp_w"] += d["cp_w"] + d["rr"]
            f["bdcfg_w"] += d["bdcfg_w"] + d["rr"]
            f["shim_bd"] += d["shim_bd"]
            f["tile_bd"] += d["tile_bd"]
            f["wbytes"] += wbytes(d["kernel"])
            f["kswitch"] += d["kswitch"]
            f["kfirst"] += d["kernel"] not in seen
            seen.add(d["kernel"])
            f["rr"] += d["rr"]
            j = i + n
            n += 1
            f["ucs_b"] += len(ucs_line(j, d))
            if j > 0 and ucs_line(j, d) and ucs_line(j - 1, seq[j - 1]):
                f["ucs_b2b"] += len(ucs_line(j - 1, seq[j - 1]))
            if d["kswitch"]:
                prev_t = set(tiles.get(d["prev"], {})) if d["prev"] else set()
                f["rr_pair"] += sum(w for t, w in d["rr_tiles"].items() if t not in prev_t)
        if len(seq[i:i + s["launches"]]) != s["launches"] or sorted(ks) != s["kernels"]:
            raise ValueError("site %s%d: launches/kernels do not match the launch sequence" % (s["kind"], s["cid"]))
        i += s["launches"]
        s.update(f)
    if i != len(seq):
        raise ValueError("%d site launches vs %d in the launch sequence" % (i, len(seq)))
    return st


def sites(build, build_log):
    """base_sites + per-site launch features (static_feat.unet_site_feat) + host layout features
    (n_layout.layout_sites), aligned by (kind, cid) in graph_run order; any mismatch raises"""
    st = add_launch_feats(base_sites(build, build_log), unet_site_feat(build))
    ly = layout_sites(build)
    if len(ly) != len(st):
        raise ValueError("%s: %d n_feat sites vs %d layout sites" % (build, len(st), len(ly)))
    for i, (s, l) in enumerate(zip(st, ly)):
        lk = "attn" if s["kind"] == "fa" else s["kind"]
        if (lk, s["cid"]) != (l["kind"], l["cid"]):
            raise ValueError("site %d: n_feat %s%d vs layout %s%d" % (i, s["kind"], s["cid"], l["kind"], l["cid"]))
        s.update({x: l[x] for x in LAYOUT_FEATS})
        s["lpart"] = l["part"]
    return st


def check(st, log):
    txt = read(log)
    prof = _PROF.findall(txt)
    lm = _LAUNCH.findall(txt)
    log_launches = int(lm[-1]) if lm else -1
    ok = 0
    rows = []
    for i, s in enumerate(st):
        good = False
        if i < len(prof):
            part, op, cid, ms, site_ms, aie_ms = prof[i]
            kind_ok = (op == "conv" and s["kind"] == "conv") or (op == "linear" and s["kind"] == "linear") or \
                      (op == "attn" and s["kind"] in ("fa", "attn"))
            good = kind_ok and int(cid) == s["cid"] and ((s["launches"] > 0) == (float(aie_ms) > 0))
            s.update(part=int(part), ms=float(ms), site_ms=float(site_ms), aie_ms=float(aie_ms))
        if not good:
            print("NFEAT_MISALIGN site=%d %s cid=%d launches=%d prof=%s" %
                  (i, s["kind"], s["cid"], s["launches"], prof[i] if i < len(prof) else None))
        ok += good
        rows.append(s)
    return ok, len(prof), log_launches


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--build-log")
    ap.add_argument("--check-log")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    build = a.build.rstrip("/")
    blog = a.build_log or build + ".build.log"
    st = sites(build, blog)
    per = {"conv": 0, "linear": 0, "fa": 0}
    kern = set()
    for s in st:
        if s["kind"] in per:
            per[s["kind"]] += s["launches"]
        kern.update(s["kernels"])
    n = sum(s["launches"] for s in st)
    passed = n == sum(per.values()) and n > 0
    aligned = "-"
    log_l = "-"
    if a.check_log:
        ok, nprof, log_l = check(st, a.check_log)
        aligned = "%d/%d" % (ok, len(st))
        passed = passed and ok == len(st) and nprof == len(st) and log_l == n
        if nprof != len(st):
            print("NFEAT_MISALIGN prof_sites=%d build_sites=%d" % (nprof, len(st)))
    for i, s in enumerate(st):
        print("NFEAT_SITE i=%d part=%s kind=%s cid=%d launches=%d MK=%d MN=%d ms=%s aie_ms=%s" %
              (i, s.get("part", "-"), s["kind"], s["cid"], s["launches"], s["MK"], s["MN"],
               s.get("ms", "-"), s.get("aie_ms", "-")))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(st, f, indent=1)
    print("NFEAT build=%s conv=%d linear=%d fa=%d kernels=%d launches=%d log_launches=%s aligned=%s %s" %
          (os.path.basename(build), per["conv"], per["linear"], per["fa"], len(kern), n, log_l, aligned,
           "PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
