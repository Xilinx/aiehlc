#!/usr/bin/env python3
"""Static per-site N (host glue) features of an aiegraph CNN build on the cnn_aie_conv (kacc_chunk) path
(yolo26n, aiek_cnnconv; costmodel M4.2 P2). The ap_sites.py counterpart for aiek_cnnconv.cc.

  cc_sites.py --build <root>/yolo26n_<cfg> [--build-log <build>.build.log] [--check-log <board log>]

Sites = the cnn_aie_conv calls of ag_work/arm_build/graph_driver.c:
  cnn_aie_conv(cid, args, nargs, sh, sw, ph, pw, act, dw, np, kk, kp, mc, kch, split, ks);  /* [<part>] group conv<cid>: fn */
part = the board log's 'PROF part=<part> op=aie_host_conv<cid>' line. Geometry M/N/K/launches from the build log
'aie conv <cid> <name> -> call <part> M= N= K= ... dw= act= -> NpxKkxMC split=S launches=L' (net_sites.parse_yolo_log
fields), the kernel instance from aiek_src/aiek_cnnconv.cc uc_kacc_launch, its single-launch BD call counts from
ag_work/aiek/aiek_cnnconv/worklocal/host.cc (static_feat.host_counts / feat_of_counts, as ap_sites). No board number
enters a feature.

Host work of one site (aiek_cnnconv.cc uc_chunk_conv), all static:
  W port once per site: Np*Kp loop iterations (index math + uc_seg bf16 split)          -> wpack_elems
  per launch: X port MC*Kp iterations (im2col bound check + uc_seg)                    -> xpack_elems = L*MC*Kp
              of which loads of real input (k < Kk, kk < K, p < M; no padding cut)    -> xload_elems = M*K*split
              y sentinel memset Np*MC*4 B, flush W (first) + X + Y, invalidate Y       -> sync_bytes
              launch (PH_BDCFG host calls / BD words of the kernel)                     -> glue_bd, glue_w
  scatter of N*M fp32 (+ per-channel bias sum)                                           -> scatter_elems
              SiLU (expf) when act=1                                                     -> silu_elems = N*M*act
              same-shape (residual) adds: fused fn adds after the first (bias)          -> add_elems = N*M*n_full
  kfirst: 1 on the first site of a kernel instance in graph_run order; launches = L = ceil(M/MC)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # python -I

from aiegraph.costmodel.static_feat import feat_of_counts, host_counts  # noqa: E402

CC_FEATS = ("wpack_elems", "xpack_elems", "xload_elems", "sync_bytes", "scatter_elems", "silu_elems", "add_elems",
            "glue_bd", "glue_w", "kfirst", "launches")
SHIM_MD_WORDS = 8
CC_WL = os.path.join("ag_work", "aiek", "aiek_cnnconv", "worklocal")

_CALL = re.compile(r"^\s*cnn_aie_conv\((\d+), args, (\d+), ([-\d, ]+)\);\s*/\* \[(\d+)\] group conv(\d+): (\w+) \*/",
                   re.M)
_GEOM = re.compile(r"aie conv\s+(\d+) \S+ -> call (\d+) M=(\d+) N=(\d+) K=(\d+) s=(\d+) p=(\d+) dw=([01]) "
                   r"act=(\d+) -> (\d+)x(\d+)x(\d+) split=(\d+) launches=(\d+)\s*$", re.M)
_INST = re.compile(r"if \(N == (\d+) && K == (\d+) && M == (\d+)\) \{\s*gemm_kacc<([\d, ]+)><<<", re.S)
_PROF = re.compile(r"^PROF part=(\d+) op=aie_host_conv(\d+) ms=([\d.]+) .*site_ms=([\d.]+) aie_ms=([\d.]+)", re.M)


def read(p):
    with open(p, errors="replace") as f:
        return f.read()


def driver_calls(build):
    """[(cid, nargs, part, fused fn, dict of the int args)] in graph_run order"""
    txt = read(os.path.join(build, "ag_work", "arm_build", "graph_driver.c"))
    names = ("sh", "sw", "ph", "pw", "act", "dw", "np", "kk", "kp", "mc", "kch", "split", "ks")
    out = []
    for cid, nargs, rest, part, cid2, fn in _CALL.findall(txt):
        if cid != cid2:
            continue
        v = [int(x) for x in rest.split(",")]
        if len(v) != len(names):
            raise ValueError("%s: cnn_aie_conv(%s) has %d int args" % (build, cid, len(v)))
        out.append((int(cid), int(nargs), int(part), fn, dict(zip(names, v))))
    if not out:
        raise ValueError("%s: no cnn_aie_conv call sites" % build)
    return out


def geometry(build_log):
    g = {}
    for cid, part, M, N, K, s, p, dw, act, np_, kk, mc, split, L in _GEOM.findall(read(build_log)):
        g[int(cid)] = dict(part=int(part), M=int(M), N=int(N), K=int(K), dw=int(dw), act=int(act), Np=int(np_),
                           Kk=int(kk), MC=int(mc), split=int(split), L=int(L))
    if not g:
        raise ValueError("no 'aie conv <cid> ... -> call' lines in " + build_log)
    return g


def kernels(build):
    txt = read(os.path.join(build, "aiek_src", "aiek_cnnconv.cc"))
    return {(int(n), int(k), int(m)): "gemm_kacc__" + "_".join(x.strip() for x in t.split(","))
            for n, k, m, t in _INST.findall(txt)}


def kernel_feat(build, kname, cache):
    if kname not in cache:
        wl = os.path.join(build, CC_WL)
        host = os.path.join(wl, "host.cc")
        c = host_counts(host, "host_canonicalized_" + kname, cache.setdefault("_txt", read(host)))
        f = feat_of_counts(c, os.path.join(wl, "host_ctrlpkt_%s.h" % kname))
        md = c.get("__Runtime_dma_bd_config_multidim", 0) + c.get("__Runtime_dma_bd_config_multidim_ooo", 0)
        cache[kname] = dict(bdcfg_calls=f["bdcfg_calls"], bdcfg_w=f["bdcfg_w"] + SHIM_MD_WORDS * md)
    return cache[kname]


def sites(build, build_log=None):
    """static rows (graph_run order) of a cnn_aie_conv build: kind, op, cid, part, launches, kernel + CC_FEATS"""
    build = build.rstrip("/")
    geo = geometry(build_log or build + ".build.log")
    inst = kernels(build)
    cache, seen, out = {}, set(), []
    for cid, nargs, part, fn, d in driver_calls(build):
        g = geo[cid]
        M, N, K, Np, Kk, MC, sp = g["M"], g["N"], g["K"], g["Np"], g["Kk"], g["MC"], g["split"]
        if (g["part"], g["dw"], g["act"], Np, Kk, MC, sp) != (part, d["dw"], d["act"], d["np"], d["kk"], d["mc"],
                                                               d["split"]):
            raise ValueError("cid %d: driver args %s vs build log %s" % (cid, d, g))
        L = -(-M // MC)
        if L != g["L"]:
            raise ValueError("cid %d: ceil(M/MC)=%d vs build log launches=%d" % (cid, L, g["L"]))
        Kp = d["kp"]
        ne = nargs - 3
        n_full = max(0, ne - 1) if fn.count("_add") else 0
        kname = inst[(Np, Kk, MC)]
        kf = kernel_feat(build, kname, cache)
        out.append(dict(
            kind="conv", op="conv", cid=cid, part=part, launches=L, kernel=kname, kernels=[kname],
            M=M, N=N, K=K, Np=Np, Kk=Kk, Kp=Kp, MC=MC, split=sp, dw=g["dw"], act=g["act"], n_full=n_full,
            wpack_elems=Np * Kp, xpack_elems=L * MC * Kp, xload_elems=M * K * sp,
            sync_bytes=Np * Kp * 2 + L * (MC * Kp * 2 + 2 * Np * MC * 4),
            scatter_elems=N * M, silu_elems=N * M * g["act"], add_elems=N * M * n_full,
            glue_bd=L * kf["bdcfg_calls"], glue_w=L * kf["bdcfg_w"],
            kfirst=int(kname not in seen)))
        seen.add(kname)
    return sorted(out, key=lambda r: r["part"])


def check(st, log):
    """(aligned sites, PROF aie_host_conv<cid> sites): part, cid order and aie_ms>0 must match the board log"""
    prof = _PROF.findall(read(log))
    ok = sum(1 for s, p in zip(st, prof) if int(p[0]) == s["part"] and int(p[1]) == s["cid"] and float(p[4]) > 0)
    return ok, len(prof)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--build-log")
    ap.add_argument("--check-log")
    a = ap.parse_args(argv)
    st = sites(a.build, a.build_log)
    for s in st:
        print("CCSITE part=%d conv%d kernel=%s %s" % (s["part"], s["cid"], s["kernel"],
                                                     " ".join("%s=%s" % (f, s[f]) for f in CC_FEATS)))
    good = len(st) > 0
    al = "-"
    if a.check_log:
        ok, n = check(st, a.check_log)
        al = "%d/%d" % (ok, n)
        good &= ok == n == len(st)
    print("CCSITES build=%s sites=%d launches=%d kernels=%d aligned=%s %s" % (
        os.path.basename(a.build.rstrip("/")), len(st), sum(s["launches"] for s in st),
        len({s["kernel"] for s in st}), al, "PASS" if good else "FAIL"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
