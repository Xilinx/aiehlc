#!/usr/bin/env python3
"""Static per-site N (host glue) features of an aiegraph CNN build on the ap_gemm path (resnet18; costmodel M4.1 P2).

  ap_sites.py --build <root>/resnet18_<cfg> [--build-log <build>.build.log] [--check-log <board log>]

Sites = the aiednn_conv / aiednn_dense calls of ag_work/arm_build/graph_driver.c in graph_run order; the call's
trailing comment '/* [<part>] group conv<cid>: <fused fn> */' gives the PROF part, which is the order of the board
log's 'PROF part=<part> op=aie_host_conv' lines (ap_gemm prints op=aie_host_conv for conv AND dense sites).
Geometry per cid from the build log '  aie conv|linear <cid> ... M= N= K= s= p= act= inst=<Ng>x<Kg>x<PX>
launches=<L>'; the kernel instance from aiek_src/aiek_cnn.cc uc_kacc_launch (gemm_kacc<Ng, Kg, PX, ...>), its
single-launch BD/startio call counts from ag_work/aiek/aiek_cnn/worklocal/host.cc host_canonicalized_<kernel>
(static_feat.host_counts / feat_of_counts, the L-segment formulas). No board number enters a feature.

Host work of one site (aiek_cnn.cc ap_gemm), all static:
  W re-tile once per site: memset Ng*Kp*2 B + N*K scalar bf16 stores (ap_woff)
  per launch: memset of X PX*Kp*2 B, im2col of mv*K elements (bound check + fp32->bf16, ap_xoff),
              y sentinel (UC_OPT bit0: one word per 4 KiB), flush W+X (uc_sync_pre), launch, invalidate Y,
              scatter of N*mv fp32 with the fused adds (bias per channel; residual = the output's own shape) + ReLU
Features (bytes / counts over the whole site):
  pack_bytes        2*(N*K + M*K)           bf16 bytes stored by the W re-tile + im2col scalar loops
  wpack_bytes       2*N*K                   of which the W re-tile (once per site, ap_woff stores)
  xpack_bytes       2*M*K                   of which the im2col (per launch, bound check + RNE + ap_xoff)
  tail_elems        Ng*Kp + L*PX*Kp - N*K - M*K   zero-filled (memset) elements that no pack loop writes
                                            (N/K/pixel padding, incl. a small K on a larger-K instance)
  scatter_bytes     4*N*M                   fp32 output bytes written by the scatter
  scatter_add_bytes scatter_bytes * n_res   residual operand bytes read by the fused adds (n_res = adds of the
                                            output's shape: fused fn '_add' count - 1 bias; dense: 0)
  sync_bytes        L*(Ng*Kp*2 + PX*Kp*2 + Ng*PX*4)   cache flush (W, X) + invalidate (Y) range bytes
  glue_bd           L * bdcfg_calls(kernel)  PH_BDCFG host calls in the launch (tile/shim BD configs, windows)
  glue_w            L * (bdcfg_w + 8*shim_md)(kernel)  host-written BD words (n_feat SHIM_MD_WORDS rule)
  kfirst            1 on the first site of a kernel instance in graph_run order (cold kernel load + plan)
  launches          L = ceil(M / PX)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python -I

from costmodel.static_feat import feat_of_counts, host_counts  # noqa: E402

AP_FEATS = ("pack_bytes", "wpack_bytes", "xpack_bytes", "tail_elems", "scatter_bytes", "scatter_add_bytes",
            "sync_bytes", "glue_bd", "glue_w", "kfirst", "launches")
SHIM_MD_WORDS = 8
AP_WL = os.path.join("ag_work", "aiek", "aiek_cnn", "worklocal")

_CALL = re.compile(r"^\s*aiednn_(conv|dense)\((\d+), args, (\d+), \(const uint16_t \*\)\(g_params \+ \d+\), "
                   r"([-\d, ]+)\);\s*/\* \[(\d+)\] group (?:conv|lin)(\d+): (\w+) \*/", re.M)
_GEOM = re.compile(r"^\[build_\w+\]\s+aie (conv|linear)\s+(\d+) \S+ -> (\w+) M=(\d+) N=(\d+) K=(\d+) s=(\d+) "
                   r"p=(\d+) act=(\d+) inst=(\d+)x(\d+)x(\d+) launches=(\d+)", re.M)
_INST = re.compile(r"if \(N == (\d+) && K == (\d+) && M == (\d+)\) \{\s*gemm_kacc<([\d, ]+)><<<", re.S)
_PROF = re.compile(r"^PROF part=(\d+) op=aie_host_conv ms=([\d.]+) .*site_ms=([\d.]+) aie_ms=([\d.]+)", re.M)


def read(p):
    with open(p, errors="replace") as f:
        return f.read()


def ap_geom(N, K, M):
    """aiek_cnn.cc ap_geom (kacc_geom of aiegraph.cnn): (kch, tr, Kp) of instance (Ng, Kg, PX)"""
    tkw = 4096 // M if 4096 // M < 32 else 32
    tr = 2 * (16 * tkw * 4 + (M // 4) * tkw * 4 + 16 * (M // 4) * 4) > 29696
    if tr:
        tkw = 32
        while (N // 4) * tkw > 1024 or (K // 2) % tkw:
            tkw //= 2
    if tkw < 4 or (K // 2) % tkw:
        raise ValueError("ap_geom(%d,%d,%d): no K chunk" % (N, K, M))
    kch = 2 * tkw
    return kch, int(tr), (K + kch if N == K // 2 else K)


def driver_calls(build):
    """[(kind conv|dense, cid, nargs, part, fused fn)] in graph_run order"""
    txt = read(os.path.join(build, "ag_work", "arm_build", "graph_driver.c"))
    out = [(k, int(c), int(n), int(p), fn) for k, c, n, _a, p, cid2, fn in _CALL.findall(txt) if c == cid2]
    if not out:
        raise ValueError("%s: no aiednn_conv/aiednn_dense call sites" % build)
    return out


def geometry(build_log):
    g = {}
    for kind, cid, fn, M, N, K, s, p, act, ng, kg, px, L in _GEOM.findall(read(build_log)):
        g[int(cid)] = dict(gkind=kind, fn=fn, M=int(M), N=int(N), K=int(K), Ng=int(ng), Kg=int(kg), PX=int(px),
                           L=int(L))
    if not g:
        raise ValueError("no '  aie conv|linear <cid>' lines in " + build_log)
    return g


def kernels(build):
    """{(Ng, Kg, PX): kernel name gemm_kacc__<template args>} from uc_kacc_launch"""
    txt = read(os.path.join(build, "aiek_src", "aiek_cnn.cc"))
    return {(int(n), int(k), int(m)): "gemm_kacc__" + "_".join(x.strip() for x in t.split(","))
            for n, k, m, t in _INST.findall(txt)}


def kernel_feat(build, kname, cache):
    if kname not in cache:
        wl = os.path.join(build, AP_WL)
        host = os.path.join(wl, "host.cc")
        c = host_counts(host, "host_canonicalized_" + kname, cache.setdefault("_txt", read(host)))
        f = feat_of_counts(c, os.path.join(wl, "host_ctrlpkt_%s.h" % kname))
        md = c.get("__Runtime_dma_bd_config_multidim", 0) + c.get("__Runtime_dma_bd_config_multidim_ooo", 0)
        cache[kname] = dict(bdcfg_calls=f["bdcfg_calls"], bdcfg_w=f["bdcfg_w"] + SHIM_MD_WORDS * md)
    return cache[kname]


def sites(build, build_log=None):
    """static rows (graph_run order) of an ap_gemm build: kind, cid, part, launches, kernel + AP_FEATS"""
    build = build.rstrip("/")
    geo = geometry(build_log or build + ".build.log")
    inst = kernels(build)
    cache, seen, out = {}, set(), []
    for kind, cid, nargs, part, fn in driver_calls(build):
        g = geo[cid]
        if g["fn"] != fn:
            raise ValueError("cid %d: driver fn %s vs build log %s" % (cid, fn, g["fn"]))
        M, N, K, Ng, Kg, PX = g["M"], g["N"], g["K"], g["Ng"], g["Kg"], g["PX"]
        L = -(-M // PX)
        if L != g["L"]:
            raise ValueError("cid %d: ceil(M/PX)=%d vs build log launches=%d" % (cid, L, g["L"]))
        n_add = fn.count("_add")
        if nargs != 3 + n_add:
            raise ValueError("cid %d: nargs=%d vs %d fused adds in %s" % (cid, nargs, n_add, fn))
        n_res = max(0, n_add - 1)
        kch, tr, Kp = ap_geom(Ng, Kg, PX)
        kname = inst[(Ng, Kg, PX)]
        kf = kernel_feat(build, kname, cache)
        sb = 4 * N * M
        out.append(dict(
            kind="conv", op=kind, cid=cid, part=part, launches=L, kernel=kname, kernels=[kname],
            M=M, N=N, K=K, Ng=Ng, Kg=Kg, PX=PX, Kp=Kp, tr=tr, n_res=n_res,
            pack_bytes=2 * (N * K + M * K), wpack_bytes=2 * N * K, xpack_bytes=2 * M * K,
            tail_elems=Ng * Kp + L * PX * Kp - N * K - M * K,
            scatter_bytes=sb, scatter_add_bytes=sb * n_res,
            sync_bytes=L * (Ng * Kp * 2 + PX * Kp * 2 + Ng * PX * 4),
            glue_bd=L * kf["bdcfg_calls"], glue_w=L * kf["bdcfg_w"],
            kfirst=int(kname not in seen)))
        seen.add(kname)
    return out


def check(st, log):
    """(aligned sites, PROF aie_host_conv sites): part order and launches>0 must match the board log"""
    prof = _PROF.findall(read(log))
    ok = sum(1 for s, p in zip(st, prof) if int(p[0]) == s["part"] and float(p[3]) > 0)
    return ok, len(prof)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--build-log")
    ap.add_argument("--check-log")
    a = ap.parse_args(argv)
    st = sites(a.build, a.build_log)
    for s in st:
        print("APSITE part=%d %s%d kernel=%s %s" % (s["part"], s["op"], s["cid"], s["kernel"],
                                                   " ".join("%s=%s" % (f, s[f]) for f in AP_FEATS)))
    good = len(st) > 0
    al = "-"
    if a.check_log:
        ok, n = check(st, a.check_log)
        al = "%d/%d" % (ok, n)
        good &= ok == n == len(st)
    print("APSITES build=%s sites=%d launches=%d kernels=%d aligned=%s %s" % (
        os.path.basename(a.build.rstrip("/")), len(st), sum(s["launches"] for s in st),
        len({s["kernel"] for s in st}), al, "PASS" if good else "FAIL"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
