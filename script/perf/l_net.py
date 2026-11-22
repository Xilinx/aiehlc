#!/usr/bin/env python3
"""Frozen L on a cnn net (resnet18, M4.1 P1): static L calls per launch from the build only, checked against the
board's LAUNCH split_detail, and the frozen L params (no refit) against the board L per launch.

  l_net.py --net resnet18 --params script/perf/params/l_palmyra213.m3121.json --build-root R CFG=LOG[,LOG..] ...

Build R/<net>_<cfg> (read only):
  ag_work/arm_build/graph_driver.c   aiednn_conv / aiednn_dense calls in order: part [n], cid, px, kg
  ag_work/network.md                 Layers table row n: out shape -> N (channels) and M (pixels / rows)
  ag_work/aiek/aiek_cnn/worklocal/host.cc (+ host_ctrlpkt_<k>.h)  per-instance host function
  aiek_src/aiek_cnn.cc               ceil(M / PX) launches per call on instance gemm_kacc__<ceil64(N)>_<kg>_<px>_*
Static calls per launch (static_feat P1 formulas on host_canonicalized_<k>): bdcfg = PH_BDCFG phases, startio,
kload = coreen = 1, core_reset = rst_static (2 n_core + 1 MMIO, 0 ctrl; first load has no DMA reset -> -1).
kload_w: kload_static per-tile words replayed in launch order (PM_WARM_SKIP resident ELF per tile).
Board side (comparison only): split_detail calls, split (kload, coreen), split_detail rt_bdcfg / rt_startio.
  L_meas = kload (ELF + reset, the reset is timed inside kload) + rt_bdcfg + rt_startio + coreen   (ms / launch)
  L_hat  = launch_l.predict_l(static features, cfg, frozen params) with core_reset = 0 (the reset is inside
           kload; the params' rst term carries the double-counted reset of the fit target, m3_site convention)
Prints:
  LSTATIC_CALLS net cfg run k=static/board ...          per log
  LSTATIC net=<net> cfg=<cfg> calls_err=<sum |static-board| over calls x logs>
  LSITE net cfg run part kind launches kernel L_hat_ms                                 (per site, frozen L)
  LCHK net=<net> cfg=<cfg> run=<r> board=<b> launches=<n> lsum_rel=<|L_hat-L_meas|/L_meas %.4f> PASS|FAIL
  LNET net=<net> lstatic=<cfgs with calls_err 0> lchk_pass=<n> logs=<n>
"""
import argparse
import hashlib
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from aiegraph.costmodel import kload_static as ks  # noqa: E402
from aiegraph.costmodel import launch_l  # noqa: E402
from aiegraph.costmodel import static_feat as sf  # noqa: E402
from aiegraph.costmodel.kswitch import parse_host  # noqa: E402
from aiegraph.costmodel.launch_split import lines, parse  # noqa: E402
from aiegraph.costmodel.rst_static import AIE_TILE_ROW_START, rst_static  # noqa: E402

sys.path.insert(0, str(HERE))
import net_sites  # noqa: E402

LSUM_TOL = 0.20
CALLS = ("kload", "bdcfg", "startio", "coreen", "core_reset")
NET = "resnet18"


def wl(net=None):
    """kernel worklocal dir under the build: ag_work/aiek/<net_sites.aiek_dir(net)>/worklocal"""
    return os.path.join("ag_work", "aiek", net_sites.aiek_dir(net or NET), "worklocal")
_CALL = re.compile(r"\baiednn_(conv|dense)\((\d+), args, \d+, [^;]*?, (\d+), (\d+)\);\s*/\* \[(\d+)\] group")
_ROW = re.compile(r"^\| (\d+) \|.*\|\s*([^|]*)\|[^|]*\|\s*$")
_SHAPE = re.compile(r"\[([0-9,]+)\]`")


def sites_yolo(build):
    """yolo26n: net_sites.sites (build log 'aie conv <cid> .. -> call <part> .. -> AxBxC .. launches=n'), kernel =
    the one host_canonicalized_gemm_kacc__A_B_C_* in host.cc"""
    host = open(os.path.join(build, wl(), "host.cc")).read()
    kern = sorted(set(re.findall(r"void host_canonicalized_(gemm_kacc__\w+)\(", host)))
    root, base = os.path.split(os.path.normpath(build))
    cfg = base[len(NET) + 1:]
    out = []
    for part, kind, cid, inst, n in net_sites.sites(root, NET, cfg):
        ks_ = [k for k in kern if k.startswith("gemm_kacc__%d_%d_%d_" % inst)]
        if len(ks_) != 1:
            raise ValueError("part %d: instance %s matches %s" % (part, inst, ks_))
        out.append(dict(part=part, kind=kind, cid=cid, px=inst[2], kg=inst[1], N=None, M=None, launches=n,
                        kernel=ks_[0]))
    if not out:
        raise ValueError("no aie conv site in %s.build.log" % build)
    return out


def sites(build):
    """[dict(part, kind, cid, px, kg, N, M, launches, kernel)] in graph order, build products only"""
    if NET == "yolo26n":
        return sites_yolo(build)
    drv = open(os.path.join(build, "ag_work", "arm_build", "graph_driver.c")).read()
    outs = {}
    for ln in open(os.path.join(build, "ag_work", "network.md")):
        m = _ROW.match(ln.rstrip("\n"))
        if m:
            sh = _SHAPE.findall(m.group(2))
            if sh:
                outs[int(m.group(1))] = [int(x) for x in sh[-1].split(",")]
    host = open(os.path.join(build, wl(), "host.cc")).read()
    kern = sorted(set(re.findall(r"void host_canonicalized_(gemm_kacc__\w+)\(", host)))
    out = []
    for kind, cid, px, kg, part in _CALL.findall(drv):
        px, kg, part = int(px), int(kg), int(part)
        sh = outs[part]
        if kind == "conv":   # NCHWc [B, C/c, H, W, c]
            N, M = sh[1] * sh[4], sh[0] * sh[2] * sh[3]
        else:                # dense [B, N]
            N, M = sh[1], sh[0]
        ng = -(-N // 64) * 64
        ks_ = [k for k in kern if k.startswith("gemm_kacc__%d_%d_%d_" % (ng, kg, px))]
        if len(ks_) != 1:
            raise ValueError("part %d: instance %d_%d_%d matches %s" % (part, ng, kg, px, ks_))
        out.append(dict(part=part, kind=kind, cid=int(cid), px=px, kg=kg, N=N, M=M,
                        launches=-(-M // px), kernel=ks_[0]))
    if not out:
        raise ValueError("no aiednn_conv/dense call in %s" % build)
    return out


def launch_feats(build):
    """per-launch static dicts in launch order + per-cfg rst info"""
    b = Path(build)
    st = sites(build)
    wl_ = os.path.join(build, wl())
    host = os.path.join(wl_, "host.cc")
    txt = open(host).read()
    per = {}
    for s in st:
        k = s["kernel"]
        if k not in per:
            c = sf.host_counts(host, "host_canonicalized_" + k, txt)
            f = sf.feat_of_counts(c, os.path.join(wl_, "host_ctrlpkt_%s.h" % k))
            f["ctrl"] = bool(c.get("__Runtime_load_kernel_group_16t_ctrl", 0) or c.get("__Runtime_load_kernel_group_8t_ctrl", 0)
                             or c.get("__Runtime_load_kernel_group_4t_ctrl", 0))
            per[k] = f
    rs = rst_static(b)
    defs = ks.build_defs(b)
    warm = bool(defs.get("PM_WARM_SKIP"))
    elf, tiles = parse_host(txt)
    objdir = b / wl() / "build"
    cache, res = {}, {}
    seq = []
    i = 0
    for s in st:
        k = s["kernel"]
        f = per[k]
        cores = [t for t in tiles[k] if t[1] >= AIE_TILE_ROW_START]
        for _ in range(s["launches"]):
            if f["ctrl"]:
                kw, rst = float(ks.CTRL_KLOAD_W), 0.0
            else:
                blob = elf[k]
                if blob not in cache:
                    obj = objdir / (blob[len("_binary_"):-len("_start")] + ".o")
                    cache[blob] = ks.tile_words(ks.elf_segments(ks.elf_from_obj(obj, blob)), defs)
                pm, other = cache[blob]
                kw = 0
                for t in cores:
                    cold = not (warm and res.get(t) == blob)
                    res[t] = blob
                    kw += (ks.DMA_RESET_W if i else 0) + ks.CORE_CTRL_W + other + (pm if cold else 0)
                rst = 2.0 * len(cores) + (1.0 if i else 0.0) if rs["rst"] else 0.0
            seq.append(dict(part=s["part"], kind=s["kind"], kernel=k, bdcfg=f["bdcfg_calls"],
                            startio=f["startio_calls"], bdcfg_w=f["bdcfg_w"], cp_w=f["cp_w"], kload_w=float(kw),
                            core_reset=rst))
            i += 1
    return st, seq, rs


def board(log):
    p = parse(log)
    sp, det = p["split"], p["split_detail"]
    n = bname = None
    for ln in lines(log):
        m = re.match(r"LAUNCH (?:warm|cold) .*?\blaunches=(\d+)", ln)
        if m:
            n = int(m.group(1))
        m = re.match(r"CMP \S+ board=(\S+)", ln)
        if m:
            bname = m.group(1)
    l_meas = sp["kload"] + sp["coreen"] + det["rt"]["bdcfg"] + det["rt"]["startio"]
    return dict(calls=det["calls"], n=n, board=bname, l_meas=l_meas)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="resnet18")
    ap.add_argument("--params", required=True)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("runs", nargs="+", help="CFG=LOG[,LOG...]")
    a = ap.parse_args(argv)
    global NET
    NET = net_sites.check_net(a.net)
    prm = launch_l.load_params(a.params)
    md5 = hashlib.md5(open(a.params, "rb").read()).hexdigest()
    print("LPARAMS file=%s md5=%s rst_off=%s rst_aot-kco=%s" % (a.params, md5, prm["cfg"]["off"]["coef"].get("rst"),
                                                                prm["cfg"].get("aot-kco", {}).get("coef", {}).get("rst")))
    nstat = npass = nlog = 0
    for arg in a.runs:
        cfg, logs = arg.split("=", 1)
        build = os.path.join(a.build_root, "%s_%s" % (a.net, cfg))
        st, seq, rs = launch_feats(build)
        n = len(seq)
        stat = {"kload": float(n), "coreen": float(n), "bdcfg": sum(d["bdcfg"] for d in seq),
                "startio": sum(d["startio"] for d in seq), "core_reset": sum(d["core_reset"] for d in seq)}
        print("LSTATIC_SRC net=%s cfg=%s build=%s sites=%d launches=%d kernels=%d rst=%s"
              % (a.net, cfg, build, len(st), n, len({d["kernel"] for d in seq}), rs["why"]))
        # frozen L per launch (core_reset = 0, see doc)
        lh = []
        for d in seq:
            calls = {"bdcfg": d["bdcfg"], "startio": d["startio"], "core_reset": 0.0}
            words = {"bdcfg_w": d["bdcfg_w"], "cp_w": d["cp_w"], "kload_w": d["kload_w"]}
            lh.append(launch_l.predict_l(calls, words, cfg, prm))
        l_hat = sum(lh) / n
        err = 0.0
        runs = []
        for log in logs.split(","):
            bd = board(log)
            run = re.search(r"\.([a-z])\.log$", log).group(1)
            e = sum(abs(stat[c] - float(bd["calls"].get(c, 0))) for c in CALLS)
            err += e
            print("LSTATIC_CALLS net=%s cfg=%s run=%s %s err=%g" % (a.net, cfg, run, " ".join(
                "%s=%g/%g" % (c, stat[c], float(bd["calls"].get(c, 0))) for c in CALLS), e))
            runs.append((run, bd))
        print("LSTATIC net=%s cfg=%s calls_err=%g" % (a.net, cfg, err))
        nstat += err == 0
        i = 0
        for s in st:
            print("LSITE net=%s cfg=%s part=%d kind=%s launches=%d kernel=%s L_hat_ms=%.4f"
                  % (a.net, cfg, s["part"], s["kind"], s["launches"], s["kernel"], sum(lh[i:i + s["launches"]])))
            i += s["launches"]
        for run, bd in runs:
            rel = abs(l_hat - bd["l_meas"]) / bd["l_meas"]
            ok = bd["n"] == n and rel <= LSUM_TOL
            print("LMEAS net=%s cfg=%s run=%s L_meas_ms=%.4f L_hat_ms=%.4f" % (a.net, cfg, run, bd["l_meas"], l_hat))
            print("LCHK net=%s cfg=%s run=%s board=%s launches=%s lsum_rel=%.4f %s"
                  % (a.net, cfg, run, bd["board"], bd["n"], rel, "PASS" if ok else "FAIL"))
            npass += ok
            nlog += 1
    print("LNET net=%s lstatic=%d lchk_pass=%d logs=%d params_md5=%s" % (a.net, nstat, npass, nlog, md5))
    return 0 if nstat == len(a.runs) and npass == nlog else 1


if __name__ == "__main__":
    sys.exit(main())
