#!/usr/bin/env python3
"""net_sites -- per-net naming for the frozen K+L+N chain (costmodel M4.1 P0) + board-log site/launch census.

Shared helpers (imported by m3_site / k_diag / k_dmafeat / kln_site / n_fit via set_net):
  NETS                 known nets; DEFAULT_NET = unet (every script's --net default, UNet output unchanged)
  CFGS[net]            the two board configs of the net (unet/resnet18: off aot-kco; yolo26n: off aot)
  cmp_re(net)          '^CMP <net> board=.. aie_launches=.. run_rc=..' regex (the old hard-coded 'CMP unet')
  run_re(net)          '<net>_<cfg>.<run>.log' -> run regex (the old kln_site _RUN 'unet_' regex)
  build_dir(root, net, cfg) = root/<net>_<cfg>
  set_net(net, *mods)  sets mod.NET and mod._CMP (if the module has one) for each module
  aiek_dir(net)        AIE kernel family dir under ag_work/aiek: resnet18/unet -> aiek_cnn, yolo26n -> aiek_cnnconv
  kpar_dir(root, net, cfg) = build_dir/ag_work/aiek/<aiek_dir>/worklocal/build/kpar
  sites(root, net, cfg)    static AIE site table [(part, kind, cid, (A,B,C), launches)] sorted by part (M4.2 P0):
      resnet18/unet: build log 'aie conv|linear <cid> ... inst=AxBxC launches=n' joined with graph_driver.c
                     'GRAPH_PROF_END(<part>, "aie aiednn_<conv|dense> <cid>")' (= kcore_net.sites at d8144cdb);
      yolo26n:       build log 'aie conv <cid> <name> -> call <part> M= N= K= ... dw= ... -> AxBxC split=3 launches=n'
                     (dw=0 and dw=1 sites alike; part = call number = board 'PROF part=<part> op=aie_host_conv<cid>').
  inst_dir(kpar, inst)     the one kpar/gemm_kacc__A_B_C_* dir of an instance (ValueError if not exactly one)

CLI (census, reads only the board log):
  net_sites.py [--net NET] LOG...      one line per log, sum_ok last:
    NETSITES net= cfg= run= board= launches= sites= sum_ok=
  launches = CMP aie_launches; sites = 'PROF aie_host_split' lines (one per AIE site);
  sum_ok=1 iff the CMP line is for NET with run_rc=0, every site has one 'PROF part=<p> op=aie_host_* site_ms=' line,
  and the sum of the sites' launches= equals aie_launches.
  rc 0 all logs sum_ok=1, 1 otherwise, 2 bad args.
  net_sites.py --selftest
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

NETS = ("unet", "resnet18", "yolo26n")
DEFAULT_NET = "unet"
CFGS = {"unet": ("off", "aot-kco"), "resnet18": ("off", "aot-kco"), "yolo26n": ("off", "aot")}

_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) .*\blaunches=(\d+)\s*$", re.M)
_SITE = re.compile(r"^PROF part=(\d+) op=aie_host_\w+ ms=[\d.]+ .*site_ms=[\d.]+", re.M)


def check_net(net):
    if net not in NETS:
        raise ValueError("unknown net %r (known: %s)" % (net, " ".join(NETS)))
    return net


def cmp_re(net):
    return re.compile(r"^CMP %s board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+)" % re.escape(check_net(net)), re.M)


def run_re(net):
    return re.compile(r"^(?:.*/)?(?:%s_[\w-]+\.)?([a-z])(?:\.log)?$" % re.escape(check_net(net)))


def build_dir(root, net, cfg):
    return str(Path(root) / ("%s_%s" % (check_net(net), cfg)))


def set_net(net, *mods):
    check_net(net)
    for m in mods:
        m.NET = net
        if hasattr(m, "_CMP"):
            m._CMP = cmp_re(net)
    return net


AIEK = {"unet": "aiek_cnn", "resnet18": "aiek_cnn", "yolo26n": "aiek_cnnconv"}
_BL = re.compile(r"aie (conv|linear)\s+(\d+) .*?\binst=(\d+)x(\d+)x(\d+) launches=(\d+)")
_GD = re.compile(r'GRAPH_PROF_END\((\d+), "aie aiednn_(conv|dense) (\d+)"\)')
_YBL = re.compile(r"aie conv\s+(\d+) (\S+) -> call (\d+) M=(\d+) N=(\d+) K=(\d+) .*?\bdw=([01]) .*?"
                  r"-> (\d+)x(\d+)x(\d+) split=\d+ launches=(\d+)\s*$", re.M)


def aiek_dir(net):
    return AIEK[check_net(net)]


def kpar_dir(root, net, cfg):
    return os.path.join(build_dir(root, net, cfg), "ag_work", "aiek", aiek_dir(net), "worklocal", "build", "kpar")


def inst_dir(kpar, inst):
    g = [x for x in sorted(glob.glob(os.path.join(kpar, "gemm_kacc__%d_%d_%d_*" % tuple(inst)))) if os.path.isdir(x)]
    if len(g) != 1:
        raise ValueError("%s: %d dirs for instance %s" % (kpar, len(g), inst))
    return g[0]


def parse_yolo_log(txt):
    """yolo build-log text -> [(part, 'conv', cid, (A,B,C), launches, dw)]"""
    out = []
    for m in _YBL.finditer(txt):
        out.append((int(m.group(3)), "conv", int(m.group(1)),
                    (int(m.group(8)), int(m.group(9)), int(m.group(10))), int(m.group(11)), int(m.group(7))))
    return out


def sites(build_root, net, cfg, with_dw=False):
    """[(part, kind, cid, (A,B,C), launches)] in part order ((..., dw) for yolo26n if with_dw)"""
    bdir = build_dir(build_root, net, cfg)
    txt = open(bdir + ".build.log", errors="replace").read()
    if net == "yolo26n":
        out = parse_yolo_log(txt)
        if len(set(x[0] for x in out)) != len(out):
            raise ValueError("%s: duplicate call/part numbers" % bdir)
        return sorted(out) if with_dw else sorted(x[:5] for x in out)
    bl = {}
    for m in _BL.finditer(txt):
        kind = "conv" if m.group(1) == "conv" else "dense"
        bl[(kind, int(m.group(2)))] = ((int(m.group(3)), int(m.group(4)), int(m.group(5))), int(m.group(6)))
    gd = open(os.path.join(bdir, "ag_work", "arm_build", "graph_driver.c"), errors="replace").read()
    out = []
    for m in _GD.finditer(gd):
        part, kind, cid = int(m.group(1)), m.group(2), int(m.group(3))
        inst, ln = bl[(kind, cid)]
        out.append((part, kind, cid, inst, ln))
    return sorted(out)


_NAME = re.compile(r"^(?P<net>[a-z0-9]+)_(?P<cfg>[\w-]+)\.(?P<run>[a-z])\.log$")


def census(path, net):
    """-> dict(net cfg run board launches sites sum_ok) of one board log"""
    p = Path(path)
    m = _NAME.match(p.name)
    cfg, run = (m.group("cfg"), m.group("run")) if m and m.group("net") == net else ("?", "?")
    txt = p.read_text(errors="replace") if p.is_file() else ""
    c = cmp_re(net).findall(txt)
    board, launches, rc = (c[-1][0], int(c[-1][1]), int(c[-1][2])) if len(c) == 1 else ("?", -1, -1)
    split = _SPLIT.findall(txt)
    parts = [int(x[0]) for x in split]
    site_parts = [int(x) for x in _SITE.findall(txt)]
    lsum = sum(int(x[2]) for x in split)
    ok = (len(c) == 1 and rc == 0 and cfg != "?" and len(split) > 0 and len(set(parts)) == len(parts)
          and sorted(parts) == sorted(site_parts) and lsum == launches)
    return dict(net=net, cfg=cfg, run=run, board=board, launches=launches, sites=len(split),
                lsum=lsum, sum_ok=int(ok))


def line(d):
    return ("NETSITES net=%(net)s cfg=%(cfg)s run=%(run)s board=%(board)s launches=%(launches)d "
            "sites=%(sites)d sum_ok=%(sum_ok)d" % d)


def selftest():
    import tempfile
    n = 0
    bad = 0

    def chk(name, cond):
        nonlocal n, bad
        n += 1
        bad += 0 if cond else 1
        print("NETSITES_SELFTEST %s %s" % (name, "ok" if cond else "FAIL"))

    chk("run_unet", run_re("unet").match("/x/unet_aot-kco.b.log").group(1) == "b")
    chk("run_r18", run_re("resnet18").match("resnet18_off.c.log").group(1) == "c")
    chk("run_wrong_net", run_re("unet").match("resnet18_off.c.log") is None)
    chk("cmp", cmp_re("resnet18").findall("CMP resnet18 board=b1 x aie_launches=7 y run_rc=0 PASS\n")
        == [("b1", "7", "0")])
    chk("cmp_other_net", cmp_re("unet").findall("CMP resnet18 board=b1 x aie_launches=7 y run_rc=0\n") == [])
    chk("build_dir", build_dir("/r", "yolo26n", "aot") == "/r/yolo26n_aot")
    chk("aiek_dir", (aiek_dir("resnet18"), aiek_dir("unet"), aiek_dir("yolo26n")) ==
        ("aiek_cnn", "aiek_cnn", "aiek_cnnconv"))
    yl = ("[build_yolov26]   aie conv  35 /model.10/m/m.0/attn/pe/conv/Conv -> call 59 M=400 N=128 K=1152 s=1 p=1 "
          "dw=1 act=0 -> 128x3456x256 split=3 launches=2\n"
          "[build_yolov26]   aie conv   0 /model.0/conv/Conv -> call 1 M=102400 N=16 K=27 s=2 p=1 dw=0 act=1 "
          "-> 64x192x256 split=3 launches=400\n")
    chk("yolo_parse", parse_yolo_log(yl) == [(59, "conv", 35, (128, 3456, 256), 2, 1),
                                           (1, "conv", 0, (64, 192, 256), 400, 0)])
    try:
        check_net("vgg")
        chk("unknown_net", False)
    except ValueError:
        chk("unknown_net", True)
    body = ("PROF part=1 op=aie_host_conv ms=1.0 bf16_expand_ms=0 site_ms=2.0 aie_ms=1\n"
            "PROF aie_host_split part=1 kind=conv pack_ms=1 dma_ms=1 launches=3\n"
            "PROF part=2 op=aie_host_linear ms=1.0 bf16_expand_ms=0 site_ms=2.0 aie_ms=1\n"
            "PROF aie_host_split part=2 kind=linear pack_ms=1 dma_ms=1 launches=4\n")
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "resnet18_off.a.log"
        good.write_text(body + "CMP resnet18 board=palmyra213 x aie_launches=7 y run_rc=0 PASS\n")
        badl = Path(d) / "resnet18_aot-kco.b.log"
        badl.write_text(body + "CMP resnet18 board=palmyra213 x aie_launches=8 y run_rc=0 PASS\n")
        g, b = census(good, "resnet18"), census(badl, "resnet18")
        chk("census_ok", line(g) == "NETSITES net=resnet18 cfg=off run=a board=palmyra213 launches=7 sites=2 sum_ok=1")
        chk("census_sum_bad", b["sum_ok"] == 0 and b["cfg"] == "aot-kco" and b["run"] == "b")
        chk("census_net_bad", census(good, "unet")["sum_ok"] == 0)
    print("NETSITES_SELFTEST n=%d bad=%d %s" % (n, bad, "PASS" if bad == 0 else "FAIL"))
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=DEFAULT_NET, choices=NETS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("logs", nargs="*")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    if not a.logs:
        ap.error("no logs")
    ok = True
    for p in a.logs:
        d = census(p, a.net)
        print(line(d))
        ok = ok and d["sum_ok"] == 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
