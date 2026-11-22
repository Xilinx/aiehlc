#!/usr/bin/env python3
"""net_sites -- per-net naming for the frozen K+L+N chain (costmodel M4.1 P0) + board-log site/launch census.

Shared helpers (imported by m3_site / k_diag / k_dmafeat / kln_site / n_fit via set_net):
  NETS                 known nets; DEFAULT_NET = unet (every script's --net default, UNet output unchanged)
  CFGS[net]            the two board configs of the net (unet/resnet18: off aot-kco; yolo26n: off aot)
  cmp_re(net)          '^CMP <net> board=.. aie_launches=.. run_rc=..' regex (the old hard-coded 'CMP unet')
  run_re(net)          '<net>_<cfg>.<run>.log' -> run regex (the old kln_site _RUN 'unet_' regex)
  build_dir(root, net, cfg) = root/<net>_<cfg>
  set_net(net, *mods)  sets mod.NET and mod._CMP (if the module has one) for each module

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
