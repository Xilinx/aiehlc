#!/usr/bin/env python3
"""M6 P2: a per-build cfg decision made ONLY by aiegraph.costmodel.estimate() on build dirs, for unet / resnet18 /
yolo26n, scored against what palmyra213 measured.

  python3 costmodel_cfg_decision.py --all [--tie 0.03] [--alt] [--selftest]
          [--unet-build ROOT] [--net-build ROOT] [--net unet|resnet18|yolo26n ...]

Decision per net: build the whole net with cfg a (off, MMIO) or cfg b (aot-kco; yolo26n: aot).  aot / aot-kco need a
free west column (Maxwell r1), so the decision is per whole build, not per site.  estimate({"net", "build", "cfg"})
prices every launches>0 site of both builds (build dirs only); pick = the cfg with the smaller predicted total.
The board logs are read ONLY for scoring (FIXED log list, one log per (net, cfg), no directory scan):
  default  unet     shuhua M34 unet_{off,aot-kco}.a.log
           resnet18 shuhua M41 resnet18_{off,aot-kco}.d.log
           yolo26n  shuhua M42 yolo26n_{off,aot}.d.log
  --alt    unet *.b.log, resnet18 / yolo26n *.e.log
meas_<cfg>_ms = sum of site_ms over the launches>0 sites of the log (same site set the model prices, = M6.1 demo).
agree = pick == meas_pick; tie = measured |a-b|/mean < --tie.  A net that neither agrees nor ties -> FAIL.
cmp_ok counts '^CMP <net> board=palmyra213 .* aie_launches=<258|147|1770> .* run_rc=0' lines (6 expected).
--selftest swaps the measured values of one net whose cfgs are not tied: it must FAIL (rc=1).

Output:
  M6DEC net=<n> aie_launches=<L> pred_<a>_ms=X pred_<b>_ms=X meas_<a>_ms=X meas_<b>_ms=X pick=<cfg>
        meas_pick=<cfg> agree|tie|FAIL
  M6DEC ALL nets=<n> board=<B> cmp_ok=<n> agree_or_tie=<n> PASS|FAIL
rc=1 whenever FAIL is printed, rc=2 on missing / inconsistent logs (no CMP line, boards differ).
"""
import argparse
import re
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "src"))

from aiegraph.costmodel import estimate  # noqa: E402

SH = Path("/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks")
NETS = {  # net: (cfg a, cfg b, aie_launches, {run: (log a, log b)})
    "unet": ("off", "aot-kco", 258, {
        "main": (SH / "2026-11-22_M34/unet_off.a.log", SH / "2026-11-22_M34/unet_aot-kco.a.log"),
        "alt": (SH / "2026-11-22_M34/unet_off.b.log", SH / "2026-11-22_M34/unet_aot-kco.b.log")}),
    "resnet18": ("off", "aot-kco", 147, {
        "main": (SH / "2026-11-23_M41/resnet18_off.d.log", SH / "2026-11-23_M41/resnet18_aot-kco.d.log"),
        "alt": (SH / "2026-11-23_M41/resnet18_off.e.log", SH / "2026-11-23_M41/resnet18_aot-kco.e.log")}),
    "yolo26n": ("off", "aot", 1770, {
        "main": (SH / "2026-11-23_M42/yolo26n_off.d.log", SH / "2026-11-23_M42/yolo26n_aot.d.log"),
        "alt": (SH / "2026-11-23_M42/yolo26n_off.e.log", SH / "2026-11-23_M42/yolo26n_aot.e.log")}),
}
UNET_BUILD = "/scratch/staff/huaj/tmp/costmodel_m31223"   # unet_<cfg>/ (M6.1 / M34 PZ builds)
NET_BUILD = "/scratch/staff/huaj/tmp/costmodel_m40"       # resnet18_<cfg>/, yolo26n_<cfg>/ (M4.1 / M4.2 builds)

# board-log formats (kln_site.log_sites view, op name with or without a numeric cid)
_PROF = re.compile(r"^PROF part=(\d+) op=aie_host_\w+ ms=[\d.]+ .*site_ms=([\d.]+) ", re.M)
_LOGMSG = re.compile(r"\[\d\d:\d\d:\d\d [A-Z]+\] [^\n]*\n")
_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=\w+ .*launches=(\d+)$", re.M)


def read_log(path, net, launches):
    txt = Path(path).read_text(errors="replace")
    # the host runner's "[hh:mm:ss DEBUG] ..." messages can land in the middle of a PROF line and split it
    # (yolo26n_off.e.log part=151): drop each message with its newline, which rejoins the split line
    txt = _LOGMSG.sub("", txt)
    cmp_ = re.findall(r"^CMP %s board=(\S+) .*aie_launches=%d .*run_rc=0" % (re.escape(net), launches), txt, re.M)
    sp = {int(p): int(n) for p, n in _SPLIT.findall(txt)}
    sites = {int(p): float(ms) for p, ms in _PROF.findall(txt) if sp.get(int(p), 0) > 0}
    return dict(path=str(path), boards=cmp_, sites=sites, total=sum(sites.values()))


def decide(net, build, board):
    a, b, _L, _logs = NETS[net]
    pred = {}
    for cfg in (a, b):
        r = estimate({"net": net, "build": build, "cfg": cfg}, board=board)
        for f, m in sorted(r["params"].items()):
            print("PARAMS net=%s cfg=%s file=%s md5=%s" % (net, cfg, f, m))
        pred[cfg] = r["total_ms"]
    return pred, min((a, b), key=lambda c: pred[c])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="all 3 nets (default when --net is not given)")
    ap.add_argument("--net", action="append", choices=sorted(NETS))
    ap.add_argument("--tie", type=float, default=0.03)
    ap.add_argument("--alt", action="store_true", help="score on the second independent board run")
    ap.add_argument("--selftest", action="store_true", help="swap the measured cfgs of one non-tied net: must FAIL")
    ap.add_argument("--unet-build", default=UNET_BUILD)
    ap.add_argument("--net-build", default=NET_BUILD)
    ap.add_argument("--board", default="palmyra213")
    a = ap.parse_args(argv)
    nets = list(NETS) if (a.all or not a.net) else a.net
    run = "alt" if a.alt else "main"

    rc = 0
    rows = []
    boards = set()
    cmp_ok = 0
    for net in nets:
        ca, cb, L, logs = NETS[net]
        pred, pick = decide(net, a.unet_build if net == "unet" else a.net_build, a.board)
        meas = {ca: read_log(logs[run][0], net, L), cb: read_log(logs[run][1], net, L)}
        nb = set()
        for cfg, m in meas.items():
            print("LOG net=%s cfg=%s path=%s cmp=%s sites=%d" % (net, cfg, m["path"], ",".join(m["boards"]) or "-",
                                                                len(m["sites"])))
            if len(m["boards"]) != 1:
                print("ERROR net=%s cfg=%s: %d passing CMP lines in %s" % (net, cfg, len(m["boards"]), m["path"]))
                rc = 2
            nb |= set(m["boards"])
            cmp_ok += len(m["boards"])
        if len(nb) != 1:
            print("ERROR net=%s boards differ / missing: %s" % (net, sorted(nb)))
            rc = 2
        if set(meas[ca]["sites"]) != set(meas[cb]["sites"]):
            print("ERROR net=%s site sets differ between cfgs" % net)
            rc = 2
        boards |= nb
        rows.append(dict(net=net, a=ca, b=cb, L=L, pred=pred, pick=pick,
                         meas={c: m["total"] for c, m in meas.items()}))

    flipped = None
    if a.selftest:
        for r in rows:
            ma, mb = r["meas"][r["a"]], r["meas"][r["b"]]
            if abs(ma - mb) / ((ma + mb) / 2) >= a.tie:
                r["meas"] = {r["a"]: mb, r["b"]: ma}
                flipped = r["net"]
                break
        if flipped is None:
            print("ERROR selftest: every net is tied, nothing to flip")
            rc = 2

    good = 0
    for r in rows:
        ca, cb, m, p = r["a"], r["b"], r["meas"], r["pred"]
        meas_pick = min((ca, cb), key=lambda c: m[c])
        if r["pick"] == meas_pick:
            v = "agree"
        elif abs(m[ca] - m[cb]) / ((m[ca] + m[cb]) / 2) < a.tie:
            v = "tie"
        else:
            v = "FAIL"
        good += v != "FAIL"
        print("M6DEC net=%s aie_launches=%d pred_%s_ms=%.1f pred_%s_ms=%.1f meas_%s_ms=%.1f meas_%s_ms=%.1f pick=%s "
              "meas_pick=%s %s" % (r["net"], r["L"], ca, p[ca], cb, p[cb], ca, m[ca], cb, m[cb], r["pick"],
                                   meas_pick, v))
    ok = good == len(rows) and rc == 0 and len(boards) == 1 and cmp_ok == 2 * len(rows)
    board = ",".join(sorted(boards)) or "-"
    tag = "M6DEC SELFTEST flipped=%s" % flipped if a.selftest else "M6DEC ALL"
    print("%s nets=%d board=%s cmp_ok=%d agree_or_tie=%d %s" % (tag, len(rows), board, cmp_ok, good,
                                                               "PASS" if ok else "FAIL"))
    if not ok:
        rc = rc or 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
