#!/usr/bin/env python3
"""M6.1 demo: a UNet partition decision made by costmodel.estimate() (frozen K+L+N, M34 PZ),
checked against what the board measured.

  python3 costmodel_partition_demo.py --build-root DIR --partition partition_p4a.json \
      --meas-off unet_off.a.log --meas-kco unet_aot-kco.a.log [--tie 0.03]

Decision: build partition_p4a with cfg off (MMIO) or aot-kco.  estimate() prices every launches>0 site of both
builds (DIR/unet_off, DIR/unet_aot-kco, build dirs only); the board logs are read ONLY to score the decision.
Per site the pick is the cfg with the smaller T; strict = sites whose predicted pick == measured pick,
tie = the remaining sites whose measured |T_off-T_kco|/mean < --tie.
Caveat (Maxwell r1): aot-kco needs a free west column -- the per-site table is a report, not a buildable mix;
the decision is per partition (pick / meas_pick).

Output (last line, field order fixed):
  M61DEMO board=<B> cmp_ok=<n> part=<stem> sites=<n> pred_off_ms=X pred_kco_ms=X meas_off_ms=X meas_kco_ms=X
          pick=<cfg> meas_pick=<cfg> agree=0|1 strict=<n> tie=<n> unmapped=<n>
rc != 0 if a log has no passing CMP line or the two logs come from different boards.
"""
import argparse
import json
import re
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "src"))

from costmodel import estimate  # noqa: E402

CFGS = ("off", "aot-kco")
# board-log formats (same regexes as script/perf n_fit._PROF / k_diag._SPLIT used by kln_site.log_sites)
_PROF = re.compile(r"^PROF part=(\d+) op=aie_host_(conv|linear|attn)(\d+) ms=([\d.]+) .*site_ms=([\d.]+) "
                   r"aie_ms=([\d.]+)", re.M)
_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) .*wait_ms=([\d.]+) .*inval_ms=([\d.]+) "
                    r"launches=(\d+)$", re.M)
_CMP = re.compile(r"^CMP unet board=(\S+) .*aie_launches=258 .*run_rc=0 .*PASS$", re.M)
_SITEPERF = re.compile(r"^PROF aie_site_perf ", re.M)


def log_sites(txt):
    """{part: (site, launches, site_ms)} of the sites with launches > 0 (= kln_site.log_sites)"""
    sp = {int(p): int(n) for p, _k, _w, _i, n in _SPLIT.findall(txt)}
    out = {}
    for part, op, cid, _ms, site_ms, _aie in _PROF.findall(txt):
        n = sp.get(int(part), 0)
        if n > 0:
            out[int(part)] = ("%s%s" % ("fa" if op == "attn" else op, cid), n, float(site_ms))
    return out


def read_log(path):
    txt = Path(path).read_text(errors="replace")
    cmp_ = _CMP.findall(txt)
    return dict(path=path, boards=cmp_, nperf=len(_SITEPERF.findall(txt)), sites=log_sites(txt))


def part_index(path):
    """{(cls, idx): layer} of the partition JSON (site cid == layer idx, site kind == layer cls)"""
    d = json.loads(Path(path).read_text())
    return {(l["cls"], int(l["idx"])): l for l in d["layers"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", required=True, help="dir holding unet_off/ and unet_aot-kco/ builds")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--meas-off", required=True)
    ap.add_argument("--meas-kco", required=True)
    ap.add_argument("--tie", type=float, default=0.03)
    ap.add_argument("--board", default="palmyra213")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)

    # ---- the decision: estimate() on the build dirs only
    pred = {}
    for cfg in CFGS:
        r = estimate({"unet_build": a.build_root, "cfg": cfg, "sites": None}, board=a.board, params="m3122")
        for f, m in sorted(r["params"].items()):
            print("PARAMS cfg=%s file=%s md5=%s" % (cfg, f, m))
        pred[cfg] = {int(s["part"]): s for s in r["sites"]}
        pred[cfg + "_total"] = r["total_ms"]
    pick = min(CFGS, key=lambda c: pred[c + "_total"])

    # ---- the board measurement (scoring only)
    meas = {"off": read_log(a.meas_off), "aot-kco": read_log(a.meas_kco)}
    rc = 0
    boards = set()
    for cfg, m in meas.items():
        if not m["boards"]:
            print("ERROR no passing CMP line in %s" % m["path"])
            rc = 2
        boards |= set(m["boards"])
    if len(boards) != 1:
        print("ERROR boards differ / missing: %s" % sorted(boards))
        rc = 2
    board = ",".join(sorted(boards)) or "-"
    cmp_ok = sum(len(m["boards"]) for m in meas.values())
    nperf = {c: m["nperf"] for c, m in meas.items()}
    if nperf["off"] != nperf["aot-kco"]:
        print("ERROR aie_site_perf counts differ: %s" % nperf)
        rc = 2
    sites = nperf["off"]
    meas_tot = {c: sum(v[2] for v in m["sites"].values()) for c, m in meas.items()}
    meas_pick = min(CFGS, key=lambda c: meas_tot[c])

    # ---- per site
    pidx = part_index(a.partition)
    parts = sorted(set(meas["off"]["sites"]) | set(meas["aot-kco"]["sites"]))
    if len(parts) != sites:
        print("WARN log_sites=%d vs aie_site_perf=%d" % (len(parts), sites))
    strict = tie = unmapped = 0
    rows = []
    print("SITE part site launches pred_off pred_kco meas_off meas_kco pred_pick meas_pick class")
    for p in parts:
        so, sk = meas["off"]["sites"].get(p), meas["aot-kco"]["sites"].get(p)
        po, pk = pred["off"].get(p), pred["aot-kco"].get(p)
        name = (so or sk)[0]
        kind, cid = re.match(r"([a-z]+)(\d+)$", name).groups()
        if (("attn" if kind == "fa" else kind), int(cid)) not in pidx:
            print("UNMAPPED %s part=%d" % (name, p))
            unmapped += 1
        if not (so and sk and po and pk):
            print("MISSING %s part=%d meas_off=%s meas_kco=%s pred_off=%s pred_kco=%s"
                  % (name, p, bool(so), bool(sk), bool(po), bool(pk)))
            continue
        t_po, t_pk = po["T_hat"] * po["launches"], pk["T_hat"] * pk["launches"]
        t_mo, t_mk = so[2], sk[2]
        pp = "off" if t_po < t_pk else "aot-kco"
        mp = "off" if t_mo < t_mk else "aot-kco"
        if pp == mp:
            strict += 1
            c = "strict"
        elif abs(t_mo - t_mk) / ((t_mo + t_mk) / 2) < a.tie:
            tie += 1
            c = "tie"
        else:
            c = "disagree"
        rows.append((name, p, abs(t_po - t_mo), abs(t_pk - t_mk), t_po, t_mo, t_pk, t_mk))
        print("SITE %d %s %d %.4f %.4f %.4f %.4f %s %s %s" % (p, name, so[1], t_po, t_pk, t_mo, t_mk, pp, mp, c))
    print("TOP |pred-meas| per site and cfg:")
    top = sorted([(r[2], "off", r) for r in rows] + [(r[3], "aot-kco", r) for r in rows], key=lambda x: -x[0])
    for d, cfg, r in top[:a.top]:
        pr, me = (r[4], r[5]) if cfg == "off" else (r[6], r[7])
        print("  top site=%s part=%d cfg=%s pred_ms=%.3f meas_ms=%.3f absdiff_ms=%.3f" % (r[0], r[1], cfg, pr, me, d))
    agree = int(pick == meas_pick)
    n = {r[1]: meas["off"]["sites"][r[1]][1] for r in rows}
    print("SUMT per-launch T summed over sites (site_pairs.tsv view): pred_off=%.1f pred_kco=%.1f meas_off=%.1f "
          "meas_kco=%.1f" % tuple(sum(r[i] / n[r[1]] for r in rows) for i in (4, 6, 5, 7)))
    print("M61DEMO board=%s cmp_ok=%d part=%s sites=%d pred_off_ms=%.1f pred_kco_ms=%.1f meas_off_ms=%.1f "
          "meas_kco_ms=%.1f pick=%s meas_pick=%s agree=%d strict=%d tie=%d unmapped=%d"
          % (board, cmp_ok, Path(a.partition).stem, sites, pred["off_total"], pred["aot-kco_total"],
             meas_tot["off"], meas_tot["aot-kco"], pick, meas_pick, agree, strict, tie, unmapped))
    return rc


if __name__ == "__main__":
    sys.exit(main())
