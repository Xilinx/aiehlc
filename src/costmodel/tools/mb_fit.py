#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Fit the raw "MB ..." lines of example/perf/aie_microbench.cc into src/costmodel/params/<board>.json (costmodel K1a).

usage: mb_fit.py <console log> <out json> <board name> [aie|ddr]
Exits non-zero (no JSON written) if the BOARD marker is missing, any point is data=BAD, MB END is not PASS, or a
parameter is not a positive number.

"MB ddr" semantics (both sections): bytes= is the TOTAL over all reps and all columns (aie_microbench.cc prints
n*DDR_NCOL*DDR_REPS, ddr_microbench.cc n*cols*reps) and cntpct= is the SUM of the per-rep wall times, so
bytes/cntpct is the MEAN rep rate. section=aie (K1a) keeps that mean; section=ddr (K2d) replaces ddr_rd_gbps /
ddr_wr_gbps with the aggregate over ddr_rd_cols columns of the MEDIAN rep (bytes/reps / cntpct_med_rep), which must lie
within 10 % of the best rep (cntpct_min_rep), and MERGES into the existing <out json> (every aie key kept).
"""
import json
import re
import sys


def kv(line):
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))


DDR_CAP_GBPS_PER_COL = 14.0  # plausibility ruler: one shim column NMU <= 14 GB/s (Amit, ASSUMED, not measured)


def fit_ddr(log, out, name, lines, mb, errs):
    """section=ddr: per-rep DDR rates (cols 1..4) + strided small-burst cost; merge into the existing params JSON."""
    p = {}
    for kind in ("rd", "wr"):
        pts = {}
        for l in mb:
            if l.startswith("MB ddr kind=%s " % kind):
                c = kv(l)
                cols, reps, frq = int(c["cols"]), int(c["reps"]), int(c["cntfrq"])
                per_rep = int(c["bytes"]) / reps  # bytes= is the total over reps x cols
                med = per_rep / (int(c["cntpct_med_rep"]) / frq) / 1e9
                best = per_rep / (int(c["cntpct_min_rep"]) / frq) / 1e9
                pts[cols] = (med, best, reps)
        if not pts:
            errs.append("missing MB ddr kind=%s" % kind)
            continue
        cmax = max(pts)
        med, best, reps = pts[cmax]
        if abs(med - best) / best > 0.10:
            errs.append("ddr %s cols=%d median rep %.3f GB/s not within 10%% of best rep %.3f" % (kind, cmax, med, best))
        for cols, (m, _, _) in pts.items():
            if m > DDR_CAP_GBPS_PER_COL * cols:
                errs.append("ddr %s cols=%d %.3f GB/s > %.0f GB/s/col cap" % (kind, cols, m, DDR_CAP_GBPS_PER_COL))
        p["ddr_%s_gbps" % kind] = round(med, 3)
        p["ddr_%s_gbps_best" % kind] = round(best, 3)
        p["ddr_%s_cols" % kind] = cmax
        p["ddr_%s_reps" % kind] = reps
        p["ddr_%s_cols_points" % kind] = [[c, round(pts[c][0], 3)] for c in sorted(pts)]
        if len(pts) < 4:
            errs.append("ddr %s: %d column points, need 4" % (kind, len(pts)))
    # strided bursts: per size two n; cyc per burst = slope over n (start-up cancelled)
    bs = {}
    for l in mb:
        if l.startswith("MB ddrburst bytes="):
            c = kv(l)
            bs.setdefault(int(c["bytes"]), {})[int(c["n"])] = int(c["cyc"])
    bpts = []
    for b in sorted(bs):
        ns = sorted(bs[b])
        if len(ns) < 2:
            errs.append("ddrburst bytes=%d: need 2 n" % b)
            continue
        lo, hi = ns[0], ns[-1]
        bpts.append([b, round((bs[b][hi] - bs[b][lo]) / float(hi - lo), 3)])
    if len(bpts) >= 2:  # cyc_per_burst = cyc0 + bytes / b_per_cyc (least squares)
        n = len(bpts)
        mx = sum(x for x, _ in bpts) / n
        my = sum(y for _, y in bpts) / n
        sxx = sum((x - mx) ** 2 for x, _ in bpts)
        slope = sum((x - mx) * (y - my) for x, y in bpts) / sxx
        p["ddr_burst_points"] = bpts
        p["ddr_burst_cyc0"] = round(my - slope * mx, 2)
        p["ddr_burst_b_per_cyc"] = round(1.0 / slope, 3) if slope > 0 else None
        p["ddr_burst_stride"] = 4096
        p["ddr_burst_unit"] = "AIE cycles per burst (memtile S2MM task time slope over n bursts), shim MM2S 2-D BD, 4 KiB stride"
        if p["ddr_burst_cyc0"] < 0:
            errs.append("ddr_burst_cyc0=%s < 0" % p["ddr_burst_cyc0"])
    else:
        errs.append("ddrburst: <2 sizes")
    p["ddr_log"] = log
    try:
        j = json.load(open(out))
    except (OSError, ValueError) as e:
        errs.append("cannot read existing %s (run --section aie first): %s" % (out, e))
        j = {}
    if j and j.get("board") != name:
        errs.append("existing %s is board=%r, not %s" % (out, j.get("board"), name))
    for k in ("ddr_rd_gbps", "ddr_wr_gbps", "ddr_rd_cols", "ddr_rd_reps", "ddr_burst_cyc0"):
        print("PARAM %s=%s" % (k, p.get(k)))
    print("PARAM ddr_rd_cols_points=%s" % p.get("ddr_rd_cols_points"))
    print("PARAM ddr_wr_cols_points=%s" % p.get("ddr_wr_cols_points"))
    print("PARAM ddr_burst_points=%s" % p.get("ddr_burst_points"))
    if errs:
        for e in errs:
            print("[mb_fit] ERROR " + e)
        print("PARAMS %s.json NOT written (%d errors) section=ddr" % (name, len(errs)))
        return 1
    j.update(p)
    json.dump(j, open(out, "w"), indent=1, sort_keys=True)
    sys.stdout.flush()
    print("PARAMS %s.json written (%d params) section=ddr" % (name, len(j)))
    return 0


def main(log, out, name, section="aie"):
    lines = [l.rstrip("\n") for l in open(log, errors="replace")]
    board = [l for l in lines if re.match(r"^BOARD machine=aarch64 midr=0x41[0-9a-f]{6} board=" + re.escape(name) + "$", l)]
    mb = [l for l in lines if l.startswith("MB ")]
    errs = []
    if not board:
        errs.append("no BOARD machine=aarch64 ... board=%s line" % name)
    end = [l for l in mb if l.startswith("MB END")]
    if not end or "PASS" not in end[-1]:
        errs.append("MB END not PASS: %s" % (end[-1] if end else "missing"))
    bad = [l for l in mb if "data=BAD" in l or "TIMEOUT" in l or "rcfail" in l]
    errs += ["bad point: " + l for l in bad]
    if section == "ddr":
        return fit_ddr(log, out, name, lines, mb, errs)

    def one(prefix):
        r = [kv(l) for l in mb if l.startswith(prefix)]
        if not r:
            errs.append("missing " + prefix)
            return None
        return r[-1]

    p = {}
    c = one("MB clk ")
    if c:
        p["aie_clk_mhz"] = round(int(c["aie_cyc"]) / (int(c["cntpct"]) / int(c["cntfrq"])) / 1e6, 2)
    c = one("MB pmu ")
    if c:
        p["pmu_hz"] = int(round(int(c["pmccntr"]) / (int(c["cntpct"]) / int(c["cntfrq"]))))
    for kind in ("rd", "wr"):
        c = one("MB ddr kind=%s " % kind)
        if c:  # wall clock (CNTPCT), all columns concurrently, GB = 1e9 B
            p["ddr_%s_gbps" % kind] = round(int(c["bytes"]) / (int(c["cntpct"]) / int(c["cntfrq"])) / 1e9, 3)
            p["ddr_%s_cols" % kind] = int(c["cols"])
    dma = [kv(l) for l in mb if l.startswith("MB dma ")]
    pts = [(int(d["bytes"]), int(d["cyc"])) for d in dma if int(d["bytes"]) >= 4096]
    if len(pts) >= 2:  # least squares cyc = a + bytes/b over the >= 4 KiB points (steady state)
        n = len(pts)
        mx = sum(x for x, _ in pts) / n
        my = sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        slope = sxy / sxx
        p["dma_b_per_cyc"] = round(1.0 / slope, 3)
        p["dma_first_byte_cyc"] = round(my - slope * mx, 1)
        p["dma_points"] = [[x, y] for x, y in sorted((int(d["bytes"]), int(d["cyc"])) for d in dma)]
    else:
        errs.append("dma: <2 points >= 4 KiB")
    c = one("MB bd ")
    if c:
        nbd = int(c["nbd"])
        p["bd_switch_cyc"] = round((int(c["chain_cyc"]) - int(c["single_cyc"])) / (nbd - 1), 2)
    c = one("MB lock handoff")
    if c and "dma_b_per_cyc" in p:
        # lock hand-off: producer S2MM0 BD done (releases lock) -> waiting MM2S1 acquires -> its data lands in S2MM1;
        # minus the per-byte stream time (bytes / dma_b_per_cyc) = release->acquire->first-beat sync cost per hand-off.
        # leg_cyc (fresh task start, same bytes, no lock) is kept as a cross-check: hand-off < task start.
        p["lock_acq_cyc"] = round(int(c["handoff_cyc"]) - int(c["bytes"]) / p["dma_b_per_cyc"], 2)
        p["lock_handoff_points"] = {"handoff_cyc": int(c["handoff_cyc"]), "leg_cyc": int(c["leg_cyc"]), "bytes": int(c["bytes"])}
    c = [kv(l) for l in mb if l.startswith("MB lockchain ")]
    if c:  # uncontended lock on every chained BD (hidden by BD prefetch); report only
        p["lock_chain_extra_cyc_per_bd"] = round((int(c[-1]["lock_cyc"]) - int(c[-1]["chain_cyc"])) / int(c[-1]["nbd"]), 2)
    bs, bdf = one("MB bank case=same "), one("MB bank case=diff ")
    if bs and bdf:
        kib = int(bs["bytes_per_ch"]) / 1024.0
        es = (int(bs["elapsed0"]) + int(bs["elapsed1"])) / 2.0
        ed = (int(bdf["elapsed0"]) + int(bdf["elapsed1"])) / 2.0
        # DMA-vs-DMA: the arbiter conflicts (CONFLICT_DM_BANK_0, level count) but the 8 B/cyc streams absorb them, so
        # the elapsed penalty is reported separately (dm_bank_conflict_dma_elapsed_cyc_per_kib, 0 on palmyra213).
        # dm_bank_conflict_cyc = bank-0 conflict cycles (same - diff) per KiB of same-bank traffic (4 streams).
        p["dm_bank_conflict_dma_elapsed_cyc_per_kib"] = round((es - ed) / kib, 2)
        p["dm_bank_conflict_cyc"] = round((int(bs["conflict_bank0"]) - int(bdf["conflict_bank0"])) / (4 * kib), 2)
        p["dm_bank_conflict_unit"] = ("CONFLICT_DM_BANK_0 cycles per KiB of bank traffic, 2 MM2S + 2 S2MM streams in one "
                                      "8 KiB bank minus the 4-bank case; DMA elapsed penalty in dm_bank_conflict_dma_elapsed_cyc_per_kib")
        p["dm_bank_conflict_evt"] = {"same": [int(bs["conflict_bank0"]), int(bs["starv_mm2s0"])],
                                     "diff": [int(bdf["conflict_bank0"]), int(bdf["starv_mm2s0"])],
                                     "elapsed_same": [int(bs["elapsed0"]), int(bs["elapsed1"])],
                                     "elapsed_diff": [int(bdf["elapsed0"]), int(bdf["elapsed1"])]}
    hop = {int(kv(l)["links"]): int(kv(l)["cyc"]) for l in mb if l.startswith("MB hop ")}
    if 4 in hop and 6 in hop:
        p["ss_hop_cyc"] = round((hop[6] - hop[4]) / 2.0, 2)
        p["ss_hop_points"] = sorted([k, v] for k, v in hop.items())
    else:
        errs.append("hop: need links=4 and links=6")
    p["board"] = name
    p["path"] = "baremetal"
    p["log"] = log
    req = ["aie_clk_mhz", "pmu_hz", "ddr_rd_gbps", "ddr_wr_gbps", "dma_b_per_cyc", "dma_first_byte_cyc", "bd_switch_cyc",
           "lock_acq_cyc", "dm_bank_conflict_cyc", "ss_hop_cyc"]
    for k in req:
        v = p.get(k)
        print("PARAM %s=%s" % (k, v))
        if not isinstance(v, (int, float)) or v <= 0:
            errs.append("param %s=%r not > 0" % (k, v))
    if errs:
        for e in errs:
            print("[mb_fit] ERROR " + e)
        print("PARAMS %s.json NOT written (%d errors) section=aie" % (name, len(errs)))
        return 1
    json.dump(p, open(out, "w"), indent=1, sort_keys=True)
    print("PARAMS %s.json written (%d params) section=aie" % (name, len(req)))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
