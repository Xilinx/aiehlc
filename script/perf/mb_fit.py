#!/usr/bin/env python3
"""Fit the raw "MB ..." lines of example/perf/aie_microbench.cc into script/perf/params/<board>.json (costmodel K1a).

usage: mb_fit.py <console log> <out json> <board name>
Exits non-zero (no JSON written) if the BOARD marker is missing, any point is data=BAD, MB END is not PASS, or a
parameter is not a positive number.
"""
import json
import re
import sys


def kv(line):
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))


def main(log, out, name):
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
    sys.exit(main(*sys.argv[1:4]))
