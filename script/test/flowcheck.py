#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""flowcheck: do routing.cc and host.cc deliver every dataflow where the dmaphop
IR says it goes?  Static, seconds, no board.

    python3 script/test/flowcheck.py aout/worklocal [--json]

Intent  : dmaphopprovenacemap.json (written from the dmaphop IR the pipeline
          actually used, so it follows AIEHLC_DMAPHOP_IR_IN): per flow the
          producer (shim MM2S channel, or core MM2S channel + packet id) and the
          consumers (core S2MM channels, or a shim S2MM channel).
Routing : routing.cc stream-switch programming, simulated: circuit connections,
          packet-switch slots / arbiters / msel, the shim mux / demux, and the
          neighbour links (master NORTH i -> slave SOUTH i of the tile above, ...).
Host    : host.cc "Create IO" comments: a DMA IO on the producer / consumer shim
          tile and channel of every flow.

A flow passes when the endpoints its data reaches are exactly its consumers, no
other flow reaches them, and host.cc has the shim IO. Exit 0 = consistent.
Why: RoutingHWVerify only checks that neighbour ports pair up; an edited dmaphop
IR (doc/design/routing_ir_reentry.md) built with 0 errors there and computed wrong
data on the board because flows were re-paired.

Shim channel <-> stream port: MM2S ch0 -> SOUTH 3, ch1 -> SOUTH 7; S2MM ch0 <-
SOUTH 1, ch1 <- SOUTH 3 on Gen5 (AIE2PS), SOUTH 2 / 3 otherwise (the ports the
driver accepts, in channel order; matches every board-verified build).
"""
import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

LOC = r"XAie_TileLoc\((\d+),\s*(\d+)\)"
NEIGHBOUR = {"NORTH": (0, 1, "SOUTH"), "SOUTH": (0, -1, "NORTH"), "EAST": (1, 0, "WEST"), "WEST": (-1, 0, "EAST")}


def shim_ports(aie_gen):
    mm2s = {0: 3, 1: 7}
    s2mm = {0: 1, 1: 3} if aie_gen in ("Gen5", "AIE2PS") else {0: 2, 1: 3}
    return mm2s, s2mm


class Switches:
    """routing.cc as a port graph."""

    def __init__(self, text):
        self.circuit = defaultdict(list)    # (c,r,slvT,slvI) -> [(c,r,mstT,mstI)]
        self.pkt_slave = set()               # (c,r,slvT,slvI) in packet mode
        self.slots = defaultdict(list)       # (c,r,slvT,slvI) -> [(id, mask, msel, arbiter)]
        self.pkt_master = defaultdict(list)  # (c,r) -> [(mstT, mstI, arbiter, mselEn)]
        self.shim_in = set()                 # (c, port): shim DMA -> stream mux
        self.shim_out = set()                # (c, port): stream -> shim DMA demux
        for line in text.splitlines():
            m = re.search(rf"XAie_StrmConnCctEnable\(\w+, {LOC}, (\w+), (\d+), (\w+), (\d+)\)", line)
            if m:
                c, r, st, si, mt, mi = int(m[1]), int(m[2]), m[3], int(m[4]), m[5], int(m[6])
                self.circuit[(c, r, st, si)].append((c, r, mt, mi))
                continue
            m = re.search(rf"XAie_StrmPktSwSlavePortEnable\(\w+, {LOC}, (\w+), (\d+)\)", line)
            if m:
                self.pkt_slave.add((int(m[1]), int(m[2]), m[3], int(m[4])))
                continue
            m = re.search(rf"XAie_StrmPktSwSlaveSlotEnable\(\w+, {LOC}, (\w+), (\d+), \d+, "
                          r"XAie_PacketInit\((\d+), \d+\), (\d+), (\d+), (\d+)\)", line)
            if m:
                key = (int(m[1]), int(m[2]), m[3], int(m[4]))
                self.slots[key].append((int(m[5]), int(m[6]), int(m[7]), int(m[8])))
                continue
            m = re.search(rf"XAie_StrmPktSwMstrPortEnable\(\w+, {LOC}, (\w+), (\d+), \w+, (\d+), (\d+)\)", line)
            if m:
                self.pkt_master[(int(m[1]), int(m[2]))].append((m[3], int(m[4]), int(m[5]), int(m[6])))
                continue
            m = re.search(rf"XAie_EnableShimDmaToAieStrmPort\(\w+, {LOC}, (\d+)\)", line)
            if m:
                self.shim_in.add((int(m[1]), int(m[3])))
                continue
            m = re.search(rf"XAie_EnableAieToShimDmaStrmPort\(\w+, {LOC}, (\d+)\)", line)
            if m:
                self.shim_out.add((int(m[1]), int(m[3])))

    def masters_of(self, slave, pkt):
        """Master ports a word entering `slave` (with packet id `pkt`) leaves through."""
        out = list(self.circuit.get(slave, []))
        if slave in self.pkt_slave:
            if pkt is None:
                return out, f"packet-mode slave {slave} reached without a packet id"
            c, r = slave[0], slave[1]
            for sid, mask, msel, arb in self.slots.get(slave, []):
                if (pkt & mask) == (sid & mask):
                    for mt, mi, marb, msel_en in self.pkt_master.get((c, r), []):
                        if marb == arb and (msel_en >> msel) & 1:
                            out.append((c, r, mt, mi))
                    break
        return out, None

    def deliver(self, start, pkt, s2mm_port_to_ch):
        """Endpoints reachable from slave port `start`: ("core", c, r, ch) or ("shim", c, ch)."""
        seen, ends, notes = set(), set(), []
        todo = deque([start])
        while todo:
            slave = todo.popleft()
            if slave in seen:
                continue
            seen.add(slave)
            masters, note = self.masters_of(slave, pkt)
            if note:
                notes.append(note)
            for c, r, mt, mi in masters:
                if mt == "DMA":
                    ends.add(("core", c, r, mi))
                elif mt == "SOUTH" and r == 0 and (c, mi) in self.shim_out:
                    ends.add(("shim", c, s2mm_port_to_ch.get(mi, -1)))
                elif mt in NEIGHBOUR:
                    dc, dr, side = NEIGHBOUR[mt]
                    if r + dr >= 0:
                        todo.append((c + dc, r + dr, side, mi))
        return ends, notes


def host_ios(text):
    ios = set()
    for m in re.finditer(r"Create IO: channel_id=(\d+), bd_id=\d+, tile=\((\d+),(\d+)\), direction=(\w+)", text):
        ios.add((int(m[2]), int(m[3]), int(m[1]), m[4]))
    return ios


def check(worklocal):
    wl = Path(worklocal)
    prov = json.loads((wl / "dmaphopprovenacemap.json").read_text())
    sw = Switches((wl / "routing.cc").read_text())
    ios = host_ios((wl / "host.cc").read_text())
    mm2s, s2mm = shim_ports(prov.get("aie_gen", ""))
    s2mm_port_to_ch = {p: ch for ch, p in s2mm.items()}
    errors, flows, owner = [], [], defaultdict(list)
    for path in prov["communication_paths"]:
        fid = path["id"]
        st = {s["role"]: s for s in path["stages"]}
        prod, cons = st["producer"], st["consumer"]
        if path["direction"] == "push":
            c, ch = prod["tile"]["col"], prod["channel"]
            want = {("core", t["col"], t["row"], t["dma_port"]) for t in cons["tiles"]}
            port = mm2s.get(ch)
            if (c, port) not in sw.shim_in:
                errors.append(f"{fid}: shim ({c},0) MM2S ch{ch} -> SOUTH {port} is not enabled in routing.cc")
            if (c, 0, ch, "MM2S") not in ios:
                errors.append(f"{fid}: host.cc creates no MM2S IO on shim ({c},0) ch{ch}")
            got, notes = sw.deliver((c, 0, "SOUTH", port), None, s2mm_port_to_ch)
            sources = [(f"shim({c},0).mm2s{ch}", got, notes)]
        else:
            c, ch = cons["tile"]["col"], cons["channel"]
            want = {("shim", c, ch)}
            if (c, s2mm.get(ch)) not in sw.shim_out:
                errors.append(f"{fid}: SOUTH {s2mm.get(ch)} -> shim ({c},0) S2MM ch{ch} is not enabled in routing.cc")
            if (c, 0, ch, "S2MM") not in ios:
                errors.append(f"{fid}: host.cc creates no S2MM IO on shim ({c},0) ch{ch}")
            sources = []
            for t in prod["tiles"]:
                got, notes = sw.deliver((t["col"], t["row"], "DMA", t["dma_port"]), t.get("pkt_id"), s2mm_port_to_ch)
                sources.append((f"core({t['col']},{t['row']}).mm2s{t['dma_port']}", got, notes))
        for name, got, notes in sources:
            for n in sorted(set(notes)):
                errors.append(f"{fid} from {name}: {n}")
            for e in got:
                owner[e].append(fid)
            missing, extra = want - got, got - want
            if missing:
                errors.append(f"{fid} from {name}: does not reach {sorted(missing)}")
            if extra:
                errors.append(f"{fid} from {name}: also reaches {sorted(extra)}")
        flows.append({"id": fid, "direction": path["direction"], "want": sorted(map(list, want)),
                      "reached": sorted({tuple(e) for _, g, _ in sources for e in g})})
    for e, fids in owner.items():
        if len(set(fids)) > 1:
            errors.append(f"{e} is reached by several flows: {sorted(set(fids))}")
    return {"ok": not errors, "errors": errors, "flows": flows}


def main():
    ap = argparse.ArgumentParser(description="routing.cc / host.cc vs the dmaphop IR's dataflows")
    ap.add_argument("worklocal", help="aout/worklocal of an aiehlc build (or a routing_ir check dir)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = check(a.worklocal)
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        for e in r["errors"]:
            print("FLOWCHECK ERROR:", e)
        print(f"flowcheck: {'OK' if r['ok'] else 'FAIL'} -- {len(r['flows'])} flows, {len(r['errors'])} error(s)")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
