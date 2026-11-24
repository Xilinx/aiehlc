#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Rebuild end-to-end routing flows from scanned stream-switch registers alone.

`switch_scan` answers "does the hardware match the routing map".  This answers
the prior question -- "what is the hardware actually routing" -- with no
provenance map involved, for the cases where the map is absent (raw-XAie flows),
stale (the board holds a different binary), or was never right (routing
reprogrammed at runtime).

The switch registers describe single hops, so flows are recovered by traversal:
intra-tile slave->master edges from the master config (circuit) or the
arbiter/msel pairing (packet), inter-tile edges from the fixed port wiring that
`aiediag.master_neighbor` models.  A DFS from every terminal slave yields one
fan-out tree per source; trees are then grouped by their shim endpoint, because
that -- not the tree -- is what the compiler's map calls a flow.  A broadcast to
four cores is one push flow with four sinks; four DMAs draining to one shim port
are one pull flow, not four.

Output is emitted in the two shapes the rest of the tooling already speaks:
`routing_groups` (routingprovenancemap.json) and `comm_paths` (the browser).
Emission is the inverse of `switch_scan.expected_records`, so re-scanning a
reconstructed map against the registers it came from comes back clean.

Known gap: a slot register carries no packet type (`aiediag.decode_strm_slot`
decodes id/mask/enable/msel/arbiter), so `pkttype` is reported as 0.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiediag

TERMINALS = aiediag.STRM_SW_TERMINALS

PUSH, PULL, LOCAL = "push", "pull", "local"


def _is_terminal(port_dir):
    return port_dir in TERMINALS


def _is_shim_ext(col_row, port_dir):
    """A row-0 SOUTH port is the PL/NoC/DDR boundary, not a tile hop."""
    return col_row[1] == 0 and port_dir == "SOUTH"


# ── per-tile edges ───────────────────────────────────────────────────────────

def intra_edges(tile_type, decoded):
    """({(slave_dir,slave_idx): [((master_dir,master_idx), meta)]}, orphans).

    meta is None for a circuit connection, else the slot/master packet fields
    the emitted packet_connect record needs.  Read off `decoded` rather than
    `switch_scan.live_records` because the Pkt namedtuple drops msel and slot.

    `orphans` are enabled packet masters no slot feeds -- the gather tile that
    hands a packet segment off to circuit switching looks like this.  They drive
    no edge but still have to be emitted, or the round trip through
    `switch_scan.expected_records` loses their Mst record.
    """
    slave_ports = aiediag.STRM_SW_SLAVE_PORTS[tile_type]
    out, orphans = {}, {}
    for m in decoded["masters"]:
        if not m["enable"]:
            continue
        mport = (m["type"], m["num"])
        if not m["packet"]:
            idx = m["slave_idx"]
            if idx is None or not 0 <= idx < len(slave_ports):
                continue
            out.setdefault(slave_ports[idx], []).append((mport, None))
            continue
        fed = False
        for s in decoded["slaves"]:
            if not (s["enable"] and s["packet"]):
                continue
            for slot in s["slots"]:
                if slot["arbiter"] != m["arbiter"]:
                    continue
                if not (m["msel_en"] >> slot["msel"]) & 1:
                    continue
                fed = True
                out.setdefault((s["type"], s["num"]), []).append((mport, {
                    "pktid": slot["id"], "mask": slot["mask"],
                    "msel": slot["msel"], "slot": slot["slot"],
                    "arbiter": m["arbiter"], "msel_en": m["msel_en"],
                    "drop_header": m["drop_header"],
                }))
        if not fed:
            orphans[mport] = {"arbiter": m["arbiter"], "msel_en": m["msel_en"],
                              "drop_header": m["drop_header"]}
    return out, orphans


def _sources(graph):
    """Every port a flow can originate at: a terminal slave, or DDR via a shim."""
    for cr in sorted(graph):
        for sport in sorted(graph[cr]):
            if _is_terminal(sport[0]) or _is_shim_ext(cr, sport[0]):
                yield cr, sport


# ── traversal ────────────────────────────────────────────────────────────────

class _Tree(object):
    """One source's fan-out: the tiles, hops and per-tile connections it drives."""

    def __init__(self, source):
        self.source = source          # (col_row, port)
        self.tiles = set()
        self.edges = set()            # (from_cr, to_cr)
        self.sinks = set()            # (col_row, master_port)
        # Path order, not sorted: `routing_edges_for_flow` stitches the
        # packet-to-circuit junction off the *first* circuit_connect, so a
        # reloaded map only rebuilds its edges if the walk order is preserved.
        self.conns = []               # [(col_row, slave_port, master_port)]
        self.pkt_meta = {}            # conn key -> meta dict
        self.partial = False          # a hop died on an unscanned tile

    def shim_sink(self):
        s = sorted(x for x in self.sinks if _is_shim_ext(x[0], x[1][0]))
        return s[0] if s else None

    def direction(self):
        if _is_shim_ext(self.source[0], self.source[1][0]):
            return PUSH
        return PULL if self.shim_sink() else LOCAL

    def group_key(self):
        """Trees sharing a shim endpoint are one flow. A gather has several
        sources and one shim sink, so keying a pull on the sink merges them."""
        d = self.direction()
        if d == PUSH:
            return (d,) + self.source[0] + self.source[1]
        if d == PULL:
            cr, port = self.shim_sink()
            return (d,) + cr + port
        return (d,) + self.source[0] + self.source[1]


def _walk(graph, tree, cr, sport, guard, aie_version):
    if (cr, sport) in guard:
        return
    guard = guard | {(cr, sport)}
    tree.tiles.add(cr)
    for mport, meta in graph.get(cr, {}).get(sport, []):
        key = (cr, sport, mport)
        if key not in tree.conns:
            tree.conns.append(key)
        if meta is not None:
            tree.pkt_meta[key] = meta
        nb = aiediag.master_neighbor(cr[0], cr[1], mport[0], mport[1],
                                     aie_version)
        if nb is None or nb[0] == "terminal":
            tree.sinks.add((cr, mport))
            continue
        ncr = (nb[1], nb[2])
        if ncr not in graph:
            tree.sinks.add((cr, mport))
            tree.partial = True
            continue
        tree.edges.add((cr, ncr))
        _walk(graph, tree, ncr, (nb[3], nb[4]), guard, aie_version)


def discover_flows(switches, aie_version="5"):
    """Reconstruct flows from {(col,row): (tile_type, decoded)}.

    Tiles that could not be read are simply absent from `switches`; a flow
    transiting one splits into partial fragments rather than being bridged.
    Returns flow dicts ordered so ids are stable across rescans.
    """
    graph, orphans, types = {}, {}, {}
    for cr, (ttype, dec) in switches.items():
        graph[cr], orph = intra_edges(ttype, dec)
        types[cr] = ttype
        if orph:
            orphans[cr] = orph

    groups = {}
    for cr, sport in _sources(graph):
        tree = _Tree((cr, sport))
        _walk(graph, tree, cr, sport, set(), aie_version)
        if not tree.conns:
            continue
        groups.setdefault(tree.group_key(), []).append(tree)

    flows = []
    for i, key in enumerate(sorted(groups)):
        flows.append(_merge(key, groups[key], types, i, orphans))
    return flows


def reconcile(flows, static_comm_paths):
    """Adopt the static map's id/flow_index where a flow matches it exactly.

    The browser keys DMA-channel and lock tables off `flow_index` against the
    statically-built `DATA.tiles`, so a reconstructed flow that IS the static
    flow should carry its index or those tables render empty.  A flow whose
    tiles or edges differ keeps its own index -- it is genuinely not that flow,
    and an empty table is the honest answer.
    """
    by_shape = {}
    for p in static_comm_paths or []:
        key = (frozenset(map(tuple, p.get("tiles") or [])),
               frozenset((tuple(a), tuple(b)) for a, b in (p.get("edges") or [])))
        by_shape.setdefault(key, []).append(p)
    for f in flows:
        key = (frozenset(f["tiles"]), frozenset(f["edges"]))
        cand = by_shape.get(key)
        if not cand:
            continue
        p = cand.pop(0)
        f["flow_index"] = p.get("flow_index", f["flow_index"])
        f["id"] = p.get("id", f["id"])
        f["reconciled"] = True
    return flows


def _merge(key, trees, types, flow_index, orphans):
    """Fold the trees sharing a shim endpoint into one flow dict."""
    direction = key[0]
    tiles, edges, sinks, pkt_meta = set(), set(), set(), {}
    conns, seen = [], set()
    partial = False
    for t in trees:
        tiles |= t.tiles
        edges |= t.edges
        sinks |= t.sinks
        pkt_meta.update(t.pkt_meta)
        partial = partial or t.partial
        for key in t.conns:
            if key not in seen:
                seen.add(key)
                conns.append(key)

    sources = [t.source for t in trees]
    if direction == PUSH:
        shim = trees[0].source[0]
    elif direction == PULL:
        shim = trees[0].shim_sink()[0]
    else:
        shim = sources[0][0]

    dma_tiles = sorted({cr for cr, port in sinks if port[0] == "DMA"} |
                       {cr for cr, port in sources if port[0] == "DMA"})
    packet_tiles = sorted({k[0] for k in pkt_meta})
    return {
        "flow_index": flow_index,
        "id": "%s_%d" % (direction, flow_index),
        "direction": direction,
        "shim": shim,
        "sources": sorted(sources),
        "sinks": sorted(sinks),
        "tiles": sorted(tiles),
        "edges": sorted(edges),
        "dma_tiles": dma_tiles,
        "packet_tiles": packet_tiles,
        "conns": conns,
        "pkt_meta": pkt_meta,
        "orphan_masters": {cr: orphans[cr] for cr in tiles if cr in orphans},
        "partial": partial,
        "types": {cr: types.get(cr, "core") for cr in tiles},
    }


# ── emission ─────────────────────────────────────────────────────────────────

_NONE_PORT = {"dir": "NONE", "idx": 0, "pktid": 0, "pkttype": 0,
              "mask": 0, "msel": 0, "arbiter": 0, "slot": 0}


def _tile_ref(flow, cr):
    return {"col": cr[0], "row": cr[1], "type": flow["types"].get(cr, "core")}


def _pkt_port(port, meta):
    return {"dir": port[0], "idx": port[1], "pktid": meta["pktid"],
            "pkttype": 0, "mask": meta["mask"], "msel": meta["msel"],
            "arbiter": meta["arbiter"], "slot": meta["slot"]}


def _packet_records(flow):
    """One packet_connect per (tile, forward master).

    The map frames a packet hop as recv_slave + local_dma + forward_master, so
    the legs arriving at one master are folded back into that shape; a leg with
    no counterpart is emitted as dir NONE, which is what the renderer and
    `switch_scan.expected_records` both expect.
    """
    by_master = {}
    for key in flow["conns"]:          # path order, see _Tree.conns
        meta = flow["pkt_meta"].get(key)
        if meta is None:
            continue
        cr, sport, mport = key
        by_master.setdefault((cr, mport), []).append((sport, meta))
    # A packet master no slot feeds still needs its record, or the Mst is lost.
    for cr, ports in sorted(flow["orphan_masters"].items()):
        for mport in sorted(ports):
            by_master.setdefault((cr, mport), [])

    out = []
    for (cr, mport), legs in by_master.items():
        recv, dma = dict(_NONE_PORT), dict(_NONE_PORT)
        drop = False
        for sport, meta in sorted(legs):
            drop = drop or meta["drop_header"]
            if _is_terminal(sport[0]):
                dma = _pkt_port(sport, meta)
            else:
                recv = _pkt_port(sport, meta)
        meta0 = legs[0][1] if legs else flow["orphan_masters"][cr][mport]
        drop = drop or (not legs and meta0["drop_header"])
        out.append({
            "kind": "packet_connect", "tile": _tile_ref(flow, cr),
            "recv_slave": recv, "local_dma": dma,
            "forward_master": {"dir": mport[0], "idx": mport[1],
                               "arbiter": meta0["arbiter"],
                               "msel_en": meta0["msel_en"]},
            "preserve_header": not drop,
        })
    return out


def _shim_record(flow, kind, cr, port):
    """Both spellings: the UI's GMIO table reads only `stream_id`, while the
    tiling-style consumers read `port`."""
    return {"kind": kind, "tile": _tile_ref(flow, cr), "stream_id": port[1],
            "port": {"dir": port[0], "idx": port[1]}}


def _connections(flow):
    """routing_connections for one flow: shim entry, hops, shim exit.

    Order matters. `schedule_view.routing_edges_for_flow` splits a group at the
    first `shim_aie_to_ext` and, for a pull, reads packet edges only from the
    prefix -- so a pull group must put its packet records before the marker or
    every packet forward edge is dropped on reload.
    """
    circuit = [{"kind": "circuit_connect", "tile": _tile_ref(flow, cr),
                "slave": {"dir": sport[0], "idx": sport[1]},
                "master": {"dir": mport[0], "idx": mport[1]}}
               for cr, sport, mport in flow["conns"]
               if (cr, sport, mport) not in flow["pkt_meta"]]
    packet = _packet_records(flow)
    exits = [_shim_record(flow, "shim_aie_to_ext", cr, port)
             for cr, port in flow["sinks"] if _is_shim_ext(cr, port[0])]

    if flow["direction"] == PUSH:
        cr, port = flow["sources"][0]
        return ([_shim_record(flow, "shim_ext_to_aie", cr, port)]
                + circuit + packet + exits)

    # Pull. The reload derives packet hops from the records before the marker
    # and stitches the packet-to-circuit junction from the ones after it, so
    # the record whose master leaves the packet segment has to go after.
    pkt_tiles = {(c["tile"]["col"], c["tile"]["row"]) for c in packet}
    segment, junction = [], []
    for c in packet:
        fm = c["forward_master"]
        d = aiediag._DIR_DELTA.get(fm["dir"])
        dest = (c["tile"]["col"] + d[0], c["tile"]["row"] + d[1]) if d else None
        if dest in pkt_tiles:
            segment.append(c)
            continue
        junction.append(c)
        # The hop *into* the junction is only rebuilt if the junction tile is
        # already known to be a packet tile before the marker, so leave a
        # forward-less twin there -- the same placeholder the compiler emits.
        twin = dict(c)
        twin["forward_master"] = {"dir": "NONE", "idx": 0}
        segment.append(twin)
    return segment + exits + junction + circuit


def _stages(flow):
    """Minimal producer/consumer so the net panel shows endpoints."""
    out = []
    if flow["sources"]:
        cr, port = flow["sources"][0]
        out.append({"role": "producer", "tile": _tile_ref(flow, cr),
                    "port_sym": aiediag.port_label(*port)})
    if flow["sinks"]:
        cr, port = flow["sinks"][0]
        out.append({"role": "consumer", "tile": _tile_ref(flow, cr),
                    "port_sym": aiediag.port_label(*port)})
    return out


def to_comm_paths(flows):
    """The comm_paths list the browser's routing panels consume."""
    out = []
    for f in flows:
        out.append({
            "id": f["id"], "flow_index": f["flow_index"],
            "direction": f["direction"],
            "prod_col": f["shim"][0], "prod_row": f["shim"][1],
            "tiles": [list(t) for t in f["tiles"]],
            "edges": [[list(a), list(b)] for a, b in f["edges"]],
            "dma_tiles": [list(t) for t in f["dma_tiles"]],
            "packet_tiles": [list(t) for t in f["packet_tiles"]],
            "hops": [{"from_col": a[0], "from_row": a[1],
                      "to_col": b[0], "to_row": b[1], "type": "stream"}
                     for a, b in f["edges"]],
            "routing_connections": _connections(f),
            "stages": _stages(f),
            "dynamic": True,
            "partial": f["partial"],
        })
    return out


def to_routing_groups(flows, startcol=0, aie_gen="Gen5"):
    """A routingprovenancemap.json-shaped dict the CLI crossref tools accept."""
    groups = []
    for f in flows:
        groups.append({
            "id": "dynamic_%d" % f["flow_index"], "memo": "live-scan",
            "scf_idx": f["flow_index"], "ioid": f["flow_index"],
            "flow_index": f["flow_index"], "direction": f["direction"],
            "tiles": [_tile_ref(f, cr) for cr in f["tiles"]],
            "dma_tiles": [list(t) for t in f["dma_tiles"]],
            "connections": _connections(f),
        })
    return {"version": 1, "startcol": startcol, "aie_gen": aie_gen,
            "module_attrs": {}, "source": "live-stream-switch-scan",
            "routing_groups": groups}
