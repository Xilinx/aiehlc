#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert routingresourcemap.json into a self-contained C header.

The generated header (aie_resource_map.h) declares a flat table of routing
resource entries -- for each connection, which tile / stream port / pkt id /
pkt mask it uses -- plus a dependency-free __Runtime_print_resource_map()
that prints every entry via plain printf.

Usage:
    resource_json_to_header.py <routingresourcemap.json> --out <aie_resource_map.h>
"""

import argparse
import json
import sys


def _port(obj, key):
    """Return (dir, idx) for a nested port object, or ("NONE", -1)."""
    p = obj.get(key) if obj else None
    if not isinstance(p, dict):
        return ("NONE", -1)
    return (str(p.get("dir", "NONE")), int(p.get("idx", -1)))


def _slot(obj, key):
    """Return (dir, idx, pktid, mask, msel, arbiter) for a recv_slave / local_dma object."""
    p = obj.get(key) if obj else None
    if not isinstance(p, dict):
        return ("NONE", -1, -1, 0, 0, 0)
    return (
        str(p.get("dir", "NONE")),
        int(p.get("idx", -1)),
        int(p.get("pktid", -1)),
        int(p.get("mask", 0)),
        int(p.get("msel", 0)),
        int(p.get("arbiter", 0)),
    )


def _master(obj, key):
    """Return (dir, idx, mselen, arbiter) for a forward_master object.

    The master port carries the MSelEn bitmask (MSelEn |= 1 << slave_msel), not
    a single slave slot's msel. Accepts the enriched object form
    (dir/idx/mselen/arbiter); falls back to the legacy "msel" key, then to the
    plain (dir, idx) port form with mselen/arbiter defaulted for older JSON.
    """
    p = obj.get(key) if obj else None
    if not isinstance(p, dict):
        return ("NONE", -1, 0, 0)
    mselen = p.get("mselen", p.get("msel", 0))
    return (
        str(p.get("dir", "NONE")),
        int(p.get("idx", -1)),
        int(mselen),
        int(p.get("arbiter", 0)),
    )


def _new_row(kind, col, row, tkind):
    """A blank table row: every optional port empty (NONE/-1)."""
    return {
        "kind": kind,
        "col": col,
        "row": row,
        "tkind": tkind,
        # packet_connect: receive slave slot
        "recv_dir": "NONE",
        "recv_idx": -1,
        "recv_pktid": -1,
        "recv_mask": 0,
        "recv_msel": 0,
        "recv_arbiter": 0,
        # packet_connect: local DMA slave slot
        "dma_dir": "NONE",
        "dma_idx": -1,
        "dma_pktid": -1,
        "dma_mask": 0,
        "dma_msel": 0,
        "dma_arbiter": 0,
        # packet_connect: forward master port
        "fwd_dir": "NONE",
        "fwd_idx": -1,
        "fwd_mselen": 0,
        "fwd_arbiter": 0,
        # circuit_connect / circuit_connect_pair / shim_*: generic slave+master
        "slave_dir": "NONE",
        "slave_idx": -1,
        "master_dir": "NONE",
        "master_idx": -1,
        "preserve": 0,
    }


def build_entries(model):
    """Flatten the JSON connections into a stable list of table rows (dicts).

    Each row is one tile's use of a stream-switch resource. A packet_connect
    fills the recv/dma/fwd fields; a circuit_connect fills the generic
    slave/master fields; a circuit_connect_pair spans two tiles so it emits two
    rows (one per src_tile/dst_tile); shim_* carry their single port in the
    master fields to keep the table complete for collision queries.
    """
    rows = []
    for conn in model.get("connections", []):
        kind = str(conn.get("kind", "unknown"))

        if kind == "circuit_connect_pair":
            for tkey, skey, mkey in (
                ("src_tile", "src_slave", "src_master"),
                ("dst_tile", "dst_slave", "dst_master"),
            ):
                tile = conn.get(tkey) or {}
                r = _new_row(kind, int(tile.get("col", -1)), int(tile.get("row", -1)),
                             str(tile.get("kind", "unknown")))
                r["slave_dir"], r["slave_idx"] = _port(conn, skey)
                r["master_dir"], r["master_idx"] = _port(conn, mkey)
                rows.append(r)
            continue

        tile = conn.get("tile") or {}
        r = _new_row(kind, int(tile.get("col", -1)), int(tile.get("row", -1)),
                     str(tile.get("kind", "unknown")))

        if kind == "packet_connect":
            (r["recv_dir"], r["recv_idx"], r["recv_pktid"], r["recv_mask"],
             r["recv_msel"], r["recv_arbiter"]) = _slot(conn, "recv_slave")
            (r["dma_dir"], r["dma_idx"], r["dma_pktid"], r["dma_mask"],
             r["dma_msel"], r["dma_arbiter"]) = _slot(conn, "local_dma")
            r["fwd_dir"], r["fwd_idx"], r["fwd_mselen"], r["fwd_arbiter"] = _master(conn, "forward_master")
            r["preserve"] = 1 if conn.get("preserve_header") else 0
        elif kind == "circuit_connect":
            r["slave_dir"], r["slave_idx"] = _port(conn, "slave")
            r["master_dir"], r["master_idx"] = _port(conn, "master")
        elif kind in ("shim_ext_to_aie", "shim_aie_to_ext"):
            # single directional port on shim row 0; carry it as the master port
            r["master_dir"], r["master_idx"] = _port(conn, "port")
        elif kind == "shim_stream_switch_port":
            sm = conn.get("shim_master") or {}
            r["master_dir"] = str(sm.get("port", "NONE"))
            r["master_idx"] = int(sm.get("idx", -1))
        else:
            # unknown kind: keep the tile row with empty ports rather than drop.
            pass

        rows.append(r)

    # Deterministic ordering so rebuilds are diff-clean.
    rows.sort(
        key=lambda e: (
            e["col"],
            e["row"],
            e["kind"],
            e["recv_dir"],
            e["recv_idx"],
            e["recv_pktid"],
            e["dma_dir"],
            e["dma_idx"],
            e["dma_pktid"],
            e["fwd_dir"],
            e["fwd_idx"],
            e["slave_dir"],
            e["slave_idx"],
            e["master_dir"],
            e["master_idx"],
        )
    )
    return rows


def _cstr(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_header(model, rows):
    start_col = model.get("partition_start_col", model.get("startcol", -1))
    aie_gen = model.get("aie_gen", "")

    out = []
    w = out.append

    w("/* Auto-generated by resource_json_to_header.py -- do not edit. */")
    w("#ifndef AIE_RESOURCE_MAP_H")
    w("#define AIE_RESOURCE_MAP_H")
    w("")
    w("#include <stdio.h>")
    w("")
    w("struct AieResourceEntry {")
    w("    const char *kind;   /* connection kind (packet_connect, circuit_connect, ...) */")
    w("    int col;")
    w("    int row;")
    w("    const char *tile_kind;")
    w("    /* receive slave slot */")
    w("    const char *recv_dir;")
    w("    int recv_idx;")
    w("    int recv_pktid;")
    w("    int recv_mask;")
    w("    int recv_msel;")
    w("    int recv_arbiter;")
    w("    /* local DMA slave slot */")
    w("    const char *dma_dir;")
    w("    int dma_idx;")
    w("    int dma_pktid;")
    w("    int dma_mask;")
    w("    int dma_msel;")
    w("    int dma_arbiter;")
    w("    /* forward master port (mselen = MSelEn bitmask) */")
    w("    const char *fwd_dir;")
    w("    int fwd_idx;")
    w("    int fwd_mselen;")
    w("    int fwd_arbiter;")
    w("    /* circuit_connect / circuit_connect_pair / shim_* generic ports */")
    w("    const char *slave_dir;")
    w("    int slave_idx;")
    w("    const char *master_dir;")
    w("    int master_idx;")
    w("    int preserve_header;")
    w("};")
    w("")
    w("static const struct AieResourceEntry __aie_resource_map[] = {")
    for e in rows:
        w(
            "    { %s, %d, %d, %s, %s, %d, %d, %d, %d, %d, %s, %d, %d, %d, %d, %d, "
            "%s, %d, %d, %d, %s, %d, %s, %d, %d },"
            % (
                _cstr(e["kind"]),
                e["col"],
                e["row"],
                _cstr(e["tkind"]),
                _cstr(e["recv_dir"]),
                e["recv_idx"],
                e["recv_pktid"],
                e["recv_mask"],
                e["recv_msel"],
                e["recv_arbiter"],
                _cstr(e["dma_dir"]),
                e["dma_idx"],
                e["dma_pktid"],
                e["dma_mask"],
                e["dma_msel"],
                e["dma_arbiter"],
                _cstr(e["fwd_dir"]),
                e["fwd_idx"],
                e["fwd_mselen"],
                e["fwd_arbiter"],
                _cstr(e["slave_dir"]),
                e["slave_idx"],
                _cstr(e["master_dir"]),
                e["master_idx"],
                e["preserve"],
            )
        )
    w("};")
    w("")
    w("static const int __aie_resource_map_count =")
    w("    (int)(sizeof(__aie_resource_map) / sizeof(__aie_resource_map[0]));")
    w("")
    w("static inline void __Runtime_print_resource_map(void) {")
    w('    printf("[aie_resource_map] partition_start_col=%d aie_gen=%s entries=%d\\n",')
    w("           %d, %s, __aie_resource_map_count);" % (int(start_col), _cstr(str(aie_gen))))
    w("    for (int i = 0; i < __aie_resource_map_count; ++i) {")
    w("        const struct AieResourceEntry *e = &__aie_resource_map[i];")
    w('        printf("[aie_resource_map] #%d %s tile(%d,%d,%s)"')
    w('               " recv[%s:%d pktid=%d mask=0x%x msel=%d arb=%d]"')
    w('               " dma[%s:%d pktid=%d mask=0x%x msel=%d arb=%d]"')
    w('               " fwd[%s:%d mselen=%d arb=%d]"')
    w('               " slave[%s:%d] master[%s:%d] preserve=%d\\n",')
    w("               i, e->kind, e->col, e->row, e->tile_kind,")
    w("               e->recv_dir, e->recv_idx, e->recv_pktid, e->recv_mask,")
    w("               e->recv_msel, e->recv_arbiter,")
    w("               e->dma_dir, e->dma_idx, e->dma_pktid, e->dma_mask,")
    w("               e->dma_msel, e->dma_arbiter,")
    w("               e->fwd_dir, e->fwd_idx, e->fwd_mselen, e->fwd_arbiter,")
    w("               e->slave_dir, e->slave_idx,")
    w("               e->master_dir, e->master_idx, e->preserve_header);")
    w("    }")
    w("}")
    w("")
    w("#endif /* AIE_RESOURCE_MAP_H */")
    w("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert routingresourcemap.json to a C header.")
    ap.add_argument("json", help="path to routingresourcemap.json")
    ap.add_argument("--out", required=True, help="output header path (aie_resource_map.h)")
    args = ap.parse_args(argv)

    try:
        with open(args.json, "r") as f:
            model = json.load(f)
    except (OSError, ValueError) as exc:
        sys.stderr.write("resource_json_to_header: cannot read %s: %s\n" % (args.json, exc))
        return 1

    rows = build_entries(model)
    text = render_header(model, rows)

    try:
        with open(args.out, "w") as f:
            f.write(text)
    except OSError as exc:
        sys.stderr.write("resource_json_to_header: cannot write %s: %s\n" % (args.out, exc))
        return 1

    sys.stderr.write(
        "resource_json_to_header: wrote %s (%d entries)\n" % (args.out, len(rows))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
