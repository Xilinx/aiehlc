#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""
work2provenance.py — generate debug-UI provenance JSON files from an
aiecompiler Work directory.

Produces three files consumed by aiehlc_aiesim's schedule_view.py:
  dfscheduleprovenancemap.json   — per-tile DMA channels + BD chains
  dmaphopprovenacemap.json       — end-to-end communication paths
  routingprovenancemap.json      — AIE stream-switch routing

Also copies host.cc (aie_control.cpp) and kernel.cc for line attribution.

Data sources (preference: JSON over C++):
  ps/c_rts/aie_control_config.json — HW geometry, GMIOs, kernel mapping
  reports/compiler_report.json    — BD offset/len/locks, channel assignment
  reports/dma_lock_report.json    — port-variable names per channel
  temp/router_soln.json           — stream-switch routing trees
  ps/c_rts/aie_control.cpp       — only used as text copy for host.cc

Usage:
  python work2provenance.py <work_dir> [--out <out_dir>]

Default out_dir: <work_dir>/../worklocal
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hw_gen_str(hw_gen):
    return {1: "Gen1", 2: "Gen2", 5: "Gen5"}.get(int(hw_gen), f"Gen{hw_gen}")


def _tile_type(row, dcfg):
    if row == dcfg.get("shim_row", 0):
        return "shim"
    if row < dcfg["aie_tile_row_start"]:
        return "mem"
    return "core"


# ---------------------------------------------------------------------------
# Parse compiler_report.json for BD chains and channel info
# ---------------------------------------------------------------------------

def parse_compiler_report(report, dcfg, port2tile=None):
    """Build BD chains and channel list from a loaded compiler_report dict.

    Channels are attributed to the owning kernel's tile via ``port2tile`` (the DMA
    hardware may live on an adjacent tile; keying by the kernel tile keeps the
    channel, its kernel, and its window buffers consistent).  When ``port2tile``
    lacks a port, the DMA hardware row is used as a fallback.

    Returns:
      bds      : {(col, abs_row): {bd_id: {bd_id, buffer_offset, len,
                                           acquire_lock, release_lock, next_bd}}}
      channels : [(col, abs_row, ch_idx, direction, start_bd)]
    """
    port2tile = port2tile or {}
    aie_tile_row_start = dcfg["aie_tile_row_start"]
    bds = defaultdict(dict)
    channels = []

    port_mapping = report.get("mapping", {}).get("portInstanceMapping", {})
    for port_id, port_data in port_mapping.items():
        buf_infos = port_data.get("bufferInfo", [])
        dma_info  = port_data.get("dmaInfo", {})
        if not buf_infos or not dma_info:
            continue

        direction    = "S2MM" if dma_info["direction"] == "s2mm" else "MM2S"
        ch           = dma_info["channel"]
        bd_ids       = dma_info.get("bufferDescriptor", [])

        # Attribute to the owning kernel's tile when known, else the DMA hw tile.
        ktile = port2tile.get(port_id)
        if ktile:
            col, abs_row = ktile
        else:
            col     = dma_info["column"]
            abs_row = aie_tile_row_start + dma_info["row"]

        # Standard double-buffer lock value convention:
        #   S2MM (DMA writing INTO core memory):
        #     acquire when empty (val=-1), release as full (val=1)
        #   MM2S (DMA reading OUT of core memory):
        #     acquire when full (val=1),  release as empty (val=-1)
        if direction == "S2MM":
            acq_val, rel_val = -1, 1
        else:
            acq_val, rel_val = 1, -1

        n = len(bd_ids)
        for i, (buf, bd_id) in enumerate(zip(buf_infos, bd_ids)):
            lock_pair   = buf.get("lock", [-1, -1])
            acq_lock_id = lock_pair[0]
            rel_lock_id = lock_pair[1] if len(lock_pair) > 1 else lock_pair[0]

            # Circular next_bd: last BD wraps back to first
            next_bd_id = bd_ids[(i + 1) % n] if n > 1 else -1

            bds[(col, abs_row)][bd_id] = {
                "bd_id":         bd_id,
                "buffer_offset": buf["offset"],
                "len":           buf["size"],
                "enable_packet": False,
                "packet_id":     0,
                "next_bd":       next_bd_id,
                "acquire_lock":  [{"id": acq_lock_id, "val": acq_val}],
                "release_lock":  [{"id": rel_lock_id, "val": rel_val}],
            }

        start_bd = bd_ids[0] if bd_ids else 0
        channels.append((col, abs_row, ch, direction, start_bd))

    return dict(bds), channels


# ---------------------------------------------------------------------------
# Build stream connections from router_soln.json same-tile port pairs
# ---------------------------------------------------------------------------

_RE_PORT = re.compile(
    r'^(?P<sm>[SM])_'
    r'(?:(?P<shim>SHIM)_)?'
    r'(?:(?P<memtile>MEMTILE)_)?'
    r'(?:(?P<dma_dir>MM2S|S2MM)_)?'
    r'(?P<dir>DMA|SOUTH|NORTH|EAST|WEST|CORE)?'
    r'(?:_ch(?P<ch>\d+))?'
    r'_C(?P<col>\d+)'
    r'(?:_R(?P<row>\d+))?$'
)


def parse_port_name(name):
    """Parse a router_soln port name string.

    Returns dict with keys: is_slave, tile_class, rel_row, dir, ch, col
    Returns None if unparseable.
    """
    m = _RE_PORT.match(name)
    if not m:
        return None
    is_slave     = m.group('sm') == 'S'
    shim_flag    = m.group('shim')
    memtile_flag = m.group('memtile')
    dma_dir      = m.group('dma_dir')
    dir_type     = m.group('dir')
    ch_str       = m.group('ch')
    col          = int(m.group('col'))
    row_str      = m.group('row')

    if shim_flag:
        tile_class = "shim"
        rel_row = 0
    elif memtile_flag:
        tile_class = "mem"
        rel_row = int(row_str) if row_str is not None else 0
    elif row_str is not None:
        tile_class = "aie"
        rel_row = int(row_str)
    else:
        tile_class = "shim"
        rel_row = 0

    actual_dir = dma_dir if dma_dir else (dir_type or "")

    return {
        "is_slave":   is_slave,
        "tile_class": tile_class,
        "rel_row":    rel_row,
        "dir":        actual_dir,
        "ch":         int(ch_str) if ch_str is not None else 0,
        "col":        col,
    }


def port_abs_row(p, dcfg):
    """Convert a parsed port's relative row to an absolute tile row."""
    if p["tile_class"] == "shim":
        return dcfg.get("shim_row", 0)
    if p["tile_class"] == "mem":
        return dcfg["mem_tile_row_start"] + p["rel_row"]
    if p["tile_class"] == "aie":
        return dcfg["aie_tile_row_start"] + p["rel_row"]
    return p["rel_row"]


# ---------------------------------------------------------------------------
# Derive shim mux/demux from GMIO metadata
# ---------------------------------------------------------------------------

def build_shim_mux_demux(gmios, dcfg):
    """Build shim mux/demux entries from aie_control_config.json GMIOs.

    GMIO type 0 (input, MM2S/push): EnableShimDmaToAieStrmPort → shim_mux
    GMIO type 1 (output, S2MM/pull): EnableAieToShimDmaStrmPort → shim_demux

    Returns:
      shim_mux   : [(col, row, stream_id)]
      shim_demux : [(col, row, stream_id)]
    """
    shim_row = dcfg.get("shim_row", 0)
    shim_mux   = []
    shim_demux = []
    for g in gmios.values():
        col       = g["shim_column"]
        stream_id = g["stream_id"]
        gtype     = g.get("type", 0)   # 0=input(push/MM2S), 1=output(pull/S2MM)
        if gtype == 0:
            shim_mux.append((col, shim_row, stream_id))
        else:
            shim_demux.append((col, shim_row, stream_id))
    return shim_mux, shim_demux


# ---------------------------------------------------------------------------
# Net direction + helper utilities
# ---------------------------------------------------------------------------

def _net_direction_from_tree(tree, dcfg):
    """Determine push/pull from the first hop source port.

    push: shim is the producer (data flows shim → cores)
    pull: a core DMA (MM2S) is the producer (data flows cores → shim)
    """
    if not tree:
        return "push"
    src_name = tree[0][0]
    p = parse_port_name(src_name)
    if p is None:
        return "push"
    if p["dir"] == "MM2S" and p["tile_class"] != "shim":
        return "pull"
    if p["tile_class"] == "shim":
        return "push"
    return "push"


def _shim_stream_id_from_tree(tree, direction, dcfg):
    """Extract the stream_id on the shim port for this net."""
    if direction == "push":
        p = parse_port_name(tree[0][0])
    else:
        p = parse_port_name(tree[-1][1])
    if p and p["tile_class"] == "shim":
        return p["ch"]
    return None


def _gmio_for_net(gmios, shim_col, stream_id, direction):
    """Find GMIO matching column + stream_id; fall back to direction match."""
    for g in gmios.values():
        if g.get("shim_column") == shim_col and g.get("stream_id") == stream_id:
            return g
    gmio_type = 0 if direction == "push" else 1
    for g in gmios.values():
        if g.get("shim_column") == shim_col and g.get("type", 0) == gmio_type:
            return g
    return None


def _net_flow_index(net_name):
    """net0→0, net1→1, etc."""
    m = re.search(r'(\d+)$', net_name)
    return int(m.group(1)) if m else 0


def _infer_direction(src_dir, dst_dir):
    """Fallback direction inference from stream-switch port directions."""
    if dst_dir == "DMA":
        return "push"
    if src_dir == "DMA":
        return "pull"
    if src_dir == "SOUTH" and dst_dir == "NORTH":
        return "push"
    return "pull"


# ---------------------------------------------------------------------------
# Build BD chain (follow next_bd links from start)
# ---------------------------------------------------------------------------

def build_bd_chain(tile_bds, start_bd):
    chain = []
    visited = set()
    cur = start_bd
    while cur != -1 and cur not in visited:
        bd = tile_bds.get(cur)
        if bd is None:
            break
        chain.append(bd)
        visited.add(cur)
        cur = bd.get("next_bd", -1)
    return chain


# ---------------------------------------------------------------------------
# Build net_flow_map: (col, abs_row, ch, direction) → flow_index
# ---------------------------------------------------------------------------

def build_net_flow_map(router_soln, dcfg, gmios):
    """Map each DMA channel endpoint (col, abs_row, ch, dir) → flow_index."""
    shim_row = dcfg.get("shim_row", 0)
    net_flow = {}

    for net_name, net_data in router_soln.items():
        fi = _net_flow_index(net_name)
        tree = net_data.get("tree", [])
        direction = _net_direction_from_tree(tree, dcfg)

        for src_name, dst_name in tree:
            for port_name in (src_name, dst_name):
                p = parse_port_name(port_name)
                if p is None:
                    continue
                if p["dir"] not in ("MM2S", "S2MM", "DMA"):
                    continue
                row_abs = port_abs_row(p, dcfg)
                dma_dir = p["dir"]
                if dma_dir == "DMA":
                    dma_dir = "MM2S" if direction == "pull" else "S2MM"
                net_flow[(p["col"], row_abs, p["ch"], dma_dir)] = fi

        # Map shim GMIO channel to flow_index via stream_id
        stream_id = _shim_stream_id_from_tree(tree, direction, dcfg)
        ref_name = tree[0][0] if direction == "push" else tree[-1][1]
        sp = parse_port_name(ref_name)
        if sp and stream_id is not None:
            shim_col = sp["col"]
            gmio_dir = "MM2S" if direction == "push" else "S2MM"
            for g in gmios.values():
                if (g.get("shim_column") == shim_col and
                        g.get("stream_id") == stream_id):
                    ch = g.get("channel_number", 0)
                    net_flow[(shim_col, shim_row, ch, gmio_dir)] = fi
                    break

    return net_flow


# ---------------------------------------------------------------------------
# Generate dfscheduleprovenancemap.json
# ---------------------------------------------------------------------------

def gen_dfschedule(dcfg, bds, channels, gmios, aie_gen, startcol,
                   kernel_mapping, net_flow_map, dma_lock_report,
                   flow_summary=None, win_base=None, tile_artifacts=None,
                   chan2port=None, port2info=None, kernel_tiles=None):
    """Build the dfscheduleprovenancemap structure."""
    tile_artifacts = tile_artifacts or {}
    chan2port = chan2port or {}
    port2info = port2info or {}
    kernel_tiles = kernel_tiles or {}

    # Build (col, abs_row, ch, direction) → port_variable_name from dma_lock_report
    chan_to_port = {}
    for direction_key in ("S2MM", "MM2S"):
        for entry in dma_lock_report.get(direction_key, []):
            port_name = entry.get("KernelPort", {}).get("VariableName", "")
            for bi in entry.get("BufferInfo", []):
                col          = bi["Column"]
                row_compiler = bi["Row"]
                abs_row      = row_compiler + dcfg["aie_tile_row_start"]
                ch           = bi["Channel"]
                chan_to_port[(col, abs_row, ch, direction_key)] = port_name

    # Shim tiles from GMIOs.
    # A single shim column can host several GMIOs of the same direction on
    # different channels (e.g. col 0 with MM2S ch0 AND MM2S ch1), so key by
    # (direction, channel) rather than direction alone to avoid overwriting.
    shim_tiles = {}   # col → {(direction, ch): gmio_entry}
    for gname, g in gmios.items():
        col       = g["shim_column"]
        gtype     = g.get("type", 0)   # 0=input(MM2S), 1=output(S2MM)
        direction = "MM2S" if gtype == 0 else "S2MM"
        ch        = g.get("channel_number", 0)
        shim_tiles.setdefault(col, {})[(direction, ch)] = g

    # Collect all tile keys
    tile_channels = defaultdict(list)
    for col, row, ch_idx, direction, start_bd in channels:
        tile_channels[(col, row)].append((ch_idx, direction, start_bd))

    all_tile_keys = set(tile_channels.keys())
    for col in shim_tiles:
        all_tile_keys.add((col, dcfg.get("shim_row", 0)))
    # Every kernel tile appears, even those that only communicate via shared
    # memory (no DMA channels) — otherwise mid-array compute tiles are invisible.
    all_tile_keys.update(kernel_tiles.keys())

    tiles_out = []
    for (col, row) in sorted(all_tile_keys):
        ttype = _tile_type(row, dcfg)
        dma_channels_out = []

        if ttype == "shim":
            dirs_for_col = shim_tiles.get(col, {})
            for (direction, _ch_key), g in sorted(dirs_for_col.items()):
                ch_idx     = g.get("channel_number", 0)
                stream_id  = g.get("stream_id", 0)
                burst      = g.get("burst_length_in_16byte", 1)
                burst_bytes = burst * 16
                logical_name = g.get("logical_name", g.get("name", ""))
                fi = net_flow_map.get((col, row, ch_idx, direction), 0)

                verb = "send" if direction == "MM2S" else "receive"
                contract = (f"{direction} ch{ch_idx}: GMIO {verb}, "
                            f"stream_id={stream_id}, burst={burst_bytes}B, "
                            f"logical={logical_name}")

                dma_channels_out.append({
                    "channel":             ch_idx,
                    "direction":           direction,
                    "enable_out_of_order": False,
                    "flow_index":          fi,
                    "stream_id":           stream_id,
                    "logical_name":        logical_name,
                    "bd_chain":            [],
                    "start_io":            [{"repeat_count": 1}],
                    "contract":            contract,
                })
        else:
            for (ch_idx, direction, start_bd) in sorted(tile_channels.get((col, row), [])):
                tile_bds = bds.get((col, row), {})
                chain = build_bd_chain(tile_bds, start_bd)
                # Prefer the flow_index resolved from the owning port's net; fall
                # back to the geometry-based net_flow_map.
                port_id = chan2port.get((col, row, ch_idx, direction))
                info = port2info.get(port_id) if port_id else None
                fi = (info.get("flow_index") if info
                      else net_flow_map.get((col, row, ch_idx, direction), 0))

                acq_id = rel_id = "?"
                length = "?"
                if chain:
                    acq_id = chain[0].get("acquire_lock", [{"id": "?"}])[0].get("id", "?")
                    rel_id = chain[0].get("release_lock", [{"id": "?"}])[0].get("id", "?")
                    length = chain[0].get("len", "?")
                verb = "receive" if direction == "S2MM" else "send"
                pingpong = "ping-pong " if len(chain) == 2 else ""
                contract = (f"{direction} ch{ch_idx}: {pingpong}{verb}, "
                            f"{length}B each, lock {acq_id}/{rel_id}")

                port_var = chan_to_port.get((col, row, ch_idx, direction), "")
                ch_entry = {
                    "channel":             ch_idx,
                    "direction":           direction,
                    "enable_out_of_order": False,
                    "flow_index":          fi,
                    "bd_chain":            chain,
                    "start_io":            [{"repeat_count": 1}],
                    "contract":            contract,
                }
                if port_var:
                    ch_entry["kernel_port"] = port_var
                # Resolve the kernel window/buffers by port name (deterministic,
                # address-independent). Consumed by schedule_view's kernel_match.
                if info:
                    ch_entry["kernel_port_id"] = port_id
                    if info.get("window"):
                        ch_entry["kernel_window"] = info["window"]
                    if info.get("buffers"):
                        ch_entry["kernel_buffers"] = info["buffers"]
                dma_channels_out.append(ch_entry)

        tile_entry = {
            "col":          col,
            "row":          row,
            "type":         ttype,
            "dma_channels": dma_channels_out,
        }
        # Attach per-tile kernel.cc / .bcf / win_base for core tiles so the debug
        # UI can show the correct kernel source and buffer map for each tile.
        art = tile_artifacts.get((col, row))
        if art:
            if art.get("kernel_cc"):
                tile_entry["kernel_cc"] = art["kernel_cc"]
            if art.get("bcf"):
                tile_entry["bcf"] = art["bcf"]
            if art.get("win_base") is not None:
                tile_entry["win_base"] = art["win_base"]
        tiles_out.append(tile_entry)

    # load_kernel_group: one group per kernel tile (placement from coreInfo).
    # kernel_for_tile() matches on these (col, row) coordinates.
    load_kernel_groups = []
    for (col, abs_row), info in sorted(kernel_tiles.items()):
        callee = info.get("callee", "")
        load_kernel_groups.append({
            "callee":    callee,
            "function":  callee,
            "tiles":     [{"col": col, "row": abs_row}],
            "exec_tile": {"col": col, "row": abs_row},
        })

    result = {
        "version":           1,
        "startcol":          startcol,
        "aie_gen":           aie_gen,
        "module_attrs":      [],
        "tiles":             tiles_out,
        "load_kernel_group": load_kernel_groups,
    }
    if flow_summary is not None:
        result["flow_summary"] = flow_summary
    if win_base is not None:
        result["win_base"] = win_base
    return result


# ---------------------------------------------------------------------------
# Generate dmaphopprovenacemap.json
# ---------------------------------------------------------------------------

def gen_dmaphop(dcfg, router_soln, gmios, net_flow_map, aie_gen, startcol,
                stream_info=None, shmem_edges=None, kernel_tiles=None):
    """Build dmaphopprovenacemap structure from router_soln.json.

    In addition to the stream-routed shim<->core hops, this appends the
    shared-memory hops of the systolic array to each push flow: the DMA-hardware
    -> kernel bridge (data crosses to the compute tile via shared memory) and the
    kernel -> kernel chain along the row.  These render as dashed shmem links in
    the device map, and carry an explicit ``hop_type`` so schedule_view does not
    have to re-classify them.
    """
    shim_row     = dcfg.get("shim_row", 0)
    stream_info  = stream_info or {}
    shmem_edges  = shmem_edges or []
    kernel_tiles = kernel_tiles or {}

    # Build a col -> set of known abs_rows index so we can fix off-by-one rows
    # in router_soln AIE port names (aiecompiler sometimes emits R1 for the first
    # core tile when the correct compiler-relative row is R0, yielding abs_row =
    # aie_tile_row_start + 1 instead of aie_tile_row_start + 0).
    _ktile_rows_by_col = {}
    for (c, r) in kernel_tiles:
        _ktile_rows_by_col.setdefault(c, set()).add(r)

    # Forward adjacency for the systolic chain (src kernel tile -> dst kernel tile).
    shmem_next = {}
    for (s, d) in shmem_edges:
        shmem_next.setdefault(s, d)

    def _chain_from(start):
        """Ordered list of kernel tiles reachable from start via shmem edges."""
        chain = [start]
        seen  = {start}
        cur   = start
        while cur in shmem_next and shmem_next[cur] not in seen:
            cur = shmem_next[cur]
            chain.append(cur)
            seen.add(cur)
        return chain

    def _shmem_hop(fi, fc, fr, tc, tr, kind):
        # kind: "window" = ping-pong buffered window between two kernels;
        #       "dma"    = direct shared-memory access bridging a DMA-hardware
        #                  tile and the adjacent compute tile (the last hops).
        return {
            "from":       f"f{fi}_shmemOut({fc},{fr})",
            "to":         f"f{fi}_shmemIn({tc},{tr})",
            "hop_type":   "shmem",
            "shmem_kind": kind,
        }

    comm_paths = []
    for net_name, net_data in router_soln.items():
        tree = net_data.get("tree", [])
        if not tree:
            continue

        fi        = _net_flow_index(net_name)
        direction = _net_direction_from_tree(tree, dcfg)

        # Shim column + stream_id from routing tree
        if direction == "push":
            shim_port_name = tree[0][0]
        else:
            shim_port_name = tree[-1][1]
        shim_p    = parse_port_name(shim_port_name)
        shim_col  = shim_p["col"] if shim_p else 0
        stream_id = shim_p["ch"] if shim_p else 0

        gmio = _gmio_for_net(gmios, shim_col, stream_id, direction)

        # Non-shim DMA endpoint
        if direction == "push":
            end_port_name = tree[-1][1]
        else:
            end_port_name = tree[0][0]
        end_p       = parse_port_name(end_port_name)
        end_col     = end_p["col"] if end_p else 0
        end_row_abs = port_abs_row(end_p, dcfg) if end_p else 0
        # Correct a known aiecompiler off-by-one: the router_soln sometimes uses
        # R1 for the first core tile (should be R0), making end_row_abs one too
        # high.  If the computed row is not a known kernel tile but there is
        # exactly one kernel tile in that column, use its row instead.
        if end_p and end_p["tile_class"] == "aie":
            col_rows = _ktile_rows_by_col.get(end_col, set())
            if (end_col, end_row_abs) not in kernel_tiles and len(col_rows) == 1:
                end_row_abs = next(iter(col_rows))

        # Single abstract hop: shim ↔ DMA tile.
        # _load_comm_paths in schedule_view.py builds nonshim_tiles from hop
        # (col,row) coords and looks up the routing group by that frozenset.
        # Including intermediate MEM tiles would blow up the set and break
        # the lookup — one abstract hop gives nonshim_tiles={(col, dma_row)}.
        if direction == "push":
            hops_out = [{
                "from": f"f{fi}_shimPortOut({shim_col},{shim_row})",
                "to":   f"f{fi}_corePortIn0({end_col},{end_row_abs})",
            }]
            # Shared-memory part of the input flow: the DMA-hw -> kernel bridge
            # (data crosses to the compute tile) then the kernel->kernel systolic
            # chain.  The final kernel -> output-DMA-hw hop is NOT added here: it
            # feeds the output stream and is attributed to that pull flow instead
            # (see the pull branch), so the last hop carries the stream it uses.
            si = stream_info.get(net_name)
            if si and si.get("kernel_tile"):
                ktile = si["kernel_tile"]
                hw    = si.get("dma_hw_tile") or ktile
                if hw != ktile:
                    hops_out.append(_shmem_hop(fi, hw[0], hw[1], ktile[0], ktile[1], "dma"))
                chain = _chain_from(ktile)
                for (fc, fr), (tc, tr) in zip(chain, chain[1:]):
                    hops_out.append(_shmem_hop(fi, fc, fr, tc, tr, "window"))
        else:
            # Stored reversed per dmaphop convention (schedule_view.py reverses)
            hops_out = [{
                "from": f"f{fi}_shimPortOut({shim_col},{shim_row})",
                "to":   f"f{fi}_corePortOut0({end_col},{end_row_abs})",
            }]
            # Shared-memory hop feeding this output stream: the producing kernel
            # (OneOutput) -> its output DMA-hardware tile.  Coloured with the pull
            # flow so the last hop is associated with the stream it actually uses.
            si = stream_info.get(net_name)
            if si and si.get("kernel_tile"):
                ktile = si["kernel_tile"]
                hw    = si.get("dma_hw_tile") or ktile
                if hw != ktile:
                    hops_out.append(_shmem_hop(fi, ktile[0], ktile[1], hw[0], hw[1], "dma"))

        if direction == "push":
            prod_ch = gmio.get("channel_number", 0) if gmio else 0
            producer_stage = {
                "role":       "producer",
                "tile":       {"col": shim_col, "row": shim_row, "type": "shim"},
                "port_sym":   f"f{fi}_shimPortIn",
                "channel":    prod_ch,
                "contract":   "produce via MM2S from DDR",
                "config_ref": f"gmio:{gmio.get('name', '') if gmio else ''}",
            }
            consumer_stage = {
                "role":  "consumer",
                "tiles": [{
                    "col":          end_col,
                    "row":          end_row_abs,
                    "type":         "core",
                    "port_sym":     f"f{fi}_corePortIn0",
                    "consumer_sym": f"f{fi}_consumer0",
                    "dma_port":     0,
                }],
            }
        else:
            producer_stage = {
                "role":     "producer",
                "tiles":    [{"col": end_col, "row": end_row_abs, "type": "core",
                              "port_sym": f"f{fi}_corePortOut0"}],
                "contract": "produce via MM2S from core",
            }
            cons_ch = gmio.get("channel_number", 0) if gmio else 0
            consumer_stage = {
                "role":       "consumer",
                "tile":       {"col": shim_col, "row": shim_row, "type": "shim"},
                "port_sym":   f"f{fi}_shimPortOut",
                "channel":    cons_ch,
                "contract":   "consume via S2MM to DDR",
                "config_ref": f"gmio:{gmio.get('name', '') if gmio else ''}",
            }

        channel_stage = {
            "role":       "channel",
            "hops":       hops_out,
            "contract":   f"lossless delivery, {len(hops_out)}-hop chain",
            "config_ref": "router_soln",
        }

        data_block = {}
        if gmio:
            data_block = {
                "logical_name":  gmio.get("logical_name", ""),
                "burst_bytes":   gmio.get("burst_length_in_16byte", 1) * 16,
                "element_type":  "i32",
                "element_bytes": 4,
            }

        comm_paths.append({
            "id":        net_name,
            "direction": direction,
            "data":      data_block,
            "stages":    [producer_stage, channel_stage, consumer_stage],
        })

    return {
        "version":             1,
        "startcol":            startcol,
        "aie_gen":             aie_gen,
        "module_attrs":        {},
        "communication_paths": comm_paths,
    }


# ---------------------------------------------------------------------------
# Generate routingprovenancemap.json
# ---------------------------------------------------------------------------

def gen_routing(dcfg, shim_mux, shim_demux, router_soln, aie_gen, startcol):
    """Build routingprovenancemap structure — one routing group per net.

    aiecompiler nets are point-to-point (one shim GMIO ↔ one core DMA), and two
    different nets can terminate at the same core tile on different channels while
    passing through the same physical stream-switch tiles.  A frozenset-of-tiles
    key (as aiehlc_aiesim's broadcast groups use) cannot distinguish them, so each
    group carries a `flow_index` field and schedule_view.py matches by that when
    present (falling back to the tile-set key for native aiehlc provenance).

    Each group's connections list drives edge reconstruction in
    routing_edges_for_flow():
      - a circuit_connect with master.dir == a compass direction draws an edge
        from that tile toward the neighbor in that direction;
      - a circuit_connect with master.dir == 'DMA' marks that tile as the DMA
        terminal (no edge) and contributes to the DMA-tile key.

    For a push net the sink core's connection is master=DMA directly.  For a pull
    net the source core's connection keeps its compass master (so the edge toward
    the shim is drawn) and gets an extra master=DMA marker entry so the terminal
    is recorded.
    """

    def _same_tile_pairs(tree):
        """Yield (parsed_src, parsed_dst) for tree hops that stay within one tile."""
        for src_name, dst_name in tree:
            sp = parse_port_name(src_name)
            dp = parse_port_name(dst_name)
            if sp is None or dp is None:
                continue
            if sp["col"] == dp["col"] and port_abs_row(sp, dcfg) == port_abs_row(dp, dcfg):
                yield sp, dp

    groups = []
    for gi, (net_name, net_data) in enumerate(sorted(
            router_soln.items(), key=lambda kv: _net_flow_index(kv[0]))):
        tree = net_data.get("tree", [])
        if not tree:
            continue
        fi        = _net_flow_index(net_name)
        direction = _net_direction_from_tree(tree, dcfg)

        connections = []
        dma_tiles   = set()

        # Shim mux/demux marker for this net's shim endpoint.
        if direction == "push":
            shim_port = parse_port_name(tree[0][0])
            if shim_port:
                connections.append({
                    "kind":      "shim_ext_to_aie",
                    "tile":      {"col": shim_port["col"], "row": dcfg.get("shim_row", 0)},
                    "stream_id": shim_port["ch"],
                })
        else:
            shim_port = parse_port_name(tree[-1][1])
            if shim_port:
                # Placed first so routing_edges_for_flow's split point puts every
                # circuit_connect in the pull section (conns[split_idx+1:]).
                connections.append({
                    "kind":      "shim_aie_to_ext",
                    "tile":      {"col": shim_port["col"], "row": dcfg.get("shim_row", 0)},
                    "stream_id": shim_port["ch"],
                })

        for sp, dp in _same_tile_pairs(tree):
            col     = sp["col"]
            row     = port_abs_row(sp, dcfg)
            s_dir   = sp["dir"]
            m_dir   = dp["dir"]

            if m_dir in ("S2MM", "MM2S"):
                # Push sink: data enters the DMA on this tile — terminal marker.
                connections.append({
                    "kind":   "circuit_connect",
                    "tile":   {"col": col, "row": row},
                    "slave":  {"dir": s_dir, "idx": sp["ch"]},
                    "master": {"dir": "DMA", "idx": dp["ch"]},
                })
                dma_tiles.add((col, row))
            elif s_dir in ("S2MM", "MM2S"):
                # Pull source: DMA emits toward m_dir (compass).  Only the one
                # real connection — the tile is keyed as a DMA endpoint through
                # the group's dma_tiles field.  Emitting a reversed twin here
                # put a connection on screen that host.cc never programs.
                connections.append({
                    "kind":   "circuit_connect",
                    "tile":   {"col": col, "row": row},
                    "slave":  {"dir": "DMA", "idx": sp["ch"]},
                    "master": {"dir": m_dir, "idx": dp["ch"]},
                })
                dma_tiles.add((col, row))
            else:
                # Pure relay tile.
                connections.append({
                    "kind":   "circuit_connect",
                    "tile":   {"col": col, "row": row},
                    "slave":  {"dir": s_dir, "idx": sp["ch"]},
                    "master": {"dir": m_dir, "idx": dp["ch"]},
                })

        tiles = []
        seen_t = set()
        for c in connections:
            t = c.get("tile", {})
            k = (t.get("col"), t.get("row"))
            if k not in seen_t:
                seen_t.add(k)
                tiles.append({"col": k[0], "row": k[1], "type": _tile_type(k[1], dcfg)})

        groups.append({
            "id":          f"group_{gi}",
            "memo":        "net",
            "scf_idx":     0,
            "ioid":        fi,
            "flow_index":  fi,
            "direction":   direction,
            "tiles":       tiles,
            "dma_tiles":   [list(t) for t in sorted(dma_tiles)],
            "connections": connections,
        })

    return {
        "version":        1,
        "startcol":       startcol,
        "aie_gen":        aie_gen,
        "module_attrs":   {},
        "routing_groups": groups,
    }


# ---------------------------------------------------------------------------
# Generate flow_summary for supply/demand balance analysis
# ---------------------------------------------------------------------------

def gen_flow_summary(router_soln, bds, channels, gmios, dcfg, port2info=None):
    """Build flow_summary consumed by compute_flow_balance() in schedule_view.py.

    Each flow (net) gets one entry with a list of participant channels and their
    BD lengths.  The shim channel is identified by GMIO metadata; core channels
    by the compiler_report BD data (attributed to the owning kernel tile).

    Schema per entry:
      {"flow_index": N, "direction": "push"|"pull",
       "entries": [{"io_direction": "MM2S"|"S2MM",
                    "tile_col": col, "tile_row": abs_row,
                    "channel": ch, "bd_len": bytes, "repeat_count": 1}]}
    """
    port2info = port2info or {}
    shim_row = dcfg.get("shim_row", 0)

    # flow_index -> core DMA port info (kernel tile, channel, direction).
    flow2core = {}
    for info in port2info.values():
        flow2core[info["flow_index"]] = info

    # Map (col, abs_row, ch, direction) → BD length from the first BD in the chain
    ch_bd_len = {}
    for (col, row), tile_bds in bds.items():
        for bd in tile_bds.values():
            pass  # handled below via channels
    # Build per-channel start BD → len lookup
    for col, row, ch_idx, direction, start_bd in channels:
        tile_bds = bds.get((col, row), {})
        bd = tile_bds.get(start_bd)
        bd_len = bd["len"] if bd else 0
        ch_bd_len[(col, row, ch_idx, direction)] = bd_len

    # Map GMIO metadata for shim channel lengths (shim BDs are runtime-configured;
    # the GMIO burst_length_in_16byte gives the per-fire chunk size)
    gmio_shim_len = {}  # (shim_col, ch, direction) → burst_bytes
    for g in gmios.values():
        col       = g["shim_column"]
        ch        = g.get("channel_number", 0)
        gtype     = g.get("type", 0)  # 0=input(MM2S push), 1=output(S2MM pull)
        direction = "MM2S" if gtype == 0 else "S2MM"
        burst_bytes = g.get("burst_length_in_16byte", 1) * 16
        gmio_shim_len[(col, ch, direction)] = burst_bytes

    flow_summary = []
    for net_name, net_data in router_soln.items():
        tree = net_data.get("tree", [])
        if not tree:
            continue

        fi        = _net_flow_index(net_name)
        direction = _net_direction_from_tree(tree, dcfg)

        entries = []

        # This net is point-to-point: exactly one core DMA endpoint, attributed to
        # the owning kernel tile (via port2info).  Only that one core channel
        # participates in this flow — adding every core channel of the right
        # direction would inflate the demand/supply totals by the number of flows.
        core_dir = "S2MM" if direction == "push" else "MM2S"
        core_col = core_row = core_ch = None
        core_bd_len = 0
        cinfo = flow2core.get(fi)
        if cinfo:
            core_col, core_row = cinfo["kernel_tile"]
            core_ch  = cinfo["channel"]
            core_bd_len = ch_bd_len.get((core_col, core_row, core_ch, core_dir), 0)

        # Shim participant: identify GMIO by (column, stream_id).  Use the core
        # BD length as the per-fire transfer size (GMIO burst_length_in_16byte is
        # an AXI burst configuration, not the transfer size; the actual per-fire
        # transfer matches what the core DMA writes/reads in one BD fire).
        if direction == "push":
            shim_port = parse_port_name(tree[0][0])
        else:
            shim_port = parse_port_name(tree[-1][1])
        if shim_port:
            shim_col = shim_port["col"]
            stream_id = shim_port["ch"]
            gmio = _gmio_for_net(gmios, shim_col, stream_id, direction)
            if gmio:
                shim_ch  = gmio.get("channel_number", 0)
                shim_dir = "MM2S" if direction == "push" else "S2MM"
                shim_len = core_bd_len or (gmio.get("burst_length_in_16byte", 1) * 16)
                entries.append({
                    "io_direction": shim_dir,
                    "tile_col":     shim_col,
                    "tile_row":     shim_row,
                    "channel":      shim_ch,
                    "bd_len":       shim_len,
                    "repeat_count": 1,
                })

        # Core participant: the single endpoint of this net.
        if core_col is not None:
            entries.append({
                "io_direction": core_dir,
                "tile_col":     core_col,
                "tile_row":     core_row,
                "channel":      core_ch,
                "bd_len":       core_bd_len,
                "repeat_count": 1,
            })

        if entries:
            flow_summary.append({
                "flow_index": fi,
                "direction":  direction,
                "entries":    entries,
            })

    return flow_summary


# ---------------------------------------------------------------------------
# Find .bcf linker script and extract WIN_BASE for the core tile
# ---------------------------------------------------------------------------

def find_bcf(work_dir, kernel_mapping):
    """Find the .bcf linker script for the first mapped core.

    Returns (path, win_base) or (None, None) if not found.
    """
    _re_reserved = re.compile(r'_reserved\s+DMb\s+0x0\s+(0x[0-9a-fA-F]+)')
    for entry in kernel_mapping:
        col          = entry["column"]
        row_compiler = entry["row"]
        label        = f"{col}_{row_compiler}"
        bcf_path     = os.path.join(work_dir, "aie", label, "scripts", f"{label}.bcf")
        if os.path.isfile(bcf_path):
            win_base = None
            try:
                with open(bcf_path) as f:
                    for line in f:
                        m = _re_reserved.search(line)
                        if m:
                            win_base = int(m.group(1), 16)
                            break
            except OSError:
                pass
            return bcf_path, win_base
    return None, None


# ---------------------------------------------------------------------------
# Collect kernel source files for kernel.cc
# ---------------------------------------------------------------------------

def collect_kernel_src(work_dir, kernel_mapping):
    """Return list of (col_row_label, src_path) for each mapped core."""
    src_files = []
    aie_dir = os.path.join(work_dir, "aie")
    if not os.path.isdir(aie_dir):
        return src_files
    for entry in kernel_mapping:
        col          = entry["column"]
        row_compiler = entry["row"]
        label        = f"{col}_{row_compiler}"
        src_path     = os.path.join(aie_dir, label, "src", f"{label}.cc")
        if os.path.isfile(src_path):
            src_files.append((label, src_path))
    return src_files


_RE_INCLUDE = re.compile(r'#include\s+"([^"]+\.cc)"')

# kernel.cc port declaration:
#   input_window_uint32 *input_window_i2_pi0 = (get_input_async_window_...(window_buf0_buf0d));
_RE_PORT_WIN = re.compile(
    r'_window_(i\d+_p[io]\d+)\s*=\s*\(?\s*get_\w*window\w*\(\s*(window_\w+)\s*\)')
# kernel invocation callee:  // Kernel call : i2:OneInput
_RE_KCALL = re.compile(r'//\s*Kernel call\s*:\s*(i\d+):(\w+)')


def build_graph_topology(work_dir, compiler_report, router_soln, dcfg):
    """Build the full kernel/connection topology from the compiler reports.

    Uses the authoritative sources:
      * ``blockInstanceMapping.coreInfo`` for kernel placement (the generated
        kernel.cc file label does NOT encode the physical tile position).
      * ``nets`` for every graph connection.  Nets present in router_soln.json are
        stream-routed (shim<->core via stream switches); the rest are shared-memory
        window connections between adjacent kernel tiles.
      * kernel.cc port declarations (``get_*_window(window_<bufs>)``) for the
        deterministic port -> window binding (buffer-address math is unreliable).

    Note: this design targets the "out of bounds" placement where a kernel's DMA
    hardware sits on an *adjacent* tile and the data crosses via shared memory.
    DMA channels are attributed to the owning kernel's tile (so tile == kernel ==
    window are consistent); the DMA hardware tile is recorded separately for the
    stream<->kernel bridge edge.

    Returns a dict with:
      kernel_tiles : {(col, abs_row): {instance, file_label, callee}}
      shmem_edges  : [((col,row), (col,row))]  deduped adjacent kernel pairs
      stream_info  : {net_name: {flow_index, direction, kernel_tile, dma_hw_tile}}
      chan2port    : {(kcol, krow, ch, direction): port_id}
      port2info    : {port_id: {kernel_tile, flow_index, window, buffers, callee,
                                file_label, dma_hw_tile, direction, channel}}
      tile2file    : {(col, abs_row): file_label}  for every kernel tile
      inst2tile    : {instance: (col, abs_row)}
    """
    aie_start = dcfg["aie_tile_row_start"]
    aie_dir   = os.path.join(work_dir, "aie")
    mapping   = compiler_report.get("mapping", {})
    bim       = mapping.get("blockInstanceMapping", {})
    pm        = mapping.get("portInstanceMapping", {})
    nets      = compiler_report.get("nets", {})

    # --- scan kernel sources: port -> (window, buffers); instance -> file/callee ---
    port2win    = {}
    inst2file   = {}
    inst2callee = {}
    if os.path.isdir(aie_dir):
        for label in sorted(os.listdir(aie_dir)):
            src = os.path.join(aie_dir, label, "src", f"{label}.cc")
            if not os.path.isfile(src):
                continue
            try:
                with open(src) as f:
                    text = f.read()
            except OSError:
                continue
            for m in _RE_KCALL.finditer(text):
                inst2file[m.group(1)]   = label
                inst2callee[m.group(1)] = m.group(2)
            for m in _RE_PORT_WIN.finditer(text):
                port_id   = m.group(1)
                win       = m.group(2)
                win_short = win[len("window_"):] if win.startswith("window_") else win
                port2win[port_id] = (win_short, re.findall(r'buf\d+d?', win_short))

    # --- instance -> kernel tile (coreInfo is the authoritative placement) ---
    inst2tile = {}
    for inst, b in bim.items():
        ci = b.get("coreInfo")
        if ci and ci.get("tile") == "aie":
            inst2tile[inst] = (ci["column"], aie_start + ci["row"])

    kernel_tiles = {}
    for inst, tile in inst2tile.items():
        kernel_tiles[tile] = {
            "instance":   inst,
            "file_label": inst2file.get(inst),
            "callee":     inst2callee.get(inst, ""),
        }

    stream_nets = set(router_soln.keys())

    # --- port -> flow_index (each DMA port belongs to exactly one net) ---
    port2flow = {}
    for name, n in nets.items():
        fi = _net_flow_index(name)
        for pkey in ("srcPort", "dstPort"):
            p = n.get(pkey)
            if p:
                port2flow[p] = fi

    # --- DMA channels attributed to their owning kernel tile ---
    chan2port = {}
    port2info = {}
    for port_id, pdata in pm.items():
        di = pdata.get("dmaInfo")
        if not di or di.get("tile") != "aie":
            continue
        prefix = port_id.split("_")[0]
        ktile  = inst2tile.get(prefix)
        if not ktile:
            continue
        direction = "S2MM" if di["direction"] == "s2mm" else "MM2S"
        ch        = di["channel"]
        chan2port[(ktile[0], ktile[1], ch, direction)] = port_id
        win, buffers = port2win.get(port_id, (None, []))
        port2info[port_id] = {
            "kernel_tile": ktile,
            "flow_index":  port2flow.get(port_id, 0),
            "window":      win,
            "buffers":     buffers,
            "callee":      inst2callee.get(prefix, ""),
            "file_label":  inst2file.get(prefix),
            "dma_hw_tile": (di["column"], aie_start + di["row"]),
            "direction":   direction,
            "channel":     ch,
        }

    # --- shared-memory edges: non-stream nets between distinct kernel tiles ---
    shmem_edges = []
    seen = set()
    for name, n in nets.items():
        if name in stream_nets:
            continue
        s = inst2tile.get(n.get("srcInstance"))
        d = inst2tile.get(n.get("dstInstance"))
        if s and d and s != d and (s, d) not in seen:
            seen.add((s, d))
            shmem_edges.append((s, d))

    # --- per stream net: kernel endpoint + DMA hardware tile (for the bridge) ---
    stream_info = {}
    for name in stream_nets:
        n  = nets.get(name, {})
        fi = _net_flow_index(name)
        direction = _net_direction_from_tree(router_soln[name].get("tree", []), dcfg)
        if direction == "push":
            kinst, port = n.get("dstInstance"), n.get("dstPort")
        else:
            kinst, port = n.get("srcInstance"), n.get("srcPort")
        ktile = inst2tile.get(kinst)
        di    = pm.get(port, {}).get("dmaInfo")
        if di and di.get("tile") == "aie":
            hw_col, hw_row = di["column"], aie_start + di["row"]
            # aiecompiler sometimes emits row+1 for the output DMA port of the
            # first core tile; if the result is not a known kernel tile but the
            # source kernel tile is in the same column, use the kernel tile row.
            if (hw_col, hw_row) not in kernel_tiles and ktile and ktile[0] == hw_col:
                hw_row = ktile[1]
            hw = (hw_col, hw_row)
        else:
            hw = ktile
        stream_info[name] = {
            "flow_index":  fi,
            "direction":   direction,
            "kernel_tile": ktile,
            "dma_hw_tile": hw,
        }

    tile2file = {t: info["file_label"]
                 for t, info in kernel_tiles.items() if info["file_label"]}

    return {
        "kernel_tiles": kernel_tiles,
        "shmem_edges":  shmem_edges,
        "stream_info":  stream_info,
        "chan2port":    chan2port,
        "port2info":    port2info,
        "tile2file":    tile2file,
        "inst2tile":    inst2tile,
    }


def _copy_kernel_with_includes(src_path, dst_path, candidate_bases):
    """Copy a kernel.cc to dst_path and copy its #include "*.cc" dependencies
    into the same destination directory tree (preserving the relative include
    path so _find_kernel_source resolves them next to the copied kernel.cc).
    """
    shutil.copy2(src_path, dst_path)
    dst_dir = os.path.dirname(dst_path)
    copied = [dst_path]
    try:
        with open(src_path) as fk:
            lines = fk.readlines()
    except OSError:
        return copied
    for line in lines:
        m = _RE_INCLUDE.search(line)
        if not m:
            continue
        rel = m.group(1)
        inc_src = next(
            (os.path.join(base, rel)
             for base in candidate_bases
             if os.path.isfile(os.path.join(base, rel))),
            None
        )
        if not inc_src:
            continue
        inc_dst = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(inc_dst), exist_ok=True)
        shutil.copy2(inc_src, inc_dst)
        copied.append(inc_dst)
    return copied


def collect_tile_artifacts(work_dir, out_dir, tile2file, dcfg):
    """Copy per-tile kernel.cc + .bcf into out_dir/kernels/<col>_<absrow>/ and
    return a map (col, abs_row) -> {kernel_cc, bcf, win_base} of out_dir-relative
    paths.  Each AIE core runs a different kernel with its own buffer layout, so
    the debug UI needs the kernel source and buffer map specific to each tile.

    tile2file maps a physical tile (col, abs_row) to its owning kernel source
    label (resolved by port name, since the generated file label does not encode
    the physical tile position).
    """
    aie_dir     = os.path.join(work_dir, "aie")
    example_root = os.path.dirname(work_dir)
    _re_reserved = re.compile(r'_reserved\s+DMb\s+0x0\s+(0x[0-9a-fA-F]+)')
    artifacts = {}
    if not os.path.isdir(aie_dir):
        return artifacts

    for (col, abs_row), label in sorted(tile2file.items()):
        src_path = os.path.join(aie_dir, label, "src", f"{label}.cc")
        bcf_path = os.path.join(aie_dir, label, "scripts", f"{label}.bcf")

        tile_dir = os.path.join(out_dir, "kernels", f"{col}_{abs_row}")
        rec = {}

        if os.path.isfile(src_path):
            os.makedirs(tile_dir, exist_ok=True)
            dst_kernel = os.path.join(tile_dir, "kernel.cc")
            _copy_kernel_with_includes(
                src_path, dst_kernel,
                [os.path.dirname(src_path), work_dir, example_root],
            )
            rec["kernel_cc"] = os.path.relpath(dst_kernel, out_dir)

        if os.path.isfile(bcf_path):
            os.makedirs(tile_dir, exist_ok=True)
            dst_bcf = os.path.join(tile_dir, "tile.bcf")
            shutil.copy2(bcf_path, dst_bcf)
            rec["bcf"] = os.path.relpath(dst_bcf, out_dir)
            try:
                with open(bcf_path) as f:
                    for line in f:
                        m = _re_reserved.search(line)
                        if m:
                            rec["win_base"] = int(m.group(1), 16)
                            break
            except OSError:
                pass

        if rec:
            artifacts[(col, abs_row)] = rec

    return artifacts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def work_to_provenance(work_dir, out_dir):
    work_dir = os.path.abspath(work_dir)
    out_dir  = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- aie_control_config.json: HW geometry, GMIOs, kernel mapping ---
    cfg_path = os.path.join(work_dir, "ps", "c_rts", "aie_control_config.json")
    with open(cfg_path) as f:
        ctrl_cfg = json.load(f)

    meta           = ctrl_cfg["aie_metadata"]
    dcfg           = meta["driver_config"]
    gmios          = meta.get("GMIOs", {})
    kernel_mapping = meta.get("TileMapping", {}).get("AIEKernelToTileMapping", [])
    hw_gen         = dcfg.get("hw_gen", 5)
    aie_gen        = _hw_gen_str(hw_gen)
    startcol       = dcfg.get("startcol", 0)

    # --- compiler_report.json (loaded once; drives BDs, topology, nets) ---
    compiler_report_path = os.path.join(work_dir, "reports", "compiler_report.json")
    with open(compiler_report_path) as f:
        compiler_report = json.load(f)

    # --- dma_lock_report.json: port variable names per channel ---
    dma_lock_path = os.path.join(work_dir, "reports", "dma_lock_report.json")
    with open(dma_lock_path) as f:
        dma_lock_report = json.load(f)

    # --- router_soln.json: routing trees ---
    router_path = os.path.join(work_dir, "temp", "router_soln.json")
    with open(router_path) as f:
        router_soln = json.load(f)

    # --- graph topology: kernel tiles, shmem edges, stream bridges, port info ---
    topo         = build_graph_topology(work_dir, compiler_report, router_soln, dcfg)
    kernel_tiles = topo["kernel_tiles"]
    shmem_edges  = topo["shmem_edges"]
    stream_info  = topo["stream_info"]
    chan2port    = topo["chan2port"]
    port2info    = topo["port2info"]
    tile2file    = topo["tile2file"]

    # BD chains attributed to the owning kernel tile.
    port2tile = {pid: info["kernel_tile"] for pid, info in port2info.items()}
    bds, channels = parse_compiler_report(compiler_report, dcfg, port2tile=port2tile)

    # --- derive shim mux/demux from JSON sources ---
    shim_mux, shim_demux   = build_shim_mux_demux(gmios, dcfg)

    # --- build net→flow_index map ---
    net_flow_map = build_net_flow_map(router_soln, dcfg, gmios)

    # --- copy .bcf now so win_base is available for dfschedule JSON ---
    # (moved up from below so we can embed win_base before writing JSONs)
    bcf_src, win_base = find_bcf(work_dir, kernel_mapping)

    # --- copy per-tile kernel.cc / .bcf so each core shows its own kernel ---
    tile_artifacts = collect_tile_artifacts(work_dir, out_dir, tile2file, dcfg)

    # --- generate the three provenance JSONs ---
    flow_summary = gen_flow_summary(router_soln, bds, channels, gmios, dcfg,
                                    port2info=port2info)
    dfschedule = gen_dfschedule(
        dcfg, bds, channels, gmios, aie_gen, startcol,
        kernel_mapping, net_flow_map, dma_lock_report,
        flow_summary=flow_summary,
        win_base=win_base,
        tile_artifacts=tile_artifacts,
        chan2port=chan2port,
        port2info=port2info,
        kernel_tiles=kernel_tiles,
    )
    dmaphop = gen_dmaphop(dcfg, router_soln, gmios, net_flow_map, aie_gen, startcol,
                          stream_info=stream_info, shmem_edges=shmem_edges,
                          kernel_tiles=kernel_tiles)
    routing = gen_routing(dcfg, shim_mux, shim_demux, router_soln, aie_gen, startcol)

    # --- write JSONs ---
    def write_json(name, data):
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  wrote {path}")

    write_json("dfscheduleprovenancemap.json", dfschedule)
    write_json("dmaphopprovenacemap.json", dmaphop)
    write_json("routingprovenancemap.json", routing)

    # --- write host.cc: aie_control.cpp + host_canonicalized wrapper ---
    # schedule_view.py looks for host_canonicalized() to attribute DMA init
    # lines to tiles.  We copy aie_control.cpp as-is (it contains XAie_TileLoc
    # / lock comments that drive attribution) and append a thin wrapper.
    aie_ctrl_path = os.path.join(work_dir, "ps", "c_rts", "aie_control.cpp")
    host_dst = os.path.join(out_dir, "host.cc")
    with open(aie_ctrl_path) as f:
        ctrl_src = f.read()

    graph_names = [g["name"] for g in meta.get("graphs", {}).values()]
    if not graph_names:
        graph_names = ["gradf"]
    init_calls = "\n".join(f"  {g}_init(\"\");" for g in graph_names)

    wrapper = (
        "\n// host_canonicalized: schedule-debug attribution wrapper"
        " (generated by work2provenance.py)\n"
        f"void host_canonicalized() {{\n{init_calls}\n}}\n"
    )
    with open(host_dst, "w") as f:
        f.write(ctrl_src)
        f.write(wrapper)
    print(f"  wrote {host_dst} (aie_control.cpp + host_canonicalized wrapper)")

    # --- copy top-level kernel.cc (first core) as the global DATA.kernel fallback ---
    # Per-tile kernel.cc files (out_dir/kernels/<col>_<row>/) are the primary
    # source; this top-level copy covers tiles without a per-tile artifact and
    # preserves the single-kernel aiehlc behaviour.
    kernel_srcs = collect_kernel_src(work_dir, kernel_mapping)
    if kernel_srcs:
        label, src_path = kernel_srcs[0]
        kernel_dst = os.path.join(out_dir, "kernel.cc")
        _copy_kernel_with_includes(
            src_path, kernel_dst,
            [os.path.dirname(src_path), work_dir, os.path.dirname(work_dir)],
        )
        print(f"  copied {src_path} → {kernel_dst}")
        if tile_artifacts:
            print(f"  copied {len(tile_artifacts)} per-tile kernel.cc/.bcf into {os.path.join(out_dir, 'kernels')}/")
    else:
        print("  warning: no kernel source files found")

    # --- copy .bcf linker script (buffer symbol map for channel↔window matching) ---
    if bcf_src:
        bcf_dst = os.path.join(out_dir, os.path.basename(bcf_src))
        shutil.copy2(bcf_src, bcf_dst)
        wb_str = f"0x{win_base:x}" if win_base is not None else "unknown"
        print(f"  copied {bcf_src} → {bcf_dst} (win_base={wb_str})")
    else:
        print("  note: no .bcf linker script found (channel-to-window matching may be limited)")
        win_base = None

    print(f"\nDone. Output in: {out_dir}")
    print("\nTo generate the debug UI:")
    print(f"  cd /scratch/staff/bkirinci/aiehlc_aiesim")
    print(f"  python src/tool/debug/schedule_view.py {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate provenance JSONs from an aiecompiler Work directory")
    ap.add_argument("work_dir", help="Path to the Work directory")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: <work_dir>/../worklocal)")
    args = ap.parse_args()

    out_dir = args.out
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.work_dir)), "worklocal")

    print(f"work_dir : {os.path.abspath(args.work_dir)}")
    print(f"out_dir  : {os.path.abspath(out_dir)}")
    work_to_provenance(args.work_dir, out_dir)


if __name__ == "__main__":
    main()
