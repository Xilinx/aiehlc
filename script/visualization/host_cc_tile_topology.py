#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified AIE visualizer: routing topology + DMA BD configuration.

Parses routing.cc (stream-switch connections) and host.cc (DMA buffer
descriptors, channels, locks) and produces a single interactive HTML
visualization showing both routing paths as SVG arrows and DMA details
inside tile cards.

Usage:
    python aie_visualizer.py --routing routing.cc --host host.cc
    python aie_visualizer.py --routing routing.cc --host host.cc -o viz.html
    python aie_visualizer.py --routing routing.cc                # routing only
    python aie_visualizer.py --host host.cc                      # DMA only
    python aie_visualizer.py --routing routing.cc --host host.cc --serve
    python aie_visualizer.py --routing routing.cc --host host.cc -m text
    python aie_visualizer.py --routing routing.cc --host host.cc -m png -o out.png
"""

import argparse
import http.server
import json
import re
import socket
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures — Routing
# ---------------------------------------------------------------------------

DIRECTION_OPPOSITES = {
    "NORTH": "SOUTH", "SOUTH": "NORTH",
    "EAST": "WEST", "WEST": "EAST",
}

DIRECTION_DELTA = {
    "NORTH": (0, 1), "SOUTH": (0, -1),
    "EAST": (1, 0), "WEST": (-1, 0),
}

TILE_TYPE_BY_ROW = lambda row: "Shim" if row == 0 else ("Mem" if row <= 2 else "AIE")

BLOCK_COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
    "#F44336", "#3F51B5", "#009688", "#CDDC39",
]


@dataclass
class PktSlaveSlot:
    tile: tuple
    port_type: str
    port_num: int
    slot_num: int
    pkt_id: int
    mask: int
    msel: int
    arbiter: int
    line: int


@dataclass
class PktMasterPort:
    tile: tuple
    port_type: str
    port_num: int
    drop_header: bool
    arbiter: int
    msel_en: int
    line: int


@dataclass
class CircuitConn:
    tile: tuple
    slave_type: str
    slave_port: int
    master_type: str
    master_port: int
    line: int


@dataclass
class ShimDmaPort:
    tile: tuple
    port_num: int
    direction: str
    line: int


@dataclass
class RoutingBlock:
    name: str
    circuit_conns: list = field(default_factory=list)
    pkt_slaves: list = field(default_factory=list)
    pkt_masters: list = field(default_factory=list)
    shim_dma: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data structures — DMA
# ---------------------------------------------------------------------------

@dataclass
class LockInfo:
    lock_id: int
    init_value: Optional[int] = None
    operate_value: Optional[int] = None


@dataclass
class BdConfig:
    var: str
    tile_var: str
    tile_loc: Tuple[int, int]
    buf_var: str
    buf_source: str
    bd_id: int
    offset: int
    length: int
    next_bd: int
    enable_packet: bool
    packet_id: int
    acquire_lock_id: int = -1
    acquire_lock_val: int = -1
    release_lock_id: int = -1
    release_lock_val: int = -1
    flow_id: int = -1  # assigned during flow grouping


@dataclass
class IoConfig:
    var: str
    tile_var: str
    tile_loc: Tuple[int, int]
    bd_var: str
    channel_id: int
    direction: str
    flow_id: int = -1  # assigned during flow grouping


@dataclass
class SliceInfo:
    var: str
    source_var: str
    off0: int
    off1: int
    size0: int
    size1: int


@dataclass
class TileDmaInfo:
    loc: Tuple[int, int]
    tile_type: str
    bds: List[BdConfig] = field(default_factory=list)
    ios: List[IoConfig] = field(default_factory=list)
    locks: Dict[int, LockInfo] = field(default_factory=dict)


@dataclass
class FlowInfo:
    """Represents a shim DMA flow: one shim tile channel+direction and its connected tiles."""
    flow_id: int
    shim_tile: Tuple[int, int]
    channel_id: int
    direction: str  # "S2MM" or "MM2S"
    connected_tiles: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"Tile({self.shim_tile[0]},{self.shim_tile[1]}) ch{self.channel_id} {self.direction}"


# ---------------------------------------------------------------------------
# Parsing — routing.cc
# ---------------------------------------------------------------------------

def parse_tile_loc(s: str) -> Optional[tuple]:
    m = re.search(r"XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def parse_routing_file(filepath: str) -> List[RoutingBlock]:
    with open(filepath) as f:
        lines = f.readlines()

    blocks: List[RoutingBlock] = []
    current_block: Optional[RoutingBlock] = None
    block_idx = 0

    for lineno_0, raw in enumerate(lines):
        lineno = lineno_0 + 1
        line = raw.strip()

        round_m = re.search(r"round\s+is\s+(\d+)", line, re.IGNORECASE)
        if round_m:
            block_idx = int(round_m.group(1))
            current_block = RoutingBlock(name=f"Round {block_idx}")
            blocks.append(current_block)
            continue

        if current_block is None and re.search(r"XAie_Strm|XAie_Enable", line):
            if re.match(r"(int32_t|void|AieRC)\s+XAie_\w+\s*\(", line):
                continue
            current_block = RoutingBlock(name=f"Block {block_idx}")
            blocks.append(current_block)

        if current_block is None:
            continue
        if re.match(r"(int32_t|void|AieRC)\s+XAie_\w+\s*\(", line):
            continue

        m = re.search(
            r"XAie_StrmConnCctEnable\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.circuit_conns.append(CircuitConn(
                tile=tile,
                slave_type=m.group(2), slave_port=int(m.group(3)),
                master_type=m.group(4), master_port=int(m.group(5)),
                line=lineno))
            continue

        m = re.search(
            r"XAie_StrmPktSwSlaveSlotEnable\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*"
            r"(?:\{[^}]*PktId\s*=\s*(\d+)[^}]*\}|XAie_PacketInit\(\s*(\d+)\s*,\s*\d+\s*\))"
            r"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            pkt_id = int(m.group(5) or m.group(6))
            current_block.pkt_slaves.append(PktSlaveSlot(
                tile=tile,
                port_type=m.group(2), port_num=int(m.group(3)),
                slot_num=int(m.group(4)),
                pkt_id=pkt_id,
                mask=int(m.group(7)), msel=int(m.group(8)),
                arbiter=int(m.group(9)),
                line=lineno))
            continue

        m = re.search(
            r"XAie_StrmPktSwMstrPortEnable\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\w+)\s*,\s*(\d+)\s*,\s*\w+\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.pkt_masters.append(PktMasterPort(
                tile=tile,
                port_type=m.group(2), port_num=int(m.group(3)),
                drop_header=True,
                arbiter=int(m.group(4)), msel_en=int(m.group(5)),
                line=lineno))
            continue

        m = re.search(
            r"XAie_EnableAieToShimDmaStrmPort\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.shim_dma.append(ShimDmaPort(
                tile=tile, port_num=int(m.group(2)),
                direction="aie_to_shim", line=lineno))
            continue

        m = re.search(
            r"XAie_EnableShimDmaToAieStrmPort\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.shim_dma.append(ShimDmaPort(
                tile=tile, port_num=int(m.group(2)),
                direction="shim_to_aie", line=lineno))
            continue

    return blocks


# ---------------------------------------------------------------------------
# Parsing — host.cc (DMA BDs)
# ---------------------------------------------------------------------------

def parse_host_cc(path: str) -> Tuple[Dict[Tuple[int, int], TileDmaInfo], List[FlowInfo]]:
    text = Path(path).read_text()

    tile_map: Dict[str, Tuple[int, int]] = {}
    for m in re.finditer(
        r"XAie_LocType\s+(\w+)\s*=\s*XAie_TileLoc\((\d+),\s*(\d+)\)", text
    ):
        tile_map[m.group(1)] = (int(m.group(2)), int(m.group(3)))

    buf_source: Dict[str, str] = {}
    for m in re.finditer(
        r"void\*\s+(\w+)\s*=\s*__runtime_buffer_arg\((.+?)\);", text
    ):
        buf_source[m.group(1)] = m.group(2).strip()

    slices: Dict[str, SliceInfo] = {}
    for m in re.finditer(
        r"PartitionTensor\s+(\w+)\s*=\s*__Runtime_extract_slice_contiguous_2d"
        r"\((\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", text,
    ):
        slices[m.group(1)] = SliceInfo(
            var=m.group(1), source_var=m.group(2),
            off0=int(m.group(3)), off1=int(m.group(4)),
            size0=int(m.group(5)), size1=int(m.group(6)),
        )

    def _slice_label(var: str) -> str:
        if var in slices:
            s = slices[var]
            return f"{var}[{s.off0}:{s.off0+s.size0}, {s.off1}:{s.off1+s.size1}]"
        return var

    bd_comment_locks: Dict[str, Tuple[int, int, int, int]] = {}
    bd_comment_pattern = re.compile(
        r"/\*\s*DMA BD Config:.*?"
        r"acquire_lock_id=(-?\d+).*?"
        r"acquire_lock_val=(-?\d+).*?"
        r"release_lock_id=(-?\d+).*?"
        r"release_lock_val=(-?\d+).*?\*/",
        re.DOTALL,
    )
    bd_comment_pattern_old = re.compile(
        r"/\*\s*DMA BD Config:.*?"
        r"acquire_lock_id=(-?\d+).*?"
        r"release_lock_id=(-?\d+).*?\*/",
        re.DOTALL,
    )
    bd_call_pattern = re.compile(
        r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
    )
    for cm in bd_comment_pattern.finditer(text):
        acq_id, acq_val = int(cm.group(1)), int(cm.group(2))
        rel_id, rel_val = int(cm.group(3)), int(cm.group(4))
        after = text[cm.end():]
        call_m = bd_call_pattern.search(after)
        if call_m:
            bd_comment_locks[call_m.group(1)] = (acq_id, acq_val, rel_id, rel_val)
    for cm in bd_comment_pattern_old.finditer(text):
        acq_id, rel_id = int(cm.group(1)), int(cm.group(2))
        after = text[cm.end():]
        call_m = bd_call_pattern.search(after)
        if call_m and call_m.group(1) not in bd_comment_locks:
            bd_comment_locks[call_m.group(1)] = (acq_id, -1, rel_id, -1)

    lock_init_map: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(dict)
    for m in re.finditer(
        r"XAie_LockSetValue\(\s*\w+,\s*XAie_TileLoc\(\s*(\d+),\s*(\d+)\s*\),"
        r"\s*XAie_LockInit\(\s*(\d+),\s*(\d+)\s*\)\)", text,
    ):
        loc = (int(m.group(1)), int(m.group(2)))
        lock_init_map[loc][int(m.group(3))] = int(m.group(4))
    for m in re.finditer(
        r"XAie_LockSetValue\(\s*\w+,\s*(\w+),\s*XAie_LockInit\(\s*(\d+),\s*(\d+)\s*\)\)",
        text,
    ):
        tvar = m.group(1)
        if tvar.startswith("XAie_TileLoc"):
            continue
        loc = tile_map.get(tvar, (-1, -1))
        lock_init_map[loc][int(m.group(2))] = int(m.group(3))

    lock_operate_map: Dict[str, Tuple[int, int]] = {}
    for m in re.finditer(
        r"XAie_DmaSetLock\(\s*&(\w+).*?"
        r"XAie_LockInit\(\s*(-?\d+),\s*(-?\d+)\s*\).*?"
        r"XAie_LockInit\(\s*(-?\d+),\s*(-?\d+)\s*\)",
        text, re.DOTALL,
    ):
        lock_operate_map[m.group(1)] = (int(m.group(3)), int(m.group(5)))

    bd_map: Dict[str, BdConfig] = {}
    bd_call_13 = re.compile(
        r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
        r"(\w+),\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
        r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
        r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
    )
    bd_call_10 = re.compile(
        r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
        r"(\w+),\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
        r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
    )
    for m in bd_call_13.finditer(text):
        tile_var = m.group(3)
        buf_ptr = m.group(4)
        bd_var = m.group(1)
        loc = tile_map.get(tile_var, (-1, -1))
        next_bd_raw = int(m.group(8))
        if next_bd_raw < 0 or next_bd_raw > 65535:
            next_bd_raw = -1
        bd = BdConfig(
            var=bd_var, tile_var=tile_var, tile_loc=loc,
            buf_var=buf_ptr,
            buf_source=_slice_label(buf_source.get(buf_ptr, buf_ptr)),
            bd_id=int(m.group(5)), offset=int(m.group(6)), length=int(m.group(7)),
            next_bd=next_bd_raw,
            enable_packet=int(m.group(9)) != 0, packet_id=int(m.group(10)),
            acquire_lock_id=int(m.group(11)), acquire_lock_val=int(m.group(12)),
            release_lock_id=int(m.group(13)), release_lock_val=int(m.group(14)),
        )
        bd_map[bd.var] = bd

    for m in bd_call_10.finditer(text):
        bd_var = m.group(1)
        if bd_var in bd_map:
            continue
        tile_var = m.group(3)
        buf_ptr = m.group(4)
        loc = tile_map.get(tile_var, (-1, -1))
        next_bd_raw = int(m.group(8))
        if next_bd_raw < 0 or next_bd_raw > 65535:
            next_bd_raw = -1
        acq_lock, acq_val, rel_lock, rel_val = -1, -1, -1, -1
        if bd_var in bd_comment_locks:
            acq_lock, acq_val, rel_lock, rel_val = bd_comment_locks[bd_var]
        bd = BdConfig(
            var=bd_var, tile_var=tile_var, tile_loc=loc,
            buf_var=buf_ptr,
            buf_source=_slice_label(buf_source.get(buf_ptr, buf_ptr)),
            bd_id=int(m.group(5)), offset=int(m.group(6)), length=int(m.group(7)),
            next_bd=next_bd_raw,
            enable_packet=int(m.group(9)) != 0, packet_id=int(m.group(10)),
            acquire_lock_id=acq_lock, acquire_lock_val=acq_val,
            release_lock_id=rel_lock, release_lock_val=rel_val,
        )
        bd_map[bd.var] = bd

    io_comment = re.compile(r"/\*\s*Create IO:.*?direction=(\w+)\s*\*/", re.DOTALL)
    # __Runtime_dma_createio_4(tile, bd_desc, channel_id, hw_bd_id, DMA_DIRECTION)
    # The 5th arg (DMA_S2MM / DMA_MM2S) is optional for backward compat.
    io_call = re.compile(
        r"io\s+(\w+)\s*=\s*__Runtime_dma_createio_4\("
        r"(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+)(?:,\s*\w+)?\)"
    )
    io_configs: List[IoConfig] = []
    for cm in io_comment.finditer(text):
        direction = cm.group(1)
        after = text[cm.end():]
        call_m = io_call.search(after)
        if call_m:
            tile_var = call_m.group(2)
            loc = tile_map.get(tile_var, (-1, -1))
            io_configs.append(IoConfig(
                var=call_m.group(1), tile_var=tile_var, tile_loc=loc,
                bd_var=call_m.group(3), channel_id=int(call_m.group(4)),
                direction=direction,
            ))

    # --- Build flow groups ---
    # A "flow" starts at each shim tile IO (row==0) and includes all subsequent
    # non-shim IOs until the next shim IO. This captures the fan-out pattern:
    # shim MM2S -> [compute S2MM tiles], shim S2MM <- [compute MM2S tiles].
    flows: List[FlowInfo] = []
    flow_id = 0
    current_flow: Optional[FlowInfo] = None
    io_to_flow: Dict[str, int] = {}  # io var -> flow_id

    for io in io_configs:
        is_shim = (io.tile_loc[1] == 0)
        if is_shim:
            current_flow = FlowInfo(
                flow_id=flow_id,
                shim_tile=io.tile_loc,
                channel_id=io.channel_id,
                direction=io.direction,
            )
            flows.append(current_flow)
            io.flow_id = flow_id
            io_to_flow[io.var] = flow_id
            flow_id += 1
        elif current_flow is not None:
            current_flow.connected_tiles.append(io.tile_loc)
            io.flow_id = current_flow.flow_id
            io_to_flow[io.var] = current_flow.flow_id

    # Map BD vars to flow_id via their IO associations
    bd_var_to_flow: Dict[str, int] = {}
    for io in io_configs:
        if io.flow_id >= 0 and io.bd_var in bd_map:
            bd_map[io.bd_var].flow_id = io.flow_id
            bd_var_to_flow[io.bd_var] = io.flow_id
            # Also tag chained BDs (next_bd chains)
            bd = bd_map[io.bd_var]
            visited_bds = {bd.bd_id}
            while bd.next_bd >= 0 and bd.next_bd not in visited_bds:
                visited_bds.add(bd.next_bd)
                # Find the chained BD on the same tile
                for other_bd in bd_map.values():
                    if other_bd.tile_loc == bd.tile_loc and other_bd.bd_id == bd.next_bd:
                        other_bd.flow_id = io.flow_id
                        break
                else:
                    break
                bd = other_bd

    tiles: Dict[Tuple[int, int], TileDmaInfo] = {}
    for bd in bd_map.values():
        loc = bd.tile_loc
        if loc not in tiles:
            tiles[loc] = TileDmaInfo(loc=loc, tile_type=TILE_TYPE_BY_ROW(loc[1]))
        tiles[loc].bds.append(bd)
    for io in io_configs:
        loc = io.tile_loc
        if loc not in tiles:
            tiles[loc] = TileDmaInfo(loc=loc, tile_type=TILE_TYPE_BY_ROW(loc[1]))
        tiles[loc].ios.append(io)

    for info in tiles.values():
        info.bds.sort(key=lambda b: b.bd_id)

    for info in tiles.values():
        loc = info.loc
        for bd in info.bds:
            if bd.acquire_lock_id >= 0:
                if bd.acquire_lock_id not in info.locks:
                    info.locks[bd.acquire_lock_id] = LockInfo(lock_id=bd.acquire_lock_id)
                info.locks[bd.acquire_lock_id].operate_value = bd.acquire_lock_val
            if bd.release_lock_id >= 0:
                if bd.release_lock_id not in info.locks:
                    info.locks[bd.release_lock_id] = LockInfo(lock_id=bd.release_lock_id)
                info.locks[bd.release_lock_id].operate_value = bd.release_lock_val
            if bd.var in lock_operate_map:
                acq_val, rel_val = lock_operate_map[bd.var]
                if bd.acquire_lock_id >= 0:
                    info.locks[bd.acquire_lock_id].operate_value = acq_val
                if bd.release_lock_id >= 0:
                    info.locks[bd.release_lock_id].operate_value = rel_val
        for lid, lk in info.locks.items():
            if loc in lock_init_map and lid in lock_init_map[loc]:
                lk.init_value = lock_init_map[loc][lid]

    return tiles, flows


# ---------------------------------------------------------------------------
# Routing analysis
# ---------------------------------------------------------------------------

def resolve_pkt_internal_edges(block: RoutingBlock):
    edges = []
    masters_by_tile = defaultdict(list)
    for mp in block.pkt_masters:
        masters_by_tile[mp.tile].append(mp)
    for ss in block.pkt_slaves:
        for mp in masters_by_tile.get(ss.tile, []):
            if mp.arbiter == ss.arbiter and (mp.msel_en >> ss.msel) & 1:
                edges.append((ss, mp))
    return edges


def trace_paths(blocks: List[RoutingBlock]):
    all_paths = []
    for block in blocks:
        adj = defaultdict(list)
        for cc in block.circuit_conns:
            adj[(cc.tile, cc.slave_type, cc.slave_port)].append(
                (cc.tile, cc.master_type, cc.master_port, "circuit"))
        for ss, mp in resolve_pkt_internal_edges(block):
            adj[(ss.tile, ss.port_type, ss.port_num)].append(
                (mp.tile, mp.port_type, mp.port_num, "packet"))

        sources = set()
        for ss in block.pkt_slaves:
            if ss.port_type == "DMA":
                sources.add((ss.tile, ss.port_type, ss.port_num))
        for cc in block.circuit_conns:
            if cc.slave_type not in DIRECTION_OPPOSITES:
                sources.add((cc.tile, cc.slave_type, cc.slave_port))

        def follow(tile, mstr_type, mstr_port, visited):
            if mstr_type not in DIRECTION_DELTA:
                return [[(tile, mstr_type, mstr_port, None, None, "endpoint")]]
            if mstr_type == "SOUTH" and tile[1] == 0:
                return [[(tile, "SOUTH", mstr_port, None, None, "shim_exit")]]
            dc, dr = DIRECTION_DELTA[mstr_type]
            nb = (tile[0] + dc, tile[1] + dr)
            opp = DIRECTION_OPPOSITES[mstr_type]
            nb_key = (nb, opp, mstr_port)
            if nb_key in visited:
                return [[(tile, mstr_type, mstr_port, None, None, "loop")]]
            visited.add(nb_key)
            nexts = adj.get(nb_key, [])
            if not nexts:
                return [[(nb, opp, mstr_port, None, None, "dangling")]]
            results = []
            for (nt, mt, mp, mode) in nexts:
                sub = follow(nt, mt, mp, visited)
                for s in sub:
                    results.append([(nb, opp, mstr_port, mt, mp, mode)] + s)
            return results

        for src_key in sources:
            visited = {src_key}
            nexts = adj.get(src_key, [])
            for (nt, mt, mp, mode) in nexts:
                sub_paths = follow(nt, mt, mp, visited.copy())
                for sp in sub_paths:
                    full = [(src_key[0], src_key[1], src_key[2], mt, mp, mode)] + sp
                    all_paths.append((block.name, full))

    return all_paths


def collect_routing_tiles(blocks: List[RoutingBlock]) -> set:
    tiles = set()
    for b in blocks:
        for cc in b.circuit_conns:
            tiles.add(cc.tile)
        for ss in b.pkt_slaves:
            tiles.add(ss.tile)
        for mp in b.pkt_masters:
            tiles.add(mp.tile)
        for sd in b.shim_dma:
            tiles.add(sd.tile)
    return tiles


# ---------------------------------------------------------------------------
# Build routing data for JSON export (used by HTML/JS)
# ---------------------------------------------------------------------------

def _build_routing_json(blocks: List[RoutingBlock]) -> dict:
    """Build JSON-serializable routing data for the HTML renderer."""
    cross_tile_arrows = []
    internal_conns = []

    for bi, block in enumerate(blocks):
        color = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
        block_key = f"{block.name}__{bi}"

        for cc in block.circuit_conns:
            internal_conns.append({
                "tile": list(cc.tile),
                "slave_type": cc.slave_type,
                "slave_port": cc.slave_port,
                "master_type": cc.master_type,
                "master_port": cc.master_port,
                "mode": "circuit",
                "block": block.name,
                "block_key": block_key,
                "color": color,
            })

            mdir = cc.master_type
            if mdir in DIRECTION_DELTA:
                dc, dr = DIRECTION_DELTA[mdir]
                nb = (cc.tile[0] + dc, cc.tile[1] + dr)
                cross_tile_arrows.append({
                    "from_tile": list(cc.tile),
                    "to_tile": list(nb),
                    "from_dir": mdir,
                    "to_dir": DIRECTION_OPPOSITES[mdir],
                    "port": cc.master_port,
                    "mode": "circuit",
                    "block": block.name,
                    "block_key": block_key,
                    "color": color,
                })

        pkt_edges = resolve_pkt_internal_edges(block)
        for ss, mp in pkt_edges:
            internal_conns.append({
                "tile": list(ss.tile),
                "slave_type": ss.port_type,
                "slave_port": ss.port_num,
                "master_type": mp.port_type,
                "master_port": mp.port_num,
                "mode": "packet",
                "block": block.name,
                "block_key": block_key,
                "color": color,
                "pkt_id": ss.pkt_id,
            })

            mdir = mp.port_type
            if mdir in DIRECTION_DELTA:
                dc, dr = DIRECTION_DELTA[mdir]
                nb = (mp.tile[0] + dc, mp.tile[1] + dr)
                cross_tile_arrows.append({
                    "from_tile": list(mp.tile),
                    "to_tile": list(nb),
                    "from_dir": mdir,
                    "to_dir": DIRECTION_OPPOSITES[mdir],
                    "port": mp.port_num,
                    "mode": "packet",
                    "block": block.name,
                    "block_key": block_key,
                    "color": color,
                })

        for sd in block.shim_dma:
            internal_conns.append({
                "tile": list(sd.tile),
                "slave_type": "ShimDMA",
                "slave_port": sd.port_num,
                "master_type": "ShimDMA",
                "master_port": sd.port_num,
                "mode": "shim_dma",
                "direction": sd.direction,
                "block": block.name,
                "block_key": block_key,
                "color": color,
            })

    return {
        "cross_tile_arrows": cross_tile_arrows,
        "internal_conns": internal_conns,
        "blocks": [{"name": b.name, "color": BLOCK_COLORS[i % len(BLOCK_COLORS)]}
                    for i, b in enumerate(blocks)],
    }


# ---------------------------------------------------------------------------
# Build per-tile routing summary for tile cards
# ---------------------------------------------------------------------------

def _map_blocks_to_flows(
    blocks: List[RoutingBlock], flows: List[FlowInfo],
) -> Dict[str, List[int]]:
    """Map routing block keys to flow IDs by matching shim tiles and DMA endpoints.

    Block keys use the format "blockname__idx" to handle duplicate block names.
    The same key format is used in _build_routing_json and _build_routing_tile_summary.

    Returns a dict mapping block_key -> list of flow_ids.  A routing block that
    carries data for both an input flow (shim_to_aie) and an output flow
    (aie_to_shim) will map to multiple flow_ids so that checking either
    flow's checkbox makes the block's arrows visible.
    """
    block_to_flow: Dict[str, List[int]] = {}
    for bi, block in enumerate(blocks):
        block_key = f"{block.name}__{bi}"
        # Determine routing directions from shim_dma entries
        shim_tiles_by_dir: Dict[str, set] = defaultdict(set)
        shim_tiles = set()
        for sd in block.shim_dma:
            shim_tiles.add(sd.tile)
            shim_tiles_by_dir[sd.direction].add(sd.tile)
        has_mixed_dirs = len(shim_tiles_by_dir) > 1
        shim_direction = None
        if not has_mixed_dirs and shim_tiles_by_dir:
            shim_direction = next(iter(shim_tiles_by_dir))
        # Collect compute tiles that have DMA endpoints in circuit connections
        dma_compute_tiles = set()
        for cc in block.circuit_conns:
            if cc.master_type == "DMA" and cc.tile[1] > 0:
                dma_compute_tiles.add(cc.tile)
            if cc.slave_type == "DMA" and cc.tile[1] > 0:
                dma_compute_tiles.add(cc.tile)
        # Also check packet connections
        for ss in block.pkt_slaves:
            if ss.port_type == "DMA" and ss.tile[1] > 0:
                dma_compute_tiles.add(ss.tile)
        for mp in block.pkt_masters:
            if mp.port_type == "DMA" and mp.tile[1] > 0:
                dma_compute_tiles.add(mp.tile)
        # If no shim tile found, try inferring from circuit connections at row 0
        if not shim_tiles:
            for cc in block.circuit_conns:
                if cc.tile[1] == 0:
                    shim_tiles.add(cc.tile)
        # For output paths (aie_to_shim), compute tiles are not explicit DMA endpoints
        # in the routing block. Try to find the topmost tile in the routing chain.
        if not dma_compute_tiles and shim_direction == "aie_to_shim":
            all_block_tiles = set()
            for cc in block.circuit_conns:
                all_block_tiles.add(cc.tile)
            # The topmost tiles (highest row) are likely the source compute tiles
            if all_block_tiles:
                max_r = max(t[1] for t in all_block_tiles)
                top_tiles = {t for t in all_block_tiles if t[1] == max_r and t[1] > 0}
                dma_compute_tiles = top_tiles

        # Match against flows: find overlapping flows.
        # For mixed-direction blocks, match each direction independently so the
        # block maps to both the input and output flow.
        matched_flow_ids: List[int] = []
        if has_mixed_dirs:
            # Match each direction separately against its corresponding shim tiles
            for direction, dir_shim_tiles in shim_tiles_by_dir.items():
                expected_flow_dir = "MM2S" if direction == "shim_to_aie" else "S2MM"
                best_fid = -1
                best_score = 0
                candidates = []
                for f in flows:
                    if f.shim_tile not in dir_shim_tiles:
                        continue
                    if f.direction != expected_flow_dir:
                        continue
                    overlap = len(dma_compute_tiles & set(f.connected_tiles))
                    if overlap > best_score:
                        best_score = overlap
                        best_fid = f.flow_id
                    candidates.append(f)
                if best_fid < 0 and len(candidates) == 1:
                    best_fid = candidates[0].flow_id
                if best_fid < 0 and candidates:
                    best_fid = candidates[0].flow_id
                if best_fid >= 0:
                    matched_flow_ids.append(best_fid)
        else:
            # Single-direction block: find best matching flow
            best_flow_id = -1
            best_score = 0
            candidates = []
            for f in flows:
                if shim_tiles and f.shim_tile not in shim_tiles:
                    continue
                if shim_direction == "shim_to_aie" and f.direction != "MM2S":
                    continue
                if shim_direction == "aie_to_shim" and f.direction != "S2MM":
                    continue
                overlap = len(dma_compute_tiles & set(f.connected_tiles))
                if overlap > best_score:
                    best_score = overlap
                    best_flow_id = f.flow_id
                candidates.append(f)
            if best_flow_id < 0 and len(candidates) == 1:
                best_flow_id = candidates[0].flow_id
            if best_flow_id < 0 and candidates:
                best_flow_id = candidates[0].flow_id
            if best_flow_id >= 0:
                matched_flow_ids.append(best_flow_id)

        if matched_flow_ids:
            block_to_flow[block_key] = matched_flow_ids
    return block_to_flow


def _classify_mixed_block_connections(
    block: RoutingBlock,
) -> Dict[str, set]:
    """For a mixed-direction block, classify each circuit connection as belonging
    to the 'input' path (shim_to_aie) or the 'output' path (aie_to_shim).

    Returns a dict with keys 'shim_to_aie' and 'aie_to_shim', each mapping to a
    set of connection indices (into block.circuit_conns).

    Strategy: Build a directed graph of connections within the block. Then BFS
    forward from each shim DMA source to tag reachable connections.
    - For shim_to_aie: the shim tile's SOUTH port is the source, data flows
      northward/outward to compute tiles.
    - For aie_to_shim: the compute tiles' DMA/SOUTH are sources, data flows
      southward to the output shim tile.

    We trace forward through the connection graph: a connection's master port
    becomes the next connection's slave port (via cross-tile neighbor lookup
    for directional ports like NORTH/SOUTH/EAST/WEST).
    """
    # Build adjacency: (tile, port_type, port_num) -> list of connection indices
    # Each connection has a slave input and master output.
    # The master output of one connection feeds the slave input of the next.
    slave_to_conn: Dict[tuple, List[int]] = defaultdict(list)
    for ci, cc in enumerate(block.circuit_conns):
        slave_to_conn[(cc.tile, cc.slave_type, cc.slave_port)].append(ci)

    def _next_slave_key(cc: CircuitConn) -> Optional[tuple]:
        """Given a connection's master port, compute the slave key of the next tile."""
        mdir = cc.master_type
        if mdir in DIRECTION_DELTA:
            dc, dr = DIRECTION_DELTA[mdir]
            nb = (cc.tile[0] + dc, cc.tile[1] + dr)
            opp = DIRECTION_OPPOSITES[mdir]
            return (nb, opp, cc.master_port)
        return None  # DMA or other non-directional endpoint

    # Identify shim DMA tiles and their directions
    shim_dma_by_dir: Dict[str, List[ShimDmaPort]] = defaultdict(list)
    for sd in block.shim_dma:
        shim_dma_by_dir[sd.direction].append(sd)

    result: Dict[str, set] = {"shim_to_aie": set(), "aie_to_shim": set()}

    # For shim_to_aie: start from the SOUTH port of the input shim tile
    # The shim DMA enables a SOUTH port on the shim tile, and the first
    # circuit connection takes SOUTH:N as slave input.
    for sd in shim_dma_by_dir.get("shim_to_aie", []):
        # Find connections on the shim tile that have SOUTH as slave
        start_key = (sd.tile, "SOUTH", sd.port_num)
        # BFS forward
        queue = list(slave_to_conn.get(start_key, []))
        visited = set(queue)
        while queue:
            ci = queue.pop(0)
            result["shim_to_aie"].add(ci)
            cc = block.circuit_conns[ci]
            # Follow master port to next tile's slave
            nk = _next_slave_key(cc)
            if nk:
                for nci in slave_to_conn.get(nk, []):
                    if nci not in visited:
                        visited.add(nci)
                        queue.append(nci)

    # For aie_to_shim: start from connections whose slave is DMA or NORTH
    # on compute tiles, flowing southward to the output shim tile.
    # We find the output shim tile and trace backward... actually, let's trace
    # forward from compute DMA outputs.
    # The output path: compute tile DMA -> SOUTH -> ... -> shim tile SOUTH exit
    # Find connections with DMA or NORTH slave on non-shim tiles that are NOT
    # already tagged as input path
    for sd in shim_dma_by_dir.get("aie_to_shim", []):
        # The output path ends at the shim tile. Find the connection on the shim
        # tile whose master is SOUTH (exiting to shim DMA).
        # Actually for aie_to_shim, the connection chain goes:
        # compute tile: NORTH:N (slave from above) -> SOUTH:M (master going down)
        # repeated until reaching shim tile row 0.
        # We need to find the top of this chain.
        # Strategy: find all connections NOT in input set, BFS from any that
        # have DMA or NORTH slave type on compute tiles.
        pass

    # Simpler approach for aie_to_shim: any connection NOT reached by shim_to_aie
    # BFS belongs to the output path.
    all_indices = set(range(len(block.circuit_conns)))
    result["aie_to_shim"] = all_indices - result["shim_to_aie"]

    return result


def _build_routing_tile_summary(blocks: List[RoutingBlock]) -> Dict[tuple, List[dict]]:
    result: Dict[tuple, List[dict]] = defaultdict(list)
    for bi, block in enumerate(blocks):
        color = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
        block_key = f"{block.name}__{bi}"
        for cc in block.circuit_conns:
            result[cc.tile].append({
                "type": "circuit",
                "slave": f"{cc.slave_type}:{cc.slave_port}",
                "master": f"{cc.master_type}:{cc.master_port}",
                "block": block.name,
                "block_key": block_key,
                "color": color,
            })
        for ss, mp in resolve_pkt_internal_edges(block):
            result[ss.tile].append({
                "type": "packet",
                "slave": f"{ss.port_type}:{ss.port_num}",
                "master": f"{mp.port_type}:{mp.port_num}",
                "block": block.name,
                "block_key": block_key,
                "color": color,
                "pkt_id": ss.pkt_id,
            })
        for sd in block.shim_dma:
            arrow = "AIE->Shim" if sd.direction == "aie_to_shim" else "Shim->AIE"
            result[sd.tile].append({
                "type": "shim_dma",
                "port": sd.port_num,
                "direction": arrow,
                "block": block.name,
                "block_key": block_key,
                "color": color,
            })
    return dict(result)


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def render_text(blocks: Optional[List[RoutingBlock]],
                paths: Optional[list],
                dma_tiles: Optional[Dict[Tuple[int, int], TileDmaInfo]],
                routing_summary: Optional[Dict[tuple, List[dict]]],
                out) -> None:
    if blocks:
        out.write("=" * 70 + "\n")
        out.write("  ROUTING TOPOLOGY\n")
        out.write("=" * 70 + "\n")
        for block in blocks:
            out.write(f"\n--- {block.name} ---\n")
            if block.circuit_conns:
                out.write("  Circuit-Switch Connections:\n")
                for cc in block.circuit_conns:
                    out.write(f"    tile({cc.tile[0]},{cc.tile[1]}) "
                              f"{cc.slave_type}:{cc.slave_port} -> "
                              f"{cc.master_type}:{cc.master_port}\n")
            pkt_edges = resolve_pkt_internal_edges(block)
            if pkt_edges:
                out.write("  Packet-Switch Connections:\n")
                for ss, mp in pkt_edges:
                    out.write(f"    tile({ss.tile[0]},{ss.tile[1]}) "
                              f"{ss.port_type}:{ss.port_num} [PktId={ss.pkt_id}] -> "
                              f"{mp.port_type}:{mp.port_num}  (Arb={ss.arbiter})\n")
            if block.shim_dma:
                out.write("  Shim DMA Ports:\n")
                for sd in block.shim_dma:
                    d = "AIE->Shim" if sd.direction == "aie_to_shim" else "Shim->AIE"
                    out.write(f"    tile({sd.tile[0]},{sd.tile[1]}) port {sd.port_num} ({d})\n")

    if paths:
        out.write(f"\n{'=' * 70}\n")
        out.write("  TRACED END-TO-END PATHS\n")
        out.write(f"{'=' * 70}\n")
        for i, (bname, path) in enumerate(paths):
            hops = []
            for (tile, stype, sport, mtype, mport, mode) in path:
                if mtype:
                    hops.append(f"({tile[0]},{tile[1]}) {stype}:{sport}->{mtype}:{mport}")
                elif mode == "shim_exit":
                    hops.append(f"({tile[0]},{tile[1]}) -> ShimDMA (SOUTH:{sport})")
                else:
                    hops.append(f"({tile[0]},{tile[1]}) {stype}:{sport} [{mode}]")
            out.write(f"\n  [{bname}] Path {i+1}:\n")
            for j, h in enumerate(hops):
                prefix = "    +---> " if j == len(hops) - 1 else "    |---> "
                out.write(f"{prefix}{h}\n")

    if dma_tiles:
        out.write(f"\n{'=' * 70}\n")
        out.write("  DMA BUFFER DESCRIPTOR CONFIGURATION\n")
        out.write(f"{'=' * 70}\n\n")
        for loc in sorted(dma_tiles, key=lambda k: (k[1], k[0])):
            info = dma_tiles[loc]
            header = f"Tile ({loc[0]},{loc[1]}) [{info.tile_type}]"
            out.write(f"--- {header} ---\n")
            if info.ios:
                dirs = ", ".join(f"ch{io.channel_id}:{io.direction}" for io in info.ios)
                out.write(f"  IO Channels: {dirs}\n")
            if info.bds:
                out.write("  BD Configurations:\n")
                for bd in info.bds:
                    chain = f" -> BD{bd.next_bd}" if bd.next_bd >= 0 else " (no chain)"
                    pkt = f"pkt={bd.packet_id}" if bd.enable_packet else "no-pkt"
                    acq = f"lock={bd.acquire_lock_id} val={bd.acquire_lock_val}" if bd.acquire_lock_id >= 0 else "none"
                    rel = f"lock={bd.release_lock_id} val={bd.release_lock_val}" if bd.release_lock_id >= 0 else "none"
                    out.write(f"    [BD{bd.bd_id}] len={bd.length:>4}  {pkt:<8}"
                              f"  next{chain:<12}  buf={bd.buf_source}\n")
                    out.write(f"           DMA lock: acquire({acq})  release({rel})\n")
            if info.locks:
                out.write("  Lock Summary:\n")
                for lid in sorted(info.locks):
                    lk = info.locks[lid]
                    dma_s = f"dma_request_val={lk.operate_value}" if lk.operate_value is not None else "dma_request_val=unset"
                    init_s = f"init={lk.init_value}" if lk.init_value is not None else "no init"
                    out.write(f"    Lock {lid}: {dma_s} | {init_s}\n")
            out.write("\n")


# ---------------------------------------------------------------------------
# HTML output — combined routing + DMA
# ---------------------------------------------------------------------------

_PKT_COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D", "#E57373",
    "#BA68C8", "#4DB6AC", "#FF8A65", "#A1887F",
]


def render_html(
    blocks: Optional[List[RoutingBlock]],
    dma_tiles: Optional[Dict[Tuple[int, int], TileDmaInfo]],
    routing_tiles: Optional[set],
    routing_summary: Optional[Dict[tuple, List[dict]]],
    output_path: str,
    flows: Optional[List[FlowInfo]] = None,
) -> str:
    all_tile_locs = set()
    if routing_tiles:
        all_tile_locs |= routing_tiles
    if dma_tiles:
        all_tile_locs |= set(dma_tiles.keys())

    if not all_tile_locs:
        print("No tiles found.", file=sys.stderr)
        return output_path

    all_cols = sorted({t[0] for t in all_tile_locs})
    all_rows = sorted({t[1] for t in all_tile_locs}, reverse=True)
    min_col = all_cols[0]
    max_row = all_rows[0]

    if routing_summary is None:
        routing_summary = {}
    if dma_tiles is None:
        dma_tiles = {}

    routing_json_data = {}
    if blocks:
        routing_json_data = _build_routing_json(blocks)

    # Build block-name -> flow_ids mapping for routing elements
    block_to_flow: Dict[str, List[int]] = {}
    per_conn_flow: Dict[tuple, List[int]] = {}
    if blocks and flows:
        block_to_flow = _map_blocks_to_flows(blocks, flows)

        # For mixed-direction blocks, build per-connection flow_id mappings.
        # Key: (block_key, tuple(tile), slave_type, slave_port, master_type, master_port)
        # Value: list of flow_ids for that specific connection.
        per_conn_flow: Dict[tuple, List[int]] = {}
        for bi, block in enumerate(blocks):
            block_key = f"{block.name}__{bi}"
            all_flow_ids = block_to_flow.get(block_key, [])
            if len(all_flow_ids) <= 1:
                continue  # Single-direction block, no splitting needed

            # Classify connections as input vs output
            classification = _classify_mixed_block_connections(block)

            # Map directions to flow_ids
            dir_to_fid: Dict[str, List[int]] = {}
            for sd in block.shim_dma:
                direction = sd.direction
                # Find matching flow_id for this direction+shim tile
                expected_flow_dir = "MM2S" if direction == "shim_to_aie" else "S2MM"
                for f in flows:
                    if f.flow_id in all_flow_ids and f.direction == expected_flow_dir:
                        dir_to_fid.setdefault(direction, []).append(f.flow_id)
                        break

            # Tag each circuit connection with its direction's flow_ids
            for ci, cc in enumerate(block.circuit_conns):
                conn_key = (block_key, tuple(cc.tile), cc.slave_type,
                            cc.slave_port, cc.master_type, cc.master_port)
                if ci in classification.get("shim_to_aie", set()):
                    per_conn_flow[conn_key] = dir_to_fid.get("shim_to_aie", all_flow_ids)
                elif ci in classification.get("aie_to_shim", set()):
                    per_conn_flow[conn_key] = dir_to_fid.get("aie_to_shim", all_flow_ids)

            # Tag shim_dma entries per direction
            for sd in block.shim_dma:
                conn_key = (block_key, tuple(sd.tile), "ShimDMA",
                            sd.port_num, "ShimDMA", sd.port_num)
                per_conn_flow[conn_key] = dir_to_fid.get(sd.direction, all_flow_ids)

        # Inject flow_ids into routing JSON entries, using per-connection
        # granularity for mixed blocks and block-level for single-direction blocks.
        if routing_json_data:
            for arrow in routing_json_data.get("cross_tile_arrows", []):
                bk = arrow.get("block_key", arrow["block"])
                # Try per-connection lookup first: the cross_tile_arrow is derived
                # from a circuit connection at from_tile with the master direction.
                # We need to find the originating connection's classification.
                # The arrow's from_tile + from_dir + port corresponds to a connection
                # with master_type=from_dir, master_port=port at from_tile.
                # Look up all internal_conns that match this.
                found = False
                if bk in block_to_flow and len(block_to_flow[bk]) > 1:
                    # Mixed block: find which connections at from_tile have this master port
                    from_t = tuple(arrow["from_tile"])
                    from_dir = arrow["from_dir"]
                    port = arrow["port"]
                    # Search for matching circuit connection entry in per_conn_flow
                    for conn_key, fids in per_conn_flow.items():
                        if (conn_key[0] == bk and conn_key[1] == from_t
                                and conn_key[4] == from_dir and conn_key[5] == port):
                            arrow["flow_ids"] = fids
                            found = True
                            break
                if not found:
                    arrow["flow_ids"] = block_to_flow.get(bk, [])

            for conn in routing_json_data.get("internal_conns", []):
                bk = conn.get("block_key", conn["block"])
                conn_key = (bk, tuple(conn["tile"]), conn["slave_type"],
                            conn["slave_port"], conn["master_type"], conn["master_port"])
                if conn_key in per_conn_flow:
                    conn["flow_ids"] = per_conn_flow[conn_key]
                else:
                    conn["flow_ids"] = block_to_flow.get(bk, [])

            # Fallback: infer flow_ids for unmatched entries from neighboring arrows
            # sharing the same block_key that already have valid flow_ids.
            block_key_to_flows: Dict[str, List[int]] = {}
            for arrow in routing_json_data.get("cross_tile_arrows", []):
                bk = arrow.get("block_key", arrow["block"])
                if arrow.get("flow_ids"):
                    block_key_to_flows[bk] = arrow["flow_ids"]
            for conn in routing_json_data.get("internal_conns", []):
                bk = conn.get("block_key", conn["block"])
                if conn.get("flow_ids"):
                    block_key_to_flows[bk] = conn["flow_ids"]
            # Apply inferred flow_ids to unmatched entries
            for arrow in routing_json_data.get("cross_tile_arrows", []):
                if not arrow.get("flow_ids"):
                    bk = arrow.get("block_key", arrow["block"])
                    arrow["flow_ids"] = block_key_to_flows.get(bk, [])
            for conn in routing_json_data.get("internal_conns", []):
                if not conn.get("flow_ids"):
                    bk = conn.get("block_key", conn["block"])
                    conn["flow_ids"] = block_key_to_flows.get(bk, [])

    def _pkt_color(pid: int) -> str:
        return _PKT_COLORS[pid % len(_PKT_COLORS)]

    def _lock_html(lock_id, lock_val, locks, label):
        if lock_id < 0:
            return f'<span class="lock unset">{label}: none</span>'
        lk = locks.get(lock_id)
        init_val = str(lk.init_value) if lk and lk.init_value is not None else "?"
        tooltip = f"DMA {label}: lock_id={lock_id}, request_val={lock_val}, init={init_val}"
        init_badge = ""
        if lk and lk.init_value is not None:
            init_badge = f'<span class="lock-init-badge">init={lk.init_value}</span>'
        return (
            f'<span class="lock" title="{tooltip}">'
            f'{label}=L{lock_id}(val={lock_val}) {init_badge}</span>'
        )

    tile_cards = []
    for loc in sorted(all_tile_locs, key=lambda k: (-k[1], k[0])):
        tile_type = TILE_TYPE_BY_ROW(loc[1])
        type_cls = tile_type.lower()
        dma_info = dma_tiles.get(loc)
        rt_info = routing_summary.get(loc, [])
        is_active = loc in (routing_tiles or set()) or loc in dma_tiles

        # --- Routing section ---
        routing_rows = []
        for r in rt_info:
            # Use per-connection flow_ids for mixed blocks when available
            _r_bk = r.get("block_key", r["block"])
            if r["type"] == "circuit":
                _s_type, _s_port = r["slave"].split(":")
                _m_type, _m_port = r["master"].split(":")
                _r_conn_key = (_r_bk, loc, _s_type, int(_s_port), _m_type, int(_m_port))
                r_flow_ids = per_conn_flow.get(_r_conn_key, block_to_flow.get(_r_bk, []))
            elif r["type"] == "shim_dma":
                _r_conn_key = (_r_bk, loc, "ShimDMA", r["port"], "ShimDMA", r["port"])
                r_flow_ids = per_conn_flow.get(_r_conn_key, block_to_flow.get(_r_bk, []))
            else:
                r_flow_ids = block_to_flow.get(_r_bk, [])
            r_flow_attr = f' data-flow-ids="{",".join(str(fid) for fid in r_flow_ids)}"' if r_flow_ids else ''
            if r["type"] == "circuit":
                routing_rows.append(
                    f'<div class="route-row circuit"{r_flow_attr} style="border-left:3px solid {r["color"]}">'
                    f'<span class="route-label">CCT</span> '
                    f'{r["slave"]} &rarr; {r["master"]}'
                    f'<span class="route-block">{r["block"]}</span></div>'
                )
            elif r["type"] == "packet":
                routing_rows.append(
                    f'<div class="route-row packet"{r_flow_attr} style="border-left:3px solid {r["color"]}">'
                    f'<span class="route-label">PKT</span> '
                    f'{r["slave"]} &rarr; {r["master"]} '
                    f'<span class="pkt" style="background:{_pkt_color(r.get("pkt_id", 0))}">'
                    f'pkt{r.get("pkt_id", "?")}</span>'
                    f'<span class="route-block">{r["block"]}</span></div>'
                )
            elif r["type"] == "shim_dma":
                arrow = "&#x25BC;" if "AIE" in r["direction"] else "&#x25B2;"
                routing_rows.append(
                    f'<div class="route-row shim-dma"{r_flow_attr} style="border-left:3px solid {r["color"]}">'
                    f'{arrow} ShimDMA:{r["port"]} ({r["direction"]})'
                    f'<span class="route-block">{r["block"]}</span></div>'
                )

        routing_section = ""
        if routing_rows:
            collapsed = "".join(routing_rows)
            routing_section = (
                f'<div class="section-header routing-header" onclick="toggleSection(this)">'
                f'&#x25BC; Routing ({len(routing_rows)})</div>'
                f'<div class="routing-section expanded">{collapsed}</div>'
            )

        # --- DMA section ---
        dma_section = ""
        if dma_info:
            io_rows = []
            for io in dma_info.ios:
                dir_cls = "s2mm" if io.direction == "S2MM" else "mm2s"
                flow_attr = f' data-flow="{io.flow_id}"' if io.flow_id >= 0 else ''
                io_rows.append(
                    f'<div class="io-row {dir_cls}"{flow_attr}>ch{io.channel_id}: {io.direction}</div>'
                )

            bd_rows = []
            bd_by_id = {bd.bd_id: bd for bd in dma_info.bds}
            for bd in dma_info.bds:
                is_pp = (bd.next_bd >= 0 and bd.next_bd != bd.bd_id
                         and bd.next_bd in bd_by_id
                         and bd_by_id[bd.next_bd].next_bd == bd.bd_id)
                pp_class = " ping-pong" if is_pp else ""
                flow_attr = f' data-flow="{bd.flow_id}"' if bd.flow_id >= 0 else ''
                nxt = f"&rarr;BD{bd.next_bd}" if bd.next_bd >= 0 else "none"
                chain_html = f'<span class="chain">BD{bd.bd_id} next={nxt}</span>'

                pkt_html = ""
                if bd.enable_packet:
                    c = _pkt_color(bd.packet_id)
                    pkt_html = f'<span class="pkt" style="background:{c}">pkt{bd.packet_id}</span>'

                acq_html = _lock_html(bd.acquire_lock_id, bd.acquire_lock_val, dma_info.locks, "acq")
                rel_html = _lock_html(bd.release_lock_id, bd.release_lock_val, dma_info.locks, "rel")

                bd_rows.append(
                    f'<div class="bd-row{pp_class}"{flow_attr}>'
                    f'{chain_html} {pkt_html} '
                    f'<span class="len">len={bd.length}</span> '
                    f'<span class="buf">buf={bd.buf_source}</span>'
                    f'<div class="lock-row">{acq_html} {rel_html}</div>'
                    f'</div>'
                )

            lock_rows = []
            for lid in sorted(dma_info.locks):
                lk = dma_info.locks[lid]
                dma_s = f"req_val={lk.operate_value}" if lk.operate_value is not None else "unset"
                if lk.init_value is not None:
                    init_s = f'<span class="lock-init-badge">init={lk.init_value}</span>'
                else:
                    init_s = '<span class="lock-no-init">no init</span>'
                lock_rows.append(
                    f'<div class="lock-summary-row">L{lid}: {dma_s} {init_s}</div>'
                )
            lock_section = ""
            if lock_rows:
                lock_section = f'<div class="lock-section">{"".join(lock_rows)}</div>'

            io_html = f'<div class="io-section">{"".join(io_rows)}</div>' if io_rows else ""
            bd_html = f'<div class="bd-section">{"".join(bd_rows)}</div>' if bd_rows else ""
            sources = list(dict.fromkeys(bd.buf_source for bd in dma_info.bds))
            src_text = ", ".join(sources[:2]) + ("..." if len(sources) > 2 else "")
            buf_html = f'<div class="buf-line">buf: {src_text}</div>' if sources else ""

            dma_section = (
                f'<div class="section-header dma-header" onclick="toggleSection(this)">'
                f'&#x25B6; DMA ({len(dma_info.bds)} BDs)</div>'
                f'<div class="dma-detail-section collapsed">'
                f'{io_html}{bd_html}{lock_section}{buf_html}</div>'
            )

        gc = loc[0] - min_col + 1
        gr = max_row - loc[1] + 1

        card = (
            f'<div class="tile-card {type_cls}{"" if is_active else " inactive"}" '
            f'id="tile-{loc[0]}-{loc[1]}" '
            f'style="grid-column:{gc}; grid-row:{gr}" '
            f'data-col="{loc[0]}" data-row="{loc[1]}">'
            f'<div class="tile-header">({loc[0]},{loc[1]}) {tile_type}</div>'
            f'{routing_section}'
            f'{dma_section}'
            f'</div>'
        )
        tile_cards.append(card)

    ncols = len(all_cols)
    nrows = len(all_rows)

    has_routing = bool(blocks)
    has_dma = bool(dma_tiles)
    title_parts = []
    if has_routing:
        title_parts.append("Routing Topology")
    if has_dma:
        title_parts.append("DMA BD Configuration")
    title = " + ".join(title_parts) or "AIE Visualization"

    block_legend = ""
    if blocks:
        items = []
        for i, b in enumerate(blocks):
            c = BLOCK_COLORS[i % len(BLOCK_COLORS)]
            items.append(f'<div class="legend-item">'
                         f'<div class="legend-swatch" style="background:{c}"></div> {b.name}</div>')
        block_legend = "".join(items)

    routing_json_str = json.dumps(routing_json_data) if routing_json_data else "{}"

    # --- Build Device Map data (full dataflow view) ---
    map_tile_data = []
    for loc in sorted(all_tile_locs):
        tile_type = TILE_TYPE_BY_ROW(loc[1])
        rt_info = routing_summary.get(loc, [])
        dma_info_loc = dma_tiles.get(loc)
        connected_flow_ids = set()
        for r in rt_info:
            bk = r.get("block_key", r["block"])
            for fid in block_to_flow.get(bk, []):
                connected_flow_ids.add(fid)
        if dma_info_loc:
            for io in dma_info_loc.ios:
                if io.flow_id >= 0:
                    connected_flow_ids.add(io.flow_id)
        # Collect per-flow labels inside this tile (buf names + stream IDs)
        tile_labels = []
        if dma_info_loc:
            seen_bufs = set()
            for bd in dma_info_loc.bds:
                if bd.buf_source and bd.buf_source not in seen_bufs:
                    seen_bufs.add(bd.buf_source)
                    fid = bd.flow_id
                    tile_labels.append({"text": bd.buf_source[:14], "flow_id": fid})
            for io in dma_info_loc.ios:
                tile_labels.append({"text": f"ch{io.channel_id} {io.direction}", "flow_id": io.flow_id})
        # Routing edges passing through this tile (for gray core-to-core lines)
        rt_edges = []
        for r in rt_info:
            if r["type"] == "circuit":
                rt_edges.append({"slave": r["slave"], "master": r["master"],
                                 "block": r["block"],
                                 "flow_ids": list(block_to_flow.get(r.get("block_key", r["block"]), []))})
        map_tile_data.append({
            "col": loc[0], "row": loc[1],
            "type": tile_type,
            "active": loc in (routing_tiles or set()) or loc in dma_tiles,
            "flow_ids": sorted(connected_flow_ids),
            "labels": tile_labels,
            "rt_edges": rt_edges,
        })
    map_tile_data_str = json.dumps(map_tile_data)

    # --- Build flow filter panel ---
    flow_panel_html = ""
    flow_json_str = "[]"
    if flows:
        flow_json_data = []
        for f in flows:
            flow_json_data.append({
                "flow_id": f.flow_id,
                "shim_tile": list(f.shim_tile),
                "channel_id": f.channel_id,
                "direction": f.direction,
                "connected_tiles": [list(t) for t in f.connected_tiles],
                "label": f.label,
            })
        flow_json_str = json.dumps(flow_json_data)

        # Group flows by shim tile
        shim_groups: Dict[Tuple[int, int], List[FlowInfo]] = defaultdict(list)
        for f in flows:
            shim_groups[f.shim_tile].append(f)

        panel_items = []
        for stile in sorted(shim_groups.keys()):
            tile_flows = shim_groups[stile]
            panel_items.append(
                f'<div class="flow-group">'
                f'<div class="flow-group-header">Shim ({stile[0]},{stile[1]})</div>'
            )
            for f in tile_flows:
                dir_cls = "s2mm" if f.direction == "S2MM" else "mm2s"
                arrow = "&#x25BC;" if f.direction == "S2MM" else "&#x25B2;"
                n_tiles = len(f.connected_tiles)
                panel_items.append(
                    f'<label class="flow-toggle {dir_cls}">'
                    f'<input type="checkbox" checked data-flow-id="{f.flow_id}" '
                    f'onchange="toggleFlow({f.flow_id}, this.checked)">'
                    f'{arrow} ch{f.channel_id} {f.direction} ({n_tiles} tiles)'
                    f'</label>'
                    f'<button class="flow-solo-btn" onclick="soloOneFlow({f.flow_id})" '
                    f'title="Show only this flow">Solo</button>'
                )
            panel_items.append('</div>')

        flow_panel_html = (
            '<div class="flow-filter-panel">'
            '<div class="flow-panel-header" onclick="toggleFlowPanel(this)">'
            '&#x25BC; DMA Flow Filter</div>'
            '<div class="flow-panel-body expanded">'
            '<div class="flow-panel-controls">'
            '<button onclick="showAllFlows()">Show All</button>'
            '<button onclick="hideAllFlows()">Hide All</button>'
            '<button onclick="soloFlowMode()">Solo Mode</button>'
            '</div>'
            + "".join(panel_items) +
            '</div></div>'
        )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5; padding: 20px;
    color: #333;
}}
h1 {{ text-align: center; margin-bottom: 6px; font-size: 22px; color: #1a1a2e; }}
.subtitle {{ text-align: center; margin-bottom: 16px; font-size: 12px; color: #888; }}
.controls {{
    display: flex; justify-content: center; gap: 12px;
    margin-bottom: 16px; flex-wrap: wrap;
}}
.controls button {{
    padding: 6px 16px; border: 1px solid #bbb; border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 12px; font-weight: 600;
    transition: all .15s;
}}
.controls button:hover {{ background: #e3f2fd; border-color: #42a5f5; }}
.controls button.active {{ background: #1976d2; color: #fff; border-color: #1976d2; }}
.legend {{
    display: flex; gap: 14px; justify-content: center;
    margin-bottom: 14px; font-size: 11px; flex-wrap: wrap;
}}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-swatch {{
    width: 14px; height: 14px; border-radius: 3px; border: 1px solid #aaa;
}}
.grid-wrapper {{
    position: relative; display: inline-block;
    margin: 0 auto;
}}
.grid-container {{
    display: flex; justify-content: center;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat({ncols}, 300px);
    grid-template-rows: repeat({nrows}, auto);
    gap: 14px;
    position: relative;
    z-index: 1;
}}
svg.routing-overlay {{
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 10;
}}
svg.routing-overlay line, svg.routing-overlay path {{
    pointer-events: stroke;
}}
.tile-card {{
    border: 2px solid #bbb; border-radius: 10px;
    padding: 10px; min-height: 80px;
    background: #fff; position: relative;
    transition: box-shadow .15s, transform .15s;
}}
.tile-card:hover {{ box-shadow: 0 4px 20px rgba(0,0,0,.15); transform: translateY(-1px); }}
.tile-card.inactive {{ opacity: 0.4; }}
.tile-card.shim {{ background: #e3f2fd; border-color: #42a5f5; }}
.tile-card.mem  {{ background: #e8f5e9; border-color: #66bb6a; }}
.tile-card.aie  {{ background: #fffde7; border-color: #fdd835; }}
.tile-header {{
    font-weight: 700; font-size: 13px; margin-bottom: 6px;
    padding-bottom: 4px; border-bottom: 1px solid #ddd;
    display: flex; justify-content: space-between; align-items: center;
}}
.section-header {{
    font-size: 11px; font-weight: 700; color: #555;
    cursor: pointer; padding: 3px 0; user-select: none;
    transition: color .1s;
}}
.section-header:hover {{ color: #1976d2; }}
.routing-section, .dma-detail-section {{
    overflow: hidden; transition: max-height .3s ease;
}}
.collapsed {{ max-height: 0 !important; }}
.expanded {{ max-height: 2000px; }}
.route-row {{
    font-size: 10px; font-family: monospace;
    padding: 2px 6px; margin-bottom: 2px; border-radius: 3px;
    background: rgba(255,255,255,0.7);
}}
.route-label {{
    display: inline-block; width: 28px; font-weight: 700;
    font-size: 9px; color: #fff; text-align: center;
    border-radius: 3px; padding: 0 2px; margin-right: 4px;
}}
.route-row.circuit .route-label {{ background: #1976d2; }}
.route-row.packet .route-label {{ background: #7b1fa2; }}
.route-row.shim-dma .route-label {{ background: #e65100; }}
.route-block {{
    float: right; font-size: 9px; color: #999; font-style: italic;
}}
.io-section {{ margin-bottom: 4px; }}
.io-row {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-right: 4px; margin-bottom: 2px;
}}
.io-row.s2mm {{ background: #bbdefb; color: #0d47a1; }}
.io-row.mm2s {{ background: #ffe0b2; color: #e65100; }}
.bd-section {{ font-size: 11px; }}
.bd-row {{
    padding: 4px 6px; margin-bottom: 3px; border-radius: 4px;
    background: #fafafa; border: 1px solid #e0e0e0;
    display: flex; flex-wrap: wrap; gap: 5px; align-items: center;
}}
.bd-row.ping-pong {{ background: #e8eaf6; border-color: #7986cb; }}
.chain {{ font-weight: 700; color: #1565c0; font-family: monospace; }}
.pkt {{
    padding: 1px 6px; border-radius: 3px; font-size: 10px;
    color: #fff; font-weight: 600;
}}
.len {{ color: #555; }}
.buf {{ color: #888; font-style: italic; font-size: 10px; }}
.lock-row {{ width: 100%; display: flex; gap: 8px; margin-top: 2px; }}
.lock {{
    font-size: 10px; font-family: monospace; color: #4a148c;
    background: #f3e5f5; padding: 1px 5px; border-radius: 3px;
}}
.lock.unset {{ color: #999; background: #f0f0f0; }}
.lock-init-badge {{
    font-size: 9px; font-weight: 700; color: #fff; background: #7b1fa2;
    padding: 0 4px; border-radius: 3px; margin-left: 2px;
}}
.lock-no-init {{ color: #aaa; font-style: italic; }}
.lock-section {{
    margin-top: 4px; padding-top: 3px; border-top: 1px dashed #ccc;
    font-size: 10px; color: #666;
}}
.lock-summary-row {{ padding: 1px 0; font-family: monospace; }}
.buf-line {{
    font-size: 10px; color: #777; font-style: italic;
    margin-top: 3px; padding-top: 3px; border-top: 1px dotted #ddd;
}}
.arrow-tooltip {{
    position: absolute; background: rgba(30,30,30,.92); color: #fff;
    padding: 4px 10px; border-radius: 5px; font-size: 11px;
    pointer-events: none; z-index: 100; white-space: nowrap;
    display: none; font-family: monospace;
}}
/* Flow filter panel */
.flow-filter-panel {{
    max-width: 900px; margin: 0 auto 16px auto;
    background: #fff; border: 1px solid #ccc; border-radius: 8px;
    padding: 10px 14px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
}}
.flow-panel-header {{
    font-size: 13px; font-weight: 700; color: #333;
    cursor: pointer; user-select: none; padding: 2px 0;
}}
.flow-panel-header:hover {{ color: #1976d2; }}
.flow-panel-body {{
    overflow: hidden; transition: max-height .3s ease;
}}
.flow-panel-body.collapsed {{ max-height: 0 !important; padding: 0; }}
.flow-panel-body.expanded {{ max-height: 2000px; }}
.flow-panel-controls {{
    display: flex; gap: 8px; margin: 8px 0;
}}
.flow-panel-controls button {{
    padding: 4px 12px; border: 1px solid #bbb; border-radius: 5px;
    background: #fff; cursor: pointer; font-size: 11px; font-weight: 600;
    transition: all .15s;
}}
.flow-panel-controls button:hover {{ background: #e3f2fd; border-color: #42a5f5; }}
.flow-group {{
    margin-bottom: 6px; padding: 4px 0;
    border-bottom: 1px solid #eee;
}}
.flow-group:last-child {{ border-bottom: none; }}
.flow-group-header {{
    font-size: 11px; font-weight: 700; color: #555; margin-bottom: 4px;
}}
.flow-toggle {{
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; font-family: monospace;
    padding: 2px 8px; margin: 2px 4px; border-radius: 4px;
    cursor: pointer; user-select: none; transition: opacity .15s;
}}
.flow-toggle.s2mm {{ background: #bbdefb; color: #0d47a1; }}
.flow-toggle.mm2s {{ background: #ffe0b2; color: #e65100; }}
.flow-toggle input[type="checkbox"] {{ margin: 0; cursor: pointer; }}
[data-flow].flow-hidden, [data-flow-ids].flow-hidden {{
    display: none !important;
}}
.solo-active {{ outline: 2px solid #f44336; outline-offset: -2px; }}
.tile-card.flow-dimmed {{ opacity: 0.15; pointer-events: none; transition: opacity .2s; }}
.flow-solo-btn {{
    padding: 1px 7px; border: 1px solid #bbb; border-radius: 3px;
    background: #fff; cursor: pointer; font-size: 10px; font-weight: 600;
    margin-left: 2px; vertical-align: middle; transition: all .15s;
}}
.flow-solo-btn:hover {{ background: #ffecb3; border-color: #ffa000; }}

/* ── Tab bar ─────────────────────────────────────────────────── */
.tab-bar {{
    display: flex; gap: 0; border-bottom: 2px solid #ddd;
    max-width: 1200px; margin: 0 auto 20px auto;
}}
.tab-btn {{
    padding: 8px 22px; font-size: 13px; font-weight: 600;
    border: none; background: none; cursor: pointer; color: #888;
    border-bottom: 3px solid transparent; margin-bottom: -2px;
    transition: color .15s, border-color .15s;
}}
.tab-btn:hover {{ color: #1976d2; }}
.tab-btn.active {{ color: #1976d2; border-bottom-color: #1976d2; }}
.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}

/* ── Pan/zoom canvas (shared by both tabs) ────────────────────── */
.pan-viewport {{
    width: 100%; height: calc(100vh - 180px); min-height: 500px;
    overflow: hidden; position: relative; background: #f6f7f9;
    border: 1px solid #e0e0e0; border-radius: 6px; cursor: grab;
    user-select: none;
}}
.pan-viewport.panning {{ cursor: grabbing; }}
.pan-canvas {{
    position: absolute; top: 0; left: 0;
    transform-origin: 0 0; will-change: transform;
}}
.pan-hint {{
    position: absolute; bottom: 8px; right: 12px;
    font-size: 10px; color: #bbb; pointer-events: none;
}}
.pan-reset-btn {{
    position: absolute; top: 8px; right: 8px;
    padding: 4px 10px; font-size: 11px; font-weight: 600;
    border: 1px solid #ccc; border-radius: 4px; background: #fff;
    cursor: pointer; z-index: 20; transition: all .15s;
}}
.pan-reset-btn:hover {{ background: #e3f2fd; border-color: #42a5f5; }}

/* ── Device Map net bar ───────────────────────────────────────── */
.map-header {{
    padding: 10px 16px 0; background: #fff;
    border-bottom: 1px solid #e8e8e8;
}}
.map-header-title {{
    font-size: 12px; color: #666; margin-bottom: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.map-net-bar {{
    display: flex; flex-wrap: wrap; gap: 6px;
    padding-bottom: 10px; align-items: center;
}}
.map-net-chip {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px 3px 8px; border-radius: 14px; font-size: 11px;
    font-family: monospace; font-weight: 600; cursor: pointer;
    border: 2px solid transparent; user-select: none;
    transition: opacity .15s, border-color .15s, background .15s;
    white-space: nowrap;
}}
.map-net-chip .chip-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
.map-net-chip.all-chip {{ background: #f0f2f5; border-color: #aaa; color: #333; }}
.map-net-chip.all-chip.active {{ background: #333; color: #fff; border-color: #333; }}
.map-net-chip.net-chip {{ background: #f9f9f9; border-color: #ddd; color: #555; }}
.map-net-chip.net-chip.active {{ opacity: 1; }}
.map-net-chip.net-chip:not(.active) {{ opacity: 0.35; }}

/* ── Device Map SVG canvas tiles (drawn by JS into SVG) ────────── */
/* no DOM tile elements — all drawn in SVG for pan/zoom performance */

/* ── Map legend bar ───────────────────────────────────────────── */
.map-legend-bar {{
    display: flex; gap: 20px; align-items: center;
    padding: 8px 16px; font-size: 11px; color: #666;
    border-top: 1px solid #e8e8e8; background: #fff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    flex-wrap: wrap;
}}
.map-legend-item {{ display: flex; align-items: center; gap: 5px; }}
.map-legend-swatch {{
    width: 28px; height: 14px; border-radius: 3px; border: 1.5px solid #aaa;
}}
.map-legend-line {{
    width: 28px; height: 3px; border-radius: 2px;
}}
.map-legend-dot {{
    width: 10px; height: 10px; border-radius: 50%;
}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">Generated from routing.cc + host.cc</div>

<nav class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('detail', this)">Routing Detail</button>
    <button class="tab-btn" onclick="switchTab('map', this)">Device Map</button>
</nav>

<div id="panel-detail" class="tab-panel active">

<div class="controls">
    <button id="btn-expand" onclick="expandAll()">Expand All</button>
    <button id="btn-collapse" onclick="collapseAll()">Collapse All</button>
    <button id="btn-arrows" class="active" onclick="toggleArrows(this)">Routing Arrows</button>
</div>

{flow_panel_html}

<div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#e3f2fd;border-color:#42a5f5"></div> Shim</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#e8f5e9;border-color:#66bb6a"></div> Mem</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#fffde7;border-color:#fdd835"></div> AIE</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#e8eaf6;border-color:#7986cb"></div> Ping-Pong</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#bbdefb"></div> S2MM</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#ffe0b2"></div> MM2S</div>
    {block_legend}
    <div class="legend-item"><div class="legend-swatch" style="background:#fff;border:2px solid #1976d2"></div> Circuit (solid)</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#fff;border:2px dashed #7b1fa2"></div> Packet (dashed)</div>
</div>

<div class="pan-viewport" id="detail-viewport">
    <button class="pan-reset-btn" onclick="resetDetailPan()">Reset View</button>
    <div class="pan-canvas" id="detail-canvas">
        <div style="padding:20px; position:relative;">
        <div class="grid-wrapper">
        <div class="grid" id="tile-grid">
        {"".join(tile_cards)}
        </div>
        <svg class="routing-overlay" id="routing-svg">
            <defs>
                <marker id="arrowhead" markerWidth="8" markerHeight="6"
                        refX="7" refY="3" orient="auto" fill="currentColor">
                    <polygon points="0 0, 8 3, 0 6"/>
                </marker>
                <marker id="arrowhead-dashed" markerWidth="8" markerHeight="6"
                        refX="7" refY="3" orient="auto" fill="currentColor">
                    <polygon points="0 0, 8 3, 0 6"/>
                </marker>
                <marker id="arrowhead-sm" markerWidth="6" markerHeight="5"
                        refX="5" refY="2.5" orient="auto" fill="currentColor">
                    <polygon points="0 0, 6 2.5, 0 5"/>
                </marker>
            </defs>
        </svg>
        <div class="arrow-tooltip" id="arrow-tooltip"></div>
        </div>
        </div>
    </div>
    <div class="pan-hint">Scroll to zoom · Drag to pan</div>
</div>

</div><!-- /panel-detail -->

<div id="panel-map" class="tab-panel">
    <div class="map-header">
        <div class="map-header-title">Device map — full dataflow (streams + core-to-core) &nbsp;·&nbsp; <span style="float:right;color:#aaa">col 0–{ncols-1} &middot; row 0 (shim) at bottom</span></div>
        <div class="map-net-bar" id="map-net-bar"></div>
    </div>
    <div class="pan-viewport" id="map-viewport" style="border-radius:0;border-left:none;border-right:none;">
        <button class="pan-reset-btn" onclick="resetMapPan()">Reset View</button>
        <div class="pan-canvas" id="map-canvas">
            <svg id="map-svg" style="display:block;"></svg>
        </div>
        <div class="pan-hint">Scroll to zoom · Drag to pan · Click net chip to isolate</div>
    </div>
    <div class="map-legend-bar">
        <div class="map-legend-item">
            <div class="map-legend-swatch" style="background:#dbeafe;border-color:#93c5fd"></div> SHIM (PL/NoC gateway)
        </div>
        <div class="map-legend-item">
            <div class="map-legend-swatch" style="background:#dcfce7;border-color:#86efac"></div> MEM (memory tiles)
        </div>
        <div class="map-legend-item">
            <div class="map-legend-swatch" style="background:#fef9c3;border-color:#fde047"></div> AIE (compute cores)
        </div>
        <div class="map-legend-item">
            <div class="map-legend-line" style="background:#6366f1;height:2px"></div> colored = stream-switch route
        </div>
        <div class="map-legend-item">
            <div class="map-legend-line" style="background:#9ca3af;height:1.5px"></div> gray = core-to-core
        </div>
        <div class="map-legend-item">
            <div class="map-legend-dot" style="background:#333"></div> source (DMA / NoC)
        </div>
        <div class="map-legend-item">
            <div class="map-legend-dot" style="background:#fff;border:2px solid #333"></div> destination
        </div>
    </div>
</div><!-- /panel-map -->

<script>
const routingData = {routing_json_str};
const flowData = {flow_json_str};
let arrowsVisible = true;
let soloMode = false;
let soloFlowId = -1;

// Collect the set of currently-visible flow IDs from checkboxes
function getVisibleFlowIds() {{
    const visible = new Set();
    document.querySelectorAll('.flow-toggle input[type="checkbox"]').forEach(cb => {{
        if (cb.checked) visible.add(parseInt(cb.dataset.flowId));
    }});
    return visible;
}}

// Check if an element matches any visible flow.
// Handles both data-flow (single, DMA) and data-flow-ids (multi, routing).
function elMatchesAnyFlow(el, visibleSet) {{
    const singleFlow = el.getAttribute('data-flow');
    if (singleFlow !== null) {{
        return visibleSet.has(parseInt(singleFlow));
    }}
    const multiFlow = el.getAttribute('data-flow-ids');
    if (multiFlow !== null && multiFlow !== '') {{
        return multiFlow.split(',').some(id => visibleSet.has(parseInt(id)));
    }}
    return false;
}}

// Apply flow visibility to all flow-tagged DOM elements
function applyFlowVisibility() {{
    const visible = getVisibleFlowIds();
    // data-flow elements (DMA items)
    document.querySelectorAll('[data-flow]').forEach(el => {{
        el.classList.toggle('flow-hidden', !visible.has(parseInt(el.getAttribute('data-flow'))));
    }});
    // data-flow-ids elements (routing items)
    document.querySelectorAll('[data-flow-ids]').forEach(el => {{
        const ids = el.getAttribute('data-flow-ids').split(',');
        const anyMatch = ids.some(id => visible.has(parseInt(id)));
        el.classList.toggle('flow-hidden', !anyMatch);
    }});
}}

function updateTileCardDimming() {{
    document.querySelectorAll('.tile-card').forEach(card => {{
        const flowEls = card.querySelectorAll('[data-flow], [data-flow-ids]');
        if (flowEls.length === 0) return;
        const anyVisible = [...flowEls].some(el => !el.classList.contains('flow-hidden'));
        card.classList.toggle('flow-dimmed', !anyVisible);
    }});
}}

function toggleFlow(flowId, visible) {{
    applyFlowVisibility();
    updateTileCardDimming();
    setTimeout(drawArrows, 100);
}}

function showAllFlows() {{
    soloMode = false;
    soloFlowId = -1;
    document.querySelectorAll('.flow-toggle input[type="checkbox"]').forEach(cb => {{
        cb.checked = true;
    }});
    applyFlowVisibility();
    document.querySelectorAll('.flow-toggle').forEach(el => el.classList.remove('solo-active'));
    document.querySelectorAll('.tile-card').forEach(c => c.classList.remove('flow-dimmed'));
    setTimeout(drawArrows, 100);
}}

function hideAllFlows() {{
    soloMode = false;
    soloFlowId = -1;
    document.querySelectorAll('.flow-toggle input[type="checkbox"]').forEach(cb => {{
        cb.checked = false;
    }});
    applyFlowVisibility();
    document.querySelectorAll('.flow-toggle').forEach(el => el.classList.remove('solo-active'));
    updateTileCardDimming();
    setTimeout(drawArrows, 100);
}}

function soloFlowMode() {{
    soloMode = !soloMode;
    if (!soloMode) {{
        showAllFlows();
        return;
    }}
    // In solo mode, clicking a flow toggle shows only that flow
    // Start by showing none; user clicks to solo one
    hideAllFlows();
    soloMode = true;  // re-set since hideAllFlows resets it
}}

function soloOneFlow(flowId) {{
    // Hide all flows first
    document.querySelectorAll('.flow-toggle input[type="checkbox"]').forEach(cb => {{
        cb.checked = false;
    }});
    document.querySelectorAll('.flow-toggle').forEach(el => el.classList.remove('solo-active'));
    // Show only the selected flow
    document.querySelectorAll('.flow-toggle input[data-flow-id="' + flowId + '"]').forEach(cb => {{
        cb.checked = true;
        cb.closest('.flow-toggle').classList.add('solo-active');
    }});
    applyFlowVisibility();
    soloMode = true;
    soloFlowId = flowId;
    updateTileCardDimming();
    setTimeout(drawArrows, 100);
}}

function toggleFlowPanel(header) {{
    const body = header.nextElementSibling;
    if (body.classList.contains('collapsed')) {{
        body.classList.remove('collapsed');
        body.classList.add('expanded');
        header.innerHTML = header.innerHTML.replace('\\u25B6', '\\u25BC');
    }} else {{
        body.classList.remove('expanded');
        body.classList.add('collapsed');
        header.innerHTML = header.innerHTML.replace('\\u25BC', '\\u25B6');
    }}
}}

function toggleSection(header) {{
    const section = header.nextElementSibling;
    if (section.classList.contains('collapsed')) {{
        section.classList.remove('collapsed');
        section.classList.add('expanded');
        header.innerHTML = header.innerHTML.replace('\\u25B6', '\\u25BC');
    }} else {{
        section.classList.remove('expanded');
        section.classList.add('collapsed');
        header.innerHTML = header.innerHTML.replace('\\u25BC', '\\u25B6');
    }}
    setTimeout(drawArrows, 350);
}}

function expandAll() {{
    document.querySelectorAll('.collapsed').forEach(el => {{
        el.classList.remove('collapsed');
        el.classList.add('expanded');
    }});
    document.querySelectorAll('.section-header').forEach(h => {{
        h.innerHTML = h.innerHTML.replace('\\u25B6', '\\u25BC');
    }});
    setTimeout(drawArrows, 350);
}}

function collapseAll() {{
    document.querySelectorAll('.routing-section, .dma-detail-section').forEach(el => {{
        el.classList.remove('expanded');
        el.classList.add('collapsed');
    }});
    document.querySelectorAll('.section-header').forEach(h => {{
        h.innerHTML = h.innerHTML.replace('\\u25BC', '\\u25B6');
    }});
    setTimeout(drawArrows, 350);
}}

function toggleArrows(btn) {{
    arrowsVisible = !arrowsVisible;
    btn.classList.toggle('active', arrowsVisible);
    document.getElementById('routing-svg').style.display = arrowsVisible ? '' : 'none';
}}

function drawArrows() {{
    const svg = document.getElementById('routing-svg');
    const wrapper = document.querySelector('.grid-wrapper');
    const grid = document.getElementById('tile-grid');

    // Clear existing arrows
    while (svg.lastChild && svg.lastChild.tagName !== 'defs') {{
        svg.removeChild(svg.lastChild);
    }}

    const wrapperRect = wrapper.getBoundingClientRect();
    svg.setAttribute('width', wrapper.offsetWidth);
    svg.setAttribute('height', wrapper.offsetHeight);
    svg.setAttribute('viewBox', `0 0 ${{wrapper.offsetWidth}} ${{wrapper.offsetHeight}}`);

    if (!routingData.cross_tile_arrows) return;

    const tileElements = {{}};
    document.querySelectorAll('.tile-card').forEach(card => {{
        const col = parseInt(card.dataset.col);
        const row = parseInt(card.dataset.row);
        tileElements[col + ',' + row] = card;
    }});

    const dirOffset = {{
        'NORTH': {{ dx: 0, dy: -1 }},
        'SOUTH': {{ dx: 0, dy: 1 }},
        'EAST':  {{ dx: 1, dy: 0 }},
        'WEST':  {{ dx: -1, dy: 0 }},
    }};

    const drawn = new Set();

    // Collect visible flow IDs for filtering SVG arrows
    const visibleFlows = getVisibleFlowIds();

    // Helper: check if an arrow/conn should be shown based on its flow_ids
    function shouldShowElement(flowIds) {{
        if (!flowIds || flowIds.length === 0) return true;  // no flow info = always show
        return flowIds.some(fid => visibleFlows.has(fid));
    }}

    routingData.cross_tile_arrows.forEach(arrow => {{
        // Skip arrows where all associated flows are hidden
        if (!shouldShowElement(arrow.flow_ids)) return;

        const fromKey = arrow.from_tile[0] + ',' + arrow.from_tile[1];
        const toKey = arrow.to_tile[0] + ',' + arrow.to_tile[1];
        const fromEl = tileElements[fromKey];
        const toEl = tileElements[toKey];
        if (!fromEl || !toEl) return;

        const arrowKey = `${{fromKey}}-${{toKey}}-${{arrow.port}}-${{arrow.mode}}`;
        if (drawn.has(arrowKey)) return;
        drawn.add(arrowKey);

        const fromRect = fromEl.getBoundingClientRect();
        const toRect = toEl.getBoundingClientRect();

        const portSpread = 12;
        const portOffset = (arrow.port - 0.5) * portSpread;

        let x1, y1, x2, y2;
        const dir = arrow.from_dir;

        if (dir === 'NORTH') {{
            x1 = fromRect.left + fromRect.width / 2 + portOffset - wrapperRect.left;
            y1 = fromRect.top - wrapperRect.top;
            x2 = toRect.left + toRect.width / 2 + portOffset - wrapperRect.left;
            y2 = toRect.top + toRect.height - wrapperRect.top;
        }} else if (dir === 'SOUTH') {{
            x1 = fromRect.left + fromRect.width / 2 + portOffset - wrapperRect.left;
            y1 = fromRect.top + fromRect.height - wrapperRect.top;
            x2 = toRect.left + toRect.width / 2 + portOffset - wrapperRect.left;
            y2 = toRect.top - wrapperRect.top;
        }} else if (dir === 'EAST') {{
            x1 = fromRect.left + fromRect.width - wrapperRect.left;
            y1 = fromRect.top + fromRect.height / 2 + portOffset - wrapperRect.top;
            x2 = toRect.left - wrapperRect.left;
            y2 = toRect.top + toRect.height / 2 + portOffset - wrapperRect.top;
        }} else if (dir === 'WEST') {{
            x1 = fromRect.left - wrapperRect.left;
            y1 = fromRect.top + fromRect.height / 2 + portOffset - wrapperRect.top;
            x2 = toRect.left + toRect.width - wrapperRect.left;
            y2 = toRect.top + toRect.height / 2 + portOffset - wrapperRect.top;
        }}

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', arrow.color);
        line.setAttribute('stroke-width', arrow.mode === 'circuit' ? '2.5' : '2');
        line.style.color = arrow.color;
        if (arrow.mode === 'packet') {{
            line.setAttribute('stroke-dasharray', '6,3');
        }}
        line.setAttribute('marker-end', 'url(#arrowhead)');
        line.style.pointerEvents = 'stroke';
        line.style.cursor = 'pointer';

        const tooltip = document.getElementById('arrow-tooltip');
        line.addEventListener('mouseenter', (e) => {{
            tooltip.textContent = `${{arrow.block}}: ${{arrow.mode}} port=${{arrow.port}} (${{fromKey}} -> ${{toKey}})`;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX - wrapperRect.left + 10) + 'px';
            tooltip.style.top = (e.clientY - wrapperRect.top - 20) + 'px';
        }});
        line.addEventListener('mousemove', (e) => {{
            tooltip.style.left = (e.clientX - wrapperRect.left + 10) + 'px';
            tooltip.style.top = (e.clientY - wrapperRect.top - 20) + 'px';
        }});
        line.addEventListener('mouseleave', () => {{
            tooltip.style.display = 'none';
        }});

        svg.appendChild(line);
    }});

    // --- Draw internal routing connections within tiles (curved arrows) ---
    function getPortPos(tileRect, portType, portNum, isMaster) {{
        const ps = 12;
        const off = (portNum - 0.5) * ps;
        const r = tileRect;
        const inset = 6;
        if (portType === 'NORTH') return {{x: r.left + r.width/2 + off - wrapperRect.left, y: r.top - wrapperRect.top + inset}};
        if (portType === 'SOUTH') return {{x: r.left + r.width/2 + off - wrapperRect.left, y: r.top + r.height - wrapperRect.top - inset}};
        if (portType === 'EAST')  return {{x: r.left + r.width - wrapperRect.left - inset, y: r.top + r.height/2 + off - wrapperRect.top}};
        if (portType === 'WEST')  return {{x: r.left - wrapperRect.left + inset, y: r.top + r.height/2 + off - wrapperRect.top}};
        const xp = isMaster ? (r.left + r.width * 0.65 - wrapperRect.left) : (r.left + r.width * 0.35 - wrapperRect.left);
        return {{x: xp, y: r.top + r.height * 0.4 + portNum * 10 - wrapperRect.top}};
    }}

    const drawnInternal = new Set();
    if (routingData.internal_conns) {{
        routingData.internal_conns.forEach(conn => {{
            if (conn.mode === 'shim_dma') return;
            // Skip connections where all associated flows are hidden
            if (!shouldShowElement(conn.flow_ids)) return;

            const tKey = conn.tile[0] + ',' + conn.tile[1];
            const tileEl = tileElements[tKey];
            if (!tileEl) return;

            const connKey = `${{tKey}}-${{conn.slave_type}}-${{conn.slave_port}}-${{conn.master_type}}-${{conn.master_port}}-${{conn.mode}}`;
            if (drawnInternal.has(connKey)) return;
            drawnInternal.add(connKey);

            const tr = tileEl.getBoundingClientRect();
            const s = getPortPos(tr, conn.slave_type, conn.slave_port, false);
            const m = getPortPos(tr, conn.master_type, conn.master_port, true);

            const tcx = tr.left + tr.width / 2 - wrapperRect.left;
            const tcy = tr.top + tr.height / 2 - wrapperRect.top;

            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', `M ${{s.x}} ${{s.y}} Q ${{tcx}} ${{tcy}} ${{m.x}} ${{m.y}}`);
            path.setAttribute('stroke', conn.color);
            path.setAttribute('stroke-width', '1.5');
            path.setAttribute('fill', 'none');
            path.setAttribute('opacity', '0.55');
            path.style.color = conn.color;
            if (conn.mode === 'packet') {{
                path.setAttribute('stroke-dasharray', '4,2');
            }}
            path.setAttribute('marker-end', 'url(#arrowhead-sm)');
            path.style.pointerEvents = 'stroke';
            path.style.cursor = 'pointer';

            const tooltip = document.getElementById('arrow-tooltip');
            const label = `${{conn.block}}: ${{conn.slave_type}}:${{conn.slave_port}} \\u2192 ${{conn.master_type}}:${{conn.master_port}} (${{conn.mode}})`;
            path.addEventListener('mouseenter', (e) => {{
                tooltip.textContent = label;
                tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX - wrapperRect.left + 10) + 'px';
                tooltip.style.top = (e.clientY - wrapperRect.top - 20) + 'px';
            }});
            path.addEventListener('mousemove', (e) => {{
                tooltip.style.left = (e.clientX - wrapperRect.left + 10) + 'px';
                tooltip.style.top = (e.clientY - wrapperRect.top - 20) + 'px';
            }});
            path.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});

            svg.appendChild(path);

            [s, m].forEach(pos => {{
                const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                dot.setAttribute('cx', pos.x);
                dot.setAttribute('cy', pos.y);
                dot.setAttribute('r', '3');
                dot.setAttribute('fill', conn.color);
                dot.setAttribute('opacity', '0.7');
                svg.appendChild(dot);
            }});
        }});
    }}

}}

// ── Pan/Zoom shared factory ────────────────────────────────────
function makePanZoom(viewportId, canvasId, onAfterTransform) {{
    const vp = document.getElementById(viewportId);
    const canvas = document.getElementById(canvasId);
    let tx = 0, ty = 0, scale = 1;
    let dragging = false, lastX = 0, lastY = 0;

    function applyTransform() {{
        canvas.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;
        if (onAfterTransform) onAfterTransform(scale);
    }}

    vp.addEventListener('wheel', e => {{
        e.preventDefault();
        const rect = vp.getBoundingClientRect();
        const mx = e.clientX - rect.left, my = e.clientY - rect.top;
        const delta = e.deltaY < 0 ? 1.12 : 1/1.12;
        const newScale = Math.max(0.15, Math.min(5, scale * delta));
        tx = mx - (mx - tx) * (newScale / scale);
        ty = my - (my - ty) * (newScale / scale);
        scale = newScale;
        applyTransform();
    }}, {{ passive: false }});

    vp.addEventListener('mousedown', e => {{
        if (e.button !== 0) return;
        dragging = true; lastX = e.clientX; lastY = e.clientY;
        vp.classList.add('panning');
    }});
    window.addEventListener('mousemove', e => {{
        if (!dragging) return;
        tx += e.clientX - lastX; ty += e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;
        applyTransform();
    }});
    window.addEventListener('mouseup', () => {{
        dragging = false; vp.classList.remove('panning');
    }});

    function reset(initScale, initTx, initTy) {{
        scale = initScale || 1; tx = initTx || 0; ty = initTy || 0;
        applyTransform();
    }}
    return {{ reset, applyTransform, getState: () => ({{ tx, ty, scale }}) }};
}}

// ── Tab switching ─────────────────────────────────────────────
function switchTab(name, btn) {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + name).classList.add('active');
    if (name === 'map') setTimeout(buildMapSvg, 30);
    else setTimeout(drawArrows, 60);
}}

// ── Device Map (SVG canvas) ───────────────────────────────────
const mapTileData = {map_tile_data_str};

const NET_COLORS = [
    '#3b82f6','#ec4899','#10b981','#f59e0b',
    '#8b5cf6','#06b6d4','#ef4444','#6366f1',
    '#14b8a6','#f97316','#84cc16','#a855f7',
];
const allFlowIds = [...new Set(mapTileData.flatMap(t => t.flow_ids))].sort((a,b)=>a-b);

function netColor(fid) {{
    const idx = allFlowIds.indexOf(fid);
    return NET_COLORS[idx >= 0 ? idx % NET_COLORS.length : 0];
}}
function netLabel(fid) {{
    const fd = flowData.find(f => f.flow_id === fid);
    return fd ? `net${{fid}} · ${{fd.label}}` : `net${{fid}}`;
}}

let mapActiveFid = -1;
let mapPZ = null;

// SVG drawing constants
const TW = 160, TH = 90, GAP_X = 20, GAP_Y = 16;
const ROW_LABEL_W = 36, COL_LABEL_H = 24, PAD = 16;

function tilePos(col, row, cols, rows) {{
    const ci = cols.indexOf(col);
    const ri = rows.indexOf(row);  // rows is sorted descending, so ri=0 is top
    const x = ROW_LABEL_W + PAD + ci * (TW + GAP_X);
    const y = COL_LABEL_H + PAD + ri * (TH + GAP_Y);
    return {{ x, y, cx: x + TW/2, cy: y + TH/2 }};
}}

function buildNetBar() {{
    const netBar = document.getElementById('map-net-bar');
    netBar.innerHTML = '';
    const allChip = document.createElement('span');
    allChip.className = 'map-net-chip all-chip active';
    allChip.textContent = 'All nets';
    allChip.addEventListener('click', () => selectMapNet(-1));
    netBar.appendChild(allChip);

    allFlowIds.forEach(fid => {{
        const fd = flowData.find(f => f.flow_id === fid);
        const dir = fd ? (fd.direction === 'S2MM' ? 'gradf.in' : 'gradf.out') : '';
        const chip = document.createElement('span');
        chip.className = 'map-net-chip net-chip active';
        chip.dataset.fid = fid;
        const dot = document.createElement('span');
        dot.className = 'chip-dot';
        dot.style.background = netColor(fid);
        chip.appendChild(dot);
        chip.appendChild(document.createTextNode(`net${{fid}} · ${{dir}}[${{fid}}]`));
        chip.style.borderColor = netColor(fid);
        chip.style.color = netColor(fid);
        chip.addEventListener('click', () => selectMapNet(fid));
        netBar.appendChild(chip);
    }});
}}

function selectMapNet(fid) {{
    mapActiveFid = fid;
    document.querySelector('.map-net-chip.all-chip').classList.toggle('active', fid === -1);
    document.querySelectorAll('.map-net-chip.net-chip').forEach(c => {{
        c.classList.toggle('active', fid === -1 || parseInt(c.dataset.fid) === fid);
    }});
    buildMapSvg();
}}

function svgEl(tag, attrs) {{
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
}}

function buildMapSvg() {{
    if (!mapTileData.length) return;
    const svg = document.getElementById('map-svg');
    svg.innerHTML = '';

    const cols = [...new Set(mapTileData.map(t => t.col))].sort((a,b)=>a-b);
    const rows = [...new Set(mapTileData.map(t => t.row))].sort((a,b)=>b-a);  // top row first

    const totalW = ROW_LABEL_W + PAD*2 + cols.length*(TW+GAP_X) - GAP_X;
    const totalH = COL_LABEL_H + PAD*2 + rows.length*(TH+GAP_Y) - GAP_Y;
    svg.setAttribute('width', totalW);
    svg.setAttribute('height', totalH);

    const defs = svgEl('defs', {{}});
    // Build arrow markers for each flow
    allFlowIds.forEach(fid => {{
        if (mapActiveFid !== -1 && fid !== mapActiveFid) return;
        const color = netColor(fid);
        const mid = `arr-${{fid}}`;
        const mk = svgEl('marker', {{ id: mid, markerWidth:'8', markerHeight:'6',
            refX:'7', refY:'3', orient:'auto', fill: color }});
        mk.appendChild(svgEl('polygon', {{ points:'0 0,8 3,0 6' }}));
        defs.appendChild(mk);
    }});
    // Gray arrow marker for core-to-core
    const mkGray = svgEl('marker', {{ id:'arr-gray', markerWidth:'7', markerHeight:'5',
        refX:'6', refY:'2.5', orient:'auto', fill:'#9ca3af' }});
    mkGray.appendChild(svgEl('polygon', {{ points:'0 0,7 2.5,0 5' }}));
    defs.appendChild(mkGray);
    svg.appendChild(defs);

    // Column labels
    cols.forEach((c, ci) => {{
        const x = ROW_LABEL_W + PAD + ci*(TW+GAP_X) + TW/2;
        const t = svgEl('text', {{ x, y: COL_LABEL_H-4, 'text-anchor':'middle',
            'font-size':'11', fill:'#9ca3af', 'font-family':'monospace' }});
        t.textContent = `col ${{c}}`;
        svg.appendChild(t);
    }});

    // Row labels
    rows.forEach((r, ri) => {{
        const y = COL_LABEL_H + PAD + ri*(TH+GAP_Y) + TH/2 + 4;
        const t = svgEl('text', {{ x: ROW_LABEL_W-4, y, 'text-anchor':'end',
            'font-size':'11', fill:'#9ca3af', 'font-family':'monospace' }});
        t.textContent = `r${{r}}`;
        svg.appendChild(t);
    }});

    // Tile backgrounds
    const TILE_FILL = {{ Shim:'#dbeafe', Mem:'#dcfce7', AIE:'#fef9c3' }};
    const TILE_STROKE = {{ Shim:'#93c5fd', Mem:'#86efac', AIE:'#fde047' }};
    const TILE_INACTIVE_OPACITY = '0.3';

    mapTileData.forEach(t => {{
        const p = tilePos(t.col, t.row, cols, rows);
        const fill = TILE_FILL[t.type] || '#f3f4f6';
        const stroke = TILE_STROKE[t.type] || '#d1d5db';
        const opacity = t.active ? '1' : TILE_INACTIVE_OPACITY;

        // Tile rect
        svg.appendChild(svgEl('rect', {{
            x: p.x, y: p.y, width: TW, height: TH,
            rx:'6', ry:'6', fill, stroke, 'stroke-width':'1.5', opacity
        }}));

        // Tile type badge (top right)
        const typeLabel = t.type === 'Shim' ? 'SHIM' : t.type === 'Mem' ? 'MEM' : 'AIE';
        const badge = svgEl('text', {{ x: p.x+TW-6, y: p.y+13, 'text-anchor':'end',
            'font-size':'9', fill:'#9ca3af', 'font-family':'monospace', opacity }});
        badge.textContent = typeLabel;
        svg.appendChild(badge);

        // Tile coord (top left)
        const coord = svgEl('text', {{ x: p.x+6, y: p.y+13,
            'font-size':'11', 'font-weight':'700', fill:'#374151',
            'font-family':'monospace', opacity }});
        coord.textContent = `(${{t.col}},${{t.row}})`;
        svg.appendChild(coord);

        // Buffer/stream labels inside tile
        if (t.active && t.labels && t.labels.length) {{
            const visLabels = mapActiveFid === -1
                ? t.labels
                : t.labels.filter(lb => lb.flow_id === mapActiveFid || lb.flow_id === -1);
            visLabels.slice(0,3).forEach((lb, i) => {{
                const color = lb.flow_id >= 0 ? netColor(lb.flow_id) : '#6b7280';
                const dimmed = mapActiveFid !== -1 && lb.flow_id !== mapActiveFid && lb.flow_id !== -1;
                const lbl = svgEl('text', {{ x: p.cx, y: p.y+30+i*14,
                    'text-anchor':'middle', 'font-size':'9', fill: dimmed ? '#d1d5db' : color,
                    'font-family':'monospace' }});
                lbl.textContent = lb.text;
                svg.appendChild(lbl);
            }});
        }}
    }});

    // ── Gray core-to-core arrows (routing edges, no flow association) ──
    const grp_gray = svgEl('g', {{ opacity: mapActiveFid === -1 ? '0.55' : '0.12' }});
    mapTileData.forEach(t => {{
        if (!t.rt_edges) return;
        const p = tilePos(t.col, t.row, cols, rows);
        t.rt_edges.forEach((edge, i) => {{
            // Only draw EAST/WEST edges as horizontal core-to-core lines
            // (NORTH/SOUTH are handled by stream lines)
            const masterDir = (edge.master || '').split(':')[0];
            if (!['EAST','WEST','NORTH','SOUTH'].includes(masterDir)) return;
            if (edge.flow_ids && edge.flow_ids.length > 0) return; // has flow — skip gray
            // Draw a small directed stub from tile center toward master direction
            const offX = masterDir === 'EAST' ? TW/2+GAP_X/2 : masterDir === 'WEST' ? -(TW/2+GAP_X/2) : 0;
            const offY = masterDir === 'NORTH' ? -(TH/2+GAP_Y/2) : masterDir === 'SOUTH' ? TH/2+GAP_Y/2 : 0;
            const x1 = p.cx + (offX > 0 ? TW*0.3 : offX < 0 ? -TW*0.3 : 0);
            const y1 = p.cy + (offY > 0 ? TH*0.3 : offY < 0 ? -TH*0.3 : 0);
            const x2 = p.cx + offX * 0.6;
            const y2 = p.cy + offY * 0.6;
            grp_gray.appendChild(svgEl('line', {{
                x1, y1, x2, y2, stroke:'#9ca3af', 'stroke-width':'1.5',
                'marker-end':'url(#arr-gray)'
            }}));
        }});
    }});
    svg.appendChild(grp_gray);

    // ── Colored stream lines (one vertical line per flow per column) ──
    // For each flow, draw a vertical line in the shim column and horizontal
    // lines branching to each connected tile — matching the reference image style.
    const tileMap = {{}};
    mapTileData.forEach(t => {{ tileMap[t.col+','+t.row] = tilePos(t.col, t.row, cols, rows); }});

    // Assign each flow a horizontal offset slot within its shim column
    const colFlowSlots = {{}};  // col -> [fid, ...]
    flowData.forEach(fd => {{
        const c = fd.shim_tile[0];
        if (!colFlowSlots[c]) colFlowSlots[c] = [];
        colFlowSlots[c].push(fd.flow_id);
    }});

    flowData.forEach(fd => {{
        const fid = fd.flow_id;
        const visible = mapActiveFid === -1 || mapActiveFid === fid;
        const color = netColor(fid);
        const alpha = visible ? '1' : '0.1';

        const shimP = tileMap[fd.shim_tile[0]+','+fd.shim_tile[1]];
        if (!shimP) return;

        // Slot offset so multiple flows in the same column don't overlap
        const slots = colFlowSlots[fd.shim_tile[0]] || [fid];
        const slotIdx = slots.indexOf(fid);
        const slotCount = slots.length;
        const xOffset = (slotIdx - (slotCount-1)/2) * 10;
        const lineX = shimP.cx + xOffset;

        // Shim anchor dot
        const isSource = fd.direction === 'MM2S';
        svg.appendChild(svgEl('circle', {{
            cx: lineX, cy: isSource ? shimP.y+TH : shimP.y,
            r:'5', fill: isSource ? color : 'white',
            stroke: color, 'stroke-width':'2', opacity: alpha
        }}));

        // Arrow marker at bottom of line into shim
        const markerId = `arr-${{fid}}`;

        (fd.connected_tiles || []).forEach(ct => {{
            const ctP = tileMap[ct[0]+','+ct[1]];
            if (!ctP) return;
            const ctSlots = colFlowSlots[ct[0]] || [fid];
            const ctSlotIdx = ctSlots.indexOf(fid);
            const ctSlotCount = ctSlots.length;
            const ctXOffset = (ctSlotIdx - (ctSlotCount-1)/2) * 10;
            const ctLineX = ctP.cx + ctXOffset;

            // Vertical segment from shim to compute tile row
            const shimAnchorY = isSource ? shimP.y + TH/2 : shimP.y + TH/2;
            const ctAnchorY = isSource ? ctP.cy : ctP.cy;

            // Draw L-shaped path: vertical in shim column, then step to compute col if different
            let d;
            if (fd.shim_tile[0] === ct[0]) {{
                // Same column: straight vertical
                const y1 = isSource ? shimP.y + TH : shimP.y;
                const y2 = isSource ? ctP.y + TH/2 : ctP.y + TH/2;
                d = `M${{lineX}},${{y1}} L${{lineX}},${{y2}}`;
            }} else {{
                // Different column: vertical then horizontal then vertical
                const shimTopY = Math.min(shimP.y, ctP.y) - GAP_Y;
                const y1 = isSource ? shimP.y + TH : shimP.y;
                const y2 = isSource ? ctP.y + TH/2 : ctP.y + TH/2;
                d = `M${{lineX}},${{y1}} L${{lineX}},${{shimTopY}} L${{ctLineX}},${{shimTopY}} L${{ctLineX}},${{y2}}`;
            }}

            svg.appendChild(svgEl('path', {{
                d, stroke: color, 'stroke-width':'2', fill:'none',
                opacity: alpha,
                'marker-end': visible ? `url(#${{markerId}})` : ''
            }}));

            // Destination dot on compute tile
            svg.appendChild(svgEl('circle', {{
                cx: ctLineX, cy: ctP.cy,
                r:'4', fill: isSource ? 'white' : color,
                stroke: color, 'stroke-width':'2', opacity: alpha
            }}));
        }});
    }});

}}

let detailPZ = null;

function resetDetailPan() {{
    if (detailPZ) detailPZ.reset(1, 20, 20);
}}
function resetMapPan() {{
    if (mapPZ) mapPZ.reset(0.85, 20, 20);
}}

window.addEventListener('load', () => {{
    detailPZ = makePanZoom('detail-viewport', 'detail-canvas', null);
    detailPZ.reset(1, 20, 20);
    mapPZ = makePanZoom('map-viewport', 'map-canvas', null);
    mapPZ.reset(0.85, 20, 20);
    drawArrows();
    buildNetBar();
    // Build SVG map on first switch to map tab
}});
window.addEventListener('resize', () => {{
    drawArrows();
}});
</script>
</body>
</html>
"""
    Path(output_path).write_text(html)
    print(f"HTML saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# PNG output (matplotlib)
# ---------------------------------------------------------------------------

def render_png(
    blocks: Optional[List[RoutingBlock]],
    dma_tiles: Optional[Dict[Tuple[int, int], TileDmaInfo]],
    routing_tiles: Optional[set],
    routing_summary: Optional[Dict[tuple, List[dict]]],
    output_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    all_tile_locs = set()
    if routing_tiles:
        all_tile_locs |= routing_tiles
    if dma_tiles:
        all_tile_locs |= set(dma_tiles.keys())

    if not all_tile_locs:
        print("No tiles found.", file=sys.stderr)
        return

    cols = sorted({t[0] for t in all_tile_locs})
    rows = sorted({t[1] for t in all_tile_locs})
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)
    all_cols = list(range(min_col, max_col + 1))
    all_rows = list(range(min_row, max_row + 1))

    has_dma = bool(dma_tiles)
    cell_w = 3.0
    cell_h = 2.6 if has_dma else 1.8
    pad_x, pad_y = 1.5, 1.0
    fig_w = len(all_cols) * cell_w + 2 * pad_x
    fig_h = len(all_rows) * cell_h + 2 * pad_y + 1.5

    fig, ax = plt.subplots(figsize=(max(fig_w, 8), max(fig_h, 6)))
    ax.set_xlim(-pad_x, len(all_cols) * cell_w + pad_x)
    ax.set_ylim(-pad_y - 1.0, len(all_rows) * cell_h + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    tile_centers = {}
    type_colors = {"Shim": "#FFECB3", "Mem": "#C8E6C9", "AIE": "#BBDEFB"}

    for ci, c in enumerate(all_cols):
        for ri, r in enumerate(all_rows):
            cx = ci * cell_w + cell_w / 2
            cy = ri * cell_h + cell_h / 2
            tile_centers[(c, r)] = (cx, cy)
            ttype = TILE_TYPE_BY_ROW(r)
            is_active = (c, r) in all_tile_locs
            fc = type_colors.get(ttype, "#EEE") if is_active else "#F5F5F5"
            ec = "#333" if is_active else "#CCC"
            lw = 1.5 if is_active else 0.5
            rect = mpatches.FancyBboxPatch(
                (cx - cell_w * 0.42, cy - cell_h * 0.40),
                cell_w * 0.84, cell_h * 0.80,
                boxstyle="round,pad=0.05",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1)
            ax.add_patch(rect)
            ax.text(cx, cy + cell_h * 0.25, f"({c},{r})",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#333" if is_active else "#AAA", zorder=5)
            ax.text(cx, cy + cell_h * 0.12, ttype,
                    ha="center", va="center", fontsize=6.5,
                    color="#666" if is_active else "#CCC", zorder=5)

    for ci, c in enumerate(all_cols):
        cx = ci * cell_w + cell_w / 2
        ax.text(cx, len(all_rows) * cell_h + pad_y * 0.3,
                f"Col {c}", ha="center", va="center", fontsize=9, fontweight="bold")
    for ri, r in enumerate(all_rows):
        cy = ri * cell_h + cell_h / 2
        ax.text(-pad_x * 0.55, cy,
                f"Row {r}", ha="center", va="center", fontsize=9, fontweight="bold")

    port_offsets = {
        "NORTH": (0, 0.38), "SOUTH": (0, -0.38),
        "EAST": (0.38, 0), "WEST": (-0.38, 0),
    }

    def tile_port_pos(tile, direction, port_num, spread=0.2):
        cx, cy = tile_centers[tile]
        dx, dy = port_offsets.get(direction, (0, 0))
        px = cx + dx * cell_w
        py = cy + dy * cell_h
        if direction in ("NORTH", "SOUTH"):
            px += (port_num - 0.5) * spread
        elif direction in ("EAST", "WEST"):
            py += (port_num - 0.5) * spread
        return px, py

    if blocks:
        drawn_internal = set()
        drawn_cross = set()
        for bi, block in enumerate(blocks):
            color = BLOCK_COLORS[bi % len(BLOCK_COLORS)]

            for cc in block.circuit_conns:
                if cc.tile not in tile_centers:
                    continue
                key = (cc.tile, cc.slave_type, cc.slave_port, cc.master_type, cc.master_port, bi)
                if key in drawn_internal:
                    continue
                drawn_internal.add(key)
                sx, sy = tile_port_pos(cc.tile, cc.slave_type, cc.slave_port)
                mx, my = tile_port_pos(cc.tile, cc.master_type, cc.master_port)
                ax.annotate("", xy=(mx, my), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.3, connectionstyle="arc3,rad=0.15"),
                            zorder=3)

                mdir = cc.master_type
                if mdir in DIRECTION_DELTA:
                    dc, dr = DIRECTION_DELTA[mdir]
                    nb = (cc.tile[0] + dc, cc.tile[1] + dr)
                    if nb in tile_centers:
                        ckey = (cc.tile, mdir, cc.master_port, nb, bi)
                        if ckey not in drawn_cross:
                            drawn_cross.add(ckey)
                            esx, esy = tile_port_pos(cc.tile, mdir, cc.master_port)
                            opp = DIRECTION_OPPOSITES[mdir]
                            eex, eey = tile_port_pos(nb, opp, cc.master_port)
                            ax.annotate("", xy=(eex, eey), xytext=(esx, esy),
                                        arrowprops=dict(arrowstyle="-|>", color=color,
                                                        lw=1.8),
                                        zorder=4)

            pkt_edges = resolve_pkt_internal_edges(block)
            for ss, mp in pkt_edges:
                if ss.tile not in tile_centers:
                    continue
                key = (ss.tile, ss.port_type, ss.port_num, mp.port_type, mp.port_num, bi)
                if key in drawn_internal:
                    continue
                drawn_internal.add(key)
                sx, sy = tile_port_pos(ss.tile, ss.port_type,
                                       ss.port_num if ss.port_type in DIRECTION_DELTA else 0)
                mx, my = tile_port_pos(ss.tile, mp.port_type,
                                       mp.port_num if mp.port_type in DIRECTION_DELTA else 0)
                tcx, tcy = tile_centers[ss.tile]
                if ss.port_type not in DIRECTION_DELTA:
                    sx = tcx - cell_w * 0.25
                    sy = tcy
                if mp.port_type not in DIRECTION_DELTA:
                    mx = tcx + cell_w * 0.25
                    my = tcy
                ax.annotate("", xy=(mx, my), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.3, linestyle="dashed",
                                            connectionstyle="arc3,rad=0.2"),
                            zorder=3)

                mdir = mp.port_type
                if mdir in DIRECTION_DELTA:
                    dc, dr = DIRECTION_DELTA[mdir]
                    nb = (mp.tile[0] + dc, mp.tile[1] + dr)
                    if nb in tile_centers:
                        ckey = (mp.tile, mdir, mp.port_num, nb, bi)
                        if ckey not in drawn_cross:
                            drawn_cross.add(ckey)
                            esx, esy = tile_port_pos(mp.tile, mdir, mp.port_num)
                            opp = DIRECTION_OPPOSITES[mdir]
                            eex, eey = tile_port_pos(nb, opp, mp.port_num)
                            ax.annotate("", xy=(eex, eey), xytext=(esx, esy),
                                        arrowprops=dict(arrowstyle="-|>", color=color,
                                                        lw=1.8, linestyle="dashed"),
                                        zorder=4)

            for sd in block.shim_dma:
                if sd.tile not in tile_centers:
                    continue
                tcx, tcy = tile_centers[sd.tile]
                arrow = "V" if sd.direction == "aie_to_shim" else "^"
                ax.text(tcx, tcy - cell_h * 0.0,
                        f"{arrow} ShimDMA:{sd.port_num}",
                        ha="center", va="center", fontsize=5.5,
                        color=color, fontweight="bold", zorder=6)

    if dma_tiles:
        for tile, info in dma_tiles.items():
            if tile not in tile_centers:
                continue
            tcx, tcy = tile_centers[tile]
            y_start = tcy - cell_h * 0.05
            line_i = 0
            for io in info.ios:
                y_pos = y_start - line_i * cell_h * 0.10
                io_color = "#1565C0" if io.direction == "MM2S" else "#C62828"
                ax.text(tcx, y_pos, f"ch{io.channel_id}:{io.direction}",
                        ha="center", va="center", fontsize=5, color=io_color,
                        fontfamily="monospace", zorder=7)
                line_i += 1

            for bd in info.bds[:3]:
                y_pos = y_start - line_i * cell_h * 0.10
                pkt = f"pkt{bd.packet_id}" if bd.enable_packet else ""
                txt = f"BD{bd.bd_id} {pkt} len={bd.length}"
                ax.text(tcx, y_pos, txt,
                        ha="center", va="center", fontsize=4.5,
                        fontfamily="monospace", color="#333",
                        bbox=dict(boxstyle="round,pad=0.06", fc="white",
                                  ec="#BDBDBD", alpha=0.85, lw=0.4),
                        zorder=7)
                line_i += 1
            if len(info.bds) > 3:
                y_pos = y_start - line_i * cell_h * 0.10
                ax.text(tcx, y_pos, f"+{len(info.bds)-3} more",
                        ha="center", va="center", fontsize=3.5, color="#888", zorder=7)

    legend_handles = []
    if blocks:
        for bi, block in enumerate(blocks):
            c = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
            legend_handles.append(mpatches.Patch(color=c, label=block.name))
        legend_handles.append(plt.Line2D([0], [0], color="gray", lw=1.5,
                                         label="Circuit (solid)"))
        legend_handles.append(plt.Line2D([0], [0], color="gray", lw=1.5,
                                         linestyle="dashed", label="Packet (dashed)"))
    if has_dma:
        legend_handles.append(mpatches.Patch(color="#BBDEFB", label="S2MM (blue)"))
        legend_handles.append(mpatches.Patch(color="#FFE0B2", label="MM2S (orange)"))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower center",
                  ncol=min(len(legend_handles), 6), fontsize=6.5, frameon=True,
                  bbox_to_anchor=(0.5, -0.08))

    title_parts = []
    if blocks:
        title_parts.append("Routing")
    if has_dma:
        title_parts.append("DMA BD")
    ax.set_title(" + ".join(title_parts) + " — AIE Topology",
                 fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"PNG saved to {output_path}")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def serve_html(html_path: str, port: int) -> None:
    abs_path = Path(html_path).resolve()
    directory = str(abs_path.parent)
    filename = abs_path.name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)
        def log_message(self, fmt, *args):
            pass

    hostname = socket.gethostname()
    local_url = f"http://localhost:{port}/{filename}"
    remote_url = f"http://{hostname}:{port}/{filename}"
    print(f"Serving AIE visualization:")
    print(f"  Local:  {local_url}")
    print(f"  Remote: {remote_url}")
    print(f"  (Ctrl+C to stop)")

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    try:
        webbrowser.open(local_url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_default_worklocal() -> Path:
    """Locate the worklocal directory holding routing.cc/host.cc.

    Searches likely output dirs (aiehlc.sh's aout/worklocal first, then the
    unitest worklocal) and returns the first that actually contains a
    routing.cc or host.cc. Falls back to aout/worklocal so the error message
    points at the primary location when nothing is found."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    candidates = [
        repo_root / "aout" / "worklocal",
        Path.cwd() / "aout" / "worklocal",
        repo_root / "src" / "mlir" / "mlirfront" / "tilinglinalg"
        / "pass" / "unitest" / "worklocal",
    ]
    for wl in candidates:
        if (wl / "routing.cc").exists() or (wl / "host.cc").exists():
            return wl
    return candidates[0]


def main():
    worklocal = _find_default_worklocal()
    default_routing = str(worklocal / "routing.cc") if (worklocal / "routing.cc").exists() else None
    default_host = str(worklocal / "host.cc") if (worklocal / "host.cc").exists() else None

    parser = argparse.ArgumentParser(
        description="Unified AIE visualizer: routing topology + DMA BD configuration")
    parser.add_argument("--routing", default=default_routing,
                        help=f"Path to routing.cc (default: {default_routing})")
    parser.add_argument("--host", default=default_host,
                        help=f"Path to host.cc (default: {default_host})")
    parser.add_argument("-m", "--mode", choices=["text", "png", "html"],
                        default=None, help="Output mode (default: html)")
    parser.add_argument("-o", "--output", default=None, help="Output file path")
    parser.add_argument("--serve", action="store_true", default=True,
                        help="Start HTTP server after generating HTML (default: enabled)")
    parser.add_argument("--no-serve", dest="serve", action="store_false",
                        help="Disable HTTP server")
    parser.add_argument("--port", type=int, default=8088,
                        help="HTTP server port (default: 8088)")
    args = parser.parse_args()

    if not args.routing and not args.host:
        parser.error("At least one of --routing or --host is required.\n"
                     f"  Looked for defaults in: {worklocal}")

    blocks = None
    routing_tiles = None
    paths = None
    routing_summary = None
    dma_tiles = None
    flows = None

    if args.routing:
        blocks = parse_routing_file(args.routing)
        if not blocks:
            print("No routing blocks found.", file=sys.stderr)
        else:
            routing_tiles = collect_routing_tiles(blocks)
            paths = trace_paths(blocks)
            routing_summary = _build_routing_tile_summary(blocks)

    if args.host:
        dma_tiles, flows = parse_host_cc(args.host)
        if not dma_tiles:
            print("No DMA BD configurations found.", file=sys.stderr)

    if not blocks and not dma_tiles:
        print("Nothing to visualize.", file=sys.stderr)
        sys.exit(1)

    mode = args.mode or "html"

    if mode == "text":
        out = sys.stdout
        if args.output:
            out = open(args.output, "w")
        render_text(blocks, paths, dma_tiles, routing_summary, out)
        if args.output:
            out.close()
            print(f"Text report saved to {args.output}")

    elif mode == "png":
        out_path = args.output or "aie_topology.png"
        render_png(blocks, dma_tiles, routing_tiles, routing_summary, out_path)

    elif mode == "html":
        out_path = args.output or "aie_topology.html"
        render_html(blocks, dma_tiles, routing_tiles, routing_summary, out_path,
                    flows=flows)
        if args.serve:
            serve_html(out_path, args.port)


if __name__ == "__main__":
    main()
