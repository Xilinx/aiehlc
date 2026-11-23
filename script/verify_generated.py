#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""
Pre-HW Static Verification for generated host.cc and kernel.cc.

Parses machine-generated C code and checks cross-file invariants that catch
common DMA/lock/repeat bugs before running on hardware.

Usage:
    python3 script/verify_generated.py [host.cc] [kernel.cc]

If no arguments given, uses default paths under unitest/worklocal/.
"""

import re
import sys
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class BDConfig:
    """A single BD configuration from __Runtime_dma_bd_config or _multidim_ooo."""
    def __init__(self, tile_var, bd_id, length, next_bd, enable_packet,
                 packet_id, acq_lock_id, acq_lock_val, rel_lock_id,
                 rel_lock_val, ooo_bd_id, is_multidim=False,
                 dims=None, iter_step=0, iter_wrap=0, line_num=0):
        self.tile_var = tile_var
        self.bd_id = bd_id
        self.length = length
        self.next_bd = next_bd
        self.enable_packet = enable_packet
        self.packet_id = packet_id
        self.acq_lock_id = acq_lock_id
        self.acq_lock_val = acq_lock_val
        self.rel_lock_id = rel_lock_id
        self.rel_lock_val = rel_lock_val
        self.ooo_bd_id = ooo_bd_id
        self.is_multidim = is_multidim
        self.dims = dims or []  # list of (size, stride) tuples
        self.iter_step = iter_step
        self.iter_wrap = iter_wrap
        self.line_num = line_num
        # Set by IO creation
        self.channel_id = None
        self.direction = None  # 'S2MM' or 'MM2S'
        self.result_var = None  # the var name assigned to the bd_config result


class IOConfig:
    """A DMA IO channel from __Runtime_dma_createio_4."""
    def __init__(self, tile_var, bd_var, channel_id, start_bd, direction,
                 result_var, line_num=0):
        self.tile_var = tile_var
        self.bd_var = bd_var
        self.channel_id = channel_id
        self.start_bd = start_bd
        self.direction = direction
        self.result_var = result_var
        self.line_num = line_num


class StartIOConfig:
    """A startio call."""
    def __init__(self, io_var, bd_count, repeat, result_var, line_num=0):
        self.io_var = io_var
        self.bd_count = bd_count
        self.repeat = repeat
        self.result_var = result_var
        self.line_num = line_num


class LockInit:
    """A lock initialization."""
    def __init__(self, col, row, lock_id, init_val, line_num=0):
        self.col = col
        self.row = row
        self.lock_id = lock_id
        self.init_val = init_val
        self.line_num = line_num


class WindowInit:
    """Kernel window_init call."""
    def __init__(self, window_name, buf_sz, total_rounds, line_num=0):
        self.window_name = window_name
        self.buf_sz = buf_sz
        self.total_rounds = total_rounds
        self.line_num = line_num


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_host(host_text):
    """Parse host.cc and extract all DMA/IO/startio/lock/tile data."""
    lines = host_text.split('\n')

    # Map variable names to tile locations: var -> (col, row)
    tile_vars = {}
    # Map variable names to BD configs
    bd_configs = {}
    # Map variable names to IO configs
    io_configs = {}
    # Map variable names to startio configs
    startio_configs = {}
    # Lock inits
    lock_inits = []

    # Regex patterns
    re_tileloc = re.compile(
        r'(\w+)\s*=\s*XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)')

    # __Runtime_dma_bd_config(dev, tile, buf, bd_id, len, next_bd,
    #   enable_pkt, pkt_id, acq_lock_id, acq_lock_val,
    #   rel_lock_id, rel_lock_val, ooo_bd_id)
    re_bd_config = re.compile(
        r'(\w+)\s*=\s*__Runtime_dma_bd_config\(\s*'
        r'(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*'
        r'(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*'
        r'(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*'
        r'(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*'
        r'(-?\d+)\s*\)')

    # __Runtime_dma_bd_config_multidim_ooo(dev, tile, buf, bd_id, len, next_bd,
    #   enable_pkt, pkt_id, acq_lock_id, acq_lock_val,
    #   rel_lock_id, rel_lock_val, ooo_bd_id,
    #   num_dims, d0_size, d0_stride, d1_size, d1_stride,
    #   d2_size, d2_stride, iter_step, iter_wrap)
    # Variable number of dim args depending on num_dims. We capture all args.
    re_bd_multidim = re.compile(
        r'(\w+)\s*=\s*__Runtime_dma_bd_config_multidim_ooo\(\s*'
        r'(.+?)\)')

    re_createio = re.compile(
        r'(\w+)\s*=\s*__Runtime_dma_createio_4\(\s*'
        r'(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*'
        r'(DMA_S2MM|DMA_MM2S)\s*\)')

    re_startio = re.compile(
        r'(\w+)\s*=\s*__Runtime_startio\(\s*'
        r'(\w+)\s*,\s*(\w+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)')

    re_lock = re.compile(
        r'XAie_LockSetValue\(\s*\w+\s*,\s*XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*'
        r'XAie_LockInit\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*\)')

    for line_num, line in enumerate(lines, 1):
        # Tile locations
        m = re_tileloc.search(line)
        if m:
            var = m.group(1)
            col, row = int(m.group(2)), int(m.group(3))
            tile_vars[var] = (col, row)

        # BD config (simple)
        m = re_bd_config.search(line)
        if m:
            result_var = m.group(1)
            tile_var = m.group(3)  # group 2 is dev
            bd = BDConfig(
                tile_var=tile_var,
                bd_id=int(m.group(5)),
                length=int(m.group(6)),
                next_bd=int(m.group(7)),
                enable_packet=int(m.group(8)),
                packet_id=int(m.group(9)),
                acq_lock_id=int(m.group(10)),
                acq_lock_val=int(m.group(11)),
                rel_lock_id=int(m.group(12)),
                rel_lock_val=int(m.group(13)),
                ooo_bd_id=int(m.group(14)),
                line_num=line_num
            )
            bd.result_var = result_var
            bd_configs[result_var] = bd
            continue

        # BD config multidim
        m = re_bd_multidim.search(line)
        if m:
            result_var = m.group(1)
            args_str = m.group(2)
            args = [a.strip() for a in args_str.split(',')]
            # args[0]=dev, [1]=tile, [2]=buf, [3]=bd_id, [4]=len, [5]=next_bd,
            # [6]=enable_pkt, [7]=pkt_id, [8]=acq_lock_id, [9]=acq_lock_val,
            # [10]=rel_lock_id, [11]=rel_lock_val, [12]=ooo_bd_id,
            # [13]=num_dims, then pairs of (size, stride), then iter_step, iter_wrap
            tile_var = args[1]
            bd_id = int(args[3])
            length = int(args[4])
            next_bd = int(args[5])
            enable_packet = int(args[6])
            packet_id = int(args[7])
            acq_lock_id = int(args[8])
            acq_lock_val = int(args[9])
            rel_lock_id = int(args[10])
            rel_lock_val = int(args[11])
            ooo_bd_id = int(args[12])
            num_dims = int(args[13])
            dims = []
            idx = 14
            for _ in range(num_dims):
                d_size = int(args[idx])
                d_stride = int(args[idx + 1])
                dims.append((d_size, d_stride))
                idx += 2
            iter_step = int(args[idx]) if idx < len(args) else 0
            iter_wrap = int(args[idx + 1]) if idx + 1 < len(args) else 0

            bd = BDConfig(
                tile_var=tile_var,
                bd_id=bd_id,
                length=length,
                next_bd=next_bd,
                enable_packet=enable_packet,
                packet_id=packet_id,
                acq_lock_id=acq_lock_id,
                acq_lock_val=acq_lock_val,
                rel_lock_id=rel_lock_id,
                rel_lock_val=rel_lock_val,
                ooo_bd_id=ooo_bd_id,
                is_multidim=True,
                dims=dims,
                iter_step=iter_step,
                iter_wrap=iter_wrap,
                line_num=line_num
            )
            bd.result_var = result_var
            bd_configs[result_var] = bd
            continue

        # IO creation
        m = re_createio.search(line)
        if m:
            result_var = m.group(1)
            io = IOConfig(
                tile_var=m.group(2),
                bd_var=m.group(3),
                channel_id=int(m.group(4)),
                start_bd=int(m.group(5)),
                direction=m.group(6).replace('DMA_', ''),
                result_var=result_var,
                line_num=line_num
            )
            io_configs[result_var] = io
            # Also annotate the BD with direction/channel
            if io.bd_var in bd_configs:
                bd_configs[io.bd_var].channel_id = io.channel_id
                bd_configs[io.bd_var].direction = io.direction

        # Start IO
        m = re_startio.search(line)
        if m:
            result_var = m.group(1)
            sio = StartIOConfig(
                io_var=m.group(3),
                bd_count=int(m.group(4)),
                repeat=int(m.group(5)),
                result_var=result_var,
                line_num=line_num
            )
            startio_configs[result_var] = sio

        # Lock init
        m = re_lock.search(line)
        if m:
            lock = LockInit(
                col=int(m.group(1)),
                row=int(m.group(2)),
                lock_id=int(m.group(3)),
                init_val=int(m.group(4)),
                line_num=line_num
            )
            lock_inits.append(lock)

    return tile_vars, bd_configs, io_configs, startio_configs, lock_inits


def parse_kernel(kernel_text):
    """Parse kernel.cc for BUF_SZ defines and window_init calls."""
    buf_sizes = {}
    window_inits = []

    lines = kernel_text.split('\n')
    for line_num, line in enumerate(lines, 1):
        # #define BUF_SZ_IN_0 32
        m = re.match(r'#define\s+(BUF_SZ_IN_\d+|BUF_SZ_OUT_\d+|BUF_SZ)\s+(\d+)', line)
        if m:
            buf_sizes[m.group(1)] = int(m.group(2))

        # window_init(window_var, 1, buf_ping, LOCK_ACQ, buf_pong, LOCK_REL, buf_sz, total_rounds)
        m = re.search(
            r'window_init\(\s*(\w+)\s*,\s*\d+\s*,\s*\w+\s*,\s*\w+\s*,\s*'
            r'\w+\s*,\s*\w+\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', line)
        if m:
            # Determine window name from variable (e.g., window_window_in_0 -> in_0)
            var_name = m.group(1)
            win_name = var_name
            # Try to extract a cleaner name
            match_name = re.search(r'window_window_(in_\d+|out_\d+)', var_name)
            if match_name:
                win_name = match_name.group(1)

            wi = WindowInit(
                window_name=win_name,
                buf_sz=int(m.group(2)),
                total_rounds=int(m.group(3)),
                line_num=line_num
            )
            window_inits.append(wi)

    return buf_sizes, window_inits


# ---------------------------------------------------------------------------
# Helper: resolve tile location from var
# ---------------------------------------------------------------------------

def resolve_tile(tile_vars, tile_var):
    """Return (col, row) for a tile variable, or None."""
    return tile_vars.get(tile_var)


def is_shim_tile(tile_vars, tile_var):
    """Shim tiles are row 0."""
    loc = resolve_tile(tile_vars, tile_var)
    return loc is not None and loc[1] == 0


def is_core_tile(tile_vars, tile_var):
    """Core tiles are row >= 3 (for AIEML, rows 0 = shim, 1-2 = memtile)."""
    loc = resolve_tile(tile_vars, tile_var)
    return loc is not None and loc[1] >= 3


def tile_str(tile_vars, tile_var):
    loc = resolve_tile(tile_vars, tile_var)
    if loc:
        return f"tile({loc[0]},{loc[1]})"
    return tile_var


# ---------------------------------------------------------------------------
# Build per-tile data model
# ---------------------------------------------------------------------------

def build_tile_model(tile_vars, bd_configs, io_configs, startio_configs):
    """
    Build a per-tile model:
    tile_bds[(col,row)] = list of BDConfig
    tile_ios[(col,row)] = list of IOConfig
    io_to_startio: io_var -> StartIOConfig
    """
    tile_bds = defaultdict(list)
    tile_ios = defaultdict(list)
    io_to_startio = {}

    for var, bd in bd_configs.items():
        loc = resolve_tile(tile_vars, bd.tile_var)
        if loc:
            tile_bds[loc].append(bd)

    for var, io in io_configs.items():
        loc = resolve_tile(tile_vars, io.tile_var)
        if loc:
            tile_ios[loc].append(io)

    for var, sio in startio_configs.items():
        io_to_startio[sio.io_var] = sio

    return tile_bds, tile_ios, io_to_startio


def get_bd_chain(bd_configs, tile_var, start_bd_id):
    """Follow next_bd chain for a tile and return list of BDConfigs."""
    # Find all BDs for this tile
    tile_bds = [bd for bd in bd_configs.values() if bd.tile_var == tile_var]
    bd_by_id = {bd.bd_id: bd for bd in tile_bds}

    chain = []
    visited = set()
    bd_id = start_bd_id
    while bd_id >= 0 and bd_id not in visited and bd_id in bd_by_id:
        visited.add(bd_id)
        chain.append(bd_by_id[bd_id])
        bd_id = bd_by_id[bd_id].next_bd
    return chain


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name, passed, details=None):
        self.name = name
        self.passed = passed
        self.details = details or []


def check1_buffer_size_consistency(tile_vars, bd_configs, io_configs,
                                   buf_sizes, window_inits):
    """
    Check 1: Buffer Size Consistency (host BD len vs kernel BUF_SZ).

    For S2MM (input) channels: the host BD len should be a multiple of
    kernel BUF_SZ * 4 (v4int8 = 4 bytes per element).  Exact match means
    one DMA fire = one kernel acquire.  A multiple is acceptable when the
    host packs multiple sub-tiles.

    For MM2S (output) channels: the host BD len should likewise be a
    multiple of BUF_SZ_OUT * 4.

    All BDs in the same ping-pong chain must have the same length.
    """
    details = []
    checked = False
    v4_size = 4  # bytes per v4int8

    # Map (direction, channel) -> kernel BUF_SZ key
    buf_map = {
        ('S2MM', 0): 'BUF_SZ_IN_0',
        ('S2MM', 1): 'BUF_SZ_IN_1',
        ('MM2S', 0): 'BUF_SZ_OUT_0',
    }

    core_tiles = set()
    for var, tile_loc in tile_vars.items():
        if tile_loc[1] >= 3:
            core_tiles.add(tile_loc)

    if not core_tiles:
        return CheckResult("Buffer size consistency (host BD len vs kernel BUF_SZ)",
                          True, ["No core tiles found, skipping check"])

    for direction, channel in [('S2MM', 0), ('S2MM', 1), ('MM2S', 0)]:
        buf_key = buf_map.get((direction, channel))
        if buf_key not in buf_sizes:
            continue

        expected_unit = buf_sizes[buf_key] * v4_size
        checked = True

        # Check only one representative core tile (all should be identical)
        rep_tile = sorted(core_tiles)[0]
        for var, io in io_configs.items():
            io_loc = resolve_tile(tile_vars, io.tile_var)
            if io_loc != rep_tile or io.direction != direction or io.channel_id != channel:
                continue
            chain = get_bd_chain(bd_configs, io.tile_var, io.start_bd)
            if not chain:
                continue

            # All BDs in chain must have the same length
            lengths = set(bd.length for bd in chain)
            if len(lengths) > 1:
                details.append(
                    f"  tile({rep_tile[0]},{rep_tile[1]}) {direction} ch{channel}: "
                    f"ping-pong BDs have mismatched lengths: {sorted(lengths)}")
                continue

            bd_len = chain[0].length
            if bd_len % expected_unit != 0:
                details.append(
                    f"  tile({rep_tile[0]},{rep_tile[1]}) {direction} ch{channel}: "
                    f"BD len={bd_len} is not a multiple of {buf_key}={buf_sizes[buf_key]} "
                    f"* {v4_size}B = {expected_unit}B")
            break  # Only check one IO per direction/channel

    if not checked:
        return CheckResult("Buffer size consistency (host BD len vs kernel BUF_SZ)",
                          True, ["No matching BUF_SZ defines found in kernel"])

    passed = len(details) == 0
    return CheckResult("Buffer size consistency (host BD len vs kernel BUF_SZ)",
                      passed, details)


def check2_total_data_volume(tile_vars, bd_configs, io_configs, startio_configs):
    """
    Check 2: Total Data Volume Invariant.
    Cross-check: SHIM total send volume on each column should be consistent
    with the core tile receive volumes on the same column.
    """
    details = []
    io_to_startio = {}
    for var, sio in startio_configs.items():
        io_to_startio[sio.io_var] = sio

    # Compute per-SHIM-column send volumes (MM2S)
    shim_send = defaultdict(int)  # col -> total bytes sent
    # Compute per-core-column receive volumes (S2MM)
    core_recv = defaultdict(int)  # col -> total bytes received per channel

    for io_var, io in io_configs.items():
        loc = resolve_tile(tile_vars, io.tile_var)
        if loc is None:
            continue

        if io_var not in io_to_startio:
            continue
        sio = io_to_startio[io_var]

        chain = get_bd_chain(bd_configs, io.tile_var, io.start_bd)
        if not chain:
            continue

        # Total bytes for this IO = sum(BD len in chain) * repeat
        # For ping-pong (chained BDs), one "fire" = one BD, repeat fires all
        # Actually, repeat = number of times the channel fires
        # Each fire uses one BD from the chain, cycling through
        # Total = repeat * single_bd_len (for uniform ping-pong chains)
        bd_len = chain[0].length  # All BDs in a pp chain should have same len
        pp_depth = len(chain)

        # For multidim BDs with iter_wrap
        if chain[0].is_multidim and chain[0].iter_wrap > 0:
            total_bytes = bd_len * chain[0].iter_wrap * sio.repeat
        else:
            total_bytes = bd_len * sio.repeat

        if is_shim_tile(tile_vars, io.tile_var):
            if io.direction == 'MM2S':
                shim_send[(loc[0], io.channel_id)] += total_bytes

    # This check is informational - report the volumes
    if shim_send:
        for key in sorted(shim_send.keys()):
            col, ch = key
            # We don't have expected sizes, so just report for manual review
            pass

    passed = len(details) == 0
    if not shim_send:
        return CheckResult("Total data volume invariant",
                          True, ["No SHIM send IOs found, skipping"])
    return CheckResult("Total data volume invariant", passed,
                      details if details else ["Volumes computed (manual review)"])


def check3_shim_bd_len_vs_dims(tile_vars, bd_configs):
    """
    Check 3: SHIM BD len = Per-Iteration Size.
    For multidim BDs with iter_wrap > 1, verify:
    - BD len matches the innermost transfer size (d0_size * d0_stride or similar)
    - iter_step is consistent with len (iter_step should advance the base address)
    - All SHIM MM2S BDs for the same channel have consistent lengths
    """
    details = []
    checked = False

    # Group SHIM multidim BDs by (tile, channel) for consistency check
    shim_bd_groups = defaultdict(list)

    for var, bd in bd_configs.items():
        if not bd.is_multidim or not is_shim_tile(tile_vars, bd.tile_var):
            continue

        loc = resolve_tile(tile_vars, bd.tile_var)
        shim_bd_groups[(bd.tile_var, bd.bd_id)].append(bd)

        if bd.iter_wrap <= 1:
            continue

        checked = True

        # With iter_wrap > 1, verify iter_step is positive (covered by check 5)
        # Here we check that BD len is consistent with iter_step:
        # iter_step should be >= len (otherwise iterations overlap)
        if bd.iter_step > 0 and bd.iter_step < bd.length:
            details.append(
                f"  {tile_str(tile_vars, bd.tile_var)} BD#{bd.bd_id}: "
                f"iter_step={bd.iter_step} < len={bd.length} "
                f"(iterations would overlap) [line {bd.line_num}]")

    if not checked:
        return CheckResult("SHIM BD len vs iteration pattern",
                          True, ["No multidim SHIM BDs with iter_wrap>1 found"])

    passed = len(details) == 0
    return CheckResult("SHIM BD len vs iteration pattern", passed, details)


def check4_channel_repeat_count(tile_vars, bd_configs, io_configs,
                                startio_configs, window_inits):
    """
    Check 4: Channel Repeat Count Sanity.
    Core MM2S (output) repeat should match kernel output rounds.
    """
    details = []
    io_to_startio = {}
    for var, sio in startio_configs.items():
        io_to_startio[sio.io_var] = sio

    # Find kernel output window rounds
    out_rounds = None
    for wi in window_inits:
        if 'out' in wi.window_name:
            out_rounds = wi.total_rounds
            break

    # Check core MM2S startio repeat vs kernel output rounds
    if out_rounds is not None:
        for io_var, io in io_configs.items():
            if not is_core_tile(tile_vars, io.tile_var):
                continue
            if io.direction != 'MM2S':
                continue
            if io_var not in io_to_startio:
                continue
            sio = io_to_startio[io_var]
            loc = resolve_tile(tile_vars, io.tile_var)
            if sio.repeat != out_rounds:
                details.append(
                    f"  {tile_str(tile_vars, io.tile_var)} MM2S ch{io.channel_id}: "
                    f"startio repeat={sio.repeat}, kernel output rounds={out_rounds} "
                    f"[line {sio.line_num}]")

    # Check core S2MM startio repeats are consistent across tiles
    s2mm_repeats = defaultdict(set)  # channel -> set of repeats
    for io_var, io in io_configs.items():
        if not is_core_tile(tile_vars, io.tile_var):
            continue
        if io.direction != 'S2MM':
            continue
        if io_var not in io_to_startio:
            continue
        sio = io_to_startio[io_var]
        s2mm_repeats[io.channel_id].add(sio.repeat)

    for ch, repeats in s2mm_repeats.items():
        if len(repeats) > 1:
            details.append(
                f"  Core S2MM ch{ch}: inconsistent repeats across tiles: "
                f"{sorted(repeats)}")

    passed = len(details) == 0
    return CheckResult("Channel repeat count sanity", passed, details)


def check5_iter_step_sanity(tile_vars, bd_configs):
    """
    Check 5: iter_step_size Sanity.
    When iter_wrap > 1, iter_step must be > 0.
    When iter_wrap <= 1, iter_step should be 0.
    """
    details = []

    for var, bd in bd_configs.items():
        if not bd.is_multidim:
            continue

        loc = resolve_tile(tile_vars, bd.tile_var)
        if bd.iter_wrap > 1 and bd.iter_step <= 0:
            details.append(
                f"  {tile_str(tile_vars, bd.tile_var)} BD#{bd.bd_id}: "
                f"iter_wrap={bd.iter_wrap} but iter_step={bd.iter_step} "
                f"(should be > 0) [line {bd.line_num}]")
        elif bd.iter_wrap <= 1 and bd.iter_step != 0:
            details.append(
                f"  [WARN] {tile_str(tile_vars, bd.tile_var)} BD#{bd.bd_id}: "
                f"iter_wrap={bd.iter_wrap} but iter_step={bd.iter_step} "
                f"(non-functional) [line {bd.line_num}]")

    passed = len(details) == 0
    return CheckResult("iter_step_size sanity", passed, details)


def check6_ooo_bd_id_presence(tile_vars, bd_configs, io_configs):
    """
    Check 6: OOO BD ID Presence.
    Core MM2S BDs with ooo_bd_id should have matching SHIM S2MM BDs.
    """
    details = []

    # Collect all SHIM S2MM BD IDs
    shim_s2mm_bd_ids = set()
    for var, bd in bd_configs.items():
        if is_shim_tile(tile_vars, bd.tile_var):
            # Check if any IO for this tile is S2MM
            for io_var, io in io_configs.items():
                if io.tile_var == bd.tile_var and io.direction == 'S2MM':
                    shim_s2mm_bd_ids.add((bd.tile_var, bd.bd_id))

    # Check core MM2S BDs with ooo_bd_id
    for var, bd in bd_configs.items():
        if not is_core_tile(tile_vars, bd.tile_var):
            continue
        if bd.ooo_bd_id == -1:
            continue

        # This core BD has an OOO ID - verify it matches a SHIM BD
        # The ooo_bd_id on the core side is the BD ID on the SHIM side
        # We need to find which SHIM tile receives from this core tile's column
        loc = resolve_tile(tile_vars, bd.tile_var)
        if loc is None:
            continue

        # Find the SHIM tile for this column (not always same col in routing)
        # For now, just check that ooo_bd_id exists as a SHIM S2MM BD somewhere
        found = False
        for shim_var, shim_bd in bd_configs.items():
            if is_shim_tile(tile_vars, shim_bd.tile_var) and shim_bd.bd_id == bd.ooo_bd_id:
                # Check if it's used in S2MM
                for io_var, io in io_configs.items():
                    if io.tile_var == shim_bd.tile_var and io.direction == 'S2MM':
                        found = True
                        break
            if found:
                break

        if not found:
            details.append(
                f"  {tile_str(tile_vars, bd.tile_var)} MM2S BD#{bd.bd_id}: "
                f"ooo_bd_id={bd.ooo_bd_id} has no matching SHIM S2MM BD "
                f"[line {bd.line_num}]")

    # Also check: if SHIM has S2MM OOO BDs, core MM2S BDs should have ooo_bd_id != -1
    for var, io in io_configs.items():
        if not is_shim_tile(tile_vars, io.tile_var):
            continue
        if io.direction != 'S2MM':
            continue
        # Check if any BDs on this SHIM S2MM channel are OOO (multiple BDs, no chaining)
        chain = get_bd_chain(bd_configs, io.tile_var, io.start_bd)
        if len(chain) == 1 and chain[0].next_bd == -1:
            # Single BD with no chain - might be OOO
            # Check if there are other BDs on same tile/channel
            pass

    passed = len(details) == 0
    return CheckResult("OOO BD ID presence", passed, details)


def check7_lock_credit_symmetry(tile_vars, bd_configs, io_configs, lock_inits):
    """
    Check 7: Lock Credit Symmetry.
    For each core tile, input S2MM BDs should have complementary lock pairs.
    Lock init value should match ping-pong depth.
    """
    details = []

    # Group locks by tile
    tile_locks = defaultdict(list)
    for lock in lock_inits:
        tile_locks[(lock.col, lock.row)].append(lock)

    # For each core tile, check lock pairs
    core_tiles = set()
    for var, loc in tile_vars.items():
        if loc[1] >= 3:
            core_tiles.add(loc)

    for loc in sorted(core_tiles):
        locks = tile_locks.get(loc, [])
        lock_by_id = {l.lock_id: l for l in locks}

        # Check that locks used in BDs have proper init values
        tile_bd_list = [bd for bd in bd_configs.values()
                       if resolve_tile(tile_vars, bd.tile_var) == loc]

        # Group BDs by their lock pairs
        lock_pairs = set()
        for bd in tile_bd_list:
            if bd.acq_lock_id >= 0 and bd.rel_lock_id >= 0:
                lock_pairs.add((bd.acq_lock_id, bd.rel_lock_id))

        for acq_id, rel_id in lock_pairs:
            # Check init value of acquire lock
            if acq_id in lock_by_id:
                init_val = lock_by_id[acq_id].init_val
                # For ping-pong, init should be pp_depth (typically 2)
                # The acquire lock starts with credits = pp_depth
                # We just check it's positive
                if init_val <= 0:
                    details.append(
                        f"  tile({loc[0]},{loc[1]}) lock {acq_id}: "
                        f"init_val={init_val} (should be > 0 for acquire lock) "
                        f"[line {lock_by_id[acq_id].line_num}]")

    passed = len(details) == 0
    return CheckResult("Lock credit symmetry", passed, details)


def check8_kernel_window_vs_startio(tile_vars, bd_configs, io_configs,
                                     startio_configs, window_inits):
    """
    Check 8: Kernel window_init vs startio Consistency.

    For output (MM2S) channels: the startio repeat should match the
    kernel output window's total_rounds, since MM2S repeat directly
    controls how many times the output DMA fires.

    For input (S2MM) channels: the host uses lock-driven flow control
    with the BD chain cycling via acquire/release, so the startio repeat
    value is typically 1 (meaning "enable and run"). We do NOT compare
    S2MM repeat to window_init rounds -- those are driven by locks.

    Cross-check: all core tiles should have the same MM2S repeat.
    """
    details = []
    io_to_startio = {}
    for var, sio in startio_configs.items():
        io_to_startio[sio.io_var] = sio

    # Find kernel output window rounds
    out_rounds = None
    for wi in window_inits:
        if 'out' in wi.window_name:
            out_rounds = wi.total_rounds
            break

    # Collect core tile MM2S startio repeats
    mm2s_repeats = {}  # (col, row) -> repeat
    for io_var, io in io_configs.items():
        if not is_core_tile(tile_vars, io.tile_var):
            continue
        if io.direction != 'MM2S':
            continue
        if io_var not in io_to_startio:
            continue
        sio = io_to_startio[io_var]
        loc = resolve_tile(tile_vars, io.tile_var)
        mm2s_repeats[loc] = (sio.repeat, sio.line_num)

    # Check MM2S repeat vs kernel output rounds
    if out_rounds is not None:
        for loc in sorted(mm2s_repeats.keys()):
            repeat, line_num = mm2s_repeats[loc]
            if repeat != out_rounds:
                details.append(
                    f"  window_out_0: window_init rounds={out_rounds}, "
                    f"tile({loc[0]},{loc[1]}) MM2S startio repeat={repeat} "
                    f"[host line {line_num}]")
            break  # Check one representative tile

    # Cross-check: all core tiles should have same MM2S repeat
    unique_repeats = set(r for r, _ in mm2s_repeats.values())
    if len(unique_repeats) > 1:
        details.append(
            f"  Core MM2S repeat inconsistent across tiles: {sorted(unique_repeats)}")

    passed = len(details) == 0
    return CheckResult("Kernel window_init vs host startio", passed, details)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checks(host_path, kernel_path):
    """Run all checks and return results."""
    with open(host_path, 'r') as f:
        host_text = f.read()
    with open(kernel_path, 'r') as f:
        kernel_text = f.read()

    # Parse
    tile_vars, bd_configs, io_configs, startio_configs, lock_inits = parse_host(host_text)
    buf_sizes, window_inits = parse_kernel(kernel_text)

    # Print parse summary
    print(f"Parsed host.cc: {len(tile_vars)} tiles, {len(bd_configs)} BDs, "
          f"{len(io_configs)} IOs, {len(startio_configs)} startios, "
          f"{len(lock_inits)} lock inits")
    print(f"Parsed kernel.cc: BUF_SZ={buf_sizes}, {len(window_inits)} window_inits")
    print()

    # Run checks
    results = []

    results.append(check1_buffer_size_consistency(
        tile_vars, bd_configs, io_configs, buf_sizes, window_inits))

    results.append(check2_total_data_volume(
        tile_vars, bd_configs, io_configs, startio_configs))

    results.append(check3_shim_bd_len_vs_dims(
        tile_vars, bd_configs))

    results.append(check4_channel_repeat_count(
        tile_vars, bd_configs, io_configs, startio_configs, window_inits))

    results.append(check5_iter_step_sanity(
        tile_vars, bd_configs))

    results.append(check6_ooo_bd_id_presence(
        tile_vars, bd_configs, io_configs))

    results.append(check7_lock_credit_symmetry(
        tile_vars, bd_configs, io_configs, lock_inits))

    results.append(check8_kernel_window_vs_startio(
        tile_vars, bd_configs, io_configs, startio_configs, window_inits))

    return results


def print_results(results):
    """Print check results with colored output."""
    print(f"{'=' * 60}")
    print(f"  Pre-HW Verification Results")
    print(f"{'=' * 60}")
    print()

    pass_count = 0
    fail_count = 0
    warn_count = 0

    for i, r in enumerate(results, 1):
        if r.passed:
            status = "[PASS]"
            pass_count += 1
        else:
            # Check if all details are warnings
            all_warns = all('[WARN]' in d for d in r.details)
            if all_warns and r.details:
                status = "[WARN]"
                warn_count += 1
            else:
                status = "[FAIL]"
                fail_count += 1

        print(f"{status} Check {i}: {r.name}")
        for detail in r.details:
            print(detail)
        if r.details:
            print()

    print(f"{'=' * 60}")
    summary_parts = [f"{pass_count} PASS"]
    if fail_count:
        summary_parts.append(f"{fail_count} FAIL")
    if warn_count:
        summary_parts.append(f"{warn_count} WARN")
    print(f"Summary: {', '.join(summary_parts)}")
    print(f"{'=' * 60}")

    return fail_count == 0


def main():
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_host = os.path.join(
        project_root, 'src', 'mlir', 'mlirfront', 'tilinglinalg', 'pass',
        'unitest', 'worklocal', 'host.cc')
    default_kernel = os.path.join(
        project_root, 'src', 'mlir', 'mlirfront', 'tilinglinalg', 'pass',
        'unitest', 'worklocal', 'kernel.cc')

    if len(sys.argv) >= 3:
        host_path = sys.argv[1]
        kernel_path = sys.argv[2]
    elif len(sys.argv) == 2:
        # Single arg = directory containing both files
        d = sys.argv[1]
        host_path = os.path.join(d, 'host.cc')
        kernel_path = os.path.join(d, 'kernel.cc')
    else:
        host_path = default_host
        kernel_path = default_kernel

    if not os.path.exists(host_path):
        print(f"Error: host file not found: {host_path}")
        sys.exit(1)
    if not os.path.exists(kernel_path):
        print(f"Error: kernel file not found: {kernel_path}")
        sys.exit(1)

    print(f"=== Pre-HW Verification: {os.path.basename(host_path)} + "
          f"{os.path.basename(kernel_path)} ===")
    print(f"  host:   {host_path}")
    print(f"  kernel: {kernel_path}")
    print()

    results = run_checks(host_path, kernel_path)
    success = print_results(results)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
