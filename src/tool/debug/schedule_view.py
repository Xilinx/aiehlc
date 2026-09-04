#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""
schedule_view.py - Readable spatial schema + HTML viewer for the machine-generated
`host_canonicalized` function in <workdir>/host.cc.

It consumes:
  - <workdir>/dfscheduleprovenancemap.json  (high-level, already grouped by tile/data)
  - <workdir>/host.cc                        (low-level, final emitted C++)

and produces:
  - <workdir>/schedule_view.json             (readable schema: grid + per-tile hi/lo)
  - <workdir>/host_schedule.html             (self-contained interactive tile grid)

host.cc is never modified.

Usage:
    python3 src/tool/debug/schedule_view.py [workdir]
    (workdir defaults to aout/worklocal)
"""

import glob
import json
import os
import re
import sys


# ----------------------------------------------------------------------------
# host.cc parsing -> attribute each line to a spatial owner: (col,row) or "__global__"
# ----------------------------------------------------------------------------

RE_TILELOC_DECL = re.compile(
    r'XAie_LocType\s+(v\d+)\s*=\s*XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)')
RE_BD_CONFIG = re.compile(
    r'__Runtime_dma_bd_config\w*\(\s*v\d+\s*,\s*(v\d+)\s*,')
RE_CREATEIO = re.compile(
    r'__Runtime_dma_createio_\d+\(\s*(v\d+)\s*,')
RE_ENABLE_OOO = re.compile(
    r'__Runtime_dma_channel_enable_ooo\(\s*v\d+\s*,\s*(v\d+)\s*,')
RE_LOCKSET = re.compile(
    r'XAie_LockSetValue\(\s*\w+\s*,\s*XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)')
RE_COMMENT_TILE = re.compile(
    r'tile[ =]\(\s*(\d+)\s*,\s*(\d+)\s*\)')
RE_LAUNCH = re.compile(
    r'(v\d+)\s*=\s*__Runtime_launch_kernel_group')

# for-loop context: header detection + provenance-comment boundary for preamble.
RE_FOR = re.compile(r'\bfor\s*\(')
RE_PROV_COMMENT = re.compile(
    r'/\*\s*(Allocated BD|DMA BD Config|Create IO|Enable out-of-order)')


def find_loops(lines, fstart, fend):
    """Flat/nested for-loops in [fstart,fend]. Returns list of dicts:
    {header, end, preamble} (0-based line indices). preamble = the loop's
    index/offset math: statement lines after the `for(` header up to the first
    provenance comment (/* Allocated BD | DMA BD Config | ... */)."""
    loops, stack, depth = [], [], 0
    for i in range(fstart, fend + 1):
        ln = lines[i]
        if RE_FOR.search(ln):
            stack.append([i, depth])          # body { opens on this line
        depth += ln.count('{') - ln.count('}')
        while stack and depth <= stack[-1][1]:
            hdr, _ = stack.pop()
            loops.append({'header': hdr, 'end': i})
    for lp in loops:                          # compute preamble (index math)
        pre = []
        for j in range(lp['header'] + 1, lp['end']):
            s = lines[j]
            if RE_PROV_COMMENT.search(s):
                break
            if _is_comment_or_blank(s):
                continue
            pre.append(j)
        lp['preamble'] = pre
    return loops


# Simple SSA constant assignments: `<type> vN = <literal>;` (RHS is a bare int).
RE_CONST_ASSIGN = re.compile(r'\b(v\d+)\s*=\s*(-?\d+)\s*;')
RE_VAR = re.compile(r'\bv\d+\b')
# LHS assignment to vN (`<type> vN = ...` or `vN = ...`). Negative lookahead on
# `=` excludes the `==` comparison operator; `<=`/`>=`/`!=` never match because
# the leading operator char sits between vN and `=`.
RE_VAR_DEF = re.compile(r'\b(v\d+)\s*=(?!=)')


def build_const_map(lines, fstart, fend):
    """Map SSA var -> literal string for `vN = <int>;` assignments (SSA: once)."""
    cmap = {}
    for i in range(fstart, fend + 1):
        m = RE_CONST_ASSIGN.search(lines[i])
        if m:
            cmap.setdefault(m.group(1), m.group(2))
    return cmap


def build_var_def_map(lines, fstart, fend):
    """Map SSA var -> 1-based line where it is first assigned/declared.

    Function parameters (the `vN` on the signature line at `fstart`) have no `=`
    assignment, so they are seeded first, all pointing at the signature line;
    this lets the full-block trace-back surface the whole function signature
    (every parameter) when a param like `v1`/`v2` is referenced. SSA guarantees a
    single def per var, so the first match wins (setdefault)."""
    var_def = {}
    for m in RE_VAR.finditer(lines[fstart]):        # function parameters
        var_def.setdefault(m.group(0), fstart + 1)
    for i in range(fstart, fend + 1):
        for m in RE_VAR_DEF.finditer(lines[i]):
            var_def.setdefault(m.group(1), i + 1)
    return var_def


def channel_full_block(host_lines, entries, cmap, var_def, var2loc):
    """Contiguous host.cc block for ONE channel, with arg-def trace-back.

    `entries` are the channel-scoped mapping records (already include the
    enclosing loop header/index/close context). We union those lines with the
    defining line of every SSA `vN` referenced on them, following the def chain
    TRANSITIVELY (a pulled-in def line like `v174 = buffer_arg(v44)` drags in
    v44's def too, and so on) so the whole argument provenance is visible. SSA
    (one def per var) guarantees the closure terminates. Returns the same shape
    as the tile full block (`ranges/line_start/line_end/code_lines`) plus a
    `params` list annotating each referenced var's resolved value.
    """
    base = {e['line'] for e in entries}          # 1-based, channel scoped

    def resolve(v):
        val = cmap.get(v)
        if val is None and v in var2loc:
            c, r = var2loc[v]
            val = 'TileLoc(%d,%d)' % (c, r)
        return val

    # Transitive closure over the def chain. Seed the worklist with every var
    # referenced on a base line, then for each var pull its def line into the
    # block and enqueue the vars referenced on THAT def line. `varset` collects
    # all vars seen (for param annotation); SSA bounds the loop to finite vars.
    union = set(base)
    varset = set()
    work = []
    for n in base:
        work.extend(m.group(0) for m in RE_VAR.finditer(host_lines[n - 1]))
    while work:
        v = work.pop()
        if v in varset:
            continue
        varset.add(v)
        dl = var_def.get(v)
        if dl is None or dl in union:
            continue
        union.add(dl)                            # pull arg-def line into block
        work.extend(m.group(0)                   # trace vars on the def line
                    for m in RE_VAR.finditer(host_lines[dl - 1]))

    ordered = sorted(union)
    ranges = contiguous_ranges(ordered)
    code_lines = [{'line': n, 'code': host_lines[n - 1]} for n in ordered]

    params = []
    for v in sorted(varset):
        val = resolve(v)
        dl = var_def.get(v)
        if val is None and dl is None:
            continue                             # nothing meaningful to annotate
        params.append({'var': v, 'value': val, 'def_line': dl})

    return {
        'ranges': ranges,
        'line_start': ranges[0][0] if ranges else None,
        'line_end': ranges[-1][1] if ranges else None,
        'code_lines': code_lines,
        'params': params,
    }


def loop_note(header_text, cmap):
    """`v7=0, v11=7168, v5=1` for every const-valued var referenced in the header.
    The induction var (assigned in the header) is not a const, so it's skipped."""
    seen, parts = [], []
    for m in RE_VAR.finditer(header_text):
        v = m.group(0)
        if v in cmap and v not in seen:
            seen.append(v)
            parts.append('%s=%s' % (v, cmap[v]))
    return ', '.join(parts)

# ----------------------------------------------------------------------------
# Rename-robust anchors for the filtered per-channel mapping.
#
# These key on the provenance COMMENTS emitted by passdfscheduletoapi.cpp (from
# IR attributes, independent of the C API name) plus the stable runtime RETURN
# TYPES (io / ioevent / XAie_DmaDesc, typedef'd in aie_runtime.h).  They do NOT
# depend on the __Runtime_* wrapper names, so renaming any wrapper changes nothing
# the mapper reads.  Join key everywhere is (col,row,direction,channel).
# ----------------------------------------------------------------------------

# /* Create IO: channel_id=C, bd_id=B, tile=(col,row), direction=DIR */
RE_C_IO = re.compile(
    r'/\*\s*Create IO:\s*channel_id=(\d+),\s*bd_id=(-?\d+),\s*'
    r'tile=\((\d+),\s*(\d+)\),\s*direction=(\w+)')
# /* DMA BD Config: bd_id=B, len=L, ..., acquire_lock_id=A, ..., release_lock_id=R, ... */
RE_C_BD = re.compile(
    r'/\*\s*DMA BD Config:\s*bd_id=(-?\d+),\s*len=(\d+).*?'
    r'acquire_lock_id=(-?\d+),\s*acquire_lock_val=(-?\d+),\s*'
    r'release_lock_id=(-?\d+),\s*release_lock_val=(-?\d+)')
# /* Enable out-of-order BD on tile(col,row) ch=C dir=DIR */
RE_C_OOO = re.compile(
    r'/\*\s*Enable out-of-order BD on tile\((\d+),\s*(\d+)\)\s*ch=(\d+)\s*dir=(\w+)')
# __Runtime_dma_bd_config(dev, tile, BUFVAR, BD_ID, ...) -> buffer var + bd_id
RE_C_BDCALL = re.compile(
    r'__Runtime_dma_bd_config\(\s*\w+\s*,\s*\w+\s*,\s*(\w+)\s*,\s*(-?\d+)')
# void* vN = __runtime_buffer_arg((void*)ADDR);  -> literal DMA-view address
RE_C_BUFARG = re.compile(r'__runtime_buffer_arg\(\s*\(void\s*\*\)\s*(\d+)\s*\)')

# Stable runtime return types -> capture the LHS SSA var of a call statement.
RE_RET_BD = re.compile(r'\bXAie_DmaDesc\s+(v\d+)\s*=')
RE_RET_IO = re.compile(r'\bio\s+(v\d+)\s*=')       # NOT 'ioevent' (no ws after 'io')
RE_RET_SIO = re.compile(r'\bioevent\s+(v\d+)\s*=')
# Per-channel wait on a startio event: __Runtime_wait(device, <event_var>). The
# event var is the LHS of the matching `ioevent vN = __Runtime_startio(...)`.
RE_WAIT_EVENT = re.compile(r'__Runtime_wait\w*\(\s*\w+\s*,\s*(v\d+)\s*\)')
# LockSetValue is an XAie_ driver call (stable, not a __Runtime_ wrapper).
RE_LOCK_FULL = re.compile(
    r'XAie_LockSetValue\(\s*\w+\s*,\s*XAie_TileLoc\((\d+),\s*(\d+)\)\s*,\s*XAie_LockInit\((-?\d+)')

GLOBAL_MARKERS = (
    'load_kernel_group',
    'launch_kernel_group',
    'AieRt_DebugSnapshot',
    '_dbg_',
)


def find_function_range(lines, fname='host_canonicalized'):
    """Return (start_idx, end_idx) 0-based inclusive line indices of the function body."""
    start = None
    for i, ln in enumerate(lines):
        if fname in ln and '(' in ln and ln.rstrip().endswith('{'):
            start = i
            break
    if start is None:
        raise RuntimeError('could not find function %s in host.cc' % fname)
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0 and i > start:
            return start, i
    return start, len(lines) - 1


def attribute_lines(lines, fstart, fend):
    """Return owner[] (same length as lines) for the function body.

    owner[i] is a (col,row) tuple, the string '__global__', or None (outside fn).
    """
    n = len(lines)
    owner = [None] * n
    var2loc = {}
    launch_var = None

    # First pass: collect var->loc and explicit per-line anchors.
    for i in range(fstart, fend + 1):
        ln = lines[i]
        m = RE_TILELOC_DECL.search(ln)
        if m:
            var2loc[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        m = RE_LAUNCH.search(ln)
        if m:
            launch_var = m.group(1)

    for i in range(fstart, fend + 1):
        ln = lines[i]

        # Global markers first (kernel group load/launch, debug snapshot arrays).
        if any(mk in ln for mk in GLOBAL_MARKERS):
            owner[i] = '__global__'
            continue
        # The single outer wait on the launch event.
        if launch_var and '__Runtime_wait' in ln and re.search(
                r'\b%s\b' % re.escape(launch_var), ln):
            owner[i] = '__global__'
            continue

        # Explicit tile-typed function arguments (highest fidelity).
        m = RE_BD_CONFIG.search(ln) or RE_CREATEIO.search(ln) or RE_ENABLE_OOO.search(ln)
        if m and m.group(1) in var2loc:
            owner[i] = var2loc[m.group(1)]
            continue
        m = RE_LOCKSET.search(ln)
        if m:
            owner[i] = (int(m.group(1)), int(m.group(2)))
            continue
        m = RE_TILELOC_DECL.search(ln)
        if m:
            owner[i] = var2loc[m.group(1)]
            continue
        # Comment mentioning a tile (fallback anchor).
        m = RE_COMMENT_TILE.search(ln)
        if m:
            owner[i] = (int(m.group(1)), int(m.group(2)))
            continue

    # Backward-fill: an unanchored helper line belongs to the nearest anchored
    # line *below* it (helpers precede the call they feed).
    current = None
    for i in range(fend, fstart - 1, -1):
        if owner[i] is not None:
            current = owner[i]
        else:
            owner[i] = current if current is not None else '__global__'

    return owner, var2loc, launch_var


def contiguous_ranges(line_nums):
    """Given sorted 1-based line numbers, return list of [start,end] inclusive ranges."""
    ranges = []
    for ln in line_nums:
        if ranges and ln == ranges[-1][1] + 1:
            ranges[-1][1] = ln
        else:
            ranges.append([ln, ln])
    return ranges


# ----------------------------------------------------------------------------
# Rename-robust, filtered line -> schedule mapping.
#
# For each schedule element (bd_config / createio / startio / enable_ooo / lock)
# we discover its directly-implementing host.cc line via provenance COMMENTS and
# stable return TYPES, never via the renameable __Runtime_* wrapper names.
# Result: channel_map[(col,row,dir,channel)] = [ {line,kind,bd_id,comment_line,code}, ... ]
# ----------------------------------------------------------------------------

def _is_comment_or_blank(s):
    t = s.strip()
    return (not t) or t.startswith('/*') or t.startswith('//')


def _next_type_line(lines, i, end, rx):
    """First index in [i,end] whose text matches a return-type regex (call stmt)."""
    for j in range(i, end + 1):
        if rx.search(lines[j]):
            return j
    return None


def _next_stmt_line(lines, i, end):
    """First index in [i,end] that is not a comment/blank (the actual statement)."""
    for j in range(i, end + 1):
        if not _is_comment_or_blank(lines[j]):
            return j
    return None


def map_relevant_lines(host_lines, fstart, fend, prov, loops=None):
    """Map each schedule element to only its directly-implementing host.cc line(s).

    Returns channel_map[(col,row,dir,channel)] -> list of entries
        {line, kind, bd_id, comment_line, code}
    where kind in {createio, bd_config, enable_ooo, startio, wait, lock} and `line` is
    a 1-based host.cc line number.
    """
    # Precompute, per channel key, the bd-id set and lock-id set for exact join.
    chan_bd = {}                 # key -> set(bd_id)
    tile_lock_keys = {}          # (col,row) -> {lock_id: set(key)}
    for t in prov.get('tiles', []):
        col, row = t['col'], t['row']
        for c in t.get('dma_channels', []):
            key = (col, row, c['direction'], c['channel'])
            bds = set()
            locks = set()
            for bd in c.get('bd_chain', []):
                bds.add(bd.get('bd_id'))
                for lk in (bd.get('acquire_lock', []) + bd.get('release_lock', [])):
                    lid = lk.get('id')
                    if lid is not None and lid >= 0:
                        locks.add(lid)
            chan_bd[key] = bds
            for lid in locks:
                tile_lock_keys.setdefault((col, row), {}).setdefault(lid, set()).add(key)

    channel_map = {}

    def emit(key, entry):
        channel_map.setdefault(key, []).append(entry)

    pending_bd = []      # [{comment_line, bd_id, code_line}] since last createio
    io_var2key = {}      # io SSA var -> channel key (for name-free startio linking)

    for i in range(fstart, fend + 1):
        ln = host_lines[i]

        m = RE_C_BD.search(ln)
        if m:
            bd_id = int(m.group(1))
            j = _next_type_line(host_lines, i + 1, fend, RE_RET_BD)
            pending_bd.append({
                'comment_line': i + 1,
                'bd_id': bd_id,
                'code_line': (j + 1) if j is not None else (i + 1),
            })
            continue

        m = RE_C_OOO.search(ln)
        if m:
            # The OOO comment carries the full channel key, so emit directly
            # (its call line is the next statement, the void enable_ooo call).
            col, row = int(m.group(1)), int(m.group(2))
            ch, dr = int(m.group(3)), m.group(4)
            key = (col, row, dr, ch)
            j = _next_stmt_line(host_lines, i + 1, fend)
            code_line = (j + 1) if j is not None else (i + 1)
            emit(key, {
                'line': code_line, 'kind': 'enable_ooo', 'bd_id': None,
                'comment_line': i + 1, 'code': host_lines[code_line - 1].strip(),
            })
            continue

        m = RE_C_IO.search(ln)
        if m:
            ch, bd_id = int(m.group(1)), int(m.group(2))
            col, row, dr = int(m.group(3)), int(m.group(4)), m.group(5)
            key = (col, row, dr, ch)
            # createio call = next stmt whose LHS type is `io` (skips OOO call).
            j = _next_type_line(host_lines, i + 1, fend, RE_RET_IO)
            code_line = (j + 1) if j is not None else (i + 1)
            if j is not None:
                vm = RE_RET_IO.search(host_lines[j])
                if vm:
                    io_var2key[vm.group(1)] = key
            emit(key, {
                'line': code_line, 'kind': 'createio', 'bd_id': bd_id,
                'comment_line': i + 1,
                'code': host_lines[code_line - 1].strip(),
            })
            # Flush every pending BD whose bd_id belongs to this channel.
            keep = []
            for pb in pending_bd:
                if pb['bd_id'] in chan_bd.get(key, set()):
                    emit(key, {
                        'line': pb['code_line'], 'kind': 'bd_config',
                        'bd_id': pb['bd_id'], 'comment_line': pb['comment_line'],
                        'code': host_lines[pb['code_line'] - 1].strip(),
                        'bd_comment': host_lines[pb['comment_line'] - 1].strip(),
                    })
                else:
                    keep.append(pb)
            pending_bd = keep
            continue

    # startio: any `ioevent v = ...` statement referencing a known io var.
    # Also record event_var -> key so the matching __Runtime_wait can be linked.
    event_var2key = {}   # ioevent SSA var -> channel key
    for i in range(fstart, fend + 1):
        ms = RE_RET_SIO.search(host_lines[i])
        if not ms:
            continue
        for var, key in io_var2key.items():
            if re.search(r'\b%s\b' % re.escape(var), host_lines[i]):
                emit(key, {
                    'line': i + 1, 'kind': 'startio', 'bd_id': None,
                    'comment_line': None, 'code': host_lines[i].strip(),
                })
                event_var2key[ms.group(1)] = key
                break

    # wait: __Runtime_wait(device, <event_var>) where event_var is a per-channel
    # startio result. The single outer wait on the launch event is not a startio
    # event (not in event_var2key), so it is correctly excluded here.
    for i in range(fstart, fend + 1):
        m = RE_WAIT_EVENT.search(host_lines[i])
        if not m:
            continue
        key = event_var2key.get(m.group(1))
        if key is None:
            continue
        emit(key, {
            'line': i + 1, 'kind': 'wait', 'bd_id': None,
            'comment_line': None, 'code': host_lines[i].strip(),
        })

    # locks: XAie_LockSetValue(...) -> channel on same tile whose lock set has id.
    for i in range(fstart, fend + 1):
        m = RE_LOCK_FULL.search(host_lines[i])
        if not m:
            continue
        col, row, lid = int(m.group(1)), int(m.group(2)), int(m.group(3))
        keys = tile_lock_keys.get((col, row), {}).get(lid, set())
        if not keys:
            continue
        if len(keys) == 1:
            chosen = next(iter(keys))
        else:
            # Disambiguate by nearest preceding createio line.
            chosen, best = None, -1
            for key in keys:
                for e in channel_map.get(key, []):
                    if e['kind'] == 'createio' and e['line'] <= i + 1 and e['line'] > best:
                        best, chosen = e['line'], key
            if chosen is None:
                chosen = sorted(keys)[0]
        emit(chosen, {
            'line': i + 1, 'kind': 'lock', 'bd_id': None,
            'comment_line': None, 'code': host_lines[i].strip(),
        })

    # Attach enclosing-loop context (for-header + index math + closing brace) to
    # every channel whose captured ops fall inside a loop body. Emits new kinds
    # loop / loopidx / loopend; the header + idx sort ahead of the ops (lowest
    # line numbers) and the closing brace sorts after them (highest line number).
    # The loop header also carries a `note` resolving its SSA const parameters
    # (e.g. v7=0, v11=7168, v5=1).
    cmap = build_const_map(host_lines, fstart, fend)
    for key, entries in list(channel_map.items()):
        lset = {e['line'] for e in entries}                 # 1-based
        for lp in (loops or []):
            body_lo, body_hi = lp['header'] + 2, lp['end'] + 1
            if any(body_lo <= ln <= body_hi for ln in lset):
                hdr_code = host_lines[lp['header']].strip()
                emit(key, {'line': lp['header'] + 1, 'kind': 'loop',
                           'bd_id': None, 'comment_line': None,
                           'code': hdr_code, 'note': loop_note(hdr_code, cmap)})
                for pj in lp['preamble']:
                    emit(key, {'line': pj + 1, 'kind': 'loopidx',
                               'bd_id': None, 'comment_line': None,
                               'code': host_lines[pj].strip()})
                emit(key, {'line': lp['end'] + 1, 'kind': 'loopend',
                           'bd_id': None, 'comment_line': None,
                           'code': host_lines[lp['end']].strip()})

    return channel_map


# ----------------------------------------------------------------------------
# high-level rollup from provenance JSON
# ----------------------------------------------------------------------------

def kernel_for_tile(prov, col, row):
    for grp in prov.get('load_kernel_group', []):
        for t in grp.get('tiles', []):
            if t.get('col') == col and t.get('row') == row:
                return grp.get('callee')
    return None


def tile_role(ttype, channels):
    dirs = {c.get('direction') for c in channels}
    if ttype == 'shim':
        if dirs == {'MM2S'}:
            return 'dma-in (host->array)'
        if dirs == {'S2MM'}:
            return 'dma-out (array->host)'
        return 'shim-io (in+out)'
    return 'compute'


def build_summary(channels):
    out = []
    for c in channels:
        d = c.get('direction')
        ch = c.get('channel')
        bd = c.get('bd_chain', [])
        length = bd[0].get('len') if bd else '?'
        acq = bd[0].get('acquire_lock', [{}])[0] if bd else {}
        rel = bd[0].get('release_lock', [{}])[0] if bd else {}
        verb = 'recv' if d == 'S2MM' else 'send'
        reps = ''
        si = c.get('start_io', [])
        if si:
            rc = si[0].get('repeat_count')
            if rc and rc != 1:
                reps = ' x%d' % rc
        lock = ''
        if acq or rel:
            lock = ' lock%s/%s' % (acq.get('id'), rel.get('id'))
        out.append('%s %sB ch%s %s%s%s' % (verb, length, ch, d, reps, lock))
    return out


# ----------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------

def _loop_trips(entry):
    """Number of outer-loop iterations from a flow_summary entry's loop_range
    ('a..b' -> b-a). Returns 1 when there is no enclosing scf.for."""
    lr = entry.get('loop_range')
    if not lr or '..' not in lr:
        return 1
    try:
        a, b = lr.split('..', 1)
        n = int(b) - int(a)
        return n if n > 0 else 1
    except (ValueError, TypeError):
        return 1


def _entry_fires(entry):
    """Explicit BD fire count = repeat_count * outer-loop trips (default 1)."""
    rc = entry.get('repeat_count') or 1
    return rc * _loop_trips(entry)


def compute_flow_balance(flow_summary):
    """Analyze each flow's producer (MM2S) vs consumer (S2MM) byte coverage.

    Returns a list of per-flow dicts:
      {flow_index, direction, pattern, supply_per_round, demand_per_round,
       balanced, note, participants:[{loc,io_direction,channel,bd_len,fires,
       is_shim,role}]}
    `balanced` is None when the pattern is not one we can check reliably.
    """
    out = []
    for f in flow_summary or []:
        entries = f.get('entries', []) or []
        parts = []
        producers, consumers = [], []
        for e in entries:
            io = e.get('io_direction')
            is_shim = e.get('tile_row') == 0
            rec = {
                'loc': [e.get('tile_col'), e.get('tile_row')],
                'io_direction': io,
                'channel': e.get('channel'),
                'bd_len': e.get('bd_len') or 0,
                'fires': _entry_fires(e),
                'repeat_count': e.get('repeat_count') or 1,
                'is_shim': is_shim,
                'role': 'producer' if io == 'MM2S' else 'consumer',
            }
            parts.append(rec)
            (producers if io == 'MM2S' else consumers).append(rec)

        shim_prod = [p for p in producers if p['is_shim']]
        core_prod = [p for p in producers if not p['is_shim']]
        shim_cons = [c for c in consumers if c['is_shim']]
        core_cons = [c for c in consumers if not c['is_shim']]

        pattern = 'unknown'
        supply = demand = None
        balanced = None
        note = ''

        if len(shim_prod) == 1 and core_cons and not core_prod and not shim_cons:
            pattern = 'broadcast'
            supply = shim_prod[0]['bd_len']
            demand = sum(c['bd_len'] for c in core_cons)
            balanced = (supply == demand)
            note = 'shim per-fire bytes vs sum of %d core reads' % len(core_cons)
        elif len(shim_cons) == 1 and core_prod and not core_cons and not shim_prod:
            pattern = 'gather'
            supply = sum(p['bd_len'] for p in core_prod)
            demand = shim_cons[0]['bd_len'] * len(core_prod)
            balanced = (supply == demand)
            note = 'sum of %d core writes vs shim per-fire x %d gather fires' % (
                len(core_prod), len(core_prod))
        elif len(producers) == 1 and len(consumers) == 1:
            pattern = 'p2p'
            supply = producers[0]['bd_len']
            demand = consumers[0]['bd_len']
            balanced = (supply == demand)
            note = 'producer per-fire vs consumer per-fire'
        else:
            note = 'unrecognized fan pattern; coverage check skipped'

        out.append({
            'flow_index': f.get('flow_index'),
            'direction': f.get('direction'),
            'pattern': pattern,
            'supply_per_round': supply,
            'demand_per_round': demand,
            'balanced': balanced,
            'note': note,
            'participants': parts,
        })
    return out


# ----------------------------------------------------------------------------
# Kernel code show (core tiles): parse kernel.cc and correlate each core-tile
# DMA channel with the kernel window/argument it feeds or drains.
#
# The generated kernel.cc carries stable, parseable markers:
#   #define LOCK_window_<name>_ACQ <n> / _REL <n>   -> per-window lock ids
#   // window_def window_<name>                       -> window declaration
#   window_init(window_window_<name>, .., LOCK_..ACQ, .., LOCK_..REL, size, ..)
#   // kernel_invoke <fn>
#   <fn>(get_input_async_window_..(window_window_<name>), ...)  -> arg order/dir
# ----------------------------------------------------------------------------

RE_KLOCK = re.compile(r'#define\s+LOCK_window_(\w+)_(ACQ|REL)\s+(\d+)')
RE_KWINDEF = re.compile(r'//\s*window_def\s+window_(\w+)')
RE_KWININIT = re.compile(
    r'window_init\(\s*window_window_(\w+)\s*,.*?LOCK_window_\w+_ACQ\s*,'
    r'.*?LOCK_window_\w+_REL\s*,\s*(\d+)\s*,')
RE_KINVOKE = re.compile(r'//\s*kernel_invoke\s+(\w+)')
RE_KARG = re.compile(r'get_(input|output)_async_window\w*\(\s*window_window_(\w+)\s*\)')
# Ping/pong buffer names referenced by a window_init(...) call (positional
# buffer args) and the .bcf symbol lines that map each buffer name -> address.
RE_KBUF = re.compile(r'\bbuf_\w+')
RE_BCF_SYMBOL = re.compile(r'^\s*_symbol\s+(\S+)\s+(0x[0-9A-Fa-f]+)')

RE_KWININIT_BC = re.compile(
    r'window_init\(\s*window_(\w+)\s*,\s*\d+\s*,'
    r'\s*(\w+)\s*,'
    r'\s*LOCK_\w+\s*,'
    r'\s*(\w+)\s*,'
    r'\s*LOCK_\w+\s*,'
    r'\s*(\d+)\s*,')
RE_KARG_BC = re.compile(r'get_(input|output)_async_window\w*\(\s*window_(\w+)\s*\)')
RE_KINVOKE_BC = re.compile(r'//\s*Kernel call\s*:\s*\w+:(\w+)')
RE_KBUF_BC = re.compile(r'\b(buf\w+)\b')
RE_KWININIT_RAW = re.compile(
    r'window_init\(\s*\w+\s*,\s*\d+\s*,\s*(\w+)\s*,\s*([^,\s]+)')
RE_KARG_RAW = re.compile(
    r'(input|output)_window\w*\s*\*\s*(\w+)_win_ptr\s*='
    r'\s*get_(?:input|output)_async_window\w*\(')
RE_KINVOKE_RAW = re.compile(
    r'^\s*(\w+)\s*\([^;]*\b\w+_win_ptr\b[^;]*\)\s*;\s*$')


def _win_dir(name):
    """Window direction from its generated name prefix (in_* / out_*)."""
    head = name.split('_', 1)[0]
    if head == 'in':
        return 'input'
    if head == 'out':
        return 'output'
    return 'unknown'


RE_KINCLUDE = re.compile(r'#include\s+"([^"]+\.cc)"')


def _strip_cc_comments(s):
    """Drop // and /* */ comments so structural scanning ignores commas/braces
    that appear inside comments (kernel signatures carry [K, tile_N] comments)."""
    s = re.sub(r'/\*.*?\*/', '', s)
    s = re.sub(r'//.*$', '', s)
    return s


def _split_top_commas(s):
    """Split on top-level commas (ignore commas nested in ()/[]/{})."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append(''.join(cur))
    return out


def _parse_kernel_func(flines, def_idx, function):
    """Parse the kernel function definition starting at/near line def_idx.

    Returns None if this occurrence is a call/declaration (no '{' body) rather
    than a definition. On success returns {function, start_line, end_line,
    lines[], params[]} where each param is {pos, name, dir, ref_lines[]}.
    """
    stripped = [_strip_cc_comments(l) for l in flines]
    # flatten (line_idx, char) from the def line onward for brace/paren matching
    flat = []
    for li in range(def_idx, len(flines)):
        for ch in stripped[li]:
            flat.append((li, ch))
        flat.append((li, '\n'))
    s = ''.join(ch for _, ch in flat)
    mname = re.search(r'\b' + re.escape(function) + r'\s*\(', s)
    if not mname:
        return None
    popen = s.index('(', mname.start())
    depth, pclose = 0, -1
    for k in range(popen, len(s)):
        if s[k] == '(':
            depth += 1
        elif s[k] == ')':
            depth -= 1
            if depth == 0:
                pclose = k
                break
    if pclose < 0:
        return None
    sig = s[popen + 1:pclose]
    # a definition has '{' after the signature; a call/decl has ';'
    j = pclose + 1
    while j < len(s) and s[j] in ' \t\n':
        j += 1
    if j >= len(s) or s[j] != '{':
        return None
    depth, bclose = 0, -1
    for k in range(j, len(s)):
        if s[k] == '{':
            depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0:
                bclose = k
                break
    if bclose < 0:
        return None
    start_li = def_idx
    end_li = flat[bclose][0]

    params = []
    for pos, part in enumerate(p for p in _split_top_commas(sig) if p.strip()):
        toks = re.findall(r'[A-Za-z_]\w*', part)
        if not toks:
            continue
        if 'output_window' in part or re.search(r'\boutput\b', part):
            d = 'output'
        elif 'input_window' in part or re.search(r'\binput\b', part):
            d = 'input'
        else:
            d = 'unknown'
        params.append({'pos': pos, 'name': toks[-1], 'dir': d, 'ref_lines': []})

    lines = [{'line': n + 1, 'code': flines[n]}
             for n in range(start_li, end_li + 1)]
    for p in params:
        rex = re.compile(r'\b' + re.escape(p['name']) + r'\b')
        p['ref_lines'] = [n + 1 for n in range(start_li, end_li + 1)
                          if rex.search(stripped[n])]
    return {'function': function, 'start_line': start_li + 1,
            'end_line': end_li + 1, 'lines': lines, 'params': params}


def _find_kernel_source(kernel_cc_path, src_lines, function):
    """Locate the file+lines that DEFINE `function`, following #include "*.cc".

    Searches the included .cc files first (that is where the user kernel body
    lives), then kernel.cc itself. Returns the parsed function dict augmented
    with {file, path}, or None when no definition is found (graceful).
    """
    if not function:
        return None
    base = os.path.dirname(kernel_cc_path)
    candidates = []
    for ln in src_lines:
        m = RE_KINCLUDE.search(ln)
        if m:
            p = os.path.join(base, m.group(1))
            if os.path.isfile(p):
                try:
                    with open(p) as f:
                        candidates.append((m.group(1), p, f.read().split('\n')))
                except OSError:
                    pass
    candidates.append((os.path.basename(kernel_cc_path), kernel_cc_path, src_lines))
    defre = re.compile(r'\b' + re.escape(function) + r'\s*\(')
    for fname, fpath, flines in candidates:
        for i, ln in enumerate(flines):
            if defre.search(ln):
                parsed = _parse_kernel_func(flines, i, function)
                if parsed:
                    parsed['file'] = fname
                    parsed['path'] = os.path.abspath(fpath)
                    return parsed
    return None


def parse_kernel(kernel_cc_path):
    """Parse a generated kernel.cc into a window/argument model.

    Returns None if the file is absent (graceful skip). On success returns a
    dict {path, function, windows[], args[], lines[]} where each window has
    {name, dir, acq_lock, rel_lock, buf_size, def_line, init_line} and each arg
    has {arg, window, dir, line}. `lines` holds the marker line records for the
    relevant-slice render.
    """
    if not kernel_cc_path or not os.path.isfile(kernel_cc_path):
        return None
    try:
        with open(kernel_cc_path) as f:
            text = f.read()
    except OSError:
        return None
    src = text.split('\n')
    raw_defs = {}
    for ln in src:
        match = re.match(r'\s*#define\s+(\w+)\s+(\d+)\s*$', ln)
        if match:
            raw_defs[match.group(1)] = int(match.group(2))

    windows = {}         # name -> window dict
    order = []           # window names in first-seen order
    marker_lines = []    # (line_no, code) records for the render slice

    def _win(name):
        if name not in windows:
            windows[name] = {'name': name, 'dir': _win_dir(name),
                             'acq_lock': None, 'rel_lock': None,
                             'buf_size': None, 'def_line': None,
                             'init_line': None, 'lock_lines': [],
                             'buffers': [], 'buf_decl_lines': []}
            order.append(name)
        return windows[name]

    function = None
    invoke_line = None
    for i, ln in enumerate(src, 1):
        m = RE_KLOCK.search(ln)
        if m:
            w = _win(m.group(1))
            w['acq_lock' if m.group(2) == 'ACQ' else 'rel_lock'] = int(m.group(3))
            w['lock_lines'].append(i)
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KWINDEF.search(ln)
        if m:
            w = _win(m.group(1))
            w['def_line'] = i
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KWININIT.search(ln)
        if m:
            w = _win(m.group(1))
            w['buf_size'] = int(m.group(2))
            w['init_line'] = i
            # Positional ping/pong buffer names -> .bcf address correlation.
            for bm in RE_KBUF.finditer(ln):
                if bm.group(0) not in w['buffers']:
                    w['buffers'].append(bm.group(0))
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KWININIT_BC.search(ln)
        if m and not RE_KWININIT.search(ln):
            win_name  = m.group(1)
            ping_buf  = m.group(2)
            pong_buf  = m.group(3)
            buf_size  = int(m.group(4))
            w = _win(win_name)
            w['buf_size']  = buf_size
            w['init_line'] = i
            for buf in (ping_buf, pong_buf):
                if buf and buf not in w['buffers']:
                    w['buffers'].append(buf)
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KWININIT_RAW.search(ln)
        if m:
            w = _win(m.group(1))
            w['buffers'].append(m.group(1))
            w['buf_size'] = raw_defs.get(m.group(2))
            w['init_line'] = i
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KINVOKE.search(ln)
        if m:
            function = m.group(1)
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KINVOKE_BC.search(ln)
        if m and function is None:
            function = m.group(1)
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KINVOKE_RAW.search(ln)
        if m and function is None:
            function = m.group(1)
            marker_lines.append({'line': i, 'code': ln})
            continue

    # Buffer declaration lines (`<type> buf_xxx[N];`) per window, for highlight.
    for w in windows.values():
        for buf in w['buffers']:
            decl_re = re.compile(r'\b' + re.escape(buf) + r'\s*\[')
            for i, ln in enumerate(src, 1):
                if decl_re.search(ln) and '=' not in ln.split(buf, 1)[0]:
                    if i not in w['buf_decl_lines']:
                        w['buf_decl_lines'].append(i)

    # Invoke call line: the call to <function>( ... ) with the arg windows.
    args = []
    win_dir_by_line = {}
    for i, ln in enumerate(src, 1):
        for am in list(RE_KARG.finditer(ln)) + list(RE_KARG_BC.finditer(ln)):
            d = 'input' if am.group(1) == 'input' else 'output'
            wn = am.group(2)
            win_dir_by_line[i] = (d, wn)
            if wn in windows:
                windows[wn]['dir'] = d
        for am in RE_KARG_RAW.finditer(ln):
            d = 'input' if am.group(1) == 'input' else 'output'
            wn = am.group(2)
            win_dir_by_line[i] = (d, wn)
            _win(wn)['dir'] = d
    if function:
        call_re = re.compile(r'\b' + re.escape(function) + r'\s*\(')
        for i, ln in enumerate(src, 1):
            if call_re.search(ln):
                arg_matches = list(RE_KARG.finditer(ln)) + list(RE_KARG_BC.finditer(ln))
                if arg_matches:
                    invoke_line = i
                    marker_lines.append({'line': i, 'code': ln})
                    for pos, am in enumerate(arg_matches):
                        d = 'input' if am.group(1) == 'input' else 'output'
                        win_name = am.group(2)
                        args.append({'arg': pos, 'window': win_name, 'dir': d})
                        if win_name in windows:
                            windows[win_name]['dir'] = d
                    break
                if 'get_' not in ln and win_dir_by_line:
                    invoke_line = i
                    marker_lines.append({'line': i, 'code': ln})
                    for pos, (d, wn) in enumerate(
                            v for _, v in sorted(win_dir_by_line.items())):
                        args.append({'arg': pos, 'window': wn, 'dir': d})
                    break

    marker_lines.sort(key=lambda r: r['line'])
    # Resolve and parse the actual kernel function body (follows #include "*.cc").
    source = _find_kernel_source(kernel_cc_path, src, function)
    return {
        'path': os.path.abspath(kernel_cc_path),
        'file': os.path.basename(kernel_cc_path),
        'function': function,
        'invoke_line': invoke_line,
        'windows': [windows[n] for n in order],
        'args': args,
        'lines': marker_lines,
        # Full kernel.cc text (window_init / lock / buffer decls) for the new
        # "kernel.cc" sub-tab; highlighted per focused window.
        'kernel_lines': [{'line': i, 'code': ln} for i, ln in enumerate(src, 1)],
        'source': source,
    }


def find_bcf(workdir):
    """Locate the linker .bcf (buffer symbol map) in workdir; None if absent."""
    matches = sorted(glob.glob(os.path.join(workdir, '*.bcf')))
    return matches[0] if matches else None


def parse_bcf(bcf_path):
    """Parse a .bcf into {path, file, lines[], symbols[]}.

    Each `_symbol <name> <0xADDR>` line maps a kernel buffer name to its tile
    memory address; `symbols` records {name, addr, line} for buffer<->address
    correlation with the focused channel's window. Graceful None if absent.
    """
    if not bcf_path or not os.path.isfile(bcf_path):
        return None
    try:
        with open(bcf_path) as f:
            text = f.read()
    except OSError:
        return None
    src = text.split('\n')
    symbols = []
    for i, ln in enumerate(src, 1):
        m = RE_BCF_SYMBOL.match(ln)
        if m:
            symbols.append({'name': m.group(1), 'addr': m.group(2), 'line': i})
    return {
        'path': os.path.abspath(bcf_path),
        'file': os.path.basename(bcf_path),
        'lines': [{'line': i, 'code': ln} for i, ln in enumerate(src, 1)],
        'symbols': symbols,
    }


# Buffer base offset: a core-tile proc address minus this yields the DMA-view
# address literal that host.cc passes to __runtime_buffer_arg((void*)ADDR); the
# .bcf `_symbol <name> 0x...` addresses are proc addresses (ADDR + WIN_BASE).
WIN_BASE = 0x70000


def match_channels_to_kernel(channels, kernel_view, bcf_view, host_lines, var_def,
                             win_base=WIN_BASE):
    """Correlate each core-tile DMA channel with a kernel window/argument by the
    BD buffer address (deterministic, no lock/positional heuristics).

    Chain per channel BD:
      1. host.cc `__Runtime_dma_bd_config(dev, tile, BUFVAR, bd_id, ...)` names a
         buffer SSA var whose def is
         `void* BUFVAR = __runtime_buffer_arg((void*)ADDR);` where ADDR is the
         DMA-view tile address (proc_addr - WIN_BASE).
      2. ADDR + WIN_BASE == a `.bcf` `_symbol <name> 0x...` address -> buffer
         name (e.g. buf_in_ping_1).
      3. That buffer name appears in a kernel.cc `window_init(...)` -> its window.

    Primary path uses host.cc SSA (`method='addr'`). When the host.cc parse
    fails, falls back to the provenance `bd_chain[i]['buffer_offset']` (already
    the DMA-view ADDR, `method='addr(prov)'`). `method='none'` if no address is
    recoverable. Shim tiles use a symbolic (DDR) buffer arg and are not matched
    here (this runs for core tiles only).
    """
    if not kernel_view:
        return {'matches': []}
    wins = kernel_view.get('windows', [])
    arg_of = {}
    for a in kernel_view.get('args', []):
        arg_of[a['window']] = a['arg']
    # buffer name -> owning window (from kernel.cc window_init buffer lists).
    buf2win = {}
    for w in wins:
        for buf in w.get('buffers', []):
            buf2win.setdefault(buf, w)
    # DMA-view address -> window / symbol name (via .bcf symbol table).
    dmaview2win = {}
    addr2sym = {}
    for s in (bcf_view or {}).get('symbols', []):
        try:
            proc_addr = int(s['addr'], 16)
        except (ValueError, TypeError, KeyError):
            continue
        dmaview = proc_addr - win_base
        addr2sym[dmaview] = s['name']
        w = buf2win.get(s['name'])
        if w is not None:
            dmaview2win.setdefault(dmaview, w)

    def _bd_addr(entry):
        """Resolve one bd_config entry -> DMA-view ADDR via host.cc SSA, or None.

        Reads the __Runtime_dma_bd_config call line for the buffer SSA var, then
        follows that var's def (`__runtime_buffer_arg((void*)ADDR)`)."""
        ln = host_lines[entry['line'] - 1]
        mc = RE_C_BDCALL.search(ln)
        if not mc:
            return None
        bufvar = mc.group(1)
        def_line = var_def.get(bufvar)
        if not def_line:
            return None
        ma = RE_C_BUFARG.search(host_lines[def_line - 1])
        if not ma:
            return None
        return int(ma.group(1))

    win_by_name = {w['name']: w for w in wins}

    matches = []
    for c in channels:
        direction = c.get('direction')

        pwin = c.get('kernel_window')
        if pwin and pwin in win_by_name:
            w = win_by_name[pwin]
            matches.append({
                'direction': direction,
                'channel': c.get('channel'),
                'window': w['name'],
                'arg': arg_of.get(w['name']),
                'buf_size': w.get('buf_size'),
                'method': 'port',
                'bcf_syms': list(c.get('kernel_buffers', []) or w.get('buffers', [])),
                'addrs': [],
                'addrs_hex': [],
                'port_id': c.get('kernel_port_id'),
            })
            continue

        addrs = []
        method = 'none'
        # Primary: host.cc SSA (user's path) for every BD line of this channel.
        for e in c.get('host_lines', []) or []:
            if e.get('kind') != 'bd_config':
                continue
            a = _bd_addr(e)
            if a is not None:
                addrs.append(a)
        if addrs:
            method = 'addr'
        else:
            # Fallback: provenance bd_chain buffer_offset (already DMA-view ADDR).
            for bd in c.get('bd_chain', []) or []:
                off = bd.get('buffer_offset')
                if off is not None:
                    addrs.append(int(off))
            if addrs:
                method = 'addr(prov)'
        # Dedupe addresses preserving first-seen order (ping/pong collapse).
        seen = set()
        uaddrs = []
        for a in addrs:
            if a not in seen:
                seen.add(a)
                uaddrs.append(a)
        # Map addresses -> .bcf symbols + the single owning window.
        chosen = None
        bcf_syms = []
        for a in uaddrs:
            bcf_syms.append(addr2sym.get(a, '(unmatched-addr)'))
            w = dmaview2win.get(a)
            if w is not None and chosen is None:
                chosen = w
        matches.append({
            'direction': direction,
            'channel': c.get('channel'),
            'window': chosen['name'] if chosen else None,
            'arg': arg_of.get(chosen['name']) if chosen else None,
            'buf_size': chosen.get('buf_size') if chosen else None,
            'method': method,
            'bcf_syms': bcf_syms,
            'addrs': uaddrs,
            'addrs_hex': ['0x%x' % a for a in uaddrs],
        })
    return {'matches': matches}


# ----------------------------------------------------------------------------
# dfschedule IR slicing (Middle tab): verbatim per-tile / per-channel slice of
# the executable schedule dump 6_BlueprintToSchedulePass.mlir.
#
# The dfschedule dialect is a clean SSA def-use chain, so a scope-aware
# reachability slice extracts a tile's/channel's ops WITHOUT any compiler change.
# SSA names reset per region (each RoutingCreate round reuses %4, %5, ...), so
# operand resolution is scope-aware: a use binds to the nearest prior def whose
# enclosing region is an ancestor-or-equal of the use's region.
# ----------------------------------------------------------------------------

DFSCHED_IR_NAME = '6_BlueprintToSchedulePass.mlir'

# Region-carrying ops in this IR. A line ending with '{' that mentions one of
# these opens a nested-op region; any other line ending with '{' opens an
# attribute dict on a single op (closed later by `} : <types>`).
_MLIR_REGION_KEYWORDS = ('module', 'func.func', 'scf.', 'affine.', 'RoutingCreate')

RE_MLIR_DEF = re.compile(r'^\s*(%[A-Za-z0-9_]+)\s*=')
RE_MLIR_SSA = re.compile(r'%[A-Za-z0-9_]+')
RE_MLIR_DECLTILE = re.compile(
    r'dfschedule\.declaretile\s*\{\s*col\s*=\s*(-?\d+)\s*:\s*i32\s*,\s*'
    r'row\s*=\s*(-?\d+)\s*:\s*i32')


def find_dfschedule_ir(workdir):
    """Locate 6_BlueprintToSchedulePass.mlir; return path or None (graceful)."""
    candidates = []
    env = os.environ.get('SCHEDULE_VIEW_IR_DIR')
    if env:
        candidates.append(os.path.join(env, DFSCHED_IR_NAME))
    wd = os.path.abspath(workdir)
    for up in ('ir/dfschedule', '../ir/dfschedule', '../../ir/dfschedule'):
        candidates.append(os.path.join(wd, up, DFSCHED_IR_NAME))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _mlir_is_region_open(s):
    return s.endswith('{') and any(k in s for k in _MLIR_REGION_KEYWORDS)


def _brace_delta(s):
    return (s.count('(') + s.count('[') + s.count('{')
            - s.count(')') - s.count(']') - s.count('}'))


class DfscheduleSlicer:
    """Statement-level def-use slicer over a dfschedule .mlir dump."""

    def __init__(self, text):
        self.stmts = []          # {idx,text,result,operands,scope,region_stack}
        self.def_by_name = {}    # ssa name -> [(idx, scope), ...]
        self.tile_ssa = {}       # (col,row) -> set(declaretile stmt idx)
        self._parse(text)
        self._resolve()

    # -- parsing --------------------------------------------------------------
    def _parse(self, text):
        lines = text.split('\n')
        region_stack = []        # raw header lines (context)
        scope_stack = []         # region ids -> current scope tuple
        rid = [0]
        cur, depth, cur_start = [], 0, 0
        for lineno, raw in enumerate(lines, 1):   # 1-based source line numbers
            s = raw.strip()
            if depth == 0:
                if s == '':
                    continue
                if s == '}':                      # region close
                    if scope_stack:
                        scope_stack.pop()
                    if region_stack:
                        region_stack.pop()
                    continue
                if _mlir_is_region_open(s):        # region open (context header)
                    region_stack.append(raw)
                    rid[0] += 1
                    scope_stack.append(rid[0])
                    continue
                if s.startswith('^') and s.endswith(':'):   # block label
                    continue
                cur = [raw]
                cur_start = lineno
                depth += _brace_delta(s)
                if depth <= 0:
                    depth = 0
                    self._add_stmt(cur, cur_start, tuple(scope_stack), region_stack)
                    cur = []
            else:                                 # inside a multi-line statement
                cur.append(raw)
                depth += _brace_delta(s)
                if depth <= 0:
                    depth = 0
                    self._add_stmt(cur, cur_start, tuple(scope_stack), region_stack)
                    cur = []

    def _add_stmt(self, raw_lines, start_line, scope, region_stack):
        idx = len(self.stmts)
        text = '\n'.join(raw_lines)
        mdef = RE_MLIR_DEF.match(raw_lines[0])
        result = mdef.group(1) if mdef else None
        operands = []
        for j, rl in enumerate(raw_lines):
            body = rl
            if j == 0 and result and '=' in rl:
                body = rl.split('=', 1)[1]
            operands.extend(m.group(0) for m in RE_MLIR_SSA.finditer(body))
        self.stmts.append({
            'idx': idx, 'text': text, 'result': result, 'operands': operands,
            'scope': scope, 'region_stack': list(region_stack),
            'line': start_line, 'lines': list(raw_lines),
        })
        if result:
            self.def_by_name.setdefault(result, []).append((idx, scope))
        m = RE_MLIR_DECLTILE.search(text)
        if m:
            self.tile_ssa.setdefault((int(m.group(1)), int(m.group(2))),
                                     set()).add(idx)

    # -- def-use resolution ---------------------------------------------------
    def _resolve(self):
        self.fwd = {}    # producer idx -> set(consumer idx)
        self.bwd = {}    # consumer idx -> set(producer idx)
        for st in self.stmts:
            pset = set()
            for name in st['operands']:
                p = self._resolve_name(name, st['idx'], st['scope'])
                if p is not None:
                    pset.add(p)
            self.bwd[st['idx']] = pset
            for p in pset:
                self.fwd.setdefault(p, set()).add(st['idx'])

    def _resolve_name(self, name, use_idx, use_scope):
        best = None
        for (didx, dscope) in self.def_by_name.get(name, ()):
            if didx >= use_idx:
                continue
            if len(dscope) <= len(use_scope) and use_scope[:len(dscope)] == dscope:
                if best is None or didx > best:
                    best = didx
        return best

    def _close_backward(self, sel, seeds):
        stack = list(seeds)
        while stack:
            c = stack.pop()
            for p in self.bwd.get(c, ()):
                if p not in sel:
                    sel.add(p)
                    stack.append(p)

    # -- slices ---------------------------------------------------------------
    def tile_slice(self, col, row):
        seeds = self.tile_ssa.get((col, row))
        if not seeds:
            return '(no matching ops)'
        sel = set(seeds)
        stack = list(seeds)                       # forward reachability
        while stack:
            for c in self.fwd.get(stack.pop(), ()):
                if c not in sel:
                    sel.add(c)
                    stack.append(c)
        self._close_backward(sel, list(sel))      # backward completion
        return self._emit(sel)

    def channel_slice(self, col, row, direction, channel):
        tiles = self.tile_ssa.get((col, row), set())
        seeds = []
        for st in self.stmts:
            if 'dfschedule.config.create_io' not in st['text']:
                continue
            if not (self.bwd.get(st['idx'], set()) & tiles):
                continue
            if not re.search(r'channel\s*=\s*%d\b' % channel, st['text']):
                continue
            if not re.search(r'direction\s*=\s*"%s"' % re.escape(direction),
                             st['text']):
                continue
            seeds.append(st['idx'])
        if not seeds:
            return '(no matching ops)'
        sel = set(seeds)
        self._close_backward(sel, seeds)          # bd chain -> buffers -> tile
        frontier = list(seeds)                    # create_io -> start_io -> wait
        while frontier:
            nxt = []
            for f in frontier:
                for c in self.fwd.get(f, ()):
                    txt = self.stmts[c]['text']
                    if ('schedule.start_io' in txt or 'schedule.wait' in txt) \
                            and c not in sel:
                        sel.add(c)
                        nxt.append(c)
                        self._close_backward(sel, [c])   # pull getbdid, etc.
            frontier = nxt
        return self._emit(sel)

    # -- emission -------------------------------------------------------------
    def _emit(self, sel):
        # Returns a list of {line, code} rows preserving the ORIGINAL source line
        # number from 6_BlueprintToSchedulePass.mlir. Context headers, blank
        # separators and '// ...' elision markers carry line=None (no source
        # line). Multi-line statements number each physical line.
        idxs = sorted(sel)
        MAXN = 200
        trunc = len(idxs) > MAXN
        idxs = idxs[:MAXN]
        out, last_ctx, prev = [], None, None
        for idx in idxs:
            st = self.stmts[idx]
            ctx = [h.strip() for h in st['region_stack']
                   if ('scf.' in h or 'RoutingCreate' in h)]
            if ctx != last_ctx:
                if out:
                    out.append({'line': None, 'code': ''})
                out.extend({'line': None, 'code': '// ' + h} for h in ctx)
                last_ctx = ctx
            elif prev is not None and idx != prev + 1:
                out.append({'line': None, 'code': '// ...'})
            for off, code in enumerate(st['lines']):
                out.append({'line': st['line'] + off, 'code': code})
            prev = idx
        if trunc:
            out.append({'line': None,
                        'code': '// ... (slice truncated at %d statements)' % MAXN})
        return out


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

import re as _re



def _normalize_pkt_connect(conn):
    """Fill PKT slot metadata matching routinghwlower.cpp constants."""
    if conn.get('kind') != 'packet_connect':
        return
    for leg, defaults in (
        ('recv_slave', {'mask': 0, 'msel': 0, 'arbiter': 0, 'slot': 0}),
        ('local_dma', {'mask': 0x1f, 'msel': 0, 'arbiter': 0, 'slot': 0}),
    ):
        port = conn.setdefault(leg, {})
        if port.get('dir') in (None, 'NONE'):
            continue
        for k, v in defaults.items():
            port.setdefault(k, v)
    fm = conn.setdefault('forward_master', {})
    if fm.get('dir') not in (None, 'NONE'):
        fm.setdefault('arbiter', 0)
        fm.setdefault('msel_en', 1)


def _normalize_routing_groups(groups):
    for g in groups:
        for c in g.get('connections', []):
            _normalize_pkt_connect(c)


def _load_comm_paths(workdir):
    """Load comm_paths from dmaphopprovenacemap.json + optional routingprovenancemap.json.

    For each communication path, builds an ordered tile waypoint list and hop list.

    Push paths: shim is producer, data flows shim→cores.
      - dmaphop hops are in order: (shim→core0), (core0→core1), ...
    Pull paths: cores are producers, data flows cores→shim.
      - dmaphop hops are in REVERSE order: (coreN→shim), ..., (core0→core1)
      - We reverse them to get data-flow order.

    When routingprovenancemap.json exists, intermediate routing tiles (stream-switch path
    through MEM tiles) are filled in via circuit_connect chain reconstruction.
    """
    _OPPOSITE = {'NORTH': 'SOUTH', 'SOUTH': 'NORTH', 'EAST': 'WEST', 'WEST': 'EAST'}
    _DELTA    = {'NORTH': (0, 1), 'SOUTH': (0, -1), 'EAST': (1, 0), 'WEST': (-1, 0)}

    def parse_coord(s):
        m = _re.search(r'\((\d+),(\d+)\)', s)
        return (int(m.group(1)), int(m.group(2))) if m else None

    hop_path = os.path.join(workdir, 'dmaphopprovenacemap.json')
    if not os.path.exists(hop_path):
        return []
    try:
        with open(hop_path) as f:
            hop_data = json.load(f)
    except Exception:
        return []

    shim_row = 0

    rp_path = os.path.join(workdir, 'routingprovenancemap.json')
    rp_groups = []
    if os.path.exists(rp_path):
        try:
            with open(rp_path) as f:
                rp_groups = json.load(f).get('routing_groups', [])
            _normalize_routing_groups(rp_groups)
        except Exception:
            pass

    rp_by_dma_tiles = {}
    rp_by_flow_index = {}
    for g in rp_groups:
        conns = g.get('connections', [])
        dma_key = frozenset(
            (c['tile']['col'], c['tile']['row'])
            for c in conns
            if c.get('kind') == 'circuit_connect'
            and c.get('master', {}).get('dir') == 'DMA'
        )
        if dma_key:
            rp_by_dma_tiles[dma_key] = g
        if 'flow_index' in g:
            rp_by_flow_index[g['flow_index']] = g

    def routing_edges_for_flow(direction, nonshim_tiles, flow_index=None):
        """Extract adjacency edges from the routing group for this flow.

        When groups carry a `flow_index` (baremetal provenance), match on that —
        point-to-point nets can traverse identical tiles so a tile-set key is
        ambiguous.  Otherwise fall back to the DMA-tile frozenset key used by
        native aiehlc broadcast/gather groups.

        Row groups store both push and pull circuit_connect entries in one list,
        split at the shim_aie_to_ext marker. Col groups have push entries only.
        Off-grid edges (destination row < 0) are dropped.
        Returns (edges, tiles_seen, dma_tiles) where dma_tiles is the set of
        core tiles that are DMA consumers (used for intermediate-tap markers)."""
        g = None
        if rp_by_flow_index and flow_index is not None:
            g = rp_by_flow_index.get(flow_index)
        if g is None:
            g = rp_by_dma_tiles.get(frozenset(nonshim_tiles))
        if not g:
            return [], set(), set(), set()
        conns = g.get('connections', [])

        split_idx = next(
            (i for i, c in enumerate(conns) if c.get('kind') == 'shim_aie_to_ext'),
            None
        )
        if split_idx is not None:
            cc_conns = conns[:split_idx] if direction == 'push' else conns[split_idx + 1:]
        else:
            cc_conns = conns

        push_conns = conns[:split_idx] if split_idx is not None else conns
        packet_tiles = {(c['tile']['col'], c['tile']['row'])
                        for c in push_conns if c.get('kind') == 'packet_connect'}

        pc_tiles = {(c['tile']['col'], c['tile']['row'])
                    for c in cc_conns if c.get('kind') == 'packet_connect'}

        edges = []
        tiles_seen = set()
        dma_tiles = set()
        first_cc = True
        for c in cc_conns:
            if c.get('kind') != 'circuit_connect':
                continue
            t = c['tile']
            fc, fr = t['col'], t['row']
            mdir = c['master']['dir']
            if mdir == 'DMA':
                tiles_seen.add((fc, fr))
                dma_tiles.add((fc, fr))
                continue
            d = _DELTA.get(mdir)
            if not d:
                continue
            tc, tr = fc + d[0], fr + d[1]
            if tr < 0:
                continue
            if first_cc and pc_tiles:
                first_cc = False
                sdir = c.get('slave', {}).get('dir')
                sd = _DELTA.get(sdir)
                if sd:
                    src_c, src_r = fc + sd[0], fr + sd[1]
                    if (src_c, src_r) in pc_tiles and src_r >= 0:
                        edges.append([[src_c, src_r], [fc, fr]])
                        tiles_seen.add((src_c, src_r))
            else:
                first_cc = False
            edges.append([[fc, fr], [tc, tr]])
            tiles_seen.add((fc, fr))
            tiles_seen.add((tc, tr))

        # Groups that name their DMA endpoints directly (work2provenance) are
        # taken at their word; deriving them from a master.dir=='DMA' entry
        # only works for push, where the DMA is the master.
        for t in g.get('dma_tiles', []):
            dma_tiles.add((t[0], t[1]))
            tiles_seen.add((t[0], t[1]))

        if direction == 'pull' and split_idx is not None:
            for c in push_conns:
                if c.get('kind') != 'packet_connect':
                    continue
                t = c['tile']
                fc, fr = t['col'], t['row']
                fm = c.get('forward_master', {})
                fmdir = fm.get('dir')
                d = _DELTA.get(fmdir)
                if not d:
                    continue
                tc, tr = fc + d[0], fr + d[1]
                if tr < 0 or (tc, tr) not in packet_tiles:
                    continue
                edge = [[fc, fr], [tc, tr]]
                if edge not in edges:
                    edges.append(edge)
                    tiles_seen.add((fc, fr))
                    tiles_seen.add((tc, tr))

        return edges, tiles_seen, dma_tiles, packet_tiles

    result = []
    for p in hop_data.get('communication_paths', []):
        path_id  = p.get('id', '')
        direction = p.get('direction', 'push')

        producer = next((s for s in p.get('stages', []) if s.get('role') == 'producer'), None)
        consumer = next((s for s in p.get('stages', []) if s.get('role') == 'consumer'), None)
        channel  = next((s for s in p.get('stages', []) if s.get('role') == 'channel'),  None)

        if direction == 'push':
            port_sym = (producer or {}).get('port_sym', '')
            shim_tile = (producer or {}).get('tile') or {}
        else:
            port_sym = (consumer or {}).get('port_sym', '')
            shim_tile = (consumer or {}).get('tile') or {}

        fm = _re.match(r'f(\d+)_', port_sym)
        seq_m = _re.search(r'(\d+)$', path_id)
        flow_index = int(fm.group(1)) if fm else (int(seq_m.group(1)) if seq_m else 0)

        shim_col = shim_tile.get('col')
        shim_row_val = shim_tile.get('row', shim_row)

        raw_pairs = []
        if channel:
            for h in channel.get('hops', []):
                src = parse_coord(h.get('from', ''))
                dst = parse_coord(h.get('to', ''))
                if src and dst:
                    raw_pairs.append((src, dst, h.get('hop_type'), h.get('shmem_kind')))

        if direction == 'pull':
            raw_pairs = list(reversed(raw_pairs))

        nonshim_tiles = list({(c, r) for (c, r) in
                              [p for (s, d, hint, kind) in raw_pairs if hint != 'shmem'
                               for p in (s, d)]
                              if r != shim_row})

        routing_edges, routing_tiles, routing_dma_tiles, routing_packet_tiles = routing_edges_for_flow(direction, nonshim_tiles, flow_index)

        if routing_edges:
            edges_out      = routing_edges
            tiles_list     = [list(t) for t in routing_tiles]
            dma_tiles_out  = [list(t) for t in routing_dma_tiles]
            packet_tiles_out = [list(t) for t in routing_packet_tiles]
        else:
            edges_out  = [[[fc, fr], [tc, tr]] for (fc, fr), (tc, tr), hint, kind in raw_pairs
                          if hint != 'shmem' and abs(fc - tc) + abs(fr - tr) == 1]
            tiles_list = []
            seen = set()
            for (fc, fr), (tc, tr), hint, kind in raw_pairs:
                for t in ((fc, fr), (tc, tr)):
                    if t not in seen:
                        seen.add(t); tiles_list.append(list(t))
            dma_tiles_out    = []
            packet_tiles_out = []

        # Synthesize the missing shim-edge: routing_edges_for_flow drops the
        # DMA tile (row 0 shim) because it has master.dir=='DMA' and skips it.
        # Connect the shim to its row-1 neighbour so the arc reaches row 0.
        if shim_col is not None:
            shim_key  = (shim_col, shim_row_val)
            edge_set  = {(e[0][0], e[0][1], e[1][0], e[1][1]) for e in edges_out}
            edge_set |= {(e[1][0], e[1][1], e[0][0], e[0][1]) for e in edges_out}
            row1_key  = (shim_col, shim_row_val + 1)
            shim_edge_missing = (
                shim_key not in {(e[0][0], e[0][1]) for e in edges_out} and
                shim_key not in {(e[1][0], e[1][1]) for e in edges_out}
            )
            row1_present = row1_key in (
                {(e[0][0], e[0][1]) for e in edges_out} |
                {(e[1][0], e[1][1]) for e in edges_out}
            )
            if shim_edge_missing and row1_present:
                if direction == 'pull':
                    edges_out.append([list(row1_key), list(shim_key)])
                else:
                    edges_out.insert(0, [list(shim_key), list(row1_key)])
                tiles_list.append(list(shim_key))

        routing_edge_set = {(e[0][0], e[0][1], e[1][0], e[1][1]) for e in routing_edges}
        packet_tile_set  = {(t[0], t[1]) for t in routing_packet_tiles}
        hops_out = []
        for (fc, fr), (tc, tr), hint, kind in raw_pairs:
            dist = abs(fc - tc) + abs(fr - tr)
            if hint:
                htype = hint
            elif dist != 1:
                htype = 'stream'
            elif ((fc, fr, tc, tr) in routing_edge_set or
                  (tc, tr, fc, fr) in routing_edge_set):
                htype = 'stream'
            elif (fc, fr) in packet_tile_set and (tc, tr) in packet_tile_set:
                htype = 'stream'
            else:
                htype = 'shmem'
            hop = {'from_col': fc, 'from_row': fr,
                   'to_col': tc, 'to_row': tr,
                   'type': htype}
            if kind:
                hop['kind'] = kind
            hops_out.append(hop)

        prod = [shim_col, shim_row_val]

        rg = None
        if rp_by_flow_index and flow_index is not None:
            rg = rp_by_flow_index.get(flow_index)
        if rg is None:
            rg = rp_by_dma_tiles.get(frozenset(nonshim_tiles))
        rg_conns_all = rg.get('connections', []) if rg else []
        if rg_conns_all:
            _split = next((i for i, c in enumerate(rg_conns_all)
                           if c.get('kind') == 'shim_aie_to_ext'), None)
            if _split is not None:
                if direction == 'push':
                    # Packet entries in a split group are the gather segment:
                    # local_dma is a packet *slave*, i.e. the tile sourcing
                    # into the stream, which only happens on the way out.  A
                    # push consumer takes delivery on a circuit_connect whose
                    # master is DMA.  Leaving them in attributed the pull
                    # flow's gather config to the push flow as well.
                    rg_connections = [c for c in rg_conns_all[:_split]
                                      if c.get('kind') != 'packet_connect']
                else:
                    # Pull flows: keep only packet_connect entries from the push
                    # section (packet-switch gather config is shared) plus all
                    # entries after shim_aie_to_ext (the pull-side routing).
                    # Exclude push-section circuit_connect entries: they configure
                    # the input distribution stream switch and must not appear on
                    # pull flows or they get wrongly attributed to both.
                    push_pkts = [c for c in rg_conns_all[:_split]
                                 if c.get('kind') == 'packet_connect']
                    # Keep the shim_aie_to_ext marker itself: it is this pull
                    # flow's egress port, and dropping it left pull nets with
                    # no GMIO section and the port attributed to no flow.
                    pull_conns = rg_conns_all[_split:]
                    rg_connections = push_pkts + pull_conns
            else:
                rg_connections = rg_conns_all
        else:
            rg_connections = []

        stages_raw = p.get('stages', [])

        result.append({
            'id': path_id,
            'flow_index': flow_index,
            'direction': direction,
            'prod_col': prod[0],
            'prod_row': prod[1],
            'tiles':     tiles_list,
            'edges':     edges_out,
            'dma_tiles':    dma_tiles_out,
            'packet_tiles': packet_tiles_out,
            'hops':      hops_out,
            'routing_connections': rg_connections,
            'stages':    stages_raw,
        })

    return result


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build_view(workdir):
    prov_path = os.path.join(workdir, 'dfscheduleprovenancemap.json')
    host_path = os.path.join(workdir, 'host.cc')
    with open(prov_path) as f:
        prov = json.load(f)
    with open(host_path) as f:
        host_lines = f.read().split('\n')

    kernel_view = parse_kernel(os.path.join(workdir, 'kernel.cc'))
    # .bcf buffer symbol map (buffer name -> tile address), for the new "*.bcf"
    # sub-tab correlated to the focused channel's window buffers.
    bcf_view = parse_bcf(find_bcf(workdir))

    _kernel_cache = {}
    _bcf_cache = {}
    def _tile_kernel_view(rel):
        if not rel:
            return None
        if rel not in _kernel_cache:
            p = os.path.join(workdir, rel)
            _kernel_cache[rel] = parse_kernel(p) if os.path.isfile(p) else None
        return _kernel_cache[rel]
    def _tile_bcf_view(rel):
        if not rel:
            return None
        if rel not in _bcf_cache:
            p = os.path.join(workdir, rel)
            _bcf_cache[rel] = parse_bcf(p) if os.path.isfile(p) else None
        return _bcf_cache[rel]

    # The tiling flow names its function host_canonicalized; the raw-XAie
    # single-kernel flow (static-xaie provenance) wraps its calls in a
    # differently named function surfaced as host_entry_fn. Fall back to the
    # default, then to the whole file, so the view renders rather than crashing.
    _entry_fn = prov.get('host_entry_fn') or 'host_canonicalized'
    try:
        fstart, fend = find_function_range(host_lines, _entry_fn)
    except RuntimeError:
        try:
            fstart, fend = find_function_range(host_lines)
        except RuntimeError:
            fstart, fend = 0, len(host_lines) - 1
    owner, var2loc, _ = attribute_lines(host_lines, fstart, fend)

    # for-loop context (headers + per-iteration index math), computed once.
    loops = find_loops(host_lines, fstart, fend)

    # Rename-robust, filtered per-channel line mapping (comment/type anchored).
    channel_map = map_relevant_lines(host_lines, fstart, fend, prov, loops)

    supply_demand = compute_flow_balance(prov.get('flow_summary', []))
    balance_by_flow = {b['flow_index']: b for b in supply_demand}

    # SSA value/def maps for the per-channel full-block arg trace-back.
    cmap = build_const_map(host_lines, fstart, fend)
    var_def = build_var_def_map(host_lines, fstart, fend)

    ir_path = None if prov.get('provenance_source') == 'static-xaie' \
        else find_dfschedule_ir(workdir)
    slicer = None
    ir_text = None
    if ir_path:
        try:
            with open(ir_path) as f:
                ir_text = f.read()
            slicer = DfscheduleSlicer(ir_text)
        except Exception:
            slicer = None
            ir_text = None
    mid_fallback = ('(dfschedule IR dump not found)' if slicer is None
                    else '(no matching ops)')

    # per-owner 1-based line numbers
    owner_lines = {}
    for i in range(fstart, fend + 1):
        o = owner[i]
        owner_lines.setdefault(o, []).append(i + 1)

    # Full-block view: augment each tile owner's line set with the for-header +
    # index math of any loop that wraps one of its lines, so multi-tile loops
    # show the `for(...)` header for every participating tile, not just the first.
    for okey, lns in list(owner_lines.items()):
        if not isinstance(okey, tuple):        # skip '__global__'
            continue
        lnset = set(lns)
        extra = set()
        for lp in loops:
            body_lo, body_hi = lp['header'] + 2, lp['end'] + 1
            if any(body_lo <= x <= body_hi for x in lnset):
                extra.add(lp['header'] + 1)
                extra.update(p + 1 for p in lp['preamble'])
                extra.add(lp['end'] + 1)
        owner_lines[okey] = sorted(lnset | extra)

    # grid derivation from JSON tiles
    cols = sorted({t['col'] for t in prov['tiles']})
    rows = sorted({t['row'] for t in prov['tiles']})
    shim_rows = sorted({t['row'] for t in prov['tiles'] if t['type'] == 'shim'})
    core_rows = sorted({t['row'] for t in prov['tiles'] if t['type'] == 'core'})
    # Physical device core start row: for all known AIE generations (Gen2, Gen5)
    # rows 0=shim, 1-2=mem, 3+=core. Emit this so the device map can synthesize
    # phantom mem tiles only for the real mem rows (1-2), not schedule-derived ones.
    _AIE_DEVICE_CORE_MIN_ROW = 3

    def code_for(line_nums):
        ordered = sorted(line_nums)
        ranges = contiguous_ranges(ordered)
        chunks = []
        for a, b in ranges:
            chunks.append('\n'.join(host_lines[a - 1:b]))
        # Per-line records (raw text, preserving indentation) so the full-block
        # view can prefix each line with its host.cc line number.
        code_lines = [{'line': n, 'code': host_lines[n - 1]} for n in ordered]
        return ranges, '\n'.join(chunks), code_lines

    tiles_out = []
    for t in prov['tiles']:
        col, row, ttype = t['col'], t['row'], t['type']
        channels = t.get('dma_channels', [])
        lns = owner_lines.get((col, row), [])
        ranges, code, code_lines = code_for(lns) if lns else ([], '', [])

        # Filtered, rename-robust mapping: attach directly-implementing lines to
        # each channel and aggregate a per-tile "relevant only" view.
        line_kinds = {}
        line_bd = {}
        line_note = {}
        for c in channels:
            key = (col, row, c['direction'], c['channel'])
            entries = sorted(channel_map.get(key, []), key=lambda e: e['line'])
            c['host_lines'] = entries
            # Channel-scoped full block (contiguous ops + enclosing loop context
            # + arg-def trace-back lines), plus resolved-value params for inline
            # annotation in the "Full block" sub-tab.
            c['low_level'] = channel_full_block(host_lines, entries, cmap,
                                                var_def, var2loc)
            c['middle_ir'] = (slicer.channel_slice(col, row, c['direction'],
                                                    c['channel'])
                              if slicer else mid_fallback)
            fb = balance_by_flow.get(c.get('flow_index'))
            if fb is not None:
                c['flow_balance'] = {
                    'flow_index': fb['flow_index'],
                    'pattern': fb['pattern'],
                    'supply_per_round': fb['supply_per_round'],
                    'demand_per_round': fb['demand_per_round'],
                    'balanced': fb['balanced'],
                    'note': fb['note'],
                }
            for e in entries:
                line_kinds.setdefault(e['line'], set()).add(e['kind'])
                if e.get('bd_comment'):
                    line_bd[e['line']] = e['bd_comment']
                if e.get('note'):
                    line_note[e['line']] = e['note']
        relevant_lines = sorted(line_kinds)
        relevant_detail = []
        for n in relevant_lines:
            d = {'line': n, 'kinds': sorted(line_kinds[n]), 'code': host_lines[n - 1].strip()}
            if n in line_bd:
                d['bd_comment'] = line_bd[n]
            if n in line_note:
                d['note'] = line_note[n]
            relevant_detail.append(d)
        relevant_code = '\n'.join(
            'L%d: %s' % (n, host_lines[n - 1].strip()) for n in relevant_lines)

        t_kernel_view = _tile_kernel_view(t.get('kernel_cc')) or kernel_view
        t_bcf_view    = _tile_bcf_view(t.get('bcf')) or bcf_view
        t_win_base    = t.get('win_base', prov.get('win_base', WIN_BASE))

        # Kernel channel<->argument correlation (core tiles only). None for
        # shim/other tiles or when kernel.cc is absent.
        kernel_match = (match_channels_to_kernel(channels, t_kernel_view, t_bcf_view,
                                                 host_lines, var_def,
                                                 win_base=t_win_base)
                        if ttype == 'core' and t_kernel_view else None)

        tile_out = {
            'loc': [col, row],
            'type': ttype,
            'high_level': {
                'role': tile_role(ttype, channels),
                'kernel': kernel_for_tile(prov, col, row),
                'summary': build_summary(channels),
                'contracts': [c.get('contract') for c in channels if c.get('contract')],
                'kernel_match': kernel_match,
            },
            'low_level': {
                'ranges': ranges,
                'line_start': ranges[0][0] if ranges else None,
                'line_end': ranges[-1][1] if ranges else None,
                'code': code,
                'code_lines': code_lines,
            },
            'middle_ir': (slicer.tile_slice(col, row) if slicer
                          else mid_fallback),
            'relevant_lines': relevant_lines,
            'relevant_detail': relevant_detail,
            'relevant_code': relevant_code,
            'dma_channels': channels,
        }
        if ttype == 'core':
            tk = _tile_kernel_view(t.get('kernel_cc'))
            tb = _tile_bcf_view(t.get('bcf'))
            if tk is not None:
                tile_out['kernel'] = tk
            if tb is not None:
                tile_out['bcf'] = tb
        tiles_out.append(tile_out)

    host_mapped = any(t['relevant_lines'] or t['low_level']['ranges']
                      for t in tiles_out)
    kernel_code = any(t.get('kernel') or t.get('bcf') for t in tiles_out) \
        or bool(kernel_view) or bool(bcf_view)

    # global (non-tile) lines: kernel group load/launch, final wait, dbg snapshot
    glns = owner_lines.get('__global__', [])
    granges, gcode, _ = code_for(glns) if glns else ([], '', [])

    view = {
        'capabilities': {
            'host_lines': bool(host_mapped),
            'ir': slicer is not None,
            'kernel_code': bool(kernel_code),
        },
        'source': {
            'host_cc': os.path.abspath(host_path),
            'provenance': os.path.abspath(prov_path),
            'function': 'host_canonicalized',
            'function_lines': [fstart + 1, fend + 1],
        },
        'grid': {
            'cols': (max(cols) + 1) if cols else 0,
            'rows': (max(rows) + 1) if rows else 0,
            'col_list': cols,
            'row_list': rows,
            'shim_rows': shim_rows,
            'core_rows': core_rows,
            'device_core_min_row': _AIE_DEVICE_CORE_MIN_ROW,
            # Partition origin (absolute physical start column) from the
            # provenance map; phys_col = col + startcol. None = no partition.
            'startcol': prov.get('startcol'),
        },
        'module_attrs': prov.get('module_attrs', []),
        'buffers': [
            {'arg': 'v2', 'role': 'aux'},
            {'arg': 'v3', 'role': 'input (A)'},
            {'arg': 'v4', 'role': 'output (C)'},
        ],
        'global': {
            'ranges': granges,
            'code': gcode,
            'note': 'kernel group load/launch, outer wait, debug snapshot',
        },
        'tiles': tiles_out,
        'kernel': kernel_view,
        'bcf': bcf_view,
        'invariant_checks': prov.get('invariant_checks', []),
        'flow_summary': prov.get('flow_summary', []),
        'supply_demand': supply_demand,
        'comm_paths': _load_comm_paths(workdir),
        'dfschedule_ir': {
            'name': DFSCHED_IR_NAME,
            'path': os.path.abspath(ir_path) if ir_path else None,
            'text': ir_text or '',
        },
    }
    return view


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AIE Debug</title>
<style>
  /* ── design tokens ───────────────────────────────────────────── */
  :root {
    --shim: #2a5a90;
    --core: #2a5a3a;
    --tile-shim-fill: rgba(42, 90, 144, 0.11);
    --tile-shim-stroke: rgba(58, 104, 152, 0.36);
    --tile-core-fill: rgba(42, 90, 58, 0.11);
    --tile-core-stroke: rgba(56, 104, 72, 0.36);
    --tile-mem-fill: rgba(72, 64, 104, 0.12);
    --tile-mem-stroke: rgba(88, 88, 120, 0.36);
    --sel: #f0b840;
    /* backgrounds — three-level stack with clear contrast steps */
    --bg-base:    #282828;
    --bg-surface: #323232;
    --bg-raised:  #3e3e3e;
    --bg-code:    #1e1e1e;
    --bg-input:   #242424;
    --bg-hover:   #484848;
    /* borders */
    --bd:        #555555;
    --bd-soft:   #404040;
    --bd-subtle: #333333;
    --bd-accent: #3a5070;
    /* text */
    --tx-hi:   #f0f0f0;
    --tx-mid:  #a8a8a8;
    --tx-lo:   #686868;
    /* accent — electric blue */
    --accent:     #5ab0ff;
    --accent-dim: #2a3a48;
    --accent-fg:  #a0d0ff;
    /* status */
    --green-bg: #0c2018; --green-fg: #7aefa0;
    --red-bg:   #1e0c10; --red-fg:   #ff9090;
    --amber-fg: #f0c040;
    /* peer highlight colors */
    --peer-send-border: #d058c0;
    --peer-recv-border: #38d0e0;
    --right-pad: 16px;
  }

  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
         background:var(--bg-base); color:var(--tx-hi); display:flex; height:100vh;
         font-size:14px; line-height:1.45; }

  /* ── layout skeleton ─────────────────────────────────────────── */
  #left { flex:0 0 50%; min-width:200px; display:flex; flex-direction:column;
          overflow:hidden; border-right:1px solid var(--bd); }
  #lefttop { flex:3 1 0; overflow:auto; padding:16px; min-height:80px; }
  #leftbottom { flex:1 1 0; overflow:hidden; min-height:60px;
               display:flex; flex-direction:column; }
  #right { flex:1 1 0; min-width:200px; padding:var(--right-pad); overflow:hidden;
           display:flex; flex-direction:column; }
  #panel { flex:1 1 0; min-height:0; display:flex; flex-direction:column; position:relative; }
  #panel-body { flex:1 1 0; overflow-y:auto; min-height:0; }
  #panel-toc { position:absolute; right:0; top:0; width:112px; overflow-y:auto;
               max-height:100%; background:rgba(28,28,28,0.40);
               border-left:1px solid rgba(255,255,255,0.08);
               padding:0; z-index:5; transition:width 0.15s ease; }
  #panel-toc.no-items { display:none; }
  #panel-toc.collapsed { width:22px; overflow:hidden; }
  #panel-toc.collapsed .ptoc-item { display:none; }
  .ptoc-toggle { display:block; width:100%; padding:4px 0; text-align:center;
                 font-size:14px; color:var(--tx); background:none; border:none;
                 border-bottom:1px solid rgba(255,255,255,0.08); cursor:pointer; line-height:1; }
  .ptoc-toggle:hover { color:var(--accent); }
  .ptoc-top { display:block; width:100%; padding:3px 0; text-align:center;
              font-size:13px; color:var(--tx); background:none; border:none;
              border-bottom:1px solid rgba(255,255,255,0.06); cursor:pointer; line-height:1; }
  .ptoc-top:hover { color:var(--tx-hi); }
  .ptoc-item { display:block; padding:2px 8px; font-size:10px; color:var(--tx);
               cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
               line-height:1.5; }
  .ptoc-item:hover { color:var(--tx-hi); background:rgba(255,255,255,0.06); }
  /* Title + item tabs share one row, and it sits OUTSIDE #panel: the strip used
     to scroll away with the body, and hoisting it costs no extra height. */
  #panel-hdr { flex:0 0 auto; display:flex; align-items:baseline; flex-wrap:wrap;
               gap:8px; padding:0 0 6px; margin-bottom:6px;
               border-bottom:1px solid var(--bd-soft); }
  /* ── panel item tabs (multi-select net/tile strip) ─────────── */
  #panel-itemtabs { display:flex; flex-wrap:wrap; gap:4px; }
  #panel-itemtabs:empty { display:none; }
  .pitab { display:inline-flex; align-items:center; gap:5px; padding:2px 8px 2px 6px;
           border-radius:9999px; font-size:10px; font-weight:500; cursor:pointer;
           border:1px solid var(--bd); background:var(--bg-raised);
           color:var(--tx-lo); user-select:none; transition:opacity .12s, border-color .12s; }
  .pitab.act { border-color:rgba(228,228,228,.5); color:var(--tx-hi);
               background:rgba(228,228,228,.07); }
  .pitab:hover:not(.act) { color:var(--tx-mid); }
  .pitab .pit-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
  .pitab .pit-x { margin-left:3px; font-size:12px; line-height:1; color:var(--tx-lo);
                  cursor:pointer; }
  .pitab .pit-x:hover { color:var(--tx-hi); }

  /* ── splitters ───────────────────────────────────────────────── */
  #splitter { flex:0 0 5px; cursor:col-resize; background:var(--bg-base);
              border-left:1px solid var(--bd); border-right:1px solid var(--bd);
              transition:background .15s; position:relative; }
  #splitter:hover, #splitter.drag { background:var(--accent-dim); border-color:var(--accent); }
  #splitter::after { content:''; position:absolute; top:50%; left:50%;
                     transform:translate(-50%,-50%);
                     width:1px; height:20px; background:var(--bd); border-radius:1px; }
  #splitter:hover::after, #splitter.drag::after { background:var(--accent); }

  #lhsplitter { flex:0 0 5px; cursor:row-resize; background:var(--bg-base);
                border-top:1px solid var(--bd); border-bottom:1px solid var(--bd);
                transition:background .15s; position:relative; }
  #lhsplitter:hover, #lhsplitter.drag { background:var(--accent-dim); border-color:var(--accent); }
  #lhsplitter::after { content:''; position:absolute; left:50%; top:50%;
                       transform:translate(-50%,-50%);
                       height:1px; width:20px; background:var(--bd); border-radius:1px; }
  #lhsplitter:hover::after, #lhsplitter.drag::after { background:var(--accent); }

  #rhsplitter { display:none; flex:0 0 5px; cursor:row-resize; background:var(--bg-base);
                margin:8px calc(-1 * var(--right-pad)) 0;
                border-top:1px solid var(--bd); border-bottom:1px solid var(--bd);
                transition:background .15s; position:relative; }
  #right:has(#cmdconsole:not(.hide)) #rhsplitter { display:block; }
  #rhsplitter:hover, #rhsplitter.drag { background:var(--accent-dim); border-color:var(--accent); }

  body.resizing { cursor:col-resize; user-select:none; }
  body.vresizing { cursor:row-resize; user-select:none; }

  /* ── typography ──────────────────────────────────────────────── */
  #lefttop-header { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  h1 { font-size:17px; font-weight:700; margin:0; letter-spacing:-.01em;
       transform:scale(1.118); transform-origin:left center;
       margin-right:calc((1.118 - 1) * 10ch); }
  /* One name per pane. Each rides the pane's existing first row — a control
     row or a tab strip — rather than taking a row of its own. Scale up visually
     without changing the flex/inline footprint (font-size drives layout box);
     margin-right absorbs the scaled overflow to the right. */
  .pane-title { flex:0 0 auto; font-size:13px; font-weight:700; margin:0;
                color:var(--tx-hi); letter-spacing:.01em; white-space:nowrap;
                transform:scale(1.154); transform-origin:left center;
                margin-right:calc((1.154 - 1) * 9ch); }
  #contabs > .pane-title { display:inline-block; vertical-align:bottom;
                           padding-bottom:3px;
                           margin-right:calc(8px + (1.154 - 1) * 5ch);
                           transform-origin:left bottom; }
  .sub { color:var(--tx-lo); font-size:11px; letter-spacing:.01em; }
  .panel h2 { font-size:12px; font-weight:600; margin:16px 0 6px;
              padding-bottom:5px; border-bottom:1px solid var(--bd-soft);
              color:var(--tx-mid); text-transform:uppercase; letter-spacing:.06em; }

  /* ── tile grid ───────────────────────────────────────────────── */
  #grid { display:grid; gap:5px; }
  .tile { border:1px solid var(--bd); border-radius:5px; padding:8px 9px;
          cursor:pointer; font-size:11px; line-height:1.35;
          transition:box-shadow .12s, border-color .12s; min-height:60px;
          position:relative; user-select:none; }
  .tile.shim { background:var(--shim); border-color:#3a6898; }
  .tile.core { background:var(--core); border-color:#386848; }
  .tile:hover { border-color:var(--tx-lo); box-shadow:0 0 0 1px var(--tx-lo) inset; }
  /* selected tile: inner ring glow — the "precision instrument" signature */
  .tile.sel { border-color:var(--sel);
              box-shadow:0 0 0 1px var(--sel) inset, 0 0 14px 0 rgba(240,184,64,.25); }
  .tile.sdmismatch { border-color:#e04040;
                     box-shadow:0 0 0 1px #e04040 inset, 0 0 12px 0 rgba(224,64,64,.22); }
  .tile .badge.sdwarn { background:var(--red-bg); color:var(--red-fg); cursor:default; }
  .tile .loc { font-weight:700; font-size:11.5px; color:var(--tx-hi); }
  .tile .badge { display:inline-block; background:rgba(0,0,0,.32);
                 border:1px solid rgba(255,255,255,.10);
                 border-radius:3px; padding:0 4px; margin:2px 2px 0 0;
                 font-size:10px; font-family:ui-monospace,monospace; cursor:pointer; }
  .tile .badge:hover { border-color:rgba(255,255,255,.35); }
  .tile .badge.selbadge { border-color:var(--sel); background:rgba(0,0,0,.45);
                           box-shadow:0 0 4px 0 rgba(232,168,64,.35); }
  .tile .badge.peerbadge-partner { border-color:#e8943a; border-style:dotted; background:rgba(0,0,0,.45); }
  .tile .badge.peerbadge-coop    { border-color:#6cd080; border-style:dashed; background:rgba(0,0,0,.45); }
  .tile.peer-recv { border-color:var(--peer-recv-border);
                    box-shadow:0 0 0 1px var(--peer-recv-border) inset,
                               0 0 12px 0 rgba(56,208,224,.25); }
  .tile.peer-send { border-color:var(--peer-send-border); border-style:dashed;
                    box-shadow:0 0 12px 0 rgba(208,88,192,.25); }
  .tile.flowsel   { box-shadow:0 0 0 1px rgba(255,255,255,.25) inset; }
  .livebar { display:block; margin-top:4px; font-size:9px; font-weight:700; color:#fff;
             border-radius:2px; padding:0 4px; text-align:center; letter-spacing:.5px; }

  /* ── legend ──────────────────────────────────────────────────── */
  .legend { margin-top:14px; font-size:11px; color:var(--tx-mid); }
  .legend .sw { display:inline-block; width:11px; height:11px; border-radius:2px;
                vertical-align:-2px; margin-right:4px; border:1px solid rgba(255,255,255,.1); }

  /* ── key-value pairs ─────────────────────────────────────────── */
  .kv { font-size:12px; margin:3px 0; }
  .kv b { color:var(--accent-fg); font-weight:500; }
  ul.sum { margin:4px 0; padding-left:18px; font-size:12px; }
  ul.sum li { margin:2px 0; }
  .contract { color:#c9a; font-size:11.5px; }
  .contract .ct-pktid { display:inline-block; font-size:9px; color:#c07fd4;
                        background:#1a0d24; border-radius:2px; padding:0 3px;
                        margin-left:3px; vertical-align:middle; }

  /* ── supply/demand badges ────────────────────────────────────── */
  .sdrow { margin:4px 0 7px; }
  .sdbadge { display:inline-block; font-size:10px; font-weight:700; padding:1px 6px;
             border-radius:3px; margin-right:6px; vertical-align:1px;
             letter-spacing:.02em; }
  .sdbadge.sdok  { background:var(--green-bg); color:var(--green-fg);
                   border:1px solid #205a30; }
  .sdbadge.sdbad { background:var(--red-bg); color:var(--red-fg);
                   border:1px solid #582020; }
  .sdbadge.sdna  { background:var(--bg-raised); color:var(--tx-mid);
                   border:1px solid var(--bd); }
  .sdmeta, .sdnote { color:var(--tx-lo); font-size:11px; }
  .sdfig  { color:var(--tx-mid); font-size:12px; margin:2px 0 0 2px; }
  .sdnote { font-style:italic; margin-left:2px; }

  /* ── code / pre blocks ───────────────────────────────────────── */
  pre.code { background:var(--bg-code); border:1px solid var(--bd-subtle);
             border-radius:5px; padding:10px; overflow:auto;
             font-size:11.5px; line-height:1.45; white-space:pre;
             font-family:ui-monospace, 'Cascadia Code', monospace; }
  .lref { color:var(--tx-lo); font-size:11px; margin:4px 0; }

  /* ── source viewer ───────────────────────────────────────────── */
  .srcref { cursor:pointer; }
  .srcref:hover { text-decoration:underline dotted; }
  .con-ln.con-src { cursor:pointer; }
  .srcwrap { overflow:auto; border:1px solid var(--bd-subtle); border-radius:4px;
             padding:6px 0; background:var(--bg-code); }
  /* One id of specificity, so these beat the style sheet the daemon injects
     regardless of which <style> the browser saw last. */
  #panel-body .srcview { background:var(--bg-code); margin:0;
                         font-family:ui-monospace,'Cascadia Code',monospace;
                         font-size:11.5px; line-height:1.55; }
  #panel-body .srcview pre { margin:0; padding:0; border:none; background:none;
                             line-height:inherit; white-space:pre; }
  /* user-select:none mirrors .lno, so copying out of the viewer (which feeds
     "+ Add context") yields code without line numbers. */
  #panel-body .srcview .linenos { display:inline-block; min-width:5ch;
                                  text-align:right; margin-right:12px;
                                  color:var(--tx-lo); user-select:none; }
  /* inset shadow, not border-left: a border would shift the highlighted line
     horizontally relative to its neighbours inside the <pre>. */
  #panel-body .srcview .hlline { display:block; background:#26240e;
                                 box-shadow:inset 3px 0 0 var(--amber-fg); }
  .srcbanner .srcmeta { color:var(--tx-lo); margin-left:8px; font-weight:400; }
  .srcbanner .srcmore { font-size:10px; padding:1px 7px; margin-left:4px;
                        border-radius:3px; cursor:pointer; }
  .srcnote { color:#f6c177; font-size:11px; margin:4px 0; }

  /* context echoed into the transcript after sending */
  .llm-msg-sctx { background:transparent; border-left:2px solid var(--bd-subtle);
                  padding:3px 9px; }
  .sctx-lead { color:var(--tx-lo); font-size:10.5px; margin-right:6px; }
  .llm-sent-pill { display:inline-block; margin:1px 4px 1px 0; padding:0 7px;
                   border-radius:9999px; font-size:10.5px; cursor:pointer;
                   background:var(--bg-raised); color:var(--tx-mid);
                   border:1px solid var(--bd-subtle); }
  .llm-sent-pill:hover { color:var(--tx-hi); border-color:var(--bd); }
  .llm-sent-pill.act { border-color:var(--accent); color:var(--accent-fg); }
  .sctx-body { margin:4px 0 2px; padding:5px 8px; max-height:220px; overflow:auto;
               background:var(--bg-code); border:1px solid var(--bd-subtle);
               border-radius:4px; font-size:11px; white-space:pre-wrap; }
  .codepath { color:var(--accent-fg); font-size:11px; margin:2px 0 10px;
              word-break:break-all; padding:5px 9px;
              background:var(--accent-dim); border:1px solid var(--bd-accent); border-radius:4px; }
  .codepath b { color:var(--accent); font-weight:600; }
  .codepath .cpath { color:#b0d8f8; -webkit-user-select:all; user-select:all; }
  .placeholder { color:var(--tx-lo); margin-top:40px; font-size:12px; }

  /* ── unified button ──────────────────────────────────────────── */
  button, .btn {
    padding:3px 11px; cursor:pointer; font-size:12px;
    background:var(--bg-raised); color:var(--tx-mid);
    border:1px solid var(--bd); border-radius:4px;
    transition:background .12s, border-color .12s, color .12s;
    font-family:inherit;
  }
  button:hover, .btn:hover { background:var(--bg-hover); color:var(--tx-hi); border-color:var(--bd); }
  button:disabled, .btn:disabled { opacity:.38; cursor:not-allowed; }
  #runbtn { font-weight:600; }
  #ctrlbar[data-connected="true"] #runbtn,
  #ctrlbar:not([data-connected="true"]) #testconn {
    background:var(--accent); color:#101820; border-color:var(--accent);
  }
  #ctrlbar[data-connected="true"] #runbtn:hover,
  #ctrlbar:not([data-connected="true"]) #testconn:hover {
    filter:brightness(1.12);
  }
  #stopbtn { background:#2a1516; color:#f8b0b0; border-color:#5c2828; }
  #stopbtn:hover { background:#3a1e1e; border-color:#7a3535; }

  /* ── unified input ───────────────────────────────────────────── */
  input[type=text], input[type=password], .ui-input, #boardHost, #dbgcmd {
    background:var(--bg-input); color:var(--tx-hi);
    border:1px solid var(--bd); border-radius:4px;
    padding:3px 7px; font-family:ui-monospace,monospace; font-size:12px;
    outline:none; transition:border-color .15s;
  }
  input[type=text]:focus, input[type=password]:focus, .ui-input:focus,
  #boardHost:focus, #dbgcmd:focus { border-color:var(--accent); }
  #dbgcmd { padding-top:4px; }

  /* ── unified tab system ──────────────────────────────────────── */
  /* One accent identity for every tab family: inactive tabs are raised-grey,
     hover lifts them, and the selected one takes the blue accent. Shape still
     varies (pill / folder / segmented) because it encodes what the strip does,
     but the *color* language is shared so "selected" reads the same anywhere.
     Note: .pitab (the net/tile chip strip) is deliberately NOT part of this —
     its chips carry per-item identity dots and stay neutral. */
  .ltab.act, .vsw.act, .tab.act, .contab.act, .subtab.act {
    background:var(--accent-dim); color:var(--accent-fg); border-color:var(--bd-accent); }
  .ltab:hover:not(.act), .vsw:hover:not(.act), .tab:hover:not(.act),
  .contab:hover:not(.act), .subtab:hover:not(.act) {
    color:var(--tx-mid); background:var(--bg-hover); }

  /* Pill tabs (.ltab, .vsw overlay tabs) */
  .ltab { display:inline-block; padding:2px 9px; margin:5px 3px 0 0;
          border:1px solid var(--bd); border-radius:4px;
          background:var(--bg-raised); color:var(--tx-lo);
          cursor:pointer; font-size:11px; transition:background .12s, color .12s, border-color .12s; }

  /* Folder tabs (.tab, .contab — top-edge tabs that connect to a content border) */
  .tabs { margin-top:10px; }
  .tab { display:inline-block; padding:4px 11px; border:1px solid var(--bd);
         border-bottom:none; cursor:pointer; background:var(--bg-raised);
         color:var(--tx-lo); border-radius:5px 5px 0 0; font-size:12px;
         margin-right:3px; transition:color .12s, background .12s, border-color .12s; }
  /* The active folder tab no longer matches .tabbody's background, so give the
     body an accent top edge — that line is what keeps the tab reading as
     attached to the box below it now that the two surfaces differ. */
  .tabbody { border:1px solid var(--bd); border-top-color:var(--bd-accent); padding:10px;
             border-radius:0 5px 5px 5px; background:var(--bg-code); }

  /* Subtabs (.subtab — compact secondary tabs) */
  .subtabs { margin:3px 0 9px; }
  .subtab { display:inline-block; padding:2px 9px; border:1px solid var(--bd);
            cursor:pointer; background:var(--bg-raised); color:var(--tx-lo);
            border-radius:4px; margin-right:5px; font-size:11.5px;
            transition:background .12s, color .12s, border-color .12s; }
  .codefile-tabs { display:flex; gap:5px; margin:0 0 10px; padding-bottom:2px;
                   overflow-x:auto; scrollbar-width:thin; }
  .codefile-tabs .subtab { flex:0 0 auto; margin-right:0; white-space:nowrap;
                           font-family:ui-monospace,'Cascadia Code',monospace;
                           font-size:10.5px; }
  .codefile-tabs .subtab:first-child { font-family:inherit; }
  .codefile-view { min-width:0; }

  /* Segmented control (.vsw — Grid/Device Map switcher) */
  #viewswitcher { display:flex; gap:0; }
  .vsw { padding:3px 14px; font-size:12px; font-weight:500;
         border:1px solid var(--bd); background:var(--bg-raised);
         color:var(--tx-lo); cursor:pointer; transition:background .12s, color .12s, border-color .12s; }
  .vsw:first-child { border-radius:4px 0 0 4px; }
  .vsw:last-child  { border-radius:0 4px 4px 0; }
  .vsw + .vsw { border-left:none; }
  /* Segments share borders, so an active segment would show accent on only
     three sides; paint the left edge of the following segment to close it. */
  .vsw.act + .vsw { border-left:1px solid var(--bd-accent); }

  /* Profile tab is inert until a run renders a timeline.png. */
  .vsw-disabled { opacity:.4; pointer-events:none; cursor:default; }

  /* ── Profile panel ───────────────────────────────────────────── */
  #profileview { display:none; flex-direction:column; gap:8px; }
  #profileview.show { display:flex; }
  #profile-status { font-size:12px; color:var(--tx-mid); }
  #profile-img { max-width:100%; height:auto; border:1px solid var(--bd);
                 border-radius:4px; background:#fff; }

  /* ── Targets panel ───────────────────────────────────────────── */
  #targetsview { display:none; flex-direction:column; gap:0; }
  #targetsview.show { display:flex; }
  #tgt-toolbar { display:flex; align-items:center; gap:8px; margin-bottom:10px; flex-wrap:wrap; }
  #tgtRefreshBtn { font-size:11px; padding:3px 13px; border-radius:4px; cursor:pointer;
                   border:1px solid var(--accent); background:var(--accent);
                   color:var(--bg-code); font-weight:600; letter-spacing:.02em;
                   transition:filter .12s; }
  #tgtRefreshBtn:hover { filter:brightness(1.12); }
  #tgtRefreshBtn:active { filter:brightness(.92); }
  #tgt-status { font-size:11.5px; color:var(--tx-lo); }
  #tgt-list { flex:1 1 auto; overflow:auto; }

  /* Card: left-border is the primary state color so you can skim at a glance */
  .tgt-card { border:1px solid var(--bd); border-left-width:4px; border-radius:6px;
              margin-bottom:8px; background:var(--bg-raised); overflow:hidden; }
  .tgt-card.tc-running   { border-left-color:#3fb950; }
  .tgt-card.tc-suspended { border-left-color:#d29922; }
  .tgt-card.tc-crashed   { border-left-color:#f85149; }
  .tgt-card.tc-noelf     { border-left-color:#484f58; }
  .tgt-card.tc-other     { border-left-color:var(--bd); }
  .tgt-card.tgt-sel      { outline:2px solid var(--accent); outline-offset:-2px; }

  /* Header row: always visible */
  .tgt-hdr { display:flex; align-items:center; gap:8px; padding:7px 10px 6px;
             background:var(--bg-raised); }

  /* State badge (pill) */
  .tgt-badge { font-size:10px; font-weight:700; padding:2px 8px; border-radius:10px;
               letter-spacing:.04em; white-space:nowrap; flex-shrink:0; }
  .tgt-badge.running   { background:#1a3c1e; color:#3fb950; }
  .tgt-badge.suspended { background:#3b2800; color:#d29922; }
  .tgt-badge.crashed   { background:#3b0a0a; color:#f85149; }
  .tgt-badge.noelf     { background:var(--bg-code); color:#484f58; }
  .tgt-badge.other     { background:var(--bg-code); color:var(--tx-lo); }

  /* Name + id */
  .tgt-name { font-size:12.5px; font-weight:600; flex:1 1 auto; overflow:hidden;
              text-overflow:ellipsis; white-space:nowrap; }
  .tgt-id   { font-size:11px; color:var(--tx-lo); opacity:.55; flex-shrink:0; }

  /* PC row: monospace line below the header, visible whenever PC is known */
  .tgt-pcrow { display:flex; align-items:baseline; gap:8px;
               padding:0 10px 5px; font-family:ui-monospace,monospace; font-size:11px; }
  .tgt-pclabel { color:var(--tx-lo); flex-shrink:0; }
  .tgt-pcval   { color:#a8d8ff; flex-shrink:0; }
  .tgt-pcsrc   { color:var(--accent-fg); overflow:hidden; text-overflow:ellipsis;
                 white-space:nowrap; cursor:pointer; }
  .tgt-pcsrc:hover { text-decoration:underline dotted; }

  /* Stack trace: collapsible <details> below the PC row */
  .tgt-btdetails { padding:0; border-top:1px solid var(--bd); }
  .tgt-btsummary { font-size:11px; color:var(--tx-lo); padding:4px 10px;
                   cursor:pointer; user-select:none; list-style:none; }
  .tgt-btsummary::-webkit-details-marker { display:none; }
  .tgt-btsummary::before { content:'▶'; font-size:9px; margin-right:5px;
                            display:inline-block; transition:transform .15s; }
  .tgt-btdetails[open] .tgt-btsummary::before { transform:rotate(90deg); }
  .tgt-stack { font-family:ui-monospace,monospace; font-size:11px; width:100%;
               border-collapse:collapse; }
  .tgt-stack tr { border-bottom:1px solid var(--bd); }
  .tgt-stack tr:last-child { border-bottom:none; }
  .tgt-stack td { padding:3px 8px; vertical-align:top; }
  .tgt-stack .sf-frame { color:var(--tx-lo); width:2ch; text-align:right; padding-right:8px;
                         font-size:10px; }
  .tgt-stack .sf-addr  { color:#6e8fa8; width:9ch; white-space:nowrap; }
  .tgt-stack .sf-func  { font-weight:500; color:var(--tx); }
  /* top frame gets accent color so it stands out */
  .tgt-stack tr:first-child .sf-func { color:var(--accent-fg); }
  .tgt-stack .sf-loc   { color:var(--tx-lo); cursor:pointer; }
  .tgt-stack .sf-loc:hover { color:var(--accent-fg); text-decoration:underline dotted; }
  .tgt-stack .sf-raw   { color:var(--tx-lo); font-style:italic; }

  .tgt-elfdot { font-size:10px; flex-shrink:0; margin-left:2px; }
  .tgt-elfdot.sym   { color:#3fb950; }
  .tgt-elfdot.nosym { color:#d29922; }
  .tgt-elfdot.none  { color:#484f58; }
  .tgt-nonote { padding:4px 10px 8px; font-size:11px; color:var(--tx-lo); font-style:italic; }

  .tgt-haltrow { display:flex; align-items:center; gap:8px; padding:4px 10px 6px; }
  .tgt-haltbtn { font-size:11px; padding:3px 10px; border-radius:4px; cursor:pointer;
                 background:#3b2800; color:#d29922; border:1px solid #5c4200;
                 transition:filter .12s; }
  .tgt-haltbtn:hover { filter:brightness(1.2); }
  .tgt-haltbtn:active { filter:brightness(.85); }
  .tgt-haltbtn:disabled { opacity:.45; cursor:default; filter:none; }
  .tgt-haltnote { font-size:10px; color:var(--tx-lo); }

  .tgt-empty { color:var(--tx-lo); font-size:12px; padding:24px 0; text-align:center; }

  /* Cluster (APU/RPU group) headers */
  .tgt-cluster { margin-bottom:4px; }
  .tgt-cluster-hdr { display:flex; align-items:center; gap:6px; padding:4px 6px 4px 8px;
                     border-radius:4px; background:var(--bg-raised); cursor:pointer;
                     user-select:none; }
  .tgt-cluster-hdr:hover { background:var(--bg-hover,var(--bg-raised)); filter:brightness(1.1); }
  .tgt-cluster-arrow { font-size:9px; color:var(--tx-lo); transition:transform .15s;
                       flex-shrink:0; display:inline-block; }
  details.tgt-cluster[open] > summary .tgt-cluster-arrow { transform:rotate(90deg); }
  .tgt-cluster-name { font-size:12px; font-weight:600; color:var(--tx); flex:1 1 auto; }
  .tgt-cluster-badge { font-size:10px; padding:1px 7px; border-radius:10px;
                       background:var(--bg-code); color:var(--tx-lo); flex-shrink:0; }
  .tgt-cluster-badge.cl-running { background:#1a3c1e; color:#3fb950; }
  .tgt-cluster-body { padding-left:10px; padding-top:3px; display:flex;
                      flex-direction:column; gap:4px; }
  details.tgt-cluster:not([open]) > .tgt-cluster-body { display:none; }
  details.tgt-cluster > summary { list-style:none; }
  details.tgt-cluster > summary::-webkit-details-marker { display:none; }

  /* Console header tabs (#contabs) — folder style */
  #contabs { flex:0 0 auto; margin-bottom:6px; }
  .contab { display:inline-block; padding:3px 12px; border:1px solid var(--bd);
            border-bottom:none; cursor:pointer; background:var(--bg-raised);
            color:var(--tx-lo); border-radius:5px 5px 0 0; font-size:12px; margin-right:3px;
            transition:color .12s, background .12s, border-color .12s; }

  .hide { display:none; }

  /* ── console / terminal ──────────────────────────────────────── */
  #cmdconsole { flex:0 0 260px; border-top:1px solid var(--bd);
                margin:8px calc(-1 * var(--right-pad)) 0;
                padding:10px var(--right-pad) 0; display:flex; flex-direction:column;
                min-height:0; overflow:hidden; }
  #conhdr, #conhelp { flex:0 0 auto; }
  #conhdr { font-weight:600; margin-bottom:4px; font-size:12px; }
  #llmpane #conhdr { display:flex; align-items:center; gap:0; }
  .llm-hint { font-size:11px; font-weight:400; color:var(--tx-lo); margin-left:10px; }
  #contarget { color:var(--accent-fg); }
  #conhelp { display:flex; flex-wrap:wrap; align-items:center; gap:2px;
             margin-bottom:6px; }
  .qcmd-sep { display:inline-block; width:1px; height:14px; background:var(--bd);
              margin:0 4px; vertical-align:middle; flex-shrink:0; }
  #conreload { margin-left:8px; }
  /* aiegdb header: name, live scope and every action on ONE row, reclaiming the
     vertical strip the separate #conhelp button row used to take. Scoped by
     class because #conhdr/#conhelp are reused by the LLM and Search panes. */
  #conhdr.conhdr-row { display:flex; align-items:center; gap:4px;
                       flex-wrap:nowrap; margin-bottom:5px; overflow:hidden; }
  .conhdr-row .chname { flex:0 0 auto; }
  /* the scope breadcrumb is the only elastic item — it ellipsizes so the
     buttons never wrap to a second row */
  .conhdr-row #contarget { flex:0 1 auto; min-width:0; font-size:11px;
                           font-weight:400; white-space:nowrap;
                           overflow:hidden; text-overflow:ellipsis; }
  .conhdr-row .chgap { flex:1 1 auto; min-width:4px; }
  .conhdr-row .qcmd { flex:0 0 auto; }
  .conhdr-row #conreload { margin-left:0; }
  #conterm { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             margin-top:6px; padding:7px; background:var(--bg-code);
             border:1px solid var(--bd-subtle); border-radius:5px;
             overflow:hidden; cursor:text; }
  #conout { flex:1 1 auto; min-height:0; overflow:auto; margin:0; background:transparent; }
  #conpromptline { flex:0 0 auto; display:flex; align-items:center;
                   margin-bottom:5px; padding-bottom:5px;
                   border-bottom:1px solid var(--bd-subtle); }
  #conprompt { color:var(--accent-fg); margin-right:6px; white-space:nowrap;
               font-family:ui-monospace,monospace; font-size:11.5px; }
  #conin { flex:1 1 auto; background:transparent; border:none; outline:none;
           color:var(--tx-hi); font-family:ui-monospace,monospace;
           font-size:11.5px; padding:0; }
  #conin.disabled, #conin:disabled { opacity:.4; cursor:not-allowed; }
  #conpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0; }
  #llmpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             position:relative; }
  #conpane.hide, #llmpane.hide { display:none; }

  /* ── aiegdb console: per-command blocks ──────────────────────────
     #conout was a <pre class="code">; it is a <div> now so each command can be
     its own foldable block, so the typography pre.code used to supply is
     re-declared here. pre-wrap on .con-ln keeps aiegdb's leading indentation,
     which is load-bearing both for reading and for the line classifier. */
  #conout { flex:1 1 auto; min-height:0; overflow:auto; margin:0;
            background:transparent; font-family:ui-monospace,monospace;
            font-size:11.5px; line-height:1.5; color:var(--tx-mid); }
  .con-blk { border:1px solid var(--bd-subtle); border-radius:4px;
             margin:0 0 5px; background:#00000026; }
  .con-blk.err { border-color:#5a2a2a; }
  .con-blk.cur { border-color:var(--accent-fg); }
  .con-blk.cur.err { border-color:#8a3a3a; }
  .con-blk.cur > .con-echo { background:var(--bg-hover); cursor:default; }
  .con-blk.cur > .con-echo .cf { display:none; }
  .con-echo { position:sticky; top:0; z-index:2;
              display:flex; align-items:center; gap:6px; cursor:pointer;
              padding:2px 7px; background:var(--bg-surface);
              border-radius:3px 3px 0 0;
              border-bottom:1px solid var(--bd-subtle); user-select:none; }
  .con-echo:hover { background:var(--bg-hover); }
  .con-echo .cf { color:var(--tx-lo); width:9px; flex:0 0 9px; }
  .con-echo .cs { color:var(--tx-lo); }
  .con-echo .cc { color:var(--accent-fg); font-weight:600; }
  .con-echo .cn { margin-left:auto; color:var(--tx-lo); font-size:10px; }
  .con-blk.fold .con-body { display:none; }
  .con-body { padding:3px 7px 4px; }
  .con-ln { white-space:pre-wrap; word-break:break-word; }
  /* line kinds — mapped from aiegdb/aiediag output shapes */
  .con-hdr   { color:var(--tx-hi); font-weight:600; }
  .con-key   { color:var(--tx-lo); }
  .con-val   { color:var(--tx-hi); }
  .con-hex   { color:#c8a0f0; }
  .con-ok    { color:var(--green-fg); }
  .con-bad   { color:var(--red-fg); }
  .con-warn  { color:var(--amber-fg); }
  .con-scope { color:var(--tx-hi); }
  .con-dim   { color:var(--tx-lo); }
  .con-src   { color:#7fd0c0; }
  /* the "[registers read] { … }" appendix aiegdb adds after most commands —
     the single biggest source of crowding, so it folds away by default */
  .con-reg { margin:3px 0 0; border-left:2px solid var(--bd-soft);
             padding-left:6px; }
  .con-reg > summary { cursor:pointer; color:var(--tx-lo); font-size:10.5px;
                       list-style:none; user-select:none; }
  .con-reg > summary:hover { color:var(--tx-mid); }
  .con-reg[open] > summary { color:var(--tx-mid); }

  /* ── aiegdb console: autocomplete popup ──────────────────────── */
  /* position:fixed, not absolute: #conterm sets overflow:hidden and the prompt
     line is its FIRST child, so an absolutely-positioned popup gets clipped away
     entirely. Fixed + viewport coords from getBoundingClientRect escapes every
     overflow container; conSugPlace() sets left/top/width. */
  #consug { position:fixed; z-index:60; max-height:190px; overflow:auto;
            background:var(--bg-raised); border:1px solid var(--bd);
            border-radius:5px; box-shadow:0 6px 18px #00000070; }
  #consug.hide { display:none; }
  .con-sug { display:flex; align-items:baseline; gap:7px; padding:3px 8px;
             cursor:pointer; font-family:ui-monospace,monospace;
             font-size:11px; }
  .con-sug.act { background:var(--accent-dim); }
  .con-sug:hover { background:var(--bg-hover); }
  .con-sug .sname { color:var(--tx-hi); font-weight:600; white-space:nowrap; }
  .con-sug .sargs { color:var(--amber-fg); white-space:nowrap; }
  .con-sug .ssum  { color:var(--tx-lo); margin-left:auto; font-size:10px;
                    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .con-sug .swrite { color:var(--red-fg); font-size:9.5px; border:1px solid #5a2a2a;
                     border-radius:3px; padding:0 3px; white-space:nowrap; }
  /* dimmed + struck-through: these cannot run here at all (live TUI grids) */
  .con-sug.blocked .sname { color:var(--tx-lo); text-decoration:line-through; }
  .con-sug .sblock { color:var(--tx-lo); font-size:9.5px; border:1px solid var(--bd-soft);
                     border-radius:3px; padding:0 3px; white-space:nowrap; }
  .con-sug .sslow  { color:var(--amber-fg); font-size:9.5px; border:1px solid #5a4a1a;
                     border-radius:3px; padding:0 3px; white-space:nowrap; }
  .con-sug .sscope { color:var(--tx-lo); font-size:9.5px; opacity:.75; }
  /* an argument description, not a completion — nothing to click */
  .con-sug.hintrow { cursor:default; background:none; }
  .con-sug.hintrow:hover { background:none; }
  .con-sug.hintrow .sname { color:var(--amber-fg); font-weight:400; }
  .con-sug.hintrow .ssum { color:var(--tx-mid); margin-left:0; }

  /* ── aiegdb console: ⌘ Commands palette ──────────────────────── */
  /* fixed for the same reason as #consug — conPalOpen() computes viewport coords */
  #conpal { position:fixed; z-index:60; width:340px; max-height:280px;
            display:flex; flex-direction:column;
            background:var(--bg-raised); border:1px solid var(--bd);
            border-radius:6px; box-shadow:0 8px 24px #00000080; }
  #conpal.hide { display:none; }
  #conpalq { flex:0 0 auto; margin:6px; padding:4px 7px;
             background:var(--bg-input); border:1px solid var(--bd-soft);
             border-radius:4px; color:var(--tx-hi);
             font-family:ui-monospace,monospace; font-size:11.5px; outline:none; }
  #conpalq:focus { border-color:var(--accent); }
  #conpallist { flex:1 1 auto; overflow:auto; padding:0 4px 5px; }
  .con-palgrp { color:var(--tx-lo); font-size:9.5px; text-transform:uppercase;
                letter-spacing:.05em; padding:5px 6px 2px; }

  /* ── live debug ──────────────────────────────────────────────── */
  #leftbottom label, #overlayctl label { cursor:pointer; }
  #overlayctl { font-size:12px; }
  #testconn:disabled { opacity:.38; cursor:not-allowed; }
  #leftbottom label.disabled, #overlayctl label.disabled { opacity:.45; pointer-events:none; }
  #livestatus { color:var(--accent-fg); font-size:11px; margin-top:4px; min-height:0; }
  #livestatus:empty { display:none; }
  #runstatus  { color:var(--tx-mid);   font-size:11px; min-height:14px; }
  /* ── DMA issue bar ───────────────────────────────────────────── */
  #issue-bar { display:none; margin-top:6px; border:1px solid #6b2222;
               border-radius:4px; background:#2a1212; padding:5px 8px;
               font-size:11px; max-height:120px; overflow-y:auto; }
  #issue-bar .ib-hdr { color:#e06060; font-weight:700; margin-bottom:4px;
                       display:flex; align-items:center; gap:6px; }
  #issue-bar .ib-hdr .ib-clear { margin-left:auto; cursor:pointer;
                                  color:var(--tx-lo); font-size:10px;
                                  background:none; border:none; padding:0; }
  #issue-bar .ib-hdr .ib-clear:hover { color:var(--tx-hi); }
  #issue-bar .ib-row { display:flex; align-items:center; gap:6px;
                       padding:2px 0; border-top:1px solid #3a1a1a;
                       cursor:pointer; }
  #issue-bar .ib-row:first-of-type { border-top:none; }
  #issue-bar .ib-row:hover { background:#3a1818; border-radius:3px; }
  #issue-bar .ib-icon { font-size:12px; flex:0 0 14px; text-align:center; }
  #issue-bar .ib-loc  { color:var(--tx-hi); font-weight:700; min-width:46px; }
  #issue-bar .ib-ch   { color:#d4a0a0; min-width:52px; }
  #issue-bar .ib-msg  { color:var(--tx-mid); flex:1; }
  #issue-bar .ib-flow { color:var(--tx-lo); font-size:10px; margin-left:auto; }
  .consolebox { background:var(--bg-code); border:1px solid var(--bd-subtle);
                border-radius:5px; padding:9px; font-size:11px; line-height:1.4;
                max-height:220px; overflow:auto; white-space:pre-wrap;
                word-break:break-word; max-width:100%; min-width:0; margin-top:9px;
                font-family:ui-monospace,monospace; }
  #console { flex:1 1 0; max-height:none; min-height:0; overflow:auto; }

  /* ── device map ──────────────────────────────────────────────── */
  #devmap { display:none; flex-direction:column; }
  #devmap.show { display:flex; }
  /* The topbar stays nowrap so the scan controls keep their pinned-right slot
     no matter how long the net list gets; each half absorbs pressure inside
     its own box. #devmap-scan has to be shrinkable or the first control added
     to it starves the netbar to zero, and the netbar needs a real basis: with
     basis:auto its shrink factor scales with its content width, so it absorbs
     essentially all of the overflow on its own. */
  /* Two columns: pills left, controls right. The topbar itself never wraps —
     each side absorbs its own growth, the pills by wrapping to more rows and
     the controls by stacking down the right column. */
  #devmap-topbar { display:flex; flex-wrap:nowrap; align-items:flex-start; gap:8px;
                   margin-bottom:8px; }
  /* min-content floor so a control row can never be squeezed narrower than the
     buttons on it; below that the pills give way rather than Clear sliding out
     of the pane. */
  /* max-width:50% is what actually holds the split. A column flex box sizes to
     its widest row and will not yield on its own, so without a cap the scan
     status text — which is perfectly able to ellipsize — silently takes width
     from the pills and they pay for it in rows. */
  #devmap-right { display:flex; flex-direction:column; align-items:flex-end;
                  gap:4px; flex:0 1 auto; min-width:min-content; max-width:50%;
                  margin-left:auto; }
  #dmRsrc { margin-left:0; }
  /* A half-width basis: the pills are navigation and claim their own half,
     rather than being left whatever the scan status text — transient, and able
     to ellipsize — happens not to want. */
  #devmap-netbar { display:flex; flex-wrap:wrap; gap:4px; align-items:center;
                   flex:1 1 50%; min-width:0; }
  /* Deliberately not flex-wrap: the status text changes length every 2s in
     live mode, so a wrapping cluster would move the Scan button between rows
     while the user is aiming at it. */
  #devmap-scan { display:flex; align-items:center; gap:4px; }
  /* Scan mode is a one-of-N selection that grew past the width a pill strip
     can spend on it, so it is a dropdown wearing the unselected .ltab skin —
     it is a selection, and must not read as the accent-filled active tab or as
     the solid-accent Scan verb beside it. */
  .scan-what { padding:2px 6px; border:1px solid var(--bd); border-radius:4px;
               background:var(--bg-raised); color:var(--tx-mid);
               cursor:pointer; font-size:11px; font-family:inherit;
               transition:background .12s, color .12s, border-color .12s; }
  .scan-what:hover { color:var(--tx-hi); background:var(--bg-hover); }
  .scan-what:focus-visible { outline:1px solid var(--bd-accent); outline-offset:1px; }
  #overlayWhat { margin-left:8px; vertical-align:middle; }
  /* Scan progress. A switch scan costs one board round-trip per tile, so the
     status text alone leaves the page looking hung. Revealed on a delay (see
     setScanBusy) so the 2s DMA poll does not blink it. */
  .spin { display:inline-block; width:11px; height:11px; margin-left:7px;
          vertical-align:-1px; box-sizing:border-box;
          border:2px solid var(--bd); border-top-color:var(--accent);
          border-radius:50%; animation:scanspin .7s linear infinite; }
  .spin[hidden] { display:none; }
  @keyframes scanspin { to { transform:rotate(360deg); } }
  /* Reduced motion: the ring appearing at all is the signal; spinning it is
     decoration the user has asked not to see. */
  @media (prefers-reduced-motion: reduce) { .spin { animation:none; } }
  /* Routing source. Hidden until a switch scan has actually reconstructed a
     map, because until then there is nothing to switch to. It is a one-of-2
     selection whose label carries the flow count, so a pill strip would cost
     width proportional to the design — the same reason scan mode above became
     a dropdown. Save is a verb, so it stays a button beside it. */
  /* flex:0 0 auto, not min-width:0: as a shrinkable flex child this collapsed
     to the width of the selector alone and the Save button spilled out on top
     of the next control. The status text next to it is the one that gives. */
  .rsrc { display:inline-flex; align-items:center; gap:6px; flex:0 0 auto;
          margin-left:10px; font-size:11px; vertical-align:middle; }
  .rsrc[hidden] { display:none; }
  .rsrc-diff-wrap { color:var(--tx-dim); white-space:nowrap; cursor:pointer;
                     flex:0 0 auto; }
  .rsrc-diff-wrap input { margin:0 4px 0 0; vertical-align:middle; }
  .rsrc-diff-wrap:has(input:checked) { color:var(--tx-hi); }
  /* A floor, not min-width:0: the whole point of the label is the source name
     and the flow count, and a selector shrunk to "rout…" carries neither. The
     status text beside it already ellipsizes, so it absorbs the pressure. */
  .rsrc-sel { flex:0 0 auto; min-width:112px; }
  /* Dynamic stays selected across every panel and view, so the selector has
     to carry that state at rest. Border and text only: an --accent-dim fill
     here is the active-tab reading the .scan-what skin exists to avoid. */
  .rsrc-sel[data-src="dynamic"] { color:var(--tx-hi); border-color:var(--bd-accent); }
  .rsrc .rs-dl { flex:0 0 auto; font-size:11px; padding:2px 8px;
                 border-radius:4px; cursor:pointer; font-family:inherit;
                 color:var(--tx-dim); border:1px solid var(--bd);
                 background:var(--bg-raised); }
  .rsrc .rs-dl:hover { color:var(--tx-hi); background:var(--bg-hover); }
  /* nowrap: the failure message is longer than the selector it replaces, and
     wrapping it would push the Scan button onto a second row — the one thing
     the surrounding layout is built to prevent. */
  .rsrc-err { color:var(--tx-dim); white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis; min-width:0; }
  .rsrc-sel[hidden], .rsrc .rs-dl[hidden], .rsrc-err[hidden] { display:none; }
  #gridRsrc { margin-left:8px; }
  /* Scan is an ACTION, not a selection. It deliberately avoids both tab states:
     --accent-dim fill would read as a selected tab, and the default button
     surface (--bg-raised on --bd) is nearly identical to an *un*selected .ltab.
     A solid accent fill is used nowhere else, so it can only read as a verb. */
  #dmScanBtn, #gridScanBtn {
               font-size:11px; padding:3px 13px; border-radius:4px; cursor:pointer;
               border:1px solid var(--accent); background:var(--accent);
               color:var(--bg-code); font-weight:600; letter-spacing:.02em;
               transition:filter .12s; }
  #gridScanBtn { margin-left:6px; vertical-align:middle; }
  #dmScanBtn:hover:not(:disabled), #gridScanBtn:hover:not(:disabled) { filter:brightness(1.12); }
  #dmScanBtn:active:not(:disabled), #gridScanBtn:active:not(:disabled) { filter:brightness(.92); }
  #dmScanBtn:disabled, #gridScanBtn:disabled { opacity:.38; cursor:not-allowed; filter:none; }
  /* Secondary to Scan: same size, neutral surface, so the filled accent stays
     the one thing in the row that reads as the primary action. */
  #dmClearBtn { font-size:11px; padding:3px 11px; border-radius:4px; }
  #dmLiveWrap { font-size:10px; color:var(--tx-lo); cursor:pointer; display:flex;
                align-items:center; gap:3px; }
  #dmLiveWrap input { margin:0; }
  /* A px cap, not a percentage: nowrap text contributes its full width to the
     column's min-content, and min-width beats max-width — so a percentage cap
     on the column cannot hold it back. Capped here it ellipsizes instead, and
     the column's width stops depending on how long the last message was. */
  #dmScanStatus { font-size:10px; color:var(--accent-fg); min-width:0; white-space:nowrap;
                  overflow:hidden; text-overflow:ellipsis; max-width:100%; margin-top:4px;
                  align-self:stretch; }
  /* No empty line above the buttons on the runs where there is no status yet. */
  #dmScanStatus:empty { display:none; }
  #dmScanStatus.err { color:var(--red-fg); }
  .dm-vsep { width:1px; height:16px; background:var(--bd-soft); flex-shrink:0; }
  /* Status ring on a net chip — sits left of the identity dot so the net's own
     color stays readable next to it. */
  .dm-chip .chstat { width:7px; height:7px; border-radius:50%; flex-shrink:0;
                     border:1.5px solid transparent; }
  .dm-chip { display:inline-flex; align-items:center; gap:4px; padding:2px 8px 2px 6px;
             border-radius:9999px; font-size:10px; font-weight:500; cursor:pointer;
             border:1px solid var(--stroke,rgba(228,228,228,.2)); user-select:none;
             transition:opacity .15s, background .15s; white-space:nowrap;
             background:transparent; color:var(--fg,#e4e4e4); }
  .dm-chip .chdot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
  .dm-chip.all-chip { border-color:rgba(228,228,228,.2); }
  .dm-chip.all-chip.act { background:rgba(228,228,228,.12); border-color:rgba(228,228,228,.5); }
  .dm-chip.net-chip.act { background:rgba(228,228,228,.07); border-color:rgba(228,228,228,.2); opacity:1; }
  .dm-chip.net-chip:not(.act) { opacity:0.22; }
  #devmap-vp { overflow:hidden; position:relative; height:calc(100vh - 280px); min-height:420px;
               border:1px solid rgba(228,228,228,.1); border-radius:5px;
               background:#0f1018; cursor:grab; }
  #devmap-vp.panning { cursor:grabbing; }
  #devmap-canvas { position:absolute; top:0; left:0; transform-origin:0 0; }
  #devmap-hint { position:absolute; bottom:7px; right:9px; font-size:9px;
                 color:rgba(228,228,228,.35); pointer-events:none; letter-spacing:.2px; }
  #devmap-spacehint { position:absolute; top:7px; left:9px; font-size:9px;
                      color:rgba(228,228,228,.35); pointer-events:none; }
  #devmap-reset { position:absolute; top:7px; right:8px; font-size:10px; padding:2px 9px;
                  border:1px solid rgba(228,228,228,.2); background:rgba(228,228,228,.07);
                  color:rgba(228,228,228,.7); border-radius:4px; cursor:pointer; z-index:10; }
  #devmap-reset:hover { background:rgba(228,228,228,.13); color:rgba(228,228,228,.95); }
  #dmSwWrap { position:absolute; top:34px; right:8px; font-size:10px;
              color:rgba(228,228,228,.6); cursor:pointer; z-index:10;
              display:flex; align-items:center; gap:4px; user-select:none; }
  #dmSwWrap:hover { color:rgba(228,228,228,.9); }
  #dmSwWrap input { cursor:pointer; margin:0; }
  #devmap-legend { display:flex; gap:10px; flex-wrap:wrap; margin-top:6px; font-size:10px;
                   color:rgba(228,228,228,.35); align-items:center; }
  .dml-item { display:flex; align-items:center; gap:4px; }
  .dml-swatch { width:16px; height:7px; border-radius:2px; }
  .dml-line { width:18px; height:0; }
  .dml-dot { width:7px; height:7px; border-radius:50%; }
  .dbgpanel { margin-top:10px; border:1px solid var(--bd); border-radius:5px; padding:10px;
              background:var(--bg-surface); }
  .dbgpanel h2 { margin-top:0; }
  .dbgpresets { margin-bottom:6px; }
  .dbgpresets .ltab { margin-top:0; }

  /* ── op-kind label badges ────────────────────────────────────── */
  .kb { display:inline-block; border-radius:3px; padding:0 5px; margin-right:4px;
        font-size:10px; font-weight:600; vertical-align:1px;
        border:1px solid transparent; font-family:ui-monospace,monospace; }
  .kb.bd_config  { background:#162a08; color:#c8ff80; border-color:#3a6010; }
  .kb.createio   { background:#0c2048; color:#80d0ff; border-color:#1e50a0; }
  .kb.startio    { background:#301800; color:#ffc860; border-color:#704000; }
  .kb.lock       { background:#280a30; color:#e890ff; border-color:#602878; }
  .kb.wait       { background:#300808; color:#ff9090; border-color:#702020; }
  .kb.enable_ooo { background:#082828; color:#60f0f0; border-color:#106060; }
  .kb.loop, .kb.loopend { background:#101038; color:#c0c0ff; border-color:#303088; }
  .kb.loopidx    { background:#0c0c28; color:#9898e0; border-color:#282860; }
  .rnote { color:#9090c0; font-style:italic; margin-left:8px; font-size:11px; }

  /* ── code viewer tokens ──────────────────────────────────────── */
  .kw { color:#7faadf; } .fn { color:#dcdcaa; } .num { color:#a8cfa0; }
  .cm { color:#5c7a50; }
  .str { color:#c99070; } .pp { color:#b48ac0; } .ty { color:#5fb3b3; }
  .pann { color:#7aaf7a; font-style:italic; margin-left:1ch; }
  .gap { color:#c89050; font-style:italic; opacity:0.75; }
  .bdpretty { font-family:ui-monospace,monospace; font-size:11.5px; line-height:1.45;
              color:#80e060; background:#081808; border-left:3px solid #206010;
              border-radius:4px; margin:2px 0 9px 22px;
              padding:6px 10px; white-space:pre; overflow-x:auto; }
  .rline { font-family:ui-monospace,monospace; font-size:11.5px; line-height:1.55;
           white-space:pre-wrap; }
  .lno, .mlno { color:var(--tx-lo); user-select:none; }
  .lno  { margin-right:6px; }
  .mlno { display:inline-block; min-width:3.2em; text-align:right; margin-right:10px; }
  .midctrls { margin:6px 0; }
  .midctrls button { margin-right:8px; }
  .irfull { max-height:420px; overflow:auto; }
  .irln { display:block; }
  .irlno { color:#40445a; user-select:none; display:inline-block;
           min-width:3.6em; text-align:right; margin-right:10px; }
  .irhi { background:#26240e; }
  .irhi .irlno { color:var(--amber-fg); }
  .irhidden { display:block; }
  .irhidden.hide { display:none; }
  .irfold { display:block; color:#6890d8; cursor:pointer; user-select:none;
            background:#141a2c; padding:0 6px; }
  .irfold:hover { background:#1c2438; }
  .irfold.irfold-open { color:#80c880; }
  .khl { background:#26240e; border-left:3px solid var(--amber-fg); }
  .kshowall { margin:4px 0 9px; padding:3px 11px; font-size:12px; cursor:pointer; }
  .kfileref { color:var(--accent-fg); font-size:11px; }
  /* ── selection-to-LLM popup ─────────────────────────────────── */
  #sel-popup { position:fixed; z-index:900; display:none;
               background:var(--accent); color:#000; font-size:11px; font-weight:600;
               padding:3px 10px; border-radius:10px; cursor:pointer;
               box-shadow:0 2px 8px rgba(0,0,0,.5); white-space:nowrap;
               transform:translateX(-50%); pointer-events:auto; }
  #sel-popup:hover { background:var(--accent-fg); }
  /* ── LLM context pills ───────────────────────────────────────── */

  /* ── search pane ─────────────────────────────────────────────── */
  #searchpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0; }
  #searchpane.hide { display:none; }
  #sr-inputrow { flex:0 0 auto; display:flex; align-items:center; gap:6px;
                 margin-bottom:7px; position:relative; }
  #sr-input { flex:1 1 auto; background:var(--bg-input); border:1px solid var(--bd);
              border-radius:4px; color:var(--tx-hi); font-size:12px;
              font-family:ui-monospace,monospace; padding:4px 9px; outline:none;
              transition:border-color .15s; }
  #sr-input:focus { border-color:var(--accent); }
  #sr-suggest { position:absolute; top:100%; left:0; right:0; z-index:200;
                background:var(--bg-surface); border:1px solid var(--bd);
                border-radius:0 0 5px 5px; max-height:200px; overflow-y:auto;
                font-size:12px; font-family:ui-monospace,monospace;
                box-shadow:0 6px 20px rgba(0,0,0,.5); }
  .sr-sug-item { padding:4px 10px; cursor:pointer; display:flex; gap:8px;
                 align-items:baseline; white-space:nowrap; overflow:hidden; }
  .sr-sug-item:hover, .sr-sug-item.sel { background:var(--bg-raised); }
  .sr-sug-kind { color:var(--accent-fg); font-size:10px; min-width:52px; }
  .sr-sug-label { color:var(--tx-hi); overflow:hidden; text-overflow:ellipsis; }
  .sr-sug-match { color:#f8d040; }
  .sr-sug-tile { color:var(--tx-lo); font-size:10px; margin-left:auto; white-space:nowrap; }
  #sr-chips { flex:0 0 auto; display:flex; flex-wrap:wrap; gap:4px; margin-bottom:7px; }
  #sr-results { flex:1 1 0; overflow-y:auto; min-height:0; }
  #sr-results table { border-collapse:collapse; width:100%; font-size:12px; }
  #sr-results th, #sr-results td { border:1px solid var(--bd-subtle); padding:3px 7px;
                                    text-align:left; white-space:nowrap; }
  #sr-results th { background:var(--bg-surface); color:var(--tx-lo); font-weight:600;
                   font-size:11px; }
  #sr-results td:first-child { font-family:ui-monospace,monospace; }
  .sr-group-hdr { padding:5px 8px 3px; color:var(--accent-fg); font-size:11px;
                  font-weight:700; background:var(--bg-base); border-bottom:1px solid var(--bd-subtle); }
  .sr-empty { color:var(--tx-lo); font-size:12px; padding:10px; }
  .sr-stat  { color:var(--tx-lo); font-size:11px; padding:2px 8px 6px; }

  /* ── search chip ─────────────────────────────────────────────── */
  .lk-chip { display:inline-flex; align-items:center; gap:4px;
             background:#1e1600; border:1px solid #7a5a20; border-radius:10px;
             color:#d8a840; font-size:11px; padding:1px 8px 1px 6px;
             font-family:ui-monospace,monospace; }
  .lk-chip .lk-x { cursor:pointer; color:#a08030; font-size:13px;
                    line-height:1; margin-left:2px; }
  .lk-chip .lk-x:hover { color:#e0a840; }

  /* ── table styles ────────────────────────────────────────────── */
  /* ── right panel sections ────────────────────────────────────── */
  .sec { margin:14px 0 4px; padding-top:10px; border-top:1px solid var(--bd-subtle); }
  .sec:first-child { margin-top:4px; padding-top:0; border-top:none; }
  .sec-hdr { font-size:10px; font-weight:700; text-transform:uppercase;
             letter-spacing:.07em; color:var(--tx-lo); margin-bottom:6px; }
  /* ── mismatch status bar ─────────────────────────────────────── */
  .statusbar { display:flex; align-items:center; gap:8px; padding:7px 10px;
               border-radius:4px; margin-bottom:10px; font-size:12px; }
  .statusbar.ok   { background:var(--green-bg); border:1px solid #205a30; color:var(--green-fg); }
  .statusbar.bad  { background:var(--red-bg);   border:1px solid #582020; color:var(--red-fg); }
  .statusbar .sb-icon { font-size:14px; flex-shrink:0; }
  .statusbar .sb-detail { color:var(--tx-mid); font-size:11px; }
  /* ── collapsible code sections ───────────────────────────────── */
  details.codesec { margin:10px 0; }
  details.codesec > summary { cursor:pointer; font-size:11px; font-weight:600;
                               color:var(--accent-fg); user-select:none;
                               padding:4px 0; list-style:none; display:flex;
                               align-items:center; gap:5px; }
  details.codesec > summary::before { content:'\25B6'; font-size:9px;
                                       color:var(--tx-lo); transition:transform .15s; }
  details.codesec[open] > summary::before { transform:rotate(90deg); }
  details.codesec > summary:hover { color:var(--accent); }
  details.codesec > .codesec-body { min-width:0; padding:2px 0 4px 14px; }
  .kmap { border-collapse:collapse; margin:7px 0 9px; font-size:12px; }
  .kmap th, .kmap td { border:1px solid var(--bd); padding:3px 9px; text-align:left; }
  .kmap th, .rctbl th { background:var(--bg-raised); color:var(--tx-mid); font-weight:600; }
  .kmap td.arrow { color:var(--accent-fg); text-align:center; }
  .kmap .win   { color:#dcdcaa; }
  .kmap .mlock { color:#6aaa80; }
  .kmap .morder { color:#c8a030; }
  .kmap .mnone  { color:#a05050; }
  .rctbl { border-collapse:collapse; margin:7px 0 9px; font-size:12px; width:100%; }
  .rctbl th, .rctbl td { border:1px solid var(--bd); padding:3px 8px; text-align:left;
                           white-space:nowrap; }
  .rctbl td:last-child { white-space:normal; }
  .rctbl td:first-child { color:var(--accent-fg); font-family:ui-monospace,monospace; }

  /* ── tile routing + DMA BD mini-sections ─────────────────────── */
  .rt-row { display:flex; align-items:center; gap:6px; font-size:11px;
            font-family:ui-monospace,monospace; padding:2px 4px;
            border-radius:2px; margin:1px 0; }
  .rt-row .rt-kind { flex:0 0 32px; font-size:9px; font-weight:700;
                      text-transform:uppercase; color:var(--tx-lo); }
  .rt-row .rt-ports { flex:1; color:var(--tx-hi); }
  .rt-row .rt-flow  { flex:0 0 auto; color:var(--tx-lo); font-size:9px; }
  .rt-pktid { font-size:9px; color:#c07fd4; background:#1a0d24; border-radius:2px;
              padding:1px 5px; margin-left:6px; display:inline-block; white-space:nowrap; }
  .rt-pktmask { font-size:9px; color:#e87850; background:#3a1010; border-radius:2px;
                padding:1px 5px; margin-left:6px; display:inline-block; white-space:nowrap; }
  .rt-row .rt-pktid, .rt-row .rt-pktmask { flex:0 0 auto; margin-left:0; }
  /* stream-switch scan verdicts */
  .sw-ok { font-size:9px; color:#7fd48a; background:#0f2413; border-radius:2px;
           padding:0 3px; margin-left:4px; flex:0 0 auto; }
  .sw-bad { font-size:9px; color:#ff9270; background:#361208; border-radius:2px;
            padding:0 3px; margin-left:4px; flex:0 0 auto; }
  .rt-row.sw-extra { opacity:0.95; }
  .sw-state { font-size:9px; border-radius:2px; padding:0 4px; margin-left:6px; }
  .sw-state.verified { color:#7fd48a; background:#0f2413; }
  .sw-state.mismatch { color:#ff9270; background:#361208; }
  .sw-state.unreachable { color:#c0a080; background:#2a2118; }
  .sw-state.idle { color:#8a94a0; background:#1a1e24; }
  .rt-s2mm { font-size:9px; color:#30c0d0; background:#0a2830; border-radius:2px;
             padding:1px 4px; margin-left:6px; display:inline-block; white-space:nowrap; }
  .rt-mm2s { font-size:9px; color:#c050b0; background:#2a1028; border-radius:2px;
             padding:1px 4px; margin-left:6px; display:inline-block; white-space:nowrap; }
  .rt-row .rt-s2mm, .rt-row .rt-mm2s { flex:0 0 auto; margin-left:0; }
  .rt-route { font-size:9px; color:#687080; background:#141820; border-radius:2px;
              padding:1px 4px; margin-left:6px; display:inline-block; white-space:nowrap;
              font-style:italic; opacity:.88; }
  .rt-row .rt-route { flex:0 0 auto; margin-left:0; }
  .rt-row .rt-fwd   { color:var(--tx-lo); font-size:9px; margin-left:2px; }
  .rt-row.cct  { border-left:2px solid #4a7fd4; }
  .rt-row.pkt  { border-left:2px solid #9c4fd4; }
  .rt-row.mst  { border-left:2px solid #7a5098; opacity:.92; }
  .rt-row.shim { border-left:2px solid #4aa4d4; }
  .rt-mst-hdr { font-size:10px; color:var(--tx-lo); margin:8px 0 3px 4px;
                text-transform:uppercase; letter-spacing:.04em; }
  .bd-mini { font-size:11px; font-family:ui-monospace,monospace;
             padding:2px 4px; margin:1px 0; border-left:2px solid var(--bd); }
  .bd-mini .bd-id, .bd-id   { color:var(--accent-fg); font-weight:700; margin-right:3px; }
  .bd-mini .bd-len, .bd-len { color:var(--tx-hi); margin-right:3px; }
  .bd-mini .bd-next,.bd-next{ color:var(--tx-lo); margin-right:3px; }
  .bd-mini .bd-lock,.bd-lock{ color:#6aaa80; font-size:10px; }
  .bd-mini-ch { font-size:11px; font-weight:600; margin:6px 0 2px;
                color:var(--tx-mid); }

  .dimtxt { color:var(--tx-lo); font-size:12px; }

  /* ── LLM pane ────────────────────────────────────────────────── */
  #llmterm { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             margin-top:7px; padding:7px; background:var(--bg-code);
             border:1px solid var(--bd-subtle); border-radius:5px;
             overflow:hidden; cursor:text; }
  #llmmsg { flex:1 1 0; min-height:0; overflow-y:auto; padding:2px 0; }
  .llm-msg { margin:3px 0; padding:5px 9px; border-radius:3px;
             font-family:ui-monospace,monospace; font-size:12px; line-height:1.45;
             white-space:pre-wrap; word-break:break-word; }
  .llm-msg-you { background:#0d1c10; border-left:2px solid var(--accent-fg); color:#b8ddf0; }
  .llm-msg-ai  { background:transparent; color:#c0c4d8; border-left:2px solid var(--bd); }
  .llm-msg-ctx { background:transparent; color:var(--tx-lo); font-style:italic;
                 border-left:2px solid var(--bd-subtle); font-size:11px; }
  .llm-msg-ai .llm-you       { color:var(--accent-fg); font-weight:bold; }
  .llm-msg-ai .llm-error     { color:#f07880; font-weight:bold; }
  /* One row per tool call, result folded onto the same row. white-space:normal
     escapes the bubble's pre-wrap so the row cannot wrap or inherit the blank
     lines the markers arrive surrounded by. */
  .llm-tools { margin:3px 0; white-space:normal; }
  .llm-tc { display:flex; align-items:baseline; gap:6px; padding:0 6px;
            font-size:10.5px; line-height:1.65; border-radius:0 3px 3px 0;
            background:var(--bg-surface); border-left:2px solid var(--bd); }
  .llm-tc + .llm-tc { margin-top:1px; }
  /* Sans for the prettified name, monospace for the argument values. */
  .llm-tc .tn { flex:0 0 auto; color:#e0af68; font-weight:600;
                font-family:ui-sans-serif, system-ui, sans-serif; }
  .llm-tc .ta { flex:1 1 auto; min-width:0; overflow:hidden; white-space:nowrap;
                text-overflow:ellipsis; color:var(--tx-lo); }
  /* Background-only boxes: a border would grow the row height. */
  .llm-tc .tg { display:inline-block; margin-right:4px; padding:0 5px;
                border-radius:3px; background:var(--bg-raised); color:var(--tx-mid); }
  .llm-tc .tg i { font-style:normal; margin-right:5px; color:var(--tx-lo);
                  opacity:.75; font-family:ui-sans-serif, system-ui, sans-serif; }
  .llm-tc .targ { color:var(--tx-lo); }
  .llm-tc .tr { flex:0 0 auto; color:var(--tx-lo); opacity:.85; }
  .llm-tc.done { border-left-color:var(--accent); }
  .llm-tc.err  { border-left-color:#f07880; }
  .llm-tc.err .tr { color:#f07880; opacity:1; }
  .llm-msg-ai .llm-file      { color:#9ece6a; }
  .llm-msg-ai .llm-line      { color:#e0af68; }
  .llm-msg-ai .md-h      { color:#7dcfff; font-weight:bold; display:block; margin-top:4px; }
  .llm-msg-ai .md-bullet { color:#7aa2f7; }
  .llm-msg-ai strong     { color:#c0caf5; font-weight:bold; }
  .llm-msg-ai .md-code   { background:#181828; color:#e0af68; padding:0 3px; border-radius:3px; }
  .llm-msg-ai .md-block  { background:#0c0c18; border:1px solid var(--bd-subtle);
                            border-radius:4px; padding:5px 9px; margin:3px 0;
                            overflow-x:auto; white-space:pre; }
  .llm-msg-ai .md-block code { background:none; padding:0; color:#c0caf5; }
  .llm-msg-ai .md-table  { border-collapse:collapse; margin:4px 0; font-size:.85em; }
  .llm-msg-ai .md-table th,
  .llm-msg-ai .md-table td { border:1px solid var(--bd-subtle); padding:3px 8px; text-align:left; }
  .llm-msg-ai .md-table th { background:var(--bg-surface); color:#7dcfff; font-weight:bold; }
  .llm-msg-ai em           { color:#c0caf5; font-style:italic; }
  .llm-msg-ai .md-ol       { margin:2px 0 2px 18px; }
  .llm-msg-ai .md-a        { color:#7aa2f7; text-decoration:underline; cursor:pointer; }
  .llm-msg-ai .cm-keyword { color:#bb9af7; }
  .llm-msg-ai .cm-string  { color:#9ece6a; }
  .llm-msg-ai .cm-comment { color:#485070; font-style:italic; }
  .llm-msg-ai .cm-number  { color:#ff9e64; }
  #llmthink { padding:4px 8px 2px; }
  .llm-dot  { display:inline-block; width:4px; height:4px; border-radius:50%;
              background:var(--accent); margin-right:3px;
              animation:llmblink 1.1s ease-in-out infinite; }
  .llm-dot:nth-child(2){ animation-delay:.18s; }
  .llm-dot:nth-child(3){ animation-delay:.36s; }
  @keyframes llmblink{ 0%,80%,100%{opacity:.15} 40%{opacity:.85} }
  #llminline { flex:0 0 auto; display:flex; align-items:flex-start; margin-top:6px;
               border-top:1px solid var(--bd-subtle); padding-top:6px; }
  #llmprompt { color:var(--accent-fg); margin-right:6px; white-space:nowrap;
               font-family:ui-monospace,monospace; padding-top:2px; font-size:11.5px; }
  /* contenteditable, not a textarea: context pills are real elements sitting
     inline in the message, which a textarea cannot hold. */
  #llmin { flex:1 1 auto; background:transparent; border:none; outline:none;
           color:var(--tx-hi); font-family:ui-monospace,monospace; padding:0;
           overflow-y:auto; max-height:120px; line-height:1.5; font-size:inherit;
           white-space:pre-wrap; word-break:break-word; min-height:1.5em; }
  #llmin:empty::before { content:attr(data-ph); color:var(--tx-lo); }
  .msg-pill { display:inline-flex; align-items:baseline; gap:4px; margin:0 2px;
              padding:0 6px; border-radius:9999px; font-size:10.5px;
              font-family:ui-sans-serif,system-ui,sans-serif;
              background:var(--accent-dim); color:var(--accent-fg);
              border:1px solid var(--bd-accent);
              vertical-align:baseline; white-space:nowrap; }
  /* Only the × is unselectable — the chip itself must highlight and copy like
     the text around it. */
  .msg-pill .mp-x { cursor:pointer; opacity:.6; font-size:11px; user-select:none; }
  .msg-pill .mp-x:hover { opacity:1; color:var(--tx-hi); }
  #llmsend, #llmreset { margin-left:6px; }
  #llmauth { position:absolute; inset:0; display:flex; align-items:center;
             justify-content:center; background:rgba(0,0,0,.72); z-index:50; }
  #llmauth.hide { display:none; }
  #llmauthbox { background:var(--bg-surface); border:1px solid var(--bd);
                border-radius:6px; padding:18px 20px; width:290px;
                font-family:ui-monospace,monospace; color:var(--tx-hi);
                box-shadow:0 8px 32px rgba(0,0,0,.7); }
  #llmauthbox .t { color:var(--accent-fg); font-weight:600; margin-bottom:10px; font-size:13px; }
  #llmauthin { width:100%; box-sizing:border-box; background:var(--bg-input);
               border:1px solid var(--bd); border-radius:4px; color:var(--tx-hi);
               font-family:ui-monospace,monospace; padding:5px 7px; outline:none; }
  #llmauthin:focus { border-color:var(--accent); }
  #llmauthok { margin-top:11px; }
  #llmautherr { color:var(--red-fg); font-size:11px; margin-top:7px; min-height:14px; }

  /* ── run controls toolbar ────────────────────────────────────── */
  #ctrlbar { flex:0 0 auto; padding:10px 14px; background:var(--bg-surface); }
  .run-head { display:flex; align-items:center; flex-wrap:wrap;
              gap:6px 10px; margin-bottom:8px; }
  #connstatus { display:inline-flex; align-items:center; gap:6px; min-width:0;
                margin-left:auto; flex:0 1 auto;
                max-width:100%; color:var(--tx-lo); font-size:10px;
                overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #connstatus::before { content:""; flex:0 0 7px; width:7px; height:7px;
                        border-radius:50%; background:var(--tx-lo); }
  #connstatus[data-state="connected"] { color:var(--accent-fg); }
  #connstatus[data-state="connected"]::before { background:var(--accent); }
  #connstatus[data-state="busy"] { color:var(--amber-fg); }
  #connstatus[data-state="busy"]::before { background:var(--amber-fg); }
  #connstatus[data-state="error"] { color:var(--red-fg); }
  #connstatus[data-state="error"]::before { background:var(--red-fg); }
  .run-fields { display:flex; flex-direction:column; gap:5px; }
  .run-row { display:flex; align-items:center; gap:8px; min-width:0; }
  .run-label { flex:0 0 46px; color:var(--tx-mid); font-size:11px; font-weight:600; }
  .run-row > select { flex:0 1 210px; min-width:110px; }
  #appinfo { flex:1 1 auto; min-width:0; overflow:hidden;
             text-overflow:ellipsis; white-space:nowrap; color:var(--tx-lo);
             font-size:11px; }
  #boardHost { flex:1 1 auto; min-width:0; box-sizing:border-box; }
  #ctrlbar[data-apps="one"] #row-app { display:none; }
  #ctrlbar[data-simonly="true"] #row-board,
  #ctrlbar[data-simonly="true"] #attachbtn,
  #ctrlbar[data-simonly="true"] #testconn,
  #ctrlbar[data-simonly="true"] #connhint { display:none; }
  #ctrlbar[data-apps="one"][data-simonly="true"] .run-fields { display:none; }
  #appbadge { flex:0 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis;
              white-space:nowrap; font-size:12px; color:var(--tx-mid); }
  #appbadge::before { content:'\2014\00a0'; color:var(--tx-lo); }
  .run-actions { display:flex; align-items:center; flex-wrap:wrap; gap:6px;
                 margin-top:0; }
  .run-actions #attachbtn { margin-left:4px; }
  #connhint { margin-top:6px; padding:6px 8px; font-size:11px; color:#f6c177;
              background:#2a1f14; border:1px solid #5a3d1a; border-radius:4px; }

  /* ── overlay toolbar above grid ─────────────────────────────── */
  #overlaybar { flex:0 0 auto; display:flex; align-items:center; gap:8px;
                padding:4px 0 6px; font-size:12px; }

  /* ── collapsible legend ──────────────────────────────────────── */
  #legend-detail { margin-bottom:8px; }
  #legend-detail summary { cursor:pointer; font-size:11px; color:var(--tx-lo);
                            user-select:none; padding:2px 0; }
  #legend-detail summary:hover { color:var(--tx-mid); }
  #legend-detail .legend { margin-top:8px; }
  #legend-detail .legend-actions { margin-top:8px; }

  /* ── aiegdb quick-command chips ─────────────────────────────── */
  .qcmd { padding:1px 7px; font-size:11px; margin:2px 2px 2px 0;
          font-family:ui-monospace,monospace; background:var(--bg-raised);
          border:1px solid var(--bd); border-radius:3px; color:var(--tx-mid);
          cursor:pointer; transition:background .1s, color .1s; }
  .qcmd:hover { background:var(--bg-hover); color:var(--tx-hi); }
</style>
</head>
<body>
<div id="left">
  <!-- ── grid area ─────────────────────────────────────────────── -->
  <div id="lefttop">
  <div id="lefttop-header">
    <h1>AIE Debug</h1>
    <span id="appbadge" class="hide" title=""></span>
    <div id="viewswitcher">
      <span class="vsw act" data-v="grid" onclick="switchView('grid')">Grid</span>
      <span class="vsw" data-v="map" onclick="switchView('map')">Device Map</span>
      <span class="vsw" data-v="targets" onclick="switchView('targets')">Host</span>
      <!-- Profile tab: disabled until a run finishes and timeline.png renders. -->
      <span class="vsw vsw-disabled" id="vsw-profile" data-v="profile"
            onclick="switchView('profile')" title="Timeline appears after a run finishes">Profile</span>
    </div>
  </div>
  <div class="sub" id="meta" style="display:none"></div>
  <!-- live status overlay: pinned above the tile grid -->
  <div id="overlayctl" style="margin-bottom:6px;">
    <label><input type="checkbox" id="liveToggle"> Live status overlay</label>
    <span class="rsrc" id="gridRsrc" hidden></span>
    <select id="overlayWhat" class="scan-what" title="what to read on the next scan">
      <option value="dma" title="DMA channel state and BD progress">DMA</option>
      <option value="cores" title="core status per tile">Cores</option>
      <option value="events" title="DMA start/finish/error events">Events</option>
      <option value="switch" title="stream-switch registers, diffed against the routing map">Switch</option>
    </select>
    <button id="gridScanBtn" title="read live status from the board / simulator">Scan</button><span class="spin" id="gridSpin" hidden title="scan in progress"></span>
    <div id="livestatus"></div>
    <div id="runstatus"></div>
    <div id="issue-bar"></div>
  </div>
  <div id="grid"></div>
  <div id="devmap">
    <!-- Scan controls live in their own static container: buildNetBar() wipes
         #devmap-netbar on every rebuild, so anything placed inside it would be
         destroyed the first time a net chip is clicked. -->
    <div id="devmap-topbar">
      <div id="devmap-netbar"></div>
      <span class="dm-vsep"></span>
      <!-- Right column. The net pills own the left and grow downward there;
           anything control-shaped stacks on this side instead of stealing
           width from them. -->
      <div id="devmap-right">
      <div id="devmap-scan">
        <label id="dmLiveWrap" title="re-scan every 2s"><input type="checkbox" id="dmLiveToggle"> live</label>
        <select id="dmScanWhat" class="scan-what" title="what to read on the next scan">
          <option value="dma" title="DMA channel state and BD progress">DMA</option>
          <option value="cores" title="core status per tile">Cores</option>
          <option value="events" title="DMA start/finish/error events">Events</option>
          <option value="switch" title="stream-switch registers, diffed against the routing map">Switch</option>
        </select>
        <button id="dmScanBtn" title="read live status from the board / simulator">Scan</button><span class="spin" id="dmSpin" hidden title="scan in progress"></span>
        <button id="dmClearBtn" title="clear scan status and search highlights">Clear</button>
      </div>
        <!-- Second row of the right column: this selects which routing map the
             whole view reads, and the scan row has no width left to lend it. -->
        <span class="rsrc" id="dmRsrc" hidden></span>
        <span id="dmScanStatus"></span>
      </div>
    </div>
    <div id="devmap-vp">
      <button id="devmap-reset" onclick="dmReset(true)">Reset view</button>
      <label id="dmSwWrap" title="show/hide CCT/PKT routing config inside tiles"><input type="checkbox" id="dmSwToggle" checked> routing info</label>
      <div id="devmap-spacehint">scroll to zoom · click tile to inspect · right-click tile for routing/isolate menu · click stream to isolate</div>
      <div id="devmap-canvas"><svg id="devmap-svg"></svg></div>
      <div id="devmap-hint">col 0–3 · row 0 (shim) at bottom</div>
    </div>
    <div id="devmap-legend">
      <div class="dml-item"><div class="dml-swatch" style="background:var(--tile-shim-fill);border:1px solid var(--tile-shim-stroke)"></div>SHIM = PL/NoC gateway</div>
      <div class="dml-item"><div class="dml-swatch" style="background:var(--tile-mem-fill);border:1px solid var(--tile-mem-stroke)"></div>MEM = memory tiles</div>
      <div class="dml-item"><div class="dml-swatch" style="background:var(--tile-core-fill);border:1px solid var(--tile-core-stroke)"></div>AIE = compute cores</div>
      <div class="dml-item"><div class="dml-line" style="border-top:2.5px solid #599ce7"></div>solid = stream-switch route</div>
      <div class="dml-item"><div class="dml-line" style="border-top:2px solid #599ce7;border-bottom:2px solid #599ce7;height:5px"></div><span style="color:#599ce7">&#9642;</span> double line + square = ping-pong window (kernel&harr;kernel)</div>
      <div class="dml-item"><div class="dml-line" style="border-top:2px dashed #599ce7"></div>dashed = direct shared-memory (DMA&harr;kernel)</div>
      <div class="dml-item"><div class="dml-dot" style="background:#599ce7;border:1.2px solid #181818"></div>● solid = injects into stream (source / contributor)</div>
      <div class="dml-item"><div class="dml-dot" style="background:#181818;border:2.2px solid #599ce7"></div>○ hollow = takes from stream (destination / tap)</div>
      <div class="dml-item"><span style="font:600 9px monospace;color:#30c0d0;background:#0a2830;padding:0 3px;border-radius:2px">s2mm</span> receive &nbsp; <span style="font:600 9px monospace;color:#c050b0;background:#2a1028;padding:0 3px;border-radius:2px">mm2s</span> send &nbsp; <span style="font:600 9px monospace;color:#687080;background:#141820;padding:0 3px;border-radius:2px;font-style:italic">route</span> passthrough</div>
      <!-- Live-status swatches: hidden until a scan returns, so the legend does
           not advertise colors that are not on screen yet. Colors are injected
           from LSTATE at load so this list cannot drift from the renderer. -->
      <span id="devmap-stlegend" class="hide"></span>
    </div>
  </div>
  <!-- ── Targets panel ───────────────────────────────────────────── -->
  <div id="targetsview">
    <div id="tgt-toolbar">
      <button id="tgtRefreshBtn" onclick="tgtRefresh()">Refresh</button>
      <label><input type="checkbox" id="tgtLiveToggle" onchange="tgtSetLive(this.checked)"> Live</label>
      <span id="tgt-status"></span>
    </div>
    <div id="tgt-list"><div class="tgt-empty">Click Refresh to read targets from the board.</div></div>
  </div>
  <!-- ── Profile panel ───────────────────────────────────────────── -->
  <!-- Shows the host+AIE timeline.png rendered on the server when a run ends
       (natural finish or Stop run). Cleared and the tab disabled on a new run. -->
  <div id="profileview">
    <div id="profile-status">No timeline yet — it appears after a run finishes.</div>
    <img id="profile-img" class="hide" alt="host + AIE execution timeline"
         onerror="profileImgError()"/>
  </div>
  <!-- collapsible legend — collapsed by default -->
  <details id="legend-detail">
    <summary>Legend &amp; interaction guide</summary>
    <div class="legend">
      <div><span class="sw" style="background:var(--shim)"></span>shim (row 0, host&lt;-&gt;array)</div>
      <div><span class="sw" style="background:var(--core)"></span>core (compute)</div>
      <div>badge = channel dir (S2MM recv / MM2S send)</div>
      <div>click a badge to highlight its flow peers:</div>
      <div><span class="sw" style="background:#30c0d0"></span>receivers (S2MM) of the clicked flow</div>
      <div><span class="sw" style="background:#c050b0"></span>senders (MM2S) of the clicked flow (dashed)</div>
      <div>peer channel badges of the clicked flow:</div>
      <div><span class="sw" style="border:2px dotted #e8943a;background:transparent"></span>source/destination channel (opposite dir, dotted)</div>
      <div><span class="sw" style="border:2px dashed #6cd080;background:transparent"></span>cooperating channel (same dir group, dashed)</div>
    </div>
    <div class="legend-actions">
      <button id="gbtn">Show global / kernel-group code</button>
    </div>
  </details>
  </div>
  <!-- ── run controls ─────────────────────────────────────────── -->
  <div id="lhsplitter" title="Drag to resize (top / bottom)"></div>
  <div id="leftbottom">
  <div id="ctrlbar" data-connected="false">
    <!-- The buttons ride the pane-title row: they are the pane's whole point,
         and under --sim-only with one app they are the ONLY thing left, so a
         title row of their own would be a wasted line. #connstatus is pushed
         right by margin-left:auto and wraps to its own line when the pane is
         too narrow, rather than squeezing the buttons. -->
    <div class="run-head">
      <span class="pane-title">Execution</span>
      <div class="run-actions">
        <button id="testconn" disabled>Connect</button>
        <button id="runbtn" disabled>Run</button>
        <button id="attachbtn" disabled
          title="Attach to a run you started outside this UI (CLI / already-programmed board)">Attach existing run</button>
        <button id="stopbtn" disabled title="Terminate the run owned by this UI">Stop run</button>
      </div>
      <span id="connstatus" data-state="idle">Not connected</span>
    </div>
    <!-- One flex row per selector, below the buttons — Board first, App under
         it. Whole rows, not grid cells, so hiding one (sim-only / single app)
         takes its row-gap with it instead of leaving a seam. -->
    <div class="run-fields">
      <div class="run-row" id="row-board">
        <label class="run-label" for="deviceSel">Target</label>
        <select id="deviceSel">
          <option value="">&mdash; select target &mdash;</option>
<!--__DEVICE_OPTIONS__-->
        </select>
        <input type="text" id="boardHost" class="hide" placeholder="vek385 board hostname">
      </div>
      <div class="run-row" id="row-app">
        <label class="run-label" for="appSel">App</label>
        <select id="appSel"><option value="">&mdash; loading &mdash;</option></select>
        <span id="appinfo" title=""></span>
      </div>
    </div>
    <div id="connhint" class="hide">
      Connection failed. On the target test board, start the hw_server via xsdb:
      <code style="color:#e0def4; background:#1a1622; padding:1px 4px;
            border-radius:3px;">exec hw_server -stcp:0.0.0.0:3121</code>
    </div>
  </div>
  <pre id="console" class="consolebox hide"></pre>
  </div>
</div>
<div id="splitter" title="Drag to resize"></div>
<div id="right">
  <div id="panel-hdr">
    <span class="pane-title">Info</span>
    <div id="panel-itemtabs"></div>
  </div>
  <div id="panel" class="panel">
    <div id="panel-body"><div class="placeholder">Select a tile or net for details.</div></div>
    <div id="panel-toc" class="collapsed"></div>
  </div>
  <div id="rhsplitter" title="Drag to resize (panel / console)"></div>
  <div id="cmdconsole" class="hide">
    <div id="contabs">
      <span class="pane-title">Tools</span>
      <span class="contab act" data-pane="conpane">aiegdb</span>
      <span class="contab" data-pane="llmpane">LLM</span>
      <span class="contab" data-pane="searchpane">Search</span>
    </div>
    <div id="conpane">
    <div id="conhdr" class="conhdr-row">
      <span class="chname">aiegdb</span>
      <span id="contarget" title="current scope">partition</span>
      <span class="chgap"></span>
      <button class="qcmd" id="conpalbtn" title="browse every command for this scope">&#8984; Commands</button>
      <button class="qcmd" onclick="conSend('where')">where</button>
      <button class="qcmd" onclick="conSend('up')">up</button>
      <button class="qcmd" onclick="conSend('top')">top</button>
      <button class="qcmd" id="conclear" title="clear the console">Clear</button>
      <button class="qcmd" id="conreload" title="kill + restart aiegdb.py (reloads edited code)">Reload</button>
    </div>
    <div id="conterm">
      <div id="conpromptline"><span id="conprompt">partition&gt;</span><input id="conin"
          placeholder="type a command (Tab to accept suggestions)"
          autocomplete="off" spellcheck="false">
        <div id="consug" class="hide"></div></div>
      <div id="conout"><div class="con-ln con-dim">(aiegdb console &mdash; click a tile, press &#8984; Commands, or type 'help')</div></div>
    </div>
    <div id="conpal" class="hide">
      <input id="conpalq" placeholder="search commands…" autocomplete="off" spellcheck="false">
      <div id="conpallist"></div>
    </div>
    </div>
    <div id="llmpane" class="hide">
      <div id="conhdr">Claude Code
        <button id="llmreset" title="kill + respawn claude (new conversation)">New chat</button>
        <span class="llm-hint">Ask about anything related to this design, application, or the codebase.</span></div>
      <div id="llmterm">
        <div id="llmmsg"></div>
        <div id="llmthink" class="hide"><span class="llm-dot"></span><span class="llm-dot"></span><span class="llm-dot"></span></div>
        <div id="llminline"><span id="llmprompt">you&gt;</span><div id="llmin"
          contenteditable="true" spellcheck="false"
          data-ph="ask about this design. Enter to send, Shift+Enter for newline"></div>
          <button id="llmsend">Send</button></div>
      </div>
      <div id="llmauth" class="hide">
        <div id="llmauthbox">
          <div class="t">LLM locked</div>
          <input type="password" id="llmauthin" placeholder="password"
                 autocomplete="off">
          <button id="llmauthok">Unlock</button>
          <div id="llmautherr"></div>
        </div>
      </div>
    </div>
    <div id="searchpane" class="hide">
      <div id="sr-inputrow">
        <input id="sr-input" type="text" autocomplete="off" spellcheck="false"
               placeholder="kernel, window, net, GMIO, port, len&hellip;">
        <div id="sr-suggest" class="hide"></div>
      </div>
      <div id="sr-chips"></div>
      <div id="sr-results"><div class="sr-empty">Type above to search, Enter or click a suggestion to pin a term and see results.</div></div>
    </div>
  </div>
</div>
<script>
const DATA = /*__DATA__*/ null;
const CAPS = Object.assign({host_lines:true, ir:true, kernel_code:true},
                           (DATA && DATA.capabilities) || {});
// aiegdb's command grammar, baked in at render time from aiegdb.COMMAND_SPEC.
// The daemon also serves a live copy at /aiegdb/spec; this copy is what makes
// the console's autocomplete work in the standalone (daemon-less) HTML.
const GDBSPEC_STATIC = /*__GDBSPEC__*/ null;
// What the user currently has open, mirrored to the daemon so the embedded
// agent can answer questions about the view in front of the human. Declared
// here so it precedes every reportUIState() call site.
const UISTATE = {view:'grid', selected_tile:null, tile_tab:null, net_tab:null,
                 console_pane:'conpane', flow:null, channel:null, search:null,
                 source:null};

function esc(s){ return (s==null?'':(''+s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
const KIND_LABEL = {bd_config:'BD', createio:'IO', startio:'START', wait:'WAIT',
  lock:'LOCK', enable_ooo:'OOO', loop:'FOR', loopidx:'IDX', loopend:'END'};
function renderRelevant(t){
  const det = t.relevant_detail || [];
  if (!det.length) return '<div class="placeholder">(no relevant lines)</div>';
  return det.map(d => {
    const badges = (d.kinds||[]).map(k =>
      '<span class="kb '+k+'">'+(KIND_LABEL[k]||k)+'</span>').join('');
    const note = d.note ? '<span class="rnote">// '+esc(d.note)+'</span>' : '';
    // Comment-first layout: parameter explanation, then the code line.
    const pp = paramPretty(d.code, d.bd_comment);
    let html = pp ? '<pre class="bdpretty">'+esc(pp)+'</pre>' : '';
    html += '<div class="rline"><span class="lno">L'+d.line+'</span>'+badges+hl(d.code)+note+'</div>';
    return html;
  }).join('');
}
// Kernel channel<->argument mapping (High level tab, core tiles). Renders the
// channel->window/arg correlation table plus the relevant kernel.cc slice so the
// user can match each DMA channel to the kernel input/output parameter.
// Per-tile kernel/bcf when present (baremetal: one kernel per core), else the
// global fallback (native aiehlc: shared kernel).
function tileKernel(t){ return (t && t.kernel) || DATA.kernel; }
function tileBcf(t){ return (t && t.bcf) || DATA.bcf; }
function tileHasKernel(t){
  return !!(t && t.type === 'core' &&
    (((t.high_level||{}).kernel) || t.kernel));
}
function renderKernelMatch(t){
  const km = t.high_level && t.high_level.kernel_match;
  const kv = tileKernel(t);
  if(!km || !kv) return '';
  const rows = (km.matches||[]).map(m => {
    const mcls = m.method==='addr'?'mlock':(m.method==='addr(prov)'?'morder':'mnone');
    const win = m.window ? '<span class="win">'+esc(m.window)+'</span>' : '<span class="mnone">(unmatched)</span>';
    const arg = (m.arg!=null) ? ('arg'+m.arg) : '\u2014';
    const adr = (m.addrs_hex||[]).join('/') || '\u2014';
    const bsy = (m.bcf_syms||[]).join('/') || '\u2014';
    return '<tr>'+
      '<td>'+esc(m.direction)+esc(''+m.channel)+'</td>'+
      '<td>'+esc(adr)+'</td>'+
      '<td class="arrow">\u2192</td>'+
      '<td>'+win+'</td>'+
      '<td>'+arg+'</td>'+
      '<td>'+esc(bsy)+'</td>'+
      '<td class="'+mcls+'">'+esc(m.method)+'</td>'+
    '</tr>';
  }).join('');
  const table =
    '<table class="kmap"><thead><tr>'+
      '<th>channel</th><th>bd addr(s)</th><th></th><th>window</th>'+
      '<th>arg</th><th>bcf buffer</th><th>via</th>'+
    '</tr></thead><tbody>'+rows+'</tbody></table>';
  const src = kv.source;
  const fn = kv.function
    ? '<div class="kv"><b>kernel fn:</b> '+esc(kv.function)+
      (src&&src.file?' <span class="lref">('+esc(src.file)+')</span>':'')+'</div>'
    : '';
  const hint = src
    ? '<div class="lref">see the <b>'+esc(src.file)+'</b> sub-tab in Low level for the kernel body; click a channel to isolate its argument\u2019s code.</div>'
    : '';
  return table+fn+hint;
}
// Kernel function source (Low level tab, core tiles). For a focused channel it
// isolates the "code piece" = the lines that directly use the mapped parameter
// (win_a/win_b/win_c) and offers a "Show all" toggle that reveals the whole
// function with those lines highlighted. For a tile it shows the full body.
function renderKernelSource(t, ch, focused){
  const kv = tileKernel(t);
  const src = kv && kv.source;
  if(!src) return '<div class="placeholder">(no kernel source found)</div>';
  const km = t.high_level && t.high_level.kernel_match;
  let refset = null, param = null, argn = null;
  if(focused && km){
    const m = (km.matches||[]).find(x =>
      x.direction===ch.direction && x.channel===ch.channel);
    if(m && m.arg!=null && src.params && src.params[m.arg]){
      param = src.params[m.arg];
      refset = {}; (param.ref_lines||[]).forEach(n => refset[n]=1);
      argn = m.arg;
    }
  }
  const hdr = '<div class="lref">kernel: <b class="srcref" data-p="'+esc(src.path||'')+'"'
    +' data-l="'+(src.start_line||'')+'">'+esc(src.file)+'</b> &mdash; '+
    esc(src.function)+'() lines '+src.start_line+'-'+src.end_line+'</div>';
  const hs = hlSeq();
  const full = (src.lines||[]).map(r => {
    const hlcls = (refset && refset[r.line]) ? ' khl' : '';
    return '<div class="rline'+hlcls+'"><span class="lno">L'+r.line+'</span>'+
      hs(r.code)+'</div>';
  }).join('');
  if(focused && refset && param){
    // Show the related lines; fold the rest into "//line a-b" markers.
    const piece = renderKFolded(src.lines||[], refset, 'src', hlSeq());
    return hdr +
      '<div class="kv">channel <b>'+esc(ch.direction)+esc(''+ch.channel)+
        '</b> \u2192 arg'+argn+' <span class="win">'+esc(param.name)+'</span> ('+
        esc(param.dir)+')</div>' +
      '<button class="kshowall">Show all</button>' +
      '<div class="kern-piece">'+
        (piece||'<div class="placeholder">(no direct uses found)</div>')+'</div>' +
      '<div class="kern-full hide">'+full+'</div>';
  }
  return hdr + '<div class="kern-full">'+full+'</div>';
}
// Resolve the kernel window a focused channel maps to (via kernel_match), or
function focusedWindow(t, ch, focused){
  if(!focused || !ch) return null;
  const km = t.high_level && t.high_level.kernel_match;
  if(!km) return null;
  const m = (km.matches||[]).find(x =>
    x.direction===ch.direction && x.channel===ch.channel);
  if(!m || !m.window) return null;
  const kv = tileKernel(t);
  return ((kv && kv.windows) || [])
    .find(w => w.name===m.window) || null;
}
// "kernel.cc" sub-tab: the generated wrapper (window_init / LOCK #defines /
// buffer decls). When a channel is focused, highlight its window's lines.
function renderKernelCC(t, ch, focused){
  const k = tileKernel(t);
  if(!k || !k.kernel_lines) return '<div class="placeholder">(no kernel.cc found)</div>';
  const win = focusedWindow(t, ch, focused);
  const hset = {};
  if(win){
    (win.lock_lines||[]).forEach(n => hset[n]=1);
    (win.buf_decl_lines||[]).forEach(n => hset[n]=1);
    if(win.def_line) hset[win.def_line]=1;
    if(win.init_line) hset[win.init_line]=1;
  }
  const hdr = '<div class="lref">kernel: <b class="srcref" data-p="'+esc(k.path||'')+'"'
    +(win&&win.def_line?' data-l="'+win.def_line+'"':'')+'>'+esc(k.file||'kernel.cc')+'</b>'+
    (win ? ' &mdash; window <span class="win">'+esc(win.name)+'</span>'+
      ' (buffers '+esc((win.buffers||[]).join(', ')||'-')+')' : '')+'</div>';
  // Focused (window matched): show only the window's lines, fold the rest.
  const hs = hlSeq();
  const body = win
    ? renderKFolded(k.kernel_lines||[], hset, 'kcc', hs)
    : (k.kernel_lines||[]).map(r =>
        '<div class="rline"><span class="lno">L'+r.line+'</span>'+
        hs(r.code)+'</div>').join('');
  return hdr + '<div>'+body+'</div>';
}
// "*.bcf" sub-tab: buffer name -> tile address map. When a channel is focused,
// highlight the address lines of that window's ping/pong buffers.
function renderBcf(t, ch, focused){
  const b = tileBcf(t);
  if(!b || !b.lines) return '<div class="placeholder">(no .bcf found)</div>';
  const win = focusedWindow(t, ch, focused);
  const hset = {};
  if(win){
    const bufs = {}; (win.buffers||[]).forEach(x => bufs[x]=1);
    (b.symbols||[]).forEach(s => { if(bufs[s.name]) hset[s.line]=1; });
  }
  const hdr = '<div class="lref">bcf: <b class="srcref" data-p="'+esc(b.path||'')+'">'
    +esc(b.file)+'</b>'+
    (win ? ' &mdash; buffers for <span class="win">'+esc(win.name)+'</span>' : '')+'</div>';
  // Focused (window matched): show only this window's buffer lines, fold the rest.
  const hs = hlSeq();
  const body = win
    ? renderKFolded(b.lines||[], hset, 'bcf', hs)
    : (b.lines||[]).map(r =>
        '<div class="rline"><span class="lno">L'+r.line+'</span>'+
        hs(r.code)+'</div>').join('');
  return hdr + '<div>'+body+'</div>';
}
// Render a list of {line,code} rows with only the highlighted (hset) lines shown
// and each run of non-related lines collapsed into a clickable "//line a-b"
// fold marker (default collapsed). `prefix` keeps fold ids unique across the
// stacked kernel-code sections. Reuses the shared fold machinery (wireFolds /
// setFold / foldLabel).
function renderKFolded(lines, hset, prefix, h){
  const N = lines.length;
  const hf = h || hl;
  const row = r => '<div class="rline'+(hset[r.line]?' khl':'')+'">'+
    '<span class="lno">L'+r.line+'</span>'+hf(r.code)+'</div>';
  let out = [], i = 0, fid = 0;
  while(i < N){
    const r = lines[i];
    if(hset[r.line]){ out.push(row(r)); i++; continue; }
    let j = i; while(j < N && !hset[lines[j].line]) j++;   // hidden run [i, j)
    let run = ''; for(let k = i; k < j; k++) run += row(lines[k]);
    const cnt = j - i, a = lines[i].line, b = lines[j-1].line;
    const id = prefix + '-' + (++fid);
    out.push('<span class="irfold" data-fold="'+id+'" data-cnt="'+cnt+
             '" data-a="'+a+'" data-b="'+b+'">'+foldLabel(cnt, true, a, b)+'</span>'+
             '<span class="irhidden hide" data-fold="'+id+'">'+run+'</span>');
    i = j;
  }
  return out.join('');
}
function renderCodeSection(title, html, open){
  return '<details class="codesec"'+(open?' open':'')+'>'+
    '<summary>'+title+'</summary><div class="codesec-body">'+html+'</div></details>';
}
function renderKernelCode(t, ch, focused, openFirst){
  const sections = [];
  const kv = tileKernel(t), b = tileBcf(t);
  if(kv && kv.source)
    sections.push([(kv.source.file||'kernel source')+' — kernel source',
                   renderKernelSource(t, ch, focused)]);
  if(kv && kv.kernel_lines)
    sections.push([(kv.file||'kernel.cc')+' — generated wrapper',
                   renderKernelCC(t, ch, focused)]);
  if(b && b.lines)
    sections.push([(b.file||'.bcf')+' — buffer address map',
                   renderBcf(t, ch, focused)]);
  if(!sections.length)
    return '<div class="placeholder">(no kernel code found)</div>';
  return sections.map((s, i) =>
    renderCodeSection(esc(s[0]), s[1], !!openFirst)
  ).join('');
}
function renderTileCodeKernelFirst(t, ch, focused, banner){
  const hasKernel = tileHasKernel(t);
  const kv = hasKernel ? tileKernel(t) : null;
  const b = hasKernel ? tileBcf(t) : null;
  let out = banner || '';
  let any = false;
  if(kv && kv.source){
    const f = kv.source.file || 'kernel source';
    out += renderCodeSection('Kernel source &mdash; '+esc(f)+
      (kv.source.function?' &mdash; '+esc(kv.source.function)+'()':''),
      renderKernelSource(t, ch, focused), true);
    any = true;
  }
  if(kv && kv.kernel_lines){
    out += renderCodeSection('Generated wrapper &mdash; '+esc(kv.file||'kernel.cc'),
      renderKernelCC(t, ch, focused), true);
    any = true;
  }
  if(b && b.lines){
    out += renderCodeSection('Buffer address map &mdash; '+esc(b.file||'.bcf'),
      renderBcf(t, ch, focused), true);
    any = true;
  }
  if(!any)
    out += '<div class="placeholder">(no code for this tile &mdash; '+
      esc(t.type)+' tiles carry no kernel; see the Schedule tab for its '+
      'transfers)</div>';
  return out;
}
function renderCodeFileTabs(t, ch, focused, defaultHtml, hostHtml){
  const hasKernel = tileHasKernel(t);
  const kv = hasKernel ? tileKernel(t) : null;
  const b = hasKernel ? tileBcf(t) : null;
  const views = [{key:'default', label:'Default', path:'', html:defaultHtml}];
  if(CAPS.host_lines && hostHtml)
    views.push({key:'host', label:'host.cc',
                path:(DATA.source||{}).host_cc||'', html:hostHtml});
  if((kv && (kv.source || kv.kernel_lines)) || (b && b.lines))
    views.push({key:'kernel', label:'Kernel files', path:'',
                html:renderKernelCode(t, ch, focused, true)});
  const tabs = views.map((v, i) =>
    '<button type="button" class="subtab codefile-tab'+(i?'':' act')+
    '" data-codefile="'+v.key+'" title="'+
    esc(v.path||v.label).replace(/"/g,'&quot;')+'">'+
    esc(v.label)+'</button>').join('');
  const bodies = views.map((v, i) =>
    '<div class="codefile-view'+(i?' hide':'')+'" data-codefile="'+v.key+'">'+
    v.html+'</div>').join('');
  return '<div class="subtabs codefile-tabs" role="tablist" '+
    'aria-label="Code files">'+tabs+'</div>'+bodies;
}
const HL_TOK = new RegExp([
  '(\\/\\*[\\s\\S]*?(?:\\*\\/|$)|\\/\\/.*$)',                        // 1 comment
  '("(?:[^"\\\\]|\\\\.)*"|\'(?:[^\'\\\\]|\\\\.)*\')',                // 2 string
  '(^\\s*#\\s*[a-z_]+)',                                             // 3 preproc
  '\\b(__Runtime_\\w+|XAie_\\w+|__runtime_[a-z_]+|_symbol|_stack|_reserved)\\b',
  '\\b(alignas|auto|bool|break|case|catch|class|const|constexpr|continue|' +
    'default|delete|do|else|enum|explicit|extern|false|for|goto|if|inline|' +
    'namespace|new|nullptr|operator|private|protected|public|register|' +
    'return|sizeof|static|struct|switch|template|this|throw|true|try|' +
    'typedef|typename|union|using|virtual|volatile|while)\\b',       // 5 keyword
  '\\b(char|double|float|int|long|short|signed|unsigned|void|size_t|' +
    'ptrdiff_t|u?int(?:8|16|32|64)_t)\\b',                           // 6 type
  '\\b(0[xX][0-9a-fA-F]+|\\d+(?:\\.\\d+)?(?:[uUlLfF]+)?)\\b',        // 7 number
].join('|'), 'g');
function hlOpen(code, open){
  let s = (code == null ? '' : '' + code), pre = '';
  if(open){
    const e = s.indexOf('*/');
    if(e < 0) return {html: '<span class="cm">'+esc(s)+'</span>', open: true};
    pre = '<span class="cm">'+esc(s.slice(0, e+2))+'</span>';
    s = s.slice(e+2);
  }
  let stillOpen = false;
  const html = esc(s).replace(HL_TOK, (m, c, str, p, f, k, t, n) => {
    if(c){
      if(c.slice(0,2) === '/*' && c.slice(-2) !== '*/') stillOpen = true;
      return '<span class="cm">' + c + '</span>';
    }
    return str ? '<span class="str">' + str + '</span>' :
           p   ? '<span class="pp">'  + p   + '</span>' :
           f   ? '<span class="fn">'  + f   + '</span>' :
           k   ? '<span class="kw">'  + k   + '</span>' :
           t   ? '<span class="ty">'  + t   + '</span>' :
           n   ? '<span class="num">' + n   + '</span>' : m;
  });
  return {html: pre + html, open: stillOpen};
}
function hl(code){ return hlOpen(code, false).html; }
function hlSeq(){
  let open = false;
  return code => { const r = hlOpen(code, open); open = r.open; return r.html; };
}
// Reformat a "/* DMA BD Config: k=v, ... */" comment into a friendly multi-line block.
// Returns the block text (to be placed in a <pre>), or '' when code is not a BD comment.
function prettyBd(code){
  const m = code.match(/DMA BD Config:\s*([\s\S]*?)\*\//);
  if(!m) return '';
  const kv = {};
  m[1].split(',').forEach(p=>{
    const i = p.indexOf('=');
    if(i>0) kv[p.slice(0,i).trim()] = p.slice(i+1).trim();
  });
  const g = k => kv[k];
  let o = '/* DMA BD Config \u2014 bd_id='+g('bd_id')+'\n *\n';
  o += ' *   Packet:      enable_packet='+g('enable_packet')+', packet_id='+g('packet_id')+'\n';
  o += ' *   Chaining:    next_bd='+g('next_bd')+', ooo_bd_id='+g('ooo_bd_id')+'\n';
  o += ' *   Locks:       acquire_lock_id='+g('acquire_lock_id')+',  acquire_lock_val='+g('acquire_lock_val')+'\n';
  o += ' *                release_lock_id='+g('release_lock_id')+',  release_lock_val='+g('release_lock_val')+'\n';
  o += ' *\n';
  o += ' *   Transfer:    len='+g('len')+'\n';
  const nd = g('num_dims')!=null ? parseInt(g('num_dims'),10) : 0;
  if(nd>0){
    o += ' *   Dimensions:  num_dims='+nd+'\n';
    const sstr = [];
    for(let i=0;i<nd;i++) sstr.push('stride='+g('d'+i+'_stride')+',');
    const w = Math.max.apply(null, sstr.map(s=>s.length));
    for(let i=0;i<nd;i++)
      o += ' *                d'+i+': '+sstr[i].padEnd(w)+' wrap='+g('d'+i+'_wrap')+'\n';
  }
  if(g('iter_step_size')!=null)
    o += ' *   Iteration:   step_size='+g('iter_step_size')+', wrap='+g('iter_wrap')+'\n';
  o += ' */';
  return o;
}
// Positional parameter names + one-line descriptions for the DMA createio
// runtime calls, so the "Relevant only" view can explain each argument.
const CREATEIO_PARAMS = {
  '__Runtime_dma_createio_4': ['tile_loc','dma_desc','channel_id','bd_id','direction'],
  '__Runtime_dma_createio':   ['tile_loc','dma_desc','channel_id','bd_id','direction','mem'],
};
const CREATEIO_DESC = {
  tile_loc:   'col,row of the owning tile',
  dma_desc:   'pre-built DMA buffer descriptor',
  channel_id: 'DMA channel index on the tile',
  bd_id:      'buffer-descriptor ID for this transfer',
  direction:  'DMA_MM2S: tile mem->stream, DMA_S2MM: stream->tile mem',
  mem:        'optional backing XAie_MemInst (NULL if none)',
};
// Split a call's argument list on top-level commas (ignore nested ()/[]).
function splitTopCommas(s){
  const out=[]; let depth=0, cur='';
  for(const ch of s){
    if(ch==='('||ch==='['||ch==='{') depth++;
    else if(ch===')'||ch===']'||ch==='}') depth--;
    if(ch===','&&depth===0){ out.push(cur); cur=''; } else cur+=ch;
  }
  if(cur.trim()!=='') out.push(cur);
  return out;
}
// Reformat a __Runtime_dma_createio[_4](...) call into a friendly block that
// names each positional argument. Returns '' when code is not such a call.
function createioPretty(code){
  const m = code.match(/(__Runtime_dma_createio(?:_4)?)\s*\(([\s\S]*?)\)\s*;/);
  if(!m) return '';
  const names = CREATEIO_PARAMS[m[1]];
  if(!names) return '';
  const args = splitTopCommas(m[2]).map(a=>a.trim());
  let o = '/* '+m[1]+' \u2014 argument map\n *\n';
  const w = Math.max.apply(null, names.map(n=>n.length));
  for(let i=0;i<names.length;i++){
    const val = (i<args.length) ? args[i] : '\u2014';
    o += ' *   '+names[i].padEnd(w)+' = '+val+
         '   ('+(CREATEIO_DESC[names[i]]||'')+')\n';
  }
  o += ' */';
  return o;
}
// Positional parameter names + one-line descriptions for the startio runtime
// calls (queue a BD chain on a DMA channel and return a wait handle).
// Signature: __Runtime_startio(dev, io, bd_id, repeat).
const STARTIO_PARAMS = {
  '__Runtime_startio':    ['dev','io','bd_id','repeat'],
  '_Runtime_startio_ooo': ['dev','io','bd_id','repeat'],
};
const STARTIO_DESC = {
  dev:    'XAie device instance handle',
  io:     'IO channel handle from createio (tile, channel, direction)',
  bd_id:  'starting buffer-descriptor ID enqueued on the channel',
  repeat: 'number of times the BD (chain) is started (repeat count)',
};
// Reformat a __Runtime_startio[_ooo](...) call into a friendly block that names
// each positional argument. Returns '' when code is not such a call.
function startioPretty(code){
  const m = code.match(/(__Runtime_startio|_Runtime_startio_ooo)\s*\(([\s\S]*?)\)\s*;/);
  if(!m) return '';
  const names = STARTIO_PARAMS[m[1]];
  if(!names) return '';
  const args = splitTopCommas(m[2]).map(a=>a.trim());
  let o = '/* '+m[1]+' \u2014 argument map\n *\n';
  const w = Math.max.apply(null, names.map(n=>n.length));
  for(let i=0;i<names.length;i++){
    const val = (i<args.length) ? args[i] : '\u2014';
    o += ' *   '+names[i].padEnd(w)+' = '+val+
         '   ('+(STARTIO_DESC[names[i]]||'')+')\n';
  }
  o += ' */';
  return o;
}
// First matching parameter-explanation block for a host.cc line: DMA BD Config,
// createio, or startio. `bdComment` is the attached "/* DMA BD Config */" text
// when the entry carries one. Returns '' when no decoder applies. The layout is
// comment-first (this block precedes the code line it explains).
function paramPretty(code, bdComment){
  return prettyBd(bdComment||code) || createioPretty(code) || startioPretty(code);
}
// Render a host.cc block from per-line {line,code} records, prefixing each line
// with its host.cc line number and injecting a friendly bdpretty block BEFORE
// every line that a decoder recognizes (comment explains the parameters, then
// the code line follows).
function renderFullBlock(codeLines, params){
  if(!codeLines || !codeLines.length) return '<pre class="code">(no lines attributed)</pre>';
  const byVar = {};
  (params||[]).forEach(p => { byVar[p.var] = p; });
  let out = '', buf = [], prev = null;
  const flush = () => { if(buf.length){ out += '<pre class="code">'+buf.join('\n')+'</pre>'; buf = []; } };
  codeLines.forEach(item => {
    // Non-contiguous jump: mark the elided host.cc lines with a gap comment.
    if(prev !== null && item.line > prev + 1){
      const a = prev + 1, b = item.line - 1;
      const span = (a === b) ? ('line '+a) : ('lines '+a+'-'+b);
      buf.push('<span class="gap">//... ('+span+' hidden)</span>');
    }
    prev = item.line;
    let row = '<span class="lno">L'+item.line+'</span>'+hl(item.code);
    // Inline arg annotation: resolved values of every vN referenced on this
    // line (deduped, skipping the var's own def line and unresolved vars).
    const seen = {}, anns = [];
    (item.code.match(/\bv\d+\b/g) || []).forEach(v => {
      const p = byVar[v];
      if(!p || p.value == null || seen[v]) return;
      if(p.def_line === item.line) return;
      seen[v] = 1;
      anns.push(v+'='+p.value);
    });
    // __runtime_buffer_arg((void*)<dec>): show the address in hex too.
    let bam;
    const bre = /__runtime_buffer_arg\(\s*(?:\(\s*void\s*\*\s*\)\s*)?(\d+)\s*\)/g;
    while((bam = bre.exec(item.code)) !== null){
      const dec = parseInt(bam[1], 10);
      anns.push(bam[1]+'=0x'+dec.toString(16).toUpperCase());
    }
    if(anns.length) row += '<span class="pann">// '+esc(anns.join(', '))+'</span>';
    // Parameter-explanation block precedes the code line it describes.
    const pb = paramPretty(item.code);
    if(pb){ flush(); out += '<pre class="bdpretty">'+esc(pb)+'</pre>'; }
    buf.push(row);
  });
  flush();
  return out;
}
// Render the dfschedule Middle IR slice. Accepts either a string (fallback /
// '(no matching ops)') or an array of {line, code} rows; array rows show the
// ORIGINAL 6_BlueprintToSchedule.mlir line number in a right-aligned gutter
// (context/separator rows carry line=null -> blank gutter, preserving columns).
function renderMiddleIR(mid){
  if (!mid || typeof mid === 'string')
    return '<pre class="code">' + esc(mid || '(no dfschedule IR)') + '</pre>';
  const body = mid.map(it =>
    '<span class="mlno">' + (it.line != null ? it.line : '') + '</span>'
      + esc(it.code)).join('\n');
  return '<pre class="code">' + body + '</pre>';
}

// Render the FULL 6_BlueprintToSchedule.mlir with the given 1-based source line
// numbers highlighted (amber). Runs of non-highlighted lines are folded into a
// clickable "N lines hidden" marker so the target (highlighted) lines are easy
// to locate. Uses the IR text embedded at generation time.
function renderFullIr(highlightLines){
  const ir = (DATA.dfschedule_ir || {});
  const txt = ir.text || '';
  if (!txt) return '<pre class="code">(full dfschedule IR not embedded)</pre>';
  const hi = new Set(highlightLines || []);
  const lines = txt.split('\n');
  const N = lines.length;
  const rowHtml = (code, n) =>
    '<span class="'+(hi.has(n)?'irln irhi':'irln')+'">' +
    '<span class="irlno">'+n+'</span>'+esc(code)+'</span>';
  const hdr = '<div class="lref"><b class="srcref" data-p="'+esc(ir.path||'')+'">'
    +esc(ir.name||'')+'</b> &mdash; '+N+
    ' lines, '+hi.size+' highlighted'+(hi.size?' (non-target folded)':'')+'</div>';
  // No highlights -> nothing to locate; show the whole file unfolded.
  if (!hi.size){
    let all=''; for (let i=0;i<N;i++) all += rowHtml(lines[i], i+1);
    return hdr + '<pre class="code irfull">'+all+'</pre>';
  }
  let out=[], i=0, fid=0;
  while (i<N){
    const n=i+1;
    if (hi.has(n)){ out.push(rowHtml(lines[i], n)); i++; continue; }
    let j=i; while (j<N && !hi.has(j+1)) j++;    // hidden run [i, j)
    let run=''; for (let k=i;k<j;k++) run += rowHtml(lines[k], k+1);
    const cnt=j-i; fid++;
    out.push('<span class="irfold" data-fold="'+fid+'" data-cnt="'+cnt+'">' +
             foldLabel(cnt, true) + '</span>' +
             '<span class="irhidden hide" data-fold="'+fid+'">'+run+'</span>');
    i=j;
  }
  return hdr + '<pre class="code irfull">'+out.join('')+'</pre>';
}

// Inline fold marker label: triangle reflects state so the marker itself is the
// expand/collapse control (no separate row). When a line range (a[,b]) is
// given, prefix a "//line a-b" tag (used by the kernel-code sub-tab to show the
// hidden line span for non-related code).
function foldLabel(cnt, collapsed, a, b){
  const s = cnt > 1 ? 's' : '';
  const rng = (a != null)
    ? '//line ' + ((b != null && b !== a) ? (a + '-' + b) : a) + '   ' : '';
  return collapsed
    ? rng + '\u25b6 \u22ef ' + cnt + ' line' + s + ' hidden \u2014 click to expand'
    : rng + '\u25bc ' + cnt + ' line' + s + ' shown \u2014 click to collapse';
}

// Apply a fold state to one marker/hidden pair (classes + visibility + label).
function setFold(f, h, collapsed){
  h.classList.toggle('hide', collapsed);
  f.classList.toggle('irfold-open', !collapsed);
  const a = (f.dataset.a !== undefined) ? +f.dataset.a : null;
  const b = (f.dataset.b !== undefined) ? +f.dataset.b : null;
  f.textContent = foldLabel(+f.dataset.cnt, collapsed, a, b);
}

// Attach inline fold/expand click handlers within a rendered full-IR container.
// onChange (optional) is called after any toggle so callers can refresh an
// Expand-all/Collapse-all label.
function wireFolds(box, onChange){
  box.querySelectorAll('.irfold').forEach(f => {
    const scope = f.closest('.codesec-body') || box;
    const h = scope.querySelector('.irhidden[data-fold="'+f.dataset.fold+'"]');
    if (!h) return;
    f.onclick = () => {
      setFold(f, h, !h.classList.contains('hide'));   // toggle to opposite state
      if (onChange) onChange();
    };
  });
}

const g = DATA.grid;
document.getElementById('meta').textContent =
  'AIE Debug  ' + g.cols + '\u00d7' + g.rows + ' grid';

// build a loc->tile map
const byLoc = {};
DATA.tiles.forEach(t => byLoc[t.loc[0]+','+t.loc[1]] = t);

// loc -> DOM cell (filled during grid build)
const cellByLoc = {};
// loc -> live status bar element (filled during grid build)
const liveBar = {};

// flow membership: fi -> {send:[[c,r]], recv:[[c,r]]}
const flowMembers = {};
(DATA.flow_summary||[]).forEach(f => {
  const m = flowMembers[f.flow_index] = {send:[], recv:[]};
  (f.entries||[]).forEach(e => {
    (e.io_direction==='MM2S'?m.send:m.recv).push([e.tile_col,e.tile_row]);
  });
});

function clearPeers(){
  document.querySelectorAll('.tile.peer-recv,.tile.peer-send,.tile.flowsel')
    .forEach(el=>el.classList.remove('peer-recv','peer-send','flowsel'));
  document.querySelectorAll('.badge.peerbadge-partner,.badge.peerbadge-coop')
    .forEach(el=>el.classList.remove('peerbadge-partner','peerbadge-coop'));
}
// Highlight the cooperating channel badges of the same flow as the clicked one:
//   partner = opposite direction (the source for an S2MM / the destination for an
//             MM2S) -> dotted amber
//   coop    = same direction, different badge (broadcast peers / ping-pong)
//             -> dashed green
// The clicked badge itself keeps its solid .selbadge and is excluded here.
function highlightPeerBadges(fi, dir, clickedBadge){
  if(fi==null) return;
  document.querySelectorAll('.badge[data-flow="'+fi+'"]').forEach(b=>{
    if(b===clickedBadge) return;
    b.classList.add(b.dataset.role===dir ? 'peerbadge-coop' : 'peerbadge-partner');
  });
}
function highlightFlow(fi, loc, badgeEl){
  clearPeers();
  highlightPeerBadges(fi, badgeEl ? badgeEl.dataset.role : null, badgeEl);
  const fm = flowMembers[fi]; if(!fm) return;
  const mark=(list,cls)=>list.forEach(([c,r])=>{
    const el=cellByLoc[c+','+r]; if(!el) return;
    if(c===loc[0]&&r===loc[1]) el.classList.add('flowsel');
    else el.classList.add(cls);
  });
  mark(fm.recv,'peer-recv');
  mark(fm.send,'peer-send');
}

// grid: columns left->right, rows top (highest) -> bottom (0)
const grid = document.getElementById('grid');
grid.style.gridTemplateColumns = 'repeat(' + g.cols + ', 1fr)';
const rowsDesc = g.row_list.slice().sort((a,b)=>b-a);
rowsDesc.forEach(r => {
  for (let c=0; c<g.cols; c++){
    const t = byLoc[c+','+r];
    const cell = document.createElement('div');
    if (!t){ cell.style.visibility='hidden'; cell.className='tile'; grid.appendChild(cell); continue; }
    cell.className = 'tile ' + t.type;
    let badges = (t.dma_channels||[]).map((ch,ci) =>
      '<span class="badge" data-flow="'+ch.flow_index+'" data-role="'+ch.direction+'" data-idx="'+ci+'">'
        + ch.direction + ch.channel + '</span>').join('');
    const sdBad = (t.dma_channels||[]).some(ch => ch.flow_balance && ch.flow_balance.balanced===false);
    if (sdBad){ cell.classList.add('sdmismatch');
      badges += '<span class="badge sdwarn" title="supply/demand mismatch">&#9888;</span>'; }
    cell.innerHTML = '<div class="loc">('+t.loc[0]+','+t.loc[1]+')</div>' +
                     '<div>'+t.type+'</div>' + badges;
    cell.title = (t.high_level.contracts||[]).join('\n') || t.type;
    if (sdBad) cell.title += '\n\u26a0 supply/demand mismatch on a flow';
    cellByLoc[c+','+r] = cell;
    const lbar = document.createElement('div');
    lbar.className = 'livebar hide';
    cell.appendChild(lbar);
    liveBar[c+','+r] = lbar;
    cell.onclick = (ev) => {
      const ctrl = ev.ctrlKey || ev.metaKey || ctrlHeld;
      const b = ev.target.closest('.badge');
      if (b && b.dataset.idx !== undefined) {
        const ch = (t.dma_channels||[])[+b.dataset.idx];
        select(t, cell, ch, b, ctrl);
        highlightFlow(+b.dataset.flow, t.loc, b);
      } else {
        select(t, cell, null, null, ctrl);
        clearPeers();
      }
    };
    grid.appendChild(cell);
  }
});

// ── View switcher (Grid / Device Map / Targets) ──────────────
// Device Map is the default view; the initial call at the bottom of this script
// runs through here too, so the shown/hidden state has exactly one definition.
function switchView(name){
  // The Profile tab's enabled state is owned by the run lifecycle, not by which
  // view is showing. Re-assert it from LIVE.profileReady on every switch so a
  // view change (Device Map, Host, Grid) can never leave the tab stuck disabled
  // once a timeline has rendered — it clears only when the next Run starts.
  if(LIVE.profileReady) setProfileTabEnabled(true);
  // The Profile tab is inert until a run renders a timeline — ignore clicks
  // on it while disabled so a disabled tab can never take over the view.
  if(name==='profile'){
    const t=document.getElementById('vsw-profile');
    if(t && t.classList.contains('vsw-disabled')) return;
  }
  document.querySelectorAll('.vsw').forEach(b=>b.classList.toggle('act', b.dataset.v===name));
  const showGrid    = name==='grid';
  const showMap     = name==='map';
  const showTargets = name==='targets';
  const showProfile = name==='profile';
  document.getElementById('grid').style.display = showGrid ? '' : 'none';
  document.getElementById('overlayctl').style.display = showGrid ? '' : 'none';
  document.getElementById('legend-detail').style.display = showGrid ? '' : 'none';
  document.getElementById('devmap').classList.toggle('show', showMap);
  document.getElementById('targetsview').classList.toggle('show', showTargets);
  document.getElementById('profileview').classList.toggle('show', showProfile);
  if(showMap) buildDeviceMap();
  if(showTargets) tgtMaybeAutoRefresh();
  reportUIState({view: name});
}

// ── Profile tab (host+AIE timeline.png) ───────────────────────────
// Enable/disable the Profile tab; disabling also flips the view back to Grid
// if the Profile panel is the one on screen.
function setProfileTabEnabled(on){
  const t=document.getElementById('vsw-profile');
  if(!t) return;
  t.classList.toggle('vsw-disabled', !on);
  if(!on && t.classList.contains('act')) switchView('grid');
}
// A run started: wipe any prior timeline image and lock the Profile tab.
function clearTimelineProfile(){
  const img=document.getElementById('profile-img');
  const st =document.getElementById('profile-status');
  if(img){ img.removeAttribute('src'); img.classList.add('hide'); }
  if(st){ st.textContent='Run in progress — timeline will appear when it finishes.'; }
  // Clear the flag BEFORE disabling: setProfileTabEnabled(false) may bounce the
  // view to Grid via switchView, which re-reads this flag.
  LIVE.profileReady = false;
  setProfileTabEnabled(false);
}
// A run ended and the server rendered timeline.png: show it and unlock the tab.
function showTimelineProfile(){
  const img=document.getElementById('profile-img');
  const st =document.getElementById('profile-status');
  if(img){ img.src='/timeline.png?ts='+Date.now(); img.classList.remove('hide'); }
  if(st){ st.textContent=''; }
  // Latch the tab ON: from here it stays enabled across any view switch until
  // the next Run calls clearTimelineProfile().
  LIVE.profileReady = true;
  setProfileTabEnabled(true);
}
function profileImgError(){
  const img=document.getElementById('profile-img');
  const st =document.getElementById('profile-status');
  if(img) img.classList.add('hide');
  if(st) st.textContent='timeline.png could not be loaded.';
}

// ── Targets panel ─────────────────────────────────────────────
const TGT = { live: false, timer: null, loading: false, data: [] };

function tgtSetLive(on){
  TGT.live = on;
  clearInterval(TGT.timer);
  TGT.timer = null;
  if(on){ tgtRefresh(); TGT.timer = setInterval(tgtRefresh, 4000); }
}

function tgtMaybeAutoRefresh(){
  // Refresh once when first opening the panel if there are no results yet.
  const list = document.getElementById('tgt-list');
  if(list && list.querySelector('.tgt-empty')) tgtRefresh();
}

function tgtRefresh(){
  if(TGT.loading) return;
  TGT.loading = true;
  const st = document.getElementById('tgt-status');
  if(st) st.textContent = 'Reading…';
  const btn = document.getElementById('tgtRefreshBtn');
  if(btn) btn.disabled = true;
  const dev  = deviceSel   ? deviceSel.value            : '';
  const host = boardHost   ? boardHost.value.trim()      : '';
  const base = '?device=' + encodeURIComponent(dev) + '&host=' + encodeURIComponent(host);
  api('/targets' + base)
    .then(data => {
      TGT.loading = false;
      if(btn) btn.disabled = false;
      if(data.error){
        if(st) st.textContent = '⚠ ' + data.error;
        tgtRenderError(data.error);
        return;
      }
      TGT.data = data.targets || [];
      if(data.raw) console.log('[tgtRaw xsdb output]\n' + data.raw);
      tgtRender(TGT.data);
      tgtSyncPanel();

      // Fan out detail reads in parallel for all stopped targets.
      const stopped = TGT.data.filter(t => t.suspended);
      if(!stopped.length){ if(st) st.textContent = 'Updated ' + new Date().toLocaleTimeString(); return; }
      if(st) st.textContent = 'Reading registers (' + stopped.length + ' stopped)…';
      let done = 0;
      stopped.forEach(t => {
        fetch('/targets/detail?id=' + t.id + '&' + base.slice(1))
          .then(r => r.json())
          .then(d => {
            if(!d.error){
              if(d.pc)    t.pc   = d.pc;
              if(d.sp)    t.sp   = d.sp;
              if(d.lr)    t.lr   = d.lr;
              if(d.cpsr)  t.cpsr = d.cpsr;
              if(d.elf)   t.elf  = true;
              if(d.stack && d.stack.length) t.stack = d.stack;
              tgtRender(TGT.data);
              tgtSyncPanel();
            }
          })
          .catch(() => {})
          .finally(() => {
            done++;
            if(done === stopped.length && st)
              st.textContent = 'Updated ' + new Date().toLocaleTimeString();
          });
      });
    })
    .catch(e => {
      TGT.loading = false;
      if(btn) btn.disabled = false;
      if(st) st.textContent = '⚠ ' + e.message;
    });
}

function tgtSyncPanel(){
  TGT.data.forEach(t => {
    const key = 'tgt:' + t.id;
    if(panelItems.has(key)){
      panelItems.get(key).buildBody = () => tgtBuildBody(t);
      panelItems.get(key).llmCtx = tgtLlmCtx(t);
      if(panelActiveKey === key) panelRenderBody(key);
    }
  });
}

// Whether the stack trace carries real symbol names (ELF with debug info loaded).
function tgtHasSymbols(t){
  if(!t.stack || !t.stack.length) return false;
  const f = t.stack[0];
  if(f.raw) return false;
  return f.func && f.func !== '??' && !f.func.startsWith('0x');
}

// Whether any ELF was loaded at all.
// Server sets t.elf=true when memmap returns data; also infer from PC / symbol.
function tgtElfLoaded(t){
  return !!(t.elf || t.pc || tgtHasSymbols(t));
}

// State class drives the left border and badge color.
// running+ELF = green, stopped = amber, stopped+no-symbols = red, no-ELF = grey.
function tgtStateClass(t){
  const s = (t.state||'').toLowerCase();
  // Server-parsed booleans are authoritative; fall back to state string scan.
  const running = t.running  || s.includes('run');
  const stopped = t.suspended || s.includes('sus') || s.includes('stop') || s.includes('halt');
  if(running)  return 'running';
  if(stopped && !tgtHasSymbols(t) && t.stack && t.stack.length) return 'crashed';
  if(stopped)  return 'suspended';
  if(!tgtElfLoaded(t)) return 'noelf';
  return 'other';
}

function tgtBadgeLabel(t){
  const s = (t.state||'').toLowerCase();
  const running = t.running  || s.includes('run');
  const stopped = t.suspended || s.includes('sus') || s.includes('stop') || s.includes('halt');
  if(running) return 'RUNNING';
  if(stopped) return 'STOPPED';
  return (t.state||'IDLE').toUpperCase();
}

// Halt one running core, read its registers, resume, then patch TGT.data
// and re-render the card + Info pane in place.
function tgtHaltRead(e, tid){
  if(e) e.stopPropagation();
  // Disable the button immediately to prevent double-clicks.
  const btn = e && e.currentTarget;
  if(btn){ btn.disabled = true; btn.textContent = 'Halting…'; }

  const params = new URLSearchParams({id: tid});
  const dv = document.getElementById('deviceSel');
  if(dv && dv.value) params.set('device', dv.value);
  const hv = document.getElementById('hostInput');
  if(hv && hv.value) params.set('host', hv.value);

  fetch('/targets/halt_read?' + params)
    .then(r => r.json())
    .then(data => {
      if(data.error){
        if(btn){ btn.disabled = false; btn.textContent = 'Halt & read registers'; }
        alert('Halt-read failed: ' + data.error);
        return;
      }
      // Merge result into the target object in TGT.data.
      const t = TGT.data.find(x => x.id === tid);
      if(!t) return;
      if(data.pc)    t.pc   = data.pc;
      if(data.sp)    t.sp   = data.sp;
      if(data.lr)    t.lr   = data.lr;
      if(data.cpsr)   t.cpsr   = data.cpsr;
      if(data.elf)    t.elf    = true;
      if(data.pc_loc) t.pc_loc = data.pc_loc;
      if(data.pc_sym) t.pc_sym = data.pc_sym;
      if(data.stack && data.stack.length) t.stack = data.stack;
      // Re-render the full list (preserves open state) and refresh Info pane.
      tgtRender(TGT.data);
      if(panelActiveKey === 'tgt:' + tid) tgtSelectCard(t);
    })
    .catch(err => {
      if(btn){ btn.disabled = false; btn.textContent = 'Halt & read registers'; }
      alert('Halt-read error: ' + err);
    });
}

function tgtRenderError(msg){
  const list = document.getElementById('tgt-list');
  if(!list) return;
  list.innerHTML = '<div class="tgt-empty">Error: ' + esc(msg) + '</div>';
}

// Build a tree from the flat targets list.
//
// Primary strategy: use xsdb indent depth (works when xsdb emits explicit
// group rows like "APU" / "RPU" at indent 0 with cores indented under them).
//
// Fallback strategy: if all entries have the same indent (flat list — xsdb
// version that emits only leaf cores), group by core-family prefix:
//   "Cortex-A*"  → APU cluster
//   "Cortex-R*"  → RPU cluster
//   "Cortex-M*"  → MCU cluster
//   anything else → top-level
//
// Either way every cluster node auto-collapses when all descendants are idle.
function tgtBuildTree(targets){
  if(!targets.length) return [];

  // --- Primary: indent-based tree ---
  const allSameIndent = targets.every(t => (t.indent || 0) === (targets[0].indent || 0));
  if(!allSameIndent){
    const nodes = targets.map(t => ({ t, children: [] }));
    const roots = [];
    const stack = []; // [{indent, node}]
    for(const node of nodes){
      const ind = node.t.indent || 0;
      while(stack.length && stack[stack.length-1].indent >= ind) stack.pop();
      if(stack.length) stack[stack.length-1].node.children.push(node);
      else             roots.push(node);
      stack.push({indent: ind, node});
    }
    return roots;
  }

  // --- Fallback: name-prefix grouping for flat lists ---
  function clusterKey(name){
    if(/cortex-a/i.test(name)) return 'APU (Cortex-A)';
    if(/cortex-r/i.test(name)) return 'RPU (Cortex-R)';
    if(/cortex-m/i.test(name)) return 'MCU (Cortex-M)';
    if(/microblaze/i.test(name)) return 'MicroBlaze';
    return null;  // top-level
  }
  const clusterMap = {};   // key → synthetic group node
  const clusterOrder = []; // insertion order
  const roots = [];
  for(const t of targets){
    const key = clusterKey(t.name);
    if(key){
      if(!clusterMap[key]){
        // Synthetic group target: not a real xsdb target, just a label.
        clusterMap[key] = { t: {id: -1, indent: 0, name: key, state: '', running: false,
                                suspended: false, pc: null, sp: null, lr: null,
                                cpsr: null, elf: false, stack: []},
                            children: [], synthetic: true };
        clusterOrder.push(key);
        roots.push(clusterMap[key]);
      }
      clusterMap[key].children.push({ t, children: [] });
    } else {
      roots.push({ t, children: [] });
    }
  }
  // Update synthetic group's running flag to reflect children.
  clusterOrder.forEach(key => {
    const grp = clusterMap[key];
    grp.t.running = grp.children.some(c => c.t.running);
  });
  return roots;
}

function tgtClusterAllIdle(node){
  // True when this subtree has no Running descendants.
  if(node.children.length === 0) return !node.t.running;
  return node.children.every(tgtClusterAllIdle);
}

function tgtRender(targets){
  const list = document.getElementById('tgt-list');
  if(!list) return;
  if(!targets.length){
    list.innerHTML = '<div class="tgt-empty">No targets found.</div>';
    return;
  }
  console.log('[tgtRender] targets:', JSON.stringify(targets.map(t=>({id:t.id,indent:t.indent,name:t.name,running:t.running}))));
  console.log('[tgtRender] tree:', JSON.stringify(tgtBuildTree(targets).map(function s(n){return {id:n.t.id,name:n.t.name,children:n.children.map(s)};})));
  // Preserve open/closed state across live refreshes.
  const bodyOpen     = new Set();
  const clusterOpen  = new Set();
  const clusterClosed = new Set();
  list.querySelectorAll('.tgt-body-det[open]').forEach(d => bodyOpen.add(+d.dataset.id));
  list.querySelectorAll('details.tgt-cluster[open]').forEach(d => clusterOpen.add(d.dataset.cname));
  list.querySelectorAll('details.tgt-cluster:not([open])').forEach(d => clusterClosed.add(d.dataset.cname));

  const roots = tgtBuildTree(targets);
  list.innerHTML = '';

  function renderNode(node, container){
    const t = node.t;
    if(node.children.length > 0){
      // Group node → cluster <details>
      const allIdle = tgtClusterAllIdle(node);
      // User-overridden state wins; default: collapse when all idle.
      let openAttr = '';
      if(clusterOpen.has(t.name))        openAttr = ' open';
      else if(clusterClosed.has(t.name)) openAttr = '';
      else if(!allIdle)                   openAttr = ' open';

      const runningCount = node.children.filter(n => n.t.running).length;
      const badgeCls = runningCount ? ' cl-running' : '';
      const badgeTxt = runningCount
        ? runningCount + ' running'
        : node.children.length + ' core' + (node.children.length !== 1 ? 's' : '') + ' idle';

      const det = document.createElement('details');
      det.className = 'tgt-cluster';
      det.dataset.cname = t.name;
      if(openAttr) det.open = true;

      det.innerHTML = '<summary class="tgt-cluster-hdr">'
        + '<span class="tgt-cluster-arrow">&#9654;</span>'
        + '<span class="tgt-cluster-name">' + esc(t.name) + '</span>'
        + '<span class="tgt-cluster-badge' + badgeCls + '">' + badgeTxt + '</span>'
        + '</summary>';

      const body = document.createElement('div');
      body.className = 'tgt-cluster-body';
      node.children.forEach(child => renderNode(child, body));
      det.appendChild(body);
      container.appendChild(det);
      return;
    }

    // Leaf node → card (same as before, no indent margin needed — nesting handles it)
    const sc  = tgtStateClass(t);
    const lbl = tgtBadgeLabel(t);
    const hasStack = t.stack && t.stack.length;
    const elfLoaded = tgtElfLoaded(t);
    const hasSyms   = tgtHasSymbols(t);

    const card = document.createElement('div');
    card.className = 'tgt-card tc-' + sc;
    card.dataset.id = t.id;
    card.title = 'Click to view in Info pane';
    card.style.cursor = 'pointer';

    // ── Header: badge + name + id + ELF indicator (always visible) ──
    const elfDot = elfLoaded
      ? (hasSyms
          ? '<span class="tgt-elfdot sym" title="ELF loaded, debug symbols present">&#9679;</span>'
          : '<span class="tgt-elfdot nosym" title="ELF loaded, no debug symbols">&#9675;</span>')
      : '<span class="tgt-elfdot none" title="No ELF loaded">&#8212;</span>';
    let html = '<div class="tgt-hdr">'
      + '<span class="tgt-badge ' + sc + '">' + lbl + '</span>'
      + '<span class="tgt-name">' + esc(t.name) + '</span>'
      + elfDot
      + '<span class="tgt-id">#' + t.id + '</span>'
      + '</div>';

    // ── Halt & read button: only for running cores with no register data ──
    if(sc === 'running' && !t.pc){
      html += '<div class="tgt-haltrow">'
        + '<button class="tgt-haltbtn" data-tid="' + t.id + '" onclick="tgtHaltRead(event,' + t.id + ')">'
        + 'Halt &amp; read registers'
        + '</button>'
        + '<span class="tgt-haltnote">briefly halts the core (~100 ms)</span>'
        + '</div>';
    }

    // ── PC row: shown whenever PC data is available ───────────────
    if(t.pc){
      let srcHtml = '';
      // Prefer addr2line-resolved pc_loc (server-side), then top stack frame.
      const pcLoc = t.pc_loc || (hasStack && t.stack[0].file ? t.stack[0] : null);
      if(pcLoc && pcLoc.file){
        srcHtml = '<span class="tgt-pcsrc srcref"'
          + ' data-p="' + esc(pcLoc.file) + '"'
          + ' data-l="' + pcLoc.line + '"'
          + ' title="' + esc(pcLoc.file + ':' + pcLoc.line) + '">'
          + esc(pcLoc.file.split('/').pop()) + ':' + pcLoc.line
          + '</span>';
        if(pcLoc.func && pcLoc.func !== '??')
          srcHtml += '<span class="tgt-pcsrc" style="color:var(--tx-lo)"> in ' + esc(pcLoc.func) + '</span>';
      } else {
        const sym = (t.pc_sym) || (hasStack && t.stack[0].func) || '';
        if(sym) srcHtml = '<span class="tgt-pcsrc" style="color:var(--tx-lo)">in ' + esc(sym) + '</span>';
      }
      html += '<div class="tgt-pcrow">'
        + '<span class="tgt-pclabel">PC</span>'
        + '<span class="tgt-pcval">' + esc(t.pc) + '</span>'
        + srcHtml
        + '</div>';
    }

    // ── Body <details>: open by default when stopped, preserved on refresh ──
    // Always rendered so the user has a clickable element on every card.
    const wasOpen = bodyOpen.has(t.id);
    const openAttr = (t.suspended || wasOpen) ? ' open' : '';
    html += '<details class="tgt-body-det" data-id="' + t.id + '"' + openAttr + '>';

    if(hasStack){
      const fc = t.stack.length;
      html += '<summary class="tgt-btsummary">Stack trace (' + fc + ' frame' + (fc !== 1 ? 's' : '') + ')</summary>';
      html += '<table class="tgt-stack">';
      t.stack.forEach(function(f){
        if(f.raw !== undefined){
          html += '<tr><td colspan="4" class="sf-raw">' + esc(f.raw) + '</td></tr>';
        } else {
          const locStr = f.file ? f.file + ':' + f.line : '';
          const locCell = locStr
            ? '<span class="sf-loc srcref" data-p="' + esc(f.file) + '" data-l="' + f.line + '" title="' + esc(locStr) + '">' + esc(locStr) + '</span>'
            : '';
          html += '<tr>'
            + '<td class="sf-frame">' + f.frame + '</td>'
            + '<td class="sf-addr">' + esc(f.addr || '') + '</td>'
            + '<td class="sf-func">' + esc(f.func || '') + '</td>'
            + '<td>' + locCell + '</td>'
            + '</tr>';
        }
      });
      html += '</table>';
    } else {
      const bodyNote = t.suspended
        ? 'Stopped, no backtrace available.'
        : t.pc ? 'Running, backtrace not captured.' : 'No debug info.';
      html += '<summary class="tgt-btsummary">Details</summary>'
        + '<div class="tgt-nonote">' + bodyNote + '</div>';
    }

    html += '</details>';

    card.innerHTML = html;
    container.appendChild(card);
  }

  roots.forEach(node => renderNode(node, list));
}

// ── Target Info pane integration ──────────────────────────────
// Clicking a target card opens it in the Info pane (top-right), same model
// as tile/net clicks.  The card is highlighted with .tgt-sel; the Info pane
// shows PC, ELF/symbol status, full stack trace with clickable file:line links.

function tgtLlmCtx(t){
  const sc  = tgtStateClass(t);
  const sym = tgtHasSymbols(t) ? 'symbols present' : 'no debug symbols';
  const elf = tgtElfLoaded(t) ? 'ELF loaded' : 'no ELF';
  let ctx = 'Processor ' + t.name + ' #' + t.id + ': ' + (t.state || 'unknown') + ', ' + elf + ', ' + sym;
  if(t.pc){
    ctx += '; PC ' + t.pc;
    const loc = t.pc_loc;
    if(loc && loc.file) ctx += ' (' + loc.file + ':' + loc.line + (loc.func ? ' in ' + loc.func : '') + ')';
    else if(t.pc_sym)   ctx += ' (in ' + t.pc_sym + ')';
  }
  if(t.sp)   ctx += '; SP ' + t.sp;
  if(t.lr)   ctx += '; LR ' + t.lr;
  if(t.cpsr) ctx += '; CPSR ' + t.cpsr;
  if(t.stack && t.stack.length && t.stack[0].func)
    ctx += '; top frame: ' + t.stack[0].func;
  if(t.stack && t.stack.length && t.stack[0].file)
    ctx += ' at ' + t.stack[0].file + ':' + t.stack[0].line;
  return ctx;
}

function tgtBuildBody(t){
  const sc   = tgtStateClass(t);
  const lbl  = tgtBadgeLabel(t);
  const sym  = tgtHasSymbols(t);
  const elf  = tgtElfLoaded(t);
  const hasStack = t.stack && t.stack.length;

  // Status line
  const stateColor = {
    running:'#3fb950', suspended:'#d29922', crashed:'#f85149', noelf:'#484f58', other:'var(--tx-lo)'
  }[sc] || 'var(--tx-lo)';

  let h = '<div class="sec">'
    + '<div class="sec-hdr">' + esc(t.name) + ' &mdash; Target #' + t.id + '</div>'
    + '<div class="kv"><b>State:</b> <span style="color:' + stateColor + ';font-weight:600">' + lbl + '</span>'
    + (t.state ? ' <span style="color:var(--tx-lo);font-size:11px">(' + esc(t.state) + ')</span>' : '') + '</div>'
    + '<div class="kv"><b>ELF:</b> '
    + (elf ? (sym ? '<span style="color:#3fb950">loaded, debug symbols present</span>'
                  : '<span style="color:#d29922">loaded, no debug symbols</span>')
           : '<span style="color:#484f58">not loaded</span>')
    + '</div>'
    + '</div>';

  // Halt & read button in Info pane (running core, no register data yet)
  if(sc === 'running' && !t.pc && !t.sp){
    h += '<div class="sec"><div style="display:flex;align-items:center;gap:10px">'
      + '<button class="tgt-haltbtn" onclick="tgtHaltRead(null,' + t.id + ')">Halt &amp; read registers</button>'
      + '<span style="color:var(--tx-lo);font-size:11px">briefly halts the core (~100 ms) then resumes</span>'
      + '</div></div>';
  }

  // Registers section (PC, SP, LR, CPSR) + source location
  if(t.pc || t.sp || t.lr || t.cpsr){
    // PC source: prefer server addr2line result, fall back to top stack frame.
    const pcLoc = t.pc_loc || (hasStack && t.stack[0].file ? t.stack[0] : null);
    let pcLocHtml = '';
    if(pcLoc && pcLoc.file){
      pcLocHtml = ' &mdash; <span class="srcref" data-p="' + esc(pcLoc.file) + '" data-l="' + pcLoc.line
        + '" style="color:var(--accent-fg);cursor:pointer" title="' + esc(pcLoc.file+':'+pcLoc.line) + '">'
        + esc(pcLoc.file.split('/').pop()) + ':' + pcLoc.line + '</span>';
      if(pcLoc.func && pcLoc.func !== '??')
        pcLocHtml += ' <span style="color:var(--tx-lo)">in ' + esc(pcLoc.func) + '</span>';
    } else {
      const sym = t.pc_sym || (hasStack && t.stack[0].func) || '';
      if(sym) pcLocHtml = ' &mdash; <span style="color:var(--tx-lo)">in ' + esc(sym) + '</span>';
    }
    h += '<div class="sec"><div class="sec-hdr">Registers</div>'
      + '<table style="font-family:ui-monospace,monospace;font-size:12px;border-collapse:collapse">';
    const regs = [['PC',t.pc,pcLocHtml],['SP',t.sp,''],['LR',t.lr,''],['CPSR',t.cpsr,'']];
    regs.forEach(function(r){
      if(!r[1]) return;
      h += '<tr>'
        + '<td style="padding:2px 12px 2px 4px;color:var(--tx-lo);width:4ch">' + r[0] + '</td>'
        + '<td style="padding:2px 8px;color:#a8d8ff;font-weight:600">' + esc(r[1]) + '</td>'
        + '<td style="padding:2px 4px">' + r[2] + '</td>'
        + '</tr>';
    });
    h += '</table></div>';
  }

  // Stack trace
  if(hasStack){
    h += '<div class="sec"><div class="sec-hdr">Stack Trace</div>'
      + '<table style="font-family:ui-monospace,monospace;font-size:11px;width:100%;border-collapse:collapse">';
    t.stack.forEach(function(f, i){
      if(f.raw !== undefined){
        h += '<tr><td colspan="4" style="padding:3px 6px;color:var(--tx-lo);font-style:italic">' + esc(f.raw) + '</td></tr>';
      } else {
        const locStr  = f.file ? f.file + ':' + f.line : '';
        const locCell = locStr
          ? '<span class="srcref" data-p="' + esc(f.file) + '" data-l="' + f.line
            + '" style="color:var(--tx-lo);cursor:pointer" title="' + esc(locStr) + '">' + esc(locStr) + '</span>'
          : '';
        const funcColor = i === 0 ? 'color:var(--accent-fg);font-weight:600' : 'color:var(--tx)';
        h += '<tr style="border-bottom:1px solid var(--bd)">'
          + '<td style="padding:3px 8px 3px 4px;color:var(--tx-lo);text-align:right;width:2ch">' + f.frame + '</td>'
          + '<td style="padding:3px 8px;color:#6e8fa8;width:9ch;white-space:nowrap">' + esc(f.addr||'') + '</td>'
          + '<td style="padding:3px 8px;' + funcColor + '">' + esc(f.func||'??') + '</td>'
          + '<td style="padding:3px 8px">' + locCell + '</td>'
          + '</tr>';
      }
    });
    h += '</table></div>';
  } else if(sc === 'running'){
    // Running and no stack yet — "Halt & read" button already shown above.
    h += '<div class="sec"><div style="color:var(--tx-lo);font-size:12px">Use <b>Halt &amp; read registers</b> above to capture the current stack.</div></div>';
  } else if(!elf && (t.suspended || sc === 'suspended')){
    // Stopped but memmap/ELF probe returned nothing — genuinely no ELF.
    h += '<div class="sec"><div style="color:var(--tx-lo);font-size:12px">No ELF loaded on this processor.</div></div>';
  } else if(t.suspended || sc === 'suspended'){
    h += '<div class="sec"><div style="color:var(--tx-lo);font-size:12px">Stopped but backtrace unavailable (debug symbols may be stripped).</div></div>';
  } else {
    // Unknown / idle container node — no register data expected.
    h += '<div class="sec"><div style="color:var(--tx-lo);font-size:12px">No debug data available for this target.</div></div>';
  }

  return h;
}

function tgtSelectCard(t){
  // Highlight the clicked card.
  document.querySelectorAll('#tgt-list .tgt-card').forEach(c => c.classList.remove('tgt-sel'));
  const clicked = document.querySelector('#tgt-list .tgt-card[data-id="' + t.id + '"]');
  if(clicked) clicked.classList.add('tgt-sel');

  const key = 'tgt:' + t.id;
  const label = t.name + ' #' + t.id;
  panelItems.forEach((_,k) => { if(k.startsWith('tgt:')) panelItems.delete(k); });
  panelItems.set(key, {
    kind: 'tgt',
    label: label,
    color: null,
    buildBody: () => tgtBuildBody(t),
    llmCtx: tgtLlmCtx(t),
  });
  panelActiveKey = key;
  panelSync();
}

// Delegated click on the targets list.
// Cards can be nested inside cluster <details> elements, so we must NOT guard
// on 'details' — that would catch the cluster wrapper and kill every card click.
// We DO guard on 'summary' (the native toggle) and 'button'/'a' (their own actions).
document.getElementById('tgt-list').addEventListener('click', function(e){
  if(e.target.closest('summary, a, button')) return;
  const card = e.target.closest('.tgt-card');
  if(!card) return;
  const id = +card.dataset.id;
  const t = TGT.data.find(x => x.id === id);
  if(t) tgtSelectCard(t);
});

// ── Device Map (SVG, pan/zoom) ────────────────────────────────
const DM_COLORS = [
  '#7BAFE9','#81A1C1','#9386F2','#B48EAD',
  '#3FA266','#F1B467','#DD7F76','#FC6B83',
  '#88C0D0','#A3BE8C','#EBCB8B','#BF616A',
];
// Build from comm_paths so flows that appear in edges but have no core-tile DMA
// entries (e.g. GMIO-to-memtile flows absent from flow_summary) still get a color.
// Declared here: recomputeFlowIds() runs at module scope and reads it.
let ROUTING_SRC = 'static';
let ROUTING_DIFF = false;
let dmFlowIds = [];
function recomputeFlowIds(){
  const base = ROUTING_SRC === 'dynamic'
    ? [] : Object.keys(flowMembers).map(Number);
  dmFlowIds = [...new Set([
    ...base,
    ...(DATA.comm_paths||[]).map(p=>p.flow_index),
  ])].sort((a,b)=>a-b);
}
recomputeFlowIds();
function dmColor(fi){ return DM_COLORS[dmFlowIds.indexOf(fi)%DM_COLORS.length]; }

let dmActiveNets = new Set();   // empty = all nets active (unless dmHideAll)
let dmHideAll = false;          // true = All Flows toggled off
let dmShowSW = true;
// Per-tile routing-info overrides, driven by the tile right-click menu. The
// dmSwToggle checkbox stays the master switch; this set carves individual tiles
// out of it so a dense map can be thinned one tile/row/column at a time.
// Collapsed tiles also drop out of the SW_ROWS max, so collapsing everything in
// view shrinks the boxes instead of leaving a field of empty rows.
let dmSwCollapsed = new Set();  // 'c,r' keys whose routing rows are hidden
let dmBuilt = false;
// One-shot: refit on the next rebuild. Set only by "Reset view" — every other
// rebuild (net chips, tile menu, search, filters) keeps the user's pan/zoom.
let dmRefitNext = false;
// Map tile selection lives at module scope so it survives a buildDeviceMap()
// rebuild (e.g. triggered by a net click) instead of being wiped each time.
let dmSelKeys = new Set();      // 'c,r' keys of highlighted map tiles
const dmTileStroke = {};        // key -> current rect stroke, for deselect/hover-out restore
const dmTileFill = {};          // key -> base rect fill, for status-tint restore

// Live scan results for the map. Same payload the grid overlay consumes, kept
// at module scope because buildDeviceMap() wipes the SVG — the paint pass is
// re-run after every rebuild instead of the colors living only in the DOM.
// nets[] is derived, not served: /grid reports per-tile and per-channel state,
// so a flow's state is the worst state across the DMA channels carrying it.
let dmStatus = { what:null, cells:null, nets:{}, err:null, ts:null };

// Pan/zoom state
let dmTx=20, dmTy=20, dmScale=0.9, dmDragging=false, dmLx=0, dmLy=0;
let dmSpaceHeld=false;

// Ctrl state captured at mousedown (before Linux keyup races with click).
let ctrlHeld=false;
document.addEventListener('keydown',  e=>{ if(e.key==='Control'||e.key==='Meta') ctrlHeld=true; });
document.addEventListener('keyup',    e=>{ if(e.key==='Control'||e.key==='Meta') ctrlHeld=false; });
window.addEventListener('blur',       ()=>{ ctrlHeld=false; });
document.addEventListener('mousedown',e=>{ ctrlHeld = e.ctrlKey||e.metaKey; });

function dmApply(){
  document.getElementById('devmap-canvas').style.transform=
    'translate('+dmTx+'px,'+dmTy+'px) scale('+dmScale+')';
}
// Centre the map at fit-zoom. Reads the SVG's current width/height, so it must
// run *after* a rebuild has resized it, never before.
function dmFitView(){
  const vp=document.getElementById('devmap-vp');
  const svg=document.getElementById('devmap-svg');
  const vpW=vp.clientWidth, vpH=vp.clientHeight;
  const svgW=parseFloat(svg.getAttribute('width')||'400');
  const svgH=parseFloat(svg.getAttribute('height')||'300');
  if(svgW>0&&svgH>0){
    const scaleX=vpW/svgW, scaleY=vpH/svgH;
    dmScale=Math.max(scaleX*0.88, Math.min(scaleX,scaleY)*0.92);
    dmTx=(vpW-svgW*dmScale)/2;
    dmTy=(vpH-svgH*dmScale)/2;
  } else { dmTx=20; dmTy=20; dmScale=0.9; }
  dmApply();
}

function dmReset(rebuild){
  // Reset filter state only when called from the Reset button, not from buildDeviceMap itself.
  if(rebuild){
    dmActiveNets=new Set();
    dmHideAll=false;
    dmSelKeys=new Set();
    dmSwCollapsed=new Set();
    const allChip=document.querySelector('.dm-chip.all-chip');
    if(allChip) allChip.classList.add('act');
    document.querySelectorAll('.dm-chip.net-chip').forEach(c=>c.classList.add('act'));
    // Refit after the rebuild, not before: clearing the filters changes the tile
    // geometry, so fitting against the old SVG size would land slightly wrong.
    dmRefitNext=true;
    buildDeviceMap();
    return;
  }
  dmFitView();
}

(function initPanZoom(){
  const vp = document.getElementById('devmap-vp');

  // Spacebar: suppress page scroll when devmap is active (capture phase).
  // Skip whenever focus is somewhere typeable. isContentEditable matters as
  // much as the tag check: the LLM prompt box is a contenteditable div, so a
  // tag-only test ate every space typed into it while the map was showing.
  document.addEventListener('keydown',e=>{
    if(e.code!=='Space'||e.repeat) return;
    if(!document.getElementById('devmap').classList.contains('show')) return;
    const ae=document.activeElement||{}, tag=ae.tagName||'';
    if(tag==='INPUT'||tag==='TEXTAREA'||ae.isContentEditable) return;
    e.preventDefault();
  }, true);

  // Scroll to zoom (always)
  vp.addEventListener('wheel', e=>{
    e.preventDefault();
    const r=vp.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    const d=Math.pow(1.0016,-e.deltaY);
    const ns=Math.max(0.12,Math.min(6,dmScale*d));
    dmTx=mx-(mx-dmTx)*(ns/dmScale);
    dmTy=my-(my-dmTy)*(ns/dmScale);
    dmScale=ns; dmApply();
  },{passive:false});

  // Suppress text selection on the viewport at all times.
  vp.addEventListener('selectstart', e=>e.preventDefault());

  // Left-click always pans. Track start position to distinguish click from drag.
  let dmDownX=0, dmDownY=0;
  const DM_DRAG_THRESHOLD=4; // px — move less than this = it's a click, not a drag

  vp.addEventListener('mousedown',e=>{
    if(e.button!==0) return;
    e.preventDefault();
    dmDownX=e.clientX; dmDownY=e.clientY;
    dmDragging=false;
    dmLx=e.clientX; dmLy=e.clientY;
    vp.classList.add('panning');
  });

  // Suppress right-click context menu on the viewport.
  vp.addEventListener('contextmenu',e=>e.preventDefault());

  window.addEventListener('mousemove',e=>{
    if(!vp.classList.contains('panning')) return;
    const dx=e.clientX-dmDownX, dy=e.clientY-dmDownY;
    if(!dmDragging && Math.sqrt(dx*dx+dy*dy)>DM_DRAG_THRESHOLD) dmDragging=true;
    if(dmDragging){
      dmTx+=e.clientX-dmLx; dmTy+=e.clientY-dmLy;
      dmLx=e.clientX; dmLy=e.clientY; dmApply();
    }
  });

  window.addEventListener('mouseup',e=>{
    if(e.button!==0) return;
    vp.classList.remove('panning');
    // Short tap (no drag) — let click events on child elements fire normally.
    // dmDragging stays true until next mousedown so child click handlers can check it.
    if(!dmDragging) dmDragging=false;
    // Reset after a tick so child click handlers see the correct dmDragging value.
    setTimeout(()=>{ dmDragging=false; }, 0);
  });
})();

// ── General symbol search ─────────────────────────────────────────────────────
// searchIndex: array of hit objects, built once at page load.
// Each hit: {kind, tkey, fi, description, label}
//   kind: 'kernel'|'window'|'buffer'|'port'|'gmio'|'net'|'flow'|'bd_len'|'contract'
//   tkey: "col,row" (always set; for flow-only hits the DMA tile)
//   fi:   flow_index or null
//   description: human-readable detail string
//   label: the raw string that was indexed (used for substring match)
const searchIndex = [];

(function buildSearchIndex(){
  // Helper to add an entry
  function add(kind, tkey, fi, label, description){
    // searchText covers kind + label + description so any of those words are findable.
    const searchText=(kind+' '+label+' '+description).toLowerCase();
    searchIndex.push({kind,tkey,fi,label:label.toLowerCase(),labelRaw:label,description,searchText});
  }

  // Index from DATA.tiles
  (DATA.tiles||[]).forEach(t=>{
    if(!t||!t.loc) return;
    const [tc,tr]=t.loc;
    const tkey=tc+','+tr;

    // Kernel names — high_level.kernel and legacy top-level callee
    const hl=t.high_level||{};
    if(hl.kernel) add('kernel',tkey,null,hl.kernel,'kernel on ('+tc+','+tr+')');
    (t.kernels||[]).forEach(k=>{
      if(k.callee) add('kernel',tkey,null,k.callee,'kernel on ('+tc+','+tr+')');
    });
    if(t.callee) add('kernel',tkey,null,t.callee,'kernel on ('+tc+','+tr+')');

    // DMA channels — direction, window, buffers, port, logical name, BD lengths, contracts
    (t.dma_channels||[]).forEach(ch=>{
      const fi=ch.flow_index!=null?ch.flow_index:null;
      const chDesc=ch.direction+' ch'+ch.channel+' on ('+tc+','+tr+')';
      // Direction string itself (e.g. "MM2S", "S2MM") + channel number
      if(ch.direction) add('channel',tkey,fi,ch.direction+ch.channel,chDesc);
      if(ch.direction) add('channel',tkey,fi,ch.direction,chDesc);
      // kernel_window: "buf31_buf31d"
      if(ch.kernel_window) add('window',tkey,fi,ch.kernel_window,chDesc);
      // kernel_buffers: ["buf31","buf31d"]
      (ch.kernel_buffers||[]).forEach(b=>{
        if(b) add('buffer',tkey,fi,b,chDesc);
      });
      // kernel_port: graph port string e.g. "gradf.g_kernel[3][3].out[0]"
      if(ch.kernel_port) add('port',tkey,fi,ch.kernel_port,chDesc);
      // logical_name: GMIO name e.g. "gmioin0"
      if(ch.logical_name) add('gmio',tkey,fi,ch.logical_name,chDesc);
      // contract string
      if(ch.contract) add('contract',tkey,fi,ch.contract,chDesc);
      // BD chain — lengths and BCF buffer symbols
      (ch.bd_chain||[]).forEach(bd=>{
        if(bd.len!=null) add('bd_len',tkey,fi,String(bd.len),
          'BD'+bd.bd_id+' len='+bd.len+' '+ch.direction+' ch'+ch.channel+' ('+tc+','+tr+')');
        (bd.bcf_syms||[]).forEach(sym=>{
          if(sym) add('buffer',tkey,fi,sym,
            'BD'+bd.bd_id+' buffer on ('+tc+','+tr+') '+ch.direction+' ch'+ch.channel);
        });
      });
    });

    // Legacy port connections array (aiehlc_aiesim path)
    (t.ports||[]).forEach(p=>{
      if(p.port_id) add('port',tkey,null,p.port_id,'port on ('+tc+','+tr+')');
      if(p.kernel_port) add('port',tkey,null,p.kernel_port,'port on ('+tc+','+tr+')');
      if(p.graph_port) add('port',tkey,null,p.graph_port,'port on ('+tc+','+tr+')');
    });
  });

  // Index from DATA.comm_paths
  (DATA.comm_paths||[]).forEach(p=>{
    if(!p) return;
    const fi=p.flow_index!=null?p.flow_index:null;
    // Representative tile for the hit. Prefer a DMA tile, but pull flows have
    // none (routing_edges_for_flow fills dma_tiles only from circuit_connects
    // whose master is DMA, and a pull source has DMA as the slave), so fall
    // back to a tile the flow actually touches instead of the literal 0,0.
    // Sorted because dma_tiles is serialized from a Python set.
    const cand=[...(p.dma_tiles||[]), ...(p.tiles||[]),
                ...(p.edges||[]).map(e=>e[0])].filter(t=>t&&t.length===2);
    const repTile=cand.length
      ?cand.slice().sort((a,b)=>(a[0]-b[0])||(a[1]-b[1]))[0]
      :null;
    const tkey=repTile?repTile[0]+','+repTile[1]:null;

    // Net ID: "net7"
    if(p.net_id) add('net',tkey,fi,p.net_id,'net ('+p.net_id+') f'+fi);
    // Flow index string: "f7"
    if(fi!=null){ add('flow',tkey,fi,'f'+fi,'flow index '+fi); }
    // GMIO logical name (from producer/consumer data blocks)
    const gmioNames=[];
    (p.producer||[]).forEach(s=>{ if(s.gmio_name) gmioNames.push(s.gmio_name); });
    (p.consumer||[]).forEach(s=>{ if(s.gmio_name) gmioNames.push(s.gmio_name); });
    // Also from nested data blocks
    ['producer_data','consumer_data'].forEach(dk=>{
      const d=p[dk]; if(!d) return;
      if(d.logical_name) gmioNames.push(d.logical_name);
      if(d.name) gmioNames.push(d.name);
    });
    // config_ref GMIO names
    if(p.config_ref) add('gmio',tkey,fi,p.config_ref,'config ref for f'+fi);
    gmioNames.forEach(n=>{ if(n) add('gmio',tkey,fi,n,'GMIO for f'+fi); });

    // Hops — pick up any extra names embedded there
    (p.hops||[]).forEach(h=>{
      if(h.port_name) add('port',tkey,fi,h.port_name,'hop port in f'+fi);
    });
  });

  // Index from DATA.flow_summary (supply/demand)
  (DATA.flow_summary||[]).forEach(fs=>{
    if(!fs) return;
    const fi=fs.flow_index!=null?fs.flow_index:null;
    const tkey=fs.tile?fs.tile[0]+','+fs.tile[1]:'0,0';
    if(fs.kernel_port) add('port',tkey,fi,fs.kernel_port,'supply/demand port f'+fi);
  });
})();

// ── Search pane state ─────────────────────────────────────────────────────────
// srSearchTerms: pinned search terms that drive LAYER 2.5 highlights.
let srSearchTerms=new Set();

// Kind badge colours (shared by suggest and results table)
const SR_KIND_COLOR={
  kernel:'#7ac',window:'#9f9',buffer:'#9f9',port:'#c9a',
  gmio:'#f0c070',net:'#aaf',flow:'#aaf',bd_len:'#f9c',contract:'#ccc'
};

// Split query on whitespace into tokens; every token must appear in searchText.
function srTokens(term){ return term.toLowerCase().trim().split(/\s+/).filter(Boolean); }

// Exact per-token substring match — used by highlight layer and results table.
function srResolve(term){
  const tokens=srTokens(term);
  if(!tokens.length) return [];
  return searchIndex.filter(h=>tokens.every(t=>h.searchText.includes(t))).map(h=>({
    kind:h.kind, tkey:h.tkey, fi:h.fi, labelRaw:h.labelRaw, description:h.description
  }));
}

// Returns {tileLockKeys, flowLockFis} for LAYER 2.5 SVG highlight.
function lkActiveSets(){
  const tileLockKeys=new Set();
  const flowLockFis=new Set();
  srSearchTerms.forEach(term=>{
    srResolve(term).forEach(h=>{
      if(h.tkey) tileLockKeys.add(h.tkey);
      if(h.fi!=null) flowLockFis.add(h.fi);
    });
  });
  return {tileLockKeys,flowLockFis};
}

// ── Fuzzy suggest ─────────────────────────────────────────────────────────────
// Score a single candidate label against the query using a contiguous-run bonus
// fuzzy algorithm: each matched character scores 1; consecutive run adds bonus.
// Spaces split into tokens; each token is scored independently and summed.
function srFuzzyScore(label_lc, q){
  function scoreOne(lc, tok){
    let li=0, qi=0, score=0, run=0;
    while(li<lc.length && qi<tok.length){
      if(lc[li]===tok[qi]){ score+=1+run*0.5; run++; qi++; }
      else { run=0; }
      li++;
    }
    return qi===tok.length ? score : -1;
  }
  const tokens=q.split(/\s+/).filter(Boolean);
  let total=0;
  for(const tok of tokens){
    const s=scoreOne(label_lc, tok);
    if(s<0) return -1;
    total+=s;
  }
  return total;
}

// Return up to `limit` unique label candidates matching `q` fuzzily, sorted by score.
function srSuggest(q, limit=12){
  const ql=q.toLowerCase().trim();
  if(!ql) return [];
  const scored=[];
  const seen=new Set();
  searchIndex.forEach(h=>{
    if(seen.has(h.labelRaw)) return;
    // Score primarily against label; fall back to searching searchText so kind/description hits surface.
    let s=srFuzzyScore(h.label, ql);
    if(s<0) s=srFuzzyScore(h.searchText, ql)*0.5; // half-weight for description/kind matches
    if(s<0) return;
    seen.add(h.labelRaw);
    scored.push({s, kind:h.kind, labelRaw:h.labelRaw, tkey:h.tkey, fi:h.fi});
  });
  scored.sort((a,b)=>b.s-a.s);
  return scored.slice(0,limit);
}

// Highlight matched chars in a label string for the suggestion dropdown.
function srHighlightMatch(labelRaw, q){
  const lc=labelRaw.toLowerCase(), ql=q.toLowerCase();
  let result='', qi=0;
  for(let i=0;i<labelRaw.length;i++){
    if(qi<ql.length && lc[i]===ql[qi]){
      result+='<span class="sr-sug-match">'+esc(labelRaw[i])+'</span>';
      qi++;
    } else {
      result+=esc(labelRaw[i]);
    }
  }
  return result;
}

// ── Suggest dropdown ──────────────────────────────────────────────────────────
let srSugIdx=-1; // keyboard-selected row index

function srShowSuggest(q){
  const box=document.getElementById('sr-suggest');
  if(!q.trim()){ box.classList.add('hide'); box.innerHTML=''; srSugIdx=-1; return; }
  const candidates=srSuggest(q);
  if(!candidates.length){ box.classList.add('hide'); box.innerHTML=''; srSugIdx=-1; return; }
  box.innerHTML=candidates.map((c,i)=>{
    const kc=SR_KIND_COLOR[c.kind]||'#ccc';
    const tileTxt=c.tkey?'('+c.tkey+')':'';
    return '<div class="sr-sug-item" data-i="'+i+'" data-label="'+esc(c.labelRaw)+'">'
      +'<span class="sr-sug-kind" style="color:'+kc+'">'+esc(c.kind)+'</span>'
      +'<span class="sr-sug-label">'+srHighlightMatch(c.labelRaw,q)+'</span>'
      +'<span class="sr-sug-tile">'+esc(tileTxt)+'</span>'
      +'</div>';
  }).join('');
  srSugIdx=-1;
  box.classList.remove('hide');
  box.querySelectorAll('.sr-sug-item').forEach(item=>{
    item.onmousedown=e=>{
      e.preventDefault();
      srPinTerm(item.dataset.label);
      document.getElementById('sr-input').value='';
      box.classList.add('hide'); box.innerHTML=''; srSugIdx=-1;
    };
  });
}

function srMoveSug(dir){
  const box=document.getElementById('sr-suggest');
  const items=box.querySelectorAll('.sr-sug-item');
  if(!items.length) return;
  items.forEach(x=>x.classList.remove('sel'));
  srSugIdx=Math.max(0,Math.min(items.length-1,(srSugIdx<0?-1:srSugIdx)+dir));
  items[srSugIdx].classList.add('sel');
  items[srSugIdx].scrollIntoView({block:'nearest'});
}

function srSugSelected(){
  const box=document.getElementById('sr-suggest');
  const sel=box.querySelector('.sr-sug-item.sel');
  return sel ? sel.dataset.label : null;
}

// ── Pin term + chip + results ─────────────────────────────────────────────────
function srPinTerm(raw){
  if(!raw||!raw.trim()) return;
  srSearchTerms.add(raw.trim());
  srRenderChips();
  srRenderResults();
  if(document.getElementById('devmap').classList.contains('show')) buildDeviceMap();
  if(srSearchTerms.size>0)
    llmPushCtx('[context] Search: pinned "'+[...srSearchTerms].join('", "')+'"',
                 'search');
}

function srRenderChips(){
  const wrap=document.getElementById('sr-chips');
  wrap.innerHTML='';
  srSearchTerms.forEach(term=>{
    const chip=document.createElement('span');
    chip.className='lk-chip';
    chip.appendChild(document.createTextNode(term));
    const x=document.createElement('span');
    x.className='lk-x'; x.textContent='×';
    x.onclick=()=>{ srSearchTerms.delete(term); srRenderChips(); srRenderResults(); if(document.getElementById('devmap').classList.contains('show')) buildDeviceMap(); if(!srSearchTerms.size) llmPushCtx(null,'search'); };
    chip.appendChild(x);
    wrap.appendChild(chip);
  });
  dmSyncClearBtn();
}

function srRenderResults(){
  const out=document.getElementById('sr-results');
  if(!srSearchTerms.size){
    out.innerHTML='<div class="sr-empty">Type above to search, Enter or click a suggestion to pin a term and see results.</div>';
    return;
  }
  let html='';
  srSearchTerms.forEach(term=>{
    const hits=srResolve(term);
    html+='<div class="sr-group-hdr">'+esc(term)
      +(hits.length?' <span style="color:var(--tx-lo);font-weight:400">('+hits.length+' match'+(hits.length!==1?'es':'')+')</span>':'')
      +'</div>';
    if(!hits.length){ html+='<div class="sr-empty">no matches</div>'; return; }
    html+='<table><thead><tr>'
      +'<th>kind</th><th>tile</th><th>flow</th><th>matched</th><th>detail</th>'
      +'</tr></thead><tbody>';
    const seen=new Set();
    hits.forEach(h=>{
      const rowKey=h.kind+':'+h.tkey+':'+h.fi+':'+h.labelRaw;
      if(seen.has(rowKey)) return; seen.add(rowKey);
      const [tc,tr]=(h.tkey||'?,?').split(',');
      const kc=SR_KIND_COLOR[h.kind]||'#ccc';
      html+='<tr>'
        +'<td><span style="color:'+kc+'">'+esc(h.kind)+'</span></td>'
        +'<td>('+esc(tc)+','+esc(tr)+')</td>'
        +'<td>'+(h.fi!=null?'f'+h.fi:'-')+'</td>'
        +'<td style="font-family:monospace;max-width:120px;overflow:hidden;text-overflow:ellipsis">'+esc(h.labelRaw)+'</td>'
        +'<td style="color:var(--tx-lo);font-size:11px">'+esc(h.description)+'</td>'
        +'</tr>';
    });
    html+='</tbody></table>';
  });
  out.innerHTML=html;
}

// ── Wire up the search pane input ─────────────────────────────────────────────
(function initSearchPane(){
  const inp=document.getElementById('sr-input');
  const sug=document.getElementById('sr-suggest');
  if(!inp) return;

  inp.addEventListener('input',()=>srShowSuggest(inp.value));

  inp.addEventListener('keydown',e=>{
    if(e.key==='ArrowDown'){ e.preventDefault(); srMoveSug(1); return; }
    if(e.key==='ArrowUp'){ e.preventDefault(); srMoveSug(-1); return; }
    if(e.key==='Escape'){ sug.classList.add('hide'); sug.innerHTML=''; srSugIdx=-1; return; }
    if(e.key==='Enter'){
      e.preventDefault();
      const sel=srSugSelected();
      const raw=sel||inp.value.trim();
      if(!raw) return;
      srPinTerm(raw);
      inp.value='';
      sug.classList.add('hide'); sug.innerHTML=''; srSugIdx=-1;
    }
  });

  // Hide suggest when focus leaves the input (but not on mousedown in the list)
  inp.addEventListener('blur',()=>{ setTimeout(()=>{ sug.classList.add('hide'); sug.innerHTML=''; srSugIdx=-1; },150); });
})();

// ── Device-map live status ────────────────────────────────────
// Worst-wins ordering, mirroring the server's _worst(): a flow carrying one
// stalled channel is stalled even if its other channels are running.
const DM_ST_ORDER = ['unreachable','error','stalled','running','completed','idle'];
function dmWorst(states){
  for(const s of DM_ST_ORDER) if(states.indexOf(s)>=0) return s;
  return states.length?states[0]:null;
}
// A flow's state, from the channels that carry it. Only meaningful for the
// 'dma' scan — 'cores'/'events' report nothing per-flow, so nets stay uncolored.
function dmDeriveNets(cells){
  const out={};
  if(!cells) return out;
  (DATA.tiles||[]).forEach(t=>{
    const cell=cells[t.loc[0]+','+t.loc[1]];
    if(!cell||!cell.channels) return;
    (t.dma_channels||[]).forEach(ch=>{
      if(ch.flow_index==null||!ch.direction) return;
      const c=cell.channels[ch.direction.toLowerCase()+ch.channel];
      if(!c||!c.state) return;
      (out[ch.flow_index]=out[ch.flow_index]||[]).push(c.state);
    });
  });
  const nets={};
  Object.keys(out).forEach(fi=>{ nets[fi]=dmWorst(out[fi]); });
  return nets;
}
function dmStColor(state){ return (LSTATE[state]||LSTATE.unknown)[0]; }
function dmStLabel(state){ return (LSTATE[state]||LSTATE.unknown)[1]; }

// Paint scan results onto the already-built SVG. Deliberately a DOM patch
// rather than a rebuild: a rebuild is far more work than setting a few
// attributes, and doing it on every 2s poll would churn the whole SVG under a
// user who is mid-interaction.
function dmPaintStatus(){
  const cells = dmStatus.cells;
  const svg = document.getElementById('devmap-svg');
  if(!svg) return;
  // Tiles.
  svg.querySelectorAll('.dm-tile').forEach(g => {
    const key = g.getAttribute('data-key');
    const rect = g.querySelector('rect');
    const lbl  = g.querySelector('.dm-stlabel');
    const cell = cells ? cells[key] : null;
    const base = dmTileFill[key];
    if(!cell || !cell.state){
      if(rect && base!=null) rect.setAttribute('fill', base);
      dmTileStroke[key] = g.getAttribute('data-basestroke') || dmTileStroke[key];
      if(rect && !dmSelKeys.has(key)){
        rect.setAttribute('stroke', dmTileStroke[key]);
        rect.setAttribute('stroke-width','1');
      }
      if(lbl){ lbl.textContent=''; }
      g.removeAttribute('data-state');
      return;
    }
    const col = dmStColor(cell.state);
    g.setAttribute('data-state', cell.state);
    if(rect){
      rect.setAttribute('fill', col+'55');
      // Selection outranks status on the stroke, but the fill tint still shows
      // through — so a selected tile does not go blind to its own state.
      dmTileStroke[key] = col;
      if(!dmSelKeys.has(key)){
        rect.setAttribute('stroke', col);
        rect.setAttribute('stroke-width','2');
      }
    }
    if(lbl){
      lbl.textContent = dmStLabel(cell.state);
      lbl.setAttribute('fill', col);
    }
  });
  // Net casings: a wider translucent stroke *under* each edge, so the net keeps
  // its identity color on top and the status reads as a glow around it.
  svg.querySelectorAll('.dm-edgecase').forEach(ln => {
    const st = dmStatus.nets[ln.getAttribute('data-fi')];
    ln.setAttribute('stroke', st ? dmStColor(st) : 'transparent');
  });
  // Net chips.
  document.querySelectorAll('#devmap-netbar .net-chip').forEach(chip => {
    let dot = chip.querySelector('.chstat');
    const st = dmStatus.nets[chip.dataset.fi];
    if(!st){ if(dot) dot.remove(); return; }
    if(!dot){
      dot = document.createElement('span');
      dot.className = 'chstat';
      chip.insertBefore(dot, chip.firstChild);
    }
    dot.style.background = dmStColor(st);
    dot.title = 'net'+chip.dataset.fi+': '+st;
  });
  dmSyncLegend();
  dmSyncClearBtn();
}
// Legend lists only the states actually on screen — a fixed six-swatch strip
// would be mostly noise on a healthy run.
function dmSyncLegend(){
  const el = document.getElementById('devmap-stlegend');
  if(!el) return;
  const seen = [];
  if(dmStatus.cells){
    Object.values(dmStatus.cells).forEach(c => {
      if(c && c.state && seen.indexOf(c.state)<0) seen.push(c.state);
    });
  }
  Object.values(dmStatus.nets).forEach(s => { if(s && seen.indexOf(s)<0) seen.push(s); });
  if(!seen.length){ el.className='hide'; el.innerHTML=''; return; }
  seen.sort((a,b)=>DM_ST_ORDER.indexOf(a)-DM_ST_ORDER.indexOf(b));
  el.className = 'dml-item';
  el.innerHTML = '<span style="margin-right:2px">'+esc(dmStatus.what||'')+':</span>' +
    seen.map(s => '<span class="dml-item" style="margin-right:7px">' +
      '<span class="dml-dot" style="background:'+dmStColor(s)+'"></span>' +
      esc(dmStLabel(s))+'</span>').join('');
}
function dmSetScanStatus(msg, isErr){
  const e = document.getElementById('dmScanStatus');
  if(!e) return;
  e.textContent = msg;
  e.className = isErr ? 'err' : '';
}
// Entry point called from applyGrid() so both views share one fetch.
function dmApplyStatus(res){
  dmStatus.what  = res.error ? dmStatus.what : (res.what || LIVE.what);
  dmStatus.cells = res.error ? null : (res.cells || {});
  dmStatus.nets  = res.error ? {} : dmDeriveNets(dmStatus.cells);
  dmStatus.err   = res.error || null;
  dmStatus.ts    = new Date().toLocaleTimeString();
  dmSetScanStatus(res.error ? res.error
                            : (dmStatus.what+' @ '+dmStatus.ts), !!res.error);
  dmPaintStatus();
}
function dmClearStatus(){
  dmStatus = { what:null, cells:null, nets:{}, err:null, ts:null };
  dmSetScanStatus('');
  dmPaintStatus();
}
// Greyed out when there is nothing to clear, so the button reports whether the
// map is actually carrying any overlay.
function dmSyncClearBtn(){
  const b = document.getElementById('dmClearBtn');
  if (b) b.disabled = !dmStatus.cells && !srSearchTerms.size;
}
// "Back to normal" for the two things that tint the map: a live scan and pinned
// search terms. Net isolation and pan/zoom are deliberately left alone — the
// "Reset view" button in the viewport already owns those, and clearing them
// here would silently throw away a filter the user set up by hand.
function dmClearAll(){
  // Stop the poll first: clearing while live is on just gets repainted 2s later.
  if (LIVE.enabled) setLive(false);
  else dmClearStatus();
  if (ROUTING_SRC !== 'static') setRoutingSource('static');
  if (srSearchTerms.size){
    srSearchTerms.clear();
    llmPushCtx(null, 'search');
    srRenderChips();
    srRenderResults();
    if (document.getElementById('devmap').classList.contains('show')) buildDeviceMap();
  }
  dmSyncClearBtn();
}

function buildNetBar(){
  const bar=document.getElementById('devmap-netbar');
  bar.innerHTML='';
  // "All Flows" chip
  const allc=document.createElement('button');
  allc.className='dm-chip all-chip'+(!dmHideAll&&dmActiveNets.size===0?' act':'');
  allc.textContent='Toggle All Flows';
  allc.onclick=()=>dmSelectNet(-1,false);
  bar.appendChild(allc);
  // Per-flow chips — derive label from flow_summary if available
  const fsSummary={};
  (DATA.flow_summary||[]).forEach(f=>{ fsSummary[f.flow_index]=f; });
  const fiDir={};
  (DATA.comm_paths||[]).forEach(p=>{ fiDir[p.flow_index]=p.direction; });
  dmFlowIds.forEach(fi=>{
    const isAct=!dmHideAll&&(dmActiveNets.size===0||dmActiveNets.has(fi));
    const chip=document.createElement('button');
    chip.className='dm-chip net-chip'+(isAct?' act':'');
    chip.dataset.fi=fi;
    const dot=document.createElement('span');
    dot.className='chdot'; dot.style.background=dmColor(fi);
    chip.appendChild(dot);
    const dir=fiDir[fi]==='push'?'→':'←';
    chip.appendChild(document.createTextNode('net'+fi+' '+dir));
    chip.style.color=dmColor(fi);
    chip.title='Click to isolate net '+fi+' · Ctrl+click to multi-select';
    chip.onclick=ev=>dmSelectNet(fi,ev.ctrlKey||ev.metaKey);
    bar.appendChild(chip);
  });
}

function dmSelectNet(fi, ctrl){
  if(fi===-1){
    // "All Flows" chip: if all are already showing, hide all (toggle); otherwise restore all
    if(!dmHideAll&&dmActiveNets.size===0) { dmHideAll=true; }
    else { dmHideAll=false; dmActiveNets=new Set(); }
  } else if(ctrl){
    // Ctrl+click: toggle this net in/out of multi-selection
    dmHideAll=false;
    if(dmActiveNets.has(fi)) dmActiveNets.delete(fi);
    else dmActiveNets.add(fi);
  } else {
    // Plain click: exclusively select this net, or clear if it was the only one
    dmHideAll=false;
    if(dmActiveNets.size===1&&dmActiveNets.has(fi)) dmActiveNets=new Set();
    else { dmActiveNets=new Set(); dmActiveNets.add(fi); }
  }
  dmSyncNetSelection();
}

// Repaint the chip bar, rebuild the map and reconcile the side panel against
// whatever dmActiveNets/dmHideAll now say. Split out of dmSelectNet so the tile
// right-click menu can set the selection itself and still land in one place.
function dmSyncNetSelection(){
  const allActive=!dmHideAll&&dmActiveNets.size===0;
  document.querySelector('.dm-chip.all-chip').classList.toggle('act',allActive);
  document.querySelectorAll('.dm-chip.net-chip').forEach(c=>{
    c.classList.toggle('act',!dmHideAll&&(allActive||dmActiveNets.has(parseInt(c.dataset.fi))));
  });
  buildDeviceMap();
  // Sync panelItems: remove net entries no longer selected, add newly selected ones.
  // Preserve existing tile entries.
  panelItems.forEach((_,key)=>{ if(key.startsWith('net:')&&!dmActiveNets.has(parseInt(key.slice(4)))) panelItems.delete(key); });
  dmActiveNets.forEach(fi2=>{
    const key=panelKey('net',fi2);
    if(!panelItems.has(key)){
      const path=(DATA.comm_paths||[]).find(p=>p.flow_index===fi2)||null;
      const color=dmColor(fi2);
      const dir=path?(path.direction==='push'?'push →':'pull ←'):'';
      const prod=path&&path.producer?(path.producer.gmio_name||path.producer.logical_name||'?'):'?';
      const cons=path&&path.consumer?(path.consumer.gmio_name||path.consumer.logical_name||'?'):'?';
      panelItems.set(key,{
        kind:'net', label:'net'+fi2+(dir?' '+dir:''), color,
        buildBody:()=>buildNetBody(path),
        llmCtx:'net f'+fi2+(dir?' ('+dir+')':'')+' producer:'+prod+' consumer:'+cons
      });
    }
    panelActiveKey = panelKey('net',fi2);  // last added becomes active
  });
  if(!dmActiveNets.size){
    panelItems.forEach((_,key)=>{ if(key.startsWith('net:')) panelItems.delete(key); });
  }
  panelSync();
}

// ── Tile right-click menu ─────────────────────────────────────
// Two actions the chip bar can't express: thinning routing rows per tile/row/
// column, and isolating by tile instead of by net.

// Flow indices whose path touches (tc,tr). Uses the same _flowTileSet() the
// renderer uses to decide which tiles a flow draws through, so the isolated set
// is exactly the flows that would still paint this tile.
function dmFlowsAtTile(tc, tr){
  const key=tc+','+tr, out=new Set();
  (DATA.comm_paths||[]).forEach(p=>{
    if(_flowTileSet(p).has(key)) out.add(p.flow_index);
  });
  return out;
}

// Keys of every rendered tile box in a row / column / the whole map. Waypoint
// tiles are excluded: they draw no box and so have no routing rows to collapse.
function dmTileKeysIn(scope, tc, tr){
  const svg=document.getElementById('devmap-svg');
  const out=[];
  svg.querySelectorAll('.dm-tile').forEach(g=>{
    if(g.getAttribute('data-waypoint')==='1') return;
    const k=g.getAttribute('data-key'); if(!k) return;
    const [c,r]=k.split(',').map(Number);
    if(scope==='row'&&r!==tr) return;
    if(scope==='col'&&c!==tc) return;
    out.push(k);
  });
  return out;
}

// Collapse unless every key in scope is already collapsed, in which case expand.
// One menu entry per scope that reads as a toggle, matching the net chips.
function dmToggleRouting(keys){
  if(!keys.length) return;
  const allHidden=keys.every(k=>dmSwCollapsed.has(k));
  keys.forEach(k=>{ if(allHidden) dmSwCollapsed.delete(k); else dmSwCollapsed.add(k); });
  buildDeviceMap();
}

function dmHideMenu(){
  const m=document.getElementById('dm-ctxmenu');
  if(m) m.remove();
}

function dmShowTileMenu(clientX, clientY, tc, tr){
  dmHideMenu();
  dmHideTip();
  const key=tc+','+tr;
  const rowKeys=dmTileKeysIn('row',tc,tr), colKeys=dmTileKeysIn('col',tc,tr);
  const flows=dmFlowsAtTile(tc,tr);
  const lbl=(keys,what)=>(keys.every(k=>dmSwCollapsed.has(k))?'Expand ':'Collapse ')+what;

  const items=[
    {head:'routing info'},
    {text:lbl([key],'this tile'), on:()=>dmToggleRouting([key])},
    {text:lbl(rowKeys,'row '+tr+'  ('+rowKeys.length+' tiles)'), on:()=>dmToggleRouting(rowKeys)},
    {text:lbl(colKeys,'column '+tc+'  ('+colKeys.length+' tiles)'), on:()=>dmToggleRouting(colKeys)},
    {sep:true},
    {head:'flows'},
    {text:'Isolate flows through this tile  ('+flows.size+')',
     dis:flows.size===0,
     on:()=>{ dmHideAll=false; dmActiveNets=new Set(flows); dmSyncNetSelection(); }},
    {text:'Show all flows', dis:!dmHideAll&&dmActiveNets.size===0,
     on:()=>{ dmHideAll=false; dmActiveNets=new Set(); dmSyncNetSelection(); }},
  ];

  const m=document.createElement('div');
  m.id='dm-ctxmenu';
  m.style.cssText='position:fixed;z-index:10000;min-width:210px;'
    +'background:#1a1c27;border:1px solid #2a2e46;border-radius:5px;padding:4px 0;'
    +'font:11px/1.6 ui-monospace,monospace;color:#e2e4f0;'
    +'box-shadow:0 6px 22px rgba(0,0,0,.7);';
  items.forEach(it=>{
    const d=document.createElement('div');
    if(it.sep){
      d.style.cssText='height:1px;background:#2a2e46;margin:4px 0;';
    } else if(it.head){
      d.style.cssText='padding:2px 10px;color:#7c8099;font-size:9px;'
        +'letter-spacing:.08em;text-transform:uppercase;';
      d.textContent=it.head;
    } else {
      d.style.cssText='padding:3px 12px;cursor:'+(it.dis?'default':'pointer')
        +';color:'+(it.dis?'#5a5e74':'#e2e4f0')+';white-space:nowrap;';
      d.textContent=it.text;
      if(!it.dis){
        d.onmouseenter=()=>d.style.background='#262a3d';
        d.onmouseleave=()=>d.style.background='';
        d.onclick=()=>{ dmHideMenu(); it.on(); };
      }
    }
    m.appendChild(d);
  });
  document.body.appendChild(m);
  // Flip against the viewport edges the same way dmShowTip does.
  let lx=clientX+2, ly=clientY+2;
  if(lx+m.offsetWidth>window.innerWidth-8) lx=clientX-m.offsetWidth-2;
  if(ly+m.offsetHeight>window.innerHeight-8) ly=Math.max(8,clientY-m.offsetHeight-2);
  m.style.left=lx+'px'; m.style.top=ly+'px';
}
document.addEventListener('mousedown',e=>{
  const m=document.getElementById('dm-ctxmenu');
  if(m&&!m.contains(e.target)) dmHideMenu();
});
document.addEventListener('keydown',e=>{ if(e.key==='Escape') dmHideMenu(); });

function buildNetBody(p){
  if(!p) return '<div class="placeholder">Click a tile or net for details</div>';
  const fi=p.flow_index;
  const color=dmColor(fi);
  const dir=p.direction==='push'?'push (host → array)':'pull (array → host)';
  const dot='<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:'+color+';vertical-align:middle;margin-right:6px;border:1.2px solid #181818"></span>';

  // ── Producer / Consumer from stages ──────────────────────────────────────
  const stages=p.stages||[];
  const prodStage=stages.find(s=>s.role==='producer');
  const consStage=stages.find(s=>s.role==='consumer');
  const chanStage=stages.find(s=>s.role==='channel');

  function fmtTile(t){ return t?'('+t.col+','+t.row+') '+esc(t.type||''):'-'; }
  function fmtTiles(arr){ return (arr||[]).map(fmtTile).join(', ')||'-'; }

  let prodHtml='', consHtml='';
  if(prodStage){
    const t=prodStage.tile||((prodStage.tiles||[])[0]);
    prodHtml='<div class="kv"><b>producer:</b> '+fmtTile(t)
      +(prodStage.contract?' &mdash; <span class="dimtxt">'+esc(prodStage.contract)+'</span>':'')+'</div>';
    if(prodStage.config_ref)
      prodHtml+='<div class="kv"><span class="dimtxt">config: '+esc(prodStage.config_ref)+'</span></div>';
  }
  if(consStage){
    const t=consStage.tile||((consStage.tiles||[])[0]);
    consHtml='<div class="kv"><b>consumer:</b> '+fmtTile(t)
      +(consStage.contract?' &mdash; <span class="dimtxt">'+esc(consStage.contract)+'</span>':'')+'</div>';
    if(consStage.config_ref)
      consHtml+='<div class="kv"><span class="dimtxt">config: '+esc(consStage.config_ref)+'</span></div>';
  }

  // ── Participating tile DMA channels (from DATA.tiles) ────────────────────
  const tileMap={};
  (DATA.tiles||[]).forEach(t=>{ tileMap[t.loc[0]+','+t.loc[1]]=t; });

  const allTilePairs=new Set();
  (p.edges||[]).forEach(e=>{ allTilePairs.add(e[0][0]+','+e[0][1]); allTilePairs.add(e[1][0]+','+e[1][1]); });
  (p.dma_tiles||[]).forEach(t=>allTilePairs.add(t[0]+','+t[1]));

  let chanRows='';
  allTilePairs.forEach(key=>{
    const t=tileMap[key]; if(!t) return;
    const ch=(t.dma_channels||[]).filter(c=>c.flow_index===fi);
    ch.forEach(c=>{
      const bds=c.bd_chain||[];
      const bdChain=bds.slice(0,4).map((bd,i)=>{
        const hasNext=bd.next_bd!=null&&bd.next_bd>=0&&i<bds.length-1;
        return '<span class="bd-id">BD'+bd.bd_id+'</span>'
          +'<span class="bd-len">['+bd.len+'B]</span>'
          +(hasNext?'<span class="bd-next">→</span>':'');
      }).join('')+(bds.length>4?'…':'');
      const acqSet=new Set(), relSet=new Set();
      bds.forEach(bd=>{
        (bd.acquire_lock||[]).forEach(l=>l.id!=null&&acqSet.add('L'+l.id+'('+l.val+')'));
        (bd.release_lock||[]).forEach(l=>l.id!=null&&relSet.add('L'+l.id+'('+l.val+')'));
      });
      const lockStr=(acqSet.size||relSet.size)
        ?'<span class="bd-lock">acq:'+[...acqSet].join(',')+' rel:'+[...relSet].join(',')+'</span>'
        :'-';
      chanRows+='<tr>'
        +'<td>('+t.loc[0]+','+t.loc[1]+')</td>'
        +'<td>'+esc(t.type)+'</td>'
        +'<td>'+esc(c.direction)+' ch'+c.channel+'</td>'
        +'<td style="font-family:ui-monospace,monospace;white-space:nowrap">'+(bdChain||'—')+'</td>'
        +'<td style="font-size:10px">'+lockStr+'</td>'
        +(c.kernel_port?'<td>'+esc(c.kernel_port)+'</td>':'<td>-</td>')
        +'</tr>';
    });
  });
  const chanTable=chanRows
    ?'<table class="rctbl"><thead><tr><th>tile</th><th>type</th><th>ch</th>'
      +'<th>BD chain</th><th>locks</th><th>port</th></tr></thead>'
      +'<tbody>'+chanRows+'</tbody></table>'
    :'<div class="placeholder">(no DMA channels on participating tiles)</div>';

  // ── Stream-switch connections + GMIO (routing_connections) ──────────────
  // Split: circuit_connect/packet_connect are stream-switch port configs;
  // shim_aie_to_ext/shim_ext_to_aie are GMIO DMA channel registrations —
  // different hardware, different config, kept separate.
  const shimKinds=new Set(['shim_aie_to_ext','shim_ext_to_aie']);
  const gmioConns=(p.routing_connections||[]).filter(c=>shimKinds.has(c.kind));
  const swConnsRaw=(p.routing_connections||[]).filter(c=>!shimKinds.has(c.kind));
  // Sort stream-switch in data-flow order: pull → high row first; push → low row first.
  const pktOnPath=swConnsRaw.filter(c=>c.kind==='packet_connect');
  const sharedPktFwd=_sharedPktForwardMaster(pktOnPath);
  const swExpanded=[];
  swConnsRaw.forEach(c=>{
    if(c.kind!=='packet_connect'){ swExpanded.push(c); return; }
    _expandPktConnectRows(c,sharedPktFwd).forEach(row=>{
      swExpanded.push({kind:'packet_hw', tile:c.tile, flow_index:c.flow_index, ...row});
    });
  });
  swExpanded.sort((a,b)=>
    p.direction==='pull'
      ?(b.tile?.row??0)-(a.tile?.row??0)
      :(a.tile?.row??0)-(b.tile?.row??0)
  );
  let swRows='';
  swExpanded.forEach(c=>{
    const t=c.tile||{};
    let kind='', detail='', extra='';
    if(c.kind==='packet_hw'){
      kind='PKT';
      const sl=c.slave||{};
      detail=sl.dir
        ?esc(sl.dir)+':'+sl.idx+'&nbsp;&rarr;&nbsp;'+esc(c.master.dir)+':'+c.master.idx
        :'fwd&nbsp;&rarr;&nbsp;'+esc(c.master.dir)+':'+c.master.idx;
      if(c.pktid!=null) extra='<span class="rt-pktid" title="packet match id">pkt'+c.pktid+'</span>';
      if(c.mask!=null) extra+=_fmtPktMaskBadge(c.mask,c.leg);
    } else {
      kind='CCT';
      const s=c.slave||{}, m=c.master||{};
      if(s.dir!=null&&m.dir!=null){
        detail=esc(s.dir)+':'+s.idx+'&nbsp;&rarr;&nbsp;'+esc(m.dir)+':'+m.idx;
      }
    }
    // The panel is already scoped to one net: use its flow, not c.flow_index —
    // routing_connections records carry no flow_index, so that was always null
    // and every row fell back to the grey "no local DMA" badge.
    extra+=_flowDmaSpanHtml(p.flow_index,t.col,t.row);
    swRows+='<tr><td>('+t.col+','+t.row+')</td><td>'+kind+'</td>'
      +'<td>'+detail+extra+'</td></tr>';
  });
  const swTable=swRows
    ?'<table class="rctbl"><thead><tr><th>tile</th><th>kind</th><th>ports</th></tr></thead>'
      +'<tbody>'+swRows+'</tbody></table>'
    :'<div class="placeholder">(no stream-switch connections)</div>';

  // GMIO channel table — separate from stream switch
  const gmioDir={'shim_aie_to_ext':'S2MM (array → DDR)','shim_ext_to_aie':'MM2S (DDR → array)'};
  let gmioRows='';
  gmioConns.forEach(c=>{
    const t=c.tile||{};
    const dir=gmioDir[c.kind]||esc(c.kind);
    gmioRows+='<tr><td>('+t.col+','+t.row+')</td><td>'+dir+'</td>'
      +'<td>'+(c.stream_id!=null?c.stream_id:'-')+'</td></tr>';
  });
  const gmioTable=gmioRows
    ?'<table class="rctbl"><thead><tr><th>tile</th><th>direction</th><th>stream_id</th></tr></thead>'
      +'<tbody>'+gmioRows+'</tbody></table>'
    :'';

  // ── All hops (stream + shmem) ─────────────────────────────────────────────
  let hopRows='';
  (p.hops||[]).forEach(h=>{
    const kind=h.kind||'';
    const typeLabel=h.type==='shmem'
      ?(kind==='window'?'shmem (window)':'shmem (dma bridge)')
      :'stream';
    hopRows+='<tr><td>('+h.from_col+','+h.from_row+')</td>'
      +'<td>→</td>'
      +'<td>('+h.to_col+','+h.to_row+')</td>'
      +'<td>'+typeLabel+'</td></tr>';
  });
  const hopTable=hopRows
    ?'<table class="rctbl"><thead><tr><th>from</th><th></th><th>to</th><th>type</th></tr></thead>'
      +'<tbody>'+hopRows+'</tbody></table>'
    :'<div class="placeholder">(no hops)</div>';

  // ── Supply/demand balance ─────────────────────────────────────────────────
  const bal=(DATA.supply_demand||[]).find(b=>b.flow_index===fi);
  const balHtml=bal?renderFlowBalance(bal):'';

  return '<div class="tabs">'+
      '<span class="tab act" data-t="nd-hi">Overview</span>'+
      '<span class="tab" data-t="nd-sw">Stream switch</span>'+
      '<span class="tab" data-t="nd-hops">All hops</span>'+
    '</div>'+
    '<div class="tabbody">'+
      '<div id="tab-nd-hi">'+
        prodHtml+consHtml+
        (balHtml?'<div class="kv"><b>supply / demand:</b></div>'+balHtml:'')+
        '<div class="kv"><b>DMA channels:</b></div>'+chanTable+
      '</div>'+
      '<div id="tab-nd-sw" class="hide">'+
        '<div class="lref">stream-switch connections from routingprovenancemap</div>'+
        swTable+
        (gmioTable?'<div class="kv" style="margin-top:10px"><b>GMIO channel</b>'
          +' <span class="dimtxt">(DMA engine on shim, data path: MemTile SRAM to NoC to DDR)</span></div>'
          +gmioTable:'')
      +'</div>'+
      '<div id="tab-nd-hops" class="hide">'+
        '<div class="lref">hop-by-hop path from dmaphopprovenacemap</div>'+
        hopTable+
      '</div>'+
    '</div>';
}

function svgN(tag,attrs){
  const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const[k,v] of Object.entries(attrs)) el.setAttribute(k,v);
  return el;
}
function svgT(el, txt){ el.textContent=txt; return el; }

let dmTooltipEl=null;
let dmFlowHoverFi=null;
let dmFlowHoverTimer=null;
function dmTagFlowLine(el, fi, cls, normW, normO, hoverW, hoverO){
  el.setAttribute('class', cls);
  el.setAttribute('data-fi', String(fi));
  el.dataset.normW=normW; el.dataset.normO=normO;
  el.dataset.hoverW=hoverW; el.dataset.hoverO=hoverO;
}
function dmTagFlowSq(el, fi, normFo, hoverFo){
  el.setAttribute('class', 'dm-shmemsq');
  el.setAttribute('data-fi', String(fi));
  el.dataset.normFo=normFo; el.dataset.hoverFo=hoverFo;
}
function dmApplyFlowHover(fi, on){
  const svg=document.getElementById('devmap-svg');
  if(!svg||fi==null) return;
  svg.querySelectorAll('.dm-flowvis[data-fi="'+fi+'"],.dm-shmemvis[data-fi="'+fi+'"]')
    .forEach(v=>{
      v.setAttribute('stroke-width', on?v.dataset.hoverW:v.dataset.normW);
      v.setAttribute('stroke-opacity', on?v.dataset.hoverO:v.dataset.normO);
    });
  svg.querySelectorAll('.dm-shmemsq[data-fi="'+fi+'"]').forEach(r=>{
    r.setAttribute('fill-opacity', on?r.dataset.hoverFo:r.dataset.normFo);
  });
}
function dmFlowHoverIn(fi, e, tipLines){
  if(dmFlowHoverTimer){ clearTimeout(dmFlowHoverTimer); dmFlowHoverTimer=null; }
  if(dmFlowHoverFi!==fi){
    if(dmFlowHoverFi!=null) dmApplyFlowHover(dmFlowHoverFi, false);
    dmFlowHoverFi=fi;
    dmApplyFlowHover(fi, true);
  }
  if(tipLines&&tipLines.length) dmShowTip(e.clientX, e.clientY, tipLines);
}
function dmFlowHoverLeave(fi){
  if(dmFlowHoverTimer) clearTimeout(dmFlowHoverTimer);
  dmFlowHoverTimer=setTimeout(()=>{
    if(dmFlowHoverFi===fi){
      dmApplyFlowHover(fi, false);
      dmFlowHoverFi=null;
      dmHideTip();
    }
    dmFlowHoverTimer=null;
  }, 40);
}
function dmWireFlowHit(hit, fi, tipLines){
  if(!hit) return;
  hit.addEventListener('mouseenter', e=>dmFlowHoverIn(fi, e, tipLines));
  hit.addEventListener('mousemove', dmMoveTip);
  hit.addEventListener('mouseleave', ()=>dmFlowHoverLeave(fi));
}
function dmShowTip(clientX, clientY, lines){
  dmHideTip();
  const d=document.createElement('div');
  d.style.cssText='position:fixed;z-index:9999;pointer-events:none;'
    +'background:#1a1c27;border:1px solid #2a2e46;border-radius:4px;'
    +'padding:5px 9px;font:10px/1.5 ui-monospace,monospace;color:#e2e4f0;'
    +'white-space:nowrap;box-shadow:0 4px 16px rgba(0,0,0,.6);';
  d.innerHTML=lines.map(l=>'<div>'+l+'</div>').join('');
  document.body.appendChild(d);
  const tw=d.offsetWidth, th=d.offsetHeight;
  let lx=clientX+14, ly=clientY-th-10;
  if(lx+tw>window.innerWidth-8) lx=clientX-tw-14;
  if(ly<8) ly=clientY+14;
  d.style.left=lx+'px'; d.style.top=ly+'px';
  dmTooltipEl=d;
}
function dmHideTip(){
  if(dmTooltipEl){ dmTooltipEl.remove(); dmTooltipEl=null; }
}
function dmMoveTip(e){
  if(!dmTooltipEl) return;
  const tw=dmTooltipEl.offsetWidth, th=dmTooltipEl.offsetHeight;
  let lx=e.clientX+14, ly=e.clientY-th-10;
  if(lx+tw>window.innerWidth-8) lx=e.clientX-tw-14;
  if(ly<8) ly=e.clientY+14;
  dmTooltipEl.style.left=lx+'px'; dmTooltipEl.style.top=ly+'px';
}

function buildDeviceMap(){
  if(!DATA.tiles||!DATA.tiles.length){ console.warn('buildDeviceMap: no tiles'); return; }
  dmFlowHoverFi=null;
  if(dmFlowHoverTimer){ clearTimeout(dmFlowHoverTimer); dmFlowHoverTimer=null; }
  dmHideTip();
  buildNetBar();
  const svg=document.getElementById('devmap-svg');
  svg.innerHTML='';

  // Layout: compute tile size to fill the viewport well at fit-zoom.
  // Gaps are 30% of tile size so tiles dominate the space.
  const vp=document.getElementById('devmap-vp');
  const vpW=vp.clientWidth||800, vpH=vp.clientHeight||600;
  const ML=44, MT=28, MR=16, MB=12;

  const _swShimKinds=new Set(['shim_aie_to_ext','shim_ext_to_aie']);
  const _tileConnKeys=new Map();
  if(dmShowSW&&!dmHideAll){
    (DATA.comm_paths||[]).forEach(p=>{
      if(dmActiveNets.size>0&&!dmActiveNets.has(p.flow_index)) return;
      const fts=_flowTileSet(p);
      const pathPkt=(p.routing_connections||[]).filter(c=>c.kind==='packet_connect');
      const sharedPktFwd=_sharedPktForwardMaster(pathPkt);
      (p.routing_connections||[]).forEach(c=>{
        const t=c.tile||{};
        if(_swShimKinds.has(c.kind)) return;
        const tkey=t.col+','+t.row;
        if(dmSwCollapsed.has(tkey)) return;
        if(fts.size>0&&!fts.has(tkey)) return;
        let ck;
        if(c.kind==='packet_connect'){
          _expandPktConnectRows(c,sharedPktFwd).forEach(row=>{
            if(!_tileConnKeys.has(tkey)) _tileConnKeys.set(tkey,new Set());
            _tileConnKeys.get(tkey).add(p.flow_index+'|'+_pktRowKey(row));
          });
          return;
        }
        const s=c.slave||{},m=c.master||{};
        ck=p.flow_index+'|'+c.kind+'|'+s.dir+'|'+s.idx+'|'+m.dir+'|'+m.idx;
        if(!_tileConnKeys.has(tkey)) _tileConnKeys.set(tkey,new Set());
        _tileConnKeys.get(tkey).add(ck);
      });
    });
  }
  const _maxConns=_tileConnKeys.size?Math.max(...[..._tileConnKeys.values()].map(s=>s.size)):0;
  // No rows to draw anywhere (routing info off, all flows hidden, or every tile
  // collapsed) → zero rows, so the boxes shrink instead of reserving dead space.
  const SW_ROWS=(dmShowSW&&_maxConns)?Math.min(24,Math.max(4,_maxConns)):0;

  // Max number of DMA badges any single SW row needs (one per flow with a local DMA channel).
  let _maxDmaBadges=1;
  if(dmShowSW){
    (DATA.tiles||[]).forEach(t=>{
      if(dmSwCollapsed.has(t.loc[0]+','+t.loc[1])) return;
      const localFiSet=new Set((t.dma_channels||[]).map(ch=>ch.flow_index));
      _tileRoutingConns(t.loc[0],t.loc[1]).forEach(c=>{
        const withDma=(c.flow_indices||[c.flow_index]).filter(fi=>localFiSet.has(fi));
        if(withDma.length>_maxDmaBadges) _maxDmaBadges=withDma.length;
      });
    });
  }
  // Extra tile width: each additional badge is ~28px (label ≤7 chars * 3.55 + 4 + 2px gap).
  const TW=148+(_maxDmaBadges-1)*30, TH=dmShowSW?(52+SW_ROWS*8):72, GX=24, GY=14;
  const COLSTEP=TW+GX, ROWSTEP=TH+GY;

  // Merge DATA.tiles with every tile referenced in comm_paths (waypoints + hops)
  const tileMap={};
  DATA.tiles.forEach(t=>{ tileMap[t.loc[0]+','+t.loc[1]]=t; });
  // Track which tiles are shmem hop endpoints so they always get a box.
  const shmemEndKeys=new Set();
  // Track tiles explicitly listed in comm_paths.tiles — these are named routing
  // tiles that must render a box regardless of whether they appear in DATA.tiles.
  const routedTileKeys=new Set();
  (DATA.comm_paths||[]).forEach(p=>{
    (p.hops||[]).filter(h=>h.type==='shmem').forEach(h=>{
      shmemEndKeys.add(h.from_col+','+h.from_row);
      shmemEndKeys.add(h.to_col+','+h.to_row);
    });
    // Register tiles from both edges and the tiles list
    (p.tiles||[]).forEach(([c,r])=>{
      const k=c+','+r;
      routedTileKeys.add(k);
      if(!tileMap[k]) tileMap[k]={loc:[c,r],type:'mem',dma_channels:[]};
    });
    (p.edges||[]).forEach(([src,dst])=>{
      [[src[0],src[1]],[dst[0],dst[1]]].forEach(([c,r])=>{
        const k=c+','+r;
        if(!tileMap[k]) tileMap[k]={loc:[c,r],type:r===0?'shim':'mem',dma_channels:[]};
      });
    });
  });
  // Pure stream-switch waypoints: tiles synthesised from comm_paths edges only
  // (not in DATA.tiles, not in comm_paths.tiles, not a shim, not a shmem endpoint,
  // no DMA channels). These are implicit path endpoints that the stream line passes
  // through invisibly. Tiles in comm_paths.tiles are explicitly named routing tiles
  // and must always render a visible box.
  const waypointKeys=new Set();
  const dataTileKeys=new Set(DATA.tiles.map(t=>t.loc[0]+','+t.loc[1]));
  Object.values(tileMap).forEach(t=>{
    const k=t.loc[0]+','+t.loc[1];
    if(!dataTileKeys.has(k) && !routedTileKeys.has(k) && t.loc[1]!==0 && !shmemEndKeys.has(k) && !(t.dma_channels&&t.dma_channels.length))
      waypointKeys.add(k);
  });

  // Synthesize MEM tiles for columns that have a shim (row 0) and a core row.
  // The stream-switch path always traverses these rows physically even when the
  // abstract hop chain skips them. Use device_core_min_row (physical device
  // geometry) rather than the schedule's lowest core row — the schedule may only
  // mention a core tile at row 4 while the device's mem rows are only 1-2.
  const coreRowSet=new Set(DATA.grid.core_rows||[]);
  const deviceCoreMinRow=DATA.grid.device_core_min_row||3;
  const shimCols=new Set(), coreCols=new Set();
  Object.values(tileMap).forEach(t=>{
    if(t.loc[1]===0) shimCols.add(t.loc[0]);
    if(coreRowSet.has(t.loc[1])) coreCols.add(t.loc[0]);
  });
  const memCols=new Set([...shimCols].filter(c=>coreCols.has(c)));
  // Mem rows are 1..(deviceCoreMinRow-1) inclusive — fixed device geometry.
  const memRowsToSynth=[];
  for(let r=1;r<deviceCoreMinRow;r++) memRowsToSynth.push(r);
  memCols.forEach(c=>{
    memRowsToSynth.forEach(r=>{
      const k=c+','+r;
      if(!tileMap[k]) tileMap[k]={loc:[c,r],type:'mem',dma_channels:[]};
    });
  });
  // Drop synthesized MEM tiles whose row exceeds the declared grid — these are
  // implicit waypoints the edge-tracing added with the wrong type. But keep any
  // tile that was explicitly named in comm_paths.tiles, a shmem endpoint, or a
  // DATA.tiles entry — those are real tiles the schedule uses.
  const gridMaxRow=(DATA.grid.rows||0)-1;
  Object.keys(tileMap).forEach(k=>{
    const t=tileMap[k];
    if(t.loc[1]>gridMaxRow
       && !dataTileKeys.has(k)
       && !routedTileKeys.has(k)
       && !shmemEndKeys.has(k))
      delete tileMap[k];
  });

  const allTiles=Object.values(tileMap);
  const allCols=[...new Set(allTiles.map(t=>t.loc[0]))].sort((a,b)=>a-b);
  const allRows=[...new Set(allTiles.map(t=>t.loc[1]))].sort((a,b)=>a-b);
  const minC=allCols[0], maxC=allCols[allCols.length-1];
  const maxR=allRows[allRows.length-1];
  const NCOLS=maxC-minC+1, NROWS=maxR+1;

  const SVG_W=ML+NCOLS*TW+(NCOLS-1)*GX+MR;
  const SVG_H=MT+NROWS*TH+(NROWS-1)*GY+MB;
  svg.setAttribute('width',SVG_W); svg.setAttribute('height',SVG_H);
  svg.setAttribute('viewBox','0 0 '+SVG_W+' '+SVG_H);

  // Position helpers (mirrors the reference exactly)
  const tx=c=>ML+(c-minC)*COLSTEP;
  const ty=r=>MT+(maxR-r)*ROWSTEP;
  const cx=c=>tx(c)+TW/2;
  const cy=r=>ty(r)+TH/2;
  const fcx=c=>tx(c)+Math.round(TW*0.70);
  const fcy=r=>ty(r)+Math.round(TH*0.38);

  // Lane spacing. Scaled down as the flow count grows so the whole fan stays a
  // fraction of the tile pitch: 12 flows at a fixed 5px would span 55px of a
  // 72px row step and stray off the tiles.
  const OX_STEP=Math.max(2, Math.min(5,
    Math.round(0.25*COLSTEP/Math.max(1, dmFlowIds.length-1))));
  const OY_STEP=Math.max(1, Math.min(3,
    Math.round(0.25*ROWSTEP/Math.max(1, dmFlowIds.length-1))));
  // One lane per flow, held for the whole path.
  //
  // This used to fan flows out per EDGE, by how many flows shared that
  // particular hop — so a flow's own offset changed from segment to segment as
  // the sharing changed along its route, and the line arrived at a tile on one
  // lane and left on another. That is the gap-and-jog: e.g. net3 ran its first
  // hop unshared at offset 0 and the rest at -12, a 12px break mid-route.
  //
  // A global lane also removes an overlap the per-edge scheme allowed: two
  // flows that never share an edge both got offset 0 and drew on top of each
  // other. The band is the same width the surrounding code already reserves
  // for streams (streamMaxX/Y = OX_STEP*(N-1)/2).
  function flowLane(fi){
    const i=dmFlowIds.indexOf(fi), n=dmFlowIds.length;
    return {
      ox: Math.round((i-(n-1)/2)*OX_STEP),
      oy: Math.round((i-(n-1)/2)*OY_STEP),
    };
  }
  function edgeOffset(e, fi){ return flowLane(fi); }
  function dotOffset(p, tc, tr){ return flowLane(p.flow_index); }
  // N is used by shmem streamMaxX/Y to reserve lanes outside the stream offset band.
  const N=dmFlowIds.length;

  // ── Defs: arrowhead markers (userSpaceOnUse = no scale distortion) ──
  const defs=svgN('defs',{});
  svg.appendChild(defs);
  dmFlowIds.forEach(fi=>{
    const mk=svgN('marker',{id:'ar-'+fi,markerWidth:'9',markerHeight:'9',
      refX:'7',refY:'4.5',orient:'auto','markerUnits':'userSpaceOnUse'});
    mk.appendChild(svgN('path',{d:'M0,1 L8,4.5 L0,8 Z',fill:dmColor(fi)}));
    defs.appendChild(mk);
  });
  const mkInt=svgN('marker',{id:'ar-int',markerWidth:'8',markerHeight:'8',
    refX:'6.5',refY:'4',orient:'auto','markerUnits':'userSpaceOnUse'});
  mkInt.appendChild(svgN('path',{d:'M0,1 L7,4 L0,7 Z',fill:'#e4e4e45e'}));
  defs.appendChild(mkInt);

  // ── LAYER 1: axis labels (bottom layer) ─────────────────────────
  for(let c=minC;c<=maxC;c++){
    svg.appendChild(svgT(svgN('text',{x:cx(c),y:'20','text-anchor':'middle',
      'font-size':'12','font-family':'monospace',fill:'#e4e4e45e'}), 'col '+c));
  }
  for(let r=0;r<=maxR;r++){
    svg.appendChild(svgT(svgN('text',{x:ML-12,y:cy(r)+4,'text-anchor':'end',
      'font-size':'11','font-family':'monospace',fill:'#e4e4e45e'}), 'r'+r));
  }

  // ── LAYER 2: tile rectangles (drawn first → lines go on top) ─────
  const tileType=r=>{
    if(r===0) return 'shim';
    if(r>=deviceCoreMinRow) return 'core';
    return 'mem';
  };
  function dmTileStyle(ttype){
    if(ttype==='shim') return {fill:'var(--tile-shim-fill)', stroke:'var(--tile-shim-stroke)',
      label:'#78a8e0'};
    if(ttype==='mem') return {fill:'var(--tile-mem-fill)', stroke:'var(--tile-mem-stroke)',
      label:'#a898d8'};
    return {fill:'var(--tile-core-fill)', stroke:'var(--tile-core-stroke)', label:'#78c890'};
  }
  const tileGroups={};

  allTiles.forEach(t=>{
    const [tc,tr]=t.loc;
    const key=tc+','+tr;
    // Pure stream-switch waypoints: no box, but still register in tileGroups so
    // the hover and click handlers can find them (they show pass-through info).
    if(waypointKeys.has(key)){
      const g=svgN('g',{class:'dm-tile','data-key':key,'data-waypoint':'1',
        opacity:'0',cursor:'pointer'});
      tileGroups[key]=g;
      dmTileStroke[key]='';
      dmTileFill[key]='';
      svg.appendChild(g);
      return;
    }
    const ttype=tileType(tr);
    const ts=dmTileStyle(ttype);
    const fill=ts.fill;
    const stroke=ts.stroke;

    const g=svgN('g',{opacity:'1',cursor:'pointer',class:'dm-tile','data-key':key,
      'data-basestroke':stroke});
    tileGroups[key]=g;
    dmTileStroke[key]=stroke;
    dmTileFill[key]=fill;

    const rect=svgN('rect',{x:tx(tc),y:ty(tr),width:TW,height:TH,rx:'6',
      fill,stroke,'stroke-width':'1'});
    g.appendChild(rect);
    // Empty placeholder so dmPaintStatus() only has to set textContent — it must
    // not add nodes, since it runs on every poll.
    g.appendChild(svgN('text',{class:'dm-stlabel',x:tx(tc)+TW-6,y:ty(tr)+TH-5,
      'text-anchor':'end','font-size':'8','font-family':'monospace',
      'font-weight':'700','pointer-events':'none'}));

    // Coord top-left, type badge top-right
    const typStr=ttype==='shim'?'SHIM':ttype==='mem'?'MEM':'AIE';
    g.appendChild(svgT(svgN('text',{x:tx(tc)+6,y:ty(tr)+12,
      'font-size':'8','font-family':'monospace',fill:'#e4e4e848'}),
      '('+tc+','+tr+')'));
    g.appendChild(svgT(svgN('text',{x:tx(tc)+TW-6,y:ty(tr)+12,
      'text-anchor':'end','font-size':'7','font-family':'monospace',fill:ts.label}),
      typStr));

    if(dmShowSW&&!dmSwCollapsed.has(key)){
      let rconns=_tileRoutingConns(tc,tr, ROUTING_DIFF ? STATIC_PATHS : undefined);
      if(dmHideAll){
        rconns=[];
      } else if(dmActiveNets.size>0){
        rconns=rconns.filter(c=>(c.flow_indices||[c.flow_index]).some(fi=>dmActiveNets.has(fi)));
      }
      const shortDir=_shortDir;
      rconns.slice(0,SW_ROWS).forEach((c,i)=>{
        const isCct=c.kind==='circuit_connect';
        const kc=isCct?'#4a7fd4':'#9c4fd4';
        const y=ty(tr)+20+i*8;
        let label; let pktBadges=[];
        if(c.kind==='packet_hw'){
          label=_fmtPktLabel(c, true);
          pktBadges=_dmPktBadges(c);
        } else {
          const s=c.slave||{}, m=c.master||{};
          label='CCT '+shortDir(s.dir)+':'+s.idx+' → '+shortDir(m.dir)+':'+m.idx;
        }
        g.appendChild(svgN('rect',{x:tx(tc)+5,y:y-6,width:'2',height:'6',
          fill:kc,rx:'1','pointer-events':'none'}));
        g.appendChild(svgT(svgN('text',{x:tx(tc)+10,y,
          'font-size':'6.5','font-family':'monospace',fill:'#c8cad8',
          'pointer-events':'none'}), label));
        let badgeX=tx(tc)+10+label.length*3.9;
        pktBadges.forEach(b=>{
          const bw=_svgBadgeWidth(b.text);
          g.appendChild(svgN('rect',{x:badgeX,y:y-5,width:String(bw),height:'7',
            rx:'2',fill:b.bg,'pointer-events':'none'}));
          g.appendChild(svgT(svgN('text',{x:badgeX+2,y,
            'font-size':'6','font-family':'monospace',fill:b.fg,
            'pointer-events':'none'}), b.text));
          badgeX+=bw+2;
        });
        // Draw one badge per flow that has a local DMA channel on this tile, right-aligned.
        const localFiSet=new Set((t.dma_channels||[]).map(ch=>ch.flow_index));
        const dmaCandidates=(c.flow_indices||[c.flow_index]).filter(fi=>localFiSet.has(fi));
        const dmaBadges=dmaCandidates.length
          ?dmaCandidates.map(fi=>{const lbl=_flowDmaLabel(fi,tc,tr);return _flowDmaBadgeStyle(lbl);})
          :[_flowDmaBadgeStyle('')];
        let bx=tx(tc)+TW-4;
        dmaBadges.forEach(dmaSt=>{
          const bw=_svgBadgeWidth(dmaSt.text);
          bx-=bw;
          g.appendChild(svgN('rect',{x:bx,y:y-5,width:String(bw),height:'7',
            rx:'2',fill:dmaSt.bg,'pointer-events':'none'}));
          g.appendChild(svgT(svgN('text',{x:bx+2,y,
            'font-size':'6','font-family':'monospace',
            fill:dmaSt.fg,'pointer-events':'none'}), dmaSt.text));
          bx-=2;
        });
      });
    }

    if(t.dma_channels&&t.dma_channels.length){
      const chans=(!dmHideAll&&dmActiveNets.size===0)?t.dma_channels
        :t.dma_channels.filter(ch=>!dmHideAll&&dmActiveNets.has(ch.flow_index));
      const ins=chans.filter(ch=>ch.direction==='S2MM');
      const outs=chans.filter(ch=>ch.direction==='MM2S');
      // A collapsed tile drew no routing rows, so pull its DMA badges up into
      // the freed space instead of leaving a gap the height of the global rows.
      const swRowsHere=(dmShowSW&&!dmSwCollapsed.has(key))?SW_ROWS:0;
      const dmaBaseY=ty(tr)+(swRowsHere?(22+swRowsHere*8):22);
      ins.slice(0,3).forEach((ch,i)=>{
        const fc=dmColor(ch.flow_index);
        const y=dmaBaseY+i*10;
        const bd0=(ch.bd_chain||[])[0];
        const bdTag=bd0?' BD'+bd0.bd_id:'';
        g.appendChild(svgT(svgN('text',{x:tx(tc)+7,y,
          'font-size':'7.5','font-family':'monospace',fill:fc}),
          '▶f'+ch.flow_index+bdTag));
      });
      outs.slice(0,3).forEach((ch,i)=>{
        const fc=dmColor(ch.flow_index);
        const y=dmaBaseY+i*10;
        const bd0=(ch.bd_chain||[])[0];
        const bdTag=bd0?' BD'+bd0.bd_id:'';
        g.appendChild(svgT(svgN('text',{x:tx(tc)+TW-7,y,
          'text-anchor':'end','font-size':'7.5','font-family':'monospace',fill:fc}),
          'f'+ch.flow_index+bdTag+'▶'));
      });
    }

    g.addEventListener('mouseenter',e=>{
      rect.setAttribute('stroke','#e4e4e488');
      rect.setAttribute('stroke-width','1.5');
      const lines=['('+tc+','+tr+') '+typStr];
      if(t.dma_channels&&t.dma_channels.length){
        const vis=(!dmHideAll&&dmActiveNets.size===0)?t.dma_channels
          :t.dma_channels.filter(ch=>!dmHideAll&&dmActiveNets.has(ch.flow_index));
        // Terminal channels
        vis.filter(ch=>ch.direction==='S2MM').forEach(ch=>lines.push('in  f'+ch.flow_index));
        vis.filter(ch=>ch.direction==='MM2S').forEach(ch=>lines.push('out f'+ch.flow_index));
        // Pass-through flows (edge touches this tile but it's not a DMA terminal)
        const termFis=new Set(vis.map(ch=>ch.flow_index));
        const tileInEdges=p=>(p.edges||[]).some(e=>
          (e[0][0]===tc&&e[0][1]===tr)||(e[1][0]===tc&&e[1][1]===tr));
        const thru=(DATA.comm_paths||[]).filter(p=>{
          if(dmHideAll||dmActiveNets.size>0&&!dmActiveNets.has(p.flow_index)) return false;
          return tileInEdges(p)&&!termFis.has(p.flow_index);
        });
        if(thru.length) lines.push('── pass-through ──');
        thru.slice(0,4).forEach(p=>lines.push('    f'+p.flow_index+' ('+p.direction+')'));
      } else {
        // Ghost tile — only pass-through
        const tileInEdgesG=p=>(p.edges||[]).some(e=>
          (e[0][0]===tc&&e[0][1]===tr)||(e[1][0]===tc&&e[1][1]===tr));
        const thru=(DATA.comm_paths||[]).filter(p=>{
          if(dmHideAll||dmActiveNets.size>0&&!dmActiveNets.has(p.flow_index)) return false;
          return tileInEdgesG(p);
        });
        if(thru.length) thru.slice(0,4).forEach(p=>lines.push('pass f'+p.flow_index));
        else lines.push('routing tile');
      }
      // Live scan detail, appended to the existing hover card rather than a
      // native SVG <title> — two tooltips on one element fight each other.
      const stCell = dmStatus.cells ? dmStatus.cells[key] : null;
      if(stCell && stCell.state){
        lines.push('── '+(dmStatus.what||'scan')+': '+stCell.state+' ──');
        if(stCell.channels){
          Object.keys(stCell.channels).forEach(cn => {
            const ch = stCell.channels[cn];
            let detail = ch.state || '?';
            if(ch.cur_bd != null) detail += '  last bd=' + ch.cur_bd;
            if(ch.stalls && ch.stalls.length) detail += '  [' + ch.stalls.join(',') + ']';
            lines.push('    '+cn+': '+detail);
          });
        }
        if(stCell.pc) lines.push('    pc '+stCell.pc+(stCell.source?' · '+stCell.source:''));
        if(stCell.core_status) lines.push('    core '+stCell.core_status);
        if(stCell.words) lines.push('    events '+stCell.words.join(' '));
      }
      dmShowTip(e.clientX, e.clientY, lines);
    });
    g.addEventListener('mousemove',dmMoveTip);
    g.addEventListener('mouseleave',()=>{
      if(!dmSelKeys.has(key)){
        // Restore from dmTileStroke, not the captured `stroke`: a live scan may
        // have recolored this tile since it was built, and hovering out must
        // not wipe the status color.
        const cur=dmTileStroke[key]||stroke;
        rect.setAttribute('stroke',cur);
        rect.setAttribute('stroke-width', g.getAttribute('data-state')?'2':'1');
      }
      dmHideTip();
    });
    // Right-click menu. stopPropagation keeps the viewport's blanket
    // preventDefault from being the only handler that sees this event.
    g.addEventListener('contextmenu',e=>{
      e.preventDefault(); e.stopPropagation();
      dmShowTileMenu(e.clientX,e.clientY,tc,tr);
    });

    g.addEventListener('click',e=>{
      if(dmDragging) return;
      const ctrl=e.ctrlKey||e.metaKey||ctrlHeld;
      const selOn=k=>{ const gr=tileGroups[k]; if(!gr) return;
        const r=gr.querySelector('rect'); if(!r) return;
        r.setAttribute('stroke','var(--sel)'); r.setAttribute('stroke-width','2'); };
      const selOff=k=>{ const gr=tileGroups[k]; if(!gr) return;
        const r=gr.querySelector('rect'); if(!r) return;
        r.setAttribute('stroke',dmTileStroke[k]||'var(--stroke,#e4e4e433)');
        r.setAttribute('stroke-width','1'); };
      const match=DATA.tiles.find(dt=>dt.loc[0]===tc&&dt.loc[1]===tr);
      // Mirror the map click onto the grid cell so .sel highlighting stays in sync.
      const gridCell=()=>{ let found=null;
        document.querySelectorAll('#grid .tile').forEach(cell=>{
          const locEl=cell.querySelector('.loc');
          if(locEl&&locEl.textContent==='('+tc+','+tr+')') found=cell; });
        return found; };
      if(ctrl){
        // Ctrl+click: toggle this tile in/out of the selection, keep the rest.
        if(dmSelKeys.has(key)){
          dmSelKeys.delete(key); selOff(key);
          panelRemove(panelKey('tile', tc+','+tr));
          return;
        }
        dmSelKeys.add(key); selOn(key);
        if(match) select(match, gridCell()||g, null, null, true);
        else { selectRoutingTile(tc, tr, true); setConTargetLoc(tc, tr, null); }
      } else {
        // Plain click: clear the selection; clicking the only selected tile deselects.
        const wasOnly = dmSelKeys.size===1 && dmSelKeys.has(key);
        dmSelKeys.forEach(k=>selOff(k));
        dmSelKeys.clear();
        if(wasOnly){ panelRemove(panelKey('tile', tc+','+tr)); return; }
        dmSelKeys.add(key); selOn(key);
        if(match) select(match, gridCell()||g, null, null, false);
        else { selectRoutingTile(tc, tr, false); setConTargetLoc(tc, tr, null); }
      }
    });
    svg.appendChild(g);
  });

  // Re-apply tile selection highlights: the rects above were just recreated, so a
  // rebuild (net click, filter change) would otherwise drop the ctrl+click selection.
  dmSelKeys.forEach(k=>{
    const gr=tileGroups[k]; if(!gr) return;
    const r=gr.querySelector('rect'); if(!r) return;
    r.setAttribute('stroke','var(--sel)'); r.setAttribute('stroke-width','2');
  });

  // ── LAYER 3: packet-switched and shmem links ──────────────────────
  // 'packet' hops: adjacent core tiles connected via packet-routed streams
  //   (e.g. gather path from compute cores to collector) — drawn as dashed lines.
  // 'shmem' hops: true shared-memory transfers not covered by stream-switch — drawn
  //   as shorter dashed lines with a different dash pattern.
  // Both run tile-edge to tile-edge.
  function edgeCoords(fc, fr, tc, tr){
    const dc=tc-fc, dr=tr-fr;
    if(dc>0)       return [tx(fc)+TW, cy(fr), tx(tc),    cy(tr)];
    if(dc<0)       return [tx(fc),    cy(fr), tx(tc)+TW, cy(tr)];
    if(dr>0)       return [fcx(fc), ty(fr),    fcx(tc), ty(tr)+TH];
    /* dr<0 */     return [fcx(fc), ty(fr)+TH, fcx(tc), ty(tr)   ];
  }
  // Shmem links live in a RESERVED lane outside the stream-edge offset band, so
  // dashed shared-memory links never draw on top of the solid stream edges.
  // Stream edges are fanned within +/-streamMax(X|Y) around tile centres; shmem
  // links start just beyond that band and step outward, alternating sides so
  // several links between the same tile pair (e.g. swapped-neighbour DMA/kernel
  // bridges running both ways) each get their own lane.
  const streamMaxX = OX_STEP*(N-1)/2;
  const streamMaxY = OY_STEP*(N-1)/2;
  const SH_GAP=7, SH_STEP=7;
  const shKey = h => {
    const a=h.from_col+','+h.from_row, b=h.to_col+','+h.to_row;
    return a<b ? a+'-'+b : b+'-'+a;
  };
  const shGroups={};
  (DATA.comm_paths||[]).forEach(p=>{
    (p.hops||[]).filter(h=>h.type==='shmem').forEach(h=>{
      (shGroups[shKey(h)]=shGroups[shKey(h)]||[]).push(h);
    });
  });
  const shIndex=new Map();
  Object.values(shGroups).forEach(list=>{
    list.forEach((h,i)=>shIndex.set(h,{idx:i,count:list.length}));
  });
  // Reserved-lane offset for the idx-th shmem link of a tile-pair group.
  // idx 0 -> +base, idx 1 -> -base, idx 2 -> +base+step, ... (alternating sides).
  function shOffset(idx, streamMax){
    const base = streamMax + SH_GAP;
    const side = (idx % 2 === 0) ? 1 : -1;
    const lane = Math.floor(idx / 2);
    return side * (base + lane*SH_STEP);
  }

  // ── LAYER 2.5: symbol-search highlights ──────────────────────────
  // Drawn here (after edgeCoords/shIndex/shOffset/edgeOffset are all defined)
  // so halos use the exact same coordinates as the actual drawn lines.
  if(srSearchTerms.size){
    const {tileLockKeys,flowLockFis}=lkActiveSets();

    // Tile overlays — semi-transparent yellow rect inset from the tile border
    tileLockKeys.forEach(tkey=>{
      const parts=tkey.split(','); const tc=parseInt(parts[0]), tr=parseInt(parts[1]);
      svg.appendChild(svgN('rect',{
        x:tx(tc)+2,y:ty(tr)+2,width:TW-4,height:TH-4,rx:'5',
        fill:'rgba(255,210,0,0.18)',stroke:'#ffd200',
        'stroke-width':'2','stroke-dasharray':'4 3',
        'pointer-events':'none'}));
    });

    // Stream edge halos — match the per-edge (ox,oy) spread offset used by LAYER 4.
    (DATA.comm_paths||[]).forEach(p=>{
      if(!flowLockFis.has(p.flow_index)) return;
      const fi=p.flow_index;
      (p.edges||[]).forEach(e=>{
        const {ox,oy}=edgeOffset(e,fi);
        const [fc,fr]=e[0],[tc,tr]=e[1];
        svg.appendChild(svgN('line',{
          x1:fcx(fc)+ox,y1:fcy(fr)+oy,x2:fcx(tc)+ox,y2:fcy(tr)+oy,
          stroke:'rgba(255,210,0,0.40)','stroke-width':'10',
          'stroke-linecap':'round','pointer-events':'none'}));
      });
    });

    // Shmem link halos — match edgeCoords() + shOffset() used by LAYER 3.
    (DATA.comm_paths||[]).forEach(p=>{
      (p.hops||[]).forEach(h=>{
        if(h.type!=='shmem') return;
        const fk=h.from_col+','+h.from_row, tk=h.to_col+','+h.to_row;
        if(!tileLockKeys.has(fk)&&!tileLockKeys.has(tk)&&!flowLockFis.has(p.flow_index)) return;
        let [x1,y1,x2,y2]=edgeCoords(h.from_col,h.from_row,h.to_col,h.to_row);
        const gi=shIndex.get(h)||{idx:0,count:1};
        const vertical=h.from_row!==h.to_row;
        if(vertical){ const o=shOffset(gi.idx,streamMaxX); x1+=o; x2+=o; }
        else         { const o=shOffset(gi.idx,streamMaxY); y1+=o; y2+=o; }
        svg.appendChild(svgN('line',{
          x1,y1,x2,y2,
          stroke:'rgba(255,210,0,0.35)','stroke-width':'9',
          'stroke-linecap':'round','pointer-events':'none'}));
      });
    });
  }

  (DATA.comm_paths||[]).forEach(p=>{
    const fi=p.flow_index;
    const dim=dmHideAll||(dmActiveNets.size>0&&!dmActiveNets.has(fi));
    if(dim) return;
    const color=dmColor(fi);
    const RAIL=2.4;   // half-gap between the two rails of a ping-pong window link
    const BUFSZ=6;    // buffer glyph (filled square) side
    (p.hops||[]).filter(h=>h.type==='shmem').forEach(h=>{
      let [x1,y1,x2,y2]=edgeCoords(h.from_col,h.from_row,h.to_col,h.to_row);
      const gi=shIndex.get(h)||{idx:0,count:1};
      const vertical = h.from_row!==h.to_row;
      // Horizontal link -> offset vertically; vertical link -> offset horizontally.
      if(vertical){ const o=shOffset(gi.idx,streamMaxX); x1+=o; x2+=o; }
      else        { const o=shOffset(gi.idx,streamMaxY); y1+=o; y2+=o; }
      // Differentiate the two shared-memory kinds so they read at any zoom:
      //   'dma'    = direct shared-memory write/read (DMA<->kernel):
      //             a single dashed line.
      //   'window' = ping-pong buffered window between two kernels:
      //             two parallel rails (the ping/pong buffers) + a filled
      //             square (buffer glyph) at the midpoint.
      if(h.kind==='dma'){
        const vis=svgN('line',{
          x1, y1, x2, y2,
          stroke:color,
          'stroke-width':'1.5',
          'stroke-opacity':'0.6',
          'stroke-dasharray':'5 3',
          'stroke-linecap':'round',
          'marker-end':'url(#ar-'+fi+')'});
        svg.appendChild(vis);
        dmTagFlowLine(vis, fi, 'dm-shmemvis', '1.5', '0.6', '2.6', '0.95');
      } else {
        const dx = vertical ? RAIL : 0;
        const dy = vertical ? 0 : RAIL;
        [-1, 1].forEach(s=>{
          const ln=svgN('line',{
            x1:x1+s*dx, y1:y1+s*dy, x2:x2+s*dx, y2:y2+s*dy,
            stroke:color,
            'stroke-width':'1.6',
            'stroke-opacity':'0.8',
            'stroke-linecap':'round',
            'marker-end':'url(#ar-'+fi+')'});
          svg.appendChild(ln);
          dmTagFlowLine(ln, fi, 'dm-shmemvis', '1.6', '0.8', '2.8', '0.98');
        });
        const mx=(x1+x2)/2, my=(y1+y2)/2;
        const sq=svgN('rect',{
          x:mx-BUFSZ/2, y:my-BUFSZ/2, width:BUFSZ, height:BUFSZ,
          fill:color, 'fill-opacity':'0.9',
          stroke:'#181818', 'stroke-width':'0.8'});
        svg.appendChild(sq);
        dmTagFlowSq(sq, fi, '0.9', '1');
      }
      // Wide transparent hit area (covers both rails for window links).
      const hit=svgN('line',{x1,y1,x2,y2,stroke:'transparent','stroke-width':'12',
        cursor:'pointer','pointer-events':'stroke'});
      const kindLbl=h.kind==='window'?'shmem window (ping-pong)':'shmem dma bridge';
      const flowTip=['flow f'+fi+(p.id?' ('+p.id+')':'')+' · '+p.direction, kindLbl];
      dmWireFlowHit(hit, fi, flowTip);
      hit.addEventListener('click',ev=>{
        if(dmDragging) return;
        ev.stopPropagation();
        dmSelectNet(fi,ev.ctrlKey||ev.metaKey);
      });
      svg.appendChild(hit);
    });
  });

  // ── LAYER 4: stream edges (over tiles, colored) ─────────────────
  // Two-pass render: dim flows first, then active flow on top.
  // Each visible line gets an invisible wide hit-area line on top for easy clicking.
  function drawEdges(p, dim){
    const fi=p.flow_index;
    const color=dmColor(fi);
    const edges=p.edges||[];
    const srcKeys=new Set(edges.map(e=>e[0][0]+','+e[0][1]));
    edges.forEach(e=>{
      const [fc,fr]=e[0], [tc,tr]=e[1];
      const {ox,oy}=edgeOffset(e,fi);
      const isTerminal=!srcKeys.has(tc+','+tr);
      const x1=fcx(fc)+ox, y1=fcy(fr)+oy, x2=fcx(tc)+ox, y2=fcy(tr)+oy;
      // Status casing, drawn first so it sits under the identity-colored line.
      // Starts transparent; dmPaintStatus() fills it in when a scan lands.
      if(!dim){
        svg.appendChild(svgN('line',{
          x1, y1, x2, y2, class:'dm-edgecase', 'data-fi':fi,
          stroke:'transparent', 'stroke-width':'9', 'stroke-opacity':'0.55',
          'stroke-linecap':'round', 'pointer-events':'none'}));
      }
      const vis=svgN('line',{
        x1, y1, x2, y2,
        stroke:color,
        'stroke-width':dim?'0.7':'1.5',
        'stroke-opacity':dim?'0.05':'0.6',
        'stroke-linecap':'round'});
      svg.appendChild(vis);
      if(!dim) dmTagFlowLine(vis, fi, 'dm-flowvis', '1.5', '0.6', '2.8', '0.95');
      // Wide transparent hit area — only on active lines so dim flows aren't clickable.
      if(!dim){
        const hit=svgN('line',{x1,y1,x2,y2,stroke:'transparent','stroke-width':'12',
          cursor:'pointer','pointer-events':'stroke'});
        const flowTip=['flow f'+fi+(p.id?' ('+p.id+')':'')+' · '+p.direction];
        dmWireFlowHit(hit, fi, flowTip);
        hit.addEventListener('click',e=>{
          if(dmDragging) return;
          e.stopPropagation();
          dmSelectNet(fi,e.ctrlKey||e.metaKey);
        });
        svg.appendChild(hit);
      }
    });
  }
  if(dmHideAll){
    // Hide-all mode: draw nothing.
  } else if(dmActiveNets.size===0){
    // All-nets mode: draw every flow bright.
    (DATA.comm_paths||[]).forEach(p=>drawEdges(p, false));
  } else {
    // Filtered mode: draw dim flows first, active on top.
    (DATA.comm_paths||[]).forEach(p=>{
      if(!dmActiveNets.has(p.flow_index)) drawEdges(p, true);
    });
    (DATA.comm_paths||[]).forEach(p=>{
      if(dmActiveNets.has(p.flow_index)) drawEdges(p, false);
    });
  }

  // Small directional arrow drawn next to source/destination dots.
  // (dx,dy) is the unit direction the arrow points (toward next tile for sources,
  // away from prev tile for destinations — i.e. always "the direction data travels").
  // outward=true  → source: base just outside dot, tip points away (data leaving)
  // outward=false → dest:   tip just outside dot, base further back (data arriving)
  // (dx,dy) = unit vector in the direction data flows.
  // outward=true  (source):      arrow exits the dot — base just outside on downstream side, tip further downstream.
  // outward=false (destination): arrow on the line upstream — tip just outside dot on upstream side, base further back.
  // gap overrides the clearance from dot centre (default 7px, just past r=4.5px dot edge).
  let dotsG = null;
  function svgArrow(x, y, dx, dy, color, outward, gap=4){
    const LEN=4, HALF=1.5;
    const nx=-dy, ny=dx;
    let tx, ty, bx, by;
    if(outward){
      bx=x+dx*gap;   by=y+dy*gap;
      tx=bx+dx*LEN;  ty=by+dy*LEN;
    } else {
      tx=x-dx*gap;   ty=y-dy*gap;
      bx=tx-dx*LEN;  by=ty-dy*LEN;
    }
    const b1x=bx+nx*HALF, b1y=by+ny*HALF;
    const b2x=bx-nx*HALF, b2y=by-ny*HALF;
    (dotsG||svg).appendChild(svgN('polygon',{points:`${tx},${ty} ${b1x},${b1y} ${b2x},${b2y}`,fill:color,opacity:'0.9'}));
  }
  // Normalise a grid-space direction vector to unit length (Manhattan tiles only).
  function gridDir(fromC, fromR, toC, toR){
    const dc=toC-fromC, dr=toR-fromR;
    const len=Math.sqrt(dc*dc+dr*dr)||1;
    return [dc/len, -dr/len];   // -dr because SVG y is inverted relative to row index
  }

  // ── LAYER 5a: solid dots (sources, contributors, forks) ──────────
  // Solid = injects into stream: source (push origin) or contributor (pull gather inject).
  // Fork = pure routing split (not a data producer or consumer).
  // Drawn before hollow dots so hollow dots always render on top.
  dotsG = svgN('g',{'pointer-events':'none'});
  (DATA.comm_paths||[]).forEach(p=>{
    const fi=p.flow_index;
    const active=!dmHideAll&&(dmActiveNets.size===0||dmActiveNets.has(fi));
    if(!active) return;
    const color=dmColor(fi);
    const edges=p.edges||[];
    if(!edges.length) return;

    const dmaTileSet=new Set((p.dma_tiles||[]).map(t=>t[0]+','+t[1]));
    const pktTileSet=new Set((p.packet_tiles||[]).map(t=>t[0]+','+t[1]));
    const isPull=p.direction==='pull';

    const outCount={}, inCount={};
    edges.forEach(e=>{
      const sk=e[0][0]+','+e[0][1], dk=e[1][0]+','+e[1][1];
      outCount[sk]=(outCount[sk]||0)+1;
      inCount[dk]=(inCount[dk]||0)+1;
    });

    // Source dot — solid filled circle: no incoming edges, not a gather contributor.
    const srcTile=edges.find(e=>!inCount[e[0][0]+','+e[0][1]]);
    if(srcTile){
      const [sc,sr]=srcTile[0];
      if(!(isPull&&pktTileSet.has(sc+','+sr))){
        const {ox,oy}=dotOffset(p,sc,sr);
        dotsG.appendChild(svgN('circle',{cx:fcx(sc)+ox,cy:fcy(sr)+oy,r:'1.8',
          fill:color,stroke:'#181818','stroke-width':'0.7'}));
        // Arrow at source dot (push flows only — pull contributors use solid dot without arrow).
        if(!isPull){
          const [tc2,tr2]=srcTile[1];
          if(tr2>=0){ const [dx,dy]=gridDir(sc,sr,tc2,tr2); svgArrow(fcx(sc)+ox,fcy(sr)+oy,dx,dy,color,true); }
        }
      }
    }

    // Contributor dots (pull flows) — solid: each packet_tile injects computed results
    // into the gather stream. Outward arrow shows data leaving toward the collector.
    if(isPull){
      const seen=new Set();
      (p.packet_tiles||[]).forEach(t=>{
        const [tc,tr]=t, k=tc+','+tr;
        if(seen.has(k)) return;
        seen.add(k);
        const {ox,oy}=dotOffset(p,tc,tr);
        dotsG.appendChild(svgN('circle',{cx:fcx(tc)+ox,cy:fcy(tr)+oy,r:'1.8',
          fill:color,stroke:'#181818','stroke-width':'0.7'}));
        const outEdge=edges.find(e=>e[0][0]===tc&&e[0][1]===tr);
        if(outEdge){ const [dx,dy]=gridDir(tc,tr,outEdge[1][0],outEdge[1][1]); svgArrow(fcx(tc)+ox,fcy(tr)+oy,dx,dy,color,true); }
      });
    }

    // Fork dots — solid with outer ring: pure split point (1 in, 2+ out), not a DMA tap.
    const forkSeen=new Set();
    edges.forEach(e=>{
      const [fc,fr]=e[0], k=fc+','+fr;
      if(fr<0) return;
      const outC=outCount[k]||0, inC=inCount[k]||0;
      if(outC>=2 && inC===1 && !dmaTileSet.has(k) && !forkSeen.has(k)){
        forkSeen.add(k);
        const {ox,oy}=dotOffset(p,fc,fr);
        dotsG.appendChild(svgN('circle',{
          cx:fcx(fc)+ox,cy:fcy(fr)+oy,
          r:'2',fill:color,stroke:'#e4e4e4','stroke-width':'0.8'}));
      }
    });
  });

  // ── LAYER 5b: hollow dots — drawn LAST so they always sit on top of lines ──
  // Pre-collect all terminal tiles across all active flows so tap drawing can skip them.
  // A tile that is a terminal in one flow must not also get a tap arrow from another flow.
  const globalTerminals=new Set();
  (DATA.comm_paths||[]).forEach(p=>{
    const active=!dmHideAll&&(dmActiveNets.size===0||dmActiveNets.has(p.flow_index));
    if(!active) return;
    const edges=p.edges||[];
    const outCount={};
    edges.forEach(e=>{ const sk=e[0][0]+','+e[0][1]; outCount[sk]=(outCount[sk]||0)+1; });
    edges.forEach(e=>{ const [tc,tr]=e[1]; if(!outCount[tc+','+tr]&&tr>=0) globalTerminals.add(tc+','+tr); });
  });
  (DATA.comm_paths||[]).forEach(p=>{
    const fi=p.flow_index;
    const active=!dmHideAll&&(dmActiveNets.size===0||dmActiveNets.has(fi));
    if(!active) return;
    const color=dmColor(fi);
    const edges=p.edges||[];
    if(!edges.length) return;

    const dmaTileSet=new Set((p.dma_tiles||[]).map(t=>t[0]+','+t[1]));

    const outCount={}, inCount={};
    edges.forEach(e=>{
      const sk=e[0][0]+','+e[0][1], dk=e[1][0]+','+e[1][1];
      outCount[sk]=(outCount[sk]||0)+1;
      inCount[dk]=(inCount[dk]||0)+1;
    });

    // Terminal dots — hollow ring, opaque background: final consumer, no outgoing edges.
    const dstSeen=new Set();
    edges.forEach(e=>{
      const [sc2,sr2]=e[0], [tc,tr]=e[1], dk=tc+','+tr;
      if(!outCount[dk] && tr>=0 && !dstSeen.has(dk)){
        dstSeen.add(dk);
        const {ox,oy}=dotOffset(p,tc,tr);
        dotsG.appendChild(svgN('circle',{cx:fcx(tc)+ox,cy:fcy(tr)+oy,r:'1.8',
          fill:'#181818',stroke:color,'stroke-width':'1.2'}));
        // Arrow pointing into the dot (data flows in from the previous tile).
        const [dx,dy]=gridDir(sc2,sr2,tc,tr);
        svgArrow(fcx(tc)+ox,fcy(tr)+oy,dx,dy,color,false,4);
      }
    });

    // Tap dots — hollow ring, opaque background: DMA tile receives AND relays onward.
    // Slightly larger ring to distinguish from terminal.
    // Require BOTH inCount and outCount: a pull DMA source has outgoing edges but
    // no incoming edges and must not be drawn as a tap (it gets the solid source dot).
    const tapSeen=new Set();
    edges.forEach(e=>{
      const [sc,sr]=e[0], sk=sc+','+sr;
      if(sr>=0 && dmaTileSet.has(sk) && outCount[sk] && inCount[sk] && !tapSeen.has(sk) && !globalTerminals.has(sk)){
        tapSeen.add(sk);
        const {ox,oy}=dotOffset(p,sc,sr);
        dotsG.appendChild(svgN('circle',{cx:fcx(sc)+ox,cy:fcy(sr)+oy,r:'2',
          fill:'#181818',stroke:color,'stroke-width':'1.2'}));
        // Inward arrow only — tap is a consumer (output node), data flows in.
        const inEdge=edges.find(e2=>e2[1][0]===sc&&e2[1][1]===sr);
        if(inEdge){ const [dx,dy]=gridDir(inEdge[0][0],inEdge[0][1],sc,sr); svgArrow(fcx(sc)+ox,fcy(sr)+oy,dx,dy,color,false,4); }
      }
    });
  });

  svg.appendChild(dotsG);

  // Fit only on the first build or an explicit "Reset view". Rebuilds triggered
  // by the net chips, the tile right-click menu or a search must not throw away
  // the pan/zoom the user set up — the transform is independent of the geometry,
  // so reapplying it keeps the view anchored even when tile heights change.
  if(dmRefitNext||!dmBuilt){ dmRefitNext=false; dmFitView(); }
  else dmApply();
  // The SVG was recreated from scratch above, so any live status painted on the
  // previous DOM is gone — re-apply it from module state.
  dmPaintStatus();
  dmBuilt=true;
}

buildNetBar();


function _fmtDmaChanBadge(ch){
  const p=ch.direction==='S2MM'?'s2mm':'mm2s';
  return p+(ch.channel??0);
}
function _flowDmaLabel(fi, col, row){
  if(fi==null) return '';
  const tile=(DATA.tiles||[]).find(t=>t.loc[0]===col && t.loc[1]===row);
  if(!tile) return '';
  const chans=(tile.dma_channels||[]).filter(c=>c.flow_index===fi);
  if(chans.length===1) return _fmtDmaChanBadge(chans[0]);
  if(chans.length>1){
    return chans.map(_fmtDmaChanBadge).join(' ');
  }
  return '';
}
function _flowDmaBadgeStyle(lbl){
  if(!lbl){
    return {cls:'rt-route', bg:'#141820', fg:'#687080', text:'route',
      title:'stream-switch routing only — no local DMA for this flow on this tile'};
  }
  if(lbl.startsWith('mm2s')){
    return {cls:'rt-mm2s', bg:'#2a1028', fg:'#c050b0', text:lbl,
      title:'MM2S — tile memory to stream (send)'};
  }
  return {cls:'rt-s2mm', bg:'#0a2830', fg:'#30c0d0', text:lbl,
    title:'S2MM — stream to tile memory (receive)'};
}
function _flowDmaSpanHtml(fi, col, row){
  if(fi==null){
    const st=_flowDmaBadgeStyle('');
    return '<span class="'+st.cls+'" title="'+esc(st.title)+'">'+esc(st.text)+'</span>';
  }
  const tile=(DATA.tiles||[]).find(t=>t.loc[0]===col && t.loc[1]===row);
  const chans=tile?(tile.dma_channels||[]).filter(c=>c.flow_index===fi):[];
  if(!chans.length){
    const st=_flowDmaBadgeStyle('');
    return '<span class="'+st.cls+'" title="'+esc(st.title)+'">'+esc(st.text)+'</span>';
  }
  return chans.map(ch=>{
    const lbl=_fmtDmaChanBadge(ch);
    const st=_flowDmaBadgeStyle(lbl);
    return '<span class="'+st.cls+'" title="'+esc(st.title)+'">'+esc(st.text)+'</span>';
  }).join('');
}

function _flowTileSet(p){
  const s=new Set();
  (p.edges||[]).forEach(e=>{ s.add(e[0][0]+','+e[0][1]); s.add(e[1][0]+','+e[1][1]); });
  (p.tiles||[]).forEach(t=>s.add(t[0]+','+t[1]));
  (p.dma_tiles||[]).forEach(t=>s.add(t[0]+','+t[1]));
  return s;
}
function _isPktPort(p){ return p&&p.dir&&p.dir!=='NONE'; }
function _sharedPktForwardMaster(pktConns){
  for(const c of pktConns){
    const fm=c.forward_master||{};
    if(_isPktPort(fm)) return fm;
  }
  return null;
}
function _defaultPktMask(leg){ return leg==='dma'?0x1f:0; }
function _resolvePktMask(leg, port){
  if(port&&port.mask!=null) return port.mask;
  return _defaultPktMask(leg);
}
function _shortDir(d){
  return d==='NORTH'?'N':d==='SOUTH'?'S':d==='EAST'?'E':d==='WEST'?'W':d?d[0]:'?';
}
function _fmtPktMaskHex(mask, leg){
  const m=(mask!=null)?mask:_defaultPktMask(leg);
  return '0x'+Number(m).toString(16).toUpperCase();
}
function _fmtPktMaskHtml(mask, leg){
  return _fmtPktMaskBadge(mask, leg);
}
function _fmtPktMaskBadge(mask, leg){
  return '<span class="rt-pktmask" title="slave slot match mask (XAie_StrmPktSwSlaveSlotEnable)">'
    +_fmtPktMaskHex(mask, leg)+'</span>';
}
function _expandPktConnectRows(c, sharedFwd){
  const rs=c.recv_slave||{}, ld=c.local_dma||{}, fm=c.forward_master||{};
  const master=_isPktPort(fm)?fm:sharedFwd;
  const rows=[];
  if(_isPktPort(rs) && master)
    rows.push({leg:'recv', slave:{dir:rs.dir, idx:rs.idx},
               master:{dir:master.dir, idx:master.idx},
               pktid:rs.pktid, mask:_resolvePktMask('recv', rs)});
  if(_isPktPort(ld) && master)
    rows.push({leg:'dma', slave:{dir:ld.dir, idx:ld.idx},
               master:{dir:master.dir, idx:master.idx},
               pktid:ld.pktid, mask:_resolvePktMask('dma', ld)});
  // Forward-only entry: no recv/dma slave but a valid forward_master (e.g. the
  // terminal gather tile that hands off from the packet segment to circuit-switch).
  if(!rows.length && _isPktPort(fm))
    rows.push({leg:'fwd', slave:null, master:{dir:fm.dir, idx:fm.idx},
               pktid:null, mask:null});
  return rows;
}
function _pktRowKey(row){
  const sl=row.slave||{};
  return 'pkt|'+row.leg+'|'+(sl.dir??'none')+'|'+(sl.idx??'-')+'|'+(row.pktid??'?')
    +'|'+row.mask+'|'+row.master.dir+'|'+row.master.idx;
}
function _fmtPktLabel(row, compact){
  const dir=compact?_shortDir:(d=>d);
  const arr=' → ';
  const sl=row.slave||{};
  const slavePart=sl.dir?dir(sl.dir)+':'+sl.idx:'fwd';
  let s='PKT '+slavePart+arr+dir(row.master.dir)+':'+row.master.idx;
  if(!compact){
    if(row.pktid!=null) s+=' pkt'+row.pktid;
    if(row.mask!=null) s+=' '+_fmtPktMaskHex(row.mask, row.leg);
  }
  return s;
}
function _dmPktBadges(row){
  const badges=[];
  if(row.pktid!=null) badges.push({text:'pkt'+row.pktid, bg:'#2a0838', fg:'#c07fd4'});
  if(row.mask!=null) badges.push({text:_fmtPktMaskHex(row.mask, row.leg), bg:'#3a1010', fg:'#e87850'});
  return badges;
}
function _svgBadgeWidth(text){
  return Math.round(text.length*3.55+4);
}
function _fmtMselEnHex(v){
  return '0x'+Number(v!=null?v:1).toString(16).toUpperCase();
}
function _tilePktMasters(col, row, focusFlowIdx){
  const out=new Map();
  (DATA.comm_paths||[]).forEach(p=>{
    if(focusFlowIdx!=null && p.flow_index!==focusFlowIdx) return;
    const fts=_flowTileSet(p);
    if(fts.size>0 && !fts.has(col+','+row)) return;
    (p.routing_connections||[]).forEach(c=>{
      if(c.kind!=='packet_connect') return;
      const t=c.tile||{};
      if(t.col!==col || t.row!==row) return;
      const fm=c.forward_master||{};
      if(!_isPktPort(fm)) return;
      const key=fm.dir+'|'+fm.idx;
      if(!out.has(key)){
        out.set(key, {
          dir:fm.dir, idx:fm.idx,
          arbiter:(fm.arbiter!=null)?fm.arbiter:0,
          msel_en:(fm.msel_en!=null)?fm.msel_en:1,
          flow_indices:[],
        });
      }
      const e=out.get(key);
      if(p.flow_index!=null && !e.flow_indices.includes(p.flow_index))
        e.flow_indices.push(p.flow_index);
    });
  });
  return [...out.values()].sort((a,b)=>
    a.dir.localeCompare(b.dir)||a.idx-b.idx);
}
function _renderPktMasterBlock(col, row, focusFlowIdx, scan){
  const masters=_tilePktMasters(col, row, focusFlowIdx);
  const mstExtra=_swExtraMstHtml(scan, masters);
  if(!masters.length && !mstExtra) return '';
  const rows=masters.map(m=>{
    const flows=m.flow_indices.map(fi=>'fl'+fi).join(' ');
    const flow=flows?'<span class="rt-flow">'+esc(flows)+'</span>':'';
    const swSpan=_swMarkHtmlMst(scan, m.dir+':'+m.idx, m.arbiter, m.msel_en);
    return '<div class="rt-row mst">'
      +'<span class="rt-kind">MST</span>'
      +'<span class="rt-ports">'+esc(m.dir)+':'+m.idx+'</span>'
      +'<span class="rt-pktid" title="XAie_StrmPktSwMstrPortEnable Arbitor">arb:'
        +m.arbiter+'</span>'
      +'<span class="rt-pktid" title="XAie_StrmPktSwMstrPortEnable MSelEn (bitmask of slave MSel lines)">msel_en:'
        +_fmtMselEnHex(m.msel_en)+'</span>'
      +flow+swSpan+'</div>';
  }).join('');
  return '<div class="rt-mst-hdr">PKT master ports</div>'
    +rows+mstExtra;
}
function _tileRoutingConns(col, row, paths){
  paths = paths || DATA.comm_paths || [];
  const shimKinds=new Set(['shim_aie_to_ext','shim_ext_to_aie']);
  const keyIdx=new Map();
  const out=[];
  (paths||[]).forEach(p=>{
    const fts=_flowTileSet(p);
    if(fts.size>0 && !fts.has(col+','+row)) return;
    const pathPkt=(p.routing_connections||[]).filter(c=>{
      const t=c.tile||{};
      return t.col===col && t.row===row && c.kind==='packet_connect';
    });
    const sharedPktFwd=_sharedPktForwardMaster(pathPkt);
    pathPkt.forEach(c=>{
      _expandPktConnectRows(c,sharedPktFwd).forEach(row=>{
        const key=_pktRowKey(row);
        if(keyIdx.has(key)){
          const existing=out[keyIdx.get(key)];
          if(!(existing.flow_indices||[]).includes(p.flow_index)) existing.flow_indices.push(p.flow_index);
          return;
        }
        keyIdx.set(key, out.length);
        out.push({kind:'packet_hw', ...row, flow_index:p.flow_index, flow_indices:[p.flow_index]});
      });
    });
    (p.routing_connections||[]).forEach(c=>{
      const t=c.tile||{};
      if(t.col!==col || t.row!==row || shimKinds.has(c.kind)) return;
      if(c.kind==='packet_connect') return;
      const s=c.slave||{}, m=c.master||{};
      const key=c.kind+'|'+s.dir+'|'+s.idx+'|'+m.dir+'|'+m.idx;
      if(keyIdx.has(key)){
        const existing=out[keyIdx.get(key)];
        if(!(existing.flow_indices||[]).includes(p.flow_index)) existing.flow_indices.push(p.flow_index);
        return;
      }
      keyIdx.set(key, out.length);
      out.push({...c, flow_index:p.flow_index, flow_indices:[p.flow_index]});
    });
  });
  return out;
}

function _routingConnDiffRecord(c){
  if(c.kind==='packet_hw'){
    const sl=c.slave||{};
    return {kind:'PKT',
      slave: sl.dir ? sl.dir+':'+sl.idx : null,
      master: c.master.dir+':'+c.master.idx};
  }
  const s=c.slave||{}, m=c.master||{};
  return {kind:'CCT', slave:s.dir+':'+s.idx, master:m.dir+':'+m.idx};
}
function _routingConnDiffKey(c){
  const r=_routingConnDiffRecord(c);
  return _swKey(r.kind, r.slave, r.master);
}
function _filterRoutingConns(conns){
  return conns.filter(c=>{
    if(c.kind==='packet_hw') return true;
    const s=c.slave||{}, m=c.master||{};
    return s.dir!=null && s.idx!=null && m.dir!=null && m.idx!=null;
  });
}
function _routingDiffSides(){
  const base = ROUTING_SRC==='dynamic' ? 'dynamic' : 'static';
  return {base, other: base==='dynamic' ? 'static' : 'dynamic'};
}
function _routingSrcDiffTile(col, row){
  if(!(DYNAMIC && DYNAMIC.comm_paths)) return null;
  const d=_routingDiffSides();
  const basePaths=d.base==='dynamic' ? DYNAMIC.comm_paths : STATIC_PATHS;
  const otherPaths=d.base==='dynamic' ? STATIC_PATHS : DYNAMIC.comm_paths;
  const baseConns=_filterRoutingConns(_tileRoutingConns(col, row, basePaths));
  const otherConns=_filterRoutingConns(_tileRoutingConns(col, row, otherPaths));
  const bMap=new Map();
  baseConns.forEach(c=>{ const k=_routingConnDiffKey(c); if(!bMap.has(k)) bMap.set(k, _routingConnDiffRecord(c)); });
  const oMap=new Map();
  otherConns.forEach(c=>{ const k=_routingConnDiffKey(c); if(!oMap.has(k)) oMap.set(k, _routingConnDiffRecord(c)); });
  const missing=[], unexpected=[];
  bMap.forEach((rec, k)=>{ if(!oMap.has(k)) missing.push(rec); });
  oMap.forEach((rec, k)=>{ if(!bMap.has(k)) unexpected.push(rec); });
  let state='verified';
  if(!bMap.size && !oMap.size) state='idle';
  else if(missing.length || unexpected.length) state='mismatch';
  return {state, missing, unexpected};
}

// ── live switch scan overlay ────────────────────────────────────────────────
// A scanned tile carries the set of rows the hardware is NOT programmed with
// (missing) and the rows it has that no flow accounts for (unexpected).  Rows
// are keyed the same way both sides build them: kind + slave + master.
function _swKey(kind, slave, master){ return kind+'|'+(slave||'fwd')+'|'+master; }
function _swNormSlave(slave){
  return (!slave || slave==='fwd') ? 'fwd' : slave;
}
function _swRecordKey(r){
  if(!r) return '';
  if(r.kind==='PKT'){
    return 'PKT|'+_swNormSlave(r.slave)+'|'+r.master
      +'|'+(r.pktid!=null?r.pktid:'?')+'|'+(r.mask!=null?r.mask:'?');
  }
  if(r.kind==='MST'){
    return 'MST|'+r.master+'|'+(r.arbiter!=null?r.arbiter:0)
      +'|'+(r.msel_en!=null?r.msel_en:1);
  }
  return _swKey(r.kind, r.slave, r.master);
}
function _swCoarseKey(r){
  if(!r) return '';
  if(r.kind==='PKT'){
    const sl = (r.slave==='fwd'||!r.slave) ? null : r.slave;
    return _swKey('PKT', sl, r.master);
  }
  if(r.kind==='MST') return _swKey('MST', null, r.master);
  return _swKey(r.kind, r.slave, r.master);
}
function _swConnCoarseKey(c){
  if(c.kind==='packet_hw'){
    const sl=c.slave||{};
    return _swKey('PKT', sl.dir?sl.dir+':'+sl.idx:null,
                  c.master.dir+':'+c.master.idx);
  }
  const s=c.slave||{}, m=c.master||{};
  return _swKey('CCT', s.dir+':'+s.idx, m.dir+':'+m.idx);
}
function _swScanTile(col, row){
  return SWSCAN ? SWSCAN[col+','+row] : null;
}
function _swMissingRecordKeys(scan){
  const s=new Set();
  (scan&&scan.missing||[]).forEach(r=>s.add(_swRecordKey(r)));
  return s;
}
function _swMissingKeys(scan){
  const s=new Set();
  (scan&&scan.missing||[]).forEach(r=>s.add(_swCoarseKey(r)));
  return s;
}
function _swOnlyLabel(){
  if(!ROUTING_DIFF){
    return {text:'HW only',
            title:'programmed in hardware but no flow in the routing map accounts for it'};
  }
  const d=_routingDiffSides();
  return {text:d.other+' only',
          title:'in the '+d.other+' map but not in the '+d.base+' map'};
}
function _swDiffBadHtml(){
  const d=_routingDiffSides();
  return _swBadHtml('in the '+d.base+' map but not in the '+d.other+' map',
                    d.base+' only');
}
function _swBadHtml(title, text){
  return '<span class="sw-bad" title="'+esc(title)+'">'+text+'</span>';
}
function _swOkHtml(title, text){
  return '<span class="sw-ok" title="'+esc(title)+'">'+text+'</span>';
}
function _swMarkHtmlCct(scan, slave, master){
  if(!scan || scan.state==='unreachable') return '';
  const rec={kind:'CCT', slave:slave, master:master};
  if(_swMissingRecordKeys(scan).has(_swRecordKey(rec))){
    if(ROUTING_DIFF) return _swDiffBadHtml();
    return _swBadHtml('the routing map claims this connection '
      +'but the stream-switch registers are not programmed with it',
                      'not in HW');
  }
  if(ROUTING_DIFF){
    return _swOkHtml('present in both the static and dynamic maps', 'match');
  }
  return _swOkHtml('this connection is programmed in the stream-switch registers',
                   'in HW');
}
function _swMarkHtmlPkt(scan, slave, master, pktid, mask){
  if(!scan || scan.state==='unreachable') return '';
  const sl = slave || 'fwd';
  const rec={kind:'PKT', slave:sl, master:master, pktid:pktid, mask:mask};
  if(_swMissingRecordKeys(scan).has(_swRecordKey(rec))){
    if(ROUTING_DIFF) return _swDiffBadHtml();
    return _swBadHtml('the routing map claims this connection '
      +'but the stream-switch registers are not programmed with it',
                      'not in HW');
  }
  if(ROUTING_DIFF){
    return _swOkHtml('present in both the static and dynamic maps', 'match');
  }
  return _swOkHtml('this connection is programmed in the stream-switch registers',
                   'in HW');
}
function _swMarkHtmlMst(scan, master, arbiter, msel_en){
  if(!scan || scan.state==='unreachable') return '';
  const rec={kind:'MST', slave:null, master:master,
             arbiter:arbiter, msel_en:msel_en};
  if(_swMissingRecordKeys(scan).has(_swRecordKey(rec))){
    if(ROUTING_DIFF) return '';
    return _swBadHtml('the routing map claims this master port '
      +'but the stream-switch registers are not programmed with it',
                      'not in HW');
  }
  if(ROUTING_DIFF) return '';
  return _swOkHtml('this master port is programmed in the stream-switch registers',
                   'in HW');
}
function _swExtraRowHtml(r, only){
  const ports=(r.slave?esc(r.slave):'fwd')+'&nbsp;&rarr;&nbsp;'+esc(r.master);
  return '<div class="rt-row sw-extra">'
    +'<span class="rt-kind">'+esc(r.kind)+'</span>'
    +'<span class="rt-ports">'+ports+'</span>'
    +_swBadHtml(only.title, only.text)+'</div>';
}
function _swExtraRowsHtml(scan, mainConns){
  if(!scan) return '';
  const only=_swOnlyLabel();
  const mainCoarse=new Set((mainConns||[]).map(_swConnCoarseKey));
  const missingCoarse=_swMissingKeys(scan);
  const seen=new Set();
  return (scan.unexpected||[]).filter(r=>{
    if(!r || r.kind==='MST') return false;
    const ck=_swCoarseKey(r);
    if(mainCoarse.has(ck) && !missingCoarse.has(ck)) return false;
    const fk=_swRecordKey(r);
    if(seen.has(fk)) return false;
    seen.add(fk);
    return true;
  }).map(r=>_swExtraRowHtml(r, only)).join('');
}
function _swExtraMstHtml(scan, masters){
  if(!scan) return '';
  const only=_swOnlyLabel();
  const mainCoarse=new Set((masters||[]).map(m=>_swKey('MST', null, m.dir+':'+m.idx)));
  const missingCoarse=_swMissingKeys(scan);
  const seen=new Set();
  return (scan.unexpected||[]).filter(r=>{
    if(!r || r.kind!=='MST') return false;
    const ck=_swCoarseKey(r);
    if(mainCoarse.has(ck) && !missingCoarse.has(ck)) return false;
    const fk=_swRecordKey(r);
    if(seen.has(fk)) return false;
    seen.add(fk);
    return true;
  }).map(r=>{
    const ports=(r.slave?esc(r.slave):'fwd')+'&nbsp;&rarr;&nbsp;'+esc(r.master);
    let tail='';
    if(r.arbiter!=null){
      tail+=' <span class="rt-pktid" title="arbiter">arb:'+r.arbiter+'</span>';
    }
    if(r.msel_en!=null){
      tail+=' <span class="rt-pktid" title="msel_en">msel_en:'
        +_fmtMselEnHex(r.msel_en)+'</span>';
    }
    return '<div class="rt-row mst sw-extra">'
      +'<span class="rt-kind">MST</span>'
      +'<span class="rt-ports">'+ports+'</span>'
      +tail+_swBadHtml(only.title, only.text)+'</div>';
  }).join('');
}

function renderTileRoutingSection(col, row, focusFlowIdx){
  const scan = ROUTING_DIFF
    ? (_routingSrcDiffTile(col, row) || _swScanTile(col, row))
    : null;
  const allRaw=_tileRoutingConns(col, row);

  let conns=_filterRoutingConns(allRaw);
  if(focusFlowIdx!=null) conns=conns.filter(c=>(c.flow_indices||[c.flow_index]).includes(focusFlowIdx));
  conns=conns.slice().sort((a,b)=>{
    const rank=c=>(c.kind==='packet_hw'?1:0);
    return rank(a)-rank(b);
  });
  if(!conns.length && !scan) return '';
  if(!conns.length && scan && !(scan.unexpected||[]).length) return '';

  const rows=conns.map(c=>{
    const tile=(DATA.tiles||[]).find(t=>t.loc[0]===col&&t.loc[1]===row);
    const localFis=new Set((tile&&tile.dma_channels||[]).map(ch=>ch.flow_index));
    // Under focus the row is claimed by one flow only; badging every flow that
    // shares the row would put another flow's channel on it.
    const candidates=(focusFlowIdx!=null)
      ?[focusFlowIdx]
      :(c.flow_indices||[c.flow_index]);
    const withDma=candidates.filter(fi=>localFis.has(fi));
    // Show one badge per flow that has a local DMA channel; fall back to route badge.
    const dmaSpan=withDma.length
      ?withDma.map(fi=>_flowDmaSpanHtml(fi,col,row)).join('')
      :_flowDmaSpanHtml(null,col,row);
    if(c.kind==='packet_hw'){
      const sl=c.slave||{};
      const ports=sl.dir
        ?esc(sl.dir)+':'+sl.idx+'&nbsp;&rarr;&nbsp;'+esc(c.master.dir)+':'+c.master.idx
        :'fwd&nbsp;&rarr;&nbsp;'+esc(c.master.dir)+':'+c.master.idx;
      const pktSpan=(c.pktid!=null)
        ?'<span class="rt-pktid" title="packet match id">pkt'+c.pktid+'</span>':'';
      const maskSpan=(c.mask!=null)?_fmtPktMaskBadge(c.mask,c.leg):'';
      const swSpan=_swMarkHtmlPkt(scan,
        sl.dir?sl.dir+':'+sl.idx:null, c.master.dir+':'+c.master.idx,
        c.pktid, c.mask);
      return '<div class="rt-row pkt">'
        +'<span class="rt-kind">PKT</span>'
        +'<span class="rt-ports">'+ports+'</span>'
        +pktSpan+maskSpan+dmaSpan+swSpan+'</div>';
    }
    const s=c.slave||{}, m=c.master||{};
    const ports=esc(s.dir)+':'+s.idx+'&nbsp;&rarr;&nbsp;'+esc(m.dir)+':'+m.idx;
    const swSpan=_swMarkHtmlCct(scan, s.dir+':'+s.idx, m.dir+':'+m.idx);
    return '<div class="rt-row cct">'
      +'<span class="rt-kind">CCT</span>'
      +'<span class="rt-ports">'+ports+'</span>'
      +dmaSpan+swSpan+'</div>';
  });
  const mstBlock=_renderPktMasterBlock(col, row, focusFlowIdx, scan);
  const swHdr=scan
    ?' <span class="sw-state '+esc(scan.state)+'">'+esc(scan.state)+'</span>'
    :'';
  const extra=_swExtraRowsHtml(scan, conns);
  if(!rows.filter(Boolean).join('') && !mstBlock && !extra) return '';
  return '<div class="sec"><div class="sec-hdr">Stream switch'+swHdr+'</div>'
    +rows.filter(Boolean).join('')+mstBlock+extra+'</div>';
}

function _routingTileType(row){
  const deviceCoreMinRow=DATA.grid.device_core_min_row||3;
  if(row===0) return 'shim';
  if(row>=deviceCoreMinRow) return 'core';
  return 'mem';
}
function _tileCommPaths(col, row){
  // Stream edges plus shared-memory hops: a tile reached only through shared
  // memory still carries the flow, and printing "(no comm paths through this
  // tile)" above its own Shared memory table contradicted itself.
  const onEdge=p=>(p.edges||[]).some(e=>
    (e[0][0]===col&&e[0][1]===row)||(e[1][0]===col&&e[1][1]===row));
  const onShmem=p=>(p.hops||[]).some(h=>h.type==='shmem'&&
    ((h.from_col===col&&h.from_row===row)||(h.to_col===col&&h.to_row===row)));
  return (DATA.comm_paths||[]).filter(p=>onEdge(p)||onShmem(p));
}
function _tileShmemHops(col, row){
  const out=[];
  (DATA.comm_paths||[]).forEach(p=>{
    (p.hops||[]).filter(h=>h.type==='shmem').forEach(h=>{
      if((h.from_col===col&&h.from_row===row)||(h.to_col===col&&h.to_row===row))
        out.push({flow_index:p.flow_index, ...h});
    });
  });
  return out;
}
function buildRoutingTileHtml(col, row){
  const ttype=_routingTileType(row);
  const flows=_tileCommPaths(col, row);
  const flowRows=flows.map(p=>'<tr><td>f'+p.flow_index+'</td><td>'
    +esc(p.direction||'')+'</td><td>'+esc(p.id||'')+'</td></tr>').join('');
  const flowTable=flowRows
    ?'<table class="rctbl"><thead><tr><th>flow</th><th>dir</th><th>net</th></tr></thead>'
      +'<tbody>'+flowRows+'</tbody></table>'
    :'<div class="placeholder">(no comm paths through this tile)</div>';
  const shmem=_tileShmemHops(col, row);
  const shmemRows=shmem.map(h=>'<tr><td>f'+h.flow_index+'</td><td>('+h.from_col+','+h.from_row
    +')</td><td>→</td><td>('+h.to_col+','+h.to_row+')</td><td>'
    +esc(h.kind||'shmem')+'</td></tr>').join('');
  const shmemSec=shmemRows
    ?'<div class="sec"><div class="sec-hdr">Shared memory</div>'
      +'<table class="rctbl"><thead><tr><th>flow</th><th>from</th><th></th><th>to</th>'
      +'<th>kind</th></tr></thead><tbody>'+shmemRows+'</tbody></table></div>'
    :'';
  const swSec=renderTileRoutingSection(col, row);
  return '<div class="placeholder dimtxt">No kernel or DMA schedule for this tile — routing path only.</div>'
    +'<div class="sec"><div class="sec-hdr">Tile</div>'
    +'<div class="kv"><b>location:</b> ('+col+','+row+')</div>'
    +'<div class="kv"><b>geometry:</b> '+ttype+'</div></div>'
    +'<div class="sec"><div class="sec-hdr">Flows</div>'+flowTable+'</div>'
    +shmemSec+(swSec||'');
}
function selectRoutingTile(col, row, ctrl){
  reportUIState({selected_tile:[col,row], channel:null, flow:null});
  const tileKey=panelKey('tile', col+','+row);
  const label='('+col+','+row+') routing';
  const llmCtx='tile ('+col+','+row+') routing-only (no schedule bundle)';
  const buildBody=()=>buildRoutingTileHtml(col, row);
  if(ctrl){
    if(panelItems.has(tileKey)){ panelRemove(tileKey); return; }
    panelItems.set(tileKey,{kind:'tile',label,color:null,buildBody,wireBody:()=>{},llmCtx});
  } else {
    panelItems.forEach((_,k)=>{ if(k.startsWith('tile:')) panelItems.delete(k); });
    panelItems.set(tileKey,{kind:'tile',label,color:null,buildBody,wireBody:()=>{},llmCtx});
  }
  panelActiveKey=tileKey;
  panelSync();
}

function renderTileDmaBdSection(t, ch, focused){
  const chans=focused?[ch]:(t.dma_channels||[]);
  if(!chans.length) return '';
  const parts=chans.map(c=>{
    const bds=c.bd_chain||[];
    if(!bds.length) return '';
    const bdRows=bds.map(bd=>{
      const nxt=(bd.next_bd!=null && bd.next_bd>=0)?' &rarr;BD'+bd.next_bd:'';
      const acq=(bd.acquire_lock||[])[0];
      const rel=(bd.release_lock||[])[0];
      const locks=(acq||rel)
        ?'<span class="bd-lock">'
          +(acq?'acq=L'+acq.id+'('+acq.val+')':'')
          +(acq&&rel?' ':'')
          +(rel?'rel=L'+rel.id+'('+rel.val+')':'')
          +'</span>':'';
      return '<div class="bd-mini">'
        +'<span class="bd-id">BD'+bd.bd_id+'</span>'
        +'<span class="bd-len">len='+bd.len+'</span>'
        +(nxt?'<span class="bd-next">'+nxt+'</span>':'')
        +locks+'</div>';
    });
    const hdr=focused?''
      :'<div class="bd-mini-ch">'+esc(c.direction)+' ch'+c.channel
        +(c.flow_index!=null?' fl'+c.flow_index:'')+'</div>';
    return hdr+bdRows.join('');
  }).filter(Boolean);
  if(!parts.length) return '';
  return '<div class="sec"><div class="sec-hdr">DMA BDs</div>'+parts.join('')+'</div>';
}

// One-line summary for a single channel (mirrors build_summary in schedule_view.py).
function chanSummary(ch){
  const bd = (ch.bd_chain||[])[0] || {};
  const len = (bd.len!=null) ? bd.len : '?';
  const acq = (bd.acquire_lock||[])[0] || {};
  const rel = (bd.release_lock||[])[0] || {};
  const verb = ch.direction==='S2MM' ? 'recv' : 'send';
  let reps = '';
  const si = (ch.start_io||[])[0];
  if (si && si.repeat_count && si.repeat_count!==1) reps = ' x'+si.repeat_count;
  let lock = '';
  if (acq.id!=null || rel.id!=null) lock = ' lock'+acq.id+'/'+rel.id;
  return verb+' '+len+'B ch'+ch.channel+' '+ch.direction+reps+lock;
}
// Supply/demand verdict for one flow (from DATA.supply_demand). Renders a
// balanced/mismatch badge with the per-round byte figures so the user can see
// exactly where a flow supplies more (or less) than it consumes.
function renderFlowBalance(b){
  if(!b) return '';
  const s = b.supply_per_round, d = b.demand_per_round;
  let badge, cls;
  if (b.balanced === true){ badge='balanced'; cls='sdok'; }
  else if (b.balanced === false){
    badge = (s>d ? 'OVER-SUPPLY' : 'UNDER-SUPPLY'); cls='sdbad';
  } else { badge='unchecked'; cls='sdna'; }
  let fig = '';
  if (s!=null && d!=null){
    fig = '<div class="sdfig">supply '+s+'B/round vs demand '+d+'B/round'+
          (b.balanced===false ? '  (\u0394 '+(s-d)+'B)' : '')+'</div>';
  }
  return '<div class="sdrow">'+
    '<span class="sdbadge '+cls+'">'+badge+'</span>'+
    '<span class="sdmeta">flow '+b.flow_index+' &middot; '+esc(b.pattern)+'</span>'+
    fig+
    (b.note?'<div class="sdnote">'+esc(b.note)+'</div>':'')+
  '</div>';
}
// All distinct flow balances a tile participates in (dedup by flow_index).
function renderTileBalances(t){
  const seen = {}, rows = [];
  (t.dma_channels||[]).forEach(c => {
    const b = c.flow_balance;
    if (b && !(b.flow_index in seen)){ seen[b.flow_index]=1; rows.push(b); }
  });
  if (!rows.length) return '';
  rows.sort((a,b)=>a.flow_index-b.flow_index);
  const bad = rows.some(b=>b.balanced===false);
  const hdr = '<div class="kv"><b>supply / demand'+
    (bad?' <span class="sdbadge sdbad">MISMATCH</span>':'')+':</b></div>';
  return hdr + rows.map(renderFlowBalance).join('');
}
// Relevant host.cc lines for a single channel (from its host_lines entries).
function renderChannelLines(ch){
  const es = (ch.host_lines||[]).slice().sort((a,b)=>a.line-b.line);
  if (!es.length) return '<div class="placeholder">(no relevant lines)</div>';
  return es.map(e => {
    const badge = '<span class="kb '+e.kind+'">'+(KIND_LABEL[e.kind]||e.kind)+'</span>';
    const note = e.note ? '<span class="rnote">// '+esc(e.note)+'</span>' : '';
    // Comment-first layout: parameter explanation, then the code line.
    const pp = paramPretty(e.code, e.bd_comment);
    let html = pp ? '<pre class="bdpretty">'+esc(pp)+'</pre>' : '';
    html += '<div class="rline"><span class="lno">L'+e.line+'</span>'+badge+hl(e.code)+note+'</div>';
    return html;
  }).join('');
}

let selEl = null;
let selBadge = null;
function select(t, el, ch, badgeEl, ctrl){
  if (!ctrl){
    // plain click: deselect old tile highlight
    if (selEl) selEl.classList.remove('sel');
  }
  selEl = el; el.classList.add('sel');
  reportUIState({selected_tile: t.loc,
                 channel: ch ? (ch.direction + ch.channel) : null,
                 flow: ch ? ch.flow_index : null});
  // Persistent channel selection: keep the clicked badge highlighted after the
  // cursor moves out, and clear it only when another item (tile or channel) is
  // clicked. A tile-only click (no badgeEl) clears any previous badge highlight.
  if (selBadge) selBadge.classList.remove('selbadge');
  selBadge = badgeEl || null;
  if (selBadge) selBadge.classList.add('selbadge');
  const hlv = t.high_level, lo = t.low_level;
  const focused = !!ch;
  // Full-block source: channel-scoped when a channel is focused, else the tile.
  const flo = (focused && ch.low_level) ? ch.low_level : lo;

  // --- title ---
  const title = focused
    ? 'Tile ('+t.loc[0]+','+t.loc[1]+') &mdash; channel '+ch.direction+ch.channel+
      ' <span class="lref">(flow '+ch.flow_index+')</span>'
    : 'Tile ('+t.loc[0]+','+t.loc[1]+') &mdash; '+t.type;

  // --- high-level body ---
  let hiBody;
  if (focused) {
    const con = ch.contract ? '<div class="contract">'+esc(ch.contract)+'</div>' : '';
    const fb = ch.flow_balance;
    const bad = fb && fb.balanced === false;
    const ok  = fb && fb.balanced === true;
    const statusBar = fb
      ? '<div class="statusbar '+(bad?'bad':'ok')+'">'+
          '<span class="sb-icon">'+(bad?'&#9888;':'&#10003;')+'</span>'+
          '<span>'+(bad?(fb.supply_per_round>fb.demand_per_round?'Over-supply':'Under-supply')+' on flow '+fb.flow_index:'Flow '+fb.flow_index+' balanced')+'</span>'+
          (fb.supply_per_round!=null?'<span class="sb-detail">'+fb.supply_per_round+'B/round supplied &rarr; '+fb.demand_per_round+'B/round demanded'+(bad?' (Δ '+(fb.supply_per_round-fb.demand_per_round)+'B)':'')+'</span>':'')+
        '</div>'
      : '';
    hiBody =
      (bad ? statusBar : '') +
      '<div class="sec"><div class="sec-hdr">Channel</div>' +
        '<div class="kv"><b>role:</b> '+esc(hlv.role)+'</div>' +
        (hlv.kernel?'<div class="kv"><b>kernel:</b> '+esc(hlv.kernel)+'</div>':'') +
        '<div class="kv"><b>channel:</b> '+ch.direction+ch.channel+' &mdash; flow '+ch.flow_index+'</div>' +
        '<div class="kv"><b>transfer:</b> '+esc(chanSummary(ch))+'</div>' +
      '</div>' +
      renderTileRoutingSection(t.loc[0], t.loc[1], ch.flow_index) +
      renderTileDmaBdSection(t, ch, true) +
      (con?'<div class="sec"><div class="sec-hdr">Contract</div>'+con+'</div>':'');
  } else {
    const sum = (hlv.summary||[]).map(s=>'<li>'+esc(s)+'</li>').join('');
    const con = (hlv.contracts||[]).map(s=>{
      return '<div class="contract">'+esc(s)+'</div>';
    }).join('');
    // Build status bar from tile-level balances
    const balRows = [];
    const bseen = {};
    (t.dma_channels||[]).forEach(c => {
      const b = c.flow_balance;
      if (b && !(b.flow_index in bseen)){ bseen[b.flow_index]=1; balRows.push(b); }
    });
    balRows.sort((a,b)=>a.flow_index-b.flow_index);
    const anyBad = balRows.some(b=>b.balanced===false);
    const allOk  = balRows.length && balRows.every(b=>b.balanced===true);
    const statusBar = balRows.length
      ? '<div class="statusbar '+(anyBad?'bad':'ok')+'">'+
          '<span class="sb-icon">'+(anyBad?'&#9888;':'&#10003;')+'</span>'+
          '<span>'+(anyBad?'Supply/demand mismatch on '+balRows.filter(b=>b.balanced===false).length+' flow(s)':'All '+balRows.length+' flow(s) balanced')+'</span>'+
        '</div>'
      : '';
    const kmatch = renderKernelMatch(t);
    hiBody =
      (anyBad ? statusBar : '') +
      '<div class="sec"><div class="sec-hdr">Tile</div>' +
        '<div class="kv"><b>role:</b> '+esc(hlv.role)+'</div>' +
        (hlv.kernel?'<div class="kv"><b>kernel:</b> '+esc(hlv.kernel)+'</div>':'') +
        '<div class="kv"><b>transfers:</b></div><ul class="sum">'+sum+'</ul>' +
      '</div>' +
      renderTileRoutingSection(t.loc[0], t.loc[1]) +
      renderTileDmaBdSection(t, null, false) +
      (kmatch?'<div class="sec"><div class="sec-hdr">Kernel &harr; Channel Arguments</div>'+kmatch+'</div>':'') +
      (balRows.length?'<div class="sec"><div class="sec-hdr">Supply / Demand</div>'+balRows.map(renderFlowBalance).join('')+'</div>':'') +
      (con?'<div class="sec"><div class="sec-hdr">Contracts</div>'+con+'</div>':'');
  }

  // --- relevant lines body ---
  const relCount = focused ? (ch.host_lines||[]).length : (t.relevant_lines||[]).length;
  const relBody = focused ? renderChannelLines(ch) : renderRelevant(t);
  const relLabel = focused
    ? relCount+' relevant line(s) for channel '+ch.direction+ch.channel
    : relCount+' relevant line(s)';

  // --- middle (dfschedule IR) body ---
  const midIR = focused ? (ch.middle_ir||'') : (t.middle_ir||'');

  // --- code-piece file (debugcache/code) for the file-frame header + LLM ---
  const codeFile = focused ? (ch.code_file||'') : (t.code_file||'');
  // Core tiles carry a parsed kernel function body; surface its file name.
  const hasKernel = tileHasKernel(t);
  const tkv = hasKernel ? tileKernel(t) : null;
  const tbf = hasKernel ? tileBcf(t) : null;
  const ksrc = (tkv && tkv.source) ? tkv.source : null;
  // Single "kernel code" sub-tab (core tiles): merges the kernel source
  // (conv2d_spatial.cc), the generated wrapper (kernel.cc) and the buffer
  // address map (.bcf) into one stacked view, each section headed by its file.
  const kcodeOn = hasKernel &&
    ((tkv && (tkv.source || tkv.kernel_lines)) ||
     (tbf && tbf.lines));
  const kfile = ksrc ? ksrc.file : '';
  const kfileTag = ksrc
    ? ' <span class="kfileref srcref" data-p="'+esc(ksrc.path||'')+'"'
      +' data-l="'+(ksrc.start_line||'')+'">+ kernel '+esc(kfile)+'</span>' : '';
  const codePathBanner = codeFile
    ? '<div class="codepath"><b>code piece:</b> <span class="cpath srcref" data-p="'+esc(codeFile)+'"'
      +' title="open in the source viewer">'+esc(codeFile)+'</span>'+kfileTag+'</div>'
    : (ksrc
       ? '<div class="codepath"><b>kernel:</b> <span class="cpath srcref" data-p="'
         +esc((ksrc&&ksrc.path)||'')+'" data-l="'+((ksrc&&ksrc.start_line)||'')+'">'
         +esc(kfile)+'</span></div>'
       : '');

  const defaultCodeBody = CAPS.host_lines
    ? codePathBanner +
      renderCodeSection('Relevant lines &mdash; '+relLabel, relBody, true) +
      renderCodeSection('Full block &mdash; host.cc '+
        (flo.line_start||'?')+'–'+(flo.line_end||'?')+
        ' ('+(flo.ranges||[]).length+' range(s))'+
        (focused?' &mdash; '+ch.direction+ch.channel+' scope':''),
        renderFullBlock(flo.code_lines,
          focused ? ((ch.low_level||{}).params||null) : null), true) +
      (kcodeOn ? renderKernelCode(t, ch, focused, true) : '')
    : renderTileCodeKernelFirst(t, ch, focused, codePathBanner);
  const hostFileBody = CAPS.host_lines
    ? codePathBanner +
      renderCodeSection('host.cc &mdash; '+(focused
          ? 'channel '+ch.direction+ch.channel : 'tile')+' scope',
        renderFullBlock(flo.code_lines,
          focused ? ((ch.low_level||{}).params||null) : null), true)
    : '';
  const loBody = renderCodeFileTabs(
    t, ch, focused, defaultCodeBody, hostFileBody);

  const buildTileHtml = () =>
    '<div class="tabs">' +
      '<span class="tab act" data-t="hi">Schedule</span>' +
      (CAPS.ir ? '<span class="tab" data-t="mid">IR</span>' : '') +
      '<span class="tab" data-t="lo">Code</span>' +
    '</div>' +
    '<div class="tabbody">' +
      '<div id="tab-hi">' + hiBody + '</div>' +
      (CAPS.ir ?
      '<div id="tab-mid" class="hide">' +
        '<div class="lref">dfschedule IR (6_BlueprintToSchedule) &mdash; ' +
          (focused ? 'channel '+ch.direction+ch.channel : 'tile') + ' scope</div>' +
        '<div class="midctrls">' +
          '<button id="loadFullIr">Load full dfschedule IR</button>' +
          '<button id="foldAll" class="hide">Expand all</button>' +
        '</div>' +
        '<div id="midContent">' + renderMiddleIR(midIR) + '</div>' +
      '</div>' : '') +
      '<div id="tab-lo" class="hide">' + loBody + '</div>' +
    '</div>';

  const tileKey = panelKey('tile', t.loc[0]+','+t.loc[1]+(ch?'/'+ch.direction+ch.channel:''));
  const tileLabel = focused
    ? '('+t.loc[0]+','+t.loc[1]+') '+ch.direction+ch.channel
    : '('+t.loc[0]+','+t.loc[1]+') '+t.type;
  const tileLlmCtx = '[context] tile ('+t.loc[0]+','+t.loc[1]+') type:'+t.type
    +(ch?' ch:'+ch.direction+ch.channel+' flow:'+ch.flow_index:'')
    +' role:'+(hlv.role||'?')+(hlv.kernel?' kernel:'+hlv.kernel:'');

  if(ctrl){
    // Ctrl+click: toggle this tile in/out of panelItems
    if(panelItems.has(tileKey)){ panelRemove(tileKey); return; }
    panelItems.set(tileKey,{kind:'tile',label:tileLabel,color:null,
      buildBody:buildTileHtml, wireBody:wireTileExtra.bind(null,t,ch,focused,flo,midIR,kcodeOn),
      llmCtx:tileLlmCtx});
    panelActiveKey = tileKey;
  } else {
    // Plain click: clear all tile entries, keep nets, add this tile
    panelItems.forEach((_,k)=>{ if(k.startsWith('tile:')) panelItems.delete(k); });
    panelItems.set(tileKey,{kind:'tile',label:tileLabel,color:null,
      buildBody:buildTileHtml, wireBody:wireTileExtra.bind(null,t,ch,focused,flo,midIR,kcodeOn),
      llmCtx:tileLlmCtx});
    panelActiveKey = tileKey;
  }
  panelSync();
  setConTarget(t, ch);
}

// Extra DOM wiring for tile body — called by panelRenderBody via item.wireBody(body).
function wireTileExtra(t, ch, focused, flo, midIR, kcodeOn, body){
  body.querySelectorAll('.codefile-tab').forEach(tab => {
    tab.onclick = () => {
      const key = tab.dataset.codefile;
      body.querySelectorAll('.codefile-tab').forEach(
        other => other.classList.toggle('act', other === tab));
      body.querySelectorAll('.codefile-view').forEach(
        view => view.classList.toggle('hide', view.dataset.codefile !== key));
      panelBuildToc();
    };
  });
  body.querySelectorAll('.kshowall').forEach(kbtn => {
    const section = kbtn.closest('.codesec-body') || kbtn.parentElement;
    const kpiece = section.querySelector('.kern-piece');
    const kfull  = section.querySelector('.kern-full');
    if (!kfull) return;
    kbtn.onclick = () => {
      const showAll = kfull.classList.contains('hide');
      kfull.classList.toggle('hide', !showAll);
      if (kpiece) kpiece.classList.toggle('hide', showAll);
      kbtn.textContent = showAll ? 'Show piece only' : 'Show all';
    };
  });
  const klo = body.querySelector('#tab-lo');
  if (klo) wireFolds(klo);

  const midLines = Array.isArray(midIR)
    ? midIR.filter(r => r.line != null).map(r => r.line) : [];
  const loadBtn    = body.querySelector('#loadFullIr');
  const foldAllBtn = body.querySelector('#foldAll');
  const midContent = body.querySelector('#midContent');
  if (loadBtn && midContent) {
    const sliceHtml = midContent.innerHTML;
    let showingFull = false;
    const refreshFoldAll = () => {
      if (!foldAllBtn) return;
      foldAllBtn.textContent =
        midContent.querySelector('.irhidden.hide') ? 'Expand all' : 'Collapse all';
    };
    loadBtn.onclick = () => {
      showingFull = !showingFull;
      if (showingFull) {
        midContent.innerHTML = renderFullIr(midLines);
        wireFolds(midContent, refreshFoldAll);
        loadBtn.textContent = 'Show slice only';
        if (foldAllBtn) foldAllBtn.classList.remove('hide');
        refreshFoldAll();
        const first = midContent.querySelector('.irhi');
        if (first) first.scrollIntoView({block: 'center'});
      } else {
        midContent.innerHTML = sliceHtml;
        loadBtn.textContent = 'Load full dfschedule IR';
        if (foldAllBtn) foldAllBtn.classList.add('hide');
      }
    };
    if (foldAllBtn) {
      foldAllBtn.onclick = () => {
        const collapse = foldAllBtn.textContent.indexOf('Collapse') === 0;
        midContent.querySelectorAll('.irfold').forEach(f => {
          const h = midContent.querySelector('.irhidden[data-fold="'+f.dataset.fold+'"]');
          if (h) setFold(f, h, collapse);
        });
        refreshFoldAll();
      };
    }
  }
}

function llmPushCtx(text, chan){
  const k = chan || 'session';
  if (text) LLM.ctx.set(k, text); else LLM.ctx.delete(k);
}

// Richer tile/channel context: include role, kernel, contract, port.
function setLLMContext(t, ch, codeFile){
  const hl = (t && t.high_level) || {};
  const loc = t ? 'tile ('+t.loc[0]+','+t.loc[1]+')' : '';
  const parts = ['[context] Selected: '+loc
    + (ch ? ' channel '+ch.direction+ch.channel : '')
    + ' type: '+(t&&t.type||'?')+', role: '+(hl.role||'?')];
  if (hl.kernel) parts.push('kernel: '+hl.kernel);
  if (ch && ch.contract) parts.push('contract: '+ch.contract);
  if (ch && ch.kernel_port) parts.push('port: '+ch.kernel_port);
  if (codeFile) parts.push('code: '+codeFile);
  llmPushCtx(parts.join('; '), 'selection');
}

// ─── multi-item panel (net + tile tabs) ───────────────────────────────────────
// panelItems: ordered Map of key → {kind:'net'|'tile', label, color, buildBody}
// panelActiveKey: which item is shown in the body below the tab strip
const panelItems = new Map();
let panelActiveKey = null;

function panelKey(kind, id){ return kind+':'+id; }

function panelBuildToc(){
  const toc = document.getElementById('panel-toc');
  const pb  = document.getElementById('panel-body');
  if(!toc || !pb){ return; }
  function notHidden(el){
    let p = el;
    while(p && p !== pb){ if(p.classList.contains('hide')) return false; p = p.parentElement; }
    return true;
  }
  const entries = [];
  pb.querySelectorAll('.sec-hdr, details.codesec > summary').forEach(el => {
    if(notHidden(el)) entries.push({ label: el.textContent.trim(), el });
  });
  if(!entries.length){ toc.innerHTML = ''; toc.classList.add('no-items'); return; }
  toc.classList.remove('no-items');
  const collapsed = toc.classList.contains('collapsed');
  toc.innerHTML =
    '<button class="ptoc-toggle" title="'+(collapsed?'Expand':'Collapse')+' table of contents">'+(collapsed?'◀':'▶')+'</button>' +
    '<button class="ptoc-top" title="Scroll to top">▲</button>' +
    entries.map((e, i) => '<span class="ptoc-item" data-i="'+i+'">'+esc(e.label)+'</span>').join('');
  toc.querySelector('.ptoc-toggle').onclick = () => {
    toc.classList.toggle('collapsed');
    const btn = toc.querySelector('.ptoc-toggle');
    const now = toc.classList.contains('collapsed');
    btn.textContent = now ? '◀' : '▶';
    btn.title = now ? 'Expand table of contents' : 'Collapse table of contents';
  };
  const topBtn = toc.querySelector('.ptoc-top');
  topBtn.onclick = () => pb.scrollTo({ top: 0, behavior: 'smooth' });
  const syncTopBtn = () => { topBtn.style.display = pb.scrollTop > 40 ? '' : 'none'; };
  if(pb._ptocScroll) pb.removeEventListener('scroll', pb._ptocScroll);
  pb._ptocScroll = syncTopBtn;
  pb.addEventListener('scroll', syncTopBtn, { passive: true });
  syncTopBtn();
  toc.querySelectorAll('.ptoc-item').forEach(span => {
    const target = entries[+span.dataset.i].el;
    span.onclick = () => {
      const off = target.getBoundingClientRect().top - pb.getBoundingClientRect().top + pb.scrollTop - 6;
      pb.scrollTo({ top: Math.max(0, off), behavior: 'smooth' });
    };
  });
}

function panelRenderTabs(){
  const strip = document.getElementById('panel-itemtabs');
  strip.innerHTML = '';
  panelItems.forEach((item, key) => {
    const tab = document.createElement('span');
    tab.className = 'pitab' + (key===panelActiveKey?' act':'');
    if(item.color){
      const dot = document.createElement('span');
      dot.className = 'pit-dot';
      dot.style.background = item.color;
      tab.appendChild(dot);
    }
    tab.appendChild(document.createTextNode(item.label));
    const x = document.createElement('span');
    x.className = 'pit-x'; x.textContent = '×';
    x.onclick = ev => { ev.stopPropagation(); panelRemove(key); };
    tab.appendChild(x);
    tab.onclick = () => panelShow(key);
    strip.appendChild(tab);
  });
}

function panelRenderBody(key){
  const body = document.getElementById('panel-body');
  const item = panelItems.get(key);
  if(!item){ body.innerHTML='<div class="placeholder">Select a tile or net for details.</div>'; panelBuildToc(); return; }
  body.innerHTML = item.buildBody();
  // wire folder tabs inside body
  body.querySelectorAll('.tab').forEach(tab=>{
    tab.onclick=()=>{
      body.querySelectorAll('.tab').forEach(t=>t.classList.remove('act'));
      tab.classList.add('act');
      const tabField = {net:'net_tab', tile:'tile_tab', src:'src_tab'}[item.kind];
      if (tabField) reportUIState({[tabField]: tab.dataset.t});
      const id='tab-'+tab.dataset.t;
      body.querySelectorAll('.tabbody>div').forEach(d=>d.classList.toggle('hide',d.id!==id));
      panelBuildToc();
    };
  });
  // wire any extra handlers the item needs
  if(item.wireBody) item.wireBody(body);
  panelBuildToc();
}

function panelShow(key){
  panelActiveKey = key;
  panelRenderTabs();
  panelRenderBody(key);
  panelUpdateLLM();
}

function panelRemove(key){
  panelItems.delete(key);
  if(panelActiveKey===key){
    // activate the last remaining item, or nothing
    panelActiveKey = panelItems.size ? [...panelItems.keys()].at(-1) : null;
  }
  panelRenderTabs();
  if(panelActiveKey) panelRenderBody(panelActiveKey);
  else { document.getElementById('panel-body').innerHTML='<div class="placeholder">Select a tile or net for details.</div>'; panelBuildToc(); }
  panelUpdateLLM();
}

function panelSync(){
  // After panelItems is mutated, refresh tabs and body (keeping active key if still valid).
  if(!panelItems.has(panelActiveKey)) panelActiveKey = panelItems.size ? [...panelItems.keys()][0] : null;
  panelRenderTabs();
  if(panelActiveKey) panelRenderBody(panelActiveKey);
  else { document.getElementById('panel-body').innerHTML='<div class="placeholder">Select a tile or net for details.</div>'; panelBuildToc(); }
  panelUpdateLLM();
}

function panelUpdateLLM(){
  if(!panelItems.size){ llmPushCtx(null, 'selection'); return; }
  const parts=[];
  panelItems.forEach((item,key)=>{
    const active = key===panelActiveKey;
    parts.push((active?'[viewing] ':'[also open] ')+item.llmCtx);
  });
  llmPushCtx(parts.join(' | '), 'selection');
}

// ─── source viewer ────────────────────────────────────────────────────────────
// Click a source reference anywhere in the UI and the file opens here, in the
// Info pane, highlighted at that line. Needs the daemon: the page has no way to
// read a file at file://, so SRC.on gates every entry point.
const SRC = { on:(location.protocol === 'http:' || location.protocol === 'https:'),
              idx:null, note:null,
              // raw citation -> the absolute path the daemon resolved it to.
              // The panel is keyed on the resolved path, so clicking `graph.cpp`
              // and later its full path reuses one tab instead of opening two.
              alias:{} };

function srcNote(msg){
  const b = document.getElementById('panel-body');
  if (!b) return;
  let n = b.querySelector('.srcnote');
  if (!n){ n = document.createElement('div'); n.className = 'srcnote';
           b.insertBefore(n, b.firstChild); }
  n.textContent = msg;
  clearTimeout(SRC.note);
  SRC.note = setTimeout(() => n.remove(), 6000);
}
// basename -> abspath from DATA, mirroring the daemon's index, so a bare
// `host.cc` becomes absolute before the fetch instead of round-tripping.
function srcIndex(){
  if (SRC.idx) return SRC.idx;
  const idx = {};
  const add = p => { if (p && p.charAt(0) === '/' && !idx[p.split('/').pop()])
                       idx[p.split('/').pop()] = p; };
  const s = DATA.source || {};
  add(s.host_cc); add(s.provenance);
  ['kernel','bcf','dfschedule_ir'].forEach(k => {
    const sub = DATA[k] || {}; add(sub.path); add((sub.source||{}).path); });
  (DATA.tiles || []).forEach(t => {
    add(t.code_file);
    ['kernel', 'bcf'].forEach(k => {
      const sub = t[k] || {}; add(sub.path); add((sub.source || {}).path); });
    (t.dma_channels || []).forEach(c => add(c.code_file)); });
  SRC.idx = idx;
  return idx;
}
function srcParseRef(raw){
  let s = (raw || '').trim().replace(/^[`'"(\[]+/, '').replace(/[`'")\],.;:]+$/, '');
  if (!s) return null;
  let line = null, endLine = null;
  // file:line, file:line:col, file:line-line2 (hyphen or en dash)
  const m = s.match(/^(.*?):(\d+)(?:\s*[-\u2013]\s*(\d+)|:\d+)?$/);
  if (m){ s = m[1]; line = +m[2]; if (m[3]) endLine = +m[3]; }
  if (s.indexOf('?') >= 0 || s.indexOf('<') >= 0) return null;   // ??, <unknown>
  if (s.indexOf('.') < 0 && s.indexOf('/') < 0) return null;
  if (s.slice(0,2) === './') s = s.slice(2);
  if (s.charAt(0) !== '/' && s.indexOf('/') < 0){
    const hit = srcIndex()[s];
    if (hit) s = hit;
  }
  return {path:s, line:line, endLine:endLine};
}
function srcInjectCss(r){
  if (!r.css) return;
  let el = document.getElementById('pygstyle');
  if (!el){ el = document.createElement('style'); el.id = 'pygstyle';
            document.head.appendChild(el); }
  if (el.dataset.ver !== r.css_ver){ el.textContent = r.css; el.dataset.ver = r.css_ver; }
}
function srcBuildBody(item){
  const m = item.meta;
  const win = m.truncated
    ? ' &middot; showing '+m.first+'–'+m.last+' of '+m.lines
    : ' &middot; '+m.lines+' lines';
  return '<div class="codepath srcbanner">'
     + '<b>' + esc(m.display) + '</b> <span class="cpath">' + esc(m.path) + '</span>'
     + '<span class="srcmeta">' + esc(m.lang) + win + '</span>'
     + (m.truncated ? '<button class="srcmore" data-dir="up">▲</button>'
                    + '<button class="srcmore" data-dir="down">▼</button>' : '')
     + '</div><div class="srcwrap">' + m.html + '</div>';
}
function srcWireBody(item, body){
  const panel = document.getElementById('panel-body');
  let row = null;
  if (item.line){
    for (let n = item.line; n <= (item.endLine || item.line); n++){
      const el = body.querySelector('#SL-' + n);
      if (!el) continue;
      el.classList.add('hlline');
      if (!row) row = el;
    }
  }
  // Not scrollIntoView (it walks ancestors and would move the outer flex column
  // offsetParent is <body> and offsetTop measures from the top of the document.
  // Rect deltas are correct whatever the offsetParent turns out to be.
  if (panel){
    if (row && item.scrollTop == null){
      const pr = panel.getBoundingClientRect(), rr = row.getBoundingClientRect();
      panel.scrollTop += (rr.top - pr.top) - (panel.clientHeight - rr.height) / 2;
    } else if (item.scrollTop != null){
      panel.scrollTop = item.scrollTop;
    }
    panel.onscroll = () => { item.scrollTop = panel.scrollTop; };
  }
  body.querySelectorAll('.srcmore').forEach(b => b.onclick = () => {
    const step = (b.dataset.dir === 'up') ? -400 : 400;
    srcOpen(item.path, item.line, {first:Math.max(1, item.meta.first + step),
                                   last:item.meta.last + step});
  });
}
function srcOpen(rawPath, line, span, endLine){
  const ref = srcParseRef(rawPath);
  if (!ref) return;
  if (line == null) line = ref.line;
  if (endLine == null) endLine = ref.endLine;
  if (endLine && line && endLine < line){ const t = line; line = endLine; endLine = t; }
  const tag = line ? (':' + line + (endLine ? '-' + endLine : '')) : '';
  if (!SRC.on){ srcNote('source viewer needs the debug daemon'); return; }
  const key = panelKey('src', SRC.alias[ref.path] || ref.path);
  const open = panelItems.get(key);
  // Already open and the window covers the line: no fetch, just re-highlight.
  if (open && !span && (!line || (line >= open.meta.first && line <= open.meta.last))){
    open.line = line || open.line;
    open.endLine = endLine;
    open.scrollTop = null;
    open.label = open.meta.display + tag;
    panelActiveKey = key; panelSync();
    reportUIState({source:{path:ref.path, line:open.line}});
    return;
  }
  let qs = '/source?path=' + encodeURIComponent(ref.path);
  if (line) qs += '&line=' + line;
  if (span) qs += '&first=' + span.first + '&last=' + span.last;
  api(qs).then(r => {
    if (!r || r.error){ srcNote(('cannot open ' + ref.path.split('/').pop() + ': ')
                                + ((r && r.error) || 'unknown')); return; }
    srcInjectCss(r);
    SRC.alias[ref.path] = r.path;
    const rkey = panelKey('src', r.path);
    const item = {
      kind:'src', label:r.display + tag, color:null,
      path:r.path, line:line, endLine:endLine, meta:r, scrollTop:null,
      buildBody:() => srcBuildBody(item),
      wireBody:(el) => srcWireBody(item, el),
      llmCtx:'[context] source ' + r.rel + tag
    };
    panelItems.set(rkey, item);
    panelActiveKey = rkey;
    panelSync();
    reportUIState({source:{path:r.path, line:line}});
  }).catch(() => srcNote('daemon offline: cannot open source'));
}
// Shared by every click delegate: never hijack a link, and never fire when the
// user is drag-selecting text for "+ Add context".
function srcClickOk(e){
  if (e.target.closest && e.target.closest('a')) return false;
  const s = window.getSelection();
  return !(s && String(s).trim());
}
document.getElementById('llmmsg').addEventListener('click', e => {
  if (!srcClickOk(e)) return;
  const r = e.target.closest && e.target.closest('.srcref');
  if (!r || !r.dataset.p) return;
  srcOpen(r.dataset.p, r.dataset.l ? +r.dataset.l : null, null,
          r.dataset.l2 ? +r.dataset.l2 : null);
});
// conRender keeps classified lines as textContent so "+ Add context" copies
// aiegdb verbatim; re-parse on click rather than switching it to innerHTML.
document.getElementById('conout').addEventListener('click', e => {
  if (!srcClickOk(e)) return;
  const ln = e.target.closest && e.target.closest('.con-ln.con-src');
  if (!ln) return;
  const m = ln.textContent.match(/->\s*(\S+):(\d+)/);
  if (m) srcOpen(m[1], +m[2]);
});
document.getElementById('panel-body').addEventListener('click', e => {
  if (!srcClickOk(e)) return;
  const r = e.target.closest && e.target.closest('.srcref');
  if (!r || !r.dataset.p) return;
  e.stopPropagation();          // a srcref inside <summary> must not fold it
  srcOpen(r.dataset.p, r.dataset.l ? +r.dataset.l : null, null,
          r.dataset.l2 ? +r.dataset.l2 : null);
});
document.getElementById('targetsview').addEventListener('click', e => {
  if (!srcClickOk(e)) return;
  const r = e.target.closest && e.target.closest('.srcref');
  if (!r || !r.dataset.p) return;
  srcOpen(r.dataset.p, r.dataset.l ? +r.dataset.l : null);
});

// ─── aiegdb console (right-bottom, drives aiegdb.py --server) ─────────────────
// A terminal-style console over the daemon's persistent aiegdb REPL subprocess.
// The user types any aiegdb command; scope (partition->tile->channel) is kept by
// the subprocess and reported back via r.scope. Tile/channel clicks send
// `target ...` so the console follows the UI. "Reload aiegdb.py" respawns the
// subprocess (reloads edited code), resetting to partition scope.
const CON = { scope:'partition', hist:[], histIdx:-1, draft:'', spec:GDBSPEC_STATIC };
// A live spec beats the baked-in copy (the daemon may be running an edited
// aiegdb.py); the static one keeps autocomplete working with no daemon at all.
function conLoadSpec(){
  api('/aiegdb/spec').then(r => { if (r && r.spec) CON.spec = r.spec; })
                     .catch(() => {});   // static fallback already in place
}
if (location.protocol === 'http:' || location.protocol === 'https:') conLoadSpec();

// Bare scope level from the decorated prompt breadcrumb, e.g.
// 'partition(startcol=3)/tile(0,0)/mm2s0' -> 'channel'. This is the COMMAND_SPEC key.
function conScopeLevel(){
  const depth = ((CON.scope || '').match(/\//g) || []).length;
  return depth >= 2 ? 'channel' : depth === 1 ? 'tile' : 'partition';
}

// ─── output colorizing ───────────────────────────────────────────────────────
// Matched in order against one plain-text line; first hit wins. A table rather
// than an if-chain so a new aiegdb output shape is a one-line addition.
const CON_RULES = [
  [/^scope ->/,                                     'con-scope'],
  [/^(partition|tile|channel) scope commands/,      'con-hdr'],
  [/^(Navigation|Channel scope|Tile scope|Partition scope|Universal)/, 'con-hdr'],
  // dry-run echoes are scaffolding, not results — dim them before anything else
  // gets a chance to color them by keyword.
  [/^\s*\[dry-run\]/,                               'con-dim'],
  // failures must win over the generic key:value shape below
  [/^\s*(error|Error|ERROR)\b/,                     'con-bad'],
  [/^\s*(usage|warning|WARN)\b/,                    'con-warn'],
  [/\[aiegdb:/,                                     'con-bad'],
  [/\b(STALLED?|FAULT|TIMEOUT|not found|failed)\b/, 'con-bad'],
  // INTRUSIVE only ever appears as a caution in a help listing, so it is a
  // warning rather than an error.
  [/\bINTRUSIVE\b/,                                 'con-warn'],
  [/\b(OK|PASS|DONE|RUNNING|ready|enabled)\b/,      'con-ok'],
  [/->\s*\S+:\d+/,                                  'con-src'],   // PC -> file:line
];
function conClassify(line){
  for (const r of CON_RULES) if (r[0].test(line)) return r[1];
  return null;
}
// Inline spans for lines with no whole-line class: hex literals, and the
// "  key : value" shape aiediag's decoders print.
function conInline(escaped){
  let s = escaped.replace(/\b(0x[0-9A-Fa-f]+)\b/g, '<span class="con-hex">$1</span>');
  // Leading indent is optional: `where` prints "scope:     ..." at column 0
  // while the decoders indent their key/value rows by two.
  return s.replace(/^(\s*)([A-Za-z_][\w .\/-]*?)(\s*:\s)/,
                   (m, ind, k, sep) => ind + '<span class="con-key">' + k + sep + '</span>');
}
// Render aiegdb text into `body`, folding the "[registers read] { ... }" appendix
// into a <details>: run_line appends it after nearly every decoded command and it
// is the single biggest source of the crowding this rework is fixing.
function conRender(body, text){
  let reg = null;
  (text || '').split('\n').forEach(line => {
    if (/^\s*\[registers read\]\s*\{\s*$/.test(line)){
      reg = document.createElement('details');
      reg.className = 'con-reg';
      const sum = document.createElement('summary');
      sum.textContent = 'registers read';
      reg.appendChild(sum);
      body.appendChild(reg);
      return;
    }
    if (reg && /^\s*\}\s*$/.test(line)){ reg = null; return; }
    const div = document.createElement('div');
    div.className = 'con-ln';
    const cls = conClassify(line);
    // textContent (not innerHTML) on classified lines keeps the "+ Add context"
    // selection copying exactly what aiegdb printed.
    if (cls){ div.classList.add(cls); div.textContent = line; }
    else div.innerHTML = conInline(esc(line));
    (reg || body).appendChild(div);
  });
}
// One foldable block per command; returns its .con-body for conRender to fill.
function conBlock(cmd){
  const out = document.getElementById('conout');
  if (!out) return null;
  if (!out.querySelector('.con-blk')) out.innerHTML = '';   // drop placeholder
  const blk = document.createElement('div');
  blk.className = 'con-blk';
  if (cmd){
    const prev = out.querySelector('.con-blk.cur');
    if (prev) prev.classList.remove('cur');
    blk.classList.add('cur');
  }
  const echo = document.createElement('div');
  echo.className = 'con-echo';
  echo.innerHTML = '<span class="cf">▾</span>' +
                   '<span class="cs">' + esc(CON.scope) + '&gt;</span>' +
                   '<span class="cc">' + esc(cmd) + '</span>';
  echo.onclick = () => {
    if (blk.classList.contains('cur')) return;
    const folded = blk.classList.toggle('fold');
    echo.querySelector('.cf').textContent = folded ? '▸' : '▾';
  };
  const body = document.createElement('div');
  body.className = 'con-body';
  blk.appendChild(echo); blk.appendChild(body);
  const current = out.querySelector('.con-blk.cur');
  if (cmd || !current) out.insertBefore(blk, out.firstChild);
  else out.insertBefore(blk, current.nextSibling);
  return body;
}
function conReveal(blk){
  const out = document.getElementById('conout');
  if (!out || !blk) return;
  out.scrollTop = 0;
}
function conNewestVisible(){
  const out = document.getElementById('conout');
  if (!out) return true;
  const newest = out.querySelector('.con-blk.cur, .con-blk');
  if (!newest) return true;
  const ob = out.getBoundingClientRect(), nb = newest.getBoundingClientRect();
  return nb.bottom > ob.top && nb.top < ob.bottom;
}
// Retained for the few non-command notices ([reloaded aiegdb.py], daemon offline).
function conAppend(text){
  const stick = conNewestVisible();
  const body = conBlock('');
  if (!body) return;
  body.parentNode.querySelector('.con-echo').style.display = 'none';
  conRender(body, text);
  if (stick) conReveal(body.parentNode);
}
function conSetScope(s){
  CON.scope = s || 'partition';
  const tgt = document.getElementById('contarget');
  // The header ellipsizes a long breadcrumb, so keep the full text on hover.
  if (tgt){ tgt.textContent = CON.scope; tgt.title = CON.scope; }
  const pr = document.getElementById('conprompt');
  if (pr) pr.textContent = CON.scope + '>';
}
// applyScope=false lets a caller keep an already-set (optimistic) scope instead
// of adopting this command's returned r.scope — used for the intermediate
// `target tile` step of a channel selection so it doesn't downgrade the prompt.
function conSend(cmd, echo, applyScope){
  // setConTarget probes with an empty command purely to spawn aiegdb and learn
  // the scope — that must not emit an empty block.
  const body = (echo !== false && cmd) ? conBlock(cmd) : null;
  if (body) conReveal(body.parentNode);
  const dev = deviceSel ? deviceSel.value : '';
  return api('/aiegdb', {method:'POST', headers:{'Content-Type':'application/json'},
                         body: JSON.stringify({cmd:cmd, device:dev})})
    .then(r => {
      const out = r.output ? r.output.replace(/\n+$/,'')
                : r.error ? 'ERROR: '+r.error : '';
      if (body){
        if (out) conRender(body, out);
        else { const d = document.createElement('div');
               d.className = 'con-ln con-dim'; d.textContent = '(no output)';
               body.appendChild(d); }
        if (r.error || /^\s*error/im.test(out)) body.parentNode.classList.add('err');
      } else if (out) conAppend(out);
      if (applyScope !== false && r.scope) conSetScope(r.scope);
      if (body) conReveal(body.parentNode);   // else: conAppend already scrolled
    })
    .catch(() => {
      if (body){
        const d = document.createElement('div');
        d.className = 'con-ln con-bad';
        d.textContent = 'daemon offline (static mode)';
        body.appendChild(d);
        body.parentNode.classList.add('err');
        conReveal(body.parentNode);
      } else conAppend('daemon offline (static mode)');
    });
}
// Tile/channel click: run `target tile c r`, then chain `target channel dir_ch`.
// The prompt is updated optimistically from the UI selection first, so the
// console scope follows the click immediately — independent of daemon latency,
// and even if the chained `target channel` request is slow or fails (its .catch
// would otherwise leave the prompt stuck at tile scope). A successful daemon
// response still confirms/refines the scope via r.scope. The optimistic string
// mirrors aiegdb's prompt format exactly: partition(startcol=N)/tile(c,r)/dir_ch
// (direction lowercased, matching _parse_dir_ch).
function setConTargetLoc(col, row, ch){
  if (location.protocol === 'file:') return;   // no daemon, nothing to drive
  const box = document.getElementById('cmdconsole');
  if (box) box.classList.remove('hide');
  const rsp = document.getElementById('rhsplitter');
  if (rsp) rsp.classList.remove('hide');
  const sc = (g.startcol !== null && g.startcol !== undefined) ? g.startcol : 0;
  let optScope = 'partition(startcol=' + sc + ')/tile(' + col + ',' + row + ')';
  if (ch) optScope += '/' + ch.direction.toLowerCase() + ch.channel;
  conSetScope(optScope);
  // For a channel selection, suppress the intermediate tile step's scope apply
  // (applyScope=false) so it can't downgrade the optimistic channel prompt; the
  // chained `target channel` then confirms it. A tile-only click applies scope.
  const p = conSend('target tile ' + col + ' ' + row, undefined, !ch);
  if (ch) p.then(() => conSend('target channel ' +
                               ch.direction.toLowerCase() + ch.channel));
}
function setConTarget(t, ch){ setConTargetLoc(t.loc[0], t.loc[1], ch); }
// ─── autocomplete ────────────────────────────────────────────────────────────
// Candidates = the current scope's commands plus the universal ones, drawn from
// COMMAND_SPEC so they can never drift from what aiegdb actually dispatches.
function conCandidates(){
  const sp = CON.spec;
  if (!sp) return [];
  const lvl = conScopeLevel();
  const out = [];
  (sp[lvl] || []).forEach(c => out.push(Object.assign({scope:lvl}, c)));
  (sp.universal || []).forEach(c => out.push(Object.assign({scope:'any'}, c)));
  return out;
}
// Match on the command name and its aliases; a typed alias shows the canonical
// name so the user learns it. Prefix hits rank above substring hits.
function conMatch(q){
  const s = q.trim().toLowerCase();
  const hits = [];
  conCandidates().forEach(c => {
    const names = [c.name].concat(c.aliases || []);
    let best = -1;
    names.forEach(n => {
      const ln = n.toLowerCase();
      if (ln.indexOf(s) === 0){ best = Math.max(best, 2); return; }
      // Word-prefix, not bare substring: typing "d" should offer "dma counter",
      // not "reg read" merely because it contains a d.
      if (ln.split(/[ _-]+/).some(w => w.indexOf(s) === 0)) best = Math.max(best, 1);
    });
    if (!s || best >= 0) hits.push({c:c, rank: s ? best : 2});
  });
  // Stable within a rank, so the spec's own ordering (most-used first) survives.
  hits.sort((a, b) => b.rank - a.rank);
  return hits.map(h => h.c);
}
// ─── argument completion ─────────────────────────────────────────────────────
// A template like "<scope> <register>" says nothing about what to actually type.
// Two sources fill that in: aiedbg's own per-argument help (lazily fetched), and
// — better where it applies — the loaded design itself, which knows exactly which
// tiles and channels exist.
const ARGH = {};                       // command name -> [{name,desc,values}]
function conArgHelp(name, cb){
  if (ARGH[name]){ cb(ARGH[name]); return; }
  api('/aiegdb/arghelp?cmd=' + encodeURIComponent(name))
    .then(r => { ARGH[name] = (r && r.args) || []; cb(ARGH[name]); })
    .catch(() => { ARGH[name] = []; cb([]); });   // offline: no arg values
}
function conCurTile(){
  const m = /tile\((\d+),(\d+)\)/.exec(CON.scope || '');
  return m ? [+m[1], +m[2]] : null;
}
// Values drawn from DATA — the tiles/channels this app actually has, which beats
// anything aiedbg's generic help can offer.
function conDesignValues(cmd, idx, typed){
  const tiles = (DATA && DATA.tiles) || [];
  if (cmd === 'target tile' || cmd === 'tile'){
    if (idx === 0){
      const cols = [...new Set(tiles.map(t => t.loc[0]))].sort((a, b) => a - b);
      return cols.map(c => ({ v:String(c), d:'column ' +
        tiles.filter(t => t.loc[0] === c).length + ' tile(s)' }));
    }
    if (idx === 1){
      const col = parseInt((typed || '').trim().split(/\s+/)[0], 10);
      return tiles.filter(t => isNaN(col) || t.loc[0] === col)
        .sort((a, b) => a.loc[1] - b.loc[1])
        .map(t => ({ v:String(t.loc[1]), d:t.type +
          (t.high_level && t.high_level.role ? ' ' + t.high_level.role : '') }));
    }
  }
  if (cmd === 'target channel' || cmd === 'channel' || cmd === 'dma'){
    if (idx !== 0) return null;
    const loc = conCurTile();
    if (!loc) return null;
    const t = tiles.find(x => x.loc[0] === loc[0] && x.loc[1] === loc[1]);
    if (!t) return null;
    return (t.dma_channels || []).map(ch => ({
      v: ch.direction.toLowerCase() + ch.channel,
      d: 'flow ' + ch.flow_index + (ch.contract ? ': ' + ch.contract : '') }));
  }
  return null;
}
// Split "reg lookup shi" into {cmd:'reg lookup', idx:0, partial:'shi'} — i.e.
// which argument is being typed right now. Returns null when the text is still
// a command name.
function conArgContext(q){
  let best = null;
  conCandidates().forEach(c => {
    if (!c.args) return;
    [c.name].concat(c.aliases || []).forEach(n => {
      if (q.toLowerCase().startsWith(n.toLowerCase() + ' ') &&
          (!best || n.length > best.n.length)) best = { c:c, n:n };
    });
  });
  if (!best) return null;
  const rest = q.slice(best.n.length).replace(/^\s+/, '');
  // Tokenise the TRIMMED text: "shim " would otherwise split to ["shim",""] and
  // report the next argument index one too high.
  const trimmed = rest.trim();
  const toks = trimmed.length ? trimmed.split(/\s+/) : [];
  const trailing = /\s$/.test(q) || rest.length === 0;
  return { cmd: best.c.name, spec: best.c, rest: rest,
           idx: trailing ? toks.length : toks.length - 1,
           partial: trailing ? '' : (toks[toks.length - 1] || '') };
}

const SUG = { open:false, idx:0, items:[], mode:'cmd', ctx:null };
function conSugHide(){
  SUG.open = false; SUG.items = []; SUG.idx = 0;
  const el = document.getElementById('consug');
  if (el) el.classList.add('hide');
}
// Anchor the fixed-position popup to the input. Prefers dropping below (the
// prompt sits at the TOP of the console box), flipping above only when the
// viewport bottom would cut it off.
function conSugPlace(){
  const el = document.getElementById('consug');
  const inp = document.getElementById('conin');
  if (!el || !inp) return;
  const r = inp.getBoundingClientRect();
  const line = document.getElementById('conpromptline').getBoundingClientRect();
  el.style.left  = line.left + 'px';
  el.style.width = line.width + 'px';
  const below = window.innerHeight - r.bottom;
  const h = Math.min(el.scrollHeight || 190, 190);
  if (below < h + 12 && r.top > h + 12) el.style.top = (r.top - h - 6) + 'px';
  else el.style.top = (r.bottom + 6) + 'px';
}
function conSugRender(){
  const el = document.getElementById('consug');
  if (!el) return;
  if (!SUG.items.length){ conSugHide(); return; }
  el.innerHTML = '';
  SUG.items.forEach((c, i) => {
    const row = document.createElement('div');
    row.className = 'con-sug' + (i === SUG.idx && !c.hint ? ' act' : '') +
                    (c.blocking ? ' blocked' : '') + (c.hint ? ' hintrow' : '');
    if (c.blocking) row.title = 'Live aiedbg view (never exits). Run it in a terminal instead.';
    row.innerHTML =
      '<span class="sname">' + esc(c.name) + '</span>' +
      (c.args ? '<span class="sargs">' + esc(c.args) + '</span>' : '') +
      (c.intrusive ? '<span class="swrite">WRITES HW</span>' : '') +
      (c.blocking ? '<span class="sblock">NOT HERE</span>' : '') +
      (c.slow && !c.blocking ? '<span class="sslow">SLOW</span>' : '') +
      (c.scope === 'any' ? '<span class="sscope">any</span>' : '') +
      '<span class="ssum">' + esc(c.summary) + '</span>';
    // mousedown, not click: the input's blur would tear the popup down first.
    row.onmousedown = ev => { ev.preventDefault(); conSugAccept(i); };
    el.appendChild(row);
  });
  el.classList.remove('hide');
  SUG.open = true;
  conSugPlace();
  const act = el.querySelector('.con-sug.act');
  if (act) act.scrollIntoView({block:'nearest'});
}
function conSugShow(q){
  const ctx = conArgContext(q);
  if (ctx){ conSugShowArgs(q, ctx); return; }
  SUG.mode = 'cmd'; SUG.ctx = null;
  SUG.items = conMatch(q).slice(0, 40);
  SUG.idx = 0;
  conSugRender();
}
// Offer values for the argument being typed: the design's own tiles/channels
// where we have them, otherwise aiedbg's documented choices. When neither
// enumerates, still show the argument's description — knowing "nwords: number of
// 32-bit words to read (1-256)" beats staring at "<nwords>".
function conSugShowArgs(q, ctx){
  SUG.mode = 'arg'; SUG.ctx = ctx; SUG.idx = 0;
  const emit = rows => {
    const p = (ctx.partial || '').toLowerCase();
    SUG.items = rows.filter(r => !p || r.name.toLowerCase().startsWith(p)).slice(0, 40);
    conSugRender();
  };
  const design = conDesignValues(ctx.cmd, ctx.idx, ctx.rest);
  if (design && design.length){
    emit(design.map(d => ({ name:d.v, summary:d.d, args:'', argvalue:true })));
    return;
  }
  conArgHelp(ctx.cmd, list => {
    // aiedbg still lists col/row for commands where aiegdb injects them.
    const a = list[ctx.idx + (ctx.spec.coord_skip || 0)];
    if (!a){ conSugHide(); return; }
    if (a.values && a.values.length){
      emit(a.values.map(v => ({ name:v, summary:a.desc, args:'', argvalue:true })));
      return;
    }
    // Not enumerable — one informational row, not a completion.
    SUG.items = [{ name:a.name, summary:a.desc, args:'', hint:true }];
    conSugRender();
  });
}
// Accept = fill in the command name and leave the caret ready for its args,
// rather than running it — the user still confirms with Enter.
function conSugAccept(i){
  const c = SUG.items[i];
  if (!c || c.hint) return;              // informational row, nothing to insert
  const inp = document.getElementById('conin');
  if (c.argvalue){
    // Replace only the partial token being typed, keeping the command and any
    // earlier arguments intact.
    const cut = inp.value.length - (SUG.ctx ? (SUG.ctx.partial || '').length : 0);
    inp.value = inp.value.slice(0, cut) + c.name + ' ';
  } else {
    inp.value = c.name + (c.args ? ' ' : '');
  }
  conSugHide();
  inp.focus();
  // Chain straight into the next argument's values.
  conSugShow(inp.value);
}
function conRun(cmd){
  const v = (cmd || '').trim();
  if (!v) return;
  CON.hist.push(v);
  if (CON.hist.length > 200) CON.hist.shift();
  CON.histIdx = -1;
  conSend(v);
}
const conin = document.getElementById('conin');
conin.addEventListener('input', e => {
  // Match the WHOLE input as a command prefix rather than stopping at the first
  // space: plenty of commands are multi-word ("scan dma", "reg write",
  // "dma counter setup"), so typing "scan " must still offer its completions.
  // Once the text stops being a prefix of anything — i.e. real arguments are
  // being typed, like "reg read 0x1DF10" — conMatch returns nothing and the
  // popup closes on its own.
  conSugShow(e.target.value);
});
conin.addEventListener('blur', () => setTimeout(conSugHide, 120));
// The popup is viewport-anchored, so it has to follow layout changes.
window.addEventListener('resize', () => { if (SUG.open) conSugPlace(); });
conin.addEventListener('keydown', e => {
  if (e.key === 'Escape'){ conSugHide(); return; }
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp'){
    const dir = e.key === 'ArrowDown' ? 1 : -1;
    if (SUG.open){
      e.preventDefault();
      SUG.idx = (SUG.idx + dir + SUG.items.length) % SUG.items.length;
      conSugRender();
      return;
    }
    // Popup closed: Up/Down walk command history instead.
    if (!CON.hist.length) return;
    e.preventDefault();
    if (CON.histIdx === -1){
      if (dir === -1){ CON.draft = e.target.value; CON.histIdx = CON.hist.length - 1; }
      else return;
    } else {
      CON.histIdx += dir;
      if (CON.histIdx >= CON.hist.length){ CON.histIdx = -1; e.target.value = CON.draft; return; }
      if (CON.histIdx < 0) CON.histIdx = 0;
    }
    e.target.value = CON.hist[CON.histIdx];
    return;
  }
  if (e.key === 'Tab'){
    if (SUG.open){ e.preventDefault(); conSugAccept(SUG.idx); }
    return;
  }
  if (e.key === 'Enter'){
    const typed = e.target.value.trim().toLowerCase();
    const exact = conCandidates().some(c => [c.name].concat(c.aliases || [])
      .some(n => n.toLowerCase() === typed));
    // Completing a partial command fills the input rather than running it —
    // "scan " must not silently fire "scan dma" (an array-wide, slow read the
    // user may not have meant). A second Enter then runs the completed command.
    // An exact command, or text the spec doesn't know (aiedbg passthrough,
    // or a command plus its arguments), runs immediately.
    // Fill rather than run when completing a COMMAND name, or when the user has
    // deliberately arrowed onto a suggestion. While typing ARGUMENTS, Enter runs
    // what was typed — argument values are freeform, so silently substituting
    // the top row would be wrong.
    const sel = SUG.items[SUG.idx];
    if (SUG.open && sel && !sel.hint &&
        (SUG.idx > 0 || (SUG.mode === 'cmd' && !exact))){
      e.preventDefault();
      conSugAccept(SUG.idx);
      return;
    }
    conSugHide();
    conRun(e.target.value);
    e.target.value = '';
  }
});

// ─── command palette (the "⌘ Commands" button) ───────────────────────────────
// Replaces the old fixed quick-command row: every command for the current scope,
// searchable, with the same spec-derived metadata the autocomplete uses.
const PAL = { open:false };
function conPalRender(){
  const list = document.getElementById('conpallist');
  const q = (document.getElementById('conpalq').value || '').trim().toLowerCase();
  const items = conMatch(q);
  list.innerHTML = '';
  let lastScope = null;
  if (!items.length){
    list.innerHTML = '<div class="con-palgrp">no match</div>';
    return;
  }
  items.forEach(c => {
    if (c.scope !== lastScope){
      lastScope = c.scope;
      const g2 = document.createElement('div');
      g2.className = 'con-palgrp';
      g2.textContent = c.scope === 'any' ? 'universal' : (c.scope + ' scope');
      list.appendChild(g2);
    }
    const row = document.createElement('div');
    row.className = 'con-sug' + (c.blocking ? ' blocked' : '');
    row.innerHTML =
      '<span class="sname">' + esc(c.name) + '</span>' +
      (c.args ? '<span class="sargs">' + esc(c.args) + '</span>' : '') +
      (c.intrusive ? '<span class="swrite">WRITES HW</span>' : '') +
      (c.blocking ? '<span class="sblock">NOT HERE</span>' : '') +
      (c.slow && !c.blocking ? '<span class="sslow">SLOW</span>' : '') +
      '<span class="ssum">' + esc(c.summary) + '</span>';
    row.onclick = () => {
      conPalClose();
      const inp = document.getElementById('conin');
      // A live TUI view can never run here — staging it in the prompt would just
      // invite the user to hit Enter and hang the console, so explain instead.
      if (c.blocking){
        const body = conBlock(c.name);
        if (body) conRender(body,
          'error: ' + c.name + ' is a live aiedbg view that never exits, so it '
          + 'cannot run in this console.\n'
          + '  Run it in a terminal:  aiedbg -d <device> ' + c.name + '\n'
          + '  For a live view here, use the grid overlay above.');
        if (body) conReveal(body.parentNode);
        return;
      }
      // Commands needing args are staged for editing; complete ones just run.
      if (c.args){ inp.value = c.name + ' '; inp.focus(); }
      else conRun(c.name);
    };
    list.appendChild(row);
  });
}
function conPalOpen(){
  const pal = document.getElementById('conpal');
  const btn = document.getElementById('conpalbtn');
  const r = btn.getBoundingClientRect();
  pal.classList.remove('hide');
  // Anchor under the button, clamped to the viewport.
  pal.style.left = Math.max(6, Math.min(r.left, window.innerWidth - 348)) + 'px';
  pal.style.top  = (r.bottom + 4) + 'px';
  const q = document.getElementById('conpalq');
  q.value = ''; conPalRender(); q.focus();
  PAL.open = true;
}
function conPalClose(){
  document.getElementById('conpal').classList.add('hide');
  PAL.open = false;
}
document.getElementById('conpalbtn').onclick = ev => {
  ev.stopPropagation();
  PAL.open ? conPalClose() : conPalOpen();
};
document.getElementById('conpalq').addEventListener('input', conPalRender);
document.getElementById('conpalq').addEventListener('keydown', e => {
  if (e.key === 'Escape') conPalClose();
  if (e.key === 'Enter'){
    const first = document.querySelector('#conpallist .con-sug');
    if (first) first.click();
  }
});
document.addEventListener('mousedown', e => {
  if (PAL.open && !e.target.closest('#conpal') && !e.target.closest('#conpalbtn'))
    conPalClose();
});
document.getElementById('conclear').onclick = () => {
  document.getElementById('conout').innerHTML =
    '<div class="con-ln con-dim">(cleared)</div>';
};
// Clicking anywhere in the terminal focuses the inline prompt, but don't
// steal focus mid-drag while the user is selecting text in the output.
document.getElementById('conterm').onclick = () => {
  const sel = window.getSelection();
  if (sel && sel.toString()) return;
  document.getElementById('conin').focus();
};
document.getElementById('conreload').onclick = () => {
  api('/aiegdb/reload', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => { conAppend('[reloaded aiegdb.py]'); if (r.scope) conSetScope(r.scope); })
    .catch(() => conAppend('daemon offline (static mode)'));
};

// ─── LLM console (embedded Claude Code, drives claude -p streaming) ───────────
// One persistent `claude -p --output-format stream-json` process in the repo
// root. Each user turn becomes a .llm-msg-you bubble; streamed reply tokens
// accumulate into a .llm-msg-ai bubble via llmAppendToMsg.
const LLM = { off:0, poll:null, busy:false, pendingId:null,
  ctx:new Map(), ctxSent:new Map(), generation:null, pollErrors:0 };
const LLM_CTX_ORDER = ['session', 'run', 'search', 'selection', 'scan'];
let llmMessages = [];
let llmMsgIdCtr = 0;
function llmEscape(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function llmAttr(s){ return llmEscape(s).replace(/"/g,'&quot;'); }
// Args may embed ']' (a JSON list), so take the LAST ']' on the line; the name
// stops at whitespace or '{' so `name{...}` parses the same as `name {...}`.
const LLM_TOOL_CALL = /^\[tool:\s*([^\s\]{]+)\s*(.*)\]$/;
const LLM_TOOL_RES  = /^\[tool result(?::\s*(.*))?\]$/;
const LLM_ACRONYM = {aie:'AIE', dma:'DMA', bd:'BD', pc:'PC', ir:'IR', elf:'ELF',
  gmio:'GMIO', ui:'UI', llm:'LLM', ipc:'IPC', mcp:'MCP', json:'JSON', url:'URL',
  id:'ID', hw:'HW', api:'API', cpu:'CPU', io:'IO'};
// mcp__debugui__get_ui_state -> "Get UI state"; WebFetch -> "Web fetch".
function llmToolName(raw){
  const words = raw.replace(/^mcp__[\s\S]*?__/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .split(/[_\s]+/).filter(Boolean);
  if (!words.length) return raw;
  return words.map((w, i) => {
    const a = LLM_ACRONYM[w.toLowerCase()];
    if (a) return a;
    return i === 0 ? w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()
                   : w.toLowerCase();
  }).join(' ');
}
// One labelled chip per argument. Raw text only when the JSON will not parse.
function llmToolArgs(args){
  args = (args || '').trim();
  if (!args) return '';
  let obj = null;
  try { obj = JSON.parse(args); } catch (e) { obj = null; }
  if (!obj || typeof obj !== 'object' || Array.isArray(obj))
    return '<span class="targ">' + llmEscape(args) + '</span>';
  return Object.keys(obj).map(k => {
    const v = obj[k];
    let s = v === null ? 'null'
          : typeof v === 'object' ? JSON.stringify(v) : String(v);
    if (s.length > 60) s = s.slice(0, 59) + '…';
    return '<span class="tg"><i>' + llmEscape(k) + '</i>' + llmEscape(s) + '</span>';
  }).join('');
}
function llmBuildTools(rows){
  return '<div class="llm-tools">' + rows.map(r => {
    const err = /error/i.test(r.res || '');
    const mark = r.res == null ? '…' : (err ? '✗ ' : '✓ ') + r.res;
    return '<div class="llm-tc' + (r.res == null ? '' : err ? ' err' : ' done')
      + '" title="' + llmAttr((r.name + ' ' + r.args).trim()) + '">'
      + '<span class="tn">'
      + (r.name ? llmEscape(llmToolName(r.name)) : '↳') + '</span>'
      + '<span class="ta">' + llmToolArgs(r.args) + '</span>'
      + '<span class="tr">' + mark + '</span></div>';
  }).join('') + '</div>';
}
// Colorize an ALREADY-escaped prose string: tool calls, tool results, you>
// prompts, error/offline markers, and file / file:line references.
function llmColorizeMarkers(s){
  s = s.replace(/^you&gt;.*$/gm, m => '<span class="llm-you">' + m + '</span>');
  s = s.replace(/\[(llm error[^\]]*|daemon offline[^\]]*)\]/g,
    (m,inner) => '<span class="llm-error">[' + inner + ']</span>');
  // Wrap the pair in one carrier so a click has the path and line together.
  // The char class excludes " < > &, so the attributes cannot be broken out of;
  // widening it would require llmAttr(). elf is gone (binary, always refused).
  // Lookbehind, not \b: \b never matches between a space and '/', so a leading
  // slash was dropped and every absolute path the model cited resolved nowhere.
  s = s.replace(/(?<![\w./-])([A-Za-z0-9_./-]+\.(?:py|cc|cpp|cxx|h|hpp|c|md|mlir|sh|json|txt|inc|td|html|log|bcf))(?::(\d+)(?:\s*[-\u2013]\s*(\d+))?)?\b/g,
    (m,file,line,line2) => {
      // Show the basename only — an absolute app path runs to ~60 characters and
      // swamps the sentence around it. data-p keeps the full path for the click,
      // and title exposes it on hover.
      const shown = file.slice(file.lastIndexOf('/') + 1);
      return '<span class="srcref" data-p="' + file + '"'
        + (line ? ' data-l="' + line + '"' : '')
        + (line2 ? ' data-l2="' + line2 + '"' : '')
        + (shown !== file ? ' title="' + file + '"' : '') + '>'
        + '<span class="llm-file">' + shown + '</span>'
        + (line ? ':<span class="llm-line">' + line
                  + (line2 ? '-' + line2 : '') + '</span>' : '')
        + '</span>';
    });
  return s;
}
// Minimal offline highlighter for one fenced code block.
const LLM_TOK = /(\/\*[\s\S]*?\*\/|\/\/[^\n]*|#\s[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(0x[0-9a-fA-F]+|\d+\.?\d*)\b|\b(if|else|elif|for|while|do|return|break|continue|switch|case|default|goto|int|float|double|char|bool|void|long|short|unsigned|signed|const|static|struct|class|public|private|protected|namespace|template|typename|typedef|enum|union|new|delete|sizeof|this|nullptr|true|false|NULL|auto|using|include|define|import|from|def|lambda|None|True|False|self|and|or|not|in|is|print|std)\b/g;
function llmHighlightCode(raw){
  return llmEscape(raw.replace(/\n$/,'')).replace(LLM_TOK, (m,c,s,n,k) =>
    c ? '<span class="cm-comment">' + c + '</span>' :
    s ? '<span class="cm-string">'  + s + '</span>' :
    n ? '<span class="cm-number">'  + n + '</span>' :
    k ? '<span class="cm-keyword">' + k + '</span>' : m);
}
function llmBuildTable(lines){
  // lines: array of raw (unescaped) pipe-delimited rows
  const rows = lines.map(l => l.replace(/^\s*\|/, '').replace(/\|?\s*$/, '').split('|').map(c => c.trim()));
  const isSep = r => r.every(c => /^:?-+:?$/.test(c));
  let html = '<table class="md-table">';
  let inHead = rows.length > 1 && isSep(rows[1]);
  rows.forEach((r, i) => {
    if (isSep(r)) return;
    const isHead = inHead && i === 0;
    const tag = isHead ? 'th' : 'td';
    html += '<tr>' + r.map(c => '<' + tag + '>' + llmEscape(c) + '</' + tag + '>').join('') + '</tr>';
  });
  return html + '</table>';
}
// Pre-built block HTML, keyed by placeholder index. Render-scoped rather than
// local to llmProse: the placeholders must outlive llmColorizeMarkers, whose
// file-path regex would otherwise rewrite the inside of a block's title="".
let _llmBlocks = [];
function llmProse(raw){
  // Process line-by-line to group block elements (tables, ordered lists) without
  // regex alternation or nested quantifiers that cause catastrophic backtracking.
  const blocks = _llmBlocks;
  const outLines = [];  // lines after block extraction (may contain placeholders)

  const lines = raw.split('\n');
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    // Tool block: a run of call/result markers plus the blank lines the daemon
    // wraps them in, collapsed to one row per call. Left as prose they cost
    // three lines each and the result never pairs with its call.
    if (LLM_TOOL_CALL.test(l.trim()) || LLM_TOOL_RES.test(l.trim())) {
      const rows = [];
      while (i < lines.length) {
        const t = lines[i].trim();
        if (!t) { i++; continue; }
        let m = t.match(LLM_TOOL_CALL);
        if (m) { rows.push({name:m[1], args:(m[2]||'').trim(), res:null}); i++; continue; }
        m = t.match(LLM_TOOL_RES);
        if (!m) break;
        // Fill the FIRST unresolved call, not the last. A parallel batch emits
        // N calls back to back and then N results, so pairing to the last row
        // left every earlier call stuck on '…' and the extra results orphaned.
        const open = rows.find(x => x.res == null);
        if (open) open.res = m[1] || 'done';
        else rows.push({name:'', args:'', res:m[1] || 'done'});
        i++;
      }
      while (outLines.length && !outLines[outLines.length-1].trim()) outLines.pop();
      outLines.push('\x00BLK' + blocks.length + '\x00');
      blocks.push(llmBuildTools(rows));
      continue;
    }
    // Table block: consecutive lines starting with |
    if (l.trimStart().startsWith('|')) {
      const group = [];
      while (i < lines.length && lines[i].trimStart().startsWith('|')) group.push(lines[i++]);
      outLines.push('\x00BLK' + blocks.length + '\x00');
      blocks.push(llmBuildTable(group));
      continue;
    }
    // Ordered list block: consecutive lines matching /^\d+\. /
    if (/^\d+\.\s/.test(l)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i]))
        items.push('<li>' + llmEscape(lines[i++].replace(/^\d+\.\s+/, '')) + '</li>');
      outLines.push('\x00BLK' + blocks.length + '\x00');
      blocks.push('<ol class="md-ol">' + items.join('') + '</ol>');
      continue;
    }
    outLines.push(l);
    i++;
  }

  let s = llmEscape(outLines.join('\n'));
  s = s.replace(/^(#{1,6})\s+(.*)$/gm, (m,h,t) => '<span class="md-h">' + t + '</span>');
  s = s.replace(/^(\s*)[-*+]\s+/gm, '$1<span class="md-bullet">\u2022 </span>');
  s = s.replace(/`([^`\n]+)`/g, (m,c) => '<code class="md-code">' + c + '</code>');
  // Links before bold/italic so brackets aren't consumed by other patterns.
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m,t,u) => '<a class="md-a" href="' + u + '" target="_blank" rel="noopener">' + t + '</a>');
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em>$1</em>');
  return s;
}
// Split on ``` fences: even segments are prose, odd are code blocks.
// Unclosed trailing fence mid-stream is treated as code for safe partial renders.
function llmRenderText(raw, colorize){
  _llmBlocks = [];
  const parts = raw.split('```');
  let html = '';
  for (let i = 0; i < parts.length; i++){
    if (i % 2 === 0){ html += llmProse(parts[i]); continue; }
    const blk = parts[i], nl = blk.indexOf('\n');
    const body = nl >= 0 ? blk.slice(nl + 1) : blk;
    html += '<pre class="md-block"><code>' + llmHighlightCode(body) + '</code></pre>';
  }
  if (colorize) html = llmColorizeMarkers(html);
  return html.replace(/\x00BLK(\d+)\x00/g, (m, idx) => _llmBlocks[+idx] || '');
}

// Append a new message bubble; returns its DOM id.
function llmAddMsg(role, html){
  const id = 'llmm' + (++llmMsgIdCtr);
  const rec = {id, role, html, _raw:''};
  llmMessages.push(rec);
  const list = document.getElementById('llmmsg');
  if (!list) return id;
  const near = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
  const div = document.createElement('div');
  div.id = id;
  div.className = 'llm-msg llm-msg-' + role;
  div.innerHTML = html;
  list.appendChild(div);
  if (near) list.scrollTop = list.scrollHeight;
  return id;
}
// Stream raw text into an existing AI bubble, throttled to one render per animation frame.
// llmColorizeMarkers (expensive filename regex) is deferred until streaming ends.
const _llmPending = new Map(); // id -> rafHandle
function llmAppendToMsg(id, rawChunk, finalize){
  const rec = llmMessages.find(x => x.id === id);
  if (!rec) return;
  rec._raw += rawChunk;
  if (finalize){
    // Cancel any pending frame; do one final full render including colorization.
    if (_llmPending.has(id)){ cancelAnimationFrame(_llmPending.get(id)); _llmPending.delete(id); }
    rec.html = llmRenderText(rec._raw, true);
    const el = document.getElementById(id);
    if (el) el.innerHTML = rec.html;
    const list = document.getElementById('llmmsg');
    if (list && list.scrollHeight - list.scrollTop - list.clientHeight < 60)
      list.scrollTop = list.scrollHeight;
    return;
  }
  if (_llmPending.has(id)) return; // already scheduled for this frame
  _llmPending.set(id, requestAnimationFrame(() => {
    _llmPending.delete(id);
    rec.html = llmRenderText(rec._raw, false);
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = rec.html;
    const list = document.getElementById('llmmsg');
    if (list && list.scrollHeight - list.scrollTop - list.clientHeight < 60)
      list.scrollTop = list.scrollHeight;
  }));
}
// #llmmsg shares a flex column with the think dots, the context pills and the
// input box. Anything that changes their height shrinks the transcript, and a
// view pinned to the bottom keeps its scrollTop — so the last lines slide out
// of sight. Re-pin whatever was at the bottom. Reading scrollHeight after the
// mutation forces the reflow, so the new geometry is what we scroll against.
function llmKeepBottom(mutate){
  const list = document.getElementById('llmmsg');
  const atBottom = !list ||
    (list.scrollHeight - list.scrollTop - list.clientHeight) < 60;
  mutate();
  if (atBottom && list) list.scrollTop = list.scrollHeight;
}
function llmShowThink(on){
  const t = document.getElementById('llmthink');
  if (!t) return;
  llmKeepBottom(() => t.classList.toggle('hide', !on));
}

function llmStopPoll(){ if (LLM.poll){ clearInterval(LLM.poll); LLM.poll = null; } }
function llmAdoptGeneration(r, announce){
  if (!r || r.llm_generation == null) return false;
  const previous = LLM.generation;
  LLM.generation = r.llm_generation;
  if (previous == null || previous === LLM.generation) return false;
  llmStopPoll();
  llmShowThink(false);
  LLM.off = 0;
  LLM.ctxSent.clear();
  const reason = r.llm_reset_reason || 'the LLM process restarted';
  if (LLM.pendingId){
    llmAppendToMsg(LLM.pendingId, '\n[interrupted: ' + reason + ']', true);
    LLM.pendingId = null;
  } else if (announce !== false){
    llmAddMsg('ctx', reason + ' (agent context reset)');
  }
  return true;
}
function llmPollOnce(){
  if (LLM.busy) return;
  LLM.busy = true;
  api('/llm/poll?offset=' + LLM.off).then(r => {
    LLM.pollErrors = 0;
    if (r.auth){ llmLock(); return; }
    if (r.error){ llmStopPoll(); llmShowThink(false); return; }
    if (llmAdoptGeneration(r)) return;
    if (r.stuck){
      const s = r.stuck_s ? r.stuck_s + 's' : '2min';
      llmStopPoll();
      llmShowThink(false);
      if (LLM.pendingId){
        llmAppendToMsg(LLM.pendingId,
          `\n[no response after ${s} — the turn may be stuck. Use **Reset chat** to start a fresh conversation.]`,
          true);
        LLM.pendingId = null;
      }
      return;
    }
    const done = r.active === false;
    if (r.data && LLM.pendingId) llmAppendToMsg(LLM.pendingId, r.data, done);
    else if (done && LLM.pendingId) llmAppendToMsg(LLM.pendingId, '', true);
    if (r.next != null) LLM.off = r.next;
    // Tracks the turn, not the first chunk: the model keeps working through
    // tool calls and further text long after the opening tokens land.
    llmShowThink(!done);
    if (done){ llmStopPoll(); LLM.pendingId = null; }
  }).catch(() => {
    LLM.pollErrors += 1;
    if (LLM.pollErrors > 4){ llmStopPoll(); llmShowThink(false); }
  }).finally(() => { LLM.busy = false; });
}
// Echo the context that just went out, as read-only pills above the message.
// Without this the blocks vanished from the transcript the moment they were
// drained, leaving no record of what the model was actually given.
function llmAddSentCtx(sent){
  if (!sent || !sent.length) return;
  const html = '<span class="sctx-lead">context sent:</span>'
    + sent.map(s => '<span class="llm-sent-pill" title="' + llmAttr(s.text) + '">'
        + llmEscape(s.label) + '</span>').join('');
  const id = llmAddMsg('sctx', html);
  const el = document.getElementById(id);
  if (!el) return;
  // Click a pill to reveal exactly what was sent under that label.
  el.querySelectorAll('.llm-sent-pill').forEach((p, i) => p.onclick = () => {
    const open = p.nextElementSibling &&
                 p.nextElementSibling.classList.contains('sctx-body');
    el.querySelectorAll('.sctx-body').forEach(b => b.remove());
    p.classList.toggle('act', !open);
    if (open) return;
    const pre = document.createElement('pre');
    pre.className = 'sctx-body';
    pre.textContent = sent[i].text;
    el.appendChild(pre);
  });
}
// fromInput: the message is whatever is in the editable box, pills included.
// A programmatic caller passes a literal string instead, so it cannot drain a
// half-written message out from under the user.
function llmSend(prompt, fromInput){
  prompt = (prompt || '').trim();
  const drained = (fromInput && window._drainCtxSnippets)
    ? window._drainCtxSnippets() : {text:prompt, sent:[]};
  if (!drained.text.trim() && !drained.sent.length) return;
  llmStopPoll();
  let toSend = drained.text;
  const fresh = [];
  const chans = LLM_CTX_ORDER.concat(
    [...LLM.ctx.keys()].filter(k => LLM_CTX_ORDER.indexOf(k) < 0));
  chans.forEach(chan => {
    const text = LLM.ctx.get(chan);
    if (text && LLM.ctxSent.get(chan) !== text){
      fresh.push(text);
      LLM.ctxSent.set(chan, text);
    }
  });
  if (fresh.length){
    const blob = fresh.join('\n');
    toSend = blob + '\n' + toSend;
    llmAddMsg('ctx', llmEscape(blob));
  }
  llmAddSentCtx(drained.sent);
  // The prompt is echoed with pills reduced to [[label]] — showing the expanded
  // blocks here would bury the question the user actually asked.
  if (prompt) llmAddMsg('you', llmEscape(prompt));
  if (fromInput && window._llmClearInput) window._llmClearInput();
  const aiId = llmAddMsg('ai', '');
  LLM.pendingId = aiId;
  llmShowThink(true);
  api('/llm', {method:'POST', headers:{'Content-Type':'application/json'},
               body: JSON.stringify({prompt:toSend})})
    .then(r => {
      if (r.auth){ llmLock(); return; }
      if (!r.ok){ llmShowThink(false);
        llmAppendToMsg(aiId, '[llm error: ' + (r.error || 'unknown') + ']'); return; }
      if (r.llm_generation != null) LLM.generation = r.llm_generation;
      if (r.offset != null) LLM.off = r.offset;
      llmStopPoll();
      llmPollOnce();
      LLM.poll = setInterval(llmPollOnce, 700);
    })
    .catch(() => { llmShowThink(false);
      llmAppendToMsg(aiId, '[daemon offline: LLM tab needs schedule_debug_server]'); });
}
function llmReset(){
  llmStopPoll();
  api('/llm/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => {
      if (r && r.auth){ llmLock(); return; }
      if (r && r.llm_generation != null) LLM.generation = r.llm_generation;
      LLM.ctxSent.clear(); LLM.off = 0; LLM.pendingId = null;
      llmMessages = [];
      const list = document.getElementById('llmmsg');
      if (list) list.innerHTML = '';
      llmShowThink(false);
      llmAddMsg('ctx', 'new chat \u2014 context reset');
    })
    .catch(() => { llmAddMsg('ctx', '[daemon offline: cannot reset]'); });
}
// ── LLM password modal ──────────────────────────────────────────────────────
function llmAuthShow(){ const m=document.getElementById('llmauth');
  if (m){ m.classList.remove('hide');
    const i=document.getElementById('llmauthin'); if (i){ i.value=''; i.focus(); } } }
function llmAuthHide(){ const m=document.getElementById('llmauth');
  if (m) m.classList.add('hide');
  const e=document.getElementById('llmautherr'); if (e) e.textContent=''; }
function llmLock(){
  sessionStorage.removeItem('LLM_AUTH');
  llmStopPoll();
  llmAddMsg('ctx', '[locked: enter password]');
  llmAuthShow();
}
// Ask the daemon whether auth is required; prompt if so and no token stored yet.
function llmCheckAuth(){
  api('/llm/auth').then(r => {
    if (r && r.required && !llmToken()) llmAuthShow();
  }).catch(() => {});
}
(function(){
  const inp = document.getElementById('llmin');
  const snd = document.getElementById('llmsend');
  const rst = document.getElementById('llmreset');
  const term = document.getElementById('llmterm');
  function submitLLM(){
    if (!inp) return;
    const v = window._llmInputText ? window._llmInputText() : '';
    if (v || (window._hasCtxSnippets && window._hasCtxSnippets())) llmSend(v, true);
  }
  if (inp) inp.addEventListener('keydown', e => {
    // Backspace over a chip removes the whole chip in one press. Left to the
    // browser it would eat the flanking zero-width space first, so the first
    // press looked like it did nothing.
    if (e.key === 'Backspace' || e.key === 'Delete'){
      const sel = window.getSelection();
      if (sel && sel.rangeCount && sel.isCollapsed && window._ctxChipBefore){
        const r = sel.getRangeAt(0);
        const pill = (e.key === 'Backspace')
          ? window._ctxChipBefore(r)
          : (r.startContainer.nodeType === 1
             ? r.startContainer.childNodes[r.startOffset] : null);
        if (pill && pill.classList && pill.classList.contains('msg-pill')){
          e.preventDefault();
          window._ctxRemoveChip(pill);
          return;
        }
      }
    }
    if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); submitLLM(); }
    // Shift+Enter: a plain newline. Left to the browser this produces a <div>
    // or <br> soup that the reader would then have to normalise.
    if (e.key === 'Enter' && e.shiftKey){
      e.preventDefault();
      document.execCommand('insertLineBreak');
    }
  });
  // Copy/cut: a chip is part of the text. Write both flavours — plain text so
  // it lands as [[label]] anywhere else, and HTML so pasting back into the box
  // rebuilds a real chip instead of dropping to a token.
  const clipOut = (e, cut) => {
    if (!window._ctxSelectionClip) return;
    const c = window._ctxSelectionClip();
    if (!c) return;
    e.preventDefault();
    const cd = e.clipboardData || window.clipboardData;
    cd.setData('text/plain', c.text);
    cd.setData('text/html', c.html);
    if (cut){ c.range.deleteContents(); inp.dispatchEvent(new Event('input')); }
  };
  if (inp) inp.addEventListener('copy', e => clipOut(e, false));
  if (inp) inp.addEventListener('cut',  e => clipOut(e, true));
  // Paste rebuilds chips from our own HTML flavour; anything else is inserted
  // as plain text, so foreign markup never enters the box.
  if (inp) inp.addEventListener('paste', e => {
    e.preventDefault();
    const cd = e.clipboardData || window.clipboardData;
    const html = cd.getData('text/html');
    if (html && html.indexOf('msg-pill') >= 0 && window._ctxFragFromHtml){
      window._ctxInsertFrag(window._ctxFragFromHtml(html));
      return;
    }
    document.execCommand('insertText', false, cd.getData('text'));
  });
  // Removing a pill by its ×; the caret can also just backspace over it.
  if (inp) inp.addEventListener('click', e => {
    const x = e.target.closest && e.target.closest('.mp-x');
    if (!x) return;
    const pill = x.closest('.msg-pill');
    if (pill && window._ctxRemoveChip) window._ctxRemoveChip(pill);
  });
  // The box grows with its content, which shrinks the transcript above it.
  if (inp) inp.addEventListener('input', () => llmKeepBottom(() => {}));
  if (snd) snd.onclick = submitLLM;
  if (rst) rst.onclick = llmReset;
  if (term) term.onclick = () => {
    const sel = window.getSelection();
    if (sel && sel.toString()) return;
    if (inp) inp.focus();
  };
  const authin = document.getElementById('llmauthin');
  const authok = document.getElementById('llmauthok');
  const doUnlock = () => {
    const v = authin ? authin.value : '';
    if (!v){ const e=document.getElementById('llmautherr');
      if (e) e.textContent='enter a password'; return; }
    sessionStorage.setItem('LLM_AUTH', v);
    llmAuthHide();
  };
  if (authok) authok.onclick = doUnlock;
  if (authin) authin.addEventListener('keydown', e => {
    if (e.key === 'Enter') doUnlock();
  });
  llmCheckAuth();
})();
// Console tab switching: toggle .hide on panes + .act on the tabs.
const _CON_PANES = ['conpane', 'llmpane', 'searchpane'];
document.querySelectorAll('#contabs .contab').forEach(tab => tab.onclick = () => {
  document.querySelectorAll('#contabs .contab').forEach(x => x.classList.remove('act'));
  tab.classList.add('act');
  const pane = tab.dataset.pane;
  _CON_PANES.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hide', id !== pane);
  });
  reportUIState({console_pane: pane});
  if (pane === 'llmpane') llmCheckAuth();
  if (pane === 'searchpane') document.getElementById('sr-input').focus();
});

document.getElementById('gbtn').onclick = () => {
  clearPeers();
  if (selEl){ selEl.classList.remove('sel'); selEl=null; }
  const gl = DATA.global;
  // #panel-body, not #panel: overwriting the pane destroys the container
  // panelRenderBody() writes into, so the next tile click hits a null.
  document.getElementById('panel-body').innerHTML =
    '<h2>Global / kernel-group</h2>' +
    '<div class="kv">'+esc(gl.note)+'</div>' +
    '<pre class="code">'+hl(gl.code||'')+'</pre>';
  panelBuildToc();
};

// ─── Live debug overlay (talks to schedule_debug_server on the same origin) ───
// All additive: when opened as a static file:// (no daemon) every fetch fails
// and the page silently stays in static mode.
const LIVE = { enabled:false, connected:false, what:'dma', gridTimer:null,
               conTimer:null, logoff:0, gridBusy:false, rescan:false, logBusy:false,
               runActive:false, debugUnlocked:false, device:'', host:'',
               hwsrvOff:0, hwsrvTimer:null, hwsrvBusy:false,
               // runOwned mirrors the DAEMON's bookkeeping (/runstate), not this
               // page's: a reload or dropped tail must not convince the UI that
               // a live run is over.
               runOwned:false, daemonRun:null, rsBusy:false, rsTimer:null,
               // Timeline auto-launch. tlDone dedups the final render across the
               // TIMESYNC-idle timer, natural run-end and force-stop paths.
               // tsSeen: the first [TIMESYNC] line has appeared this run.
               // tsIdleTimer: the 5s "quiet" timer that fires the render once no
               // new [TIMESYNC] line has arrived for a full 5 seconds.
               tlDone:false, tsSeen:false, tsIdleTimer:null,
               // profileReady is the single source of truth for the Profile tab:
               // true once a timeline has rendered, false again only when a new
               // Run starts. switchView re-asserts the tab from this flag so no
               // view switch (e.g. Device Map) can leave the tab stuck disabled.
               profileReady:false,
               simOnly:false, jtagHost:'' };
const HW_SERVER_CMD = 'exec hw_server -stcp:0.0.0.0:3121';
const LSTATE = {
  running:['#4caf50','RUN'], stalled:['#ffca28','STALL'], error:['#ef5350','ERR'],
  completed:['#26a69a','done'], idle:['#546e7a','idle'],
  unreachable:['#8d6e63','n/a'], unknown:['#546e7a','?'],
  verified:['#2e7d32','sw ok'], mismatch:['#d84315','sw ≠']
};
let SWSCAN = null;
let SWSCAN_RES = null;

const STATIC_PATHS = DATA.comm_paths || [];
let DYNAMIC = null;

function setRoutingSource(which){
  if (which === 'dynamic' && !(DYNAMIC && DYNAMIC.comm_paths)) {
    syncRoutingSource(); return;
  }
  ROUTING_SRC = which;
  DATA.comm_paths = which === 'dynamic' ? DYNAMIC.comm_paths : STATIC_PATHS;
  recomputeFlowIds();
  syncRoutingSource();
  if (document.getElementById('devmap').classList.contains('show')) buildDeviceMap();
  if (panelActiveKey) panelRenderBody(panelActiveKey);
  reapplySwitchScan();
}

function setRoutingDiff(on){
  ROUTING_DIFF = !!on;
  syncRoutingSource();
  if (document.getElementById('devmap').classList.contains('show')) buildDeviceMap();
  if (panelActiveKey) panelRenderBody(panelActiveKey);
  reapplySwitchScan();
}

// No interpolation: the markup is fixed and every varying string is written as
// a property below, so the reconstructor's error text never reaches an HTML
// parser.
const RSRC_HTML =
    '<label class="rsrc-diff-wrap" title="annotate the selected map\'s stream-switch rows with what the other map does or does not have">'
  + '<input type="checkbox" class="rsrc-diff" onchange="setRoutingDiff(this.checked)"> diff</label>'
  + '<select class="scan-what rsrc-sel" title="which routing map every panel reads"'
  + ' onchange="setRoutingSource(this.value)">'
  + '<option value="static" title="the map the compiler emitted"></option>'
  + '<option value="dynamic"></option>'
  + '</select>'
  + '<button class="rs-dl" title="save the reconstructed routing map as JSON"'
  + ' onclick="downloadDynamic()">Save JSON</button>'
  + '<span class="rsrc-err" hidden></span>';

function syncRoutingSource(){
  ['gridRsrc','dmRsrc'].forEach(id=>{
    const el = document.getElementById(id);
    if (!el) return;
    el.hidden = !DYNAMIC;
    if (!DYNAMIC) return;
    // applyGrid() calls this on every switch scan and `live` re-scans every 2s,
    // so the nodes are built once: replacing the markup would slam an open
    // dropdown shut under the cursor mid-selection.
    if (!el.firstChild) el.innerHTML = RSRC_HTML;
    const diff = el.querySelector('.rsrc-diff');
    const sel = el.querySelector('.rsrc-sel');
    const dl  = el.querySelector('.rs-dl');
    const msg = el.querySelector('.rsrc-err');
    const err = DYNAMIC.error || '';
    sel.hidden = !!err; dl.hidden = !!err; msg.hidden = !err;
    if (diff){ diff.checked = ROUTING_DIFF; diff.disabled = !!err; }
    if (err){ msg.textContent = 'routing: reconstruct failed'; msg.title = err; return; }
    const n = DYNAMIC.n_flows || 0;
    sel.querySelector('option[value="static"]').textContent =
      'static · ' + n + ' found';
    const od = sel.querySelector('option[value="dynamic"]');
    od.textContent = 'dynamic (' + n + ')';
    od.title = n + ' flow' + (n===1?'':'s') + ' reconstructed from the live switch'
             + ' registers — independent of the provenance map';
    sel.value = ROUTING_SRC;
    sel.dataset.src = ROUTING_SRC;
    sel.title = ROUTING_DIFF
      ? 'which routing map every panel reads — and the side diff annotates from'
      : 'which routing map every panel reads';
  });
}

function downloadDynamic(){
  if (!(DYNAMIC && DYNAMIC.routing_groups)) return;
  const blob = new Blob([JSON.stringify(DYNAMIC.routing_groups, null, 1)],
                        {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'routingprovenancemap.dynamic.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

function llmToken(){ return sessionStorage.getItem('LLM_AUTH') || ''; }
function api(path, opts){
  opts = opts || {};
  const tok = llmToken();
  if (tok) opts.headers = Object.assign({}, opts.headers, {'X-LLM-Auth': tok});
  return fetch(path, opts).then(r => r.json());
}
// Render `timeline.py <applog>` to timeline.png on the server and show it in the
// Profile tab. Trigger model: TIMESYNC-idle. The render only happens once real
// [TIMESYNC] data has been seen this run (LIVE.tsSeen) — the primary caller is
// the 5s "quiet" timer armed in pollLog when [TIMESYNC] lines stop arriving.
// The run-end and force-stop callers are gated safety nets: on a run with no
// TIMESYNC output they no-op (tsSeen stays false), which is what avoids the
// `no [TIMESYNC] cps= record found` error. LIVE.tlDone dedups across all three
// callers; both flags reset when the next run starts.
function triggerTimeline(final){
  if (!LIVE.tsSeen) return;      // no TIMESYNC data this run → nothing to render.
  if (LIVE.tlDone) return;
  LIVE.tlDone = true;
  setStatus('timeline: rendering…');
  api('/timeline', {method:'POST', headers:{'Content-Type':'application/json'},
                    body:'{}'})
    .then(r => {
      if (r && r.started){ showTimelineProfile(); setStatus('timeline: ready (Profile tab)'); }
      else { setStatus('timeline: ' + ((r && r.error) || 'not generated')); }
    })
    .catch(() => {});
}
function setStatus(msg){ const e=document.getElementById('livestatus'); if(e) e.textContent=msg; }
function setRunStatus(msg){ const e=document.getElementById('runstatus'); if(e) e.textContent=msg; }

function clearBars(){
  Object.values(liveBar).forEach(b => { b.className='livebar hide'; b.textContent=''; });
  applyIssues([]);
}


// Build a lookup: "col,row,chankey" → flow_index, from the static schedule data.
function _buildFlowLookup(){
  const m = {};
  (DATA.tiles||[]).forEach(t => {
    const [tc, tr] = t.loc;
    (t.dma_channels||[]).forEach(ch => {
      if (ch.flow_index != null)
        m[tc+','+tr+','+(ch.direction.toLowerCase()+ch.channel)] = ch.flow_index;
    });
  });
  return m;
}

// Render the issue bar. issues = [] hides it; non-empty shows it.
// Each issue: {key, col, row, chan, state, stalls, errors, flow_index}
function applyIssues(issues){
  const bar = document.getElementById('issue-bar');
  if (!bar) return;
  if (!issues || !issues.length){ bar.style.display='none'; bar.innerHTML=''; return; }
  const icon = st => st === 'error' ? '\u2297' : '\u26a0';  // ⊗ or ⚠
  const rows = issues.map(iss => {
    const detail = iss.errors.length ? iss.errors.join(',')
                 : iss.stalls.length ? iss.stalls.join(',')
                 : iss.state;
    const fi = iss.flow_index != null ? 'f'+iss.flow_index : '';
    return '<div class="ib-row" data-key="'+iss.key+'" data-col="'+iss.col+'" data-row="'+iss.row+'">'
      + '<span class="ib-icon">'+icon(iss.state)+'</span>'
      + '<span class="ib-loc">('+iss.col+','+iss.row+')</span>'
      + '<span class="ib-ch">'+iss.chan.toUpperCase()+'</span>'
      + '<span class="ib-msg">'+iss.state+' ['+detail+']</span>'
      + (fi ? '<span class="ib-flow">'+fi+'</span>' : '')
      + '</div>';
  }).join('');
  bar.innerHTML = '<div class="ib-hdr">'
    + '\u26a0 '+issues.length+' DMA issue'+(issues.length>1?'s':'')
    + '<button class="ib-clear" title="Dismiss">clear</button>'
    + '</div>' + rows;
  bar.style.display = '';
  // Clicking a row selects the tile in the grid view.
  bar.querySelectorAll('.ib-row').forEach(row => {
    row.addEventListener('click', () => {
      const col = parseInt(row.dataset.col), r2 = parseInt(row.dataset.row);
      const cell = cellByLoc[col+','+r2];
      if (cell) cell.click();
    });
  });
  bar.querySelector('.ib-clear').addEventListener('click', e => {
    e.stopPropagation();
    applyIssues([]);
  });
}

// Extract issues from a /grid response and render the bar.
function _updateIssueBar(res){
  if (res.what && res.what !== 'dma'){ applyIssues([]); return; }
  const cells = res.cells || {};
  const flowLookup = _buildFlowLookup();
  const issues = [];
  Object.entries(cells).forEach(([key, cell]) => {
    if (!cell || !cell.channels) return;
    const [col, row] = key.split(',').map(Number);
    Object.entries(cell.channels).forEach(([chan, ch]) => {
      if (ch.state !== 'stalled' && ch.state !== 'error') return;
      issues.push({
        key, col, row, chan,
        state: ch.state,
        stalls: ch.stalls || [],
        errors: ch.errors || [],
        flow_index: flowLookup[col+','+row+','+chan] != null
                    ? flowLookup[col+','+row+','+chan] : null,
      });
    });
  });
  applyIssues(issues);
}

function _swDisplayRes(res){
  const cells = {};
  Object.entries(res.cells || {}).forEach(([k, c]) => {
    if (!c) return;
    if (c.state === 'unreachable'){ cells[k] = c; return; }
    if (ROUTING_DIFF){
      const parts = k.split(',').map(Number);
      const d = _routingSrcDiffTile(parts[0], parts[1]);
      cells[k] = d ? Object.assign({}, c, d) : c;
      return;
    }
    const plain = Object.assign({}, c);
    delete plain.state;
    cells[k] = plain;
  });
  return Object.assign({}, res, { cells });
}
function reapplySwitchScan(){
  if (!SWSCAN_RES) return;
  const again = Object.assign({}, SWSCAN_RES);
  delete again.llm_summary;
  applyGrid(again);
}

function applyGrid(res){
  if (res.llm_summary){
    llmPushCtx('[context] '+res.llm_summary, 'scan');
  }
  const isSwitch = res.what === 'switch' && !res.error;
  if (isSwitch) SWSCAN_RES = res;
  const disp = isSwitch ? _swDisplayRes(res) : res;
  if (res.error){ setStatus('live: '+res.error); }
  else if (res.what === 'switch'){
    SWSCAN = res.cells || {};
    if (ROUTING_DIFF){
      const n = Object.values(disp.cells || {})
        .filter(c => c && c.state === 'mismatch').length;
      setStatus(n
        ? ('switch: '+(n===1?'1 tile differs':n+' tiles differ')
           +' between static and dynamic')
        : 'switch: static and dynamic agree on every tile');
    } else {
      const n = Object.keys(disp.cells || {}).length;
      setStatus('switch: '+n+' tile'+(n===1?'':'s')+' read · showing '+ROUTING_SRC
                +' routing — tick diff to compare the two maps');
    }
    if (res.dynamic){ DYNAMIC = res.dynamic; syncRoutingSource(); }
    if (panelActiveKey) panelRenderBody(panelActiveKey);
  }
  else setStatus('live '+LIVE.what+' @ '+new Date().toLocaleTimeString());
  // One fetch feeds both views; the device map paints from the same payload.
  dmApplyStatus(disp);
  const cells = disp.cells || {};
  Object.keys(liveBar).forEach(k => {
    const b = liveBar[k], c = cells[k];
    const cell = cellByLoc[k];
    // Only badge tiles with actionable states — idle/completed channels
    // finished cleanly and should not display a coloured bar.
    if (!c || !c.state || c.state === 'idle' || c.state === 'completed'){
      b.className='livebar hide';
      // Still surface last-BD info on the tile hover for completed channels.
      if(c && c.channels && cell){
        if(!cell._baseTitle) cell._baseTitle = cell.title;
        const bdLines = Object.entries(c.channels).map(([cn, ch]) => {
          let s = cn + ': ' + (ch.state || '?');
          if(ch.cur_bd != null) s += '  last bd=' + ch.cur_bd;
          return s;
        });
        if(bdLines.length) cell.title = cell._baseTitle + '\n── dma (completed) ──\n' + bdLines.join('\n');
      }
      return;
    }
    const m = LSTATE[c.state] || LSTATE.unknown;
    b.className = 'livebar';
    b.style.background = m[0];
    b.textContent = m[1];
    b.title = JSON.stringify(c, null, 2);
  });
  _updateIssueBar(res);
}
// userInitiated: a click (mode pill, Scan button, live checkbox) rather than the
// 2s timer. Only user scans queue a retry when the guard rejects them — letting
// skipped auto-polls retry would busy-loop the board back-to-back with no gap.
// Scan progress indicator, shared by both views.
let SCAN_SPIN_T = null;
function setScanBusy(on){
  clearTimeout(SCAN_SPIN_T); SCAN_SPIN_T = null;
  const els = ['gridSpin','dmSpin']
    .map(id => document.getElementById(id)).filter(Boolean);
  if (!on){ els.forEach(e => e.hidden = true); return; }
  // Delayed reveal: a DMA poll usually lands well inside this, and a spinner
  // that blinks every 2s reads as a fault rather than as progress. A switch
  // scan (one board round-trip per tile) is always slower than the delay.
  SCAN_SPIN_T = setTimeout(() => els.forEach(e => e.hidden = false), 250);
}

function scanOnce(userInitiated){
  // One-shot scan, always runs (user-triggered). Skips the runActive guard so
  // the user can read core/DMA state even while a run is parked on the board —
  // the same window where 'scan cores' in the aiegdb console works fine.
  if (deviceSel && deviceSel.value === 'simulator' && !simHasLiveReads()){
    showSimLiveUnavailable();
    return;
  }
  if (LIVE.gridBusy){ if (userInitiated) LIVE.rescan = true; return; }
  LIVE.gridBusy = true;
  LIVE.rescan = false;
  setScanBusy(true);
  // Pin the mode for this request. A scan can easily outlive the 2s poll, so by
  // the time it lands the user may have picked a different one.
  const what = LIVE.what;
  setStatus('scanning '+what+'…');
  const dev = deviceSel ? deviceSel.value : '';
  const host = boardHost ? boardHost.value.trim() : '';
  const qs = '/grid?what='+what +
             '&device='+encodeURIComponent(dev) +
             '&host='+encodeURIComponent(host);
  api(qs).then(res => {
      // Drop a response for a mode the user has already moved off of, instead of
      // repainting the old mode's data over the new selection.
      if (what !== LIVE.what) return;
      applyGrid(res);
    })
    .catch(() => setStatus('daemon offline (static mode)'))
    .finally(() => {
      LIVE.gridBusy = false;
      // A swallowed click always retries. A mode changed mid-scan only retries
      // under live — with the poll off, picking a mode must not read the board.
      const chained = LIVE.rescan || (LIVE.enabled && what !== LIVE.what);
      if (chained){
        // Leave the spinner up across the handoff; hiding it here would blink
        // it off and on for a scan that never actually stopped.
        LIVE.rescan = false; scanOnce(true);
      } else {
        setScanBusy(false);
      }
    });
}
function pollGridOnce(){
  // Automatic poll: skip while a board run holds JTAG exclusively during
  // device program / reset / download. The user can still trigger scanOnce()
  // manually once the board is past that phase (app running / parked).
  if (LIVE.runActive) return;
  scanOnce();
}
function startGridPoll(){
  // Clear first: setLive(true) can be reached from either checkbox, and
  // overwriting gridTimer without clearing orphans the old interval forever.
  stopGridPoll();
  scanOnce(true);
  LIVE.gridTimer = setInterval(pollGridOnce, 2000);
}
function stopGridPoll(){ if(LIVE.gridTimer){ clearInterval(LIVE.gridTimer); LIVE.gridTimer=null; } }
function hideConsole(){
  CON.scope = 'partition';
  const box = document.getElementById('cmdconsole');
  if (box) box.classList.add('hide');
  const rsp = document.getElementById('rhsplitter');
  if (rsp) rsp.classList.add('hide');
  const tgt = document.getElementById('contarget');
  if (tgt) tgt.textContent = 'partition';
}
function setLive(on){
  if (on && deviceSel && deviceSel.value === 'simulator' && !simHasLiveReads()){
    showSimLiveUnavailable();
    on = false;
  }
  // The switch is static configuration, not per-cycle state: polling it every
  // 2s would spend a board round-trip per tile to re-read bits that only the
  // host program changes. Scan it on demand instead.
  if (on && LIVE.what === 'switch'){
    on = false;
  }
  LIVE.enabled = on;
  // Both views expose the same poll as a checkbox; keep them mirrored so the
  // one that did not initiate the change does not lie about the poll state.
  const cb = document.getElementById('liveToggle'); if(cb) cb.checked = on;
  const dmcb = document.getElementById('dmLiveToggle'); if(dmcb) dmcb.checked = on;
  if (on) startGridPoll();
  else { stopGridPoll(); clearBars(); setStatus(''); dmClearStatus(); }
  // Console visibility is tied to LIVE.connected (the connection test), NOT to
  // the overlay toggle, so it persists while the overlay is switched off.
}
// Disable ALL aiedbg-driven features while a board run is live so their JTAG
// reads don't collide with apppaltest's device program/reset/dow -force over
// the single serialized xsdb://<PALIP>:3121 link (intermittent download fail).
// Auto-unlock the aiegdb console mid-run (called from pollLog when the daemon
// reports r.debuggable). Enables ONLY the console input + reload — the live
// overlay checkbox / grid-poll stay off so the only JTAG traffic is the
// commands the user types, which won't collide with the parked run's reads.
function unlockConsoleForDebug(){
  if (LIVE.debugUnlocked) return;   // idempotent: only flip once per run
  LIVE.debugUnlocked = true;
  const cr = document.getElementById('conreload');
  const ci = document.getElementById('conin');
  if (cr) cr.disabled = false;
  if (ci){ ci.disabled = false; ci.classList.remove('disabled'); }
  // The run has parked (compute done, applog flushed). Timeline rendering is no
  // longer tied to this park event — it's driven by the TIMESYNC-idle timer in
  // pollLog, so a run with no TIMESYNC output never renders.
  if (LIVE.runOwned && LIVE.daemonRun) renderRunBanner();
  else setRunStatus('run parked \u2014 aiegdb console unlocked for live debug '
                  + '(overlay stays off)');
}
function setDebugEnabled(on){
  const cb = document.getElementById('liveToggle');
  const tc = document.getElementById('testconn');
  const cr = document.getElementById('conreload');
  const ci = document.getElementById('conin');
  if (!on){
    LIVE.runActive = true;
    LIVE.debugUnlocked = false;   // new run → re-lock until it's debuggable again
    clearBars();
    if (cr) cr.disabled = true;
    if (ci){ ci.disabled = true; ci.classList.add('disabled'); }
    setRunStatus('');
  } else {
    LIVE.runActive = false;
    if (cr) cr.disabled = false;
    if (ci){ ci.disabled = false; ci.classList.remove('disabled'); }
    // Re-enable the overlay toggle only if the connection is still valid
    // (mirrors the testConnect gating).
    if (cb){ cb.disabled = !LIVE.connected;
      cb.closest('label').classList.toggle('disabled', !LIVE.connected); }
    updateRunButtons();
    setRunStatus('');
  }
}

// Device selection gates the live controls (item #4/#5). A device must be
// chosen before the "Live status overlay" checkbox and "Run" button work;
// picking vek385 also reveals the board-hostname text box.
const deviceSel = document.getElementById('deviceSel');
const boardHost = document.getElementById('boardHost');
const liveToggle = document.getElementById('liveToggle');
const runbtn = document.getElementById('runbtn');
const stopbtn = document.getElementById('stopbtn');
const testconn = document.getElementById('testconn');
const attachbtn = document.getElementById('attachbtn');
function setConnStatus(msg){
  const e = document.getElementById('connstatus');
  if (!e) return;
  const text = msg || 'Not connected';
  let state = 'idle';
  if (/^(connected|attached|simulator activated)/i.test(text)) state = 'connected';
  else if (/(testing|attaching|starting|setting target|in progress)/i.test(text)) state = 'busy';
  else if (/(failed|failure|error|offline|no simulator|cannot|enter the|select a)/i.test(text)) state = 'error';
  e.textContent = text;
  e.title = text;
  e.dataset.state = state;
}
function updateConnectionPresentation(){
  const ctrl = document.getElementById('ctrlbar');
  if (ctrl) ctrl.dataset.connected = LIVE.connected ? 'true' : 'false';
  if (!testconn) return;
  const sim = deviceSel && deviceSel.value === 'simulator';
  testconn.textContent = LIVE.connected
    ? (sim ? 'Reactivate' : 'Reconnect')
    : (sim ? 'Activate' : 'Connect');
}
// Show/hide the "start hw_server on the target board" hint (shown on failure).
function setConnHint(show){ const e=document.getElementById('connhint'); if(e) e.classList.toggle('hide', !show); }

function connectTcpHost(dev, host){
  if (dev === 'pal') return LIVE.jtagHost || null;
  return host || null;
}
function boardConnectGuideText(dev, host){
  const tcp = connectTcpHost(dev, host);
  const lines = [
    '[board setup — run on the target in xsdb, then press Connect here]',
    '',
    'On the target board, in xsdb, start hw_server:',
    '',
    '    ' + HW_SERVER_CMD,
    '',
  ];
  if (tcp){
    lines.push('This UI will probe from here (same as Connect):');
    lines.push('');
    lines.push('    connect -url TCP:' + tcp + ':3121');
    lines.push('');
  } else if (dev !== 'pal'){
    lines.push('Enter the board hostname above first.');
    lines.push('');
  }
  lines.push('Then press Connect.');
  return lines.join('\n');
}
function showBoardConnectGuide(dev, host){
  if (LIVE.simOnly || !dev || dev === 'simulator' || LIVE.connected) return hideRunConsole();
  if (LIVE.conTimer || LIVE.hwsrvTimer || SIM.timer) return;
  const con = document.getElementById('console');
  if (!con) return;
  con.textContent = boardConnectGuideText(dev, host);
  con.classList.remove('hide');
}
function hideRunConsole(){
  if (LIVE.conTimer || LIVE.hwsrvTimer || SIM.timer) return;
  const con = document.getElementById('console');
  if (!con) return;
  con.classList.add('hide');
  if (!con.textContent.startsWith('[run ') && !con.textContent.startsWith('[adopted ')
      && !con.textContent.startsWith('[auto-starting'))
    con.textContent = '';
}

// ── run-state reconciliation (UI ⇄ daemon) ───────────────────────────────────
function updateRunButtons(){
  const dev = deviceSel ? deviceSel.value : '';
  if (dev === 'simulator') return;    // the /sim/* handlers own those buttons
  if (runbtn)  runbtn.disabled  = !LIVE.connected || LIVE.runOwned;
  if (stopbtn) stopbtn.disabled = !LIVE.runOwned;
}
function fmtAge(s){
  if (s == null) return '';
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm' + (s%60 ? ' ' + (s%60) + 's' : '');
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}
// One writer for #runstatus: the 5s heartbeat and the 1s tail both know part of
// the story and would otherwise flip the line between them.
function renderRunBanner(){
  const rs = LIVE.daemonRun;
  if (!LIVE.runOwned || !rs){ setRunStatus(''); return; }
  let s = 'run #' + rs.run_id + (rs.device ? ' on ' + rs.device : '')
        + (rs.started_iso ? ' started ' + rs.started_iso : '')
        + (rs.age_s != null ? ' (' + fmtAge(rs.age_s) + ' ago)' : '')
        + ', ' + (rs.status || 'running');
  if (rs.stale) s += '; no new output for a while (press "Stop run" to release the board)';
  if (LIVE.debugUnlocked) s += '; aiegdb console unlocked for live debug (overlay stays off)';
  setRunStatus(s);
}
// The simulator half of run-state reconciliation. The page learned a sim
// existed only from the /sim/log tail it started itself, so a reload — or a sim
// started in another tab — left Run lit and Stop grey while the daemon refused
// to switch apps. Edge-triggered on `running`, exactly like applyRunState, so
// it never fights a tail that is already up.
function applySimState(ss){
  if (!ss) return;
  if (ss.kind) SIM.kind = ss.kind;
  if (ss.available === false && simRow()) simRow().available = false;
  const was = SIM.owned;
  SIM.owned = !!ss.running;
  if (ss.running && !was){
    // Adopt it: point the dropdown at the simulator so the /sim/* handlers own
    // the buttons, and start the tail from the top of the log.
    if (deviceSel && deviceSel.value !== 'simulator'){
      deviceSel.value = 'simulator';
      LIVE.device = 'simulator'; LIVE.host = '';
      if (testconn) testconn.textContent = 'Activate';
      if (boardHost) boardHost.classList.add('hide');
      if (attachbtn) attachbtn.disabled = true;
    }
    SIM.logoff = 0; SIM.applogoff = 0; SIM.applogSeen = false;
    const con = document.getElementById('console');
    if (con){ con.classList.remove('hide');
      con.textContent = '[adopted running simulator #' + ss.run_id
        + (ss.kind ? ' (' + ss.kind + ')' : '') + ' → '
        + (ss.sim_log || '') + ']\n'; }
    if (SIM.timer) clearInterval(SIM.timer);
    SIM.timer = setInterval(pollSimLog, 1000);
    pollSimLog();
  }
  if (deviceSel && deviceSel.value === 'simulator'){
    if (runbtn) runbtn.disabled = SIM.owned;
    if (stopbtn) stopbtn.disabled = !SIM.owned;
  }
  if (!ss.running && was && SIM.timer){
    clearInterval(SIM.timer); SIM.timer = null;
  }
}
// Edge-triggered on `running` so it can't fight the tail pollLog already drives.
function applyRunState(rs){
  if (!rs) return;
  applySimState(rs.sim);
  const was = LIVE.runOwned;
  LIVE.runOwned = !!rs.running;
  LIVE.daemonRun = rs;
  if (rs.running){
    if (!was){
      setDebugEnabled(false);
      if (LIVE.conTimer) clearInterval(LIVE.conTimer);
      LIVE.logoff = 0;
      const con = document.getElementById('console');
      if (con){ con.classList.remove('hide'); con.textContent = ''; }
      LIVE.conTimer = setInterval(pollLog, 1000);
      pollLog();
    }
    renderRunBanner();
  } else if (was){
    if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer = null; }
    setDebugEnabled(true);          // clears LIVE.runActive + the run banner
  }
  updateRunButtons();
}
// Runs unconditionally, not just while a tail is up: the point is to notice runs
// this page is not tracking. No JTAG, so it is cheap.
function syncRunState(){
  if (LIVE.rsBusy) return Promise.resolve(null);
  LIVE.rsBusy = true;
  return api('/runstate')
    .then(rs => { if (rs && !rs.error) applyRunState(rs); return rs; })
    .catch(() => null)
    .finally(() => { LIVE.rsBusy = false; });
}
// One writer for the device row's status line and the Connect/Activate gate.
// Called from updateDeviceUI and again when /devices lands, because the rows
// arrive asynchronously and the selection is usually already made by then.
function refreshDeviceStatus(){
  const dev = deviceSel ? deviceSel.value : '';
  const host = boardHost ? boardHost.value.trim() : '';
  if (!dev){ if (testconn) testconn.disabled = true; setConnStatus('Not connected'); hideRunConsole(); return; }
  if (dev !== 'simulator'){
    if (testconn) testconn.disabled = false;
    setConnStatus('click "Connect" to enable live features');
    if (!LIVE.connected) showBoardConnectGuide(dev, host);
    return;
  }
  hideRunConsole();
  const sr = simRow();
  if (sr && sr.available === false){
    // Offered, but nothing to run. Naming the missing artifact here is the
    // whole point of keeping the option visible — otherwise the only way to
    // learn why is to press Run and read an error.
    if (testconn) testconn.disabled = true;
    if (runbtn) runbtn.disabled = true;
    setConnStatus('no simulator for this app: ' + (sr.reason || 'not built'));
    return;
  }
  if (testconn) testconn.disabled = false;
  // A sim started before this selection keeps owning the buttons.
  if (SIM.owned){
    if (runbtn) runbtn.disabled = true;
    if (stopbtn) stopbtn.disabled = false;
  }
  setConnStatus((LIVE.simOnly ? 'simulator \u2014 press "Run"'
                              : 'click "Activate" to use the simulator')
    + ((sr && sr.note) ? ': ' + sr.note : ''));
}
// Selecting a device only enables the "Connect" button. The live overlay
// checkbox + the drill-down console stay locked until a connection test passes
// (LIVE.connected). Changing the device invalidates any prior test.
function updateDeviceUI(){
  const dev = deviceSel ? deviceSel.value : '';
  const has = !!dev;
  LIVE.connected = false;
  updateConnectionPresentation();
  setLive(false);                              // uncheck overlay + stop poll
  hideConsole();                               // connection invalidated
  if (liveToggle){ liveToggle.disabled = true;
    liveToggle.closest('label').classList.toggle('disabled', true); }
  // Also the simulator's reset: updateRunButtons leaves /sim/*'s buttons alone.
  if (runbtn) runbtn.disabled = true;
  if (stopbtn) stopbtn.disabled = true;
  updateRunButtons();
  // Attaching is only meaningful for a real board: the simulator has no run to
  // adopt — it is started by this UI or not at all.
  if (attachbtn) attachbtn.disabled = !has || dev === 'simulator';
  if (boardHost) boardHost.classList.toggle('hide', !has || dev === 'pal' || dev === 'simulator');
  updateLiveReadControls();
  refreshDeviceStatus();
  setConnHint(false);
  if(deviceSel&&has){
    const opt=deviceSel.options[deviceSel.selectedIndex];
    llmPushCtx('[context] Device selected: '+(opt?opt.text:dev), 'session');
  }
}
if (deviceSel) deviceSel.onchange = updateDeviceUI;
// Editing the hostname invalidates any prior connection test — reset to the
// "click Connect" state so stale LIVE.connected doesn't let a Run start
// against a host the user just changed.
if (boardHost) boardHost.oninput = updateDeviceUI;
function applyConnected(r){
  LIVE.connected = true;
  updateConnectionPresentation();
  updateRunButtons();                     // unlock Run now
  if (liveToggle){ liveToggle.disabled = false;
    liveToggle.closest('label').classList.remove('disabled'); }
  updateLiveReadControls();
  hideRunConsole();
  const box = document.getElementById('cmdconsole');
  if (box) box.classList.remove('hide');   // reveal the aiegdb console
  const rsp = document.getElementById('rhsplitter');
  if (rsp) rsp.classList.remove('hide');
  setConnHint(false);
  // Only now that the JTAG connect succeeded do we tell the daemon to switch the
  // live target to this host and respawn the aiegdb console against it. Without
  // this the console stays pinned to the daemon's startup target and every
  // read times out even though the host was reachable.
  setConnStatus('connected \u2014 ' + ((r && r.detail) || 'ok') + '; setting target\u2026');
  api('/settarget', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({device:LIVE.device, host:LIVE.host})})
    .then(sr => {
      if (sr && sr.ok){
        llmAdoptGeneration(sr);
        setConnStatus('connected \u2014 ' + (sr.target || ((r && r.detail) || 'ok')));
        llmPushCtx('[context] Connected to '+(LIVE.host||LIVE.device)
          +' \u2014 AIEDBG_TARGET: '+(sr.target||'unknown'), 'session');
        conSend('', false);   // spawn aiegdb with the new target; shows scope
      } else {
        setConnStatus('target switch failed: ' + ((sr && sr.detail) || 'unknown'));
        conSend('', false);   // still spawn so the console is usable
      }
    })
    .catch(() => {   // daemon offline / static mode: fall back to a plain spawn
      setConnStatus('connected \u2014 ' + ((r && r.detail) || 'ok'));
      conSend('', false);
    });
}
function markDisconnected(){
  LIVE.connected = false;
  updateConnectionPresentation();
  updateRunButtons();                      // keep Run gray
  if (liveToggle) liveToggle.disabled = true;
  hideConsole();
}
const LOG_FOLLOW_FRAC = 0.5, LOG_FOLLOW_MIN = 160;
function logFollowing(el){
  if (!el) return true;
  const slack = Math.max(LOG_FOLLOW_MIN, el.clientHeight * LOG_FOLLOW_FRAC);
  return (el.scrollHeight - el.scrollTop - el.clientHeight) <= slack;
}
function logFollow(el, was){ if (el && was) el.scrollTop = el.scrollHeight; }
// Test the daemon + JTAG target; on success unlock the overlay + reveal console.
function testConnect(){
  const dev = deviceSel ? deviceSel.value : '';
  if (!dev){ setConnStatus('select a device first'); return; }
  // Simulator: probe IPC readiness. If already ready, apply connected immediately.
  // Otherwise, unlock Run/Stop so the user can start the sim and the aiegdb
  // console will auto-connect once the IPC debug socket appears.
  if (dev === 'simulator'){
    const sr0 = simRow();
    if (sr0 && sr0.available === false){
      setConnStatus('no simulator for this app — ' + (sr0.reason || 'not built'));
      return;
    }
    LIVE.device = dev; LIVE.host = '';
    api('/sim/status').then(ss => {
      if (ss && ss.kind) SIM.kind = ss.kind;
      if (ss && ss.available === false){
        setConnStatus('no simulator for this app: ' + (ss.reason || 'not built'));
        return;
      }
      if (ss && ss.ipc_ready){
        applyConnected({detail: 'simulator IPC ready'});
        return;
      }
      // Not ready (or never will be) — still unlock Run and reveal the console.
      LIVE.connected = true;
      updateConnectionPresentation();
      if (runbtn) runbtn.disabled = !!SIM.owned;
      if (stopbtn) stopbtn.disabled = !SIM.owned;
      const box = document.getElementById('cmdconsole');
      if (box) box.classList.remove('hide');
      const rsp = document.getElementById('rhsplitter');
      if (rsp) rsp.classList.remove('hide');
      if (simHasLiveReads()){
        if (liveToggle){ liveToggle.disabled = false;
          liveToggle.closest('label').classList.remove('disabled'); }
        setConnStatus(LIVE.simOnly
          ? 'simulator ready'
          : 'simulator activated; run it to enable live grid reads');
      } else {
        // aiesim exposes no debug socket, so the overlay stays locked. Saying
        // "run it to enable live grid reads" here promised something the
        // backend can never deliver, and left the user waiting for it.
        if (liveToggle){ liveToggle.disabled = true;
          liveToggle.closest('label').classList.add('disabled'); }
        updateLiveReadControls();
        setConnStatus(LIVE.simOnly
          ? 'simulator ready \u2014 no live register access'
          : 'aiesim activated \u2014 no live register access');
        conAppend('aiesim has no debug socket: DMA/Cores/Events scans and '
          +'aiedbg register commands are unavailable. aiegdb navigation and '
          +'help commands still work.');
      }
    }).catch(() => {
      LIVE.connected = true;
      updateConnectionPresentation();
      if (runbtn) runbtn.disabled = false;
      if (liveToggle){ liveToggle.disabled = false;
        liveToggle.closest('label').classList.remove('disabled'); }
      setConnStatus('simulator activated (daemon offline)');
    });
    return;
  }
  const host = boardHost ? boardHost.value.trim() : '';
  if (dev !== 'pal' && !host){ setConnStatus('enter the ' + dev + ' board hostname'); return; }
  // Remember the selection so applyConnected can tell the daemon which target
  // to switch the aiegdb console to.
  LIVE.device = dev; LIVE.host = host;
  setConnStatus('testing\u2026');
  const qs = '?device='+encodeURIComponent(dev)+'&host='+encodeURIComponent(host);
  api('/ping'+qs).then(r => {
    if (r && r.ok){
      applyConnected(r);
    } else if (r && r.busy){
      // A live run holds the link; the link is not dead. Auto-starting
      // hw_server would reset a working service and refuse again anyway.
      applyRunState(r.run);
      setConnStatus('a board run is still in progress; press "Stop run" then Connect again');
      setConnHint(false);
    } else {
      // Daemon answered but the JTAG connect failed. Auto-starting hw_server
      // on the board over ssh proved unstable (intermittent failures), so we
      // no longer attempt it; guide the user to start it manually instead.
      markDisconnected();
      const con = document.getElementById('console');
      if (con){ con.classList.remove('hide'); con.textContent =
        '[connection failed: ' + ((r && r.detail) || 'no response') + ']\n'
        + 'Connection failed. On the target hw board, start hw_server via xsdb:\n'
        + '    exec hw_server -stcp:0.0.0.0:3121\n'; }
      setConnStatus('connection failed \u2014 start hw_server on the target board: '
        + 'exec hw_server -stcp:0.0.0.0:3121');
      setConnHint(true);   // show the persistent manual-start guidance banner
    }
  }).catch(() => {
    // Daemon itself is unreachable (static mode) → can't ssh from a dead
    // daemon, so don't attempt auto-launch; just guide the user.
    markDisconnected();
    setConnStatus('daemon offline (static mode)');
    setConnHint(true);   // guide user to start hw_server on the target board
  });
}
if (testconn) testconn.onclick = testConnect;

function attachSession(){
  const dev  = deviceSel ? deviceSel.value : '';
  const host = boardHost ? boardHost.value.trim() : '';
  if (!dev) return;
  LIVE.device = dev; LIVE.host = host;
  setConnStatus('attaching…');
  api('/attach', {method:'POST', headers:{'Content-Type':'application/json'},
                  body: JSON.stringify({device:dev, board_host:host})})
    .then(r => {
      if (r && r.ok){
        applyConnected(r);
        setConnStatus('attached to existing session; board state predates this UI');
        llmPushCtx('[context] User attached to a board session started outside the UI; '
                 + 'prior board history is unknown to the daemon.', 'session');
      } else if (r && r.busy){
        applyRunState(r.run);
        setConnStatus('this UI is already running a test; press "Stop run" first');
        setConnHint(false);
      } else {
        markDisconnected();
        setConnStatus('attach failed: ' + ((r && r.detail) || 'no response'));
        setConnHint(true);
      }
    })
    .catch(() => { markDisconnected();
                   setConnStatus('daemon offline (static mode)'); });
}
if (attachbtn) attachbtn.onclick = attachSession;

document.getElementById('liveToggle').onchange = e => setLive(e.target.checked);

// LIVE.what is shared by the grid overlay and the device-map scanner — there is
// one /grid fetch and one selection, so both pill strips mirror each other.
function setOverlayWhat(w){
  const changed = LIVE.what !== w;
  LIVE.what = w;
  // Both views expose the same selection; keep them mirrored so the one that
  // did not initiate the change does not name a mode it is not showing.
  ['overlayWhat','dmScanWhat'].forEach(id => {
    const s = document.getElementById(id);
    if (s && s.value !== w) s.value = w;
  });
  if (!changed) return;
  // Drop the previous mode's colors immediately: leaving DMA tints on screen
  // under a "Cores" selection reads as live data for a mode never read.
  dmClearStatus();
  // Same reasoning for the per-row switch verdicts: they are only true for the
  // scan that produced them.
  SWSCAN = null;
  SWSCAN_RES = null;
  if (panelActiveKey) panelRenderBody(panelActiveKey);
  // Static config: never let the 2s poll carry over into this mode.
  if (w === 'switch' && LIVE.enabled) setLive(false);
  if (LIVE.enabled){
    const msg = 'scanning '+w+'…';
    dmSetScanStatus(msg);
    setStatus(msg);
  } else {
    dmSetScanStatus('');
    setStatus('');
  }
}
// Picking a mode is a SELECTION, not an action. It reads the board only while
// the live poll is on; otherwise "Scan" is the trigger.
function pickOverlayWhat(w, setMsg){
  setOverlayWhat(w);
  if (!LIVE.connected) return;
  if (LIVE.enabled) scanOnce(true);
}
(function(){
  const s = document.getElementById('overlayWhat');
  if (s) s.onchange = () => pickOverlayWhat(s.value, setStatus);
})();

// ── Device-map scan controls ──────────────────────────────────
// Deliberately routed through the same scanOnce()/setLive() pair as the grid
// overlay rather than a parallel fetch path: LIVE.gridBusy is a single-flight
// guard and LIVE.gridTimer a single handle, so a second poller here would
// fight the first for the one aiedbg subprocess the daemon runs per scan.
(function(){
  const s = document.getElementById('dmScanWhat');
  if (s) s.onchange = () => pickOverlayWhat(s.value, dmSetScanStatus);
})();
function runScanNow(setMsg){
  if (deviceSel && deviceSel.value === 'simulator' && !simHasLiveReads()){
    updateLiveReadControls();
    return;
  }
  if (!LIVE.connected) return;
  setMsg('scanning '+LIVE.what+'…');
  scanOnce(true);
}
(function(){
  const btn = document.getElementById('dmScanBtn');
  if (btn) btn.onclick = () => runScanNow(dmSetScanStatus);
  const gbtn = document.getElementById('gridScanBtn');
  if (gbtn) gbtn.onclick = () => runScanNow(setStatus);
  const clr = document.getElementById('dmClearBtn');
  if (clr) clr.onclick = dmClearAll;
  const live = document.getElementById('dmLiveToggle');
  if (live) live.onchange = e => {
    if (e.target.checked && !LIVE.connected){
      e.target.checked = false;
      return;
    }
    setLive(e.target.checked);
  };
  const swToggle = document.getElementById('dmSwToggle');
  if (swToggle) swToggle.onchange = e => {
    dmShowSW = e.target.checked;
    buildDeviceMap();
  };
})();

// Keep the console DOM bounded so a long AIE trace run (multi-MB of
// STREAM_STALL/DMA_FINISH lines) doesn't jank the tab with an ever-growing
// textContent. Trim from the front, keeping the most recent MAX lines.
function capConsole(con){
  const MAX = 5000;
  const txt = con.textContent;
  if (txt.length < 400000) return;   // cheap: only split when it's large enough
  const nl = txt.split('\n');
  if (nl.length > MAX) con.textContent = nl.slice(nl.length - MAX).join('\n');
}

function pollLog(){
  // Guard against overlapping tails so a slow response can't pile up. A stuck
  // fetch can no longer wedge the tail forever: an AbortController aborts it
  // after a timeout, so `.finally()` always runs and clears logBusy — the next
  // 1s tick then retries instead of early-returning forever.
  if (LIVE.logBusy) return;
  LIVE.logBusy = true;
  const ctrl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
  const abTimer = ctrl ? setTimeout(() => ctrl.abort(), 8000) : null;
  api('/applog?offset='+LIVE.logoff, ctrl ? {signal: ctrl.signal} : undefined).then(r => {
    const con = document.getElementById('console');
    const follow = logFollowing(con);
    if (r.data){ con.textContent += r.data; capConsole(con); }
    // TIMESYNC-idle trigger: the first [TIMESYNC] line arms a 5s "quiet" window;
    // each new TIMESYNC-bearing chunk re-arms it. When a full 5 seconds pass with
    // no further [TIMESYNC] output the block is complete → render the timeline.
    if (r.data && /\[TIMESYNC\]/.test(r.data)){
      LIVE.tsSeen = true;
      if (LIVE.tsIdleTimer) clearTimeout(LIVE.tsIdleTimer);
      LIVE.tsIdleTimer = setTimeout(() => {
        LIVE.tsIdleTimer = null;
        triggerTimeline(true);   // gated by tsSeen; dedups via tlDone
      }, 5000);
    }
    if (r.next != null) LIVE.logoff = r.next;
    logFollow(con, follow);
    if (LIVE.runOwned && LIVE.daemonRun){
      LIVE.daemonRun.status = r.status;
      LIVE.daemonRun.stale = (r.status === 'hang');
      renderRunBanner();
    } else {
      setRunStatus('run: ' + r.status);
    }
    // Auto-unlock the aiegdb console mid-run once the daemon reports the run is
    // past the exclusive-JTAG setup phase (download/program/reset done → app
    // running/parked/hung). Only typed commands hit JTAG; the live overlay
    // grid-poll stays OFF so we don't compete with the run's own reads.
    if (r.running === true && r.debuggable) unlockConsoleForDebug();
    // Timeline rendering is driven by the TIMESYNC-idle timer above, not by the
    // derived pass/fail status — a run with no TIMESYNC output never renders.
    // Stop only when the subprocess has actually exited — NOT on a derived
    // pass/fail. apppaltest keeps writing (summary/cleanup/reboot) long after
    // the teardown marker first appears mid-run.
    if (r.running === false){
      if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer=null; }
      // Drop any pending idle timer so it can't fire into the next run.
      if (LIVE.tsIdleTimer){ clearTimeout(LIVE.tsIdleTimer); LIVE.tsIdleTimer=null; }
      // Run ended (force-stop OR natural completion) → re-enable aiedbg features.
      LIVE.runOwned = false;
      setDebugEnabled(true);
      setRunStatus('');
      triggerTimeline(true);   // safety net; gated by tsSeen, dedups via tlDone.
    }
  }).catch(() => {
    // Transient tail failure: a slow/interrupted/aborted poll or a brief daemon
    // hiccup over a remote link. While the run is still ours, DO NOT tear down
    // the tail — one bad poll must not permanently freeze the console. Note it
    // and let the next 1s tick retry; the tail is only stopped on a confirmed
    // `running === false` response in the then-branch above.
    if (LIVE.runOwned){
      setRunStatus('run: tail retrying…');
    } else {
      // Not a live run we own (adopted run already ended, or never running) →
      // the daemon is gone and nothing holds JTAG, so unblock aiedbg features.
      if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer = null; }
      setDebugEnabled(true);
    }
  }).finally(() => { if (abTimer) clearTimeout(abTimer); LIVE.logBusy = false; });
}
document.getElementById('runbtn').onclick = () => {
  const dev = deviceSel ? deviceSel.value : '';
  if (!dev){ setStatus('select a device first'); return; }
  const con = document.getElementById('console');
  con.classList.remove('hide'); con.textContent = '';
  // New run → drop the previous run's timeline and lock the Profile tab; it
  // re-populates when this run ends and the server re-renders timeline.png.
  clearTimelineProfile();
  // Simulator path: route to /sim/run and tail /sim/log.
  if (dev === 'simulator'){
    SIM.logoff = 0; SIM.applogoff = 0; SIM.applogSeen = false;
    if (runbtn) runbtn.disabled = true;
    if (stopbtn) stopbtn.disabled = false;
    api('/sim/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
      .then(r => {
        if (r.error){ con.textContent = 'sim error: ' + r.error + '\n';
          if (runbtn) runbtn.disabled = false; if (stopbtn) stopbtn.disabled = true;
          setConnStatus(r.reason ? ('no simulator for this app \u2014 ' + r.reason)
                                 : 'simulator did not start');
          return; }
        if (r.sim_kind) SIM.kind = r.sim_kind;
        SIM.owned = true;
        con.textContent = '[simulator (' + (r.sim_kind||'?') + ') started \u2192 '
          + (r.sim_log||'') + ']\n'
          // aiesim never opens a debug socket, so announcing a wait for one
          // described a different backend's flow and read as a hang.
          + (simHasLiveReads() ? '[waiting for IPC debug socket\u2026]\n'
                               : '[console output only \u2014 this backend has no '
                                 + 'debug socket]\n')
          + (r.engine_log ? '[simulator engine log: ' + r.engine_log + ']\n' : '');
        llmPushCtx('[context] Simulator run started (' + (r.sim_kind||'?') + ')', 'run');
        SIM.ipcReady = false;
        if (SIM.timer) clearInterval(SIM.timer);
        SIM.timer = setInterval(pollSimLog, 1000);
        pollSimLog();
      })
      .catch(() => { con.textContent = 'daemon offline: cannot start simulator.';
        if (runbtn) runbtn.disabled = false; if (stopbtn) stopbtn.disabled = true; });
    return;
  }
  const host = boardHost ? boardHost.value.trim() : '';
  if (dev !== 'pal' && !host){ setStatus('enter the ' + dev + ' board hostname'); return; }
  LIVE.logoff = 0;
  // New run → reset the TIMESYNC-idle trigger state and drop any stale timer.
  LIVE.tlDone = false; LIVE.tsSeen = false;
  if (LIVE.tsIdleTimer){ clearTimeout(LIVE.tsIdleTimer); LIVE.tsIdleTimer = null; }
  // Stop aiedbg polling/console BEFORE the download starts so its JTAG reads
  // don't collide with device program/reset/dow -force. Re-enabled in pollLog
  // when the run ends (force-stop or natural completion).
  setDebugEnabled(false);
  api('/run', {method:'POST', headers:{'Content-Type':'application/json'},
               body: JSON.stringify({device:dev, board_host:host})})
    .then(r => {
      // Run didn't actually start → re-enable debug (no pollLog will run).
      // A refusal usually means an earlier run is still held; adopt its state
      if (r.error){ con.textContent = 'run error: ' + r.error;
        setDebugEnabled(true);
        if (r.run) applyRunState(r.run); else syncRunState();
        return; }
      // Own it now so the heartbeat doesn't re-adopt it as an unknown run.
      LIVE.runOwned = true;
      LIVE.daemonRun = {run_id:r.run_id, device:dev, status:'starting',
                        started_iso:'', age_s:null, stale:false};
      updateRunButtons();
      con.textContent = '[run ' + r.run_id + ' started \u2192 ' + (r.applog||'applog') + ']\n';
      llmPushCtx('[context] Hardware run '+r.run_id+' started on '+dev, 'run');
      if (LIVE.conTimer) clearInterval(LIVE.conTimer);
      LIVE.conTimer = setInterval(pollLog, 1000);
    })
    .catch(() => { con.textContent = 'daemon offline: cannot start a run (open via schedule_debug_server).'; setDebugEnabled(true); });
};
document.getElementById('stopbtn').onclick = () => {
  const dev = deviceSel ? deviceSel.value : '';
  if (dev === 'simulator'){
    api('/sim/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
      .then(r => {
        // "no simulator running" is not a failure: nothing holds the sim, which
        // is what the click asked for. Clearing on it unwedges the buttons.
        SIM.owned = false;
        if (runbtn) runbtn.disabled = false;
        if (stopbtn) stopbtn.disabled = true;
        if (r.error){ setStatus('stop: ' + r.error); return; }
        setStatus('simulator stopped (pid ' + r.pid + ')');
        pollSimLog();
      })
      .catch(() => { setStatus('daemon offline: cannot stop simulator.'); });
    return;
  }
  api('/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => {
      // Re-enable debug features immediately — don't wait for the next pollLog
      // tick, which may never fire if the run was in a bad state.
      if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer = null; }
      // Drop any pending idle timer so it can't fire into the next run.
      if (LIVE.tsIdleTimer){ clearTimeout(LIVE.tsIdleTimer); LIVE.tsIdleTimer = null; }
      // "no run in progress" is not a failure: the board is free, which is what
      // the click asked for. Clearing on that answer is what unwedges the UI.
      LIVE.runOwned = false;
      setDebugEnabled(true);
      setRunStatus('');
      // Every Stop click should re-render and reload timeline.png. Clearing
      // tlDone defeats the dedup so triggerTimeline regenerates from the final
      // applog and showTimelineProfile re-fetches the image (cache-busted ts).
      LIVE.tlDone = false;
      triggerTimeline(true);   // gated by tsSeen; forced fresh render on Stop.
      if (r.run) applyRunState(r.run);
      updateRunButtons();
      if (r.error){ setStatus('stop: ' + r.error); }
      else if (r.abandoned){
        // #livestatus is hidden in Device Map view, so use the banner too.
        const msg = 'run ' + r.run_id + ' (pid ' + r.pid + ') survived SIGKILL; '
                  + 'the daemon released it but it may still hold the board';
        setStatus(msg); setRunStatus(msg);
      } else {
        setStatus('run: stopped (pid ' + r.pid + ')');
      }
    })
    .catch(() => { setStatus('daemon offline: cannot stop a run.'); setDebugEnabled(true); });
};
// Load the existing applog from the start (one-shot, no live tail). Useful for
// inspecting the last run without launching a new one.

// ── Extra devices (simulator etc): populated from /devices at startup ──
// The server reads debug_ui_config.json from the workdir and returns any
// extra devices (e.g. simulator) that it can run.  Each entry is injected
// as an <option> in the board dropdown; selecting "simulator" short-circuits
// the JTAG "Connect" and routes Run/Stop/Load-log to /sim/* endpoints.
const SIM = { timer: null, logoff: 0, logBusy: false, ipcReady: false,
              applogoff: 0, applogSeen: false, kind: '', owned: false };
// value -> row from /devices. The simulator row is always present; `available`
// says whether it can run and `reason` says what to build if it cannot, so a
// greyed-out simulator names its own remedy instead of just vanishing.
const DEVINFO = {};
function simRow(){ return DEVINFO['simulator'] || null; }
// aiesim has no debug socket and never will; only the IPC flow unlocks reads.
function simHasLiveReads(){
  const row = simRow() || {};
  if (row.live_reads !== undefined) return !!row.live_reads;
  return (SIM.kind || row.sim_kind) === 'ipc';
}
function simLiveUnavailableText(){
  return 'live scans unavailable: aiesim exposes no debug socket';
}
function showSimLiveUnavailable(){
  updateLiveReadControls();
}
function updateLiveReadControls(){
  const blocked = !!(deviceSel && deviceSel.value === 'simulator'
    && !simHasLiveReads());
  ['gridScanBtn','dmScanBtn'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.disabled = blocked;
    el.title = blocked
      ? 'Unavailable: aiesim (aie2pssimmsm) has no live register transport'
      : 'read live status from the board / simulator';
  });
  ['liveToggle','dmLiveToggle'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    if (blocked) el.checked = false;
    el.disabled = blocked || !LIVE.connected;
    const label = el.closest('label');
    if (label) label.classList.toggle('disabled', el.disabled);
  });
  const name = document.querySelector('#conhdr .chname');
  if (name) name.textContent = blocked ? 'aiegdb (navigation only)' : 'aiegdb';
}
function pollSimLog(){
  if (SIM.logBusy) return;
  SIM.logBusy = true;
  const con = document.getElementById('console');
  // Poll both the simulator engine log and the PS application log in parallel.
  Promise.all([
    api('/sim/log?offset='+SIM.logoff),
    api('/sim/applog?offset='+SIM.applogoff).catch(() => ({data:'',next:SIM.applogoff,running:false}))
  ]).then(([r, ra]) => {
    const follow = logFollowing(con);
    if (r.kind) SIM.kind = r.kind;
    if (r.data && con){ con.textContent += r.data; }
    if (r.next != null) SIM.logoff = r.next;
    // PS app output: prefix with a header the first time it appears, then stream.
    if (ra.data && con){
      if (!SIM.applogSeen){
        con.textContent += '\n--- PS application output (ipc_app.log) ---\n';
        SIM.applogSeen = true;
      }
      con.textContent += ra.data;
    }
    if (ra.next != null) SIM.applogoff = ra.next;
    logFollow(con, follow);
    // Enable live grid reads once the IPC debug socket is ready.
    if (r.ipc_ready && !SIM.ipcReady){
      SIM.ipcReady = true;
      LIVE.device = 'simulator'; LIVE.host = '';
      if (liveToggle){ liveToggle.disabled = false;
        liveToggle.closest('label').classList.remove('disabled'); }
      setConnStatus('simulator IPC ready \u2014 live grid reads active');
    }
    if (simHasLiveReads())
      setStatus('sim: ' + (r.running ? 'running' : 'stopped')
              + (SIM.kind ? ' (' + SIM.kind + ')' : ''));
    else
      setStatus('sim: ' + (r.running ? 'running' : 'stopped')
              + ' \u2014 live scans unavailable');
    SIM.owned = !!r.running;
    if (!r.running){
      if (SIM.timer){ clearInterval(SIM.timer); SIM.timer = null; }
      SIM.ipcReady = false;
      if (runbtn)  runbtn.disabled  = false;
      if (stopbtn) stopbtn.disabled = true;
    }
  }).catch(() => {}).finally(() => { SIM.logBusy = false; });
}
// The template bakes in aiedbg's device names AND `simulator`, so a row naming
// one must replace that option rather than append a duplicate value.
function loadDevices(){
  return api('/devices').then(r => {
    if (!r || !r.devices || !deviceSel) return;
    r.devices.forEach(d => {
      DEVINFO[d.value] = d;
      const existing = Array.from(deviceSel.options).find(o => o.value === d.value);
      if (existing){ existing.textContent = d.label; return; }
      const opt = document.createElement('option');
      opt.value = d.value; opt.textContent = d.label;
      deviceSel.appendChild(opt);
    });
    const sr = simRow();
    if (sr && sr.sim_kind) SIM.kind = sr.sim_kind;
    updateLiveReadControls();
    // The labels just changed under whatever is already selected; re-derive the
    // status line so a picked-but-unbuildable simulator says why straight away.
    if (deviceSel.value) refreshDeviceStatus();
  }).catch(() => {});
}
// Once per page. Switching apps reloads the page (see appSel.onchange), which
// is what re-reads these for the newly selected bundle — a different app has a
// different simulator, or none.
loadDevices();
// Splitter helper: pointer-capture drag with anchor-relative sizing.
// onStart(e) → opaque state recorded at pointerdown.
// onMove(e, state) → applies the drag delta using that state.
function _makeSplitter(spId, bodyCls, onStart, onMove){
  const sp = document.getElementById(spId);
  if (!sp) return;
  let state = null;
  sp.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    sp.setPointerCapture(e.pointerId);
    sp.classList.add('drag');
    document.body.classList.add(bodyCls);
    state = onStart(e);
    e.preventDefault();
  });
  sp.addEventListener('pointermove', e => {
    if (!sp.hasPointerCapture(e.pointerId)) return;
    onMove(e, state);
  });
  const _end = e => {
    if (!sp.hasPointerCapture(e.pointerId)) return;
    sp.releasePointerCapture(e.pointerId);
    sp.classList.remove('drag');
    document.body.classList.remove(bodyCls);
    state = null;
  };
  sp.addEventListener('pointerup', _end);
  sp.addEventListener('pointercancel', _end);
}

// Left/right pane splitter.
(function(){
  const left = document.getElementById('left');
  const sp   = document.getElementById('splitter');
  if (!left || !sp) return;
  _makeSplitter('splitter', 'resizing',
    e => ({ x: e.clientX, w: left.getBoundingClientRect().width }),
    (e, s) => {
      const min = 200, max = window.innerWidth - 200 - sp.offsetWidth;
      left.style.flex = '0 0 ' + Math.max(min, Math.min(max, s.w + e.clientX - s.x)) + 'px';
    });
})();

// Top/bottom left-pane splitter.
(function(){
  const top  = document.getElementById('lefttop');
  const left = document.getElementById('left');
  const sp   = document.getElementById('lhsplitter');
  if (!top || !left || !sp) return;
  _makeSplitter('lhsplitter', 'vresizing',
    e => ({ y: e.clientY, h: top.getBoundingClientRect().height }),
    (e, s) => {
      const rect = left.getBoundingClientRect();
      const min = 80, max = rect.height - 80 - sp.offsetHeight;
      top.style.flex = '0 0 ' + Math.max(min, Math.min(max, s.h + e.clientY - s.y)) + 'px';
    });
})();

// Panel/console right-pane splitter.
// Dragging up (negative dy) grows the console; dragging down shrinks it.
(function(){
  const con   = document.getElementById('cmdconsole');
  const right = document.getElementById('right');
  if (!con || !right) return;
  _makeSplitter('rhsplitter', 'vresizing',
    e => ({ y: e.clientY, h: con.getBoundingClientRect().height }),
    (e, s) => {
      const min = 80, max = right.getBoundingClientRect().height - 120;
      con.style.flex = '0 0 ' + Math.max(min, Math.min(max, s.h - (e.clientY - s.y))) + 'px';
    });
})();

// Served over HTTP by the daemon: leave live controls gated on device choice
// (updateDeviceUI disables Run/overlay until a device is picked). On file://
// there is no daemon, so stay static.
// Ask the daemon for board defaults (device + hostname) so the dropdown and
// hostname box come pre-filled instead of empty. Falls back silently on
// file:// or if the daemon doesn't answer.
function applyBoardDefaults(){
  api('/config').then(c => {
    if (!c || c.error) return;
    LIVE.simOnly = !!c.sim_only;
    const bar = document.getElementById('ctrlbar');
    if (bar) bar.dataset.simonly = LIVE.simOnly ? 'true' : 'false';
    if (LIVE.simOnly && deviceSel) deviceSel.value = 'simulator';
    else if (c.device && deviceSel){
      const has = Array.from(deviceSel.options).some(o => o.value === c.device);
      if (has) deviceSel.value = c.device;
    }
    if (c.board_host && boardHost) boardHost.value = c.board_host;
    if (c.jtag_host) LIVE.jtagHost = c.jtag_host;
    if (c.source_viewer === false) SRC.on = false;
    updateDeviceUI();   // reveal/enable controls for the preselected device
    if (LIVE.simOnly) testConnect();
  }).catch(() => {});
}
if (location.protocol === 'http:' || location.protocol === 'https:') {
  updateDeviceUI();   // device empty ⇒ controls disabled, overlay off
  applyBoardDefaults();
  syncRunState();
  LIVE.rsTimer = setInterval(syncRunState, 5000);
} else {
  setStatus('static mode; open via schedule_debug_server.py for live status');
  updateDeviceUI();
}

// ── app switching ────────────────────────────────────────────────────────────
// The daemon owns app selection and injects the chosen app's DATA when serving
// this page, so switching is: POST the choice, then reload.
const appSel = document.getElementById('appSel');
// #appinfo sits beside the selector in #ctrlbar, which is visible in every view
// — unlike #livestatus. On success it goes back to showing the app's path.
function setAppMsg(msg, isErr){
  const info = document.getElementById('appinfo');
  if (!info) return;
  info.textContent = msg;
  info.title = msg;
  info.style.color = isErr ? 'var(--red-fg)' : 'var(--tx-lo)';
  info.dataset.err = isErr ? '1' : '';
}
function loadApps(){
  api('/apps').then(r => {
    if (!r || !r.apps || !appSel) return;
    const one = r.apps.length <= 1;
    const bar = document.getElementById('ctrlbar');
    if (bar) bar.dataset.apps = one ? 'one' : 'many';
    const badge = document.getElementById('appbadge');
    const cur = r.apps.find(a => a.current) || r.apps[0];
    if (cur) document.title = 'AIE Debug — ' + cur.label;
    if (badge){
      badge.classList.toggle('hide', !(one && cur));
      if (cur){ badge.textContent = cur.label; badge.title = cur.path; }
    }
    appSel.innerHTML = '';
    r.apps.forEach(a => {
      const o = document.createElement('option');
      o.value = a.id;
      o.textContent = a.label + (a.has_hw ? '  [hw]' : '') + (a.has_sim ? '  [sim]' : '');
      if (a.current) { o.selected = true;
        const info = document.getElementById('appinfo');
        // Keep a visible error: loadApps() is called by the failure path itself.
        if (info && !info.dataset.err){
          info.textContent = a.path;
          info.title = a.path;
        }
      }
      appSel.appendChild(o);
    });
  }).catch(() => {});
}
if (appSel) appSel.onchange = () => {
  setAppMsg('');            // drop a previous failure before retrying
  api('/apps/select', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify({id: appSel.value})})
    .then(r => {
      // Report next to the selector, NOT via setStatus(): #livestatus lives
      // inside #overlayctl, which switchView() hides in Device Map view — the
      // default — so a failure there is completely invisible and the dropdown
      // just silently snaps back.
      if (r && r.error) { setAppMsg(r.error, true); loadApps(); return; }
      setAppMsg('');
      location.reload();
    }).catch(() => setAppMsg('daemon offline; cannot switch apps', true));
};

// ── UI state reporting ───────────────────────────────────────────────────────
// Tell the daemon what the user currently has open so the embedded agent can
// answer questions about the view the human is actually looking at.
let _uiStateTimer = null;
function reportUIState(patch){
  if (patch) Object.assign(UISTATE, patch);
  if (location.protocol === 'file:') return;
  clearTimeout(_uiStateTimer);
  _uiStateTimer = setTimeout(() => {
    api('/uistate', {method:'POST', headers:{'Content-Type':'application/json'},
                     body: JSON.stringify(UISTATE)}).catch(() => {});
  }, 250);
}
if (location.protocol === 'http:' || location.protocol === 'https:') loadApps();

// LLM tab is independent of the JTAG connection: probe /llm/poll on load and,
// if the daemon answers (and the tab is enabled), reveal the console and
// default-select the LLM tab. The aiegdb tab stays gated on LIVE.connected.
function probeLLM(){
  api('/llm/poll?offset=0').then(r => {
    if (!r || r.error) return;   // disabled (--no-llm) or non-daemon response
    const box = document.getElementById('cmdconsole');
    if (box) box.classList.remove('hide');
    const rsp = document.getElementById('rhsplitter');
    if (rsp) rsp.classList.remove('hide');
    // Default-select the LLM tab.
    const tab = document.querySelector('#contabs .contab[data-pane="llmpane"]');
    if (tab) tab.click();
    // Skip past any existing transcript — it's gone from the DOM on page reload.
    // Replaying the full history would block the main thread on large sessions.
    if (r.next != null) LLM.off = r.next;
    if (r.llm_generation != null) LLM.generation = r.llm_generation;
    if (r.active){
      // Reloaded mid-turn: adopt the in-flight answer instead of letting it
      // land in a transcript nobody is tailing.
      LLM.pendingId = llmAddMsg('ai', '');
      llmShowThink(true);
      llmStopPoll();
      LLM.poll = setInterval(llmPollOnce, 700);
    }
  }).catch(() => {});   // no daemon → stay static
}
if (location.protocol === 'http:' || location.protocol === 'https:') probeLLM();

// ── Selection → LLM popup ──────────────────────────────────────────────────
(function(){
  const popup = document.createElement('div');
  popup.id = 'sel-popup';
  popup.textContent = '+ Add context';
  document.body.appendChild(popup);

  // Accumulated context snippets: unique id -> {label, text}. Keyed by id rather
  // than label so several selections from the same pane accumulate instead of
  // overwriting each other; the "#n" suffix is derived at render time.
  const ctxSnippets = new Map();
  let ctxSeq = 0;

  // Regions where a selection is meaningful to send to the LLM.
  const REGIONS = [
    { id:'panel',  pane:'Info pane' },
    { id:'conout', pane:'aiegdb output' },
    { id:'llmmsg', pane:'LLM history' },
  ];

  function selectionRegion(){
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.toString().trim()) return null;
    const anchor = sel.anchorNode;
    if (!anchor) return null;
    for (const r of REGIONS){
      const el = document.getElementById(r.id);
      if (el && el.contains(anchor)) return r;
    }
    return null;
  }

  // Build a human-readable source label describing exactly where the text came from.
  function buildSourceLabel(region){
    const parts = [region.pane];
    if (UISTATE.selected_tile){
      const [c, r] = UISTATE.selected_tile;
      parts.push('tile('+c+','+r+')');
      if (UISTATE.channel) parts.push(UISTATE.channel);
    } else if (UISTATE.flow != null){
      parts.push('flow f'+UISTATE.flow);
    }
    if (region.id === 'panel' && UISTATE.tile_tab){
      const tabNames = {hi:'Schedule tab', mid:'IR tab', lo:'Code tab'};
      parts.push(tabNames[UISTATE.tile_tab] || UISTATE.tile_tab);
    }
    if (UISTATE.console_pane === 'searchpane' && UISTATE.search){
      parts.push('search "'+UISTATE.search+'"');
    }
    return parts.join(' · ');
  }

  function ctxBlock(label, text){ return '[context from ' + label + ']\n' + text; }

  // Labels are assigned once, at creation, and never renumbered: a pill is a
  // real element in the message, and renaming it under the user is worse than
  // a stale "#2".
  function ctxUniqueLabel(base){
    let n = 0;
    ctxSnippets.forEach(s => { if (s.base === base) n++; });
    return n ? base + ' #' + (n + 1) : base;
  }

  // Build the inline chip. contenteditable="false" makes it an atom: the caret
  // steps over it and backspace removes the whole thing.
  function ctxPillNode(id, snip){
    const pill = document.createElement('span');
    pill.className = 'msg-pill';
    pill.contentEditable = 'false';
    pill.dataset.cid = id;
    pill.title = ctxBlock(snip.label, snip.text);
    const t = document.createElement('span');
    t.textContent = snip.label;
    const x = document.createElement('span');
    x.className = 'mp-x';
    x.textContent = '\u00d7';
    pill.appendChild(t);
    pill.appendChild(x);
    return pill;
  }

  // Zero-width space. Every chip is flanked by one so the caret always has a
  // text position on each side: without them the browser cannot put the caret
  // before a chip that starts the message, or between two adjacent chips.
  const ZW = '\u200b';

  // The caret inside #llmin, remembered across focus loss. Selecting text in
  // another pane moves the document selection out of the box, so by the time
  // "+ Add context" is clicked there is no live caret to insert at — which is
  // why every pill used to land at the end regardless of where you were typing.
  let ctxCaret = null;
  document.addEventListener('selectionchange', () => {
    const inp = document.getElementById('llmin');
    const sel = window.getSelection();
    if (!inp || !sel || !sel.rangeCount) return;
    const r = sel.getRangeAt(0);
    if (inp.contains(r.startContainer)) ctxCaret = r.cloneRange();
  });

  function ctxPlaceCaret(node, offset){
    const sel = window.getSelection();
    const r = document.createRange();
    r.setStart(node, offset);
    r.collapse(true);
    sel.removeAllRanges();
    sel.addRange(r);
    ctxCaret = r.cloneRange();
  }

  // Insert a chip at the remembered caret, then put the caret after it.
  function ctxInsertNode(node){
    const inp = document.getElementById('llmin');
    if (!inp) return;
    let range = (ctxCaret && inp.contains(ctxCaret.startContainer))
      ? ctxCaret.cloneRange() : null;
    if (!range){
      range = document.createRange();
      range.selectNodeContents(inp);
      range.collapse(false);
    }
    range.deleteContents();
    const before = document.createTextNode(ZW);
    const after  = document.createTextNode(ZW);
    const frag = document.createDocumentFragment();
    frag.appendChild(before);
    frag.appendChild(node);
    frag.appendChild(after);
    range.insertNode(frag);
    inp.focus();
    ctxPlaceCaret(after, after.nodeValue.length);
    inp.dispatchEvent(new Event('input'));
  }

  // The chip immediately before a collapsed caret, or null. Only the flanking
  // ZW counts as "nothing in between" — a space the user typed is real text and
  // must be deleted first, so backspace stays predictable.
  function ctxChipBefore(range){
    let n = range.startContainer;
    if (n.nodeType === 3){
      if (n.nodeValue.slice(0, range.startOffset) !== ZW) return null;
      n = n.previousSibling;
    } else {
      n = n.childNodes[range.startOffset - 1] || null;
    }
    return (n && n.nodeType === 1 && n.classList
            && n.classList.contains('msg-pill')) ? n : null;
  }

  // Remove a chip together with the ZW flanks we added around it.
  function ctxRemoveChip(pill){
    const inp = document.getElementById('llmin');
    const prev = pill.previousSibling, next = pill.nextSibling;
    const anchor = (prev && prev.nodeType === 3) ? prev : null;
    if (next && next.nodeType === 3 && next.nodeValue === ZW) next.remove();
    if (prev && prev.nodeType === 3 && prev.nodeValue === ZW) prev.remove();
    pill.remove();
    if (inp){
      if (anchor && anchor.parentNode) ctxPlaceCaret(anchor, anchor.nodeValue.length);
      else ctxPlaceCaret(inp, inp.childNodes.length);
      inp.dispatchEvent(new Event('input'));
    }
  }

  // Read the message back out: text as typed, pills as whatever `pill` maps them
  // to. One walker serves both the outgoing prompt (full context blocks) and the
  // plain-text check for "is there anything to send".
  function ctxReadInput(pill){
    const inp = document.getElementById('llmin');
    if (!inp) return {text:'', sent:[]};
    return ctxReadNode(inp, pill);
  }

  function ctxReadNode(root, pill){
    const sent = [], out = [];
    (function walk(node){
      node.childNodes.forEach(n => {
        if (n.nodeType === 3){
          out.push(n.nodeValue.replace(/\u200b/g, '').replace(/\u00a0/g, ' '));
          return;
        }
        if (n.nodeType !== 1) return;
        if (n.classList && n.classList.contains('msg-pill')){
          const snip = ctxSnippets.get(+n.dataset.cid);
          if (snip){ sent.push(snip); out.push(pill(snip)); }
          return;
        }
        if (n.tagName === 'BR'){ out.push('\n'); return; }
        // A block the browser produced (paste, or an Enter we did not intercept)
        // starts its own line, so it needs a break before as well as after.
        const block = (n.tagName === 'DIV' || n.tagName === 'P');
        if (block && out.length && !/\n$/.test(out[out.length - 1])) out.push('\n');
        walk(n);
        if (block) out.push('\n');
      });
    })(root);
    return {text:out.join(''), sent:sent};
  }

  // Insert a whole fragment (paste) at the remembered caret.
  function ctxInsertFrag(frag){
    const inp = document.getElementById('llmin');
    if (!inp || !frag.childNodes.length) return;
    const last = frag.childNodes[frag.childNodes.length - 1];
    let range = (ctxCaret && inp.contains(ctxCaret.startContainer))
      ? ctxCaret.cloneRange() : null;
    if (!range){
      range = document.createRange();
      range.selectNodeContents(inp);
      range.collapse(false);
    }
    range.deleteContents();
    range.insertNode(frag);
    inp.focus();
    if (last.nodeType === 3) ctxPlaceCaret(last, last.nodeValue.length);
    else if (last.parentNode)
      ctxPlaceCaret(last.parentNode, Array.prototype.indexOf.call(
        last.parentNode.childNodes, last) + 1);
    inp.dispatchEvent(new Event('input'));
  }

  // Rebuild a pasted selection. Chips are recreated from data-cid; EVERYTHING
  // else becomes plain text, so foreign markup can never enter the box. Parsed
  // with DOMParser rather than innerHTML: that document is inert, so a pasted
  // <img onerror> neither loads nor fires.
  function ctxFragFromHtml(html){
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const frag = document.createDocumentFragment();
    // A newline goes BEFORE a block, never after. Emitting one after each block
    // added a trailing blank line per wrapper — and a copy out of the box is
    // wrapped, so every paste grew by a line.
    //
    // A chip is inline text and must never start or end a line of its own. The
    // wrapper markup around a copied chip would otherwise put a break on each
    // side, dropping the pasted chip onto its own line. `justPill` swallows the
    // wrapper break that follows one, `dropNl` removes the one that preceded
    // it. An explicit <br> is user intent and always stands.
    let justPill = false;
    const nl = () => {
      if (!frag.childNodes.length) return;
      if (justPill){ justPill = false; return; }
      const last = frag.childNodes[frag.childNodes.length - 1];
      if (last && last.nodeType === 3 && /\n$/.test(last.nodeValue)) return;
      frag.appendChild(document.createTextNode('\n'));
    };
    const dropNl = () => {
      const last = frag.childNodes[frag.childNodes.length - 1];
      if (last && last.nodeType === 3 && /^\n+$/.test(last.nodeValue))
        frag.removeChild(last);
    };
    (function walk(node){
      node.childNodes.forEach(n => {
        if (n.nodeType === 3){
          const t = n.nodeValue.replace(/\u200b/g, '');
          if (t) justPill = false;
          frag.appendChild(document.createTextNode(t));
          return;
        }
        if (n.nodeType !== 1) return;
        if (n.classList && n.classList.contains('msg-pill')){
          const cid = +n.dataset.cid;
          const snip = ctxSnippets.get(cid);
          dropNl();
          if (snip){
            frag.appendChild(document.createTextNode(ZW));
            frag.appendChild(ctxPillNode(cid, snip));
            frag.appendChild(document.createTextNode(ZW));
          } else {
            // Pasted from another page/session: the snippet is gone, so keep
            // the readable token rather than a chip that resolves to nothing.
            frag.appendChild(document.createTextNode(
              '[[' + (n.textContent || '').replace(/\u00d7/g, '').trim() + ']]'));
          }
          justPill = true;
          return;
        }
        if (n.tagName === 'BR'){
          justPill = false;
          frag.appendChild(document.createTextNode('\n'));
          return;
        }
        if (n.tagName === 'DIV' || n.tagName === 'P') nl();
        walk(n);
      });
    })(doc.body);
    return frag;
  }

  // The selection inside the box, as {text, html} — text via the same walker
  // the sender uses (so a chip reads as [[label]], never as its × glyph).
  function ctxSelectionClip(){
    const sel = window.getSelection();
    const inp = document.getElementById('llmin');
    if (!inp || !sel || !sel.rangeCount || sel.isCollapsed) return null;
    const r = sel.getRangeAt(0);
    if (!inp.contains(r.commonAncestorContainer)) return null;
    const frag = r.cloneContents();
    const holder = document.createElement('div');
    holder.appendChild(frag.cloneNode(true));
    return {range:r,
            text: ctxReadNode(frag, s => '[[' + s.label + ']]').text,
            html: holder.innerHTML};
  }

  function ctxClearInput(){
    const inp = document.getElementById('llmin');
    if (inp) inp.innerHTML = '';
    ctxCaret = null;      // its containers are detached now
  }

  // llmSend calls this: the pills ARE the message, so substitution is just the
  // walk above with pills rendered as their context block.
  function drainCtxSnippets(){
    const r = ctxReadInput(s => '\n' + ctxBlock(s.label, s.text) + '\n');
    ctxSnippets.clear();
    return r;
  }
  window._drainCtxSnippets = drainCtxSnippets;
  // Plain text of the message, pills reduced to their label — used for the
  // transcript echo and the "is there anything to send" check.
  window._llmInputText = () => ctxReadInput(s => '[[' + s.label + ']]').text.trim();
  window._hasCtxSnippets = () => {
    const inp = document.getElementById('llmin');
    return !!(inp && inp.querySelector('.msg-pill'));
  };
  window._llmClearInput = ctxClearInput;
  // The keydown wiring lives with the input; it needs these internals.
  window._ctxChipBefore = ctxChipBefore;
  window._ctxRemoveChip = ctxRemoveChip;
  window._ctxSelectionClip = ctxSelectionClip;
  window._ctxFragFromHtml = ctxFragFromHtml;
  window._ctxInsertFrag = ctxInsertFrag;

  function hidePopup(){ popup.style.display = 'none'; }

  document.addEventListener('selectionchange', () => {
    if (popup.style.display === 'none') return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.toString().trim()) hidePopup();
  });

  document.addEventListener('mouseup', e => {
    if (e.target === popup) return;
    hidePopup();
    if (!selectionRegion()) return;
    const sel = window.getSelection();
    const range = sel.getRangeAt(0);
    const rect  = range.getBoundingClientRect();
    if (!rect.width && !rect.height) return;
    popup.style.left = (rect.left + rect.width / 2) + 'px';
    popup.style.top  = (rect.top + window.scrollY - 30) + 'px';
    popup.style.display = 'block';
  });

  document.addEventListener('mousedown', e => {
    if (e.target !== popup) hidePopup();
  });
  document.addEventListener('scroll', hidePopup, true);

  popup.addEventListener('mousedown', e => { e.preventDefault(); e.stopPropagation(); });
  popup.addEventListener('click', e => {
    e.preventDefault();
    const region = selectionRegion();
    const sel = window.getSelection();
    const text = sel ? sel.toString().trim() : '';
    if (!text || !region){ hidePopup(); return; }
    const label = buildSourceLabel(region);
    // Every selection becomes its own pill; the same text from the same place
    // twice is two pills, which is what placing it in two spots means now.
    const id = ++ctxSeq;
    const snip = {base:label, label:ctxUniqueLabel(label), text:text};
    ctxSnippets.set(id, snip);
    const cmdconsole = document.getElementById('cmdconsole');
    if (cmdconsole && cmdconsole.classList.contains('hide'))
      cmdconsole.classList.remove('hide');
    const llmTab = document.querySelector('#contabs .contab[data-pane="llmpane"]');
    if (llmTab) llmTab.click();
    sel.removeAllRanges();
    ctxInsertNode(ctxPillNode(id, snip));
    hidePopup();
  });
})();

// Open on the Grid. Deferred to the very end of the script for symmetry with
// the previous default (buildDeviceMap() reaches for consts declared above it
// and would hit their temporal dead zone if called any earlier).
switchView('grid');
</script>
</body>
</html>
"""


_AIEDBG_DEVICE_FALLBACK = ("pal", "vck190", "vek280", "vek385")


def _aiedbg_devices():
    """Board options for the dropdown, taken from aiedbg's own device table.

    These values ARE the `aiedbg -d` names, so the UI selection and the debug
    device can no longer drift apart. Falls back to a fixed tuple when aiedbg
    is not importable (static host_schedule.html generation)."""
    try:
        from aiedbg.device_config import DEVICE_CONFIGS
        return sorted(DEVICE_CONFIGS)
    except Exception:
        return list(_AIEDBG_DEVICE_FALLBACK)


# ─────────────────────────────────────────────────────────────────────────────
# Per-tile / per-channel code-piece cache
# ─────────────────────────────────────────────────────────────────────────────
# For local-LLM cooperation we materialize the host.cc code that implements each
# tile and each DMA channel into standalone files under ./debugcache/code/. Each
# tile/channel in the view gets a `code_file` (absolute path) so the browser can
# (a) show the code-piece path at the top of the file frame and (b) auto-tell the
# LLM "the related code is <path>" on click.

def _tile_cache_content(t):
    col, row = t['loc']
    hl = t.get('high_level', {}) or {}
    lo = t.get('low_level', {}) or {}
    out = ['// AIE Schedule debug code piece',
           '// Tile (%d,%d)  type=%s' % (col, row, t.get('type'))]
    if hl.get('role'):
        out.append('// role: %s' % hl['role'])
    if hl.get('kernel'):
        out.append('// kernel: %s' % hl['kernel'])
    for s in hl.get('summary', []) or []:
        out.append('//   transfer: %s' % s)
    seen_fb = set()
    for c in t.get('dma_channels', []) or []:
        fb = c.get('flow_balance')
        if not fb or fb['flow_index'] in seen_fb:
            continue
        seen_fb.add(fb['flow_index'])
        if fb.get('balanced') is False:
            verd = 'OVER-SUPPLY' if (fb['supply_per_round'] or 0) > (fb['demand_per_round'] or 0) else 'UNDER-SUPPLY'
        elif fb.get('balanced') is True:
            verd = 'balanced'
        else:
            verd = 'unchecked'
        out.append('//   supply/demand flow %s (%s): %s  supply=%s demand=%s'
                   % (fb['flow_index'], fb['pattern'], verd,
                      fb['supply_per_round'], fb['demand_per_round']))
    km = hl.get('kernel_match')
    if km and km.get('matches'):
        out.append('// channel <-> kernel argument (by BD buffer address):')
        for m in km['matches']:
            adr = '/'.join(m.get('addrs_hex') or []) or '-'
            bsy = '/'.join(m.get('bcf_syms') or []) or '-'
            out.append('//   %s%s -> window %s arg %s  [bd %s = %s] via %s'
                       % (m.get('direction'), m.get('channel'),
                          m.get('window'),
                          ('arg%s' % m['arg']) if m.get('arg') is not None else '-',
                          adr, bsy, m.get('method')))
    rngs = lo.get('ranges') or []
    if rngs:
        out.append('// host.cc lines %s-%s (%d range(s))'
                   % (lo.get('line_start'), lo.get('line_end'), len(rngs)))
    out.append('')
    code_lines = lo.get('code_lines') or []
    if not code_lines:
        out.append('// (no attributed host.cc lines for this tile)')
    for rec in code_lines:
        out.append('/* L%-5d */ %s' % (rec['line'], rec['code']))
    return '\n'.join(out) + '\n'


def _channel_cache_content(t, c):
    col, row = t['loc']
    d, ch = c['direction'], c['channel']
    out = ['// AIE Schedule debug code piece',
           '// Tile (%d,%d) channel %s%s (flow %s)'
           % (col, row, d, ch, c.get('flow_index'))]
    if c.get('contract'):
        out.append('// contract: %s' % c['contract'])
    fb = c.get('flow_balance')
    if fb:
        if fb.get('balanced') is False:
            verd = 'OVER-SUPPLY' if (fb['supply_per_round'] or 0) > (fb['demand_per_round'] or 0) else 'UNDER-SUPPLY'
        elif fb.get('balanced') is True:
            verd = 'balanced'
        else:
            verd = 'unchecked'
        out.append('// supply/demand (%s): %s  supply=%s demand=%s  [%s]'
                   % (fb['pattern'], verd, fb['supply_per_round'],
                      fb['demand_per_round'], fb['note']))
    out.append('')
    entries = sorted(c.get('host_lines', []) or [], key=lambda e: e['line'])
    if not entries:
        out.append('// (no relevant host.cc lines for this channel)')
    for e in entries:
        if e.get('bd_comment'):
            out.append(e['bd_comment'])
        if e.get('note'):
            out.append('// %s' % e['note'])
        out.append('/* L%-5d %s */ %s' % (e['line'], e.get('kind', ''), e['code']))
    return '\n'.join(out) + '\n'


def write_code_cache(view, cache_dir=None, workdir=None):
    """Write one code-piece file per tile and per channel; annotate the view
    with absolute `code_file` paths. Returns (cache_dir_abs, file_count).

    Defaults to <workdir>/debugcache/code so each app owns its pieces. A
    CWD-relative default would make two apps generated from the same directory
    overwrite each other, and the paths are handed to the LLM to read."""
    if cache_dir is None:
        base = workdir if workdir else '.'
        cache_dir = os.path.join(base, 'debugcache', 'code')
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    count = 0
    for t in view.get('tiles', []):
        col, row = t['loc']
        tpath = os.path.join(cache_dir, 'tile_c%d_r%d.cc' % (col, row))
        with open(tpath, 'w') as f:
            f.write(_tile_cache_content(t))
        t['code_file'] = tpath
        count += 1
        for c in t.get('dma_channels', []) or []:
            cpath = os.path.join(
                cache_dir,
                'tile_c%d_r%d_%s%s.cc' % (col, row, c['direction'].lower(),
                                          c['channel']))
            with open(cpath, 'w') as f:
                f.write(_channel_cache_content(t, c))
            c['code_file'] = cpath
            count += 1
    return cache_dir, count


def _aiegdb_spec_json():
    """The aiegdb command grammar, baked into the page as the autocomplete's
    offline fallback. The daemon serves a live copy at /aiegdb/spec; the
    standalone host_schedule.html has no daemon, so it needs this. Importing
    aiegdb must never be fatal — the viewer is useful without a console."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import aiegdb
        return json.dumps(aiegdb.command_spec(), indent=None)
    except Exception:
        return 'null'


def render_html(view):
    """Return the full UI page with `view` injected. Used both to write the
    standalone file and to serve a selected app live from
    schedule_debug_server."""
    data_json = json.dumps(view, indent=None)
    html = HTML_TEMPLATE.replace('/*__DATA__*/ null', data_json)
    html = html.replace('/*__GDBSPEC__*/ null', _aiegdb_spec_json())
    opts = "".join('          <option value="%s">%s</option>\n' % (d, d)
                   for d in _aiedbg_devices())
    # `simulator` is baked in beside the boards rather than appended by
    # /devices, so it is present in the static page and before the daemon
    # answers. Whether it can RUN is a separate question /devices answers by
    # relabelling this option; an option that only appears when it happens to
    # be buildable reads as a UI that cannot simulate at all.
    opts += '          <option value="simulator">simulator</option>\n'
    return html.replace('<!--__DEVICE_OPTIONS__-->', opts)


def write_html(view, out_path):
    with open(out_path, 'w') as f:
        f.write(render_html(view))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    json_only = '--json-only' in sys.argv[1:]
    workdir = args[0] if args else 'aout/worklocal'
    view = build_view(workdir)
    # Materialize per-tile/channel code pieces (annotates view with code_file
    # paths) BEFORE serializing so both the JSON and the embedded HTML DATA carry
    # the paths for the file-frame header + LLM auto-notify.
    cache_dir, ncode = write_code_cache(view, workdir=workdir)
    json_out = os.path.join(workdir, 'schedule_view.json')
    with open(json_out, 'w') as f:
        json.dump(view, f, indent=2)
    ntiles = len(view['tiles'])
    covered = sum(len(t['low_level']['ranges']) for t in view['tiles'])
    print('wrote %s (%d tiles, %d line-ranges attributed)' % (json_out, ntiles, covered))
    if not json_only:
        html_out = os.path.join(workdir, 'host_schedule.html')
        write_html(view, html_out)
        print('wrote %s' % html_out)
    print('wrote %d code pieces to %s' % (ncode, cache_dir))


if __name__ == '__main__':
    main()
