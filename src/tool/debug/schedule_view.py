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
# supply/demand balance analysis (per data flow)
#
# Flags flows where the producer side and consumer side do not move a matching
# number of bytes PER ROUND. We deliberately use per-round byte *coverage*
# (bd_len based) instead of total static bytes, because core-tile DMA channels
# are lock-driven: their static start_io repeat_count is 1 even though they
# re-fire many times under kernel/lock control. bd_len (the per-fire chunk) is
# always explicit and reliable, so comparing per-round coverage avoids false
# positives on correct lock-driven designs. Verified against a 4x4 matmul:
#   broadcast  shim_bd_len 16384 == 4 cores x 4096   (balanced)
#   gather     4 cores x 1024   == shim_bd_len 1024 x 4 fires (balanced)
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
            # Broadcast: one shim MM2S feeds N core S2MM. The shim's per-fire
            # chunk must cover the sum of what all cores read that round.
            pattern = 'broadcast'
            supply = shim_prod[0]['bd_len']
            demand = sum(c['bd_len'] for c in core_cons)
            balanced = (supply == demand)
            note = 'shim per-fire bytes vs sum of %d core reads' % len(core_cons)
        elif len(shim_cons) == 1 and core_prod and not core_cons and not shim_prod:
            # Gather: N core MM2S drain into one shim S2MM. We compare PER-FIRE
            # byte coverage (shim per-fire vs core per-fire), NOT startio repeat
            # counts. The shim's startio repeat is numCoreTiles x oooIterWrap
            # (e.g. 4x4=16), which is NOT an over-demand: each core is lock-driven
            # and re-fires m_rounds*n_rounds times under kernel control (its static
            # repeat=1 hides this), so aggregate shim fires == aggregate core fires.
            # Comparing shim repeat against the producer count would false-flag this
            # correct, byte-balanced design (verified against the 4x4 256^3 matmul).
            pattern = 'gather'
            supply = sum(p['bd_len'] for p in core_prod)
            demand = shim_cons[0]['bd_len'] * len(core_prod)
            balanced = (supply == demand)
            note = 'sum of %d core writes vs shim per-fire x %d gather fires' % (
                len(core_prod), len(core_prod))
        elif len(producers) == 1 and len(consumers) == 1:
            # Point-to-point: per-fire chunks must match.
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

# Alternate patterns for aiecompiler-generated kernel.cc (baremetal flow).
# Naming convention differs: window names are pingpong buffer pairs (buf0_buf0d),
# lock #defines are tile-scoped (LOCK_<col>_<row>_<id>_<PRD|CNS>), and the
# kernel invocation comment is absent (direct call).
#   window_init(window_buf0_buf0d, 1, buf0, LOCK_4_0_0_PRD, buf0d, LOCK_4_0_1_CNS, 8, 8)
#   get_input_async_window_int32(window_buf0_buf0d)
#   // Kernel call : i2:square  -> kernel function name
RE_KWININIT_BC = re.compile(
    r'window_init\(\s*window_(\w+)\s*,\s*\d+\s*,'      # window_<pingpong_name>
    r'\s*(\w+)\s*,'                                      # ping_buf
    r'\s*LOCK_\w+\s*,'                                   # ACQ lock (any name)
    r'\s*(\w+)\s*,'                                      # pong_buf
    r'\s*LOCK_\w+\s*,'                                   # REL lock (any name)
    r'\s*(\d+)\s*,')                                     # buf_size (words)
RE_KARG_BC = re.compile(r'get_(input|output)_async_window\w*\(\s*window_(\w+)\s*\)')
RE_KINVOKE_BC = re.compile(r'//\s*Kernel call\s*:\s*\w+:(\w+)')
# Also match buf0/buf0d style buffer names (not just buf_xxx)
RE_KBUF_BC = re.compile(r'\b(buf\w+)\b')


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
        # Alternate: aiecompiler baremetal kernel.cc format.
        # window_init(window_<name>, 1, ping_buf, LOCK_x, pong_buf, LOCK_y, size, size)
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
        m = RE_KINVOKE.search(ln)
        if m:
            function = m.group(1)
            marker_lines.append({'line': i, 'code': ln})
            continue
        # Alternate: aiecompiler kernel invoke comment format.
        # // Kernel call : i2:square
        m = RE_KINVOKE_BC.search(ln)
        if m and function is None:
            function = m.group(1)
            marker_lines.append({'line': i, 'code': ln})
            continue

    # Buffer declaration lines (`<type> buf_xxx[N];`) per window, for highlight.
    # Also handle baremetal buf0/buf0d style names.
    for w in windows.values():
        for buf in w['buffers']:
            decl_re = re.compile(r'\b' + re.escape(buf) + r'\s*\[')
            for i, ln in enumerate(src, 1):
                if decl_re.search(ln) and '=' not in ln.split(buf, 1)[0]:
                    if i not in w['buf_decl_lines']:
                        w['buf_decl_lines'].append(i)

    # Invoke call line: the call to <function>( ... ) with the arg windows.
    # In aiehlc_aiesim format the get_input/output_async_window calls appear on
    # the same line as the kernel call.  In the aiecompiler baremetal format they
    # appear earlier as separate variable assignments; we collect those first,
    # then find the bare invoke line.
    args = []
    # Pre-pass: collect window→direction from standalone get_input/output lines.
    # Maps window_name -> (direction, line_no) for later arg ordering.
    win_dir_by_line = {}   # line_no -> (direction, win_name)
    for i, ln in enumerate(src, 1):
        for am in list(RE_KARG.finditer(ln)) + list(RE_KARG_BC.finditer(ln)):
            d = 'input' if am.group(1) == 'input' else 'output'
            wn = am.group(2)
            win_dir_by_line[i] = (d, wn)
            if wn in windows:
                windows[wn]['dir'] = d
    if function:
        call_re = re.compile(r'\b' + re.escape(function) + r'\s*\(')
        for i, ln in enumerate(src, 1):
            if call_re.search(ln):
                # Primary: get_* calls on same line (aiehlc_aiesim format).
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
                # Baremetal: invoke is a bare call like square((var1),(var2)).
                # Reconstruct arg order from the pre-collected win_dir_by_line,
                # sorted by line number (assignment order == param order).
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
# The default 0x70000 matches aiehlc_aiesim's memory map.  Baremetal designs
# (aiecompiler Work directory flow) embed win_base in dfscheduleprovenancemap.json
# from the .bcf _reserved DMb 0x0 <WIN_BASE> line; build_view() reads it and
# overrides this default when present.
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

    # window name -> window object (baremetal port-name resolution path).
    win_by_name = {w['name']: w for w in wins}

    matches = []
    for c in channels:
        direction = c.get('direction')

        # Baremetal: work2provenance resolved the window by port name (address
        # arithmetic is unreliable because buffers land in different memory
        # modules).  Use it directly when present.
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
    candidates.append(os.path.join(os.getcwd(), 'ir/dfschedule', DFSCHED_IR_NAME))
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
# Communication path loader (dmaphop provenance map → structured hop list)
# ----------------------------------------------------------------------------

import re as _re

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

    # ── Load dmaphop ──────────────────────────────────────────────────────────
    hop_path = os.path.join(workdir, 'dmaphopprovenacemap.json')
    if not os.path.exists(hop_path):
        return []
    try:
        with open(hop_path) as f:
            hop_data = json.load(f)
    except Exception:
        return []

    shim_row = 0

    # ── Load routing provenance (optional) ────────────────────────────────────
    rp_path = os.path.join(workdir, 'routingprovenancemap.json')
    rp_groups = []
    if os.path.exists(rp_path):
        try:
            with open(rp_path) as f:
                rp_groups = json.load(f).get('routing_groups', [])
        except Exception:
            pass

    # Build routing group lookup keyed by the frozenset of DMA terminal (col,row) tiles.
    # Col groups (ioid=13-16) have 4 DMA tiles in a single column.
    # Row groups (ioid=17,19,21,23) have 4 DMA tiles in a single row.
    # This key is unambiguous regardless of IOID/flow_index mismatch and regardless of
    # which shim column feeds the group.
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
        # Baremetal (work2provenance) groups are point-to-point and tag each group
        # with its flow_index, since two nets can share the same physical tile set.
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

        # Find split point: connections before shim_aie_to_ext = push path,
        # connections after = pull path. Col groups have no shim_aie_to_ext.
        split_idx = next(
            (i for i, c in enumerate(conns) if c.get('kind') == 'shim_aie_to_ext'),
            None
        )
        if split_idx is not None:
            cc_conns = conns[:split_idx] if direction == 'push' else conns[split_idx + 1:]
        else:
            cc_conns = conns

        # packet_connect tiles in the push section are the DMA core tiles that also
        # send results back via packet routing. Any adjacent hop between two of these
        # tiles is a packet-routed stream connection, not shmem.
        push_conns = conns[:split_idx] if split_idx is not None else conns
        packet_tiles = {(c['tile']['col'], c['tile']['row'])
                        for c in push_conns if c.get('kind') == 'packet_connect'}

        # For pull paths: packet_connect tile in cc_conns is the packet collector.
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
                continue   # off-grid — south of shim row
            # Stitch packet_connect collector → this relay (pull path only, once).
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

        # For pull paths: extract the gather chain from packet_connect forward_master.
        # Each packet_connect tile has a forward_master direction that shows the
        # lateral routing of gathered results (e.g. EAST for (0,4)→(1,4)→(2,4)→(3,4)).
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
                    continue  # NONE or unknown — this is the chain terminus
                tc, tr = fc + d[0], fr + d[1]
                if tr < 0 or (tc, tr) not in packet_tiles:
                    continue
                edge = [[fc, fr], [tc, tr]]
                if edge not in edges:
                    edges.append(edge)
                    tiles_seen.add((fc, fr))
                    tiles_seen.add((tc, tr))

        return edges, tiles_seen, dma_tiles, packet_tiles

    # ── Parse each communication path ────────────────────────────────────────
    result = []
    for p in hop_data.get('communication_paths', []):
        path_id  = p.get('id', '')
        direction = p.get('direction', 'push')

        producer = next((s for s in p.get('stages', []) if s.get('role') == 'producer'), None)
        consumer = next((s for s in p.get('stages', []) if s.get('role') == 'consumer'), None)
        channel  = next((s for s in p.get('stages', []) if s.get('role') == 'channel'),  None)

        # Flow index: push → producer port_sym; pull → consumer port_sym
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

        # Raw hops from channel stage. A hop may carry an explicit 'hop_type'
        # (e.g. baremetal shared-memory hops) which overrides classification.
        raw_pairs = []
        if channel:
            for h in channel.get('hops', []):
                src = parse_coord(h.get('from', ''))
                dst = parse_coord(h.get('to', ''))
                if src and dst:
                    raw_pairs.append((src, dst, h.get('hop_type'), h.get('shmem_kind')))

        # Pull hops are stored in reverse (last→first); reverse to get data-flow order
        if direction == 'pull':
            raw_pairs = list(reversed(raw_pairs))

        # Non-shim tiles for routing-group matching = stream-hop tiles only
        # (explicit shmem hops are not part of the stream-switch routing group).
        nonshim_tiles = list({(c, r) for (c, r) in
                              [p for (s, d, hint, kind) in raw_pairs if hint != 'shmem'
                               for p in (s, d)]
                              if r != shim_row})

        # Extract exact adjacency edges from routing provenance circuit_connect entries.
        # Matched by DMA terminal tile set so push and pull are split correctly.
        routing_edges, routing_tiles, routing_dma_tiles, routing_packet_tiles = routing_edges_for_flow(direction, nonshim_tiles, flow_index)

        if routing_edges:
            edges_out      = routing_edges
            tiles_list     = [list(t) for t in routing_tiles]
            dma_tiles_out  = [list(t) for t in routing_dma_tiles]
            packet_tiles_out = [list(t) for t in routing_packet_tiles]
        else:
            # Fallback: derive edges directly from dmaphop pairs (may skip MEM rows).
            # Skip non-adjacent hops and explicit shmem hops (shmem is drawn in
            # its own layer, not as a stream edge).
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

        # Classify dmaphop hops:
        # - Multi-tile hops (manhattan > 1) are abstract aggregations — not drawable as shmem.
        # - Adjacent hops covered by routing provenance circuit_connect are circuit-switched stream.
        # - Adjacent hops between packet_connect tiles are packet-switched stream.
        # - Adjacent hops in none of the above are true shared-memory connections.
        routing_edge_set = {(e[0][0], e[0][1], e[1][0], e[1][1]) for e in routing_edges}
        packet_tile_set  = {(t[0], t[1]) for t in routing_packet_tiles}
        hops_out = []
        for (fc, fr), (tc, tr), hint, kind in raw_pairs:
            dist = abs(fc - tc) + abs(fr - tr)
            if hint:
                htype = hint      # explicit (e.g. baremetal shared-memory hop)
            elif dist != 1:
                htype = 'stream'
            elif ((fc, fr, tc, tr) in routing_edge_set or
                  (tc, tr, fc, fr) in routing_edge_set):
                htype = 'stream'  # circuit-switched stream
            elif (fc, fr) in packet_tile_set and (tc, tr) in packet_tile_set:
                htype = 'stream'  # packet-switched stream — still a hardware stream-switch connection
            else:
                htype = 'shmem'   # true shared memory
            hop = {'from_col': fc, 'from_row': fr,
                   'to_col': tc, 'to_row': tr,
                   'type': htype}
            if kind:
                hop['kind'] = kind   # 'window' (ping-pong) | 'dma' (direct DMA bridge)
            hops_out.append(hop)

        prod = [shim_col, shim_row_val]

        # Attach raw routing group (connections list) for the detail panel.
        rg = None
        if rp_by_flow_index and flow_index is not None:
            rg = rp_by_flow_index.get(flow_index)
        if rg is None:
            rg = rp_by_dma_tiles.get(frozenset(nonshim_tiles))
        rg_connections = rg.get('connections', []) if rg else []

        # Attach dmaphop raw stages (producer/consumer info) from the original hop_data.
        stages_raw = p.get('stages', [])

        result.append({
            'id': path_id,
            'flow_index': flow_index,
            'direction': direction,
            'prod_col': prod[0],
            'prod_row': prod[1],
            'tiles':     tiles_list,     # used only for tileMap merging in JS
            'edges':     edges_out,      # [[fc,fr],[tc,tr]] adjacency pairs for drawing
            'dma_tiles':    dma_tiles_out,    # core tiles that are DMA consumers (tap markers)
            'packet_tiles': packet_tiles_out, # tiles connected via packet-routed streams
            'hops':      hops_out,       # dmaphop hop pairs (all stream-switch; pull core hops = packet gather)
            'routing_connections': rg_connections,  # raw stream-switch circuit_connect list
            'stages':    stages_raw,     # producer/channel/consumer stage metadata
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

    # Kernel code show: parse the top-level kernel.cc once (graceful skip if
    # absent).  This is the global fallback; when a tile carries per-tile
    # `kernel_cc`/`bcf` paths (baremetal, one kernel per core) those take
    # precedence and are loaded/cached below.
    kernel_view = parse_kernel(os.path.join(workdir, 'kernel.cc'))
    # .bcf buffer symbol map (buffer name -> tile address), for the new "*.bcf"
    # sub-tab correlated to the focused channel's window buffers.
    bcf_view = parse_bcf(find_bcf(workdir))

    # Per-tile kernel.cc / .bcf caches (baremetal: each core runs its own kernel
    # with its own buffer layout). Keyed by out_dir-relative path.
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

    fstart, fend = find_function_range(host_lines)
    owner, var2loc, _ = attribute_lines(host_lines, fstart, fend)

    # for-loop context (headers + per-iteration index math), computed once.
    loops = find_loops(host_lines, fstart, fend)

    # Rename-robust, filtered per-channel line mapping (comment/type anchored).
    channel_map = map_relevant_lines(host_lines, fstart, fend, prov, loops)

    # Supply/demand balance per data flow (flag over/under supply). Indexed by
    # flow_index so each tile channel can surface its flow's balance verdict.
    supply_demand = compute_flow_balance(prov.get('flow_summary', []))
    balance_by_flow = {b['flow_index']: b for b in supply_demand}

    # SSA value/def maps for the per-channel full-block arg trace-back.
    cmap = build_const_map(host_lines, fstart, fend)
    var_def = build_var_def_map(host_lines, fstart, fend)

    # Middle tab: verbatim dfschedule IR slicer (built once, graceful if absent).
    ir_path = find_dfschedule_ir(workdir)
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
            # Attach this channel's flow supply/demand verdict (if any).
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

        # Per-tile kernel/bcf (baremetal) with graceful fallback to the global
        # top-level kernel.cc/.bcf (native aiehlc, shared kernel).
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
        # Attach per-tile kernel/bcf views (baremetal) so the UI shows the kernel
        # source specific to this core rather than the global fallback. Only set
        # when a distinct per-tile artifact was found.
        if ttype == 'core':
            tk = _tile_kernel_view(t.get('kernel_cc'))
            tb = _tile_bcf_view(t.get('bcf'))
            if tk is not None:
                tile_out['kernel'] = tk
            if tb is not None:
                tile_out['bcf'] = tb
        tiles_out.append(tile_out)

    # global (non-tile) lines: kernel group load/launch, final wait, dbg snapshot
    glns = owner_lines.get('__global__', [])
    granges, gcode, _ = code_for(glns) if glns else ([], '', [])

    view = {
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
<title>AIE Schedule View</title>
<style>
  :root { --shim:#2d6a9f; --core:#3a7d44; --sel:#e0a800; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         background:#1e1e1e; color:#ddd; display:flex; height:100vh; }
  #left { flex:0 0 50%; min-width:200px; display:flex; flex-direction:column;
          overflow:hidden; }
  #lefttop { flex:0 1 auto; overflow:auto; padding:16px; min-height:0; }
  /* flex-basis:0 so the bottom frame is sized only by leftover space, never by
     its console content (which would otherwise inflate the frame). */
  #leftbottom { flex:1 1 0; overflow:auto; padding:16px; min-height:80px;
                display:flex; flex-direction:column; }
  #lhsplitter { flex:0 0 6px; cursor:row-resize; background:#333;
                border-top:1px solid #444; border-bottom:1px solid #444; }
  #lhsplitter:hover, #lhsplitter.drag { background:#5a5a5a; }
  #splitter { flex:0 0 6px; cursor:col-resize; background:#333;
              border-left:1px solid #444; border-right:1px solid #444; }
  #splitter:hover, #splitter.drag { background:#5a5a5a; }
  #right { flex:1 1 0; min-width:200px; padding:16px; overflow:hidden;
           display:flex; flex-direction:column; }
  #panel { flex:1 1 0; overflow:auto; min-height:0; }
  #rhsplitter { display:none; flex:0 0 6px; cursor:row-resize; background:#333;
                margin-top:8px; border-top:1px solid #444; border-bottom:1px solid #444; }
  #right:has(#cmdconsole:not(.hide)) #rhsplitter { display:block; }
  #rhsplitter:hover, #rhsplitter.drag { background:#5a5a5a; }
  /* fixed default height + flex column so console output scrolls inside the
     frame instead of inflating it; drag #rhsplitter to resize. */
  #cmdconsole { flex:0 0 260px; border-top:1px solid #333; margin-top:8px;
                padding-top:8px; display:flex; flex-direction:column;
                min-height:0; overflow:hidden; }
  /* keep the header/help at natural height; the terminal box (#conterm)
     flexes and holds both the scrolling output and the inline prompt. */
  #conhdr, #conhelp { flex:0 0 auto; }
  #conhdr { font-weight:700; margin-bottom:4px; }
  #contarget { color:#8ec; }
  #conhelp { color:#789; font-size:10px; margin-bottom:6px; }
  #conreload { margin-left:8px; padding:3px 10px; cursor:pointer; }
  #conterm { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             margin-top:6px; padding:6px; background:#111; border:1px solid #444;
             border-radius:4px; overflow:hidden; cursor:text; }
  #conout { flex:1 1 auto; min-height:0; overflow:auto; margin:0;
            background:transparent; }
  #conpromptline { flex:0 0 auto; display:flex; align-items:center;
                   margin-bottom:4px; padding-bottom:4px;
                   border-bottom:1px solid #333; }
  #conprompt { color:#8ec; margin-right:6px; white-space:nowrap;
               font-family:monospace; }
  #conin { flex:1 1 auto; background:transparent; border:none; outline:none;
           color:#ddd; font-family:monospace; padding:0; }
  /* highlight of the most recent command + result, pinned at the top of the
     terminal box just under the prompt/input line (before the scrolling log) */
  /* cap + scroll the "last result" so a big command output (help/bd/event/klog)
     scrolls inside this box instead of ballooning past #conterm and getting
     clipped by its overflow:hidden (which left no scrollbar and squeezed the
     #conout log to zero height). */
  #conlast { flex:0 0 auto; margin-bottom:6px; padding:6px 8px; background:#161616;
             border:1px solid #444; border-radius:4px; font-family:monospace;
             font-size:11px; font-weight:700; white-space:pre-wrap;
             word-break:break-word; max-height:96px; overflow:auto; }
  #conlastlbl { color:#789; font-weight:700; margin-right:6px; }
  #conlastcmd { color:#8ec; font-weight:700; }
  #conlastres { color:#f5b942; font-weight:700; }
  body.resizing { cursor:col-resize; user-select:none; }
  body.vresizing { cursor:row-resize; user-select:none; }
  h1 { font-size:16px; margin:0 0 4px; }
  .sub { color:#888; font-size:12px; margin-bottom:12px; }
  #grid { display:grid; gap:6px; }
  .tile { border:1px solid #444; border-radius:6px; padding:8px; cursor:pointer;
          font-size:11px; line-height:1.35; transition:.1s; min-height:64px; }
  .tile.shim { background:var(--shim); }
  .tile.core { background:var(--core); }
  .tile:hover { outline:2px solid #fff6; }
  .tile.sel { outline:3px solid var(--sel); }
  .tile.sdmismatch { box-shadow:0 0 0 2px #d64545 inset; }
  .tile .badge.sdwarn { background:#5a2020; color:#ffb0b0; cursor:default; }
  .tile .loc { font-weight:700; font-size:12px; }
  .tile .badge { display:inline-block; background:#0006; border-radius:3px;
                 padding:0 4px; margin:1px 2px 0 0; font-size:10px; }
  .tile .badge { cursor:pointer; }
  .tile .badge:hover { outline:1px solid #fff; }
  /* persistent selection: the clicked channel badge stays highlighted (survives
     mouseout) until another item (tile or channel) is clicked. */
  .tile .badge.selbadge { outline:2px solid var(--sel); background:#0009; }
  /* peer channel badges of the same flow as the clicked channel: partner =
     opposite direction (the source/destination), coop = same direction
     (cooperating group, e.g. broadcast peers / ping-pong). Distinct dotted vs
     dashed lines and colors so they read differently from the solid selbadge. */
  .tile .badge.peerbadge-partner { outline:2px dotted #ffb347; background:#0009; }  /* source/dest = amber dotted */
  .tile .badge.peerbadge-coop    { outline:2px dashed #7ee081; background:#0009; }  /* same group  = green dashed */
  .tile.peer-recv { outline:3px solid #35d0e0; }          /* receivers = cyan  */
  .tile.peer-send { outline:3px dashed #e05fd0; }          /* senders  = magenta*/
  .tile.flowsel   { box-shadow:0 0 0 2px #fff inset; }     /* the clicked badge's tile */
  .legend { margin-top:14px; font-size:12px; color:#aaa; }
  .legend .sw { display:inline-block; width:12px; height:12px; border-radius:3px;
                vertical-align:-2px; margin-right:4px; }
  .panel h2 { font-size:14px; margin:14px 0 6px; border-bottom:1px solid #333;
              padding-bottom:4px; }
  .kv { font-size:13px; margin:2px 0; }
  .kv b { color:#9cc; }
  ul.sum { margin:4px 0; padding-left:18px; font-size:13px; }
  ul.sum li { margin:2px 0; }
  .contract { color:#c9a; font-size:12px; }
  /* supply/demand balance flags */
  .sdrow { margin:3px 0 6px; }
  .sdbadge { display:inline-block; font-size:10px; font-weight:700; padding:1px 6px;
             border-radius:3px; margin-right:6px; vertical-align:1px; }
  .sdbadge.sdok  { background:#1f4d2e; color:#8fe0a8; }
  .sdbadge.sdbad { background:#5a2020; color:#ff9a9a; }
  .sdbadge.sdna  { background:#3a3a3a; color:#bbb; }
  .sdmeta { color:#9aa; font-size:11px; }
  .sdfig  { color:#ccd; font-size:12px; margin:2px 0 0 2px; }
  .sdnote { color:#888; font-size:11px; font-style:italic; margin-left:2px; }
  pre.code { background:#111; border:1px solid #333; border-radius:6px; padding:10px;
             overflow:auto; font-size:12px; line-height:1.4; white-space:pre; }
  .lref { color:#888; font-size:12px; margin:4px 0; }
  /* code-piece file path banner shown at the top of the file (host.cc) frame */
  .codepath { color:#8ec; font-size:11px; margin:2px 0 8px; word-break:break-all;
              padding:4px 8px; background:#12252b; border:1px solid #244;
              border-radius:4px; }
  .codepath b { color:#6ab; font-weight:600; }
  .codepath .cpath { color:#bfe; -webkit-user-select:all; user-select:all; }
  .tabs { margin-top:8px; }
  .tab { display:inline-block; padding:4px 10px; border:1px solid #333;
         border-bottom:none; cursor:pointer; background:#252525; border-radius:6px 6px 0 0; }
  .tab.act { background:#111; color:#fff; }
  .tabbody { border:1px solid #333; padding:10px; border-radius:0 6px 6px 6px; }
  .hide { display:none; }
  .kw { color:#569cd6; } .fn { color:#dcdcaa; } .num { color:#b5cea8; }
  .cm { color:#6a9955; }
  .pann { color:#8a8; font-style:italic; margin-left:1ch; }
  .gap { color:#c8905a; font-style:italic; opacity:0.75; }
  .bdpretty { font-family:monospace; font-size:11.5px; line-height:1.4; color:#6a9955;
              background:#0f1a10; border-left:3px solid #4a5a2a; border-radius:4px;
              margin:2px 0 8px 22px; padding:6px 10px; white-space:pre; overflow-x:auto; }
  .placeholder { color:#666; margin-top:40px; }
  /* kernel channel<->argument mapping table (High level tab, core tiles) */
  .kmap { border-collapse:collapse; margin:6px 0 8px; font-size:12px; }
  .kmap th, .kmap td { border:1px solid #333; padding:3px 8px; text-align:left; }
  .kmap th { background:#252525; color:#bbb; font-weight:600; }
  .kmap td.arrow { color:#8ec; text-align:center; }
  .kmap .win { color:#dcdcaa; }
  .kmap .mlock { color:#6a9; }
  .kmap .morder { color:#c93; }
  .kmap .mnone { color:#a55; }
  /* net detail panel tables (stream-switch connections, hops, channels) */
  .rctbl { border-collapse:collapse; margin:6px 0 8px; font-size:12px; width:100%; }
  .rctbl th, .rctbl td { border:1px solid #333; padding:3px 7px; text-align:left;
                          white-space:nowrap; }
  .rctbl th { background:#252525; color:#bbb; font-weight:600; }
  .rctbl td:first-child { color:#9cc; font-family:monospace; }
  .dimtxt { color:#888; font-size:12px; }
  /* search chip shared style (used in #sr-chips) */
  .lk-chip { display:inline-flex; align-items:center; gap:4px;
             background:#2a1e0a; border:1px solid #c8905a; border-radius:12px;
             color:#f0c070; font-size:11px; padding:1px 8px 1px 6px;
             font-family:monospace; }
  .lk-chip .lk-x { cursor:pointer; color:#c8905a; font-size:13px;
                    line-height:1; margin-left:2px; }
  .lk-chip .lk-x:hover { color:#ff9a6c; }
  /* kernel source view (Low level tab, core tiles) */
  .khl { background:#3a3410; border-left:3px solid #e0a800; }
  .kshowall { margin:4px 0 8px; padding:3px 10px; font-size:12px; cursor:pointer;
              background:#252525; color:#ddd; border:1px solid #444; border-radius:4px; }
  .kshowall:hover { background:#333; }
  .kfileref { color:#8ec; font-size:11px; }
  /* merged "kernel code" sub-tab: stacked source/wrapper/bcf sections */
  .ksec { margin:0 0 6px; }
  .ksechdr { color:#8ec; font-size:11px; text-transform:uppercase; letter-spacing:.5px;
             margin:6px 0 2px; opacity:.8; }
  .ksecsep { border-top:1px dashed #444; margin:10px 0; }
  .subtabs { margin:2px 0 8px; }
  .subtab { display:inline-block; padding:2px 9px; border:1px solid #444; cursor:pointer;
            background:#252525; border-radius:4px; margin-right:6px; font-size:12px; }
  .subtab.act { background:#0d3a2a; color:#fff; border-color:#3a7d44; }
  .rline { font-family:monospace; font-size:12px; line-height:1.55; white-space:pre-wrap; }
  .lno { color:#666; user-select:none; margin-right:6px; }
  /* right-aligned gutter for the dfschedule Middle IR original line numbers */
  .mlno { color:#666; user-select:none; display:inline-block; min-width:3.2em;
          text-align:right; margin-right:10px; }
  .midctrls { margin:6px 0; }
  .midctrls button { margin-right:8px; padding:3px 10px; cursor:pointer; }
  .irfull { max-height:420px; overflow:auto; }
  .irln { display:block; }
  .irlno { color:#555; user-select:none; display:inline-block; min-width:3.6em;
           text-align:right; margin-right:10px; }
  .irhi { background:#3a3320; }              /* highlighted tile/channel line */
  .irhi .irlno { color:#e0a800; }
  .irhidden { display:block; }               /* revealed folded lines */
  .irhidden.hide { display:none; }           /* beats global .hide (equal spec) */
  .irfold { display:block; color:#7fb0ff; cursor:pointer; user-select:none;
            background:#1b2330; padding:0 6px; }
  .irfold:hover { background:#243043; }
  .irfold.irfold-open { color:#88cc88; }
  .kb { display:inline-block; border-radius:3px; padding:0 4px; margin-right:5px;
        font-size:10px; font-weight:700; vertical-align:1px; }
  .kb.bd_config{background:#4a5a2a;color:#dfffb0} .kb.createio{background:#2d4a6a;color:#cfe6ff}
  .kb.startio{background:#6a4a2d;color:#ffe0c0} .kb.lock{background:#5a2d5a;color:#f0c0f0}
  .kb.wait{background:#6a2d2d;color:#ffc0c0}
  .kb.enable_ooo{background:#2d5a5a;color:#c0f0f0}
  .kb.loop{background:#3a3a6a;color:#cfd0ff}
  .kb.loopidx{background:#33334d;color:#b0b0e0}
  .kb.loopend{background:#3a3a6a;color:#cfd0ff}
  .rnote{color:#9a9ad0;font-style:italic;margin-left:8px;font-size:11px;}
  /* ── live debug overlay (active only when served by schedule_debug_server) ── */
  #live { font-size:12px; flex:1 1 auto; display:flex; flex-direction:column;
          min-height:0; }
  #live label, #overlayctl label { cursor:pointer; }
  /* overlay controls live in the top-left array/partition view (after #grid) */
  #overlayctl { margin-top:8px; font-size:12px; }
  .ltab { display:inline-block; padding:2px 8px; margin:6px 4px 0 0; border:1px solid #444;
          border-radius:4px; background:#252525; cursor:pointer; font-size:11px; }
  .ltab.act { background:#0d2a3a; color:#fff; border-color:#2d6a9f; }
  #runbtn { margin-left:6px; padding:4px 12px; cursor:pointer; }
  #stopbtn { margin-left:6px; padding:4px 12px; cursor:pointer;
             background:#5a2d2d; color:#fdd; border:1px solid #843; border-radius:4px; }
  #loadlogbtn { margin-left:6px; padding:4px 12px; cursor:pointer; }
  #stopbtn:hover { background:#7a3535; }
  #runbtn:disabled, #stopbtn:disabled { opacity:.4; cursor:not-allowed; }
  #testconn:disabled, #conreload:disabled { opacity:.4; cursor:not-allowed; }
  #conin.disabled, #conin:disabled { opacity:.4; cursor:not-allowed; }
  #live label.disabled, #overlayctl label.disabled { opacity:.5; }
  #boardHost { background:#111; color:#ddd; border:1px solid #444; border-radius:4px;
               padding:3px 6px; font-family:monospace; }
  #livestatus { color:#8ab; font-size:11px; margin-top:6px; min-height:14px; }
  .consolebox { background:#0a0a0a; border:1px solid #333; border-radius:6px; padding:8px;
                font-size:11px; line-height:1.35; max-height:220px; overflow:auto;
                white-space:pre-wrap; word-break:break-word; max-width:100%;
                min-width:0; margin-top:8px; }
  /* the run/log console fills the resizable bottom-left region */
  #console { flex:1 1 auto; max-height:none; min-height:100px; }
  .livebar { display:block; margin-top:3px; font-size:9px; font-weight:700; color:#fff;
             border-radius:3px; padding:0 4px; text-align:center; letter-spacing:.5px; }
  /* ── view switcher (Grid / Device Map) ─────────────────────── */
  #viewswitcher { display:flex; gap:0; margin-bottom:10px; }
  .vsw { padding:4px 14px; font-size:12px; font-weight:600; border:1px solid #444;
         background:#252525; color:#888; cursor:pointer; transition:all .12s; }
  .vsw:first-child { border-radius:5px 0 0 5px; }
  .vsw:last-child  { border-radius:0 5px 5px 0; }
  .vsw.act { background:#0d2a3a; color:#7ec8e3; border-color:#2d6a9f; }
  /* ── Device Map panel ───────────────────────────────────────── */
  #devmap { display:none; flex-direction:column; }
  #devmap.show { display:flex; }
  #devmap-netlabel { font-size:10px; color:#888; margin-bottom:4px; user-select:none; }
  #devmap-netbar { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:6px; align-items:center; }
  .dm-chip { display:inline-flex; align-items:center; gap:4px; padding:2px 8px 2px 6px;
             border-radius:9999px; font-size:10px; font-weight:500;
             cursor:pointer; border:1px solid var(--stroke,#e4e4e433); user-select:none;
             transition:opacity .15s, background .15s; white-space:nowrap;
             background:transparent; color:var(--fg,#e4e4e4); }
  .dm-chip .chdot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
  .dm-chip.all-chip { border-color:#e4e4e433; }
  .dm-chip.all-chip.act { background:#e4e4e41e; border-color:#e4e4e488; }
  .dm-chip.net-chip.act { background:#e4e4e411; border-color:#e4e4e433; opacity:1; }
  .dm-chip.net-chip:not(.act) { opacity:0.25; }
  #devmap-vp { overflow:hidden; position:relative; height:calc(100vh - 280px); min-height:420px;
               border:1px solid #e4e4e420; border-radius:6px; background:#181818;
               cursor:grab; }
  #devmap-vp.panning { cursor:grabbing; }
  #devmap-canvas { position:absolute; top:0; left:0; transform-origin:0 0; }
  #devmap-hint { position:absolute; bottom:6px; right:8px; font-size:9px;
                 color:#e4e4e45e; pointer-events:none; letter-spacing:.2px; }
  #devmap-spacehint { position:absolute; top:6px; left:8px; font-size:9px;
                      color:#e4e4e45e; pointer-events:none; }
  #devmap-reset { position:absolute; top:6px; right:7px; font-size:10px; padding:2px 9px;
                  border:1px solid #e4e4e433; background:#e4e4e411; color:#e4e4e4bb;
                  border-radius:4px; cursor:pointer; z-index:10; }
  #devmap-reset:hover { background:#e4e4e41e; color:#e4e4e4; }
  #devmap-legend { display:flex; gap:10px; flex-wrap:wrap; margin-top:5px; font-size:10px;
                   color:#e4e4e45e; align-items:center; }
  .dml-item { display:flex; align-items:center; gap:4px; }
  .dml-swatch { width:16px; height:7px; border-radius:2px; }
  .dml-line { width:18px; height:0; }
  .dml-dot { width:7px; height:7px; border-radius:50%; }
  .dbgpanel { margin-top:10px; border:1px solid #333; border-radius:6px; padding:10px; }
  .dbgpanel h2 { margin-top:0; }
  .dbgpresets { margin-bottom:6px; }
  .dbgpresets .ltab { margin-top:0; }
  #dbgcmd { background:#111; color:#ddd; border:1px solid #444; border-radius:4px;
            padding:4px 6px; font-family:monospace; }
  #dbgrun { margin-left:6px; padding:3px 10px; cursor:pointer; }
  /* ── tabbed command console (aiegdb | LLM) ── */
  #contabs { flex:0 0 auto; margin-bottom:6px; }
  .contab { display:inline-block; padding:3px 12px; border:1px solid #444;
            border-bottom:none; cursor:pointer; background:#252525;
            border-radius:6px 6px 0 0; font-size:12px; margin-right:4px; }
  .contab.act { background:#111; color:#fff; }
  #conpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0; }
  #llmpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             position:relative; }
  /* ID rules above (specificity 1,0,0) beat the base .hide (0,1,0); this
     ID-qualified rule (1,1,0) lets the tab switch actually hide a pane. */
  #conpane.hide, #llmpane.hide { display:none; }
  /* LLM (embedded Claude Code) pane */
  #llmterm { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             margin-top:6px; padding:6px; background:#111; border:1px solid #444;
             border-radius:4px; overflow:hidden; cursor:text; }
  /* Scrolling message list */
  #llmmsg { flex:1 1 0; min-height:0; overflow-y:auto; padding:2px 0; }
  /* Per-message bubbles */
  .llm-msg { margin:3px 0; padding:5px 8px; border-radius:3px;
             font-family:monospace; font-size:12px; line-height:1.45;
             white-space:pre-wrap; word-break:break-word; }
  .llm-msg-you { background:#152015; border-left:2px solid #6ab; color:#cee; }
  .llm-msg-ai  { background:transparent; color:#ccc; border-left:2px solid #333; }
  .llm-msg-ctx { background:transparent; color:#555; font-style:italic;
                 border-left:2px solid #2a2a2a; font-size:11px; }
  /* Marker colors inside AI bubbles */
  .llm-msg-ai .llm-you       { color:#8ec; font-weight:bold; }
  .llm-msg-ai .llm-tool      { color:#7aa2f7; }
  .llm-msg-ai .llm-toolname  { color:#e0af68; font-weight:bold; }
  .llm-msg-ai .llm-toolresult{ color:#555; }
  .llm-msg-ai .llm-error     { color:#f7768e; font-weight:bold; }
  .llm-msg-ai .llm-file      { color:#9ece6a; }
  .llm-msg-ai .llm-line      { color:#e0af68; }
  /* Markdown inside AI bubbles */
  .llm-msg-ai .md-h      { color:#7dcfff; font-weight:bold; display:block; margin-top:4px; }
  .llm-msg-ai .md-bullet { color:#7aa2f7; }
  .llm-msg-ai strong     { color:#c0caf5; font-weight:bold; }
  .llm-msg-ai .md-code   { background:#1b1b2b; color:#e0af68; padding:0 3px; border-radius:3px; }
  .llm-msg-ai .md-block  { background:#0d0d16; border:1px solid #2a2a3a; border-radius:3px;
                            padding:5px 8px; margin:3px 0; overflow-x:auto; white-space:pre; }
  .llm-msg-ai .md-block code { background:none; padding:0; color:#c0caf5; }
  .llm-msg-ai .cm-keyword { color:#bb9af7; }
  .llm-msg-ai .cm-string  { color:#9ece6a; }
  .llm-msg-ai .cm-comment { color:#565f89; font-style:italic; }
  .llm-msg-ai .cm-number  { color:#ff9e64; }
  /* Thinking indicator */
  #llmthink { padding:4px 8px 2px; }
  .llm-dot  { display:inline-block; width:4px; height:4px; border-radius:50%;
              background:#555; margin-right:3px;
              animation:llmblink 1.1s ease-in-out infinite; }
  .llm-dot:nth-child(2){ animation-delay:.18s; }
  .llm-dot:nth-child(3){ animation-delay:.36s; }
  @keyframes llmblink{ 0%,80%,100%{opacity:.15} 40%{opacity:.8} }
  /* Input row */
  #llminline { flex:0 0 auto; display:flex; align-items:flex-start; margin-top:5px;
               border-top:1px solid #2a2a2a; padding-top:5px; }
  #llmprompt { color:#6ab; margin-right:6px; white-space:nowrap;
               font-family:monospace; padding-top:2px; }
  #llmin { flex:1 1 auto; background:transparent; border:none; outline:none;
           color:#ddd; font-family:monospace; padding:0; resize:none;
           overflow-y:auto; max-height:120px; line-height:1.4;
           font-size:inherit; }
  #llmsend, #llmreset { margin-left:6px; padding:3px 10px; cursor:pointer; }
  /* LLM password modal overlay (matches the #llmout dark palette). */
  #llmauth { position:absolute; inset:0; display:flex; align-items:center;
             justify-content:center; background:rgba(0,0,0,0.7); z-index:50; }
  /* ID rule (1,0,0) beats base .hide (0,1,0); this ID-qualified rule (1,1,0)
     lets the JS actually hide the modal (same trap as #llmpane.hide above). */
  #llmauth.hide { display:none; }
  #llmauthbox { background:#111; border:1px solid #444; border-radius:6px;
                padding:16px 18px; width:280px; font-family:monospace;
                color:#ddd; box-shadow:0 4px 20px rgba(0,0,0,0.6); }
  #llmauthbox .t { color:#8ec; font-weight:bold; margin-bottom:8px; }
  #llmauthin { width:100%; box-sizing:border-box; background:#0d0d16;
               border:1px solid #2a2a3a; border-radius:4px; color:#ddd;
               font-family:monospace; padding:5px 6px; outline:none; }
  #llmauthok { margin-top:10px; padding:4px 12px; cursor:pointer; }
  #llmautherr { color:#f7768e; font-size:11px; margin-top:6px; min-height:14px; }
  /* ── Search pane ── */
  #searchpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0; }
  #searchpane.hide { display:none; }
  #sr-inputrow { flex:0 0 auto; display:flex; align-items:center; gap:6px;
                 margin-bottom:6px; position:relative; }
  #sr-input { flex:1 1 auto; background:#111; border:1px solid #444; border-radius:4px;
              color:#ddd; font-size:12px; font-family:monospace; padding:4px 8px;
              outline:none; }
  #sr-input:focus { border-color:#7ac; }
  #sr-suggest { position:absolute; top:100%; left:0; right:0; z-index:200;
                background:#1a1a1a; border:1px solid #555; border-radius:0 0 6px 6px;
                max-height:200px; overflow-y:auto; font-size:12px; font-family:monospace;
                box-shadow:0 4px 12px rgba(0,0,0,0.5); }
  .sr-sug-item { padding:4px 10px; cursor:pointer; display:flex; gap:8px;
                 align-items:baseline; white-space:nowrap; overflow:hidden; }
  .sr-sug-item:hover, .sr-sug-item.sel { background:#2a2a2a; }
  .sr-sug-kind { color:#7ac; font-size:10px; min-width:52px; }
  .sr-sug-label { color:#ddd; overflow:hidden; text-overflow:ellipsis; }
  .sr-sug-match { color:#ffd200; }
  .sr-sug-tile { color:#666; font-size:10px; margin-left:auto; white-space:nowrap; }
  #sr-chips { flex:0 0 auto; display:flex; flex-wrap:wrap; gap:4px; margin-bottom:6px; }
  #sr-results { flex:1 1 0; overflow-y:auto; min-height:0; }
  #sr-results table { border-collapse:collapse; width:100%; font-size:12px; }
  #sr-results th, #sr-results td { border:1px solid #2a2a2a; padding:3px 7px;
                                    text-align:left; white-space:nowrap; }
  #sr-results th { background:#1c1c1c; color:#888; font-weight:600; font-size:11px; }
  #sr-results td:first-child { font-family:monospace; }
  .sr-group-hdr { padding:5px 7px 2px; color:#7ac; font-size:11px; font-weight:700;
                  background:#141414; border-bottom:1px solid #222; }
  .sr-empty { color:#555; font-size:12px; padding:8px; }
  .sr-stat { color:#666; font-size:11px; padding:2px 7px 6px; }
</style>
</head>
<body>
<div id="left">
  <div id="lefttop">
  <h1>AIE Schedule View</h1>
  <div class="sub" id="meta"></div>
  <div id="viewswitcher">
    <span class="vsw act" onclick="switchView('grid',this)">Grid</span>
    <span class="vsw" onclick="switchView('map',this)">Device Map</span>
  </div>
  <div id="grid"></div>
  <div id="devmap">
    <div id="devmap-netlabel">Click a stream to isolate its route in the map below.</div>
    <div id="devmap-netbar"></div>
    <div id="devmap-vp">
      <button id="devmap-reset" onclick="dmReset()">Reset view</button>
      <div id="devmap-spacehint">double-drag · space+drag · right-drag to pan · scroll to zoom · click tile to inspect</div>
      <div id="devmap-canvas"><svg id="devmap-svg"></svg></div>
      <div id="devmap-hint">col 0–3 · row 0 (shim) at bottom</div>
    </div>
    <div id="devmap-legend">
      <div class="dml-item"><div class="dml-swatch" style="background:#e4e4e41e;border:1px solid #e4e4e433"></div>SHIM = PL/NoC gateway</div>
      <div class="dml-item"><div class="dml-swatch" style="background:#e4e4e411;border:1px solid #e4e4e41f"></div>MEM = memory tiles</div>
      <div class="dml-item"><div class="dml-swatch" style="background:#e4e4e41e;border:1px solid #e4e4e433"></div>AIE = compute cores</div>
      <div class="dml-item"><div class="dml-line" style="border-top:2.5px solid #599ce7"></div>solid = stream-switch route</div>
      <div class="dml-item"><div class="dml-line" style="border-top:2px solid #599ce7;border-bottom:2px solid #599ce7;height:5px"></div><span style="color:#599ce7">&#9642;</span> double line + square = ping-pong window (kernel&harr;kernel)</div>
      <div class="dml-item"><div class="dml-line" style="border-top:2px dashed #599ce7"></div>dashed = direct shared-memory (DMA&harr;kernel)</div>
      <div class="dml-item"><div class="dml-dot" style="background:#599ce7;border:1.2px solid #181818"></div>● solid = injects into stream (source / contributor)</div>
      <div class="dml-item"><div class="dml-dot" style="background:#181818;border:2.2px solid #599ce7"></div>○ hollow = takes from stream (destination / tap)</div>
    </div>
  </div>
  <!-- live status overlay controls belong to the array/partition view: they
       drive the per-tile overlay rendered onto #grid above. -->
  <div id="overlayctl">
    <label><input type="checkbox" id="liveToggle"> Live status overlay</label>
    <div id="overlaytabs">
      <span class="ltab act" data-w="dma">DMA</span>
      <span class="ltab" data-w="cores">Cores</span>
      <span class="ltab" data-w="events">Events</span>
    </div>
    <div id="livestatus"></div>
  </div>
  <div id="gridlegend" class="legend">
    <div><span class="sw" style="background:var(--shim)"></span>shim (row 0, host&lt;-&gt;array)</div>
    <div><span class="sw" style="background:var(--core)"></span>core (compute)</div>
    <div>badge = channel dir (S2MM recv / MM2S send)</div>
    <div>click a badge to highlight its flow peers:</div>
    <div><span class="sw" style="background:#35d0e0"></span>receivers (S2MM) of the clicked flow</div>
    <div><span class="sw" style="background:#e05fd0"></span>senders (MM2S) of the clicked flow (dashed)</div>
    <div>peer channel badges of the clicked flow:</div>
    <div><span class="sw" style="border:2px dotted #ffb347;background:transparent"></span>source/destination channel (opposite dir, dotted)</div>
    <div><span class="sw" style="border:2px dashed #7ee081;background:transparent"></span>cooperating channel (same dir group, dashed)</div>
    <button id="gbtn" style="margin-top:10px">Show global / kernel-group code</button>
  </div>
  </div>
  <div id="lhsplitter" title="Drag to resize (top / bottom)"></div>
  <div id="leftbottom">
  <div id="live">
    <div id="devrow" style="margin-bottom:6px;">
      <label>Board:
        <select id="deviceSel">
          <option value="">&mdash; select device &mdash;</option>
<!--__PALMYRA_OPTION__-->          <option value="vek385">vek385</option>
        </select>
      </label>
      <input type="text" id="boardHost" class="hide"
             placeholder="vek385 board hostname" style="margin-left:6px;">
      <button id="testconn" disabled style="margin-left:6px;">Test connect</button>
      <button id="runbtn" disabled>Run test</button>
      <button id="stopbtn" disabled>Force stop</button>
      <button id="loadlogbtn">Load log</button>
      <span id="connstatus" style="margin-left:6px; font-size:11px; color:#8ab;"></span>
      <div id="connhint" class="hide" style="margin-top:6px; padding:6px 8px;
           font-size:11px; color:#f6c177; background:#2a1f14;
           border:1px solid #5a3d1a; border-radius:4px;">
        Connection failed. On the target test board, start the hw_server via xsdb:
        <code style="color:#e0def4; background:#1a1622; padding:1px 4px;
              border-radius:3px;">exec hw_server -stcp:0.0.0.0:3121</code>
      </div>
    </div>
    <pre id="console" class="consolebox hide"></pre>
  </div>
  </div>
</div>
<div id="splitter" title="Drag to resize"></div>
<div id="right">
  <div id="panel" class="panel">
    <div class="placeholder">Click a tile or a net (in device map) to see details.</div>
  </div>
  <div id="rhsplitter" title="Drag to resize (panel / console)"></div>
  <div id="cmdconsole" class="hide">
    <div id="contabs">
      <span class="contab act" data-pane="conpane">aiegdb</span>
      <span class="contab" data-pane="llmpane">LLM</span>
      <span class="contab" data-pane="searchpane">Search</span>
    </div>
    <div id="conpane">
    <div id="conhdr">aiegdb &mdash; <span id="contarget">partition</span>
      <button id="conreload" title="kill + restart aiegdb.py (reloads edited code)">Reload aiegdb.py</button></div>
    <div id="conhelp">aiegdb commands: target tile 0 0 | target channel mm2s0 | dma status |
      bd | event | pc | dma counter | reg read OFF | up | top | help</div>
    <div id="conterm">
      <div id="conpromptline"><span id="conprompt">partition&gt;</span><input id="conin"
        placeholder="type an aiegdb command, press Enter (e.g. dma status, help)"></div>
      <div id="conlast"><span id="conlastlbl">last:</span><span id="conlastcmd">&mdash;</span>
        <span id="conlastres"></span></div>
      <pre id="conout" class="code">(aiegdb console &mdash; click a tile or type 'help')</pre>
    </div>
    </div>
    <div id="llmpane" class="hide">
      <div id="conhdr">LLM &mdash; embedded Claude Code (repo-aware AIE debug agent)
        <button id="llmreset" title="kill + respawn claude (new conversation)">New chat</button></div>
      <div id="conhelp">Ask about this schedule, the applog, or the codebase.
        Tool calls appear as [tool: ...] markers. Streams live.</div>
      <div id="llmterm">
        <div id="llmmsg"></div>
        <div id="llmthink" class="hide"><span class="llm-dot"></span><span class="llm-dot"></span><span class="llm-dot"></span></div>
        <div id="llminline"><span id="llmprompt">you&gt;</span><textarea id="llmin"
          rows="1" placeholder="ask about this design — Enter to send, Shift+Enter for newline"></textarea>
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
  return '<div class="kv"><b>channel \u2194 kernel argument (by BD buffer address):</b></div>'+
         table+fn+hint;
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
  const hdr = '<div class="lref">kernel: <b>'+esc(src.file)+'</b> &mdash; '+
    esc(src.function)+'() lines '+src.start_line+'-'+src.end_line+'</div>';
  const full = (src.lines||[]).map(r => {
    const hlcls = (refset && refset[r.line]) ? ' khl' : '';
    return '<div class="rline'+hlcls+'"><span class="lno">L'+r.line+'</span>'+
      hl(r.code)+'</div>';
  }).join('');
  if(focused && refset && param){
    // Show the related lines; fold the rest into "//line a-b" markers.
    const piece = renderKFolded(src.lines||[], refset, 'src');
    return hdr +
      '<div class="kv">channel <b>'+esc(ch.direction)+esc(''+ch.channel)+
        '</b> \u2192 arg'+argn+' <span class="win">'+esc(param.name)+'</span> ('+
        esc(param.dir)+')</div>' +
      '<button class="kshowall">Show all</button>' +
      '<div id="kern-piece">'+
        (piece||'<div class="placeholder">(no direct uses found)</div>')+'</div>' +
      '<div id="kern-full" class="hide">'+full+'</div>';
  }
  return hdr + '<div id="kern-full">'+full+'</div>';
}
// Resolve the kernel window a focused channel maps to (via kernel_match), or
// null when not focused / no match. Shared by the kernel.cc + .bcf sub-tabs.
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
  const hdr = '<div class="lref">kernel: <b>'+esc(k.file||'kernel.cc')+'</b>'+
    (win ? ' &mdash; window <span class="win">'+esc(win.name)+'</span>'+
      ' (buffers '+esc((win.buffers||[]).join(', ')||'-')+')' : '')+'</div>';
  // Focused (window matched): show only the window's lines, fold the rest.
  const body = win
    ? renderKFolded(k.kernel_lines||[], hset, 'kcc')
    : (k.kernel_lines||[]).map(r =>
        '<div class="rline"><span class="lno">L'+r.line+'</span>'+
        hl(r.code)+'</div>').join('');
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
  const hdr = '<div class="lref">bcf: <b>'+esc(b.file)+'</b>'+
    (win ? ' &mdash; buffers for <span class="win">'+esc(win.name)+'</span>' : '')+'</div>';
  // Focused (window matched): show only this window's buffer lines, fold the rest.
  const body = win
    ? renderKFolded(b.lines||[], hset, 'bcf')
    : (b.lines||[]).map(r =>
        '<div class="rline"><span class="lno">L'+r.line+'</span>'+
        hl(r.code)+'</div>').join('');
  return hdr + '<div>'+body+'</div>';
}
// Render a list of {line,code} rows with only the highlighted (hset) lines shown
// and each run of non-related lines collapsed into a clickable "//line a-b"
// fold marker (default collapsed). `prefix` keeps fold ids unique across the
// stacked kernel-code sections. Reuses the shared fold machinery (wireFolds /
// setFold / foldLabel).
function renderKFolded(lines, hset, prefix){
  const N = lines.length;
  const row = r => '<div class="rline'+(hset[r.line]?' khl':'')+'">'+
    '<span class="lno">L'+r.line+'</span>'+hl(r.code)+'</div>';
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
// Merged "kernel code" sub-tab content. Sections are stacked in dependency
// order (bcf buffer addresses -> generated kernel.cc wrapper -> conv2d_spatial.cc
// source) so the reader follows the data from where it lands in tile memory up
// to where the kernel body consumes it.
function renderKernelCode(t, ch, focused){
  const sections = [];
  const kv = tileKernel(t), b = tileBcf(t);
  if(b && b.lines)
    sections.push(['buffer address map', renderBcf(t, ch, focused)]);
  if(kv && kv.kernel_lines)
    sections.push(['generated wrapper', renderKernelCC(t, ch, focused)]);
  if(kv && kv.source)
    sections.push(['kernel source', renderKernelSource(t, ch, focused)]);
  if(!sections.length)
    return '<div class="placeholder">(no kernel code found)</div>';
  return sections.map(s =>
    '<div class="ksec"><div class="ksechdr">'+esc(s[0])+'</div>'+s[1]+'</div>'
  ).join('<div class="ksecsep"></div>');
}
function hl(code){
  let s = esc(code);
  s = s.replace(/\/\*[\s\S]*?\*\//g, m=>'<span class="cm">'+m+'</span>');
  s = s.replace(/\b(void|int|int32_t|int64_t|size_t|for|return|uint8_t)\b/g,'<span class="kw">$1</span>');
  s = s.replace(/\b(__Runtime_[a-zA-Z0-9_]+|XAie_[A-Za-z0-9_]+|__runtime_[a-z_]+)\b/g,'<span class="fn">$1</span>');
  s = s.replace(/\b(\d+)\b/g,'<span class="num">$1</span>');
  return s;
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
  const hdr = '<div class="lref">'+esc(ir.name||'')+' &mdash; '+N+
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
    const h = box.querySelector('.irhidden[data-fold="'+f.dataset.fold+'"]');
    if (!h) return;
    f.onclick = () => {
      setFold(f, h, !h.classList.contains('hide'));   // toggle to opposite state
      if (onChange) onChange();
    };
  });
}

const g = DATA.grid;
document.getElementById('meta').textContent =
  DATA.source.function + '  (host.cc lines ' + DATA.source.function_lines[0] +
  '-' + DATA.source.function_lines[1] + ')  grid ' + g.cols + '\u00d7' + g.rows +
  ((g.startcol !== null && g.startcol !== undefined)
    ? '  partition startcol=' + g.startcol + ' (phys_col = col + ' + g.startcol + ')'
    : '');

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
      const b = ev.target.closest('.badge');
      if (b && b.dataset.idx !== undefined) {
        const ch = (t.dma_channels||[])[+b.dataset.idx];
        select(t, cell, ch, b);
        highlightFlow(+b.dataset.flow, t.loc, b);
      } else {
        select(t, cell);
        clearPeers();
      }
    };
    grid.appendChild(cell);
  }
});

// ── View switcher (Grid / Device Map) ────────────────────────
function switchView(name, btn){
  document.querySelectorAll('.vsw').forEach(b=>b.classList.remove('act'));
  btn.classList.add('act');
  const showGrid = name==='grid';
  document.getElementById('grid').style.display = showGrid ? '' : 'none';
  document.getElementById('overlayctl').style.display = showGrid ? '' : 'none';
  document.getElementById('gridlegend').style.display = showGrid ? '' : 'none';
  document.getElementById('devmap').classList.toggle('show', !showGrid);
  if(!showGrid) buildDeviceMap();
}

// ── Device Map (SVG, pan/zoom) ────────────────────────────────
const DM_COLORS = [
  '#7BAFE9','#81A1C1','#9386F2','#B48EAD',
  '#3FA266','#F1B467','#DD7F76','#FC6B83',
  '#88C0D0','#A3BE8C','#EBCB8B','#BF616A',
];
const dmFlowIds = Object.keys(flowMembers).map(Number).sort((a,b)=>a-b);
function dmColor(fi){ return DM_COLORS[dmFlowIds.indexOf(fi)%DM_COLORS.length]; }

let dmActiveFi = -1;
let dmBuilt = false;

// Pan/zoom state
let dmTx=20, dmTy=20, dmScale=0.9, dmDragging=false, dmLx=0, dmLy=0;
let dmSpaceHeld=false;

function dmApply(){
  document.getElementById('devmap-canvas').style.transform=
    'translate('+dmTx+'px,'+dmTy+'px) scale('+dmScale+')';
}
function dmReset(){
  const vp=document.getElementById('devmap-vp');
  const svg=document.getElementById('devmap-svg');
  const vpW=vp.clientWidth, vpH=vp.clientHeight;
  const svgW=parseFloat(svg.getAttribute('width')||'400');
  const svgH=parseFloat(svg.getAttribute('height')||'300');
  if(svgW>0&&svgH>0){
    const scaleX=vpW/svgW, scaleY=vpH/svgH;
    dmScale=Math.min(scaleX,scaleY)*0.92;
    dmTx=(vpW-svgW*dmScale)/2;
    dmTy=(vpH-svgH*dmScale)/2;
  } else { dmTx=20; dmTy=20; dmScale=0.9; }
  dmApply();
}

(function initPanZoom(){
  const vp = document.getElementById('devmap-vp');

  // Spacebar: suppress page scroll when devmap is active (capture phase).
  // Skip when focus is on an input/textarea so the search bar can type spaces.
  document.addEventListener('keydown',e=>{
    if(e.code!=='Space'||e.repeat) return;
    if(!document.getElementById('devmap').classList.contains('show')) return;
    const tag=(document.activeElement||{}).tagName||'';
    if(tag==='INPUT'||tag==='TEXTAREA') return;
    e.preventDefault();
  }, true);

  // Scroll to zoom (always)
  vp.addEventListener('wheel', e=>{
    e.preventDefault();
    const r=vp.getBoundingClientRect();
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    const d=e.deltaY<0?1.12:1/1.12;
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
    // Derive a representative tkey from the first DMA tile
    const dmaTiles=(p.dma_tiles||[]);
    const repTile=dmaTiles[0]||null;
    const tkey=repTile?repTile[0]+','+repTile[1]:'0,0';

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
      tileLockKeys.add(h.tkey);
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
  buildDeviceMap();
  if(srSearchTerms.size>0)
    llmPushCtx('[context] Search: pinned "'+[...srSearchTerms].join('", "')+'"');
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
    x.onclick=()=>{ srSearchTerms.delete(term); srRenderChips(); srRenderResults(); buildDeviceMap(); };
    chip.appendChild(x);
    wrap.appendChild(chip);
  });
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
      +(hits.length?' <span style="color:#555;font-weight:400">('+hits.length+' match'+(hits.length!==1?'es':'')+')</span>':'')
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
        +'<td>'+(h.fi!=null?'f'+h.fi:'—')+'</td>'
        +'<td style="font-family:monospace;max-width:120px;overflow:hidden;text-overflow:ellipsis">'+esc(h.labelRaw)+'</td>'
        +'<td style="color:#666;font-size:11px">'+esc(h.description)+'</td>'
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

function buildNetBar(){
  const bar=document.getElementById('devmap-netbar');
  bar.innerHTML='';
  // "All nets" chip
  const allc=document.createElement('button');
  allc.className='dm-chip all-chip'+(dmActiveFi===-1?' act':'');
  allc.textContent='All nets';
  allc.onclick=()=>dmSelectNet(-1);
  bar.appendChild(allc);
  // Per-flow chips — derive label from flow_summary if available
  const fsSummary={};
  (DATA.flow_summary||[]).forEach(f=>{ fsSummary[f.flow_index]=f; });
  const fiDir={};
  (DATA.comm_paths||[]).forEach(p=>{ fiDir[p.flow_index]=p.direction; });
  dmFlowIds.forEach(fi=>{
    const isAct=dmActiveFi===-1||dmActiveFi===fi;
    const chip=document.createElement('button');
    chip.className='dm-chip net-chip'+(isAct?' act':'');
    chip.dataset.fi=fi;
    const dot=document.createElement('span');
    dot.className='chdot'; dot.style.background=dmColor(fi);
    chip.appendChild(dot);
    const dir=fiDir[fi]==='push'?'→':'←';
    chip.appendChild(document.createTextNode('net'+fi+' '+dir));
    chip.style.color=dmColor(fi);
    chip.title='Click to isolate net '+fi+' in map';
    chip.onclick=()=>dmSelectNet(dmActiveFi===fi?-1:fi);
    bar.appendChild(chip);
  });
}

function dmSelectNet(fi){
  dmActiveFi=fi;
  document.querySelector('.dm-chip.all-chip').classList.toggle('act',fi===-1);
  document.querySelectorAll('.dm-chip.net-chip').forEach(c=>{
    c.classList.toggle('act',fi===-1||parseInt(c.dataset.fi)===fi);
  });
  buildDeviceMap();
  if(fi===-1){ renderNetDetail(null); llmPushCtx(null); }
  else {
    const path=(DATA.comm_paths||[]).find(p=>p.flow_index===fi)||null;
    renderNetDetail(path);
    if(path){
      const prod=path.producer?(path.producer.gmio_name||path.producer.logical_name||'?'):'?';
      const cons=path.consumer?(path.consumer.gmio_name||path.consumer.logical_name||'?'):'?';
      llmPushCtx('[context] Selected net/flow f'+fi+' — producer: '+prod+', consumer: '+cons
        +(path.direction?' ('+path.direction+')':''));
    }
  }
}

function renderNetDetail(p){
  const panel=document.getElementById('panel');
  if(!p){
    // Deselected — restore placeholder
    panel.innerHTML='<div class="placeholder">Click a tile or net for details</div>';
    return;
  }
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
      const bd=(c.bd_chain||[])[0]||{};
      const acq=(bd.acquire_lock||[])[0]||{};
      const rel=(bd.release_lock||[])[0]||{};
      const lock=(acq.id!=null||rel.id!=null)?' lock '+acq.id+'/'+rel.id:'';
      chanRows+='<tr><td>('+t.loc[0]+','+t.loc[1]+')</td><td>'+esc(t.type)+'</td>'
        +'<td>'+esc(c.direction)+' ch'+c.channel+'</td>'
        +'<td>'+(bd.len!=null?bd.len+'B':'?')+'</td>'
        +'<td>'+(c.bd_chain||[]).length+'</td>'
        +'<td>'+esc(lock)+'</td>'
        +(c.kernel_port?'<td>'+esc(c.kernel_port)+'</td>':'<td>-</td>')
        +'</tr>';
    });
  });
  const chanTable=chanRows
    ?'<table class="rctbl"><thead><tr><th>tile</th><th>type</th><th>ch</th>'
      +'<th>len</th><th>BDs</th><th>lock acq/rel</th><th>port</th></tr></thead>'
      +'<tbody>'+chanRows+'</tbody></table>'
    :'<div class="placeholder">(no DMA channels on participating tiles)</div>';

  // ── Stream-switch connections (routing_connections) ───────────────────────
  let swRows='';
  const connKindLabel={'circuit_connect':'circuit','packet_connect':'packet',
    'shim_ext_to_aie':'shim→AIE','shim_aie_to_ext':'AIE→shim'};
  (p.routing_connections||[]).forEach(c=>{
    const t=c.tile||{};
    const kind=connKindLabel[c.kind]||esc(c.kind);
    let detail='';
    if(c.slave&&c.master) detail=esc(c.slave.dir)+'['+c.slave.idx+'] → '+esc(c.master.dir)+'['+c.master.idx+']';
    else if(c.stream_id!=null) detail='stream_id='+c.stream_id;
    swRows+='<tr><td>('+t.col+','+t.row+')</td><td>'+kind+'</td><td>'+detail+'</td></tr>';
  });
  const swTable=swRows
    ?'<table class="rctbl"><thead><tr><th>tile</th><th>kind</th><th>ports</th></tr></thead>'
      +'<tbody>'+swRows+'</tbody></table>'
    :'<div class="placeholder">(no routing data)</div>';

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

  panel.innerHTML=
    '<h2>'+dot+'net'+fi+' &mdash; '+esc(dir)+'</h2>'+
    '<div class="tabs">'+
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
      '</div>'+
      '<div id="tab-nd-hops" class="hide">'+
        '<div class="lref">hop-by-hop path from dmaphopprovenacemap</div>'+
        hopTable+
      '</div>'+
    '</div>';

  // Attach tab switching (reuses same logic as tile panel)
  panel.querySelectorAll('.tab').forEach(tab=>{
    tab.onclick=()=>{
      panel.querySelectorAll('.tab').forEach(t=>t.classList.remove('act'));
      tab.classList.add('act');
      const id='tab-'+tab.dataset.t;
      panel.querySelectorAll('.tabbody>div').forEach(d=>{
        d.classList.toggle('hide',d.id!==id);
      });
    };
  });
}

function svgN(tag,attrs){
  const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const[k,v] of Object.entries(attrs)) el.setAttribute(k,v);
  return el;
}
function svgT(el, txt){ el.textContent=txt; return el; }

// ── Hover tooltip state ────────────────────────────────────────
let dmTooltipEl=null;
function dmShowTip(clientX, clientY, lines){
  dmHideTip();
  const d=document.createElement('div');
  d.style.cssText='position:fixed;z-index:9999;pointer-events:none;'
    +'background:#252525;border:1px solid #e4e4e433;border-radius:4px;'
    +'padding:5px 9px;font:10px/1.5 monospace;color:#e4e4e4;'
    +'white-space:nowrap;box-shadow:0 2px 10px #0008;';
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

function buildDeviceMap(){
  if(!DATA.tiles||!DATA.tiles.length){ console.warn('buildDeviceMap: no tiles'); return; }
  buildNetBar();
  const svg=document.getElementById('devmap-svg');
  svg.innerHTML='';

  // Layout: compute tile size to fill the viewport well at fit-zoom.
  // Gaps are 30% of tile size so tiles dominate the space.
  const vp=document.getElementById('devmap-vp');
  const vpW=vp.clientWidth||800, vpH=vp.clientHeight||600;
  const ML=44, MT=28, MR=16, MB=12;

  // Compute col/row counts first (need tileMap for this)
  // Defer sizing until after tileMap is built — use temp values here,
  // then recalculate SVG size after allCols/allRows are known.
  const TW=148, TH=56, GX=24, GY=16;
  const COLSTEP=TW+GX, ROWSTEP=TH+GY;

  // Merge DATA.tiles with every tile referenced in comm_paths (waypoints + hops)
  const tileMap={};
  DATA.tiles.forEach(t=>{ tileMap[t.loc[0]+','+t.loc[1]]=t; });
  (DATA.comm_paths||[]).forEach(p=>{
    // Register tiles from both edges and the tiles list
    (p.tiles||[]).forEach(([c,r])=>{
      const k=c+','+r;
      if(!tileMap[k]) tileMap[k]={loc:[c,r],type:'mem',dma_channels:[]};
    });
    (p.edges||[]).forEach(([src,dst])=>{
      [[src[0],src[1]],[dst[0],dst[1]]].forEach(([c,r])=>{
        const k=c+','+r;
        if(!tileMap[k]) tileMap[k]={loc:[c,r],type:'mem',dma_channels:[]};
      });
    });
  });

  // Synthesize MEM tiles (rows 1–2) for columns that have a shim (row 0) and a
  // core (row 3+). The stream-switch path always traverses these rows physically
  // even when the abstract hop chain skips them.
  const shimCols=new Set(), coreCols=new Set();
  Object.values(tileMap).forEach(t=>{
    if(t.loc[1]===0) shimCols.add(t.loc[0]);
    if(t.loc[1]>=3) coreCols.add(t.loc[0]);
  });
  const memCols=new Set([...shimCols].filter(c=>coreCols.has(c)));
  memCols.forEach(c=>{
    [1,2].forEach(r=>{
      const k=c+','+r;
      if(!tileMap[k]) tileMap[k]={loc:[c,r],type:'mem',dma_channels:[]};
    });
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

  // Per-flow (ox,oy) offsets — spreads parallel polylines apart so they're all visible.
  // Mirrors the reference: evenly stepped from -MAX to +MAX across all flows.
  const N=dmFlowIds.length;
  const dmOx={}, dmOy={};
  const OX_STEP=8, OY_STEP=5;           // pixels between adjacent lines
  const OX_MAX=OX_STEP*(N-1)/2;
  dmFlowIds.forEach((fi,i)=>{
    dmOx[fi]=Math.round((i-(N-1)/2)*OX_STEP);
    dmOy[fi]=Math.round((i-(N-1)/2)*OY_STEP);
  });

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
    if(r<=2)  return 'mem';
    return 'core';
  };
  const usedKeys=new Set(DATA.tiles.map(t=>t.loc[0]+','+t.loc[1]));
  let dmSelKey=null;
  const tileGroups={};

  allTiles.forEach(t=>{
    const [tc,tr]=t.loc;
    const key=tc+','+tr;
    const ttype=tileType(tr);
    const used=usedKeys.has(key);
    const fill=ttype==='mem'?'var(--fill,#e4e4e411)':'var(--fill2,#e4e4e41e)';
    const stroke=used?'var(--stroke,#e4e4e433)':'var(--stroke2,#e4e4e41f)';
    const baseOp=used?1:0.32;

    const g=svgN('g',{opacity:baseOp,cursor:'pointer',class:'dm-tile','data-key':key});
    tileGroups[key]=g;

    const rect=svgN('rect',{x:tx(tc),y:ty(tr),width:TW,height:TH,rx:'6',
      fill,stroke,'stroke-width':'1'});
    g.appendChild(rect);

    // Coord top-left, type badge top-right
    const typStr=ttype==='shim'?'SHIM':ttype==='mem'?'MEM':'AIE';
    g.appendChild(svgT(svgN('text',{x:tx(tc)+6,y:ty(tr)+13,
      'font-size':'9','font-family':'monospace',fill:'#e4e4e848'}),
      '('+tc+','+tr+')'));
    g.appendChild(svgT(svgN('text',{x:tx(tc)+TW-6,y:ty(tr)+13,
      'text-anchor':'end','font-size':'8','font-family':'monospace',fill:'#e4e4e43a'}),
      typStr));

    // Show only terminal DMA channels (S2MM=input, MM2S=output) — not pass-through.
    // Each line: colored arrow + "f{fi}" compactly at bottom of tile.
    if(used&&t.dma_channels&&t.dma_channels.length){
      const chans=dmActiveFi===-1?t.dma_channels
        :t.dma_channels.filter(ch=>ch.flow_index===dmActiveFi);
      // Split into inputs (S2MM) and outputs (MM2S)
      const ins=chans.filter(ch=>ch.direction==='S2MM');
      const outs=chans.filter(ch=>ch.direction==='MM2S');
      // Render inputs on left half, outputs on right half, centered vertically
      const midY=ty(tr)+TH/2+4;
      ins.slice(0,3).forEach((ch,i)=>{
        const fc=dmColor(ch.flow_index);
        const y=midY+(i-(ins.length-1)/2)*11;
        // Arrow pointing in: ▶ f{fi}
        g.appendChild(svgT(svgN('text',{x:tx(tc)+7,y,
          'font-size':'8.5','font-family':'monospace',fill:fc}),
          '▶ f'+ch.flow_index));
      });
      outs.slice(0,3).forEach((ch,i)=>{
        const fc=dmColor(ch.flow_index);
        const y=midY+(i-(outs.length-1)/2)*11;
        // Arrow pointing out: f{fi} ▶
        g.appendChild(svgT(svgN('text',{x:tx(tc)+TW-7,y,
          'text-anchor':'end','font-size':'8.5','font-family':'monospace',fill:fc}),
          'f'+ch.flow_index+' ▶'));
      });
    }

    g.addEventListener('mouseenter',e=>{
      rect.setAttribute('stroke','#e4e4e488');
      rect.setAttribute('stroke-width','1.5');
      const lines=['('+tc+','+tr+') '+typStr];
      if(t.dma_channels&&t.dma_channels.length){
        const vis=dmActiveFi===-1?t.dma_channels
          :t.dma_channels.filter(ch=>ch.flow_index===dmActiveFi);
        // Terminal channels
        vis.filter(ch=>ch.direction==='S2MM').forEach(ch=>lines.push('in  f'+ch.flow_index));
        vis.filter(ch=>ch.direction==='MM2S').forEach(ch=>lines.push('out f'+ch.flow_index));
        // Pass-through flows (edge touches this tile but it's not a DMA terminal)
        const termFis=new Set(vis.map(ch=>ch.flow_index));
        const tileInEdges=p=>(p.edges||[]).some(e=>
          (e[0][0]===tc&&e[0][1]===tr)||(e[1][0]===tc&&e[1][1]===tr));
        const thru=(DATA.comm_paths||[]).filter(p=>{
          if(dmActiveFi!==-1&&p.flow_index!==dmActiveFi) return false;
          return tileInEdges(p)&&!termFis.has(p.flow_index);
        });
        if(thru.length) lines.push('── pass-through ──');
        thru.slice(0,4).forEach(p=>lines.push('    f'+p.flow_index+' ('+p.direction+')'));
      } else {
        // Ghost tile — only pass-through
        const tileInEdgesG=p=>(p.edges||[]).some(e=>
          (e[0][0]===tc&&e[0][1]===tr)||(e[1][0]===tc&&e[1][1]===tr));
        const thru=(DATA.comm_paths||[]).filter(p=>{
          if(dmActiveFi!==-1&&p.flow_index!==dmActiveFi) return false;
          return tileInEdgesG(p);
        });
        if(thru.length) thru.slice(0,4).forEach(p=>lines.push('pass f'+p.flow_index));
        else lines.push('routing tile');
      }
      dmShowTip(e.clientX, e.clientY, lines);
    });
    g.addEventListener('mousemove',e=>{
      if(!dmTooltipEl) return;
      const tw=dmTooltipEl.offsetWidth, th=dmTooltipEl.offsetHeight;
      let lx=e.clientX+14, ly=e.clientY-th-10;
      if(lx+tw>window.innerWidth-8) lx=e.clientX-tw-14;
      if(ly<8) ly=e.clientY+14;
      dmTooltipEl.style.left=lx+'px'; dmTooltipEl.style.top=ly+'px';
    });
    g.addEventListener('mouseleave',()=>{
      if(dmSelKey!==key){
        rect.setAttribute('stroke',stroke);
        rect.setAttribute('stroke-width','1');
      }
      dmHideTip();
    });
    g.addEventListener('click',e=>{
      if(dmDragging) return;
      if(dmSelKey&&tileGroups[dmSelKey]){
        const prev=tileGroups[dmSelKey];
        const pr=prev.querySelector('rect');
        if(pr){ pr.setAttribute('stroke','var(--stroke,#e4e4e433)'); pr.setAttribute('stroke-width','1'); }
      }
      if(dmSelKey===key){ dmSelKey=null; return; }
      dmSelKey=key;
      rect.setAttribute('stroke','#599ce7'); rect.setAttribute('stroke-width','2');
      const match=DATA.tiles.find(dt=>dt.loc[0]===tc&&dt.loc[1]===tr);
      if(match){
        const cells=document.querySelectorAll('#grid .tile');
        let found=null;
        cells.forEach(cell=>{
          const locEl=cell.querySelector('.loc');
          if(locEl&&locEl.textContent==='('+tc+','+tr+')') found=cell;
        });
        select(match, found||g);
      }
    });
    svg.appendChild(g);
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
    if(dr>0)       return [cx(fc), ty(fr),    cx(tc), ty(tr)+TH];
    /* dr<0 */     return [cx(fc), ty(fr)+TH, cx(tc), ty(tr)   ];
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
  // Drawn here (after edgeCoords/shIndex/shOffset/dmOx/dmOy are all defined)
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

    // Stream edge halos — match the per-flow (ox,oy) spread offset used by LAYER 4.
    (DATA.comm_paths||[]).forEach(p=>{
      if(!flowLockFis.has(p.flow_index)) return;
      const fi=p.flow_index;
      const ox=dmOx[fi]||0, oy=dmOy[fi]||0;
      (p.edges||[]).forEach(e=>{
        const [fc,fr]=e[0],[tc,tr]=e[1];
        svg.appendChild(svgN('line',{
          x1:cx(fc)+ox,y1:cy(fr)+oy,x2:cx(tc)+ox,y2:cy(tr)+oy,
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
    const dim=dmActiveFi!==-1&&dmActiveFi!==fi;
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
        svg.appendChild(svgN('line',{
          x1, y1, x2, y2,
          stroke:color,
          'stroke-width':'1.5',
          'stroke-opacity':'0.6',
          'stroke-dasharray':'5 3',
          'stroke-linecap':'round',
          'marker-end':'url(#ar-'+fi+')'}));
      } else {
        const dx = vertical ? RAIL : 0;   // rails offset perpendicular to the link
        const dy = vertical ? 0 : RAIL;
        [-1, 1].forEach(s=>{
          svg.appendChild(svgN('line',{
            x1:x1+s*dx, y1:y1+s*dy, x2:x2+s*dx, y2:y2+s*dy,
            stroke:color,
            'stroke-width':'1.6',
            'stroke-opacity':'0.8',
            'stroke-linecap':'round',
            'marker-end':'url(#ar-'+fi+')'}));
        });
        const mx=(x1+x2)/2, my=(y1+y2)/2;
        svg.appendChild(svgN('rect',{
          x:mx-BUFSZ/2, y:my-BUFSZ/2, width:BUFSZ, height:BUFSZ,
          fill:color, 'fill-opacity':'0.9',
          stroke:'#181818', 'stroke-width':'0.8'}));
      }
      // Wide transparent hit area (covers both rails for window links).
      const hit=svgN('line',{x1,y1,x2,y2,stroke:'transparent','stroke-width':'12',
        cursor:'pointer','pointer-events':'stroke'});
      hit.addEventListener('click',ev=>{
        if(dmDragging) return;
        ev.stopPropagation();
        dmSelectNet(dmActiveFi===fi?-1:fi);
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
    const ox=dmOx[fi]||0, oy=dmOy[fi]||0;
    const edges=p.edges||[];
    const srcKeys=new Set(edges.map(e=>e[0][0]+','+e[0][1]));
    edges.forEach(e=>{
      const [fc,fr]=e[0], [tc,tr]=e[1];
      const isTerminal=!srcKeys.has(tc+','+tr);
      const x1=cx(fc)+ox, y1=cy(fr)+oy, x2=cx(tc)+ox, y2=cy(tr)+oy;
      svg.appendChild(svgN('line',{
        x1, y1, x2, y2,
        stroke:color,
        'stroke-width':dim?'1':'3',
        'stroke-opacity':dim?'0.05':'0.95',
        'stroke-linecap':'round',
        'marker-end':(dim||!isTerminal)?'':'url(#ar-'+fi+')'}));
      // Wide transparent hit area — only on active lines so dim flows aren't clickable.
      if(!dim){
        const hit=svgN('line',{x1,y1,x2,y2,stroke:'transparent','stroke-width':'12',
          cursor:'pointer','pointer-events':'stroke'});
        hit.addEventListener('click',e=>{
          if(dmDragging) return;
          e.stopPropagation();
          dmSelectNet(dmActiveFi===fi?-1:fi);
        });
        svg.appendChild(hit);
      }
    });
  }
  if(dmActiveFi===-1){
    // All-nets mode: draw every flow (all bright, no dimming needed since all active).
    (DATA.comm_paths||[]).forEach(p=>drawEdges(p, false));
  } else {
    // Single-net mode: draw dim flows first, active on top.
    // Dim lines are very faint so shared segments don't occlude the active color.
    (DATA.comm_paths||[]).forEach(p=>{
      if(p.flow_index!==dmActiveFi) drawEdges(p, true);
    });
    (DATA.comm_paths||[]).forEach(p=>{
      if(p.flow_index===dmActiveFi) drawEdges(p, false);
    });
  }

  // ── LAYER 5a: solid dots (sources, contributors, forks) ──────────
  // Solid = injects into stream: source (push origin) or contributor (pull gather inject).
  // Fork = pure routing split (not a data producer or consumer).
  // Drawn before hollow dots so hollow dots always render on top.
  (DATA.comm_paths||[]).forEach(p=>{
    const fi=p.flow_index;
    const active=dmActiveFi===-1||dmActiveFi===fi;
    if(!active) return;
    const color=dmColor(fi);
    const ox=dmOx[fi]||0, oy=dmOy[fi]||0;
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
        svg.appendChild(svgN('circle',{cx:cx(sc)+ox,cy:cy(sr)+oy,r:'4.5',
          fill:color,stroke:'#181818','stroke-width':'1.2'}));
      }
    }

    // Contributor dots (pull flows) — solid: each packet_tile injects computed results
    // into the gather stream. Same solid style as source (both are injectors).
    if(isPull){
      const seen=new Set();
      (p.packet_tiles||[]).forEach(t=>{
        const [tc,tr]=t, k=tc+','+tr;
        if(seen.has(k)) return;
        seen.add(k);
        svg.appendChild(svgN('circle',{cx:cx(tc)+ox,cy:cy(tr)+oy,r:'4.5',
          fill:color,stroke:'#181818','stroke-width':'1.2'}));
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
        svg.appendChild(svgN('circle',{
          cx:cx(fc)+ox,cy:cy(fr)+oy,
          r:'5',fill:color,stroke:'#e4e4e4','stroke-width':'1.5'}));
      }
    });
  });

  // ── LAYER 5b: hollow dots — drawn LAST so they always sit on top of lines ──
  // Hollow = takes from stream: terminal consumer or mid-stream tap (receives + relays).
  // All hollow dots use fill:'#181818' so the opaque background covers any line passing
  // through the dot center, making the ring visually "cut into" the stream.
  (DATA.comm_paths||[]).forEach(p=>{
    const fi=p.flow_index;
    const active=dmActiveFi===-1||dmActiveFi===fi;
    if(!active) return;
    const color=dmColor(fi);
    const ox=dmOx[fi]||0, oy=dmOy[fi]||0;
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
      const [tc,tr]=e[1], dk=tc+','+tr;
      if(!outCount[dk] && tr>=0 && !dstSeen.has(dk)){
        dstSeen.add(dk);
        svg.appendChild(svgN('circle',{cx:cx(tc)+ox,cy:cy(tr)+oy,r:'4.5',
          fill:'#181818',stroke:color,'stroke-width':'2.2'}));
      }
    });

    // Tap dots — hollow ring, opaque background: DMA tile receives AND relays onward.
    // Slightly larger ring to distinguish from terminal.
    // Require BOTH inCount and outCount: a pull DMA source has outgoing edges but
    // no incoming edges and must not be drawn as a tap (it gets the solid source dot).
    const tapSeen=new Set();
    edges.forEach(e=>{
      const [sc,sr]=e[0], sk=sc+','+sr;
      if(sr>=0 && dmaTileSet.has(sk) && outCount[sk] && inCount[sk] && !tapSeen.has(sk)){
        tapSeen.add(sk);
        svg.appendChild(svgN('circle',{cx:cx(sc)+ox,cy:cy(sr)+oy,r:'5',
          fill:'#181818',stroke:color,'stroke-width':'2.5'}));
      }
    });
  });

  dmReset();
  dmBuilt=true;
}

buildNetBar();

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
function select(t, el, ch, badgeEl){
  if (selEl) selEl.classList.remove('sel');
  selEl = el; el.classList.add('sel');
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
  const p = document.getElementById('panel');

  // --- title ---
  const title = focused
    ? 'Tile ('+t.loc[0]+','+t.loc[1]+') &mdash; channel '+ch.direction+ch.channel+
      ' <span class="lref">(flow '+ch.flow_index+')</span>'
    : 'Tile ('+t.loc[0]+','+t.loc[1]+') &mdash; '+t.type;

  // --- high-level body ---
  let hiBody;
  if (focused) {
    const con = ch.contract ? '<div class="contract">'+esc(ch.contract)+'</div>' : '';
    const sd = ch.flow_balance ? renderFlowBalance(ch.flow_balance) : '';
    hiBody =
      '<div class="kv"><b>role:</b> '+esc(hlv.role)+'</div>' +
      (hlv.kernel?'<div class="kv"><b>kernel:</b> '+esc(hlv.kernel)+'</div>':'') +
      '<div class="kv"><b>channel:</b> '+ch.direction+ch.channel+' (flow '+ch.flow_index+')</div>' +
      '<div class="kv"><b>transfer:</b> '+esc(chanSummary(ch))+'</div>' +
      (sd?'<div class="kv"><b>supply / demand:</b></div>'+sd:'') +
      (con?'<div class="kv"><b>contract:</b></div>'+con:'');
  } else {
    const sum = (hlv.summary||[]).map(s=>'<li>'+esc(s)+'</li>').join('');
    const con = (hlv.contracts||[]).map(s=>'<div class="contract">'+esc(s)+'</div>').join('');
    hiBody =
      '<div class="kv"><b>role:</b> '+esc(hlv.role)+'</div>' +
      (hlv.kernel?'<div class="kv"><b>kernel:</b> '+esc(hlv.kernel)+'</div>':'') +
      '<div class="kv"><b>transfers:</b></div><ul class="sum">'+sum+'</ul>' +
      renderTileBalances(t) +
      (con?'<div class="kv"><b>contracts:</b></div>'+con:'') +
      renderKernelMatch(t);
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
  const tkv = tileKernel(t), tbf = tileBcf(t);
  const ksrc = (t.type==='core' && tkv && tkv.source) ? tkv.source : null;
  const isCore = !!ksrc;
  // Single "kernel code" sub-tab (core tiles): merges the kernel source
  // (conv2d_spatial.cc), the generated wrapper (kernel.cc) and the buffer
  // address map (.bcf) into one stacked view, each section headed by its file.
  const kcodeOn = (t.type==='core' &&
    ((tkv && (tkv.source || tkv.kernel_lines)) ||
     (tbf && tbf.lines)));
  const kfile = ksrc ? ksrc.file : '';
  const kfileTag = isCore
    ? ' <span class="kfileref">+ kernel '+esc(kfile)+'</span>' : '';
  const codePathBanner = codeFile
    ? '<div class="codepath"><b>code piece:</b> <span class="cpath">'+esc(codeFile)+'</span>'+kfileTag+'</div>'
    : (isCore
       ? '<div class="codepath"><b>kernel:</b> <span class="cpath">'+esc(kfile)+'</span></div>'
       : '');

  p.innerHTML =
    '<h2>'+title+'</h2>' +
    '<div class="tabs">' +
      '<span class="tab act" data-t="hi">High level</span>' +
      '<span class="tab" data-t="mid">Middle (dfschedule)</span>' +
      '<span class="tab" data-t="lo">Low level (host.cc)</span>' +
    '</div>' +
    '<div class="tabbody">' +
      '<div id="tab-hi">' + hiBody + '</div>' +
      '<div id="tab-mid" class="hide">' +
        '<div class="lref">dfschedule IR (6_BlueprintToSchedule) &mdash; ' +
          (focused ? 'channel '+ch.direction+ch.channel : 'tile') + ' scope</div>' +
        '<div class="midctrls">' +
          '<button id="loadFullIr">Load full dfschedule IR</button>' +
          '<button id="foldAll" class="hide">Expand all</button>' +
        '</div>' +
        '<div id="midContent">' + renderMiddleIR(midIR) + '</div>' +
      '</div>' +
      '<div id="tab-lo" class="hide">' +
        codePathBanner +
        '<div class="subtabs">' +
          '<span class="subtab act" data-lo="rel">Relevant only</span>' +
          '<span class="subtab" data-lo="full">Full block</span>' +
          (kcodeOn ? '<span class="subtab" data-lo="kern">kernel code</span>' : '') +
        '</div>' +
        '<div id="lo-rel">' +
          '<div class="lref">'+relLabel+'</div>' +
          relBody +
        '</div>' +
        '<div id="lo-full" class="hide">' +
          '<div class="lref">host.cc lines '+ (flo.line_start||'?') +'-'+ (flo.line_end||'?') +
            '  ('+(flo.ranges||[]).length+' range(s))' +
            (focused ? ' &mdash; channel '+ch.direction+ch.channel+' scope' : '') + '</div>' +
          renderFullBlock(flo.code_lines, focused ? ((ch.low_level||{}).params||null) : null) +
        '</div>' +
        (kcodeOn ? '<div id="lo-kern" class="hide">'+renderKernelCode(t, ch, focused)+'</div>' : '') +
      '</div>' +
    '</div>';
  const panes = {hi:'#tab-hi', mid:'#tab-mid', lo:'#tab-lo'};
  p.querySelectorAll('.tab').forEach(tab => tab.onclick = () => {
    p.querySelectorAll('.tab').forEach(x=>x.classList.remove('act'));
    tab.classList.add('act');
    Object.entries(panes).forEach(([k,sel]) =>
      p.querySelector(sel).classList.toggle('hide', tab.dataset.t!==k));
  });
  p.querySelectorAll('.subtab').forEach(st => st.onclick = () => {
    p.querySelectorAll('.subtab').forEach(x=>x.classList.remove('act'));
    st.classList.add('act');
    p.querySelector('#lo-rel').classList.toggle('hide', st.dataset.lo!=='rel');
    p.querySelector('#lo-full').classList.toggle('hide', st.dataset.lo!=='full');
    const kp = p.querySelector('#lo-kern');
    if (kp) kp.classList.toggle('hide', st.dataset.lo!=='kern');
  });
  // Kernel source "Show all" toggle: swap the isolated param code-piece for the
  // whole function body (with the param's lines highlighted), and back.
  const kbtn = p.querySelector('.kshowall');
  if (kbtn) {
    const kpiece = p.querySelector('#kern-piece');
    const kfull  = p.querySelector('#kern-full');
    kbtn.onclick = () => {
      const showAll = kfull.classList.contains('hide');
      kfull.classList.toggle('hide', !showAll);
      if (kpiece) kpiece.classList.toggle('hide', showAll);
      kbtn.textContent = showAll ? 'Show piece only' : 'Show all';
    };
  }
  // Kernel-code sub-tab: make the "//line a-b" fold markers (non-related code,
  // collapsed by default) clickable to expand/collapse in place.
  const klo = p.querySelector('#lo-kern');
  if (klo) wireFolds(klo);

  // --- full dfschedule IR (embedded) load/highlight/fold ---
  // The slice rows carry the ORIGINAL 6_BlueprintToSchedule.mlir line numbers;
  // reuse them as the highlight set for the full-file view. The button toggles
  // the SAME window (#midContent) between the slice piece and the full IR (with
  // non-target runs folded); a second button expands/collapses all folds.
  const midLines = Array.isArray(midIR)
    ? midIR.filter(r => r.line != null).map(r => r.line) : [];
  const loadBtn    = p.querySelector('#loadFullIr');
  const foldAllBtn = p.querySelector('#foldAll');
  const midContent = p.querySelector('#midContent');
  if (loadBtn && midContent) {
    const sliceHtml = midContent.innerHTML;     // cache the slice view
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

  // --- aiegdb console (only after a passing connection test) ---
  // Clicking a tile runs `target tile <col> <row>`; a channel click chains
  // `target channel <dir_ch>` so the console scope follows the UI.
  if (LIVE.connected) setConTarget(t, ch);

  // --- local-LLM cooperation: prepare (do NOT send) the tile/file location
  // context for the clicked tile/channel. It is attached to the NEXT user chat
  // message, and only if it differs from the last one attached (see llmSend). ---
  setLLMContext(t, ch, codeFile);
}

// Set LLM context from anywhere; llmSend attaches it to the next message (deduped).
function llmPushCtx(text){ LLM.ctx = text || null; }

// Richer tile/channel context: include role, kernel, contract, port.
function setLLMContext(t, ch, codeFile){
  const hl = (t && t.high_level) || {};
  const loc = t ? 'tile ('+t.loc[0]+','+t.loc[1]+')' : '';
  const parts = ['[context] Selected: '+loc
    + (ch ? ' channel '+ch.direction+ch.channel : '')
    + ' — type: '+(t&&t.type||'?')+', role: '+(hl.role||'?')];
  if (hl.kernel) parts.push('kernel: '+hl.kernel);
  if (ch && ch.contract) parts.push('contract: '+ch.contract);
  if (ch && ch.kernel_port) parts.push('port: '+ch.kernel_port);
  if (codeFile) parts.push('code: '+codeFile);
  LLM.ctx = parts.join('; ');
}

// ─── aiegdb console (right-bottom, drives aiegdb.py --server) ─────────────────
// A terminal-style console over the daemon's persistent aiegdb REPL subprocess.
// The user types any aiegdb command; scope (partition->tile->channel) is kept by
// the subprocess and reported back via r.scope. Tile/channel clicks send
// `target ...` so the console follows the UI. "Reload aiegdb.py" respawns the
// subprocess (reloads edited code), resetting to partition scope.
const CON = { scope:'partition' };
function conAppend(text){
  const out = document.getElementById('conout');
  if (!out) return;
  out.textContent += (out.textContent && !out.textContent.endsWith('\n') ? '\n' : '') + text;
  out.scrollTop = out.scrollHeight;
}
function conSetScope(s){
  CON.scope = s || 'partition';
  const tgt = document.getElementById('contarget');
  if (tgt) tgt.textContent = CON.scope;
  const pr = document.getElementById('conprompt');
  if (pr) pr.textContent = CON.scope + '>';
}
function conSetLast(cmd, res){
  const c = document.getElementById('conlastcmd');
  const r = document.getElementById('conlastres');
  if (c) c.textContent = CON.scope + '> ' + cmd;
  if (r) r.textContent = res == null ? '' : (' \u2192 ' + res);
}
// applyScope=false lets a caller keep an already-set (optimistic) scope instead
// of adopting this command's returned r.scope — used for the intermediate
// `target tile` step of a channel selection so it doesn't downgrade the prompt.
function conSend(cmd, echo, applyScope){
  if (echo !== false) conAppend(CON.scope + '> ' + cmd);
  conSetLast(cmd, null);
  return api('/aiegdb', {method:'POST', headers:{'Content-Type':'application/json'},
                         body: JSON.stringify({cmd:cmd})})
    .then(r => {
      const out = r.output ? r.output.replace(/\n+$/,'') : '';
      if (out) conAppend(out);
      if (applyScope !== false && r.scope) conSetScope(r.scope);
      conSetLast(cmd, out || '(no output)');
    })
    .catch(() => { conAppend('daemon offline (static mode)');
                   conSetLast(cmd, 'daemon offline (static mode)'); });
}
// Tile/channel click: run `target tile c r`, then chain `target channel dir_ch`.
// The prompt is updated optimistically from the UI selection first, so the
// console scope follows the click immediately — independent of daemon latency,
// and even if the chained `target channel` request is slow or fails (its .catch
// would otherwise leave the prompt stuck at tile scope). A successful daemon
// response still confirms/refines the scope via r.scope. The optimistic string
// mirrors aiegdb's prompt format exactly: partition(startcol=N)/tile(c,r)/dir_ch
// (direction lowercased, matching _parse_dir_ch).
function setConTarget(t, ch){
  const box = document.getElementById('cmdconsole');
  if (box) box.classList.remove('hide');
  const rsp = document.getElementById('rhsplitter');
  if (rsp) rsp.classList.remove('hide');
  const sc = (g.startcol !== null && g.startcol !== undefined) ? g.startcol : 0;
  let optScope = 'partition(startcol=' + sc + ')/tile(' + t.loc[0] + ',' + t.loc[1] + ')';
  if (ch) optScope += '/' + ch.direction.toLowerCase() + ch.channel;
  conSetScope(optScope);
  // For a channel selection, suppress the intermediate tile step's scope apply
  // (applyScope=false) so it can't downgrade the optimistic channel prompt; the
  // chained `target channel` then confirms it. A tile-only click applies scope.
  const p = conSend('target tile ' + t.loc[0] + ' ' + t.loc[1], undefined, !ch);
  if (ch) p.then(() => conSend('target channel ' +
                               ch.direction.toLowerCase() + ch.channel));
}
document.getElementById('conin').addEventListener('keydown', e => {
  if (e.key === 'Enter'){ const v = e.target.value.trim();
    if (v){ conSend(v); e.target.value=''; } }
});
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
const LLM = { off:0, poll:null, busy:false, pendingId:null, ctx:null, ctxSent:null };
let llmMessages = [];
let llmMsgIdCtr = 0;
function llmEscape(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
// Colorize an ALREADY-escaped prose string: tool calls, tool results, you>
// prompts, error/offline markers, and file / file:line references.
function llmColorizeMarkers(s){
  s = s.replace(/^you&gt;.*$/gm, m => '<span class="llm-you">' + m + '</span>');
  s = s.replace(/\[tool: ([^\s\]]+)([^\]]*)\]/g,
    (m,name,rest) => '<span class="llm-tool">[tool: <span class="llm-toolname">'
      + name + '</span>' + rest + ']</span>');
  s = s.replace(/\[tool result\]/g, '<span class="llm-toolresult">[tool result]</span>');
  s = s.replace(/\[(llm error[^\]]*|daemon offline[^\]]*)\]/g,
    (m,inner) => '<span class="llm-error">[' + inner + ']</span>');
  s = s.replace(/\b([A-Za-z0-9_./-]+\.(?:py|cc|cpp|cxx|h|hpp|c|md|mlir|sh|json|txt|inc|td|html|elf|log))(?::(\d+))?\b/g,
    (m,file,line) => '<span class="llm-file">' + file + '</span>'
      + (line ? ':<span class="llm-line">' + line + '</span>' : ''));
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
function llmProse(raw){
  let s = llmEscape(raw);
  s = s.replace(/^(#{1,6})\s+(.*)$/gm, (m,h,t) => '<span class="md-h">' + t + '</span>');
  s = s.replace(/^(\s*)[-*+]\s+/gm, '$1<span class="md-bullet">\u2022 </span>');
  s = s.replace(/`([^`\n]+)`/g, (m,c) => '<code class="md-code">' + c + '</code>');
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  return llmColorizeMarkers(s);
}
// Split on ``` fences: even segments are prose, odd are code blocks.
// Unclosed trailing fence mid-stream is treated as code for safe partial renders.
function llmRenderText(raw){
  const parts = raw.split('```');
  let html = '';
  for (let i = 0; i < parts.length; i++){
    if (i % 2 === 0){ html += llmProse(parts[i]); continue; }
    const blk = parts[i], nl = blk.indexOf('\n');
    const body = nl >= 0 ? blk.slice(nl + 1) : blk;
    html += '<pre class="md-block"><code>' + llmHighlightCode(body) + '</code></pre>';
  }
  return html;
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
// Stream raw text into an existing AI bubble, re-rendering markdown each chunk.
function llmAppendToMsg(id, rawChunk){
  const rec = llmMessages.find(x => x.id === id);
  if (!rec) return;
  const list = document.getElementById('llmmsg');
  const near = list ? (list.scrollHeight - list.scrollTop - list.clientHeight < 60) : false;
  rec._raw += rawChunk;
  rec.html = llmRenderText(rec._raw);
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = rec.html;
  if (near && list) list.scrollTop = list.scrollHeight;
}
function llmShowThink(on){
  const t = document.getElementById('llmthink');
  if (t) t.classList.toggle('hide', !on);
}

function llmStopPoll(){ if (LLM.poll){ clearInterval(LLM.poll); LLM.poll = null; } }
function llmPollOnce(){
  if (LLM.busy) return;
  LLM.busy = true;
  api('/llm/poll?offset=' + LLM.off).then(r => {
    if (r.auth){ llmLock(); return; }
    if (r.error){ llmStopPoll(); return; }
    if (r.data && LLM.pendingId){ llmShowThink(false); llmAppendToMsg(LLM.pendingId, r.data); }
    if (r.next != null) LLM.off = r.next;
    if (r.active === false){ llmStopPoll(); LLM.pendingId = null; llmShowThink(false); }
  }).catch(() => { llmStopPoll(); llmShowThink(false); })
    .finally(() => { LLM.busy = false; });
}
function llmSend(prompt){
  prompt = (prompt || '').trim();
  if (!prompt) return;
  llmStopPoll();
  let toSend = prompt;
  if (LLM.ctx && LLM.ctx !== LLM.ctxSent){
    toSend = LLM.ctx + '\n' + prompt;
    LLM.ctxSent = LLM.ctx;
    llmAddMsg('ctx', llmEscape(LLM.ctx));
  }
  llmAddMsg('you', llmEscape(prompt));
  const aiId = llmAddMsg('ai', '');
  LLM.pendingId = aiId;
  llmShowThink(true);
  api('/llm', {method:'POST', headers:{'Content-Type':'application/json'},
               body: JSON.stringify({prompt:toSend})})
    .then(r => {
      if (r.auth){ llmLock(); return; }
      if (!r.ok){ llmShowThink(false);
        llmAppendToMsg(aiId, '[llm error: ' + (r.error || 'unknown') + ']'); return; }
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
      LLM.ctxSent = null; LLM.off = 0; LLM.pendingId = null;
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
  function autoGrow(){
    if (!inp) return;
    inp.style.height = 'auto';
    inp.style.height = inp.scrollHeight + 'px';
  }
  function submitLLM(){
    if (!inp) return;
    const v = inp.value.trim();
    if (v){ llmSend(v); inp.value=''; autoGrow(); }
  }
  if (inp) inp.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); submitLLM(); }
  });
  if (inp) inp.addEventListener('input', autoGrow);
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
  if (pane === 'llmpane') llmCheckAuth();
  if (pane === 'searchpane') document.getElementById('sr-input').focus();
});

document.getElementById('gbtn').onclick = () => {
  clearPeers();
  if (selEl){ selEl.classList.remove('sel'); selEl=null; }
  const gl = DATA.global;
  document.getElementById('panel').innerHTML =
    '<h2>Global / kernel-group</h2>' +
    '<div class="kv">'+esc(gl.note)+'</div>' +
    '<pre class="code">'+hl(gl.code||'')+'</pre>';
};

// ─── Live debug overlay (talks to schedule_debug_server on the same origin) ───
// All additive: when opened as a static file:// (no daemon) every fetch fails
// and the page silently stays in static mode.
const LIVE = { enabled:false, connected:false, what:'dma', gridTimer:null,
               conTimer:null, logoff:0, gridBusy:false, logBusy:false,
               runActive:false, debugUnlocked:false, device:'', host:'',
               hwsrvOff:0, hwsrvTimer:null, hwsrvBusy:false };
const LSTATE = {
  running:['#2e7d32','RUN'], stalled:['#b8860b','STALL'], error:['#c62828','ERR'],
  idle:['#3a3a3a','idle'], unreachable:['#5a2d2d','n/a'], unknown:['#3a3a3a','?']
};
function llmToken(){ return sessionStorage.getItem('LLM_AUTH') || ''; }
function api(path, opts){
  opts = opts || {};
  const tok = llmToken();
  if (tok) opts.headers = Object.assign({}, opts.headers, {'X-LLM-Auth': tok});
  return fetch(path, opts).then(r => r.json());
}
function setStatus(msg){ const e=document.getElementById('livestatus'); if(e) e.textContent=msg; }

function clearBars(){
  Object.values(liveBar).forEach(b => { b.className='livebar hide'; b.textContent=''; });
}
function applyGrid(res){
  if (res.error){ setStatus('live: '+res.error); }
  else setStatus('live '+LIVE.what+' @ '+new Date().toLocaleTimeString());
  const cells = res.cells || {};
  Object.keys(liveBar).forEach(k => {
    const b = liveBar[k], c = cells[k];
    if (!c){ b.className='livebar hide'; return; }
    const m = LSTATE[c.state] || LSTATE.unknown;
    b.className = 'livebar';
    b.style.background = m[0];
    b.textContent = m[1];
    b.title = JSON.stringify(c, null, 2);
  });
}
function pollGridOnce(){
  // Belt-and-suspenders: never poll the JTAG while a board run is live; the
  // aiedbg reads would collide with apppaltest's device program/reset/download.
  if (LIVE.runActive) return;
  // Skip if a /grid is still in flight: aiedbg reads can take longer than the
  // 2s tick, and piled-up requests would exhaust the browser's ~6-connection
  // limit and starve /applog (freezing the run console).
  if (LIVE.gridBusy) return;
  LIVE.gridBusy = true;
  const dev = deviceSel ? deviceSel.value : '';
  const host = boardHost ? boardHost.value.trim() : '';
  const qs = '/grid?what='+LIVE.what +
             '&device='+encodeURIComponent(dev) +
             '&host='+encodeURIComponent(host);
  api(qs).then(applyGrid)
    .catch(() => setStatus('daemon offline (static mode)'))
    .finally(() => { LIVE.gridBusy = false; });
}
function startGridPoll(){ pollGridOnce(); LIVE.gridTimer = setInterval(pollGridOnce, 2000); }
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
  LIVE.enabled = on;
  const cb = document.getElementById('liveToggle'); if(cb) cb.checked = on;
  if (on) startGridPoll();
  else { stopGridPoll(); clearBars(); setStatus(''); }
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
  setStatus('run parked \u2014 aiegdb console unlocked for live debug (overlay stays off)');
}
function setDebugEnabled(on){
  const cb = document.getElementById('liveToggle');
  const tc = document.getElementById('testconn');
  const cr = document.getElementById('conreload');
  const ci = document.getElementById('conin');
  if (!on){
    LIVE.runActive = true;
    LIVE.debugUnlocked = false;   // new run → re-lock until it's debuggable again
    stopGridPoll(); clearBars();
    if (cb){ cb.checked = false; cb.disabled = true;
      cb.closest('label').classList.toggle('disabled', true); }
    if (tc) tc.disabled = true;
    if (cr) cr.disabled = true;
    if (ci){ ci.disabled = true; ci.classList.add('disabled'); }
    setStatus('debug disabled during run setup');
  } else {
    LIVE.runActive = false;
    if (tc) tc.disabled = false;
    if (cr) cr.disabled = false;
    if (ci){ ci.disabled = false; ci.classList.remove('disabled'); }
    // Re-enable the overlay toggle only if the connection is still valid
    // (mirrors the testConnect gating).
    if (cb){ cb.disabled = !LIVE.connected;
      cb.closest('label').classList.toggle('disabled', !LIVE.connected); }
    setStatus('');
  }
}

// Device selection gates the live controls (item #4/#5). A device must be
// chosen before the "Live status overlay" checkbox and "Run test" button work;
// picking vek385 also reveals the board-hostname text box.
const deviceSel = document.getElementById('deviceSel');
const boardHost = document.getElementById('boardHost');
const liveToggle = document.getElementById('liveToggle');
const runbtn = document.getElementById('runbtn');
const stopbtn = document.getElementById('stopbtn');
const testconn = document.getElementById('testconn');
function setConnStatus(msg){ const e=document.getElementById('connstatus'); if(e) e.textContent=msg; }
// Show/hide the "start hw_server on the target board" hint (shown on failure).
function setConnHint(show){ const e=document.getElementById('connhint'); if(e) e.classList.toggle('hide', !show); }
// Selecting a device only enables the "Test connect" button. The live overlay
// checkbox + the drill-down console stay locked until a connection test passes
// (LIVE.connected). Changing the device invalidates any prior test.
function updateDeviceUI(){
  const dev = deviceSel ? deviceSel.value : '';
  const has = !!dev;
  LIVE.connected = false;
  setLive(false);                              // uncheck overlay + stop poll
  hideConsole();                               // connection invalidated
  if (liveToggle){ liveToggle.disabled = true;
    liveToggle.closest('label').classList.toggle('disabled', true); }
  // Run test / Force stop stay gray until "Test connect" passes; changing the
  // device invalidates any prior test (LIVE.connected cleared above).
  if (runbtn) runbtn.disabled = true;
  if (stopbtn) stopbtn.disabled = true;
  if (testconn) testconn.disabled = !has;
  if (boardHost) boardHost.classList.toggle('hide', dev !== 'vek385');
  // For simulator, "Test connect" label stays but skips JTAG; still require
  // the test step so the user explicitly enables Run/Stop for it.
  if (testconn) testconn.textContent = (dev === 'simulator') ? 'Activate' : 'Test connect';
  setConnStatus(has ? 'click "Test connect" to enable live features' : '');
  setConnHint(false);
  if(deviceSel&&has){
    const opt=deviceSel.options[deviceSel.selectedIndex];
    llmPushCtx('[context] Device selected: '+(opt?opt.text:dev));
  }
}
if (deviceSel) deviceSel.onchange = updateDeviceUI;
// Apply a successful connection: unlock Run test / Force stop / overlay and
// reveal the aiegdb console. Shared by the direct test and the auto-launch path.
function applyConnected(r){
  LIVE.connected = true;
  if (runbtn) runbtn.disabled = false;    // unlock Run test now
  if (stopbtn) stopbtn.disabled = false;  // unlock Force stop now
  if (liveToggle){ liveToggle.disabled = false;
    liveToggle.closest('label').classList.remove('disabled'); }
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
        setConnStatus('connected \u2014 ' + (sr.target || ((r && r.detail) || 'ok')));
        llmPushCtx('[context] Connected to '+(LIVE.host||LIVE.device)
          +' \u2014 AIEDBG_TARGET: '+(sr.target||'unknown'));
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
// Mark disconnected: re-gray Run test / Force stop / overlay and hide console.
function markDisconnected(){
  LIVE.connected = false;
  if (runbtn) runbtn.disabled = true;      // keep Run test gray
  if (stopbtn) stopbtn.disabled = true;    // keep Force stop gray
  if (liveToggle) liveToggle.disabled = true;
  hideConsole();
}
// Recovery: when the JTAG connect fails but the daemon is up, have the daemon
// ssh to the board, start hw_server (ssh -> systest -> xsdb -> exec hw_server),
// then re-probe once. The launch runs async on the daemon; we tail its per-step
// progress into the left-bottom console (#console) via /hwsrv_log so the user
// sees each step live. On success we're connected; otherwise show the hint.
function autoLaunchHwServer(dev, host, why){
  markDisconnected();
  // Keep the selection so applyConnected (reached via pollHwSrv on success)
  // switches the aiegdb console target to this same host.
  LIVE.device = dev; LIVE.host = host;
  setConnStatus('connect failed (' + why + ') \u2014 starting hw_server on board\u2026');
  setConnHint(false);
  // Reveal + reset the left-bottom console so the launch steps stream in.
  const con = document.getElementById('console');
  if (con){ con.classList.remove('hide'); con.textContent =
    '[auto-starting hw_server on board \u2014 live progress below]\n'; }
  LIVE.hwsrvOff = 0;
  api('/launch_hwserver', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({device:dev, host:host})})
    .then(r => {
      if (r && r.started){
        // Begin tailing the daemon's hw_server session log into #console.
        if (LIVE.hwsrvTimer) clearInterval(LIVE.hwsrvTimer);
        LIVE.hwsrvTimer = setInterval(pollHwSrv, 1000);
        pollHwSrv();   // immediate first tail so steps show without a 1s delay
      } else {
        markDisconnected();
        setConnStatus('auto-start failed: ' + ((r && r.detail) || 'unknown'));
        setConnHint(true);   // fall back to the manual start-hw_server guidance
      }
    })
    .catch(() => {
      markDisconnected();
      setConnStatus('daemon offline: cannot auto-start hw_server.');
      setConnHint(true);
    });
}
// Tail the daemon's hw_server launch session into #console and, once the
// background worker is done, apply the connect result (single-retry outcome).
function pollHwSrv(){
  if (LIVE.hwsrvBusy) return;           // guard against overlapping tails
  LIVE.hwsrvBusy = true;
  api('/hwsrv_log?offset='+LIVE.hwsrvOff).then(r => {
    const con = document.getElementById('console');
    if (con){
      const atBottom = (con.scrollHeight - con.scrollTop - con.clientHeight) < 4;
      if (r.data){ con.textContent += r.data; }
      if (atBottom) con.scrollTop = con.scrollHeight;
    }
    if (r.next != null) LIVE.hwsrvOff = r.next;
    setConnStatus('hw_server: ' + (r.status || 'starting') + '\u2026');
    if (r.done){
      if (LIVE.hwsrvTimer){ clearInterval(LIVE.hwsrvTimer); LIVE.hwsrvTimer=null; }
      if (r.ok){ applyConnected(r); }    // connected on the single retry
      else {
        // Final retry still failed. Re-gray Run test / Force stop, but KEEP the
        // console visible (don't call markDisconnected -> hideConsole) so the
        // user still sees the launch steps, and append the original
        // "connection failed -> start hw_server manually" guidance in place.
        LIVE.connected = false;
        if (runbtn) runbtn.disabled = true;
        if (stopbtn) stopbtn.disabled = true;
        if (liveToggle) liveToggle.disabled = true;
        if (con){
          con.textContent += '\n[auto-start failed \u2014 connection still down: '
            + ((r && r.detail) || 'unknown') + ']\n'
            + 'Connection failed. On the target hw board, start hw_server via xsdb:\n'
            + '    exec hw_server -stcp:0.0.0.0:3121\n';
          con.scrollTop = con.scrollHeight;
        }
        setConnStatus('connection failed \u2014 run "exec hw_server -stcp:0.0.0.0:3121" on the target board');
        setConnHint(true);   // also show the persistent manual-start hint banner
      }
    }
  }).catch(() => {
    if (LIVE.hwsrvTimer){ clearInterval(LIVE.hwsrvTimer); LIVE.hwsrvTimer=null; }
    markDisconnected();
    setConnStatus('daemon offline: cannot auto-start hw_server.');
    setConnHint(true);
  }).finally(() => { LIVE.hwsrvBusy = false; });
}
// Test the daemon + JTAG target; on success unlock the overlay + reveal console.
function testConnect(){
  const dev = deviceSel ? deviceSel.value : '';
  if (!dev){ setConnStatus('select a device first'); return; }
  // Simulator: probe IPC readiness. If already ready, apply connected immediately.
  // Otherwise, unlock Run/Stop so the user can start the sim and the aiegdb
  // console will auto-connect once the IPC debug socket appears.
  if (dev === 'simulator'){
    LIVE.device = dev; LIVE.host = '';
    api('/sim/status').then(ss => {
      if (ss && ss.ipc_ready){
        applyConnected({detail: 'simulator IPC ready'});
      } else {
        // Not ready yet — still unlock all live controls so Run sim works and
        // the overlay checkbox is available once the sim is running.
        LIVE.connected = true;
        if (runbtn) runbtn.disabled = false;
        if (stopbtn) stopbtn.disabled = true;
        if (liveToggle){ liveToggle.disabled = false;
          liveToggle.closest('label').classList.remove('disabled'); }
        const box = document.getElementById('cmdconsole');
        if (box) box.classList.remove('hide');
        const rsp = document.getElementById('rhsplitter');
        if (rsp) rsp.classList.remove('hide');
        setConnStatus('simulator activated — run it to enable live grid reads');
      }
    }).catch(() => {
      LIVE.connected = true;
      if (runbtn) runbtn.disabled = false;
      if (liveToggle){ liveToggle.disabled = false;
        liveToggle.closest('label').classList.remove('disabled'); }
      setConnStatus('simulator activated (daemon offline)');
    });
    return;
  }
  const host = boardHost ? boardHost.value.trim() : '';
  if (dev === 'vek385' && !host){ setConnStatus('enter vek385 board hostname'); return; }
  // Remember the selection so applyConnected can tell the daemon which target
  // to switch the aiegdb console to (mirrors autoLaunchHwServer's closure).
  LIVE.device = dev; LIVE.host = host;
  setConnStatus('testing\u2026');
  const qs = '?device='+encodeURIComponent(dev)+'&host='+encodeURIComponent(host);
  api('/ping'+qs).then(r => {
    if (r && r.ok){
      applyConnected(r);
    } else {
      // Daemon answered but the JTAG connect failed → try to auto-start
      // hw_server on the board and retry once.
      autoLaunchHwServer(dev, host, (r && r.detail) || 'no response');
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

document.getElementById('liveToggle').onchange = e => setLive(e.target.checked);
document.querySelectorAll('#overlaytabs .ltab').forEach(tab => tab.onclick = () => {
  document.querySelectorAll('#overlaytabs .ltab').forEach(x => x.classList.remove('act'));
  tab.classList.add('act');
  LIVE.what = tab.dataset.w;
  // Make the tab self-activating: if the overlay is already live, just refetch
  // for the new 'what'. Otherwise auto-enable the overlay when we're connected,
  // so clicking Cores/Events actually shows its data (core PCs / DMA events)
  // without also having to tick the "Live status overlay" box. If we can't poll
  // (no connection or a run is holding the JTAG) explain why nothing appears.
  if (LIVE.enabled){ pollGridOnce(); return; }
  if (LIVE.runActive){ setStatus('overlay paused during run (JTAG busy)'); return; }
  if (LIVE.connected){
    const cb = document.getElementById('liveToggle');
    if (cb) cb.checked = true;
    setLive(true);   // flips LIVE.enabled + starts the grid poll for this 'what'
  } else {
    setStatus('pick a device, "Test connect", then this tab shows live '+LIVE.what);
  }
});

function pollLog(){
  // Guard against overlapping tails so a slow response can't pile up.
  if (LIVE.logBusy) return;
  LIVE.logBusy = true;
  api('/applog?offset='+LIVE.logoff).then(r => {
    const con = document.getElementById('console');
    const atBottom = (con.scrollHeight - con.scrollTop - con.clientHeight) < 4;
    if (r.data){ con.textContent += r.data; }
    if (r.next != null) LIVE.logoff = r.next;
    if (atBottom) con.scrollTop = con.scrollHeight;
    setStatus('run: ' + r.status);
    // Auto-unlock the aiegdb console mid-run once the daemon reports the run is
    // past the exclusive-JTAG setup phase (download/program/reset done → app
    // running/parked/hung). Only typed commands hit JTAG; the live overlay
    // grid-poll stays OFF so we don't compete with the run's own reads.
    if (r.running === true && r.debuggable) unlockConsoleForDebug();
    // Stop only when the subprocess has actually exited — NOT on a derived
    // pass/fail. apppaltest keeps writing (summary/cleanup/reboot) long after
    // the teardown marker first appears mid-run.
    if (r.running === false){
      if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer=null; }
      // Run ended (force-stop OR natural completion) → re-enable aiedbg features.
      setDebugEnabled(true);
    }
  }).catch(() => {}).finally(() => { LIVE.logBusy = false; });
}
document.getElementById('runbtn').onclick = () => {
  const dev = deviceSel ? deviceSel.value : '';
  if (!dev){ setStatus('select a device first'); return; }
  const con = document.getElementById('console');
  con.classList.remove('hide'); con.textContent = '';
  // Simulator path: route to /sim/run and tail /sim/log.
  if (dev === 'simulator'){
    SIM.logoff = 0; SIM.applogoff = 0; SIM.applogSeen = false;
    if (runbtn) runbtn.disabled = true;
    if (stopbtn) stopbtn.disabled = false;
    api('/sim/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
      .then(r => {
        if (r.error){ con.textContent = 'sim error: ' + r.error;
          if (runbtn) runbtn.disabled = false; if (stopbtn) stopbtn.disabled = true; return; }
        con.textContent = '[simulator started \u2192 ' + (r.sim_log||'ipc_sim.log') + ']\n'
          + '[waiting for IPC debug socket\u2026]\n';
        llmPushCtx('[context] Simulator run started');
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
  if (dev === 'vek385' && !host){ setStatus('enter vek385 board hostname'); return; }
  LIVE.logoff = 0;
  // Stop aiedbg polling/console BEFORE the download starts so its JTAG reads
  // don't collide with device program/reset/dow -force. Re-enabled in pollLog
  // when the run ends (force-stop or natural completion).
  setDebugEnabled(false);
  api('/run', {method:'POST', headers:{'Content-Type':'application/json'},
               body: JSON.stringify({device:dev, board_host:host})})
    .then(r => {
      // Run didn't actually start → re-enable debug (no pollLog will run).
      if (r.error){ con.textContent = 'run error: ' + r.error; setDebugEnabled(true); return; }
      con.textContent = '[run ' + r.run_id + ' started \u2192 ' + (r.applog||'applog') + ']\n';
      llmPushCtx('[context] Hardware run '+r.run_id+' started on '+dev);
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
        if (r.error){ setStatus('stop: ' + r.error); return; }
        setStatus('simulator stopped (pid ' + r.pid + ')');
        pollSimLog();
      })
      .catch(() => { setStatus('daemon offline: cannot stop simulator.'); });
    return;
  }
  api('/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => {
      if (r.error){ setStatus('stop: ' + r.error); return; }
      setStatus('run: stopped (pid ' + r.pid + ')');
      // The next pollLog tick sees running=false, drains the force-stop line,
      // and clears the timer. Force one immediate tail so it shows at once.
      if (typeof pollLog === 'function') pollLog();
    })
    .catch(() => { setStatus('daemon offline: cannot stop a run.'); });
};
// Load the existing applog from the start (one-shot, no live tail). Useful for
// inspecting the last run without launching a new one.
document.getElementById('loadlogbtn').onclick = () => {
  const dev = deviceSel ? deviceSel.value : '';
  const con = document.getElementById('console');
  con.classList.remove('hide'); con.textContent = '';
  if (dev === 'simulator'){
    SIM.logoff = 0; SIM.applogoff = 0; SIM.applogSeen = false;
    if (SIM.timer){ clearInterval(SIM.timer); SIM.timer=null; }
    Promise.all([
      api('/sim/log?offset=0'),
      api('/sim/applog?offset=0').catch(() => ({data:'',next:0}))
    ]).then(([r, ra]) => {
      con.textContent = r.data || '(sim log is empty)';
      if (r.next != null) SIM.logoff = r.next;
      if (ra.data){
        con.textContent += '\n--- PS application output (ipc_app.log) ---\n' + ra.data;
        SIM.applogSeen = true;
      }
      if (ra.next != null) SIM.applogoff = ra.next;
      con.scrollTop = con.scrollHeight;
      setStatus('loaded sim log (' + (r.running ? 'running' : 'idle') + ')');
    }).catch(() => { con.textContent = 'daemon offline: cannot load sim log.'; });
    return;
  }
  LIVE.logoff = 0;
  // Cancel any active tail so the one-shot load isn't fought by polling.
  if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer=null; }
  api('/applog?offset=0').then(r => {
    con.textContent = r.data || '(applog is empty)';
    if (r.next != null) LIVE.logoff = r.next;
    con.scrollTop = con.scrollHeight;
    setStatus('loaded existing log (' + (r.status || 'idle') + ')');
  }).catch(() => { con.textContent = 'daemon offline: cannot load applog.'; });
};

// ── Extra devices (simulator etc): populated from /devices at startup ──
// The server reads debug_ui_config.json from the workdir and returns any
// extra devices (e.g. simulator) that it can run.  Each entry is injected
// as an <option> in the board dropdown; selecting "simulator" short-circuits
// the JTAG "Test connect" and routes Run/Stop/Load-log to /sim/* endpoints.
const SIM = { timer: null, logoff: 0, logBusy: false, ipcReady: false,
              applogoff: 0, applogSeen: false };
function pollSimLog(){
  if (SIM.logBusy) return;
  SIM.logBusy = true;
  const con = document.getElementById('console');
  // Poll both the simulator engine log and the PS application log in parallel.
  Promise.all([
    api('/sim/log?offset='+SIM.logoff),
    api('/sim/applog?offset='+SIM.applogoff).catch(() => ({data:'',next:SIM.applogoff,running:false}))
  ]).then(([r, ra]) => {
    const atBottom = con ? (con.scrollHeight - con.scrollTop - con.clientHeight) < 4 : true;
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
    if (atBottom && con) con.scrollTop = con.scrollHeight;
    // Enable live grid reads once the IPC debug socket is ready.
    if (r.ipc_ready && !SIM.ipcReady){
      SIM.ipcReady = true;
      LIVE.device = 'simulator'; LIVE.host = '';
      if (liveToggle){ liveToggle.disabled = false;
        liveToggle.closest('label').classList.remove('disabled'); }
      setConnStatus('simulator IPC ready \u2014 live grid reads active');
    }
    setStatus('sim: ' + (r.running ? 'running' : 'stopped'));
    if (!r.running){
      if (SIM.timer){ clearInterval(SIM.timer); SIM.timer = null; }
      SIM.ipcReady = false;
      if (runbtn)  runbtn.disabled  = false;
      if (stopbtn) stopbtn.disabled = true;
    }
  }).catch(() => {}).finally(() => { SIM.logBusy = false; });
}
api('/devices').then(r => {
  if (!r || !r.devices || !deviceSel) return;
  r.devices.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.value; opt.textContent = d.label;
    deviceSel.appendChild(opt);
  });
}).catch(() => {});
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
    if (c.device && deviceSel){
      const has = Array.from(deviceSel.options).some(o => o.value === c.device);
      if (has) deviceSel.value = c.device;
    }
    if (c.board_host && boardHost) boardHost.value = c.board_host;
    updateDeviceUI();   // reveal/enable controls for the preselected device
  }).catch(() => {});
}
if (location.protocol === 'http:' || location.protocol === 'https:') {
  updateDeviceUI();   // device empty ⇒ controls disabled, overlay off
  applyBoardDefaults();
} else {
  setStatus('static mode — open via schedule_debug_server.py for live status');
  updateDeviceUI();
}

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
    // Show any transcript already buffered (e.g. after a page reload).
    if (r.data){ llmAddMsg('ai', llmRenderText(r.data)); }
    if (r.next != null) LLM.off = r.next;
  }).catch(() => {});   // no daemon → stay static
}
if (location.protocol === 'http:' || location.protocol === 'https:') probeLLM();
</script>
</body>
</html>
"""


def _palmyra_enabled():
    """The palmyra board option is shown only when ~/USERENV/ENABLEPAL holds '1'.
    Missing/unreadable file (or any other value) hides it."""
    try:
        with open(os.path.expanduser('~/USERENV/ENABLEPAL')) as f:
            return f.read().strip() == '1'
    except OSError:
        return False


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


def write_code_cache(view, cache_dir='debugcache/code'):
    """Write one code-piece file per tile and per channel; annotate the view
    with absolute `code_file` paths. Returns (cache_dir_abs, file_count)."""
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


def write_html(view, out_path):
    data_json = json.dumps(view, indent=None)
    html = HTML_TEMPLATE.replace('/*__DATA__*/ null', data_json)
    palmyra_opt = ('          <option value="palmyra">palmyra</option>\n'
                   if _palmyra_enabled() else '')
    html = html.replace('<!--__PALMYRA_OPTION__-->', palmyra_opt)
    with open(out_path, 'w') as f:
        f.write(html)


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else 'aout/worklocal'
    view = build_view(workdir)
    # Materialize per-tile/channel code pieces (annotates view with code_file
    # paths) BEFORE serializing so both the JSON and the embedded HTML DATA carry
    # the paths for the file-frame header + LLM auto-notify.
    cache_dir, ncode = write_code_cache(view)
    json_out = os.path.join(workdir, 'schedule_view.json')
    html_out = os.path.join(workdir, 'host_schedule.html')
    with open(json_out, 'w') as f:
        json.dump(view, f, indent=2)
    write_html(view, html_out)
    ntiles = len(view['tiles'])
    covered = sum(len(t['low_level']['ranges']) for t in view['tiles'])
    print('wrote %s (%d tiles, %d line-ranges attributed)' % (json_out, ntiles, covered))
    print('wrote %s' % html_out)
    print('wrote %d code pieces to %s' % (ncode, cache_dir))


if __name__ == '__main__':
    main()
