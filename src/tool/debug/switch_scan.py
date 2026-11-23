#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Read the live stream switch of a tile and diff it against the provenance map.

aiegdb's `show switch` / `scan switch` decode one tile over `aiedbg reg read`,
one subprocess per register -- 228 of them for a core tile.  The whole switch
register block is contiguous and fits inside aiedbg's 256-word `mem read`
limit, so on hardware this reads it in a single call per tile and decodes with
the same `aiediag` helpers.  On the simulator the per-register IPC reader is
already cheap and is used directly.

Both sides reduce to the record types `streamswitch_crossref` already defines,
so "what the hardware is programmed with" and "what the UI claims" are
comparable without a translation layer.

Packet routes are resolved the way the switch arbiters do: a slave slot drives
every master port on the tile whose arbiter matches and whose MSelEn has the
slot's msel line set.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiediag
from streamswitch_crossref import Cct, Mst, Pkt

OK, MISMATCH, UNREACHABLE, EMPTY = 'verified', 'mismatch', 'unreachable', 'idle'


# ── one contiguous register block per tile ───────────────────────────────────

def block_span(tile_type):
    """(base_offset, nwords) covering masters, slaves and every slave slot."""
    base = aiediag.STRM_SW_BASE[tile_type]['master']
    n_slaves = len(aiediag.STRM_SW_SLAVE_PORTS[tile_type])
    last = aiediag.strm_slot_off(tile_type, n_slaves - 1,
                                 aiediag.STRM_SW_SLOTS_PER_SLAVE - 1)
    return base, (last + 4 - base) // 4


def reader_from_words(tile_type, words):
    """Turn one block read into the reader(off) signature read_switch wants."""
    base, _ = block_span(tile_type)

    def read(off):
        idx = (off - base) // 4
        return words[idx] if 0 <= idx < len(words) else None
    return read


def read_tile_switch(tile_type, phys_col, row, reg_read_fn=None,
                     target=None, device=None):
    """Decoded switch for one tile, or None if it could not be read.

    reg_read_fn(phys_col, row, off) -> int|None is the simulator path; without
    it this issues a single `aiedbg mem read` for the whole block.
    """
    if reg_read_fn is not None:
        return aiediag.read_switch(
            tile_type, lambda off: reg_read_fn(phys_col, row, off))
    base, nwords = block_span(tile_type)
    words = aiediag.run_aiedbg_mem_read(phys_col, row, base, nwords,
                                        target=target, device=device)
    if not words:
        return None
    return aiediag.read_switch(tile_type, reader_from_words(tile_type, words))


# ── decoded registers -> canonical records ───────────────────────────────────

def live_records(tile_type, col, row, decoded):
    """(cct, pkt, mst) sets for one tile, from its decoded registers."""
    slave_ports = aiediag.STRM_SW_SLAVE_PORTS[tile_type]
    cct, mst = set(), set()
    for m in decoded['masters']:
        if not m['enable']:
            continue
        if m['packet']:
            mst.add(Mst(col, row, m['type'], m['num'],
                        m['arbiter'], m['msel_en']))
            continue
        idx = m['slave_idx']
        if idx is None or not 0 <= idx < len(slave_ports):
            continue
        sdir, sidx = slave_ports[idx]
        cct.add(Cct(col, row, sdir, sidx, m['type'], m['num']))

    pkt = set()
    for s in decoded['slaves']:
        if not (s['enable'] and s['packet']):
            continue
        for slot in s['slots']:
            for m in mst:
                if m.arbiter == slot['arbiter'] and \
                        ((m.msel_en >> slot['msel']) & 1):
                    pkt.add(Pkt(col, row, s['type'], s['num'],
                                m.mdir, m.midx, slot['id'], slot['mask']))
    return cct, pkt, mst


def expected_records(comm_paths, col, row):
    """(cct, pkt, mst) the provenance map says this tile should hold.

    Mirrors what the tile panel renders: circuit connections verbatim, packet
    legs expanded per slave, and the forward master of each packet group.
    """
    cct, pkt, mst = set(), set(), set()
    for path in comm_paths:
        conns = [c for c in path.get('routing_connections', [])
                 if (c.get('tile') or {}).get('col') == col
                 and (c.get('tile') or {}).get('row') == row]
        shared = next((c['forward_master'] for c in conns
                       if c.get('kind') == 'packet_connect'
                       and (c.get('forward_master') or {}).get('dir')
                       not in (None, 'NONE')), None)
        for c in conns:
            if c.get('kind') == 'circuit_connect':
                s, m = c.get('slave') or {}, c.get('master') or {}
                cct.add(Cct(col, row, s.get('dir'), s.get('idx'),
                            m.get('dir'), m.get('idx')))
            elif c.get('kind') == 'packet_connect':
                fm = c.get('forward_master') or {}
                master = fm if fm.get('dir') not in (None, 'NONE') else shared
                if not master:
                    continue
                mst.add(Mst(col, row, master['dir'], master['idx'],
                            master.get('arbiter', 0), master.get('msel_en', 1)))
                for leg, default_mask in (('recv_slave', 0), ('local_dma', 0x1f)):
                    port = c.get(leg) or {}
                    if port.get('dir') in (None, 'NONE'):
                        continue
                    pkt.add(Pkt(col, row, port['dir'], port['idx'],
                                master['dir'], master['idx'],
                                port.get('pktid'),
                                port.get('mask', default_mask)))
    return cct, pkt, mst


# ── comparison ───────────────────────────────────────────────────────────────

def compare_tile(live, expected):
    """{state, missing, unexpected, matched} for one tile.

    `missing`  — the provenance map claims it, the hardware is not programmed
                 with it.  A flow that never moves data looks like this.
    `unexpected` — the hardware has it and no flow accounts for it.  Left over
                 from a previous run, or a routing bug the UI cannot see.
    """
    lc, lp, lm = live
    ec, ep, em = expected
    missing = sorted(ec - lc, key=repr) + sorted(ep - lp, key=repr) + \
        sorted(em - lm, key=repr)
    unexpected = sorted(lc - ec, key=repr) + sorted(lp - ep, key=repr) + \
        sorted(lm - em, key=repr)
    matched = len(ec & lc) + len(ep & lp) + len(em & lm)
    if missing or unexpected:
        state = MISMATCH
    elif matched:
        state = OK
    else:
        state = EMPTY
    return {'state': state, 'matched': matched,
            'missing': [_json(r) for r in missing],
            'unexpected': [_json(r) for r in unexpected]}


def _json(rec):
    """Record -> a flat dict the browser can render."""
    if isinstance(rec, Cct):
        return {'kind': 'CCT', 'slave': '%s:%s' % (rec.sdir, rec.sidx),
                'master': '%s:%s' % (rec.mdir, rec.midx)}
    if isinstance(rec, Pkt):
        slave = 'fwd' if rec.sdir is None else '%s:%s' % (rec.sdir, rec.sidx)
        return {'kind': 'PKT', 'slave': slave,
                'master': '%s:%s' % (rec.mdir, rec.midx),
                'pktid': rec.pktid, 'mask': rec.mask}
    return {'kind': 'MST', 'slave': None,
            'master': '%s:%s' % (rec.mdir, rec.midx),
            'arbiter': rec.arbiter, 'msel_en': rec.msel_en}


def scan_tile(tile_type, col, row, phys_col, comm_paths, reg_read_fn=None,
              target=None, device=None):
    """Read one tile's switch and diff it against the provenance map."""
    cell, _ = scan_tile_decoded(tile_type, col, row, phys_col, comm_paths,
                               reg_read_fn=reg_read_fn, target=target,
                               device=device)
    return cell


def scan_tile_decoded(tile_type, col, row, phys_col, comm_paths,
                      reg_read_fn=None, target=None, device=None):
    """(cell, decoded) -- the diff plus the registers it was derived from.

    `switch_reconstruct` needs the decoded switch and the diff needs the same
    read; handing both back keeps a scan at one board round-trip per tile.
    """
    decoded = read_tile_switch(tile_type, phys_col, row,
                               reg_read_fn=reg_read_fn, target=target,
                               device=device)
    if decoded is None:
        return ({'state': UNREACHABLE, 'phys_col': phys_col, 'matched': 0,
                 'missing': [], 'unexpected': []}, None)
    result = compare_tile(live_records(tile_type, col, row, decoded),
                          expected_records(comm_paths, col, row))
    result['phys_col'] = phys_col
    return result, decoded
