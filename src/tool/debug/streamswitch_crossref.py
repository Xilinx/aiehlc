#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Cross-reference the debug UI's per-tile stream-switch panel against source.

Two independent extractors reduce to the same canonical model:

  source  routing.cc / host.cc  -> XAie_Strm* call parse -> arbiter/msel pairing
  ui      routingprovenancemap.json -> schedule_view._load_comm_paths
          -> the real browser JS (renderTileRoutingSection) under node
          -> rendered rows parsed back into records

The source side never reads the provenance JSON and the UI side never reads
routing.cc, so a disagreement is a genuine UI accuracy defect.

Packet routes are resolved the way the switch itself resolves them: a slave
slot drives every master port on its tile whose arbiter matches the slot's
arbiter and whose MSelEn bitmask has the slot's msel line set.  The UI instead
borrows a "shared forward master" from a sibling provenance record, so the two
sides agree only when that heuristic happens to be right.

Static parsing covers generated code only.  Stream-switch writes issued at
runtime with computed arguments (core-trace setup, shim loopback) are out of
scope and will show up as UI rows with no source backing if ever enabled.

CLI:
    python3 streamswitch_crossref.py <workdir> [-v]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── canonical model ──────────────────────────────────────────────────────────

Cct  = namedtuple('Cct',  'col row sdir sidx mdir midx')
Pkt  = namedtuple('Pkt',  'col row sdir sidx mdir midx pktid mask')
Mst  = namedtuple('Mst',  'col row mdir midx arbiter msel_en')
Shim = namedtuple('Shim', 'col row kind idx')
Slot = namedtuple('Slot', 'col row sdir sidx slot pktid pkttype mask msel arbiter')

PORT_DIRS = ('SOUTH', 'NORTH', 'EAST', 'WEST', 'DMA', 'CORE', 'CTRL', 'FIFO',
             'TRACE', 'UCTRLR')

# Tokens routinghwlower.cpp emits when it fails to resolve a direction.
BAD_DIR_TOKENS = ('fixme', 'NONE', 'CONTROL', 'Unknown')

SOURCE_FILES = ('routing.cc', 'host.cc')


class SourceConfig(object):
    """Stream-switch state programmed by the generated source."""

    def __init__(self):
        self.cct = set()
        self.pkt = set()
        self.mst = set()
        self.shim = set()
        self.slots = []
        self.slave_enables = set()
        self.drop_header = set()
        self.warnings = []
        self.sources = []

    def tiles(self):
        return ({(r.col, r.row) for r in self.cct} |
                {(r.col, r.row) for r in self.pkt} |
                {(r.col, r.row) for r in self.mst})


class UiConfig(object):
    """Stream-switch rows the debug UI actually renders."""

    def __init__(self):
        self.cct = set()
        self.pkt = set()
        self.mst = set()
        self.fwd = set()
        self.shim = set()
        self.html = {}
        self.focus_rows = {}

    def tiles(self):
        return ({(r.col, r.row) for r in self.cct} |
                {(r.col, r.row) for r in self.pkt} |
                {(r.col, r.row) for r in self.mst})


class UiUnavailable(RuntimeError):
    pass


# ── C source scanning ────────────────────────────────────────────────────────

def _strip_comments(src):
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.S)
    return re.sub(r'//[^\n]*', '', src)


def _match_delim(src, i, opener, closer):
    depth = 0
    while i < len(src):
        if src[i] == opener:
            depth += 1
        elif src[i] == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _split_args(text):
    """Comma-split at depth 0.  Braces matter: XAie_Packet arrives both as
    XAie_PacketInit(id, type) and as {.PktId=0, .PktType=0}."""
    out, depth, cur = [], 0, ''
    for ch in text:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _iter_calls(src, name):
    for m in re.finditer(r'\b' + re.escape(name) + r'\s*\(', src):
        i = m.end() - 1
        j = _match_delim(src, i, '(', ')')
        if j < 0:
            continue
        yield _split_args(src[i + 1:j]), m.start()


def _tile_loc(arg):
    m = re.match(r'XAie_TileLoc\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$', arg)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _int(arg):
    arg = arg.strip()
    return int(arg, 0) if re.fullmatch(r'[-+]?(0[xX][0-9a-fA-F]+|\d+)', arg) else None


def _packet(arg):
    m = re.match(r'XAie_PacketInit\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$', arg)
    if m:
        return int(m.group(1)), int(m.group(2))
    pid = re.search(r'\.PktId\s*=\s*(\d+)', arg)
    ptype = re.search(r'\.PktType\s*=\s*(\d+)', arg)
    if arg.startswith('{') and pid and ptype:
        return int(pid.group(1)), int(ptype.group(1))
    return None


def _drop_header(arg):
    arg = arg.strip()
    if arg == 'XAIE_SS_PKT_DROP_HEADER':
        return True
    if arg == 'XAIE_SS_PKT_DONOT_DROP_HEADER':
        return False
    return None


# ── execution guards ─────────────────────────────────────────────────────────

def _const_bools(src):
    return {m.group(1): m.group(2) == 'true'
            for m in re.finditer(r'\bbool\s+(\w+)\s*=\s*(true|false)\s*;', src)}


def _pred_truth(pred, consts):
    pred = pred.strip()
    if pred in ('true', '1'):
        return True
    if pred in ('false', '0'):
        return False
    return consts.get(pred)


def _guards(src):
    """[(body_start, body_end, predicate, truth)] for if/for/while blocks."""
    consts = _const_bools(src)
    out = []
    for m in re.finditer(r'\b(if|for|while)\s*\(', src):
        i = m.end() - 1
        j = _match_delim(src, i, '(', ')')
        if j < 0:
            continue
        k = src.find('{', j)
        if k < 0 or src[j + 1:k].strip():
            continue
        end = _match_delim(src, k, '{', '}')
        if end < 0:
            continue
        pred = src[i + 1:j].strip()
        truth = _pred_truth(pred, consts) if m.group(1) == 'if' else None
        out.append((k, end, pred, truth))
    return out


def _guard_state(guards, pos):
    """(reached, unknown_predicates) for a call at byte offset pos."""
    unknown = []
    for start, end, pred, truth in guards:
        if not start <= pos <= end:
            continue
        if truth is False:
            return False, unknown
        if truth is None:
            unknown.append(pred)
    return True, unknown


# ── source parse ─────────────────────────────────────────────────────────────

class _Scanner(object):
    def __init__(self, src, cfg, origin):
        self.src = src
        self.cfg = cfg
        self.origin = origin
        self.guards = _guards(src)

    def live(self, pos, what):
        reached, unknown = self.guard_state(pos, what)
        return reached and not unknown

    def guard_state(self, pos, what):
        reached, unknown = _guard_state(self.guards, pos)
        if not reached:
            self.warn('%s skipped, guard is constant-false' % what)
        for pred in unknown:
            self.warn('%s under non-constant guard "%s"' % (what, pred))
        return reached, unknown

    def warn(self, msg):
        self.cfg.warnings.append('%s: %s' % (self.origin, msg))

    def port(self, arg, what):
        arg = arg.strip()
        if arg in PORT_DIRS:
            return arg
        if arg in BAD_DIR_TOKENS:
            self.warn('%s has unresolved port direction %r (emitter bug marker)'
                      % (what, arg))
        else:
            self.warn('%s has unrecognised port direction %r' % (what, arg))
        return arg


def _scan_circuit(sc):
    for name, add in (('XAie_StrmConnCctEnable', True),
                      ('XAie_StrmConnCctDisable', False)):
        for args, pos in _iter_calls(sc.src, name):
            if len(args) != 6:
                sc.warn('%s has %d args, expected 6' % (name, len(args)))
                continue
            loc = _tile_loc(args[1])
            if not loc:
                sc.warn('%s has non-literal tile loc %r' % (name, args[1]))
                continue
            reached, _ = sc.guard_state(pos, name)
            if not reached:
                continue
            rec = Cct(loc[0], loc[1], sc.port(args[2], name), _int(args[3]),
                      sc.port(args[4], name), _int(args[5]))
            sc.cfg.cct.add(rec) if add else sc.cfg.cct.discard(rec)


def _scan_slots(sc):
    name = 'XAie_StrmPktSwSlaveSlotEnable'
    for args, pos in _iter_calls(sc.src, name):
        if len(args) != 9:
            sc.warn('%s has %d args, expected 9' % (name, len(args)))
            continue
        loc = _tile_loc(args[1])
        pkt = _packet(args[5])
        if not loc or not _guard_state(sc.guards, pos)[0]:
            continue
        if not pkt:
            sc.warn('%s has unparsed packet %r' % (name, args[5]))
            continue
        sc.cfg.slots.append(Slot(loc[0], loc[1], sc.port(args[2], name),
                                 _int(args[3]), _int(args[4]), pkt[0], pkt[1],
                                 _int(args[6]), _int(args[7]), _int(args[8])))


def _scan_slave_enables(sc):
    name = 'XAie_StrmPktSwSlavePortEnable'
    for args, pos in _iter_calls(sc.src, name):
        loc = _tile_loc(args[1]) if len(args) == 4 else None
        if not loc or not _guard_state(sc.guards, pos)[0]:
            continue
        sc.cfg.slave_enables.add((loc[0], loc[1], sc.port(args[2], name),
                                  _int(args[3])))


def _scan_masters(sc):
    name = 'XAie_StrmPktSwMstrPortEnable'
    for args, pos in _iter_calls(sc.src, name):
        if len(args) != 7:
            sc.warn('%s has %d args, expected 7' % (name, len(args)))
            continue
        loc = _tile_loc(args[1])
        if not loc or not _guard_state(sc.guards, pos)[0]:
            continue
        mdir, midx = sc.port(args[2], name), _int(args[3])
        sc.cfg.mst.add(Mst(loc[0], loc[1], mdir, midx, _int(args[5]), _int(args[6])))
        drop = _drop_header(args[4])
        if drop is None:
            sc.warn('%s has unrecognised drop-header %r' % (name, args[4]))
        elif drop:
            sc.cfg.drop_header.add((loc[0], loc[1], mdir, midx))


def _scan_shim(sc):
    for name, kind in (('XAie_EnableShimDmaToAieStrmPort', 'ext_to_aie'),
                       ('XAie_EnableAieToShimDmaStrmPort', 'aie_to_ext')):
        for args, pos in _iter_calls(sc.src, name):
            loc = _tile_loc(args[1]) if len(args) == 3 else None
            if not loc or not _guard_state(sc.guards, pos)[0]:
                continue
            sc.cfg.shim.add(Shim(loc[0], loc[1], kind, _int(args[2])))


def pair_packet_routes(cfg):
    """Resolve slave slot -> master port the way the switch arbiters do."""
    by_tile = {}
    for m in cfg.mst:
        by_tile.setdefault((m.col, m.row), []).append(m)
    for s in cfg.slots:
        if (s.col, s.row, s.sdir, s.sidx) not in cfg.slave_enables:
            cfg.warnings.append(
                'tile(%d,%d) slot %s:%s has no StrmPktSwSlavePortEnable'
                % (s.col, s.row, s.sdir, s.sidx))
        hits = [m for m in by_tile.get((s.col, s.row), [])
                if m.arbiter == s.arbiter and ((m.msel_en >> s.msel) & 1)]
        if not hits:
            cfg.warnings.append(
                'tile(%d,%d) slot %s:%s (arbiter=%s msel=%s) drives no master'
                % (s.col, s.row, s.sdir, s.sidx, s.arbiter, s.msel))
        for m in hits:
            cfg.pkt.add(Pkt(s.col, s.row, s.sdir, s.sidx, m.mdir, m.midx,
                            s.pktid, s.mask))


def parse_source_text(text, origin='<text>', cfg=None):
    cfg = cfg or SourceConfig()
    sc = _Scanner(_strip_comments(text), cfg, origin)
    _scan_circuit(sc)
    _scan_slots(sc)
    _scan_slave_enables(sc)
    _scan_masters(sc)
    _scan_shim(sc)
    return cfg


def parse_source(workdir, names=SOURCE_FILES):
    """Union of the stream-switch programming in every generated source file."""
    cfg = SourceConfig()
    for name in names:
        path = os.path.join(workdir, name)
        if os.path.exists(path):
            with open(path, errors='replace') as f:
                parse_source_text(f.read(), name, cfg)
            cfg.sources.append(name)
    if not cfg.sources:
        raise FileNotFoundError('no %s in %s' % (' or '.join(names), workdir))
    pair_packet_routes(cfg)
    return cfg


# ── UI extraction (runs the real browser JS) ─────────────────────────────────

UI_FUNCS = (
    'esc', '_fmtDmaChanBadge', '_flowDmaBadgeStyle', '_flowDmaSpanHtml',
    '_flowTileSet', '_isPktPort', '_sharedPktForwardMaster', '_defaultPktMask',
    '_resolvePktMask', '_fmtPktMaskHex', '_fmtPktMaskBadge',
    '_expandPktConnectRows', '_pktRowKey', '_fmtMselEnHex', '_tilePktMasters',
    '_renderPktMasterBlock', '_tileRoutingConns', '_filterRoutingConns',
    'renderTileRoutingSection', '_tileCommPaths', '_routingDiffSides',
    '_routingSrcDiffTile', '_swKey', '_swNormSlave', '_swRecordKey',
    '_swCoarseKey', '_swConnCoarseKey', '_swScanTile', '_swMissingRecordKeys',
    '_swMissingKeys', '_swOnlyLabel', '_swDiffBadHtml', '_swBadHtml',
    '_swOkHtml', '_swMarkHtmlCct', '_swMarkHtmlPkt', '_swMarkHtmlMst',
    '_swExtraRowHtml', '_swExtraRowsHtml', '_swExtraMstHtml',
)

# Top-level declarations the extracted functions close over.
UI_PRELUDE = ('SWSCAN',)

_JS_DRIVER = r'''
const fs = require('fs');
const inp = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const DATA = inp.data;
const STATIC_PATHS = DATA.comm_paths || [];
const ROUTING_DIFF = false;
const ROUTING_SRC = 'static';
let DYNAMIC = null;
const key = t => t[0] + ',' + t[1];
const out = {none: {}, focus: {}, flows_table: {}};
for (const t of inp.tiles) {
  out.none[key(t)] = renderTileRoutingSection(t[0], t[1], null);
  out.flows_table[key(t)] = _tileCommPaths(t[0], t[1]).map(p => p.flow_index);
}
for (const fi of inp.flows) {
  const m = {};
  for (const t of inp.tiles) m[key(t)] = renderTileRoutingSection(t[0], t[1], fi);
  out.focus[fi] = m;
}
process.stdout.write(JSON.stringify(out));
'''


def extract_js_function(template, name):
    m = re.search(r'^function\s+' + re.escape(name) + r'\s*\(', template, re.M)
    if not m:
        raise KeyError('function %s not found in HTML_TEMPLATE' % name)
    i = template.index('{', m.end() - 1)
    end = _match_delim(template, i, '{', '}')
    if end < 0:
        raise ValueError('unbalanced body for %s' % name)
    return template[m.start():end + 1]


def extract_js_statement(template, name):
    """A top-level `const NAME = ...;` / `let NAME = ...;`, braces balanced."""
    m = re.search(r'^(?:const|let|var)\s+' + re.escape(name) + r'\b', template, re.M)
    if not m:
        raise KeyError('statement %s not found in HTML_TEMPLATE' % name)
    depth, i = 0, m.end()
    while i < len(template):
        ch = template[i]
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == ';' and depth == 0:
            return template[m.start():i + 1]
        i += 1
    raise ValueError('unterminated statement %s' % name)


def extract_js_expression(template, anchor):
    """The statement beginning at `anchor` (a regex), braces balanced."""
    m = re.search(anchor, template, re.M)
    if not m:
        raise KeyError('anchor %r not found in HTML_TEMPLATE' % anchor)
    depth, i = 0, m.start()
    while i < len(template):
        ch = template[i]
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == ';' and depth == 0:
            return template[m.start():i + 1]
        i += 1
    raise ValueError('unterminated expression %r' % anchor)


def build_ui_script(template, names=UI_FUNCS, prelude=UI_PRELUDE):
    parts = [extract_js_statement(template, n) for n in prelude]
    parts += [extract_js_function(template, n) for n in names]
    return '\n'.join(parts) + _JS_DRIVER


def _ui_data(workdir, schedule_view):
    comm_paths = schedule_view._load_comm_paths(workdir)
    try:
        tiles = schedule_view.build_view(workdir)['tiles']
    except Exception:
        tiles = []
    return {'comm_paths': comm_paths, 'tiles': tiles}


def _ui_tile_list(data, extra_tiles=()):
    """Full rectangle covering every tile either side knows about.

    Deliberately not DATA.tiles: memtile rows carry real circuit connections
    but never appear in the schedule grid.
    """
    seen = set(extra_tiles)
    for p in data['comm_paths']:
        for e in p.get('edges', []):
            seen.add((e[0][0], e[0][1]))
            seen.add((e[1][0], e[1][1]))
        for group in ('tiles', 'dma_tiles', 'packet_tiles'):
            seen.update((t[0], t[1]) for t in p.get(group, []))
        for h in p.get('hops', []):
            seen.add((h['from_col'], h['from_row']))
            seen.add((h['to_col'], h['to_row']))
        for c in p.get('routing_connections', []):
            t = c.get('tile') or {}
            if t.get('col') is not None:
                seen.add((t['col'], t['row']))
    seen.update(tuple(t['loc']) for t in data['tiles'])
    if not seen:
        return []
    cols = max(c for c, _ in seen)
    rows = max(r for _, r in seen)
    return [[c, r] for c in range(cols + 1) for r in range(rows + 1)]


def _run_node(script, payload):
    node = shutil.which('node') or shutil.which('nodejs')
    if not node:
        raise UiUnavailable('node is required to run the debug UI JS')
    tmp = tempfile.mkdtemp(prefix='ssxref')
    try:
        js, inp = os.path.join(tmp, 'render.js'), os.path.join(tmp, 'data.json')
        with open(js, 'w') as f:
            f.write(script)
        with open(inp, 'w') as f:
            json.dump(payload, f)
        proc = subprocess.run([node, js, inp], capture_output=True, text=True)
        if proc.returncode:
            raise UiUnavailable('node failed: %s' % proc.stderr.strip()[:2000])
        return json.loads(proc.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def render_ui_html(workdir, extra_tiles=()):
    """{'none': {tile: html}, 'focus': {flow: {tile: html}}} from the real UI."""
    import schedule_view

    data = _ui_data(workdir, schedule_view)
    flows = sorted({p['flow_index'] for p in data['comm_paths']
                    if p.get('flow_index') is not None})
    payload = {'data': data, 'tiles': _ui_tile_list(data, extra_tiles),
               'flows': flows}
    rendered = _run_node(build_ui_script(schedule_view.HTML_TEMPLATE), payload)
    rendered['shim'] = _ui_shim_records(data['comm_paths'])
    return rendered


def _ui_shim_records(comm_paths):
    """Shim port enables that survive into the UI's data at all.

    The panel never renders these; this only distinguishes "filtered out of the
    tile view" from "destroyed before any UI surface could show it".
    """
    kinds = {'shim_ext_to_aie': 'ext_to_aie', 'shim_aie_to_ext': 'aie_to_ext'}
    out = set()
    for p in comm_paths:
        for c in p.get('routing_connections', []):
            kind = kinds.get(c.get('kind'))
            if not kind:
                continue
            t = c.get('tile') or {}
            idx = c.get('stream_id')
            if idx is None:
                idx = (c.get('port') or {}).get('idx')
            out.add(Shim(t.get('col'), t.get('row'), kind, idx))
    return out


# ── UI HTML -> records ───────────────────────────────────────────────────────

_ROW_RE   = re.compile(r'<div class="rt-row (cct|pkt|mst)">(.*?)(?=<div class="rt-|</div>\s*$)')
_PORTS_RE = re.compile(r'<span class="rt-ports">(.*?)</span>')
_PKTID_RE = re.compile(r'>pkt(\d+)</span>')
_MASK_RE  = re.compile(r'class="rt-pktmask"[^>]*>0x([0-9A-Fa-f]+)<')
_ARB_RE   = re.compile(r'>arb:(\d+)</span>')
_MSEL_RE  = re.compile(r'>msel_en:0x([0-9A-Fa-f]+)<')


def _decode_ports(text):
    text = text.replace('&nbsp;', ' ').replace('&rarr;', '->').replace('&gt;', '>')
    m = re.match(r'\s*(\w+):(\d+)\s*->\s*(\w+):(\d+)\s*$', text)
    if m:
        return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    m = re.match(r'\s*(?:fwd\s*->\s*)?(\w+):(\d+)\s*$', text)
    if m:
        return None, None, m.group(1), int(m.group(2))
    raise ValueError('unrecognised rt-ports %r' % text)


def _collect_rows(html_by_tile, ui):
    for key, html in html_by_tile.items():
        if not html:
            continue
        col, row = (int(x) for x in key.split(','))
        for kind, chunk in _ROW_RE.findall(html):
            ports = _PORTS_RE.search(chunk)
            if not ports:
                continue
            sdir, sidx, mdir, midx = _decode_ports(ports.group(1))
            if kind == 'cct':
                ui.cct.add(Cct(col, row, sdir, sidx, mdir, midx))
            elif kind == 'pkt':
                pktid, mask = _PKTID_RE.search(chunk), _MASK_RE.search(chunk)
                rec = Pkt(col, row, sdir, sidx, mdir, midx,
                          int(pktid.group(1)) if pktid else None,
                          int(mask.group(1), 16) if mask else None)
                (ui.fwd if sdir is None else ui.pkt).add(rec)
            else:
                arb, msel = _ARB_RE.search(chunk), _MSEL_RE.search(chunk)
                ui.mst.add(Mst(col, row, mdir, midx,
                               int(arb.group(1)) if arb else None,
                               int(msel.group(1), 16) if msel else None))


def parse_ui_html(rendered):
    ui = UiConfig()
    ui.html = rendered['none']
    ui.shim = set(rendered.get('shim', ()))
    _collect_rows(rendered['none'], ui)
    for flow, tiles in rendered.get('focus', {}).items():
        sub = UiConfig()
        _collect_rows(tiles, sub)
        ui.focus_rows[int(flow)] = sub
    return ui


def render_ui(workdir, extra_tiles=()):
    return parse_ui_html(render_ui_html(workdir, extra_tiles))


# ── cross-reference ──────────────────────────────────────────────────────────

PROVENANCE_FILES = ('routingprovenancemap.json', 'dmaphopprovenacemap.json')


def missing_provenance(workdir):
    """Inputs the UI needs before it can show any stream-switch config."""
    return [n for n in PROVENANCE_FILES
            if not os.path.exists(os.path.join(workdir, n))]


class Report(object):
    def __init__(self, workdir, source, ui):
        self.workdir = workdir
        self.source = source
        self.ui = ui
        self.provenance_missing = missing_provenance(workdir)
        self.cct_missing = sorted(source.cct - ui.cct)
        self.cct_extra = sorted(ui.cct - source.cct)
        self.pkt_missing = sorted(source.pkt - ui.pkt)
        self.pkt_extra = sorted(ui.pkt - source.pkt)
        self.mst_missing = sorted(source.mst - ui.mst)
        self.mst_extra = sorted(ui.mst - source.mst)
        self.fwd_rows = sorted(ui.fwd)
        self.fwd_unbacked = sorted(
            r for r in ui.fwd
            if not any(m[:4] == (r.col, r.row, r.mdir, r.midx) for m in source.mst))
        self.shim_unreachable = sorted(source.shim - ui.shim)
        self.tiles_missing = sorted(source.tiles() - ui.tiles())
        self.drop_header_hidden = sorted(source.drop_header)
        self.focus_lost = self._focus_lost()

    def _focus_lost(self):
        """Rows visible unfocused that no focus selection can bring back."""
        if not self.ui.focus_rows:
            return []
        union_cct, union_pkt = set(), set()
        for sub in self.ui.focus_rows.values():
            union_cct |= sub.cct
            union_pkt |= sub.pkt
        return sorted((self.ui.cct - union_cct) | (self.ui.pkt - union_pkt),
                      key=repr)

    @property
    def deviations(self):
        """Rows the UI gets wrong.  Empty while provenance input is missing:
        an absent routingprovenancemap.json is a pipeline gap, not a UI that
        misreports what it was given."""
        if self.provenance_missing:
            return []
        return (self.cct_missing + self.cct_extra + self.pkt_missing +
                self.pkt_extra + self.mst_missing + self.mst_extra +
                self.fwd_unbacked + self.focus_lost)

    def ok(self):
        return not self.deviations and not self.provenance_missing


def crossref(workdir):
    source = parse_source(workdir)
    ui = render_ui(workdir, extra_tiles=source.tiles())
    return Report(workdir, source, ui)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _fmt(rec):
    if isinstance(rec, Cct):
        return 'CCT (%d,%d) %s:%s -> %s:%s' % rec
    if isinstance(rec, Pkt):
        slave = 'fwd' if rec.sdir is None else '%s:%s' % (rec.sdir, rec.sidx)
        mask = rec.mask if rec.mask is None else hex(rec.mask)
        return 'PKT (%d,%d) %s -> %s:%s pkt%s mask=%s' % (
            rec.col, rec.row, slave, rec.mdir, rec.midx, rec.pktid, mask)
    if isinstance(rec, Mst):
        return 'MST (%d,%d) %s:%s arbiter=%s msel_en=%s' % rec
    if isinstance(rec, Shim):
        return 'SHIM (%s,%s) %s port %s' % rec
    return str(rec)


def _section(title, rows):
    if not rows:
        return
    print('  %s (%d)' % (title, len(rows)))
    for r in rows:
        print('    %s' % _fmt(r))


def report_text(rep, verbose=False):
    src, ui = rep.source, rep.ui
    print('workdir : %s' % rep.workdir)
    print('sources : %s' % ', '.join(src.sources))
    print('source  : %d circuit, %d packet routes, %d packet masters, %d shim'
          % (len(src.cct), len(src.pkt), len(src.mst), len(src.shim)))
    print('ui      : %d circuit, %d packet routes, %d packet masters, %d fwd-only'
          % (len(ui.cct), len(ui.pkt), len(ui.mst), len(ui.fwd)))
    for w in src.warnings:
        print('warning : %s' % w)
    if rep.provenance_missing:
        print('BLOCKED : %s absent — the UI has no stream-switch data to show '
              'for any of the %d connections in source'
              % (', '.join(rep.provenance_missing), len(src.cct) + len(src.pkt)))
        print('result  : NO UI DATA')
        return
    _section('circuit connections missing from UI', rep.cct_missing)
    _section('circuit connections the UI invents', rep.cct_extra)
    _section('packet routes missing from UI', rep.pkt_missing)
    _section('packet routes the UI invents', rep.pkt_extra)
    _section('packet master ports missing from UI', rep.mst_missing)
    _section('packet master ports the UI invents', rep.mst_extra)
    _section('forward-only rows with no master port behind them', rep.fwd_unbacked)
    _section('rows no flow focus can reach', rep.focus_lost)
    if rep.tiles_missing:
        print('  tiles with config but no UI panel: %s' % rep.tiles_missing)
    _section('shim port enables absent from every UI surface', rep.shim_unreachable)
    if rep.drop_header_hidden:
        print('  note: %d master port(s) drop the packet header; the UI has no '
              'field for it' % len(rep.drop_header_hidden))
    if verbose:
        _section('forward-only rows (view artifact, backed by a master)',
                 rep.fwd_rows)
    print('result  : %s' % ('MATCH' if rep.ok()
                            else '%d DEVIATIONS' % len(rep.deviations)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('workdir')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)
    rep = crossref(args.workdir)
    report_text(rep, args.verbose)
    return 0 if rep.ok() else 1


if __name__ == '__main__':
    sys.exit(main())
