#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Does the debug UI attribute the right info to every tile and every flow?

streamswitch_crossref answers "is this connection real".  This answers "does it
belong to the flow and the tile the UI says it does".

Three inputs, three different emitters:

  routing.cc                    routinghw lowering  -> per-group connections,
                                                       split push / pull
  dmaphopprovenacemap.json      dataflowmap passes  -> per-flow DMA endpoints
                                                       and hop chain
  routingprovenancemap.json     routing provenance  -> what the UI actually
                                                       consumes

routing.cc emits one `if (v2) { ... }` block per routing group, in the same
order as routing_groups.  Each block's AIE->DDR egress call splits it the way
`shim_aie_to_ext` splits the group, so a block yields a push record set and a
pull record set.

Groups carry no flow_index in the tiling flow, so the UI joins group to flow on
a frozenset of DMA tiles -- its own comments call that ambiguous.  This module
joins independently, on DMA endpoints including the port index plus the shim
column, and reports where the two joins disagree.

CLI:
    python3 flow_crossref.py <workdir> [-v]
"""

import argparse
import json
import os
import re
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamswitch_crossref as sx

Endpoint = namedtuple('Endpoint', 'col row port')
ShimPort = namedtuple('ShimPort', 'col row stream')

PUSH, PULL = 'push', 'pull'


class SourceGroup(object):
    """One `if (v2) { ... }` block of routing.cc."""

    def __init__(self, index, memo, rnd):
        self.index = index
        self.memo = memo
        self.round = rnd
        self.records = {PUSH: set(), PULL: set()}
        self.dma = {PUSH: set(), PULL: set()}
        self.ingress = []
        self.egress = []

    def tiles(self, section):
        return {(r.col, r.row) for r in self.records[section]}

    def sections(self):
        return [s for s in (PUSH, PULL) if self.records[s]]

    def __repr__(self):
        return 'block%d(%s round %s)' % (self.index, self.memo, self.round)


class Flow(object):
    """One communication_path of dmaphopprovenacemap.json."""

    def __init__(self, index, path_id, direction):
        self.index = index
        self.id = path_id
        self.direction = direction
        self.producers = set()
        self.consumers = set()
        self.hop_tiles = set()

    @property
    def aie_endpoints(self):
        """DMA endpoints on the AIE side -- consumers when pushing into the
        array, producers when pulling out of it."""
        return self.consumers if self.direction == PUSH else self.producers

    @property
    def shim_endpoints(self):
        return self.producers if self.direction == PUSH else self.consumers

    def __repr__(self):
        return 'f%d(%s %s)' % (self.index, self.direction, self.id)


# ── routing.cc -> ordered groups ─────────────────────────────────────────────

_ROUND_RE = re.compile(r'//\s*round is (\d+) hw split in\s*:\s*(\w+)')


def _iter_blocks(raw):
    """(round, memo, body) per top-level `if (...) { ... }`, in source order."""
    for m in re.finditer(r'\bif\s*\(', raw):
        i = m.end() - 1
        j = sx._match_delim(raw, i, '(', ')')
        if j < 0:
            continue
        k = raw.find('{', j)
        if k < 0 or raw[j + 1:k].strip():
            continue
        end = sx._match_delim(raw, k, '{', '}')
        if end < 0:
            continue
        head = _ROUND_RE.findall(raw[max(0, m.start() - 300):m.start()])
        rnd, memo = head[-1] if head else (None, None)
        yield rnd, memo, raw[k:end + 1]


def _section_at(cut, pos):
    return PUSH if pos < cut else PULL


def _block_records(body, group):
    """Fill a SourceGroup from one block body, split at the egress marker."""
    src = sx._strip_comments(body)
    egress = list(sx._iter_calls(src, 'XAie_EnableAieToShimDmaStrmPort'))
    cut = egress[0][1] if egress else len(src)

    for args, pos in sx._iter_calls(src, 'XAie_StrmConnCctEnable'):
        loc = sx._tile_loc(args[1])
        if not loc:
            continue
        sec = _section_at(cut, pos)
        group.records[sec].add(sx.Cct(loc[0], loc[1], args[2].strip(),
                                      sx._int(args[3]), args[4].strip(),
                                      sx._int(args[5])))
        if args[4].strip() == 'DMA':
            group.dma[sec].add(Endpoint(loc[0], loc[1], sx._int(args[5])))

    for args, pos in sx._iter_calls(src, 'XAie_StrmPktSwSlaveSlotEnable'):
        loc = sx._tile_loc(args[1])
        if loc and args[2].strip() == 'DMA':
            # Packet gather legs sit before the egress marker but feed the pull
            # direction; the DDR-bound section is the one that owns them.
            sec = PULL if egress else PUSH
            group.dma[sec].add(Endpoint(loc[0], loc[1], sx._int(args[3])))

    for args, pos in sx._iter_calls(src, 'XAie_StrmPktSwMstrPortEnable'):
        loc = sx._tile_loc(args[1])
        if loc:
            group.records[PULL if egress else PUSH].add(
                sx.Mst(loc[0], loc[1], args[2].strip(), sx._int(args[3]),
                       sx._int(args[5]), sx._int(args[6])))

    for name, bucket in (('XAie_EnableShimDmaToAieStrmPort', group.ingress),
                         ('XAie_EnableAieToShimDmaStrmPort', group.egress)):
        for args, _ in sx._iter_calls(src, name):
            loc = sx._tile_loc(args[1])
            if loc:
                bucket.append(ShimPort(loc[0], loc[1], sx._int(args[2])))


def parse_source_groups(workdir, name='routing.cc'):
    path = os.path.join(workdir, name)
    if not os.path.exists(path):
        return []
    with open(path, errors='replace') as f:
        raw = f.read()
    groups = []
    for index, (rnd, memo, body) in enumerate(_iter_blocks(raw)):
        group = SourceGroup(index, memo, rnd)
        _block_records(body, group)
        groups.append(group)
    return groups


# ── dmaphopprovenacemap.json -> flows ────────────────────────────────────────

_COORD_RE = re.compile(r'\((\d+),\s*(\d+)\)')
_FSYM_RE = re.compile(r'f(\d+)_')


def _stage_endpoints(stage):
    if not stage:
        return set()
    if 'tiles' in stage:
        return {Endpoint(t['col'], t['row'], t.get('dma_port'))
                for t in stage['tiles']}
    tile = stage.get('tile') or {}
    if tile.get('col') is None:
        return set()
    return {Endpoint(tile['col'], tile['row'], stage.get('channel'))}


def _flow_index(path, stages):
    for stage in stages:
        m = _FSYM_RE.match(stage.get('port_sym') or '')
        if m:
            return int(m.group(1))
        for t in stage.get('tiles', []):
            m = _FSYM_RE.match(t.get('port_sym') or '')
            if m:
                return int(m.group(1))
    m = re.search(r'(\d+)$', path.get('id', ''))
    return int(m.group(1)) if m else None


def parse_flows(workdir):
    path = os.path.join(workdir, 'dmaphopprovenacemap.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    flows = []
    for p in data.get('communication_paths', []):
        stages = p.get('stages', [])
        index = _flow_index(p, stages)
        if index is None:
            continue
        flow = Flow(index, p.get('id', ''), p.get('direction', PUSH))
        for stage in stages:
            if stage.get('role') == 'producer':
                flow.producers |= _stage_endpoints(stage)
            elif stage.get('role') == 'consumer':
                flow.consumers |= _stage_endpoints(stage)
            for h in stage.get('hops', []):
                for side in ('from', 'to'):
                    m = _COORD_RE.search(h.get(side, ''))
                    if m:
                        flow.hop_tiles.add((int(m.group(1)), int(m.group(2))))
        flows.append(flow)
    return flows


# ── independent group <-> flow join ──────────────────────────────────────────

def join_flows(groups, flows):
    """{flow_index: (group, section)} matched on DMA endpoints + shim column.

    Returns (join, problems).  Never guesses: a flow that matches zero or more
    than one (group, section) is left unjoined and reported.
    """
    join, problems = {}, []
    for flow in flows:
        section = PUSH if flow.direction == PUSH else PULL
        wanted = flow.aie_endpoints
        cands = [g for g in groups if g.dma.get(section) == wanted]
        if len(cands) > 1:
            cands = [g for g in cands if _shim_matches(g, flow, section)]
        if len(cands) == 1:
            join[flow.index] = (cands[0], section)
        else:
            problems.append(
                'f%d (%s) matches %d source blocks on DMA endpoints %s'
                % (flow.index, flow.direction, len(cands), sorted(wanted)))
    return join, problems


def _shim_matches(group, flow, section):
    ports = group.ingress if section == PUSH else group.egress
    cols = {p.col for p in ports}
    return not cols or cols & {e.col for e in flow.shim_endpoints}


def group_key_collisions(workdir):
    """Groups the UI cannot tell apart: identical DMA-tile frozensets.

    routing_edges_for_flow keys on this set whenever groups carry no
    flow_index, and silently keeps the last writer.
    """
    path = os.path.join(workdir, 'routingprovenancemap.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        groups = json.load(f).get('routing_groups', [])
    if any('flow_index' in g for g in groups):
        return []
    seen = {}
    for g in groups:
        key = frozenset(
            (c['tile']['col'], c['tile']['row']) for c in g.get('connections', [])
            if c.get('kind') == 'circuit_connect'
            and c.get('master', {}).get('dir') == 'DMA')
        seen.setdefault(key, []).append(g.get('id'))
    return [ids for ids in seen.values() if len(ids) > 1]


# ── UI side ──────────────────────────────────────────────────────────────────

def _conn_records(conns):
    """routing_connections -> the same record types the source side produces."""
    cct, pkt_slaves, mst = set(), set(), set()
    for c in conns:
        tile = c.get('tile') or {}
        col, row = tile.get('col'), tile.get('row')
        if c.get('kind') == 'circuit_connect':
            s, m = c.get('slave') or {}, c.get('master') or {}
            cct.add(sx.Cct(col, row, s.get('dir'), s.get('idx'),
                           m.get('dir'), m.get('idx')))
        elif c.get('kind') == 'packet_connect':
            for leg in ('recv_slave', 'local_dma'):
                port = c.get(leg) or {}
                if port.get('dir') not in (None, 'NONE'):
                    pkt_slaves.add((col, row, port['dir'], port['idx']))
            fm = c.get('forward_master') or {}
            if fm.get('dir') not in (None, 'NONE'):
                mst.add(sx.Mst(col, row, fm['dir'], fm['idx'],
                               fm.get('arbiter', 0), fm.get('msel_en', 1)))
    return cct, pkt_slaves, mst


class UiFlows(object):
    """What the UI attributes to each flow and each tile."""

    def __init__(self, comm_paths, tiles, focus_rows):
        self.paths = {p['flow_index']: p for p in comm_paths}
        self.tiles = tiles
        self.focus_rows = focus_rows
        self.focus_html = {}
        self.net_html = {}
        self.flows_table_html = {}

    def channel_owners(self):
        """{(tile, badge_label): {flow_index}} — who each DMA badge belongs to."""
        out = {}
        for t in self.tiles:
            for ch in t.get('dma_channels') or []:
                label = ('mm2s' if ch.get('direction') == 'MM2S' else 's2mm')
                key = (tuple(t['loc']), label + str(ch.get('channel')))
                out.setdefault(key, set()).add(ch.get('flow_index'))
        return out

    def flow_tiles_with_dma(self, flow_index):
        out = set()
        for t in self.tiles:
            if any(ch.get('flow_index') == flow_index
                   for ch in t.get('dma_channels') or []):
                out.add(tuple(t['loc']))
        return out

    def gmio_conns(self, flow_index):
        kinds = {'shim_ext_to_aie': 'ext_to_aie', 'shim_aie_to_ext': 'aie_to_ext'}
        out = set()
        for c in self.paths[flow_index].get('routing_connections', []):
            kind = kinds.get(c.get('kind'))
            if not kind:
                continue
            t = c.get('tile') or {}
            idx = c.get('stream_id')
            if idx is None:
                idx = (c.get('port') or {}).get('idx')
            out.add(sx.Shim(t.get('col'), t.get('row'), kind, idx))
        return out

    def search_rep_tile(self, flow_index):
        """searchIndex's representative tile: lowest DMA tile, else any tile."""
        p = self.paths[flow_index]
        cand = ([tuple(t) for t in p.get('dma_tiles') or []] +
                [tuple(t) for t in p.get('tiles') or []] +
                [(e[0][0], e[0][1]) for e in p.get('edges') or []])
        cand = [t for t in cand if len(t) == 2]
        return (min(cand), bool(p.get('dma_tiles'))) if cand else (None, False)

    def connections(self, flow_index):
        return _conn_records(self.paths[flow_index].get('routing_connections', []))

    def tile_set(self, flow_index):
        p = self.paths[flow_index]
        out = set()
        for e in p.get('edges', []):
            out.add((e[0][0], e[0][1]))
            out.add((e[1][0], e[1][1]))
        for group in ('tiles', 'dma_tiles', 'packet_tiles'):
            out.update((t[0], t[1]) for t in p.get(group, []))
        return out

    def flows_table(self, col, row):
        """The tile panel's Flows table, from the real _tileCommPaths()."""
        return set(self.flows_table_html.get('%d,%d' % (col, row), []))

    def panel_flows(self, col, row):
        """Flows whose focus selection leaves at least one row on this tile."""
        out = set()
        for fi, sub in self.focus_rows.items():
            if any((r.col, r.row) == (col, row)
                   for r in list(sub.cct) + list(sub.pkt) + list(sub.mst)):
                out.add(fi)
        return out

    def dma_channels(self, col, row):
        for t in self.tiles:
            if tuple(t['loc']) == (col, row):
                return t.get('dma_channels') or []
        return []

    def dma_endpoints(self):
        """{flow_index: {Endpoint}} from the DMA badges the UI renders."""
        out = {}
        for t in self.tiles:
            col, row = t['loc']
            for ch in t.get('dma_channels') or []:
                fi = ch.get('flow_index')
                if fi is not None:
                    out.setdefault(fi, set()).add(
                        Endpoint(col, row, ch.get('channel')))
        return out


def load_ui(workdir):
    import schedule_view

    comm_paths = schedule_view._load_comm_paths(workdir)
    view, tiles = None, []
    try:
        view = schedule_view.build_view(workdir)
        tiles = view['tiles']
    except Exception:
        pass
    rendered = sx.render_ui_html(workdir)
    ui = UiFlows(comm_paths, tiles, sx.parse_ui_html(rendered).focus_rows)
    ui.focus_html = rendered['focus']
    ui.flows_table_html = rendered.get('flows_table', {})
    ui.net_html = render_net_panels(view) if view else {}
    return ui


# ── the net panel, rendered by the real JS ───────────────────────────────────

_NET_PRELUDE = (
    ('statement', 'flowMembers'),
    ('expression', r'^\(DATA\.flow_summary\|\|\[\]\)\.forEach'),
    ('statement', 'DM_COLORS'),
    ('statement', 'dmFlowIds'),
)

_NET_FUNCS = ('dmColor', 'esc', '_isPktPort', '_sharedPktForwardMaster',
              '_defaultPktMask', '_resolvePktMask', '_fmtPktMaskHex',
              '_fmtPktMaskBadge', '_expandPktConnectRows', '_fmtDmaChanBadge',
              '_flowDmaBadgeStyle', '_flowDmaSpanHtml', 'renderFlowBalance',
              'buildNetBody')

_NET_DRIVER = r'''
const out = {};
for (const p of DATA.comm_paths) {
  try { out[p.flow_index] = buildNetBody(p); }
  catch (e) { out[p.flow_index] = 'ERROR: ' + e.message; }
}
process.stdout.write(JSON.stringify(out));
'''


def render_net_panels(view):
    """{flow_index: html} from the real buildNetBody()."""
    import schedule_view

    tpl = schedule_view.HTML_TEMPLATE
    parts = ['const DATA = JSON.parse('
             'require("fs").readFileSync(process.argv[2], "utf8"));']
    for kind, name in _NET_PRELUDE:
        parts.append(sx.extract_js_statement(tpl, name) if kind == 'statement'
                     else sx.extract_js_expression(tpl, name))
    parts += [sx.extract_js_function(tpl, n) for n in _NET_FUNCS]
    parts.append(_NET_DRIVER)
    return sx._run_node('\n'.join(parts), view)


_NET_ROW_RE = re.compile(r'<tr><td>\((\d+),(\d+)\)</td><td>(CCT|PKT)</td>'
                         r'<td>(.*?)</td></tr>')


def net_rows(html):
    """[(col, row, kind, has_dma_badge)] from one net panel."""
    return [(int(c), int(r), kind, 'rt-route' not in body)
            for c, r, kind, body in _NET_ROW_RE.findall(html)]


def _json_group_records(group):
    """Every stream-switch record a provenance group holds, split ignored."""
    cct, mst = set(), set()
    for c in group.get('connections', []):
        tile = c.get('tile') or {}
        col, row = tile.get('col'), tile.get('row')
        if c.get('kind') == 'circuit_connect':
            s, m = c.get('slave') or {}, c.get('master') or {}
            cct.add(sx.Cct(col, row, s.get('dir'), s.get('idx'),
                           m.get('dir'), m.get('idx')))
        elif c.get('kind') == 'packet_connect':
            fm = c.get('forward_master') or {}
            if fm.get('dir') not in (None, 'NONE'):
                mst.add(sx.Mst(col, row, fm['dir'], fm['idx'],
                               fm.get('arbiter', 0), fm.get('msel_en', 1)))
    return cct, mst


def _kind(record, name):
    return record.__class__.__name__ == name


def _split(records, name):
    return {r for r in records if _kind(r, name)}


# ── report ───────────────────────────────────────────────────────────────────

class Report(object):
    """Every tile-and-flow attribution the UI makes, checked against source."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.groups = parse_source_groups(workdir)
        self.flows = parse_flows(workdir)
        self.join, self.join_problems = (
            join_flows(self.groups, self.flows) if self.groups else ({}, []))
        self.ui = load_ui(workdir)
        self.key_collisions = group_key_collisions(workdir)
        self.unjoined = [f.index for f in self.flows if f.index not in self.join]
        self.flat_source = sx.parse_source(workdir)
        self.grouped = bool(self.groups)
        self.badges_checkable = bool(self.ui.tiles)

        # Always available: does every attributed connection exist in source,
        # and does every source connection reach some flow?
        self.flow_unbacked = self._check_unbacked()
        self.unattributed = self._check_unattributed()

        # Only with routing.cc's per-group blocks: which flow owns what.
        self.group_mismatch = self._check_groups(workdir) if self.grouped else []
        self.flow_conn = self._check_flow_connections() if self.grouped else {}
        self.cross_direction = self._check_cross_direction() if self.grouped else {}
        self.flow_tiles = self._check_flow_tiles() if self.grouped else {}
        self.tile_flows = self._check_tile_flows() if self.grouped else []

        self.hop_gaps = self._check_hop_coverage()
        self.shmem_orphans = self._check_shmem_flows()
        self.search_rep = self._check_search_rep()
        self.shim_unclaimed, self.flows_without_gmio = self._check_flow_shim()
        self.dma_badges = self._check_dma_badges() if self.badges_checkable else {}
        self.focus_badges = (self._check_focus_badges()
                             if self.badges_checkable else [])
        self.net_badges = self._check_net_badges() if self.ui.net_html else []

    # -- checks ---------------------------------------------------------------

    def _ui_flow_cct(self, flow_index):
        return self.ui.connections(flow_index)[0]

    def _check_unbacked(self):
        """Connections a flow claims that the source never programs."""
        out = {}
        for fi in self.ui.paths:
            extra = self._ui_flow_cct(fi) - self.flat_source.cct
            if extra:
                out[fi] = sorted(extra)
        return out

    def _check_unattributed(self):
        """Connections the source programs that no flow claims."""
        claimed = set()
        for fi in self.ui.paths:
            claimed |= self._ui_flow_cct(fi)
        return sorted(self.flat_source.cct - claimed)

    def _check_groups(self, workdir):
        """Does each provenance group hold exactly its source block's records?"""
        path = os.path.join(workdir, 'routingprovenancemap.json')
        if not os.path.exists(path) or not self.groups:
            return []
        with open(path) as f:
            json_groups = json.load(f).get('routing_groups', [])
        out = []
        if len(json_groups) != len(self.groups):
            out.append(('count', len(self.groups), len(json_groups)))
            return out
        for block, jg in zip(self.groups, json_groups):
            src = block.records[PUSH] | block.records[PULL]
            jcct, jmst = _json_group_records(jg)
            for name, got in (('Cct', jcct), ('Mst', jmst)):
                want = _split(src, name)
                if want != got:
                    out.append((jg.get('id'), sorted(want - got), sorted(got - want)))
        return out

    def _flow_records(self, flow_index):
        ui_cct, _, ui_mst = self.ui.connections(flow_index)
        block, section = self.join[flow_index]
        return (ui_cct, _split(block.records[section], 'Cct'),
                ui_mst, _split(block.records[section], 'Mst'))

    def _check_flow_connections(self):
        """Circuit connections the UI attributes to the wrong flow."""
        out = {}
        for fi in self.join:
            ui_cct, src_cct, _, _ = self._flow_records(fi)
            if ui_cct != src_cct:
                out[fi] = (sorted(src_cct - ui_cct), sorted(ui_cct - src_cct))
        return out

    def _check_cross_direction(self):
        """Config from the other direction shown under this flow.

        A row group programs distribution (push) and gather (pull) in one
        block.  The UI slices at the shim_aie_to_ext marker, but the packet
        gather config is emitted before that marker, so it lands in the push
        slice too.
        """
        out = {}
        for fi, (block, section) in self.join.items():
            other = PULL if section == PUSH else PUSH
            _, _, ui_mst, src_mst = self._flow_records(fi)
            leaked = ui_mst & _split(block.records[other], 'Mst')
            if leaked - src_mst:
                out[fi] = sorted(leaked - src_mst)
        return out

    def _source_tiles(self, flow_index):
        block, section = self.join[flow_index]
        shim = block.ingress if section == PUSH else block.egress
        return (block.tiles(section) |
                {(e.col, e.row) for e in block.dma[section]} |
                {(p.col, p.row) for p in shim})

    def _check_flow_tiles(self):
        out = {}
        for fi in self.join:
            ui_t, src_t = self.ui.tile_set(fi), self._source_tiles(fi)
            if ui_t != src_t:
                out[fi] = (sorted(src_t - ui_t), sorted(ui_t - src_t))
        return out

    def _check_hop_coverage(self):
        """Hops that ride the stream switch must land in the flow's tile set.

        Shared-memory hops legitimately do not: no switch is programmed for
        them, so they are filtered out by intersecting with the tiles the
        source actually configures.
        """
        stream_tiles = {(r.col, r.row) for r in self.flat_source.cct}
        out = {}
        for flow in self.flows:
            if flow.index not in self.ui.paths:
                continue
            gap = (flow.hop_tiles & stream_tiles) - self.ui.tile_set(flow.index)
            if gap:
                out[flow.index] = sorted(gap)
        return out

    def _check_shmem_flows(self):
        """A tile showing a shared-memory hop for a flow should list that flow.

        _tileCommPaths keys the Flows table on stream edges only, so a tile
        reached purely through shared memory renders a Shared memory row for a
        flow it does not admit to carrying.
        """
        out = []
        for fi, p in self.ui.paths.items():
            for h in p.get('hops', []):
                if h.get('type') != 'shmem':
                    continue
                for tile in ((h['from_col'], h['from_row']),
                             (h['to_col'], h['to_row'])):
                    if fi not in self.ui.flows_table(*tile):
                        out.append((tile, fi))
        return sorted(set(out))

    def _check_tile_flows(self):
        """Flows table vs panel rows vs source, per tile."""
        by_tile = {}
        for fi in self.join:
            for tile in self._source_tiles(fi):
                by_tile.setdefault(tile, set()).add(fi)
        for fi in self.join:
            for tile in self.ui.tile_set(fi):
                by_tile.setdefault(tile, set())
        out = []
        for tile, src in sorted(by_tile.items()):
            table = self.ui.flows_table(*tile)
            panel = self.ui.panel_flows(*tile)
            if not (table == panel == src):
                out.append((tile, sorted(table), sorted(panel), sorted(src)))
        return out

    def _check_focus_badges(self):
        """A row shown under flow N must not wear flow M's DMA badge.

        renderTileRoutingSection filters rows by the focused flow but builds
        the badge from the row's unfiltered flow_indices, so a shared row keeps
        both flows' channel badges after focusing one of them.
        """
        owners = self.ui.channel_owners()
        badge = re.compile(r'class="rt-(?:s2mm|mm2s)"[^>]*>([a-z0-9]+)<')
        row = re.compile(r'<div class="rt-row (?:cct|pkt)">(.*?)'
                         r'(?=<div class="rt-|</div>\s*$)')
        out = []
        for fi, per_tile in self.ui.focus_html.items():
            for key, html in per_tile.items():
                if not html:
                    continue
                tile = tuple(int(x) for x in key.split(','))
                for chunk in row.findall(html):
                    for label in badge.findall(chunk):
                        own = owners.get((tile, label), set())
                        if int(fi) not in own:
                            out.append((int(fi), tile, label, tuple(sorted(own))))
        return sorted(set(out))

    def _check_net_badges(self):
        """The net panel must badge a connection the tile panel badges.

        buildNetBody passes c.flow_index from a routing_connections record;
        those records carry no flow_index, so every row falls back to the grey
        'no local DMA for this flow' badge.
        """
        out = []
        for fi, html in self.ui.net_html.items():
            fi = int(fi)
            if str(html).startswith('ERROR'):
                out.append((fi, None, str(html)[:80]))
                continue
            with_dma = self.ui.flow_tiles_with_dma(fi)
            for col, row, _, badged in net_rows(html):
                if (col, row) in with_dma and not badged:
                    out.append((fi, (col, row), 'shown as route-only'))
        return out

    def _check_flow_shim(self):
        """Every shim port enable must be attributed to some flow."""
        claimed = set()
        for fi in self.ui.paths:
            claimed |= self.ui.gmio_conns(fi)
        missing = self.flat_source.shim - claimed
        flowless = [fi for fi in sorted(self.ui.paths)
                    if not self.ui.gmio_conns(fi)]
        return sorted(missing), flowless

    def _check_search_rep(self):
        """A search hit must point at a tile the flow actually touches."""
        out = []
        for fi in self.ui.paths:
            tile, real = self.ui.search_rep_tile(fi)
            if tile not in self.ui.tile_set(fi):
                out.append((fi, tile, 'dma_tiles[0]' if real else 'fallback'))
        return out

    def _check_dma_badges(self):
        """Does each DMA badge name the flow that channel serves?

        Some dmaphop stages carry no dma_port; those endpoints are matched on
        the tile alone rather than reported as a channel-number mismatch.
        """
        badges = self.ui.dma_endpoints()
        out = {}
        for flow in self.flows:
            want = flow.producers | flow.consumers
            got = badges.get(flow.index, set())
            loose = {(e.col, e.row) for e in want if e.port is None}
            want_exact = {e for e in want if e.port is not None}
            got_exact = {e for e in got if (e.col, e.row) not in loose}
            got_tiles = {(e.col, e.row) for e in got}
            missing = (sorted(want_exact - got_exact) +
                       sorted(t for t in loose if t not in got_tiles))
            extra = sorted(got_exact - want_exact)
            if missing or extra:
                out[flow.index] = (missing, extra)
        return out

    # -- verdict --------------------------------------------------------------

    @property
    def deviations(self):
        out = []
        out += [('group %s' % g[0], g[1], g[2]) for g in self.group_mismatch]
        out += [('flow f%d connections' % f, m, e)
                for f, (m, e) in sorted(self.flow_conn.items())]
        out += [('flow f%d unbacked by source' % f, e, [])
                for f, e in sorted(self.flow_unbacked.items())]
        out += [('flow f%d tiles' % f, m, e)
                for f, (m, e) in sorted(self.flow_tiles.items())]
        out += [('flow f%d shows the other direction\'s packet config' % f, r, [])
                for f, r in sorted(self.cross_direction.items())]
        out += [('flow f%d dma badges' % f, m, e)
                for f, (m, e) in sorted(self.dma_badges.items())]
        out += [('tile %s flows' % (t,), p, s)
                for t, tab, p, s in self.tile_flows]
        out += [('flow f%d hop coverage' % f, g, [])
                for f, g in sorted(self.hop_gaps.items())]
        out += [('tile %s omits flow f%d from its Flows table' % (t, f), [], [])
                for t, f in self.shmem_orphans]
        out += [('focus f%d shows flow %s\'s %s badge on %s' % (f, o, b, t), [], [])
                for f, t, b, o in self.focus_badges]
        out += [('net panel f%d tile %s %s' % row, [], [])
                for row in self.net_badges]
        out += [('search hit for f%d points at %s (%s)' % r, [], [])
                for r in self.search_rep]
        if self.shim_unclaimed:
            out.append(('shim port enables no flow claims', self.shim_unclaimed, []))
        if self.unattributed:
            out.append(('connections no flow claims', self.unattributed, []))
        return out

    def ok(self):
        return not self.deviations and not self.join_problems


# ── CLI ──────────────────────────────────────────────────────────────────────

def _dump(title, rows, limit=None):
    if not rows:
        return
    print('  %s (%d)' % (title, len(rows)))
    shown = rows if limit is None else rows[:limit]
    for row in shown:
        print('    %s' % (row,))
    if len(shown) < len(rows):
        print('    ... %d more' % (len(rows) - len(shown)))


def report_text(rep, verbose=False):
    print('workdir : %s' % rep.workdir)
    print('source  : %d routing blocks, %d circuit connections'
          % (len(rep.groups), len(rep.flat_source.cct)))
    print('flows   : %d, %d joined to a source block'
          % (len(rep.flows), len(rep.join)))
    if not rep.grouped:
        print('note    : no per-group blocks in source — only whole-design '
              'attribution is checkable, not which flow owns what')
    if not rep.badges_checkable:
        print('note    : no dfscheduleprovenancemap.json/host.cc — DMA badge '
              'attribution not checked')
    for problem in rep.join_problems:
        print('warning : %s' % problem)
    if rep.unjoined:
        print('warning : flows with no routing group, UI falls back to raw hops: %s'
              % rep.unjoined)
    for ids in rep.key_collisions:
        print('warning : routing groups %s share a DMA-tile key; the UI keeps '
              'the last one for every flow that matches' % ids)

    _dump('provenance groups that do not match their source block',
          rep.group_mismatch)
    _dump('flows whose circuit connections are wrong',
          [(f, m, e) for f, (m, e) in sorted(rep.flow_conn.items())])
    _dump('flows claiming connections the source never programs',
          sorted(rep.flow_unbacked.items()))
    _dump('connections the source programs that no flow claims',
          rep.unattributed)
    _dump('flows whose tile set is wrong',
          [(f, m, e) for f, (m, e) in sorted(rep.flow_tiles.items())])
    _dump('flows whose DMA badges are wrong',
          [(f, m, e) for f, (m, e) in sorted(rep.dma_badges.items())])
    _dump('tiles where Flows table, panel and source disagree', rep.tile_flows)
    _dump('flows missing a tile their hop chain traverses',
          sorted(rep.hop_gaps.items()))
    _dump('tiles with a shared-memory hop for a flow they do not list',
          rep.shmem_orphans)
    limit = None if verbose else 6
    _dump("rows wearing another flow's DMA badge under focus",
          rep.focus_badges, limit)
    _dump('net panel rows shown as route-only on a tile that has a DMA '
          'channel for that flow', rep.net_badges, limit)
    _dump('search hits pointing at a tile the flow does not touch', rep.search_rep)
    _dump('shim port enables no flow claims', rep.shim_unclaimed)
    if rep.flows_without_gmio:
        print('  flows whose panel never says where data enters or leaves the '
              'array: %s' % rep.flows_without_gmio)

    if rep.cross_direction:
        total = sum(len(v) for v in rep.cross_direction.values())
        print('  packet config shown under the opposite direction (%d ports)' % total)
        for fi, recs in sorted(rep.cross_direction.items()):
            print('    f%d (%s) also shows the gather config of its paired flow'
                  % (fi, self_direction(rep, fi)))
            if verbose:
                for r in recs:
                    print('      %s' % (r,))
    print('result  : %s' % ('MATCH' if rep.ok()
                            else '%d DEVIATIONS' % len(rep.deviations)))


def self_direction(rep, flow_index):
    return next(f.direction for f in rep.flows if f.index == flow_index)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('workdir')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)
    rep = Report(args.workdir)
    report_text(rep, args.verbose)
    return 0 if rep.ok() else 1


if __name__ == '__main__':
    sys.exit(main())
