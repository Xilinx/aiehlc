###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import aiediag as d
import aiegdb


# ── decode round-trip ────────────────────────────────────────────────────────

def test_decode_master_circuit():
    m = d.decode_strm_master(0x80000005)
    assert m["enable"] is True
    assert m["packet"] is False
    assert m["drop_header"] is False
    assert m["config"] == 5

def test_decode_master_packet_drophdr():
    # bit31 enable, bit30 packet, bit7 drop_header, config = 0x05.
    m = d.decode_strm_master(0xC0000085)
    assert m["enable"] and m["packet"] and m["drop_header"]
    assert m["config"] == 5

def test_decode_master_disabled():
    m = d.decode_strm_master(0x00000003)
    assert m["enable"] is False
    assert m["config"] == 3

def test_decode_slave():
    assert d.decode_strm_slave(0x80000000)["enable"] is True
    assert d.decode_strm_slave(0x80000000)["packet"] is False
    assert d.decode_strm_slave(0xC0000000)["packet"] is True
    assert d.decode_strm_slave(0x00000000)["enable"] is False

def test_decode_slot():
    # ID[28:24]=3, MASK[20:16]=0x1F, ENABLE bit8, MSEL[5:4]=1, ARBITER[2:0]=2.
    raw = (3 << 24) | (0x1F << 16) | (1 << 8) | (1 << 4) | 2
    s = d.decode_strm_slot(raw)
    assert s["id"] == 3
    assert s["mask"] == 0x1F
    assert s["enable"] is True
    assert s["msel"] == 1
    assert s["arbiter"] == 2


# ── port-map offset math ─────────────────────────────────────────────────────

def test_core_offsets():
    assert d.strm_master_off("core", 0) == 0x3F000
    assert d.strm_master_off("core", 1) == 0x3F004
    assert d.strm_slave_off("core", 0) == 0x3F100
    assert d.strm_slave_off("core", 2) == 0x3F108
    # 4 slots/slave, stride 0x4 -> 0x10 per slave.
    assert d.strm_slot_off("core", 0, 0) == 0x3F200
    assert d.strm_slot_off("core", 0, 1) == 0x3F204
    assert d.strm_slot_off("core", 1, 0) == 0x3F210

def test_shim_shares_core_bases():
    assert d.strm_master_off("shim", 0) == 0x3F000
    assert d.strm_slave_off("shim", 0) == 0x3F100

def test_memtile_offsets():
    assert d.strm_master_off("memtile", 0) == 0xB0000
    assert d.strm_slave_off("memtile", 0) == 0xB0100
    assert d.strm_slot_off("memtile", 0, 0) == 0xB0200
    assert d.strm_slot_off("memtile", 1, 0) == 0xB0210


# ── port maps ────────────────────────────────────────────────────────────────

def test_core_slave_south0_index():
    # CORE, DMA0-1, CTRL, FIFO, then SOUTH0 -> physical index 5.
    assert d.STRM_SW_SLAVE_PORTS["core"].index(("SOUTH", 0)) == 5
    assert d.STRM_SW_SLAVE_PORTS["core"].index(("DMA", 0)) == 1

def test_core_master_north0_index():
    # CORE, DMA0-1, CTRL, FIFO, SOUTH0-3, WEST0-3, then NORTH0 -> index 13.
    assert d.STRM_SW_MASTER_PORTS["core"].index(("NORTH", 0)) == 13
    assert d.STRM_SW_MASTER_PORTS["core"].index(("CORE", 0)) == 0

def test_memtile_has_no_east_west():
    for ptype, _ in d.STRM_SW_MASTER_PORTS["memtile"]:
        assert ptype not in ("EAST", "WEST")
    for ptype, _ in d.STRM_SW_SLAVE_PORTS["memtile"]:
        assert ptype not in ("EAST", "WEST")


# ── inter-tile adjacency ─────────────────────────────────────────────────────

def test_adjacency_north_south():
    assert d.master_neighbor(0, 3, "NORTH", 0) == ("tile", 0, 4, "SOUTH", 0)
    assert d.master_neighbor(0, 4, "SOUTH", 0) == ("tile", 0, 3, "NORTH", 0)

def test_adjacency_east_west():
    assert d.master_neighbor(0, 3, "EAST", 1) == ("tile", 1, 3, "WEST", 1)
    assert d.master_neighbor(1, 3, "WEST", 1) == ("tile", 0, 3, "EAST", 1)

def test_shim_south_is_terminal():
    assert d.master_neighbor(0, 0, "SOUTH", 0) == ("terminal", "PL/NoC/DDR:SOUTH0")

def test_nondirectional_is_terminal():
    assert d.master_neighbor(0, 3, "CORE", 0) == ("terminal", "CORE0")
    assert d.master_neighbor(0, 3, "DMA", 1) == ("terminal", "DMA1")

def test_offgrid_hop_is_none():
    assert d.master_neighbor(0, 3, "WEST", 0) is None   # col -1


# ── read_switch + format_switch with an injected fake reader ─────────────────

def _core_reader(master_cfgs=None, slave_ens=None, slot_map=None):
    """Build a fake reader(off) for a core tile. master_cfgs: {phys_idx: cfg},
    slave_ens: {phys_idx: packet_bool}, slot_map: {(slave_idx,slot): raw}."""
    master_cfgs = master_cfgs or {}
    slave_ens = slave_ens or {}
    slot_map = slot_map or {}
    regs = {}
    for idx, cfg in master_cfgs.items():
        regs[d.strm_master_off("core", idx)] = 0x80000000 | cfg
    for idx, pkt in slave_ens.items():
        regs[d.strm_slave_off("core", idx)] = 0x80000000 | (0x40000000 if pkt else 0)
    for (si, sl), raw in slot_map.items():
        regs[d.strm_slot_off("core", si, sl)] = raw
    return lambda off: regs.get(off, 0)

def test_read_switch_reports_enabled_master():
    # NORTH0 master (idx13) sourced from SOUTH0 slave (idx5); SOUTH0 slave on.
    reader = _core_reader(master_cfgs={13: 5}, slave_ens={5: False})
    sw = d.read_switch("core", reader)
    en = [m for m in sw["masters"] if m["enable"]]
    assert len(en) == 1
    assert en[0]["port"] == "NORTH0"
    assert en[0]["config"] == 5
    assert [s["port"] for s in sw["slaves"] if s["enable"]] == ["SOUTH0"]

def test_format_switch_shows_source_and_mode():
    reader = _core_reader(master_cfgs={13: 5}, slave_ens={5: False})
    sw = d.read_switch("core", reader)
    out = d.format_switch(0, 3, "core", sw)
    assert "SOUTH0 -> NORTH0" in out
    assert "[circuit]" in out
    assert "enabled slaves: SOUTH0" in out

def test_format_switch_packet_slot():
    slot_raw = (3 << 24) | (0x1F << 16) | (1 << 8) | (1 << 4) | 2
    reader = _core_reader(master_cfgs={13: 5}, slave_ens={5: True},
                          slot_map={(5, 0): slot_raw})
    sw = d.read_switch("core", reader)
    out = d.format_switch(0, 3, "core", sw)
    assert "slot SOUTH0 #0: id=0x3" in out
    assert "msel=1" in out and "arbiter=2" in out

def test_format_switch_no_masters():
    out = d.format_switch(0, 3, "core", d.read_switch("core", lambda off: 0))
    assert "no enabled master ports" in out


# ── _scan_switch flow-trace on a mocked mesh ─────────────────────────────────

def _mesh_reg_read(regmap):
    """Return a bound-method-shaped fake: fn(self, phys_col, row, off)."""
    def _fn(self, phys_col, row, off):
        return regmap.get(row, {}).get(off, 0)
    return _fn

def _core_tile_regs(masters=None, slaves=None):
    masters = masters or {}
    slaves = slaves or {}
    regs = {}
    for idx, cfg in masters.items():
        regs[d.strm_master_off("core", idx)] = 0x80000000 | cfg
    for idx in slaves:
        regs[d.strm_slave_off("core", idx)] = 0x80000000
    return regs

def _build_vertical_mesh():
    """3-tile vertical flow through origin (0,4):
       DMA0@(0,3) -> NORTH0 -> NORTH0@(0,4) -> CORE0@(0,5).
    core slave SOUTH0 idx=5, DMA0 idx=1; core master NORTH0 idx=13, CORE0 idx=0.
    """
    S0, DMA0 = 5, 1            # slave phys indices
    NORTH0, CORE0 = 13, 0      # master phys indices
    return {
        3: _core_tile_regs(masters={NORTH0: DMA0}, slaves={DMA0}),
        4: _core_tile_regs(masters={NORTH0: S0}, slaves={S0}),
        5: _core_tile_regs(masters={CORE0: S0}, slaves={S0}),
    }

def _run_scan(regmap, col, row, monkeypatch_reg):
    g = aiegdb.AieGdb(startcol=0, aie_version="5", dry_run=True)
    g._reg_read = monkeypatch_reg.__get__(g, aiegdb.AieGdb)
    g.tile = {"col": col, "row": row, "type": "core"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g._scan_switch()
    return buf.getvalue()

def test_scan_switch_assembles_3tile_flow():
    regmap = _build_vertical_mesh()
    out = _run_scan(regmap, 0, 4, _mesh_reg_read(regmap))
    assert "DMA0@(0,3) -> NORTH0@(0,3) -> NORTH0@(0,4) -> CORE0@(0,5)" in out
    # BFS reached all three tiles.
    assert "(3 tiles)" in out

def test_scan_switch_visits_upstream_and_downstream():
    regmap = _build_vertical_mesh()
    out = _run_scan(regmap, 0, 4, _mesh_reg_read(regmap))
    assert "stream switch tile(0,3)" in out
    assert "stream switch tile(0,4)" in out
    assert "stream switch tile(0,5)" in out

def test_scan_switch_cycle_guard_terminates():
    # Two core tiles whose NORTH masters each claim to be fed by the other's
    # SOUTH slave -> a misconfigured loop. BFS visited + recursion guard must
    # terminate rather than hang.
    S0, NORTH0 = 5, 13
    regmap = {
        3: _core_tile_regs(masters={NORTH0: S0}, slaves={S0}),
        4: _core_tile_regs(masters={NORTH0: S0}, slaves={S0}),
        5: _core_tile_regs(masters={NORTH0: S0}, slaves={S0}),
    }
    out = _run_scan(regmap, 0, 4, _mesh_reg_read(regmap))
    # Completes and produces some flow text (the assertion is that it returns).
    assert "flows through tile(0, 4)" in out

def test_scan_switch_no_connections():
    # All-zero tile: no enabled masters -> single tile, no flows.
    regmap = {4: {}}
    out = _run_scan(regmap, 0, 4, _mesh_reg_read(regmap))
    assert "(1 tiles)" in out
    assert "no enabled connections through this tile" in out
