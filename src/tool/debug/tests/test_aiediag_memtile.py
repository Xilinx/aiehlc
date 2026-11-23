###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import aiediag as d


# ── tile_type_for_row ────────────────────────────────────────────────────────

def test_row0_is_shim():
    assert d.tile_type_for_row(0, "5") == "shim"

def test_gen5_memtile_rows_1_and_2():
    # gen5/aie2ps cores start at row 3 -> rows 1,2 are memtiles.
    assert d.tile_type_for_row(1, "5") == "memtile"
    assert d.tile_type_for_row(2, "5") == "memtile"

def test_gen5_core_rows():
    assert d.tile_type_for_row(3, "5") == "core"
    assert d.tile_type_for_row(6, "5") == "core"

def test_aieml_single_memtile_row():
    # aieml cores start at row 2 -> only row 1 is a memtile.
    assert d.tile_type_for_row(1, "2") == "memtile"
    assert d.tile_type_for_row(2, "2") == "core"

def test_unknown_version_defaults_to_gen5_band():
    assert d.tile_type_for_row(2, "unknown") == "memtile"


# ── compute_reg_offset ───────────────────────────────────────────────────────

def test_memtile_s2mm_status_offset_ch0():
    # XAIE2PSGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0 = 0xA0660.
    assert d.compute_reg_offset("memtile", "s2mm", 0, "5") == 0xA0660

def test_memtile_s2mm_status_offset_ch1():
    # Per-channel stride 0x4 (STATUS_1 = 0xA0664).
    assert d.compute_reg_offset("memtile", "s2mm", 1, "5") == 0xA0664

def test_memtile_mm2s_status_offset_ch0():
    assert d.compute_reg_offset("memtile", "mm2s", 0, "5") == 0xA0680

def test_core_offset_unchanged():
    # Regression: the core-tile memory-module offset (the one the old code used
    # for a memtile) must stay put.
    assert d.compute_reg_offset("core", "s2mm", 0, "5") == 0x1DF00
    assert d.compute_reg_offset("core", "s2mm", 1, "5") == 0x1DF04


# ── BD offsets ───────────────────────────────────────────────────────────────

def test_memtile_bd_type_key_and_length():
    assert d._bd_type_key("memtile", "5") == "memtile"
    # BD0 base 0xA0000, stride 0x20; BD3 word0 length register.
    assert d.bd_length_offset("memtile", "5", 0) == 0xA0000
    assert d.bd_length_offset("memtile", "5", 3) == 0xA0060
    assert d.BD_LEN_MASK["memtile"] == 0x1FFFF


# ── MemTile event decode ─────────────────────────────────────────────────────

def test_memtile_evt_status_regs():
    # 6 event status registers (events 0-191) vs core memory module's 4.
    assert d.MEMTILE_EVT_STATUS_REGS == (
        0x94200, 0x94204, 0x94208, 0x9420C, 0x94210, 0x94214)

def test_memtile_dma_event_ids():
    # From driver xaie_events_aie2ps.h XAIE2PS_EVENTS_MEM_TILE_DMA_*.
    assert d.MEMTILE_DMA_EVENT_IDS[("s2mm", 0)] == {
        "START_TASK": 21, "FINISHED_BD": 25, "FINISHED_TASK": 29, "ERROR": 133}
    assert d.MEMTILE_DMA_EVENT_IDS[("mm2s", 1)] == {
        "START_TASK": 24, "FINISHED_BD": 28, "FINISHED_TASK": 32, "ERROR": 134}
    # ERROR is direction-wide: both S2MM slots share 133, both MM2S share 134.
    assert d.MEMTILE_DMA_EVENT_IDS[("s2mm", 1)]["ERROR"] == 133
    assert d.MEMTILE_DMA_EVENT_IDS[("mm2s", 0)]["ERROR"] == 134

def test_memtile_dma_event_sel_reg():
    assert d.MEMTILE_DMA_EVENT_SEL_REG == 0xA06A0
    assert d.MEMTILE_DMA_EVENT_SEL_LSB[("s2mm", 0)] == 0
    assert d.MEMTILE_DMA_EVENT_SEL_LSB[("s2mm", 1)] == 8
    assert d.MEMTILE_DMA_EVENT_SEL_LSB[("mm2s", 0)] == 16
    assert d.MEMTILE_DMA_EVENT_SEL_LSB[("mm2s", 1)] == 24

def test_memtile_sel_for_channel():
    # S2MM SEL0 -> ch3, SEL1 -> ch5.
    sel_reg = 3 | (5 << 8)
    assert d.memtile_dma_sel_for_channel(sel_reg, "s2mm", 3) == 0
    assert d.memtile_dma_sel_for_channel(sel_reg, "s2mm", 5) == 1
    assert d.memtile_dma_sel_for_channel(sel_reg, "s2mm", 2) is None
    # MM2S SEL0 -> ch1.
    mm2s_reg = 1 << 16
    assert d.memtile_dma_sel_for_channel(mm2s_reg, "mm2s", 1) == 0
    # None sel_reg -> no mapping.
    assert d.memtile_dma_sel_for_channel(None, "s2mm", 0) is None

def test_memtile_evt_active_high_events():
    # ERROR events live at 133/134 which land in register index 4 (0x94210).
    regs = [0, 0, 0, 0, 1 << (133 % 32), 0]
    assert d._evt_active(133, regs) is True
    assert d._evt_active(134, regs) is False

def test_format_memtile_dma_events_started_finished():
    # S2MM SEL0: START_TASK=21, FINISHED_BD=25, FINISHED_TASK=29 all in reg0.
    regs = [(1 << 21) | (1 << 25) | (1 << 29), 0, 0, 0, 0, 0]
    sel_reg = 3  # S2MM SEL0 -> ch3
    out = d.format_memtile_dma_events(4, 2, regs, sel_reg)
    assert "S2MM SEL0 (-> ch3)" in out
    assert "started and finished" in out

def test_format_memtile_dma_events_error():
    # S2MM direction-wide ERROR (133).
    regs = [0, 0, 0, 0, 1 << (133 % 32), 0]
    out = d.format_memtile_dma_events(4, 2, regs, sel_reg=0)
    assert "DMA ERROR event active" in out


# ── Core-module event list ───────────────────────────────────────────────────

def test_core_evt_status_regs():
    assert d.CORE_EVT_STATUS_REGS == (0x34200, 0x34204, 0x34208, 0x3420C)

def test_core_evt_names_length_and_lookup():
    # 128 entries; id 54 reserved (None); matches driver s_core_evt_names[].
    assert len(d.CORE_EVT_NAMES) == 128
    assert d.core_event_name(37) == "INSTR_VECTOR"
    assert d.core_event_name(28) == "ACTIVE"
    assert d.core_event_name(24) == "STREAM_STALL"
    assert d.core_event_name(46) == "GROUP_ERRORS_0"
    assert d.core_event_name(127) == "USER_EVENT_3"
    assert d.core_event_name(54) is None
    assert d.core_event_name(999) is None

def _set_events(*ids):
    regs4 = [0, 0, 0, 0]
    for e in ids:
        regs4[e // 32] |= 1 << (e % 32)
    return regs4

def test_format_core_module_events_lists_active_by_name():
    out = d.format_core_module_events(3, 3, _set_events(24, 28, 37))
    assert "[ 24] STREAM_STALL" in out
    assert "[ 28] ACTIVE" in out
    assert "[ 37] INSTR_VECTOR" in out

def test_format_core_module_events_flags_errors():
    # GROUP_ERRORS_0 (46) and PM_ECC_ERROR_2BIT (64) are errors.
    out = d.format_core_module_events(3, 3, _set_events(46, 64))
    assert "GROUP_ERRORS_0" in out and "error" in out
    assert "PM_ECC_ERROR_2BIT" in out

def test_format_core_module_events_none_active():
    out = d.format_core_module_events(3, 3, [0, 0, 0, 0])
    assert "no active core-module events" in out

def test_format_core_module_events_high_event_id():
    # USER_EVENT_3 = 127 lives in register index 3.
    out = d.format_core_module_events(3, 3, _set_events(127))
    assert "[127] USER_EVENT_3" in out
