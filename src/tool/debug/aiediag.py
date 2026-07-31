#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiediag - Flow-aware DMA diagnostic tool for AIE.

Reads DMA status registers from live hardware via aiedbg, cross-references
with provenance map JSONs to identify connected tiles and routing paths,
and prints an automated diagnosis.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# ─── ANSI color helpers ───────────────────────────────────────────────────────

def _supports_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_USE_COLOR = _supports_color()

def _c(code, text):
    if _USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text

def red(t):    return _c("31", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def lightyellow(t): return _c("93", t)  # bright/light yellow
def cyan(t):   return _c("36", t)
def bold(t):   return _c("1", t)

# ─── Constants ────────────────────────────────────────────────────────────────

DMA_STATUS_OFFSETS = {
    "core":    {"s2mm": 0x1DF00, "mm2s": 0x1DF10},
    "shim_5":  {"s2mm": 0x1D220, "mm2s": 0x1D228},
    "shim_2ps": {"s2mm": 0x9320, "mm2s": 0x9328},
}
CH_STRIDE = 0x4

# Core module Program Counter register (AIE2PS / AIEML core tiles, row != 0).
# Source: XAIE2PSGBL_CORE_MODULE_CORE_PC = 0x00030F00, value mask 0x000FFFFF.
CORE_PC_OFFSET = 0x30F00
CORE_PC_MASK   = 0xFFFFF

# Core module Core_Status / Core_Control registers (core tiles, row != 0).
# The register OFFSET differs by AIE generation; the bit layout is identical.
#   AIEML  (aie=5):   Core_Control=0x32000  Core_Status=0x32004
#   AIE2PS (aie=2ps): Core_Control=0x38000  Core_Status=0x38004
# Source: XAIEMLGBL_/XAIE2PSGBL_CORE_MODULE_CORE_{CONTROL,STATUS}.
CORE_STATUS_OFFSET  = {"5": 0x32004, "2ps": 0x38004}
CORE_CONTROL_OFFSET = {"5": 0x32000, "2ps": 0x38000}

# Kernel data-memory log (klog) region. Mirrors src/mlir/runtime/kernel_log.h:
# the last 2 KB of core-tile data memory hold 512 int32 slots at DM byte offset
# 0xF800.  slot[0] = write_index (count of int32s written, NOT entries);
# entries are [tag_packed, value] pairs at slot[1..].  klog_init() writes a first
# entry with tag "KLOG" and value KLOG_MAGIC.  Core tiles only (data memory).
KLOG_DM_OFFSET   = 0xF800   # DM byte offset (== XAie_DataMemBlockRead offset)
KLOG_REGION_WORDS = 512     # total int32 slots in the 2 KB region
KLOG_MAX_SLOTS   = 511      # slot 0 = write_index, slots 1..511 = data
KLOG_MAGIC       = 0x4B4C4F47  # "KLOG" ASCII big-endian

# Core_Status bit fields: name -> bit position (LSB). All width 1.
# Order chosen so the decoded print reads high-level state first, then stalls.
# Source: XAIE*GBL_CORE_MODULE_CORE_STATUS_*_LSB.
CORE_STATUS_BITS = [
    ("ENABLE",             0),
    ("RESET",              1),
    ("MEMORY_STALL_S",     2),
    ("MEMORY_STALL_W",     3),
    ("MEMORY_STALL_N",     4),
    ("MEMORY_STALL_E",     5),
    ("LOCK_STALL_S",       6),
    ("LOCK_STALL_W",       7),
    ("LOCK_STALL_N",       8),
    ("LOCK_STALL_E",       9),
    ("STREAM_STALL_SS0",  10),
    ("STREAM_STALL_MS0",  12),
    ("CASCADE_STALL_SCD", 14),
    ("CASCADE_STALL_MCD", 15),
    ("DEBUG_HALT",        16),
    ("ECC_ERROR_STALL",   17),
    ("ECC_SCRUBBING_STALL", 18),
    ("ERROR_HALT",        19),
    ("CORE_DONE",         20),
    ("CORE_PROCESSOR_BUS_STALL", 21),
]

# DMA BD register layout (word 0 holds Buffer_Length for both tile types).
# tile_type -> (bd0_base, bd_stride)
BD_BASE_STRIDE = {
    "core":     (0x1D000, 0x20),
    "shim_5":   (0x9000,  0x30),
    "shim_2ps": (0x9000,  0x30),
}
# Buffer_Length field mask in word 0. Core: bits[13:0]; shim: bits[31:0].
# Value is in 32-bit words; bytes = words * 4.
BD_LEN_MASK = {"core": 0x3FFF, "shim_5": 0xFFFFFFFF, "shim_2ps": 0xFFFFFFFF}

# Shim PL module event status registers
SHIM_EVT_STATUS_REG0 = 0x00034200  # event IDs 0-31
SHIM_EVT_STATUS_REG1 = 0x00034204  # event IDs 32-63

# Event IDs for DMA channels on shim tiles (NoC 0), for the PL-module event
# status registers on AIE2PS/AIEML.
#
# Authoritative source: the XAie driver PL-module event enum, mirrored in
# src/mlir/runtime/aie_runtime_debug.c s_pl_evt_names[] (uses
# AieRt_EventName(XAIE_PL_MOD, id)). Layout per aie2psshimevent.md Table 7-12:
#   START_TASK    : S2MM0=14 S2MM1=15 MM2S0=16 MM2S1=17
#   FINISHED_BD   : S2MM0=18 S2MM1=19 MM2S0=20 MM2S1=21
#   FINISHED_TASK : S2MM0=22 S2MM1=23 MM2S0=24 MM2S1=25
#   STALLED_LOCK  : S2MM0=26 S2MM1=27 MM2S0=28 MM2S1=29
#   STREAM (S2MM=STARVATION / MM2S=BACKPRESSURE): 30 31 32 33
#   MEMORY (S2MM=BACKPRESSURE / MM2S=STARVATION): 34 35 36 37
# NOTE: an earlier version of this table omitted FINISHED_BD, shifting every
# event from FINISHED_TASK onward down by 4 (e.g. STREAM_BACKPRESSURE read bit
# 29 = STALLED_LOCK), which made real stream-backpressure stalls read as "not
# set". Keep this in sync with aie_runtime_debug.c.
# Keyed by (direction, channel) -> dict of event_name -> event_id
SHIM_DMA_EVENT_IDS = {
    ("s2mm", 0): {"START_TASK": 14, "FINISHED_BD": 18, "FINISHED_TASK": 22,
                   "STALLED_LOCK": 26, "STREAM_STARVATION": 30,
                   "MEMORY_BACKPRESSURE": 34},
    ("s2mm", 1): {"START_TASK": 15, "FINISHED_BD": 19, "FINISHED_TASK": 23,
                   "STALLED_LOCK": 27, "STREAM_STARVATION": 31,
                   "MEMORY_BACKPRESSURE": 35},
    ("mm2s", 0): {"START_TASK": 16, "FINISHED_BD": 20, "FINISHED_TASK": 24,
                   "STALLED_LOCK": 28, "STREAM_BACKPRESSURE": 32,
                   "MEMORY_STARVATION": 36},
    ("mm2s", 1): {"START_TASK": 17, "FINISHED_BD": 21, "FINISHED_TASK": 25,
                   "STALLED_LOCK": 29, "STREAM_BACKPRESSURE": 33,
                   "MEMORY_STARVATION": 37},
}

# Core-tile (row != 0) module event status registers (AIE2PS)
MEM_EVT_STATUS_REGS  = (0x14200, 0x14204, 0x14208, 0x1420C)  # events 0-127
CORE_EVT_STATUS_REGS = (0x34200, 0x34204, 0x34208, 0x3420C)  # events 0-127

# Memory-module DMA events per (direction, channel) (from core_mem_module_events.json)
# Layout (authoritative: driver s_mem_evt_names[] in aie_runtime_debug.c):
#   START_TASK   : S2MM0=19 S2MM1=20 MM2S0=21 MM2S1=22
#   FINISHED_BD  : S2MM0=23 S2MM1=24 MM2S0=25 MM2S1=26
#   FINISHED_TASK: S2MM0=27 S2MM1=28 MM2S0=29 MM2S1=30
# NOTE: FINISHED_BD (one BD completed) is distinct from FINISHED_TASK (whole
# task completed). A channel can fire FINISHED_BD repeatedly while its overall
# task is still running; only FINISHED_TASK means the transfer is done.
CORE_MEM_DMA_EVENT_IDS = {
    ("s2mm", 0): {"START_TASK": 19, "FINISHED_BD": 23, "FINISHED_TASK": 27, "ERROR": 107},
    ("s2mm", 1): {"START_TASK": 20, "FINISHED_BD": 24, "FINISHED_TASK": 28, "ERROR": 108},
    ("mm2s", 0): {"START_TASK": 21, "FINISHED_BD": 25, "FINISHED_TASK": 29, "ERROR": 109},
    ("mm2s", 1): {"START_TASK": 22, "FINISHED_BD": 26, "FINISHED_TASK": 30, "ERROR": 110},
}
CORE_DMA_CHANNELS = [("s2mm", 0), ("s2mm", 1), ("mm2s", 0), ("mm2s", 1)]

# Core-module error events (from core_events.json)
CORE_ERROR_EVENT_IDS = {
    "GROUP_ERRORS_0": 46, "GROUP_ERRORS_1": 47,
    "STREAM_PKT_PARITY_ERROR": 56, "CONTROL_PKT_ERROR": 57,
    "AXI_MM_SLAVE_ERROR": 58, "INSTR_DECOMPRSN_ERROR": 59,
    "PM_ECC_ERROR_SCRUB_2BIT": 62, "PM_ECC_ERROR_1BIT": 63,
    "PM_ECC_ERROR_2BIT": 64,
}

STATUS_NAMES = {0: "Idle", 1: "Running", 2: "Paused"}

# Bit positions
BIT_STALL_LOCK_ACQ = 2
BIT_STALL_LOCK_REL = 3
BIT_STALL_STREAM   = 4
BIT_STALL_TCT      = 5
BIT_ERR_BD_UNAVAIL  = 10
BIT_ERR_BD_INVALID  = 11
BIT_CH_RUNNING      = 19
Q_SIZE_LSB          = 20
Q_SIZE_BITS         = 3
CUR_BD_LSB          = 24
CUR_BD_BITS         = 4

# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args(argv=None):
    """Custom argument parser to handle -mm2s0/-s2mm1 style dir_ch args.

    argparse would interpret -mm2s0 as a flag, so we extract it manually
    from sys.argv before passing the rest to argparse.
    """
    raw_args = argv if argv is not None else sys.argv[1:]

    # Extract the dir_ch argument: first arg matching -mm2s\d or -s2mm\d
    dir_ch = None
    remaining = []
    for arg in raw_args:
        if dir_ch is None and re.match(r"^-(mm2s|s2mm)\d+$", arg, re.IGNORECASE):
            dir_ch = arg
        else:
            remaining.append(arg)

    parser = argparse.ArgumentParser(
        prog="aiediag",
        description="Flow-aware DMA diagnostic tool for AIE",
    )
    sub = parser.add_subparsers(dest="command")
    dig = sub.add_parser("dig", help="Dig into a tile's DMA channel")
    dig.add_argument("col", type=int, help="Logical column (from IR/JSON, 0-based)")
    dig.add_argument("row", type=int, help="Physical row (3-6 for core, 0 for shim)")
    dig.add_argument("startcol_kw", nargs="?", default=None,
                     help="Literal 'startcol' keyword")
    dig.add_argument("startcol_val", nargs="?", type=int, default=None,
                     help="startcol value (physical column offset)")
    dig.add_argument("--json-dir", default=None,
                     help="Directory containing JSON provenance maps (auto-searches common locations)")
    dig.add_argument("--aie-version", choices=["5", "2ps"], default=None,
                     help="AIE version: 5 (AIEML) or 2ps; default auto-detected from provenance JSON")
    dig.add_argument("--target", default=None,
                     help="Pass-through to aiedbg (e.g., baremetal://192.168.0.1:9999)")
    dig.add_argument("-dev", "--device", default=None,
                     help="Device type for aiedbg (e.g., pal)")
    dig.add_argument("--dry-run", action="store_true",
                     help="Skip aiedbg calls; print what would be read")
    dig.add_argument("--shim-events-json",
                     default=os.path.expanduser("~/aiejson/shimtile_events.json"),
                     help="Path to shimtile_events.json (default: ~/aiejson/shimtile_events.json)")

    pc = sub.add_parser("pc", help="Read a core's PC and map it to kernel source file:line")
    pc.add_argument("col", type=int, help="Logical column (from IR/JSON, 0-based)")
    pc.add_argument("row", type=int, help="Physical row (3-6 for core tiles)")
    pc.add_argument("startcol_kw", nargs="?", default=None,
                    help="Literal 'startcol' keyword")
    pc.add_argument("startcol_val", nargs="?", type=int, default=None,
                    help="startcol value (physical column offset)")
    pc.add_argument("--pc", default=None,
                    help="Use this PC value (hex/dec) instead of reading from HW")
    pc.add_argument("--linemap", default=None,
                    help="Path to kernel.linemap.json (auto-searched if omitted)")
    pc.add_argument("--aie-version", choices=["5", "2ps"], default=None,
                    help="AIE version: 5 (AIEML) or 2ps; default auto-detected from provenance JSON")
    pc.add_argument("--target", default=None,
                    help="Pass-through to aiedbg (e.g., baremetal://192.168.0.1:9999)")
    pc.add_argument("-dev", "--device", default=None,
                    help="Device type for aiedbg (e.g., pal)")
    pc.add_argument("--dry-run", action="store_true",
                    help="Skip aiedbg calls; print what would be read")

    args = parser.parse_args(remaining)
    args.dir_ch = dir_ch
    return args

def parse_dir_ch(raw):
    """Parse '-mm2s0' into ('mm2s', 0)."""
    s = raw.lstrip("-").lower()
    m = re.match(r"^(mm2s|s2mm)(\d+)$", s)
    if not m:
        print(f"Error: invalid dir_ch '{raw}'. Expected -mm2s0, -mm2s1, -s2mm0, -s2mm1", file=sys.stderr)
        sys.exit(1)
    return m.group(1), int(m.group(2))

# ─── JSON loading ─────────────────────────────────────────────────────────────

DFSCHE_FILENAME = "dfscheduleprovenancemap.json"
DMAPHOP_FILENAMES = ["dmaphopprovenacemap.json", "provenance_map.json"]

# Common directories to search (relative to cwd) when --json-dir is not given
_SEARCH_DIRS = [
    "./worklocal",
    "./aout/worklocal",
    "./src/mlir/mlirfront/tilinglinalg/pass/unitest/build/worklocal",
    "./src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal",
]

def _find_json_dir():
    """Auto-detect the directory containing the provenance map JSONs."""
    for d in _SEARCH_DIRS:
        if os.path.isfile(os.path.join(d, DFSCHE_FILENAME)):
            return d
    return None

def load_jsons(json_dir):
    if json_dir is None:
        json_dir = _find_json_dir()
        if json_dir is None:
            print(f"Error: cannot find {DFSCHE_FILENAME} in any of:", file=sys.stderr)
            for d in _SEARCH_DIRS:
                print(f"  {d}", file=sys.stderr)
            print("Use --json-dir to specify the directory.", file=sys.stderr)
            return None, None
        print(f"  Auto-detected JSON dir: {os.path.abspath(json_dir)}")

    dfsche_path = os.path.join(json_dir, DFSCHE_FILENAME)
    dmaphop_path = None
    for name in DMAPHOP_FILENAMES:
        candidate = os.path.join(json_dir, name)
        if os.path.exists(candidate):
            dmaphop_path = candidate
            break

    dfsche = None
    dmaphop = None

    if os.path.exists(dfsche_path):
        with open(dfsche_path) as f:
            dfsche = json.load(f)
    else:
        print(f"Warning: {dfsche_path} not found", file=sys.stderr)

    if dmaphop_path and os.path.exists(dmaphop_path):
        with open(dmaphop_path) as f:
            dmaphop = json.load(f)
    else:
        searched = [os.path.join(json_dir, n) for n in DMAPHOP_FILENAMES]
        print(f"Warning: dmaphop provenance map not found, searched: {searched}", file=sys.stderr)

    return dfsche, dmaphop

def startcol_from_jsons(dfsche, dmaphop):
    """Return the partition origin (absolute startcol) emitted into the provenance
    map JSONs by the compiler, or None if not present. dfsche takes precedence
    over dmaphop. phys_col = col + startcol."""
    for j in (dfsche, dmaphop):
        if not isinstance(j, dict):
            continue
        sc = j.get("partition", {}).get("startcol") if isinstance(j.get("partition"), dict) else None
        if sc is None:
            sc = j.get("startcol")
        if sc is not None:
            return sc
    return None

def debug_aie_version_from_gen(aie_gen):
    """Normalize any AIE version string to the debug offset-map string.

    The compiler emits "Gen1"/"Gen2"/"Gen5" (from --aie-version 1/2/5):
    "Gen5" = A78/Versal Gen2 = AIE2PS (debug "2ps"); "Gen1"/"Gen2" = A72/Gen1
    = AIEML (debug "5"). The mapping lives here (Python) so it is adjustable
    without a C++ rebuild.

    This function is the single normalization point and is idempotent, so it is
    safe to call on values that are already debug strings (e.g. "2ps"):
      "5" / "Gen5"        -> "2ps"   (AIE2PS)
      "1" / "2" / "Gen1"  -> "5"     (AIEML)
      "2ps"               -> "2ps"   (passthrough)
    """
    g = str(aie_gen).strip().lower().removeprefix("gen")
    if g == "2ps":
        return "2ps"
    return "2ps" if g == "5" else "5"

def aie_version_from_jsons(dfsche, dmaphop):
    """Return the debug offset-map string derived from the compiler aie-gen
    emitted into the provenance JSONs, or None if not present. dfsche takes
    precedence over dmaphop."""
    for j in (dfsche, dmaphop):
        if not isinstance(j, dict):
            continue
        gen = j.get("partition", {}).get("aie_gen") if isinstance(j.get("partition"), dict) else None
        if gen is None:
            gen = j.get("aie_gen")
        if gen is not None:
            return debug_aie_version_from_gen(gen)
    return None

# ─── Kernel line map (PC -> source) ───────────────────────────────────────────

LINEMAP_FILENAME = "kernel.linemap.json"

# Directories searched for kernel.linemap.json (built next to the kernel ELF).
_LINEMAP_SEARCH_DIRS = [
    "./build",
    "./worklocal/build",
    "./aout/worklocal/build",
    "./src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/build",
    "./src/mlir/mlirfront/tilinglinalg/pass/unitest/build/worklocal/build",
    "./worklocal",
    "./aout/worklocal",
]

def find_linemap(explicit=None):
    """Locate kernel.linemap.json. Returns a path or None."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for d in _LINEMAP_SEARCH_DIRS:
        candidate = os.path.join(d, LINEMAP_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    return None

def load_linemap(explicit=None):
    """Load kernel.linemap.json. Returns (entries, path) or (None, None).

    entries is a list sorted by integer address, each {addr, addr_int, file, line}.
    """
    path = find_linemap(explicit)
    if path is None:
        return None, None
    with open(path) as f:
        data = json.load(f)
    entries = data.get("entries", [])
    # Ensure sorted by integer address for bisect.
    entries.sort(key=lambda e: e.get("addr_int", int(e.get("addr", "0x0"), 16)))
    return entries, path

def pc_to_source(entries, pc):
    """Map a program counter to {file, line, addr} via greatest addr <= pc.

    entries: sorted list from load_linemap. Returns the matching entry dict or None.
    """
    if not entries:
        return None
    import bisect
    addrs = [e.get("addr_int", int(e.get("addr", "0x0"), 16)) for e in entries]
    idx = bisect.bisect_right(addrs, pc) - 1
    if idx < 0:
        return None
    return entries[idx]

# ─── Register access ─────────────────────────────────────────────────────────

def compute_reg_offset(tile_type, direction, channel, aie_version):
    """Compute the register offset for DMA status."""
    if tile_type == "core":
        base_map = DMA_STATUS_OFFSETS["core"]
    elif tile_type == "shim":
        key = "shim_5" if aie_version == "5" else "shim_2ps"
        base_map = DMA_STATUS_OFFSETS[key]
    else:
        base_map = DMA_STATUS_OFFSETS["core"]
    base = base_map[direction]
    return base + channel * CH_STRIDE

def _bd_type_key(tile_type, aie_version):
    """Map tile_type+aie_version to a BD_BASE_STRIDE / BD_LEN_MASK key."""
    if tile_type == "shim":
        return "shim_5" if aie_version == "5" else "shim_2ps"
    return "core"

def bd_length_offset(tile_type, aie_version, bd_id):
    """Register offset of word 0 (Buffer_Length) of the given BD."""
    base, stride = BD_BASE_STRIDE[_bd_type_key(tile_type, aie_version)]
    return base + bd_id * stride

def core_status_offset(aie_version):
    """Core_Status register offset for the given AIE generation.
    Defaults to AIE2PS when the version is unrecognized (matches the other
    core-register defaults in this module)."""
    return CORE_STATUS_OFFSET.get(str(aie_version), CORE_STATUS_OFFSET["2ps"])

def decode_core_status(raw):
    """Decode a Core_Status register value into a dict of set bit-field names.

    Returns {"raw": raw, "enable": bool, "reset": bool, "done": bool,
             "error_halt": bool, "debug_halt": bool, "stalls": [names...],
             "set": [all set field names]}.
    `stalls` excludes ENABLE/RESET/CORE_DONE and the two halt bits so it holds
    only the "why is it not running" reasons."""
    non_stall = {"ENABLE", "RESET", "CORE_DONE"}
    set_names = [name for name, lsb in CORE_STATUS_BITS if raw & (1 << lsb)]
    stalls = [n for n in set_names
              if n not in non_stall and not n.endswith("_HALT")]
    return {
        "raw": raw,
        "enable": bool(raw & 0x1),
        "reset": bool(raw & 0x2),
        "done": bool(raw & (1 << 20)),
        "error_halt": bool(raw & (1 << 19)),
        "debug_halt": bool(raw & (1 << 16)),
        "stalls": stalls,
        "set": set_names,
    }

def format_core_status(phys_col, row, decoded):
    """Human-readable Core_Status decode with a one-line summary verdict."""
    raw = decoded["raw"]
    lines = [f"  Core status tile({phys_col},{row}):  raw=0x{raw:08X}"]
    # Headline state.
    if decoded["reset"]:
        lines.append(yellow("    STATE: RESET (core held in reset / not enabled)"))
    elif decoded["error_halt"]:
        lines.append(red("    STATE: ERROR_HALT (core halted on error)"))
    elif decoded["done"]:
        lines.append(green("    STATE: DONE (core finished, Core_Done set)"))
    elif decoded["debug_halt"]:
        lines.append(yellow("    STATE: DEBUG_HALT (halted by debugger)"))
    elif decoded["enable"] and decoded["stalls"]:
        lines.append(yellow("    STATE: ENABLED but STALLED "
                            f"({', '.join(decoded['stalls'])})"))
    elif decoded["enable"]:
        lines.append(green("    STATE: RUNNING (enabled, no stall)"))
    else:
        lines.append(yellow("    STATE: not enabled (Enable=0, Reset=0)"))
    # Flag lines.
    lines.append(f"    Enable={int(decoded['enable'])}  "
                 f"Reset={int(decoded['reset'])}  "
                 f"Done={int(decoded['done'])}  "
                 f"Error_Halt={int(decoded['error_halt'])}  "
                 f"Debug_Halt={int(decoded['debug_halt'])}")
    if decoded["stalls"]:
        lines.append("    Stalls: " + ", ".join(decoded["stalls"]))
    return "\n".join(lines)

def format_reg_read_trace(phys_col, row, offset, value):
    """One-line raw address + raw value trace for a single register read.
    Absolute address follows the runtime style: (row<<20)|offset."""
    abs_off = (row << 20) | offset
    vtxt = "----------" if value is None else f"0x{value:08X}"
    return (f"  reg read {phys_col} {row} 0x{offset:X}  "
            f"(abs 0x{abs_off:08X}) = {vtxt}")

def format_reg_read_block(reads):
    """Render collected register reads as a light-yellow brace-delimited block,
    intended to be printed at the *bottom* of a command's output.

    `reads` is a list of (phys_col, row, offset, value) tuples. Returns None
    when the list is empty (nothing to show)."""
    if not reads:
        return None
    lines = ["[registers read] {"]
    for phys_col, row, offset, value in reads:
        lines.append(format_reg_read_trace(phys_col, row, offset, value))
    lines.append("}")
    return lightyellow("\n".join(lines))

def run_aiedbg_reg_read(phys_col, row, offset, target=None, device=None,
                        dry_run=False, sink=None):
    """Run aiedbg reg read and return the raw 32-bit value.

    When `sink` is a list, append a (phys_col, row, offset, value) tuple to it
    for each read instead of printing inline. The caller flushes the collected
    reads as a light-yellow block at the bottom of its output (see
    format_reg_read_block). `sink` is opt-in so JSON consumers (e.g.
    schedule_debug_server) that pass nothing stay silent.
    """
    cmd = ["aiedbg", "--json"]
    if target:
        cmd += ["--target", target]
    if device:
        cmd += ["--device", device]
    cmd += ["reg", "read", str(phys_col), str(row), f"0x{offset:X}"]

    if dry_run:
        print(f"  [dry-run] would execute: {' '.join(cmd)}")
        if sink is not None:
            sink.append((phys_col, row, offset, None))
        return None

    def _traced(val):
        if sink is not None:
            sink.append((phys_col, row, offset, val))
        return val

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        print(f"Error: 'aiedbg' not found in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"Error: aiedbg timed out for tile({phys_col},{row})", file=sys.stderr)
        return _traced(None)

    if result.returncode != 0:
        print(f"Warning: aiedbg returned error for tile({phys_col},{row}): {result.stderr.strip()}", file=sys.stderr)
        return _traced(None)

    # Parse the register value from aiedbg's machine-readable JSON output.
    out = result.stdout.strip()
    try:
        data = json.loads(out)
        if "value_hex" in data:
            return _traced(int(data["value_hex"], 16))
        if "value" in data:
            return _traced(int(data["value"]))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Defensive fallback for older aiedbg without --json: the human-readable
    # line looks like "----0x1DF10 --- REG_0x1DF10-- 0x04080012". The register
    # VALUE is the LAST hex token, not the first (which is the offset/address).
    tokens = re.findall(r"0x[0-9a-fA-F]+", out)
    if tokens:
        return _traced(int(tokens[-1], 16))
    print(f"Warning: could not parse aiedbg output: {out}", file=sys.stderr)
    return _traced(None)

# ─── Memory read + klog decode ─────────────────────────────────────────────────

def run_aiedbg_mem_read(phys_col, row, addr, nwords, target=None, device=None,
                        dry_run=False):
    """Run `aiedbg mem read` and return the words as a list of ints.

    Mirrors run_aiedbg_reg_read's house style.  aiedbg reads at most 256 words
    per request; callers must chunk larger reads.  `addr` is a byte offset into
    the tile (data memory base = tile offset 0), matching _tile_to_mmio.  In
    dry_run the command is printed and [] returned.
    """
    nwords = min(int(nwords), 256)
    cmd = ["aiedbg", "--json"]
    if target:
        cmd += ["--target", target]
    if device:
        cmd += ["--device", device]
    cmd += ["mem", "read", str(phys_col), str(row), f"0x{addr:X}", str(nwords)]

    if dry_run:
        print(f"  [dry-run] would execute: {' '.join(cmd)}")
        return []

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        print("Error: 'aiedbg' not found in PATH", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"Error: aiedbg timed out for tile({phys_col},{row})", file=sys.stderr)
        return []

    if result.returncode != 0:
        print(f"Warning: aiedbg returned error for tile({phys_col},{row}): "
              f"{result.stderr.strip()}", file=sys.stderr)
        return []

    out = result.stdout.strip()
    try:
        data = json.loads(out)
        if "words" in data and isinstance(data["words"], list):
            vals = []
            for w in data["words"]:
                if isinstance(w, str):
                    vals.append(int(w, 16) if w.lower().startswith("0x") else int(w))
                else:
                    vals.append(int(w))
            return vals
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback for older aiedbg without --json: scrape hex tokens.
    tokens = re.findall(r"0x[0-9a-fA-F]+", out)
    if tokens:
        return [int(t, 16) for t in tokens]
    print(f"Warning: could not parse aiedbg mem output: {out}", file=sys.stderr)
    return []

def read_klog(phys_col, row, target=None, device=None, dry_run=False):
    """Read the klog region from a core tile.

    Returns (write_index, words) where `words` is the int32 slot array starting
    at slot 0, or None if nothing could be read.  Reads slot[0] first; if the
    write_index is out of range, returns (wi, []) so the caller can report "no
    log".  Otherwise reads min(wi+1, 512) words in <=256-word chunks.
    """
    head = run_aiedbg_mem_read(phys_col, row, KLOG_DM_OFFSET, 1,
                               target=target, device=device, dry_run=dry_run)
    if dry_run:
        return (0, [])
    if not head:
        return None
    wi = head[0]
    if wi <= 0 or wi > KLOG_MAX_SLOTS:
        return (wi, [])

    total = min(wi + 1, KLOG_REGION_WORDS)
    words = []
    read = 0
    while read < total:
        chunk = min(256, total - read)
        addr = KLOG_DM_OFFSET + read * 4
        got = run_aiedbg_mem_read(phys_col, row, addr, chunk,
                                  target=target, device=device, dry_run=False)
        if not got:
            break
        words.extend(got)
        read += len(got)
        if len(got) < chunk:
            break
    return (wi, words)

def _klog_decode_tag(tag_raw):
    """Decode a packed int32 tag into an ASCII string (big-endian).

    Mirrors kernel_log.h's printf("%s"): stops at the first null byte (tags
    shorter than 4 chars are null-padded), and renders non-printable bytes as
    '.' so binary junk stays readable.
    """
    out = []
    for shift in (24, 16, 8, 0):
        c = (tag_raw >> shift) & 0xFF
        if c == 0:
            break
        out.append(chr(c) if 32 <= c < 127 else ".")
    return "".join(out)

def format_klog(phys_col, row, wi, words):
    """Render the klog contents, mirroring kernel_log.h klog_read()."""
    if wi <= 0 or wi > KLOG_MAX_SLOTS:
        return f"[klog] tile({phys_col},{row}): no log (write_index={wi})"

    lines = [f"[klog] tile({phys_col},{row}): {wi // 2} entries"]
    n = 0
    for i in range(0, wi, 2):
        if i + 2 >= len(words):
            break
        tag_raw = words[i + 1]
        val = words[i + 2]
        tag = _klog_decode_tag(tag_raw)
        sval = val - (1 << 32) if val & 0x80000000 else val
        lines.append(f"  [{i // 2}] {tag} = {sval} (0x{val & 0xFFFFFFFF:08x})")
        n += 1
    if n and words[1] == KLOG_MAGIC:
        lines.append(green("  (klog_init magic OK)"))
    elif n:
        lines.append(yellow("  (warning: first entry is not the KLOG magic; "
                            "klog_init may not have run)"))
    return "\n".join(lines)

# ─── Status decode ────────────────────────────────────────────────────────────

def decode_dma_status(raw):
    """Decode the 32-bit DMA status register into a dict."""
    def bit(n):
        return bool(raw & (1 << n))
    def bits(lsb, width):
        return (raw >> lsb) & ((1 << width) - 1)

    return {
        "raw": raw,
        "status": raw & 0x3,
        "status_str": STATUS_NAMES.get(raw & 0x3, f"Unknown({raw & 0x3})"),
        "stall_lock_acq": bit(BIT_STALL_LOCK_ACQ),
        "stall_lock_rel": bit(BIT_STALL_LOCK_REL),
        "stall_stream":   bit(BIT_STALL_STREAM),
        "stall_tct":      bit(BIT_STALL_TCT),
        "err_bd_unavail":  bit(BIT_ERR_BD_UNAVAIL),
        "err_bd_invalid":  bit(BIT_ERR_BD_INVALID),
        "channel_running": bit(BIT_CH_RUNNING),
        "q_size":          bits(Q_SIZE_LSB, Q_SIZE_BITS),
        "cur_bd":          bits(CUR_BD_LSB, CUR_BD_BITS),
    }

def format_dma_status(col, row, direction, channel, decoded):
    """Format decoded DMA status as a multi-line string."""
    lines = []
    raw = decoded["raw"]
    lines.append(f"  raw=0x{raw:08X}  {decoded['status_str']}  "
                 f"running={int(decoded['channel_running'])}  "
                 f"q_size={decoded['q_size']}  "
                 f"cur_bd={decoded['cur_bd']}")

    stalls = []
    if decoded["stall_lock_acq"]: stalls.append("lock_acq")
    if decoded["stall_lock_rel"]: stalls.append("lock_rel")
    if decoded["stall_stream"]:   stalls.append("stream")
    if decoded["stall_tct"]:      stalls.append("tct")

    stall_line = (f"  STALLED: lock_acq={int(decoded['stall_lock_acq'])} "
                  f"lock_rel={int(decoded['stall_lock_rel'])} "
                  f"stream={int(decoded['stall_stream'])} "
                  f"tct={int(decoded['stall_tct'])}")
    lines.append(stall_line)

    errs = []
    if decoded["err_bd_unavail"]: errs.append("BD_UNAVAIL")
    if decoded["err_bd_invalid"]: errs.append("BD_INVALID")
    if errs:
        lines.append(red(f"  >> ERRORS: {', '.join(errs)}"))

    if decoded["stall_stream"]:
        if direction == "mm2s":
            lines.append(yellow(f"  >> STREAM STALL: backpressure from downstream"))
        else:
            lines.append(yellow(f"  >> STREAM STALL: no data from upstream"))
    if decoded["stall_lock_acq"]:
        if direction == "mm2s":
            lines.append(yellow(f"  >> LOCK ACQ STALL: kernel hasn't released output buffer"))
        else:
            lines.append(yellow(f"  >> LOCK ACQ STALL: DMA waiting for buffer to be available"))
    if decoded["stall_lock_rel"]:
        lines.append(yellow(f"  >> LOCK REL STALL: cannot release lock"))
    if decoded["status"] == 0 and not decoded["channel_running"]:
        lines.append(f"  >> IDLE: channel completed or never started")

    return "\n".join(lines)

# ─── Shim event status ────────────────────────────────────────────────────

def load_shim_events_json(path):
    """Load shimtile_events.json. Returns events dict or None if not found."""
    if not os.path.isfile(path):
        print(yellow(f"Warning: {path} not found, skipping shim event check"), file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)

def read_shim_event_status(phys_col, direction, channel, target, device, dry_run,
                           sink=None):
    """Read event_status reg0+reg1 from shim tile (row=0), decode DMA events.

    Returns dict {event_name: bool} or None on failure.
    """
    key = (direction.lower(), channel)
    if key not in SHIM_DMA_EVENT_IDS:
        return None

    event_map = SHIM_DMA_EVENT_IDS[key]

    # Read reg0 (events 0-31) and reg1 (events 32-63)
    reg0 = run_aiedbg_reg_read(phys_col, 0, SHIM_EVT_STATUS_REG0,
                                target=target, device=device, dry_run=dry_run,
                                sink=sink)
    reg1 = run_aiedbg_reg_read(phys_col, 0, SHIM_EVT_STATUS_REG1,
                                target=target, device=device, dry_run=dry_run,
                                sink=sink)

    if dry_run:
        return None  # no data in dry-run, but commands were printed

    if reg0 is None and reg1 is None:
        return None

    result = {}
    for name, eid in event_map.items():
        if eid < 32:
            result[name] = bool((reg0 or 0) & (1 << eid))
        else:
            result[name] = bool((reg1 or 0) & (1 << (eid - 32)))
    return result

def decode_shim_event_status(direction, channel, reg0, reg1):
    """Decode shim event status from already-read register values.
    Returns dict {event_name: bool} or None if both regs are None."""
    key = (direction.lower(), channel)
    if key not in SHIM_DMA_EVENT_IDS:
        return None
    if reg0 is None and reg1 is None:
        return None
    event_map = SHIM_DMA_EVENT_IDS[key]
    result = {}
    for name, eid in event_map.items():
        if eid < 32:
            result[name] = bool((reg0 or 0) & (1 << eid))
        else:
            result[name] = bool((reg1 or 0) & (1 << (eid - 32)))
    return result

def format_shim_event_status(phys_col, direction, channel, event_bits, shim_events):
    """Format the shim event status for printing.

    Args:
        phys_col: physical column
        direction: 'mm2s' or 's2mm'
        channel: channel number
        event_bits: dict {event_name: bool} from read_shim_event_status
        shim_events: loaded shimtile_events.json (for display names), may be None
    """
    lines = []
    lines.append(f"  Shim events for {direction.upper()} ch{channel} (col {phys_col}):")

    key = (direction.lower(), channel)
    event_map = SHIM_DMA_EVENT_IDS.get(key, {})

    for name, eid in event_map.items():
        fired = event_bits.get(name, False)
        # Build display name from json if available
        display = f"DMA_{direction.upper()}_{channel}_{name}"
        if shim_events and str(eid) in shim_events:
            display = shim_events[str(eid)]
        tag = red("SET") if fired else "not set"
        lines.append(f"    {display + ':':<44s} {tag}")

    # Summary line
    started = event_bits.get("START_TASK", False)
    finished = event_bits.get("FINISHED_TASK", False)
    if started and finished:
        lines.append(green(f"    >> {direction.upper()} ch{channel} started and finished (DMA completed)"))
    elif started and not finished:
        lines.append(yellow(f"    >> {direction.upper()} ch{channel} was started but never finished (still running or stuck)"))
    elif not started:
        lines.append(yellow(f"    >> {direction.upper()} ch{channel} was never started (start_io not issued?)"))

    return "\n".join(lines)

# ─── Core-tile event status (memory + core modules) ───────────────────────────

def read_event_status_4(phys_col, row, regs, target, device, dry_run, sink=None):
    """Read 4 event_status registers (events 0-127) for a core tile.

    Returns [r0, r1, r2, r3] (None entries tolerated), or None in dry_run.
    """
    vals = []
    for off in regs:
        vals.append(run_aiedbg_reg_read(phys_col, row, off,
                                        target=target, device=device,
                                        dry_run=dry_run, sink=sink))
    if dry_run:
        return None  # commands printed, but no data to decode
    if all(v is None for v in vals):
        return None
    return vals

def _evt_active(eid, regs4):
    """True if event id `eid` (0-127) is set in the 4-register vector."""
    return bool((regs4[eid // 32] or 0) & (1 << (eid % 32)))

def format_core_mem_dma_events(phys_col, row, regs4):
    """Format memory-module DMA start/finish/error per all 4 channels.

    The raw register addresses + values are collected by run_aiedbg_reg_read
    into the caller's sink and flushed as a light-yellow block at the bottom of
    the command, so this only renders the decoded per-channel view."""
    lines = []
    lines.append(f"  Memory-module DMA events tile({phys_col},{row}):")
    for direction, channel in CORE_DMA_CHANNELS:
        event_map = CORE_MEM_DMA_EVENT_IDS[(direction, channel)]
        started = _evt_active(event_map["START_TASK"], regs4)
        finished_bd = _evt_active(event_map["FINISHED_BD"], regs4)
        finished = _evt_active(event_map["FINISHED_TASK"], regs4)
        error = _evt_active(event_map["ERROR"], regs4)
        lines.append(f"    {direction.upper()} ch{channel}:")
        lines.append(f"      {'START_TASK:':<16s} {red('SET') if started else 'not set'}")
        lines.append(f"      {'FINISHED_BD:':<16s} {red('SET') if finished_bd else 'not set'}")
        lines.append(f"      {'FINISHED_TASK:':<16s} {red('SET') if finished else 'not set'}")
        lines.append(f"      {'ERROR:':<16s} {red('SET') if error else 'not set'}")
        if error:
            lines.append(red(f"      >> {direction.upper()} ch{channel} DMA ERROR event active"))
        elif started and finished:
            lines.append(green(f"      >> {direction.upper()} ch{channel} started and finished (DMA completed)"))
        elif started and not finished:
            lines.append(yellow(f"      >> {direction.upper()} ch{channel} started but never finished (still running or stuck)"))
        else:
            lines.append(yellow(f"      >> {direction.upper()} ch{channel} never started (start_io not issued?)"))
    return "\n".join(lines)

def format_core_module_errors(phys_col, row, regs4):
    """List any active core-module error events."""
    lines = []
    lines.append(f"  Core-module errors tile({phys_col},{row}):")
    active = [name for name, eid in CORE_ERROR_EVENT_IDS.items()
              if _evt_active(eid, regs4)]
    if active:
        for name in active:
            lines.append(red(f"    {name}: SET"))
    else:
        lines.append(green("    no core errors"))
    return "\n".join(lines)

# ─── JSON lookup helpers ──────────────────────────────────────────────────────

def find_tile_in_json(dfsche, col, row):
    """Find the tile entry matching (col, row) in tiles[]."""
    if not dfsche or "tiles" not in dfsche:
        return None
    for t in dfsche["tiles"]:
        if t["col"] == col and t["row"] == row:
            return t
    return None

def find_channel_in_tile(tile_entry, direction, channel):
    """Find the dma_channels entry matching direction+channel."""
    if not tile_entry or "dma_channels" not in tile_entry:
        return None
    dir_upper = direction.upper()
    for ch in tile_entry["dma_channels"]:
        if ch["channel"] == channel and ch["direction"] == dir_upper:
            return ch
    return None

def find_flow(dfsche, flow_index):
    """Find the flow_summary entry for the given flow_index."""
    if not dfsche or "flow_summary" not in dfsche:
        return None
    for f in dfsche["flow_summary"]:
        if f["flow_index"] == flow_index:
            return f
    return None

def find_connected_tiles(flow_entry, col, row, direction):
    """Given a flow_summary entry, find tiles that are NOT the queried tile.

    Returns list of dicts with tile_col, tile_row, channel, io_direction, etc.
    """
    if not flow_entry:
        return []
    connected = []
    for entry in flow_entry.get("entries", []):
        ec, er = entry["tile_col"], entry["tile_row"]
        if ec == col and er == row and entry["io_direction"].lower() == direction:
            continue
        connected.append(entry)
    return connected

def find_routing_paths(dmaphop, flow_index, flow_entry):
    """Find communication_paths from dmaphopprovenacemap.json that match this flow.

    Matching strategy: the path's producer+consumer tile set must match
    the flow_summary's tile set closely (endpoints, not just any overlap).
    We require that the path's endpoint tiles (producer+consumer, excluding
    passthrough hops) are a subset of the flow's tiles.
    """
    if not dmaphop or "communication_paths" not in dmaphop:
        return []

    if not flow_entry:
        return []

    # Build set of (col, row) from flow_summary entries
    flow_tiles = set()
    for e in flow_entry.get("entries", []):
        flow_tiles.add((e["tile_col"], e["tile_row"]))

    matches = []
    for path in dmaphop["communication_paths"]:
        path_endpoint_tiles = set()
        for stage in path.get("stages", []):
            if stage["role"] in ("producer", "consumer"):
                if "tile" in stage:
                    t = stage["tile"]
                    path_endpoint_tiles.add((t["col"], t["row"]))
                if "tiles" in stage:
                    for t in stage["tiles"]:
                        path_endpoint_tiles.add((t["col"], t["row"]))

        # Match if ALL path endpoint tiles are in the flow's tile set
        if path_endpoint_tiles and path_endpoint_tiles <= flow_tiles:
            matches.append(path)

    return matches

# ─── BD chain printing ───────────────────────────────────────────────────────

def read_bd_hw_lengths(phys_col, row, tile_type, aie_version, bd_ids,
                       target, device, dry_run, sink=None):
    """Read the real Buffer_Length (in bytes) of each BD from hardware.

    Returns {bd_id: bytes} (None entries tolerated), or None in dry-run.
    """
    mask = BD_LEN_MASK[_bd_type_key(tile_type, aie_version)]
    result = {}
    for bd_id in bd_ids:
        off = bd_length_offset(tile_type, aie_version, bd_id)
        word0 = run_aiedbg_reg_read(phys_col, row, off,
                                    target=target, device=device,
                                    dry_run=dry_run, sink=sink)
        if word0 is None:
            result[bd_id] = None
        else:
            result[bd_id] = (word0 & mask) * 4  # words -> bytes
    if dry_run:
        return None  # commands printed, no data to compare
    if all(v is None for v in result.values()):
        return None
    return result

def format_bd_chain(channel_entry, hw_lengths=None):
    """Format the BD chain from a dma_channels entry.

    If hw_lengths ({bd_id: bytes}) is provided, append a per-BD
    intended-vs-real comparison block.
    """
    lines = []
    bd_chain = channel_entry.get("bd_chain", [])
    bd_ids = []
    for bd in bd_chain:
        bd_id = bd["bd_id"]
        bd_ids.append(bd_id)
        offset = bd.get("buffer_offset", 0)
        length = bd.get("len", 0)
        pkt_id = bd.get("packet_id", 0)
        enable_pkt = bd.get("enable_packet", False)
        next_bd = bd.get("next_bd", -1)
        ooo = bd.get("out_of_order_bd_id", None)

        acq = bd.get("acquire_lock", [{}])
        rel = bd.get("release_lock", [{}])
        acq_str = ",".join(f"lock({l['id']},{l['val']:+d})" for l in acq) if acq else "none"
        rel_str = ",".join(f"lock({l['id']},{l['val']:+d})" for l in rel) if rel else "none"

        pkt_str = f" pkt_id={pkt_id}" if enable_pkt else ""
        next_str = f" next->BD{next_bd}" if next_bd >= 0 else " next->NONE"
        ooo_str = f" ooo_bd={ooo}" if ooo is not None else ""

        # Shim-specific fields
        dim_str = ""
        if "dim_strides" in bd:
            strides = bd["dim_strides"]
            wraps = bd.get("dim_wraps", [])
            dim_str += f" strides={strides} wraps={wraps}"
        if "iter_step_size" in bd:
            dim_str += f" iter_step={bd['iter_step_size']} iter_wrap={bd.get('iter_wrap', 0)}"

        lines.append(f"  BD{bd_id}: offset={offset} len={length}B{pkt_str}{next_str}"
                     f" acq={acq_str} rel={rel_str}{ooo_str}{dim_str}")

    # Check ping-pong cycle
    if len(bd_ids) == 2:
        bd_a, bd_b = bd_chain[0], bd_chain[1]
        if bd_a.get("next_bd") == bd_b["bd_id"] and bd_b.get("next_bd") == bd_a["bd_id"]:
            lines.append(green(f"  Ping-pong: BD{bd_b['bd_id']}↔BD{bd_a['bd_id']} cycle OK"))
        elif bd_a.get("next_bd", -1) < 0 and bd_b.get("next_bd", -1) < 0:
            lines.append(f"  No ping-pong chain (next_bd=-1, likely shim OOO)")
        else:
            lines.append(yellow(f"  WARNING: ping-pong chain broken"))

    contract = channel_entry.get("contract", "")
    if contract:
        lines.append(f"  Contract: \"{contract}\"")

    start_io = channel_entry.get("start_io", [])
    repeat_total = 1
    for si in start_io:
        repeat = si.get("repeat_count", 0)
        scf = si.get("inside_scf_for", False)
        lr = si.get("loop_range", "")
        scf_str = f" (inside scf.for {lr})" if scf else ""
        lines.append(f"  start_io: repeat={repeat}{scf_str}")
        if repeat:
            repeat_total = repeat

    # Intended data volume (sum of per-BD len, scaled by repeat count)
    per_run = sum(bd.get("len", 0) for bd in bd_chain)
    total = per_run * repeat_total
    lines.append(f"  Total intended: {per_run}B/run x repeat {repeat_total} = {total}B")

    # Intended vs real HW BD length comparison
    if hw_lengths is not None:
        lines.append("  Intended vs real BD length (from HW):")
        for bd in bd_chain:
            bd_id = bd["bd_id"]
            intended = bd.get("len", 0)
            hw = hw_lengths.get(bd_id)
            if hw is None:
                tag = yellow("hw read failed")
                hw_str = "?"
            elif hw == 0:
                tag = yellow("hw not configured (len=0)")
                hw_str = "0B (0 words)"
            elif hw == intended:
                tag = green("OK")
                hw_str = f"{hw}B ({hw // 4} words)"
            else:
                tag = red("MISMATCH")
                hw_str = f"{hw}B ({hw // 4} words)"
            lines.append(f"    BD{bd_id}: intended={intended}B  hw={hw_str}  {tag}")

    return "\n".join(lines)

# ─── Routing path printing ───────────────────────────────────────────────────

def format_routing_path(path):
    """Format a communication_path entry."""
    lines = []
    pid = path.get("id", "?")
    direction = path.get("direction", "?")
    lines.append(f"  {pid} ({direction}):")

    for stage in path.get("stages", []):
        role = stage["role"]
        if role == "producer":
            if "tile" in stage:
                t = stage["tile"]
                lines.append(f"    producer: tile({t['col']},{t['row']}) [{t['type']}]"
                             f" ch={stage.get('channel', '?')}")
            if "tiles" in stage:
                for t in stage["tiles"]:
                    pkt = t.get("pkt_id", "")
                    pkt_str = f" pkt_id={pkt}" if pkt else ""
                    lines.append(f"    producer: tile({t['col']},{t['row']}) [{t['type']}]"
                                 f" port={t.get('dma_port', '?')}{pkt_str}")
        elif role == "channel":
            for hop in stage.get("hops", []):
                lines.append(f"    hop: {hop['from']} -> {hop['to']}")
        elif role == "consumer":
            if "tile" in stage:
                t = stage["tile"]
                lines.append(f"    consumer: tile({t['col']},{t['row']}) [{t['type']}]"
                             f" ch={stage.get('channel', '?')}")
            if "tiles" in stage:
                for t in stage["tiles"]:
                    pkt = t.get("pkt_id", "")
                    pkt_str = f" pkt_id={pkt}" if pkt else ""
                    lines.append(f"    consumer: tile({t['col']},{t['row']}) [{t['type']}]"
                                 f" port={t.get('dma_port', '?')}{pkt_str}")

    return "\n".join(lines)

# ─── Diagnosis engine ────────────────────────────────────────────────────────

def diagnose(direction, queried_decoded, connected_results,
             queried_shim_events=None, connected_shim_events=None,
             queried_row=None):
    """Produce diagnostic messages based on status of queried and connected tiles.

    Args:
        direction: 'mm2s' or 's2mm'
        queried_decoded: decoded status dict of the queried tile (or None)
        connected_results: list of (col, row, dir, ch, decoded_or_None) for connected tiles
        queried_shim_events: dict {event_name: bool} for queried tile if shim, or None
        connected_shim_events: dict keyed by (col, row, dir, ch) -> {event_name: bool}
        queried_row: row of queried tile (0 = shim)
    """
    if connected_shim_events is None:
        connected_shim_events = {}
    lines = []

    if queried_decoded is None:
        lines.append("Cannot diagnose: no register data for queried tile (dry-run or error)")
        return "\n".join(lines)

    q = queried_decoded

    # Check errors first
    if q["err_bd_unavail"]:
        lines.append(red("HW ERROR: BD_UNAVAIL — no BD queued for this channel"))
    if q["err_bd_invalid"]:
        lines.append(red("HW ERROR: BD_INVALID — BD configuration error"))

    # Idle
    if q["status"] == 0 and not q["channel_running"]:
        # Use shim events to disambiguate "never started" vs "completed"
        if queried_shim_events and queried_row == 0:
            started = queried_shim_events.get("START_TASK", False)
            finished = queried_shim_events.get("FINISHED_TASK", False)
            if started and finished:
                lines.append("Channel is IDLE — completed all programmed transfers")
            elif started and not finished:
                lines.append(yellow("Channel is IDLE (status) but START_TASK set without FINISHED — abnormal"))
            else:
                lines.append("Channel is IDLE — never started (START_TASK not set)")
        else:
            lines.append("Channel is IDLE — completed all transfers or never started")
        # Check if connected tiles are still active
        for cc, cr, cd, cch, cdec in connected_results:
            if cdec and (cdec["channel_running"] or cdec["status"] != 0):
                lines.append(yellow(f"  But connected tile({cc},{cr}) {cd.upper()} ch{cch} is still active!"))
        return "\n".join(lines)

    # Stream stall diagnosis
    # Track which connected tiles were already diagnosed via shim events
    shim_diagnosed = set()
    if q["stall_stream"]:
        if direction == "mm2s":
            lines.append(f"{direction.upper()} ch has STREAM BACKPRESSURE")
            for cc, cr, cd, cch, cdec in connected_results:
                if cdec is None:
                    lines.append(f"  tile({cc},{cr}) {cd.upper()} ch{cch}: no register data")
                    continue
                if cd.lower() == "s2mm":
                    if cdec["status"] == 0 and not cdec["channel_running"]:
                        # Use shim events to disambiguate idle shim tiles
                        evt = connected_shim_events.get((cc, cr, cd, cch))
                        if evt:
                            shim_diagnosed.add((cc, cr, cd, cch))
                            started = evt.get("START_TASK", False)
                            finished = evt.get("FINISHED_TASK", False)
                            if started and finished:
                                lines.append(yellow(f"  >> Receiver tile({cc},{cr}) S2MM ch{cch} completed all programmed transfers — repeat/iter_wrap too low?"))
                            elif not started:
                                lines.append(yellow(f"  >> Receiver tile({cc},{cr}) S2MM ch{cch} never started (START_TASK not set — start_io not issued)"))
                            else:
                                lines.append(yellow(f"  >> Receiver tile({cc},{cr}) S2MM ch{cch} started but not finished — stuck"))
                        else:
                            lines.append(yellow(f"  >> Receiver tile({cc},{cr}) S2MM ch{cch} is IDLE — never started or completed"))
                    elif cdec["stall_stream"]:
                        lines.append(yellow(f"  >> Receiver tile({cc},{cr}) S2MM ch{cch} also stream-stalled — possible routing congestion"))
                    elif cdec["stall_lock_acq"]:
                        lines.append(yellow(f"  >> Receiver tile({cc},{cr}) S2MM ch{cch} lock-stalled — kernel not consuming input"))
        else:
            lines.append(f"{direction.upper()} ch has STREAM STARVATION")
            for cc, cr, cd, cch, cdec in connected_results:
                if cdec is None:
                    lines.append(f"  tile({cc},{cr}) {cd.upper()} ch{cch}: no register data")
                    continue
                if cd.lower() == "mm2s":
                    if cdec["status"] == 0 and not cdec["channel_running"]:
                        # Use shim events to disambiguate idle shim tiles
                        evt = connected_shim_events.get((cc, cr, cd, cch))
                        if evt:
                            shim_diagnosed.add((cc, cr, cd, cch))
                            started = evt.get("START_TASK", False)
                            finished = evt.get("FINISHED_TASK", False)
                            if started and finished:
                                lines.append(yellow(f"  >> Sender tile({cc},{cr}) MM2S ch{cch} completed all programmed transfers — repeat/iter_wrap too low?"))
                            elif not started:
                                lines.append(yellow(f"  >> Sender tile({cc},{cr}) MM2S ch{cch} never started (START_TASK not set — start_io not issued)"))
                            else:
                                lines.append(yellow(f"  >> Sender tile({cc},{cr}) MM2S ch{cch} started but not finished — stuck"))
                        else:
                            lines.append(yellow(f"  >> Sender tile({cc},{cr}) MM2S ch{cch} is IDLE — never started (check start_io)"))
                    elif cdec["stall_lock_acq"]:
                        lines.append(yellow(f"  >> Sender tile({cc},{cr}) MM2S ch{cch} lock-stalled — kernel hasn't produced output"))

    # Lock stall diagnosis
    if q["stall_lock_acq"]:
        if direction == "mm2s":
            lines.append(f"{direction.upper()} ch stalled on LOCK ACQUIRE — kernel hasn't released output buffer (check core status)")
        else:
            lines.append(f"{direction.upper()} ch stalled on LOCK ACQUIRE — DMA waiting for kernel to consume input buffer")

    if q["stall_lock_rel"]:
        lines.append(f"{direction.upper()} ch stalled on LOCK RELEASE")

    if q["stall_tct"]:
        lines.append(f"{direction.upper()} ch stalled on TCT (task count)")

    if not lines:
        if q["status"] == 1:
            lines.append(green("Channel is Running normally"))
        elif q["status"] == 2:
            lines.append(yellow("Channel is Paused"))
        else:
            lines.append(f"Channel status: {q['status_str']}")

    # Shim event diagnosis (only for tiles not already diagnosed above)
    # Queried tile shim events
    if queried_shim_events and queried_row == 0:
        started = queried_shim_events.get("START_TASK", False)
        finished = queried_shim_events.get("FINISHED_TASK", False)
        if not started:
            lines.append(yellow(f"Shim {direction.upper()}: START_TASK not set — start_io not issued"))
        elif started and not finished:
            lines.append(yellow(f"Shim {direction.upper()}: started but not finished (transfer in progress or stuck)"))
        elif started and finished:
            lines.append(green(f"Shim {direction.upper()}: DMA completed normally"))

    # Connected shim tile events — skip tiles already covered in stream stall diagnosis
    for cc, cr, cd, cch, cdec in connected_results:
        if (cc, cr, cd, cch) in shim_diagnosed:
            continue
        evt = connected_shim_events.get((cc, cr, cd, cch))
        if not evt:
            continue
        started = evt.get("START_TASK", False)
        finished = evt.get("FINISHED_TASK", False)
        if not started:
            lines.append(yellow(f"Shim tile({cc},{cr}) {cd.upper()} ch{cch}: START_TASK not set — start_io not issued"))
        elif started and not finished:
            lines.append(yellow(f"Shim tile({cc},{cr}) {cd.upper()} ch{cch}: started but not finished (transfer in progress or stuck)"))
        elif started and finished:
            lines.append(green(f"Shim tile({cc},{cr}) {cd.upper()} ch{cch}: DMA completed normally"))

    return "\n".join(lines)

# ─── Main ─────────────────────────────────────────────────────────────────────

def _resolve_startcol(startcol_kw, startcol_val):
    """Shared startcol parsing for 'dig'/'pc' positional args."""
    if startcol_kw and startcol_kw.lower() == "startcol" and startcol_val is not None:
        return startcol_val
    if startcol_kw is not None:
        try:
            return int(startcol_kw)
        except ValueError:
            if startcol_kw.lower() == "startcol":
                return 0
            print(f"Error: unexpected argument '{startcol_kw}'", file=sys.stderr)
            sys.exit(1)
    return 0

def cmd_pc(args):
    """Read a core tile's Program Counter and map it to kernel source file:line."""
    col = args.col
    row = args.row
    startcol = _resolve_startcol(args.startcol_kw, args.startcol_val)
    phys_col = col + startcol

    print(bold("=" * 58))
    print(bold(" aiediag: PC -> source"))
    print(bold("=" * 58))
    print(f"Target tile: ({phys_col},{row})  [logical: col={col}, row={row}, startcol={startcol}]")

    if row == 0:
        print(red("Error: shim tiles (row 0) have no core PC register"), file=sys.stderr)
        return 1

    # ── Resolve the PC value ──────────────────────────────────────────────
    if args.pc is not None:
        pc_val = int(args.pc, 0)
        print(f"PC (supplied): 0x{pc_val:05X}")
    else:
        raw = run_aiedbg_reg_read(phys_col, row, CORE_PC_OFFSET,
                                  target=args.target, device=args.device,
                                  dry_run=args.dry_run)
        if raw is None:
            if args.dry_run:
                print("  (dry-run: PC not read)")
                return 0
            print(red("Error: could not read core PC register"), file=sys.stderr)
            return 1
        pc_val = raw & CORE_PC_MASK
        print(f"PC register (0x{CORE_PC_OFFSET:05X}): raw=0x{raw:08X} -> PC=0x{pc_val:05X}")

    # ── Load the line map and resolve ─────────────────────────────────────
    entries, lm_path = load_linemap(args.linemap)
    if entries is None:
        print(red(f"Error: {LINEMAP_FILENAME} not found "
                  f"(build kernel with -g, or pass --linemap)"), file=sys.stderr)
        return 1
    print(f"Line map: {lm_path} ({len(entries)} entries)")

    hit = pc_to_source(entries, pc_val)
    print()
    print(bold("--- PC -> source ---"))
    if hit is None:
        print(yellow(f"  PC 0x{pc_val:05X} is below the first mapped address — no match"))
        return 0
    print(green(f"  PC 0x{pc_val:05X}  ->  {hit['file']}:{hit['line']}  "
                f"(entry {hit['addr']})"))
    print(bold("=" * 58))
    return 0

def main(argv=None):
    args = parse_args(argv)

    if args.command == "pc":
        sys.exit(cmd_pc(args))

    if args.command != "dig":
        print("Usage:", file=sys.stderr)
        print("  aiediag dig COL ROW -DIR_CH [startcol N] [--json-dir PATH]", file=sys.stderr)
        print("  aiediag pc  COL ROW [startcol N] [--pc VAL] [--linemap PATH]", file=sys.stderr)
        sys.exit(1)

    if args.dir_ch is None:
        print("Error: missing -DIR_CH argument (e.g., -mm2s0, -s2mm1)", file=sys.stderr)
        sys.exit(1)

    col = args.col
    row = args.row
    direction, channel = parse_dir_ch(args.dir_ch)

    # None sentinel = not explicitly provided; resolve from provenance JSON later.
    startcol = None
    if args.startcol_kw and args.startcol_kw.lower() == "startcol" and args.startcol_val is not None:
        startcol = args.startcol_val
    elif args.startcol_kw is not None:
        # Maybe user passed startcol as a positional without the keyword
        try:
            startcol = int(args.startcol_kw)
        except ValueError:
            if args.startcol_kw.lower() == "startcol":
                pass  # startcol keyword without value, resolve from JSON/default 0
            else:
                print(f"Error: unexpected argument '{args.startcol_kw}'", file=sys.stderr)
                sys.exit(1)

    tile_type = "shim" if row == 0 else "core"
    aie_version = args.aie_version
    aie_version_source = "explicit"
    dry_run = args.dry_run
    target = args.target
    device = args.device
    json_dir = args.json_dir

    # ── Load JSONs ────────────────────────────────────────────────────────
    dfsche, dmaphop = load_jsons(json_dir)
    shim_events = load_shim_events_json(args.shim_events_json)

    # ── Resolve startcol: explicit flag wins, else from provenance JSON ─────
    startcol_source = "explicit"
    if startcol is None:
        json_sc = startcol_from_jsons(dfsche, dmaphop)
        if json_sc is not None:
            startcol = json_sc
            startcol_source = "json"
        else:
            startcol = 0
            startcol_source = "default"
    phys_col = col + startcol

    # ── Resolve aie_version: explicit flag wins, else from provenance JSON ──
    if aie_version is None:
        json_ver = aie_version_from_jsons(dfsche, dmaphop)
        if json_ver is not None:
            aie_version = json_ver
            aie_version_source = "from provenance JSON"
        else:
            aie_version = "5"
            aie_version_source = "default"

    # ── Header ────────────────────────────────────────────────────────────
    print(bold("=" * 58))
    print(bold(" aiediag: DMA Diagnostic"))
    print(bold("=" * 58))
    print(f"Target tile: ({phys_col},{row})  [logical: col={col}, row={row}, startcol={startcol} ({startcol_source})]")
    print(f"Channel: {direction.upper()} ch{channel}")
    print(f"Tile type: {tile_type}, AIE version: {aie_version} ({aie_version_source})")
    print()

    # ── Step 1: Read DMA status of queried tile ───────────────────────────
    offset = compute_reg_offset(tile_type, direction, channel, aie_version)
    print(bold(f"--- [1] DMA Status: tile({phys_col},{row}) {direction.upper()} ch{channel} ---"))
    raw = run_aiedbg_reg_read(phys_col, row, offset, target=target, device=device, dry_run=dry_run)
    queried_decoded = None
    if raw is not None:
        queried_decoded = decode_dma_status(raw)
        print(format_dma_status(phys_col, row, direction, channel, queried_decoded))
    if tile_type == "core":
        mem_regs = read_event_status_4(phys_col, row, MEM_EVT_STATUS_REGS,
                                       target, device, dry_run)
        if mem_regs is not None:
            print(format_core_mem_dma_events(phys_col, row, mem_regs))
        core_regs = read_event_status_4(phys_col, row, CORE_EVT_STATUS_REGS,
                                        target, device, dry_run)
        if core_regs is not None:
            print(format_core_module_errors(phys_col, row, core_regs))
        # PC -> source: where is this core stuck? (best-effort, needs -g line map)
        pc_raw = run_aiedbg_reg_read(phys_col, row, CORE_PC_OFFSET,
                                     target=target, device=device, dry_run=dry_run)
        if pc_raw is not None:
            pc_val = pc_raw & CORE_PC_MASK
            entries, lm_path = load_linemap()
            if entries:
                hit = pc_to_source(entries, pc_val)
                if hit:
                    print(green(f"  Core PC=0x{pc_val:05X} -> {hit['file']}:{hit['line']}"))
                else:
                    print(f"  Core PC=0x{pc_val:05X} (no line-map match)")
            else:
                print(f"  Core PC=0x{pc_val:05X} (no {LINEMAP_FILENAME}; build with -g)")
    print()

    # ── Step 2: BD chain from JSON ────────────────────────────────────────
    tile_entry = find_tile_in_json(dfsche, col, row)
    ch_entry = find_channel_in_tile(tile_entry, direction, channel) if tile_entry else None
    flow_index = None

    print(bold(f"--- [2] BD Chain (from JSON) ---"))
    if ch_entry:
        flow_index = ch_entry.get("flow_index")
        print(f"  Flow index: {flow_index} ({ch_entry.get('direction', '').upper()})")
        bd_ids = [bd["bd_id"] for bd in ch_entry.get("bd_chain", [])]
        hw_lengths = read_bd_hw_lengths(phys_col, row, tile_type, aie_version,
                                        bd_ids, target, device, dry_run) if bd_ids else None
        print(format_bd_chain(ch_entry, hw_lengths))
    else:
        if tile_entry:
            avail = [(c["direction"], c["channel"]) for c in tile_entry.get("dma_channels", [])]
            print(f"  Channel {direction.upper()} ch{channel} not found for tile({col},{row})")
            print(f"  Available channels: {avail}")
        else:
            print(f"  Tile ({col},{row}) not found in dfscheduleprovenancemap.json")
    print()

    # ── Step 3: Find connected tiles ──────────────────────────────────────
    flow_entry = find_flow(dfsche, flow_index) if flow_index is not None else None

    print(bold(f"--- [3] Connected tiles (flow {flow_index}) ---"))
    connected = []
    if flow_entry:
        flow_dir = flow_entry.get("direction", "?")
        print(f"  Flow direction: {flow_dir}")
        connected = find_connected_tiles(flow_entry, col, row, direction)
        if connected:
            senders = [c for c in connected if c["io_direction"].upper() == "MM2S"]
            receivers = [c for c in connected if c["io_direction"].upper() == "S2MM"]

            if receivers:
                print("  Receivers:")
                for c in receivers:
                    scf_str = ""
                    if c.get("inside_scf_for"):
                        scf_str = f" (scf.for {c.get('loop_range', '')})"
                    print(f"    tile({c['tile_col']},{c['tile_row']}) "
                          f"{c['io_direction']} ch{c['channel']} "
                          f"repeat={c.get('repeat_count', '?')} "
                          f"bd_len={c.get('bd_len', '?')}B{scf_str}")
            if senders:
                print("  Senders:")
                for c in senders:
                    scf_str = ""
                    if c.get("inside_scf_for"):
                        scf_str = f" (scf.for {c.get('loop_range', '')})"
                    print(f"    tile({c['tile_col']},{c['tile_row']}) "
                          f"{c['io_direction']} ch{c['channel']} "
                          f"repeat={c.get('repeat_count', '?')} "
                          f"bd_len={c.get('bd_len', '?')}B{scf_str}")
        else:
            print("  No other tiles found in this flow")
    else:
        print("  Flow summary not available")
    print()

    # ── Step 4: Read DMA status of connected tiles ────────────────────────
    print(bold(f"--- [4] Connected tile DMA Status ---"))
    connected_results = []
    for c in connected:
        cc = c["tile_col"]
        cr = c["tile_row"]
        c_dir = c["io_direction"].lower()
        c_ch = c["channel"]
        c_type = "shim" if cr == 0 else "core"
        c_phys_col = cc + startcol
        c_offset = compute_reg_offset(c_type, c_dir, c_ch, aie_version)

        print(f"  tile({c_phys_col},{cr}) {c_dir.upper()} ch{c_ch}:")
        c_raw = run_aiedbg_reg_read(c_phys_col, cr, c_offset, target=target, device=device, dry_run=dry_run)
        c_decoded = None
        if c_raw is not None:
            c_decoded = decode_dma_status(c_raw)
            print(format_dma_status(c_phys_col, cr, c_dir, c_ch, c_decoded))
        connected_results.append((cc, cr, c_dir, c_ch, c_decoded))
    if not connected:
        print("  (none)")
    print()

    # ── Step 4b: Shim event status ───────────────────────────────────────
    queried_shim_events = None
    connected_shim_events = {}  # keyed by (col, row, dir, ch)
    if shim_events is not None:
        print(bold(f"--- [4b] Shim Event Status ---"))
        # Queried tile
        if row == 0:
            queried_shim_events = read_shim_event_status(
                phys_col, direction, channel, target, device, dry_run)
            if queried_shim_events:
                print(format_shim_event_status(phys_col, direction, channel,
                                               queried_shim_events, shim_events))
            elif not dry_run:
                print(f"  Could not read shim events for tile({phys_col},0)")
        # Connected shim tiles
        for c in connected:
            cc = c["tile_col"]
            cr = c["tile_row"]
            if cr != 0:
                continue
            c_dir = c["io_direction"].lower()
            c_ch = c["channel"]
            c_phys_col = cc + startcol
            evt = read_shim_event_status(c_phys_col, c_dir, c_ch, target, device, dry_run)
            if evt:
                print(format_shim_event_status(c_phys_col, c_dir, c_ch, evt, shim_events))
                connected_shim_events[(cc, cr, c_dir, c_ch)] = evt
            elif not dry_run:
                print(f"  Could not read shim events for tile({c_phys_col},0)")
        if row != 0 and not any(c["tile_row"] == 0 for c in connected):
            print("  No shim tiles in this flow")
        print()
    else:
        print(bold(f"--- [4b] Shim Event Status ---"))
        print(yellow("  Skipped (shimtile_events.json not loaded)"))
        print()

    # ── Step 5: BD info for connected tiles ───────────────────────────────
    print(bold(f"--- [5] Connected tile BD chains ---"))
    for c in connected:
        cc = c["tile_col"]
        cr = c["tile_row"]
        c_dir = c["io_direction"]
        c_ch = c["channel"]
        c_tile = find_tile_in_json(dfsche, cc, cr)
        c_ch_entry = find_channel_in_tile(c_tile, c_dir.lower(), c_ch) if c_tile else None
        if c_ch_entry:
            print(f"  tile({cc},{cr}) {c_dir} ch{c_ch}:")
            print(format_bd_chain(c_ch_entry))
        else:
            print(f"  tile({cc},{cr}) {c_dir} ch{c_ch}: not found in JSON")
    if not connected:
        print("  (none)")
    print()

    # ── Step 6: Routing path from dmaphopprovenacemap.json ────────────────
    print(bold(f"--- [6] Routing path (from dmaphopprovenacemap.json) ---"))
    paths = find_routing_paths(dmaphop, flow_index, flow_entry)
    if paths:
        for p in paths:
            print(format_routing_path(p))
    else:
        print("  No matching routing path found")
    print()

    # ── Step 7: Diagnosis ─────────────────────────────────────────────────
    print(bold(f"--- [7] Diagnosis ---"))
    diag = diagnose(direction, queried_decoded, connected_results,
                    queried_shim_events=queried_shim_events,
                    connected_shim_events=connected_shim_events,
                    queried_row=row)
    print(diag)
    print()
    print(bold("=" * 58))


if __name__ == "__main__":
    main()
