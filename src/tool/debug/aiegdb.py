#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegdb - a GDB-like scoped CLI frontend over aiedbg for live AIE debug.

A stateful, interactive console that keeps a *current scope*
(partition -> tile -> channel). A `target` command descends a level, e.g.

    partition(startcol=3)> target tile 0 0
    partition(startcol=3)/tile(0,0)> target channel mm2s0
    partition(startcol=3)/tile(0,0)/mm2s0> dma status

Commands default to the current scope; tile scope auto-injects col/row, channel
scope auto-injects col/row + direction/channel. Decoding/offsets come from
aiediag.py (imported as a library); raw register access shells out to `aiedbg`.

Design reference: doc/design/live_debug_framework.md
"""

import argparse
import os
import re
import subprocess
import sys

# Import aiediag as a library (offsets, reg read via aiedbg --json, decoders,
# provenance, startcol). aiediag.py lives in the same dir (src/tool/debug/); it guards
# main() behind __main__, so importing it has no side effects. (Same shim as
# schedule_debug_server.py:48-52.)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import aiediag  # noqa: E402

try:
    import readline  # noqa: F401  (enables line editing + history)
    _HAVE_READLINE = True
except ImportError:
    _HAVE_READLINE = False

HISTORY_FILE = os.path.expanduser("~/.aiegdb_history")

# Framing marker for --server mode: a daemon writes a command to stdin and reads
# stdout until a line starting with this marker, whose remainder is the current
# prompt/scope. The record-separator prefix (\x1e) never appears in normal output.
_SERVER_MARKER = "\x1e@AIEGDB@ "

# ── AIE performance-counter register offsets (AIE2PS) ─────────────────────────
# From thirdparty/alib/aie-rt/driver/src/global/xaie2psgbl_params.h.
# DMA start/finish events live in the MEMORY module on core tiles, which is why
# the runtime debug path counts "MM2S BD finished" via MEM_MOD perf counters
# (src/mlir/runtime/aie_runtime_debug.c:1494,1973).
PERF_OFFSETS = {
    # memory module of a core tile (DMA events counted here)
    "core_mem": {"ctrl0": 0x11000, "cnt": (0x11020, 0x11024)},
    # core (compute) module of a core tile
    "core":     {"ctrl0": 0x37500, "cnt": (0x37520, 0x37524)},
    # PL module of a shim tile (row 0)
    "shim_pl":  {"ctrl0": 0x31000, "cnt": (0x31020, 0x31024)},
}
# CONTROL0 field layout (7-bit event ids for core/mem; PL uses 8-bit but the
# same LSB positions, so a 7-bit read still yields the low 7 bits which cover
# the DMA event ids we care about).
_PERF_FIELDS = {
    "cnt0_start": (0, 0x7F),
    "cnt0_stop":  (8, 0x7F),
    "cnt1_start": (16, 0x7F),
    "cnt1_stop":  (24, 0x7F),
}


def _perf_key(tile_type):
    """Perf-counter register bank used for a tile's *DMA* counters."""
    return "shim_pl" if tile_type == "shim" else "core_mem"


def _reverse_event_ids():
    """Build {event_id: 'S2MM0.FINISHED_TASK'} for core-mem and shim DMA events."""
    core_rev = {}
    for (d, ch), m in aiediag.CORE_MEM_DMA_EVENT_IDS.items():
        for name, eid in m.items():
            core_rev[eid] = f"{d.upper()}{ch}.{name}"
    shim_rev = {}
    for (d, ch), m in aiediag.SHIM_DMA_EVENT_IDS.items():
        for name, eid in m.items():
            shim_rev[eid] = f"{d.upper()}{ch}.{name}"
    return core_rev, shim_rev


_CORE_EVT_REV, _SHIM_EVT_REV = _reverse_event_ids()


def _evt_label(tile_type, eid):
    rev = _SHIM_EVT_REV if tile_type == "shim" else _CORE_EVT_REV
    if eid == 0:
        return "NONE(0)"
    return rev.get(eid, f"evt{eid}")


# ── small parse helpers ───────────────────────────────────────────────────────

_DIR_CH_RE = re.compile(r"^(mm2s|s2mm)(\d+)$", re.IGNORECASE)


def _is_dir_ch(tok):
    return bool(_DIR_CH_RE.match(tok.strip().lstrip("-")))


def _parse_dir_ch(tok):
    """'mm2s0' / '-s2mm1' -> ('mm2s', 0). Returns (None, None) on bad input."""
    m = _DIR_CH_RE.match(tok.strip().lstrip("-"))
    if not m:
        return None, None
    return m.group(1).lower(), int(m.group(2))


def _expand_parens(line):
    """Turn parenthesized forms into space forms: tile(0,0) -> tile 0 0."""
    line = re.sub(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", r" \1 \2", line)
    line = re.sub(r"\(\s*(\d+)\s*\)", r" \1", line)
    return line


# ── the REPL ──────────────────────────────────────────────────────────────────

class AieGdb:
    """Stateful scoped console. Partition config always present; tile/channel
    are pushed/popped by `target`/`up`."""

    def __init__(self, target=None, device="pal", startcol=0, aie_version="5",
                 json_dir=None, dry_run=False):
        # partition scope (always present)
        self.target = target
        self.device = device
        self.startcol = int(startcol) if startcol is not None else 0
        self.aie_version = str(aie_version)
        self.json_dir = json_dir
        self.dry_run = dry_run
        # Hard cap on raw aiedbg passthroughs (scan/show/reg/mem). An array-wide
        # `scan dma` can block forever when a board run is holding the JTAG or
        # the target/hw_server is unresponsive — without a timeout that hangs the
        # caller (e.g. the MCP tool call, freezing the embedded LLM). Override via
        # $AIEGDB_PASSTHROUGH_TIMEOUT.
        try:
            self.passthrough_timeout = float(
                os.environ.get("AIEGDB_PASSTHROUGH_TIMEOUT", "120"))
        except (TypeError, ValueError):
            self.passthrough_timeout = 120.0
        # deeper scopes (None until pushed)
        self.tile = None       # dict {col, row, type}
        self.channel = None    # dict {direction, channel}
        # lazily-loaded provenance
        self._dfsche = None
        self._dmaphop = None
        self._jsons_loaded = False
        self._shim_events = None
        self._shim_loaded = False
        self._tiles = None     # schedule_view.json tiles
        # per-command register-read sink: run_line resets it at the start of
        # each command and flushes it as a light-yellow block at the bottom.
        self._reg_trace = []

    # ---- scope helpers ---------------------------------------------------
    @property
    def phys_col(self):
        if self.tile is None:
            return None
        return self.tile["col"] + self.startcol

    def prompt(self):
        s = f"partition(startcol={self.startcol})"
        if self.tile is not None:
            s += f"/tile({self.tile['col']},{self.tile['row']})"
        if self.channel is not None:
            s += f"/{self.channel['direction']}{self.channel['channel']}"
        return s + "> "

    def _load_jsons(self):
        if not self._jsons_loaded:
            self._dfsche, self._dmaphop = aiediag.load_jsons(self.json_dir)
            self._jsons_loaded = True
        return self._dfsche, self._dmaphop

    def _load_shim_events(self):
        if not self._shim_loaded:
            path = os.path.expanduser("~/aiejson/shimtile_events.json")
            self._shim_events = aiediag.load_shim_events_json(path)
            self._shim_loaded = True
        return self._shim_events

    def _load_tiles(self):
        """Tile list from schedule_view.json (for `channels`)."""
        if self._tiles is not None:
            return self._tiles
        import json
        candidates = []
        if self.json_dir:
            candidates.append(os.path.join(self.json_dir, "schedule_view.json"))
        candidates += [
            "./worklocal/schedule_view.json",
            "./aout/worklocal/schedule_view.json",
        ]
        for p in candidates:
            if os.path.isfile(p):
                with open(p) as f:
                    self._tiles = json.load(f).get("tiles", [])
                return self._tiles
        self._tiles = []
        return self._tiles

    # ---- aiedbg backends -------------------------------------------------
    def _passthrough(self, args):
        """Run a verbatim aiedbg command (raw passthrough). Echoes stdout."""
        cmd = ["aiedbg", "-d", self.device]
        if self.target:
            cmd += ["--target", self.target]
        cmd += [str(a) for a in args]
        if self.dry_run:
            print("  [dry-run] would execute: " + " ".join(cmd))
            return
        try:
            subprocess.run(cmd, timeout=self.passthrough_timeout)
        except FileNotFoundError:
            print("Error: 'aiedbg' not found in PATH", file=sys.stderr)
        except subprocess.TimeoutExpired:
            # subprocess.run already SIGKILLed the child before re-raising.
            print("Error: aiedbg timed out after %gs (%s). The JTAG link may be "
                  "blocked \u2014 a board run can hold it, or the target/"
                  "hw_server is unresponsive. Prefer a scoped read (e.g. "
                  "'target tile C R' then 'dma status') over an array-wide "
                  "'scan'." % (self.passthrough_timeout, " ".join(cmd[3:])),
                  file=sys.stderr)

    def _reg_read(self, phys_col, row, off):
        """Decoded-path register read via aiediag (respects dry_run).

        Collects the raw address + raw value into the per-command sink so
        run_line can flush a light-yellow block at the bottom of the output."""
        return aiediag.run_aiedbg_reg_read(
            phys_col, row, off, target=self.target, device=self.device,
            dry_run=self.dry_run, sink=self._reg_trace)

    # ---- dispatch --------------------------------------------------------
    def run_line(self, line):
        """Run one command, then flush any collected register reads as a
        light-yellow brace block at the bottom of the output.

        The per-command sink is reset here so each command shows only its own
        reads. Passthrough commands (reg/mem/scan) run aiedbg uncaptured and
        print addresses natively, so they don't feed the sink."""
        self._reg_trace = []
        try:
            return self._dispatch(line)
        finally:
            block = aiediag.format_reg_read_block(self._reg_trace)
            if block:
                print(block)

    def _dispatch(self, line):
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        line = _expand_parens(line)
        parts = line.split()
        verb = parts[0].lower()
        args = parts[1:]

        # universal commands (available in every scope)
        if verb in ("exit", "quit", "q"):
            return False
        if verb == "help":
            self._help()
            return True
        if verb in ("?", "commands", "cmds"):
            self._commands()
            return True
        if verb in ("where", "info", "pwd"):
            self._where()
            return True
        if verb in ("up", ".."):
            self._up()
            return True
        if verb == "top":
            self.tile = None
            self.channel = None
            return True
        if verb in ("target", "tar"):
            self._target(args)
            return True
        if verb == "set":
            self._set(args)
            return True

        # bare navigation shortcuts
        if verb == "tile":
            # `tile list` is an aiedbg passthrough; `tile 0 0` is navigation.
            if len(args) >= 2 and args[0].lstrip("-").isdigit():
                self._target(["tile"] + args)
                return True
            if len(args) == 1 and args[0].lower() == "list":
                self._scope_passthrough(parts)
                return True
            self._target(["tile"] + args)
            return True
        if verb == "channel":
            self._target(["channel"] + args)
            return True
        if _is_dir_ch(verb) and not args:
            # bare `mm2s0` selects a channel
            self._target(["channel", verb])
            return True

        # scope-specific handling
        if self.channel is not None:
            self._channel_cmd(verb, args, parts)
        elif self.tile is not None:
            self._tile_cmd(verb, args, parts)
        else:
            self._partition_cmd(verb, args, parts)
        return True

    # ---- navigation ------------------------------------------------------
    def _target(self, args):
        if not args:
            print("usage: target partition|tile <c> <r>|channel <dir_ch>")
            return
        obj = args[0].lower()
        rest = args[1:]
        if obj == "partition":
            self._config_partition(rest)
            self.tile = None
            self.channel = None
            print(f"scope -> {self.prompt().rstrip()}")
            return
        if obj == "tile":
            if len(rest) < 2 or not (rest[0].lstrip("-").isdigit()
                                     and rest[1].lstrip("-").isdigit()):
                print("usage: target tile <col> <row>   (e.g. tar tile(0,0))")
                return
            col, row = int(rest[0]), int(rest[1])
            self.tile = {"col": col, "row": row,
                         "type": "shim" if row == 0 else "core"}
            self.channel = None
            print(f"scope -> {self.prompt().rstrip()}  "
                  f"(phys_col={self.phys_col}, row={row})")
            return
        if obj == "channel":
            if not rest:
                print("usage: target channel <dir_ch>   (e.g. mm2s0, s2mm1)")
                return
            self._push_channel(rest[0])
            return
        if _is_dir_ch(obj):
            self._push_channel(obj)
            return
        print(f"target: unknown object {obj!r} "
              f"(partition|tile|channel|<dir_ch>)")

    def _push_channel(self, tok):
        if self.tile is None:
            print("error: select a tile first (e.g. `target tile 0 0`)")
            return
        d, c = _parse_dir_ch(tok)
        if d is None:
            print(f"error: bad channel {tok!r}; expected mm2s0/s2mm1")
            return
        self.channel = {"direction": d, "channel": c}
        print(f"scope -> {self.prompt().rstrip()}")

    def _config_partition(self, kwargs):
        for kv in kwargs:
            key, _, val = kv.partition("=")
            key = key.lower()
            if key == "startcol":
                try:
                    self.startcol = int(val)
                except ValueError:
                    print(f"bad startcol: {val}")
            elif key == "target":
                self.target = val or None
            elif key == "device":
                self.device = val or self.device
            elif key in ("aie", "aie-version", "aie_version"):
                self.aie_version = val or self.aie_version
            else:
                print(f"unknown partition option: {kv}")

    def _set(self, args):
        if len(args) < 2:
            print("usage: set startcol N | set target ... | set device ... | "
                  "set aie 5|2ps")
            return
        self._config_partition([f"{args[0]}={args[1]}"])
        print(f"set {args[0]}={args[1]}")

    def _up(self):
        if self.channel is not None:
            self.channel = None
        elif self.tile is not None:
            self.tile = None
        else:
            print("already at partition scope")

    def _where(self):
        print(f"scope:     {self.prompt().rstrip()}")
        print(f"partition: startcol={self.startcol} device={self.device} "
              f"aie={self.aie_version} target={self.target or 'NONE'}")
        if self.tile is not None:
            print(f"tile:      logical col={self.tile['col']} row={self.tile['row']} "
                  f"type={self.tile['type']}  -> phys_col={self.phys_col}")
        if self.channel is not None:
            print(f"channel:   {self.channel['direction'].upper()} "
                  f"ch{self.channel['channel']}")

    # ---- partition scope -------------------------------------------------
    def _partition_cmd(self, verb, args, parts):
        # everything else is a verbatim aiedbg passthrough (explicit coords).
        self._scope_passthrough(parts)

    def _scope_passthrough(self, parts):
        self._passthrough(parts)

    # ---- tile scope ------------------------------------------------------
    def _tile_cmd(self, verb, args, parts):
        col, row = self.tile["col"], self.tile["row"]
        ttype = self.tile["type"]
        pc = self.phys_col

        if verb == "dma" and args:  # `dma <dir_ch>` decoded status
            self._dma_status_for(args[0])
            return
        if verb == "pc":
            self._pc()
            return
        if verb == "status" or (verb == "core" and (not args or args[0] == "status")):
            self._core_status()
            return
        if verb == "event":
            self._tile_events()
            return
        if verb in ("log", "klog"):
            self._klog()
            return
        if verb in ("channels", "chans"):
            self._list_channels()
            return
        if verb == "reg" and args and args[0] in ("read", "write"):
            self._passthrough(["reg", args[0], str(pc), str(row)] + args[1:])
            return
        if verb == "mem" and args:
            self._passthrough(["mem", args[0], str(pc), str(row)] + args[1:])
            return
        # scan / tile list / show / anything else: verbatim (array-wide)
        self._scope_passthrough(parts)

    def _dma_status_for(self, tok):
        d, c = _parse_dir_ch(tok)
        if d is None:
            print(f"error: bad channel {tok!r}")
            return
        ttype = self.tile["type"]
        off = aiediag.compute_reg_offset(ttype, d, c, self.aie_version)
        raw = self._reg_read(self.phys_col, self.tile["row"], off)
        if raw is None:
            return
        dec = aiediag.decode_dma_status(raw)
        print(aiediag.format_dma_status(self.phys_col, self.tile["row"],
                                        d, c, dec))

    def _pc(self):
        if self.tile["row"] == 0:
            print("error: shim tiles (row 0) have no core PC register")
            return
        raw = self._reg_read(self.phys_col, self.tile["row"],
                             aiediag.CORE_PC_OFFSET)
        if raw is None:
            return
        pc = raw & aiediag.CORE_PC_MASK
        entries, lm = aiediag.load_linemap()
        hit = aiediag.pc_to_source(entries, pc) if entries else None
        if hit:
            print(aiediag.green(f"  PC=0x{pc:05X} -> {hit['file']}:{hit['line']}"))
        else:
            print(f"  PC=0x{pc:05X} (no line-map match)")

    def _core_status(self):
        if self.tile["row"] == 0:
            print("error: shim tiles (row 0) have no core status register")
            return
        off = aiediag.core_status_offset(self.aie_version)
        raw = self._reg_read(self.phys_col, self.tile["row"], off)
        if raw is None:
            return
        dec = aiediag.decode_core_status(raw)
        print(aiediag.format_core_status(self.phys_col, self.tile["row"], dec))

    def _klog(self):
        if self.tile["row"] == 0:
            print("error: shim tiles (row 0) have no core klog buffer")
            return
        res = aiediag.read_klog(self.phys_col, self.tile["row"],
                                self.target, self.device, self.dry_run)
        if res is None:
            print(f"  klog read failed for tile({self.phys_col},{self.tile['row']})")
            return
        wi, words = res
        print(aiediag.format_klog(self.phys_col, self.tile["row"], wi, words))

    def _tile_events(self):
        ttype = self.tile["type"]
        pc, row = self.phys_col, self.tile["row"]
        if ttype == "shim":
            # _reg_read collects addr+value into the sink for each read.
            for off in (aiediag.SHIM_EVT_STATUS_REG0,
                        aiediag.SHIM_EVT_STATUS_REG1):
                self._reg_read(pc, row, off)
        else:
            regs4 = [self._reg_read(pc, row, off)
                     for off in aiediag.MEM_EVT_STATUS_REGS]
            if not all(v is None for v in regs4):
                print(aiediag.format_core_mem_dma_events(pc, row, regs4))

    def _list_channels(self):
        col, row = self.tile["col"], self.tile["row"]
        tiles = self._load_tiles()
        found = False
        for t in tiles:
            if t.get("loc", [None, None])[:2] == [col, row]:
                found = True
                chans = t.get("dma_channels", [])
                if not chans:
                    print("  (no dma channels)")
                for ch in chans:
                    print(f"  {ch.get('direction', '?')}{ch.get('channel', '?')}"
                          f"  flow_index={ch.get('flow_index')}")
                break
        if not found:
            print(f"  tile ({col},{row}) not found in schedule_view.json")

    # ---- channel scope ---------------------------------------------------
    def _channel_cmd(self, verb, args, parts):
        d = self.channel["direction"]
        c = self.channel["channel"]
        ttype = self.tile["type"]
        pc, row = self.phys_col, self.tile["row"]

        if verb in ("status",) or (verb == "dma" and args and args[0] == "status"):
            self._dma_status_for(f"{d}{c}")
            return
        if verb == "dma" and args and args[0] in ("counter",):
            self._dma_counter(args[1:])
            return
        if verb == "counter":
            self._dma_counter(args)
            return
        if verb == "dma":  # bare `dma` -> status
            self._dma_status_for(f"{d}{c}")
            return
        if verb == "bd":
            self._bd()
            return
        if verb == "event":
            self._channel_events()
            return
        if verb in ("log", "klog"):
            self._klog()
            return
        if verb == "reg" and args and args[0] in ("read", "write"):
            self._passthrough(["reg", args[0], str(pc), str(row)] + args[1:])
            return
        if verb == "mem" and args:
            self._passthrough(["mem", args[0], str(pc), str(row)] + args[1:])
            return
        self._scope_passthrough(parts)

    def _bd(self):
        d = self.channel["direction"]
        c = self.channel["channel"]
        col, row = self.tile["col"], self.tile["row"]
        dfsche, _ = self._load_jsons()
        tile = aiediag.find_tile_in_json(dfsche, col, row)
        ch = aiediag.find_channel_in_tile(tile, d, c) if tile else None
        if ch is None:
            print(f"  channel {d.upper()} ch{c} not found for tile({col},{row}) "
                  f"in provenance JSON")
            return
        bd_ids = [bd["bd_id"] for bd in ch.get("bd_chain", [])]
        if bd_ids:
            mask = aiediag.BD_LEN_MASK[aiediag._bd_type_key(
                self.tile["type"], self.aie_version)]
            hw = {}
            for bd_id in bd_ids:
                off = aiediag.bd_length_offset(
                    self.tile["type"], self.aie_version, bd_id)
                word0 = self._reg_read(self.phys_col, row, off)
                hw[bd_id] = None if word0 is None else (word0 & mask) * 4
            if all(v is None for v in hw.values()):
                hw = None
        else:
            hw = None
        print(aiediag.format_bd_chain(ch, hw))

    def _channel_events(self):
        d = self.channel["direction"]
        c = self.channel["channel"]
        ttype = self.tile["type"]
        pc, row = self.phys_col, self.tile["row"]
        if ttype == "shim":
            shim_ev = self._load_shim_events()
            reg0 = self._reg_read(pc, 0, aiediag.SHIM_EVT_STATUS_REG0)
            reg1 = self._reg_read(pc, 0, aiediag.SHIM_EVT_STATUS_REG1)
            bits = aiediag.decode_shim_event_status(d, c, reg0, reg1)
            if bits:
                print(aiediag.format_shim_event_status(pc, d, c, bits, shim_ev))
        else:
            emap = aiediag.CORE_MEM_DMA_EVENT_IDS.get((d, c))
            if emap is None:
                print(f"  no event map for {d}{c}")
                return
            regs4 = [self._reg_read(pc, row, off)
                     for off in aiediag.MEM_EVT_STATUS_REGS]
            if all(v is None for v in regs4):
                return
            started = aiediag._evt_active(emap["START_TASK"], regs4)
            finished_bd = aiediag._evt_active(emap["FINISHED_BD"], regs4)
            finished = aiediag._evt_active(emap["FINISHED_TASK"], regs4)
            error = aiediag._evt_active(emap["ERROR"], regs4)
            print(f"  {d.upper()} ch{c} events tile({pc},{row}):")
            print(f"    START_TASK:    {'SET' if started else 'not set'}")
            print(f"    FINISHED_BD:   {'SET' if finished_bd else 'not set'}")
            print(f"    FINISHED_TASK: {'SET' if finished else 'not set'}")
            print(f"    ERROR:         {'SET' if error else 'not set'}")
            if error:
                print(aiediag.red("    >> DMA ERROR event active"))
            elif started and finished:
                print(aiediag.green("    >> started and finished (completed)"))
            elif started and finished_bd:
                print(aiediag.yellow("    >> started, some BD(s) finished, but "
                                     "task never finished (still running or stuck)"))
            elif started:
                print(aiediag.yellow("    >> started but never finished (stuck)"))
            else:
                print(aiediag.yellow("    >> never started (start_io not issued?)"))

    # ---- performance counters (dma counter) ------------------------------
    def _dma_counter(self, args):
        """Read (default) or set up AIE performance counters for this channel's
        DMA. Read-only unless `setup` is given (an explicit write)."""
        if args and args[0].lower() == "setup":
            self._dma_counter_setup(args[1:])
            return
        d = self.channel["direction"]
        c = self.channel["channel"]
        ttype = self.tile["type"]
        pc, row = self.phys_col, self.tile["row"]
        bank = PERF_OFFSETS[_perf_key(ttype)]
        print(f"  Perf counters ({_perf_key(ttype)}) tile({pc},{row}) "
              f"for {d.upper()} ch{c}:")
        ctrl = self._reg_read(pc, row, bank["ctrl0"])
        cnt0 = self._reg_read(pc, row, bank["cnt"][0])
        cnt1 = self._reg_read(pc, row, bank["cnt"][1])
        if self.dry_run:
            return
        fields = {}
        if ctrl is not None:
            for name, (lsb, mask) in _PERF_FIELDS.items():
                fields[name] = (ctrl >> lsb) & mask
        self._print_counter("cnt0", cnt0, fields.get("cnt0_start"),
                            fields.get("cnt0_stop"), ttype)
        self._print_counter("cnt1", cnt1, fields.get("cnt1_start"),
                            fields.get("cnt1_stop"), ttype)

    def _print_counter(self, label, val, start, stop, ttype):
        if start is None:
            print(f"    {label}: value={'?' if val is None else val} "
                  f"(control0 unread)")
            return
        if start == 0 and stop == 0:
            print(f"    {label}: value={val}  (inactive: start==stop==0)")
            return
        s_lbl = _evt_label(ttype, start)
        e_lbl = _evt_label(ttype, stop)
        print(f"    {label}: value={val}  (start={s_lbl} stop={e_lbl})")

    def _dma_counter_setup(self, args):
        """INTRUSIVE: program CNT0 to count this channel's FINISHED_TASK (default)
        or START_TASK event via a reg write to CONTROL0."""
        d = self.channel["direction"]
        c = self.channel["channel"]
        ttype = self.tile["type"]
        pc, row = self.phys_col, self.tile["row"]
        which = (args[0].lower() if args else "finished")
        evt_name = "START_TASK" if which.startswith("start") else "FINISHED_TASK"
        if ttype == "shim":
            emap = aiediag.SHIM_DMA_EVENT_IDS.get((d, c), {})
        else:
            emap = aiediag.CORE_MEM_DMA_EVENT_IDS.get((d, c), {})
        eid = emap.get(evt_name)
        if eid is None:
            print(f"  no {evt_name} event id for {d}{c} on {ttype}")
            return
        bank = PERF_OFFSETS[_perf_key(ttype)]
        # CNT0_START = CNT0_STOP = eid so the counter increments on each event.
        ctrl_val = (eid & 0x7F) | ((eid & 0x7F) << 8)
        print(aiediag.yellow(
            f"  [intrusive] programming CONTROL0=0x{ctrl_val:08X} "
            f"(CNT0 start/stop={evt_name}={eid})"))
        self._passthrough(["reg", "write", str(pc), str(row),
                           f"0x{bank['ctrl0']:X}", f"0x{ctrl_val:X}"])
        val = self._reg_read(pc, row, bank["cnt"][0])
        if not self.dry_run:
            print(f"  cnt0 value now: {val}")

    # ---- help ------------------------------------------------------------
    def _help(self):
        print("Navigation (any scope):")
        print("  target partition [startcol=N] [target=..] [device=..] [aie=5|2ps]")
        print("  target tile <col> <row>   (or: tile 0 0 / tar tile(0,0))")
        print("  target channel <dir_ch>   (or: channel mm2s0 / tar s2mm1)")
        print("  up | ..    pop one level      top    go to partition")
        print("  where|info breadcrumb + phys_col/row + target/device")
        print("  set startcol N | set target .. | set device .. | set aie ..")
        print("  ? | commands  list only the current-level commands")
        print("  help | exit | quit")
        if self.channel is not None:
            print("Channel scope:")
            print("  dma status | status     decoded DMA status register")
            print("  bd                      BD chain (JSON) + live HW lengths")
            print("  event                   per-channel DMA start/finish/error")
            print("  dma counter | counter   AIE perf counters (read-only)")
            print("  dma counter setup [finished|started]   (INTRUSIVE write)")
            print("  log | klog              dump kernel klog buffer (core tiles only)")
            print("  reg read OFF            passthrough (col/row auto-filled)")
        elif self.tile is not None:
            print("Tile scope (col/row auto-injected):")
            print("  dma <dir_ch>            decoded DMA status")
            print("  pc                      core PC -> source (linemap)")
            print("  status | core [status]  decoded core status (enable/reset/stall)")
            print("  event                   event-status regs (decoded)")
            print("  log | klog              dump kernel klog buffer (core tiles only)")
            print("  channels | chans        list this tile's channels")
            print("  reg read OFF | reg write OFF VAL | mem read ADDR LEN")
            print("  scan .. | tile list | show ..   (array-wide passthrough)")
        else:
            print("Partition scope (explicit col/row):")
            print("  reg read C R OFF | reg write C R OFF VAL | mem read ..")
            print("  scan dma|cores | tile list | show ..  (aiedbg passthrough)")

    def _commands(self):
        """List only the commands available at the *current* scope level
        (`?`). Unlike `help`, this omits the universal navigation block and
        shows just what applies here, plus how to descend/ascend."""
        if self.channel is not None:
            print(f"channel scope commands  ({self.prompt().rstrip()}):")
            print("  dma status | status     decoded DMA status register")
            print("  bd                      BD chain (JSON) + live HW lengths")
            print("  event                   per-channel DMA start/finish/error")
            print("  dma counter | counter   AIE perf counters (read-only)")
            print("  dma counter setup [finished|started]   (INTRUSIVE write)")
            print("  log | klog              dump kernel klog buffer (core tiles only)")
            print("  reg read OFF | reg write OFF VAL | mem read ADDR LEN")
            print("  up | ..    -> tile scope     top -> partition")
        elif self.tile is not None:
            print(f"tile scope commands  ({self.prompt().rstrip()}):")
            print("  dma <dir_ch>            decoded DMA status")
            print("  pc                      core PC -> source (linemap)")
            print("  status | core [status]  decoded core status (enable/reset/stall)")
            print("  event                   event-status regs (decoded)")
            print("  log | klog              dump kernel klog buffer (core tiles only)")
            print("  channels | chans        list this tile's channels")
            print("  reg read OFF | reg write OFF VAL | mem read ADDR LEN")
            print("  scan .. | tile list | show ..   (array-wide passthrough)")
            print("  target channel <dir_ch> (or: channel mm2s0) -> descend")
            print("  up | ..    -> partition scope")
        else:
            print(f"partition scope commands  ({self.prompt().rstrip()}):")
            print("  reg read C R OFF | reg write C R OFF VAL | mem read ..")
            print("  scan dma|cores | tile list | show ..  (aiedbg passthrough)")
            print("  target tile <col> <row> (or: tile 0 0) -> descend")
        print("Universal: target|tar | set | where|info | help | exit  "
              "(help = full reference)")

    # ---- loops -----------------------------------------------------------
    def interactive(self):
        if _HAVE_READLINE:
            try:
                readline.read_history_file(HISTORY_FILE)
            except (OSError, FileNotFoundError):
                pass
        print("aiegdb - GDB-like AIE debug console. Type 'help' or 'exit'.")
        print(f"  device={self.device} startcol={self.startcol} "
              f"aie={self.aie_version} target={self.target or 'NONE'}"
              f"{'  [DRY-RUN]' if self.dry_run else ''}")
        try:
            while True:
                try:
                    line = input(self.prompt())
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print("^C")
                    continue
                if not self.run_line(line):
                    break
        finally:
            if _HAVE_READLINE:
                try:
                    readline.write_history_file(HISTORY_FILE)
                except OSError:
                    pass

    def batch(self, commands):
        for line in commands:
            for sub in line.split(";"):
                sub = sub.strip()
                if not sub:
                    continue
                print(f"{self.prompt()}{sub}")
                if not self.run_line(sub):
                    return

    def serve_stdin(self):
        """Framed REPL over stdin/stdout for a long-lived daemon subprocess.

        One command per stdin line; after each, emit a marker line whose
        remainder is the current prompt/scope so the reader knows the command
        finished and can update its breadcrumb. An exception in run_line must
        not kill the server. Exit on EOF or an exit/quit command.
        """
        def emit_marker():
            sys.stdout.write(_SERVER_MARKER + self.prompt().rstrip() + "\n")
            sys.stdout.flush()

        emit_marker()  # initial scope so the daemon learns the starting prompt
        for raw in sys.stdin:
            line = raw.rstrip("\n")
            keep_going = True
            try:
                keep_going = self.run_line(line)
            except Exception as e:  # never let a bad command kill the server
                print(f"error: {e}")
            sys.stdout.flush()
            emit_marker()
            if keep_going is False:
                break


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="aiegdb",
        description="GDB-like scoped CLI frontend over aiedbg for live AIE debug")
    ap.add_argument("--target", default=None,
                    help="aiedbg target xsdb://host:port "
                         "(default: $AIEDBG_TARGET, then ~/.aiedbg_env)")
    ap.add_argument("--device", default="pal", help="aiedbg -d device (default pal)")
    ap.add_argument("--startcol", type=int, default=None,
                    help="physical column offset: phys_col = col + startcol "
                         "(default: read from provenance JSON, else 0)")
    ap.add_argument("--aie-version", default=None,
                    help="AIE version for register offsets; accepts compiler "
                         "numbers (1/2/5) or debug strings (2ps), normalized via "
                         "debug_aie_version_from_gen "
                         "(default: auto-detected from provenance JSON)")
    ap.add_argument("--json-dir", default=None,
                    help="dir with provenance/schedule_view JSONs (auto-detected)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the aiedbg commands instead of running them")
    ap.add_argument("-c", "--command", default=None,
                    help="run ';'-separated commands non-interactively, then exit")
    ap.add_argument("--script", default=None,
                    help="run commands from FILE non-interactively, then exit")
    ap.add_argument("--server", action="store_true",
                    help="framed stdin/stdout REPL for a daemon subprocess "
                         "(one command per line; marker line after each)")
    args = ap.parse_args(argv)

    target = (args.target or os.environ.get("AIEDBG_TARGET")
              or _target_from_aiedbg_env())

    gdb = AieGdb(target=target, device=args.device, startcol=args.startcol,
                 aie_version=args.aie_version, json_dir=args.json_dir,
                 dry_run=args.dry_run)

    # Resolve startcol: explicit --startcol wins, else read from provenance JSON.
    if args.startcol is None:
        gdb._load_jsons()
        gdb.startcol = aiediag.startcol_from_jsons(gdb._dfsche, gdb._dmaphop) or 0
    else:
        gdb.startcol = args.startcol

    # Resolve aie_version (JSON wins so a compiler number from aiehlc.sh cannot
    # select the wrong offset map):
    #   1. provenance JSON aie_gen (authoritative, already a debug string)
    #   2. explicit --aie-version, normalized through the mapper (5 -> 2ps)
    #   3. warn loudly and fall back to a default
    gdb._load_jsons()
    ver = aiediag.aie_version_from_jsons(gdb._dfsche, gdb._dmaphop)
    if ver is None and args.aie_version is not None:
        ver = aiediag.debug_aie_version_from_gen(args.aie_version)
    if ver is None:
        ver = "2ps"
        print(f"warning: could not resolve aie_version from provenance JSON or "
              f"--aie-version flag; falling back to '{ver}'. Register reads may "
              f"be wrong.", file=sys.stderr)
    gdb.aie_version = ver

    if args.server:
        gdb.serve_stdin()
        return
    if args.command:
        gdb.batch([args.command])
        return
    if args.script:
        with open(args.script) as f:
            gdb.batch(f.readlines())
        return
    gdb.interactive()


def _target_from_aiedbg_env():
    """Read AIEDBG_TARGET from ~/.aiedbg_env (mirrors schedule_debug_server.py:690)."""
    path = os.path.expanduser("~/.aiedbg_env")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, val = line.partition("=")
                if key.strip() == "AIEDBG_TARGET":
                    return val.strip().strip('"').strip("'") or None
    except OSError:
        pass
    return None


if __name__ == "__main__":
    main()
