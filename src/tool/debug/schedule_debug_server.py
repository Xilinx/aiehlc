#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""schedule_debug_server - live AIE debug/test daemon behind host_schedule.html.

Bridges the (offline-capable) static schedule viewer to the runtime tools so a
browser can deploy the ELF, watch per-tile DMA/core/event status change in near
real time, and issue read-only per-tile debug commands.

Tiers (all on localhost):

    Browser: host_schedule.html (enhanced)
        |  HTTP fetch / periodic poll (stdlib http.server, 127.0.0.1)
        v
    This daemon  -- imports aiediag.py (offsets, reg read via aiedbg --json,
                    decoders, provenance, startcol); spawns apppaltest.py
        |  subprocess: aiedbg --json reg read ... ; apppaltest.py -nonreboot
        v
    hw_server @ xxx.xxx.xxx.213:3121 (JTAG)  <->  board (AIE array)

The daemon issues READ-ONLY hardware ops only (never stop/con/writes), so it
cannot disturb a running target while apppaltest (-nonreboot) holds the primary
xsdb session on the same JTAG bridge.

Design reference: doc/design/live_debug_framework.md
"""

import argparse
import errno
import getpass
import hmac
import json
import os
import pwd
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Import aiediag as a library (offsets, reg read, decoders, provenance, startcol).
# aiediag.py lives in the same directory (src/tool/debug/); it guards main()
# behind __main__, so importing it has no side effects. aiegdb is imported for
# its _SERVER_MARKER framing constant (also guarded behind __main__).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# This file lives at src/tool/debug/ → repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))

# ── Simulator IPC debug socket helpers ───────────────────────────────────────
# The aeg_ipc_server opens a second read-only socket at <main_socket>.dbg that
# accepts unlimited concurrent connections and handles only READ32/NPI_READ32.
# Wire format: 16-byte request {u8 cmd, u8[3] rsvd, u64 arg1, u32 arg2} (LE),
#              16-byte response {u8 status, u8[7] rsvd, u64 value} (LE).

import struct as _struct

_IPC_READ32     = 0x11
_IPC_NPI_READ32 = 0x13
_IPC_STATUS_OK  = 0x00
_IPC_REQ_FMT   = "<BxxxQI"    # 1+3+8+4 = 16 bytes (packed)
_IPC_RESP_FMT  = "<BxxxxxxxQ" # 1+7+8 = 16 bytes (packed)
_IPC_REQ_SIZE  = _struct.calcsize(_IPC_REQ_FMT)   # 16
_IPC_RESP_SIZE = _struct.calcsize(_IPC_RESP_FMT)  # 16


def _ipc_recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def sim_ipc_ping(dbg_socket_path):
    """Return True if the debug socket is reachable (quick connect + close)."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(dbg_socket_path)
        s.close()
        return True
    except OSError:
        return False


def sim_ipc_read32(dbg_socket_path, addr):
    """Send one READ32 over the debug socket; return the 32-bit value or None."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(dbg_socket_path)
        req = _struct.pack(_IPC_REQ_FMT, _IPC_READ32, addr, 0)
        s.sendall(req)
        raw = _ipc_recvall(s, _IPC_RESP_SIZE)
        s.close()
        if raw is None or len(raw) < _IPC_RESP_SIZE:
            return None
        status, value = _struct.unpack(_IPC_RESP_FMT, raw)
        return int(value) if status == _IPC_STATUS_OK else None
    except OSError:
        return None


def _sim_tile_addr(phys_col, row, offset, base_addr, col_shift, row_shift):
    """Compute the absolute AIE register address from tile coordinates."""
    return base_addr + (phys_col << col_shift) + (row << row_shift) + offset


def _load_aie_addr_params(example_dir):
    """Return (base_addr, col_shift, row_shift) from aie_control_config.json,
    or None if the file is absent or malformed.
    example_dir: the AIE example directory (parent of Work/)."""
    cfg_path = os.path.join(example_dir, "Work", "ps", "c_rts",
                            "aie_control_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            d = json.load(f)
        dc = d["aie_metadata"]["driver_config"]
        return (int(dc["base_address"]),
                int(dc["column_shift"]),
                int(dc["row_shift"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_ui_config(workdir):
    """Load debug_ui_config.json from workdir if it exists.

    This file is written by run_debug_ui.sh (or equivalent) and describes
    extra devices (e.g. the simulator) that this server can run.  If the
    file is absent the server still works — it just has no extra devices.

    Schema:
      {
        "extra_devices": [
          {
            "value": "simulator",
            "label": "Simulator",
            "sim_script": "/abs/path/to/runsim_ipc.sh",
            "sim_example_dir": "/abs/path/to/example"
          },
          {
            "value": "vek385",
            "label": "VEK385",
            "hw_run_script": "/abs/path/to/runhw_vek385.py",
            "hw_env": {
              "VEK385IP": "portobello13",
              "USERNAME": "bkirinci",
              "VEK385PDI": "/home/bkirinci/naiebaremetal/vek385.BIN",
              "AIEDBG_TARGET": "xsdb://portobello13:3121"
            }
          }
        ]
      }
    """
    path = os.path.join(workdir, "debug_ui_config.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_default_elf(workdir):
    """Pick the project's default tiling ELF when --elf is omitted.

    The tiling build emits the host ELF to <workdir>/build/host and copies it to
    aout/main.elf (see script/aiehlc.sh). Probe those canonical locations so the
    UI deploys the freshly-built ELF instead of falling back to apppaltest's own
    default. Returns an absolute path, or None if nothing is found.
    """
    candidates = [
        os.path.join(workdir, "build", "host"),
        os.path.join(_REPO_ROOT, "aout", "main.elf"),
        os.path.join(_REPO_ROOT, "aout", "worklocal", "build", "host"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import aiediag  # noqa: E402
import aiegdb  # noqa: E402

# pexpect drives the interactive ssh -> systest -> xsdb -> hw_server recovery
# session (see DebugState.start_hwserver_async). Optional: without it the auto-start
# path degrades to the manual "start hw_server" hint.
try:
    import pexpect  # noqa: E402
except ImportError:
    pexpect = None

# Vitis settings + xsdb fallback for the hw_server-launch recovery path, kept in
# sync with script/test/apppaltest.py and script/test/connecttest.py.
VITIS_SETTINGS = ("/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/"
                  "Vitis/settings64.sh")
XSDB_ALT_PATH = ("/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/"
                 "Vitis/bin/xsdb")


def _strip_ansi(s):
    """Defensive strip of ANSI SGR color codes (piped stdout is not a tty, so
    aiediag emits none, but keep captured console output clean regardless)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


# ── op whitelist for /cmd (no arbitrary shell) ────────────────────────────────
_CMD_OPS = {"dma", "core", "event", "pc", "reg", "chans", "chanevent"}


def _parse_dir_ch(dir_ch):
    """'mm2s0' / 's2mm1' -> ('mm2s', 0). Returns (None, None) on bad input."""
    if not dir_ch:
        return None, None
    s = dir_ch.strip().lower().lstrip("-")
    for d in ("mm2s", "s2mm"):
        if s.startswith(d):
            tail = s[len(d):]
            if tail.isdigit():
                return d, int(tail)
    return None, None


class DebugState:
    """Shared server config + live apppaltest run state."""

    def __init__(self, workdir, elf, aie_version, device, target,
                 apppaltest, startcol, applog,
                 claude_bin="claude", claude_cwd=None, claude_model=None,
                 llm_enabled=True, llm_password=None):
        self.workdir = os.path.abspath(workdir)
        self.elf = elf
        self.aie_version = str(aie_version)
        self.device = device
        self.target = target
        self.apppaltest = apppaltest
        self.startcol = int(startcol)
        self.applog = os.path.abspath(applog)

        # Cached schedule tiles from schedule_view.json.
        self._tiles = None

        # Live run bookkeeping.
        self._lock = threading.Lock()
        self._run_proc = None         # subprocess.Popen or None
        self._run_fh = None           # open applog file handle (subprocess stdout)
        self._run_id = 0

        # hw_server auto-launch session, started on the connect-failure recovery
        # path (/launch_hwserver). One long-lived pexpect ssh child runs
        # `exec hw_server` on the board; a drain thread keeps it alive.
        self._hwsrv_child = None       # pexpect child or None
        self._hwsrv_buf = []           # captured session output (tail-able)
        self._hwsrv_lock = threading.Lock()
        self._hwsrv_status = "idle"    # idle|starting|connecting|connected|failed
        self._hwsrv_done = True        # False while a launch worker is running
        self._hwsrv_ok = False         # did the post-launch probe connect?
        self._hwsrv_detail = ""        # last probe/launch detail message
        self._hwsrv_target = None      # xsdb://host:port being probed
        self._hwsrv_thread = None      # background launch worker thread

        # Persistent aiegdb REPL subprocess (stateful scope over HTTP).
        self.aiegdb = os.path.join(_THIS_DIR, "aiegdb.py")
        self._gdb_proc = None         # subprocess.Popen or None
        self._gdb_q = None            # queue.Queue fed by a reader thread
        self._gdb_scope = "partition"
        self._gdb_lock = threading.Lock()

        # Persistent Claude Code (headless streaming) subprocess for the LLM tab.
        # Driven exactly like aiegdb: one long-lived process + a reader thread
        # that decodes the stream-json output into a growing text buffer the
        # browser tails via /llm/poll?offset=N (mirrors the applog tail idiom).
        self.claude_bin = claude_bin or "claude"
        self.claude_cwd = os.path.abspath(
            claude_cwd or _REPO_ROOT)  # repo root
        self.claude_model = claude_model
        self.llm_enabled = bool(llm_enabled)
        # Optional password gating the LLM endpoints only. None => auth disabled
        # (backward compatible). Compared against the browser's X-LLM-Auth header.
        self.llm_password = llm_password or None
        self._llm_proc = None          # subprocess.Popen or None
        self._llm_reader_thread = None
        self._llm_buf = ""             # decoded transcript (browser tails this)
        self._llm_active = False       # True while a turn is being generated
        self._llm_lock = threading.Lock()
        self._llm_log_dir = self.workdir  # dir where per-session logs are written
        self._llm_log_fh = None          # open file handle for the current session log
        self._llm_log_in_asst = False    # True after first text delta of a turn
        self._llm_system_prompt_text = None  # prepended to first user turn
        self._llm_first_turn = True      # True until first message is sent

        # Path to the auto-generated MCP config handed to the claude subprocess
        # via --mcp-config (written lazily by _write_mcp_config; cleaned up on
        # shutdown). None => fall back to cwd .mcp.json auto-discovery.
        self._mcp_config_path = None
        # Tracks which backend the current MCP config was written for.
        # When this differs from the actual backend (e.g. sim IPC just became
        # ready), the cached config is invalidated so a fresh one is written.
        self._mcp_config_backend = None  # None | "simulator" | "hardware"

        # Extra devices (e.g. simulator) declared in debug_ui_config.json.
        # Each entry: {"value", "label", "sim_script", "sim_example_dir"}
        ui_cfg = _load_ui_config(self.workdir)
        self._extra_devices = ui_cfg.get("extra_devices", [])
        # Resolve the simulator entry (first extra_device with value=="simulator").
        sim_dev = next((d for d in self._extra_devices
                        if d.get("value") == "simulator"), None)
        self.sim_script = sim_dev.get("sim_script") if sim_dev else None
        self.sim_example_dir = sim_dev.get("sim_example_dir") if sim_dev else None
        self.sim_log = (os.path.join(self.sim_example_dir, "ipc_sim.log")
                        if self.sim_example_dir else None)
        self.sim_applog = (os.path.join(self.sim_example_dir, "ipc_app.log")
                           if self.sim_example_dir else None)
        self._sim_lock = threading.Lock()
        self._sim_proc = None     # subprocess.Popen or None
        self._sim_fh = None       # open log file handle
        self._sim_run_id = 0
        # Path to the debug socket (<main_socket>.dbg) discovered after sim start.
        # Once set, the debug UI uses IPC READ32 for live register reads instead
        # of aiedbg, since the main IPC socket is held exclusively by ipc_app.
        self._sim_dbg_socket = None   # path to *.sock.dbg once discovered
        self._sim_ipc_ready = False   # True once debug socket is reachable
        # AIE address parameters for converting (phys_col, row, offset) → abs addr.
        self._sim_addr_params = (
            _load_aie_addr_params(self.sim_example_dir)
            if self.sim_example_dir else None
        )

    # ---- static data -----------------------------------------------------
    def html_path(self):
        return os.path.join(self.workdir, "host_schedule.html")

    def json_path(self):
        return os.path.join(self.workdir, "schedule_view.json")

    def tiles(self):
        """Return the tile list from schedule_view.json (cached)."""
        if self._tiles is None:
            with open(self.json_path()) as f:
                view = json.load(f)
            self._tiles = view.get("tiles", [])
        return self._tiles

    # ---- run orchestration ----------------------------------------------
    def run_in_progress(self):
        """True while a board run is live. Used to gate aiedbg endpoints so
        their JTAG reads don't collide with apppaltest's device program/reset/
        download over the single serialized xsdb://<PALIP>:3121 link."""
        with self._lock:
            return self._run_proc is not None and self._run_proc.poll() is None

    def run_blocks_debug(self):
        """True only while a live run still holds the JTAG link EXCLUSIVELY, i.e.
        the setup phase (device program / dow -force / rst). Once the run is
        `debuggable` (download done; app running / parked / hung) a second aiedbg
        client can safely read over hw_server, so the console endpoints unblock.

        Used to gate /cmd, /aiegdb, /aiegdb/reload (typed console commands) — but
        NOT /grid or /ping, which drive a background reg-read storm that would
        still compete with the run's own reads."""
        if not self.run_in_progress():
            return False
        try:
            with open(self.applog, "rb") as f:
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return True  # no log yet → assume still in setup
        last_ts = os.path.getmtime(self.applog) if os.path.isfile(self.applog) else 0
        status = self._derive_status(text, True, last_ts)
        return not self._is_debuggable(text, status)

    def start_run(self, device=None, board_host=None):
        """Spawn the board test script -y -nonreboot [elf], output → applog.

        Device routing (in priority order):
          1. If the requested device matches an extra_device entry in
             debug_ui_config.json that has a `hw_run_script` field, that
             script is used.  Any `hw_env` dict in the entry is merged on
             top of os.environ.  board_host (if provided) is set as VEK385IP.
          2. vek385 (built-in): appvek385.py in the same dir as apppaltest;
             requires board_host → VEK385IP.
          3. palmyra (default): apppaltest.py; inherits the daemon's env.

        The elf positional is omitted unless configured (--elf); with -y, the
        script auto-picks its default elf, matching the manual
        `<script>.py -y -nonreboot` invocation.
        """
        device = (device or "palmyra").strip().lower()
        with self._lock:
            if self._run_proc and self._run_proc.poll() is None:
                return {"error": "a run is already in progress",
                        "run_id": self._run_id}

            # ── 1. config-driven hw_run_script from debug_ui_config.json ───
            cfg_dev = next(
                (d for d in self._extra_devices
                 if (d.get("value") or "").strip().lower() == device
                 and d.get("hw_run_script")),
                None)

            env = None
            if cfg_dev:
                script = cfg_dev["hw_run_script"]
                if not os.path.isfile(script):
                    return {"error": f"hw_run_script not found: {script}"}
                env = os.environ.copy()
                # Merge any static env overrides declared in the config.
                for k, v in (cfg_dev.get("hw_env") or {}).items():
                    env[k] = str(v)
                # board_host overrides VEK385IP so the UI host dropdown works.
                if board_host:
                    env["VEK385IP"] = board_host

            # ── 2. built-in vek385 path ────────────────────────────────────
            elif device == "vek385":
                script = os.path.join(os.path.dirname(self.apppaltest),
                                      "appvek385.py")
                if not board_host:
                    return {"error": "vek385 requires a board hostname"}
                if not os.path.isfile(script):
                    return {"error": f"appvek385.py not found: {script}"}
                env = os.environ.copy()
                env["USERNAME"] = getpass.getuser()
                env["VEK385IP"] = board_host

            # ── 3. palmyra fallback ────────────────────────────────────────
            else:
                device = "palmyra"
                script = self.apppaltest

            self._run_id += 1
            run_id = self._run_id
            # -u: unbuffered stdout for realtime tail; -y: auto-confirm ELF pick.
            cmd = [sys.executable, "-u", script, "-y", "-nonreboot"]
            # Config-driven hw_run_script gets its paths from env (VEK385_LOCAL_BIN
            # etc.) — don't append self.elf as a positional arg for those.
            if not cfg_dev:
                if self.elf and not os.path.isfile(self.elf):
                    return {"error": f"ELF not found: {self.elf}"}
                if self.elf:
                    cmd.append(self.elf)
            try:
                fh = open(self.applog, "w")
            except OSError as e:
                return {"error": f"cannot open applog {self.applog}: {e}"}
            fh.write(f"$ {' '.join(cmd)}\n")
            if env and "VEK385IP" in env:
                fh.write(f"# env: USERNAME={env.get('USERNAME', '')} "
                         f"VEK385IP={env['VEK385IP']}\n")
            fh.flush()
            try:
                # start_new_session=True → child leads its own process group so
                # /stop can SIGTERM/SIGKILL the whole tree (the test spawns ssh
                # + pexpect children that would otherwise be orphaned).
                self._run_proc = subprocess.Popen(
                    cmd, stdout=fh, stderr=subprocess.STDOUT,
                    start_new_session=True, env=env)
            except FileNotFoundError as e:
                fh.close()
                return {"error": f"cannot launch {os.path.basename(script)}: {e}"}
            self._run_fh = fh
        return {"run_id": run_id, "applog": self.applog, "device": device}

    def stop_run(self):
        """Force-kill the running apppaltest (and its ssh/pexpect children)."""
        with self._lock:
            proc = self._run_proc
            if proc is None or proc.poll() is not None:
                return {"stopped": False, "error": "no run in progress"}
            pid = proc.pid
            run_id = self._run_id
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # Wait for graceful exit outside the lock (don't block /applog polls),
        # then SIGKILL the group if it's still alive.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        with self._lock:
            if self._run_fh:
                try:
                    self._run_fh.close()
                except OSError:
                    pass
                self._run_fh = None
        # Note the kill in the applog so the live tail reflects it.
        try:
            with open(self.applog, "a") as f:
                f.write(f"\n[force-stop: killed run {run_id} (pid {pid})]\n")
        except OSError:
            pass
        return {"stopped": True, "run_id": run_id, "pid": pid}

    # ---- simulator run orchestration ------------------------------------
    def sim_in_progress(self):
        with self._sim_lock:
            return (self._sim_proc is not None
                    and self._sim_proc.poll() is None)

    def _sim_watch_dbg_socket(self, run_id):
        """Background thread: poll for a *.sock.dbg file in sim_example_dir/ipc/
        (written by aeg_ipc_server when the debug socket is ready), then mark
        the simulator as IPC-ready.  Clears state when the sim process exits."""
        ipc_dir = (os.path.join(self.sim_example_dir, "ipc")
                   if self.sim_example_dir else None)
        deadline = time.time() + 120
        while time.time() < deadline:
            with self._sim_lock:
                proc = self._sim_proc
                current_run = self._sim_run_id
            if current_run != run_id or proc is None or proc.poll() is not None:
                break
            if ipc_dir and os.path.isdir(ipc_dir):
                try:
                    dbg_files = [
                        os.path.join(ipc_dir, f)
                        for f in os.listdir(ipc_dir)
                        if f.endswith(".sock.dbg")
                    ]
                    if dbg_files:
                        dbg_path = dbg_files[0]
                        # Quick probe: connect + disconnect.
                        if sim_ipc_ping(dbg_path):
                            with self._sim_lock:
                                self._sim_dbg_socket = dbg_path
                                self._sim_ipc_ready = True
                            print(f"[sim] IPC debug socket ready → {dbg_path}",
                                  flush=True)
                            self._invalidate_mcp_config()
                            self._write_backend_status()
                            break
                except OSError:
                    pass
            time.sleep(0.5)
        # Wait for sim to exit, then clear IPC state.
        with self._sim_lock:
            proc = self._sim_proc
        if proc is not None:
            proc.wait()
        with self._sim_lock:
            if self._sim_run_id == run_id:
                self._sim_ipc_ready = False
                self._sim_dbg_socket = None
        self._invalidate_mcp_config()
        self._write_backend_status()

    def start_sim(self):
        """Spawn runsim_ipc.sh <sim_example_dir>, stream output to sim_log,
        and start a watcher thread that retargets aiegdb once the ISS port file
        appears."""
        if not self.sim_script:
            return {"error": "simulator not configured (no debug_ui_config.json)"}
        if not os.path.isfile(self.sim_script):
            return {"error": f"sim script not found: {self.sim_script}"}
        if not self.sim_example_dir:
            return {"error": "no sim_example_dir configured"}
        with self._sim_lock:
            if self._sim_proc and self._sim_proc.poll() is None:
                return {"error": "simulator already running",
                        "run_id": self._sim_run_id}
            # Remove stale debug socket files so the watcher doesn't pick up
            # sockets from a previous run.
            ipc_dir = os.path.join(self.sim_example_dir, "ipc")
            if os.path.isdir(ipc_dir):
                for f in os.listdir(ipc_dir):
                    if f.endswith(".sock.dbg"):
                        try:
                            os.unlink(os.path.join(ipc_dir, f))
                        except OSError:
                            pass
            self._sim_ipc_ready = False
            self._sim_dbg_socket = None
            self._sim_run_id += 1
            run_id = self._sim_run_id
            cmd = ["/usr/bin/env", "bash", self.sim_script, self.sim_example_dir]
            # Truncate the PS app log so the new run starts clean.
            if self.sim_applog:
                try:
                    open(self.sim_applog, "w").close()
                except OSError:
                    pass
            try:
                fh = open(self.sim_log, "w")
            except OSError as e:
                return {"error": f"cannot open sim log {self.sim_log}: {e}"}
            fh.write(f"$ {' '.join(cmd)}\n")
            fh.flush()
            try:
                self._sim_proc = subprocess.Popen(
                    cmd, stdout=fh, stderr=subprocess.STDOUT,
                    start_new_session=True)
            except FileNotFoundError as e:
                fh.close()
                return {"error": f"cannot launch sim script: {e}"}
            self._sim_fh = fh
        threading.Thread(target=self._sim_watch_dbg_socket, args=(run_id,),
                         daemon=True).start()
        return {"run_id": run_id, "sim_log": self.sim_log,
                "example_dir": self.sim_example_dir}

    def stop_sim(self):
        """Kill the running simulator and its process group."""
        with self._sim_lock:
            proc = self._sim_proc
            if proc is None or proc.poll() is not None:
                return {"stopped": False, "error": "no simulator running"}
            pid = proc.pid
            run_id = self._sim_run_id
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        with self._sim_lock:
            if self._sim_fh:
                try:
                    self._sim_fh.close()
                except OSError:
                    pass
                self._sim_fh = None
        try:
            with open(self.sim_log, "a") as f:
                f.write(f"\n[force-stop: killed sim run {run_id} (pid {pid})]\n")
        except OSError:
            pass
        with self._sim_lock:
            self._sim_ipc_ready = False
            self._sim_dbg_socket = None
        return {"stopped": True, "run_id": run_id, "pid": pid}

    def sim_status(self):
        """Return current simulator state including IPC debug socket readiness."""
        with self._sim_lock:
            running = (self._sim_proc is not None
                       and self._sim_proc.poll() is None)
            ready = self._sim_ipc_ready
            dbg = self._sim_dbg_socket
        return {"running": running, "ipc_ready": ready, "dbg_socket": dbg}

    def simlog_since(self, offset):
        """Tail the sim log file from byte offset; includes IPC readiness."""
        if not self.sim_log or not os.path.isfile(self.sim_log):
            return {"data": "", "next": 0, "running": False,
                    "ipc_ready": False, "dbg_socket": None}
        with self._sim_lock:
            running = (self._sim_proc is not None
                       and self._sim_proc.poll() is None)
            ipc_ready = self._sim_ipc_ready
            dbg_socket = self._sim_dbg_socket
        with open(self.sim_log, "rb") as f:
            full = f.read()
        chunk = full[offset:]
        data = chunk.decode("utf-8", errors="replace")
        nxt = offset + len(chunk)
        return {"data": data, "next": nxt, "running": running,
                "ipc_ready": ipc_ready, "dbg_socket": dbg_socket}

    def sim_applog_since(self, offset):
        """Tail ipc_app.log (PS application stdout) from byte offset."""
        if not self.sim_applog or not os.path.isfile(self.sim_applog):
            return {"data": "", "next": offset, "running": False}
        with self._sim_lock:
            running = (self._sim_proc is not None
                       and self._sim_proc.poll() is None)
        with open(self.sim_applog, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        data = chunk.decode("utf-8", errors="replace")
        return {"data": data, "next": offset + len(chunk), "running": running}

    def sim_ipc_reg_read(self, phys_col, row, offset):
        """Read a register via the IPC debug socket.  Returns the 32-bit value
        or None if the socket is not ready or the read fails."""
        with self._sim_lock:
            dbg = self._sim_dbg_socket
            ready = self._sim_ipc_ready
            params = self._sim_addr_params
        if not ready or not dbg or not params:
            return None
        base, col_shift, row_shift = params
        addr = _sim_tile_addr(phys_col, row, offset, base, col_shift, row_shift)
        return sim_ipc_read32(dbg, addr)

    # ---- hw_server auto-launch (connect-failure recovery) ----------------
    def _hwsrv_log(self, msg):
        """Append a line to the streamable hw_server session buffer (tailed by
        the browser via /hwsrv_log)."""
        with self._hwsrv_lock:
            self._hwsrv_buf.append(msg)

    def _hwsrv_spawn(self, device, host, logsink):
        """Open ssh -> systest -> [become] -> xsdb and start hw_server on the
        board. Returns the live pexpect child (still running the foreground
        `exec hw_server`, so no xsdb prompt comes back). Raises RuntimeError
        with a human message on any step failure. Each step is echoed to the
        session buffer so the browser can stream live progress.

        Per-device sequence (as specified by the user):
          palmyra: ssh <USERNAME>@<PALIP> -> /bin/systest -> become "<BOARD>"
          vek385 : ssh <host>             -> /proj/systest/bin/systest
        both then: xsdb -> exec hw_server -stcp:0.0.0.0:3121
        """
        device = (device or "").strip().lower()
        if device == "vek385":
            if not host:
                raise RuntimeError("vek385 requires a board hostname")
            ssh_target = host
            systest = "/proj/systest/bin/systest"
            board = None
        else:
            user = os.environ.get("USERNAME") or getpass.getuser()
            palip = os.environ.get("PALIP")
            if not palip:
                raise RuntimeError("PALIP not set in daemon environment")
            ssh_target = f"{user}@{palip}"
            systest = "/bin/systest"
            board = os.environ.get("BOARDNAME", "palmyra")

        self._hwsrv_log(f"[hwserver] ssh -X {ssh_target} ...\n")
        child = pexpect.spawn(f"ssh -X {ssh_target}", encoding="utf-8",
                              timeout=60)
        child.logfile_read = logsink            # stream the raw session output
        child.expect([r'\$\s*$', r'#\s*$', r'>\s*$'], timeout=60)  # shell prompt
        self._hwsrv_log(f"\n[hwserver] launch {systest} ...\n")
        child.sendline(systest)
        child.expect(r'Systest[#>]', timeout=60)
        if board:
            self._hwsrv_log(f"\n[hwserver] become \"{board}\" ...\n")
            child.sendline(f'become "{board}"')
            child.expect(r'Systest[#>]', timeout=60)
        # Resolve xsdb: try PATH, else source Vitis / fall back to the alt path.
        self._hwsrv_log("\n[hwserver] start xsdb ...\n")
        child.sendline("xsdb")
        idx = child.expect([r'xsdb%', r'command not found', r'Unrecognized',
                            pexpect.TIMEOUT], timeout=15)
        if idx != 0:
            self._hwsrv_log("[hwserver] xsdb not on PATH; sourcing Vitis ...\n")
            child.sendline(f"source {VITIS_SETTINGS}")
            child.expect(r'Systest[#>]', timeout=30)
            child.sendline("xsdb")
            j = child.expect([r'xsdb%', pexpect.TIMEOUT], timeout=30)
            if j != 0:
                child.sendline(XSDB_ALT_PATH)
                child.expect(r'xsdb%', timeout=60)
        # Start hw_server. This is a foreground xsdb exec that never returns a
        # prompt, so DON'T expect one — the drain thread consumes its output.
        self._hwsrv_log("\n[hwserver] exec hw_server -stcp:0.0.0.0:3121\n")
        child.sendline("exec hw_server -stcp:0.0.0.0:3121")
        return child

    def start_hwserver_async(self, device, host):
        """Kick off the hw_server launch + single-retry probe in a background
        thread so the browser can tail per-step progress via /hwsrv_log.
        Returns immediately: {started: bool, detail: str}."""
        if pexpect is None:
            return {"started": False,
                    "detail": "pexpect not installed on daemon"}
        with self._hwsrv_lock:
            if not self._hwsrv_done:
                return {"started": False,
                        "detail": "hw_server launch already in progress"}
            # Reset streamable state for a fresh attempt.
            self._hwsrv_buf = []
            self._hwsrv_status = "starting"
            self._hwsrv_done = False
            self._hwsrv_ok = False
            self._hwsrv_detail = ""
            self._hwsrv_target = resolve_target(self, device, host)
            t = threading.Thread(target=self._hwsrv_worker,
                                 args=(device, host), daemon=True)
            self._hwsrv_thread = t
            t.start()
        return {"started": True, "detail": "hw_server launch started"}

    def _hwsrv_worker(self, device, host):
        """Background worker: spawn the ssh/hw_server session (or reuse a live
        one), wait for the port to bind, then probe the JTAG connection once
        (single retry). Updates the streamable status/result throughout."""
        with self._hwsrv_lock:
            existing = self._hwsrv_child
        if existing is not None and existing.isalive():
            self._hwsrv_log("[hwserver] session already running; reusing it.\n")
        else:
            sink = _BufWriter(self._hwsrv_buf, self._hwsrv_lock)
            try:
                child = self._hwsrv_spawn(device, host, sink)
            except Exception as e:  # pexpect timeout / RuntimeError / EOF
                self._hwsrv_log(f"\n[hwserver] launch FAILED: {e}\n")
                with self._hwsrv_lock:
                    self._hwsrv_status = "failed"
                    self._hwsrv_detail = f"launch failed: {e}"
                    self._hwsrv_ok = False
                    self._hwsrv_done = True
                return
            with self._hwsrv_lock:
                self._hwsrv_child = child
            threading.Thread(target=_hwsrv_drain, args=(child,),
                             daemon=True).start()
        # Single retry: give hw_server time to bind, then probe once.
        with self._hwsrv_lock:
            self._hwsrv_status = "connecting"
            tgt = self._hwsrv_target
        self._hwsrv_log("\n[hwserver] waiting 8s for hw_server to bind ...\n")
        time.sleep(8)
        self._hwsrv_log(f"[hwserver] probing xsdb connect {tgt} ...\n")
        ok, detail = run_xsdb_connect(tgt)
        self._hwsrv_log(
            f"[hwserver] {'CONNECTED' if ok else 'STILL FAILED'}: {detail}\n")
        with self._hwsrv_lock:
            self._hwsrv_ok = ok
            self._hwsrv_detail = detail
            self._hwsrv_status = "connected" if ok else "failed"
            self._hwsrv_done = True

    def hwsrv_log_since(self, offset):
        """Tail the hw_server session buffer from char offset; return new text
        + status so the browser can stream launch progress into its console."""
        with self._hwsrv_lock:
            text = "".join(self._hwsrv_buf)
            status, done = self._hwsrv_status, self._hwsrv_done
            ok, detail, target = (self._hwsrv_ok, self._hwsrv_detail,
                                  self._hwsrv_target)
        chunk = text[offset:]
        return {"data": chunk, "next": offset + len(chunk), "status": status,
                "done": done, "ok": ok, "detail": detail, "target": target}

    def applog_since(self, offset):
        """Tail the applog file from byte offset; return new bytes + status.

        `running` reflects the live subprocess (poll()) and is read BEFORE the
        file so that, once the process has exited, the read below still captures
        every trailing byte it wrote (no lost tail). The frontend stops polling
        on `running == false` — NOT on a derived pass/fail — because apppaltest
        keeps writing (summary, cleanup, reboot messages) long after the
        `device_teardown done` marker first appears mid-run.
        """
        if not os.path.isfile(self.applog):
            return {"data": "", "next": 0, "status": "idle", "running": False}
        with self._lock:
            running = self._run_proc is not None and self._run_proc.poll() is None
        with open(self.applog, "rb") as f:
            full = f.read()
        chunk = full[offset:]
        data = chunk.decode("utf-8", errors="replace")
        nxt = offset + len(chunk)
        last_ts = os.path.getmtime(self.applog)
        full_text = full.decode("utf-8", errors="replace")
        status = self._derive_status(full_text, running, last_ts)
        # `debuggable`: the run is past the JTAG-exclusive setup phase (device
        # program / dow -force / rst), so a second aiedbg client can safely read
        # over hw_server. The frontend uses this to auto-unlock the aiegdb console
        # mid-run once the app is downloaded / running / parked / hung.
        return {"data": data, "next": nxt, "status": status, "running": running,
                "debuggable": self._is_debuggable(full_text, status)}

    # Markers appvek385/apppaltest emit AFTER dow -force + rst complete. Seeing
    # any of these means the exclusive-JTAG setup is done and reads are safe.
    _DEBUGGABLE_MARKERS = (
        "ELF download complete!",         # dow -force finished
        "Execution started!",             # con issued, core running
        "board stays powered on for debug",  # -nonreboot parked (incl. PLM-fail)
        "wait_io TIMEOUT",                # app hung waiting on DMA
        "Waiting for console output",     # download done, now polling UART
    )

    @classmethod
    def _is_debuggable(cls, text, status):
        if status in ("hang", "pass", "fail"):
            return True
        return any(m in text for m in cls._DEBUGGABLE_MARKERS)

    @staticmethod
    def _derive_status(text, running, last_ts):
        if "device_teardown done" in text or "Not tearing down partition" in text:
            return "pass"
        if "AIE ERROR" in text:
            return "fail"
        if running:
            # Hang heuristic: no new console output for a while.
            if last_ts and (time.time() - last_ts) > 60:
                return "hang"
            return "running"
        # Process ended without a pass/fail marker.
        return "fail" if text else "idle"

    # ---- aiegdb REPL subprocess ------------------------------------------
    # One long-lived `aiegdb.py --server` process preserves scope (partition ->
    # tile -> channel) across typed commands. Each console command is written to
    # its stdin; stdout is read up to the _SERVER_MARKER framing line. "Reload"
    # kills + respawns it so edits to aiegdb.py/aiediag.py take effect.
    def _gdb_spawn(self):
        """Spawn the aiegdb --server subprocess + a reader thread. Caller holds
        self._gdb_lock. Drains the initial marker to learn the starting scope."""
        cmd = [sys.executable, "-u", self.aiegdb, "--server",
               "--startcol", str(self.startcol), "--device", self.device,
               "--aie-version", str(self.aie_version),
               "--json-dir", self.workdir]
        if self.target:
            cmd += ["--target", self.target]
        self._gdb_proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._gdb_q = queue.Queue()
        proc, q = self._gdb_proc, self._gdb_q

        def _reader():
            for line in proc.stdout:
                q.put(line)
            q.put(None)  # EOF sentinel

        threading.Thread(target=_reader, daemon=True).start()
        # Drain the initial marker line so _gdb_scope reflects the fresh start.
        self._gdb_read_until_marker()

    def _gdb_read_until_marker(self, timeout=60):
        """Accumulate stdout lines until the marker line; update _gdb_scope and
        return the accumulated (non-marker) text. On EOF/timeout return what was
        collected plus an error note and mark the proc dead."""
        lines = []
        q = self._gdb_q
        while True:
            try:
                line = q.get(timeout=timeout)
            except queue.Empty:
                self._gdb_proc = None
                return "".join(lines) + "\n[aiegdb: timed out]\n"
            if line is None:  # process died
                self._gdb_proc = None
                return "".join(lines) + "\n[aiegdb: subprocess exited]\n"
            if line.startswith(aiegdb._SERVER_MARKER):
                scope = line[len(aiegdb._SERVER_MARKER):].rstrip("\n")
                self._gdb_scope = scope.rstrip("> ")
                return "".join(lines)
            lines.append(line)

    def gdb_command(self, cmd):
        """Send one command to the aiegdb REPL; return {output, scope}."""
        with self._gdb_lock:
            if self._gdb_proc is None or self._gdb_proc.poll() is not None:
                self._gdb_spawn()
            try:
                self._gdb_proc.stdin.write(cmd + "\n")
                self._gdb_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._gdb_proc = None
                return {"output": "[aiegdb: pipe broken; retry]",
                        "scope": self._gdb_scope}
            out = self._gdb_read_until_marker()
            return {"output": _strip_ansi(out), "scope": self._gdb_scope}

    def gdb_reload(self):
        """Terminate + respawn the aiegdb subprocess (reloads edited code);
        lands back at partition scope."""
        with self._gdb_lock:
            proc = self._gdb_proc
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                self._gdb_proc = None
            self._gdb_spawn()
            return {"ok": True, "scope": self._gdb_scope}

    def retarget(self, tgt):
        """Point live debug at a new JTAG target and drop the aiegdb REPL so its
        next command respawns against it.

        Called from /settarget after the browser's `Test connect` (/ping) has
        already verified `tgt` is reachable — so this only *switches* the target,
        it does not re-probe. The persistent aiegdb subprocess is spawned once
        with `--target self.target` (_gdb_spawn), so it must be killed for the
        new target to take effect; the next gdb_command lazily respawns it. Also
        invalidates the cached MCP config (which embeds the old target) so the
        LLM tab picks up the new target on its next spawn.
        """
        with self._gdb_lock:
            self.target = tgt or None
            proc = self._gdb_proc
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                self._gdb_proc = None
        self._invalidate_mcp_config()
        # Reset the LLM subprocess so the next claude -p spawn picks up the new
        # MCP config (which embeds AIEDBG_TARGET for the just-selected board).
        if self.llm_enabled:
            self.llm_reset()
        return {"ok": True, "target": self.target}

    # ---- Claude Code (LLM) streaming subprocess --------------------------
    # `claude -p --input-format stream-json --output-format stream-json ...`
    # stays alive across turns (multi-turn conversation). One JSON line per
    # user turn is written to stdin; the reader thread decodes the JSON-line
    # output stream into self._llm_buf, which the browser tails via /llm/poll.
    def llm_auth_required(self):
        """True when the LLM tab is enabled AND a password is configured, i.e.
        the browser must supply X-LLM-Auth to use the LLM endpoints."""
        return bool(self.llm_enabled and self.llm_password)

    def _llm_log_write(self, text):
        """Write text to the session log only (not the browser buffer). Caller holds lock."""
        if self._llm_log_fh:
            try:
                self._llm_log_fh.write(text)
                self._llm_log_fh.flush()
            except OSError:
                pass

    def _llm_append(self, text):
        """Thread-safe append of decoded text to the transcript buffer and log."""
        with self._llm_lock:
            self._llm_buf += text
            if self._llm_log_fh:
                try:
                    self._llm_log_fh.write(text)
                    self._llm_log_fh.flush()
                except OSError:
                    pass

    def _invalidate_mcp_config(self):
        """Delete the cached MCP config file so the next LLM call rewrites it.

        Called when the backend changes (e.g. simulator IPC becomes ready)
        so the new config picks up the current IPC socket path and backend flag.
        """
        old_cfg = self._mcp_config_path
        self._mcp_config_path = None
        self._mcp_config_backend = None
        if old_cfg:
            try:
                os.unlink(old_cfg)
            except OSError:
                pass

    def _write_backend_status(self):
        """Write workdir/backend_status.json with live backend state.

        Called whenever backend changes (sim IPC ready/gone). The debugui MCP
        tool reads this file instead of frozen env vars, so get_backend_status()
        always reflects current state regardless of when the LLM was spawned.
        """
        with self._sim_lock:
            sim_ready = self._sim_ipc_ready
            dbg_socket = self._sim_dbg_socket
        backend = "simulator" if sim_ready else "hardware"
        addr_params = self._sim_addr_params  # (base, col_shift, row_shift) or None
        data = {
            "backend": backend,
            "ipc_ready": sim_ready,
            "dbg_socket": dbg_socket or "",
            "target": self.target or "",
            "device": self.device or "",
            "startcol": self.startcol,
            "aie_version": str(self.aie_version),
            "sim_log": self.sim_log or "",
            "sim_applog": self.sim_applog or "",
            "applog": self.applog,
            "sim_base_addr": str(addr_params[0]) if addr_params else "",
            "sim_col_shift": str(addr_params[1]) if addr_params else "",
            "sim_row_shift": str(addr_params[2]) if addr_params else "",
        }
        path = os.path.join(self.workdir, "backend_status.json")
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            print(f"warning: could not write backend_status.json: {e}", file=sys.stderr)

    def _write_mcp_config(self):
        """Write a temp .mcp.json registering the MCP servers the LLM tab uses,
        with the SAME hardware config the daemon already resolved; return path.

        Two servers, both UI-scoped (this temp config is passed to `claude` via
        --mcp-config --strict-mcp-config, so it does NOT affect general Claude
        Code / CLI sessions that use the repo-root .mcp.json):
          - `aiegdb`  (aiemcp.py)      : LIVE hardware debug (DMA/core/regs).
                                         When AIEMCP_BACKEND=simulator, aiemcp.py
                                         reads registers directly via the IPC debug
                                         socket instead of calling aiedbg.
          - `debugui` (debug_ui_mcp.py): STATIC schedule-view UI data (tile_info,
                                         tile_list, get_backend_status, symbol_search).
        Returns None on failure (caller falls back to cwd discovery).
        """
        # Determine what backend the current call wants to serve.
        with self._sim_lock:
            sim_ready = self._sim_ipc_ready
            dbg_socket = self._sim_dbg_socket
            addr_params = self._sim_addr_params
        want_backend = "simulator" if sim_ready else "hardware"

        # Return cached config if still valid for the same backend.
        if (self._mcp_config_path and os.path.isfile(self._mcp_config_path)
                and self._mcp_config_backend == want_backend):
            return self._mcp_config_path

        # Invalidate stale cached config first.
        self._invalidate_mcp_config()

        aiemcp = os.path.join(_THIS_DIR, "aiemcp.py")
        debugui = os.path.join(_THIS_DIR, "debug_ui_mcp.py")

        # Build aiegdb env — common fields always present.
        aiegdb_env = {
            "AIEDBG_TARGET": self.target or "",
            "AIEMCP_DEVICE": self.device or "pal",
            "AIEMCP_STARTCOL": str(self.startcol),
            "AIEMCP_AIE_VERSION": str(self.aie_version),
            "AIEMCP_JSON_DIR": self.workdir,
            "AIEMCP_BACKEND": want_backend,
        }

        # Simulator-specific: IPC socket path + address layout so aiemcp.py can
        # compute tile addresses and issue READ32 directly, without aiedbg.
        if sim_ready and dbg_socket:
            aiegdb_env["AEG_PS_IPC_DBG_SOCKET"] = dbg_socket
            if addr_params:
                base, col_shift, row_shift = addr_params
                aiegdb_env["AEG_SIM_BASE_ADDR"] = str(base)
                aiegdb_env["AEG_SIM_COL_SHIFT"] = str(col_shift)
                aiegdb_env["AEG_SIM_ROW_SHIFT"] = str(row_shift)

        # debugui env — also expose backend status so get_backend_status works.
        # AIEDBG_TARGET must be set explicitly; without it the subprocess inherits
        # the shell env (e.g. hwlocal.sh) which may point at a different board.
        debugui_env = {
            "DEBUGUI_JSON_DIR": self.workdir,
            "AIEMCP_JSON_DIR": self.workdir,
            "AIEMCP_BACKEND": want_backend,
            "AIEDBG_TARGET": self.target or "",
            "DEBUGUI_APPLOG": self.applog,
            "DEBUGUI_SIM_APPLOG": self.sim_applog or "",
        }
        if sim_ready and dbg_socket:
            debugui_env["AEG_PS_IPC_DBG_SOCKET"] = dbg_socket

        cfg = {
            "mcpServers": {
                "aiegdb": {
                    "command": sys.executable,
                    "args": [aiemcp],
                    "env": aiegdb_env,
                },
                "debugui": {
                    "command": sys.executable,
                    "args": [debugui],
                    "env": debugui_env,
                },
            }
        }
        try:
            fd, path = tempfile.mkstemp(suffix=".mcp.json", prefix="aiegdb_")
            with os.fdopen(fd, "w") as f:
                json.dump(cfg, f, indent=2)
            self._mcp_config_path = path
            self._mcp_config_backend = want_backend
            return path
        except OSError as e:
            print(f"warning: could not write MCP config ({e}); "
                  f"falling back to cwd .mcp.json discovery", file=sys.stderr)
            self._mcp_config_path = None
            self._mcp_config_backend = None
            return None

    def probe_mcp(self, timeout=90):
        """One-shot check that the LLM tab's claude can load + call the aiegdb
        MCP server. Returns (ok: bool, detail: str). Never raises."""
        claude = shutil.which(self.claude_bin) or (
            self.claude_bin if os.path.isfile(self.claude_bin) else None)
        if not claude:
            return False, "claude binary not found"
        cfg = self._write_mcp_config()
        if not cfg:
            return False, "MCP config not written"
        # Prompt via stdin: Claude CLI 2.x treats --allowedTools as variadic
        # and would swallow a trailing positional prompt as extra tool names.
        probe_prompt = ("Call the aiegdb aie_scope tool and reply with only "
                        "its scope line.")
        cmd = [claude, "-p", "--output-format", "json",
               "--permission-mode", "bypassPermissions",
               "--mcp-config", cfg, "--strict-mcp-config",
               "--allowedTools=mcp__aiegdb__aie_scope"]
        if self.claude_model:
            cmd += ["--model", self.claude_model]
        try:
            proc = subprocess.run(cmd, cwd=self.claude_cwd,
                                  input=probe_prompt,
                                  capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"probe timed out after {timeout}s"
        except OSError as e:
            return False, f"probe failed to run: {e}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, (f"claude exited {proc.returncode}: "
                           f"{tail[-1] if tail else '(no output)'}")
        # Success heuristic: the real aie_scope breadcrumb is
        # "partition(startcol=N)...". Key on tokens that only that tool output
        # produces (NOT the word "scope", which the prompt itself contains and
        # the model tends to echo, giving false positives).
        out = proc.stdout or ""
        ok = "partition(" in out or "startcol=" in out
        return ok, ("scope tool responded" if ok else
                    "claude ran but aie_scope output not detected")

    def _llm_system_prompt(self):
        """Build the system prompt describing the debug UI context and available tools."""
        target_str = self.target or "not connected"
        backend_str = "hardware" if self.target else "simulator"
        sim_avail = bool(self.sim_script)

        # Summarise the schedule if loaded.
        view = None
        view_path = os.path.join(self.workdir, "schedule_view.json")
        if os.path.isfile(view_path):
            try:
                with open(view_path) as f:
                    view = json.load(f)
            except (OSError, ValueError):
                view = None

        grid_summary = ""
        if view:
            grid = view.get("grid") or {}
            n_flows = len(view.get("comm_paths") or [])
            n_tiles = len(view.get("tiles") or [])
            grid_summary = (
                f"\nThe loaded design has {n_tiles} tiles "
                f"({grid.get('cols')} cols × {grid.get('rows')} rows, "
                f"startcol={grid.get('startcol')}), {n_flows} communication flows."
            )

        return f"""\
You are an embedded AIE (AI Engine) debug assistant running inside the naiebaremetal \
debug UI — a browser-based schedule viewer and live debug tool for AMD Versal AIE designs.

## Your primary purpose: live AIE debugging

Your job is to help diagnose real failures and performance problems in AIE designs \
running on hardware or the simulator. This means actively reading live register state, \
not just describing the static schedule.

When a user reports a hang, incorrect output, DMA stall, lock contention, core fault, \
or unexpected behaviour, your default response is to **go read the hardware**: use \
`aie_exec` to check DMA status, BD descriptors, lock state, core PC, and event \
registers on the relevant tiles. Cross-reference what the live registers say against \
what the compiled schedule expects, and tell the user concisely what is wrong and why.

Typical debug scenarios you should handle proactively:
- **DMA stall / hang**: check `dma status` on producer and consumer tiles; read BDs \
  to find the stalled descriptor; check lock state; identify whether it is a supply \
  or demand problem.
- **Wrong output / data corruption**: check BD address and length against the schedule; \
  verify byte counters with `dma counter`; check for wrap/offset errors.
- **Core fault / PC stuck**: read `pc` and `event` on the core tile; compare PC to \
  known kernel addresses via `symbol_search`.
- **IPC / stream deadlock**: trace the flow with `get_flow_detail`, then walk each hop \
  checking DMA enables and BD queue depth.
- **Performance**: use BD counters and the supply/demand verdict from `tile_info` to \
  identify the bottleneck stage.

Always prefer concrete register evidence over speculation. If you are not connected \
to hardware or the simulator, say so and explain what you would check once connected.

## Your role (overview)
Help the user understand their compiled AIE design, interpret DMA schedule data, \
diagnose performance issues (supply/demand imbalance, BD chain misconfiguration), \
and control live hardware or simulator sessions. You have direct access to the compiled \
schedule and live register state via MCP tools.

## The debug UI — what the user sees

The browser is divided into two halves:

**Left panel — AIE tile grid (Schedule View)**
A grid of clickable tiles representing the AIE array. Each cell shows:
- Tile type (shim / mem tile / core)
- DMA channel badges (click to select a specific channel)
- Colour coding: contract balance status (green=OK, amber/red=imbalance)
Clicking a tile or channel badge opens its detail panel (right side) and \
pre-loads context for your next message.

**Right panel — tile detail**
Shows three sections (High / Middle / Low) for the selected tile:
- High: role, kernel name, transfer summary, supply/demand verdict, channel↔kernel arg map
- Middle: dfschedule IR slice
- Low: attributed host.cc source lines

**Right panel — Device Map tab**
An SVG map showing the physical AIE array and communication paths (flows). \
Each coloured arc or arrow is one DMA flow (f0, f1, …). Clicking a flow arc \
highlights all tiles it touches and opens a net detail panel. \
The "All nets / f0 / f1 …" chip buttons filter which flows are highlighted.

**Bottom-right pane — three tabs**

1. **aiegdb** — interactive AIE debug console. Type commands like `dma status`, \
`bd 0`, `pc`, `event`, `target tile 4 3`, `target channel S2MM 0`. \
Use `help` for the full command reference.

2. **LLM (this chat)** — you. The user can ask questions; clicking a tile or \
flow first automatically prepends context to the next message you receive.

3. **Search** — symbol search across the compiled design. Type any kernel name, \
window, net ID, buffer symbol, BD length, flow index, or GMIO name. \
Matching tiles and flows are highlighted on the grid.

**Top-right — Run controls**
- Device selector: choose "Simulator" or a hardware board (e.g. VEK385 portobello13)
- Board hostname field (hardware only)
- "Test connect" / "Activate" — probe the JTAG or IPC connection; \
  must succeed before Run/overlay are enabled
- "Run test" — flash BOOT.BIN (hardware) or start the simulator binary; \
  streams stdout to the Run console below
- "Live status overlay" — when checked, polls DMA/core/event registers \
  every 2 s and overlays colour on the grid tiles

**Current session state**
- Working directory: {self.workdir}{grid_summary}
- Debug target: {target_str}
- Backend: {backend_str}{"  (simulator available)" if sim_avail else ""}

## MCP tools available to you

### Static schedule tools (debugui server)
- `get_design_overview()` — grid geometry, all flows, supply/demand summary. \
  **Call this first** in any new session.
- `tile_info(col, row, section)` — full tile detail (high/mid/low). \
  section: "hi" | "mid" | "lo" | "all"
- `tile_list()` — list all tiles with type and role
- `get_flow_detail(flow_index)` — producer/consumer stages, hop routing, \
  stream-switch connections for one flow
- `symbol_search(query, kinds)` — find kernels, windows, buffers, nets, \
  flows, GMIOs by substring. kinds: kernel,window,buffer,contract,bd_len,net,flow,gmio,port
- `get_backend_status()` — current backend, AIEDBG_TARGET, IPC readiness
- `get_applog(lines)` — last N lines of the hardware run log
- `get_sim_log(lines)` — last N lines of the simulator application log (ipc_app.log)
- `get_ipc_log(lines, side)` — recent IPC transaction CSV log; side="client"|"server"|"both"

### Live register tools (aiegdb server — hardware or simulator IPC)
- `aie_scope()` — show current scope (partition/tile/channel)
- `aie_exec(cmd)` — run one aiegdb command. Key commands:
    dma status         — DMA channel enable/disable, BD queue state
    bd <id>            — BD descriptor detail (addr, len, lock, next_bd)
    dma counter        — DMA MM2S/S2MM byte counters
    pc                 — core program counter
    event              — core event status
    target tile C R    — navigate to tile (C, R)
    target channel DIR N — navigate to DMA channel
    up / top           — navigate to parent scope
- `aie_commands()` — list commands valid at current scope
- `aie_help()` — full aiegdb command reference

## PS process and simulator debugging

The simulator run has two processes:
- **ipc_app** (PS client): `{self.sim_example_dir + "/ipc/ipc_app" if self.sim_example_dir else "<sim_example_dir>/ipc/ipc_app"}` \
  Drives graph init, GMIO transfers, and result checks. \
  stdout/stderr → `get_sim_log()` (tails `ipc_app.log`).
- **aiesimulator** (server): Synopsys AIE functional simulator. \
  stdout → `ipc_sim.log` (shown in the Run console). \
  Receives `WRITE32`/`READ32`/`WRITE_GM` etc. from ipc_app over a Unix socket.

### Finding what the simulator run is currently doing

To check whether the simulator is still running and what it last did:
1. Call `get_sim_log(lines=20)` — last 20 lines of `ipc_app.log` (PS app output).
2. Call `get_ipc_log(lines=20)` — last 20 IPC transactions; the final client entry \
   is the transaction ipc_app is currently blocked waiting for a response to.
3. The simulator PID file is at: \
   `{self.sim_example_dir + "/ipc_sim.pid" if self.sim_example_dir else "<sim_example_dir>/ipc_sim.pid"}`

### Inspecting what the PS process is doing (hung/stalled)

When ipc_app appears hung, read its kernel wait state from /proc \
(use `aie_exec` with a bash command, or the aiegdb console Bash tab):

```bash
# Simulator PID:
cat {self.sim_example_dir + "/ipc_sim.pid" if self.sim_example_dir else "<sim_example_dir>/ipc_sim.pid"}

# ipc_app PID (find via socket or binary name):
pgrep -f ipc_app

# What syscall is the main thread blocking on?
cat /proc/<ipc_app_pid>/wchan

# All threads:
for tid in $(ls /proc/<ipc_app_pid>/task/); do echo "=== tid $tid ==="; cat /proc/<ipc_app_pid>/task/$tid/wchan; echo; done

# Stack trace (if perf_event_paranoid allows):
cat /proc/<ipc_app_pid>/task/<tid>/stack
```

Common wchan values:
- `unix_stream_data_wait` — blocked in `recv()` waiting for simulator ack (normal mid-transaction)
- `futex_wait_queue_me` — waiting on a mutex/condvar (normal for idle sim threads)
- `ep_poll` — epoll_wait (normal for sim's socket-dispatch thread)
- `0` — thread running or runnable

### IPC transaction log

Every run writes timestamped CSV logs:
- `{self.sim_example_dir + "/ipc_client.log" if self.sim_example_dir else "ipc_client.log"}` — transactions from ipc_app
- `{self.sim_example_dir + "/ipc_server.log" if self.sim_example_dir else "ipc_server.log"}` — same transactions as dispatched by simulator

CSV columns: `seq, ts_ns, side, cmd, arg1, arg2, status, value, note`

`ts_ns` is CLOCK_MONOTONIC nanoseconds. Delta between client and server rows = round-trip latency.

To find a hang:
1. Call `get_ipc_log(lines=20)` — last client entry is what ipc_app is blocked on.
2. If server log has matching seq with OK but client doesn't → response in flight or socket broken.
3. If neither has it → simulator hasn't dispatched it yet (SystemC scheduling lag).

Key IPC commands:
- `GRAPH_INIT` — loads AIE ELFs + runs graph init (BD programming). Must finish before GMIO.
- `WRITE_GM` / `READ_GM` — GMIO data transfer. arg1=AIE addr, arg2=byte count.
- `WRITE32` / `READ32` — single AIE register r/w (high-frequency during graph init).
- `START_PLIO` — starts PLIO streams after graph init.
- `EXIT` — ipc_app teardown; sim shuts down after ack.

### addr2line for ipc_app

The binary `{self.sim_example_dir + "/ipc/ipc_app" if self.sim_example_dir else "<sim_example_dir>/ipc/ipc_app"}` \
has DWARF debug symbols. Map an address to source:

```bash
addr2line -e {self.sim_example_dir + "/ipc/ipc_app" if self.sim_example_dir else "<sim_example_dir>/ipc/ipc_app"} -f -p 0x<addr>
```

PS app source: `{self.sim_example_dir + "/src/graph.cpp" if self.sim_example_dir else "example/*/src/graph.cpp"}` \
IPC backend: `src/ipc/aeg_ps_ipc_backend.cpp` — `do_transaction()` is the central send/recv loop.

## Suggested workflow

1. Call `get_design_overview()` to orient yourself.
2. When the user clicks a tile, you receive `[context] Selected: tile (C,R) …` — \
   call `tile_info(C, R)` for the full detail.
3. When the user clicks a flow, you receive `[context] Selected net/flow fN …` — \
   call `get_flow_detail(N)` for the routing detail.
4. For live AIE state, check `get_backend_status()` first; if connected, use `aie_exec` \
   commands to read DMA/core registers.
5. If the user has just run and wants to check results, call `get_applog()` or \
   `get_sim_log()` depending on the backend.
6. For a hang or stall, call `get_ipc_log()` to find the last IPC transaction, \
   then use `/proc/<pid>/wchan` inspection via `aie_exec` Bash or the aiegdb console \
   to confirm what each process is waiting on.
"""

    def _llm_backend_context(self):
        """Return a [context] line describing live backend state for injection into
        every LLM message. Reads backend_status.json so it reflects current state."""
        path = os.path.join(self.workdir, "backend_status.json")
        try:
            with open(path) as f:
                s = json.load(f)
        except (OSError, ValueError):
            return ""
        backend = s.get("backend", "unknown")
        ipc_ready = s.get("ipc_ready", False)
        target = s.get("target", "")
        if backend == "simulator":
            state = "IPC ready — live register reads active" if ipc_ready else "IPC not ready — simulator not started yet"
            return f"[context] Backend: simulator ({state})"
        elif backend == "hardware":
            return f"[context] Backend: hardware, target={target or 'unknown'}"
        return f"[context] Backend: {backend}"

    def _llm_spawn(self):
        """Spawn the persistent claude streaming subprocess + reader thread.
        Caller holds self._llm_lock is NOT required (we lock internally)."""
        cmd = [self.claude_bin, "-p",
               "--input-format", "stream-json",
               "--output-format", "stream-json",
               "--include-partial-messages", "--verbose",
               "--permission-mode", "bypassPermissions"]
        # Deterministically bind the aiegdb MCP server (src/tool/debug/aiemcp.py) so the
        # LLM tab connects regardless of cwd; --strict-mcp-config ignores any
        # other .mcp.json and --allowedTools grants the four aiegdb tools.
        mcp_cfg = self._write_mcp_config()
        if mcp_cfg:
            cmd += ["--mcp-config", mcp_cfg, "--strict-mcp-config",
                    "--allowedTools",
                    "mcp__aiegdb__aie_exec", "mcp__aiegdb__aie_scope",
                    "mcp__aiegdb__aie_commands", "mcp__aiegdb__aie_help",
                    "mcp__debugui__tile_info", "mcp__debugui__tile_list",
                    "mcp__debugui__get_backend_status",
                    "mcp__debugui__symbol_search",
                    "mcp__debugui__get_design_overview",
                    "mcp__debugui__get_flow_detail",
                    "mcp__debugui__get_sim_log",
                    "mcp__debugui__get_applog",
                    "mcp__debugui__get_ipc_log"]
        self._llm_system_prompt_text = self._llm_system_prompt()
        self._llm_first_turn = True
        if self.claude_model:
            cmd += ["--model", self.claude_model]
        self._llm_proc = subprocess.Popen(
            cmd, cwd=self.claude_cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        proc = self._llm_proc

        # Open a new per-session transcript log with a timestamp in the filename.
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self._llm_log_dir, f"llm_{ts}.log")
            if self._llm_log_fh:
                try:
                    self._llm_log_fh.close()
                except OSError:
                    pass
            self._llm_log_fh = open(log_path, "w", encoding="utf-8")
            ts_pretty = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fh = self._llm_log_fh
            fh.write(f"[session start {ts_pretty}  workdir={self.workdir}]\n")
            fh.write(f"model: {self.claude_model or '(default)'}\n")
            fh.write(f"target: {self.target or '(none)'}\n\n")
            # Log the MCP config: server names and allowed tools.
            allowed_tools = [
                "mcp__aiegdb__aie_exec", "mcp__aiegdb__aie_scope",
                "mcp__aiegdb__aie_commands", "mcp__aiegdb__aie_help",
                "mcp__debugui__tile_info", "mcp__debugui__tile_list",
                "mcp__debugui__get_backend_status", "mcp__debugui__symbol_search",
                "mcp__debugui__get_design_overview", "mcp__debugui__get_flow_detail",
                "mcp__debugui__get_sim_log", "mcp__debugui__get_applog",
                "mcp__debugui__get_ipc_log",
            ]
            fh.write("=== MCP SERVERS ===\n")
            fh.write("  aiegdb   (aiemcp.py)      — live DMA/core/reg reads\n")
            fh.write("  debugui  (debug_ui_mcp.py) — schedule/flow/tile/log queries\n\n")
            fh.write("=== ALLOWED TOOLS ===\n")
            for tool in allowed_tools:
                fh.write(f"  {tool}\n")
            fh.write("\n")
            if self._llm_system_prompt_text:
                fh.write("=== SYSTEM PROMPT ===\n")
                fh.write(self._llm_system_prompt_text.strip())
                fh.write("\n\n")
            fh.flush()
        except OSError as e:
            print(f"warning: cannot open LLM transcript log: {e}", file=sys.stderr)
            self._llm_log_fh = None

        def _reader():
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue  # skip non-JSON banner/log lines
                try:
                    self._llm_handle_event(obj)
                except Exception as e:
                    print(f"warning: _llm_handle_event error: {e}", file=sys.stderr)
            # stdout closed → process exited; end any in-flight turn.
            with self._llm_lock:
                self._llm_active = False

        self._llm_reader_thread = threading.Thread(target=_reader, daemon=True)
        self._llm_reader_thread.start()

    def _llm_handle_event(self, obj):
        """Decode one stream-json event into readable transcript text.

        text arrives via stream_event/content_block_delta.text_delta; tool
        calls appear as tool_use blocks in the assistant message; tool results
        as tool_result blocks in the user message; each turn ends with a
        {"type":"result",...} line (clears the active flag).
        """
        t = obj.get("type")
        if t == "stream_event":
            ev = obj.get("event", {}) or {}
            if ev.get("type") == "content_block_delta":
                delta = ev.get("delta", {}) or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        with self._llm_lock:
                            if not self._llm_log_in_asst:
                                self._llm_log_in_asst = True
                                self._llm_log_write("\n=== ASSISTANT ===\n")
                            self._llm_buf += text
                            if self._llm_log_fh:
                                try:
                                    self._llm_log_fh.write(text)
                                    self._llm_log_fh.flush()
                                except OSError:
                                    pass
            return
        if t == "assistant":
            # Emit tool_use markers only (text already streamed via deltas).
            msg = obj.get("message", {}) or {}
            for blk in msg.get("content", []) or []:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    name = blk.get("name", "?")
                    inp = blk.get("input", {})
                    try:
                        compact = json.dumps(inp, separators=(",", ":"))
                    except (TypeError, ValueError):
                        compact = str(inp)
                    if len(compact) > 200:
                        compact = compact[:197] + "..."
                    try:
                        full_json = json.dumps(inp, indent=2)
                    except (TypeError, ValueError):
                        full_json = compact
                    with self._llm_lock:
                        self._llm_log_write(f"\n=== TOOL CALL: {name} ===\n")
                        self._llm_log_write(full_json + "\n")
                    self._llm_append(f"\n[tool: {name} {compact}]\n")
            return
        if t == "user":
            # Tool results returned to the model.
            msg = obj.get("message", {}) or {}
            content = msg.get("content", [])
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        result_content = blk.get("content", "")
                        if isinstance(result_content, list):
                            # content blocks: extract text parts
                            result_text = "\n".join(
                                c.get("text", "") for c in result_content
                                if isinstance(c, dict) and c.get("type") == "text"
                            )
                        else:
                            result_text = str(result_content) if result_content else ""
                        with self._llm_lock:
                            self._llm_log_write("\n=== TOOL RESULT ===\n")
                            if result_text:
                                self._llm_log_write(result_text + "\n")
                        self._llm_append("\n[tool result]\n")
            return
        if t == "result":
            with self._llm_lock:
                self._llm_active = False
                self._llm_log_in_asst = False
            return

    def llm_send(self, prompt):
        """Send one user turn to the claude subprocess; return {ok, offset}.

        offset is the current buffer length so the browser can tail only the
        new (assistant) output that follows this prompt.
        """
        if not self.llm_enabled:
            return {"ok": False, "error": "LLM tab disabled (--no-llm)"}
        prompt = (prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "empty prompt"}
        with self._llm_lock:
            dead = self._llm_proc is None or self._llm_proc.poll() is not None
        if dead:
            self._llm_spawn()
        with self._llm_lock:
            is_first = self._llm_first_turn
            sys_text = self._llm_system_prompt_text if is_first else None
            self._llm_first_turn = False
        # Inject live backend state as a [context] line so the LLM always knows
        # what is currently running, regardless of when it was spawned.
        backend_ctx = self._llm_backend_context()
        prompt_with_ctx = (backend_ctx + "\n" + prompt) if backend_ctx else prompt
        if sys_text:
            full_text = sys_text + "\n\n---\n\n" + prompt_with_ctx
        else:
            full_text = prompt_with_ctx
        line = json.dumps({
            "type": "user",
            "message": {"role": "user",
                        "content": [{"type": "text", "text": full_text}]},
        })
        with self._llm_lock:
            offset = len(self._llm_buf)
            self._llm_active = True
            self._llm_log_in_asst = False
            # Split context prefix (lines starting with "[context]") from real user text.
            lines = prompt_with_ctx.splitlines()
            ctx_lines = []
            user_lines = []
            ctx_done = False
            for ln in lines:
                if not ctx_done and ln.startswith("[context]"):
                    ctx_lines.append(ln)
                else:
                    ctx_done = True
                    user_lines.append(ln)
            if ctx_lines:
                self._llm_log_write("\n=== CONTEXT (auto) ===\n")
                self._llm_log_write("\n".join(ctx_lines) + "\n")
            self._llm_log_write("\n=== USER ===\n")
            self._llm_log_write("\n".join(user_lines) + "\n")
        try:
            self._llm_proc.stdin.write(line + "\n")
            self._llm_proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            with self._llm_lock:
                self._llm_active = False
            return {"ok": False, "error": f"claude pipe broken: {e}"}
        return {"ok": True, "offset": offset}

    def llm_poll(self, offset):
        """Return the transcript slice past `offset` plus the turn-active flag."""
        with self._llm_lock:
            buf = self._llm_buf
            active = self._llm_active
        if offset < 0:
            offset = 0
        return {"data": buf[offset:], "next": len(buf), "active": active}

    def llm_reset(self):
        """Terminate + respawn the claude subprocess (new conversation);
        clear the transcript buffer."""
        with self._llm_lock:
            proc = self._llm_proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
        with self._llm_lock:
            self._llm_proc = None
            self._llm_buf = ""
            self._llm_active = False
            self._llm_log_in_asst = False
            self._llm_first_turn = True
            old_fh = self._llm_log_fh
            self._llm_log_fh = None
        if old_fh:
            try:
                old_fh.write("\n[session end]\n")
                old_fh.close()
            except OSError:
                pass
        if self.llm_enabled:
            self._llm_spawn()
        return {"ok": True}


# ── hardware read helpers (read-only) ─────────────────────────────────────────

def _hw_available():
    return shutil.which("aiedbg") is not None


def _xsdb_available():
    return shutil.which("xsdb") is not None


def _target_to_tcp(target):
    """'xsdb://host:port' (or 'host:port') -> ('host', 'port'); default port 3121."""
    t = (target or "").strip()
    if t.startswith("xsdb://"):
        t = t[len("xsdb://"):]
    host, _, port = t.partition(":")
    return host, (port or "3121")


def run_xsdb_connect(target, timeout=30):
    """Probe a live JTAG connection by having xsdb connect to the hw_server.

    Feeds a short tcl script over stdin so the process always exits and the
    connect result is unambiguous (a catch marker), regardless of whether the
    connect succeeds. Read-only: only `connect` is issued, never stop/reset.
    Returns (ok: bool, detail: str).
    """
    host, port = _target_to_tcp(target)
    if not host:
        return False, "no host in target"
    script = (
        "if {[catch {connect -url TCP:%s:%s} e]} "
        "{puts \"PROBE_FAIL $e\"} else {puts PROBE_OK}\n"
        "exit\n" % (host, port)
    )
    try:
        result = subprocess.run(
            ["xsdb"], input=script,
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, "xsdb not found in PATH"
    except subprocess.TimeoutExpired:
        return False, f"xsdb connect timed out (TCP:{host}:{port})"
    out = (result.stdout or "") + (result.stderr or "")
    if "PROBE_OK" in out:
        return True, f"xsdb connect ok TCP:{host}:{port}"
    # Extract the tcl error after PROBE_FAIL if present.
    detail = "connect failed"
    for line in out.splitlines():
        if "PROBE_FAIL" in line:
            detail = line.split("PROBE_FAIL", 1)[1].strip() or detail
            break
    return False, f"{detail} (TCP:{host}:{port})"


def _dma_state(decoded):
    if decoded is None:
        return "unreachable"
    if decoded["err_bd_unavail"] or decoded["err_bd_invalid"]:
        return "error"
    if (decoded["stall_lock_acq"] or decoded["stall_lock_rel"]
            or decoded["stall_stream"] or decoded["stall_tct"]):
        return "stalled"
    if decoded["channel_running"] or decoded["status"] == 1:
        return "running"
    return "idle"


def _read_dma_channel(st, tile_type, phys_col, row, direction, channel,
                      target=None, reg_read_fn=None):
    off = aiediag.compute_reg_offset(tile_type, direction, channel,
                                     st.aie_version)
    if reg_read_fn is not None:
        raw = reg_read_fn(phys_col, row, off)
    else:
        raw = aiediag.run_aiedbg_reg_read(phys_col, row, off,
                                          target=target or st.target,
                                          device=st.device)
    if raw is None:
        return {"state": "unreachable", "offset": f"0x{off:X}"}
    decoded = aiediag.decode_dma_status(raw)
    return {
        "state": _dma_state(decoded),
        "offset": f"0x{off:X}",
        "raw": f"0x{raw:08X}",
        "status": decoded["status_str"],
        "running": bool(decoded["channel_running"]),
        "q_size": decoded["q_size"],
        "cur_bd": decoded["cur_bd"],
        "stalls": [k for k in ("stall_lock_acq", "stall_lock_rel",
                               "stall_stream", "stall_tct") if decoded[k]],
        "errors": [k for k in ("err_bd_unavail", "err_bd_invalid")
                   if decoded[k]],
    }


def _read_channel_events(st, tile_type, phys_col, row, direction, channel,
                         target=None):
    """Decode DMA start/finish/(stall/error) events for one channel.

    shim → read_shim_event_status; core/mem → read_event_status_4 + CORE map.
    """
    tgt = target or st.target
    d = direction.lower()
    if tile_type == "shim":
        bits = aiediag.read_shim_event_status(phys_col, d, channel,
                                              target=tgt, device=st.device,
                                              dry_run=False)
        if bits is None:
            return {"state": "unreachable"}
        events = dict(bits)
    else:
        emap = aiediag.CORE_MEM_DMA_EVENT_IDS.get((d, channel))
        if emap is None:
            return {"error": f"no event map for {d}{channel} on {tile_type}"}
        regs4 = aiediag.read_event_status_4(phys_col, row,
                                            aiediag.MEM_EVT_STATUS_REGS,
                                            target=tgt, device=st.device,
                                            dry_run=False)
        if regs4 is None:
            return {"state": "unreachable"}
        events = {name: aiediag._evt_active(eid, regs4)
                  for name, eid in emap.items()}
    started = events.get("START_TASK", False)
    finished = events.get("FINISHED_TASK", False)
    if events.get("ERROR"):
        summary = "DMA ERROR event active"
    elif started and finished:
        summary = "started and finished (completed)"
    elif started:
        summary = "started but never finished (running/stuck)"
    else:
        summary = "never started (start_io not issued?)"
    return {"events": events, "summary": summary}


def _worst(states):
    order = ["unreachable", "error", "stalled", "running", "idle"]
    for s in order:
        if s in states:
            return s
    return "idle"


def _make_reg_read_fn(st, device, target):
    """Return a reg-read callable for (phys_col, row, offset) -> int|None.

    For the simulator device, use the IPC debug socket directly (no aiedbg).
    For all other devices, delegate to aiediag.run_aiedbg_reg_read with target.
    """
    if (device or "").strip().lower() == "simulator":
        return st.sim_ipc_reg_read
    tgt = target or st.target
    def _aiedbg_read(phys_col, row, offset):
        return aiediag.run_aiedbg_reg_read(phys_col, row, offset,
                                           target=tgt, device=st.device)
    return _aiedbg_read


def grid_dma(st, target=None, reg_read_fn=None):
    cells = {}
    for t in st.tiles():
        col, row = t["loc"][0], t["loc"][1]
        phys_col = col + st.startcol
        chans = {}
        for ch in t.get("dma_channels", []):
            d = str(ch.get("direction", "")).lower()
            c = int(ch.get("channel", 0))
            if d not in ("mm2s", "s2mm"):
                continue
            chans[f"{d}{c}"] = _read_dma_channel(st, t["type"], phys_col,
                                                 row, d, c,
                                                 target=target or st.target,
                                                 reg_read_fn=reg_read_fn)
        state = _worst([v["state"] for v in chans.values()]) if chans else "idle"
        cells[f"{col},{row}"] = {"state": state, "type": t["type"],
                                 "phys_col": phys_col, "channels": chans}
    return {"what": "dma", "cells": cells}


def grid_cores(st, target=None, reg_read_fn=None):
    cells = {}
    entries, _ = aiediag.load_linemap()
    for t in st.tiles():
        col, row = t["loc"][0], t["loc"][1]
        if t["type"] != "core":
            continue
        phys_col = col + st.startcol
        if reg_read_fn is not None:
            raw = reg_read_fn(phys_col, row, aiediag.CORE_PC_OFFSET)
        else:
            raw = aiediag.run_aiedbg_reg_read(phys_col, row,
                                              aiediag.CORE_PC_OFFSET,
                                              target=target or st.target,
                                              device=st.device)
        if raw is None:
            cells[f"{col},{row}"] = {"state": "unreachable"}
            continue
        pc = raw & aiediag.CORE_PC_MASK
        src = aiediag.pc_to_source(entries, pc) if entries else None
        cells[f"{col},{row}"] = {
            "state": "running" if pc else "idle",
            "pc": f"0x{pc:05X}",
            "source": (f"{os.path.basename(src['file'])}:{src['line']}"
                       if src else None),
        }
    return {"what": "cores", "cells": cells}


def grid_events(st, target=None, reg_read_fn=None):
    cells = {}
    for t in st.tiles():
        col, row = t["loc"][0], t["loc"][1]
        phys_col = col + st.startcol
        if t["type"] == "shim":
            regs = (aiediag.SHIM_EVT_STATUS_REG0, aiediag.SHIM_EVT_STATUS_REG1)
        else:
            regs = aiediag.MEM_EVT_STATUS_REGS
        words = []
        for off in regs:
            if reg_read_fn is not None:
                raw = reg_read_fn(phys_col, row, off)
            else:
                raw = aiediag.run_aiedbg_reg_read(phys_col, row, off,
                                                  target=target or st.target,
                                                  device=st.device)
            words.append(None if raw is None else f"0x{raw:08X}")
        if all(w is None for w in words):
            cells[f"{col},{row}"] = {"state": "unreachable"}
            continue
        any_set = any(w not in (None, "0x00000000") for w in words)
        cells[f"{col},{row}"] = {"state": "running" if any_set else "idle",
                                 "words": words}
    return {"what": "events", "cells": cells}


def do_cmd(st, body, target=None, reg_read_fn=None):
    tgt = target or st.target
    op = str(body.get("op", "")).strip().lower()
    parts = op.split()
    verb = parts[0] if parts else ""
    if verb not in _CMD_OPS:
        return {"error": f"op not allowed: {op!r}; allowed: {sorted(_CMD_OPS)}"}
    try:
        col = int(body["col"])
        row = int(body["row"])
    except (KeyError, ValueError, TypeError):
        return {"error": "col and row (integers) are required"}
    startcol = int(body.get("startcol", st.startcol))
    phys_col = col + startcol
    tile_type = str(body.get("type", "core")).lower()

    def _read(pc, r, off):
        if reg_read_fn is not None:
            return reg_read_fn(pc, r, off)
        return aiediag.run_aiedbg_reg_read(pc, r, off, target=tgt,
                                           device=st.device)

    if verb == "reg":
        if len(parts) < 2:
            return {"error": "reg requires an offset, e.g. 'reg 0x1DF00'"}
        try:
            off = int(parts[1], 0)
        except ValueError:
            return {"error": f"bad offset: {parts[1]}"}
        raw = _read(phys_col, row, off)
        return {"op": op, "phys_col": phys_col, "row": row,
                "offset": f"0x{off:X}",
                "value": None if raw is None else f"0x{raw:08X}"}

    if verb == "dma":
        d, c = _parse_dir_ch(body.get("dir_ch"))
        if d is None:
            return {"error": "dma needs dir_ch like 'mm2s0' or 's2mm1'"}
        res = _read_dma_channel(st, tile_type, phys_col, row, d, c,
                                target=tgt, reg_read_fn=reg_read_fn)
        res.update({"op": op, "phys_col": phys_col, "row": row,
                    "dir_ch": f"{d}{c}"})
        return res

    if verb in ("pc", "core"):
        raw = _read(phys_col, row, aiediag.CORE_PC_OFFSET)
        if raw is None:
            return {"op": op, "phys_col": phys_col, "row": row,
                    "error": "unreachable"}
        pc = raw & aiediag.CORE_PC_MASK
        entries, _ = aiediag.load_linemap()
        src = aiediag.pc_to_source(entries, pc) if entries else None
        return {"op": op, "phys_col": phys_col, "row": row,
                "pc": f"0x{pc:05X}",
                "source": (f"{os.path.basename(src['file'])}:{src['line']}"
                           if src else None)}

    if verb == "event":
        if tile_type == "shim":
            regs = (aiediag.SHIM_EVT_STATUS_REG0, aiediag.SHIM_EVT_STATUS_REG1)
        else:
            regs = aiediag.MEM_EVT_STATUS_REGS
        words = []
        for off in regs:
            raw = _read(phys_col, row, off)
            words.append({"offset": f"0x{off:X}",
                          "value": None if raw is None else f"0x{raw:08X}"})
        return {"op": op, "phys_col": phys_col, "row": row, "words": words}

    if verb == "chans":
        chans = []
        for t in st.tiles():
            if t["loc"][0] == col and t["loc"][1] == row:
                for ch in t.get("dma_channels", []):
                    d = str(ch.get("direction", "")).lower()
                    c = int(ch.get("channel", 0))
                    if d not in ("mm2s", "s2mm"):
                        continue
                    r = _read_dma_channel(st, tile_type, phys_col, row, d, c,
                                          target=tgt, reg_read_fn=reg_read_fn)
                    chans.append({"dir_ch": f"{d}{c}",
                                  "flow_index": ch.get("flow_index"),
                                  "state": r.get("state")})
                break
        return {"op": op, "col": col, "row": row, "phys_col": phys_col,
                "channels": chans}

    if verb == "chanevent":
        if reg_read_fn is not None:
            return {"error": "chanevent not supported for simulator (IPC)"}
        d, c = _parse_dir_ch(body.get("dir_ch"))
        if d is None:
            return {"error": "chanevent needs dir_ch like 'mm2s0'"}
        res = _read_channel_events(st, tile_type, phys_col, row, d, c, target=tgt)
        res.update({"op": op, "phys_col": phys_col, "row": row,
                    "dir_ch": f"{d}{c}"})
        return res

    return {"error": f"unhandled op: {op}"}


# ── HTTP handler ──────────────────────────────────────────────────────────────

def _check_llm_auth(st, handler):
    """Return True if the request may use the LLM endpoints.

    Auth disabled (no password configured) => always True. Otherwise the
    browser must send the configured password in the X-LLM-Auth header; compared
    with hmac.compare_digest to avoid timing leaks.
    """
    if not st.llm_password:
        return True
    supplied = handler.headers.get("X-LLM-Auth", "")
    return hmac.compare_digest(str(supplied), str(st.llm_password))


class Handler(BaseHTTPRequestHandler):
    state = None  # injected in main()

    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write("[schedule_debug_server] " + (fmt % args) + "\n")

    def _send_json(self, obj, code=200):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path, content_type):
        if not os.path.isfile(path):
            self._send_json({"error": f"not found: {path}"}, code=404)
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        st = self.state
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if path in ("/", "/index.html", "/host_schedule.html"):
            self._send_file(st.html_path(), "text/html; charset=utf-8")
        elif path == "/schedule_view.json":
            self._send_file(st.json_path(), "application/json")
        elif path == "/config":
            # Board defaults so the browser preselects the device + hostname.
            self._send_json(_ui_defaults(st))
        elif path == "/applog":
            offset = int(q.get("offset", ["0"])[0])
            self._send_json(st.applog_since(offset))
        elif path == "/hwsrv_log":
            offset = int(q.get("offset", ["0"])[0])
            self._send_json(st.hwsrv_log_since(offset))
        elif path == "/sim/log":
            offset = int(q.get("offset", ["0"])[0])
            self._send_json(st.simlog_since(offset))
        elif path == "/sim/applog":
            offset = int(q.get("offset", ["0"])[0])
            self._send_json(st.sim_applog_since(offset))
        elif path == "/sim/status":
            self._send_json(st.sim_status())
        elif path == "/devices":
            # Extra device options for the board dropdown (e.g. simulator).
            # Returns list of {value, label} objects ready for <option> injection.
            self._send_json({"devices": [
                {"value": d["value"], "label": d["label"]}
                for d in st._extra_devices
                if d.get("value") and d.get("label")
            ]})
        elif path == "/llm/auth":
            # Unprotected: lets the browser learn whether to prompt for a password.
            self._send_json({"required": st.llm_auth_required()})
        elif path == "/llm/poll":
            # No _hw_available gate: the LLM tab works without a JTAG board.
            if not st.llm_enabled:
                self._send_json({"error": "LLM tab disabled", "data": "",
                                 "next": 0, "active": False}, code=404)
                return
            if not _check_llm_auth(st, self):
                self._send_json({"error": "unauthorized", "auth": True,
                                 "data": "", "next": 0, "active": False},
                                code=401)
                return
            try:
                offset = int(q.get("offset", ["0"])[0])
            except (ValueError, TypeError):
                offset = 0
            self._send_json(st.llm_poll(offset))
        elif path == "/ping":
            if st.run_in_progress():
                self._send_json({"ok": False, "detail": "disabled during run"})
                return
            device = q.get("device", [""])[0]
            host = q.get("host", [""])[0]
            self._send_json(ping(st, device, host))
        elif path == "/grid":
            what = q.get("what", ["dma"])[0]
            if st.run_in_progress():
                self._send_json({"error": "disabled during run", "cells": {}})
                return
            device = q.get("device", [""])[0]
            host = q.get("host", [""])[0]
            is_sim = (device or "").strip().lower() == "simulator"
            if not is_sim and not _hw_available():
                self._send_json({"error": "aiedbg not found in PATH",
                                 "cells": {}})
                return
            tgt = resolve_target(st, device, host)
            rrfn = _make_reg_read_fn(st, device, tgt)
            try:
                if what == "cores":
                    self._send_json(grid_cores(st, target=tgt,
                                               reg_read_fn=rrfn))
                elif what == "events":
                    self._send_json(grid_events(st, target=tgt,
                                                reg_read_fn=rrfn))
                else:
                    self._send_json(grid_dma(st, target=tgt,
                                             reg_read_fn=rrfn))
            except Exception as e:  # never crash the poll loop
                self._send_json({"error": str(e), "cells": {}}, code=500)
        else:
            self._send_json({"error": f"unknown path: {path}"}, code=404)

    def do_POST(self):
        st = self.state
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, code=400)
            return
        if u.path == "/run":
            if "startcol" in body:
                try:
                    st.startcol = int(body["startcol"])
                except (ValueError, TypeError):
                    pass
            self._send_json(st.start_run(device=body.get("device"),
                                         board_host=body.get("board_host")))
        elif u.path == "/stop":
            self._send_json(st.stop_run())
        elif u.path == "/sim/run":
            self._send_json(st.start_sim())
        elif u.path == "/sim/stop":
            self._send_json(st.stop_sim())
        elif u.path == "/cmd":
            if st.run_blocks_debug():
                self._send_json({"error": "disabled during run setup"})
                return
            dev = body.get("device", "")
            is_sim = (dev or "").strip().lower() == "simulator"
            if not is_sim and not _hw_available():
                self._send_json({"error": "aiedbg not found in PATH"})
                return
            tgt = resolve_target(st, dev, body.get("host"))
            rrfn = _make_reg_read_fn(st, dev, tgt)
            try:
                self._send_json(do_cmd(st, body, target=tgt, reg_read_fn=rrfn))
            except Exception as e:
                self._send_json({"error": str(e)}, code=500)
        elif u.path == "/settarget":
            # Switch the live-debug target (and respawn the aiegdb console so its
            # reads/aiedbg use it). The browser only POSTs here AFTER a
            # successful /ping, so the resolved target is already known reachable;
            # we just resolve device+host the same way /ping did and hand it to
            # retarget(). Without this, the aiegdb console stays pinned to the
            # daemon's startup target no matter what host the UI selects.
            dev = body.get("device", "")
            is_sim = (dev or "").strip().lower() == "simulator"
            # Simulator is not a board run, so don't block on run_in_progress().
            if not is_sim and st.run_in_progress():
                self._send_json({"ok": False, "detail": "disabled during run"})
                return
            if is_sim:
                # Simulator reads go through IPC debug socket, not aiegdb.
                # Just report readiness without touching the aiegdb subprocess.
                ready = st._sim_ipc_ready
                self._send_json({
                    "ok": True,
                    "target": "simulator-ipc",
                    "detail": ("IPC debug socket ready" if ready
                               else "simulator not started yet"),
                })
                return
            tgt = resolve_target(st, dev, body.get("host"))
            if not tgt:
                self._send_json({"ok": False, "target": None,
                                 "detail": "no target configured"})
                return
            try:
                self._send_json(st.retarget(tgt))
            except Exception as e:
                self._send_json({"ok": False, "target": tgt,
                                 "detail": str(e)}, code=500)
        elif u.path == "/aiegdb":
            if st.run_blocks_debug():
                self._send_json({"error": "disabled during run setup"})
                return
            # No _hw_available hard-gate: navigation/help work without aiedbg;
            # live reads fail gracefully inside aiegdb.
            try:
                self._send_json(st.gdb_command(body.get("cmd", "")))
            except Exception as e:
                self._send_json({"error": str(e)}, code=500)
        elif u.path == "/aiegdb/reload":
            if st.run_blocks_debug():
                self._send_json({"error": "disabled during run setup"})
                return
            try:
                self._send_json(st.gdb_reload())
            except Exception as e:
                self._send_json({"error": str(e)}, code=500)
        elif u.path == "/llm":
            # No _hw_available gate: the LLM tab works without a JTAG board.
            if not st.llm_enabled:
                self._send_json({"ok": False, "error": "LLM tab disabled"},
                                code=404)
                return
            if not _check_llm_auth(st, self):
                self._send_json({"ok": False, "error": "unauthorized",
                                 "auth": True}, code=401)
                return
            try:
                self._send_json(st.llm_send(body.get("prompt", "")))
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, code=500)
        elif u.path == "/llm/reset":
            if not st.llm_enabled:
                self._send_json({"ok": False, "error": "LLM tab disabled"},
                                code=404)
                return
            if not _check_llm_auth(st, self):
                self._send_json({"ok": False, "error": "unauthorized",
                                 "auth": True}, code=401)
                return
            try:
                self._send_json(st.llm_reset())
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, code=500)
        elif u.path == "/launch_hwserver":
            # Connect-failure recovery: kick off (async) an ssh session that
            # starts hw_server on the board, then re-probes JTAG once. Returns
            # immediately with {started}; the browser tails per-step progress
            # via GET /hwsrv_log and applies the final result there.
            if st.run_in_progress():
                self._send_json({"started": False,
                                 "detail": "disabled during run"})
                return
            if not _xsdb_available():
                self._send_json({"started": False,
                                 "detail": "xsdb not found in PATH"})
                return
            try:
                self._send_json(st.start_hwserver_async(body.get("device"),
                                                        body.get("host")))
            except Exception as e:
                self._send_json({"started": False, "detail": str(e)}, code=500)
        else:
            self._send_json({"error": f"unknown path: {u.path}"}, code=404)


def ping(st, device, host):
    """Probe the live-debug connection.

    For the simulator: probe the IPC debug socket directly (no xsdb needed).
    For boards: have xsdb connect to the JTAG hw_server.
    Returns {ok, xsdb, target, detail}.
    """
    if (device or "").strip().lower() == "simulator":
        with st._sim_lock:
            ready = st._sim_ipc_ready
            dbg = st._sim_dbg_socket
        if ready and dbg and sim_ipc_ping(dbg):
            return {"ok": True, "xsdb": False, "target": "simulator-ipc",
                    "detail": f"IPC debug socket ready ({os.path.basename(dbg)})"}
        if st.sim_in_progress():
            return {"ok": False, "xsdb": False, "target": None,
                    "detail": "simulator running but IPC not ready yet"}
        return {"ok": False, "xsdb": False, "target": None,
                "detail": "simulator not running — press Run to start it"}
    if not _xsdb_available():
        return {"ok": False, "xsdb": False, "target": None,
                "detail": "xsdb not found in PATH"}
    tgt = resolve_target(st, device, host)
    if not tgt:
        return {"ok": False, "xsdb": True, "target": None,
                "detail": "no target configured"}
    ok, detail = run_xsdb_connect(tgt)
    return {"ok": ok, "xsdb": True, "target": tgt, "detail": detail}


class _BufWriter:
    """Minimal file-like sink for pexpect `logfile_read`: appends written text
    to a shared list under a lock so the hw_server session can be tailed."""

    def __init__(self, buf, lock):
        self._buf, self._lock = buf, lock

    def write(self, s):
        with self._lock:
            self._buf.append(s)

    def flush(self):
        pass


def _hwsrv_drain(child):
    """Keep the hw_server pexpect child alive by consuming its output until EOF
    (feeds the child's logfile_read sink). hw_server runs forever, so this
    normally blocks until the daemon/ssh dies."""
    try:
        child.expect(pexpect.EOF, timeout=None)
    except Exception:
        pass


def resolve_target(st, device, host):
    """Device-aware aiedbg target for live reads.

    * vek385 + host → xsdb://<host>:3121 (the board hostname from the UI).
    * palmyra       → xsdb://<PALIP>:3121 ($PALIP, fallback xx.xx.xx.213).
    * simulator     → None (reads go through IPC debug socket, not aiedbg).
    * otherwise     → the daemon's configured/env target (st.target).
    """
    device = (device or "").strip().lower()
    if device == "vek385" and host:
        return f"xsdb://{host}:3121"
    if device == "palmyra":
        return f"xsdb://{os.environ.get('PALIP', 'xx.xx.xx.213')}:3121"
    if device == "simulator":
        return None  # reads go through IPC debug socket, not aiedbg
    return st.target


def _ui_defaults(st):
    """Default board selection for the browser's device dropdown + hostname box.

    Derives the board hostname from $VEK385IP (what envlocal.sh exports) or, as a
    fallback, the host component of the daemon's resolved JTAG target
    (xsdb://<host>:port). When a hostname is known we preselect the vek385 device
    so the user doesn't have to pick it manually on every load.
    """
    host = os.environ.get("VEK385IP", "").strip()
    if not host and st.target:
        m = re.match(r"xsdb://([^:/]+)", st.target)
        if m:
            host = m.group(1)
    return {"device": "vek385" if host else "", "board_host": host}


def _target_from_aiedbg_env():
    """Read AIEDBG_TARGET from ~/.aiedbg_env (the file aiedbg-setup writes),
    so the daemon works after a restart without a manual `source`."""
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


def _lan_ip():
    """Best-effort primary LAN IPv4 for the display URL when bound to 0.0.0.0.
    Uses a UDP socket to a public IP (no packets sent) to learn the outbound
    interface address; falls back to the resolved hostname, then localhost."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


# ── occupied-port policy (same-user => exit + list pid; else pick next port) ──

def _username(uid):
    """Resolve a numeric uid to a login name, falling back to the number."""
    try:
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        return str(uid)


def _listen_sockets_on_port(port):
    """Scan /proc/net/tcp{,6} for LISTEN sockets bound to `port`.

    Returns a list of (uid:int, inode:str) for every match. The socket owner's
    uid is world-readable in /proc/net/tcp, so this works even when the port is
    held by another user's process (unlike `ss -p`, which needs root to show
    other users' pids).
    """
    out = []
    for pf in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(pf) as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines[1:]:  # skip header
            parts = line.split()
            if len(parts) < 10:
                continue
            if parts[3] != "0A":  # TCP_LISTEN
                continue
            hexport = parts[1].rsplit(":", 1)[-1]
            try:
                if int(hexport, 16) != port:
                    continue
                uid = int(parts[7])
            except ValueError:
                continue
            out.append((uid, parts[9]))
    return out


def _pids_for_inode(inode):
    """Return pids whose /proc/<pid>/fd contains socket:[<inode>] (only those
    the current user can see — sufficient for the same-user case)."""
    target = f"socket:[{inode}]"
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        fdd = os.path.join("/proc", name, "fd")
        try:
            fds = os.listdir(fdd)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(fdd, fd)) == target:
                    pids.append(int(name))
                    break
            except OSError:
                continue
    return pids


def _proc_cmdline(pid):
    """Best-effort command line for a pid ('' -> comm fallback -> None)."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if cmd:
            return cmd
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip() or None
    except OSError:
        return None


def make_server(host, requested_port, handler, max_scan=50):
    """Bind the daemon's HTTP server, applying the occupied-port policy.

    * requested_port free            -> bind and return it.
    * busy, held by the SAME user    -> print the offending pid(s)/cmdline and
                                        exit(1) (it's this user's own daemon).
    * busy, held by a DIFFERENT user -> scan upward for the next free port.

    Returns (server, actual_port).
    """
    try:
        return ThreadingHTTPServer((host, requested_port), handler), \
            requested_port
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise

    owners = _listen_sockets_on_port(requested_port)
    my_uid = os.getuid()
    if any(uid == my_uid for uid, _ in owners):
        print(f"error: port {requested_port} is already in use by YOU "
              f"(user '{_username(my_uid)}').", file=sys.stderr)
        printed = False
        for uid, inode in owners:
            if uid != my_uid:
                continue
            for pid in _pids_for_inode(inode):
                cmd = _proc_cmdline(pid) or "(unknown command)"
                print(f"  pid {pid}: {cmd}", file=sys.stderr)
                printed = True
        if not printed:
            print("  (could not resolve the owning pid from /proc)",
                  file=sys.stderr)
        print("Stop that process or pass a different --port.", file=sys.stderr)
        sys.exit(1)

    # Held by another user (or owner unknown): find the next free port.
    other = f" (used by user '{_username(owners[0][0])}')" if owners else ""
    for port in range(requested_port + 1, requested_port + 1 + max_scan):
        try:
            server = ThreadingHTTPServer((host, port), handler)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                continue
            raise
        print(f"notice: port {requested_port} busy{other}; "
              f"using port {port} instead.", file=sys.stderr)
        return server, port

    print(f"error: port {requested_port} busy{other} and no free port found "
          f"in [{requested_port + 1}, {requested_port + max_scan}].",
          file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Live AIE debug/test daemon for host_schedule.html")
    ap.add_argument("workdir", nargs="?", default="aout/worklocal",
                    help="dir with host_schedule.html + schedule_view.json "
                         "+ provenance JSONs (default: aout/worklocal)")
    ap.add_argument("--elf", default=None,
                    help="ELF to deploy via /run (default: project aout ELF "
                         "<workdir>/build/host or aout/main.elf; falls back to "
                         "apppaltest -y auto-pick if none exists)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (default: 0.0.0.0 = all interfaces, "
                         "reachable from other machines; use 127.0.0.1 for "
                         "localhost only)")
    ap.add_argument("--port", type=int, default=8091,
                    help="HTTP port (default: 8091)")
    ap.add_argument("--aie-version", default=None,
                    help="AIE version for register offsets: 5 or 2ps "
                         "(default: auto-detected from provenance JSON, else 5)")
    ap.add_argument("--device", default="pal", help="aiedbg --device (default pal)")
    ap.add_argument("--target", default=None,
                    help="aiedbg --target (xsdb://host:port); "
                         "default: $AIEDBG_TARGET, then ~/.aiedbg_env")
    ap.add_argument("--startcol", type=int, default=None,
                    help="physical column offset: phys_col = col + startcol "
                         "(default: read from provenance JSON in workdir, else 0)")
    ap.add_argument("--apppaltest", default=None,
                    help="path to apppaltest.py (default: script/test/apppaltest.py)")
    ap.add_argument("--applog", default=None,
                    help="run log file to write + tail (default: repo-root applog)")
    ap.add_argument("--open", action="store_true",
                    help="open the served URL in a browser after binding")
    ap.add_argument("--claude-bin", default="claude",
                    help="path to the claude CLI for the LLM tab (default: claude)")
    ap.add_argument("--claude-model", default=None,
                    help="model for the LLM tab (default: claude CLI default)")
    ap.add_argument("--claude-cwd", default=None,
                    help="working dir for the claude subprocess so it loads "
                         "CLAUDE.md/skills (default: repo root)")
    ap.add_argument("--no-llm", action="store_true",
                    help="disable the LLM tab and its /llm* endpoints")
    ap.add_argument("--password", default=None,
                    help="password required to use the LLM tab; fallback env "
                         "SCHEDULE_DEBUG_PASSWORD; if unset and stdin is a TTY "
                         "you are prompted at startup (empty => no auth)")
    ap.add_argument("--no-password", action="store_true",
                    help="skip the interactive password prompt (no LLM auth)")
    ap.add_argument("--no-mcp-probe", action="store_true",
                    help="skip the startup probe that verifies the LLM tab's "
                         "claude can reach the aiegdb MCP server")
    args = ap.parse_args()

    workdir = os.path.abspath(args.workdir)
    # --elf wins. Otherwise default to the project's freshly-built tiling ELF
    # (<workdir>/build/host or aout/main.elf); only if none exists do we leave
    # elf=None so apppaltest -y auto-picks its own default.
    elf = os.path.abspath(args.elf) if args.elf else _resolve_default_elf(workdir)
    apppaltest = args.apppaltest or os.path.join(
        _REPO_ROOT, "script", "test", "apppaltest.py")
    applog = args.applog or os.path.join(_REPO_ROOT, "applog")
    # aiedbg needs a JTAG target. Resolve in priority: --target, then
    # $AIEDBG_TARGET, then ~/.aiedbg_env (the file aiedbg-setup writes) so a
    # restart works without a manual `source`. Without any, aiedbg uses its own
    # default and live reads fail with "timed out" / "connection closed".
    target = (args.target or os.environ.get("AIEDBG_TARGET")
              or _target_from_aiedbg_env())

    if not os.path.isfile(os.path.join(workdir, "host_schedule.html")):
        print(f"warning: {workdir}/host_schedule.html not found; run "
              f"schedule_view.py first (or aiehlc.sh).", file=sys.stderr)

    # Resolve startcol + aie_version from the provenance JSONs in the workdir
    # (single source of truth for both). The JSONs always win for aie_version so
    # the HTML->aiegdb path is correct regardless of what number aiehlc.sh
    # passes; load them once.
    dfsche, dmaphop = aiediag.load_jsons(workdir)
    if args.startcol is None:
        startcol = aiediag.startcol_from_jsons(dfsche, dmaphop) or 0
    else:
        startcol = args.startcol
    # aie_version resolution order (JSON wins so the HTML->aiegdb path is correct
    # regardless of what number aiehlc.sh passes):
    #   1. provenance JSON aie_gen (authoritative, already normalized to a debug
    #      string by aie_version_from_jsons)
    #   2. explicit --aie-version flag, normalized through the mapper so compiler
    #      numbers (5) become debug strings (2ps)
    #   3. warn loudly and fall back to a default
    aie_version = aiediag.aie_version_from_jsons(dfsche, dmaphop)
    if aie_version is None and args.aie_version is not None:
        aie_version = aiediag.debug_aie_version_from_gen(args.aie_version)
    if aie_version is None:
        aie_version = "2ps"
        print(f"warning: could not resolve aie_version from provenance JSONs in "
              f"{workdir} or --aie-version flag; falling back to "
              f"'{aie_version}'. Register reads may be wrong.", file=sys.stderr)

    claude_cwd = os.path.abspath(args.claude_cwd) if args.claude_cwd \
        else _REPO_ROOT
    # LLM auth password: --password wins, else env SCHEDULE_DEBUG_PASSWORD,
    # else prompt interactively at startup (only when the LLM tab is enabled and
    # stdin is a TTY). Pressing Enter with no input => None => auth disabled.
    llm_password = args.password or os.environ.get("SCHEDULE_DEBUG_PASSWORD")
    if (llm_password is None and not args.no_llm and not args.no_password
            and sys.stdin.isatty()):
        try:
            entered = getpass.getpass(
                "LLM chat password (press Enter for no password): ")
        except (EOFError, KeyboardInterrupt):
            entered = ""
        llm_password = entered.strip() or None
    llm_password = llm_password or None
    Handler.state = DebugState(workdir, elf, aie_version, args.device,
                               target, apppaltest, startcol, applog,
                               claude_bin=args.claude_bin,
                               claude_cwd=claude_cwd,
                               claude_model=args.claude_model,
                               llm_enabled=not args.no_llm,
                               llm_password=llm_password)
    # Bind the server, applying the occupied-port policy: exit + list pid(s) if
    # this same user already holds the port, else fall forward to the next free
    # port when another user holds it.
    server, port = make_server(args.host, args.port, Handler)
    # When bound to all interfaces, show the LAN IP so the URL is reachable from
    # another machine (0.0.0.0 is not itself a connectable address).
    disp_host = _lan_ip() if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{disp_host}:{port}/"
    print(f"schedule_debug_server serving {workdir}")
    print(f"  URL:        {url}")
    print(f"  bind:       {args.host}:{port}")
    print(f"  ELF:        {elf or 'auto (apppaltest -y default)'}")
    print(f"  applog:     {applog}")
    print(f"  aiedbg:     {'found' if _hw_available() else 'NOT FOUND (live reads disabled)'}")
    if args.no_llm:
        print("  LLM:        disabled (--no-llm)")
    else:
        _claude = shutil.which(args.claude_bin) or (
            args.claude_bin if os.path.isfile(args.claude_bin) else None)
        print(f"  LLM:        claude {'found' if _claude else 'NOT FOUND'} "
              f"(bin={args.claude_bin}, cwd={claude_cwd}, "
              f"model={args.claude_model or 'CLI default'})")
        # Never print the password itself.
        print(f"  LLM auth:   {'enabled' if llm_password else 'disabled'}")
        if not args.no_mcp_probe and _claude:
            ok, detail = Handler.state.probe_mcp()
            status = "connected" if ok else "NOT connected"
            print(f"  LLM MCP:    aiegdb {status} ({detail})")
            if not ok:
                print("  WARNING: LLM tab claude could not reach the aiegdb "
                      "MCP server; the chat still works but AIE debug tools "
                      "may be unavailable.", file=sys.stderr)
    st = Handler.state
    st._write_backend_status()
    if st.sim_script:
        print(f"  sim:        {st.sim_script}")
        print(f"  sim-dir:    {st.sim_example_dir or 'NOT SET'}")
    else:
        print("  sim:        not configured (add debug_ui_config.json to workdir)")
    print(f"  target:     {target or 'NONE'}")
    print(f"  startcol:   {startcol}{'' if args.startcol is not None else ' (from provenance JSON)'}   device={args.device}   aie={aie_version}{'' if args.aie_version is not None else ' (from provenance JSON)'}")
    if _hw_available() and not target:
        print("  WARNING: no aiedbg target set — live grid/cmd reads will fail with "
              "'timed out' / 'connection closed'.\n"
              "           Fix: export AIEDBG_TARGET=xsdb://<host>:3121  (or pass "
              "--target xsdb://<host>:3121)", file=sys.stderr)
    print("  Ctrl-C to stop.")
    if args.open:
        threading.Thread(target=lambda: (time.sleep(0.5),
                                         webbrowser.open(url)),
                        daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nschedule_debug_server stopped.")
        gdb_proc = Handler.state._gdb_proc
        if gdb_proc is not None:
            try:
                gdb_proc.terminate()
            except OSError:
                pass
        llm_proc = Handler.state._llm_proc
        if llm_proc is not None:
            try:
                llm_proc.terminate()
            except OSError:
                pass
        mcp_cfg = Handler.state._mcp_config_path
        if mcp_cfg:
            try:
                os.unlink(mcp_cfg)
            except OSError:
                pass
        server.shutdown()


if __name__ == "__main__":
    main()
