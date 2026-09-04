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
import collections
import errno
import getpass
import hmac
import html
import ipaddress
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
from urllib.parse import urlparse, parse_qs, unquote

try:
    from pygments import highlight as _pyg_highlight
    from pygments.lexers import get_lexer_for_filename, get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    from pygments.util import ClassNotFound
    _HAVE_PYGMENTS = True
except ImportError:      # viewer still works, just monochrome
    _HAVE_PYGMENTS = False

# Import aiediag as a library (offsets, reg read, decoders, provenance, startcol).
# aiediag.py lives in the same directory (src/tool/debug/); it guards main()
# behind __main__, so importing it has no side effects. aiegdb is imported for
# its _SERVER_MARKER framing constant (also guarded behind __main__).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# This file lives at src/tool/debug/ → repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
_DEFAULT_WORKDIR = "aout/worklocal"
# Skills written FOR the embedded LLM (not for Claude Code sessions in this repo).
# Laid out as a Claude Code *plugin* so the spawned assistant loads them natively
# via --plugin-dir: that keeps one copy, versioned beside the tooling it documents,
# instead of a duplicate under .claude/skills/ that would drift. Being explicit
# also survives the announced change making `--bare` (no auto-discovery) the
# default for `claude -p`.
_LLM_PLUGIN_DIR = os.path.join(_THIS_DIR, "dbg_llm_skills")
_LLM_SKILLS_DIR = os.path.join(_LLM_PLUGIN_DIR, "skills")
_LLM_STUCK_S = 120
_LLM_STUCK_HARD_S = 600


import struct as _struct

_IPC_READ32     = 0x11
_IPC_NPI_READ32 = 0x13
_IPC_STATUS_OK  = 0x00
_IPC_REQ_FMT   = "<BxxxQI"
_IPC_RESP_FMT  = "<BxxxxxxxQ"
_IPC_REQ_SIZE  = _struct.calcsize(_IPC_REQ_FMT)
_IPC_RESP_SIZE = _struct.calcsize(_IPC_RESP_FMT)


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


def _detect_aiesim_device(workdir):
    """Synthesize a Simulator device for aiehlc apps built with --platform sim.

    aiehlc's simulator is script/runsim.sh driving the Vitis aie2pssimmsm model.
    It is detectable rather than declared: aiehlc.sh's write_sim_config drops a
    sim_config.sh next to the build. Returns a device dict tagged
    sim_kind='aiesim', or None when this app has no sim build.
    """
    runsim = os.path.join(_REPO_ROOT, "script", "runsim.sh")
    if not os.path.isfile(runsim):
        return None
    for cand in (workdir, os.path.dirname(workdir.rstrip(os.sep))):
        if cand and os.path.isfile(os.path.join(cand, "sim_config.sh")):
            return {
                "value": "simulator",
                "label": "Simulator (aiesim)",
                "sim_script": runsim,
                "sim_example_dir": cand,
                "sim_kind": "aiesim",
            }
    return None


def _find_up(start, relpath, levels=8):
    """Walk up from `start` looking for `relpath`; return the absolute hit."""
    d = os.path.abspath(start)
    for _ in range(levels):
        cand = os.path.join(d, relpath)
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _detect_ipcsim_device(workdir):
    """Simulator device for a naiebaremetal example (IPC flow), or None.

    The two path checks are runsim_ipc.sh's own preconditions, so a detected
    device can never fail at launch for a missing build.
    """
    example = os.path.dirname(workdir.rstrip(os.sep))
    if not example:
        return None
    if not os.path.isfile(os.path.join(example, "ipc", "build_sim.env")):
        return None
    if not os.path.isdir(os.path.join(example, "Work", "ps", "c_rts", "systemC")):
        return None
    script = _find_up(example, os.path.join("script", "runsim_ipc.sh"))
    if not script:
        return None
    return {
        "value": "simulator",
        "label": "Simulator (IPC)",
        "sim_script": script,
        "sim_example_dir": example,
        "sim_kind": "ipc",
    }


def _detect_sim_device(workdir):
    """Shared by caps() and _load_app_profile() so the app list and /devices
    can never disagree about whether an app has a simulator."""
    return _detect_aiesim_device(workdir) or _detect_ipcsim_device(workdir)


def _sim_unavailable_reason(workdir):
    """Which precondition sim detection missed.

    The Simulator option is offered for every app, so this is the text that
    makes an unavailable one actionable: an option that silently disappears
    reads as "this UI cannot simulate" when the real answer is one build
    command. Flow is picked by the same up-tree probe the detectors use.
    """
    example = os.path.dirname(workdir.rstrip(os.sep)) if workdir else ""
    if example and _find_up(example, os.path.join("script", "runsim_ipc.sh")):
        if not os.path.isfile(os.path.join(example, "ipc", "build_sim.env")):
            return ("no ipc/build_sim.env — not built for sim "
                    "(run build_sim.sh <example>)")
        if not os.path.isdir(os.path.join(example, "Work", "ps", "c_rts",
                                          "systemC")):
            return "no Work/ps/c_rts/systemC — aiecompiler output missing"
        return "IPC sim detection failed"
    return ("no sim_config.sh — not built for sim (rebuild with "
            "aiehlc.sh --platform sim, which writes one beside the bundle)")


def _hwlocal_env(runner):
    path = os.path.join(os.path.dirname(runner), "hwlocal.sh")
    if not os.path.isfile(path):
        return {}
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("export ") or line.startswith("#"):
                    continue
                key, _, val = line[len("export "):].partition("=")
                key, val = key.strip(), val.split("#")[0].strip().strip('"\'')
                if key and val:
                    # hwlocal.sh interpolates ${USERNAME} / ${VEK385IP}.
                    for k2, v2 in list(env.items()):
                        val = val.replace("${%s}" % k2, v2).replace("$" + k2, v2)
                    env[key] = val
    except OSError:
        return {}
    return env


def _detect_vek385_device(workdir):
    """VEK385 device for a naiebaremetal example, or None.

    Carries no board on purpose: the host is chosen per run in the UI, and
    freezing one here is what makes a written debug_ui_config.json go stale.
    """
    example = os.path.dirname(workdir.rstrip(os.sep))
    if not example:
        return None
    boot_bin = os.path.join(example, "build", "vek385.BIN")
    if not os.path.isfile(boot_bin):
        return None
    runner = _find_up(example, os.path.join("src", "tool", "test",
                                            "runhw_vek385.py"))
    if not runner:
        return None
    hw_env = {"VEK385_LOCAL_BIN": boot_bin}
    elf = os.path.join(example, "vek385.elf")
    if os.path.isfile(elf):
        hw_env["VEK385_LOCAL_ELF"] = elf
    return {
        "value": "vek385",
        "label": "vek385",
        "hw_run_script": runner,
        "hw_env": hw_env,
    }


# A declared hw_env is a snapshot of $VEK385IP/hwlocal.sh from whenever
# run_debug_ui.sh wrote the file, so these keys are stale by construction.
_BOARD_KEYS = ("VEK385IP", "AIEDBG_TARGET")


def _board_run_env(base_env, cfg_dev, board_host, workdir):
    """Overlay a hw profile onto base_env, deriving the board from live sources
    only: UI hostname box -> $VEK385IP -> hwlocal.sh. Returns (env, host)."""
    env = dict(base_env)
    for k, v in (cfg_dev.get("hw_env") or {}).items():
        if k in _BOARD_KEYS:
            continue
        env[k] = str(v)
    host = (board_host or "").strip() or _expected_board_host(workdir, base_env)
    if not host:
        return env, ""
    env["VEK385IP"] = host
    # Keeping a target that names another board would program one machine and
    # debug another; matching hosts preserves a non-default hw_server port.
    tgt = base_env.get("AIEDBG_TARGET", "")
    m = re.match(r"xsdb://([^:/]+)", tgt or "")
    env["AIEDBG_TARGET"] = (tgt if m and m.group(1) == host
                            else f"xsdb://{host}:3121")
    env.setdefault("USERNAME", getpass.getuser())
    return env, host


def _device_label(dev):
    """Device label with its own board name stripped — run_debug_ui.sh bakes
    "VEK385 (${VEK385IP})" into the config, which then advertises a board the
    app has nothing to do with. A custom label survives."""
    label = (dev.get("label") or dev.get("value") or "").strip()
    value = dev.get("value") or ""
    host = ((dev.get("hw_env") or {}).get("VEK385IP") or "").strip()
    if host:
        label = re.sub(r"\s*\(\s*" + re.escape(host) + r"\s*\)", "", label)
        label = label.replace(host, "").strip(" -—")
    label = label.strip()
    if dev.get("hw_run_script") and label.lower() == value.lower():
        return value
    return label or value


def _detect_hw_device(workdir):
    """Shared by caps() and _load_app_profile(), as _detect_sim_device is."""
    return _detect_vek385_device(workdir)


# Backend -> what the row means once selected. The aiesim model exposes no debug
# socket, so promising live reads there is a promise the backend cannot keep.
_SIM_KIND_NOTE = {
    "ipc": "IPC simulator — opens a debug socket, so live register reads work",
    "aiesim": ("aiesim (aie2pssimmsm) — console output only, no debug socket "
               "and no live register reads"),
}
_AIESIM_LIVE_ERROR = (
    "live register reads are unavailable for aiesim (aie2pssimmsm): "
    "this simulator exposes no debug socket. DMA/Cores/Events scans and "
    "aiedbg register commands require the AEG IPC simulator or hardware.")


_AIEGDB_LOCAL_VERBS = frozenset((
    "target", "tar", "tile", "channel", "up", "..", "top",
    "where", "info", "pwd", "set", "help", "?", "commands", "cmds",
    "spec", "exit", "quit", "q", "channels", "chans",
))


def _aiegdb_needs_live_transport(line):
    parts = (line or "").strip().split()
    if not parts:
        return False
    verb = parts[0].lower()
    if verb in _AIEGDB_LOCAL_VERBS:
        return verb == "tile" and len(parts) > 1 and parts[1].lower() == "list"
    return True


def _devices_for_ui(st):
    """Rows for the board dropdown, Simulator included unconditionally.

    Detection decides whether a simulator can RUN, never whether the option is
    offered: an option that silently disappears reads as "this UI cannot
    simulate" when the real answer is one build command, and the user has no
    way to discover which. An unavailable row carries the reason instead.
    """
    out = []
    for d in st._extra_devices:
        if not d.get("value") or not d.get("label"):
            continue
        row = {"value": d["value"], "label": _device_label(d),
               "available": True, "reason": ""}
        if d.get("value") == "simulator":
            # st.sim_kind, not d["sim_kind"]: a declared debug_ui_config.json
            # predates the field and carries none, and _load_app_profile
            # defaults it to "ipc". Reading the resolved value is what keeps
            # the label from disagreeing with the backend that will actually
            # run — the same rule caps() follows for the app listing.
            kind = st.sim_kind if st.sim_script else ""
            row["sim_kind"] = kind
            row["live_reads"] = kind == "ipc"
            row["note"] = _SIM_KIND_NOTE.get(kind, "")
        out.append(row)
    if not any(r["value"] == "simulator" for r in out):
        out.append({"value": "simulator", "label": "simulator (not built)",
                    "available": False, "sim_kind": "", "live_reads": False,
                    "note": "",
                    "reason": st.sim_reason})
    return out


def _expected_board_host(workdir, env=None):
    """The board this checkout points at: $VEK385IP, else its hwlocal.sh."""
    host = (env if env is not None else os.environ).get("VEK385IP", "").strip()
    if host:
        return host
    example = os.path.dirname(workdir.rstrip(os.sep))
    runner = _find_up(example, os.path.join("src", "tool", "test",
                                            "runhw_vek385.py")) if example else None
    return _hwlocal_env(runner).get("VEK385IP", "") if runner else ""


_AIEDBG_PATHS_CACHE = None


def _aiedbg_paths():
    """Locate the aiedbg clone, so the assistant can read its docs.

    aiedbg's ~2600 lines of documentation exist ONLY in a git clone — the
    pip-installed package in site-packages ships no .md files — and the
    aiedbg-reference skill is useless without a real path to point at.

    Resolution order, first hit wins:
      1. $AIEHLC_AIEDBG_SRC        — the override ensure_aiedbg.py honours
      2. <repo>/thirdparty/aiedbg  — where ensure_aiedbg.py clones on first run
      3. the installed dist's direct_url.json — covers a machine that was set up
         against a clone living somewhere else entirely, which is otherwise
         invisible to us
    A candidate only counts if it actually holds the docs, so a half-made
    directory does not get advertised as the reference.
    """
    global _AIEDBG_PATHS_CACHE
    # Scans dist metadata; ~150 ms. /source needs it per request, so
    # memoise for the process rather than re-resolving a static layout.
    if _AIEDBG_PATHS_CACHE is not None:
        return _AIEDBG_PATHS_CACHE
    cands = []
    override = os.environ.get("AIEHLC_AIEDBG_SRC", "").strip()
    if override:
        cands.append(os.path.abspath(override))
    cands.append(os.path.join(_REPO_ROOT, "thirdparty", "aiedbg"))
    try:
        import importlib.metadata as _md
        for dist in _md.distributions():
            if (dist.metadata.get("Name") or "").lower() != "aiedbg":
                continue
            raw = dist.read_text("direct_url.json")
            if not raw:
                continue
            url = (json.loads(raw).get("url") or "")
            if url.startswith("file://"):
                cands.append(unquote(url[len("file://"):]))
    except Exception:  # metadata layout varies; never block startup on it
        pass

    src = ""
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "CLAUDE.md")):
            src = c
            break
    _AIEDBG_PATHS_CACHE = {"src": src, "bin": shutil.which("aiedbg") or ""}
    return _AIEDBG_PATHS_CACHE


# ── source viewer ────────────────────────────────────────────────────────────
# Serves a file from a small allowlist of roots, Pygments-highlighted, so the UI
# can open a source reference the LLM or the aiegdb console printed.

_SRC_MAX_BYTES   = 4 * 1024 * 1024
_SRC_FULL_LINES  = 2000      # whole file at or below this (~940 KB of HTML)
_SRC_WINDOW      = 400       # lines each side of the target beyond that
_SRC_STYLE       = "one-dark"
_SRC_BINARY_EXTS = {".elf", ".o", ".a", ".so", ".bin", ".pdi", ".png", ".jpg", ".pdf"}
# get_lexer_for_filename picks POVRay for .inc, and has nothing for the rest.
_SRC_LEXER_OVERRIDE = {".mlir": "text", ".bcf": "text", ".td": "text",
                       ".log": "text", ".inc": "c"}
_SRC_CACHE = collections.OrderedDict()
_SRC_CACHE_MAX = 16
_SRC_CACHE_LOCK = threading.Lock()
_SRC_CSS_CACHE = None


def _source_roots(st):
    """Ordered [(name, realpath)] that /source may read under.

    Recomputed per request: select_app() reassigns st.workdir, so a cached list
    would keep serving a de-selected app. Callers that need it more than once
    per request must hoist it — _aiedbg_paths() alone costs ~150 ms.
    """
    ap = st.app_paths()
    cands = [("app", ap.get("app_dir")), ("bundle", ap.get("bundle_dir")),
             ("work", ap.get("work_dir"))]
    if st.registry is not None:          # None until main() builds it
        for a in st.registry.list():
            cands.append(("app", a.path))
            if os.path.basename(a.path) == "worklocal":
                cands.append(("app", os.path.dirname(a.path)))
    cands.append(("sim", getattr(st, "sim_example_dir", None)))
    cands.append(("repo", _REPO_ROOT))
    cands.append(("aiedbg", _aiedbg_paths().get("src")))

    banned = {"/", "/home", "/tmp", "/usr", "/etc", "/var",
              os.path.realpath(os.path.expanduser("~"))}
    out = []
    for name, p in cands:
        if not p:
            continue
        rp = os.path.realpath(p)
        # A shallow root would hand out most of the filesystem.
        if rp in banned or len([x for x in rp.split(os.sep) if x]) < 2:
            continue
        if not os.path.isdir(rp) or any(rp == e for _n, e in out):
            continue
        out.append((name, rp))
    # Nested roots are kept on purpose. They are also the join bases for a
    # relative citation, and dropping aout/worklocal because it sits under the
    # repo root would make `worklocal/host.cc` unresolvable. It widens nothing:
    # a root inside another root reaches no file the outer one did not already.
    return out


def _path_within(rp, root):
    """True when rp is root or below it. commonpath, not startswith: the latter
    says /a/bc is inside /a/b. Callers must realpath rp FIRST — that is what
    defeats `..` and a symlink pointing out of the tree."""
    try:
        return os.path.commonpath([rp, root]) == root
    except ValueError:      # different drives / one is relative
        return False


_SRC_NAME_IDX = (None, None, 0.0)     # (cache key, index, built_at)
_SRC_IDX_TTL = 20.0
_SRC_WALK_SKIP = {".git", "__pycache__", "node_modules", ".Xil", "ipc",
                  "aiesimulator_output", "Release", "Debug"}
_SRC_WALK_EXT = {".cc", ".cpp", ".cxx", ".c", ".h", ".hpp", ".py", ".md",
                 ".mlir", ".sh", ".td", ".inc", ".bcf", ".json", ".txt"}


def _source_walk(root, idx, max_depth=6, cap=4000):
    """Add basename -> abspath for source-like files under root, first wins."""
    root = os.path.realpath(root)
    if not os.path.isdir(root):
        return
    n = 0
    for dirpath, dirnames, files in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames
                       if d not in _SRC_WALK_SKIP and not d.startswith(".")]
        for f in files:
            if os.path.splitext(f)[1].lower() in _SRC_WALK_EXT:
                idx.setdefault(f, os.path.join(dirpath, f))
                n += 1
                if n >= cap:
                    return


def _source_name_index(st):
    """basename -> abspath, so a bare `graph.cpp` from the LLM resolves.

    Two tiers, view first so a generated file the tools actually cite wins over
    a same-named file elsewhere in the tree:
      1. paths the view names (host.cc, kernel.cc, the .bcf, per-tile code_file)
      2. a bounded walk of the app's own dirs — the app root, the bundle and
         Work/ — which is what makes hand-written sources like `src/graph.cpp`
         clickable. Joining a bare name against a root only ever found files
         sitting directly at that root.

    The repo root and the aiedbg clone are deliberately NOT walked: too large,
    and citations into them are repo-relative rather than bare. Measured at
    ~4 ms for a full naiebaremetal example, cached for _SRC_IDX_TTL because a
    directory tree has no single mtime to key on.
    """
    global _SRC_NAME_IDX
    jp = st.json_path()
    ap = st.app_paths()
    try:
        key = (jp, os.path.getmtime(jp), ap.get("app_dir"), ap.get("work_dir"))
    except OSError:
        key = (jp, 0, ap.get("app_dir"), ap.get("work_dir"))
    if _SRC_NAME_IDX[0] == key and (time.time() - _SRC_NAME_IDX[2]) < _SRC_IDX_TTL:
        return _SRC_NAME_IDX[1]

    idx = {}
    try:
        with open(jp) as f:
            view = json.load(f)
    except (OSError, ValueError):
        return idx

    def add(p):
        if p and os.path.isabs(p):
            idx.setdefault(os.path.basename(p), p)

    src = view.get("source") or {}
    add(src.get("host_cc"))
    add(src.get("provenance"))
    for sect in ("kernel", "bcf", "dfschedule_ir"):
        sub = view.get(sect) or {}
        add(sub.get("path"))
        add((sub.get("source") or {}).get("path"))
    for t in view.get("tiles") or []:
        add(t.get("code_file"))
        for c in t.get("dma_channels") or []:
            add(c.get("code_file"))

    for d in (ap.get("app_dir"), ap.get("bundle_dir"), ap.get("work_dir")):
        if d:
            _source_walk(d, idx)
    _SRC_NAME_IDX = (key, idx, time.time())
    return idx


# ---------------------------------------------------------------------------
# App source manifest
#
# app_paths() tells the assistant WHERE the app is; it never said what was in
# it. So "read the kernel that runs on this tile" meant guessing a filename, and
# the assistant fell back on describing the compiled schedule in the abstract —
# accurate about registers, silent about the code that programmed them. This
# enumerates the app's own hand-written sources, keeps them distinct from the
# compiler's output (which the static tools already quote), and maps each kernel
# the schedule names to the file and line that define it. That map is the link
# that turns "tile (5,3) is stalled" into a line of code to open.
# ---------------------------------------------------------------------------

_APP_SRC_EXT = {".cc", ".cpp", ".cxx", ".c"}
_APP_HDR_EXT = {".h", ".hpp", ".hh"}
_APP_BUILD_NAMES = {"build.sh", "compile.sh", "run.sh", "Makefile", "makefile",
                    "CMakeLists.txt"}
_APP_BUILD_EXT = {".bif"}
# Output trees, not sources: build/Release/Debug hold objects, Work, worklocal,
# aiesimulator_output and .Xil are compiler and simulator output, and arch/ipc
# are vendor scaffolding copied verbatim into every example.
_APP_SKIP_DIRS = {"build", "Work", "worklocal", "aiesimulator_output", ".Xil",
                  "Release", "Debug", "arch", "ipc", "reports", "temp",
                  "debugcache", "__pycache__", "node_modules"}
# The aiehlc flow copies and rewrites its input into aout/, so for that flow
# neither location nor extension separates a hand-written kernel from an emitted
# one. Both banners are literals from the emitters — the "Auto-generated by
# aiehlc" header block, and src/llvm/aiehlc.cc's Clang-stub preamble.
_APP_GEN_MARKERS = ("Auto-generated by", "Stub type declarations for Clang")
_APP_SRC_CAP   = 60          # entries surfaced per group
_APP_DEF_CAP   = 1500000     # bytes scanned looking for kernel definitions
_APP_SRC_TTL   = 20.0
_APP_SRC_CACHE = (None, None, 0.0)


def _app_is_generated(path):
    """True when the file's own banner says a tool wrote it."""
    try:
        with open(path, errors="replace") as f:
            head = f.read(400)
    except OSError:
        return False
    return any(m in head for m in _APP_GEN_MARKERS)


def _app_walk_sources(root, max_depth=5, cap=400):
    """[abspath] of hand-written sources and headers under root, walk order."""
    root = os.path.realpath(root or "")
    if not os.path.isdir(root):
        return []
    hits = []
    for dirpath, dirnames, files in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames
                       if d not in _APP_SKIP_DIRS and not d.startswith(".")]
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in (_APP_SRC_EXT | _APP_HDR_EXT):
                hits.append(os.path.join(dirpath, f))
                if len(hits) >= cap:
                    return hits
    return hits


def _app_entry_source(st):
    """The aiehlc flow's --runtime-source-file, or "".

    Worth a special case because that flow's app_dir is a build output directory
    (aout/) holding only generated code, while the source it was built from lives
    anywhere in the tree — walking the app dir can never reach it. Two records,
    both written by aiehlc.sh: app_source.txt (always) and write_sim_config's
    HOST_SRC (sim builds only, and kept as the fallback for builds that predate
    app_source.txt).
    """
    dirs = [st.workdir, os.path.dirname((st.workdir or "").rstrip(os.sep))]
    for cand in dirs:
        marker = os.path.join(cand or "", "app_source.txt")
        try:
            with open(marker, errors="replace") as f:
                path = f.read().strip()
        except OSError:
            path = ""
        if path and os.path.isfile(path):
            return path
    for cand in dirs:
        cfg = os.path.join(cand or "", "sim_config.sh")
        if not os.path.isfile(cfg):
            continue
        try:
            with open(cfg, errors="replace") as f:
                m = re.search(r'^HOST_SRC="([^"]*)"', f.read(), re.M)
        except OSError:
            continue
        if m and os.path.isfile(m.group(1)):
            return m.group(1)
    return ""


def _app_kernel_names(view):
    """Kernel names the compiled schedule names, first seen first.

    Both sources matter. Per-tile `high_level.kernel` is what the UI shows and is
    what the user will ask about; `kernel.function` is the function the frontend
    actually lifted, and for the aiehlc flow it is the only one that names a
    function in the user's file (the per-tile names there are role labels like
    `dskernel_receiver`, which no source defines).
    """
    out = []
    for t in view.get("tiles") or []:
        name = ((t.get("high_level") or {}).get("kernel") or "").strip()
        if name and name not in out:
            out.append(name)
    fn = ((view.get("kernel") or {}).get("function") or "").strip()
    if fn and fn not in out:
        out.append(fn)
    return out


def _app_def_line(text, pat):
    """Line of the most definition-like match of pat in text, or 0.

    A heuristic, and labelled as one wherever it surfaces: a match whose line
    ends in `;` is a prototype or a call and is only kept as a fallback, while
    one that does not is a signature with a body following. Good enough to open
    the right file at roughly the right place, which is the whole job.
    """
    fallback = 0
    for m in pat.finditer(text):
        ls = text.rfind("\n", 0, m.start()) + 1
        le = text.find("\n", m.start())
        line = text[ls:le if le != -1 else len(text)].strip()
        n = text.count("\n", 0, ls) + 1
        if line.endswith(";"):
            fallback = fallback or n
        elif not line.startswith(("return", "if", "for", "while", "}")):
            return n
    return fallback


def _app_kernel_defs(names, paths):
    """{kernel name: "abspath:line"} for the kernels the schedule names."""
    if not names:
        return {}
    pats = {n: re.compile(r"(?<![\w:])%s\s*\(" % re.escape(n)) for n in names}
    found, budget = {}, _APP_DEF_CAP
    for path in paths:
        if len(found) == len(names) or budget <= 0:
            break
        if os.path.splitext(path)[1].lower() not in _APP_SRC_EXT:
            continue
        try:
            with open(path, errors="replace") as f:
                text = f.read(budget)
        except OSError:
            continue
        budget -= len(text)
        for name, pat in pats.items():
            if name in found:
                continue
            line = _app_def_line(text, pat)
            if line:
                found[name] = "%s:%d" % (path, line)
    return found


def _app_build_files(app_dir, cap=8):
    """Build entry points at the app root — how the app was compiled."""
    out = []
    try:
        entries = sorted(os.listdir(app_dir or ""))
    except OSError:
        return out
    for f in entries:
        path = os.path.join(app_dir, f)
        if not os.path.isfile(path):
            continue
        if f in _APP_BUILD_NAMES or os.path.splitext(f)[1].lower() in _APP_BUILD_EXT:
            out.append(path)
        if len(out) >= cap:
            break
    return out


def _app_generated_files(st, view):
    """[(path, what)] — compiler output, with what each one is good for."""
    out = []

    def add(path, what):
        if path and os.path.isfile(path) and all(path != p for p, _w in out):
            out.append((path, what))

    add((view.get("source") or {}).get("host_cc"),
        "emitted host program — the XAie calls that program DMA, locks, routing")
    add((view.get("kernel") or {}).get("path"),
        "emitted kernel wrapper — window and lock plumbing around the kernel body")
    add((view.get("bcf") or {}).get("path"),
        "buffer map — symbol to tile-memory address")
    add((view.get("dfschedule_ir") or {}).get("path"),
        "dfschedule MLIR the host program was emitted from")
    add(os.path.join(st.workdir or "", "routing.cc"),
        "emitted stream-switch routing")
    return out


def _app_source_manifest(st):
    """The app's own sources, the compiler's output, and the map between them.

    Cached for _APP_SRC_TTL — a directory tree has no single mtime to key on,
    and this is rebuilt on every backend_status write.
    """
    global _APP_SRC_CACHE
    ap = st.app_paths()
    key = (ap.get("app_dir"), ap.get("bundle_dir"), st.json_path())
    if _APP_SRC_CACHE[0] == key and (time.time() - _APP_SRC_CACHE[2]) < _APP_SRC_TTL:
        return _APP_SRC_CACHE[1]

    try:
        with open(st.json_path()) as f:
            view = json.load(f)
    except (OSError, ValueError):
        view = {}

    app_dir = ap.get("app_dir") or ""
    walked = [p for p in _app_walk_sources(app_dir) if not _app_is_generated(p)]
    entry = _app_entry_source(st)
    if entry and entry not in walked:
        walked.insert(0, entry)

    def group(exts):
        return [p for p in walked if os.path.splitext(p)[1].lower() in exts]

    names = _app_kernel_names(view)
    defs = _app_kernel_defs(names, walked)
    man = {
        "app_dir": app_dir,
        "entry": entry,
        "sources": group(_APP_SRC_EXT)[:_APP_SRC_CAP],
        "headers": group(_APP_HDR_EXT)[:_APP_SRC_CAP],
        "build": _app_build_files(app_dir),
        "kernels": [{"name": n, "at": defs.get(n, "")} for n in names],
        "generated": [{"path": p, "what": w}
                      for p, w in _app_generated_files(st, view)],
        "truncated": len(group(_APP_SRC_EXT)) > _APP_SRC_CAP,
    }
    _APP_SRC_CACHE = (key, man, time.time())
    return man


def _fmt_app_sources(man, rel_to=None):
    """Render a manifest as prompt/tool text. Paths are shown relative to the
    app dir where possible — the absolute prefix repeats on every line and buys
    nothing, and both the browser and _resolve_source take a relative path."""
    def show(path):
        for base in (rel_to, _REPO_ROOT):
            if base and path.startswith(base.rstrip(os.sep) + os.sep):
                return path[len(base.rstrip(os.sep)) + 1:]
        return path

    out = []
    if man.get("entry"):
        out.append("Entry source (what this app was built from):")
        out.append("  %s" % man["entry"])
    kernels = [k for k in man.get("kernels") or [] if k.get("name")]
    if kernels:
        out.append("Kernels the schedule runs, and where each is defined:")
        for k in kernels:
            out.append("  %-20s %s" % (k["name"], show(k["at"]) if k.get("at")
                                       else "(definition not found by name)"))
    if man.get("sources"):
        out.append("Application sources:")
        out.extend("  %s" % show(p) for p in man["sources"])
        if man.get("truncated"):
            out.append("  … list truncated; walk %s for the rest" % man.get("app_dir"))
    if man.get("headers"):
        out.append("Headers:")
        out.extend("  %s" % show(p) for p in man["headers"])
    if man.get("build"):
        out.append("Build:")
        out.extend("  %s" % show(p) for p in man["build"])
    if man.get("generated"):
        out.append("Generated by the compiler (NOT hand-written — describes what "
                   "the tools built, not what the developer asked for):")
        for g in man["generated"]:
            out.append("  %-28s %s" % (show(g["path"]), g["what"]))
    # Said explicitly, because the silent version of this is the bug: with only
    # the generated group rendered, "no application sources" is indistinguishable
    # from "did not look", and the reader fills the gap with a guessed filename.
    if not man.get("sources") and not man.get("entry"):
        out.append("")
        out.append("NO application sources were found under %s. Everything above is "
                   "compiler output. Do not guess where the source might be — say "
                   "this and ask the user for the path. (An aiehlc build records it "
                   "in worklocal/app_source.txt; a build older than that feature has "
                   "no record of it at all.)" % (man.get("app_dir") or "the app dir"))
    return "\n".join(out)


def _resolve_source(st, raw, roots=None):
    """(realpath, root_name) or (None, reason). Accepts an absolute path, a path
    relative to any root, or a bare basename present in the view."""
    raw = (raw or "").strip()
    if not raw or "\x00" in raw or len(raw) > 4096:
        return None, "bad path"
    if raw.startswith("./"):
        raw = raw[2:]

    if roots is None:
        roots = _source_roots(st)
    cands = []
    if os.path.isabs(raw):
        cands.append(raw)
    else:
        if os.sep not in raw:
            hit = _source_name_index(st).get(raw)
            if hit:
                cands.append(hit)
        cands.extend(os.path.join(root, raw) for _n, root in roots)

    for cand in cands:
        rp = os.path.realpath(cand)       # BEFORE containment — see _path_within
        if not os.path.isfile(rp):
            continue
        for name, root in roots:
            if _path_within(rp, root):
                return rp, name
    return None, "outside"


def _source_lexer(rp):
    """A lexer for rp, never guess_lexer (slow, and wrong on IR dumps)."""
    ext = os.path.splitext(rp)[1].lower()
    alias = _SRC_LEXER_OVERRIDE.get(ext)
    try:
        if alias:
            return get_lexer_by_name(alias, stripnl=False)
        return get_lexer_for_filename(os.path.basename(rp), stripnl=False)
    except ClassNotFound:
        return get_lexer_by_name("text", stripnl=False)


def _source_css():
    """Pygments style defs with every selector forced under .srcview.

    get_style_defs(prefix) leaks unscoped rules — `pre { line-height:125% }`,
    `span.linenos`, `td.linenos .normal|.special` — which would otherwise
    restyle pre.code, .bdpretty, .md-block and the aiegdb console app-wide.
    """
    global _SRC_CSS_CACHE
    if _SRC_CSS_CACHE is not None:
        return _SRC_CSS_CACHE
    if not _HAVE_PYGMENTS:
        _SRC_CSS_CACHE = ""
        return _SRC_CSS_CACHE
    out = []
    for line in HtmlFormatter(style=_SRC_STYLE).get_style_defs(".srcview").splitlines():
        head, brace, tail = line.partition("{")
        if not brace:
            out.append(line)
            continue
        sels = [s.strip() for s in head.split(",") if s.strip()]
        sels = [s if s.startswith(".srcview") else ".srcview " + s for s in sels]
        out.append("%s {%s" % (", ".join(sels), tail))
    _SRC_CSS_CACHE = "\n".join(out)
    return _SRC_CSS_CACHE


def _source_plain(text, start):
    """Monochrome fallback with the same DOM shape Pygments produces, so the
    client's #SL-<n> highlight and scroll work without Pygments installed."""
    rows = []
    for i, ln in enumerate(text.splitlines(), start):
        rows.append('<span id="SL-%d"><span class="linenos">%d</span>%s\n</span>'
                    % (i, i, html.escape(ln)))
    return '<div class="srcview"><pre>%s</pre></div>' % "".join(rows)


def _source_render(rp, text, start):
    """Highlighted HTML. linespans (not hl_lines) so the render is independent
    of the requested line and therefore cacheable; hl_lines is also
    slice-relative while the gutter follows linenostart, which silently
    off-by-ones any windowed file."""
    if not _HAVE_PYGMENTS:
        return _source_plain(text, start), "plain text"
    lexer = _source_lexer(rp)
    fmt = HtmlFormatter(style=_SRC_STYLE, cssclass="srcview",
                        linenos="inline", linespans="SL", linenostart=start)
    return _pyg_highlight(text, lexer, fmt), lexer.name


def _serve_source(st, q):
    """(payload, http_code) for GET /source. Errors never echo the input path."""
    raw = (q.get("path") or [""])[0]
    try:
        line = int((q.get("line") or ["0"])[0] or 0)
    except (TypeError, ValueError):
        line = 0
    if not raw:
        return {"error": "path required"}, 400

    roots = _source_roots(st)          # hoisted: ~150 ms, needed 2-3 times below
    rp, why = _resolve_source(st, raw, roots)
    if rp is None:
        if why == "bad path":
            return {"error": "bad path"}, 400
        # Names only — an absolute path here would leak the layout to a caller
        # who just proved they cannot read it. Deduped so the count of
        # registered apps is not inferable either.
        return {"error": "not found, or outside the readable roots",
                "roots": sorted({n for n, _r in roots})}, 404

    if os.path.splitext(rp)[1].lower() in _SRC_BINARY_EXTS:
        return {"error": "binary file"}, 415
    try:
        stt = os.stat(rp)
        if stt.st_size > _SRC_MAX_BYTES:
            return {"error": "file too large", "bytes": stt.st_size,
                    "limit": _SRC_MAX_BYTES}, 413
        with open(rp, "rb") as f:
            head = f.read(8192)
            if b"\x00" in head:
                return {"error": "binary file"}, 415
            body = head + f.read()
    except OSError:
        return {"error": "unreadable"}, 404

    text = body.decode("utf-8", errors="replace")
    lines = text.splitlines()
    n = len(lines)
    first, last, truncated = 1, n, False
    if n > _SRC_FULL_LINES:
        centre = line if 1 <= line <= n else 1
        first = max(1, centre - _SRC_WINDOW)
        last = min(n, centre + _SRC_WINDOW)
        truncated = True
    span = "\n".join(lines[first - 1:last])

    key = (rp, stt.st_mtime_ns, stt.st_size, first, last, _SRC_STYLE)
    with _SRC_CACHE_LOCK:
        hit = _SRC_CACHE.get(key)
        if hit is not None:
            _SRC_CACHE.move_to_end(key)
    if hit is None:
        try:
            hit = _source_render(rp, span, first)
        except Exception as e:
            return {"error": "highlight failed: %s" % type(e).__name__}, 500
        with _SRC_CACHE_LOCK:
            _SRC_CACHE[key] = hit
            while len(_SRC_CACHE) > _SRC_CACHE_MAX:
                _SRC_CACHE.popitem(last=False)
    body_html, lang = hit

    rel = rp
    for _name, root in roots:
        if _path_within(rp, root):
            rel = os.path.relpath(rp, root)
            break
    return {"ok": True, "path": rp, "display": os.path.basename(rp), "rel": rel,
            "root": why, "lang": lang, "lines": n, "first": first, "last": last,
            "truncated": truncated, "line": line or None, "html": body_html,
            "css": _source_css(), "css_ver": "%s/%s" % (_SRC_STYLE, _HAVE_PYGMENTS)}, 200


def _read_skill_frontmatter(path):
    """Pull (name, description) out of a SKILL.md YAML header.

    Deliberately a hand-rolled scan rather than a YAML dependency: the header is
    a fixed two-key shape and the daemon is stdlib-only. Returns (None, None) if
    the file does not match the expected shape, so a malformed skill is skipped
    rather than breaking prompt construction.
    """
    try:
        with open(path) as f:
            text = f.read(8192)
    except OSError:
        return None, None
    # Header is the first `---` fenced block; a copyright comment may precede it.
    start = text.find("---")
    if start < 0:
        return None, None
    end = text.find("\n---", start + 3)
    if end < 0:
        return None, None
    block = text[start + 3:end]
    name = desc = None
    key = None
    for line in block.splitlines():
        m = re.match(r"^(name|description):\s*(.*)$", line)
        if m:
            key = m.group(1)
            val = m.group(2).strip()
            if key == "name":
                name = val
            else:
                desc = val
        elif key == "description" and line.startswith((" ", "\t")):
            # Folded continuation line of a multi-line description.
            desc = (desc + " " + line.strip()).strip()
        elif line.strip():
            key = None
    return name, desc


def _llm_tool_args(inp, max_keys=8, max_val=80):
    """Compact tool arguments for the browser's tool row.

    Truncates each VALUE rather than the finished string, so the result is
    always parseable JSON — the browser renders one labelled chip per key and
    falls back to raw text when a parse fails.
    """
    out = {}
    for k, v in list((inp or {}).items())[:max_keys]:
        if isinstance(v, bool) or isinstance(v, (int, float)) or v is None:
            out[k] = v
            continue
        if isinstance(v, str):
            s = " ".join(v.split())
        else:
            try:
                s = json.dumps(v, separators=(",", ":"))
            except (TypeError, ValueError):
                s = str(v)
        out[k] = s if len(s) <= max_val else s[:max_val - 1] + "…"
    try:
        return json.dumps(out, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def _llm_skills():
    """Discover skills written for the embedded LLM. Returns [(name, desc, path)]."""
    out = []
    try:
        entries = sorted(os.listdir(_LLM_SKILLS_DIR))
    except OSError:
        return out
    for d in entries:
        path = os.path.join(_LLM_SKILLS_DIR, d, "SKILL.md")
        if not os.path.isfile(path):
            continue
        name, desc = _read_skill_frontmatter(path)
        if name and desc:
            out.append((name, desc, path))
    return out


def _resolve_work_dir(workdir):
    """Locate the aiecompiler Work/ directory for a given app workdir.

    For naiebaremetal apps, workdir is <example>/worklocal and Work/ sits at
    <example>/Work.  For aiehlc apps there is no Work/ at all (the compiler
    emits provenance JSONs directly).  Returns the absolute path to Work/ if
    found, otherwise None.
    """
    # Direct child: aiehlc-style layout (rare but kept for symmetry)
    direct = os.path.join(workdir, "Work")
    if os.path.isdir(direct):
        return direct
    # Sibling: naiebaremetal worklocal sits beside Work/
    if os.path.basename(workdir) == "worklocal":
        sibling = os.path.join(os.path.dirname(workdir), "Work")
        if os.path.isdir(sibling):
            return sibling
    return None


def _resolve_default_elf(workdir):
    """Pick the project's default host ELF when --elf is omitted.

    Search order:
      1. <workdir>/build/host              — aiehlc tiling build output
      2. parent of workdir: *.elf          — naiebaremetal: vek385.elf sits beside Work/
      3. aout/main.elf                     — global aiehlc fallback
      4. aout/worklocal/build/host         — aiehlc worklocal fallback

    Returns an absolute path, or None if nothing is found.
    """
    candidates = [
        os.path.join(workdir, "build", "host"),
    ]
    # naiebaremetal style: workdir is <example>/worklocal, ELF is in <example>/
    parent = os.path.dirname(workdir)
    if os.path.basename(workdir) == "worklocal" and os.path.isdir(parent):
        for name in ("vek385.elf", "main.elf"):
            candidates.append(os.path.join(parent, name))
        # also accept any single *.elf in parent
        try:
            elfs = [f for f in os.listdir(parent) if f.endswith(".elf")]
            if len(elfs) == 1:
                candidates.append(os.path.join(parent, elfs[0]))
        except OSError:
            pass
    candidates += [
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
import switch_scan  # noqa: E402
import switch_reconstruct  # noqa: E402
import live_scan_summary  # noqa: E402
import schedule_view  # noqa: E402  (render_html for server-side app injection)
import work2provenance  # noqa: E402  (auto-generate worklocal/ from Work/)

# pexpect drives the interactive ssh -> systest -> xsdb -> hw_server recovery
# session (see DebugState.start_hwserver_async). Optional: without it the auto-start
# path degrades to the manual "start hw_server" hint.
try:
    import pexpect  # noqa: E402
except ImportError:
    pexpect = None

_SSH_PEXPECT_DIR = os.path.join(_REPO_ROOT, "script", "test")
if _SSH_PEXPECT_DIR not in sys.path:
    sys.path.insert(0, _SSH_PEXPECT_DIR)
from ssh_pexpect import expect_shell_after_ssh, ssh_command  # noqa: E402

# Vitis settings + xsdb fallback for the hw_server-launch recovery path, kept in
# sync with script/test/apppaltest.py and script/test/connecttest.py.
VITIS_SETTINGS = ("/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/"
                  "Vitis/settings64.sh")
XSDB_ALT_PATH = ("/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/"
                 "Vitis/bin/xsdb")


# Written for the LLM as much as the user: the model must not fill the gap with
# assumptions about board state when a read is refused.
_NO_SESSION_MSG = "not connected; press Connect first"


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


class App:
    """One loadable app: a workdir bundle produced by schedule_view.py.

    Both producer flows land here — aiehlc's compiler passes emit the provenance
    JSONs directly, naiebaremetal's work2provenance.py derives them from an
    aiecompiler Work/ dir — so an app is identified purely by the bundle it
    contains, not by which repo built it.
    """

    def __init__(self, path, label=None):
        self.path = os.path.abspath(path)
        base = os.path.basename(self.path)
        parent = os.path.basename(os.path.dirname(self.path))
        self.id = parent if base == "worklocal" and parent else base
        self.label = label or self.id
        self._view = None
        self._view_mtime = None

    @property
    def view_json(self):
        return os.path.join(self.path, "schedule_view.json")

    @property
    def mtime(self):
        try:
            return os.path.getmtime(self.view_json)
        except OSError:
            return 0.0

    def caps(self):
        """Per-app capabilities. The two producer flows differ only in optional
        bundle files, so consumers must check rather than assume. Shares
        _detect_sim_device with _load_app_profile so the label in the app list
        cannot disagree with the devices actually offered in /devices."""
        ui_cfg = _load_ui_config(self.path)
        devs = list(ui_cfg.get("extra_devices", []) or [])
        if not any(d.get("value") == "simulator" for d in devs):
            auto = _detect_sim_device(self.path)
            if auto:
                devs.append(auto)
        if not any(d.get("hw_run_script") for d in devs):
            auto = _detect_hw_device(self.path)
            if auto:
                devs.append(auto)
        has_sim = any(d.get("sim_script") for d in devs)
        stale = ""
        declared_hw = next((d for d in (ui_cfg.get("extra_devices") or [])
                            if d.get("hw_run_script")), None)
        if declared_hw:
            named = (declared_hw.get("hw_env") or {}).get("VEK385IP", "")
            expected = _expected_board_host(self.path)
            if named and expected and named != expected:
                stale = (f"debug_ui_config.json names VEK385IP={named} — ignored; "
                         f"the board comes from the UI box / $VEK385IP / "
                         f"hwlocal.sh (currently {expected})")
        return {
            "has_ui_config": bool(ui_cfg),
            "has_backend_status": os.path.isfile(
                os.path.join(self.path, "backend_status.json")),
            "has_sim": has_sim,
            "has_hw": any(d.get("hw_run_script") for d in devs),
            "hw_stale": stale,
            "no_sim_reason": "" if has_sim else self._no_sim_reason(),
        }

    def _no_sim_reason(self):
        """Which precondition sim detection missed, so a `view-only` row in the
        startup listing is actionable. Shared with the Simulator dropdown row,
        which is now always present and needs the same sentence."""
        return _sim_unavailable_reason(self.path)

    def load_view(self):
        """Parsed schedule_view.json, cached until the file changes on disk so a
        rebuild is picked up on the next page load."""
        m = self.mtime
        if self._view is None or self._view_mtime != m:
            with open(self.view_json) as f:
                self._view = json.load(f)
            self._view_mtime = m
        return self._view

    def info(self, current=False):
        d = {"id": self.id, "label": self.label, "path": self.path,
             "mtime": self.mtime, "current": current}
        d.update(self.caps())
        return d


def _try_generate_worklocal(example_dir):
    """If example_dir has a Work/ but no up-to-date worklocal/schedule_view.json,
    run work2provenance + schedule_view to produce it.
    Returns the worklocal path on success (whether generated or already current),
    None on failure."""
    work_dir = os.path.join(example_dir, "Work")
    worklocal = os.path.join(example_dir, "worklocal")
    svjson = os.path.join(worklocal, "schedule_view.json")
    html = os.path.join(worklocal, "host_schedule.html")
    if not os.path.isdir(work_dir):
        return None
    # Use the mtime of aie_control_config.json as a cheap proxy for Work/ freshness —
    # avoids walking the whole tree on every scan.
    control_cfg = os.path.join(work_dir, "ps", "c_rts", "aie_control_config.json")
    work_mtime = (os.path.getmtime(control_cfg)
                  if os.path.isfile(control_cfg) else 0)
    sv_mtime = os.path.getmtime(svjson) if os.path.isfile(svjson) else 0
    if sv_mtime >= work_mtime and sv_mtime > 0:
        if not os.path.isfile(html):
            with open(svjson) as f:
                schedule_view.write_html(json.load(f), html)
            print(f"[AppRegistry]   wrote {html}")
        return worklocal  # already up-to-date
    try:
        print(f"[AppRegistry] generating worklocal for {os.path.basename(example_dir)} ...")
        work2provenance.work_to_provenance(work_dir, worklocal)
        view = schedule_view.build_view(worklocal)
        schedule_view.write_code_cache(view, workdir=worklocal)
        sv_path = os.path.join(worklocal, "schedule_view.json")
        with open(sv_path, "w") as _f:
            json.dump(view, _f, indent=2)
        print(f"[AppRegistry]   wrote {sv_path}")
        schedule_view.write_html(view, html)
        print(f"[AppRegistry]   wrote {html}")
        return worklocal
    except Exception as e:
        print(f"[AppRegistry] warning: could not generate worklocal for "
              f"{os.path.basename(example_dir)}: {e}", file=sys.stderr)
        return None


class AppRegistry:

    def __init__(self, explicit=None, roots=None, auto_roots=None):
        self._apps = {}
        self._order = []
        # Apps the user actually named (positional workdir, then --app), in the
        # order given. default() prefers these; discovered ones are a fallback.
        self._explicit = []
        for spec in (explicit or []):
            path, _, label = spec.partition("=")
            bundle = self._resolve_explicit(path)
            app = self._add(bundle, label or None) if bundle else None
            if app is None:
                print(f"[AppRegistry] warning: {path} is neither a provenance "
                      f"bundle nor an app directory containing Work/",
                      file=sys.stderr)
            if app is not None and app.id not in self._explicit:
                self._explicit.append(app.id)
        for root in (roots or []):
            self._scan(root)
        for root in (auto_roots or []):
            self._scan(root)

    @staticmethod
    def _resolve_explicit(path):
        path = os.path.abspath(path)
        if os.path.isfile(os.path.join(path, "schedule_view.json")):
            return path
        if os.path.isdir(os.path.join(path, "Work")):
            return _try_generate_worklocal(path)
        worklocal = os.path.join(path, "worklocal")
        if os.path.isfile(os.path.join(worklocal, "schedule_view.json")):
            return worklocal
        return None

    def _add(self, path, label=None):
        path = os.path.abspath(path)
        if not os.path.isfile(os.path.join(path, "schedule_view.json")):
            return None
        app = App(path, label)
        if app.id in self._apps and self._apps[app.id].path != path:
            n = 2
            while f"{app.id}-{n}" in self._apps:
                n += 1
            app.id = f"{app.id}-{n}"
        if app.id not in self._apps:
            self._apps[app.id] = app
            self._order.append(app.id)
        return self._apps[app.id]

    def _scan(self, root, max_depth=4):
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            return
        root_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - root_depth > max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "debugcache", "build", "Work")]
            if "schedule_view.json" in filenames:
                self._add(dirpath)
                dirnames[:] = []
            elif "Work" in os.listdir(dirpath):
                # aiecompiler example without a worklocal yet — auto-generate
                wl = _try_generate_worklocal(dirpath)
                if wl:
                    self._add(wl)
                dirnames[:] = []

    def list(self):
        """Apps newest-first by schedule_view.json mtime."""
        return sorted((self._apps[i] for i in self._order),
                      key=lambda a: a.mtime, reverse=True)

    def get(self, app_id):
        return self._apps.get(app_id)

    def default(self):
        """The app to open with.

        An explicitly named one wins. list() is newest-first by
        schedule_view.json mtime, so without this a positional workdir — the
        historical single-app invocation, and what run_debug_ui.sh passes on a
        cold start — lost to whichever unrelated app happened to be rebuilt
        most recently. That also silently scoped bare-filename lookups to the
        wrong app.
        """
        for app_id in self._explicit:
            app = self._apps.get(app_id)
            if app is not None:
                return app
        apps = self.list()
        return apps[0] if apps else None


class DebugState:
    """Shared server config + live apppaltest run state."""

    def __init__(self, workdir, elf, aie_version, device, target,
                 apppaltest, startcol, applog,
                 claude_bin="claude", claude_cwd=None, claude_model=None,
                 llm_enabled=True, llm_password=None, sim_only=False):
        self.workdir = os.path.abspath(workdir)
        self.elf = elf
        self.aie_version = str(aie_version)
        self.device = device
        self.target = target
        self.apppaltest = apppaltest
        self.startcol = int(startcol)
        self.applog = os.path.abspath(applog)
        self.sim_only = bool(sim_only)

        # Cached slices of schedule_view.json.
        self._tiles = None
        self._comm_paths = None
        self._grid_info = None
        self._last_scan = None
        self._last_dynamic_routing = None
        self._scan_lock = threading.Lock()

        # Live run bookkeeping.
        self._lock = threading.Lock()
        self._run_proc = None         # subprocess.Popen or None
        self._run_fh = None           # open applog file handle (subprocess stdout)
        self._run_id = 0

        # applog tail bookkeeping. Deriving the run status no longer decodes the
        # whole file each poll; instead we remember which status markers have
        # been seen so far in this tail and update that set from each new chunk.
        # `_applog_scan_tail` carries the last few hundred bytes so a marker
        # split across a chunk boundary is still caught. Both reset when the
        # client restarts the tail from offset 0 (a new run).
        self._applog_seen = set()
        self._applog_scan_tail = ""

        # ---- session provenance ------------------------------------------
        # self.target is populated at startup from $AIEDBG_TARGET (envlocal.sh),
        # so its mere presence proves nothing about whether THIS session ever
        # reached the board. Without the flags below every consumer answered
        # "are we connected?" with bool(self.target) — which is true from process
        # start — and the LLM would read a physically reachable board still
        # holding a previous run's state and report it as the current run.
        #
        # _hw_session is None until the user does one of three things:
        #   connected  a /ping probe succeeded         ("Connect")
        #   ran        a run was started from this UI  ("Run")
        #   attached   the user adopted a run they started outside the UI
        #              ("Open Current Session") — the board's prior history is
        #              explicitly unknown in this mode, which is why it is not
        #              folded into "ran".
        self._session_started_at = time.time()
        self._hw_session = None       # None | {mode, at, target, detail}
        self._last_run = None         # None | {run_id, started_at, device}

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
        self._llm_log_dir = self.workdir
        self._llm_log_fh = None
        self._llm_log_in_asst = False
        self._llm_system_prompt_text = None
        self._llm_first_turn = True
        self._llm_generation = 0
        self._llm_reset_reason = ""
        self._llm_last_output = None   # float (time.monotonic) or None

        # Path to the auto-generated MCP config handed to the claude subprocess
        # via --mcp-config (written lazily by _write_mcp_config; cleaned up on
        # shutdown). None => fall back to cwd .mcp.json auto-discovery.
        self._mcp_config_path = None
        self._mcp_config_backend = None

        # App registry + last-reported browser UI state (set by main()).
        # self_url is filled in once the port is known so the debugui MCP can
        # call back for /apps and /uistate.
        self.registry = None
        self.app = None
        self.self_url = None
        self._uistate = {}
        self._uistate_lock = threading.Lock()

        self._load_app_profile(self.workdir)
        self._sim_lock = threading.Lock()
        self._sim_proc = None     # subprocess.Popen or None
        self._sim_fh = None
        self._sim_run_id = 0
        self._sim_dbg_socket = None
        self._sim_ipc_ready = False
        self._sim_addr_params = (
            _load_aie_addr_params(self.sim_example_dir)
            if self.sim_example_dir else None
        )

    # ---- per-app profile -------------------------------------------------
    def _load_app_profile(self, workdir):
        """(Re)resolve everything that belongs to one app: its run profile from
        debug_ui_config.json plus the sim paths derived from it. Called at
        construction and again on every /apps/select."""
        ui_cfg = _load_ui_config(workdir)
        self._extra_devices = list(ui_cfg.get("extra_devices", []))
        sim_dev = next((d for d in self._extra_devices
                        if d.get("value") == "simulator"), None)
        if sim_dev is None:
            sim_dev = _detect_sim_device(workdir)
            if sim_dev:
                self._extra_devices.append(sim_dev)
        if not any(d.get("hw_run_script") for d in self._extra_devices):
            hw_dev = _detect_hw_device(workdir)
            if hw_dev:
                self._extra_devices.append(hw_dev)
        self.sim_kind = (sim_dev or {}).get("sim_kind", "ipc")
        self.sim_script = sim_dev.get("sim_script") if sim_dev else None
        self.sim_example_dir = sim_dev.get("sim_example_dir") if sim_dev else None
        # Why the Simulator row is offered but not runnable. Resolved once here
        # so /devices, the dropdown, start_sim and the startup listing all give
        # the same sentence instead of the option simply going missing.
        self.sim_reason = "" if self.sim_script else _sim_unavailable_reason(workdir)
        if not self.sim_example_dir:
            self.sim_log = self.sim_applog = self.sim_engine_log = None
        elif self.sim_kind == "aiesim":
            # runsim.sh streams everything to stdout; there is no separate app log.
            self.sim_log = os.path.join(workdir, "aiesim.log")
            self.sim_applog = None
            self.sim_engine_log = None
        else:
            # runsim_ipc.sh redirects the aiesimulator process to
            # <example>/ipc_sim.log ITSELF, so capturing the script's stdout to
            # that same path gave one file two writers with independent offsets
            # — they overwrote each other and the console showed shredded text.
            # Keep the engine log where the CLI leaves it (handed over via
            # $AEG_SIM_LOG) and take a separate file for the script.
            self.sim_log = os.path.join(self.sim_example_dir, "ipc_runsim.log")
            self.sim_applog = os.path.join(self.sim_example_dir, "ipc_app.log")
            self.sim_engine_log = os.path.join(self.sim_example_dir,
                                               "ipc_sim.log")

    def select_app(self, app_id):
        """Switch the whole server to another app. Refuses mid-run because the
        run profile carries board IPs / PDIs / ELF paths — swapping it under a
        live run could deploy one app's image against another's board config."""
        if self.registry is None:
            return {"error": "app registry not initialised"}
        app = self.registry.get(app_id)
        if app is None:
            return {"error": f"unknown app: {app_id}"}
        # Name the blocker and the remedy. A bare "a run is in progress" leaves
        # the user staring at a dropdown that snapped back, with no idea which
        # of the two processes to stop.
        if self.run_in_progress():
            return {"error": "cannot switch apps while a board run is in "
                             "progress; press Stop on the run first"}
        if self.sim_running():
            return {"error": "cannot switch apps while the simulator is "
                             "running; press Stop sim first. (Switching would "
                             "repoint startcol/aie_version and the provenance "
                             "JSONs at another app, so live reads from this "
                             "simulator would be decoded against the wrong "
                             "design.)"}

        self.app = app
        self.workdir = app.path
        self._tiles = None
        self._comm_paths = None
        self._grid_info = None
        self._llm_log_dir = app.path
        self.elf = _resolve_default_elf(app.path)
        self._load_app_profile(app.path)
        self._sim_addr_params = (_load_aie_addr_params(self.sim_example_dir)
                                 if self.sim_example_dir else None)
        # startcol / aie_version come from this app's provenance JSONs.
        dfsche, dmaphop = aiediag.load_jsons(app.path)
        sc = aiediag.startcol_from_jsons(dfsche, dmaphop)
        if sc is not None:
            self.startcol = int(sc)
        av = aiediag.aie_version_from_jsons(dfsche, dmaphop)
        if av:
            self.aie_version = str(av)
        # The MCP config embeds the app dir; force a rewrite on next spawn.
        self._mcp_config_path = None
        with self._uistate_lock:
            self._uistate = {}
        return {"ok": True, **app.info(current=True),
                "startcol": self.startcol, "aie_version": self.aie_version}

    def add_app(self, path, label=None, select=False):
        """Register an app workdir with a running daemon. Lets a producer flow
        (e.g. naiebaremetal's run_debug_ui.sh) hand its freshly generated bundle
        to the shared server instead of starting a second one."""
        if self.registry is None:
            return {"error": "app registry not initialised"}
        if not path:
            return {"error": "path required"}
        app = self.registry._add(path, label)
        if app is None:
            return {"error": f"no schedule_view.json in {path}"}
        if select:
            return self.select_app(app.id)
        return {"ok": True, **app.info(current=bool(self.app and self.app.id == app.id))}

    def apps_info(self):
        if self.registry is None:
            return []
        cur = self.app.id if self.app else None
        return [a.info(current=(a.id == cur)) for a in self.registry.list()]

    def set_uistate(self, state):
        """Record what the browser currently has open so the agent can see it."""
        if not isinstance(state, dict):
            return {"error": "uistate must be an object"}
        state = dict(state)
        state["app"] = self.app.id if self.app else None
        state["ts"] = time.time()
        with self._uistate_lock:
            self._uistate = state
        return {"ok": True}

    def get_uistate(self):
        with self._uistate_lock:
            return dict(self._uistate)

    def record_live_scan(self, what, res, device="", host=""):
        """Remember the latest /grid result for the LLM tab and debugui MCP."""
        summary = live_scan_summary.summarize_live_scan(what, res or {})
        summary["device"] = (device or "").strip()
        summary["host"] = (host or "").strip()
        with self._scan_lock:
            self._last_scan = summary
        self._write_backend_status()

    def get_live_scan(self):
        with self._scan_lock:
            return dict(self._last_scan) if self._last_scan else None

    def sim_running(self):
        with self._sim_lock:
            return bool(self._sim_proc and self._sim_proc.poll() is None)

    def sim_state(self):
        """Simulator bookkeeping for the browser to reconcile against, the exact
        counterpart of run_state() for the board.

        The UI learned a sim existed only from the /sim/log tail it started
        itself, so a reload — or a sim started from another tab — left it
        believing nothing was running while select_app refused to switch apps
        and Run stayed lit. Cheap to poll: no JTAG, no subprocess."""
        with self._sim_lock:
            proc = self._sim_proc
            running = proc is not None and proc.poll() is None
            ready = self._sim_ipc_ready
            dbg = self._sim_dbg_socket
            run_id = self._sim_run_id
        return {
            "available": bool(self.sim_script),
            "reason": self.sim_reason,
            "kind": self.sim_kind if self.sim_script else "",
            "running": running,
            "run_id": run_id,
            "pid": proc.pid if running else None,
            "ipc_ready": ready,
            "dbg_socket": dbg,
            "sim_log": self.sim_log or "",
            "applog": self.sim_applog or "",
            "engine_log": self.sim_engine_log or "",
            "example_dir": self.sim_example_dir or "",
        }

    # ---- static data -----------------------------------------------------
    def html_path(self):
        return os.path.join(self.workdir, "host_schedule.html")

    def json_path(self):
        return os.path.join(self.workdir, "schedule_view.json")

    def tiles(self):
        """Return the tile list from schedule_view.json (cached)."""
        if self._tiles is None:
            self._load_view_parts()
        return self._tiles

    def comm_paths(self):
        """Return the comm_paths list from schedule_view.json (cached)."""
        if self._comm_paths is None:
            self._load_view_parts()
        return self._comm_paths

    def grid_info(self):
        """Return the grid geometry block from schedule_view.json (cached)."""
        if self._grid_info is None:
            self._load_view_parts()
        return self._grid_info

    def _load_view_parts(self):
        with open(self.json_path()) as f:
            view = json.load(f)
        self._tiles = view.get("tiles", [])
        self._comm_paths = view.get("comm_paths", [])
        self._grid_info = view.get("grid", {})

    # ---- session provenance ----------------------------------------------
    def mark_hw_session(self, mode, detail="", target=None):
        """Record that this session legitimately reached the board. Called by
        the /ping probe, start_run, and /attach — never by target resolution."""
        with self._lock:
            self._hw_session = {"mode": mode, "at": time.time(),
                                "target": target or self.target or "",
                                "detail": detail}
        self._write_backend_status()

    def clear_sim_session(self):
        """Drop the authorization a simulator granted, once it exits. Unlike a
        board — which keeps answering after the run that programmed it — the
        process that vouched for those reads is gone, so the grant must go too."""
        cleared = False
        with self._lock:
            if (self._hw_session or {}).get("mode") == "simulator":
                self._hw_session = None
                cleared = True
        return cleared

    def hw_authorized(self):
        """True once the user has connected, run, attached, or brought up a
        simulator in this session. Gates every live register read so nothing
        reports stale board state as if it were current."""
        return self._hw_session is not None

    def applog_provenance(self):
        """Classify the applog on disk relative to this session.

        `applog` is a fixed repo-root path that the manual CLI flow writes too,
        so a file sitting there is NOT evidence that this session ran anything —
        which is exactly how a previous run's "PASS" gets read as current."""
        if not os.path.isfile(self.applog):
            return {"exists": False, "state": "absent"}
        mtime = os.path.getmtime(self.applog)
        if self._last_run is not None:
            state = "current"          # this session started the run that wrote it
        elif mtime < self._session_started_at:
            state = "predates_session"
        else:
            state = "foreign"          # written while we were up, but not by us
        return {"exists": True, "state": state, "mtime": mtime,
                "mtime_iso": time.strftime("%Y-%m-%d %H:%M:%S",
                                           time.localtime(mtime)),
                "session_start_iso": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(self._session_started_at))}

    def session_state(self):
        """Canonical session provenance, published to backend_status.json and
        every model-visible surface. One builder so the UI, the MCP servers and
        the LLM context line can never disagree about what happened."""
        s = self._hw_session
        return {
            "authorized": s is not None,
            "mode": (s or {}).get("mode", "none"),
            "since_iso": (time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(s["at"])) if s else ""),
            "detail": (s or {}).get("detail", ""),
            "session_start_iso": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._session_started_at)),
            "run_started": self._last_run is not None,
            "run_in_progress": self.run_in_progress(),
            "last_run": self._last_run or {},
            "applog": self.applog_provenance(),
            "sim_only": self.sim_only,
        }

    def session_summary(self):
        """One-line, model-facing rendering of session_state()."""
        st = self.session_state()
        if not st["authorized"]:
            if st["sim_only"]:
                return ("NO SESSION — this daemon runs with --sim-only: it "
                        "reaches no board, and connect/attach/board-run are "
                        "refused. Activate the simulator to get live state. "
                        "Any applog on disk is from some earlier board run, "
                        "not from anything this daemon can do.")
            return ("NO BOARD SESSION — the user has not connected, run, or "
                    "attached in this UI. Live reads are blocked and any on-disk "
                    "log predates this session.")
        mode = st["mode"]
        if mode == "simulator":
            # Returns early: the applog suffix below describes a board run, and
            # appending it here would invite reading a board log as this
            # simulator's output.
            return (f"SIMULATOR SESSION since {st['since_iso']} — a simulator "
                    f"started from this UI is live on its IPC debug socket. "
                    f"Live register reads come from the simulator, not a board, "
                    f"and are current by construction; any applog on disk is a "
                    f"hardware run and unrelated to it.")
        if mode == "attached":
            base = (f"ATTACHED at {st['since_iso']} to a run started OUTSIDE this "
                    f"UI — the board's earlier history is unknown to the daemon")
        elif mode == "ran":
            base = f"RAN from this UI at {st['since_iso']}"
        else:
            base = (f"CONNECTED at {st['since_iso']} (link verified) but NO run "
                    f"has been started in this session")
        ap = st["applog"]
        if ap["state"] == "predates_session":
            base += f"; the applog on disk was written {ap['mtime_iso']}, BEFORE this session"
        elif ap["state"] == "absent":
            base += "; no applog on disk"
        return base

    # ---- run orchestration ----------------------------------------------
    def run_in_progress(self):
        """True while a board run is live. Used to gate aiedbg endpoints so
        their JTAG reads don't collide with apppaltest's device program/reset/
        download over the single serialized xsdb://<PALIP>:3121 link."""
        with self._lock:
            return self._run_proc is not None and self._run_proc.poll() is None

    def run_state(self):
        """Run bookkeeping for the browser to reconcile against. No JTAG, so it
        is safe to poll; /ping, /settarget and /run all refuse while a run is
        live, so a browser that lost track of one has no other way back."""
        with self._lock:
            proc = self._run_proc
            running = proc is not None and proc.poll() is None
            pid = proc.pid if proc is not None else None
            last = dict(self._last_run or {})
            run_id = self._run_id
        started = last.get("started_at")
        status, text = "idle", ""
        if os.path.isfile(self.applog):
            try:
                with open(self.applog, "rb") as f:
                    text = f.read().decode("utf-8", errors="replace")
                status = self._derive_status(text, running,
                                             os.path.getmtime(self.applog))
            except OSError:
                pass
        return {
            "running": running,
            "run_id": run_id if running else last.get("run_id", 0),
            "pid": pid if running else None,
            "device": last.get("device", ""),
            "started_iso": last.get("started_iso", ""),
            "age_s": (int(time.time() - started)
                      if running and started else None),
            "status": status,
            "stale": bool(running and status == "hang"),
            "debuggable": self._is_debuggable(text, status) if running else False,
            "applog": self.applog,
            "session": self.session_state(),
            # Rides the same heartbeat as the board run for the same reason: a
            # reloaded page has no other way to learn a simulator is live.
            "sim": self.sim_state(),
        }

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
            return True
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
          3. pal (default): apppaltest.py; inherits the daemon's env.

        The elf positional is omitted unless configured (--elf); with -y, the
        script auto-picks its default elf, matching the manual
        `<script>.py -y -nonreboot` invocation.
        """
        device = (device or "pal").strip().lower()
        with self._lock:
            if self._run_proc and self._run_proc.poll() is None:
                return {"error": "a run is already in progress",
                        "run_id": self._run_id}

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
                # The UI's hostname box wins over anything the profile carries.
                env, _host = _board_run_env(os.environ, cfg_dev, board_host,
                                            self.workdir)

            elif device == "vek385":
                script = os.path.join(os.path.dirname(self.apppaltest),
                                      "appvek385.py")
                # Same resolution order as the profile path, so the built-in
                # runner also picks up the checkout's board when the UI's box
                # is empty instead of refusing outright.
                host = (board_host or "").strip() or _expected_board_host(
                    self.workdir)
                if not host:
                    return {"error": "vek385 requires a board hostname"}
                if not os.path.isfile(script):
                    return {"error": f"appvek385.py not found: {script}"}
                env, _host = _board_run_env(os.environ, {}, host, self.workdir)

            elif device in ("pal", "palmyra", ""):
                device = "pal"
                script = self.apppaltest

            else:
                # No runner for this board. Falling through to apppaltest would
                # have quietly deployed the palmyra test to whatever was picked.
                return {"error": f"no run script for {device}; add one via "
                                 f"debug_ui_config.json (hw_run_script)"}

            self._run_id += 1
            run_id = self._run_id
            # -u: unbuffered stdout for realtime tail; -y: auto-confirm ELF pick.
            cmd = [sys.executable, "-u", script, "-y", "-nonreboot"]
            if not cfg_dev:
                # The ELF is resolved once at app-select time, so a later
                # clean/rebuild (build/host removed, aout/main.elf produced)
                # leaves a stale path. Re-resolve before giving up so the
                # search order (build/host -> parent *.elf -> aout/main.elf)
                # picks whatever exists now instead of failing the run.
                if not self.elf or not os.path.isfile(self.elf):
                    fallback = _resolve_default_elf(self.workdir)
                    if fallback:
                        self.elf = fallback
                    elif self.elf:
                        return {"error": f"ELF not found: {self.elf}"}
                if self.elf:
                    cmd.append(self.elf)
            try:
                fh = open(self.applog, "w")
            except OSError as e:
                return {"error": f"cannot open applog {self.applog}: {e}"}
            fh.write(f"$ {' '.join(cmd)}\n")
            if env and "VEK385IP" in env:
                # The board is a per-run choice, so the log must name it.
                fh.write(f"# env: USERNAME={env.get('USERNAME', '')} "
                         f"VEK385IP={env['VEK385IP']} "
                         f"AIEDBG_TARGET={env.get('AIEDBG_TARGET', '')}\n")
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
            # This session now owns the applog, so its contents describe the
            # current run rather than whatever was on disk beforehand.
            self._last_run = {"run_id": run_id, "device": device,
                              "started_at": time.time(),
                              "started_iso": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.mark_hw_session("ran", f"run #{run_id} on {device}")
        return {"run_id": run_id, "applog": self.applog, "device": device}

    def stop_run(self):
        """Force-kill the running apppaltest (and its ssh/pexpect children)."""
        with self._lock:
            proc = self._run_proc
            if proc is None or proc.poll() is not None:
                self._run_proc = None
                return {"stopped": False, "error": "no run in progress",
                        "running": False}
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
        # A child parked in an uninterruptible syscall survives SIGKILL and
        # poll() then returns None forever. run_in_progress() gates /ping, /run
        # and /settarget, so holding the handle would wedge every recovery path
        # until a daemon restart. Release it regardless.
        abandoned = proc.poll() is None
        with self._lock:
            if self._run_fh:
                try:
                    self._run_fh.close()
                except OSError:
                    pass
                self._run_fh = None
            if self._run_proc is proc:
                self._run_proc = None
        # Note the kill in the applog so the live tail reflects it.
        try:
            with open(self.applog, "a") as f:
                if abandoned:
                    f.write(f"\n[force-stop: run {run_id} (pid {pid}) did not "
                            f"die after SIGKILL — handle released; the process "
                            f"may still be alive and holding the JTAG link]\n")
                else:
                    f.write(f"\n[force-stop: killed run {run_id} (pid {pid})]\n")
        except OSError:
            pass
        return {"stopped": True, "run_id": run_id, "pid": pid,
                "abandoned": abandoned, "running": False}

    def timeline_png_path(self):
        """Where `timeline.py` writes the rendered PNG — next to the applog so it
        travels with the run artifacts and is served by GET /timeline.png."""
        out_dir = os.path.dirname(self.applog) or "."
        return os.path.join(out_dir, "timeline.png")

    def launch_timeline(self):
        """Render `timeline.py <applog>` to a PNG on disk (no interactive window)
        so the browser can show it in the Profile tab via GET /timeline.png.

        Runs synchronously and waits for the render so the caller can report
        whether the PNG is ready. No-op when no applog exists yet."""
        if not os.path.isfile(self.applog):
            return {"started": False, "applog": self.applog,
                    "error": "no applog on disk yet"}
        script = os.path.join(_THIS_DIR, "timeline.py")
        out_dir = os.path.dirname(self.applog) or "."
        png_path = self.timeline_png_path()
        try:
            proc = subprocess.run(
                [sys.executable, script, self.applog,
                 "--out-dir", out_dir, "--png", png_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            return {"started": False, "applog": self.applog, "error": str(e)}
        if proc.returncode != 0 or not os.path.isfile(png_path):
            detail = (proc.stdout or b"").decode("utf-8", "replace").strip()
            return {"started": False, "applog": self.applog,
                    "error": detail[-400:] or "timeline render failed"}
        return {"started": True, "applog": self.applog, "png": png_path}

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
                        if sim_ipc_ping(dbg_path):
                            with self._sim_lock:
                                self._sim_dbg_socket = dbg_path
                                self._sim_ipc_ready = True
                            print(f"[sim] IPC debug socket ready → {dbg_path}",
                                  flush=True)
                            # Authorize live reads. /grid and the MCP servers
                            # gate on hw_authorized(), which only /ping, /run
                            # and /attach ever set — so a working simulator was
                            # refused unless the user had also connected to a
                            # board. This socket belongs to a process we own,
                            # which is a stronger guarantee than any board can
                            # give, not a weaker one.
                            self.mark_hw_session(
                                "simulator",
                                f"IPC debug socket ready "
                                f"({os.path.basename(dbg_path)})",
                                "simulator-ipc")
                            self._invalidate_mcp_config()
                            self._write_backend_status()
                            break
                except OSError:
                    pass
            time.sleep(0.5)
        with self._sim_lock:
            proc = self._sim_proc
        if proc is not None:
            proc.wait()
        with self._sim_lock:
            if self._sim_run_id == run_id:
                self._sim_ipc_ready = False
                self._sim_dbg_socket = None
        self.clear_sim_session()
        self._invalidate_mcp_config()
        self._write_backend_status()

    def start_sim(self):
        """Spawn runsim_ipc.sh <sim_example_dir>, stream output to sim_log,
        and start a watcher thread that retargets aiegdb once the ISS port file
        appears."""
        if not self.sim_script:
            # The old text blamed a missing debug_ui_config.json, which has not
            # been how a simulator is found since detection replaced declaration
            # — and left the user with nothing to act on.
            return {"error": f"simulator unavailable for this app: "
                             f"{self.sim_reason}",
                    "reason": self.sim_reason, "available": False}
        if not os.path.isfile(self.sim_script):
            return {"error": f"sim script not found: {self.sim_script}"}
        if not self.sim_example_dir:
            return {"error": "no sim_example_dir configured"}
        with self._sim_lock:
            if self._sim_proc and self._sim_proc.poll() is None:
                return {"error": "simulator already running",
                        "run_id": self._sim_run_id}
            if self.sim_kind == "ipc":
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
            if self.sim_applog:
                try:
                    open(self.sim_applog, "w").close()
                except OSError:
                    pass
            try:
                fh = open(self.sim_log, "w")
            except OSError as e:
                return {"error": f"cannot open sim log {self.sim_log}: {e}"}
            env = dict(os.environ)
            if self.sim_engine_log:
                # Hand runsim_ipc.sh its own path for the aiesimulator process
                # so it stops writing the file this capture owns (see
                # _load_app_profile). Same default location as the CLI leaves
                # it, so `less <example>/ipc_sim.log` still works.
                env["AEG_SIM_LOG"] = self.sim_engine_log
            fh.write(f"$ {' '.join(cmd)}\n")
            if self.sim_engine_log:
                fh.write(f"[simulator engine log: {self.sim_engine_log}]\n")
            fh.flush()
            try:
                self._sim_proc = subprocess.Popen(
                    cmd, stdout=fh, stderr=subprocess.STDOUT,
                    cwd=self.sim_example_dir, env=env,
                    start_new_session=True)
            except FileNotFoundError as e:
                fh.close()
                return {"error": f"cannot launch sim script: {e}"}
            self._sim_fh = fh
        # The dbg-socket watcher is IPC-only: it retargets aiegdb at the ISS
        # socket that runsim_ipc.sh creates. The aiesim backend exposes no such
        # socket, so live debug stays on whatever target is configured.
        if self.sim_kind == "ipc":
            threading.Thread(target=self._sim_watch_dbg_socket, args=(run_id,),
                             daemon=True).start()
        self._write_backend_status()
        return {"run_id": run_id, "sim_log": self.sim_log,
                "example_dir": self.sim_example_dir, "sim_kind": self.sim_kind,
                "applog": self.sim_applog or "",
                "engine_log": self.sim_engine_log or "",
                "sim": self.sim_state()}

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
        self.clear_sim_session()
        self._write_backend_status()
        return {"stopped": True, "run_id": run_id, "pid": pid,
                "sim": self.sim_state()}

    def sim_status(self):
        """Current simulator state: availability + why not, which backend, run
        and IPC-socket readiness. A superset of what it used to return."""
        return self.sim_state()

    def simlog_since(self, offset):
        """Tail the sim log file from byte offset; includes IPC readiness.

        `kind` rides along because the two backends behave differently at the
        far end: the IPC sim opens a debug socket and unlocks live reads, the
        aiesim model never will, and the browser must not promise one while
        tailing the other."""
        if not self.sim_log or not os.path.isfile(self.sim_log):
            return {"data": "", "next": 0, "running": False,
                    "ipc_ready": False, "dbg_socket": None,
                    "kind": self.sim_kind if self.sim_script else ""}
        with self._sim_lock:
            running = (self._sim_proc is not None
                       and self._sim_proc.poll() is None)
            ipc_ready = self._sim_ipc_ready
            dbg_socket = self._sim_dbg_socket
            run_id = self._sim_run_id
        with open(self.sim_log, "rb") as f:
            full = f.read()
        chunk = full[offset:]
        data = chunk.decode("utf-8", errors="replace")
        nxt = offset + len(chunk)
        return {"data": data, "next": nxt, "running": running,
                "ipc_ready": ipc_ready, "dbg_socket": dbg_socket,
                "kind": self.sim_kind, "run_id": run_id}

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
          pal: ssh <USERNAME>@<PALIP> -> /bin/systest -> become "<BOARD>"
          others : ssh <host>             -> /proj/systest/bin/systest
        both then: xsdb -> exec hw_server -stcp:0.0.0.0:3121
        """
        device = (device or "").strip().lower()
        if device != "pal":
            # Every board except pal is reached by its hostname. Keyed on the
            # host rather than on `vek385` specifically, so vek280/vck190 do not
            # fall through and ssh to $PALIP instead of the board the user picked.
            if not host:
                raise RuntimeError(f"{device or 'board'} requires a hostname")
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
        child = pexpect.spawn(ssh_command(ssh_target), encoding="utf-8",
                              timeout=60)
        child.logfile_read = logsink            # stream the raw session output
        expect_shell_after_ssh(child, timeout=60)
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
        # A fresh tail (client reset to offset 0 for a new run) invalidates the
        # cached marker state carried over from the previous run's file.
        if offset <= 0:
            offset = 0
            self._applog_seen = set()
            self._applog_scan_tail = ""
        size = os.path.getsize(self.applog)
        # Read ONLY the new bytes, not the whole file, so poll cost stays flat as
        # the AIE event trace grows to many MB.
        with open(self.applog, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        nxt = offset + len(chunk)
        truncated = False
        # Safety cap: if the client fell far behind (multi-MB backlog), ship only
        # the last CAP bytes + a truncated flag instead of tens of MB in one hop.
        if len(chunk) > self._APPLOG_TAIL_CAP:
            chunk = chunk[-self._APPLOG_TAIL_CAP:]
            truncated = True
        data = chunk.decode("utf-8", errors="replace")
        # Update the seen-marker set from this chunk only (plus a small carry so a
        # marker split across the previous boundary is still caught). No full-file
        # decode.
        scan = self._applog_scan_tail + data
        for m in self._STATUS_MARKERS:
            if m not in self._applog_seen and m in scan:
                self._applog_seen.add(m)
        self._applog_scan_tail = scan[-256:]
        last_ts = os.path.getmtime(self.applog)
        status = self._derive_status_markers(self._applog_seen, running, last_ts,
                                             size > 0)
        return {"data": data, "next": nxt, "status": status, "running": running,
                "truncated": truncated,
                "debuggable": self._is_debuggable_markers(self._applog_seen,
                                                          status)}

    # Cap on bytes shipped in a single /applog poll. Beyond this the client has
    # fallen too far behind to catch up in one hop, so we send the last CAP bytes
    # and flag the gap rather than blocking the link on a huge payload.
    _APPLOG_TAIL_CAP = 2 * 1024 * 1024

    # Pass/fail markers plus the debuggable markers, scanned out of each tail
    # chunk so status derivation never re-reads the whole file.
    _STATUS_MARKERS = ("device_teardown done", "Not tearing down partition",
                       "AIE ERROR", "ELF download complete", "Execution started",
                       "board stays powered on for debug", "wait_io TIMEOUT",
                       "Waiting for console output")

    @staticmethod
    def _derive_status_markers(seen, running, last_ts, has_content):
        """Marker-set equivalent of _derive_status (no whole-file decode)."""
        if "device_teardown done" in seen or "Not tearing down partition" in seen:
            return "pass"
        if "AIE ERROR" in seen:
            return "fail"
        if running:
            if last_ts and (time.time() - last_ts) > 60:
                return "hang"
            return "running"
        return "fail" if has_content else "idle"

    @classmethod
    def _is_debuggable_markers(cls, seen, status):
        """Marker-set equivalent of _is_debuggable."""
        if status in ("hang", "pass", "fail"):
            return True
        return any(m in seen for m in cls._DEBUGGABLE_MARKERS)

    _DEBUGGABLE_MARKERS = (
        "ELF download complete",
        "Execution started",
        "board stays powered on for debug",
        "wait_io TIMEOUT",
        "Waiting for console output",
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
        # Auto-inject --work-dir for aiecompiler apps (naiebaremetal and similar).
        # aiehlc apps emit provenance JSONs directly and have no Work/ tree.
        _work_dir = _resolve_work_dir(self.workdir)
        if _work_dir:
            cmd += ["--work-dir", _work_dir]
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
        # callstack commands open a fresh XSDB connection to read a live PC,
        # which can take 2-3 min through a farm host.  Give them extra room.
        stripped = cmd.strip().lower()
        read_timeout = 300 if stripped.startswith("callstack") else 60
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
            out = self._gdb_read_until_marker(timeout=read_timeout)
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

    def retarget(self, tgt, device=None):
        """Point live debug at a new JTAG target (and aiedbg device) and drop the
        aiegdb REPL so its next command respawns against them.

        Called from /settarget after the browser's `Test connect` (/ping) has
        already verified `tgt` is reachable — so this only *switches* the target,
        it does not re-probe. The persistent aiegdb subprocess is spawned once
        with `--target self.target` (_gdb_spawn), so it must be killed for the
        new target to take effect; the next gdb_command lazily respawns it. Also
        invalidates the cached MCP config (which embeds the old target) so the
        LLM tab picks up the new target on its next spawn.

        `device` is the aiedbg -d name, which the dropdown now carries directly.
        It is set here rather than only at startup because it feeds four places
        at once — the aiegdb subprocess, the MCP config the LLM tab spawns with,
        do_cmd's aiediag reads and backend_status.json — and staying on the
        launch value left aiedbg decoding a 38-column part as a 12-column one.
        """
        next_target = tgt or None
        next_device = device or self.device
        if next_target == self.target and next_device == self.device:
            return {
                "ok": True, "target": self.target, "device": self.device,
                "llm_reset": False, "llm_generation": self._llm_generation,
            }

        with self._gdb_lock:
            self.target = next_target
            self.device = next_device
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
        # backend_status.json carries target+device to the out-of-process MCP
        # servers; without this refresh get_backend_status() keeps reporting the
        # startup device long after Connect switched boards.
        self._write_backend_status()
        return {
            "ok": True, "target": self.target, "device": self.device,
            "llm_reset": False, "llm_generation": self._llm_generation,
        }

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
            self._llm_last_output = time.monotonic()
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

    def app_paths(self):
        """Resolved filesystem locations for the currently selected app.

        The LLM is served out of the provenance bundle, which for both producer
        flows is a *subdirectory* (`worklocal/`) of the app, with the aiecompiler
        `Work/` tree as a sibling. Neither the app root nor Work/ is discoverable
        from the bundle path alone, so an assistant told only the workdir cannot
        find the app's sources — it has to be told all three.
        """
        bundle = self.workdir
        app_dir = (os.path.dirname(bundle)
                   if os.path.basename(bundle) == "worklocal" else bundle)
        return {
            "app": (self.app.label if self.app else os.path.basename(app_dir)),
            "app_dir": app_dir,
            "bundle_dir": bundle,
            "work_dir": _resolve_work_dir(bundle) or "",
            "elf": self.elf or "",
        }

    def app_sources(self):
        """The app's own source files, the compiler's output, and the kernel map.

        Companion to app_paths(): knowing the app directory is not the same as
        knowing what to open in it. See _app_source_manifest.
        """
        return _app_source_manifest(self)

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
        addr_params = self._sim_addr_params
        _ap = self.app_paths()
        _srcs = self.app_sources()
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
            # Which simulator, and why there isn't one. The MCP servers are
            # separate processes, so the same rule as session/app_paths applies:
            # anything the model must not guess at travels in this file.
            "sim_kind": self.sim_kind if self.sim_script else "",
            "sim_available": bool(self.sim_script),
            "sim_reason": self.sim_reason,
            "sim_engine_log": self.sim_engine_log or "",
            "applog": self.applog,
            "sim_base_addr": str(addr_params[0]) if addr_params else "",
            "sim_col_shift": str(addr_params[1]) if addr_params else "",
            "sim_row_shift": str(addr_params[2]) if addr_params else "",
            # Provenance travels with the status so the MCP servers (separate
            # processes) can refuse or caveat reads without a daemon round-trip.
            "session": self.session_state(),
            "session_summary": self.session_summary(),
            # Travels with the status for the same reason as session state: the
            # app can be switched mid-conversation, and the system prompt is
            # only sent on the first turn. app_sources rides along so the
            # assistant can re-enumerate the new app's code without a restart.
            "app_paths": _ap,
            "app_sources": _srcs,
            # Rendered here, next to the structured form, for the same reason
            # session_summary is: the MCP servers are separate processes, and one
            # renderer that both sides read beats two that drift apart.
            "app_sources_text": _fmt_app_sources(_srcs, rel_to=_ap.get("app_dir")),
            "aiedbg_paths": _aiedbg_paths(),
        }
        with self._scan_lock:
            if self._last_scan:
                data["last_scan"] = dict(self._last_scan)
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
        with self._sim_lock:
            sim_ready = self._sim_ipc_ready
            dbg_socket = self._sim_dbg_socket
            addr_params = self._sim_addr_params
        want_backend = "simulator" if sim_ready else "hardware"

        if (self._mcp_config_path and os.path.isfile(self._mcp_config_path)
                and self._mcp_config_backend == want_backend):
            return self._mcp_config_path

        self._invalidate_mcp_config()

        aiemcp = os.path.join(_THIS_DIR, "aiemcp.py")
        debugui = os.path.join(_THIS_DIR, "debug_ui_mcp.py")

        _mcp_work_dir = _resolve_work_dir(self.workdir) or ""
        aiegdb_env = {
            "AIEDBG_TARGET": self.target or "",
            "AIEMCP_DEVICE": self.device or "pal",
            "AIEMCP_STARTCOL": str(self.startcol),
            "AIEMCP_AIE_VERSION": str(self.aie_version),
            "AIEMCP_JSON_DIR": self.workdir,
            "AIEMCP_WORK_DIR": _mcp_work_dir,
            "AIEMCP_BACKEND": want_backend,
        }

        if sim_ready and dbg_socket:
            aiegdb_env["AEG_PS_IPC_DBG_SOCKET"] = dbg_socket
            if addr_params:
                base, col_shift, row_shift = addr_params
                aiegdb_env["AEG_SIM_BASE_ADDR"] = str(base)
                aiegdb_env["AEG_SIM_COL_SHIFT"] = str(col_shift)
                aiegdb_env["AEG_SIM_ROW_SHIFT"] = str(row_shift)

        debugui_env = {
            "DEBUGUI_JSON_DIR": self.workdir,
            "AIEMCP_JSON_DIR": self.workdir,
            "AIEMCP_BACKEND": want_backend,
            "AIEDBG_TARGET": self.target or "",
            "DEBUGUI_APPLOG": self.applog,
            "DEBUGUI_SIM_APPLOG": self.sim_applog or "",
            # Lets the debugui MCP follow app switches and read live UI state
            # instead of being frozen to the workdir it was spawned with.
            "DEBUGUI_SERVER_URL": self.self_url or "",
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
        # Name the backend and, when there isn't one, what is missing — the bare
        # "(simulator available)" told the model nothing it could act on and
        # nothing about which of the two very different backends it would get.
        _sim_avail_str = (
            f"  (simulator available: {self.sim_kind})" if self.sim_script
            else f"  (no simulator for this app — {self.sim_reason})")

        view = None
        view_path = os.path.join(self.workdir, "schedule_view.json")
        if os.path.isfile(view_path):
            try:
                with open(view_path) as f:
                    view = json.load(f)
            except (OSError, ValueError):
                view = None

        _ap = self.app_paths()
        _work_dir_str = _ap["work_dir"] or "(none — this app has no aiecompiler Work/ tree)"
        _elf_str = _ap["elf"] or "(not resolved)"
        # Naming the app directory was not enough: with no inventory the
        # assistant had to guess filenames, and guessing wrong is indistinguishable
        # from the app having no sources, so it silently stopped trying.
        _src_body = _fmt_app_sources(self.app_sources(), rel_to=_ap["app_dir"])
        _src_block = (_src_body if _src_body else
                      "(no sources found under the app directory — say so rather "
                      "than inventing filenames, and call `app_sources()` to retry)")
        _abp = _aiedbg_paths()
        _aiedbg_str = _abp["src"] or (
            "(not found — run src/tool/debug/ensure_aiedbg.py to clone it, "
            "or start the daemon without --skip-aiedbg-bootstrap)")

        # The plugin already puts each skill's name+description in your context,
        # so this block deliberately does NOT repeat the descriptions — it only
        # names them and gives the on-disk fallback. Two reasons it exists at all:
        # skill descriptions get truncated when the listing exceeds its context
        # budget, and if plugin loading ever fails the assistant can still Read
        # the files directly.
        _skills = _llm_skills()
        if _skills:
            _skills_block = (
                "\n## Debug skills (loaded as the `dbg-llm-skills` plugin)\n\n"
                "These are verified, step-by-step procedures for THIS debug UI — "
                "exact commands, register-decoding pitfalls, and preconditions you "
                "will otherwise get wrong. Invoke one with the Skill tool as soon as "
                "a task matches it; do not improvise a procedure a skill already covers.\n\n"
                + "\n".join(f"- `{n}`" for n, _d, _p in _skills)
                + f"\n\nIf the Skill tool cannot see these, read them directly from "
                  f"{_LLM_SKILLS_DIR}/<name>/SKILL.md.\n"
            )
        else:
            _skills_block = ""

        grid_summary = ""
        if view:
            grid = view.get("grid") or {}
            n_flows = len(view.get("comm_paths") or [])
            n_tiles = len(view.get("tiles") or [])
            grid_summary = (
                f"\n- Design: {n_tiles} tiles "
                f"({grid.get('cols')} cols × {grid.get('rows')} rows, "
                f"startcol={grid.get('startcol')}), {n_flows} communication flows."
            )

        if self.sim_example_dir and self.sim_kind == "aiesim":
            # The IPC block below describes a two-process flow with a debug
            # socket and transaction logs. None of it exists here, and handing
            # it over would send the model looking for files that never appear.
            _sim_section = (
                "## Simulator debugging (aiesim backend)\n\n"
                "This app's simulator is the aiehlc `aie2pssimmsm` flow driven by\n"
                f"`{self.sim_script}` from `{self.sim_example_dir}/sim_config.sh`.\n"
                "It is NOT the IPC simulator:\n"
                "- One process. The host code is compiled into `aiehlc_ps.so` and\n"
                "  runs inside the simulator, so there is no separate PS client,\n"
                "  no `ipc_app`, no `ipc_*.log` and no `get_ipc_log()`.\n"
                "- **No debug socket, so no live register reads.** `aie_exec`\n"
                "  register commands have nothing to talk to on this backend —\n"
                f"  read `{self.sim_log}` (the Run console tails it) instead, and\n"
                "  say so rather than reporting a read that did not happen.\n"
                "- A crash before `AIEHLC PS IP started` is a PS.so load failure;\n"
                "  the `aiesimloaddebug` skill in this repo covers it.\n\n"
            )
        elif self.sim_example_dir:
            _sed = self.sim_example_dir
            _sim_section = (
                "## PS process and simulator debugging\n\n"
                "This app has an IPC simulator. The run has two processes:\n"
                f"- **ipc_app** (PS client): `{_sed}/ipc/ipc_app`\n"
                "  Drives graph init, GMIO transfers, and result checks.\n"
                "  stdout/stderr -> `get_sim_log()` (tails `ipc_app.log`).\n"
                "- **aiesimulator** (server): Synopsys AIE functional simulator.\n"
                f"  stdout -> `{_sed}/ipc_sim.log`.\n"
                f"  The Run console tails `{_sed}/ipc_runsim.log`, the launcher\n"
                "  script's own output — a separate file, because two writers on\n"
                "  one path overwrite each other.\n"
                "  Receives `WRITE32`/`READ32`/`WRITE_GM` etc. from ipc_app over a Unix socket.\n"
                "\n"
                "### Finding what the simulator run is currently doing\n\n"
                "To check whether the simulator is still running and what it last did:\n"
                "1. Call `get_sim_log(lines=20)` — last 20 lines of `ipc_app.log` (PS app output).\n"
                "2. Call `get_ipc_log(lines=20)` — last 20 IPC transactions; the final client entry\n"
                "   is the transaction ipc_app is currently blocked waiting for a response to.\n"
                f"3. The simulator PID file is at: `{_sed}/ipc_sim.pid`\n"
                "\n"
                "### Inspecting what the PS process is doing (hung/stalled)\n\n"
                "When ipc_app appears hung, read its kernel wait state from /proc\n"
                "(use the Bash tool, not `aie_exec`, which takes aiegdb commands):\n\n"
                "```bash\n"
                f"cat {_sed}/ipc_sim.pid        # simulator PID\n"
                "pgrep -f ipc_app               # ipc_app PID\n"
                "cat /proc/<ipc_app_pid>/wchan  # syscall blocking the main thread\n"
                "```\n\n"
                "Common wchan values:\n"
                "- `unix_stream_data_wait` — blocked in recv() waiting for simulator ack\n"
                "- `futex_wait_queue_me` — waiting on a mutex/condvar\n"
                "- `ep_poll` — epoll_wait (sim socket-dispatch thread)\n"
                "- `0` — thread running or runnable\n"
                "\n"
                "### IPC transaction log\n\n"
                "Every run writes timestamped CSV logs:\n"
                f"- `{_sed}/ipc_client.log` — transactions from ipc_app\n"
                f"- `{_sed}/ipc_server.log` — same transactions as dispatched by simulator\n\n"
                "CSV columns: `seq, ts_ns, side, cmd, arg1, arg2, status, value, note`\n\n"
                "To find a hang:\n"
                "1. Call `get_ipc_log(lines=20)` — last client entry is what ipc_app is blocked on.\n"
                "2. If server log has matching seq with OK but client doesn't -> response in flight or socket broken.\n"
                "3. If neither has it -> simulator hasn't dispatched it yet (SystemC scheduling lag).\n\n"
                "Key IPC commands:\n"
                "- `GRAPH_INIT` — loads AIE ELFs + runs graph init (BD programming). Must finish before GMIO.\n"
                "- `WRITE_GM` / `READ_GM` — GMIO data transfer. arg1=AIE addr, arg2=byte count.\n"
                "- `WRITE32` / `READ32` — single AIE register r/w.\n"
                "- `START_PLIO` — starts PLIO streams after graph init.\n"
                "- `EXIT` — ipc_app teardown; sim shuts down after ack.\n"
                "\n"
                "### addr2line for ipc_app\n\n"
                f"The binary `{_sed}/ipc/ipc_app` has DWARF debug symbols. Map an address to source:\n\n"
                "```bash\n"
                f"addr2line -e {_sed}/ipc/ipc_app -f -p 0x<addr>\n"
                "```\n\n"
                f"PS app source: `{_sed}/src/graph.cpp`\n"
                "IPC backend: `src/ipc/aeg_ps_ipc_backend.cpp` — `do_transaction()` is the central send/recv loop.\n\n"
            )
        else:
            _sim_section = ""

        _workflow_step6 = (
            "6. For a hang or stall on the **simulator**, call `get_ipc_log()` to find"
            " the last IPC transaction, then use `aie_exec` (`dma status`, `bd`, `event`)"
            " to trace the stall through the producer → hop → consumer chain;"
            " or inspect `/proc/<pid>/wchan` to confirm what the PS process is waiting on."
        ) if self.sim_example_dir else (
            "6. For a hang or stall, use `aie_exec` (`dma status`, `bd`, `event`) to trace"
            " the stall through the producer → hop → consumer chain."
            " Check `get_applog()` for timeout or error messages."
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
registers on the relevant tiles. Then cross-reference three things — what the live \
registers say, what the compiled schedule expects, and **what the application's own \
source asked for** — and tell the user concisely what is wrong and why, citing the \
line of their code that it comes back to.

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

Always prefer concrete register evidence over speculation.

## Ground every explanation in the application's source

The user is debugging code they wrote. A register value is the symptom; their \
source is where the cause lives and where the fix has to go. An answer built only \
from the compiled schedule and live registers describes the machine, not their \
program — it is accurate and still unusable, because it never reaches a line they \
can change.

So, by default, **open the app's source and quote it**. The inventory is in *The \
application's own source* below, with the file and line defining each kernel the \
schedule runs, so this costs you one Read.

- Explaining what a tile does → read that tile's kernel and name the loop, the \
  window access or the intrinsic that accounts for the behaviour. `tile_info` gives \
  you the kernel name; the inventory maps that name to its definition.
- Explaining a transfer, buffer size or lock → tie the schedule's numbers back to \
  the graph/host source that declared them (window sizes, `connect<>` widths, \
  repetition counts). A number that appears in both is a fact; a number that appears \
  in only one is where the bug usually is.
- Diagnosing a stall, wrong output or fault → after the register evidence, go find \
  the source that produced the mismatched expectation and quote the specific line. \
  "The consumer expects 656 B but the producer sends 48 B" is half an answer; the \
  other half is the line in their kernel or graph that sets each.
- Proposing a fix → point at the exact line to change in a hand-written file. \
  Never propose editing generated output (`host.cc`, `kernel.cc`, the `.bcf`, the \
  MLIR) as the fix — it is overwritten on the next build. Use it as evidence of what \
  the compiler decided, then trace that decision back to the source that drove it.

**Cite every one of these as `<file>:<line>`** so the user can click straight to \
the code — see the citation rule below. Quote the few lines that matter inline; do \
not paste whole functions.

Two honesty rules, which override the above:
- Read before you cite. Never cite a line number you have not actually read; do not \
  reconstruct code from the kernel name or from what the schedule implies it must do.
- If the source contradicts the live registers, say so plainly and treat the \
  registers as ground truth for what the hardware did, the source as ground truth for \
  what was intended. That gap is usually the bug — it is the most valuable thing you \
  can report, not an inconsistency to smooth over.

## MANDATORY precondition: check session provenance first

A debug target is configured from the environment at startup, so the presence of a \
target does NOT mean a board is live or that anything has been run. Every message you \
receive carries a `Session:` line. Before reading hardware or interpreting any log:

- **"NO BOARD SESSION"** — the user has not connected, run, or attached. Live reads are \
  blocked and will return an error. Do NOT speculate about board state. Tell the user to \
  press "Connect", "Run", or "Open Current Session", and say what you would check once \
  they do.
- **"CONNECTED … but NO run has been started"** — the JTAG link is verified, but whatever \
  the registers hold is left over from some earlier run, possibly days old or from another \
  user. You may read, but you MUST label the findings as pre-existing board state, not the \
  result of a run in this session.
- **"ATTACHED … started OUTSIDE this UI"** — a real run is in play but the daemon did not \
  start it and cannot vouch for what came before. Trust live registers; do not assume the \
  run began cleanly.
- **"RAN from this UI"** — only here may you treat live state and the applog as describing \
  the current run.

The same rule governs logs: if `get_applog` reports the file predates this session, it \
describes a PREVIOUS run. A "PASS" in a stale log is not evidence that anything succeeded \
now. Never present stale state as current — say plainly which it is.

## Your role (overview)
Help the user understand their compiled AIE design, interpret DMA schedule data, \
diagnose performance issues (supply/demand imbalance, BD chain misconfiguration), \
and control live hardware or simulator sessions. You have direct access to the compiled \
schedule and live register state via MCP tools.

## The debug UI — what the user sees

The window is four panes, each with its name printed at its top-left. Use these \
names when you refer to the UI, and expect the user to use them too — "the Info \
pane", "the Tools pane". The table maps each one to the tool that answers \
questions about it; `get_ui_state()` tells you what is selected in them right now \
(`view`, `selected_tile`, `tile_tab`, `net_tab`, `flow`, `channel`, `console_pane`).

| Pane | Position | Contains | Ask which tool |
|---|---|---|---|
| **AIE Debug** | top-left | the array itself, in one of two views | `get_design_overview`, `tile_list`, `get_flow_detail` |
| **Execution** | bottom-left | app + board selection, run buttons, run log | `get_backend_status`, `get_applog`, `get_sim_log` |
| **Info** | top-right | detail for whatever is selected | `tile_info`, `get_flow_detail`, `get_pane` |
| **Tools** | bottom-right | aiegdb console, this chat, Search | `aie_exec`, `symbol_search` |

**AIE Debug (top-left)** — a `Grid` / `Device Map` switcher over the same array; \
`get_ui_state().view` is `"grid"` or `"map"`.
- *Grid*: one clickable cell per tile — type (shim / mem / core), DMA channel \
badges, colour-coded contract balance (green OK, amber/red imbalance).
- *Device Map*: SVG of the physical array with each DMA flow drawn as a coloured \
arc (f0, f1, …). Clicking an arc highlights every tile it touches and opens its \
net detail in **Info**; the "All nets / f0 / f1 …" chips filter what is drawn.
- Clicking a tile or channel badge in either view opens it in **Info** and \
prepends context to your next message. A tile with no compiled schedule clears \
**Info** and says "no schedule info" — that is a real empty selection, not a bug.
- Live overlay controls sit here: the `DMA / Cores / Events / Switch` selector \
**chooses** what to read, `Scan` reads it **once**, and the `live` checkbox \
re-reads every 2s. Each scan is summarized for you — call `get_live_scan()` \
to read the latest result and how to interpret it (see the `live-scan-results` \
skill). Picking a mode does not itself read the board unless live is on.

**Execution (bottom-left)** — `App:` and `Board:` selectors, then `Connect`, \
`Open Current Session`, `Run`, `Force stop`, and the run log beneath them.
- The board is chosen in the hostname box beside the `Board:` selector, per run — \
the device entry itself names no board.
- Controls are absent when there is nothing to choose. Serving ONE app removes the \
`App:` row (its name moves beside the `AIE Debug` pane title). `--sim-only` removes \
the whole `Board:` row AND `Connect` AND `Attach existing run` — the simulator is \
activated automatically on load, so `Run` is the only button, and the daemon \
REFUSES connect / attach / board-run / retarget / hw_server with `sim_only: true`. \
Check `get_backend_status()` before telling the user to pick a board or press \
Connect: under sim-only neither exists and those endpoints will not answer.
- A run banner reports `run #N on <device> started <time> (<age> ago) — <status>`, \
and flags a run whose log has gone quiet as stale.
- Everything the log shows is `get_applog` (hardware) or `get_sim_log` (simulator).

**Info (top-right)** — detail for the current selection, as a strip of tabs (the \
user can keep several tiles and nets open at once; ctrl+click adds).
- For a tile: *Schedule* (role, kernel, transfer summary, supply/demand verdict, \
channel↔kernel arg map) = `tile.hi`, *IR* (dfschedule IR slice) = `tile.mid`, \
*Code* = `tile.lo` — `tile_info(col, row, section)`, or `get_pane("tile.hi"|…)`.
- The tab strip follows what the app carries. Only the aiehlc pipeline emits a \
dfschedule IR and a line-attributed host.cc, so for those apps *Code* leads with \
the tile's host.cc lines and folds the kernel sources beneath. A naiebaremetal \
app has neither: it shows **no IR tab at all**, and *Code* is the kernel source \
the developer wrote, the .bcf buffer address map, then the generated wrapper. \
Do not tell the user to open a tab their app does not have.
- For a net: producer/consumer stages and hop routing — `get_flow_detail(flow)`.

**Tools (bottom-right)** — three tabs; `get_ui_state().console_pane` says which is \
open (`conpane` / `llmpane` / `searchpane`).
1. **aiegdb** — the interactive console. The same commands you run with `aie_exec`, \
so anything you can do the user can reproduce there verbatim.
2. **LLM** — this chat.
3. **Search** — symbol search over the compiled design (kernel, window, net, buffer, \
BD length, flow, GMIO); matches highlight on the map. Same index as `symbol_search`.

When a user asks about "this pane" or "what I'm looking at", call `get_ui_state()` \
first and resolve it against the table above rather than guessing.

**Where this app lives on disk**
- App: {_ap["app"]}
- App directory: {_ap["app_dir"]}
  The app root. Sources, build scripts and the host ELF live here — this is the \
directory to search when the user asks about kernel code, host code or build config.
- Provenance bundle: {_ap["bundle_dir"]}
  A subdirectory of the app holding the generated schedule JSONs the static tools \
read. It is NOT the app root; do not look for sources here.
- aiecompiler Work/: {_work_dir_str}
  Sibling of the bundle. Holds aiecompiler output — generated kernel wrappers, \
.bcf buffer maps, ps/c_rts/aie_control_config.json.
- Host ELF: {_elf_str}
- aiedbg clone: {_aiedbg_str}
  Not part of the app — the debug tool itself. Its docs live ONLY in the clone \
(the pip-installed package ships none); the `aiedbg-reference` skill indexes them.

Your working directory is the aiehlc_aiesim repo root, {_REPO_ROOT} — so a repo-relative \
path like `src/tool/debug/aiegdb.py` (which is how the skills below refer to repo files) \
resolves as-is, while app paths above are outside it and must be used absolute.

You may read any of these with your file tools. Paths above are absolute; prefer them \
over guessing relative paths.

**The application's own source** (paths relative to the app directory above; \
`app_sources()` returns this same inventory, refreshed, if the user switches apps):

```
{_src_block}
```

Read these with your file tools. Do not guess at filenames that are not in this \
list — if the code you need is not here, say so and ask, rather than describing a \
plausible-sounding file that does not exist.

**Cite source locations as `<file>:<line>`** — `host.cc:412`, \
`src/tool/debug/aiegdb.py:88`, or an absolute path. The UI turns that into a click \
that opens the file in the Info pane, highlighted at that line, so the user can read \
the code you are describing without leaving the page. A bare filename with the line \
mentioned separately ("host.cc, line 412") is NOT clickable. For files outside the \
loaded app, give a path rather than a bare filename. If the user switches apps mid-conversation these change — \
the `App:` field on each message's context line is authoritative.

**Current session state**{grid_summary}
- Configured debug target: {target_str}  (from the environment — NOT proof of a live board)
- Backend: {backend_str}{_sim_avail_str}
- Session at spawn: {self.session_summary()}

This block is a snapshot from when this conversation started. The `Session:` line on each \
message is authoritative and current — prefer it.
{_skills_block}
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
- `app_sources()` — the loaded app's own source files, grouped, plus the \
  file:line defining each kernel the schedule runs. Refreshes the inventory below \
  after an app switch. Read what it names; do not guess filenames.
- `get_backend_status()` — current backend, AIEDBG_TARGET, IPC readiness
- `get_live_scan(detail)` — latest DMA/Cores/Events/Switch scan the user ran \
  in the AIE Debug pane (counts, stalled tiles, switch mismatches). \
  **Call this when the user asks about scan colours, stalls, or routing \
  mismatches** — do not re-scan with aie_exec unless you need finer detail.
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

{_sim_section}
## Suggested workflow

1. Call `get_design_overview()` to orient yourself.
2. When the user clicks a tile, you receive `[context] Selected: tile (C,R) …` — \
   call `tile_info(C, R)` for the full detail.
3. When the user clicks a flow, you receive `[context] Selected net/flow fN …` — \
   call `get_flow_detail(N)` for the routing detail.
4. When the user has scanned the array (or asks about coloured tiles / stalls / \
   switch mismatches), call `get_live_scan()` before issuing your own reads.
5. For finer per-register detail after a scan, check `get_backend_status()` first; \
   if connected, use `aie_exec` commands to read DMA/core registers.
6. If the user has just run and wants to check results, call `get_applog()` or \
   `get_sim_log()` depending on the backend.
{_workflow_step6}
7. Before you answer, read the application source behind whatever you found — the \
   kernel for the tile, the graph/host source for the transfer — and cite it as \
   `<file>:<line>`. An answer that never leaves the register dump is not finished.
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
        # The hardware branch used to report only the target — which is set from
        # $AIEDBG_TARGET at startup — so every message asserted a live board even
        # when the user had done nothing. Session provenance is now mandatory on
        # both branches.
        summary = s.get("session_summary", "")
        # The app can be switched mid-conversation but the system prompt is only
        # sent on the first turn, so the app's location rides along on every
        # message the same way session provenance does.
        ap = s.get("app_paths") or {}
        app_str = ""
        if ap.get("app_dir"):
            app_str = f" App: {ap.get('app', '?')} at {ap['app_dir']}"
            if ap.get("work_dir"):
                app_str += f" (Work/: {ap['work_dir']})"
            app_str += "."
        if backend == "simulator":
            state = ("IPC ready — live register reads active" if ipc_ready
                     else "IPC not ready — simulator not started yet")
            return (f"[context] Backend: simulator ({state}). "
                    f"Session: {summary}{app_str}")
        elif backend == "hardware":
            return (f"[context] Backend: hardware, target={target or 'unknown'}. "
                    f"Session: {summary}{app_str}")
        return f"[context] Backend: {backend}. Session: {summary}{app_str}"

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
        # Load the debug-skill plugin explicitly rather than relying on cwd
        # auto-discovery, which `--bare` will eventually switch off by default.
        if os.path.isdir(_LLM_SKILLS_DIR):
            cmd += ["--plugin-dir", _LLM_PLUGIN_DIR]
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
                    "mcp__debugui__get_ipc_log",
                    "mcp__debugui__get_live_scan",
                    # App / UI awareness. These were registered in debug_ui_mcp
                    # but never granted, so the assistant could not answer "which
                    # app is this?" or see what the user had open — the very
                    # context it needs to follow along. select_app is deliberately
                    # NOT granted: switching the app reconfigures the whole
                    # server and belongs to the user, not the assistant.
                    "mcp__debugui__list_apps",
                    "mcp__debugui__current_app",
                    "mcp__debugui__app_sources",
                    "mcp__debugui__get_ui_state",
                    "mcp__debugui__list_panes",
                    "mcp__debugui__get_pane"]
            # --allowedTools is only a permission allowlist, and permission-mode
            # bypassPermissions already waives prompting — so omitting a tool from
            # the list above does NOT make it uncallable. Denying is the only way
            # to actually withhold one, and select_app is worth withholding: it
            # reconfigures the whole server (board IPs, PDIs, ELF paths) out from
            # under the user mid-conversation.
            cmd += ["--disallowedTools", "mcp__debugui__select_app"]
        self._llm_system_prompt_text = self._llm_system_prompt()
        self._llm_first_turn = True
        if self.claude_model:
            cmd += ["--model", self.claude_model]
        self._llm_proc = subprocess.Popen(
            cmd, cwd=self.claude_cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        proc = self._llm_proc
        self._llm_generation += 1

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
                if self._llm_proc is proc:
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
                            self._llm_last_output = time.monotonic()
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
                    compact = _llm_tool_args(inp)
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
                        # A size/outcome tag, not the payload: results run to
                        # register dumps, and the browser renders this inline
                        # on the call's own row.
                        n = len(result_text.splitlines()) if result_text else 0
                        tag = ("error" if blk.get("is_error")
                               else f"{n} lines" if n else "empty")
                        self._llm_append(f"\n[tool result: {tag}]\n")
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
            self._llm_last_output = time.monotonic()
            self._llm_log_in_asst = False
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
        return {
            "ok": True, "offset": offset,
            "llm_generation": self._llm_generation,
        }

    def _llm_proc_alive(self):
        proc = self._llm_proc
        return proc is not None and proc.poll() is None

    def llm_poll(self, offset):
        """Return the transcript slice past `offset` plus the turn-active flag.

        When active is True but no stdout has arrived, the turn stays active while
        the claude subprocess is running (no client signal).  stuck=True is returned
        only when the turn is hard-abandoned: subprocess exited with no recent
        output, or silence exceeds _LLM_STUCK_HARD_S.
        """
        with self._llm_lock:
            buf = self._llm_buf
            active = self._llm_active
            last = self._llm_last_output
            stuck = False
            stuck_s = None
            if active and last is not None:
                age = time.monotonic() - last
                if age >= _LLM_STUCK_HARD_S:
                    stuck = True
                    stuck_s = int(age)
                    self._llm_active = False
                    active = False
                    self._llm_log_write(
                        f"\n[watchdog: no output for {stuck_s}s — turn abandoned]\n"
                    )
                elif age >= _LLM_STUCK_S and not self._llm_proc_alive():
                    stuck = True
                    stuck_s = int(age)
                    self._llm_active = False
                    active = False
                    self._llm_log_write(
                        f"\n[watchdog: no output for {stuck_s}s — claude exited]\n"
                    )
        if offset < 0:
            offset = 0
        result = {
            "data": buf[offset:], "next": len(buf), "active": active,
            "llm_generation": self._llm_generation,
            "llm_reset_reason": self._llm_reset_reason,
        }
        if stuck:
            result["stuck"] = True
            result["stuck_s"] = stuck_s
        return result

    def llm_reset(self, reason="new chat"):
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
            self._llm_last_output = None
            self._llm_log_in_asst = False
            self._llm_first_turn = True
            self._llm_reset_reason = reason
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
        return {
            "ok": True,
            "llm_generation": self._llm_generation,
            "llm_reset_reason": self._llm_reset_reason,
        }


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


def run_xsdb_targets(target, timeout=15):
    """List JTAG debug targets (scan chain only, no register reads).

    Returns the list immediately so the browser can render skeleton cards.
    Per-target register/bt detail is fetched separately via /targets/detail.

    Returns:
      {"targets": [{id, indent, name, state, suspended, running, pc, sp, lr,
                    cpsr, elf, stack}]}
    or {"error": str} on failure.
    """
    host, port = _target_to_tcp(target)
    if not host:
        return {"error": "no host in target"}
    script = r"""
if {[catch {connect -url TCP:%(host)s:%(port)s} e]} {
    puts "TARGETS_FAIL $e"
    exit
}
set tgt_raw ""
catch { set tgt_raw [targets] }
puts "TGTLIST_BEGIN"
puts $tgt_raw
puts "TGTLIST_END"
puts "TARGETS_OK"
exit
""" % {"host": host, "port": port}
    try:
        result = subprocess.run(
            ["xsdb"], input=script,
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"error": "xsdb not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"xsdb targets timed out (TCP:{host}:{port})"}
    out = result.stdout or ""
    if "TARGETS_FAIL" in out:
        for line in out.splitlines():
            if "TARGETS_FAIL" in line:
                detail = line.split("TARGETS_FAIL", 1)[1].strip()
                return {"error": f"connect failed: {detail} (TCP:{host}:{port})"}
        return {"error": f"xsdb connect failed (TCP:{host}:{port})"}
    if "TARGETS_OK" not in out:
        return {"error": "xsdb exited without completing"}

    # xsdb `targets` output format — two possible column layouts:
    #
    #   Layout A (id in leading column, 2-space gap, then indented name):
    #     "  2  Cortex-A72 #0 (Running)"
    #     "  3     Cortex-A72 #1 (Suspended)"
    #
    #   Layout B (no leading whitespace, id flush-left):
    #     "1  Versal xcvc1902 (chipscope, ...)"
    #
    # xsdb encodes hierarchy in the leading spaces BEFORE the id, not after.
    # The id is right-aligned in an expanding column (~3 spaces per depth level).
    # indent = len(leading_spaces) + len(id_digits) = right-edge column of id,
    # which is identical across same-depth targets regardless of id digit count.
    # Groups: (1) leading spaces, (2) id digits, (3) name text, (4) state text.
    TGT_RE = re.compile(
        r"^( *)"               # group 1: leading spaces before id (depth indicator)
        r"(\d+)\*?"            # group 2: id digits; optional * consumed but not captured
        r"  "                  # fixed 2-space separator
        r"([^\(\n]+?)"         # group 3: name text (non-greedy, stops before '(' or \n)
        r"(?:\((.+)\))?"       # group 4: optional (State ...) — greedy, handles EL3(S)/A64
        r"\s*$"
    )
    targets_map = {}
    order = []
    in_list = False
    for line in out.splitlines():
        if "TGTLIST_BEGIN" in line:
            in_list = True
            continue
        if "TGTLIST_END" in line:
            in_list = False
            continue
        if not in_list:
            continue
        m = TGT_RE.match(line)
        if not m:
            continue
        tid_str = m.group(2)
        tid = int(tid_str)
        leading = len(m.group(1))
        indent = leading + len(tid_str)
        # The xsdb `*` marker (current target) replaces one leading space, making
        # that target appear one column to the left of its true depth.  Restore.
        if len(line) > leading + len(tid_str) and line[leading + len(tid_str)] == '*':
            indent += 1
        name  = m.group(3).strip()
        state = (m.group(4) or "").strip()
        targets_map[tid] = {
            "id": tid,
            "indent": indent,
            "name": name,
            "state": state,
            "suspended": bool(re.search(r"suspend|stop|halt", state, re.I)),
            "running":   bool(re.search(r"running", state, re.I)),
            "pc": None, "sp": None, "lr": None, "cpsr": None,
            "elf": False, "stack": [],
        }
        order.append(tid)

    # Include the raw xsdb text for browser-side debugging (console.log).
    raw_block = ""
    in_b = False
    for line in out.splitlines():
        if "TGTLIST_BEGIN" in line: in_b = True; continue
        if "TGTLIST_END"   in line: in_b = False; continue
        if in_b: raw_block += line + "\n"

    return {"targets": [targets_map[tid] for tid in order], "raw": raw_block}


def run_xsdb_target_detail(target, tid, timeout=20):
    """Read registers and backtrace for one already-stopped target.

    Called per-target after the skeleton list is rendered.  Does NOT halt or
    resume — only reads stopped/suspended cores (rrd on a live core blocks).
    Returns the same structure as run_xsdb_halt_read so the browser can use
    the same merge path.
    """
    host, port = _target_to_tcp(target)
    if not host:
        return {"error": "no host in target"}
    script = r"""
if {[catch {connect -url TCP:%(host)s:%(port)s} e]} {
    puts "DETAIL_FAIL $e"
    exit
}
proc xreg {name} {
    set raw ""
    catch { set raw [rrd $name] }
    if {[regexp {0x[0-9a-fA-F]+} $raw m]} { return $m }
    # Some xsdb versions return "name: XXXXXXXX" without the 0x prefix.
    if {[regexp {:\s*([0-9a-fA-F]+)\s*$} $raw m hex]} { return "0x$hex" }
    return $raw
}
catch {
    ta %(tid)s
    set pc   [xreg pc]
    set sp   [xreg sp]
    set lr   [xreg lr]
    set cpsr [xreg cpsr]
    if {$pc ne ""}   { puts "REG|%(tid)s|pc|$pc" }
    if {$sp ne ""}   { puts "REG|%(tid)s|sp|$sp" }
    if {$lr ne ""}   { puts "REG|%(tid)s|lr|$lr" }
    if {$cpsr ne ""} { puts "REG|%(tid)s|cpsr|$cpsr" }
    set elfinfo ""
    catch { set elfinfo [memmap] }
    if {$elfinfo ne ""} { puts "ELF|%(tid)s|1" }
    set frames ""
    catch { set frames [bt] }
    set fi 0
    foreach f [split $frames "\n"] {
        set f [string trim $f]
        if {$f ne ""} {
            puts "BT|%(tid)s|$fi|$f"
            incr fi
        }
    }
}
puts "DETAIL_OK"
exit
""" % {"host": host, "port": port, "tid": int(tid)}
    try:
        result = subprocess.run(
            ["xsdb"], input=script,
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"error": "xsdb not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"detail read timed out for target {tid}"}
    out = result.stdout or ""
    if "DETAIL_FAIL" in out:
        for line in out.splitlines():
            if "DETAIL_FAIL" in line:
                detail = line.split("DETAIL_FAIL", 1)[1].strip()
                return {"error": f"connect failed: {detail}"}
        return {"error": "xsdb connect failed"}
    if "DETAIL_OK" not in out:
        return {"error": "xsdb exited without completing"}

    data: dict = {}
    for line in out.splitlines():
        if line.startswith("REG|"):
            parts = line.split("|", 3)
            if len(parts) == 4 and parts[2] in ("pc", "sp", "lr", "cpsr"):
                data[parts[2]] = parts[3].strip()
        elif line.startswith("ELF|"):
            data["elf"] = True
        elif line.startswith("BT|"):
            parts = line.split("|", 3)
            if len(parts) == 4:
                data.setdefault("stack", []).append(_parse_bt_frame(parts[3].strip()))
    return data


def run_xsdb_halt_read(target, tid, timeout=15):
    """Halt one running target, read its registers + backtrace, then resume.

    This is intentionally invasive (50–200 ms stall) and only called when the
    user explicitly clicks the button.  Returns the same structure as a single
    entry from run_xsdb_targets so the browser can merge it in.
    """
    host, port = _target_to_tcp(target)
    if not host:
        return {"error": "no host in target"}
    script = r"""
if {[catch {connect -url TCP:%(host)s:%(port)s} e]} {
    puts "HALTREAD_FAIL $e"
    exit
}
proc xreg {name} {
    set raw ""
    catch { set raw [rrd $name] }
    if {[regexp {0x[0-9a-fA-F]+} $raw m]} { return $m }
    # Some xsdb versions return "name: XXXXXXXX" without the 0x prefix.
    if {[regexp {:\s*([0-9a-fA-F]+)\s*$} $raw m hex]} { return "0x$hex" }
    return $raw
}
catch {
    ta %(tid)s
    # Halt, read, resume.  `stop` and `con` are the xsdb halt/continue commands.
    catch { stop }
    after 80
    set pc   [xreg pc]
    set sp   [xreg sp]
    set lr   [xreg lr]
    set cpsr [xreg cpsr]
    if {$pc ne ""}   { puts "REG|%(tid)s|pc|$pc" }
    if {$sp ne ""}   { puts "REG|%(tid)s|sp|$sp" }
    if {$lr ne ""}   { puts "REG|%(tid)s|lr|$lr" }
    if {$cpsr ne ""} { puts "REG|%(tid)s|cpsr|$cpsr" }
    set elfinfo ""
    catch { set elfinfo [memmap] }
    if {$elfinfo ne ""} { puts "ELF|%(tid)s|1" }
    set frames ""
    catch { set frames [bt] }
    set fi 0
    foreach f [split $frames "\n"] {
        set f [string trim $f]
        if {$f ne ""} {
            puts "BT|%(tid)s|$fi|$f"
            incr fi
        }
    }
    # Resume unconditionally so we always leave the core running.
    catch { con }
}
puts "HALTREAD_OK"
exit
""" % {"host": host, "port": port, "tid": int(tid)}
    try:
        result = subprocess.run(
            ["xsdb"], input=script,
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return {"error": "xsdb not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"halt-read timed out — core may still be halted"}
    out = result.stdout or ""
    if "HALTREAD_FAIL" in out:
        detail = ""
        for line in out.splitlines():
            if "HALTREAD_FAIL" in line:
                detail = line.split("HALTREAD_FAIL", 1)[1].strip()
        return {"error": f"connect failed: {detail}"}
    if "HALTREAD_OK" not in out:
        return {"error": "xsdb exited without completing"}

    data: dict = {}
    for line in out.splitlines():
        if line.startswith("REG|"):
            parts = line.split("|", 3)
            if len(parts) == 4 and parts[2] in ("pc", "sp", "lr", "cpsr"):
                data[parts[2]] = parts[3].strip()
        elif line.startswith("ELF|"):
            data["elf"] = True
        elif line.startswith("BT|"):
            parts = line.split("|", 3)
            if len(parts) == 4:
                data.setdefault("stack", []).append(_parse_bt_frame(parts[3].strip()))
    return data


def _parse_bt_frame(frame_txt: str) -> dict:
    """Parse one XSDB bt line: GDB `in…at`, XSDB `func(): file, line N`, or address-only."""
    m = re.match(r"#?(\d+)\s+(?:(0x[0-9a-fA-F]+)\s+in\s+)?(.+?)\s+at\s+(.+):(\d+)", frame_txt)
    if m:
        return {"frame": int(m.group(1)), "addr": m.group(2) or "",
                "func": m.group(3).strip(), "file": m.group(4).strip(), "line": int(m.group(5))}
    m = re.match(r"(\d+)\s+(0x[0-9a-fA-F]+)\s+([^\s(]+)\(\):\s*(.+?),\s*line\s+(\d+)", frame_txt)
    if m:
        return {"frame": int(m.group(1)), "addr": m.group(2),
                "func": m.group(3), "file": m.group(4).strip(), "line": int(m.group(5))}
    m = re.match(r"(\d+)\s+(0x[0-9a-fA-F]+)\s+([^\s(]+)\(\)\s+\((.+):(\d+)\)", frame_txt)
    if m:
        return {"frame": int(m.group(1)), "addr": m.group(2),
                "func": m.group(3), "file": m.group(4).strip(), "line": int(m.group(5))}
    m = re.match(r"(\d+)\s+(0x[0-9a-fA-F]+)\s*$", frame_txt)
    if m:
        return {"frame": int(m.group(1)), "addr": m.group(2)}
    return {"raw": frame_txt}


def _addr2line(elf_path: str, addr: str) -> dict:
    """Resolve a hex address to file:line using addr2line on an ELF with DWARF.

    Returns {"file": str, "line": int, "func": str} or {} on failure/no-info.
    The host ELF has DWARF for XAie driver + BSP code; generated host_fixed.cc
    was compiled without -g so those addresses return '??:?' and are filtered out.
    """
    if not elf_path or not os.path.exists(elf_path):
        return {}
    # addr2line -C (demangle) -f (function name) -e <elf> <addr>
    # Output: two lines — function name then file:line (or "??:?" if no info)
    for tool in ("addr2line", "aarch64-linux-gnu-addr2line", "aarch64-none-elf-addr2line"):
        import shutil
        if not shutil.which(tool):
            continue
        try:
            r = subprocess.run(
                [tool, "-C", "-f", "-e", elf_path, addr],
                capture_output=True, text=True, timeout=5)
            lines = r.stdout.strip().splitlines()
            if len(lines) >= 2 and lines[1] != "??:0" and not lines[1].startswith("?"):
                loc = lines[1]
                func = lines[0] if lines[0] != "??" else ""
                if ":" in loc:
                    file_part, line_part = loc.rsplit(":", 1)
                    try:
                        return {"file": file_part.strip(), "line": int(line_part), "func": func}
                    except ValueError:
                        pass
        except Exception:
            pass
        break  # only try the first tool found
    return {}


def _nm_symbol(elf_path: str, addr: str) -> str:
    """Look up the nearest symbol name below addr using nm (fallback when no DWARF)."""
    if not elf_path or not os.path.exists(elf_path):
        return ""
    import shutil
    nm = shutil.which("nm") or shutil.which("aarch64-linux-gnu-nm") or shutil.which("aarch64-none-elf-nm")
    if not nm:
        return ""
    try:
        r = subprocess.run([nm, "-n", "--demangle", elf_path],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return ""
    # Strip "name: " prefix that some xsdb versions emit before the hex digits.
    raw = addr.strip()
    m = re.search(r'([0-9a-fA-F]+)\s*$', raw)
    if not m:
        return ""
    target_int = int(m.group(1), 16)
    best_sym, best_addr = "", 0
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            sym_addr = int(parts[0], 16)
        except ValueError:
            continue
        if sym_addr <= target_int and sym_addr > best_addr:
            best_addr = sym_addr
            best_sym = parts[2]
    return best_sym


def _run_aiedbg_scan(what, st, target, device=None):
    """Run 'aiedbg --json scan <what>' and return the parsed JSON dict, or an error dict."""
    tgt = target or st.target
    dev = device or st.device
    cmd = ["aiedbg", "--json"]
    if tgt:
        cmd += ["--target", tgt]
    if dev:
        cmd += ["-d", dev]
    cmd += ["scan", what]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return {"error": "aiedbg not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"aiedbg scan {what} timed out"}
    if r.returncode != 0:
        return {"error": r.stderr.strip() or f"aiedbg scan {what} failed"}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"error": f"bad json from aiedbg scan {what}"}


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
    order = ["unreachable", "error", "stalled", "running", "completed", "idle"]
    for s in order:
        if s in states:
            return s
    return "idle"


def _make_reg_read_fn(st, device, target):
    """Return a reg-read callable for (phys_col, row, offset) -> int|None.

    Only returns a function for the simulator (IPC socket reads).
    Hardware uses the batch aiedbg scan path (reg_read_fn=None) so the
    grid functions call _run_aiedbg_scan once instead of one subprocess per tile.
    """
    if (device or "").strip().lower() == "simulator":
        return st.sim_ipc_reg_read
    return None


_DMA_CH_STATE_MAP = {
    "started": "running", "done": "idle", "idle": "idle",
    "running": "running", "stalled": "stalled", "error": "error",
}


def grid_dma(st, target=None, reg_read_fn=None, device=None):
    cells = {}

    # Simulator path: per-channel IPC reads (no xsdb).
    if reg_read_fn is not None:
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

    # Hardware path: single 'aiedbg --json scan dma' subprocess.
    data = _run_aiedbg_scan("dma", st, target, device)
    if "error" in data:
        return {"what": "dma", "cells": {}, "error": data["error"]}
    # scan dma uses underscore channel keys (s2mm_0); grid uses no-underscore (s2mm0).
    scan_by_phys = {(int(t["col"]), int(t["row"])): t for t in data.get("tiles", [])}
    for t in st.tiles():
        col, row = t["loc"][0], t["loc"][1]
        phys_col = col + st.startcol
        entry = scan_by_phys.get((phys_col, row))
        if entry is None:
            cells[f"{col},{row}"] = {"state": "unreachable", "type": t["type"],
                                     "phys_col": phys_col, "channels": {}}
            continue
        chans = {}
        for ch_key, ch_val in entry.get("channels", {}).items():
            key = ch_key.replace("_", "")  # s2mm_0 -> s2mm0
            raw_state = ch_val.get("state", "idle")
            active_evts = ch_val.get("active_events", [])
            # Derive human-readable stall/error lists from the event names that
            # aiedbg scan dma includes in active_events (e.g. "STALLED_LOCK",
            # "STREAM_STARVATION", "BD_UNAVAILABLE", "BD_INVALID").
            stalls = []
            errors = []
            for ev in active_evts:
                ev_up = ev.upper()
                if "STALLED_LOCK" in ev_up or "LOCK_STALL" in ev_up:
                    if "lock_acq" not in stalls:
                        stalls.append("lock_acq")
                if "STREAM_STARVATION" in ev_up or "STREAM_BACKPRESSURE" in ev_up or "STREAM_STALL" in ev_up:
                    if "stream" not in stalls:
                        stalls.append("stream")
                if "MEMORY_BACKPRESSURE" in ev_up or "MEMORY_STARVATION" in ev_up or "MEMORY_STALL" in ev_up:
                    if "tct" not in stalls:
                        stalls.append("tct")
                if "BD_UNAVAIL" in ev_up or "BD_UNAVAILABLE" in ev_up:
                    if "bd_unavail" not in errors:
                        errors.append("bd_unavail")
                if "BD_INVALID" in ev_up:
                    if "bd_invalid" not in errors:
                        errors.append("bd_invalid")
            mapped_state = _DMA_CH_STATE_MAP.get(raw_state, "unknown")
            # Shim event registers are sticky — stall bits latch during a run
            # and remain set after it completes.  When the scan reports "stalled"
            # we verify against the live DMA status register: if the channel is
            # not actually running and has no queued BD, it finished and the
            # latch is post-run residue.  Reclassify so the UI distinguishes a
            # genuine live stall from a completed channel.
            cur_bd = None
            q_size = None
            if mapped_state == "stalled":
                ch_dir, ch_num = key[:4], int(key[4:])  # "mm2s"/"s2mm", 0/1
                status = _read_dma_channel(st, t["type"], phys_col, row,
                                           ch_dir, ch_num,
                                           target=target or st.target,
                                           reg_read_fn=None)
                if not status.get("running") and status.get("q_size", 1) == 0:
                    mapped_state = "completed"
                cur_bd = status.get("cur_bd")
                q_size = status.get("q_size")
            chans[key] = {"state": mapped_state,
                          "active_events": active_evts,
                          "stalls": stalls,
                          "errors": errors,
                          **({"cur_bd": cur_bd, "q_size": q_size}
                             if cur_bd is not None else {})}
        state = _worst([v["state"] for v in chans.values()]) if chans else "idle"
        cells[f"{col},{row}"] = {"state": state, "type": t["type"],
                                 "phys_col": phys_col, "channels": chans}
    return {"what": "dma", "cells": cells}


def grid_cores(st, target=None, reg_read_fn=None, device=None):
    cells = {}

    # Simulator path: per-tile IPC reads via reg_read_fn (no xsdb involved).
    if reg_read_fn is not None:
        entries, _ = aiediag.load_linemap()
        for t in st.tiles():
            col, row = t["loc"][0], t["loc"][1]
            if t["type"] != "core":
                continue
            phys_col = col + st.startcol
            raw = reg_read_fn(phys_col, row, aiediag.CORE_PC_OFFSET)
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

    # Hardware path: single 'aiedbg --json scan cores' subprocess.
    _CORE_STATE_MAP = {
        "enabled": "running", "disabled": "idle", "stalled": "stalled",
        "done": "idle", "error_halt": "error", "debug_halt": "stalled",
        "reset": "idle",
    }
    data = _run_aiedbg_scan("cores", st, target, device)
    if "error" in data:
        return {"what": "cores", "cells": {}, "error": data["error"]}
    scan_by_phys = {(int(c["col"]), int(c["row"])): c for c in data.get("cores", [])}
    for t in st.tiles():
        col, row = t["loc"][0], t["loc"][1]
        if t["type"] != "core":
            continue
        phys_col = col + st.startcol
        entry = scan_by_phys.get((phys_col, row))
        if entry is None:
            cells[f"{col},{row}"] = {"state": "unreachable"}
            continue
        raw_state = entry.get("state", "unknown")
        cells[f"{col},{row}"] = {
            "state": _CORE_STATE_MAP.get(raw_state, "unknown"),
            "core_status": raw_state,
            "reg": f"0x{entry.get('core_status_reg', 0):08X}",
        }
    return {"what": "cores", "cells": cells}


def _switch_tiles(st):
    """[(col, row, tile_type)] for every tile with a stream switch to read.

    Not st.tiles(): memtile rows carry real circuit connections but never
    appear in the schedule grid, so the routing map has to be consulted too.

    The whole grid rectangle is included as well, because reconstruction has to
    work when the routing map is missing or wrong -- seeding the tile list from
    that map would make discovery find only what the map already claims.  Note
    the rows come from the grid extent, not `row_list`: that list skips the
    memtile rows, which routing demonstrably crosses.
    """
    grid = st.grid_info()
    core_min = grid.get("device_core_min_row") or 3
    locs = {tuple(t["loc"]) for t in st.tiles()}
    cols = grid.get("col_list") or list(range(grid.get("cols") or 0))
    for col in cols:
        for row in range(grid.get("rows") or 0):
            locs.add((col, row))
    for p in st.comm_paths():
        for e in p.get("edges", []):
            locs.add((e[0][0], e[0][1]))
            locs.add((e[1][0], e[1][1]))
        for c in p.get("routing_connections", []):
            t = c.get("tile") or {}
            if t.get("col") is not None:
                locs.add((t["col"], t["row"]))
    out = []
    for col, row in sorted(locs):
        ttype = "shim" if row == 0 else ("core" if row >= core_min else "memtile")
        out.append((col, row, ttype))
    return out


def grid_switch(st, target=None, reg_read_fn=None, device=None,
                routing_map='static'):
    """Live stream-switch config per tile, diffed against a routing map.

    `routing_map` selects which comm_paths expected_records uses:
      static  — the compiler map in schedule_view.json (default)
      dynamic — flows reconstructed from the registers just read on this scan

    This is static configuration, not per-cycle state, so it is a scan and
    never a poll.  On hardware each tile costs a single `aiedbg mem read` of
    the whole switch register block; reading it the way aiegdb's `show switch`
    does would be ~230 `reg read` subprocesses per tile.
    """
    cells = {}
    switches = {}
    for col, row, ttype in _switch_tiles(st):
        phys_col = col + st.startcol
        try:
            decoded = switch_scan.read_tile_switch(
                ttype, phys_col, row,
                reg_read_fn=reg_read_fn, target=target or st.target,
                device=device)
            if decoded is None:
                cells[f"{col},{row}"] = {
                    "state": switch_scan.UNREACHABLE,
                    "phys_col": phys_col, "matched": 0,
                    "missing": [], "unexpected": []}
            else:
                switches[(col, row)] = (ttype, decoded)
        except Exception as e:
            cells[f"{col},{row}"] = {
                "state": switch_scan.UNREACHABLE,
                "phys_col": phys_col, "matched": 0,
                "missing": [], "unexpected": [], "error": str(e)}
        if f"{col},{row}" in cells:
            cells[f"{col},{row}"]["type"] = ttype

    dynamic = _dynamic_routing(st, switches)
    st._last_dynamic_routing = dynamic

    comm_paths = st.comm_paths()
    if (routing_map or "static").strip().lower() == "dynamic":
        if dynamic and dynamic.get("comm_paths"):
            comm_paths = dynamic["comm_paths"]

    for (col, row), (ttype, decoded) in switches.items():
        phys_col = col + st.startcol
        live = switch_scan.live_records(ttype, col, row, decoded)
        expected = switch_scan.expected_records(comm_paths, col, row)
        cell = switch_scan.compare_tile(live, expected)
        cell["phys_col"] = phys_col
        cell["type"] = ttype
        cells[f"{col},{row}"] = cell

    n_bad = sum(1 for c in cells.values() if c["state"] == switch_scan.MISMATCH)
    return {"what": "switch", "cells": cells, "mismatch_tiles": n_bad,
            "dynamic": dynamic, "routing_map": routing_map or "static"}


def _dynamic_routing(st, switches):
    """Rebuild the routing map from the switches just read.

    Never raises: a reconstruction problem must not cost the user the scan
    result they asked for, which is still valid on its own.
    """
    if not switches:
        return None
    try:
        flows = switch_reconstruct.discover_flows(switches, st.aie_version)
        flows = switch_reconstruct.reconcile(flows, st.comm_paths())
        groups = switch_reconstruct.to_routing_groups(flows, st.startcol)
        out = {"n_flows": len(flows),
               "comm_paths": switch_reconstruct.to_comm_paths(flows),
               "routing_groups": groups,
               "scanned_tiles": len(switches)}
    except Exception as e:
        return {"error": str(e)}
    path = os.path.join(st.workdir, "routingprovenancemap.dynamic.json")
    try:
        with open(path, "w") as f:
            json.dump(groups, f, indent=1)
        out["path"] = path
    except OSError as e:
        out["write_error"] = str(e)
    return out


def grid_events(st, target=None, reg_read_fn=None, device=None):
    cells = {}

    # Simulator path: per-tile IPC reads (no xsdb).
    if reg_read_fn is not None:
        for t in st.tiles():
            col, row = t["loc"][0], t["loc"][1]
            phys_col = col + st.startcol
            if t["type"] == "shim":
                regs = (aiediag.SHIM_EVT_STATUS_REG0, aiediag.SHIM_EVT_STATUS_REG1)
            else:
                regs = aiediag.MEM_EVT_STATUS_REGS
            words = []
            for off in regs:
                raw = reg_read_fn(phys_col, row, off)
                words.append(None if raw is None else f"0x{raw:08X}")
            if all(w is None for w in words):
                cells[f"{col},{row}"] = {"state": "unreachable"}
                continue
            any_set = any(w not in (None, "0x00000000") for w in words)
            cells[f"{col},{row}"] = {"state": "running" if any_set else "idle",
                                     "words": words}
        return {"what": "events", "cells": cells}

    # Hardware path: reuse 'aiedbg --json scan dma' — it includes event_status_hex
    # for every tile in a single batched xsdb call.
    data = _run_aiedbg_scan("dma", st, target, device)
    if "error" in data:
        return {"what": "events", "cells": {}, "error": data["error"]}
    scan_by_phys = {(int(t["col"]), int(t["row"])): t for t in data.get("tiles", [])}
    for t in st.tiles():
        col, row = t["loc"][0], t["loc"][1]
        phys_col = col + st.startcol
        entry = scan_by_phys.get((phys_col, row))
        if entry is None:
            cells[f"{col},{row}"] = {"state": "unreachable"}
            continue
        words = entry.get("event_status_hex", [])
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

def _is_loopback(handler):
    try:
        return ipaddress.ip_address(handler.client_address[0]).is_loopback
    except (ValueError, IndexError, TypeError):
        return False


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

    def handle(self):
        # This is a plain-HTTP server. A browser that connects with https://
        # sends a TLS ClientHello, whose binary bytes the base parser logs as a
        # "Bad request version (<garbage>)" 400. Peek the first byte (without
        # consuming it) and, if it is a TLS handshake record (content type
        # 0x16), reply with a plaintext hint and log one clean line instead of
        # the binary noise. Normal HTTP requests fall through untouched.
        try:
            first = self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            first = b""
        if first[:1] == b"\x16":
            self.log_message("%s", "rejected TLS/https:// connection on plain "
                                    "HTTP port; use http:// instead")
            body = (b"This is a plain HTTP debug server. You connected with "
                    b"https:// (TLS). Reload the page using http:// instead.\n")
            try:
                self.connection.sendall(
                    b"HTTP/1.0 400 Bad Request\r\n"
                    b"Content-Type: text/plain; charset=utf-8\r\n"
                    b"Connection: close\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                    b"\r\n" + body)
            except OSError:
                pass
            self.close_connection = True
            return
        return super().handle()

    def _send_json(self, obj, code=200):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _refuse_sim_only(self, what):
        if not self.state.sim_only:
            return False
        self._send_json({"ok": False, "sim_only": True,
                         "error": f"{what} is unavailable: this daemon was "
                                  f"started with --sim-only and reaches no "
                                  f"board. Use the simulator, or restart "
                                  f"without --sim-only."})
        return True

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

    def _send_page(self, st):
        """Serve the UI with the selected app's data injected server-side.

        The template's JS consumes DATA synchronously at parse time, so the data
        is substituted here rather than fetched by the client. Falls back to a
        pre-generated host_schedule.html when no app is registered (e.g. a bare
        workdir with only the old artifact)."""
        app = st.app
        if app is None:
            self._send_file(st.html_path(), "text/html; charset=utf-8")
            return
        try:
            import importlib
            importlib.reload(schedule_view)
            html = schedule_view.render_html(app.load_view())
        except (OSError, ValueError) as e:
            self._send_json({"error": f"cannot render app {app.id}: {e}"}, code=500)
            return
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        st = self.state
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        if path in ("/", "/index.html", "/host_schedule.html"):
            self._send_page(st)
        elif path == "/apps":
            self._send_json({"apps": st.apps_info()})
        elif path == "/uistate":
            self._send_json(st.get_uistate())
        elif path == "/schedule_view.json":
            self._send_file(st.json_path(), "application/json")
        elif path == "/timeline.png":
            # Rendered by launch_timeline() when a run ends; the Profile tab
            # fetches it with a cache-busting ?ts= so a new run shows a new PNG.
            self._send_file(st.timeline_png_path(), "image/png")
        elif path == "/config":
            self._send_json(_ui_defaults(st))
        elif path == "/aiegdb/spec":
            # Served straight from the imported module, not the aiegdb
            # subprocess: the console's autocomplete must populate on page load
            # even before a command has been run or a board is reachable.
            self._send_json({"spec": aiegdb.command_spec()})
        elif path == "/aiegdb/arghelp":
            # Per-argument help for ONE command, fetched lazily by the console
            # when the user reaches an argument — doing all of them up front
            # would cost a subprocess call per command on every page render.
            self._send_json({"args": aiegdb.arg_help(q.get("cmd", [""])[0])})
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
            self._send_json({"devices": _devices_for_ui(st)})
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
        elif path == "/source":
            if not _check_llm_auth(st, self):
                self._send_json({"error": "unauthorized", "auth": True}, code=401)
                return
            obj, code = _serve_source(st, q)
            self._send_json(obj, code=code)
        elif path == "/runstate":
            self._send_json(st.run_state())
        elif path == "/ping":
            if self._refuse_sim_only("connecting to a board"):
                return
            if st.run_in_progress():
                # `busy` distinguishes "a run owns the link" from "the link is
                # dead"; without it the UI ssh'd to the board to restart a
                # working hw_server.
                self._send_json({"ok": False, "busy": True,
                                 "detail": "disabled during run",
                                 "run": st.run_state()})
                return
            device = q.get("device", [""])[0]
            host = q.get("host", [""])[0]
            res = ping(st, device, host)
            # A verified probe is the weakest form of authorization: the link is
            # real, but nothing has been run, so live reads stay clearly labelled
            # as "connected, no run this session".
            if res.get("ok"):
                st.mark_hw_session("connected", res.get("detail", ""),
                                   res.get("target"))
            self._send_json(res)
        elif path == "/targets":
            if self._refuse_sim_only("listing JTAG targets"):
                return
            if not st.hw_authorized():
                self._send_json({"error": _NO_SESSION_MSG, "targets": []})
                return
            device = q.get("device", [""])[0]
            host = q.get("host", [""])[0]
            tgt = resolve_target(st, device, host)
            if not tgt:
                self._send_json({"error": "no JTAG target configured", "targets": []})
                return
            self._send_json(run_xsdb_targets(tgt))
        elif path == "/targets/detail":
            if not st.hw_authorized():
                self._send_json({"error": _NO_SESSION_MSG})
                return
            tid = q.get("id", [""])[0]
            if not tid:
                self._send_json({"error": "missing ?id="})
                return
            device = q.get("device", [""])[0]
            host = q.get("host", [""])[0]
            tgt = resolve_target(st, device, host)
            if not tgt:
                self._send_json({"error": "no JTAG target configured"})
                return
            self._send_json(run_xsdb_target_detail(tgt, tid))
        elif path == "/targets/halt_read":
            if not st.hw_authorized():
                self._send_json({"error": _NO_SESSION_MSG})
                return
            tid = q.get("id", [""])[0]
            if not tid:
                self._send_json({"error": "missing ?id="})
                return
            device = q.get("device", [""])[0]
            host_param = q.get("host", [""])[0]
            tgt = resolve_target(st, device, host_param)
            if not tgt:
                self._send_json({"error": "no JTAG target configured"})
                return
            data = run_xsdb_halt_read(tgt, tid)
            if "error" not in data:
                # Enrich with addr2line / nm resolution from the app's host ELF.
                elf = (st.app_paths() or {}).get("elf") or ""
                if elf and os.path.exists(elf):
                    # Resolve the PC itself.
                    if data.get("pc"):
                        loc = _addr2line(elf, data["pc"])
                        if loc:
                            data["pc_loc"] = loc
                        elif data.get("pc"):
                            # Fall back to nearest symbol name.
                            sym = _nm_symbol(elf, data["pc"])
                            if sym:
                                data["pc_sym"] = sym
                    # Enrich each stack frame that has an address but no file.
                    for frame in data.get("stack", []):
                        if frame.get("file"):
                            continue  # already resolved by xsdb bt
                        addr = frame.get("addr")
                        if not addr:
                            continue
                        loc = _addr2line(elf, addr)
                        if loc:
                            frame["file"] = loc["file"]
                            frame["line"] = loc["line"]
                            if not frame.get("func") or frame["func"] == "??":
                                frame["func"] = loc.get("func") or frame.get("func", "??")
                        elif not frame.get("func") or frame["func"] == "??":
                            sym = _nm_symbol(elf, addr)
                            if sym:
                                frame["func"] = sym
            self._send_json(data)
        elif path == "/grid":
            what = q.get("what", ["dma"])[0]
            device = q.get("device", [""])[0]
            host = q.get("host", [""])[0]
            routing_map = q.get("routing", ["static"])[0]
            is_sim = (device or "").strip().lower() == "simulator"
            if is_sim and st.sim_kind == "aiesim":
                self._send_json({"error": _AIESIM_LIVE_ERROR, "cells": {},
                                 "live_reads": False, "sim_kind": "aiesim"})
                return
            if st.run_blocks_debug():
                self._send_json({"error": "disabled during run setup", "cells": {}})
                return
            if not st.hw_authorized():
                self._send_json({"error": _NO_SESSION_MSG, "cells": {}})
                return
            if not is_sim and not _hw_available():
                self._send_json({"error": "aiedbg not found in PATH",
                                 "cells": {}})
                return
            tgt = resolve_target(st, device, host)
            dev = resolve_device(st, device)
            rrfn = _make_reg_read_fn(st, device, tgt)
            try:
                if what == "cores":
                    res = grid_cores(st, target=tgt,
                                     reg_read_fn=rrfn, device=dev)
                elif what == "events":
                    res = grid_events(st, target=tgt,
                                      reg_read_fn=rrfn, device=dev)
                elif what == "switch":
                    res = grid_switch(st, target=tgt,
                                      reg_read_fn=rrfn, device=dev,
                                      routing_map=routing_map)
                else:
                    res = grid_dma(st, target=tgt,
                                   reg_read_fn=rrfn, device=dev)
                st.record_live_scan(what, res, device=device, host=host)
                summary = st.get_live_scan()
                if summary:
                    res["llm_summary"] = live_scan_summary.format_for_llm(summary)
                self._send_json(res)
            except Exception as e:  # never crash the poll loop
                self._send_json({"error": str(e), "cells": {}}, code=500)
        elif path == "/live_scan":
            summary = st.get_live_scan()
            if not summary:
                self._send_json({"ok": False, "detail": "no scan recorded yet"})
                return
            self._send_json({"ok": True, "summary": summary,
                             "text": live_scan_summary.format_for_llm(summary)})
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
            if self._refuse_sim_only("starting a board run"):
                return
            if "startcol" in body:
                try:
                    st.startcol = int(body["startcol"])
                except (ValueError, TypeError):
                    pass
            res = st.start_run(device=body.get("device"),
                               board_host=body.get("board_host"))
            if res.get("error"):
                res.setdefault("run", st.run_state())
            self._send_json(res)
        elif u.path == "/apps/select":
            self._send_json(st.select_app(body.get("id")))
        elif u.path == "/apps/add":
            # Takes an arbitrary path and, with select=true, repoints workdir —
            # which rewrites /source's root allowlist. Loopback is allowed
            # unauthenticated because run_debug_ui.sh registers over 127.0.0.1
            # with no header; anything off-box needs the password.
            if not (_is_loopback(self) or _check_llm_auth(st, self)):
                self._send_json({"error": "unauthorized", "auth": True}, code=401)
                return
            self._send_json(st.add_app(body.get("path"), body.get("label"),
                                       bool(body.get("select"))))
        elif u.path == "/uistate":
            self._send_json(st.set_uistate(body))
        elif u.path == "/stop":
            res = st.stop_run()
            res["run"] = st.run_state()
            self._send_json(res)
        elif u.path == "/timeline":
            self._send_json(st.launch_timeline())
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
            if is_sim and st.sim_kind == "aiesim":
                self._send_json({"error": _AIESIM_LIVE_ERROR,
                                 "live_reads": False, "sim_kind": "aiesim"})
                return
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
            if self._refuse_sim_only("retargeting live debug at a board"):
                return
            # Switch the live-debug target (and respawn the aiegdb console so its
            # reads/aiedbg use it). The browser only POSTs here AFTER a
            # successful /ping, so the resolved target is already known reachable;
            # we just resolve device+host the same way /ping did and hand it to
            # retarget(). Without this, the aiegdb console stays pinned to the
            # daemon's startup target no matter what host the UI selects.
            dev = body.get("device", "")
            is_sim = (dev or "").strip().lower() == "simulator"
            if not is_sim and st.run_in_progress():
                self._send_json({"ok": False, "busy": True,
                                 "detail": "disabled during run",
                                 "run": st.run_state()})
                return
            if is_sim:
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
                self._send_json(st.retarget(tgt, resolve_device(st, dev)))
            except Exception as e:
                self._send_json({"ok": False, "target": tgt,
                                 "detail": str(e)}, code=500)
        elif u.path == "/aiegdb":
            if st.run_blocks_debug():
                self._send_json({"error": "disabled during run setup"})
                return
            cmd = body.get("cmd", "")
            dev = (body.get("device", "") or "").strip().lower()
            if (dev == "simulator" and st.sim_kind == "aiesim"
                    and _aiegdb_needs_live_transport(cmd)):
                self._send_json({"error": _AIESIM_LIVE_ERROR,
                                 "output": "ERROR: " + _AIESIM_LIVE_ERROR,
                                 "live_reads": False,
                                 "sim_kind": "aiesim"})
                return
            # No _hw_available hard-gate: navigation/help work without aiedbg;
            # live reads fail gracefully inside aiegdb.
            try:
                self._send_json(st.gdb_command(cmd))
            except Exception as e:
                self._send_json({"error": str(e)}, code=500)
        elif u.path == "/attach":
            if self._refuse_sim_only("attaching to a board run"):
                return
            # "Open Current Session": the user ran the app themselves (CLI, or a
            # board already programmed) and then opened the UI. Probe the link so
            # we don't authorize a dead target, but record the mode as `attached`
            # — the daemon did not start this run and cannot vouch for what the
            # board did before now.
            if st.run_in_progress():
                # Nothing to adopt, and probing JTAG would collide with it.
                self._send_json({"ok": False, "busy": True,
                                 "detail": "this daemon is already running a "
                                           "test — stop it first",
                                 "run": st.run_state()})
                return
            res = ping(st, body.get("device", ""), body.get("board_host", ""))
            if res.get("ok"):
                st.mark_hw_session("attached",
                                   "adopted a session started outside the UI",
                                   res.get("target"))
                res["session"] = st.session_state()
            self._send_json(res)
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
            if self._refuse_sim_only("starting hw_server on a board"):
                return
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

    * pal           → xsdb://<PALIP>:3121 ($PALIP, fallback xx.xx.xx.213).
    * simulator     → None (reads go through IPC debug socket, not aiedbg).
    * any other board + host → xsdb://<host>:3121 (hostname from the UI).
    * otherwise     → the daemon's configured/env target (st.target).
    """
    device = (device or "").strip().lower()
    if device == "pal":
        return f"xsdb://{os.environ.get('PALIP', 'xx.xx.xx.213')}:3121"
    if device == "simulator":
        return None
    if device and host:
        return f"xsdb://{host}:3121"
    return st.target


# The dropdown now carries aiedbg's own device names, so the UI selection IS the
# `aiedbg -d` value — the two used to be separate namespaces that never synced,
# leaving aiedbg on `pal` (12 columns) while the user worked a vek385 (38).
def resolve_device(st, device):
    """aiedbg -d name for a UI board selection, or None when it does not apply."""
    device = (device or "").strip().lower()
    if not device or device == "simulator":
        return None
    return device


def _ui_defaults(st):
    """Default board selection for the browser's device dropdown + hostname box.

    Derives the board hostname from $VEK385IP (what envlocal.sh exports) or, as a
    fallback, the host component of the daemon's resolved JTAG target
    (xsdb://<host>:port). When a hostname is known we preselect the vek385 device
    so the user doesn't have to pick it manually on every load.
    """
    if st.sim_only:
        return {"device": "simulator", "board_host": "", "sim_only": True,
                "source_viewer": True, "jtag_host": ""}
    host = os.environ.get("VEK385IP", "").strip()
    if not host and st.target:
        m = re.match(r"xsdb://([^:/]+)", st.target)
        if m:
            host = m.group(1)
    if not host:
        # Last resort: the checkout's own hwlocal.sh. This is a PREFILL, not a
        # binding — the box stays editable and the typed value wins per run.
        host = _expected_board_host(st.workdir)
    device = "vek385" if host else ""
    return {"device": device, "board_host": host,
            "sim_only": False, "source_viewer": True,
            "jtag_host": _jtag_host_for(device, host)}


def _jtag_host_for(device, board_host):
    """TCP host the browser's Connect probe uses (matches resolve_target)."""
    device = (device or "").strip().lower()
    if device == "pal":
        return os.environ.get("PALIP", "xx.xx.xx.213")
    host = (board_host or "").strip()
    return host


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


# ── per-user applog (shared checkout hosts) ──

def _default_applog_path():
    raw = os.environ.get("SCHEDULE_DEBUG_APPLOG")
    if raw:
        return os.path.abspath(raw)
    return os.path.join(_REPO_ROOT, f"applog.{getpass.getuser()}")


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


def _open_browser(url):
    """Best-effort open `url` in a browser, tolerant of headless/SSH sessions.

    In a VS Code Remote-SSH terminal, $BROWSER points at the `code` CLI, which
    tries to reach the editor over a per-session IPC socket
    ($VSCODE_IPC_HOOK_CLI). When that socket is stale (window closed/reconnected)
    the opener dies with ECONNREFUSED and a noisy Node deprecation warning. None
    of that should surface here: the daemon is already serving, so a failed
    auto-open must be silent and non-fatal. We try a plain opener first
    (xdg-open / open) and only fall back to webbrowser, suppressing all output
    and exceptions either way.
    """
    time.sleep(0.5)
    # Prefer a plain OS opener that never touches the VS Code IPC socket.
    for opener in ("xdg-open", "open"):
        path = shutil.which(opener)
        if not path:
            continue
        try:
            subprocess.Popen(
                [path, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return
        except OSError:
            continue
    # Last resort: webbrowser (may invoke the `code` opener over SSH). Swallow
    # any failure so a dead IPC socket can't crash the open thread.
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Live AIE debug/test daemon for host_schedule.html")
    ap.add_argument("workdir", nargs="?", default=None,
                    help="provenance bundle, or app dir containing Work/ "
                         "(default: aout/worklocal)")
    ap.add_argument("--app", default=None, metavar="PATH[=LABEL]",
                    help="open one app from its directory or provenance bundle; "
                         "generates worklocal from Work/ when needed")
    ap.add_argument("--app-root", action="append", default=[], metavar="DIR",
                    help="scan DIR for app workdirs (repeatable), e.g. "
                         "--app-root ../naiebaremetal/example")
    ap.add_argument("--sim-only", action="store_true",
                    help="this daemon reaches no board: drop the Board "
                         "selector from the UI and refuse connect / "
                         "attach / board-run / retarget / hw_server")
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
                    help="run log file to write + tail (default: repo-root "
                         "applog.$USER, or $SCHEDULE_DEBUG_APPLOG)")
    ap.add_argument("--open", action="store_true",
                    help="open the served URL in a browser after binding")
    ap.add_argument("--claude-bin", default="claude",
                    help="path to the claude CLI for the LLM tab (default: claude)")
    ap.add_argument("--claude-model",
                    default=os.environ.get("SCHEDULE_DEBUG_CLAUDE_MODEL",
                                           "claude-opus-5[1m]"),
                    help="model for the LLM tab (default: claude-opus-5[1m], or "
                         "$SCHEDULE_DEBUG_CLAUDE_MODEL; pass '' for the CLI default)")
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
    ap.add_argument("--skip-aiedbg-bootstrap", action="store_true",
                    help="do not auto-clone/install aiedbg when missing from PATH")
    ap.add_argument("--update-aiedbg", action="store_true",
                    help="git pull thirdparty/aiedbg and pip install --upgrade "
                         "before starting the server")
    args = ap.parse_args()
    if args.workdir and args.app:
        ap.error("pass either positional workdir/app directory or --app, not both")

    if not args.skip_aiedbg_bootstrap:
        import ensure_aiedbg as _ensure_aiedbg
        _abr = _ensure_aiedbg.ensure_aiedbg(_REPO_ROOT, update=args.update_aiedbg)
        if not _abr.get("ok"):
            print(f"warning: aiedbg bootstrap failed "
                  f"({_abr.get('action')}): {_abr.get('error', 'unknown')}",
                  file=sys.stderr)
        elif _abr.get("action") != "present":
            print(f"aiedbg bootstrap: {_abr['action']} -> {_abr.get('path')}")

    # Build the app registry first: the positional workdir is treated as an
    # explicit app so the historical single-app invocation keeps working, and
    # aout/ is auto-scanned so a bare `schedule_debug_server.py` finds the most
    # recent build on its own.
    explicit = [args.app] if args.app else []
    if args.workdir:
        explicit.insert(0, os.path.abspath(args.workdir))
    auto_roots = [] if (explicit and not args.app_root) else [
        os.path.join(_REPO_ROOT, "aout"), os.path.join(_REPO_ROOT, "example")]
    registry = AppRegistry(explicit=explicit, roots=args.app_root,
                           auto_roots=auto_roots)
    selected = registry.default()
    if explicit and selected is None:
        ap.error("explicit app has no schedule_view.json and no usable Work/ tree")

    workdir = os.path.abspath(selected.path if selected
                              else (args.workdir or _DEFAULT_WORKDIR))
    elf = os.path.abspath(args.elf) if args.elf else _resolve_default_elf(workdir)
    apppaltest = args.apppaltest or os.path.join(
        _REPO_ROOT, "script", "test", "apppaltest.py")
    applog = args.applog or _default_applog_path()
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
                               llm_password=llm_password,
                               sim_only=args.sim_only)
    Handler.state.registry = registry
    Handler.state.app = selected
    # Bind the server, applying the occupied-port policy: exit + list pid(s) if
    # this same user already holds the port, else fall forward to the next free
    # port when another user holds it.
    server, port = make_server(args.host, args.port, Handler)
    # When bound to all interfaces, show the LAN IP so the URL is reachable from
    # another machine (0.0.0.0 is not itself a connectable address).
    disp_host = _lan_ip() if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{disp_host}:{port}/"
    # Loopback callback URL for the debugui MCP (always reachable regardless of
    # what interface the server advertises).
    Handler.state.self_url = f"http://127.0.0.1:{port}"
    print(f"schedule_debug_server serving {workdir}")
    _apps = registry.list()
    print(f"  apps:       {len(_apps)} registered"
          f"{' (current: ' + selected.id + ')' if selected else ''}")
    for _a in _apps:
        _c = _a.caps()
        _flags = ",".join(k[4:] for k in ("has_sim", "has_hw") if _c[k]) or "view-only"
        print(f"    {'*' if selected and _a.id == selected.id else ' '} "
              f"{_a.id:22} {_flags:12} {_a.path}")
        if not _c["has_sim"] and _c.get("no_sim_reason"):
            print(f"      {'':22} {'':12} ↳ {_c['no_sim_reason']}")
        if _c.get("hw_stale"):
            print(f"      {'':22} {'':12} ! {_c['hw_stale']}")
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
    print(f"  source:     /source "
          f"({'pygments ' + _SRC_STYLE if _HAVE_PYGMENTS else 'plain (pygments not installed)'})")
    if args.host == "0.0.0.0" and not llm_password:
        print("  WARNING: bound to all interfaces with no password — /source lets "
              "anyone on the\n           network read files under the app, Work/ and "
              "repo roots. Use --password\n           or --host 127.0.0.1.")
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
        print(f"  sim:        {st.sim_script}  ({st.sim_kind})")
        print(f"  sim-dir:    {st.sim_example_dir or 'NOT SET'}")
        print(f"  sim-log:    {st.sim_log}"
              + (f"  (engine: {st.sim_engine_log})" if st.sim_engine_log else ""))
    else:
        # The Simulator option is still offered; say what to build, not that it
        # is missing a config file nothing has used since detection replaced
        # declaration.
        print(f"  sim:        unavailable for this app — {st.sim_reason}")
    if st.sim_only:
        print("  sim-only:   yes — no Board selector; connect / attach / "
              "board-run / retarget / hw_server all refuse")
        if not st.sim_script:
            print("  WARNING: --sim-only but no simulator was detected for "
                  "this app, so nothing can be run at all.\n"
                  f"           {st.sim_reason}", file=sys.stderr)
    else:
        print(f"  target:     {target or 'NONE'}")
    print(f"  startcol:   {startcol}{'' if args.startcol is not None else ' (from provenance JSON)'}   device={args.device}   aie={aie_version}{'' if args.aie_version is not None else ' (from provenance JSON)'}")
    if _hw_available() and not target and not st.sim_only:
        print("  WARNING: no aiedbg target set — live grid/cmd reads will fail with "
              "'timed out' / 'connection closed'.\n"
              "           Fix: export AIEDBG_TARGET=xsdb://<host>:3121  (or pass "
              "--target xsdb://<host>:3121)", file=sys.stderr)
    print("  Ctrl-C to stop.")
    if args.open:
        threading.Thread(target=_open_browser, args=(url,),
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
