#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""debug_ui_mcp - MCP server exposing the *static* schedule-view UI data.

This is a UI-only companion to the `aiegdb` MCP server (aiemcp.py). It is spawned
ONLY by schedule_debug_server.py for the browser "LLM tab", so its tools are NOT
visible to general Claude Code / CLI sessions (those use the repo-root .mcp.json
`aiegdb` server, which stays untouched).

Purpose: give the embedded LLM the same per-tile information the human sees in the
schedule view (host_schedule.html) without it having to read/parse the large
schedule_view.json blob by hand. Where `aiegdb`/aiemcp expose LIVE hardware state,
this server exposes the STATIC compiled schedule.

This is intended to grow: add more UI-facing tools here over time (flow lists,
supply/demand rollups, kernel-arg maps, etc.). Keep live-hardware tools in
aiemcp.py and static schedule/UI tools here.

Configuration (env, set by schedule_debug_server's temp .mcp.json):

    DEBUGUI_JSON_DIR    dir holding schedule_view.json (default: auto-detect)
    AIEMCP_JSON_DIR     fallback dir (shared with aiemcp.py)

Run standalone (stdio transport):  python3 src/tool/debug/debug_ui_mcp.py
Protocol smoke test:               mcp dev src/tool/debug/debug_ui_mcp.py
"""

import json
import os
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("debugui")


_VIEW_CACHE = {"path": None, "data": None}


def _view_candidates():
    """Candidate schedule_view.json paths, in priority order."""
    out = []
    for env in ("DEBUGUI_JSON_DIR", "AIEMCP_JSON_DIR"):
        d = os.environ.get(env)
        if d:
            out.append(os.path.join(d, "schedule_view.json"))
    out += [
        "./aout/worklocal/schedule_view.json",
        "./worklocal/schedule_view.json",
    ]
    return out


def _load_view():
    """Load schedule_view.json (cached by resolved path). Returns dict or None."""
    for p in _view_candidates():
        if os.path.isfile(p):
            ap = os.path.abspath(p)
            if _VIEW_CACHE["path"] == ap and _VIEW_CACHE["data"] is not None:
                return _VIEW_CACHE["data"]
            try:
                with open(ap) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            _VIEW_CACHE["path"] = ap
            _VIEW_CACHE["data"] = data
            return data
    return None


def _find_tile(view, col, row):
    for t in view.get("tiles", []) or []:
        loc = t.get("loc") or []
        if len(loc) == 2 and loc[0] == col and loc[1] == row:
            return t
    return None


def _fmt_supply_demand(tile):
    """Dedup per-flow supply/demand verdicts across the tile's channels."""
    seen, rows = set(), []
    for c in tile.get("dma_channels", []) or []:
        fb = c.get("flow_balance")
        if not fb or fb.get("flow_index") in seen:
            continue
        seen.add(fb.get("flow_index"))
        s, d = fb.get("supply_per_round"), fb.get("demand_per_round")
        if fb.get("balanced") is False:
            verd = "OVER-SUPPLY" if (s or 0) > (d or 0) else "UNDER-SUPPLY"
        elif fb.get("balanced") is True:
            verd = "balanced"
        else:
            verd = "unchecked"
        delta = ""
        if fb.get("balanced") is False and s is not None and d is not None:
            delta = "  (delta %sB)" % (s - d)
        rows.append(
            "  flow %s (%s): %s  supply=%sB/round demand=%sB/round%s  [%s]"
            % (fb.get("flow_index"), fb.get("pattern"), verd, s, d, delta,
               fb.get("note", "")))
    return rows


def _fmt_kernel_match(tile):
    km = (tile.get("high_level") or {}).get("kernel_match")
    if not km or not km.get("matches"):
        return []
    out = ["channel <-> kernel argument (by BD buffer address):"]
    for m in km["matches"]:
        adr = "/".join(m.get("addrs_hex") or []) or "-"
        bsy = "/".join(m.get("bcf_syms") or []) or "-"
        arg = ("arg%s" % m["arg"]) if m.get("arg") is not None else "-"
        out.append("  %s%s -> window %s %s  [bd %s = %s] via %s"
                   % (m.get("direction"), m.get("channel"), m.get("window"),
                      arg, adr, bsy, m.get("method")))
    return out


def _fmt_lines(records, prefix="L"):
    """Render a list of {line, code} rows (line may be None)."""
    out = []
    for rec in records or []:
        ln = rec.get("line")
        code = rec.get("code", "")
        if ln is None:
            out.append("        %s" % code)
        else:
            out.append("  %s%-5s %s" % (prefix, ln, code))
    return out


def _section_hi(tile):
    hl = tile.get("high_level") or {}
    out = []
    if hl.get("role"):
        out.append("role: %s" % hl["role"])
    if hl.get("kernel"):
        out.append("kernel: %s" % hl["kernel"])
    sd = _fmt_supply_demand(tile)
    if sd:
        out.append("supply / demand:")
        out.extend(sd)
    if hl.get("summary"):
        out.append("transfers:")
        out.extend("  %s" % s for s in hl["summary"])
    if hl.get("contracts"):
        out.append("contracts:")
        out.extend("  %s" % c for c in hl["contracts"])
    km = _fmt_kernel_match(tile)
    if km:
        out.extend(km)
    return out


def _section_mid(tile):
    mid = tile.get("middle_ir")
    if isinstance(mid, str):
        return [mid]
    return _fmt_lines(mid) or ["(no dfschedule IR slice)"]


def _section_lo(tile):
    lo = tile.get("low_level") or {}
    rows = _fmt_lines(lo.get("code_lines"))
    if not rows:
        return ["(no attributed host.cc lines)"]
    hdr = "host.cc lines %s-%s:" % (lo.get("line_start"), lo.get("line_end"))
    return [hdr] + rows


@mcp.tool()
def tile_info(col: int, row: int, section: str = "all") -> str:
    """Return the schedule-view UI information for one AIE tile.

    This is the same per-tile content the human sees in host_schedule.html when
    they click a tile: the High level summary (role, kernel, transfers,
    supply/demand balance flags, channel<->kernel-arg map), the Middle
    (dfschedule IR) slice, and the Low level (attributed host.cc source lines).

    Args:
      col:     tile column (logical, as shown in the grid)
      row:     tile row (0 = shim, >=3 = core)
      section: which part to return - "hi" | "mid" | "lo" | "all" (default all)

    Returns a readable text digest. Use section="hi" for a quick summary or
    "all" for everything. Coordinates are the logical grid coords shown in the UI
    (not phys_col); this reads the static compiled schedule, not live hardware
    (use the aiegdb tools for live DMA/core/register state).
    """
    view = _load_view()
    if view is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))
    tile = _find_tile(view, col, row)
    if tile is None:
        locs = ", ".join("(%s,%s)" % (t["loc"][0], t["loc"][1])
                         for t in view.get("tiles", []) if t.get("loc"))
        return ("error: tile (%s,%s) not in schedule_view.json. Available: %s"
                % (col, row, locs))

    sec = (section or "all").strip().lower()
    valid = {"hi", "mid", "lo", "all"}
    if sec not in valid:
        return "error: section must be one of hi|mid|lo|all (got %r)" % section

    lines = ["=== Tile (%s,%s)  type=%s ==="
             % (col, row, tile.get("type"))]
    if sec in ("hi", "all"):
        lines.append("")
        lines.append("--- High level ---")
        lines.extend(_section_hi(tile))
    if sec in ("mid", "all"):
        lines.append("")
        lines.append("--- Middle (dfschedule IR) ---")
        lines.extend(_section_mid(tile))
    if sec in ("lo", "all"):
        lines.append("")
        lines.append("--- Low level (host.cc) ---")
        lines.extend(_section_lo(tile))
    return "\n".join(lines)


def _read_live_status() -> dict:
    """Read backend_status.json from the debug server's workdir (if available)."""
    json_dir = os.environ.get("DEBUGUI_JSON_DIR", "").strip()
    if not json_dir:
        return {}
    try:
        with open(os.path.join(json_dir, "backend_status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


@mcp.tool()
def get_backend_status() -> dict:
    """Return the currently active debug backend and its connection state.

    Call this first in any debug session to understand what backend is active
    and whether live hardware reads are possible.

    Returns a dict with:
      backend      "simulator" | "hardware" | "unknown"
      ipc_ready    (simulator only) True when the IPC debug socket is open
      dbg_socket   (simulator only) filesystem path to the *.sock.dbg socket
      target       (hardware only) aiedbg target string, e.g. xsdb://host:3121
      device       aiedbg device string, e.g. "pal" or "npi"
      startcol     physical column offset for this partition
      aie_version  register layout version string, e.g. "2ps" or "5"
      note         human-readable status summary

    When backend is "simulator" and ipc_ready is True, the aiegdb MCP server
    (aie_exec / aie_scope tools) is configured to issue READ32 requests directly
    over the IPC debug socket — no JTAG or xsdb needed. When ipc_ready is False,
    the simulator is not yet running; start it from the UI Run button first.

    When backend is "hardware", use aie_exec / aie_scope to interact with the
    physical board via aiedbg. The target string identifies the xsdb endpoint.
    """
    live = _read_live_status()
    if not live:
        live = {}

    backend = live.get("backend") or os.environ.get("AIEMCP_BACKEND", "unknown").strip().lower()
    dbg_socket = live.get("dbg_socket") or os.environ.get("AEG_PS_IPC_DBG_SOCKET", "").strip() or None
    ipc_ready = bool(live.get("ipc_ready")) if "ipc_ready" in live else (backend == "simulator" and bool(dbg_socket))
    target = live.get("target") or os.environ.get("AIEDBG_TARGET", "").strip() or None
    device = live.get("device") or os.environ.get("AIEMCP_DEVICE", "").strip() or None
    startcol = live.get("startcol") or os.environ.get("AIEMCP_STARTCOL", "").strip() or None
    aie_version = live.get("aie_version") or os.environ.get("AIEMCP_AIE_VERSION", "").strip() or None
    sim_log = live.get("sim_log") or ""
    sim_applog = live.get("sim_applog") or ""

    if backend == "simulator":
        if ipc_ready:
            note = ("Simulator is running and IPC debug socket is active. "
                    "Live register reads are available via aie_exec/aie_scope.")
        else:
            note = ("Simulator backend selected but IPC debug socket not yet ready. "
                    "Start the simulator from the Run button in the UI, then retry.")
    elif backend == "hardware":
        note = ("Hardware board backend. Live reads go through aiedbg/xsdb. "
                "Use aie_exec/aie_scope to read DMA, core, and event registers.")
    else:
        note = "Backend unknown — debug server may not be running."

    return {
        "backend": backend,
        "ipc_ready": ipc_ready,
        "dbg_socket": dbg_socket,
        "target": target,
        "device": device,
        "startcol": startcol,
        "aie_version": aie_version,
        "sim_log": sim_log,
        "sim_applog": sim_applog,
        "note": note,
    }


@mcp.tool()
def tile_list() -> str:
    """List every tile in the current schedule view with its type and role.

    Use this first to discover which tiles exist before calling tile_info.
    Returns one line per tile: (col,row) type - role.
    """
    view = _load_view()
    if view is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))
    out = []
    for t in view.get("tiles", []) or []:
        loc = t.get("loc") or [None, None]
        role = (t.get("high_level") or {}).get("role", "")
        out.append("(%s,%s) %-5s %s" % (loc[0], loc[1], t.get("type"), role))
    return "\n".join(out) if out else "(no tiles in schedule_view.json)"



def _build_search_index(view):
    """Return list of hit dicts: {kind, col, row, fi, label, description}."""
    hits = []

    def add(kind, col, row, fi, label, description):
        hits.append(dict(kind=kind, col=col, row=row, fi=fi,
                         label=label, label_lc=label.lower(), description=description))

    for t in view.get("tiles", []) or []:
        if not t or not t.get("loc"):
            continue
        tc, tr = t["loc"]

        hl = t.get("high_level") or {}
        if hl.get("kernel"):
            add("kernel", tc, tr, None, hl["kernel"],
                "kernel on (%s,%s)" % (tc, tr))

        for ch in t.get("dma_channels", []) or []:
            fi = ch.get("flow_index")
            if ch.get("contract"):
                add("contract", tc, tr, fi, ch["contract"],
                    "%s ch%s on (%s,%s)" % (ch.get("direction", "?"), ch.get("channel"), tc, tr))
            for bd in ch.get("bd_chain", []) or []:
                if bd.get("len") is not None:
                    add("bd_len", tc, tr, fi, str(bd["len"]),
                        "BD%s len=%s %s ch%s (%s,%s)" % (bd.get("bd_id"), bd["len"],
                                                          ch.get("direction"), ch.get("channel"), tc, tr))
                for sym in bd.get("bcf_syms", []) or []:
                    if sym:
                        add("buffer", tc, tr, fi, sym,
                            "BD%s buffer on (%s,%s) %s ch%s" % (bd.get("bd_id"), tc, tr,
                                                                  ch.get("direction"), ch.get("channel")))

    kern = view.get("kernel") or {}
    if kern.get("function"):
        for w in kern.get("windows", []) or []:
            if w.get("name"):
                add("window", None, None, None, w["name"],
                    "kernel window (%s)" % kern["function"])

    for p in view.get("comm_paths", []) or []:
        if not p:
            continue
        fi = p.get("flow_index")
        dma_tiles = p.get("dma_tiles") or []
        rep = dma_tiles[0] if dma_tiles else None
        rc, rr = (rep[0], rep[1]) if rep else (p.get("prod_col"), p.get("prod_row"))

        net_id = p.get("id") or p.get("net_id")
        if net_id:
            add("net", rc, rr, fi, net_id, "net (%s) f%s" % (net_id, fi))
        if fi is not None:
            add("flow", rc, rr, fi, "f%s" % fi, "flow index %s" % fi)
        if p.get("config_ref"):
            add("gmio", rc, rr, fi, p["config_ref"], "config ref for f%s" % fi)
        for h in p.get("hops", []) or []:
            if h.get("port_name"):
                add("port", rc, rr, fi, h["port_name"], "hop port in f%s" % fi)

    for fs in view.get("flow_summary", []) or []:
        if not fs:
            continue
        fi = fs.get("flow_index")
        for entry in fs.get("entries", []) or []:
            tc2, tr2 = entry.get("tile_col"), entry.get("tile_row")
            if entry.get("kernel_port"):
                add("port", tc2, tr2, fi, entry["kernel_port"],
                    "supply/demand port f%s" % fi)

    return hits


_SEARCH_CACHE = {"path": None, "index": None}


def _get_search_index():
    view = _load_view()
    if view is None:
        return None, None
    ap = _VIEW_CACHE["path"]
    if _SEARCH_CACHE["path"] == ap and _SEARCH_CACHE["index"] is not None:
        return view, _SEARCH_CACHE["index"]
    idx = _build_search_index(view)
    _SEARCH_CACHE["path"] = ap
    _SEARCH_CACHE["index"] = idx
    return view, idx


@mcp.tool()
def symbol_search(query: str, kinds: str = "") -> str:
    """Search the compiled AIE design for any symbol matching a substring.

    This mirrors the Search bar in host_schedule.html (the "kernel, window, net,
    GMIO, port, len…" input at the bottom of the device-map panel), returning the
    same hits the browser would highlight — but as structured text instead of
    yellow SVG halos.

    Searchable fields (kinds):
      kernel   — kernel function names (e.g. "matmul", "dskernel_receiver")
      window   — kernel window / buffer argument names (e.g. "in_0", "out")
      buffer   — BCF buffer symbols attributed to BD descriptors
      contract — DMA channel contract strings (e.g. "S2MM ch0: ping-pong receive, 1024B")
      bd_len   — BD transfer lengths in bytes (e.g. "1024", "256")
      net      — net / comm-path IDs (e.g. "push_0", "net7")
      flow     — flow index strings (e.g. "f0", "f7")
      gmio     — GMIO config-ref names
      port     — kernel-port / graph-port strings from flow_summary

    Args:
      query: case-insensitive substring to search (e.g. "receiver", "1024", "push_0")
      kinds: comma-separated list of kinds to restrict results (default: all kinds).
             Example: "kernel,contract"  or  "bd_len,buffer"

    Returns one line per match: kind  (col,row)  f<fi>  label  —  description
    Tile coords may be "(-,-)" for design-wide entries (kernel windows, etc.).
    Use tile_info(col, row) to drill into any returned tile.
    """
    view, idx = _get_search_index()
    if idx is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))

    q = (query or "").strip().lower()
    if not q:
        return "error: query must not be empty"

    kind_filter = set()
    if kinds:
        kind_filter = {k.strip().lower() for k in kinds.split(",") if k.strip()}

    hits = [h for h in idx
            if q in h["label_lc"]
            and (not kind_filter or h["kind"] in kind_filter)]

    if not hits:
        return "no matches for %r%s" % (query, (" (kinds: %s)" % kinds) if kinds else "")

    lines = ["symbol_search %r  →  %d match%s" % (query, len(hits), "es" if len(hits) != 1 else "")]
    seen = set()
    for h in hits:
        dedup_key = (h["kind"], h.get("col"), h.get("row"), h.get("fi"), h["label"])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        col_s = str(h["col"]) if h["col"] is not None else "-"
        row_s = str(h["row"]) if h["row"] is not None else "-"
        fi_s = ("f%s" % h["fi"]) if h["fi"] is not None else "—"
        lines.append("  %-10s (%s,%s)  %-5s  %s  —  %s"
                     % (h["kind"], col_s, row_s, fi_s, h["label"], h["description"]))
    return "\n".join(lines)


@mcp.tool()
def get_design_overview() -> str:
    """Return a high-level overview of the compiled AIE design: grid geometry,
    all communication flows (producer→consumer), and the supply/demand balance
    verdict for every flow.

    Use this as the first call in any session to orient yourself before drilling
    into tile_info or get_flow_detail. It is equivalent to reading the full
    device map in the browser.

    Returns structured text with three sections:
      Grid      — tile geometry (columns, rows, shim/core row ranges, startcol)
      Flows     — one line per comm_path: flow index, direction, producer GMIO,
                  consumer tile(s), and any balance warning
      Supply/demand — per-flow balanced/OVER-SUPPLY/UNDER-SUPPLY verdict
    """
    view = _load_view()
    if view is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))

    lines_out = []

    grid = view.get("grid") or {}
    lines_out.append("=== Grid geometry ===")
    lines_out.append("  cols:     %s" % grid.get("cols"))
    lines_out.append("  rows:     %s" % grid.get("rows"))
    lines_out.append("  startcol: %s" % grid.get("startcol"))
    shim = grid.get("shim_rows") or []
    core = grid.get("core_rows") or []
    if shim:
        lines_out.append("  shim rows: %s" % shim)
    if core:
        lines_out.append("  core rows: %s" % core)

    comm_paths = view.get("comm_paths") or []
    lines_out.append("")
    lines_out.append("=== Communication flows (%d) ===" % len(comm_paths))
    for p in comm_paths:
        if not p:
            continue
        fi = p.get("flow_index")
        direction = p.get("direction", "?")
        stages = p.get("stages") or []
        prod = next((s for s in stages if s.get("role") == "producer"), None)
        cons = next((s for s in stages if s.get("role") == "consumer"), None)
        prod_name = (prod or {}).get("gmio_name") or (prod or {}).get("config_ref") or "?"
        cons_tile = ""
        if cons:
            ct = cons.get("tile") or {}
            cons_tile = " tile(%s,%s)" % (ct.get("col", "?"), ct.get("row", "?"))
        dma_tiles = p.get("dma_tiles") or []
        if not cons_tile and dma_tiles:
            cons_tile = " tiles(%s)" % ",".join("(%s,%s)" % (t[0], t[1]) for t in dma_tiles[:3])
        lines_out.append("  f%-3s %-6s  %s → %s%s"
                         % (fi, direction, prod_name, direction, cons_tile))

    sd_list = view.get("supply_demand") or []
    if sd_list:
        lines_out.append("")
        lines_out.append("=== Supply/demand balance ===")
        for sd in sd_list:
            if not sd:
                continue
            balanced = sd.get("balanced")
            if balanced is False:
                s, d = sd.get("supply_per_round"), sd.get("demand_per_round")
                verdict = "OVER-SUPPLY" if (s or 0) > (d or 0) else "UNDER-SUPPLY"
                lines_out.append("  f%s  %s  supply=%sB demand=%sB  [%s]"
                                  % (sd.get("flow_index"), verdict, s, d, sd.get("note", "")))
            else:
                lines_out.append("  f%s  balanced  pattern=%s"
                                  % (sd.get("flow_index"), sd.get("pattern")))

    return "\n".join(lines_out)


@mcp.tool()
def get_flow_detail(flow_index: int) -> str:
    """Return detailed routing and staging information for one communication flow.

    This exposes what the browser shows in the net detail panel when you click
    a flow arrow in the Device Map: producer stage, consumer stage, all hop
    tiles, stream-switch connections, and BD contract strings.

    Args:
      flow_index: the f<N> index from tile_info, get_design_overview, or
                  symbol_search (e.g. 0 for f0)

    Returns a structured text block with producer, consumer, channel hops,
    routing connections, and supply/demand detail for this specific flow.
    """
    view = _load_view()
    if view is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))

    comm_paths = view.get("comm_paths") or []
    path = next((p for p in comm_paths
                 if p and p.get("flow_index") == flow_index), None)
    if path is None:
        avail = [str(p["flow_index"]) for p in comm_paths if p and "flow_index" in p]
        return ("error: flow_index %d not found. Available: %s"
                % (flow_index, ", ".join(avail) or "none"))

    out = ["=== Flow f%d  direction=%s ===" % (flow_index, path.get("direction", "?"))]

    stages = path.get("stages") or []
    for stage in stages:
        role = stage.get("role", "?")
        tile = stage.get("tile") or {}
        gmio = stage.get("gmio_name") or stage.get("config_ref") or ""
        contract = stage.get("contract") or ""
        out.append("")
        out.append("  [%s]  tile(%s,%s)%s%s"
                   % (role, tile.get("col", "?"), tile.get("row", "?"),
                      ("  gmio=" + gmio) if gmio else "",
                      ("  contract=" + contract) if contract else ""))
        hops = stage.get("hops") or []
        for h in hops:
            out.append("    hop: %s → %s%s"
                       % (h.get("from", "?"), h.get("to", "?"),
                          ("  [%s/%s]" % (h.get("hop_type"), h.get("shmem_kind"))
                           if h.get("hop_type") else "")))

    routing = path.get("routing_connections") or []
    if routing:
        out.append("")
        out.append("  [stream-switch connections]")
        for c in routing:
            kind = c.get("kind", "?")
            t = c.get("tile") or {}
            if kind == "circuit_connect":
                sl = c.get("slave") or {}
                ms = c.get("master") or {}
                out.append("    (%s,%s) %s/%s → %s/%s"
                           % (t.get("col"), t.get("row"),
                              sl.get("dir"), sl.get("idx"),
                              ms.get("dir"), ms.get("idx")))
            else:
                out.append("    (%s,%s) %s" % (t.get("col"), t.get("row"), kind))

    sd_list = view.get("supply_demand") or []
    sd = next((s for s in sd_list if s and s.get("flow_index") == flow_index), None)
    if sd:
        out.append("")
        out.append("  [supply/demand]")
        out.append("    pattern=%s  supply=%sB/round  demand=%sB/round  balanced=%s"
                   % (sd.get("pattern"), sd.get("supply_per_round"),
                      sd.get("demand_per_round"), sd.get("balanced")))
        if sd.get("note"):
            out.append("    note: %s" % sd["note"])
        for p in (sd.get("participants") or []):
            loc = p.get("loc") or []
            out.append("    (%s,%s) %s ch%s  bd_len=%s  fires=%s%s"
                       % (loc[0] if len(loc) > 0 else "?",
                          loc[1] if len(loc) > 1 else "?",
                          p.get("io_direction"), p.get("channel"),
                          p.get("bd_len"), p.get("fires"),
                          "  [shim]" if p.get("is_shim") else ""))

    return "\n".join(out)


@mcp.tool()
def get_sim_log(lines: int = 50) -> str:
    """Return the last N lines of the simulator application log (ipc_app.log).

    This is the stdout/stderr from the PS application running inside the AIE
    simulator, equivalent to the simulator applog the browser tails in the Run
    console. Use it to check for application output, errors, or OOB test results
    when running in simulator mode.

    For the board run log (hardware mode), use get_applog instead.

    Args:
      lines: number of lines to return from the end of the log (default: 50)
    """
    live = _read_live_status()
    sim_applog = live.get("sim_applog") or os.environ.get("DEBUGUI_SIM_APPLOG", "").strip()
    if not sim_applog:
        return ("error: simulator applog path not set — start the debug server "
                "with a simulator-capable config and activate the simulator")
    if not os.path.isfile(sim_applog):
        return "simulator applog not found at %s — start the simulator first" % sim_applog
    try:
        with open(sim_applog, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        tail = text.splitlines()
        if len(tail) > lines:
            tail = tail[-lines:]
        return "\n".join(tail) if tail else "(simulator applog is empty)"
    except OSError as e:
        return "error reading simulator applog: %s" % e


@mcp.tool()
def get_applog(lines: int = 50) -> str:
    """Return the last N lines of the application run log.

    This is the same log the browser tails in the Run console panel. It contains
    stdout/stderr from the application binary (simulator or hardware). Use it to
    see what the running application printed, check for runtime errors, or confirm
    that the application completed normally.

    Args:
      lines: number of lines to return from the end of the log (default: 50)

    Returns the log tail as a string, or an error message if no log exists yet.
    Logs are only present after a run has been started from the UI Run button.
    """
    live = _read_live_status()
    backend = live.get("backend", "").strip().lower()
    if backend == "simulator":
        applog = live.get("sim_applog") or os.environ.get("DEBUGUI_SIM_APPLOG", "").strip()
        source_note = "(simulator ipc_app.log)"
    else:
        applog = live.get("applog") or os.environ.get("DEBUGUI_APPLOG", "").strip()
        source_note = "(hardware applog)"
    if not applog:
        return "error: applog path not set (start the debug server to configure it)"
    if not os.path.isfile(applog):
        return "applog not found at %s — start a run first" % applog
    try:
        with open(applog, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        tail = text.splitlines()
        if len(tail) > lines:
            tail = tail[-lines:]
        result = "\n".join(tail) if tail else "(applog is empty)"
        return "[source: %s path=%s]\n%s" % (source_note, applog, result)
    except OSError as e:
        return "error reading applog: %s" % e


@mcp.tool()
def get_ipc_log(lines: int = 100, side: str = "both") -> str:
    """Return recent IPC transaction log entries from the simulator run.

    The IPC logs record every request/response between the PS application
    (ipc_app) and the AIE simulator over the Unix socket. Each row is a CSV
    with columns: seq, ts_ns, side, cmd, arg1, arg2, status, value, note.

    ts_ns is a CLOCK_MONOTONIC nanosecond timestamp — subtract the first row's
    ts_ns to get elapsed time. The delta between a client row and its matching
    server row shows the round-trip latency for that transaction.

    Commands: WRITE32 / READ32 (register r/w), NPI_WRITE32 / NPI_READ32 (NPI
    register r/w), WRITE_GM / READ_GM / ALLOC_GM / FREE_GM (global memory),
    GRAPH_INIT (graph load + init sequence), START_PLIO, EXIT.

    Use this to:
    - Find the last completed transaction before a hang
    - Measure per-transaction latency (ts_ns delta client→server)
    - Identify which register or GM address a stall is waiting on
    - Confirm GRAPH_INIT completed before DMA transactions start

    Args:
      lines: number of lines to return from the end (default: 100)
      side:  "client", "server", or "both" (default: "both")
    """
    sim_dir = os.environ.get("DEBUGUI_SIM_APPLOG", "")
    if sim_dir:
        sim_dir = os.path.dirname(sim_dir)
    if not sim_dir:
        return "error: simulator not configured — start the debug server with a simulator config"

    client_log = os.path.join(sim_dir, "ipc_client.log")
    server_log = os.path.join(sim_dir, "ipc_server.log")

    def _tail(path, n):
        if not os.path.isfile(path):
            return None, "(not found — run the simulator first)"
        try:
            with open(path, "rb") as f:
                raw = f.read()
            rows = raw.decode("utf-8", errors="replace").splitlines()
            if len(rows) > n:
                rows = rows[:1] + rows[-(n - 1):]
            return rows, None
        except OSError as e:
            return None, "error: %s" % e

    parts = []
    if side in ("client", "both"):
        rows, err = _tail(client_log, lines)
        if err:
            parts.append("--- client IPC log ---\n" + err)
        else:
            parts.append("--- client IPC log (%s) ---\n%s" % (client_log, "\n".join(rows)))
    if side in ("server", "both"):
        rows, err = _tail(server_log, lines)
        if err:
            parts.append("--- server IPC log ---\n" + err)
        else:
            parts.append("--- server IPC log (%s) ---\n%s" % (server_log, "\n".join(rows)))
    return "\n\n".join(parts) if parts else "(no logs found)"


if __name__ == "__main__":
    mcp.run()
