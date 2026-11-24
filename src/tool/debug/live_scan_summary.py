#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Summarize /grid scan payloads for the embedded LLM and debugui MCP."""

from __future__ import annotations

import datetime


def _iso_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _count_states(cells, key="state"):
    counts = {}
    for c in (cells or {}).values():
        if not c:
            continue
        st = c.get(key) or "unknown"
        counts[st] = counts.get(st, 0) + 1
    return counts


def _fmt_counts(counts):
    if not counts:
        return "no tiles"
    return ", ".join("%s=%d" % (k, counts[k])
                       for k in sorted(counts.keys()))


def _summarize_dma(cells):
    counts = _count_states(cells)
    highlights = []
    for key, cell in sorted((cells or {}).items()):
        if not cell or cell.get("state") not in ("stalled", "error", "running"):
            continue
        col, row = key.split(",")
        parts = []
        for ch, info in sorted((cell.get("channels") or {}).items()):
            st = info.get("state", "?")
            if st in ("idle", "completed"):
                continue
            extra = []
            if info.get("stalls"):
                extra.append("stall=" + "+".join(info["stalls"]))
            if info.get("errors"):
                extra.append("err=" + "+".join(info["errors"]))
            if info.get("cur_bd") is not None:
                extra.append("bd=%s" % info["cur_bd"])
            tail = (" " + " ".join(extra)) if extra else ""
            parts.append("%s:%s%s" % (ch, st, tail))
        if parts:
            highlights.append("(%s,%s) %s" % (col, row, "; ".join(parts)))
    lines = ["DMA scan: tile states {%s}" % _fmt_counts(counts)]
    if highlights:
        lines.append("active/problem channels:")
        lines.extend("  " + h for h in highlights[:24])
        if len(highlights) > 24:
            lines.append("  … +%d more tiles" % (len(highlights) - 24))
    else:
        lines.append("no stalled/error/running channels reported")
    return {"counts": counts, "highlights": highlights, "text": "\n".join(lines)}


def _summarize_cores(cells):
    counts = _count_states(cells)
    running = []
    for key, cell in sorted((cells or {}).items()):
        if cell and cell.get("state") == "running":
            running.append(key.replace(",", ","))
    lines = ["Cores scan: {%s}" % _fmt_counts(counts)]
    if running:
        lines.append("running: " + ", ".join("(%s)" % k for k in running[:20]))
    return {"counts": counts, "highlights": running, "text": "\n".join(lines)}


def _summarize_events(cells):
    counts = _count_states(cells)
    active = [k for k, c in sorted((cells or {}).items())
              if c and c.get("state") == "running"]
    lines = ["Events scan: {%s}" % _fmt_counts(counts)]
    if active:
        lines.append("event bits set: " + ", ".join("(%s)" % k for k in active[:20]))
    return {"counts": counts, "highlights": active, "text": "\n".join(lines)}


def _summarize_switch(res):
    cells = res.get("cells") or {}
    counts = _count_states(cells)
    n_bad = int(res.get("mismatch_tiles") or 0)
    bad = [k for k, c in sorted(cells.items())
           if c and c.get("state") == "mismatch"]
    disc = res.get("dynamic") or {}
    n_flows = disc.get("n_flows")
    lines = ["Switch scan: tile states {%s}" % _fmt_counts(counts)]
    if n_bad:
        lines.append("%d tile(s) disagree with the static routing map: %s"
                     % (n_bad, ", ".join("(%s)" % k for k in bad[:16])))
    else:
        lines.append("every scanned tile matches the static routing map")
    if n_flows is not None:
        lines.append("dynamic routing: %s flow(s) reconstructed from registers"
                     % n_flows)
    if disc.get("error"):
        lines.append("dynamic routing failed: %s" % disc["error"])
    return {"counts": counts, "mismatch_tiles": n_bad,
            "dynamic_flows": n_flows, "highlights": bad,
            "text": "\n".join(lines)}


def summarize_live_scan(what, res):
    """Build a compact summary dict from a /grid response."""
    what = (res.get("what") or what or "dma").strip().lower()
    out = {"what": what, "ts": _iso_now(), "error": res.get("error")}
    if res.get("error"):
        out["text"] = "%s scan failed: %s" % (what, res["error"])
        return out
    if what == "switch":
        body = _summarize_switch(res)
    elif what == "cores":
        body = _summarize_cores(res.get("cells"))
    elif what == "events":
        body = _summarize_events(res.get("cells"))
    else:
        body = _summarize_dma(res.get("cells"))
    out.update(body)
    out["one_line"] = out["text"].split("\n", 1)[0]
    return out


def format_for_llm(summary, include_detail=True):
    """Single block suitable for [context] injection or get_live_scan."""
    if not summary:
        return "(no scan recorded yet)"
    head = "Live scan (%s @ %s)" % (summary.get("what", "?"),
                                    summary.get("ts", "?"))
    if summary.get("error"):
        return "%s — failed: %s" % (head, summary["error"])
    text = summary.get("text") or summary.get("one_line") or ""
    if not include_detail:
        return "%s — %s" % (head, summary.get("one_line") or text.split("\n")[0])
    return head + "\n" + text
