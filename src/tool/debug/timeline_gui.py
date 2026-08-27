#!/usr/bin/env python3
"""Matplotlib GUI for the host<->AIE merged timeline (timeline.json).

Reads the JSON produced by host_aie_timeline.py (schema: meta / anchors /
fit_per_tile / lanes[{name, events:[{event,start_us,end_us,detail}]}]) and draws
a Gantt-style diagram of HORIZONTAL timelines on ONE microsecond axis:

  * one horizontal lane per model["lanes"] entry (host first, then tiles),
  * the "host" lane is a horizontal timeline of BARS between consecutive phase
    markers -- BLUE where the host is running (setup / DMA-in / DMA-out / verify)
    and GREY HATCHED where the host is idle-WAITING on the AIE (the run->wait_done
    gap that brackets XAie_CoreWaitForDone),
  * every tile event is a horizontal bar start_us..end_us, COLOURED by event
    type (ACTIVE=AIE running / LOCK_STALL / STREAM_STALL / MEMORY_STALL / other),
    so the tile's green ACTIVE bar lines up UNDER the host's waiting bar,
  * a mem-DMA lane ("tile C,R dma <dir> <ch>") draws the lock stall
    (DMA_*_STALLED_LOCK_MEM, orange) as a full-height background band and marks
    EVERY stream stall event (MM2S back-pressure / S2MM starvation, red) as its
    own full-height per-event vertical tick on top, so all occurrences are
    plotted instead of merging into a few blobs,
  * light dashed vertical guides mark the host phase boundaries across all lanes,
  * pan/zoom via matplotlib's built-in toolbar, hover for per-event detail.

Headless-safe: with no $DISPLAY (or with --save) it uses the Agg backend and
writes a PNG instead of opening a window.

Usage:
    timeline_gui.py <timeline.json>                 # window (or PNG if headless)
    timeline_gui.py <timeline.json> --save out.png  # force PNG
    timeline_gui.py --self-test                     # synthetic render, no file
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

# --------------------------------------------------------------------------
# Event-type -> colour. Slots 0..3 follow __Runtime_core_trace_setup.
# --------------------------------------------------------------------------
EVENT_COLORS = {
    "ACTIVE": "#2ca02c",        # green
    "LOCK_STALL": "#ff7f0e",    # orange
    "STREAM_STALL": "#d62728",  # red
    "MEMORY_STALL": "#9467bd",  # purple
    # Stream-switch port state (slots 4..6), rendered on the per-tile port lane.
    "PORT_RUNNING_0": "#17becf",  # teal
    "PORT_STALLED_0": "#e377c2",  # pink
    "PORT_IDLE_0": "#bcbcbc",     # light grey
    # Mem-module DMA trace (pkt id 2), rendered on the per-tile "mem dma" lane.
    # s_mem_trace_slot_name: DMA_START/FINISH/STALL_LOCK, STREAM_STALL, MEM_BP,
    # LOCK_GRP/ACQ/REL. STREAM_STALL reuses the red above (a stall is a stall).
    "DMA_START": "#00e676",       # spring green (task started; distinct from ACTIVE)
    "DMA_FINISH": "#1f77b4",      # blue   (BD finished)
    "DMA_STALL_LOCK": "#ff7f0e",  # orange (blocked acquiring a lock)
    "MEM_BP": "#9467bd",          # purple (memory back-pressure)
    "LOCK_GRP": "#8c564b",        # brown  (lock group event)
    "LOCK_ACQ": "#bcbd22",        # olive  (lock acquire)
    "LOCK_REL": "#7f7f7f",        # grey   (lock release)
}
# Pulse / edge events: single-cycle markers (BD start/finish, lock group /
# acquire / release). A span rectangle cannot show a zero-width pulse, so these
# are overlaid as VERTICAL LINE markers coloured by EVENT_COLORS -- visible even
# when the pulse rides in a combined label alongside a held stall level.
PULSE_EVENTS = ("DMA_START", "DMA_FINISH", "LOCK_GRP", "LOCK_ACQ", "LOCK_REL")
# Some pulses also get a point marker at the top of their line so a LONE
# occurrence pops out of the dense pulse cluster. The DMA task-start
# (DMA_S2MM_0_START_TASK) fires once per run. The per-BD FINISH needs a marker
# for a different reason: it often coincides with a stream back-pressure event,
# whose full-height opaque tick would otherwise paint over the finish line -- the
# glyph is drawn above the ticks (higher zorder) so the finish still shows.
# Start uses a down-triangle, finish an up-triangle (mirrored, and coloured
# differently) so the two edges of a BD read at a glance.
PULSE_MARKERS = {"DMA_START": "v", "DMA_FINISH": "^"}
# Mem DMA stall slots take colour priority over DMA_START in a combined label,
# mirroring STALL_ORDER for the core lane (a BD can be "running" yet blocked).
MEM_STALL_ORDER = ("DMA_STALL_LOCK", "STREAM_STALL", "MEM_BP")
# Stall slots take colour priority over ACTIVE in a combined label: a core is
# reported ACTIVE even while blocked, so the stall should be what you see.
STALL_ORDER = ("LOCK_STALL", "STREAM_STALL", "MEMORY_STALL")
# Port-lane rasterisation priority (high->low colours winning a pixel bin by
# PRESENCE, not duration). RUNNING is the actual-data-movement signal but fires
# in very short, frequent bursts that pure duration dominance drowns under the
# long IDLE/STALLED background; STALLED likewise should surface over IDLE. So a
# bin showing any RUNNING renders RUNNING, else any STALLED renders STALLED,
# else IDLE.
PORT_RASTER_PRIORITY = (EVENT_COLORS["PORT_RUNNING_0"], EVENT_COLORS["PORT_STALLED_0"])
# Core-lane rasterisation priority: genuine "core running" is an ACTIVE span with
# NO stall bit (LOCK/STREAM/MEMORY) -- event_color already returns the ACTIVE
# colour only for those, and a stall colour whenever a stall bit is present. But
# those pure-ACTIVE bursts are brief (a few % of the run) and vanish under the
# long stall background in duration-dominance rasterisation. Give the ACTIVE
# colour presence-priority so any bin containing a real running burst renders
# green; stall-only bins stay their stall colour.
CORE_RASTER_PRIORITY = (EVENT_COLORS["ACTIVE"],)
# Mem-lane rendering: a mem-DMA lane stacks its concurrent stall states as
# per-column horizontal slices (see MEM_LANE_STACK below). ALL three stall states
# are HELD-duration bands with real start->end spans: lock stall, memory
# back-pressure, and stream stall (DMA_*_STREAM_BACKPRESSURE_MEM on MM2S /
# DMA_*_STREAM_STARVATION_MEM on S2MM, both canonicalising to STREAM_STALL). A
# long stream-starvation region (tens of us) therefore fills its whole span as a
# red band rather than collapsing to a single tick at its start. Stream ALSO
# keeps a per-event vertical tick overlay -- one line per event at its true start,
# confined to the stream slice -- because binning would merge its many
# nanosecond-scale bursts into a blob; the tick plots every occurrence. Band and
# ticks share the stream slice and colour, so they do not conflict. The axis is
# inverted (host on top). When several states are concurrent on the same lane --
# e.g. a BD both lock-stalled AND memory back-pressured (DMA_*_STALLED_LOCK_MEM |
# DMA_*_MEMORY_BACKPRESSURE_MEM) -- neither hides the other: the row is split
# top->bottom into one horizontal sub-band per PRESENT category, in a fixed order
# (lock on top, mem back-pressure, then stream) for a stable, readable layout.
# Full top->bottom stack order for a mem-DMA lane. All three stall states can be
# concurrent on one BD (lock-stalled AND memory-back-pressured AND stream-starved
# at once); each gets its own horizontal slice so none hides another. The split
# is PER COLUMN and adaptive: a column paints only the states active there,
# dividing the row height equally among them, so a lone state fills the full
# height (no reserved-but-empty white strip) while two or three concurrent states
# stack. Held-band membership is exactly this tuple.
MEM_LANE_STACK = ("DMA_STALL_LOCK", "MEM_BP", "STREAM_STALL")
# Floor for the lock / mem-bp band segment width as a fraction of the visible
# x-span so a very brief stall still rasterises to a visible width instead of
# anti-aliasing into invisibility. Stream is painted at its TRUE span (no floor)
# because it has hundreds of nanosecond bursts that padding would inflate into
# one lane-wide blob; genuinely long stream regions already span many columns.
MEM_MIN_SEG_FRAC = 0.01
OTHER_COLOR = "#7f7f7f"         # grey (multi-event or unknown)
HOST_LINE_COLOR = "#1f77b4"     # blue guide lines / host-running bars
HOST_RUNNING_COLOR = "#1f77b4"  # blue: host busy (setup / DMA / verify)
HOST_WAIT_COLOR = "#c7c7c7"     # grey hatched: host idle-waiting on the AIE

# The host phase whose *following* gap is spent blocked in XAie_CoreWaitForDone:
# the bar from the "run" marker to the next marker ("wait_done") is host-idle.
HOST_WAIT_PHASES = {"run"}

# --------------------------------------------------------------------------
# Driver-specific event names. The runtime now emits the XAie driver's exact
# event string per slot (via XAie_EventGetString) instead of a generic alias --
# e.g. DMA_S2MM_0_STREAM_STARVATION_MEM rather than STREAM_STALL. These come in
# three families, each canonicalised by regex to a generic EVENT_COLORS token
# (so colouring / pulse markers / legend keep working) plus a short human phrase
# for the legend + hover label. Without this every specific name falls through
# to grey "other" and the lane loses all meaning; the regexes also absorb the
# index / suffix variation seen across different applogs (channel 0..N, lock
# index 0..15 / SEL0..7, port index 0..7, _CORE/_MEM/_PL suffix).
#   DMA:  DMA_<S2MM|MM2S>_<ch>_<KIND>_MEM
#   LOCK: LOCK_<n>_<ACQ|REL>_MEM, LOCK_SEL<n>_<...>_MEM, LOCK_ACQUIRE/ERROR_MEM
#   PORT: PORT_<RUNNING|STALLED|IDLE|TLAST>_<n>[_CORE|_MEM|_PL]
# --------------------------------------------------------------------------
_MEM_DMA_RE = re.compile(r"^DMA_(S2MM|MM2S)_(\d+)_(.+?)_MEM$")
# (kind-substring regex, canonical EVENT_COLORS token or None, human word),
# matched in order -- specific stalls before generic START/FINISH.
_MEM_KIND_RULES = (
    (re.compile(r"STALLED_LOCK"),        "DMA_STALL_LOCK", "lock stall"),
    (re.compile(r"STREAM_STARVATION"),   "STREAM_STALL",   "stream starving"),
    (re.compile(r"STREAM_BACKPRESSURE"), "STREAM_STALL",   "stream back-pressure"),
    (re.compile(r"MEMORY_BACKPRESSURE"), "MEM_BP",         "mem back-pressure"),
    (re.compile(r"MEMORY_STARVATION"),   "MEM_BP",         "mem starvation"),
    (re.compile(r"MEMORY_CONFLICT"),     "MEM_BP",         "mem conflict"),
    (re.compile(r"FINISHED"),            "DMA_FINISH",     "finished"),
    (re.compile(r"START"),               "DMA_START",      "start"),
    (re.compile(r"GO_TO_IDLE"),          None,             "idle"),
    (re.compile(r"ERROR"),               None,             "error"),
)
# Driver lock events: LOCK_<idx>_<KIND>_MEM. <idx> is either a bare number
# (LOCK_0_REL) or SEL<n> group selector (LOCK_SEL0_ACQ_GE); LOCK_ACQUIRE / ERROR
# carry no index. KIND collapses onto the three generic lock pulse tokens.
_MEM_LOCK_RE = re.compile(r"^LOCK_(.+?)_MEM$")
_LOCK_KIND_RULES = (
    (re.compile(r"EQUAL_TO_VALUE"), "LOCK_GRP", "value"),
    (re.compile(r"REL"),            "LOCK_REL", "release"),
    (re.compile(r"ACQ|ACQUIRE"),    "LOCK_ACQ", "acquire"),
    (re.compile(r"ERROR"),          None,       "error"),
)
# Stream-switch port events at any index / module suffix.
_PORT_RE = re.compile(r"^PORT_(RUNNING|STALLED|IDLE|TLAST)_(\d+)(?:_(?:CORE|MEM|PL))?$")
_PORT_STATE_CANON = {"RUNNING": "PORT_RUNNING_0", "STALLED": "PORT_STALLED_0",
                     "IDLE": "PORT_IDLE_0"}  # TLAST has no colour slot


def _parse_mem_dma(tok):
    """Parse a driver mem-DMA event name into (dir, ch, canon, word), or None if
    `tok` is not a DMA_<dir>_<ch>_<KIND>_MEM name. `canon` is the EVENT_COLORS
    category the KIND maps to (None when uncategorised, e.g. ERROR/idle); `word`
    is the short label phrase ('stream starving')."""
    m = _MEM_DMA_RE.match(tok)
    if not m:
        return None
    direction, ch, kind = m.group(1).lower(), m.group(2), m.group(3)
    for rx, canon, word in _MEM_KIND_RULES:
        if rx.search(kind):
            return direction, ch, canon, word
    return direction, ch, None, kind.lower().replace("_", " ")


def _parse_mem_lock(tok):
    """Parse a driver lock event name into (idx, canon, word), or None if `tok`
    is not a LOCK_<...>_MEM name. `idx` is the lock/group index phrase ('0',
    'sel 3', or '' when absent); `canon` is the generic lock pulse token the
    kind maps to (None when uncategorised, e.g. ERROR); `word` is the short
    label phrase ('release')."""
    m = _MEM_LOCK_RE.match(tok)
    if not m:
        return None
    inner = m.group(1)
    idx = ""
    im = re.match(r"^SEL(\d+)", inner)
    if im:
        idx = "sel %s" % im.group(1)
    else:
        im = re.match(r"^(\d+)", inner)
        if im:
            idx = im.group(1)
    for rx, canon, word in _LOCK_KIND_RULES:
        if rx.search(inner):
            return idx, canon, word
    return idx, None, inner.lower().replace("_", " ")


def _parse_port(tok):
    """Parse a stream-switch port event name into (state, idx, canon), or None
    if `tok` is not a PORT_<STATE>_<n>[_module] name. `canon` is the generic
    PORT_*_0 colour slot (None for TLAST, which has no colour)."""
    m = _PORT_RE.match(tok)
    if not m:
        return None
    state, idx = m.group(1), m.group(2)
    return state.lower(), idx, _PORT_STATE_CANON.get(state)


def canon_token(tok):
    """Reduce a raw slot token to the canonical EVENT_COLORS category it belongs
    to. Generic / core / port tokens pass through unchanged; a driver-specific
    name collapses to its generic equivalent
    (DMA_S2MM_0_STREAM_STARVATION_MEM -> STREAM_STALL, LOCK_0_REL_MEM ->
    LOCK_REL, PORT_IDLE_3_CORE -> PORT_IDLE_0). Uncategorised names return
    unchanged (so they still read as grey 'other')."""
    if tok in EVENT_COLORS:
        return tok
    parsed = _parse_mem_dma(tok)
    if parsed and parsed[2]:
        return parsed[2]
    lk = _parse_mem_lock(tok)
    if lk and lk[1]:
        return lk[1]
    p = _parse_port(tok)
    if p and p[2]:
        return p[2]
    return tok


def rasterize_lane(events, x_lo, x_hi, nbins, priority=None, surface_colors=None):
    """Per-pixel dominant-state colouring for one tile lane.

    Splits [x_lo, x_hi] into `nbins` equal bins (one per rendered pixel column)
    and, for each bin, sums how long each colour occupies it; the colour with
    the most time wins. This makes a short burst survive as its bin's dominant
    colour instead of being overpainted by an adjacent bar (the failure mode of
    per-event rectangles). Empty bins produce no segment. Adjacent bins with the
    same colour are merged into one span.

    `priority` is an optional high->low list of colour hexes that win a bin by
    *presence* rather than duration: if any priority colour has non-zero time in
    a bin, the highest-priority present one is chosen (ignoring duration); bins
    with no priority colour fall back to plain max-duration. This keeps short but
    important states visible -- e.g. a port lane's brief RUNNING bursts that the
    long IDLE/STALLED background would otherwise drown out. With no `priority`
    (the default) behaviour is pure duration dominance.

    `surface_colors` is an optional set of colour hexes treated as a foreground
    surface over the rest (the "background"): in any bin where at least one
    surface colour is present, the surface colour with the most time wins
    (ignoring non-surface background colours); bins with no surface colour fall
    back to plain max-duration. This lets a port lane pick whichever of
    RUNNING/STALLED occupied a pixel longer, showing it full-height over the IDLE
    background, without one fixed colour always winning. `surface_colors` takes
    precedence over `priority` when both are given.

    Returns [(seg_start_us, seg_end_us, color_hex), ...] left-to-right.
    """
    if nbins < 1 or x_hi <= x_lo or not events:
        return []
    priority = priority or ()
    surface_colors = surface_colors or frozenset()
    bw = (x_hi - x_lo) / nbins
    # Per-bin colour->overlap-duration accumulator.
    acc = [dict() for _ in range(nbins)]
    for ev in events:
        s = max(ev["start_us"], x_lo)
        e = min(ev["end_us"], x_hi)
        if e <= s:
            continue
        col = event_color(ev["event"])
        b0 = int((s - x_lo) / bw)
        b1 = int((e - x_lo) / bw)
        if b1 >= nbins:
            b1 = nbins - 1
        for b in range(b0, b1 + 1):
            bin_lo = x_lo + b * bw
            bin_hi = bin_lo + bw
            overlap = min(e, bin_hi) - max(s, bin_lo)
            if overlap > 0.0:
                acc[b][col] = acc[b].get(col, 0.0) + overlap

    # Winning colour per bin: a surface colour by max-duration-among-present,
    # else a priority colour by presence, else plain max-duration.
    # Then merge adjacent runs of the same colour.
    segs = []
    for b in range(nbins):
        if not acc[b]:
            continue
        col = None
        present_surface = [(c, t) for c, t in acc[b].items()
                           if c in surface_colors and t > 0.0]
        if present_surface:
            col = max(present_surface, key=lambda ct: ct[1])[0]
        if col is None:
            for pc in priority:
                if acc[b].get(pc, 0.0) > 0.0:
                    col = pc
                    break
        if col is None:
            col = max(acc[b].items(), key=lambda kv: kv[1])[0]
        bin_lo = x_lo + b * bw
        bin_hi = bin_lo + bw
        if segs and segs[-1][2] == col and abs(segs[-1][1] - bin_lo) < bw * 1e-6:
            segs[-1] = (segs[-1][0], bin_hi, col)
        else:
            segs.append((bin_lo, bin_hi, col))
    return segs


def rasterize_lane_category(events, canon, x_lo, x_hi, nbins):
    """Rasterise a lane using ONLY the tokens that canonicalise to `canon`.

    Each event label may be a '|'-joined set of slot tokens (e.g.
    "DMA_MM2S_0_STALLED_LOCK_MEM|DMA_MM2S_0_STREAM_BACKPRESSURE_MEM"). The mem
    lane split renders one horizontal band per stall category, so a band must
    count an interval's time ONLY when that interval carries a token of its
    category -- a lock-stall band ignores the stream-stall token in the same
    interval and vice versa. Intervals lacking the category contribute nothing.

    Returns the same [(seg_start_us, seg_end_us, color_hex), ...] shape as
    `rasterize_lane`, always coloured with EVENT_COLORS[canon].
    """
    masked = []
    for ev in events:
        toks = str(ev["event"]).split("|")
        if any(canon_token(t) == canon for t in toks):
            masked.append({"start_us": ev["start_us"], "end_us": ev["end_us"],
                           "event": canon})
    # A single-category masked lane has one colour, so plain duration dominance
    # (no priority) merges adjacent bins into contiguous spans.
    return rasterize_lane(masked, x_lo, x_hi, nbins)


def widen_segments(segs, min_w):
    """Floor each rasterised segment to `min_w` microseconds wide, centred on its
    midpoint. A brief mem-DMA stall burst rasterises to a sub-pixel-thin span
    that matplotlib anti-aliases into the background (the "bottom-half stream
    back-pressure not illustrated" case: real stalls totalling a few us over a
    several-hundred-us run vanish). Widening thin segments to a visible floor
    keeps the burst on screen; already-wide segments pass through untouched.

    Returns a new [(seg_start_us, seg_end_us, color_hex), ...] list.
    """
    out = []
    for s, e, c in segs:
        if e - s < min_w:
            mid = (s + e) / 2.0
            s, e = mid - min_w / 2.0, mid + min_w / 2.0
        out.append((s, e, c))
    return out


def event_color(event):
    """Colour for an event label. A '|'-joined multi-event prefers a stall
    token over ACTIVE, so an ACTIVE|LOCK_STALL span shows the stall colour
    (a core can be ACTIVE while blocked on a lock/stream/memory). Falls back
    to the first recognised token, else grey. Driver-specific mem-DMA names are
    canonicalised first, so DMA_S2MM_0_STREAM_STARVATION_MEM colours as a
    stream stall instead of falling through to grey 'other'."""
    toks = [canon_token(t) for t in str(event).split("|")]
    for tok in toks:
        if tok in STALL_ORDER or tok in MEM_STALL_ORDER:
            return EVENT_COLORS[tok]
    for tok in toks:
        if tok in EVENT_COLORS:
            return EVENT_COLORS[tok]
    return OTHER_COLOR


def pulse_tokens(event):
    """Return the pulse/edge tokens present in a (possibly '|'-joined) event
    label, in EVENT_COLORS declaration order, de-duplicated. Level/state tokens
    (ACTIVE, stalls, back-pressure) are excluded. Used to overlay a coloured
    vertical marker per pulse -- a zero-width event no span rectangle can show.
    Driver-specific mem-DMA names are canonicalised first so a
    DMA_S2MM_0_START_TASK_MEM still registers as a DMA_START pulse."""
    toks = set(canon_token(t) for t in str(event).split("|"))
    return [name for name in EVENT_COLORS
            if name in PULSE_EVENTS and name in toks]


# --------------------------------------------------------------------------
# Human-readable event labels (legend + on-plot text). A lane's role/ident
# (from host_aie_timeline.correlate) turns a raw slot into a port/DMA-aware
# phrase: PORT_RUNNING_0 on a "south slave 0" port lane -> "south slave port 0
# running"; STREAM_STALL on a "s2mm 0" mem lane -> "dma s2mm 0 stream starving".
# --------------------------------------------------------------------------
# Core-lane fallback names (no role/ident context).
FRIENDLY = {
    "ACTIVE": "AIE running",
    "LOCK_STALL": "lock stall",
    "STREAM_STALL": "stream stall",
    "MEMORY_STALL": "memory stall",
    "PORT_RUNNING_0": "port running",
    "PORT_STALLED_0": "port stalled",
    "PORT_IDLE_0": "port idle",
    "DMA_START": "DMA start",
    "DMA_FINISH": "DMA finished",
    "DMA_STALL_LOCK": "DMA lock stall",
    "MEM_BP": "mem back-pressure",
    "LOCK_GRP": "lock group",
    "LOCK_ACQ": "lock acquire",
    "LOCK_REL": "lock release",
}
# Port-lane state word per PORT_*_0 slot.
PORT_STATE = {"PORT_RUNNING_0": "running", "PORT_STALLED_0": "stalled",
              "PORT_IDLE_0": "idle"}
# Mem-DMA-lane event word per mem slot. STREAM_STALL reads "stream starving"
# (the S2MM starvation domain term); the rest mirror FRIENDLY.
MEM_WORD = {
    "DMA_START": "start", "DMA_FINISH": "finished", "DMA_STALL_LOCK": "lock stall",
    "STREAM_STALL": "stream starving", "MEM_BP": "back-pressure",
    "LOCK_GRP": "lock group", "LOCK_ACQ": "lock acquire", "LOCK_REL": "lock release",
}


def _port_prefix(ident):
    """'south slave 0' -> 'south slave port 0' (insert 'port' before the index)."""
    parts = ident.split()
    if len(parts) == 3:
        return "%s %s port %s" % (parts[0], parts[1], parts[2])
    return (ident + " port").strip() if ident else "port"


def token_label(tok, role, ident):
    """Enriched label for a single slot token given its lane's role/ident.
    Port lanes -> '<port> <intf> port <num> <state>'; mem lanes -> 'dma <dir>
    <ch> <word>'; core/host or missing context -> the FRIENDLY name.

    A driver-specific mem-DMA or lock name carries its own direction/channel or
    lock index, so it is parsed straight into 'dma <dir> <ch> <word>' /
    'lock <idx> <word>' (e.g. DMA_S2MM_0_STREAM_STARVATION_MEM -> 'dma s2mm 0
    stream starving', LOCK_SEL0_ACQ_GE_MEM -> 'lock sel 0 acquire'), independent
    of the lane's ident. Port names at any index/module suffix fold into the
    lane's port phrase via their canonical state."""
    parsed = _parse_mem_dma(tok)
    if parsed:
        direction, ch, _canon, word = parsed
        return "dma %s %s %s" % (direction, ch, word)
    lk = _parse_mem_lock(tok)
    if lk:
        idx, _canon, word = lk
        return ("lock %s %s" % (idx, word)) if idx else ("lock %s" % word)
    if role == "port":
        state_word = PORT_STATE.get(tok)
        if state_word is None:
            p = _parse_port(tok)
            if p and p[2]:
                state_word = PORT_STATE[p[2]]
        if state_word is not None:
            if ident:
                return "%s %s" % (_port_prefix(ident), state_word)
            return "port %s" % state_word
    if role == "mem" and tok in MEM_WORD:
        return "%s %s" % (("dma %s" % ident) if ident else "dma", MEM_WORD[tok])
    return FRIENDLY.get(tok, tok)


def _dominant_token(event):
    """The token that colours a '|'-joined span (event_color priority): a stall
    over ACTIVE/DMA_START, else the first recognised slot, else the first token.
    The RAW token is returned (not its canonical form) so token_label can still
    recover a mem-DMA name's direction/channel; canonicalisation is used only to
    decide priority."""
    toks = str(event).split("|")
    for tok in toks:
        c = canon_token(tok)
        if c in STALL_ORDER or c in MEM_STALL_ORDER:
            return tok
    for tok in toks:
        if canon_token(tok) in EVENT_COLORS:
            return tok
    return toks[0] if toks else ""


def event_label(event, lane):
    """On-plot label for a span: the dominant token rendered through the lane's
    role/ident (falls back gracefully for lanes lacking role/ident)."""
    lane = lane or {}
    return token_label(_dominant_token(event), lane.get("role", ""),
                       lane.get("ident", ""))


# --------------------------------------------------------------------------
# Model access.
# --------------------------------------------------------------------------
def load_model(path):
    with open(path) as f:
        return json.load(f)


def is_host_lane(lane):
    return lane.get("name") == "host"


def tile_lanes(model):
    return [l for l in model["lanes"] if not is_host_lane(l)]


def host_lane(model):
    for l in model["lanes"]:
        if is_host_lane(l):
            return l
    return None


def _parse_host_marker(event):
    """Split a host event label 'iter<N>.<phase>' into (iter, phase).
    Returns (None, event) if it does not match that shape."""
    s = str(event)
    if s.startswith("iter") and "." in s:
        head, phase = s.split(".", 1)
        try:
            return int(head[4:]), phase
        except ValueError:
            return None, s
    return None, s


def host_intervals(hl):
    """Turn the host lane's phase markers into horizontal bars, one per gap
    between consecutive markers. Each bar is a dict:
        {start_us, end_us, phase, iter, waiting, label, detail,
         color, api, hostcc_line}
    The bar spans from a phase's START marker to the NEXT phase's START (the host
    records only starts, not API durations). 'waiting' is True for the
    run->wait_done gap (host blocked in XAie_CoreWaitForDone); every other gap is
    host-running work. 'color'/'api'/'hostcc_line' are carried through from the
    leading marker when host_aie_timeline enriched it (colour-by-API + host.cc
    line); they are None for a bare model (e.g. the synthetic self-test), which
    then falls back to the running/waiting rendering."""
    if hl is None:
        return []
    evs = sorted(hl["events"], key=lambda e: e["start_us"])
    bars = []
    for cur, nxt in zip(evs, evs[1:]):
        it, phase = _parse_host_marker(cur["event"])
        x0 = cur["start_us"]
        x1 = nxt["start_us"]
        if x1 <= x0:
            continue
        waiting = phase in HOST_WAIT_PHASES
        bars.append({
            "start_us": x0, "end_us": x1, "phase": phase, "iter": it,
            "waiting": waiting,
            "label": ("wait" if waiting else phase),
            "detail": "%s -> %s" % (cur["event"], nxt["event"]),
            "color": cur.get("color"),
            "api": cur.get("api"),
            "hostcc_line": cur.get("hostcc_line"),
        })
    return bars


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------
def _pick_backend(save, want_window):
    """Choose a matplotlib backend BEFORE importing pyplot. Returns
    (show_window, save_path_or_None)."""
    has_display = bool(os.environ.get("DISPLAY")) or sys.platform == "darwin"
    if save or not (want_window and has_display):
        import matplotlib
        matplotlib.use("Agg")
        return False, (save or "timeline.png")
    return True, None


# Fraction of the run span a tile event may extend past the host window before
# it is treated as a garbled trace-decode outlier rather than real activity.
_XRANGE_MARGIN_FRAC = 1.0
# Percentile clip used only when there is no host lane to anchor the axis.
_XRANGE_CLIP_LO, _XRANGE_CLIP_HI = 0.01, 0.99


def _robust_xrange(model):
    """X-axis range that ignores stray trace-decode outliers.

    A single garbled memory/DMA-trace packet can decode to a cycle far outside
    the run (e.g. a stray event at -123620 us while the run spans 5900..6540 us).
    Left unclamped, min()/max() stretch the axis over ~130000 us and squash every
    lane into one column. The host lane is the ground-truth axis (its times come
    straight from host counts, never from trace-cycle decode), so tile events are
    kept only when they fall within `_XRANGE_MARGIN_FRAC` of the host span; the
    outlier is clamped out. With no host lane, fall back to a wide percentile
    clip so one point still can't dominate.

    Returns (x_lo, x_hi)."""
    all_x = [e["start_us"] for l in model["lanes"] for e in l["events"]] + \
            [e["end_us"] for l in model["lanes"] for e in l["events"]]
    if not all_x:
        return 0.0, 1.0
    host = host_lane(model)
    if host is not None and host["events"]:
        hx = [e["start_us"] for e in host["events"]] + \
             [e["end_us"] for e in host["events"]]
        h_lo, h_hi = min(hx), max(hx)
        margin = max(h_hi - h_lo, 1.0) * _XRANGE_MARGIN_FRAC
        lo_ok, hi_ok = h_lo - margin, h_hi + margin
        kept = [x for x in all_x if lo_ok <= x <= hi_ok]
        if kept:
            return min(kept), max(kept)
        return h_lo, h_hi
    # No host lane: clip to a wide percentile so a lone outlier can't dominate.
    xs = sorted(all_x)
    n = len(xs)
    lo = xs[int(_XRANGE_CLIP_LO * (n - 1))]
    hi = xs[int(_XRANGE_CLIP_HI * (n - 1))]
    return (lo, hi) if hi > lo else (xs[0], xs[-1])


def _draw(ax, model, nbins=2000):
    """Draw all lanes onto ax as HORIZONTAL timelines on one us axis. Returns a
    hover index: a list of (x0, x1, y0, y1, event_dict, lane_name) tuples.

    `nbins` is the number of rasterisation columns for tile lanes (set by
    render() to the figure's pixel width).

    Row 0 is the host lane: one bar per gap between consecutive phase markers,
    BLUE where the host is running and GREY HATCHED where it is idle-waiting on
    the AIE (the run->wait_done gap). Rows 1.. are the tiles, bars coloured by
    event type, so a tile's green ACTIVE bar lines up under the host wait bar."""
    hl = host_lane(model)
    tiles = tile_lanes(model)
    row_names = (["host"] if hl is not None else []) + [l["name"] for l in tiles]
    host_row = 0 if hl is not None else None
    tile_row0 = 1 if hl is not None else 0
    n_rows = len(row_names)
    row_h = 0.8
    hover = []  # (x0, x1, y0, y1, event, lane_name)

    # X-range across every lane, with stray trace-decode outliers clamped out so
    # one garbled packet can't stretch the axis and squash the run to a line
    # (see _robust_xrange). Tile lanes are rasterised into `nbins` columns over
    # this range so per-pixel dominant colouring is stable.
    x_lo, x_hi = _robust_xrange(model)

    # Host lane (row 0): one bar per gap between consecutive phase markers,
    # painted from a phase START to the NEXT phase START (the host records only
    # starts, not API durations). When host_aie_timeline enriched the markers,
    # the bar is COLOURED BY ITS HOST API (per-phase colour) and annotated with
    # the host.cc:line that issued it; otherwise it falls back to the blue
    # running / grey-hatched waiting rendering. Only iteration <=0 markers are
    # annotated so a multi-iteration run does not stack duplicate labels.
    if hl is not None:
        y0 = host_row - row_h / 2.0
        for bar in host_intervals(hl):
            x0 = bar["start_us"]
            w = bar["end_us"] - bar["start_us"]
            kw = dict(edgecolors="black", linewidth=0.4)
            if bar.get("color"):
                kw.update(facecolors=bar["color"])
            elif bar["waiting"]:
                kw.update(facecolors=HOST_WAIT_COLOR, hatch="//")
            else:
                kw.update(facecolors=HOST_RUNNING_COLOR)
            ax.broken_barh([(x0, w)], (y0, row_h), **kw)
            line = bar.get("hostcc_line")
            if line is not None and bar.get("iter") in (None, -1, 0):
                # Offset points are display-space (up on screen), independent of
                # the inverted y-axis, so va="bottom" places the label just above
                # the host row's top edge; bbox_inches="tight" keeps it visible.
                ax.annotate("host.cc:%d" % line, xy=(x0, y0),
                            xytext=(0, 6), textcoords="offset points",
                            rotation=45, fontsize=6, ha="left", va="bottom",
                            color="#333333", zorder=7)
            # No on-bar text: the phase word is available via hover instead.
            ev = {"event": bar["label"], "start_us": bar["start_us"],
                  "end_us": bar["end_us"], "detail": bar["detail"]}
            hover.append((x0, x0 + w, y0, y0 + row_h, ev, None))

    # Tile lanes: one row each, drawn with per-pixel dominant-state colouring.
    # Each bin (~one screen column) is filled with the colour that occupies most
    # of its time, so a short burst survives as its bin's dominant colour instead
    # of being overpainted by an adjacent bar. Hover still reports true per-event
    # spans (unaffected by rasterisation).
    for i, lane in enumerate(tiles):
        row = tile_row0 + i
        y0 = row - row_h / 2.0
        # A mem-DMA lane draws all three held-duration stall states (lock stall,
        # memory back-pressure, stream stall) as rasterised bands over their real
        # spans, and additionally overlays stream as per-event ticks. When several
        # states are present in the same column the row is split top->bottom into
        # one sub-band per present category (lock, mem-bp, stream) so concurrent
        # stalls read as a vertical stack instead of one hiding the other. With a
        # single category present the band fills the full row height.
        # Every other lane fills the full row height with per-pixel dominant
        # colouring; some prioritise brief bursts by presence (a port lane's
        # RUNNING/STALLED over IDLE, a core lane's ACTIVE over the long stall
        # background).
        if lane.get("role") == "mem":
            # Per-column adaptive stack of the mem-DMA stall states (lock stall,
            # memory back-pressure, stream stall). A single BD can be blocked on
            # all three at once, so any column paints every state active there,
            # dividing the row height equally among them (MEM_LANE_STACK order,
            # top->bottom). A lone state fills the full height (no white gap); two
            # or three concurrent states stack. This is what makes the lock band,
            # the purple mem-back-pressure band, and the red stream band all
            # visible together (the "lock/mem-bp/50984-cyc not shown" and the long
            # stream-starvation region drawn as blank white cases), instead of one
            # full-height layer hiding the others.
            bw = (x_hi - x_lo) / max(nbins, 1)
            stream_canon = "STREAM_STALL"
            # EVERY stall state on a mem-DMA lane is a HELD-duration band: lock
            # stall, memory back-pressure AND stream stall (starvation /
            # back-pressure) all carry real start->end spans and must fill their
            # duration. A long stream-starvation region (tens of us) was
            # previously drawn as a single tick, leaving its whole span blank
            # ("illustration not correct"); it is now a full band, matching the
            # s2mm.png ground truth (clean stacked bands, no per-event ticks).
            # Per-column PRESENCE floor: lock / mem-bp widen by `pad` so a sub-us
            # stall is still a visible run of columns. Stream is painted at its
            # TRUE span (pad 0): with hundreds of nanosecond bursts, padding would
            # inflate them into one lane-wide red blob, while genuinely long stream
            # regions already span many columns on their own.
            pad = max(int(round(MEM_MIN_SEG_FRAC * nbins / 2.0)), 1)
            cat_pad = {c: pad for c in MEM_LANE_STACK}
            cat_pad[stream_canon] = 0
            pres = {c: bytearray(nbins) for c in MEM_LANE_STACK}
            for ev in lane["events"]:
                cats = set(canon_token(t) for t in str(ev["event"]).split("|"))
                if not (cats & pres.keys()):
                    continue
                s = max(ev["start_us"], x_lo)
                e = min(ev["end_us"], x_hi)
                b0 = int(((s if e > s else max(ev["start_us"], x_lo)) - x_lo) / bw)
                b1 = int((e - x_lo) / bw) if e > s else b0
                for c in cats & pres.keys():
                    p = cat_pad[c]
                    a = pres[c]
                    for b in range(max(b0 - p, 0), min(b1 + p, nbins - 1) + 1):
                        a[b] = 1

            def _active(b):
                return tuple(c for c in MEM_LANE_STACK if pres[c][b])

            # Held bands (lock, mem-bp, stream): draw one filled rect per
            # contiguous run of columns with an identical active-set, each present
            # state in its own equal horizontal slice (MEM_LANE_STACK order,
            # top->bottom). A lone state fills the full row height; two or three
            # concurrent states stack.
            b = 0
            while b < nbins:
                act = _active(b)
                if not act:
                    b += 1
                    continue
                b2 = b
                while b2 + 1 < nbins and _active(b2 + 1) == act:
                    b2 += 1
                seg = row_h / len(act)
                x_start = x_lo + b * bw
                width = (b2 - b + 1) * bw
                for j, c in enumerate(act):
                    ax.broken_barh([(x_start, width)], (y0 + j * seg, seg),
                                   facecolors=EVENT_COLORS[c],
                                   edgecolors="none")
                b = b2 + 1
        elif lane.get("role") == "port":
            # One full-height row, coloured per pixel by whichever of RUNNING or
            # STALLED occupied that pixel column longer (the "surface" over the
            # IDLE background). Sequential stall-then-run bursts thus render as
            # adjacent per-pixel marks -- no overlay, no fixed running priority --
            # so a stall is shown wherever it out-lasted the run in its column.
            surface = {EVENT_COLORS["PORT_RUNNING_0"],
                       EVENT_COLORS["PORT_STALLED_0"]}
            for s_us, e_us, color in rasterize_lane(
                    lane["events"], x_lo, x_hi, nbins, surface_colors=surface):
                ax.broken_barh([(s_us, e_us - s_us)], (y0, row_h),
                               facecolors=color, edgecolors="none")
        else:
            prio = {"core": CORE_RASTER_PRIORITY}.get(lane.get("role"))
            for s_us, e_us, color in rasterize_lane(lane["events"], x_lo, x_hi, nbins, prio):
                ax.broken_barh([(s_us, e_us - s_us)], (y0, row_h),
                               facecolors=color, edgecolors="none")
        for ev in lane["events"]:
            x0 = ev["start_us"]
            w = max(ev["end_us"] - ev["start_us"], 0.0)
            x1 = x0 + w
            hover.append((x0, x1, y0, y0 + row_h, ev, lane))
            # No on-plot bar text: the port/DMA-aware label is available via the
            # legend (per event type) and hover (per span) instead of overlaying
            # the bars, which crowded narrow spans and long driver names.
            # Overlay each pulse/edge token as its own coloured vertical marker
            # at the event's start. A single-cycle DMA_FINISH (lone or riding a
            # stall label) is invisible as a span; the line makes it show. When
            # several pulses share a label they stack within the row height.
            ptoks = pulse_tokens(ev["event"])
            if ptoks:
                seg = row_h / len(ptoks)
                for k, tok in enumerate(ptoks):
                    ya = y0 + k * seg
                    col = EVENT_COLORS[tok]
                    ax.plot([x0, x0], [ya, ya + seg],
                            color=col, linewidth=1.2, zorder=6)
                    mk = PULSE_MARKERS.get(tok)
                    if mk:
                        # Point marker at the (visual) top of the line; the axis
                        # is inverted so the smaller y sits highest on screen.
                        # zorder 9 keeps the glyph ABOVE the stream-stall ticks
                        # (zorder 8): a DMA_FINISH that coincides with a back-
                        # pressure event would otherwise be hidden under the red
                        # tick, so the glyph must poke out on top to stay visible.
                        ax.plot([x0], [ya], marker=mk, color=col, markersize=5,
                                linestyle="none", zorder=9)

    # Light dashed vertical guides at each host phase boundary, across all rows.
    if hl is not None and n_rows > 0:
        ytop = n_rows - 1 + row_h
        ybot = -row_h
        for ev in hl["events"]:
            x = ev["start_us"]
            ax.plot([x, x], [ybot, ytop], linestyle="--", linewidth=0.6,
                    color=HOST_LINE_COLOR, alpha=0.35, zorder=0)
            # No phase-name text at the guide: the dashed line marks the boundary;
            # the phase word is available via the host-bar hover.

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_names)
    ax.set_ylim(-1.0, max(n_rows - 1 + 1.0, 1.0))
    ax.invert_yaxis()  # host on top, then tiles
    # Pin the axis to the clamped range so a stray off-range event (host guide or
    # per-event pulse marker drawn at its true position) can't re-expand it via
    # matplotlib autoscale; out-of-range artists are simply clipped.
    if x_hi > x_lo:
        ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("time (us, t=0 at host anchor0)")

    meta = model.get("meta", {})
    ax.set_title("AIE/host timeline  cps=%s  tiles=%s  host_span=%.2f us" % (
        meta.get("cps", "?"), meta.get("num_tiles", len(tiles)),
        meta.get("host_span_us", 0.0)))
    return hover


def _add_legend(ax, model):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = []
    # Host bars, if a host lane is present. When the markers are enriched
    # (colour-by-API), one legend entry per distinct phase carries its API and
    # host.cc line ("dma_start  (host.cc:710  startio (issue DMAs))"); otherwise
    # fall back to the plain running / waiting swatches.
    hl = host_lane(model)
    if hl is not None:
        bars = host_intervals(hl)
        api_bars = [b for b in bars if b.get("color")]
        if api_bars:
            seen_phase = OrderedDict()
            for b in api_bars:
                ph = b.get("phase")
                if ph in seen_phase:
                    continue
                api = b.get("api") or ph
                line = b.get("hostcc_line")
                lbl = ("%s  (host.cc:%d  %s)" % (ph, line, api)) if line is not None \
                    else ("%s  (%s)" % (ph, api))
                seen_phase[ph] = (b["color"], lbl)
            for ph, (col, lbl) in seen_phase.items():
                handles.append(Patch(facecolor=col, edgecolor="black", label=lbl))
        else:
            if any(not b["waiting"] for b in bars):
                handles.append(Patch(facecolor=HOST_RUNNING_COLOR, edgecolor="black",
                                     label="host running"))
            if any(b["waiting"] for b in bars):
                handles.append(Patch(facecolor=HOST_WAIT_COLOR, edgecolor="black",
                                     hatch="//", label="host waiting (AIE)"))
    # One legend entry per distinct enriched LABEL present across the tile lanes,
    # so the same slot on different lanes reads differently ("stream stall" on a
    # core lane vs. "dma s2mm 0 stream starving" on a mem lane, "south slave port
    # 0 running" on a port lane). Ordered by the token's EVENT_COLORS index so
    # related states group together; deduped by label.
    order = {name: i for i, name in enumerate(EVENT_COLORS)}
    seen = {}  # label -> (order_index, color_hex, pulse_marker)
    other = False
    for lane in tile_lanes(model):
        role = lane.get("role", "")
        ident = lane.get("ident", "")
        for ev in lane["events"]:
            for tok in str(ev["event"]).split("|"):
                canon = canon_token(tok)
                if canon in EVENT_COLORS:
                    label = token_label(tok, role, ident)
                    if label not in seen:
                        seen[label] = (order[canon], EVENT_COLORS[canon],
                                       PULSE_MARKERS.get(canon))
                else:
                    other = True
    for label, (_, col, mk) in sorted(seen.items(), key=lambda kv: kv[1][0]):
        if mk:
            # Match the on-plot look: a coloured line + its point marker so
            # the DMA-start swatch reads the same as the drawn event.
            handles.append(Line2D([0], [0], color=col, marker=mk,
                                  linestyle="-", markersize=6, label=label))
        else:
            handles.append(Patch(facecolor=col, edgecolor="black", label=label))
    if other:
        handles.append(Patch(facecolor=OTHER_COLOR, edgecolor="black", label="other"))
    if handles:
        # Anchor the legend OUTSIDE the axes on the right so its colour swatches
        # never overlay the timeline bars (render() saves with bbox_inches=tight
        # so the outside box is not clipped).
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=7, framealpha=0.9, borderaxespad=0.0)


def _attach_hover(fig, ax, hover):
    """Show an annotation with the event under the cursor."""
    ann = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="#ffffcc", ec="black", alpha=0.95),
                      fontsize=7, zorder=10)
    ann.set_visible(False)

    def find(x, y):
        # Prefer a bar hit; a host marker matches within a small x tolerance.
        xtol = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.004 + 1e-9
        for x0, x1, y0, y1, ev, lane in hover:
            if y0 <= y <= y1:
                if x0 == x1:
                    if abs(x - x0) <= xtol:
                        return ev, lane
                elif x0 <= x <= x1:
                    return ev, lane
        return None

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return
        hit = find(event.xdata, event.ydata)
        if hit is None:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return
        ev, lane = hit
        # `lane` is the lane dict for tile rows, None for the host lane. Show the
        # enriched label (canonicalised DMA name -> "dma s2mm 0 stream starving")
        # alongside the raw event, since the bars no longer carry on-plot text.
        lane_name = lane["name"] if lane else "host"
        friendly = event_label(ev["event"], lane) if lane else ev["event"]
        ann.xy = (event.xdata, event.ydata)
        ann.set_text("%s / %s\n%.3f .. %.3f us\n%s" % (
            lane_name, friendly, ev["start_us"], ev["end_us"], ev.get("detail", "")))
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def render(model, save=None, want_window=True):
    """Build the figure. If not showing a window, write a PNG to `save`
    (default 'timeline.png') and return its path; else show() and return None."""
    show_window, save_path = _pick_backend(save, want_window)
    import matplotlib.pyplot as plt

    n = len(tile_lanes(model)) + (1 if host_lane(model) is not None else 0)
    n = max(n, 1)
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.6 * n))
    # One rasterisation bin per horizontal pixel of the figure.
    nbins = max(int(fig.get_figwidth() * fig.dpi), 100)
    hover = _draw(ax, model, nbins)
    _add_legend(ax, model)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()

    if show_window:
        _attach_hover(fig, ax, hover)
        plt.show()
        return None
    # bbox_inches="tight" keeps the outside-right legend from being clipped.
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Self-test: synthetic 2-tile model (no file, no board).
# --------------------------------------------------------------------------
def _synthetic_model():
    return {
        "meta": {"cps": 1000000, "num_tiles": 2, "host_span_us": 1000.0},
        "anchors": {},
        "fit_per_tile": {"4,4": {"counts_per_cycle": 1.0, "aie_hz": 1e9},
                         "5,5": {"counts_per_cycle": 1.0, "aie_hz": 1e9}},
        "lanes": [
            {"name": "host", "events": [
                {"event": "iter0.iter_start", "start_us": 0.0, "end_us": 0.0, "detail": "host=1000"},
                {"event": "iter0.dma_in_start", "start_us": 50.0, "end_us": 50.0, "detail": "host=1050"},
                {"event": "iter0.dma_in_done", "start_us": 120.0, "end_us": 120.0, "detail": "host=1120"},
                {"event": "iter0.run", "start_us": 130.0, "end_us": 130.0, "detail": "host=1130"},
                {"event": "iter0.wait_done", "start_us": 700.0, "end_us": 700.0, "detail": "host=1700"},
                {"event": "iter0.dma_out_start", "start_us": 720.0, "end_us": 720.0, "detail": "host=1720"},
                {"event": "iter0.dma_out_done", "start_us": 800.0, "end_us": 800.0, "detail": "host=1800"},
            ]},
            {"name": "tile 4,4", "events": [
                {"event": "ACTIVE", "start_us": 135.0, "end_us": 690.0, "detail": "cycles 135..689"},
            ]},
            {"name": "tile 5,5", "events": [
                {"event": "ACTIVE", "start_us": 140.0, "end_us": 520.0, "detail": "cycles 140..519"},
                {"event": "LOCK_STALL", "start_us": 520.0, "end_us": 690.0, "detail": "cycles 520..689"},
            ]},
        ],
    }


def _self_test():
    import tempfile
    model = _synthetic_model()
    with tempfile.TemporaryDirectory() as d:
        png = os.path.join(d, "timeline.png")
        out = render(model, save=png, want_window=False)
        assert out == png and os.path.getsize(png) > 0, "PNG not written"
    print("self-test OK: rendered synthetic 2-tile timeline to a PNG")
    return 0


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
def main(argv):
    ap = argparse.ArgumentParser(description="Draw a host<->AIE timeline from timeline.json.")
    ap.add_argument("json", nargs="?", help="timeline.json produced by host_aie_timeline.py")
    ap.add_argument("--save", metavar="PNG", help="write a PNG instead of opening a window")
    ap.add_argument("--self-test", action="store_true", help="render a synthetic model and exit")
    args = ap.parse_args(argv[1:])

    if args.self_test:
        return _self_test()
    if not args.json:
        ap.error("json is required (or use --self-test)")

    model = load_model(args.json)
    out = render(model, save=args.save, want_window=True)
    if out:
        print("[timeline_gui] wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
