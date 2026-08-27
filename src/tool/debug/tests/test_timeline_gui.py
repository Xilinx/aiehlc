#!/usr/bin/env python3
"""Unit test for timeline_gui.py (src/tool/debug/timeline_gui.py).

Headless (Agg backend): no window, no board, no display. Asserts
  1. event_color maps each slot type + multi/other correctly,
  2. lane helpers split host vs tile lanes,
  3. render() writes a non-empty PNG,
  4. rendering a real timeline.json (built via host_aie_timeline.correlate)
     round-trips to a PNG,
  5. _self_test() returns 0.
"""

import os
import sys
import tempfile

# Force headless BEFORE timeline_gui imports pyplot.
import matplotlib  # noqa: E402
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # timeline_gui, host_aie_timeline

import timeline_gui as tg  # noqa: E402


def test_event_color_known_slots():
    assert tg.event_color("ACTIVE") == tg.EVENT_COLORS["ACTIVE"]
    assert tg.event_color("LOCK_STALL") == tg.EVENT_COLORS["LOCK_STALL"]
    assert tg.event_color("STREAM_STALL") == tg.EVENT_COLORS["STREAM_STALL"]
    assert tg.event_color("MEMORY_STALL") == tg.EVENT_COLORS["MEMORY_STALL"]


def test_event_color_multi_prefers_stall():
    # A stall token wins over ACTIVE (core is ACTIVE even while blocked).
    assert tg.event_color("ACTIVE|LOCK_STALL") == tg.EVENT_COLORS["LOCK_STALL"]
    assert tg.event_color("ACTIVE|STREAM_STALL") == tg.EVENT_COLORS["STREAM_STALL"]
    assert tg.event_color("ACTIVE|MEMORY_STALL") == tg.EVENT_COLORS["MEMORY_STALL"]
    # No stall token: first recognised token wins.
    assert tg.event_color("STREAM_STALL|EVENT5") == tg.EVENT_COLORS["STREAM_STALL"]
    assert tg.event_color("ACTIVE|EVENT5") == tg.EVENT_COLORS["ACTIVE"]


def test_event_color_unknown_is_other():
    assert tg.event_color("EVENT5") == tg.OTHER_COLOR
    assert tg.event_color("EVENT6|EVENT7") == tg.OTHER_COLOR


def test_rasterize_dominant_active_over_minority_stall():
    # One bin covering [0,20]: 19.95us ACTIVE vs 0.05us LOCK_STALL -> green.
    events = [
        {"start_us": 0.0, "end_us": 10.0, "event": "ACTIVE"},
        {"start_us": 10.0, "end_us": 10.05, "event": "LOCK_STALL"},
        {"start_us": 10.05, "end_us": 20.0, "event": "ACTIVE"},
    ]
    assert tg.rasterize_lane(events, 0.0, 20.0, 1) == \
        [(0.0, 20.0, tg.EVENT_COLORS["ACTIVE"])]


def test_rasterize_dominant_stall():
    # A stall-dominated bin colours orange (event_color prefers the stall).
    events = [{"start_us": 0.0, "end_us": 20.0, "event": "ACTIVE|LOCK_STALL"}]
    assert tg.rasterize_lane(events, 0.0, 20.0, 1) == \
        [(0.0, 20.0, tg.EVENT_COLORS["LOCK_STALL"])]


def test_rasterize_merges_adjacent_and_splits_on_color():
    # 4 bins over [0,20]: first half ACTIVE, second half LOCK_STALL.
    events = [
        {"start_us": 0.0, "end_us": 10.0, "event": "ACTIVE"},
        {"start_us": 10.0, "end_us": 20.0, "event": "LOCK_STALL"},
    ]
    assert tg.rasterize_lane(events, 0.0, 20.0, 4) == [
        (0.0, 10.0, tg.EVENT_COLORS["ACTIVE"]),
        (10.0, 20.0, tg.EVENT_COLORS["LOCK_STALL"]),
    ]


def test_rasterize_skips_empty_bins():
    # bins [0,5] and [5,10] have no events -> no segment there.
    events = [{"start_us": 12.0, "end_us": 18.0, "event": "ACTIVE"}]
    assert tg.rasterize_lane(events, 0.0, 20.0, 4) == \
        [(10.0, 20.0, tg.EVENT_COLORS["ACTIVE"])]


def test_rasterize_short_active_burst_visible_when_dominant():
    # The real bug: a sub-pixel ACTIVE run interrupted by a 1-cycle stall.
    # Per-pixel dominance keeps the bin green instead of losing it to overpaint.
    events = [
        {"start_us": 0.0, "end_us": 0.05, "event": "ACTIVE"},
        {"start_us": 0.05, "end_us": 0.051, "event": "ACTIVE|LOCK_STALL"},
        {"start_us": 0.051, "end_us": 0.10, "event": "ACTIVE"},
    ]
    assert tg.rasterize_lane(events, 0.0, 0.10, 1) == \
        [(0.0, 0.10, tg.EVENT_COLORS["ACTIVE"])]


def test_rasterize_priority_shows_short_running_over_dominant_idle():
    # The reported bug: a port lane's brief RUNNING burst (0.01us) inside an
    # IDLE-dominated bin (19.99us) vanishes under pure duration dominance. With
    # the port presence-priority it wins the bin so the transfer stays visible.
    events = [
        {"start_us": 0.0, "end_us": 10.0, "event": "PORT_IDLE_0"},
        {"start_us": 10.0, "end_us": 10.01, "event": "PORT_RUNNING_0"},
        {"start_us": 10.01, "end_us": 20.0, "event": "PORT_IDLE_0"},
    ]
    # Without priority: idle dominates the single bin (the old, buggy behaviour).
    assert tg.rasterize_lane(events, 0.0, 20.0, 1) == \
        [(0.0, 20.0, tg.EVENT_COLORS["PORT_IDLE_0"])]
    # With the port priority: RUNNING present -> RUNNING wins the bin.
    assert tg.rasterize_lane(events, 0.0, 20.0, 1, tg.PORT_RASTER_PRIORITY) == \
        [(0.0, 20.0, tg.EVENT_COLORS["PORT_RUNNING_0"])]


def test_rasterize_priority_running_beats_stalled_in_bin():
    # RUNNING outranks STALLED by presence even when STALLED occupies more time.
    events = [
        {"start_us": 0.0, "end_us": 9.99, "event": "PORT_STALLED_0"},
        {"start_us": 9.99, "end_us": 10.0, "event": "PORT_RUNNING_0"},
    ]
    assert tg.rasterize_lane(events, 0.0, 10.0, 1, tg.PORT_RASTER_PRIORITY) == \
        [(0.0, 10.0, tg.EVENT_COLORS["PORT_RUNNING_0"])]


def test_rasterize_priority_stalled_beats_idle_in_bin():
    # No RUNNING: STALLED still surfaces over the longer IDLE background.
    events = [
        {"start_us": 0.0, "end_us": 9.99, "event": "PORT_IDLE_0"},
        {"start_us": 9.99, "end_us": 10.0, "event": "PORT_STALLED_0"},
    ]
    assert tg.rasterize_lane(events, 0.0, 10.0, 1, tg.PORT_RASTER_PRIORITY) == \
        [(0.0, 10.0, tg.EVENT_COLORS["PORT_STALLED_0"])]


def test_rasterize_priority_pure_idle_bin_stays_idle():
    # A bin with no RUNNING/STALLED keeps IDLE: priority only reorders mixed bins.
    events = [{"start_us": 0.0, "end_us": 10.0, "event": "PORT_IDLE_0"}]
    assert tg.rasterize_lane(events, 0.0, 10.0, 1, tg.PORT_RASTER_PRIORITY) == \
        [(0.0, 10.0, tg.EVENT_COLORS["PORT_IDLE_0"])]


def test_rasterize_priority_shows_short_active_over_dominant_stall():
    # The reported bug: a core lane's brief pure-ACTIVE burst (0.01us, no stall)
    # inside a lock-stall-dominated bin (19.99us) vanishes under pure duration
    # dominance. The core presence-priority surfaces the running.
    events = [
        {"start_us": 0.0, "end_us": 10.0, "event": "ACTIVE|LOCK_STALL"},
        {"start_us": 10.0, "end_us": 10.01, "event": "ACTIVE"},
        {"start_us": 10.01, "end_us": 20.0, "event": "ACTIVE|LOCK_STALL"},
    ]
    # Without priority: the stall colour dominates the single bin (old behaviour).
    assert tg.rasterize_lane(events, 0.0, 20.0, 1) == \
        [(0.0, 20.0, tg.EVENT_COLORS["LOCK_STALL"])]
    # With the core priority: a genuine ACTIVE (no stall) burst -> green wins.
    assert tg.rasterize_lane(events, 0.0, 20.0, 1, tg.CORE_RASTER_PRIORITY) == \
        [(0.0, 20.0, tg.EVENT_COLORS["ACTIVE"])]


def test_rasterize_priority_stall_only_bin_stays_stall():
    # A bin with only ACTIVE|LOCK_STALL (active bit set but blocked) has no pure
    # ACTIVE, so it keeps the stall colour -- running is never falsely shown.
    events = [{"start_us": 0.0, "end_us": 10.0, "event": "ACTIVE|LOCK_STALL"}]
    assert tg.rasterize_lane(events, 0.0, 10.0, 1, tg.CORE_RASTER_PRIORITY) == \
        [(0.0, 10.0, tg.EVENT_COLORS["LOCK_STALL"])]


def test_rasterize_surface_colors_max_duration_among_present():
    # surface_colors: colours that win over the background by PRESENCE, but among
    # themselves the one with the most time in the bin wins (used by the port lane
    # so a pixel shared by stall+run shows whichever occupied it longer, not a
    # fixed running-over-stalled order). STALL (0.3) beats RUN (0.2) over IDLE.
    events = [
        {"start_us": 0.0, "end_us": 9.5, "event": "PORT_IDLE_0"},
        {"start_us": 9.5, "end_us": 9.8, "event": "PORT_STALLED_0"},
        {"start_us": 9.8, "end_us": 10.0, "event": "PORT_RUNNING_0"},
    ]
    surface = {tg.EVENT_COLORS["PORT_RUNNING_0"], tg.EVENT_COLORS["PORT_STALLED_0"]}
    assert tg.rasterize_lane(events, 0.0, 10.0, 1, surface_colors=surface) == \
        [(0.0, 10.0, tg.EVENT_COLORS["PORT_STALLED_0"])]


def test_rasterize_surface_colors_running_wins_when_longer():
    # Same rule, RUN (0.4) now occupies more of the bin than STALL (0.1) -> RUN.
    events = [
        {"start_us": 0.0, "end_us": 9.5, "event": "PORT_IDLE_0"},
        {"start_us": 9.5, "end_us": 9.6, "event": "PORT_STALLED_0"},
        {"start_us": 9.6, "end_us": 10.0, "event": "PORT_RUNNING_0"},
    ]
    surface = {tg.EVENT_COLORS["PORT_RUNNING_0"], tg.EVENT_COLORS["PORT_STALLED_0"]}
    assert tg.rasterize_lane(events, 0.0, 10.0, 1, surface_colors=surface) == \
        [(0.0, 10.0, tg.EVENT_COLORS["PORT_RUNNING_0"])]


def test_rasterize_surface_colors_pure_idle_bin_stays_idle():
    # A bin with no surface colour keeps its plain max-duration winner (IDLE).
    events = [{"start_us": 0.0, "end_us": 10.0, "event": "PORT_IDLE_0"}]
    surface = {tg.EVENT_COLORS["PORT_RUNNING_0"], tg.EVENT_COLORS["PORT_STALLED_0"]}
    assert tg.rasterize_lane(events, 0.0, 10.0, 1, surface_colors=surface) == \
        [(0.0, 10.0, tg.EVENT_COLORS["PORT_IDLE_0"])]


def test_draw_lanes_surface_short_running_bursts():
    # End-to-end: with the priority wired in _draw, a tiny running burst paints
    # its running colour on both a core lane (pure ACTIVE amid dominant stall)
    # and a port lane (RUNNING amid dominant idle). A mem lane keeps duration
    # dominance (no priority), so its brief token does not override the majority.
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 core", "role": "core", "ident": "", "events": [
                {"event": "ACTIVE|LOCK_STALL", "start_us": 0.0, "end_us": 9.99, "detail": ""},
                {"event": "ACTIVE", "start_us": 9.99, "end_us": 10.0, "detail": ""},
            ]},
            {"name": "tile 0,3 south slave 0", "role": "port", "ident": "south slave 0",
             "events": [
                {"event": "PORT_IDLE_0", "start_us": 0.0, "end_us": 9.99, "detail": ""},
                {"event": "PORT_RUNNING_0", "start_us": 9.99, "end_us": 10.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=1)
    # Collect every broken_barh facecolor on the axis.
    bar_colors = set()
    for coll in ax.collections:
        for c in coll.get_facecolors():
            bar_colors.add(mcolors.to_hex(c))
    # Both short running bursts surface despite dominant stall/idle backgrounds.
    assert tg.EVENT_COLORS["ACTIVE"] in bar_colors, bar_colors
    assert tg.EVENT_COLORS["PORT_RUNNING_0"] in bar_colors, bar_colors
    plt.close(fig)


def test_rasterize_lane_category_filters_tokens():
    # The mem lane split rasterises each half using ONLY the tokens of one canon
    # category. A DMA_STALL_LOCK band ignores the STREAM_STALL token in the same
    # '|'-joined interval, and vice versa, so each band reflects its own signal.
    events = [
        {"start_us": 0.0, "end_us": 10.0,
         "event": "DMA_MM2S_0_STALLED_LOCK_MEM|DMA_MM2S_0_STREAM_BACKPRESSURE_MEM"},
        {"start_us": 10.0, "end_us": 20.0, "event": "DMA_MM2S_0_STALLED_LOCK_MEM"},
    ]
    # Lock band: both intervals carry the lock token -> one merged orange span.
    assert tg.rasterize_lane_category(events, "DMA_STALL_LOCK", 0.0, 20.0, 2) == \
        [(0.0, 20.0, tg.EVENT_COLORS["DMA_STALL_LOCK"])]
    # Stream band: only the first interval carries the stream token -> [0,10] red.
    assert tg.rasterize_lane_category(events, "STREAM_STALL", 0.0, 20.0, 2) == \
        [(0.0, 10.0, tg.EVENT_COLORS["STREAM_STALL"])]


def test_widen_segments_floors_thin_to_min_width():
    # Sub-pixel-thin stall segments vanish under anti-aliasing. widen_segments
    # floors each to a minimum width (centred) so a brief burst stays visible;
    # already-wide segments are untouched.
    segs = [(10.0, 10.3, "#d62728"), (20.0, 40.0, "#ff7f0e")]
    out = tg.widen_segments(segs, 2.0)
    # Thin one grows to 2.0us centred on its midpoint (10.15).
    assert out[0] == (9.15, 11.15, "#d62728"), out
    # Wide one is unchanged.
    assert out[1] == (20.0, 40.0, "#ff7f0e"), out


def test_widen_segments_empty_and_exact():
    assert tg.widen_segments([], 2.0) == []
    # A segment exactly at the floor is left as-is.
    assert tg.widen_segments([(0.0, 2.0, "#000000")], 2.0) == [(0.0, 2.0, "#000000")]


def _band_yranges_by_color(ax):
    """Map colour-hex -> list of (ymin, ymax) for every broken_barh band on ax."""
    import matplotlib.colors as mcolors
    out = {}
    for coll in ax.collections:
        cols = coll.get_facecolors()
        if len(cols) == 0:
            continue
        hexc = mcolors.to_hex(cols[0])
        for path in coll.get_paths():
            ys = [v[1] for v in path.vertices]
            out.setdefault(hexc, []).append((min(ys), max(ys)))
    return out


def test_draw_mem_lane_lock_and_stream_stack_vertically():
    # A mem lane with concurrent lock stall + stream stall (no mem-bp) splits into
    # two slices: the lock band (orange) fills the TOP half and the stream band
    # (red) fills the BOTTOM half, so neither hides the other. Both are held
    # bands spanning the event's duration (no per-event ticks).
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 dma mm2s 0", "role": "mem", "ident": "mm2s 0", "events": [
                {"event":
                 "DMA_MM2S_0_STALLED_LOCK_MEM|DMA_MM2S_0_STREAM_BACKPRESSURE_MEM",
                 "start_us": 0.0, "end_us": 10.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    bands = _band_yranges_by_color(ax)
    # Row 0 spans y in [-0.4, 0.4] (row_h=0.8); 2-way split at y=0.0.
    lock = bands[tg.EVENT_COLORS["DMA_STALL_LOCK"]]
    assert lock, bands
    lymin = min(a for a, _ in lock)
    lymax = max(b for _, b in lock)
    assert lymax <= 0.0 + 1e-6, ("lock in top half", lymax)
    assert abs((lymax - lymin) - 0.4) <= 1e-6, ("lock ~half height", lymin, lymax)
    # Stream stall is a held BAND in the BOTTOM half (complementary slice), not a
    # tick; it fills ~half the row height and does not overlap the lock band.
    stream = bands.get(tg.EVENT_COLORS["STREAM_STALL"], [])
    assert stream, ("stream band missing", list(bands))
    symin = min(a for a, _ in stream)
    symax = max(b for _, b in stream)
    assert symin >= 0.0 - 1e-6, ("stream in bottom half", symin)
    assert abs((symax - symin) - 0.4) <= 1e-6, ("stream ~half height", symin, symax)
    assert lymax <= symin + 1e-6, ("no overlap", lymax, symin)
    plt.close(fig)


def test_draw_port_lane_per_pixel_duration_stall_and_run_full_height():
    # The port lane renders one row with per-pixel duration among stall/run: a
    # pixel where the stall lasted longer shows pink (full height), a pixel where
    # the run lasted longer shows teal (full height), idle elsewhere. No overlay
    # stripe -- each state is a full-height segment at its own time (sequential).
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 east master 1", "role": "port", "ident": "east master 1",
             "events": [
                {"event": "PORT_IDLE_0", "start_us": 0.0, "end_us": 50.0, "detail": ""},
                # bin 50 (50-51us): stall (0.4) dominates a brief run (0.05) -> pink
                {"event": "PORT_STALLED_0", "start_us": 50.0, "end_us": 50.4, "detail": ""},
                {"event": "PORT_RUNNING_0", "start_us": 50.4, "end_us": 50.45, "detail": ""},
                {"event": "PORT_IDLE_0", "start_us": 50.45, "end_us": 60.0, "detail": ""},
                # bin 60 (60-61us): running dominates -> teal
                {"event": "PORT_RUNNING_0", "start_us": 60.0, "end_us": 60.5, "detail": ""},
                {"event": "PORT_IDLE_0", "start_us": 60.5, "end_us": 100.0, "detail": ""},
             ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    bands = _band_yranges_by_color(ax)
    stalled = bands.get(tg.EVENT_COLORS["PORT_STALLED_0"], [])
    running = bands.get(tg.EVENT_COLORS["PORT_RUNNING_0"], [])
    assert stalled, ("stall mark missing", list(bands))
    assert running, ("run mark missing", list(bands))
    # Both are full-height segments (no inset stripe overlay).
    for ymin, ymax in stalled + running:
        assert ymin <= -0.4 + 1e-6 and ymax >= 0.4 - 1e-6, ("full height", ymin, ymax)
    plt.close(fig)


def test_draw_mem_lane_single_category_fills_full_height():
    # When only ONE stall category is present (e.g. lock stall, no stream
    # back-pressure) the mem lane should render that band at FULL row height
    # instead of a half-height band with an empty other half.
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 dma mm2s 0", "role": "mem", "ident": "mm2s 0", "events": [
                {"event": "DMA_MM2S_0_STALLED_LOCK_MEM",
                 "start_us": 0.0, "end_us": 10.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    bands = _band_yranges_by_color(ax)
    lock = bands[tg.EVENT_COLORS["DMA_STALL_LOCK"]]
    stream = bands.get(tg.EVENT_COLORS["STREAM_STALL"], [])
    assert lock, bands
    assert not stream, ("no stream band expected", bands)
    # Row 0 spans y in [-0.4, 0.4] (row_h=0.8); the lone lock band must fill it.
    ymin = min(y0 for y0, _ in lock)
    ymax = max(y1 for _, y1 in lock)
    assert ymin <= -0.4 + 1e-6, ("lock band top", ymin)
    assert ymax >= 0.4 - 1e-6, ("lock band bottom", ymax)
    plt.close(fig)


def test_draw_mem_lane_lock_and_membp_split_vertically():
    # Two concurrent HELD stall states -- lock stall AND memory back-pressure --
    # must both be drawn, split vertically: lock (orange) in the top half, mem
    # back-pressure (purple) in the bottom half, each ~half row height and NOT
    # overlapping. Neither should hide the other.
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 dma s2mm 0", "role": "mem", "ident": "s2mm 0", "events": [
                {"event":
                 "DMA_S2MM_0_STALLED_LOCK_MEM|DMA_S2MM_0_MEMORY_BACKPRESSURE_MEM",
                 "start_us": 0.0, "end_us": 10.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    bands = _band_yranges_by_color(ax)
    # Row 0 spans y in [-0.4, 0.4] (row_h=0.8); the split is at y=0.0.
    lock = bands.get(tg.EVENT_COLORS["DMA_STALL_LOCK"], [])
    membp = bands.get(tg.EVENT_COLORS["MEM_BP"], [])
    assert lock, ("lock band missing", list(bands))
    assert membp, ("mem-bp band missing", list(bands))
    l_ymin = min(a for a, _ in lock)
    l_ymax = max(b for _, b in lock)
    m_ymin = min(a for a, _ in membp)
    m_ymax = max(b for _, b in membp)
    # Lock occupies the TOP half only (y in [-0.4, 0.0]); axis is inverted so the
    # smaller y is visually higher.
    assert l_ymax <= 0.0 + 1e-6, ("lock in top half", l_ymax)
    assert abs((l_ymax - l_ymin) - 0.4) <= 1e-6, ("lock ~half height", l_ymin, l_ymax)
    # Mem back-pressure occupies the complementary BOTTOM half (y in [0.0, 0.4]).
    assert m_ymin >= 0.0 - 1e-6, ("mem-bp in bottom half", m_ymin)
    assert abs((m_ymax - m_ymin) - 0.4) <= 1e-6, ("mem-bp ~half height", m_ymin, m_ymax)
    # The two bands do not overlap.
    assert l_ymax <= m_ymin + 1e-6, ("no overlap", l_ymax, m_ymin)
    plt.close(fig)


def test_draw_mem_lane_three_states_stack_lock_membp_stream():
    # When lock stall + mem back-pressure + stream stall all coexist on a BD, the
    # lane splits into THREE stacked slices (top->bottom: lock, mem-bp, stream) so
    # all three are visible at once: orange lock band (top third), purple mem-bp
    # band (middle third), red stream band (bottom third). None hides another --
    # all three are held bands (no per-event ticks).
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 dma s2mm 0", "role": "mem", "ident": "s2mm 0", "events": [
                {"event":
                 "DMA_S2MM_0_STALLED_LOCK_MEM|DMA_S2MM_0_MEMORY_BACKPRESSURE_MEM",
                 "start_us": 0.0, "end_us": 50.0, "detail": ""},
                {"event": "DMA_S2MM_0_STREAM_STARVATION_MEM",
                 "start_us": 0.0, "end_us": 50.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    # Row 0 spans y in [-0.4, 0.4] (row_h=0.8); thirds at -0.1333 and 0.1333.
    third = 0.8 / 3.0
    top_lo, top_hi = -0.4, -0.4 + third
    mid_lo, mid_hi = -0.4 + third, -0.4 + 2 * third
    bot_lo, bot_hi = -0.4 + 2 * third, 0.4
    bands = _band_yranges_by_color(ax)
    lock = bands.get(tg.EVENT_COLORS["DMA_STALL_LOCK"], [])
    membp = bands.get(tg.EVENT_COLORS["MEM_BP"], [])
    stream = bands.get(tg.EVENT_COLORS["STREAM_STALL"], [])
    assert lock, ("lock band missing", list(bands))
    assert membp, ("mem-bp band missing", list(bands))
    assert stream, ("stream band missing", list(bands))
    # Lock occupies the TOP third.
    assert min(a for a, _ in lock) <= top_lo + 1e-6
    assert max(b for _, b in lock) <= top_hi + 1e-6, ("lock in top third", lock)
    # Mem-bp occupies the MIDDLE third.
    assert min(a for a, _ in membp) >= mid_lo - 1e-6, ("mem-bp above middle", membp)
    assert max(b for _, b in membp) <= mid_hi + 1e-6, ("mem-bp below middle", membp)
    # Stream band occupies the BOTTOM third -- clear of the mem-bp band.
    assert min(a for a, _ in stream) >= bot_lo - 1e-6, ("stream in bottom third", stream)
    assert min(a for a, _ in stream) >= mid_hi - 1e-6, ("stream clear of mem-bp", stream)
    plt.close(fig)


def test_draw_mem_lane_lone_stream_stall_fills_full_height_band():
    # A stream stall with a real (multi-us) duration and no concurrent lock/mem-bp
    # must render as a full-height RED BAND spanning its whole duration -- not
    # merely a point tick at its start. Regression: long
    # DMA_*_STREAM_STARVATION_MEM regions (tens of us) were drawn as a single tick
    # at their start, leaving the rest of the duration blank white ("illustration
    # not correct").
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 dma s2mm 0", "role": "mem", "ident": "s2mm 0", "events": [
                {"event": "DMA_S2MM_0_STREAM_STARVATION_MEM",
                 "start_us": 0.0, "end_us": 30.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    bands = _band_yranges_by_color(ax)
    stream = bands.get(tg.EVENT_COLORS["STREAM_STALL"], [])
    assert stream, ("stream band missing", list(bands))
    # Full row height (lone state, no split): y in [-0.4, 0.4].
    assert min(a for a, _ in stream) <= -0.4 + 1e-6, ("stream band top", stream)
    assert max(b for _, b in stream) >= 0.4 - 1e-6, ("stream band bottom", stream)
    # The band spans the event's duration (to ~30us), not a single bin at x=0.
    xmax = 0.0
    for coll in ax.collections:
        cols = coll.get_facecolors()
        if len(cols) and mcolors.to_hex(cols[0]) == tg.EVENT_COLORS["STREAM_STALL"]:
            for p in coll.get_paths():
                xmax = max(xmax, max(v[0] for v in p.vertices))
    assert xmax >= 29.0, ("stream band must span its duration", xmax)
    plt.close(fig)


def test_draw_mem_lane_alternating_stream_and_lockmembp_regions():
    # Real-world s2mm pattern: a long stream-starvation region, then a long
    # lock+mem-bp region, then another long stream region -- each disjoint in time.
    # The stream regions must render as full-height red bands over their spans (not
    # ticks), and the lock+mem-bp region as an orange(top)/purple(bottom) split;
    # the boundaries must line up (no white gap swallowing a stream region).
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 dma s2mm 0", "role": "mem", "ident": "s2mm 0", "events": [
                {"event": "DMA_S2MM_0_STREAM_STARVATION_MEM",
                 "start_us": 0.0, "end_us": 30.0, "detail": ""},
                {"event":
                 "DMA_S2MM_0_STALLED_LOCK_MEM|DMA_S2MM_0_MEMORY_BACKPRESSURE_MEM",
                 "start_us": 31.0, "end_us": 80.0, "detail": ""},
                {"event": "DMA_S2MM_0_STREAM_STARVATION_MEM",
                 "start_us": 90.0, "end_us": 120.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=240)

    # Collect stream (red) band x-spans and their y-heights.
    stream_c = tg.EVENT_COLORS["STREAM_STALL"]
    lock_c = tg.EVENT_COLORS["DMA_STALL_LOCK"]
    membp_c = tg.EVENT_COLORS["MEM_BP"]
    spans = {stream_c: [], lock_c: [], membp_c: []}
    for coll in ax.collections:
        cols = coll.get_facecolors()
        if not len(cols):
            continue
        hexc = mcolors.to_hex(cols[0])
        if hexc not in spans:
            continue
        for p in coll.get_paths():
            xs = [v[0] for v in p.vertices]
            ys = [v[1] for v in p.vertices]
            spans[hexc].append((min(xs), max(xs), min(ys), max(ys)))

    # Two separate stream bands, one over ~[0,30], one over ~[90,120], both full
    # height (lone state -> no vertical split).
    sbands = spans[stream_c]
    assert sbands, "stream bands missing"
    covers_first = any(x0 <= 2.0 and x1 >= 28.0 for x0, x1, _, _ in sbands)
    covers_third = any(x0 <= 92.0 and x1 >= 118.0 for x0, x1, _, _ in sbands)
    assert covers_first, ("first stream region not a band", sbands)
    assert covers_third, ("third stream region not a band", sbands)
    for _, _, ylo, yhi in sbands:
        assert ylo <= -0.4 + 1e-6 and yhi >= 0.4 - 1e-6, ("stream full height", ylo, yhi)

    # Lock+mem-bp region ~[31,80] splits: lock top half, mem-bp bottom half.
    assert spans[lock_c], "lock band missing"
    assert spans[membp_c], "mem-bp band missing"
    lock_ymax = max(yhi for _, _, _, yhi in spans[lock_c])
    membp_ymin = min(ylo for _, _, ylo, _ in spans[membp_c])
    assert lock_ymax <= 0.0 + 1e-6, ("lock in top half", lock_ymax)
    assert membp_ymin >= 0.0 - 1e-6, ("mem-bp in bottom half", membp_ymin)
    plt.close(fig)


def test_lane_helpers_split_host_and_tiles():
    model = tg._synthetic_model()
    assert tg.host_lane(model)["name"] == "host"
    names = [l["name"] for l in tg.tile_lanes(model)]
    assert names == ["tile 4,4", "tile 5,5"]
    # host lane is excluded from tile lanes
    assert all(n != "host" for n in names)


def test_render_writes_nonempty_png():
    model = tg._synthetic_model()
    with tempfile.TemporaryDirectory() as d:
        png = os.path.join(d, "out.png")
        out = tg.render(model, save=png, want_window=False)
        assert out == png
        assert os.path.getsize(png) > 0


def test_render_real_correlated_model():
    # Build a real model from a [TIMESYNC] block via host_aie_timeline, render it.
    import host_aie_timeline as hat
    block = "\n".join([
        "[TIMESYNC] cps=1000000",
        "[TIMESYNC] anchor0 host=1000",
        "[TIMESYNC] anchor0 tile=4,4 aie=0",
        "[TIMESYNC] anchor1 host=2000",
        "[TIMESYNC] anchor1 tile=4,4 aie=1000",
        "[TIMESYNC] hostevt iter=0 phase=iter_start host=1000",
        "[TIMESYNC] hostevt iter=0 phase=run host=1500",
        "[TIMESYNC] hostevt iter=0 phase=wait_done host=1800",
        "[TIMESYNC] trace tile=4,4 words=16",
        "[TIMESYNC] trace tile=4,4 510 -- 799  ACTIVE  (290 cyc)",
    ])
    model = hat.correlate(hat.parse_timesync(block))
    with tempfile.TemporaryDirectory() as d:
        png = os.path.join(d, "real.png")
        assert tg.render(model, save=png, want_window=False) == png
        assert os.path.getsize(png) > 0


# --------------------------------------------------------------------------
# Pulse / edge events (DMA_FINISH, LOCK_*): single-cycle markers a span
# rectangle cannot show. They must be drawn as VERTICAL LINE markers.
# --------------------------------------------------------------------------
def test_pulse_tokens_extraction():
    # Combined with a stall level: only the pulse token(s) are returned.
    assert tg.pulse_tokens("DMA_FINISH|DMA_STALL_LOCK") == ["DMA_FINISH"]
    # A pure-pulse combo returns both, in EVENT_COLORS order (LOCK_GRP before REL).
    assert tg.pulse_tokens("LOCK_REL|LOCK_GRP") == ["LOCK_GRP", "LOCK_REL"]
    # A single pulse alone.
    assert tg.pulse_tokens("DMA_FINISH") == ["DMA_FINISH"]
    # No pulse: level/state events only.
    assert tg.pulse_tokens("ACTIVE|LOCK_STALL") == []
    assert tg.pulse_tokens("DMA_STALL_LOCK") == []


def _vertical_lines_by_color(ax):
    """Map colour-hex -> list of x for every vertical (x0==x1) Line2D on ax."""
    import matplotlib.colors as mcolors
    out = {}
    for ln in ax.get_lines():
        xd = ln.get_xdata()
        if len(xd) == 2 and xd[0] == xd[1]:
            hexc = mcolors.to_hex(ln.get_color())
            out.setdefault(hexc, []).append(xd[0])
    return out


def test_pulse_events_drawn_as_vertical_lines():
    import matplotlib.pyplot as plt
    # A mem-dma lane: a lone DMA_FINISH pulse, and one combined with a stall.
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 mem dma", "events": [
                {"event": "DMA_STALL_LOCK", "start_us": 0.0, "end_us": 50.0, "detail": ""},
                {"event": "DMA_FINISH", "start_us": 20.0, "end_us": 20.001, "detail": ""},
                {"event": "DMA_FINISH|DMA_STALL_LOCK", "start_us": 40.0, "end_us": 40.001, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    vlines = _vertical_lines_by_color(ax)
    xs = vlines.get(tg.EVENT_COLORS["DMA_FINISH"], [])
    # Both DMA_FINISH pulses (lone + combined-with-stall) get a blue vline.
    assert 20.0 in xs and 40.0 in xs
    plt.close(fig)


def test_pulse_stacks_multiple_tokens():
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 mem dma", "events": [
                {"event": "LOCK_GRP|LOCK_REL", "start_us": 10.0, "end_us": 10.001, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    vlines = _vertical_lines_by_color(ax)
    # Both pulse tokens draw their own coloured vertical marker at x=10.
    assert 10.0 in vlines.get(tg.EVENT_COLORS["LOCK_GRP"], [])
    assert 10.0 in vlines.get(tg.EVENT_COLORS["LOCK_REL"], [])
    plt.close(fig)


def _point_markers(ax):
    """List (x, marker, color-hex, zorder) for every point-marker Line2D.

    A point marker is a single-point plot (len(xdata)==1) with linestyle 'none'.
    """
    import matplotlib.colors as mcolors
    out = []
    for ln in ax.get_lines():
        xd = ln.get_xdata()
        if len(xd) == 1 and ln.get_linestyle() in ("none", "None", ""):
            out.append((xd[0], ln.get_marker(),
                        mcolors.to_hex(ln.get_color()), ln.get_zorder()))
    return out


def test_dma_finish_marker_renders_above_stream_ticks():
    import matplotlib.pyplot as plt
    # A mem lane where a DMA_FINISH coincides with a stream back-pressure event:
    # the opaque red tick (zorder 8) would paint over the finish line, so a
    # blue "^" glyph must be drawn ABOVE the tick (zorder >= 9) to stay visible.
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 mem dma", "events": [
                {"event": "DMA_STALL_LOCK", "start_us": 0.0, "end_us": 50.0, "detail": ""},
                {"event": "STREAM_STALL", "start_us": 30.0, "end_us": 30.001, "detail": ""},
                {"event": "DMA_FINISH|STREAM_STALL", "start_us": 30.0, "end_us": 30.001, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    marks = _point_markers(ax)
    finish = [m for m in marks
              if m[1] == tg.PULSE_MARKERS["DMA_FINISH"]
              and m[2] == tg.EVENT_COLORS["DMA_FINISH"]]
    assert finish, ("no DMA_FINISH ^ marker drawn", marks)
    # The finish glyph must sit above the stream-stall tick (zorder 8).
    assert all(z >= 9 for (_, _, _, z) in finish), finish
    plt.close(fig)


# --------------------------------------------------------------------------
# Enriched labels: a lane's role/ident turns a raw slot into a port/DMA-aware
# phrase for the legend and on-plot text.
# --------------------------------------------------------------------------
def test_token_label_port_states():
    assert tg.token_label("PORT_RUNNING_0", "port", "south slave 0") == \
        "south slave port 0 running"
    assert tg.token_label("PORT_STALLED_0", "port", "south slave 0") == \
        "south slave port 0 stalled"
    assert tg.token_label("PORT_IDLE_0", "port", "south slave 0") == \
        "south slave port 0 idle"
    # Missing ident -> generic port phrasing (still direction-agnostic).
    assert tg.token_label("PORT_RUNNING_0", "port", "") == "port running"


def test_token_label_mem_dma():
    # STREAM_STALL reads as S2MM starvation, per the domain term.
    assert tg.token_label("STREAM_STALL", "mem", "s2mm 0") == \
        "dma s2mm 0 stream starving"
    assert tg.token_label("STREAM_STALL", "mem", "mm2s 1") == \
        "dma mm2s 1 stream starving"
    assert tg.token_label("DMA_START", "mem", "s2mm 0") == "dma s2mm 0 start"
    assert tg.token_label("DMA_FINISH", "mem", "mm2s 0") == "dma mm2s 0 finished"


def test_token_label_core_fallback():
    # Core lane (no role/ident): the plain friendly names.
    assert tg.token_label("ACTIVE", "core", "") == "AIE running"
    assert tg.token_label("STREAM_STALL", "core", "") == "stream stall"
    # Unknown token round-trips.
    assert tg.token_label("EVENT5", "core", "") == "EVENT5"


def test_event_label_uses_dominant_token():
    port = {"role": "port", "ident": "south slave 0"}
    mem = {"role": "mem", "ident": "s2mm 0"}
    # Dominant-token selection mirrors event_color: a stall wins over ACTIVE.
    assert tg.event_label("ACTIVE|PORT_RUNNING_0", {"role": "core", "ident": ""}) \
        == "AIE running"  # ACTIVE stays on the core lane's split
    assert tg.event_label("PORT_RUNNING_0", port) == "south slave port 0 running"
    assert tg.event_label("STREAM_STALL", mem) == "dma s2mm 0 stream starving"
    # No lane context -> friendly fallback, never crashes.
    assert tg.event_label("ACTIVE", None) == "AIE running"


def test_legend_has_enriched_labels():
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 core", "role": "core", "ident": "", "events": [
                {"event": "ACTIVE", "start_us": 0.0, "end_us": 10.0, "detail": ""},
            ]},
            {"name": "tile 0,3 south slave 0", "role": "port", "ident": "south slave 0",
             "events": [
                {"event": "PORT_RUNNING_0", "start_us": 0.0, "end_us": 10.0, "detail": ""},
            ]},
            {"name": "tile 0,3 dma s2mm 0", "role": "mem", "ident": "s2mm 0", "events": [
                {"event": "STREAM_STALL", "start_us": 0.0, "end_us": 10.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    tg._add_legend(ax, model)
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "south slave port 0 running" in labels
    assert "dma s2mm 0 stream starving" in labels
    assert "AIE running" in labels
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver-specific mem-DMA event names. The runtime now emits the XAie driver's
# exact per-slot string (e.g. DMA_S2MM_0_STREAM_STARVATION_MEM) instead of a
# generic alias; a regex must canonicalise these so they colour/label correctly
# rather than falling through to grey "other".
# --------------------------------------------------------------------------
def test_canon_token_maps_specific_mem_dma_names():
    # Each KIND collapses to the right EVENT_COLORS category token.
    assert tg.canon_token("DMA_S2MM_0_STREAM_STARVATION_MEM") == "STREAM_STALL"
    assert tg.canon_token("DMA_MM2S_0_STREAM_BACKPRESSURE_MEM") == "STREAM_STALL"
    assert tg.canon_token("DMA_S2MM_1_STALLED_LOCK_MEM") == "DMA_STALL_LOCK"
    assert tg.canon_token("DMA_MM2S_0_STALLED_LOCK_ACQUIRE_MEM") == "DMA_STALL_LOCK"
    assert tg.canon_token("DMA_S2MM_0_MEMORY_BACKPRESSURE_MEM") == "MEM_BP"
    assert tg.canon_token("DMA_MM2S_1_MEMORY_STARVATION_MEM") == "MEM_BP"
    assert tg.canon_token("DMA_S2MM_0_START_TASK_MEM") == "DMA_START"
    assert tg.canon_token("DMA_S2MM_0_FINISHED_BD_MEM") == "DMA_FINISH"
    # Already-generic / core / port tokens pass through unchanged.
    assert tg.canon_token("STREAM_STALL") == "STREAM_STALL"
    assert tg.canon_token("ACTIVE") == "ACTIVE"
    # Uncategorised driver names (no colour) stay raw -> grey "other".
    assert tg.canon_token("DMA_S2MM_0_ERROR_MEM") == "DMA_S2MM_0_ERROR_MEM"
    assert tg.canon_token("DMA_MM2S_0_GO_TO_IDLE_MEM") == "DMA_MM2S_0_GO_TO_IDLE_MEM"
    # A non-DMA unknown token is untouched.
    assert tg.canon_token("EVENT9") == "EVENT9"


def test_event_color_specific_mem_dma_not_other():
    # The reported bug: these specific names rendered grey "other". Now they take
    # their canonical colour (a stall reads as a stall).
    assert tg.event_color("DMA_S2MM_0_STREAM_STARVATION_MEM") == \
        tg.EVENT_COLORS["STREAM_STALL"]
    assert tg.event_color("DMA_MM2S_0_STALLED_LOCK_MEM") == \
        tg.EVENT_COLORS["DMA_STALL_LOCK"]
    assert tg.event_color("DMA_S2MM_1_MEMORY_BACKPRESSURE_MEM") == \
        tg.EVENT_COLORS["MEM_BP"]
    assert tg.event_color("DMA_S2MM_0_START_TASK_MEM") == \
        tg.EVENT_COLORS["DMA_START"]
    # A combined label still prefers the stall over the START pulse.
    assert tg.event_color("DMA_S2MM_0_START_TASK_MEM|DMA_S2MM_0_STREAM_STARVATION_MEM") \
        == tg.EVENT_COLORS["STREAM_STALL"]
    # An uncategorised driver name is still grey.
    assert tg.event_color("DMA_S2MM_0_ERROR_MEM") == tg.OTHER_COLOR


def test_token_label_specific_mem_dma_names():
    # The name carries its own dir/ch, so the label is correct with no ident.
    assert tg.token_label("DMA_S2MM_0_STREAM_STARVATION_MEM", "mem", "") == \
        "dma s2mm 0 stream starving"
    assert tg.token_label("DMA_MM2S_1_STREAM_BACKPRESSURE_MEM", "mem", "") == \
        "dma mm2s 1 stream back-pressure"
    assert tg.token_label("DMA_S2MM_0_STALLED_LOCK_MEM", "mem", "") == \
        "dma s2mm 0 lock stall"
    assert tg.token_label("DMA_S2MM_0_START_TASK_MEM", "mem", "") == \
        "dma s2mm 0 start"
    assert tg.token_label("DMA_S2MM_1_FINISHED_BD_MEM", "mem", "") == \
        "dma s2mm 1 finished"


def test_pulse_tokens_specific_mem_dma_names():
    # A specific START/FINISH still registers as a pulse (vertical marker).
    assert tg.pulse_tokens("DMA_S2MM_0_START_TASK_MEM") == ["DMA_START"]
    assert tg.pulse_tokens("DMA_S2MM_0_FINISHED_BD_MEM") == ["DMA_FINISH"]
    # A stall combined with a start returns only the pulse.
    assert tg.pulse_tokens(
        "DMA_S2MM_0_STREAM_STARVATION_MEM|DMA_S2MM_0_START_TASK_MEM") == ["DMA_START"]


def test_legend_specific_mem_dma_names_enriched():
    import matplotlib.pyplot as plt
    model = {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "tile 0,3 dma s2mm 0", "role": "mem", "ident": "s2mm 0", "events": [
                {"event": "DMA_S2MM_0_STREAM_STARVATION_MEM",
                 "start_us": 0.0, "end_us": 10.0, "detail": ""},
                {"event": "DMA_MM2S_0_STALLED_LOCK_MEM",
                 "start_us": 10.0, "end_us": 20.0, "detail": ""},
            ]},
        ],
    }
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    tg._add_legend(ax, model)
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "dma s2mm 0 stream starving" in labels
    assert "dma mm2s 0 lock stall" in labels
    # These are NOT lumped into a grey "other" bucket.
    assert "other" not in labels
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver-specific LOCK event names. Mem-trace lock slots now emit the exact
# driver string (LOCK_0_REL_MEM, LOCK_SEL0_ACQ_GE_MEM, ...); these previously
# fell through to grey "other". A regex must fold them onto the generic lock
# pulse tokens (LOCK_ACQ/REL/GRP) for colour + pulse marker, and keep the lock
# index for the label.
# --------------------------------------------------------------------------
def test_canon_token_maps_specific_lock_names():
    assert tg.canon_token("LOCK_0_REL_MEM") == "LOCK_REL"
    assert tg.canon_token("LOCK_5_ACQ_MEM") == "LOCK_ACQ"
    assert tg.canon_token("LOCK_SEL0_ACQ_GE_MEM") == "LOCK_ACQ"
    assert tg.canon_token("LOCK_SEL3_ACQ_EQ_MEM") == "LOCK_ACQ"
    assert tg.canon_token("LOCK_SEL0_REL_MEM") == "LOCK_REL"
    assert tg.canon_token("LOCK_SEL7_EQUAL_TO_VALUE_MEM") == "LOCK_GRP"
    assert tg.canon_token("LOCK_ACQUIRE_MEM") == "LOCK_ACQ"
    # Uncategorised lock names (no colour) stay raw -> grey "other".
    assert tg.canon_token("LOCK_ERROR_MEM") == "LOCK_ERROR_MEM"


def test_token_label_specific_lock_names():
    # The name carries its own lock/group index, so the label needs no ident.
    assert tg.token_label("LOCK_0_REL_MEM", "mem", "") == "lock 0 release"
    assert tg.token_label("LOCK_5_ACQ_MEM", "mem", "") == "lock 5 acquire"
    assert tg.token_label("LOCK_SEL0_ACQ_GE_MEM", "mem", "") == "lock sel 0 acquire"
    assert tg.token_label("LOCK_SEL7_EQUAL_TO_VALUE_MEM", "mem", "") == \
        "lock sel 7 value"
    assert tg.token_label("LOCK_ACQUIRE_MEM", "mem", "") == "lock acquire"


def test_event_color_and_pulse_specific_lock_names():
    # Lock events colour by their canonical pulse token, not grey.
    assert tg.event_color("LOCK_0_REL_MEM") == tg.EVENT_COLORS["LOCK_REL"]
    assert tg.event_color("LOCK_SEL0_ACQ_GE_MEM") == tg.EVENT_COLORS["LOCK_ACQ"]
    assert tg.event_color("LOCK_SEL7_EQUAL_TO_VALUE_MEM") == tg.EVENT_COLORS["LOCK_GRP"]
    # ...and still register as single-cycle pulse markers.
    assert tg.pulse_tokens("LOCK_0_REL_MEM") == ["LOCK_REL"]
    assert tg.pulse_tokens("LOCK_SEL0_ACQ_GE_MEM") == ["LOCK_ACQ"]


# --------------------------------------------------------------------------
# Driver-specific PORT event names at any index / module suffix. The runtime may
# emit PORT_IDLE_0, PORT_STALLED_3_CORE, etc. across different applogs; only the
# bare *_0 forms are in EVENT_COLORS, so a regex must fold the rest onto the same
# three port colour slots and generalise the port label.
# --------------------------------------------------------------------------
def test_canon_token_maps_specific_port_names():
    # The bare forms pass through (already in EVENT_COLORS).
    assert tg.canon_token("PORT_IDLE_0") == "PORT_IDLE_0"
    assert tg.canon_token("PORT_RUNNING_0") == "PORT_RUNNING_0"
    # Any index / module suffix collapses to the *_0 colour slot.
    assert tg.canon_token("PORT_IDLE_3") == "PORT_IDLE_0"
    assert tg.canon_token("PORT_RUNNING_2_CORE") == "PORT_RUNNING_0"
    assert tg.canon_token("PORT_STALLED_7_PL") == "PORT_STALLED_0"
    assert tg.canon_token("PORT_IDLE_1_MEM") == "PORT_IDLE_0"
    # TLAST has no colour slot -> stays raw -> grey "other".
    assert tg.canon_token("PORT_TLAST_0_CORE") == "PORT_TLAST_0_CORE"


def test_token_label_generalized_port_names():
    # Generalised names fold into the lane's port phrase via their state.
    assert tg.token_label("PORT_RUNNING_0", "port", "south slave 0") == \
        "south slave port 0 running"
    assert tg.token_label("PORT_IDLE_3", "port", "south slave 0") == \
        "south slave port 0 idle"
    assert tg.token_label("PORT_STALLED_2_CORE", "port", "north master 1") == \
        "north master port 1 stalled"
    # Missing ident -> generic port phrasing.
    assert tg.token_label("PORT_IDLE_5_MEM", "port", "") == "port idle"


def test_event_color_generalized_port_names_not_other():
    assert tg.event_color("PORT_IDLE_3") == tg.EVENT_COLORS["PORT_IDLE_0"]
    assert tg.event_color("PORT_RUNNING_2_CORE") == tg.EVENT_COLORS["PORT_RUNNING_0"]
    assert tg.event_color("PORT_STALLED_7_PL") == tg.EVENT_COLORS["PORT_STALLED_0"]
    # TLAST is still grey.
    assert tg.event_color("PORT_TLAST_0_CORE") == tg.OTHER_COLOR


# --------------------------------------------------------------------------
# No on-plot bar text: labels moved to the legend + hover. _draw must not draw
# any Text artists (host phase words, tile event labels, phase-guide names).
# --------------------------------------------------------------------------
def test_draw_has_no_on_plot_text():
    import matplotlib.pyplot as plt
    model = tg._synthetic_model()  # host lane + two tile lanes with events
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=200)
    # Axis title / labels are set in render(), not _draw(); _draw must add none.
    assert list(ax.texts) == [], [t.get_text() for t in ax.texts]
    plt.close(fig)


def test_parse_host_marker():
    assert tg._parse_host_marker("iter0.run") == (0, "run")
    assert tg._parse_host_marker("iter3.dma_in_done") == (3, "dma_in_done")
    assert tg._parse_host_marker("plainname") == (None, "plainname")


def test_host_intervals_running_and_waiting():
    model = tg._synthetic_model()
    bars = tg.host_intervals(tg.host_lane(model))
    # 7 markers -> 6 gaps.
    assert len(bars) == 6
    # The run->wait_done gap is the idle-waiting bar; all others are running.
    wait = [b for b in bars if b["waiting"]]
    assert len(wait) == 1
    assert wait[0]["phase"] == "run"
    assert wait[0]["start_us"] == 130.0
    assert wait[0]["end_us"] == 700.0
    assert all(not b["waiting"] for b in bars if b["phase"] != "run")


def test_host_intervals_empty_without_host():
    # A model with no host lane yields no host bars and still renders.
    model = tg._synthetic_model()
    model["lanes"] = tg.tile_lanes(model)
    assert tg.host_lane(model) is None
    assert tg.host_intervals(None) == []


def _enriched_host_model():
    """A host lane whose markers carry the host_aie_timeline enrichment
    (colour-by-API + host.cc line), for the colour/annotation/legend tests."""
    return {
        "meta": {}, "anchors": {}, "fit_per_tile": {},
        "lanes": [
            {"name": "host", "role": "host", "ident": "", "events": [
                {"event": "iter-1.launch", "start_us": 0.0, "end_us": 0.0,
                 "detail": "", "phase": "launch", "api": "launch_kernel_group",
                 "color": "#9467bd", "hostcc_line": 267},
                {"event": "iter0.iter_start", "start_us": 100.0, "end_us": 100.0,
                 "detail": "", "phase": "iter_start", "api": "input DMA BD config",
                 "color": "#1f77b4", "hostcc_line": 702},
                {"event": "iter0.dma_start", "start_us": 110.0, "end_us": 110.0,
                 "detail": "", "phase": "dma_start", "api": "startio (issue DMAs)",
                 "color": "#17becf", "hostcc_line": 710},
                {"event": "iter0.wait_done", "start_us": 200.0, "end_us": 200.0,
                 "detail": "", "phase": "wait_done", "api": "iter end",
                 "color": "#2ca02c", "hostcc_line": 928},
            ]},
        ],
    }


def test_host_intervals_carry_api_color_and_line():
    model = _enriched_host_model()
    bars = tg.host_intervals(tg.host_lane(model))
    assert len(bars) == 3  # 4 markers -> 3 gaps
    first = bars[0]
    assert first["phase"] == "launch"
    assert first["color"] == "#9467bd"
    assert first["api"] == "launch_kernel_group"
    assert first["hostcc_line"] == 267
    # Start-to-next-start span.
    assert first["start_us"] == 0.0 and first["end_us"] == 100.0


def test_draw_colors_host_bars_by_api():
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    model = _enriched_host_model()
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    bar_colors = set()
    for coll in ax.collections:
        for c in coll.get_facecolors():
            bar_colors.add(mcolors.to_hex(c))
    # Each host bar takes its LEADING phase's API colour (not the old blue/grey
    # binary). The trailing wait_done marker forms no bar, so its colour is absent.
    assert "#9467bd" in bar_colors      # launch -> iter_start
    assert "#1f77b4" in bar_colors      # iter_start -> dma_start
    assert "#17becf" in bar_colors      # dma_start -> wait_done
    plt.close(fig)


def test_draw_annotates_hostcc_line_when_present():
    import matplotlib.pyplot as plt
    model = _enriched_host_model()
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    texts = [t.get_text() for t in ax.texts]
    # The first-iteration host bars are annotated with their host.cc line.
    assert "host.cc:267" in texts
    assert "host.cc:702" in texts
    plt.close(fig)


def test_legend_has_per_phase_host_entries():
    import matplotlib.pyplot as plt
    model = _enriched_host_model()
    fig, ax = plt.subplots()
    tg._draw(ax, model, nbins=100)
    tg._add_legend(ax, model)
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "launch  (host.cc:267  launch_kernel_group)" in labels
    assert "dma_start  (host.cc:710  startio (issue DMAs))" in labels
    plt.close(fig)


def test_self_test_entry_point():
    assert tg._self_test() == 0


# --------------------------------------------------------------------------
# Standalone runner (works without pytest installed).
# --------------------------------------------------------------------------
def _main():
    fns = [g for n, g in sorted(globals().items())
           if n.startswith("test_") and callable(g)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print("PASS  %s" % fn.__name__)
        except AssertionError as e:
            print("FAIL  %s: %s" % (fn.__name__, e))
            return 1
        except Exception as e:  # noqa: BLE001
            print("ERROR %s: %r" % (fn.__name__, e))
            return 1
    print("\n%d passed" % passed)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
