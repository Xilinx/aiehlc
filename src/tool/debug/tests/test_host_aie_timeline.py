#!/usr/bin/env python3
"""Unit test for host_aie_timeline.py (src/tool/debug/host_aie_timeline.py).

Feeds a synthetic [TIMESYNC] block -- known host/AIE anchors plus the decoded
AIE interval lines the unified dump (__Runtime_aie_trace_profile_dump) emits --
and asserts:
  1. the per-tile linear fit recovers a known cycle->us mapping (exact float),
  2. host phase events land at their expected us,
  3. CSV/JSON row counts and lane ordering are correct,
  4. two tiles with different timer offsets both map correctly.

Pure Python: no board, no compiler. The dump already decoded the raw trace into
absolute-AIE-cycle interval lines, so the tool parses them directly.
"""

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))   # host_aie_timeline

import host_aie_timeline as hat  # noqa: E402


# --------------------------------------------------------------------------
# A synthetic capture. cps=1e6 so 1 host count == 1 us. Slope 1 count/cycle.
#   tile 4,4: a0=0,    a1=1000  -> us(cycle) = cycle
#   tile 5,5: a0=500,  a1=1500  -> us(cycle) = cycle - 500   (offset by 500)
# Decoded trace intervals (both tiles): ACTIVE@10 (1 cyc), STREAM_STALL@110.
# --------------------------------------------------------------------------
# What parse_timesync should yield for each tile's interval list. Each interval
# is (start_cycle, end_cycle, slot_names, stream); core lines carry no
# "stream=" tag so the parser defaults the stream to "core".
_TRACE_INTERVALS = [(10, 10, ["ACTIVE"], "core"), (110, 110, ["STREAM_STALL"], "core")]


def _trace_lines(tile):
    c, r = tile
    return [
        "[TIMESYNC] trace tile=%d,%d words=16" % (c, r),
        "[TIMESYNC] trace tile=%d,%d 10  ACTIVE" % (c, r),
        "[TIMESYNC] trace tile=%d,%d 110  STREAM_STALL" % (c, r),
    ]


def _block(two_tiles=True):
    lines = [
        "[TIMESYNC] cps=1000000",
        "[TIMESYNC] anchor0 host=1000",
        "[TIMESYNC] anchor0 tile=4,4 aie=0",
    ]
    if two_tiles:
        lines.append("[TIMESYNC] anchor0 tile=5,5 aie=500")
    lines += [
        "[TIMESYNC] anchor1 host=2000",
        "[TIMESYNC] anchor1 tile=4,4 aie=1000",
    ]
    if two_tiles:
        lines.append("[TIMESYNC] anchor1 tile=5,5 aie=1500")
    lines += [
        "[TIMESYNC] hostevt iter=0 phase=iter_start host=1000",
        "[TIMESYNC] hostevt iter=0 phase=run host=1500",
        "[TIMESYNC] hostevt iter=1 phase=dma_out_done host=2000",
    ]
    lines += _trace_lines((4, 4))
    if two_tiles:
        lines += _trace_lines((5, 5))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def test_parse_extracts_all_records():
    ts = hat.parse_timesync(_block())
    assert ts["cps"] == 1000000
    assert ts["anchors"][0]["host"] == 1000
    assert ts["anchors"][1]["host"] == 2000
    assert ts["anchors"][0]["tiles"][(4, 4)] == 0
    assert ts["anchors"][1]["tiles"][(4, 4)] == 1000
    assert ts["anchors"][0]["tiles"][(5, 5)] == 500
    assert len(ts["hostevts"]) == 3
    assert ts["hostevts"][0] == (0, "iter_start", 1000)
    assert (4, 4) in ts["traces"] and (5, 5) in ts["traces"]
    assert ts["traces"][(4, 4)] == _TRACE_INTERVALS
    assert ts["traces"][(5, 5)] == _TRACE_INTERVALS


def test_linear_fit_recovers_cycle_to_us():
    # tile 4,4: identity; tile 5,5: shifted by -500 us.
    fit44 = hat.TileFit((4, 4), 1000000, 1000, 2000, 0, 1000)
    assert fit44.counts_per_cycle == 1.0
    assert fit44.aie_hz == 1000000.0
    assert fit44.cycle_to_us(0) == 0.0
    assert fit44.cycle_to_us(10) == 10.0
    assert fit44.cycle_to_us(1000) == 1000.0

    fit55 = hat.TileFit((5, 5), 1000000, 1000, 2000, 500, 1500)
    assert fit55.cycle_to_us(500) == 0.0
    assert fit55.cycle_to_us(510) == 10.0


def test_nonunit_slope_fit():
    # 2 counts per cycle, cps=2e6 -> counts/us=2 -> us(cycle)=cycle.
    fit = hat.TileFit((4, 4), 2000000, 0, 2000, 0, 1000)
    assert fit.counts_per_cycle == 2.0
    assert fit.cycle_to_us(100) == 100.0
    assert fit.aie_hz == 1000000.0


def test_host_events_land_at_expected_us():
    model = hat.correlate(hat.parse_timesync(_block()))
    host = next(l for l in model["lanes"] if l["name"] == "host")
    us = {e["event"]: e["start_us"] for e in host["events"]}
    assert us["iter0.iter_start"] == 0.0
    assert us["iter0.run"] == 500.0
    assert us["iter1.dma_out_done"] == 1000.0
    # Host markers are zero-width.
    for e in host["events"]:
        assert e["start_us"] == e["end_us"]


def test_tile_intervals_mapped():
    model = hat.correlate(hat.parse_timesync(_block()))
    t44 = next(l for l in model["lanes"] if l["name"] == "tile 4,4 core")
    ev = {e["event"]: e for e in t44["events"]}
    assert ev["ACTIVE"]["start_us"] == 10.0
    assert ev["ACTIVE"]["end_us"] == 11.0        # end_cycle+1 -> 1-cycle width
    assert ev["STREAM_STALL"]["start_us"] == 110.0
    # Offset tile maps the same cycles 500 us earlier.
    t55 = next(l for l in model["lanes"] if l["name"] == "tile 5,5 core")
    ev5 = {e["event"]: e for e in t55["events"]}
    assert ev5["ACTIVE"]["start_us"] == -490.0
    assert ev5["STREAM_STALL"]["start_us"] == -390.0


def test_lane_ordering_host_first_then_tiles_sorted():
    model = hat.correlate(hat.parse_timesync(_block()))
    names = [l["name"] for l in model["lanes"]]
    assert names == ["host", "tile 4,4 core", "tile 5,5 core"]


def test_csv_json_row_counts_and_ordering():
    model = hat.correlate(hat.parse_timesync(_block()))
    with tempfile.TemporaryDirectory() as d:
        csv_path, json_path, nrows = hat.emit(model, d)
        # 3 host events + 2 intervals/tile x 2 tiles = 7 rows.
        assert nrows == 7
        with open(csv_path) as f:
            lines = f.read().splitlines()
        assert lines[0] == "lane,event,start_us,end_us,detail"
        assert len(lines) == 1 + 7
        # Host rows come first, then tile 4,4, then tile 5,5.
        assert lines[1].startswith("host,")
        assert lines[2].startswith("host,")
        assert lines[3].startswith("host,")
        assert lines[4].startswith('"tile 4,4 core"')
        assert lines[6].startswith('"tile 5,5 core"')

        with open(json_path) as f:
            model2 = json.load(f)
        assert model2["meta"]["cps"] == 1000000
        assert model2["meta"]["num_tiles"] == 2
        assert set(model2["fit_per_tile"]) == {"4,4", "5,5"}
        assert model2["fit_per_tile"]["4,4"]["counts_per_cycle"] == 1.0
        assert [l["name"] for l in model2["lanes"]] == ["host", "tile 4,4 core", "tile 5,5 core"]


def test_single_tile_block():
    model = hat.correlate(hat.parse_timesync(_block(two_tiles=False)))
    names = [l["name"] for l in model["lanes"]]
    assert names == ["host", "tile 4,4 core"]
    assert model["meta"]["num_tiles"] == 1


def test_mem_stream_gets_own_lane():
    # A mem-module DMA stream (pkt id 2) is tagged "stream=mem" on the dump line
    # and must land in its own "tile C,R mem dma" lane, separate from the core
    # lane -- even though STREAM_STALL also appears in the core slot table.
    block = _block(two_tiles=False).rstrip("\n") + "\n" + "\n".join([
        "[TIMESYNC] trace tile=4,4 stream=mem 300  DMA_START",
        "[TIMESYNC] trace tile=4,4 stream=mem 305 -- 309  STREAM_STALL",
    ]) + "\n"
    ts = hat.parse_timesync(block)
    # Parser keeps the stream tag on the 4-tuple.
    streams = {iv[3] for iv in ts["traces"][(4, 4)]}
    assert streams == {"core", "mem"}
    model = hat.correlate(ts)
    names = [l["name"] for l in model["lanes"]]
    assert "tile 4,4 core" in names
    assert "tile 4,4 mem dma" in names
    mem = next(l for l in model["lanes"] if l["name"] == "tile 4,4 mem dma")
    mspans = {e["event"]: (e["start_us"], e["end_us"]) for e in mem["events"]}
    assert mspans["DMA_START"] == (300.0, 301.0)
    assert mspans["STREAM_STALL"] == (305.0, 310.0)
    # The core lane must NOT contain the mem-stream STREAM_STALL span.
    core = next(l for l in model["lanes"] if l["name"] == "tile 4,4 core")
    core_stall = [e for e in core["events"] if e["event"] == "STREAM_STALL"]
    assert all(e["start_us"] != 305.0 for e in core_stall)


def test_trace_line_event_value_suffix_stripped():
    # The [TIMESYNC] profile dump appends a "{event value v1:v2:...}" suffix that
    # encodes the raw physical HW event ids. The parser must strip it so the event
    # name stays clean and the interval still maps -- otherwise the whole line is
    # silently dropped.
    block = _block(two_tiles=False).rstrip("\n") + "\n" + "\n".join([
        "[TIMESYNC] trace tile=4,4 stream=mem 300 -- 301  "
        "DMA_MM2S_0_STALLED_LOCK_MEM{event value 20:21}  (1 cyc)",
        "[TIMESYNC] trace tile=4,4 305 -- 309  ACTIVE{event value 28}  (4 cyc)",
    ]) + "\n"
    ts = hat.parse_timesync(block)
    ivs = ts["traces"][(4, 4)]
    # iv[2] is the list of "|"-split event names.
    flat = [n for iv in ivs for n in iv[2]]
    # Both suffixed lines survived parsing (not silently dropped).
    assert "DMA_MM2S_0_STALLED_LOCK_MEM" in flat, flat
    assert "ACTIVE" in flat, flat
    # No residual "{event value ...}" text leaks into the event name.
    assert all("event value" not in n for n in flat), flat
    assert all("{" not in n and "}" not in n for n in flat), flat


def test_mem_dma_config_names_lane_and_metadata():
    # A pkt_id=2 [TRACESTREAMCONFIG] line carries the watched tile-DMA
    # (dir+channel); the mem lane is named after it ("tile C,R dma mm2s 0") and
    # tagged role="mem"/ident="mm2s 0" so the renderer can build DMA-aware labels.
    # pkt_id and dma appear in either order across the two config lines.
    block = _block(two_tiles=False).rstrip("\n") + "\n" + "\n".join([
        "[TRACESTREAMCONFIG] src_tile=(4,4) in_port=TRACE:1 out_port=SOUTH(shared) "
        "pkt_id=2 slot=0 mask=0x1F msel=1 arbiter=1 dma=MM2S:0 slots=DMA_START,STREAM_STALL",
        "[TIMESYNC] trace tile=4,4 stream=mem 300  DMA_START",
        "[TIMESYNC] trace tile=4,4 stream=mem 305 -- 309  STREAM_STALL",
    ]) + "\n"
    ts = hat.parse_timesync(block)
    assert ts["mem_dmas"] == {(4, 4): "mm2s 0"}, ts["mem_dmas"]
    model = hat.correlate(ts)
    mem = next(l for l in model["lanes"] if l["name"] == "tile 4,4 dma mm2s 0")
    assert mem["role"] == "mem" and mem["ident"] == "mm2s 0"
    # The core/port lanes still carry role metadata too.
    core = next(l for l in model["lanes"] if l["name"] == "tile 4,4 core")
    assert core["role"] == "core"


def test_evt_port_map_names_port_lane_east_master():
    # With the routed-port re-target (resmap_lookup_dma_port), a traced mm2s0 on
    # tile (4,4) reports evt_port {east,master,1} instead of the trace-stream
    # ingress. The port lane must be named after that physical port ("tile 4,4
    # east master 1") and carry the PORT_* spans.
    block = _block(two_tiles=False).rstrip("\n") + "\n" + "\n".join([
        '[aie_runtime] core_trace_stream_json: {"src_tile":[4,4],'
        '"evt_port":{"tile":[4,4],"port":"east","intf":"master","num":1},'
        '"slots":["PORT_RUNNING_0"],"hops":[]}',
        "[TIMESYNC] trace tile=4,4 200 -- 204  PORT_RUNNING_0",
    ]) + "\n"
    ts = hat.parse_timesync(block)
    assert ts["evt_ports"] == {(4, 4): "east master 1"}, ts["evt_ports"]
    model = hat.correlate(ts)
    port = next(l for l in model["lanes"] if l["name"] == "tile 4,4 east master 1")
    assert port["role"] == "port" and port["ident"] == "east master 1"
    spans = {e["event"]: (e["start_us"], e["end_us"]) for e in port["events"]}
    assert spans["PORT_RUNNING_0"] == (200.0, 205.0), spans


def test_missing_cps_raises():
    ts = hat.parse_timesync("[TIMESYNC] anchor0 host=1\n[TIMESYNC] anchor1 host=2\n")
    try:
        hat.correlate(ts)
        assert False, "expected ValueError for missing cps"
    except ValueError:
        pass


def test_resolve_hostcc_lines_maps_phase_to_emit_call():
    # The two-line codegen pattern: a phase string literal bound to a var, passed
    # as the 3rd arg of __Runtime_core_trace_event on the following line. The
    # resolver maps each phase to that emit-call's 1-based line number, keeping
    # the FIRST occurrence per phase (loop-body first iteration).
    src = "\n".join([
        "  event v89 = __Runtime_launch_kernel_group(v1, v84);",   # 1
        '  const char* v91 = "launch";',                            # 2
        "  __Runtime_core_trace_event(v1, v90, v91);",              # 3
        "  for (size_t v380 = v7; v380 < v6; v380 += v5) {",        # 4
        '    const char* v381 = "iter_start";',                     # 5
        "    __Runtime_core_trace_event(v1, v380, v381);",          # 6
        '    const char* v385 = "dma_start";',                      # 7
        "    __Runtime_core_trace_event(v1, v380, v385);",          # 8
        # A second (later-iteration-style) dma_start emit must NOT overwrite.
        '    const char* v999 = "dma_start";',                      # 9
        "    __Runtime_core_trace_event(v1, v380, v999);",          # 10
        "  }",                                                      # 11
    ])
    with tempfile.NamedTemporaryFile("w", suffix="host.cc", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        m = hat.resolve_hostcc_lines(path)
    finally:
        os.unlink(path)
    assert m == {"launch": 3, "iter_start": 6, "dma_start": 8}, m


def test_resolve_hostcc_lines_missing_file_is_empty():
    assert hat.resolve_hostcc_lines("/no/such/host.cc") == {}
    assert hat.resolve_hostcc_lines(None) == {}


def test_host_phase_style_known_and_default():
    assert hat.host_phase_style("dma_start") == hat.HOST_PHASE_STYLE["dma_start"]
    # Unknown phase -> neutral colour + its own name as the description.
    assert hat.host_phase_style("mystery") == (hat.HOST_PHASE_DEFAULT_COLOR, "mystery")


def test_correlate_enriches_host_markers_with_api_and_line():
    # With a phase->line map, each zero-width host marker gains phase/api/color and
    # (when mapped) hostcc_line -- but stays zero-width so the model contract holds.
    hostcc = {"iter_start": 6, "run": 8}
    model = hat.correlate(hat.parse_timesync(_block()), hostcc)
    host = next(l for l in model["lanes"] if l["name"] == "host")
    by_phase = {e["phase"]: e for e in host["events"]}
    assert by_phase["iter_start"]["color"] == hat.HOST_PHASE_STYLE["iter_start"][0]
    assert by_phase["iter_start"]["api"] == hat.HOST_PHASE_STYLE["iter_start"][1]
    assert by_phase["iter_start"]["hostcc_line"] == 6
    assert by_phase["run"]["hostcc_line"] == 8
    # dma_out_done had no mapping -> no hostcc_line key, still enriched + zero-width.
    assert "hostcc_line" not in by_phase["dma_out_done"]
    for e in host["events"]:
        assert e["start_us"] == e["end_us"]
        assert "color" in e and "api" in e and "phase" in e


def test_correlate_without_hostcc_still_enriches_color_no_line():
    # No host.cc supplied: markers still get colour/api (so the render is coloured
    # by API), just no line annotation.
    model = hat.correlate(hat.parse_timesync(_block()))
    host = next(l for l in model["lanes"] if l["name"] == "host")
    for e in host["events"]:
        assert "color" in e and "api" in e
        assert "hostcc_line" not in e


def test_self_test_entry_point():
    assert hat._self_test() == 0


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
