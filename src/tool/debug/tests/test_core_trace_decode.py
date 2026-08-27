#!/usr/bin/env python3
"""Unit test for __Runtime_core_trace_decode (src/mlir/runtime/aie_runtime.c).

The C decoder cannot be linked standalone (aie_runtime.c pulls in xaiengine.h),
so this test extracts ONLY the self-contained decoder region from the real
source -- the `s_core_trace_slot_name[]` table plus everything from
`#define XAIE_TRACE_SYNC_CYCLES` down through the trace-profile API (init/attach/
dump) that follows __Runtime_core_trace_decode -- wraps it in a tiny main()
harness (with a minimal AieTraceProfile/XAie_LocType shim), compiles it with
gcc, and runs it. It is
the actual production bytes that are exercised, so a regression in aie_runtime.c
fails this test.

Each case is driven by an INDEPENDENT frame encoder (the inverse of the decoder,
built directly from Figure 4-14) and checked two ways:
  1. against a hand-derived golden timeline, and
  2. against the Python reference model core_trace_decode.CoreTraceDecoder,
     which must produce byte-identical faithful output.
"""

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import core_trace_decode as ctd  # noqa: E402

_RUNTIME_C = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..",
                 "mlir", "runtime", "aie_runtime.c"))

# --------------------------------------------------------------------------
# Extract the real decoder from aie_runtime.c and build a compilable harness.
# --------------------------------------------------------------------------
def _extract_c_decoder(src):
    m = re.search(r"static const char \*const s_core_trace_slot_name\[8\][^;]*;", src)
    assert m, "s_core_trace_slot_name[8] table not found in aie_runtime.c"
    slot_tbl = m.group(0)

    # The demux (__core_trace_table_for_id) also references the mem-module slot
    # table for pkt id 2, so pull it in too or the harness will not link.
    mm = re.search(r"static const char \*const s_mem_trace_slot_name\[8\][^;]*;", src)
    assert mm, "s_mem_trace_slot_name[8] table not found in aie_runtime.c"
    mem_tbl = mm.group(0)

    # The profile dump appends each slot's physical HW event id from two parallel
    # event-value tables. Both are now filled at trace-setup time (device-only),
    # so their declarations are plain uninitialized arrays; the harness never runs
    # setup, so they stay zero and the dump prints "{event value 0:...}".
    ce = re.search(r"static uint16_t s_core_trace_slot_event\[8\][^;]*;", src)
    assert ce, "s_core_trace_slot_event[8] table not found in aie_runtime.c"
    core_evt_tbl = ce.group(0)
    me = re.search(r"static uint16_t s_mem_trace_slot_event\[8\][^;]*;", src)
    assert me, "s_mem_trace_slot_event[8] table not found in aie_runtime.c"
    mem_evt_tbl = me.group(0)

    start = src.index("#define XAIE_TRACE_SYNC_CYCLES")
    # Stop right after the trace-profile dump: the declarative-session code that
    # follows pulls in board headers (xtime_l.h / xiltimer.h) that cannot be
    # compiled standalone. The harness only needs init/attach/.../decode/dump.
    end = src.index("Declarative core-trace session", start)
    end = src.rindex("/*", start, end)  # back up to that comment block's opener
    block = src[start:end].rstrip()
    assert "__Runtime_core_trace_decode" in block, "decoder body not captured"
    return slot_tbl, mem_tbl, core_evt_tbl, mem_evt_tbl, block


# Minimal shim for the types the profile path references. The real defs live in
# aie_runtime.h (which pulls in xaiengine.h and cannot be included standalone);
# only the fields the extracted code touches matter here. Must match the header.
_PROFILE_SHIM = r"""
/* The event-value tables (s_core/s_mem_trace_slot_event) are filled on-device at
 * trace setup via XAie_EventLogicalToPhysicalConv, which the standalone harness
 * never runs -- so they stay zero here and the dump prints "{event value 0:..}".
 * The mem-module dynamic-name table is likewise populated on-device from
 * XAie_EventGetString; NULL here, so the dump falls back to the generic
 * s_mem_trace_slot_name labels. Declared here because its declaration in
 * aie_runtime.c lives outside the extracted decoder region. */
static const char *s_mem_trace_slot_dynname[8];

typedef struct { uint8_t Col, Row; } XAie_LocType;

typedef struct {
    uint8_t  col, row;
    uint32_t mask;
    uint64_t start_cycle, end_cycle;
    const char *const *names;
} AieTraceInterval;

typedef struct {
    uint8_t  col, row;
    uint64_t aie0, aie1;
} AieTraceAnchor;

typedef struct {
    int         iter;
    const char *phase;
    uint64_t    host;
} AieTraceHostEvt;

#define AIE_TRACE_PROFILE_CAP 4096u
#define AIE_TRACE_PROFILE_MIN_PER_STREAM 1024u
#define AIE_TRACE_ANCHOR_CAP    8u
#define AIE_TRACE_EVENT_CAP    64u
typedef struct {
    AieTraceInterval iv[AIE_TRACE_PROFILE_CAP];
    uint32_t count;
    uint32_t dropped;
    uint32_t reserve;
    uint8_t  cur_col, cur_row;
    int      attached;
    uint64_t cps;
    uint64_t host0, host1;
    AieTraceAnchor anchor[AIE_TRACE_ANCHOR_CAP];
    uint32_t nanchor;
    AieTraceHostEvt evt[AIE_TRACE_EVENT_CAP];
    uint32_t nevt;
    uint32_t evt_dropped;
    uint8_t  words_col, words_row;
    uint32_t nwords;
} AieTraceProfile;
"""


# Default path calls decode with a NULL profile so existing transcripts are
# byte-identical. "--profile" builds a profile, attaches tile (4,4), decodes
# into it, then dumps -- exercising the real flush-append + dump bytes.
# "--profile-host" additionally records the host<->AIE time-sync fields
# (set_clock/anchor/event/set_trace_words) so the dump emits the full
# [TIMESYNC] host block ahead of the interval lines.
_HARNESS_MAIN = r"""
int main(int argc, char **argv) {
    static uint32_t buf[8192];
    uint32_t n = 0;
    int prof_mode = 0, host_mode = 0, first = 1;
    if (argc > 1 && strcmp(argv[1], "--profile") == 0) { prof_mode = 1; first = 2; }
    else if (argc > 1 && strcmp(argv[1], "--profile-host") == 0) { prof_mode = 1; host_mode = 1; first = 2; }
    for (int i = first; i < argc && n < 8192; i++)
        buf[n++] = (uint32_t)strtoul(argv[i], NULL, 0);
    if (prof_mode) {
        static AieTraceProfile prof;
        XAie_LocType t; t.Col = 4; t.Row = 4;
        __Runtime_aie_trace_profile_init(&prof);
        __Runtime_aie_trace_profile_attach(&prof, t);
        if (host_mode) {
            /* Known anchors: cps=1000000, host 0..500, tile(4,4) aie 2000..3000.
             * Effective Hz = (3000-2000)/(500-0)*1000000 = 2000000000. */
            __Runtime_aie_trace_profile_set_clock(&prof, 1000000ull);
            __Runtime_aie_trace_profile_anchor(&prof, 0, t, 2000ull, 0ull);
            __Runtime_aie_trace_profile_anchor(&prof, 1, t, 3000ull, 500ull);
            __Runtime_aie_trace_profile_event(&prof, 0, "iter_start", 10ull);
            __Runtime_aie_trace_profile_event(&prof, 0, "run", 20ull);
            __Runtime_aie_trace_profile_set_trace_words(&prof, t, n);
        }
        __Runtime_core_trace_decode(buf, n, &prof);
        __Runtime_aie_trace_profile_dump(&prof);
    } else {
        __Runtime_core_trace_decode(buf, n, NULL);
    }
    return 0;
}
"""


def _build_harness():
    """Compile the extracted decoder + harness. Returns the binary path, or
    None if gcc is unavailable (caller skips)."""
    if shutil.which("gcc") is None:
        return None
    with open(_RUNTIME_C) as f:
        src = f.read()
    slot_tbl, mem_tbl, core_evt_tbl, mem_evt_tbl, block = _extract_c_decoder(src)
    prog = ("#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n"
            "#include <string.h>\n\n"
            + _PROFILE_SHIM + "\n" + slot_tbl + "\n" + mem_tbl + "\n"
            + core_evt_tbl + "\n" + mem_evt_tbl + "\n\n" + block + "\n" + _HARNESS_MAIN)

    d = tempfile.mkdtemp(prefix="ctd_ut_")
    cpath = os.path.join(d, "harness.c")
    bpath = os.path.join(d, "harness")
    with open(cpath, "w") as f:
        f.write(prog)
    r = subprocess.run(["gcc", "-Wall", "-std=c11", "-O0", "-o", bpath, cpath],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"decoder failed to compile:\n{r.stderr}"
    return bpath


_CC_BIN = None
_CC_TRIED = False


def _cc_bin():
    global _CC_BIN, _CC_TRIED
    if not _CC_TRIED:
        _CC_TRIED = True
        _CC_BIN = _build_harness()
    return _CC_BIN


# --------------------------------------------------------------------------
# Independent frame encoder (inverse of the decoder, from Figure 4-14).
# Frames are byte-aligned (all widths are multiples of 8), MSB-first.
# --------------------------------------------------------------------------
def _u16(v):
    return bytes([(v >> 8) & 0xFF, v & 0xFF])


def _u24(v):
    return bytes([(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])


def _u32(v):
    return bytes([(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])


def start(timer, overrun=0):
    w0 = (0x1E << 27) | ((overrun & 1) << 26) | (0 << 24) | ((timer >> 32) & 0xFFFFFF)
    return _u32(w0) + _u32(timer & 0xFFFFFFFF)


def single(event, cyc):
    if cyc < 16:                                             # Single0 8b: 0 eee cccc
        return bytes([((event & 7) << 4) | (cyc & 0xF)])
    if cyc < 1024:                                           # Single1 16b: 100 eee c(10)
        return _u16((0b100 << 13) | ((event & 7) << 10) | (cyc & 0x3FF))
    return _u24((0b101 << 21) | ((event & 7) << 18) | (cyc & 0x3FFFF))  # Single2 24b


def multiple(mask, cyc):
    if cyc < 16:                                             # Multiple0 16b: 1100 m(8) c(4)
        return _u16((0b1100 << 12) | ((mask & 0xFF) << 4) | (cyc & 0xF))
    if cyc < 1024:                                           # Multiple1 24b: 110100 m(8) c(10)
        return _u24((0b110100 << 18) | ((mask & 0xFF) << 10) | (cyc & 0x3FF))
    return _u32((0b110101 << 26) | ((mask & 0xFF) << 18) | (cyc & 0x3FFFF))  # Multiple2 32b


def repeat(n):
    if n < 16:                                               # Repeat0 8b: 1110 r(4)
        return bytes([(0b1110 << 4) | (n & 0xF)])
    return _u16((0b110110 << 10) | (n & 0x3FF))             # Repeat1 16b: 110110 r(10)


def stop(cyc, x=0):                                          # Stop 32b: 110111 x(8) c(18)
    return _u32((0b110111 << 26) | ((x & 0xFF) << 18) | (cyc & 0x3FFFF))


def filler():
    return b"\xFE"


def sync():
    return b"\xFF"


PKT_HDR = 0x00000001  # packet id=1, type=0


def pack(*frames, header=PKT_HDR, terminate=True):
    """Concatenate frames, pad the final packet with Filler (as the trace unit
    does), split into 7-word payload packets each prefixed with a header word,
    and append an all-zero terminator packet."""
    data = b"".join(frames)
    while len(data) % 28 != 0:      # 28 bytes = 7 words = one packet payload
        data += b"\xFE"
    words = [(data[i] << 24) | (data[i + 1] << 16) | (data[i + 2] << 8) | data[i + 3]
             for i in range(0, len(data), 4)]
    out = []
    for i in range(0, len(words), 7):
        out.append(header)
        out.extend(words[i:i + 7])
    if terminate:
        out.extend([0] * 8)
    return out


FILLER_WORD = 0xFEFEFEFE  # four Filler bytes -> alignment pad, decodes to nothing


def pack_payload(words, header=PKT_HDR, terminate=True):
    """Place `words` VERBATIM into payload slots (buf[1..7] of each 8-word
    packet) so the header-skipping transport decodes them as ONE contiguous
    frame stream with NO packet header of their own -- i.e. word[0] is decoded
    as a real frame, not consumed as a routing header.

    Unlike pack(), which treats its arguments as an encoded frame byte stream,
    pack_payload takes already-formed 32-bit trace words (e.g. a raw capture)
    and just routes them through the transport untouched. The final packet is
    padded with Filler words (0xFEFEFEFE, a decode no-op) up to a whole packet,
    and an all-zero terminator packet is appended."""
    out = []
    for i in range(0, len(words), 7):
        chunk = list(words[i:i + 7])
        while len(chunk) < 7:
            chunk.append(FILLER_WORD)
        out.append(header)
        out.extend(chunk)
    if terminate:
        out.extend([0] * 8)
    return out


# --------------------------------------------------------------------------
# Runners: real C decoder and Python reference model, faithful output.
# --------------------------------------------------------------------------
def _norm(text):
    """C prints buf=<pointer>; the Python model prints buf=<sim>. Normalise so
    the two faithful transcripts are directly comparable."""
    return re.sub(r"buf=(0x[0-9a-fA-F]+|\(nil\))", "buf=<sim>", text)


def run_c(words):
    b = _cc_bin()
    assert b is not None
    r = subprocess.run([b] + [hex(w) for w in words], capture_output=True, text=True)
    assert r.returncode == 0, f"harness crashed: {r.stderr}"
    return _norm(r.stdout)


def run_c_profile(words):
    """Run the decoder with an attached AieTraceProfile (tile 4,4) and dump it."""
    b = _cc_bin()
    assert b is not None
    r = subprocess.run([b, "--profile"] + [hex(w) for w in words], capture_output=True, text=True)
    assert r.returncode == 0, f"harness crashed: {r.stderr}"
    return _norm(r.stdout)


def run_c_profile_host(words):
    """Run the decoder with a profile that ALSO carries host time-sync fields
    (cps/anchor/event/words), so the dump emits the full [TIMESYNC] host block."""
    b = _cc_bin()
    assert b is not None
    r = subprocess.run([b, "--profile-host"] + [hex(w) for w in words], capture_output=True, text=True)
    assert r.returncode == 0, f"harness crashed: {r.stderr}"
    return _norm(r.stdout)


def run_py(words):
    buf = io.StringIO()
    with redirect_stdout(buf):
        ctd.CoreTraceDecoder(mode="faithful").decode(words, nwords=len(words))
    return buf.getvalue()


# Faithful output is run-length compressed: a single cycle is "<cycle>  <events>",
# an interval is "<start> -- <end>  <events>  (<N> cyc)", and <events> joins slot
# names with '|'. Banners start with '[' and never match. Expanding an interval
# back to per-cycle, per-slot tuples reconstructs the exact old timeline.
_EVT = re.compile(r"^(\d+)(?:\s+--\s+(\d+))?  (\S+?)(?:  \(\d+ cyc\))?$")


def timeline(text):
    out = []
    for ln in text.splitlines():
        m = _EVT.match(ln)
        if not m:
            continue
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        names = m.group(3).split("|")
        for cyc in range(start, end + 1):
            for nm in names:
                out.append((cyc, nm))
    return out


def raw_intervals(text):
    """The decoder's own printed intervals as (start, end, names) tuples --
    NOT expanded per-cycle. Dump lines (leading spaces) never match _EVT."""
    out = []
    for ln in text.splitlines():
        m = _EVT.match(ln)
        if not m:
            continue
        s = int(m.group(1))
        e = int(m.group(2)) if m.group(2) else s
        out.append((s, e, m.group(3)))
    return out


# Profile dump interval line:
#   "[TIMESYNC] trace tile=<c>,<r> [stream=<tag> ]<start>[ -- <end>]  <events>[{event value ...}][  (<N> cyc)]".
# The optional "stream=<tag>" marks the mem-module DMA stream (pkt id 2); it is
# non-capturing here so core lines keep their 5-group layout. The trailing
# "{event value v1:v2:...}" carries each set slot's raw XAie event value and is
# stripped (non-capturing) so the names group stays clean. The
# "[TIMESYNC] trace tile=c,r words=N" header has "words=" after the tile, so it
# never matches this interval regex.
_PROF = re.compile(
    r"^\[TIMESYNC\] trace tile=(\d+),(\d+) (?:stream=\w+ )?(\d+)(?:\s+--\s+(\d+))?  "
    r"(\S+?)(?:\{event value [^}]*\})?(?:  \(\d+ cyc\))?$")


def profile_intervals(text):
    """Parse __Runtime_aie_trace_profile_dump output into
    (col, row, start, end, names) tuples."""
    out = []
    for ln in text.splitlines():
        m = _PROF.match(ln)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                    int(m.group(4)) if m.group(4) else int(m.group(3)), m.group(5)))
    return out


def format_parse_detail(label, text, head=20, tail=5):
    """Render a human-readable parse report from a faithful transcript: the API
    banner, any START/STOP control lines, event/cycle span, the per-slot
    histogram, and a head/tail sample of the decoded timeline. Used by the
    capture tests to SHOW what was parsed, not just assert pass/fail."""
    import collections
    lines = text.splitlines()
    ctrl = [l for l in lines if "START" in l or "STOP" in l]
    tl = timeline(text)
    hist = dict(collections.Counter(n for _, n in tl))
    r = ["=== parse detail: %s ===" % label]
    if lines:
        r.append(lines[0])                     # [aie_runtime] ... banner
    r.extend(ctrl)
    if tl:
        r.append("events: %d   cycles: %d..%d" % (len(tl), tl[0][0], tl[-1][0]))
    else:
        r.append("events: 0")
    r.append("slots : %s" % hist)
    sample = tl[:head]
    r.append("first %d:" % len(sample))
    r += ["  %d  %s" % (cy, nm) for cy, nm in sample]
    if len(tl) > head + tail:
        r.append("  ... (%d more) ..." % (len(tl) - head - tail))
    if len(tl) > head:
        last = tl[-tail:]
        r.append("last %d:" % len(last))
        r += ["  %d  %s" % (cy, nm) for cy, nm in last]
    return "\n".join(r)


def _skip_if_no_cc():
    if _cc_bin() is None:
        try:
            import pytest
            pytest.skip("gcc not available")
        except ImportError:
            raise SystemExit("SKIP: gcc not available")


# --------------------------------------------------------------------------
# Cross-check helper: C == Python (faithful), and timeline == golden.
# --------------------------------------------------------------------------
def _check(words, golden):
    _skip_if_no_cc()
    c_out = run_c(words)
    py_out = run_py(words)
    assert c_out == py_out, (
        "C decoder and Python model disagree:\n--- C ---\n%s\n--- PY ---\n%s"
        % (c_out, py_out))
    assert timeline(c_out) == golden, (
        "timeline mismatch:\n got %s\n want %s" % (timeline(c_out), golden))
    return c_out


# --------------------------------------------------------------------------
# Extraction sanity (runs without a compiler).
# --------------------------------------------------------------------------
def test_extract_regions_present():
    with open(_RUNTIME_C) as f:
        src = f.read()
    slot_tbl, mem_tbl, core_evt_tbl, mem_evt_tbl, block = _extract_c_decoder(src)
    assert "ACTIVE" in slot_tbl and "MEMORY_STALL" in slot_tbl
    assert "DMA_START" in mem_tbl and "LOCK_REL" in mem_tbl
    assert "s_core_trace_slot_event" in core_evt_tbl
    assert "s_mem_trace_slot_event" in mem_evt_tbl
    assert "__core_trace_bits" in block
    assert "void __Runtime_core_trace_decode(" in block
    assert "__core_trace_table_for_id" in block  # demux by pkt id
    # The profile API travels with the decoder (same extracted region).
    assert "__Runtime_aie_trace_profile_dump" in block


# --------------------------------------------------------------------------
# Trace-profile capture (AieTraceProfile: init/attach/decode-into/dump).
# --------------------------------------------------------------------------
def test_profile_captures_single_cycle_events():
    _skip_if_no_cc()
    words = pack(start(1000), single(0, 10), single(2, 100), multiple(0x03, 5))
    out = run_c_profile(words)
    ivs = profile_intervals(out)
    # Three non-contiguous 1-cycle events, tagged with the attached tile (4,4).
    assert [(c, r, s, e, nm) for c, r, s, e, nm in ivs] == [
        (4, 4, 1010, 1010, "ACTIVE"),
        (4, 4, 1110, 1110, "STREAM_STALL"),
        (4, 4, 1115, 1115, "ACTIVE|LOCK_STALL"),
    ]
    assert "trace_profile_dump: 3 intervals (dropped=0)" in out


def test_profile_coalesces_and_matches_printed_intervals():
    _skip_if_no_cc()
    # Contiguous same-slot cycles 1010..1012 coalesce; then a lone STREAM_STALL.
    words = pack(start(1000), single(0, 10), single(0, 1), single(0, 1), single(2, 100))
    out = run_c_profile(words)
    captured = [(s, e, nm) for _, _, s, e, nm in profile_intervals(out)]
    assert captured == [(1010, 1012, "ACTIVE"), (1112, 1112, "STREAM_STALL")]
    # The profile holds EXACTLY the decoder's own printed intervals.
    assert captured == raw_intervals(out)
    # Every captured interval carries the attached tile.
    assert all((c, r) == (4, 4) for c, r, *_ in profile_intervals(out))


def test_profile_null_leaves_output_unchanged():
    # The default (NULL-profile) path must be byte-identical to the plain run.
    _skip_if_no_cc()
    words = pack(start(1000), single(0, 10), single(2, 100))
    assert run_c(words) == run_py(words)     # existing invariant, NULL path
    # A --profile run's decode section prints the same intervals, plus a dump.
    prof_out = run_c_profile(words)
    assert raw_intervals(prof_out) == raw_intervals(run_c(words))


def test_profile_host_block_emits_timesync_fields():
    # The unified dump emits the host time-sync block (cps/anchors/hostevt/aiehz/
    # words) ahead of the [TIMESYNC] trace interval lines.
    _skip_if_no_cc()
    words = pack(start(1000), single(0, 10), single(2, 100))
    out = run_c_profile_host(words)
    assert "[TIMESYNC] cps=1000000" in out
    assert "[TIMESYNC] anchor0 host=0" in out
    assert "[TIMESYNC] anchor0 tile=4,4 aie=2000" in out
    assert "[TIMESYNC] anchor1 host=500" in out
    assert "[TIMESYNC] anchor1 tile=4,4 aie=3000" in out
    assert "[TIMESYNC] hostevt iter=0 phase=iter_start host=10" in out
    assert "[TIMESYNC] hostevt iter=0 phase=run host=20" in out
    # Effective Hz = (3000-2000)/(500-0)*1e6 = 2e6.
    assert "[TIMESYNC] aiehz tile=4,4 hz=2000000.000" in out
    # words = number of trace words passed on the command line.
    assert ("[TIMESYNC] trace tile=4,4 words=%u" % len(words)) in out
    # Interval lines still parse and are tagged with tile (4,4).
    ivs = profile_intervals(out)
    assert ivs == [(4, 4, 1010, 1010, "ACTIVE"), (4, 4, 1110, 1110, "STREAM_STALL")]


def test_profile_host_block_absent_without_host_fields():
    # A decode-only --profile run records no host fields, so NO [TIMESYNC] host
    # lines appear -- only the [TIMESYNC] trace interval lines.
    _skip_if_no_cc()
    words = pack(start(1000), single(0, 10), single(2, 100))
    out = run_c_profile(words)
    assert "[TIMESYNC] cps=" not in out
    assert "[TIMESYNC] anchor0" not in out
    assert "[TIMESYNC] hostevt" not in out
    assert "[TIMESYNC] aiehz" not in out
    assert "[TIMESYNC] trace tile=4,4 words=" not in out
    # But the interval lines are still present and parseable.
    assert profile_intervals(out) == [(4, 4, 1010, 1010, "ACTIVE"),
                                       (4, 4, 1110, 1110, "STREAM_STALL")]


# --------------------------------------------------------------------------
# Per-frame decode cases.
# --------------------------------------------------------------------------
def test_demo_start_single_multiple_sync():
    words = pack(start(1000), single(0, 10), single(2, 100),
                 filler(), multiple(0x03, 5), sync())
    out = _check(words, [(1010, "ACTIVE"), (1110, "STREAM_STALL"),
                         (1115, "ACTIVE"), (1115, "LOCK_STALL")])
    assert "START timer=1000 overrun=0" in out


def test_demo_words_match_reference_demo():
    # The independently packed demo must equal core_trace_decode._demo_words().
    words = pack(start(1000), single(0, 10), single(2, 100),
                 filler(), multiple(0x03, 5), sync())
    assert words[:8] == ctd._demo_words()


def test_single2_wide_cycle():
    # cyc=100000 forces the 24-bit Single2 form; slot 3 = MEMORY_STALL.
    words = pack(start(0), single(3, 100000))
    _check(words, [(100000, "MEMORY_STALL")])


def test_multiple1_and_multiple2_with_event_slots():
    words = pack(start(0), multiple(0xFF, 500), multiple(0x81, 200000))
    golden = [(500, "ACTIVE"), (500, "LOCK_STALL"), (500, "STREAM_STALL"),
              (500, "MEMORY_STALL"), (500, "PORT_IDLE_0"), (500, "PORT_RUNNING_0"),
              (500, "PORT_STALLED_0"), (500, "EVENT7"),
              (200500, "ACTIVE"), (200500, "EVENT7")]
    _check(words, golden)


def test_repeat0_after_single():
    # Repeat0 (n<16) re-emits the previous single event, +1 cycle each.
    words = pack(start(0), single(0, 10), repeat(3))
    _check(words, [(10, "ACTIVE"), (11, "ACTIVE"), (12, "ACTIVE"), (13, "ACTIVE")])


def test_repeat1_after_single():
    # n>=16 forces the 16-bit Repeat1 form.
    words = pack(start(0), single(0, 10), repeat(16))
    golden = [(10, "ACTIVE")] + [(11 + i, "ACTIVE") for i in range(16)]
    _check(words, golden)


def test_repeat_after_multiple_reemits_mask():
    words = pack(start(0), multiple(0x05, 5), repeat(2))
    golden = [(5, "ACTIVE"), (5, "STREAM_STALL"),
              (6, "ACTIVE"), (6, "STREAM_STALL"),
              (7, "ACTIVE"), (7, "STREAM_STALL")]
    _check(words, golden)


def test_stop_frame_prints_and_ends_events():
    words = pack(start(0), single(1, 5), stop(3))
    out = _check(words, [(5, "LOCK_STALL")])
    assert "STOP @ 8" in out


def test_start_overrun_flag():
    words = pack(start(1000, overrun=1), single(0, 1))
    out = _check(words, [(1001, "ACTIVE")])
    assert "START timer=1000 overrun=1" in out


# --------------------------------------------------------------------------
# Transport-framing cases.
# --------------------------------------------------------------------------
def test_frames_span_multiple_packets():
    # 1 Start (8B) + 30 x Single0 (1B) = 38B -> 2 packets; the header word of
    # the 2nd packet must be skipped mid-frame-stream.
    frames = [start(0)] + [single(0, 1) for _ in range(30)]
    words = pack(*frames)
    assert words.count(PKT_HDR) == 2  # two real packets
    _check(words, [(i + 1, "ACTIVE") for i in range(30)])


def test_all_zero_packet_terminates_before_garbage():
    # A real packet, the all-zero terminator, THEN a bogus non-zero packet that
    # must be ignored (caller passes full buffer capacity).
    good = pack(start(1000), single(0, 10), single(2, 100),
                filler(), multiple(0x03, 5), sync())
    garbage = pack(start(9), single(0, 7), terminate=False)  # must not be decoded
    words = good + garbage
    _check(words, [(1010, "ACTIVE"), (1110, "STREAM_STALL"),
                   (1115, "ACTIVE"), (1115, "LOCK_STALL")])


def test_empty_buffer_is_header_only():
    _skip_if_no_cc()
    words = [0] * 8  # a single all-zero packet: nothing to decode
    c_out = run_c(words)
    assert timeline(c_out) == []
    assert run_c(words) == run_py(words)


# --------------------------------------------------------------------------
# Packet-id demux: a shared MemTile buffer carrying the core stream (pkt id 1)
# and the mem-module DMA stream (pkt id 2) must decode each with its OWN slot
# table, ascending by id (core before mem). Legacy single-id buffers are
# unchanged (covered by every other case, which uses id 1 / id 0).
# --------------------------------------------------------------------------
MEM_HDR = 0x00000002  # packet id=2, type=0 (mem-module DMA trace)


def test_demux_core_and_mem_streams():
    # id 1 (core): ACTIVE @1010; id 2 (mem): slot0 DMA_START @2005. The two
    # streams share one buffer; the demux splits them by header id and labels
    # each with its table. Decoded ascending -> core interval then mem interval.
    _skip_if_no_cc()
    core = pack(start(1000), single(0, 10), header=PKT_HDR, terminate=False)
    mem = pack(start(2000), single(0, 5), header=MEM_HDR, terminate=False)
    words = core + mem + [0] * 8
    _check(words, [(1010, "ACTIVE"), (2005, "DMA_START")])


def test_demux_mem_slot_names_and_category():
    # Exercise several mem slots and assert the Python model tags them with the
    # mem name table and the 'mem' timeline category (distinct from core).
    mem = pack(start(0), multiple(0x0F, 5), header=MEM_HDR, terminate=False)
    words = mem + [0] * 8
    tl = ctd.CoreTraceDecoder(mode="quiet").decode(words)
    assert tl == [
        (5, "DMA_START", "mem"),
        (5, "DMA_FINISH", "mem"),
        (5, "DMA_STALL_LOCK", "mem"),
        (5, "STREAM_STALL", "mem"),
    ]


def test_demux_core_and_mem_categories_distinct():
    # Same STREAM_STALL slot name appears in BOTH tables (core slot 2, mem slot
    # 3); only the category disambiguates the stream. Core STREAM_STALL is
    # 'core', mem STREAM_STALL is 'mem'.
    core = pack(start(0), single(2, 7), header=PKT_HDR, terminate=False)
    mem = pack(start(0), single(3, 7), header=MEM_HDR, terminate=False)
    words = core + mem + [0] * 8
    tl = ctd.CoreTraceDecoder(mode="quiet").decode(words)
    assert (7, "STREAM_STALL", "core") in tl
    assert (7, "STREAM_STALL", "mem") in tl


def test_profile_dump_tags_mem_stream():
    # The profile dump must tag mem-module intervals (pkt id 2) with "stream=mem"
    # so the off-device timeline can route them to their own lane -- STREAM_STALL
    # collides with the core table, so the slot name alone cannot disambiguate.
    # Core intervals stay untagged (backward compatible with existing parsers).
    _skip_if_no_cc()
    core = pack(start(1000), single(0, 10), header=PKT_HDR, terminate=False)
    mem = pack(start(2000), single(0, 5), single(3, 2), header=MEM_HDR, terminate=False)
    words = core + mem + [0] * 8
    out = run_c_profile(words)
    lines = [l for l in out.splitlines() if l.startswith("[TIMESYNC] trace tile=")]
    core_lines = [l for l in lines if "stream=" not in l]
    mem_lines = [l for l in lines if "stream=mem " in l]
    # Each set slot now carries an "{event value ...}" suffix after the names.
    assert any("1010  ACTIVE{event value" in l for l in core_lines), core_lines
    assert any("DMA_START{event value" in l for l in mem_lines), mem_lines
    assert any("STREAM_STALL{event value" in l for l in mem_lines), mem_lines
    # The dump still parses into intervals (stream tag + suffix are non-capturing).
    ivs = [(s, e, nm) for _, _, s, e, nm in profile_intervals(out)]
    assert (2005, 2005, "DMA_START") in ivs, ivs
    assert (2007, 2007, "STREAM_STALL") in ivs, ivs


def test_profile_dump_appends_event_value_suffix():
    # Every set slot's raw AIE event value is appended after the joined names as
    # "{event value v1:v2:...}" in slot order. A core interval covering slots 0
    # and 2 (ACTIVE|STREAM_STALL) yields the two core event values, colon-joined.
    _skip_if_no_cc()
    core = pack(start(1000), multiple(0x05, 10), header=PKT_HDR, terminate=False)
    words = core + [0] * 8
    out = run_c_profile(words)
    lines = [l for l in out.splitlines() if l.startswith("[TIMESYNC] trace tile=")]
    ev = [l for l in lines if "ACTIVE" in l and "STREAM_STALL" in l]
    assert ev, lines
    m = re.search(r"\{event value ([0-9:]+)\}", ev[0])
    assert m, ev[0]
    vals = m.group(1).split(":")
    assert len(vals) == 2, ev[0]  # two set slots -> two colon-joined values
    assert all(v.isdigit() for v in vals), ev[0]


# --------------------------------------------------------------------------
# Real captured buffer -- a raw 8-word packet exactly as passed by a caller.
#
#   [0] 0xf0000000  [1] 0x00d5922e  [2] 0x00d971ff  [3] 0xd833fefe
#   [4] 0xa1373300  [5] 0xd999fffe  [6] 0xdbffdbff  [7] 0xdbffdbff
#
# NOTE: word[0]=0xf0000000 is consumed as the stream-packet ROUTING HEADER (the
# transport layer skips buf[0] of every 8-word packet), so frame decoding starts
# at buf[1]=0x00d5922e -- NOT at word[0]. Even though 0xf0000000 looks like a
# Start frame, here it is the header and is skipped. The 7 payload words contain
# Repeat1 frames (count 1023) that expand into a large timeline (1522 events).
# This is a CHARACTERIZATION test: the anchors are ground truth captured from
# the real decoder, and all three drivers (C harness, Python model, and -- in
# test_core_trace_decode_cython -- the in-process Cython bind) must agree.
# --------------------------------------------------------------------------
USER_CAPTURE = [0xf0000000, 0x00d5922e, 0x00d971ff, 0xd833fefe,
                0xa1373300, 0xd999fffe, 0xdbffdbff, 0xdbffdbff]


def test_user_capture_8word_packet():
    import collections
    _skip_if_no_cc()
    c_out = run_c(USER_CAPTURE)
    assert c_out == run_py(USER_CAPTURE)        # C decoder == Python model

    tl = timeline(c_out)
    assert c_out.splitlines()[0] == (
        "[aie_runtime] core_trace_decode: buf=<sim> nwords=8")
    # buf[0]=0xf0000000 skipped as header; first payload byte 0x00 -> Single0
    # event 0, cycle 0.
    assert tl[0] == (0, "ACTIVE")
    assert tl[1:4] == [(142848, "STREAM_STALL"), (142848, "PORT_RUNNING_0"),
                       (142848, "PORT_STALLED_0")]
    assert tl[-1] == (13854729, "ACTIVE")
    assert len(tl) == 1522
    assert max(cyc for cyc, _ in tl) == 13854729
    assert dict(collections.Counter(n for _, n in tl)) == {
        "ACTIVE": 412, "STREAM_STALL": 370, "PORT_RUNNING_0": 370, "PORT_STALLED_0": 370}


# --------------------------------------------------------------------------
# Same 8 words, but with NO packet header -- the words ARE the raw payload.
#
#   [0] 0xf0000000  [1] 0x00d5922e  [2] 0x00d971ff  [3] 0xd833fefe
#   [4] 0xa1373300  [5] 0xd999fffe  [6] 0xdbffdbff  [7] 0xdbffdbff
#
# Here word[0]=0xf0000000 is NOT a routing header: type byte 0xf0 = 11110000
# matches the Start-frame prefix (11110 O 00), so it IS decoded as Start
# (timer = 0x00d5922e = 13996590, overrun=0). pack_payload() drops the words
# straight into payload slots so the header-skipping transport sees them as one
# contiguous frame stream. The trailing Repeat1 frames (0xdbff = 1023 reps,
# 0xd971 = 369 reps, ...) expand the single ACTIVE events into an 781-entry
# timeline. This decode was cross-checked against an independent no-header
# reference (pw = identity, 256-bit stream) and matches exactly.
# --------------------------------------------------------------------------
USER_CAPTURE_NOHDR = [0xf0000000, 0x00d5922e, 0x00d971ff, 0xd833fefe,
                      0xa1373300, 0xd999fffe, 0xdbffdbff, 0xdbffdbff]


def test_user_capture_8word_no_pkt_header():
    import collections
    _skip_if_no_cc()
    words = pack_payload(USER_CAPTURE_NOHDR)
    c_out = run_c(words)
    assert c_out == run_py(words)               # C decoder == Python model

    # SHOW the parse detail, not just pass/fail.
    print("\n" + format_parse_detail("USER_CAPTURE no-pkt-header (C API)", c_out))

    tl = timeline(c_out)
    # word[0]=0xf0000000 decoded as Start: timer = 0x00d5922e = 13996590.
    assert any("START timer=13996590 overrun=0" in l for l in c_out.splitlines())
    assert tl[0] == (13996590, "ACTIVE")        # Single0 event0 at the timer base
    assert tl[-1] == (27708471, "ACTIVE")
    assert len(tl) == 781
    assert max(cyc for cyc, _ in tl) == 27708471
    assert dict(collections.Counter(n for _, n in tl)) == {"ACTIVE": 781}


# --------------------------------------------------------------------------
# Standalone runner (works without pytest installed).
# --------------------------------------------------------------------------
def _main():
    fns = [g for n, g in sorted(globals().items())
           if n.startswith("test_") and callable(g)]
    passed = skipped = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except SystemExit as e:
            skipped += 1
            print(f"SKIP  {fn.__name__}: {e}")
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            return 1
    print(f"\n{passed} passed, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
