#!/usr/bin/env python3
"""In-process unit test for __Runtime_core_trace_decode via Cython.

Unlike test_core_trace_decode.py (which compiles a main() harness and shells
out), this builds a Cython extension that binds the C API

    void __Runtime_core_trace_decode(const uint32_t *buf, uint32_t nwords,
                                     AieTraceProfile *prof);

and calls it DIRECTLY in-process (with prof=NULL, the print-only path). The decoder body is extracted from the real
aie_runtime.c (the self-contained region only -- it cannot be linked whole
because aie_runtime.c pulls in xaiengine.h). Because the API returns void and
reports through printf, the test captures C-level fd 1 around each call.

The frame encoder, Python reference model and golden checks are reused from
test_core_trace_decode so the two suites stay in lock-step.

Skips cleanly if Cython or a C compiler is unavailable.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import test_core_trace_decode as base  # frame encoder + reference model + goldens


# --------------------------------------------------------------------------
# Build a Cython extension binding __Runtime_core_trace_decode.
# --------------------------------------------------------------------------
_PYX = r'''
# cython: language_level=3
from libc.stdint cimport uint32_t
from libc.stdio cimport fflush, stdout
from libc.stdlib cimport malloc, free

cdef extern from "ctd_impl.h":
    ctypedef struct AieTraceProfile:
        pass
    void __Runtime_core_trace_decode(const uint32_t *buf, uint32_t nwords, AieTraceProfile *prof)

def decode(list words):
    """Call the C API in-process with an array built from `words`."""
    cdef uint32_t n = <uint32_t>len(words)
    cdef uint32_t *buf = <uint32_t*>malloc((n if n else 1) * sizeof(uint32_t))
    if buf == NULL:
        raise MemoryError()
    cdef uint32_t i
    try:
        for i in range(n):
            buf[i] = <uint32_t>(int(words[i]) & 0xFFFFFFFF)
        __Runtime_core_trace_decode(buf, n, NULL)   # NULL = print-only path
        fflush(stdout)                 # flush C stdio before fd 1 is restored
    finally:
        free(buf)
'''

# Opaque AieTraceProfile in the extern header: the in-process test only ever
# passes NULL, so the generated C needs the type name, not its layout. The impl
# TU defines the real struct via base._PROFILE_SHIM; the two never meet.
_IMPL_H = ("#ifndef CTD_IMPL_H\n#define CTD_IMPL_H\n#include <stdint.h>\n"
           "typedef struct AieTraceProfile AieTraceProfile;\n"
           "void __Runtime_core_trace_decode(const uint32_t *buf, uint32_t nwords, AieTraceProfile *prof);\n"
           "#endif\n")

_SETUP = (
    "from setuptools import setup, Extension\n"
    "from Cython.Build import cythonize\n"
    "setup(ext_modules=cythonize(\n"
    "    [Extension('ctd_cy', ['ctd_cy.pyx', 'ctd_impl.c'])],\n"
    "    language_level=3, quiet=True))\n")

_CY = None
_CY_TRIED = False


def _build_cy_module():
    """Extract the decoder from aie_runtime.c, wrap it in Cython, build and
    import. Returns the module, or None if the toolchain is unavailable."""
    try:
        import Cython  # noqa: F401
    except ImportError:
        return None
    if shutil.which("gcc") is None and shutil.which("cc") is None:
        return None

    with open(base._RUNTIME_C) as f:
        src = f.read()
    slot_tbl, mem_tbl, block = base._extract_c_decoder(src)
    # The extracted region now includes the profile API, which references
    # AieTraceProfile/XAie_LocType; provide the same shim the subprocess suite uses.
    # The demux (__core_trace_table_for_id) also needs the mem-module slot table.
    impl_c = ("#include <stdint.h>\n#include <stdio.h>\n\n"
              + base._PROFILE_SHIM + "\n" + slot_tbl + "\n" + mem_tbl + "\n\n" + block + "\n")

    d = tempfile.mkdtemp(prefix="ctd_cy_")
    for name, text in (("ctd_impl.h", _IMPL_H), ("ctd_impl.c", impl_c),
                       ("ctd_cy.pyx", _PYX), ("setup.py", _SETUP)):
        with open(os.path.join(d, name), "w") as f:
            f.write(text)

    r = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                       cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, f"Cython build failed:\n{r.stdout}\n{r.stderr}"

    sys.path.insert(0, d)
    import ctd_cy  # the freshly built extension
    return ctd_cy


def _cy_module():
    global _CY, _CY_TRIED
    if not _CY_TRIED:
        _CY_TRIED = True
        _CY = _build_cy_module()
    return _CY


def _skip_if_no_cy():
    if _cy_module() is None:
        try:
            import pytest
            pytest.skip("Cython or C compiler unavailable")
        except ImportError:
            raise SystemExit("SKIP: Cython or C compiler unavailable")


# --------------------------------------------------------------------------
# Call the in-process API, capturing what the C code prints to fd 1.
# --------------------------------------------------------------------------
def _call_capture(words):
    mod = _cy_module()
    sys.stdout.flush()
    saved = os.dup(1)
    tf = tempfile.TemporaryFile()
    try:
        os.dup2(tf.fileno(), 1)
        mod.decode(list(words))       # fflush happens inside the wrapper
    finally:
        os.dup2(saved, 1)
        os.close(saved)
    tf.seek(0)
    data = tf.read().decode()
    tf.close()
    return base._norm(data)           # buf=<pointer> -> buf=<sim>


def _check(words, golden):
    _skip_if_no_cy()
    out = _call_capture(words)
    py = base.run_py(words)
    assert out == py, (
        "in-process C API and Python model disagree:\n--- C ---\n%s\n--- PY ---\n%s"
        % (out, py))
    assert base.timeline(out) == golden, (
        "timeline mismatch:\n got %s\n want %s" % (base.timeline(out), golden))
    return out


# --------------------------------------------------------------------------
# Tests -- same coverage as the subprocess suite, exercised via a direct call.
# --------------------------------------------------------------------------
def test_cy_module_builds_and_binds():
    _skip_if_no_cy()
    assert hasattr(_cy_module(), "decode")


def test_cy_demo_start_single_multiple_sync():
    words = base.pack(base.start(1000), base.single(0, 10), base.single(2, 100),
                      base.filler(), base.multiple(0x03, 5), base.sync())
    out = _check(words, [(1010, "ACTIVE"), (1110, "STREAM_STALL"),
                         (1115, "ACTIVE"), (1115, "LOCK_STALL")])
    assert "START timer=1000 overrun=0" in out
    assert "core_trace_decode: buf=<sim> nwords=%d" % len(words) in out


def test_cy_single2_wide_cycle():
    words = base.pack(base.start(0), base.single(3, 100000))
    _check(words, [(100000, "MEMORY_STALL")])


def test_cy_multiple_event_slots():
    words = base.pack(base.start(0), base.multiple(0xFF, 500),
                      base.multiple(0x81, 200000))
    golden = [(500, "ACTIVE"), (500, "LOCK_STALL"), (500, "STREAM_STALL"),
              (500, "MEMORY_STALL"), (500, "PORT_IDLE_0"), (500, "PORT_RUNNING_0"),
              (500, "PORT_STALLED_0"), (500, "EVENT7"),
              (200500, "ACTIVE"), (200500, "EVENT7")]
    _check(words, golden)


def test_cy_repeat0_after_single():
    words = base.pack(base.start(0), base.single(0, 10), base.repeat(3))
    _check(words, [(10, "ACTIVE"), (11, "ACTIVE"), (12, "ACTIVE"), (13, "ACTIVE")])


def test_cy_repeat1_after_single():
    words = base.pack(base.start(0), base.single(0, 10), base.repeat(16))
    golden = [(10, "ACTIVE")] + [(11 + i, "ACTIVE") for i in range(16)]
    _check(words, golden)


def test_cy_repeat_after_multiple():
    words = base.pack(base.start(0), base.multiple(0x05, 5), base.repeat(2))
    _check(words, [(5, "ACTIVE"), (5, "STREAM_STALL"),
                   (6, "ACTIVE"), (6, "STREAM_STALL"),
                   (7, "ACTIVE"), (7, "STREAM_STALL")])


def test_cy_stop_frame():
    words = base.pack(base.start(0), base.single(1, 5), base.stop(3))
    out = _check(words, [(5, "LOCK_STALL")])
    assert "STOP @ 8" in out


def test_cy_frames_span_multiple_packets():
    frames = [base.start(0)] + [base.single(0, 1) for _ in range(30)]
    words = base.pack(*frames)
    _check(words, [(i + 1, "ACTIVE") for i in range(30)])


def test_cy_all_zero_terminates_before_garbage():
    good = base.pack(base.start(1000), base.single(0, 10), base.single(2, 100),
                     base.filler(), base.multiple(0x03, 5), base.sync())
    garbage = base.pack(base.start(9), base.single(0, 7), terminate=False)
    _check(good + garbage, [(1010, "ACTIVE"), (1110, "STREAM_STALL"),
                            (1115, "ACTIVE"), (1115, "LOCK_STALL")])


def test_cy_empty_buffer():
    _skip_if_no_cy()
    words = [0] * 8
    assert base.timeline(_call_capture(words)) == []


def test_cy_user_capture_8word_packet():
    """User-supplied real 8-word packet, decoded via the in-process C API.
    word[0]=0xf0000000 is the packet header (skipped); payload expands to 1522
    events. Must equal the Python model and the subprocess harness."""
    import collections
    _skip_if_no_cy()
    out = _call_capture(base.USER_CAPTURE)
    assert out == base.run_py(base.USER_CAPTURE)
    if base._cc_bin() is not None:
        assert out == base.run_c(base.USER_CAPTURE)
    tl = base.timeline(out)
    assert tl[0] == (0, "ACTIVE")
    assert len(tl) == 1522
    assert dict(collections.Counter(n for _, n in tl)) == {
        "ACTIVE": 412, "STREAM_STALL": 370, "PORT_RUNNING_0": 370, "PORT_STALLED_0": 370}


def test_cy_user_capture_8word_no_pkt_header():
    """Same 8 words as test_cy_user_capture_8word_packet, but decoded as RAW
    PAYLOAD with NO packet header: word[0]=0xf0000000 (type byte 0xf0) is a
    Start frame and IS decoded (timer=13996590), not consumed as a routing
    header. pack_payload() routes the words through payload slots so the
    in-process C API sees one contiguous frame stream. Prints the parse detail
    (not just pass/fail)."""
    import collections
    _skip_if_no_cy()
    words = base.pack_payload(base.USER_CAPTURE_NOHDR)
    out = _call_capture(words)
    assert out == base.run_py(words)
    if base._cc_bin() is not None:
        assert out == base.run_c(words)

    print("\n" + base.format_parse_detail(
        "USER_CAPTURE no-pkt-header (in-process Cython C API)", out))

    tl = base.timeline(out)
    assert any("START timer=13996590 overrun=0" in l for l in out.splitlines())
    assert tl[0] == (13996590, "ACTIVE")
    assert tl[-1] == (27708471, "ACTIVE")
    assert len(tl) == 781
    assert dict(collections.Counter(n for _, n in tl)) == {"ACTIVE": 781}


def test_cy_demux_core_and_mem_streams():
    """A shared buffer carrying the core stream (pkt id 1) and the mem-module
    DMA stream (pkt id 2) must decode each with its OWN slot table, ascending by
    id (core before mem), via the in-process C API."""
    core = base.pack(base.start(1000), base.single(0, 10),
                     header=base.PKT_HDR, terminate=False)
    mem = base.pack(base.start(2000), base.single(0, 5),
                    header=base.MEM_HDR, terminate=False)
    words = core + mem + [0] * 8
    _check(words, [(1010, "ACTIVE"), (2005, "DMA_START")])


def test_cy_matches_subprocess_harness():
    """The in-process Cython call and the subprocess harness must produce
    identical output for the same input."""
    _skip_if_no_cy()
    if base._cc_bin() is None:
        return
    words = base.pack(base.start(1000), base.single(0, 10), base.single(2, 100),
                      base.filler(), base.multiple(0x03, 5), base.sync())
    assert _call_capture(words) == base.run_c(words)


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
