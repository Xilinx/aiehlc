###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Dispatch preference (aiednn_set_preference) and aiednn_primitive_impl_candidates."""

from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
PREFER_AIE, PREFER_FASTEST = 0, 1


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    lib = AiednnLib.build(tmp_path_factory.mktemp("aiednn_policy"))
    yield lib
    lib.so.aiednn_set_preference(PREFER_AIE)


def candidates(lib, prim):
    buf = C.create_string_buffer(512)
    n = lib.so.aiednn_primitive_impl_candidates(prim, buf, C.c_size_t(512))
    names = buf.value.decode().split(",") if buf.value else []
    assert n == len(names)
    return names


def quantize(lib, pref):
    """f32 -> s8 contiguous quantize: ref:'s loop was measured faster on the board"""
    lib.so.aiednn_set_preference(pref)
    x = np.linspace(-3, 3, 4096, dtype=np.float32)
    try:
        got, impl = lib.call("aiednn_reorder_create",
                             [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "s8")), None, C.c_float(0.05),
                              C.c_int32(1)], {ARG["src"]: x, ARG["dst"]: np.zeros(x.shape, np.int8)}, ARG["dst"])
    finally:
        lib.so.aiednn_set_preference(PREFER_AIE)
    np.testing.assert_array_equal(got, np.clip(np.rint(x.astype(np.float64) / 0.05) + 1, -128, 127))
    return impl


def test_default_prefers_the_aie(lib):
    assert lib.so.aiednn_get_preference() == PREFER_AIE
    assert quantize(lib, PREFER_AIE).startswith("steps:")


def test_prefer_fastest_keeps_measured_refusals(lib):
    assert quantize(lib, PREFER_FASTEST) == "ref:reorder"


def test_candidates_aie_first(lib):
    x = C.byref(md((4096,), "f32"))
    prim, st = lib.create("aiednn_reorder_create", x, C.byref(md((4096,), "s8")), None, C.c_float(0.05), C.c_int32(0))
    assert st == 0
    names = candidates(lib, prim)
    lib.so.aiednn_primitive_destroy(prim)
    assert names[0].startswith("steps:") and names[-1] == "ref:reorder", names


def test_autotune_stays_on_the_aie(lib, capfd):
    """with prefer_aie, autotuning times only the AIE acceptors (here a single one: no timing run)"""
    lib.so.aiednn_set_autotune(1)
    lib.so.aiednn_set_verbose(1)
    try:
        impl = quantize(lib, PREFER_AIE)
    finally:
        lib.so.aiednn_set_verbose(0)
        lib.so.aiednn_set_autotune(0)
    C.CDLL(None).fflush(None)   # the C runtime's stdout buffer, before capfd reads
    out = capfd.readouterr().out
    assert impl.startswith("steps:")
    assert "chose ref:" not in out


def test_steps_tables_follow_the_dst_scale(lib):
    """Two batch norms with equal statistics and different dst scales must not share steps: tables."""
    from aiegraph.dnn_ctypes import attr
    x = np.linspace(-4, 4, 16 * 32 * 32, dtype=np.float32).reshape(1, 16, 32, 32)   # steps: wants long channel runs
    st = {ARG["mean"]: np.zeros(16, np.float32), ARG["variance"]: np.ones(16, np.float32),
          ARG["scale"]: np.ones(16, np.float32), ARG["shift"]: np.zeros(16, np.float32)}
    for dscale in (0.05, 0.11):
        a, _k = attr(lib.ALG, dst_scale=dscale)
        outs = []
        for filt in (None, "ref:"):
            lib.set_filter(filt)
            try:
                got, impl = lib.call("aiednn_batch_normalization_forward_inference_create_attr",
                                     [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "s8")), C.c_float(1e-5),
                                      C.c_uint(3), C.byref(a)],
                                     {ARG["src"]: x, ARG["dst"]: np.zeros(x.shape, np.int8), **st}, ARG["dst"])
            finally:
                lib.set_filter(None)
            outs.append((impl, got.copy()))
        assert outs[0][0].startswith("steps:"), outs[0][0]
        np.testing.assert_array_equal(outs[0][1], outs[1][1])
