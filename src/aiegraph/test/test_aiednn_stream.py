###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Caller-owned workspace, streams, find / set_impl (aiednn_rt_stream.h;
doc/design/aiednn_framework_api.md sections 3, 4): aiednn_stream_test.cc links the ctypes build's
libaiednn.so and counts heap allocations (interposed malloc / calloc / realloc) during an execute with
a caller workspace -- none allowed -- for an AIE GEMM, a gemm+ composite, the float GEMM through RNN
steps, FVEC and a ref:-only op.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_stream.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import AiednnLib  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")


def test_workspace_streams_find(tmp_path):
    AiednnLib.build(tmp_path)
    exe = tmp_path / "stream_test"
    subprocess.run(["g++", "-std=c++17", "-O1", f"-I{REPO / 'include'}", "-o", str(exe),
                    str(Path(__file__).with_name("aiednn_stream_test.cc")), f"-L{tmp_path}", "-laiednn",
                    f"-Wl,-rpath,{tmp_path}"], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "STREAM TEST PASS" in r.stdout, r.stdout + r.stderr
    ws = {l.split()[1]: l for l in r.stdout.splitlines() if l.startswith("WS ")}
    assert set(ws) == {"conv_s8", "conv_s8_ref", "conv3d_s8", "lstm_f32", "gelu_f32", "lrn_f32"}
    assert "impl=gemm+conv+" in ws["conv3d_s8"] and "impl=gemm+rnn+" in ws["lstm_f32"]
    assert all("allocs=0" in l for l in ws.values())
    assert "CONTROL conv3d_s8" in r.stdout   # the interposed counter does see the library's allocations
