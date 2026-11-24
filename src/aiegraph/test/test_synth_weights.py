###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Synthetic weights (dnn_torch.synthesize_weights): the board app's synth_fill and the workstation's synth_s8
must produce the same bytes, or the board's output is checked against a reference for different weights."""
import shutil
import subprocess

import numpy as np
import pytest

from aiegraph import dnn_torch as dt


@pytest.mark.skipif(not shutil.which("gcc"), reason="needs gcc")
def test_c_hash_matches_numpy(tmp_path):
    src = tmp_path / "s.c"
    src.write_text("#include <stdint.h>\n#include <stdio.h>\n" + "\n".join(dt.SYNTH_HASH_C) +
                   "\nint main(){ static int8_t b[100003]; synth_fill(b, sizeof b, 77u);"
                   " fwrite(b, 1, sizeof b, stdout); return 0; }\n")
    subprocess.check_call(["gcc", "-O2", str(src), "-o", str(tmp_path / "s")])
    c = np.frombuffer(subprocess.check_output([str(tmp_path / "s")]), np.int8)
    np.testing.assert_array_equal(c, dt.synth_s8(100003, 77))


def test_values_are_symmetric_int8():
    v = dt.synth_s8(1 << 16, 1)
    assert v.min() == -127 and v.max() == 127          # -128 never: int8 weights are symmetric


def test_only_large_s8_constants_are_synthesized():
    g = dt.AGraph(mode="int8")
    big = g.const(np.ones((128, 256), np.int8), "s8")
    small = g.const(np.ones((16,), np.int8), "s8")
    scale = g.const(np.ones((128 * 256,), np.float32), "f32")
    moved = dt.synthesize_weights(g, min_bytes=1024)
    assert g.synth == {big: big + 1} and moved == 128 * 256
    np.testing.assert_array_equal(g.tensors[big].data.ravel(), dt.synth_s8(128 * 256, big + 1))
    assert (g.tensors[small].data == 1).all() and (g.tensors[scale].data == 1).all()
