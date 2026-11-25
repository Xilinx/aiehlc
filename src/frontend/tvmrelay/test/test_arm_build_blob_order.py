###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""write_aligned_blob packs weights in a deterministic order.

TVM's ``save_param_dict`` writes params in an unordered-map order that changes
run to run. Two builds of the same config used to get different weights.bin
layouts (arm J vs K0, 11-22: 44/340 params moved). With ``order`` given, the
output must not depend on the source blob's order.

Run:  python -m pytest -q src/frontend/tvmrelay/test/test_arm_build_blob_order.py
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frontend.tvmrelay import arm_build as ab  # noqa: E402


def _blob(params):
    """Minimal relay.save_param_dict blob: fp32 1-D arrays, given order."""
    out = struct.pack("<QQQ", ab.PARAM_MAGIC, 0, len(params))
    for name, _ in params:
        out += struct.pack("<Q", len(name.encode())) + name.encode()
    out += struct.pack("<Q", len(params))
    for _, arr in params:
        raw = arr.astype("<f4").tobytes()
        out += struct.pack("<QQii", ab.NDARRAY_MAGIC, 0, 1, 0)
        out += struct.pack("<i", 1) + struct.pack("<BBH", 2, 32, 1)
        out += struct.pack("<q", arr.size) + struct.pack("<q", len(raw)) + raw
    return out


def _params():
    rng = np.random.default_rng(0)
    return [(f"p{i}", rng.standard_normal(3 + 5 * i)) for i in range(6)]


def test_layout_independent_of_source_order(tmp_path):
    ps = _params()
    order = ["p3", "p0", "p5", "p1", "p4"]          # p2 not in the graph
    outs = []
    for k, perm in enumerate([ps, ps[::-1], ps[2:] + ps[:2]]):
        src = tmp_path / f"src{k}.params"
        src.write_bytes(_blob(perm))
        offs = ab.param_offsets(src)
        dst = tmp_path / f"w{k}.bin"
        new = ab.write_aligned_blob(src, offs, dst, bf16_names={"p1"},
                                    order=order)
        outs.append((dst.read_bytes(), new))
    assert outs[0] == outs[1] == outs[2]
    new = outs[0][1]
    # graph order first, leftovers sorted after; every offset 64-B aligned
    assert sorted(new, key=lambda n: new[n][0]) == order + ["p2"]
    assert all(off % 64 == 0 for off, _ in new.values())
    # bytes round-trip (fp32 verbatim, bf16 halves)
    data = outs[0][0]
    for name, arr in ps:
        off, nb = new[name]
        if name == "p1":
            assert nb == arr.size * 2
            got = np.frombuffer(data[off:off + nb], "<u2")
            assert (got == ab.f32_to_bf16_rne(arr.astype("<f4"))).all()
        else:
            assert np.array_equal(np.frombuffer(data[off:off + nb], "<f4"),
                                  arr.astype("<f4"))


def test_no_order_is_still_deterministic(tmp_path):
    ps = _params()
    blobs = []
    for k, perm in enumerate([ps, ps[::-1]]):
        src = tmp_path / f"s{k}.params"
        src.write_bytes(_blob(perm))
        dst = tmp_path / f"o{k}.bin"
        ab.write_aligned_blob(src, ab.param_offsets(src), dst)
        blobs.append(dst.read_bytes())
    assert blobs[0] == blobs[1]
