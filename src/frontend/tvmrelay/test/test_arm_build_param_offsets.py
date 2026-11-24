"""param_offsets / read_param_names read only the headers (F14).

The payloads are skipped with seek, so a 1.29 GB blob is no longer read twice
per call. The offsets must be the same as a full in-memory walk, and a
truncated blob must still raise ValueError.

Run:  python -m pytest -q src/frontend/tvmrelay/test/test_arm_build_param_offsets.py
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frontend.tvmrelay import arm_build as ab  # noqa: E402


def _blob(params):
    """relay.save_param_dict layout with n-D fp32 arrays; returns (bytes, expected offsets)."""
    out = struct.pack("<QQQ", ab.PARAM_MAGIC, 0, len(params))
    for name, _ in params:
        out += struct.pack("<Q", len(name.encode())) + name.encode()
    out += struct.pack("<Q", len(params))
    exp = {}
    for name, arr in params:
        raw = arr.astype("<f4").tobytes()
        out += struct.pack("<QQii", ab.NDARRAY_MAGIC, 0, 1, 0)
        out += struct.pack("<i", arr.ndim) + struct.pack("<BBH", 2, 32, 1)
        out += b"".join(struct.pack("<q", d) for d in arr.shape)
        out += struct.pack("<q", len(raw))
        exp[name] = (len(out), len(raw))
        out += raw
    return out, exp


def _params():
    rng = np.random.default_rng(1)
    shapes = [(3,), (2, 5), (1, 4, 3, 3), (), (7, 1)]
    return [(f"layer{i}.weight", rng.standard_normal(s)) for i, s in enumerate(shapes)]


def test_offsets_match_full_walk(tmp_path):
    data, exp = _blob(_params())
    p = tmp_path / "net_params.bin"
    p.write_bytes(data)
    offs = ab.param_offsets(p)
    assert offs == exp
    assert list(offs) == [n for n, _ in _params()]
    assert ab.read_param_names(p) == [n for n, _ in _params()]
    for name, arr in _params():
        o, n = offs[name]
        np.testing.assert_array_equal(np.frombuffer(data[o:o + n], "<f4"),
                                      arr.astype("<f4").ravel())


def test_reads_headers_only(tmp_path, monkeypatch):
    big = [("w", np.zeros(1 << 20, np.float32))]   # 4 MiB payload
    data, _ = _blob(big)
    p = tmp_path / "big_params.bin"
    p.write_bytes(data)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("read the whole blob"))
    assert ab.param_offsets(p)["w"][1] == 4 << 20


@pytest.mark.parametrize("cut", [10, 30, -1])
def test_truncated_blob_raises(tmp_path, cut):
    data, _ = _blob(_params())
    p = tmp_path / "t_params.bin"
    p.write_bytes(data[:cut])
    with pytest.raises(ValueError):
        ab.param_offsets(p)


def test_bad_magic_raises(tmp_path):
    data, _ = _blob(_params())
    p = tmp_path / "x.bin"
    p.write_bytes(b"\0" * 8 + data[8:])
    with pytest.raises(ValueError, match="bad magic"):
        ab.read_param_names(p)
