"""F12: boardmain.input_int32 content cache (AIEHLC_I32_CACHE / AIEHLC_FAST)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import onnx                                           # noqa: E402
from onnx import TensorProto, helper                  # noqa: E402

from aiegraph import boardmain as bm                  # noqa: E402


def _model(p: Path, k=1):
    t = helper.make_tensor_value_info("timestep", TensorProto.INT64, [1])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [1])
    c = helper.make_tensor("c", TensorProto.INT64, [1], [k])
    n = helper.make_node("Add", ["timestep", "c"], ["y"])
    g = helper.make_graph([n], "g", [t], [y], initializer=[c])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), str(p))
    return p


def test_cache_hit_across_out_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("AIEHLC_I32_CACHE", str(tmp_path / "c"))
    src = _model(tmp_path / "m.onnx")
    logs = []
    (tmp_path / "o1").mkdir()
    a = bm.input_int32(src, tmp_path / "o1" / "t32.onnx", "timestep", log=logs.append)
    (tmp_path / "o2").mkdir()
    b = bm.input_int32(src, tmp_path / "o2" / "t32.onnx", "timestep", log=logs.append)
    assert "cache hit" in logs[-1] and "cache hit" not in logs[0]
    assert a.read_bytes() == b.read_bytes()
    m = onnx.load(str(b))
    assert m.graph.input[0].type.tensor_type.elem_type == TensorProto.INT32
    assert m.graph.node[0].op_type == "Cast"
    assert len(list((tmp_path / "c").glob("*.onnx"))) == 1


def test_same_bytes_as_uncached(tmp_path, monkeypatch):
    src = _model(tmp_path / "m.onnx")
    monkeypatch.setenv("AIEHLC_I32_CACHE", "0")
    ref = bm.input_int32(src, tmp_path / "ref.onnx", "timestep", log=lambda *_: None)
    monkeypatch.setenv("AIEHLC_I32_CACHE", str(tmp_path / "c"))
    x = bm.input_int32(src, tmp_path / "x.onnx", "timestep", log=lambda *_: None)
    y = bm.input_int32(src, tmp_path / "y.onnx", "timestep", log=lambda *_: None)
    assert ref.read_bytes() == x.read_bytes() == y.read_bytes()


def test_content_change_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("AIEHLC_I32_CACHE", str(tmp_path / "c"))
    src = _model(tmp_path / "m.onnx", 1)
    a = bm.input_int32(src, tmp_path / "a.onnx", "timestep", log=lambda *_: None).read_bytes()
    st = src.stat()
    _model(src, 2)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime: a stat key would wrongly hit
    logs = []
    b = bm.input_int32(src, tmp_path / "b.onnx", "timestep", log=logs.append).read_bytes()
    assert a != b and "cache hit" not in logs[-1]


def test_disabled_without_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("AIEHLC_I32_CACHE", raising=False)
    monkeypatch.delenv("AIEHLC_FAST", raising=False)
    assert bm._i32_cache_dir() is None
    monkeypatch.setenv("AIEHLC_FAST", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert bm._i32_cache_dir() == (tmp_path / "aiehlc" / "i32").resolve()
