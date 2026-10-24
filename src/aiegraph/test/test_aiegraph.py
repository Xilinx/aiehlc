###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Tests for the ``aiegraph`` facade.

    python3 -m pytest -q src/aiegraph/test          # after `pip install -e .`

Unit tests need only onnx. ``deployed`` tests adopt ``worklocal/tvmrelay_deploy``
(deploy_flow.py output) read-only and skip without it. Nothing runs on a board.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import aiegraph as ag  # noqa: E402
from aiegraph.plan import group_by_source, parse_offload  # noqa: E402

DEPLOY = REPO / "worklocal" / "tvmrelay_deploy"
STEM = REPO / "src" / "aietensorop" / "conv2dstem" / "conv2dstem_uk.cc"
CONV3 = REPO / "example" / "userkernel" / "conv3x3_requant_ref.c"
deployed = pytest.mark.skipif(not (DEPLOY / "layers" / "manifest.json").is_file(),
                              reason="needs worklocal/tvmrelay_deploy")


# ── offload specs ──────────────────────────────────────────────────────────

def test_parse_offload_spellings(tmp_path):
    a, b = tmp_path / "a.cc", tmp_path / "b.cc"
    a.write_text(""), b.write_text("")
    assert parse_offload(f"layer1={a}") == [("layer1", a.resolve())]
    assert parse_offload(f"layer1={a}; layer4-9={b}") == [("layer1", a.resolve()),
                                                          ("layer4-9", b.resolve())]
    assert parse_offload(str(a)) == [("auto", a.resolve())]
    assert parse_offload(("layer1", a)) == [("layer1", a.resolve())]
    assert parse_offload({"layer1": a, 4: b}) == [("layer1", a.resolve()), (4, b.resolve())]
    assert len(parse_offload([f"layer1={a}", ("layer4", b)])) == 2
    assert parse_offload(None) == []


def test_parse_offload_errors(tmp_path):
    with pytest.raises(ag.AiegraphError, match="does not exist"):
        parse_offload("layer1=/no/such.cc")
    with pytest.raises(ag.AiegraphError, match="cannot read"):
        parse_offload(3.5)


def test_group_by_source(tmp_path):
    a, b = tmp_path / "a.cc", tmp_path / "b.cc"
    groups = group_by_source([("layer1", a), ("layer4", a), ("auto", b), ("layer2", b)],
                             lambda r: [int(str(r).replace("layer", ""))])
    assert groups == {a: [1, 4], b: None}


# ── models ─────────────────────────────────────────────────────────────────

def _onnx(path, name="input", shape=(1, 3, 224, 224)):
    import onnx
    from onnx import TensorProto, helper
    node = helper.make_node("Relu", [name], ["y"])
    g = helper.make_graph([node], "t", [helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))])
    onnx.save(helper.make_model(g), str(path))
    return path


def test_from_onnx_renames_input(tmp_path, monkeypatch):
    import onnx
    monkeypatch.setenv("AIEGRAPH_CACHE", str(tmp_path / "cache"))
    m = ag.from_onnx(_onnx(tmp_path / "net.onnx"))
    assert m.name == "net" and m.input_shape == [1, 3, 224, 224] and m.ops == {"Relu": 1}
    proto = onnx.load(str(m.onnx_path))
    assert proto.graph.input[0].name == "data" and proto.graph.node[0].input[0] == "data"
    assert "next: ag.convert" in repr(m)
    assert ag.convert is ag.embedded_convert and callable(m.convert)


def test_load_rejects_wrong_input(tmp_path):
    with pytest.raises(ag.AiegraphError, match=r"supports image classifiers"):
        ag.from_onnx(_onnx(tmp_path / "x.onnx", "data", (1, 3, 32, 32)))
    with pytest.raises(ag.AiegraphError, match="unknown model"):
        ag.download_model("resnet9000")
    with pytest.raises(ag.AiegraphError, match="don't know how to load"):
        ag.load("no_such_model")


def test_current_without_model():
    from aiegraph import embedded
    old, embedded._CURRENT = embedded._CURRENT, None
    try:
        with pytest.raises(ag.AiegraphError, match="embedded_convert"):
            ag.offload_check("layer1", "k.cc")
    finally:
        embedded._CURRENT = old


def test_compile_aie_needs_offload():
    with pytest.raises(ag.AiegraphError, match="needs offload"):
        ag.compile("resnet18", "aie")


def test_convert_refuses_foreign_workdir(tmp_path):
    (tmp_path / "keep.txt").write_text("user data")
    m = ag.Model("x", _onnx(tmp_path / "x.onnx", "data"), "onnx", [1, 3, 224, 224])
    with pytest.raises(ag.AiegraphError, match="not written by aiegraph"):
        ag.embedded_convert(m, workdir=tmp_path)
    assert (tmp_path / "keep.txt").is_file()


# ── regions and kernel contracts ───────────────────────────────────────────

def _region(name, shape, **kw):
    from aiegraph.region import Region
    return Region(name=name, node=name + "_fwd", op=kw.get("op", "conv2d"), weight_shape=shape,
                  weight_dtype="int8", quantization="per_channel", input_layout="NCHW",
                  weight_layout="OIHW", layer=kw.get("layer"), calls=kw.get("calls", []))


def test_region_names():
    from aiegraph.region import _canonical
    assert _canonical("/layer1/layer1.0/conv1/Conv") == "layer1.0.conv1"
    assert _canonical("/layer2/layer2.0/downsample/downsample.0/Conv") == "layer2.0.downsample.0"
    assert _canonical("/conv1/Conv") == "conv1" and _canonical("/fc/Gemm") == "fc"
    assert _canonical("resnetv15_stage1_conv0_fwd") == "resnetv15_stage1_conv0"


def test_gluon_aliases_follow_conv_index_not_node_order():
    from aiegraph.region import _gluon_aliases
    k3, k1 = (128, 64, 3, 3), (128, 64, 1, 1)
    rs = [_region("resnetv15_conv0", (64, 3, 7, 7)), _region("resnetv15_stage2_conv0", k3),
          _region("resnetv15_stage2_conv2", k1),       # ONNX node order puts it here
          _region("resnetv15_stage2_conv1", k3), _region("resnetv15_stage2_conv3", k3),
          _region("resnetv15_stage2_conv4", k3), _region("resnetv15_dense0", (1000, 512))]
    _gluon_aliases(rs)
    assert [r.aliases for r in rs] == [["conv1"], ["layer2.0.conv1"], ["layer2.0.downsample.0"],
                                       ["layer2.0.conv2"], ["layer2.1.conv1"], ["layer2.1.conv2"],
                                       ["fc"]]


def test_regions_find():
    from aiegraph.region import Regions
    rs = Regions([_region("a", (1, 1, 3, 3), calls=[1]), _region("b", (1, 1, 3, 3), calls=[4])])
    rs[0].aliases, rs[1].aliases = ["layer1.0.conv1"], ["layer1.1.conv1"]
    assert [r.name for r in rs.find("layer1.0.conv1")] == ["a"]
    assert [r.name for r in rs.find("layer1")] == ["a", "b"]           # prefix = everything under
    assert [r.name for r in rs.find("layer1.*.conv1")] == ["a", "b"]
    assert [r.name for r in rs.find("call[04]")] == ["b"]
    assert rs[0].label == "layer1.0.conv1 (a)"
    with pytest.raises(ag.AiegraphError, match="no region matches"):
        rs.find("layer9")


def test_contract_validation():
    from aiegraph.contract import make_contract
    with pytest.raises(ag.AiegraphError, match="unknown constraint"):
        make_contract("k", STEM, "conv2d", constraints={"layout": "NCHW"})
    with pytest.raises(ag.AiegraphError, match="op="):
        make_contract("k", STEM, "matmul")
    with pytest.raises(ag.AiegraphError, match="contradicts"):
        make_contract("conv2dstem", STEM, "dense")
    with pytest.raises(ag.AiegraphError, match="contradicts.*dtype"):
        make_contract("conv2dstem", STEM, "conv2d", constraints={"dtype": "uint8"})
    k = make_contract("conv2dstem", STEM, "conv2d", "a78",
                      {"input_layout": "NCHW", "dtype": "int8", "quantization": "per_channel"})
    assert k.backend == "aie:conv2dstem" and k.spec.name == "conv2dstem"


def test_check_region():
    from aiegraph.contract import check_region, make_contract
    k = make_contract("conv2dstem", STEM, "conv2d", constraints={"quantization": "per_tensor"})
    assert check_region(k, _region("x", (64, 3, 7, 7), layer=1)) == \
        ["quantization is per_channel, contract says per_tensor"]
    assert "no TVM call" in check_region(k, _region("x", (64, 3, 7, 7)))[-1]


def test_doctor_reports():
    d = ag.doctor()
    names = [c.name for c in d.checks]
    assert names[0] == "aiehlc checkout" and "TVM 0.16 (Relay)" in names
    assert "ready for:" in repr(d)


# ── on the deployed network ────────────────────────────────────────────────

@deployed
def test_open_layers_and_resolve():
    emod = ag.open(DEPLOY)
    assert ag.current() is emod
    names = [l.name for l in emod.layers]
    assert names[:2] == ["layer0", "layer1"]
    assert emod.layer("layer1").label.startswith("conv7x7/2")
    assert emod.resolve("layer4-9") == [4, 5, 6, 7, 8, 9]
    assert emod.resolve("call[08]") == [4]
    assert emod.resolve(["layer1", 4, "layer4"]) == [1, 4]
    assert 9 in emod.resolve("conv3x3/2*")
    with pytest.raises(ag.AiegraphError, match="no layer matches"):
        emod.resolve("layer99")
    assert "33 layers, 41 calls" in repr(emod)


@deployed
def test_detect_and_check_read_only():
    emod = ag.open(DEPLOY)
    assert ag.detect(CONV3).matches == [4, 9, 14, 16, 21, 23, 28]
    part = (DEPLOY / "layers" / "partition.json").read_text()
    rep = emod.offload_check("layer1", STEM)
    assert rep.ok and rep.ready_layers == [1], rep
    assert "BIT-IDENTICAL" in rep.emulation and "next: emod.offload" in repr(rep)
    bad = emod.offload_check(("layer4", STEM))
    assert not bad.ok and "stopped at detect" in repr(bad)
    with pytest.raises(ag.AiegraphError, match="layer4: stopped at detect"):
        bad.raise_for_status()
    # check never writes outside <workdir>/aiek/
    assert (DEPLOY / "layers" / "partition.json").read_text() == part
    assert json.loads((DEPLOY / "aiek" / "report.json").read_text())["kernels"]


@deployed
def test_regions_map_by_weight_fingerprint():
    emod = ag.open(DEPLOY)
    rs = emod.regions
    assert len(rs) == 21 and all(r.layer is not None for r in rs)    # 20 convs + fc
    (stem,) = rs.find("conv1")
    assert stem.layer == 1 and stem.calls == [1]
    (a,) = rs.find("layer1.0.conv1")
    (b,) = rs.find("layer1.1.conv1")
    assert a.layer == b.layer and b.label in a.siblings               # one TVM kernel, two convs
    assert len(rs.find("layer2")) == 5                                 # incl. downsample.0


@deployed
def test_check_offload_by_region_read_only():
    emod = ag.open(DEPLOY)
    emod.register_kernel(name="conv2dstem", source=STEM, op="conv2d", target="a78",
                         constraints={"input_layout": "NCHW", "weight_layout": "OIHW",
                                      "dtype": "int8", "quantization": "per_channel"})
    part = (DEPLOY / "layers" / "partition.json").read_text()
    ok = emod.check_offload(region="conv1", kernel="conv2dstem")
    assert ok.ok and "BIT-IDENTICAL" in ok.summary()
    bad = emod.check_offload(region="layer1.0.conv1", kernel="conv2dstem")
    assert not bad.ok and "stopped at detect" in bad.summary() and "move together" in bad.summary()
    with pytest.raises(ag.AiegraphError, match="no kernel registered"):
        emod.check_offload(region="conv1", kernel="nope")
    with pytest.raises(ag.AiegraphError, match="backend="):
        emod.assign(region="conv1", backend="gpu")
    assert (DEPLOY / "layers" / "partition.json").read_text() == part
