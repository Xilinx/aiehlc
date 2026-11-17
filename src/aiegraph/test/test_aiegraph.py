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
import re
import shutil
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


def test_download_model_returns_convertible_model(tmp_path, monkeypatch):
    import os
    from aiegraph import model as agm
    monkeypatch.setenv("AIEGRAPH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(agm, "_fetch", lambda name: _onnx(tmp_path / "resnet18-v1-7.onnx", "data"))
    m = ag.download_model("resnet18")
    assert isinstance(m, ag.Model) and m.name == "resnet18-v1-7" and m.source == "onnx-zoo"
    assert callable(m.convert) and ag.load(m) is m
    assert os.fspath(m) == str(m.onnx_path) and Path(m) == m.onnx_path   # path-like, as before


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


# ── offload_AI_proto, with a stand-in for `claude -p` ──────────────────────

_FAKE_CLAUDE = r'''#!{py}
import json, os, pathlib, shutil, sys
cwd = pathlib.Path.cwd()
with open(cwd / "fake_calls.txt", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
assert sys.stdin.read().strip(), "the prompt comes on stdin"
if os.environ.get("FAKE_CLAUDE_MODE") == "port":
    (entry,) = [p for p in cwd.iterdir() if p.stem.endswith("_l4")]
    shutil.copy({conv3!r}, entry)
if os.environ.get("FAKE_CLAUDE_MODE") == "fill":       # ai_offload_prototype: the solved layer-1 file
    shutil.copy(os.environ["FAKE_CLAUDE_SOLUTION"], cwd / "aiek_layer01.cc")
print(json.dumps({{"type": "assistant", "message": {{"content": [
    {{"type": "tool_use", "name": "Edit", "input": {{"file_path": "x.cc"}}}}]}}}}))
print(json.dumps({{"type": "result", "session_id": "s-1", "result": "done"}}))
'''


@pytest.fixture
def fake_claude(tmp_path):
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir()
    exe.write_text(_FAKE_CLAUDE.format(py=sys.executable, conv3=str(CONV3)))
    exe.chmod(0o755)
    return str(exe)


@deployed
def test_ai_proto_ready_kernel_is_returned_as_is(fake_claude, tmp_path):
    emod = ag.open(DEPLOY)
    res = emod.offload_AI_proto("layer1", STEM, claude=fake_claude, out_dir=tmp_path / "p")
    assert res.ok and res.rounds == 0 and Path(res) == STEM
    assert not (tmp_path / "p").exists()


@deployed
def test_ai_proto_ports_and_verifies(fake_claude, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "port")
    emod = ag.open(DEPLOY)
    before = STEM.read_bytes()
    res = emod.offload_AI_proto("layer4", STEM, claude=fake_claude, out_dir=tmp_path / "p")
    assert res.ok and res.rounds == 1 and res.layers == [4], res
    assert Path(res).parent == tmp_path / "p" and Path(res).name == "conv2dstem_uk_l4.cc"
    assert (tmp_path / "p" / "conv2dstem.cc").is_file()          # the dir was staged flat
    assert STEM.read_bytes() == before
    assert emod.offload_check("layer4", res).ok                    # path-like
    argv = json.loads((tmp_path / "p" / "fake_calls.txt").read_text().splitlines()[0])
    tools = argv[argv.index("--allowedTools") + 1]
    assert f"Edit(/{tmp_path / 'p'}/**)" in tools and "-m aiegraph check" in tools


@deployed
def test_ai_proto_gives_up_after_rounds(fake_claude, tmp_path):
    emod = ag.open(DEPLOY)
    res = emod.offload_AI_proto("layer4", STEM, claude=fake_claude, out_dir=tmp_path / "p",
                                rounds=2)
    assert not res.ok and res.rounds == 2 and "NOT READY" in repr(res)
    calls = (tmp_path / "p" / "fake_calls.txt").read_text().splitlines()
    assert "--resume" not in calls[0] and '"--resume", "s-1"' in calls[1]
    with pytest.raises(ag.AiegraphError):
        res.raise_for_status()


# ── offload_prototype ───────────────────────────────────────────────────────

def test_offload_prototype_format_needs_no_model():
    fmt = ag.offload_prototype()
    assert isinstance(fmt, ag.KernelFormat)
    for word in ("AIEK_ENTRY", "AIEK_SPEC", "AIEK_REFERENCE", "TVM call order", "require="):
        assert word in fmt


@deployed
def test_offload_prototype_layer_is_proven(tmp_path):
    emod = ag.open(DEPLOY)
    itf = emod.offload_prototype("layer1")
    assert [a.kind for a in itf.args][0] == "in" and itf.args[-1].kind == "out"
    assert "matches 1/1 call(s)" in itf.verified
    # the stem's second zero-point and unused param are all zero -> require=zero
    assert sum(1 for a in itf.args if a.requires == ("zero",)) == 2
    assert "AIEK_SPEC(aiek_layer01," in itf.spec()
    # the backbone: kernel frame + host run function + entry, compute left as TODO
    src = itf.kernel_source()
    for part in ("__global__ void aiek_layer01_kernel(", "aie::shape<AK_H, AK_W, AK_CA>",
                 "aie::load_buffer(x_t", "aie::store_buffer(y_t", "int aiek_layer01_run(",
                 "device.partition(", "AIEK_ENTRY(aiek_layer01)", "TODO(compute)"):
        assert part in src, part
    assert "kernel + host + entry backbone" in itf.generator
    # conv + requantize: the reference is complete, so offload_check passes as written
    assert itf.requant and "TODO" not in itf.reference_source() and "return AIEK_OK;" in itf.reference_source()
    files = itf.write(tmp_path)
    names = {p.name for p in files}
    assert {"aiek_layer01.cc", "aiek_layer01_ref.c", "test_aiek_layer01.cc"} <= names
    # the test: an aiehlc entry file (main() + the kernel spliced in) over the entry
    test = (tmp_path / "test_aiek_layer01.cc").read_text()
    for part in ('#include "aiek_layer01.cc"', "int main()", "aiek_layer01(aiekt_args, AIEKT_NARGS)",
                 "TEST PASS", '__has_include("aiek_layer01_test_data.h")',
                 # timed: every call on the clock, first apart, PMU split, one summary line
                 "aiekt_ms(t0, aiekt_now())", "PERF SUMMARY", "__Runtime_phase_cycles",
                 "#define AIEKT_MACS 118013952.0"):
        assert part in test, part
    assert "__global__" not in test and "<<<" not in test  # silent frontend traps
    if "aiek_layer01_test_data.h" in names:  # a make TRACE=1 deploy: real tensors + golden
        hdr = (tmp_path / "aiek_layer01_test_data.h").read_text()
        assert hdr.count("#define AIEKT_DATA_") == len(itf.args)
        assert all((tmp_path / inc).is_file() for inc in re.findall(r'#include "(l\d+_\w+\.h)"', hdr))
    assert not [f for f in ag.lint(tmp_path / "aiek_layer01.cc").findings if f.level == "error"]
    with pytest.raises(ag.AiegraphError, match="already exist"):
        itf.write(tmp_path)
    # what detect sees in the written file is the same signature
    rep = emod.detect(tmp_path / "aiek_layer01.cc")
    assert 1 in rep.matches


def test_offload_prototype_test_without_dumps(tmp_path):
    from aiegraph.prototype import OffloadPrototype, ProtoArg
    p = OffloadPrototype(layer=7, symbol="s", dir="07_x", calls=[7], label="op", entry="aiek_x", op="op",
                         match="s", args=[ProtoArg(0, "in", "int8", (1, 8), "ifm", "activation"),
                                          ProtoArg(1, "param", "int32", (8,), "bias", "bias", requires=("zero",)),
                                          ProtoArg(2, "out", "uint8", (1, 8), "ofm", "output")])
    dumps, why = p.test_data()
    assert dumps is None and "TRACE=1" in why
    files = p.write(tmp_path)
    assert {f.name for f in files} == {"aiek_x.cc", "aiek_x_ref.c", "test_aiek_x.cc"}
    test = (tmp_path / "test_aiek_x.cc").read_text()
    assert "(int32_t)aiekt_rand(0LL, 0LL)" in test  # require=zero honoured by the synthetic fill
    assert "#define AIEKT_MACS 0.0" in test and "PERF SUMMARY" in test  # timed, no conv -> no GOPS
    assert p.write(tmp_path / "no_test", test=False) and not (tmp_path / "no_test" / "test_aiek_x.cc").exists()


@deployed
def test_offload_prototype_multi_call_layer():
    itf = ag.open(DEPLOY).offload_prototype(4)
    assert len(itf.calls) == 2 and "matches 2/2 call(s)" in itf.verified


@deployed
def test_sources_names_each_layers_code():
    emod = ag.open(DEPLOY)
    text = str(emod.sources())
    l0 = emod.layers[0]
    assert l0.placement == "cpu" and l0.source == f"layers/{l0.dir}/{l0.dir}.c"
    assert l0.source in text


def test_offload_reset_puts_layers_back_on_cpu(tmp_path):
    from aiegraph.embedded import EmbeddedModel
    (tmp_path / "layers").mkdir()
    rows = [{"index": 1, "dir": "01_x", "symbol": "s1", "target": "aie", "entry": "aie_s1",
             "replaces": "s1", "archive": "a.a", "user_kernel": "k1"},
            {"index": 2, "dir": "02_y", "symbol": "s2", "target": "cpu"}]
    (tmp_path / "layers" / "partition.json").write_text(json.dumps({"layers": rows}))
    emod = EmbeddedModel(tmp_path)
    assert emod.offload_reset() == ["layer1"]
    doc = json.loads((tmp_path / "layers" / "partition.json").read_text())
    assert [r["target"] for r in doc["layers"]] == ["cpu", "cpu"]
    assert doc["layers"][0]["replaces"] is None and doc["aie_count"] == 0
    assert emod.offload_reset() == []


def test_offload_false_puts_one_layer_back_on_cpu(tmp_path, monkeypatch):
    from aiegraph.embedded import EmbeddedModel
    (tmp_path / "layers").mkdir()
    rows = [{"index": 1, "dir": "01_x", "target": "aie", "entry": "aie_s1"},
            {"index": 4, "dir": "04_y", "target": "aie", "entry": "aie_s4"}]
    (tmp_path / "layers" / "partition.json").write_text(json.dumps({"layers": rows}))
    emod = EmbeddedModel(tmp_path)
    monkeypatch.setattr(emod, "resolve", lambda ref: [int(str(ref).replace("layer", ""))])
    assert emod.offload("layer1", False) == ["layer1"]
    doc = json.loads((tmp_path / "layers" / "partition.json").read_text())
    assert [r["target"] for r in doc["layers"]] == ["cpu", "aie"]     # layer4 untouched
    assert emod.offload("layer1", False) == []                        # already on the CPU


# ── board run (offline: a fake board script) ───────────────────────────────

_CONSOLE = """[aie-offload] layer 01 conv ENTER
[aie-offload] layer 01 conv EXIT rc=0 hash=1234
  1. class=258  Samoyed                        logit=12.0173
  2. class=279  Arctic fox                     logit=8.5263
resnet18: top1 class=258 logit=12.017338
  inference    1271.003 ms   <- graph_run()
device_teardown done
Test complete!
"""


def _fake_board(tmp_path, monkeypatch, console=_CONSOLE, rc=0):
    from aiegraph import board
    fake = tmp_path / "fake_board.py"
    fake.write_text("import os, sys\nsys.stdout.write(open(os.environ['FAKE_CONSOLE']).read())\n"
                    f"sys.exit({rc})\n")
    (tmp_path / "console.txt").write_text(console)
    monkeypatch.setenv("FAKE_CONSOLE", str(tmp_path / "console.txt"))
    seen = {}

    def cmd(b, elf, netload, reboot, args=()):
        seen.update(board=b.name, netload=netload, reboot=reboot,
                    real=board.board_command.__wrapped__(b, elf, netload, reboot, args))
        return [sys.executable, str(fake)]
    cmd.__wrapped__ = board.board_command
    monkeypatch.setattr(board, "board_command", cmd)
    monkeypatch.setattr(board, "_busy", lambda script: [])
    monkeypatch.setattr(board, "_envlocal", lambda d: {})
    elf = tmp_path / "main.elf"
    elf.write_bytes(b"\x7fELF")
    return elf, seen


def test_run_defaults_to_palmyra_netload(tmp_path, monkeypatch):
    from aiegraph.board import run_on_board
    monkeypatch.setenv("PALIP", "lab-host"), monkeypatch.setenv("BOARDNAME", "pal1")
    elf, seen = _fake_board(tmp_path, monkeypatch)
    rep = run_on_board(elf, verbose=False)
    assert rep.ok and rep.board == "palmyra" and rep.host == "lab-host", rep
    assert rep.top1 == (258, 12.017338) and rep.top5[1] == (279, "Arctic fox", 8.5263)
    assert rep.inference_ms == 1271.003 and (rep.aie_enter, rep.aie_exit) == (1, 1)
    assert seen["netload"] and "--netload" in seen["real"] and seen["real"][-1] == str(elf.resolve())
    assert Path(rep.log).read_text() == _CONSOLE


def test_run_vek385_by_host(tmp_path, monkeypatch):
    from aiegraph.board import board_env, board_spec, run_on_board
    monkeypatch.delenv("VEK385IP", raising=False)
    elf, seen = _fake_board(tmp_path, monkeypatch)
    rep = run_on_board(elf, ag.VEK385, "crimini2", verbose=False)
    assert rep.ok and rep.board == "vek385" and rep.host == "crimini2"
    assert not seen["netload"] and seen["reboot"]                      # powers off, no -nonreboot
    assert "-nonreboot" not in seen["real"] and "appvek385.py" in seen["real"][1]
    assert board_spec("VEK385").name == "vek385" and board_spec("pal").name == "palmyra"
    with pytest.raises(ag.AiegraphError, match="VEK385IP not set"):
        board_env(board_spec(ag.VEK385))
    with pytest.raises(ag.AiegraphError, match="unknown board"):
        board_spec("zcu102")
    with pytest.raises(ag.AiegraphError, match="no netboot"):
        run_on_board(elf, ag.VEK385, "crimini2", netload=True, verbose=False)


def test_run_failures(tmp_path, monkeypatch):
    from aiegraph import board
    monkeypatch.setenv("PALIP", "lab-host"), monkeypatch.setenv("BOARDNAME", "pal1")
    elf, _ = _fake_board(tmp_path, monkeypatch, console="[aie-offload] layer 01 ENTER\nwait_io TIMEOUT\n")
    rep = board.run_on_board(elf, verbose=False)
    assert not rep.ok and "TIMEOUT" in rep.reason
    with pytest.raises(ag.AiegraphError, match="console log"):
        rep.raise_for_status()
    monkeypatch.setattr(board, "_busy", lambda script: [4242])
    with pytest.raises(ag.AiegraphError, match="one board session at a time"):
        board.run_on_board(elf, verbose=False)


# ── per-layer profile (make PROFILE=1 rows) ────────────────────────────────

_PROF_CONSOLE = _CONSOLE.replace("device_teardown done", """PROF   0      0.500 ms  cpu divide_round_add_clip_cast
PROF   1      2.000 ms  aie nn_contrib_conv2d_NCHWc_subtract
PROF   2      1.000 ms  cpu max_pool2d
PROF   3      1.500 ms  cpu nn_contrib_conv2d_NCHWc_add
PROF   4      1.000 ms  cpu nn_contrib_conv2d_NCHWc_add
PROF sum      6.000 ms over 1 run(s)
device_teardown done""")


def _profiled_emod(tmp_path, monkeypatch):
    from aiegraph.embedded import EmbeddedModel, Layer
    emod = EmbeddedModel(tmp_path)
    layers = [Layer(0, "s0", "00_q", [0], "quantize", "", "cpu"),
              Layer(1, "s1", "01_c", [1], "conv7x7/2 ->uint8", "", "aie:k"),
              Layer(2, "s2", "02_p", [2], "max pool2d", "", "cpu"),
              Layer(3, "s3", "03_c", [3, 4], "conv3x3/1 ->int8", "", "cpu")]
    monkeypatch.setattr(EmbeddedModel, "layers", property(lambda self: layers))
    monkeypatch.setattr(emod, "resolve", lambda ref: [int(str(r).replace("layer", ""))
                                                      for r in (ref if isinstance(ref, list) else [ref])])
    return emod


def test_parse_console_profile_rows():
    from aiegraph.board import parse_console
    got = parse_console(_PROF_CONSOLE)["profile"]
    assert len(got) == 5 and got[1] == (1, 2.0, "aie", "nn_contrib_conv2d_NCHWc_subtract")


def test_profile_per_call_and_per_layer(tmp_path, monkeypatch):
    from aiegraph.board import parse_console
    from aiegraph.report import RunReport
    emod = _profiled_emod(tmp_path, monkeypatch)
    got = parse_console(_PROF_CONSOLE)
    emod.last_run = RunReport(ok=True, board="palmyra", host="h", elf="e", log="l",
                              inference_ms=got["inference_ms"], profile=got["profile"])
    p = emod.profile(ag.ALL)
    assert [r["layer"] for r in p.rows] == ["layer0", "layer1", "layer2", "layer3", "layer3"]
    assert p.total_ms == 6.0 and p.ms("aie") == 2.0 and round(p.rows[1]["pct"], 1) == 33.3
    by = emod.profile(ag.ALL, by="layer")
    l3 = [r for r in by.rows if r["layer"] == "layer3"][0]
    assert l3["calls"] == [3, 4] and l3["ms"] == 2.5 and "layer3" in repr(by)
    assert [r["call"] for r in emod.profile("layer3").rows] == [3, 4]
    assert [r["layer"] for r in emod.profile(ag.ALL, by="layer", top=2).rows] == ["layer3", "layer1"]


def test_profile_reads_log_and_needs_profile_build(tmp_path, monkeypatch, capsys):
    emod = _profiled_emod(tmp_path, monkeypatch)
    assert emod.profile(ag.ALL) is None
    assert "profiling not enabled -- no board run yet" in capsys.readouterr().out
    (tmp_path / "arm_build").mkdir()
    (tmp_path / "arm_build" / "run_vek385.log").write_text(_CONSOLE)
    assert emod.profile(ag.ALL) is None
    out = capsys.readouterr().out
    assert "profiling not enabled" in out and "emod.run(profile=True)" in out
    (tmp_path / "arm_build" / "run_vek385.log").write_text(_PROF_CONSOLE)
    p = emod.profile(ag.ALL)
    assert p.board == "vek385" and len(p.rows) == 5


def test_make_vars_change_forces_driver_rebuild(tmp_path):
    from frontend.tvmrelay import arm_build as ab
    for f in ("graph_driver.o", "main.o", "main.elf", "layer.o"):
        (tmp_path / f).write_text("x")
    assert ab.sync_make_vars(tmp_path, None) == [] and (tmp_path / "main.o").is_file()  # unchanged
    assert ab.sync_make_vars(tmp_path, {"PROFILE": 1}) == ["PROFILE=1"]
    assert not (tmp_path / "graph_driver.o").exists() and not (tmp_path / "main.elf").exists()
    assert (tmp_path / "layer.o").is_file() and ab.built_make_vars(tmp_path) == {"PROFILE": "1"}
    (tmp_path / "main.o").write_text("x")
    ab.sync_make_vars(tmp_path, {"PROFILE": "1"})
    assert (tmp_path / "main.o").is_file()                       # same vars: nothing deleted
    ab.sync_make_vars(tmp_path, None)
    assert not (tmp_path / "main.o").exists()                    # back to plain: rebuilt again


# ── AI_perf_analysis / AI_kernel_accelerate (synthetic layer, fake claude) ─────────

def _elf32(path, text=400, data=40, bss=4000):
    import struct
    ph = [(1, 0, 0, 0, text, text, 5, 1), (1, 0, 0x78000, 0, data, data + bss, 6, 1)]
    hdr = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9) + struct.pack("<HHIIIIIHHHHHH", 2, 0x108, 1, 0, 52, 0, 0,
                                                                  52, 32, len(ph), 0, 0, 0)
    path.write_bytes(hdr + b"".join(struct.pack("<8I", *p) for p in ph))


_PERF_SRC = """#include <stdint.h>
__global__ void k(aie::port<input_window_int8 *, aie::IFM, aie::shape<8>> x) { /* TODO(compute): */ }
int run(void) { aieArray device; aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 4); return -1; }
AIEK_ENTRY(aiek_t) { return 0; }
"""


def _perf_workdir(tmp_path, monkeypatch, host_cc="__Runtime_load_kernel_group_16t(d);\n"
                  "__Runtime_dma_bd_config(a);\n" * 1 + "__Runtime_teardown_all();\n"):
    from aiegraph.embedded import EmbeddedModel, Layer
    wd = tmp_path / "wd"
    adir = wd / "layers" / "01_c" / "aie"
    (adir / "build").mkdir(parents=True)
    (adir / "host.cc").write_text(host_cc)
    (adir / "routing.cc").write_text("XAie_StrmConnCctEnable(a);\n")
    (adir / "build" / "kernel.lst").write_text("MAC r1\n")
    _elf32(adir / "build" / "kernel")
    (wd / "aiek" / "aiek_t").mkdir(parents=True)
    (wd / "aiek" / "aiek_t" / "aiehlc.log").write_text("Set kernel_tiling out_rounds=14\n")
    src = tmp_path / "k" / "aiek_t.cc"
    src.parent.mkdir()
    src.write_text(_PERF_SRC)
    rows = [{"index": 1, "dir": "01_c", "target": "aie", "aie_dir": "layers/01_c/aie",
             "source": str(src), "user_kernel": "aiek_t"}]
    (wd / "layers" / "partition.json").write_text(json.dumps({"layers": rows}))
    (wd / "arm_build").mkdir()
    (wd / "arm_build" / "run_palmyra.log").write_text(
        "[aie-offload] layer 01 conv: ENTER\n[aie-offload] layer 01: EXIT rc=-4, out\n"
        "  inference    100.000 ms   <- graph_run()\n"
        "PROF   0      2.000 ms  cpu q\nPROF   1     77.244 ms  aie conv\n")
    emod = EmbeddedModel(wd)
    lay = [Layer(0, "s0", "00_q", [0], "quantize", "", "cpu"),
           Layer(1, "s1", "01_c", [1], "conv7x7/2 ->uint8", "", "aie:aiek_t", str(src))]
    monkeypatch.setattr(EmbeddedModel, "layers", property(lambda self: lay))
    monkeypatch.setattr(emod, "resolve", lambda ref: [int(str(ref).replace("layer", ""))])
    return emod, src, adir


def test_perf_facts_model_rules(tmp_path, monkeypatch):
    from aiegraph import aiperf
    emod, src, adir = _perf_workdir(tmp_path, monkeypatch)
    rep = emod.AI_perf_analysis("layer1", ai=False)
    f, m = rep.facts, rep.model
    assert f["elf"] == {"text": 400, "data": 40, "bss": 4000} and f["src"]["mesh"] == [4, 4]
    assert f["run"]["measured_ms"] == 77.244 and f["run"]["exit_rc"] == -4
    assert m["elf_words_per_core"] == 1110 and abs(m["elf_load_ms"] - 16 * 1110 * 372e-6) < 1e-3
    ids = [r["id"] for r in rep.rules]
    assert ids[:3] == ["R0", "R1", "R2"], ids
    assert (emod.workdir / "aiperf" / "layer1" / "report.md").is_file() and "AI_kernel_accelerate" in repr(rep)
    assert emod.AI_perf_analaysis == emod.AI_perf_analysis            # the alias
    with pytest.raises(ag.AiegraphError, match="runs on the CPU"):
        emod.AI_perf_analysis("layer0", ai=False)
    # control-packet ELF load prices at 2.9 ns/word, not 372
    f2 = aiperf.layer_facts(emod, "layer1")
    f2["host"]["ctrl_elf_load"] = True
    assert aiperf.cost_model(f2)["elf_load_ms"] < m["elf_load_ms"] / 100


_FAKE_PERF_CLAUDE = r'''#!{py}
import json, os, pathlib, sys
cwd = pathlib.Path.cwd()
with open(cwd / "fake_calls.txt", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
prompt = sys.stdin.read()
if "--resume" in sys.argv:                       # a gate failure coming back
    assert "Our gate rejected" in prompt
    print(json.dumps({{"type": "result", "session_id": "p-1", "result": "fixed the build"}}))
    sys.exit(0)
assert "kernelperfcheck" in prompt
if "Make one AIE-offloaded layer faster" in prompt:
    (entry,) = [p for p in cwd.glob("*.cc")]
    entry.write_text("#pragma control_plan_op_control_packet\n" + entry.read_text())
    text = "S1: control-packet ELF load"
else:
    rep = {{"summary": "launch overhead, no compute", "computing": False,
            "bottlenecks": [{{"what": "ELF over MMIO", "evidence": "host.cc", "est_ms": 37.0}}],
            "suggestions": [{{"id": "S1", "change": "control packets", "where": "pragma",
                              "expected_saving_ms": 36.0, "risk": "low", "effort": "low"}}],
            "predicted_ms_after": 40.0}}
    text = "analysis\n```json\n" + json.dumps(rep) + "\n```"
print(json.dumps({{"type": "result", "session_id": "p-1", "result": text}}))
'''


@pytest.fixture
def fake_perf_claude(tmp_path):
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir()
    exe.write_text(_FAKE_PERF_CLAUDE.format(py=sys.executable))
    exe.chmod(0o755)
    return str(exe)


def test_ai_perf_analysis_parses_claude_report(tmp_path, monkeypatch, fake_perf_claude):
    emod, _src, _ = _perf_workdir(tmp_path, monkeypatch)
    rep = emod.AI_perf_analysis("layer1", claude=fake_perf_claude, quiet=True)
    assert rep.ai["summary"] == "launch overhead, no compute" and rep.predicted_ms_after == 40.0
    assert rep.suggestions[0]["id"] == "S1" and rep.session == "p-1"
    argv = json.loads((emod.workdir / "aiperf" / "layer1" / "fake_calls.txt").read_text().splitlines()[0])
    assert argv[argv.index("--allowedTools") + 1] == "Read,Grep,Glob"      # analysis is read-only
    assert json.loads((emod.workdir / "aiperf" / "layer1" / "report.json").read_text())["ai"]["suggestions"]


def test_ai_kernel_accelerate_gates_and_predicts(tmp_path, monkeypatch, fake_perf_claude):
    from aiegraph import aiperf
    emod, src, _adir = _perf_workdir(tmp_path, monkeypatch)
    rep = emod.AI_perf_analysis("layer1", claude=fake_perf_claude, quiet=True)
    before = src.read_text()
    wl = tmp_path / "newbuild"                      # what aiehlc would emit for the edited copy
    (wl / "build").mkdir(parents=True)
    (wl / "host.cc").write_text("__Runtime_load_kernel_group_16t_ctrl(d);\n__Runtime_teardown_all();\n")
    (wl / "routing.cc").write_text("")
    _elf32(wl / "build" / "kernel")
    (wl / "aiehlc.log").write_text("[aiehlc] Detected #pragma control_plan_op_control_packet\n")
    verdicts = iter(["aiehlc build failed: boom", ""])

    def gate(emod_, idx, entry, dest, verbose):
        assert "control_plan_op_control_packet" in entry.read_text()
        return None, {"ok": True, "worklocal": str(wl), "log": str(wl / "aiehlc.log")}, next(verdicts)
    monkeypatch.setattr(aiperf, "_gate", gate)
    res = emod.AI_kernel_accelerate("layer1", rep, claude=fake_perf_claude, quiet=True)
    assert res.ok and res.rounds == 2 and src.read_text() == before            # original untouched
    assert Path(res).parent == emod.workdir / "aiopt" / "layer1_aiek_t" and Path(res).name == "aiek_t.cc"
    assert res.model_after["elf_load_ms"] < rep.model["elf_load_ms"] / 100
    assert res.predicted_ms == round(77.244 - res.saved_ms, 3) and 6.5 < res.saved_ms < 6.7   # 16x1110 words x 372 ns
    calls = (Path(res).parent / "fake_calls.txt").read_text().splitlines()
    assert "--resume" not in calls[0] and '"--resume", "p-1"' in calls[1]      # gate failure went back
    assert "predicted" in repr(res) and "measured before 77.244 ms" in repr(res)


def test_host_facts_count_calls_not_declarations():
    from aiegraph.aiperf import _call_sites
    text = ('extern "C" void __Runtime_teardown_all();\n'
            "void __Runtime_teardown_all(void);\nXAie_DmaDesc __Runtime_dma_bd_config(int a);\n"
            "  XAie_DmaDesc v1 = __Runtime_dma_bd_config(d, 1);\n  __Runtime_startio(x);\n"
            "  return __Runtime_wait(d);\n")
    assert _call_sites(text) == {"__Runtime_dma_bd_config": 1, "__Runtime_startio": 1, "__Runtime_wait": 1}


def test_perf_report_round_trips_from_disk(tmp_path, monkeypatch, fake_perf_claude):
    emod, _src, _ = _perf_workdir(tmp_path, monkeypatch)
    rep = emod.AI_perf_analysis("layer1", claude=fake_perf_claude, quiet=True)
    back = ag.PerfReport.load(rep.dest)                       # a new session: from the saved dir
    assert back.ai == rep.ai and back.facts == rep.facts and back.suggestions == rep.suggestions
    assert "analysis" in back.text and back.measured_ms == 77.244
    with pytest.raises(ag.AiegraphError, match="no saved report"):
        ag.PerfReport.load(tmp_path / "nowhere")


def test_sources_full_paths(tmp_path, monkeypatch):
    from aiegraph import _env
    from aiegraph.embedded import EmbeddedModel, Layer
    emod = EmbeddedModel(tmp_path)
    lay = [Layer(0, "s0", "00_q", [0], "q", "", "cpu", "layers/00_q/00_q.c"),
           Layer(1, "s1", "01_c", [1], "c", "", "aie:k", "src/aietensorop/conv2dstem/conv2dstem_uk.cc"),
           Layer(2, "s2", "02_c", [2], "c", "", "aie:k2", "aiopt/layer2_k/k.cc")]   # copy inside the workdir
    (tmp_path / "aiopt" / "layer2_k").mkdir(parents=True)
    (tmp_path / "aiopt" / "layer2_k" / "k.cc").write_text("")
    monkeypatch.setattr(EmbeddedModel, "layers", property(lambda self: lay))
    assert emod.source_path(lay[0]) == (tmp_path / "layers/00_q/00_q.c").resolve()
    assert emod.source_path(lay[1]) == (_env.repo_root() / "src/aietensorop/conv2dstem/conv2dstem_uk.cc").resolve()
    assert emod.source_path(lay[2]) == (tmp_path / "aiopt/layer2_k/k.cc").resolve()
    text = emod.sources()
    assert str(tmp_path / "layers/00_q/00_q.c") in text and str(_env.repo_root() / "src/aietensorop/conv2dstem/conv2dstem_uk.cc") in text
    assert "  src/aietensorop/conv2dstem/conv2dstem_uk.cc" in emod.sources(full=False)


def test_stage_never_deletes(tmp_path):
    from aiegraph.aiproto import _stage
    dest = tmp_path / "aiopt" / "layer1_k"
    dest.mkdir(parents=True)
    (dest / "k.cc").write_text("the kernel layer 1 runs")
    with pytest.raises(ag.AiegraphError, match="does not exist"):          # stale path: nothing touched
        _stage(tmp_path / "gone" / "k.cc", dest, "k.cc")
    assert (dest / "k.cc").read_text() == "the kernel layer 1 runs"
    with pytest.raises(ag.AiegraphError, match="contains the source"):     # restage over itself
        _stage(dest / "k.cc", dest, "k.cc")
    assert (dest / "k.cc").read_text() == "the kernel layer 1 runs"
    other = tmp_path / "src" / "k.cc"
    other.parent.mkdir()
    other.write_text("new")
    _stage(other, dest, "k.cc")                                             # earlier copy moved aside
    (prev,) = (tmp_path / "aiopt").glob("layer1_k.prev-*")
    assert (prev / "k.cc").read_text() == "the kernel layer 1 runs" and (dest / "k.cc").read_text() == "new"


def test_accelerate_an_accelerated_kernel_with_a_stale_report(tmp_path, monkeypatch, fake_perf_claude):
    """The 2026-11-02 loss: report made before an offload of the aiopt copy, its source
    path stale; staging must use the layer's CURRENT kernel and leave it intact."""
    from aiegraph import aiperf
    from aiegraph.embedded import EmbeddedModel, Layer
    emod, src, _ = _perf_workdir(tmp_path, monkeypatch)
    rep = emod.AI_perf_analysis("layer1", claude=fake_perf_claude, quiet=True)
    cur = emod.workdir / "aiopt" / "layer1_aiek_t" / "aiek_t.cc"           # layer 1 now runs a copy
    cur.parent.mkdir(parents=True)
    cur.write_text(_PERF_SRC + "// accelerated once\n")
    lay = [Layer(0, "s0", "00_q", [0], "quantize", "", "cpu"),
           Layer(1, "s1", "01_c", [1], "conv7x7/2 ->uint8", "", "aie:aiek_t", "aiopt/layer1_aiek_t/aiek_t.cc")]
    monkeypatch.setattr(EmbeddedModel, "layers", property(lambda self: lay))
    rep.facts["source"] = str(emod.workdir.parent / "aiopt" / "layer1_aiek_t" / "aiek_t.cc")   # stale, absent
    monkeypatch.setattr(aiperf, "_gate", lambda *a: (None, None, "aiehlc build failed: stop here"))
    res = emod.AI_kernel_accelerate("layer1", rep, claude=fake_perf_claude, quiet=True, rounds=1)
    assert cur.read_text().endswith("// accelerated once\n")               # the running kernel survived
    assert Path(res).parent.name == "layer1_aiek_t_r2" and res.original == cur
    assert "// accelerated once" in Path(res).read_text()                   # staged from the CURRENT kernel


# ── ai_offload_prototype: kemu + prototype_check + the AI loop ──────────────

SOLVED_L1 = REPO / "example" / "tileprogram" / "ccode" / "model" / "resnet18" / "layer1" / "aiek_layer01.cc"
traced = pytest.mark.skipif(not (DEPLOY / "arm_build" / "layeriohex").is_dir() or not SOLVED_L1.is_file(),
                            reason="needs a make TRACE=1 deploy and the solved layer-1 example")


@deployed
@traced
def test_kemu_judges_the_kernel_not_the_reference(tmp_path):
    from aiegraph.kemu import kemu
    proto = ag.open(DEPLOY).offload_prototype("layer1")
    assert kemu(proto, SOLVED_L1.parent).ok                      # vs TVM's own output
    proto.write(tmp_path / "backbone", test=False)
    r = kemu(proto, tmp_path / "backbone")                       # offload_check says READY here
    assert not r.ok and "returned -4" in r.detail
    bad = tmp_path / "bad"
    bad.mkdir()
    shutil.copy(SOLVED_L1.with_name("aiek_layer01_ref.c"), bad)
    (bad / "aiek_layer01.cc").write_text(SOLVED_L1.read_text().replace("(s - 1))", "(s - 2))"))  # rounding
    r = kemu(proto, bad)
    assert not r.ok and "differ from TVM" in r.detail
    proto.iohex_dir = ""                                         # no dumps: the reference is the golden
    r = kemu(proto, SOLVED_L1.parent)
    assert r.ok and r.golden == "reference"


@deployed
@traced
def test_ai_offload_prototype_fills_and_verifies(fake_claude, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "fill")
    monkeypatch.setenv("FAKE_CLAUDE_SOLUTION", str(SOLVED_L1))
    emod = ag.open(DEPLOY)
    res = emod.ai_offload_prototype("layer1", claude=fake_claude, out_dir=tmp_path / "p", build=False)
    assert res.ok and res.rounds == 1, res
    assert [g for g, ok, _ in res.verdict.rows] == ["todo", "kemu", "trace", "check"]
    assert Path(res) == tmp_path / "p" / "aiek_layer01.cc"
    argv = json.loads((tmp_path / "p" / "fake_calls.txt").read_text().splitlines()[0])
    tools = argv[argv.index("--allowedTools") + 1]
    assert f"Edit(/{tmp_path / 'p'}/**)" in tools and "-m aiegraph protocheck" in tools


@deployed
@traced
def test_ai_offload_prototype_gives_up_and_never_deletes(fake_claude, tmp_path):
    emod = ag.open(DEPLOY)
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "mine.cc").write_text("// the user's")
    res = emod.ai_offload_prototype("layer1", claude=fake_claude, out_dir=tmp_path / "p", rounds=2, build=False)
    assert not res.ok and res.rounds == 2 and "NOT CORRECT" in repr(res)
    assert dict((g, ok) for g, ok, _ in res.verdict.rows)["kemu"] is False
    prev = [d for d in tmp_path.iterdir() if d.name.startswith("p.prev-")]
    assert len(prev) == 1 and (prev[0] / "mine.cc").read_text() == "// the user's"
    with pytest.raises(ag.AiegraphError):
        res.raise_for_status()

