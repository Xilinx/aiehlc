###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""One-call group offload, step S1 (design docs/plans/2026-11-16-aiegraph-group-offload-design.md
§5.2 conv / linear): ``ag.step(..., call=, weights=g.w.as_bf16(layout=), wrap=, tag=)``,
``emod.offload_group`` -> ``layers/partition.json`` + ``layers/groups/*.bf16``, and
``arm_build.generate_driver`` emitting the step call in place of the TVM call, the weight in
``weights.bin`` in the kernel layout and no ``graph_bf16_expand`` for it.

x86 only, no TVM: a hand-made graph.json and a relay ``save_param_dict`` blob written here.
The bk-sdm-tiny match test runs only with AIEGRAPH_UNET_SLOW=1 and the tutorial deploy tree.
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

import aiegraph as ag  # noqa: E402
from aiegraph import _env  # noqa: E402
from aiegraph.callsite import WeightRef, weight_bytes  # noqa: E402
from aiegraph.embedded import EmbeddedModel  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402

_env.frontend("")
import frontend.tvmrelay.arm_build as ab  # noqa: E402

W = (np.arange(32, dtype=np.float32).reshape(8, 4) - 7.25) / 3.0
B = np.linspace(-1, 1, 8).astype(np.float32)
W2 = np.linspace(-2, 2, 64).astype(np.float32).reshape(8, 8)


def save_params(path, arrays):
    names = list(arrays)
    b = struct.pack("<QQQ", ab.PARAM_MAGIC, 0, len(names))
    for n in names:
        e = n.encode()
        b += struct.pack("<Q", len(e)) + e
    b += struct.pack("<Q", len(names))
    for n in names:
        a = np.ascontiguousarray(arrays[n], dtype="<f4")
        b += struct.pack("<QQii", ab.NDARRAY_MAGIC, 0, 1, 0) + struct.pack("<i", a.ndim)
        b += struct.pack("<BBH", 2, 32, 1) + struct.pack("<%dq" % a.ndim, *a.shape)
        b += struct.pack("<q", a.nbytes) + a.tobytes()
    path.write_bytes(b)


# x -> [0] conv(x, w, b) -> y -> [1] dense(y, w2) -> z
def net(tmp_path):
    spec = [("x", None, [], [1, 4]), ("w", None, [], [8, 4]), ("b", None, [], [8]),
            ("w2", None, [], [8, 8]),
            ("y", "nn_contrib_conv2d_NCHWc_add", ["x", "w", "b"], [1, 8]),
            ("z", "nn_contrib_dense_pack", ["y", "w2"], [1, 8])]
    ids = {s[0]: i for i, s in enumerate(spec)}
    nodes = []
    for name, fn, ins, _ in spec:
        if fn is None:
            nodes.append({"op": "null", "name": name, "inputs": []})
        else:
            nodes.append({"op": "tvm_op", "name": name, "inputs": [[ids[i], 0, 0] for i in ins],
                          "attrs": {"func_name": "tvmgen_default_fused_" + fn, "num_outputs": "1"}})
    n = len(nodes)
    g = {"nodes": nodes, "arg_nodes": [0, 1, 2, 3], "heads": [[ids["z"], 0, 0]],
         "node_row_ptr": list(range(n + 1)),
         "attrs": {"storage_id": ["list_int", [0, 1, 2, 3, 4, 5]],
                   "shape": ["list_shape", [s[3] for s in spec]],
                   "dltype": ["list_str", ["float32"] * n]}}
    (tmp_path / "net_graph.json").write_text(json.dumps(g))
    save_params(tmp_path / "net_params.bin", {"w": W, "b": B, "w2": W2})
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "partition.json").write_text(json.dumps({"layers": []}))
    return tmp_path, ids


def transpose_layout(a):
    return np.ascontiguousarray(a.T)


def conv_step(g, w, **kw):
    a = dict(reads=g.inputs[:1], writes=g.outputs, weights=w,
             consts=dict(cid=3, s=2), call=("cid", "args", "nargs", "wbf", "s"),
             wrap=("UP_B", "UP_E"), tag="aie kconv 3")
    a.update(kw)
    return ag.step("kconv", **a)


def gen(wd, out, monkeypatch):
    monkeypatch.setenv("AIEGRAPH_PARAMS_INPLACE", "1")
    monkeypatch.setenv("AIEGRAPH_WDTYPE", "bf16")
    out.mkdir(exist_ok=True)
    ab.generate_driver(wd / "net_graph.json", wd / "net_params.bin", out, verbose=False)
    return (out / "graph_driver.c").read_text(), (out / "weights.bin").read_bytes()


def run_body(txt):
    return txt.split("float *graph_run(void) {")[1]


# ── ag.step(call=...) contract ──────────────────────────────────────────────────
def test_step_call_validation():
    w = WeightRef("w", 1, (8, 4)).as_bf16()
    ag.step("k", weights=w, consts={"cid": 1}, call=("cid", "args", "nargs", "wbf"))
    with pytest.raises(AiegraphError, match="neither a const"):
        ag.step("k", consts={"cid": 1}, call=("cid", "args", "foo"))
    with pytest.raises(AiegraphError, match="passes wbf but the step has no weights"):
        ag.step("k", consts={"cid": 1}, call=("cid", "args", "wbf"))
    with pytest.raises(AiegraphError, match="does not pass wbf"):
        ag.step("k", weights=w, consts={"cid": 1}, call=("cid", "args"))
    with pytest.raises(AiegraphError, match=r"consts \['s'\] are not in call"):
        ag.step("k", consts={"cid": 1, "s": 2}, call=("cid", "args"))
    with pytest.raises(AiegraphError, match="repeat"):
        ag.step("k", consts={"cid": 1}, call=("cid", "args", "args"))
    with pytest.raises(AiegraphError, match="wrap"):
        ag.step("k", consts={"cid": 1}, call=("cid", "args"), wrap=("A",))
    with pytest.raises(AiegraphError, match="not callable"):
        WeightRef("w", 1, (8, 4)).as_bf16(layout="kacc")


# ── weights ────────────────────────────────────────────────────────────────────
def test_weight_bytes_rne_and_layout(tmp_path):
    wd, ids = net(tmp_path)
    ref = WeightRef("w", ids["w"], (8, 4))
    got = np.frombuffer(weight_bytes(ref.as_bf16(layout=transpose_layout), wd / "net_params.bin"),
                        dtype="<u2")
    assert np.array_equal(got, ab.f32_to_bf16_rne(W).reshape(8, 4).T.reshape(-1))
    # transform runs on fp32 before the rounding
    got2 = np.frombuffer(weight_bytes(ref.as_bf16(transform=lambda a: a * 2),
                                      wd / "net_params.bin"), dtype="<u2")
    assert np.array_equal(got2, ab.f32_to_bf16_rne(W * 2).reshape(-1))


def test_layout_that_is_not_a_permutation_raises(tmp_path):
    wd, ids = net(tmp_path)
    emod = EmbeddedModel(wd)
    g = emod.group([0], "c0")
    bad = WeightRef("w", ids["w"], (8, 4)).as_bf16(layout=lambda a: np.zeros_like(a))
    with pytest.raises(AiegraphError, match="not a permutation"):
        emod.offload_group(g, "k.cc", [conv_step(g, bad)])
    assert emod.groups == []          # nothing recorded


def test_weight_the_call_does_not_read_raises(tmp_path):
    wd, ids = net(tmp_path)
    emod = EmbeddedModel(wd)
    g = emod.group([0], "c0")
    other = WeightRef("w2", ids["w2"], (8, 8)).as_bf16()
    with pytest.raises(AiegraphError, match="not read by call"):
        emod.offload_group(g, "k.cc", [conv_step(g, other)])
    assert emod.groups == []


def test_args_abi_on_a_multi_call_group_raises(tmp_path):
    wd, ids = net(tmp_path)
    emod = EmbeddedModel(wd)
    g = emod.group([0, 1], "cd")
    st = ag.step("k", reads=g.inputs[:1], writes=g.outputs, consts={"cid": 1}, call=("cid", "args"))
    with pytest.raises(AiegraphError, match="one-call groups"):
        emod.offload_group(g, "k.cc", [st])


# ── driver ─────────────────────────────────────────────────────────────────────
def test_driver_step_call_and_inplace_weight(tmp_path, monkeypatch):
    wd, ids = net(tmp_path)
    base_txt, base_w = gen(wd, tmp_path / "base", monkeypatch)
    emod = EmbeddedModel(wd)
    g = emod.group([0], "c0")
    w = WeightRef("w", ids["w"], (8, 4)).as_bf16(layout=transpose_layout)
    plan = emod.offload_group(g, "k.cc", [conv_step(g, w)])
    rec = emod.groups[0]["steps"][0]
    assert rec["decl"] == "int kconv(int cid, void *args, int nargs, const uint16_t *wbf, int s);"
    assert rec["nargs"] == 4 and rec["weights"]["param"] == "w" and plan.steps[0]["at"] == 0
    assert (wd / rec["weights"]["file"]).stat().st_size == 64

    txt, wbin = gen(wd, tmp_path / "out", monkeypatch)
    offs = ab.param_offsets(wd / "net_params.bin")
    # same blob layout as without the group (a bf16 region per param, 64 B aligned)
    off_w = base_txt.index("(g_params + ") and int(
        base_txt.split("tensors[%d].data = 0;   /* w: bf16 @" % ids["w"])[1].split("+")[0])
    head, body = txt.split("float *graph_run(void) {")
    assert rec["decl"] in head.split("static uint8_t kernel_ws[")[0]
    assert ("    UP_B(0);\n    kconv(3, args, 4, (const uint16_t *)(g_params + %d), 2);" % off_w) in body
    assert "    UP_E(0);\n    GRAPH_PROF_END(0, \"aie kconv 3\");" in body
    assert "tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add(args" not in body
    assert "tvmgen_default_fused_nn_contrib_dense_pack(args" in body       # call 1 untouched
    # w read in place: no expand for it, the bias and w2 still expanded
    assert "(g_params + %d), 32ULL)" % off_w not in body
    assert "    tensors[%d].data = 0;" % ids["w"] in body
    assert body.count("graph_bf16_expand(") == run_body(base_txt).count("graph_bf16_expand(") - 1 == 2
    # weights.bin: w in the kernel layout, everything else as without the group
    want = ab.f32_to_bf16_rne(W).reshape(8, 4).T.astype("<u2").tobytes()
    assert wbin[off_w:off_w + 64] == want
    assert wbin[:off_w] == base_w[:off_w] and wbin[off_w + 64:] == base_w[off_w + 64:]
    assert len(offs) == 3


def test_record_only_groups_do_not_change_the_driver(tmp_path, monkeypatch):
    wd, ids = net(tmp_path)
    base_txt, base_w = gen(wd, tmp_path / "base", monkeypatch)
    emod = EmbeddedModel(wd)
    g = emod.group([0], "c0")
    emod.offload_group(g, "k.cc", [ag.group_step("k", reads=g.inputs[:1], writes=g.outputs)])
    txt, wbin = gen(wd, tmp_path / "out", monkeypatch)
    assert txt == base_txt and wbin == base_w


def test_two_signatures_for_one_step_raise(tmp_path, monkeypatch):
    wd, ids = net(tmp_path)
    emod = EmbeddedModel(wd)
    g0, g1 = emod.group([0], "c0"), emod.group([1], "d1")
    emod.offload_group(g0, "k.cc", [ag.step("k", reads=g0.inputs[:1], writes=g0.outputs,
                                            consts={"cid": 0}, call=("cid", "args"))])
    emod.offload_group(g1, "k.cc", [ag.step("k", reads=g1.inputs[:1], writes=g1.outputs,
                                            consts={"cid": 1, "s": 1.5}, call=("cid", "args", "s"))])
    with pytest.raises(ValueError, match="two C signatures"):
        gen(wd, tmp_path / "out", monkeypatch)


def test_tampered_weight_file_raises(tmp_path, monkeypatch):
    wd, ids = net(tmp_path)
    emod = EmbeddedModel(wd)
    g = emod.group([0], "c0")
    emod.offload_group(g, "k.cc", [conv_step(g, WeightRef("w", ids["w"], (8, 4)).as_bf16())])
    f = wd / emod.groups[0]["steps"][0]["weights"]["file"]
    f.write_bytes(f.read_bytes()[:-2])
    with pytest.raises(ValueError, match="weights.bin holds 64 B"):
        gen(wd, tmp_path / "out", monkeypatch)


def test_inplace_weight_needs_bf16_blob(tmp_path, monkeypatch):
    wd, ids = net(tmp_path)
    emod = EmbeddedModel(wd)
    g = emod.group([0], "c0")
    emod.offload_group(g, "k.cc", [conv_step(g, WeightRef("w", ids["w"], (8, 4)).as_bf16())])
    monkeypatch.setenv("AIEGRAPH_PARAMS_INPLACE", "1")
    monkeypatch.setenv("AIEGRAPH_WDTYPE", "fp32")
    (tmp_path / "o").mkdir()
    with pytest.raises(ValueError, match="need AIEGRAPH_PARAMS_INPLACE=1 AIEGRAPH_WDTYPE=bf16"):
        ab.generate_driver(wd / "net_graph.json", wd / "net_params.bin", tmp_path / "o", verbose=False)


def test_offload_reset_drops_the_step(tmp_path, monkeypatch):
    wd, ids = net(tmp_path)
    base_txt, base_w = gen(wd, tmp_path / "base", monkeypatch)
    emod = EmbeddedModel(wd)
    g = emod.group([0], "c0")
    emod.offload_group(g, "k.cc", [conv_step(g, WeightRef("w", ids["w"], (8, 4)).as_bf16(
        layout=transpose_layout))])
    assert emod.offload_reset("group:c0") == ["group:c0"]
    txt, wbin = gen(wd, tmp_path / "out", monkeypatch)
    assert txt == base_txt and wbin == base_w


# ── bk-sdm-tiny (slow, needs the tutorial deploy tree + ONNX) ──────────────────────────────
UNET_WORK = Path("/scratch/staff/huaj/tmp/unet_tutorial/build/ag_work")
UNET_ONNX = Path("/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx")
UNET_PART = Path(__file__).resolve().parents[3] / "example/aiegraph/unet_bk_sdm_tiny/partition_p4a.json"


@pytest.mark.skipif(os.environ.get("AIEGRAPH_UNET_SLOW") != "1" or not UNET_WORK.is_dir()
                    or not UNET_ONNX.is_file() or not UNET_PART.is_file(),
                    reason="AIEGRAPH_UNET_SLOW=1 + tutorial deploy tree + bk-sdm-tiny ONNX")
def test_bk_sdm_tiny_match_conv_linear():
    import onnx
    from aiegraph.callsite import conv_groups, linear_groups
    from aiegraph.group import CallGraph
    cg = CallGraph.load(UNET_WORK)
    part = json.loads(UNET_PART.read_text())
    m = onnx.load(str(UNET_ONNX))
    conv = conv_groups(cg, UNET_WORK, None, part, model=m)
    lin = linear_groups(cg, UNET_WORK, None, part, model=m)
    assert (len(conv), len(lin)) == (48, 72)
    assert all(len(g.calls) == 1 and g.w.tensor in cg.calls[g.calls[0]].ins for g in conv + lin)
    assert {g.info["onnx"] for g in conv if g.info["folded"]} <= {
        g.info["onnx"] for g in conv if "proj_in" in g.info["onnx"]}
