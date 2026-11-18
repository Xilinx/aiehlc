###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""unet-m4 P3: keep-layout / scale folds as aiegraph declarations (aiegraph.fold), replacing the driver
text pass keep_layout_codegen.py.

``emod.match("conv+layout" | "linear+layout" | "dense+silu")`` -> groups, ``emod.merge_groups`` joins
overlapping folds, ``emod.offload_group(g, steps, emit={step: {"template": "aiednn/<cat>/x.cc",
"params": {...}}})`` records them and ``arm_build.generate_driver`` emits them (labels, removed calls,
pre-packed weights). x86 only:

* the real bk-sdm-tiny graph.json + its Relay-recorded opsem (test/data): the folds the old text pass
  made (19 conv outs, 9 transformer outs, 10 dense scale folds) are found by the matchers;
* a small hand-made graph with exact op semantics for the counter-examples: another reader, a
  permutation that is not the label, an op that is not a permutation, an epilogue operand in the output
  buffer, a shared weight -- and the plan checks (label size, pin, in-place hazard);
* the driver generated from the recorded groups.
"""
import gzip
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

import aiegraph as ag  # noqa: E402
from aiegraph import _env, fold  # noqa: E402
from aiegraph.callsite import WeightRef, _groups_ok  # noqa: E402
from aiegraph.embedded import EmbeddedModel  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.group import CallGraph, make_group, plan_group, render_step  # noqa: E402

_env.frontend("")
import frontend.tvmrelay.arm_build as ab  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


# ─────────────────────────────────────────────────────────── the real graph

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    d = tmp_path_factory.mktemp("real")
    (d / "net_graph.json").write_bytes(gzip.decompress((DATA / "bk_sdm_tiny_graph.json.gz").read_bytes()))
    (d / "net_opsem.json").write_bytes(gzip.decompress((DATA / "bk_sdm_tiny_opsem.json.gz").read_bytes()))
    cg = CallGraph.load(d)
    convs = []
    for c in [c for c in cg.calls if "conv2d" in c.fn][:-1]:      # the last conv (4-channel out) stays on TVM
        g = make_group(cg, [c.call], f"c{c.call}")
        g.info = {}
        convs.append(g)
    lins = []
    rp = cg.graph["node_row_ptr"]
    for c in cg.calls:
        raw = [rp[s] + k for s, k, _ in cg.graph["nodes"][c.node]["inputs"]]
        if "dense" in c.fn and raw[1] in cg.const and len(cg.shape[raw[1]]) == 3:
            g = make_group(cg, [c.call], f"l{c.call}")
            g.info = {}
            lins.append(g)
    return d, cg, convs, lins


# the folds keep_layout_codegen made on this graph (P2 build log, KL conv out / proj_out / temb lines)
OLD_CONV_OUT = [(14, 15), (24, 25), (71, 72), (82, 83), (129, 130), (140, 141), (187, 188), (198, 199), (245, 246),
                (256, 257), (305, 306), (316, 317), (363, 364), (374, 375), (423, 424), (434, 435), (481, 482),
                (492, 493), (530, 531)]
OLD_LIN_OUT = [(60, 61, 62), (118, 119, 120), (176, 177, 178), (234, 235, 236), (292, 293, 294), (352, 353, 354),
               (410, 411, 412), (470, 471, 472), (528, 529, 530)]
OLD_SILU = [8, 11, 68, 126, 184, 242, 302, 360, 420, 478]


def test_real_conv_layout(real):
    _, cg, convs, _ = real
    logs = []
    f = fold.conv_layout(cg, convs, log=logs.append)
    assert [g.calls for g in f] == OLD_CONV_OUT, logs
    lab = {g.calls[0]: g.info["fold"].params["layout"] for g in f}
    for g in f:   # planar for layout_transform_reshape*, NHWC for the ones with a transpose
        assert lab[g.calls[0]] == ("nhwc" if "transpose" in cg.calls[g.calls[1]].fn else "planar")
        assert g.info["fold"].removed == [g.calls[1]]
        assert g.outputs and g.outputs[0].tensor == cg.calls[g.calls[1]].outs[0]
    # x_first=False: the 8 convs whose activation buffer TVM reuses for the layout output are refused
    f2 = fold.conv_layout(cg, convs, x_first=False, log=logs.append)
    assert len(f2) == 11 and any("lives in the output buffer" in x for x in logs)


def test_real_linear_layout_and_merge(real):
    _, cg, convs, lins = real
    f1 = fold.conv_layout(cg, convs)
    f2 = fold.linear_layout(cg, lins, convs)
    assert [g.calls for g in f2] == OLD_LIN_OUT
    m = fold.merge(cg, convs + lins + f1 + f2)
    big = [g for g in m if len(g.calls) > 2]
    assert [g.calls for g in big] == OLD_LIN_OUT[:-1] + [(528, 529, 530, 531)]
    g = big[-1]
    assert sorted(g.info["fold"].removed) == [529, 531] and set(g.info["base"]) == {528, 530}
    rd, wr, args = fold.step_io(cg, g, 528)          # residual into the linear scatter: one more argument
    res = g.info["fold"].edits[528][0][2].tensor
    assert [a.tensor for a in args][-2:] == [res, cg.calls[528].outs[0]] and res in [t.tensor for t in rd]
    assert wr == []
    rd, wr, args = fold.step_io(cg, g, 530)          # conv reads [M][C] as NHWC, writes the planar NCHW out
    assert args[0].shape == (1, 1, 32, 32, 320) and args[0].early
    assert args[-1].shape == (1, 320, 32, 32, 1) and [t.tensor for t in wr] == [cg.calls[531].outs[0]]
    steps = [ag.step("lin", reads=fold.step_io(cg, g, 528)[0], args=fold.step_io(cg, g, 528)[2], at=528,
                     call=("args", "nargs")),
             ag.step("conv", reads=rd, writes=wr, args=args, at=530, call=("args", "nargs"))]
    p = plan_group(cg, g, steps)
    assert p.placement == {"lin": 528, "conv": 530} and p.removed == [529, 531]
    assert p.steps[1]["args"][0] == {"tensor": args[0].tensor, "shape": [1, 1, 32, 32, 320], "early": True}


def test_real_dense_silu(real):
    _, cg, _, _ = real
    f = fold.dense_silu(cg)
    assert [g.calls[0] for g in f] == OLD_SILU
    assert f[0].info["scale"] is None and len({g.info["scale"] for g in f[1:]}) == 1   # sigmoid inline / shared
    assert [(g.info["N"], g.info["K"], g.info["block"]) for g in f[:2]] == [(1280, 1280, 8), (320, 1280, 8)]
    assert all(g.calls[1] == g.info["dense"] and len(g.outputs) == 1 for g in f)


def test_real_linear_layout_needs_the_add(real, tmp_path):
    """counter-example on the real graph: the residual add of one transformer out recorded as a multiply"""
    d, cg, convs, lins = real
    doc = json.loads((d / "net_opsem.json").read_text())
    nid = str(cg.calls[61].node)
    for o in doc["nodes"][nid]["ops"]:
        if o["op"] == "add":
            o["op"] = "multiply"
    (tmp_path / "net_graph.json").write_text((d / "net_graph.json").read_text())
    (tmp_path / "net_opsem.json").write_text(json.dumps(doc))
    cg2 = CallGraph.load(tmp_path)
    got = [g.calls for g in fold.linear_layout(cg2, [make_group(cg2, list(g.calls), g.name) for g in lins],
                                                 [make_group(cg2, list(g.calls), g.name) for g in convs])]
    assert (60, 61, 62) not in got and len(got) == 8


# ─────────────────────────────────────────────────────────── a small graph, exact op semantics

def _lt(src, dst, ins, out):
    return {"op": "layout_transform", "ins": [ins], "out": out, "attrs": {"src_layout": src, "dst_layout": dst}}


def _op(op, ins, out, **attrs):
    return {"op": op, "ins": ins, "out": out, "attrs": attrs}


CONV_PLANAR = [_lt("NCHW4c", "NCHW", [1, 2, 2, 2, 4], [1, 8, 2, 2]), _op("reshape", [[1, 8, 2, 2]], [1, 8, 4])]
CONV_NHWC = [_lt("NCHW4c", "NCHW", [1, 2, 2, 2, 4], [1, 8, 2, 2]),
             _op("transpose", [[1, 8, 2, 2]], [1, 2, 2, 8], axes=[0, 2, 3, 1]), _op("reshape", [[1, 2, 2, 8]], [1, 4, 8])]
CONV_WHC = [_lt("NCHW4c", "NCHW", [1, 2, 2, 2, 4], [1, 8, 2, 2]),        # W and H swapped: not NHWC
            _op("transpose", [[1, 8, 2, 2]], [1, 2, 2, 8], axes=[0, 3, 2, 1]), _op("reshape", [[1, 2, 2, 8]], [1, 4, 8])]
CONV_EXP = [_lt("NCHW4c", "NCHW", [1, 2, 2, 2, 4], [1, 8, 2, 2]), _op("exp", [[1, 8, 2, 2]], [1, 8, 2, 2]),
            _op("reshape", [[1, 8, 2, 2]], [1, 8, 4])]
SILU = [_op("sigmoid", [[1, 4]], [1, 4]), _op("squeeze", [[1, 4]], [4], axis=[0]),
        _op("multiply", [[16, 4], [4]], [16, 4]), _lt("NC", "NC8n", [16, 4], [2, 4, 8])]
SCALE = [_op("multiply", [[16, 4], [4]], [16, 4]), _lt("NC", "NC8n", [16, 4], [2, 4, 8])]

WC = (np.arange(32, dtype=np.float32).reshape(2, 1, 1, 1, 4, 4) - 15.5) / 7.0
BC = np.linspace(-1, 1, 8).astype(np.float32).reshape(1, 2, 1, 1, 4)
WD = (np.arange(64, dtype=np.float32).reshape(16, 4) - 31.25) / 9.0
BD = np.linspace(0, 1, 16).astype(np.float32)


def save_params(path, arrays):
    b = struct.pack("<QQQ", ab.PARAM_MAGIC, 0, len(arrays))
    for n in arrays:
        b += struct.pack("<Q", len(n.encode())) + n.encode()
    b += struct.pack("<Q", len(arrays))
    for a in arrays.values():
        a = np.ascontiguousarray(a, dtype="<f4")
        b += struct.pack("<QQii", ab.NDARRAY_MAGIC, 0, 1, 0) + struct.pack("<i", a.ndim)
        b += struct.pack("<BBH", 2, 32, 1) + struct.pack("<%dq" % a.ndim, *a.shape)
        b += struct.pack("<q", a.nbytes) + a.tobytes()
    path.write_bytes(b)


def small(tmp_path, lay=CONV_PLANAR, silu=SILU, other_reader=None, residual_sid=None, w_shared=False):
    """x -> [0] relu -> x1 -> [1] conv(x1, wc, bc [, r]) -> y -> [2] layout -> yl -> [3] relu (out)
       t0 -> [4] relu -> t -> [5] scale/pack(t | s, wd) -> wp -> [6] dense(t, wp, bd) -> z (out)"""
    spec = []

    def node(name, fn, ins, shape, ops=None, sid=None):
        spec.append((name, fn, ins, shape, ops, sid))
    node("x", None, [], [1, 1, 2, 2, 4])
    node("wc", None, [], list(WC.shape))
    node("bc", None, [], list(BC.shape))
    node("t0", None, [], [1, 4])
    node("wd", None, [], [16, 4])
    node("bd", None, [], [16])
    node("x1", "nn_relu", ["x"], [1, 1, 2, 2, 4], [_op("nn.relu", [[1, 1, 2, 2, 4]], [1, 1, 2, 2, 4])])
    if residual_sid is not None:
        node("r", "nn_relu_r", ["x1"], [1, 2, 2, 2, 4], [_op("nn.relu", [[1, 1, 2, 2, 4]], [1, 2, 2, 2, 4])],
             sid=residual_sid)
    cins = ["x1", "wc", "bc"] + (["r"] if residual_sid is not None else [])
    node("y", "nn_contrib_conv2d_NCHWc_add", cins, [1, 2, 2, 2, 4], [_op("nn.contrib_conv2d_NCHWc", [], [])])
    node("yl", "layout_transform_reshape" if lay is not CONV_EXP else "layout_transform_exp_reshape", ["y"],
         list(lay[-1]["out"]), lay, sid=residual_sid)
    if other_reader == "y":
        node("y2", "nn_relu_y", ["y"], [1, 2, 2, 2, 4], [_op("nn.relu", [[1, 2, 2, 2, 4]], [1, 2, 2, 2, 4])])
    node("o1", "nn_relu_1", ["yl"], list(lay[-1]["out"]), [_op("nn.relu", [lay[-1]["out"]], lay[-1]["out"])])
    node("t", "nn_relu_2", ["t0"], [1, 4], [_op("nn.relu", [[1, 4]], [1, 4])])
    if silu is SCALE:
        node("s", "sigmoid_s", ["t"], [4], [_op("sigmoid", [[1, 4]], [4])])
    node("wp", "sigmoid_squeeze_multiply_layout_transform" if silu is SILU else "multiply_layout_transform",
         ["t", "wd"] if silu is SILU else ["wd", "s"], [2, 4, 8], silu)
    if other_reader == "wp":
        node("wp2", "nn_relu_wp", ["wp"], [2, 4, 8], [_op("nn.relu", [[2, 4, 8]], [2, 4, 8])])
    if w_shared:
        node("wd2", "nn_relu_wd", ["wd"], [16, 4], [_op("nn.relu", [[16, 4]], [16, 4])])
    node("z", "nn_contrib_dense_pack_add", ["t", "wp", "bd"], [1, 16], [_op("nn.dense", [], [])])
    ids = {s[0]: i for i, s in enumerate(spec)}
    nodes, sids, sem = [], [], {}
    for i, (name, fn, ins, shape, ops, sid) in enumerate(spec):
        sids.append(100 + i if sid is None else sid)
        if fn is None:
            nodes.append({"op": "null", "name": name, "inputs": []})
            continue
        nodes.append({"op": "tvm_op", "name": name, "inputs": [[ids[x], 0, 0] for x in ins],
                      "attrs": {"func_name": "tvmgen_default_fused_" + fn, "num_outputs": "1"}})
        sem[str(i)] = {"func_name": "tvmgen_default_fused_" + fn, "name_ok": True, "ops": ops,
                       "params": [spec[ids[x]][3] for x in ins], "outs": [shape]}
    n = len(nodes)
    g = {"nodes": nodes, "arg_nodes": [i for i, x in enumerate(nodes) if x["op"] == "null"],
         "heads": [[ids["o1"], 0, 0], [ids["z"], 0, 0]], "node_row_ptr": list(range(n + 1)),
         "attrs": {"storage_id": ["list_int", sids], "shape": ["list_shape", [s[3] for s in spec]],
                   "dltype": ["list_str", ["float32"] * n]}}
    (tmp_path / "net_graph.json").write_text(json.dumps(g))
    (tmp_path / "net_opsem.json").write_text(json.dumps({"version": 1, "nodes": sem}))
    save_params(tmp_path / "net_params.bin", {"wc": WC, "bc": BC, "wd": WD, "bd": BD})
    (tmp_path / "layers").mkdir(exist_ok=True)
    (tmp_path / "layers" / "partition.json").write_text(json.dumps({"layers": []}))
    cg = CallGraph.load(tmp_path)
    call = {spec[cg.calls[k].node][0]: k for k in range(len(cg.calls))}
    return cg, ids, call


def conv_base(cg, call):
    g = make_group(cg, [call["y"]], "conv0")
    g.info = {}
    return [g]


def test_conv_layout_planar_and_nhwc(tmp_path):
    for lay, lab in ((CONV_PLANAR, (1, 8, 2, 2, 1)), (CONV_NHWC, (1, 1, 2, 2, 8))):
        d = tmp_path / lab.__repr__().replace(" ", "")
        d.mkdir()
        cg, ids, call = small(d, lay=lay)
        f = fold.conv_layout(cg, conv_base(cg, call))
        assert len(f) == 1 and f[0].calls == (call["y"], call["yl"])
        assert f[0].info["fold"].params["label"] == lab
        assert f[0].info["fold"].edits[call["y"]] == [("set", 3, ag.Arg(ids["yl"], lab))]


@pytest.mark.parametrize("case", ["other_reader", "not_the_label", "not_a_permutation", "epilogue_in_out_buffer"])
def test_conv_layout_counter_examples(tmp_path, case):
    logs = []
    kw = {"other_reader": dict(other_reader="y"), "not_the_label": dict(lay=CONV_WHC),
          "not_a_permutation": dict(lay=CONV_EXP), "epilogue_in_out_buffer": dict(residual_sid=777)}[case]
    cg, ids, call = small(tmp_path, **kw)
    assert fold.conv_layout(cg, conv_base(cg, call), log=logs.append) == []
    if case == "epilogue_in_out_buffer":
        assert any("lives in the output buffer" in x for x in logs)


def test_dense_silu_inline_and_vector(tmp_path):
    cg, ids, call = small(tmp_path)
    f = fold.dense_silu(cg)
    assert len(f) == 1 and f[0].calls == (call["wp"], call["z"])
    i = f[0].info
    assert (i["N"], i["K"], i["block"], i["scale"], i["x"], i["packed"]) == (16, 4, 8, None, ids["t"], ids["wp"])
    assert f[0].w.tensor == ids["wd"] and f[0].outputs[0].tensor == ids["z"]
    d = tmp_path / "vec"
    d.mkdir()
    cg, ids, call = small(d, silu=SCALE)
    f = fold.dense_silu(cg)
    assert len(f) == 1 and f[0].info["scale"] == ids["s"]


@pytest.mark.parametrize("case", ["packed_weight_other_reader", "weight_shared"])
def test_dense_silu_counter_examples(tmp_path, case):
    cg, _, _ = small(tmp_path, **({"other_reader": "wp"} if case == "packed_weight_other_reader"
                                  else {"w_shared": True}))
    assert fold.dense_silu(cg) == []


# ─────────────────────────────────────────────────────────── plan checks

def test_label_must_keep_the_element_count(tmp_path):
    cg, ids, call = small(tmp_path)
    g = fold.conv_layout(cg, conv_base(cg, call))[0]
    rd, wr, args = fold.step_io(cg, g, call["y"])
    args[-1] = ag.arg(ids["yl"], (1, 8, 2, 2, 2))
    with pytest.raises(AiegraphError, match="changes the element count"):
        plan_group(cg, g, [ag.step("k", reads=rd, writes=wr, args=args, at=call["y"], call=("args",))])


def test_pin_must_be_a_free_group_call(tmp_path):
    cg, ids, call = small(tmp_path)
    g = fold.conv_layout(cg, conv_base(cg, call))[0]
    rd, wr, args = fold.step_io(cg, g, call["y"])
    with pytest.raises(AiegraphError, match="not a call of the group"):
        plan_group(cg, g, [ag.step("k", reads=rd, writes=wr, args=args, at=call["o1"], call=("args",))])


def test_in_place_hazard(tmp_path):
    """plan_group refuses what the matcher skips: the conv writes the layout output into the buffer of its
    residual operand (TVM reused it once the residual was dead)"""
    cg, ids, call = small(tmp_path, residual_sid=777)
    g = make_group(cg, [call["y"], call["yl"]], "bad")
    args = [ag.arg(t) for t in (ids["x1"], ids["wc"], ids["bc"], ids["r"])] + [ag.arg(ids["yl"], (1, 8, 2, 2, 1))]
    with pytest.raises(AiegraphError, match="in-place hazard"):
        plan_group(cg, g, [ag.step("k", reads=[cg.tensor(ids["x1"]), cg.tensor(ids["r"])],
                                   writes=[cg.tensor(ids["yl"])], args=args,
                                   at=call["y"], call=("args",))])


def test_args_abi_write_must_be_packed(tmp_path):
    cg, ids, call = small(tmp_path)
    emod = EmbeddedModel(tmp_path)
    g = fold.conv_layout(cg, conv_base(cg, call))[0]
    rd, wr, _ = fold.step_io(cg, g, call["y"])
    with pytest.raises(AiegraphError, match="packed arguments do not carry"):
        emod.offload_group(g, "k.cc", [ag.step("k", reads=rd, writes=wr, at=call["y"], call=("args",))])


def test_template_step_and_param_checks():
    decl, body = render_step("aiednn/elementwise/silu_scale_x.cc", dict(WPACK=9, NK=64, X=7, S="0", KLEN=4))
    assert "static void aiednn_silu_scale_x" in decl and "@" not in decl
    assert body == ["graph_bf16_expand((float *)tensors[9].data, @WBF@, 64ULL);   /* pre-packed [N/b][K][b] */",
                    "aiednn_silu_scale_x(&tensors[7], 0, 4);"]
    with pytest.raises(AiegraphError, match=r"missing params \['KLEN'\]"):
        render_step("aiednn/elementwise/silu_scale_x.cc", dict(WPACK=9, NK=64, X=7, S="0"))


def test_conv_groups_param():
    assert _groups_ok(1, 1, 8, 8) and not _groups_ok(1, 8, 8, 8)
    assert _groups_ok("dw", 8, 8, 8) and _groups_ok("dw", 1, 4, 8) and not _groups_ok("dw", 2, 8, 8)
    assert _groups_ok("any", 2, 8, 8) and _groups_ok(2, 2, 8, 8)
    from aiegraph.callsite import conv_groups
    with pytest.raises(AiegraphError, match="groups='nope'"):
        conv_groups(None, None, None, {}, groups="nope")


# ─────────────────────────────────────────────────────────── the driver from the recorded groups

def test_driver_from_declarations(tmp_path, monkeypatch):
    cg, ids, call = small(tmp_path, lay=CONV_NHWC)
    emod = EmbeddedModel(tmp_path)
    g = fold.conv_layout(cg, conv_base(cg, call))[0]
    rd, wr, args = fold.step_io(cg, g, call["y"])
    w = WeightRef("wc", ids["wc"], WC.shape).as_bf16()
    p1 = emod.offload_group(g, "k.cc", [ag.step("kconv", reads=rd, writes=wr, args=args, at=call["y"], weights=w,
                                                consts=dict(cid=3), call=("cid", "args", "nargs", "wbf"),
                                                wrap=("UP_B", "UP_E"), tag="aie kconv 3")],
                            launch=dict(aie=1))
    s = emod.match("dense+silu")[0]
    i = s.info
    pack = lambda a: a.reshape(i["N"] // i["block"], i["block"], i["K"]).transpose(0, 2, 1)   # noqa: E731
    emit = {"silu_scale_x": {"template": "aiednn/elementwise/silu_scale_x.cc", "tag": "cpu silu_scale_x",
                             "params": dict(WPACK=i["packed"], NK=i["N"] * i["K"], X=i["x"], KLEN=i["K"], S="0")},
            "silu_scale_dense": {"template": "aiednn/elementwise/silu_scale_dense.cc", "tag": "cpu dense",
                                 "params": dict(FN=i["symbol"], NARGS=i["nargs"])}}
    p2 = emod.offload_group(s, steps=[ag.step("silu_scale_x", reads=s.inputs, weights=s.w.as_bf16(layout=pack),
                                              at=s.calls[0]),
                                      ag.step("silu_scale_dense", writes=s.outputs, at=i["dense"])], emit=emit)
    rec = {r["name"]: r for r in emod.groups}
    assert rec[g.name]["launch"] == {"aie": 1, "steps": 1}
    assert rec[g.name]["io"]["outputs"] == [{"name": f"tensors[{ids['yl']}]", "tensor": ids["yl"],
                                             "shape": [1, 4, 8], "dtype": "float32"}]
    assert rec[s.name]["steps"][0]["template"] == "aiednn/elementwise/silu_scale_x.cc"
    assert rec[s.name]["steps"][0]["params"]["KLEN"] == 4 and p2.removed == [] and p1.removed == [call["yl"]]

    monkeypatch.setenv("AIEGRAPH_PARAMS_INPLACE", "1")
    monkeypatch.setenv("AIEGRAPH_WDTYPE", "bf16")
    out = tmp_path / "out"
    out.mkdir()
    ab.generate_driver(tmp_path / "net_graph.json", tmp_path / "net_params.bin", out, verbose=False)
    txt = (out / "graph_driver.c").read_text()
    k = call["y"]
    assert f"static int64_t g_lbl_{k}_3_shape[5] = {{1, 1, 2, 2, 8}};" in txt
    assert (f"    g_lbl_{k}_3 = tensors[{ids['yl']}]; g_lbl_{k}_3.ndim = 5; g_lbl_{k}_3.shape = g_lbl_{k}_3_shape;"
            in txt and f"    args[3].v_handle = &g_lbl_{k}_3;" in txt)
    assert f"/* [{call['yl']}] removed: group {g.name} */" in txt
    assert "tvmgen_default_fused_layout_transform_reshape(args" not in txt.split("float *graph_run(void)")[1]
    assert txt.count("static void aiednn_silu_scale_x") == 1
    wb = (out / "weights.bin").read_bytes()
    assert re.search(r"aiednn_silu_scale_x\(&tensors\[%d\], 0, 4\);" % ids["t"], txt)
    e = re.search(r"graph_bf16_expand\(\(float \*\)tensors\[%d\]\.data, \(const uint16_t \*\)\(g_params "
                                r"\+ (\d+)\), 64ULL\);" % ids["wp"], txt)
    assert e, txt
    got = np.frombuffer(wb[int(e.group(1)):int(e.group(1)) + 128], dtype="<u2")
    want = ab.f32_to_bf16_rne(WD.reshape(-1)).reshape(2, 8, 4).transpose(0, 2, 1).reshape(-1)
    assert np.array_equal(got, want)                       # pre-packed [N/8][K][8], bit-exact
    assert f"    tensors[{ids['wd']}].data = 0;   /* group {s.name}: bf16 weight read in place by silu_scale_x */" in txt
    assert "    args[0].v_handle = &g_silu_x_t;" in txt and 'GRAPH_PROF_END(%d, "cpu dense");' % i["dense"] in txt
