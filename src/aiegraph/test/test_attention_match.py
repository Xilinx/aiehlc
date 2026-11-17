###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Attention matcher + group steps in the driver, step S3 (design
docs/plans/2026-11-16-aiegraph-group-offload-design.md §5.2 / §6 / §9 (1)).

x86 only, no TVM, no board:
* ``aiegraph.attention.match_attention`` on the real bk-sdm-tiny graph.json (S2 fixture) + its ONNX:
  18 groups / 126 calls / 6 shape classes, Q/K/V/O tensors, scale (bf16-rounded), node-list fallback,
  and the negative cases (wrong node list, a model with another Softmax count, shape mismatch);
* ``partition_attention`` (was unet_aie.map_attn): partition attn rows joined by ONNX node name;
* ``emit_steps`` + ``arm_build.group_sites`` / ``generate_driver``: the steps' code at their calls,
  removed calls gone, a stale record (other symbol at the call) or a step without code refused.
"""
import gzip
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph.attention import (attention_classes, match_attention, onnx_attention_nodes,  # noqa: E402
                                partition_attention, tvm_attention_cores)
from aiegraph.contract import step  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.group import (CallGraph, emit_steps, group_step, make_group, plan_group,  # noqa: E402
                            record_group)

UNET_GRAPH = Path(__file__).with_name("data") / "bk_sdm_tiny_graph.json.gz"
UNET_ONNX = Path("/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx")
need_onnx = pytest.mark.skipif(not UNET_ONNX.is_file(), reason="bk-sdm-tiny ONNX not on this machine")


@pytest.fixture(scope="module")
def unet():
    return CallGraph(json.loads(gzip.decompress(UNET_GRAPH.read_bytes())), UNET_GRAPH)


@pytest.fixture(scope="module")
def model():
    import onnx
    return onnx.load(str(UNET_ONNX), load_external_data=False)


@pytest.fixture(scope="module")
def groups(unet, model):
    return match_attention(unet, onnx=model, wdtype="bf16")


# ── the matcher on the real UNet ──────────────────────────────────────────────────
def test_tvm_cores_one_per_softmax(unet):
    cores = tvm_attention_cores(unet)
    assert len(cores) == 18 and all(c is not None for c in cores)
    assert cores[0][:3] == (34, 33, 36) and cores[0][4] == 37


@need_onnx
def test_unet_18_groups_126_calls_6_classes(groups):
    assert len(groups) == 18 and sum(len(g.calls) for g in groups) == 126
    assert {len(g.calls) for g in groups} == {7}
    assert attention_classes(groups) == [(8, 1024, 1024, 40), (8, 1024, 77, 40), (8, 256, 256, 80),
                                         (8, 256, 77, 80), (8, 64, 64, 160), (8, 64, 77, 160)]


@need_onnx
def test_unet_attn0_boundary_and_scale(groups):
    """design §5.2: attn0 = [31]-[37], q/k/v = tensors[58/59/60] (the 3 split outputs of the fused qkv
    linear -- picked by the ONNX Split index), o = tensors[70], ld 320; scale 40^-1/4 rounded to bf16."""
    g = groups[0]
    assert g.calls == (31, 32, 33, 34, 35, 36, 37)
    assert (g.q_call, g.k_call, g.qk, g.softmax, g.v_call, g.pv, g.merge) == (31, 32, 33, 34, 35, 36, 37)
    assert (g.q.tensor, g.k.tensor, g.v.tensor, g.o.tensor) == (58, 59, 60, 70)
    assert g.q.ld == g.o.ld == 320 and g.q.shape == (1024, 40)
    assert (g.H, g.N, g.M, g.D, g.cls) == (8, 1024, 1024, 40, 0)
    assert abs(g.scale_fp32 - 40 ** -0.25) < 1e-6 and g.scale == g.scale_k == 0.3984375
    assert g.onnx["softmax"].endswith("transformer_blocks.0/attn1/Softmax")
    assert g.bmm["qk"]["xshape"] == [8, 1024, 40] and g.bmm["pv"]["wshape"] == [8, 40, 1024]


@need_onnx
def test_unet_cross_attention_sources(groups):
    """attn1 (cross-attention): K / V come from their own linears over the text embedding (M = 77)."""
    g = groups[1]
    assert (g.M, g.cls) == (77, 1) and len({g.q.tensor, g.k.tensor, g.v.tensor}) == 3
    assert g.k.shape == (77, 40) and g.k.ld == 320


@need_onnx
def test_fp32_weights_keep_the_onnx_scale(unet, model):
    g = match_attention(unet, onnx=model, wdtype="fp32")[2]
    assert g.scale == g.scale_fp32 and abs(g.scale - 80 ** -0.25) < 1e-6


@need_onnx
def test_node_list_fallback_same_groups(unet, model, groups):
    nodes = [(d["qk"], d["softmax"], d["pv"]) for d in onnx_attention_nodes(model)]
    g2 = match_attention(unet, onnx=model, nodes=nodes, wdtype="bf16")
    key = lambda gs: [(g.calls, g.q.tensor, g.k.tensor, g.v.tensor, g.o.tensor, g.scale) for g in gs]  # noqa
    assert key(g2) == key(groups)
    sub = match_attention(unet, onnx=model, nodes=nodes[2:4], wdtype="bf16")       # a subset works too
    assert [g.calls for g in sub] == [g.calls for g in groups[2:4]]


@need_onnx
def test_node_list_wrong_op_raises(unet, model):
    d = onnx_attention_nodes(model)[0]
    with pytest.raises(AiegraphError, match="is MatMul, expected Softmax"):
        match_attention(unet, onnx=model, nodes=[(d["qk"], d["pv"], d["pv"])])
    with pytest.raises(AiegraphError, match="not in the model"):
        match_attention(unet, onnx=model, nodes=[("nope", d["softmax"], d["pv"])])
    with pytest.raises(AiegraphError, match="missing"):
        match_attention(unet, onnx=model, nodes=[{"qk": d["qk"], "softmax": d["softmax"]}])


@need_onnx
def test_node_list_pv_not_after_softmax_raises(unet, model):
    a, b = onnx_attention_nodes(model)[:2]
    with pytest.raises(AiegraphError, match="input 0 is not the output of"):
        match_attention(unet, onnx=model, nodes=[(a["qk"], a["softmax"], b["pv"])])


@need_onnx
def test_other_model_softmax_count_raises(unet, model):
    import copy
    m = copy.deepcopy(model)
    s = next(n for n in m.graph.node if n.op_type == "Softmax")
    m.graph.node.remove(s)
    with pytest.raises(AiegraphError, match="17 Softmax nodes, the TVM graph 18"):
        match_attention(unet, onnx=m)


def test_no_onnx_leaves_split_sources_unresolved(unet):
    """Without the ONNX the self-attention Q/K/V (3 split outputs read by each layout call) are not
    guessed; cross-attention K/V (one activation each) still are. No scale."""
    gs = match_attention(unet)
    assert len(gs) == 18 and gs[0].q is None and gs[0].o.tensor == 70 and gs[0].scale is None
    assert gs[1].k is not None


def test_shape_mismatch_raises(unet):
    raw = json.loads(gzip.decompress(UNET_GRAPH.read_bytes()))
    raw["attrs"]["shape"][1][64] = [8, 1024, 1000]          # attn0 scores no longer [H][N][M]
    with pytest.raises(AiegraphError, match="not one attention core"):
        match_attention(CallGraph(raw))


# ── partition rows (was unet_aie.map_attn) ────────────────────────────────────────
def _rows(groups):
    rows, i = [], 0
    for gi, g in enumerate(groups):
        for role, (M, K, N) in (("qk", (g.N, g.D, g.M)), ("pv", (g.N, g.M, g.D))):
            rows.append({"part": gi, "idx": i, "node": g.onnx[role], "cls": "attn", "batch": g.H,
                         "mkn": [[M, K, N]], "attn_group": gi, "attn_role": role, "softmax": g.onnx["softmax"]})
            i += 1
    return {"layers": rows}


@need_onnx
def test_partition_rows_join_by_onnx_name(unet, groups):
    att = partition_attention(unet, _rows(groups), groups=groups)
    assert len(att) == 36 and att[0]["role"] == "qk" and att[0]["call"] == 33 and att[1]["call"] == 36
    assert att[0]["ag_group"] is groups[0] and att[0]["xshape"] == [8, 1024, 40]


@need_onnx
def test_partition_row_missing_or_wrong_raises(unet, groups):
    p = _rows(groups)
    p["layers"].pop(3)
    with pytest.raises(AiegraphError, match="has no partition attn row"):
        partition_attention(unet, p, groups=groups)
    p = _rows(groups)
    p["layers"][0]["mkn"] = [[1024, 64, 1024]]
    with pytest.raises(AiegraphError, match="vs TVM call"):
        partition_attention(unet, p, groups=groups)
    p = _rows(groups)
    p["layers"].append(dict(p["layers"][0], node="/extra/MatMul"))
    with pytest.raises(AiegraphError, match="match no attention group"):
        partition_attention(unet, p, groups=groups)


# ── steps in the driver ────────────────────────────────────────────────────────────
def _fa(g):
    return [step("fa_step_pack_q", reads=g.q, consts=dict(cid=4, cls=g.cls, s=g.scale)),
            step("fa_step_pack_k", reads=g.k, consts=dict(cid=4, cls=g.cls, s=g.scale_k)),
            step("fa_step_pack_v", reads=g.v, consts=dict(cid=5, cls=g.cls)),
            step("fa_step_run_o", writes=g.o, consts=dict(cid=5, cls=g.cls))]


def _emit(s):
    t = (s["reads"] or s["writes"])[0]
    return {"c": [f"kl_{s['name']}(@K@, tensors[{t}].data);"], "tag": "attn %d" % s["consts"]["cid"]}


@need_onnx
def test_plan_with_steps_emits_code(unet, groups):
    g = groups[0]
    plan = emit_steps(plan_group(unet, g, _fa(g)), _emit)
    assert plan.placement == {"fa_step_pack_q": 31, "fa_step_pack_k": 32, "fa_step_pack_v": 35,
                              "fa_step_run_o": 36} and plan.removed == [33, 34, 37]
    assert plan.steps[3]["c"] == ["kl_fa_step_run_o(@K@, tensors[70].data);"]
    assert plan.record()["steps"][0]["tag"] == "attn 4"


def test_emit_without_code_raises(unet):
    g = make_group(unet, range(31, 38), "attn0")
    q, k, v = g.inputs
    plan = plan_group(unet, g, [group_step("a", reads=q), group_step("b", reads=k),
                                group_step("c", reads=v), group_step("d", writes=g.outputs[0])])
    with pytest.raises(AiegraphError, match="no driver code for step b"):
        emit_steps(plan, {"a": {"c": ["x();"]}})


def _tiny_graph():
    """x -> [0] A -> a -> [1] B -> b -> [2] C -> y (tensor id == node id)"""
    nodes = [{"op": "null", "name": "x", "inputs": []}]
    for i, (n, f) in enumerate((("a", "A"), ("b", "B"), ("y", "C"))):
        nodes.append({"op": "tvm_op", "name": n, "inputs": [[i, 0, 0]],
                      "attrs": {"func_name": "tvmgen_default_fused_" + f, "num_outputs": "1"}})
    return {"nodes": nodes, "arg_nodes": [0], "heads": [[3, 0, 0]], "node_row_ptr": [0, 1, 2, 3, 4],
            "attrs": {"storage_id": ["list_int", [0, 1, 2, 3]], "shape": ["list_shape", [[4, 8]] * 4],
                      "dltype": ["list_str", ["float32"] * 4]}}


def _workdir(tmp_path):
    (tmp_path / "net_graph.json").write_text(json.dumps(_tiny_graph()))
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "partition.json").write_text(json.dumps({"layers": []}))
    return CallGraph(_tiny_graph())


def _driver(tmp_path):
    from frontend.tvmrelay import arm_build as ab
    params = tmp_path / "net_params.bin"
    params.write_bytes(b"")
    out = tmp_path / "arm"
    out.mkdir(exist_ok=True)
    import unittest.mock as um
    with um.patch.object(ab, "param_offsets", return_value={}), \
            um.patch.object(ab, "params_inplace", return_value=False):
        ab.generate_driver(tmp_path / "net_graph.json", params, out, verbose=False)
    return (out / "graph_driver.c").read_text()


def _have_arm_build():
    try:
        from frontend.tvmrelay import arm_build  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_arm_build(), reason="frontend.tvmrelay.arm_build not importable")
def test_driver_places_steps_and_removes_calls(tmp_path):
    cg = _workdir(tmp_path)
    g = make_group(cg, [0, 1], "g0")
    plan = plan_group(cg, g, [group_step("s_ab", reads=[0], writes=[2])])
    record_group(tmp_path, emit_steps(plan, {"s_ab": {"c": ["UP(@K@);", "s_ab(tensors[0].data);"],
                                                      "tag": "s_ab"}}))
    t = _driver(tmp_path)
    run = t.split("float *graph_run(void)")[1]
    assert "tvmgen_default_fused_A(args" not in run and "tvmgen_default_fused_B(args" not in run
    assert "tvmgen_default_fused_C(args" in run
    assert "    UP(0);\n    s_ab(tensors[0].data);\n" in run and 'GRAPH_PROF_END(0, "aie s_ab");' in run
    assert "/* [1] removed: group g0 */" in run


@pytest.mark.skipif(not _have_arm_build(), reason="frontend.tvmrelay.arm_build not importable")
def test_driver_refuses_stale_or_codeless_group(tmp_path):
    cg = _workdir(tmp_path)
    plan = plan_group(cg, make_group(cg, [0, 1], "g0"), [group_step("s_ab", reads=[0], writes=[2])])
    rec = record_group(tmp_path, plan)                       # recorded without emit=
    with pytest.raises(ValueError, match="no driver code"):
        _driver(tmp_path)
    doc = json.loads((tmp_path / "layers" / "partition.json").read_text())
    doc["groups"][0]["steps"][0].update(c=["s_ab();"], tag="s")
    doc["groups"][0]["members"][1]["symbol"] = "tvmgen_default_fused_Z"
    (tmp_path / "layers" / "partition.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="stale group record"):
        _driver(tmp_path)
    assert rec["name"] == "g0"
