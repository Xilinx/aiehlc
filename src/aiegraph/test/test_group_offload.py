###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Group offload, step S2 (design docs/plans/2026-11-16-aiegraph-group-offload-design.md §6).

x86 only, no TVM, no board. Multi-call groups on graph.json: boundary, placement,
deletion (external reader), liveness (overwritten buffer), early write, the
partition.json record and offload_reset of a whole group -- on small hand-made graphs
(the negative cases) and on the real bk-sdm-tiny UNet graph.json.

``data/bk_sdm_tiny_graph.json.gz`` = gzip of the tutorial build's
``ag_work/resnet18_graph.json`` (bk-sdm-tiny ``unet_sim_t32``, 537 calls; sha1 of the
uncompressed file ab492860ba684e11b24a089563afb84aba99ebc0), the graph the board ELF
of tutorial/model/tiny-sdm-unet.md is generated from.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph.embedded import EmbeddedModel  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.group import (CallGraph, attention_groups, group_step, make_group,  # noqa: E402
                            plan_group, record_group, recorded_groups, reset_groups)

UNET_GRAPH = Path(__file__).with_name("data") / "bk_sdm_tiny_graph.json.gz"
UNET_GRAPH_SHA1 = "ab492860ba684e11b24a089563afb84aba99ebc0"
UNET_ONNX = Path("/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx")


# ── a tiny graph.json builder ─────────────────────────────────────────────────
def graph(spec, heads=None):
    """spec: ("in", name, sid) | ("op", name, fn, [input names], sid) | ("nop", name, src).
    One output per node; tensor id == node id (row_ptr is the identity)."""
    nodes, sid, ids = [], [], {}
    for item in spec:
        kind, name = item[0], item[1]
        ids[name] = len(nodes)
        if kind == "in":
            nodes.append({"op": "null", "name": name, "inputs": []})
            sid.append(item[2])
        elif kind == "nop":
            nodes.append({"op": "tvm_op", "name": name, "inputs": [[ids[item[2]], 0, 0]],
                          "attrs": {"func_name": "__nop", "num_outputs": "1"}})
            sid.append(sid[ids[item[2]]])
        else:
            nodes.append({"op": "tvm_op", "name": name,
                          "inputs": [[ids[i], 0, 0] for i in item[3]],
                          "attrs": {"func_name": "tvmgen_default_fused_" + item[2],
                                    "num_outputs": "1"}})
            sid.append(item[4])
    n = len(nodes)
    heads = heads or [spec[-1][1]]
    return {"nodes": nodes, "arg_nodes": [i for i, x in enumerate(nodes) if x["op"] == "null"],
            "heads": [[ids[h], 0, 0] for h in heads], "node_row_ptr": list(range(n + 1)),
            "attrs": {"storage_id": ["list_int", sid], "shape": ["list_shape", [[4, 8]] * n],
                      "dltype": ["list_str", ["float32"] * n]}}


def cg_of(spec, heads=None):
    return CallGraph(graph(spec, heads))


# x -> [0] A -> a -> [1] B -> b -> [2] C -> c -> [3] D -> y
CHAIN = [("in", "x", 0), ("in", "w", 1),
         ("op", "a", "A", ["x", "w"], 2), ("op", "b", "B", ["a"], 3),
         ("op", "c", "C", ["b"], 2), ("op", "y", "D", ["c"], 4)]


def T(cg, name):
    return next(i for i, n in enumerate(cg.graph["nodes"]) if n["name"] == name)


# ── the call graph ──────────────────────────────────────────────────────────────
def test_call_graph_calls_and_storage():
    cg = cg_of(CHAIN)
    assert [c.fn for c in cg.calls] == ["A", "B", "C", "D"]
    assert cg.producer[T(cg, "b")] == 1 and cg.readers[T(cg, "b")] == [2]
    assert cg.storage(T(cg, "a")) == cg.storage(T(cg, "c")) == 2
    assert T(cg, "x") in cg.const and T(cg, "w") in cg.const
    assert cg.heads == {T(cg, "y")}


def test_nop_is_an_alias_not_a_call():
    cg = cg_of([("in", "x", 0), ("op", "a", "A", ["x"], 1), ("nop", "a2", "a"),
                ("op", "y", "B", ["a2"], 2)])
    assert [c.fn for c in cg.calls] == ["A", "B"]
    assert cg.calls[1].ins == (T(cg, "a"),)
    assert cg.canon(T(cg, "a2")) == T(cg, "a")


def test_load_dir_and_gz(tmp_path):
    (tmp_path / "net_graph.json").write_text(json.dumps(graph(CHAIN)))
    assert len(CallGraph.load(tmp_path).calls) == 4
    gz = tmp_path / "g.json.gz"
    gz.write_bytes(gzip.compress(json.dumps(graph(CHAIN)).encode()))
    assert len(CallGraph.load(gz).calls) == 4
    (tmp_path / "empty").mkdir()
    with pytest.raises(AiegraphError, match="no \\*_graph.json"):
        CallGraph.load(tmp_path / "empty")


# ── groups and their boundary ─────────────────────────────────────────────────────
def test_group_boundary():
    cg = cg_of(CHAIN)
    g = make_group(cg, [2, 1], "bc")
    assert g.calls == (1, 2)
    assert [t.tensor for t in g.inputs] == [T(cg, "a")]
    assert [t.tensor for t in g.outputs] == [T(cg, "c")]
    assert [t.tensor for t in g.internal] == [T(cg, "b")]
    assert g.inputs[0].producer == 0 and g.inputs[0].sid == 2
    assert "bc" in repr(g)


def test_group_consts_are_not_inputs():
    cg = cg_of(CHAIN)
    g = make_group(cg, [0, 1])
    assert [t.tensor for t in g.consts] == [T(cg, "x"), T(cg, "w")]
    assert g.inputs == [] and g.name == "g0"


def test_empty_group_raises():
    with pytest.raises(AiegraphError, match="empty group"):
        make_group(cg_of(CHAIN), [])


def test_unknown_call_raises():
    with pytest.raises(AiegraphError, match="no call"):
        make_group(cg_of(CHAIN), [2, 9])


# ── placement and deletion ──────────────────────────────────────────────────────────
def test_plan_places_and_removes():
    cg = cg_of(CHAIN)
    g = make_group(cg, [1, 2], "bc")
    p = plan_group(cg, g, [group_step("run", reads=g.inputs[0], writes=g.outputs[0])])
    assert p.placement == {"run": 1} and p.removed == [2]
    assert p.driver_comments() == {1: "/* [1] */", 2: "/* [2] removed: group bc */"}


def test_plan_read_then_write_steps():
    cg = cg_of(CHAIN)
    g = make_group(cg, [1, 2], "bc")
    p = plan_group(cg, g, [group_step("pack", reads=g.inputs), group_step("run", writes=g.outputs)])
    assert p.placement == {"pack": 1, "run": 2} and p.removed == []
    assert "/* [2] */ run" in repr(p)


def test_external_reader_raises():
    """b is consumed by D outside the group {B, C}... here: deleting C's input producer
    B while outside call E still reads b."""
    cg = cg_of(CHAIN[:4] + [("op", "c", "C", ["b"], 2), ("op", "e", "E", ["b", "c"], 4)])
    g = make_group(cg, [1, 2], "bc")
    assert {t.tensor for t in g.outputs} == {T(cg, "b"), T(cg, "c")}
    with pytest.raises(AiegraphError, match=r"external reader .*tensors\[%d\].* call \[3\] E"
                       % T(cg, "b")):
        plan_group(cg, g, [group_step("run", reads=g.inputs[0], writes=T(cg, "c"))])


def test_graph_output_counts_as_external_reader():
    cg = cg_of(CHAIN)
    g = make_group(cg, [2, 3], "cd")
    with pytest.raises(AiegraphError, match="external reader .*graph output"):
        plan_group(cg, g, [group_step("pack", reads=T(cg, "b"))])


def test_step_reads_group_intermediate_raises():
    cg = cg_of(CHAIN)
    g = make_group(cg, [1, 2], "bc")
    with pytest.raises(AiegraphError, match="group produces"):
        plan_group(cg, g, [group_step("run", reads=T(cg, "b"), writes=T(cg, "c"))])


def test_step_writes_input_raises():
    cg = cg_of(CHAIN)
    g = make_group(cg, [1, 2], "bc")
    with pytest.raises(AiegraphError, match="does not produce"):
        plan_group(cg, g, [group_step("run", writes=T(cg, "a"))])


def test_step_reads_tensor_group_does_not_read_raises():
    cg = cg_of(CHAIN)
    g = make_group(cg, [2, 3], "cd")
    with pytest.raises(AiegraphError, match="no call of the group reads"):
        plan_group(cg, g, [group_step("run", reads=T(cg, "a"), writes=T(cg, "y"))])


def test_too_many_steps_raises():
    cg = cg_of(CHAIN)
    g = make_group(cg, [1], "b")
    with pytest.raises(AiegraphError, match="steps >"):
        plan_group(cg, g, [group_step("s1", reads=g.inputs), group_step("s2", writes=g.outputs)])


def test_no_free_call_raises():
    cg = cg_of(CHAIN)
    g = make_group(cg, [1, 2], "bc")
    # two steps reading a, but only call [1] reads it
    with pytest.raises(AiegraphError, match="no free call"):
        plan_group(cg, g, [group_step("p1", reads=g.inputs), group_step("p2", reads=g.inputs)])


def test_duplicate_step_names_raise():
    cg = cg_of(CHAIN)
    g = make_group(cg, [1, 2], "bc")
    with pytest.raises(AiegraphError, match="repeat"):
        plan_group(cg, g, [group_step("s", reads=g.inputs), group_step("s", writes=g.outputs)])


def test_bad_step_name_and_tensor():
    with pytest.raises(AiegraphError, match="step name"):
        group_step("not a name")
    cg = cg_of(CHAIN)
    with pytest.raises(AiegraphError, match="names no graph tensor"):
        plan_group(cg, make_group(cg, [1, 2]), [group_step("s", reads="q")])


def test_const_read_is_allowed():
    cg = cg_of(CHAIN)
    g = make_group(cg, [0, 1], "ab")
    p = plan_group(cg, g, [group_step("run", reads=[T(cg, "x"), T(cg, "w")], writes=T(cg, "b"))])
    assert p.placement == {"run": 0} and p.removed == [1]


# ── liveness: an input buffer overwritten before the step reads it ──────────────────
def test_overwritten_buffer_raises():
    """x lives in storage 1 until its last TVM reader [1]; outside call [2] reuses
    storage 1 for z. A step reading x AND w can only sit at [3] (w is made at [2]) --
    by then x's buffer holds z."""
    spec = [("in", "i", 0),
            ("op", "x", "P", ["i"], 1),          # [0]  x  -> storage 1
            ("op", "a", "G1", ["x"], 2),         # [1]  group, last reader of x
            ("op", "z", "K", ["i"], 1),          # [2]  outside: z reuses storage 1
            ("op", "o", "G2", ["a", "z"], 3)]    # [3]  group
    cg = cg_of(spec)
    g = make_group(cg, [1, 3], "g")
    assert {t.tensor for t in g.inputs} == {T(cg, "x"), T(cg, "z")}
    with pytest.raises(AiegraphError, match=r"overwritten buffer .*tensors\[%d\].*call \[2\] K"
                       % T(cg, "x")):
        plan_group(cg, g, [group_step("run", reads=[T(cg, "x"), T(cg, "z")], writes=T(cg, "o"))])


def test_reading_at_the_last_reader_is_fine():
    spec = [("in", "i", 0), ("op", "x", "P", ["i"], 1), ("op", "a", "G1", ["x"], 2),
            ("op", "z", "K", ["i"], 1), ("op", "o", "G2", ["a", "z"], 3)]
    cg = cg_of(spec)
    g = make_group(cg, [1, 3], "g")
    p = plan_group(cg, g, [group_step("px", reads=T(cg, "x")),
                           group_step("run", reads=T(cg, "z"), writes=T(cg, "o"))])
    assert p.placement == {"px": 1, "run": 3}


# ── early write ──────────────────────────────────────────────────────────────────
EARLY = [("in", "i", 0),
         ("op", "x", "P", ["i"], 1),             # [0]
         ("op", "a", "G1", ["x"], 2),            # [1] group
         ("op", "y", "K", ["i"], 3),             # [2] outside, y in storage 3
         ("op", "u", "U", ["y"], 4),             # [3] outside, last reader of y
         ("op", "o", "G2", ["a"], 3),            # [4] group, o reuses storage 3
         ("op", "r", "R", ["o", "u"], 5)]        # [5]


def test_early_write_conflict_raises():
    """The step writes o at [1] (it only needs x) but o's buffer (storage 3) still holds
    y, written at [2] and read at [3] -- writing early clobbers y."""
    cg = cg_of(EARLY)
    g = make_group(cg, [1, 4], "g")
    with pytest.raises(AiegraphError, match=r"early-write conflict .*tensors\[%d\] at \[1\] "
                       r"instead of \[4\].*call \[2\] K" % T(cg, "o")):
        plan_group(cg, g, [group_step("run", reads=T(cg, "x"), writes=T(cg, "o"))])


def test_write_at_producer_slot_is_fine():
    cg = cg_of(EARLY)
    g = make_group(cg, [1, 4], "g")
    p = plan_group(cg, g, [group_step("pack", reads=T(cg, "x")), group_step("run", writes=T(cg, "o"))])
    assert p.placement == {"pack": 1, "run": 4}


def test_early_write_then_read_same_buffer_raises():
    """A step writes o (storage 1) before a later step reads v, also in storage 1."""
    spec = [("in", "i", 0),
            ("op", "v", "P", ["i"], 1),          # [0] v in storage 1
            ("op", "q", "Q", ["i"], 2),          # [1]
            ("op", "a", "G1", ["q"], 3),         # [2] group
            ("op", "b", "G2", ["a"], 4),         # [3] group
            ("op", "c", "G3", ["b", "v"], 5),    # [4] group, last reader of v
            ("op", "o", "G4", ["c"], 1)]         # [5] group, o reuses storage 1
    cg = cg_of(spec)
    g = make_group(cg, [2, 3, 4, 5], "g")
    bad = [group_step("pq", reads=T(cg, "q")), group_step("run", writes=T(cg, "o")),
           group_step("pv", reads=T(cg, "v"))]
    with pytest.raises(AiegraphError, match="early-write conflict .*step run .*step pv"):
        plan_group(cg, g, bad)
    good = [group_step("pq", reads=T(cg, "q")), group_step("pv", reads=T(cg, "v")),
            group_step("run", writes=T(cg, "o"))]
    assert plan_group(cg, g, good).placement == {"pq": 2, "pv": 4, "run": 5}


# ── partition.json and offload_reset ────────────────────────────────────────────────
def _workdir(tmp_path, spec, layers=()):
    (tmp_path / "net_graph.json").write_text(json.dumps(graph(spec)))
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "partition.json").write_text(json.dumps({"layers": list(layers)}))
    return tmp_path


def test_record_group_in_partition_json(tmp_path):
    wd = _workdir(tmp_path, CHAIN)
    emod = EmbeddedModel(wd)
    g = emod.group([1, 2], "bc")
    emod.offload_group(g, "k.cc", [group_step("pack", reads=g.inputs, consts={"s": 0.5}),
                                          group_step("run", writes=g.outputs)])
    (rec,) = emod.groups
    assert rec["name"] == "bc" and rec["calls"] == [1, 2] and rec["source"] == "k.cc"
    assert rec["members"] == [{"call": 1, "symbol": "tvmgen_default_fused_B", "action": "step",
                               "step": "pack"},
                              {"call": 2, "symbol": "tvmgen_default_fused_C", "action": "step",
                               "step": "run"}]
    a = T(emod.call_graph(), "a")
    assert rec["steps"][0] == {"name": "pack", "at": 1, "reads": [a], "writes": [],
                               "consts": {"s": 0.5}}
    assert rec["inputs"] == [a]


def test_record_overlapping_group_raises(tmp_path):
    wd = _workdir(tmp_path, CHAIN)
    cg = CallGraph.load(wd)
    g1, g2 = make_group(cg, [1, 2], "bc"), make_group(cg, [2, 3], "cd")
    record_group(wd, plan_group(cg, g1, [group_step("r", reads=g1.inputs, writes=g1.outputs)]))
    with pytest.raises(AiegraphError, match="already belong to group bc"):
        record_group(wd, plan_group(cg, g2, [group_step("r", reads=g2.inputs, writes=g2.outputs)]))
    # re-recording the same group replaces it
    record_group(wd, plan_group(cg, g1, [group_step("r2", reads=g1.inputs, writes=g1.outputs)]))
    assert [g["steps"][0]["name"] for g in recorded_groups(wd)] == ["r2"]


def test_offload_group_failing_check_records_nothing(tmp_path):
    wd = _workdir(tmp_path, EARLY)
    emod = EmbeddedModel(wd)
    with pytest.raises(AiegraphError, match="early-write"):
        emod.offload_group([1, 4], None, [group_step("run", reads=T(emod.call_graph(), "x"),
                                                     writes=T(emod.call_graph(), "o"))])
    assert emod.groups == []


def test_offload_reset_whole_group(tmp_path):
    aie_row = {"index": 7, "dir": "07_x", "symbol": "s", "target": "aie", "replaces": "s"}
    wd = _workdir(tmp_path, EARLY, layers=[aie_row])
    emod = EmbeddedModel(wd)
    cg = emod.call_graph()
    emod.offload_group(emod.group([1, 4], "g14"), None,
                       [group_step("pack", reads=T(cg, "x")), group_step("run", writes=T(cg, "o"))])
    emod.offload_group(emod.group([2, 3], "g23"), None,
                       [group_step("run", reads=T(cg, "i"), writes=T(cg, "u"))])
    assert [g["name"] for g in emod.groups] == ["g14", "g23"]
    # one group by name: all its members go, the other group and the layer rows stay
    assert emod.offload_reset("group:g14") == ["group:g14"]
    assert [g["name"] for g in emod.groups] == ["g23"]
    assert json.loads((wd / "layers" / "partition.json").read_text())["layers"][0]["target"] == "aie"
    # a member call takes its whole group out (never half a group)
    assert reset_groups(wd, calls=[3]) == ["g23"] and emod.groups == []
    # reset everything: groups and AIE layer rows
    emod.offload_group(emod.group([1, 4], "g14"), None,
                       [group_step("pack", reads=T(cg, "x")), group_step("run", writes=T(cg, "o"))])
    emod.offload_group(emod.group([2, 3], "g23"), None,
                       [group_step("run", reads=T(cg, "i"), writes=T(cg, "u"))])
    with pytest.raises(AiegraphError, match="no recorded group"):
        reset_groups(wd, names=["nope"])
    out = emod.offload_reset()
    assert "group:g14" in out and "group:g23" in out and "layer7" in out
    assert emod.groups == []
    assert json.loads((wd / "layers" / "partition.json").read_text())["layers"][0]["target"] == "cpu"


# ── the real bk-sdm-tiny UNet graph ─────────────────────────────────────────────────
@pytest.fixture(scope="module")
def unet():
    raw = gzip.decompress(UNET_GRAPH.read_bytes())
    assert hashlib.sha1(raw).hexdigest() == UNET_GRAPH_SHA1
    return CallGraph(json.loads(raw), UNET_GRAPH)


def _fa_steps(g):
    q, k, v = g.inputs
    (o,) = g.outputs
    return [group_step("fa_pack_q", reads=q, consts={"cls": 0, "s": 0.398}),
            group_step("fa_pack_k", reads=k, consts={"cls": 0, "s": 0.398}),
            group_step("fa_pack_v", reads=v, consts={"cls": 0}),
            group_step("fa_run_o", writes=o, consts={"cls": 0})]


def test_bk_sdm_tiny_attn0_group(unet, tmp_path):
    """Design §3.1 / §6.1: attn0 = calls [31]-[37]; q/k/v = tensors[58/59/60], o =
    tensors[70]; pack q/k/v at [31]/[32]/[35], FA at [36] writing [37]'s output,
    [33] [34] [37] removed; recorded in partition.json and reset as a whole."""
    groups = attention_groups(unet)
    assert len(groups) == 18 and {len(g.calls) for g in groups} == {7}
    g = groups[0]
    assert g.name == "attn0" and g.calls == (31, 32, 33, 34, 35, 36, 37)
    assert [t.tensor for t in g.inputs] == [58, 59, 60]
    assert all(t.shape == (1, 1024, 320) and t.ld == 320 for t in g.inputs)
    assert [(t.tensor, t.shape) for t in g.outputs] == [(70, (1024, 320))]
    assert [unet.calls[c].fn for c in (33, 34, 36)] == ["nn_batch_matmul", "nn_softmax",
                                                         "nn_batch_matmul_1"]
    plan = plan_group(unet, g, _fa_steps(g))
    assert plan.placement == {"fa_pack_q": 31, "fa_pack_k": 32, "fa_pack_v": 35, "fa_run_o": 36}
    assert plan.removed == [33, 34, 37]
    com = plan.driver_comments()
    assert com[31] == "/* [31] */" and com[37] == "/* [37] removed: group attn0 */"
    # the same through EmbeddedModel on a workdir holding this graph.json
    (tmp_path / "unet_graph.json").write_bytes(gzip.decompress(UNET_GRAPH.read_bytes()))
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "partition.json").write_text(json.dumps({"layers": []}))
    emod = EmbeddedModel(tmp_path)
    g0 = emod.match("attention")[0]
    emod.offload_group(g0, "fa_attn.cc", _fa_steps(g0))
    (rec,) = emod.groups
    assert rec["calls"] == list(range(31, 38))
    assert [m["action"] for m in rec["members"]] == ["step", "step", "removed", "removed",
                                                     "step", "step", "removed"]
    assert [(s["name"], s["at"]) for s in rec["steps"]] == [("fa_pack_q", 31), ("fa_pack_k", 32),
                                                             ("fa_pack_v", 35), ("fa_run_o", 36)]
    assert rec["steps"][3]["writes"] == [70]
    assert emod.offload_reset("group:attn0") == ["group:attn0"] and emod.groups == []


def test_bk_sdm_tiny_all_18_groups_plan(unet):
    for g in attention_groups(unet):
        plan = plan_group(unet, g, _fa_steps(g))
        c = g.calls
        assert plan.placement == {"fa_pack_q": c[0], "fa_pack_k": c[1], "fa_pack_v": c[4],
                                  "fa_run_o": c[5]}, g.name
        assert plan.removed == [c[2], c[3], c[6]]


def test_bk_sdm_tiny_fa_before_v_pack_raises(unet):
    """Real storage plan: attn0's o (tensors[70]) reuses v's buffer (tensors[60]); running FA
    before packing V would clobber V."""
    g = attention_groups(unet)[0]
    assert g.outputs[0].sid == g.inputs[2].sid
    s = _fa_steps(g)
    with pytest.raises(AiegraphError, match="early-write conflict .*fa_run_o .*fa_pack_v"):
        plan_group(unet, g, [s[0], s[1], s[3], s[2]])


def test_bk_sdm_tiny_group_without_merge_raises(unet):
    """attn0 without [37]: the PV output tensors[69] is read by [37] outside the group."""
    g = make_group(unet, range(31, 37), "attn0_short")
    s = _fa_steps(attention_groups(unet)[0])[:3]
    with pytest.raises(AiegraphError, match=r"external reader .*tensors\[69\].*call \[37\]"):
        plan_group(unet, g, s)


def test_bk_sdm_tiny_reading_scores_raises(unet):
    g = attention_groups(unet)[0]
    with pytest.raises(AiegraphError, match="group produces"):
        plan_group(unet, g, [group_step("bad", reads=64, writes=g.outputs)])


@pytest.mark.skipif(not UNET_ONNX.is_file(), reason="bk-sdm-tiny ONNX not on this machine")
def test_bk_sdm_tiny_groups_match_onnx_attention(unet):
    """The graph-level groups are exactly the ONNX attention cores (partition.match_attention)."""
    from aiegraph.partition import match_attention
    assert len(match_attention(UNET_ONNX)) == len(attention_groups(unet)) == 18


def test_offload_reset_member_layer_takes_whole_group(tmp_path, monkeypatch):
    """Resetting a TVM layer one of whose calls is in a group reverts the whole group."""
    from aiegraph.embedded import Layer
    wd = _workdir(tmp_path, EARLY)
    emod = EmbeddedModel(wd)
    cg = emod.call_graph()
    emod.offload_group(emod.group([1, 4], "g14"), None,
                       [group_step("pack", reads=T(cg, "x")), group_step("run", writes=T(cg, "o"))])
    monkeypatch.setattr(EmbeddedModel, "layers", property(lambda self: [
        Layer(9, "tvmgen_default_fused_G2", "09_g2", [4], "g2", "-", "cpu")]))
    monkeypatch.setattr(emod, "resolve", lambda ref: [9])
    assert emod.offload_reset("layer9") == ["group:g14"]
    assert emod.groups == []
