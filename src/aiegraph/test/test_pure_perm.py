"""unet-m4 P2 (gap 2): pure-permutation check on saved Relay op semantics (aiegraph.pure_perm).

x86 only. ``pos`` tests: calls that only permute data (+ the allowed element-wise ops) pass, both on hand-written
FusedFn records and on the ``<stem>_opsem.json`` that ``deploy_flow.build_c`` records from a real (tiny) Relay
build, aligned with its graph.json. ``neg`` tests (counter-examples): transpose+multiply without the multiply
allowed, element count not conserved (padding layout_transform), multi-output, non-permutation axes, a stale
sidecar, and the FA head split with an ``exp`` (moved from test_keep_layout_codegen.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiegraph import pure_perm  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.group import CallGraph  # noqa: E402
from aiegraph.pure_perm import FusedFn, why_impure  # noqa: E402


def _op(name, ins, out, **attrs):
    return {"op": name, "ins": [list(s) for s in ins], "out": list(out), "attrs": attrs}


def _fa_q():
    """head split of Q with the softmax scale: [1,N,H*D] -> [H,N,D] * s"""
    return FusedFn("tvmgen_default_fused_reshape_transpose_multiply_reshape", [
        _op("reshape", [[1, 4, 4]], [1, 4, 2, 2], newshape=[1, 4, 2, 2]),
        _op("transpose", [[1, 4, 2, 2]], [1, 2, 4, 2], axes=[0, 2, 1, 3]),
        _op("multiply", [[1, 2, 4, 2], []], [1, 2, 4, 2]),
        _op("reshape", [[1, 2, 4, 2]], [2, 4, 2], newshape=[2, 4, 2])], [[1, 4, 4], []], [[2, 4, 2]])


# ─────────────────────────────────────────────────────────── relay: a real recorded graph

@pytest.fixture(scope="module")
def relay_graph(tmp_path_factory):
    """tiny Relay net built through deploy_flow.build_c: graph.json + the recorded <stem>_opsem.json"""
    try:
        import aiegraph._env as env
        env.attach_tvm()
        import tvm
        from tvm import relay
        from frontend.tvmrelay.deploy_flow import build_c
    except Exception as e:   # pragma: no cover - TVM-free boxes
        pytest.skip(f"TVM 0.16 not importable: {e}")
    x = relay.var("x", shape=(1, 4, 8), dtype="float32")
    d = relay.nn.dense(relay.reshape(x, (4, 8)), relay.const(np.ones((8, 8), "float32")))
    d = relay.nn.relu(d)
    # [A] pure: head split reshape + transpose(0,2,1,3) + reshape (FA V-like)
    a = relay.reshape(relay.transpose(relay.reshape(d, (1, 4, 2, 4)), (0, 2, 1, 3)), (2, 4, 4))
    a = relay.nn.softmax(a)   # opaque: fusion barrier
    # [B] transpose + multiply by a scale (FA Q-like): pure only with multiply allowed
    b = relay.reshape(relay.multiply(relay.transpose(relay.reshape(a, (1, 2, 4, 4)), (0, 2, 1, 3)),
                                     relay.const(0.5)), (1, 4, 8))
    b = relay.nn.softmax(b)
    # [C] layout_transform NCHW -> NCHW4c with C=3: pads, element count not conserved
    c = relay.layout_transform(relay.reshape(relay.strided_slice(b, [0, 0, 0], [1, 3, 8]), (1, 3, 8, 1)),
                               "NCHW", "NCHW4c")
    c = relay.nn.relu(c)
    out = relay.add(relay.sum(c), relay.sum(b))
    mod = tvm.IRModule.from_expr(relay.Function([x], out))
    out_dir = tmp_path_factory.mktemp("relay_opsem")
    build_c(mod, {}, out_dir, verbose=False)
    cg = CallGraph.load(out_dir)
    return cg, out_dir


def _call_with(cg, *ops):
    """the call whose recorded Relay op sequence is exactly ``ops``"""
    sem = pure_perm.load(cg)
    hit = [c.call for c in cg.calls if sem.fn(cg, c.call).op_names == list(ops)]
    assert hit, [sem.fn(cg, c.call).op_names for c in cg.calls]
    return hit[0]


def test_pos_relay_opsem_recorded_and_aligned(relay_graph):
    cg, out_dir = relay_graph
    p = pure_perm.opsem_path(cg.path)
    assert p == out_dir / "resnet18_opsem.json" and p.is_file()
    doc = json.loads(p.read_text())
    g = json.loads(cg.path.read_text())
    tnodes = [i for i, n in enumerate(g["nodes"]) if n.get("op") == "tvm_op"]
    assert doc["n_tvm_op"] == len(tnodes) and sorted(int(k) for k in doc["nodes"]) == tnodes
    assert doc["name_mismatch"] == 0
    for k, v in doc["nodes"].items():                     # every entry names its graph.json node
        assert v["func_name"] == g["nodes"][int(k)]["attrs"]["func_name"]
    sem = pure_perm.load(cg)
    for c in cg.calls:
        assert sem.fn(cg, c.call).name_ok


def test_pos_relay_head_split_is_pure(relay_graph):
    cg, _ = relay_graph
    k = _call_with(cg, "reshape", "transpose", "reshape")
    assert pure_perm.check(cg, k) == ["reshape", "transpose", "reshape"]
    t = [o for o in pure_perm.load(cg).fn(cg, k).ops if o["op"] == "transpose"][0]
    assert t["attrs"]["axes"] == [0, 2, 1, 3] and t["ins"] == [[1, 4, 2, 4]] and t["out"] == [1, 2, 4, 4]


def test_pos_relay_scale_multiply_allowed(relay_graph):
    cg, _ = relay_graph
    k = _call_with(cg, "reshape", "transpose", "multiply", "reshape")
    assert pure_perm.check(cg, k, extra=("multiply",), need=("multiply",)) == \
        ["reshape", "transpose", "multiply", "reshape"]


def test_pos_fa_scale_and_label_ops():
    assert why_impure(_fa_q(), extra=("multiply",), need=("multiply",)) == ""
    lt = FusedFn("tvmgen_default_fused_layout_transform_transpose_reshape", [
        _op("layout_transform", [[1, 1, 4, 4, 8]], [1, 8, 4, 4], src_layout="NCHW8c", dst_layout="NCHW"),
        _op("transpose", [[1, 8, 4, 4]], [1, 4, 4, 8], axes=[0, 2, 3, 1]),
        _op("reshape", [[1, 4, 4, 8]], [1, 16, 8])], [[1, 1, 4, 4, 8]], [[1, 16, 8]])
    assert why_impure(lt) == ""


def test_pos_temb_sigmoid_multiply_layout():
    """temb fold: W [N,K] * sigmoid(t) -> dense_pack [N/8][K][8]"""
    f = FusedFn("tvmgen_default_fused_sigmoid_squeeze_multiply_layout_transform", [
        _op("sigmoid", [[1, 4]], [1, 4]), _op("squeeze", [[1, 4]], [4]),
        _op("multiply", [[16, 4], [4]], [16, 4]),
        _op("layout_transform", [[16, 4]], [2, 4, 8], src_layout="NC", dst_layout="NC8n")],
        [[1, 4], [16, 4]], [[2, 4, 8]])
    assert why_impure(f, extra=("multiply", "sigmoid"), need=("multiply", "layout_transform")) == ""


def test_pos_keep_layout_pure_uses_opsem(tmp_path):
    """keep_layout_codegen._pure goes through pure_perm (no symbol parsing): a record whose ops differ from
    the symbol's words decides by the ops"""
    from aiegraph import keep_layout_codegen as klc
    from test_keep_layout_codegen import _fa_net, opsem
    n, _, t = _fa_net()
    (tmp_path / "net_graph.json").write_text(json.dumps(n.graph()))
    doc = opsem(n)
    (tmp_path / "net_opsem.json").write_text(json.dumps(doc))
    cg = CallGraph.load(tmp_path)
    k = cg.producer[t["o2"]]
    assert klc._pure(cg, k) == ["reshape", "transpose", "reshape"]
    doc["nodes"][str(cg.calls[k].node)]["ops"][1]["op"] = "nn.relu"     # same symbol, impure semantics
    (tmp_path / "net_opsem.json").write_text(json.dumps(doc))
    with pytest.raises(AiegraphError, match="not a pure permutation"):
        klc._pure(CallGraph.load(tmp_path), k)


# ─────────────────────────────────────────────────────────── counter-examples

def test_neg_transpose_multiply_not_allowed(relay_graph):
    cg, _ = relay_graph
    k = _call_with(cg, "reshape", "transpose", "multiply", "reshape")
    with pytest.raises(AiegraphError, match=r"not a pure permutation .*\['multiply'\] not allowed"):
        pure_perm.check(cg, k)
    assert "not allowed" in why_impure(_fa_q())


def test_neg_element_count_not_conserved(relay_graph):
    cg, _ = relay_graph
    k = _call_with(cg, "strided_slice", "reshape", "layout_transform")
    with pytest.raises(AiegraphError, match="not a pure permutation"):
        pure_perm.check(cg, k, extra=("strided_slice",))
    f = pure_perm.load(cg).fn(cg, k)
    lt = FusedFn(f.func_name, [o for o in f.ops if o["op"] == "layout_transform"], [[1, 3, 8, 1]], [f.outs[0]])
    assert lt.ops[0]["out"] == [1, 1, 8, 1, 4]                            # C=3 padded to 4
    assert "element count not conserved" in why_impure(lt)


def test_neg_multi_output():
    f = FusedFn("tvmgen_default_fused_reshape_transpose", [
        _op("reshape", [[1, 4, 4]], [1, 4, 2, 2]), _op("transpose", [[1, 4, 2, 2]], [1, 2, 4, 2], axes=[0, 2, 1, 3])],
        [[1, 4, 4]], [[1, 2, 4, 2], [1, 4, 2, 2]])
    assert "2 outputs" in why_impure(f)


def test_neg_transpose_axes_not_permutation():
    f = FusedFn("tvmgen_default_fused_transpose", [_op("transpose", [[2, 3, 4]], [2, 3, 4], axes=[0, 0, 2])],
                [[2, 3, 4]], [[2, 3, 4]])
    assert "not a permutation" in why_impure(f)


def test_neg_stale_or_missing_opsem(tmp_path):
    from test_keep_layout_codegen import _fa_net, opsem
    n, _, t = _fa_net()
    (tmp_path / "net_graph.json").write_text(json.dumps(n.graph()))
    cg = CallGraph.load(tmp_path)
    with pytest.raises(AiegraphError, match="no Relay op semantics"):
        pure_perm.check(cg, cg.producer[t["o2"]])
    doc = opsem(n)
    nid = str(cg.calls[cg.producer[t["o2"]]].node)
    doc["nodes"][nid]["func_name"] = "tvmgen_default_fused_reshape_transpose_reshape_9"
    (tmp_path / "net_opsem.json").write_text(json.dumps(doc))
    cg = CallGraph.load(tmp_path)
    with pytest.raises(AiegraphError, match="stale"):
        pure_perm.check(cg, cg.producer[t["o2"]])


def test_neg_fa_not_a_permutation_fails(tmp_path):
    """(moved from test_keep_layout_codegen.py) an ``exp`` in the Q head split: the FA rewrite must fail"""
    from test_keep_layout_codegen import _fa_net, _run
    n, w, _ = _fa_net(impure=True)
    with pytest.raises(AiegraphError, match=r"not a pure permutation .*\['exp'\] not allowed"):
        _run(tmp_path, n, w)
