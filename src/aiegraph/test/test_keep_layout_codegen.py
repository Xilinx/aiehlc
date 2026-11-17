###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""S5: keep-layout + temb fold in the arm_build codegen (aiegraph.keep_layout_codegen).

x86 only. Small synthetic graph.json + graph_driver.c in the arm_build format: each rewrite on a minimal
group, and the counter-examples that must fail the build -- an overwritten buffer (early write of O), an
intermediate tensor with a reader outside the group, an op that is not a pure permutation, a planned
keep_layout edge the codegen did not realize. Plus the strided FA pack check moved from the example's
test_keep_layout_rt.py (fa_attn.py packs straight from [rows][H*D] == packs of TVM's head split).
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph import keep_layout_codegen as klc  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.group import CallGraph  # noqa: E402

UNET = SRC.parent / "example" / "aiegraph" / "unet_bk_sdm_tiny"


# ─────────────────────────────────────────────────────────── synthetic graph + driver

class Net:
    """One output per node: tensor id == node id == ``tensors[i]`` of the driver."""

    def __init__(self):
        self.nodes, self.shape, self.sid, self.calls = [], [], [], []

    def const(self, name, shape):
        self.nodes.append({"op": "null", "name": name, "inputs": []})
        self.shape.append(list(shape))
        self.sid.append(len(self.sid) + 100)
        return len(self.nodes) - 1

    def call(self, fn, ins, shape, sid=None, site=None, expand=()):
        """site: driver text of an AIE call site instead of the TVM call; expand: [(tensor, g_params off, n)]"""
        t = len(self.nodes)
        self.nodes.append({"op": "tvm_op", "name": fn, "inputs": [[i, 0, 0] for i in ins],
                           "attrs": {"func_name": "tvmgen_default_fused_" + fn, "num_outputs": "1",
                                     "flatten_data": "0", "num_inputs": str(len(ins))}})
        self.shape.append(list(shape))
        self.sid.append(t if sid is None else sid)
        self.calls.append((fn, list(ins), t, site, list(expand)))
        return t

    def graph(self):
        n = len(self.nodes)
        return {"nodes": self.nodes, "arg_nodes": [i for i, x in enumerate(self.nodes) if x["op"] == "null"],
                "heads": [[n - 1, 0, 0]], "node_row_ptr": list(range(n + 1)),
                "attrs": {"storage_id": ["list_int", self.sid], "shape": ["list_shape", self.shape],
                          "dltype": ["list_str", ["float32"] * n]}}

    def driver(self):
        out = ["/* synthetic arm_build driver */\n"]
        for t, s in enumerate(self.shape):
            for k, d in enumerate(s):
                out.append("    shape_store[%d][%d] = %d;\n" % (t, k, d))
        out.append("float *graph_run(void) {\n")
        ws = 0
        for k, (fn, ins, o, site, expand) in enumerate(self.calls):
            for t, off, cnt in expand:
                out.append("    graph_bf16_expand((float *)(g_wscratch + %d), (const uint16_t *)(g_params + %d), "
                           "%dULL);\n    tensors[%d].data = g_wscratch + %d;\n" % (ws, off, cnt, t, ws))
                ws += 4 * cnt
            for i, t in enumerate(ins + [o]):
                out.append("    args[%d].v_handle = &tensors[%d];\n" % (i, t))
            if site:
                out.append("    GRAPH_PROF_BEGIN(%d);\n    UP_AIE_BEGIN(%d);\n    %s\n    UP_AIE_END(%d);\n"
                           "    GRAPH_PROF_END(%d, \"aie %s\");\n" % (k, k, site, k, k, site.split("(")[0]))
            else:
                out.append("    GRAPH_PROF_BEGIN(%d);\n    tvmgen_default_fused_%s(args, codes, %d, 0, 0, 0);\n"
                           "    GRAPH_PROF_END(%d, \"cpu %s\");\n" % (k, fn, len(ins) + 1, k, fn))
        out.append("    return 0;\n}\n")
        return "".join(out)


def _run(tmp_path, net, wbin=None, plan=None):
    (tmp_path / "net_graph.json").write_text(json.dumps(net.graph()))
    drv = tmp_path / "graph_driver.c"
    drv.write_text(net.driver())
    wb = tmp_path / "weights.bin"
    if wbin is None:
        wbin = np.zeros(64, np.uint16)
    wbin.astype(np.uint16).tofile(wb)
    logs = []
    rep = klc.apply(drv, tmp_path / "net_graph.json", wb, plan=plan, log=logs.append)
    return drv.read_text(), np.fromfile(wb, np.uint16), rep, logs


def _bf16(x):
    return (np.array([x], np.float32).view(np.uint32)[0] >> 16).astype(np.uint16)


def _fa_net(H=2, N=4, M=4, D=2, clobber=False, outside_reader=False, impure=False):
    """linear outs Q/K/V [1,R,H*D] -> head split Q, K (x scale), V^T -> QK site -> softmax -> PV site -> head merge"""
    C = H * D
    n = Net()
    x = n.const("x", [1, N, C])
    sq = n.const("p_scale_q", [])
    sk = n.const("p_scale_k", [])
    ql = n.call("nn_dense", [x], [1, N, C])
    kl = n.call("nn_dense_1", [x], [1, M, C])
    vl = n.call("nn_dense_2", [x], [1, M, C])
    tq = n.call("reshape_exp_transpose" if impure else "reshape_transpose_multiply_reshape", [ql, sq], [H, N, D],
                expand=[(sq, 0, 1)])
    tk = n.call("reshape_transpose_multiply_reshape_transpose", [kl, sk], [H, M, D], expand=[(sk, 2, 1)])
    ts = n.call("nn_batch_matmul", [tq, tk], [H, N, M], site="unet_aie_attn(7, args, 3, 0, 0);")
    if outside_reader:
        n.call("nn_relu", [tq], [H, N, D])
    tp = n.call("nn_softmax", [ts], [H, N, M])
    tv = n.call("reshape_transpose_reshape_transpose", [vl], [H, D, M])
    to = n.call("nn_batch_matmul_1", [tp, tv], [H, N, D], site="unet_aie_attn(8, args, 3, 1, 0);")
    o2_sid = 900
    if clobber:   # a kept call between PV and the head merge writes the merged O buffer
        n.call("nn_relu_1", [vl], [1, N, C], sid=o2_sid)
    o2 = n.call("reshape_transpose_reshape", [to], [1, N, C], sid=o2_sid)
    n.call("nn_dense_3", [o2], [1, N, C])
    w = np.zeros(64, np.uint16)
    w[0] = _bf16(0.5)
    w[1] = _bf16(0.5)
    return n, w, dict(ql=ql, kl=kl, vl=vl, o2=o2)


# ─────────────────────────────────────────────────────────── purity on graph.json

def test_fused_ops_names():
    assert klc.fused_ops("tvmgen_default_fused_reshape_transpose_multiply_reshape_3") == \
        ["reshape", "transpose", "multiply", "reshape"]
    assert klc.fused_ops("tvmgen_default_fused_layout_transform_transpose_reshape_2") == \
        ["layout_transform", "transpose", "reshape"]
    assert klc.fused_ops("tvmgen_default_fused_sigmoid_squeeze_multiply_layout_transform") == \
        ["sigmoid", "squeeze", "multiply", "layout_transform"]
    assert klc.fused_ops("tvmgen_default_fused_reshape_exp_transpose") is None
    assert klc.fused_ops("tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add") is None


# ─────────────────────────────────────────────────────────── FA group

def test_fa_group_rewrite(tmp_path):
    n, w, t = _fa_net()
    text, _, rep, logs = _run(tmp_path, n, w)
    assert rep["attn"] == 1
    sites = re.findall(r"unet_aie_attn_kl\((\d+), (\d+), (\d+), \(?(?:const float \*\)tensors\[(\d+)\]\.data|0)", text)
    assert [(s[1], s[3]) for s in sites] == [("2", str(t["ql"])), ("3", str(t["kl"])), ("5", str(t["vl"])), ("6", "")]
    assert "(float *)tensors[%d].data, 4, 1.0f" % t["o2"] in text      # O written by stride into the merge output
    assert "5.000000000e-01f" in text                                    # scale from weights.bin, in the pack
    assert text.count("removed --") == 2                                  # QK site + head merge
    assert "tvmgen_default_fused_nn_softmax(" in text                     # not part of this rewrite


def test_fa_overwritten_buffer_fails(tmp_path):
    n, w, _ = _fa_net(clobber=True)
    with pytest.raises(AiegraphError, match="overwritten buffer"):
        _run(tmp_path, n, w)


def test_fa_intermediate_outside_reader_fails(tmp_path):
    n, w, _ = _fa_net(outside_reader=True)
    with pytest.raises(AiegraphError, match="read outside the group"):
        _run(tmp_path, n, w)


def test_fa_not_a_permutation_fails(tmp_path):
    n, w, _ = _fa_net(impure=True)
    with pytest.raises(AiegraphError, match="not a pure permutation"):
        _run(tmp_path, n, w)


# ─────────────────────────────────────────────────────────── temb fold

def test_temb_fold_prepacks_weight(tmp_path):
    N, K = 16, 4
    n = Net()
    tt = n.const("t", [1, K])
    W = n.const("p_w", [N, K])
    b = n.const("p_b", [1, N])
    wp = n.call("sigmoid_squeeze_multiply_layout_transform", [tt, W], [N // 8, K, 8], expand=[(W, 0, N * K)])
    n.call("nn_contrib_dense_pack_add", [tt, wp, b], [1, N])
    w = np.arange(N * K + 8, dtype=np.uint16)
    text, w2, rep, _ = _run(tmp_path, n, w)
    assert rep["temb"] == 1
    want = np.arange(N * K, dtype=np.uint16).reshape(N // 8, 8, K).transpose(0, 2, 1).reshape(-1)
    assert np.array_equal(w2[:N * K], want) and np.array_equal(w2[N * K:], w[N * K:])   # bit-exact permutation
    assert "kl_temb_x(&tensors[%d], 0, %d);" % (tt, K) in text
    assert "args[0].v_handle = &g_kl_x_t;" in text
    assert "tvmgen_default_fused_sigmoid_squeeze_multiply_layout_transform(" not in text
    # dense(t, W*s) == dense(t*s, W): the fold the rewrite relies on (fp32)
    rng = np.random.default_rng(0)
    t_, W_ = rng.standard_normal(K).astype(np.float32), rng.standard_normal((N, K)).astype(np.float32)
    s_ = t_ / (1 + np.exp(-t_))
    assert np.allclose((W_ * s_) @ t_, W_ @ (t_ * s_), rtol=1e-5)


def test_temb_packed_weight_outside_reader_fails(tmp_path):
    N, K = 16, 4
    n = Net()
    tt = n.const("t", [1, K])
    W = n.const("p_w", [N, K])
    b = n.const("p_b", [1, N])
    wp = n.call("sigmoid_squeeze_multiply_layout_transform", [tt, W], [N // 8, K, 8], expand=[(W, 0, N * K)])
    n.call("nn_contrib_dense_pack_add", [tt, wp, b], [1, N])
    n.call("nn_relu", [wp], [N // 8, K, 8])
    with pytest.raises(AiegraphError, match="other readers"):
        _run(tmp_path, n, np.zeros(N * K, np.uint16))


# ─────────────────────────────────────────────────────────── conv out / transformer out

def _conv_net(other_reader=False, nhwc=False, epi_shares=False):
    n = Net()
    x = n.const("x", [1, 1, 4, 4, 8])
    w = n.const("p_w", [1, 1, 1, 1, 8, 8])
    if epi_shares:   # a residual epilogue operand in the buffer the layout op's output reuses
        res = n.call("nn_relu", [x], [1, 1, 4, 4, 8], sid=500)
    else:            # per-channel bias
        res = n.const("p_b", [1, 1, 1, 1, 8])
    ci = len(n.calls)   # driver call index of the AIE conv
    y = n.call("nn_contrib_conv2d_NCHWc_add", [x, w, res], [1, 1, 4, 4, 8], site="unet_aie_conv(3, args, 4);")
    if nhwc:
        z = n.call("layout_transform_transpose_reshape", [y], [1, 16, 8], sid=500 if epi_shares else None)
    else:
        z = n.call("layout_transform_reshape", [y], [1, 2, 64], sid=500 if epi_shares else None)
    if other_reader:
        n.call("nn_relu_1", [y], [1, 1, 4, 4, 8])
    n.call("nn_dense", [z], [1, 8])
    return n, y, z, ci


def test_conv_out_label(tmp_path):
    n, y, z, ci = _conv_net()
    text, _, rep, _ = _run(tmp_path, n, plan={"edges": [{"node": "c0", "call": ci}]})
    assert rep["conv_out"] == 1 and rep["planned"] == 1
    assert "g_kl_y_%d = tensors[%d]; g_kl_y_%d.data = tensors[%d].data;" % (ci, y, ci, z) in text
    assert "static int64_t g_kl_y_%d_shape[5] = {1, 8, 4, 4, 1};" % ci in text      # planar NCHW label
    assert "tvmgen_default_fused_layout_transform_reshape(" not in text


def test_conv_out_nhwc_label(tmp_path):
    n, _, _, ci = _conv_net(nhwc=True)
    text, _, rep, _ = _run(tmp_path, n)
    assert rep["conv_out"] == 1 and "g_kl_y_%d_shape[5] = {1, 1, 4, 4, 8};" % ci in text


def test_conv_out_kept_when_other_reader(tmp_path):
    n, _, _, _ = _conv_net(other_reader=True)
    text, _, rep, _ = _run(tmp_path, n)
    assert rep["conv_out"] == 0 and "tvmgen_default_fused_layout_transform_reshape(" in text


def test_conv_out_skips_clobbered_epilogue(tmp_path):
    n, _, _, _ = _conv_net(epi_shares=True)
    text, _, rep, logs = _run(tmp_path, n)
    assert rep["conv_out"] == 0 and any("SKIP" in m for m in logs)


def test_planned_edge_not_realized_fails(tmp_path):
    n, _, _, ci = _conv_net(other_reader=True)
    with pytest.raises(AiegraphError, match="not realized"):
        _run(tmp_path, n, plan={"edges": [{"node": "c0", "call": ci}]})


def _po_net(outside=False):
    n = Net()
    x = n.const("x", [1, 16, 8])
    wc = n.const("p_wc", [1, 1, 1, 1, 8, 8])
    r = n.call("nn_relu", [x], [1, 16, 8])
    tl = n.call("nn_contrib_dense_pack_add", [x], [1, 16, 8], site="unet_aie_linear(4, args, 2, 0);")
    to = n.call("reshape_add_reshape_transpose_layout_transform", [tl, r], [1, 1, 4, 4, 8])
    n.call("nn_contrib_conv2d_NCHWc", [to, wc], [1, 1, 4, 4, 8], site="unet_aie_conv(5, args, 3);")
    if outside:   # the removed layout op's output has a reader outside the group
        n.call("nn_relu_1", [to], [1, 1, 4, 4, 8])
    return n, tl, r


def test_proj_out_residual_into_scatter(tmp_path):
    n, tl, r = _po_net()
    text, _, rep, _ = _run(tmp_path, n)
    assert rep["proj_out"] == 1
    assert "unet_aie_linear(4, args, 3, 0);" in text and "args[1].v_handle = &tensors[%d];" % r in text
    assert "g_kl_nhwc_3 = tensors[%d];" % tl in text


def test_proj_out_outside_reader_fails(tmp_path):
    n, _, _ = _po_net(outside=True)
    with pytest.raises(AiegraphError, match="read outside the group"):
        _run(tmp_path, n)


def test_driver_must_match_graph(tmp_path):
    n, _, _, _ = _conv_net()
    n.calls[-1] = ("nn_relu_9",) + n.calls[-1][1:]     # driver names a call the graph does not have there
    with pytest.raises(AiegraphError, match="graph.json call"):
        _run(tmp_path, n)


def test_register_runs_in_arm_build(tmp_path):
    """the pass is registered per workdir and replaced (not stacked) on re-register"""
    import aiegraph._env as env
    try:
        ab = env.frontend("arm_build")
    except Exception as e:   # pragma: no cover - TVM-free boxes
        pytest.skip(f"arm_build not importable: {e}")
    klc.register(tmp_path)
    klc.register(tmp_path)
    assert [n for n, _ in ab._DRIVER_PASSES[str(tmp_path.resolve())]] == ["keep_layout"]
    ab.clear_driver_passes(tmp_path)
    assert str(tmp_path.resolve()) not in ab._DRIVER_PASSES


# ─────────────────────────────────────────────────────────── strided FA packs (from test_keep_layout_rt.py)

HARNESS = r'''
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#define UC_NEON 0
typedef struct { int H, N, M, D, split, dq, nkv, E, units, upr, launches, qkrow, vrow, orow; } fa_cls;
static uint16_t *g_fa_qk[1], *g_fa_v[1], *g_fa_o[1];
static float g_mu[256]; static float *g_fa_mu = g_mu;
static unsigned long long g_uc_macs; static unsigned g_fa_sites;
@SRC@
int main(int argc, char **argv) {
  int H = 8, N = atoi(argv[1]), M = atoi(argv[2]), D = atoi(argv[3]);
  int split = D > 80, dq = split ? D / 2 : D, nkv = (M + 31) / 32, E = 32 * dq;
  fa_cls c = {H, N, M, D, split, dq, nkv, E, split ? H : H * N / 128, split ? 1 : N / 128, 0, 8 * E * 2,
              nkv * 2 * (split ? 2 : 1) * E * 2, 8 * E * 2};
  size_t qkn = (size_t)(H * N / 32 + 64) * 4 * c.qkrow, vn = (size_t)H * c.vrow, on = (size_t)(H * N / 32 + 64) * 4 * c.orow;
  std::vector<uint16_t> qa(qkn), qb(qkn), va(vn), vb(vn), oo(on);
  int C = H * D; float s = powf((float)D, -0.25f);
  std::vector<float> Ql(N * C), Kl(M * C), Vl(M * C), Qt(H * N * D), Kt(H * M * D), VT(H * D * M);
  srand(1);
  for (auto *v : {&Ql, &Kl, &Vl}) for (auto &x : *v) x = (rand() % 2001 - 1000) / 1000.f;
  for (int h = 0; h < H; h++) for (int n = 0; n < N; n++) for (int d = 0; d < D; d++) Qt[(h * N + n) * D + d] = Ql[n * C + h * D + d] * s;
  for (int h = 0; h < H; h++) for (int j = 0; j < M; j++) for (int d = 0; d < D; d++) Kt[(h * M + j) * D + d] = Kl[j * C + h * D + d] * s;
  for (int h = 0; h < H; h++) for (int j = 0; j < M; j++) for (int d = 0; d < D; d++) VT[(h * D + d) * M + j] = Vl[j * C + h * D + d];
  fa_pack_q(c, Qt.data(), (size_t)N * D, D, 1.0f, qa.data()); fa_pack_q(c, Ql.data(), D, C, s, qb.data());
  fa_pack_k(c, Kt.data(), (size_t)M * D, D, 1.0f, va.data()); fa_pack_k(c, Kl.data(), D, C, s, vb.data());
  fa_pack_v(c, VT.data(), (size_t)D * M, 0, 1, va.data()); fa_pack_v(c, Vl.data(), D, C, 0, vb.data());
  long dq_ = 0, dk = 0, bigk = 0;
  for (size_t i = 0; i < qkn; i++) dq_ += qa[i] != qb[i];
  for (size_t i = 0; i < vn; i++) if (va[i] != vb[i]) { dk++; if (abs((int)va[i] - (int)vb[i]) > 2) bigk++; }
  for (size_t i = 0; i < on; i++) oo[i] = fa_f2bf((rand() % 2001 - 1000) / 1000.f);
  g_fa_o[0] = oo.data();
  std::vector<float> Yt(H * N * D, 7.f), Yl(N * C, 9.f);
  fa_run_o(c, 0, Yt.data(), (size_t)N * D, D); fa_run_o(c, 0, Yl.data(), D, C);
  long dO = 0;
  for (int h = 0; h < H; h++) for (int n = 0; n < N; n++) for (int d = 0; d < D; d++) dO += Yt[(h * N + n) * D + d] != Yl[n * C + h * D + d];
  printf("KLTEST qdiff=%ld kvdiff=%ld big=%ld odiff=%ld kvn=%zu\n", dq_, dk, bigk, dO, vn);
  return 0;
}
'''


def _fa_src() -> str:
    src = (UNET / "fa_attn.py").read_text()
    a = src.index("static inline uint16_t fa_f2bf(float f)")
    helpers = src[a:src.index("#if UC_NEON", a)]
    packs = src[src.index("static void fa_pack_k("):src.index("static int fa_run_o(")]
    runo = src[src.index("static int fa_run_o("):src.index("static int uc_fa(void *const *argv")]
    runo = runo[:runo.index("#if UC_CTRLPKT")] + runo[runo.index("    /* v5 O: o row pm"):]
    runo = runo.replace("    uint16_t *QK = g_fa_qk[k], *V = g_fa_v[k], *O = g_fa_o[k];", "    uint16_t *O = g_fa_o[k];")
    return HARNESS.replace("@SRC@", helpers + packs + runo)


@pytest.mark.skipif(shutil.which("g++") is None or not (UNET / "fa_attn.py").is_file(),
                    reason="needs g++ and the UNet example")
@pytest.mark.parametrize("N,M,D", [(1024, 1024, 40), (1024, 77, 40), (256, 256, 80), (256, 77, 80), (64, 64, 160),
                                   (64, 77, 160)])
def test_fa_strided_packs(tmp_path, N, M, D):
    (tmp_path / "t.cc").write_text(_fa_src())
    subprocess.run(["g++", "-O1", "-std=c++17", "-o", str(tmp_path / "t"), str(tmp_path / "t.cc")], check=True)
    out = subprocess.run([str(tmp_path / "t"), str(N), str(M), str(D)], check=True, capture_output=True, text=True).stdout
    m = re.search(r"KLTEST qdiff=(\d+) kvdiff=(\d+) big=(\d+) odiff=(\d+) kvn=(\d+)", out)
    assert m, out
    qd, kd, big, od, kvn = (int(x) for x in m.groups())
    assert qd == 0 and od == 0, out          # Q pack and O unpack bit-identical
    assert kd * 1000 < kvn and big <= 2, out  # K: (k - mu) * s vs k*s - mean(k*s): fp32 rounding only
