###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The aiednn graph API: partitioning, fusion, int8 dequantize read-through, numbers vs PyTorch."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aiegraph.dnn_ctypes import conv_params, pool_params
from aiegraph.dnn_graph import FIND_HEURISTIC, FIND_MEASURE, Graph, Stream, aligned_workspace, build_graph_lib


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_graph_lib(tmp_path_factory.mktemp("graphlib"))


def q8(x, s, zp=0, dt=np.int8):
    lo, hi = (-128, 127) if dt == np.int8 else (0, 255)
    return np.clip(np.round(x / s) + zp, lo, hi).astype(dt)   # numpy rounds half to even


def run(g, feeds, fallback=None):
    g.compile_all()
    return g.run(feeds, fallback)


def test_conv_bias_relu_f32_one_partition(lib):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 12, 12), np.float32)
    w = rng.standard_normal((16, 8, 3, 3), np.float32)
    b = rng.standard_normal(16).astype(np.float32)
    g = Graph(lib)
    g.tensor("x", x.shape, "f32"), g.tensor("w", w.shape, "f32", True), g.tensor("b", (16,), "f32", True)
    g.tensor("y", (1, 16, 12, 12), "f32"), g.tensor("yb", (1, 16, 12, 12), "f32"), g.tensor("z", (1, 16, 12, 12), "f32")
    g.add("convolution", ["x", "w"], "y", conv=conv_params((1, 1), (1, 1), (1, 1)))
    g.add("bias_add", ["y", "b"], "yb")
    g.add("relu", ["yb"], "z")
    parts = g.compile_all()
    assert len(parts) == 1 and parts[0].supported, parts
    assert parts[0].ops == ["convolution0", "bias_add1", "relu2"]
    assert parts[0].inputs == ["x", "w", "b"] and parts[0].outputs == ["z"]
    env = g.run({"x": x, "w": w, "b": b})
    ref = F.relu(F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), padding=1)).numpy()
    np.testing.assert_allclose(env["z"], ref, rtol=1e-4, atol=1e-4)


def _int8_block(lib, fuse=True):
    """x -Q-> xq -DQ-> conv(DQ(w s8, per channel)) + b -> relu -> + DQ(res) -Q-> out s8"""
    rng = np.random.default_rng(1)
    C, O, H = 16, 32, 10
    sx, sr, so = 0.05, 0.04, 0.08
    sw = (rng.random(O, np.float32) * 0.02 + 0.005).astype(np.float32)
    x = rng.standard_normal((1, C, H, H), np.float32)
    wq = rng.integers(-127, 128, (O, C, 3, 3), dtype=np.int8)
    b = rng.standard_normal(O).astype(np.float32)
    resq = rng.integers(-100, 100, (1, O, H, H), dtype=np.int8)
    g = Graph(lib, fuse=fuse)
    f = lambda n, s, dt="f32", c=False: g.tensor(n, s, dt, c)
    f("x", x.shape), f("xq", x.shape, "s8"), f("xd", x.shape), f("wq", wq.shape, "s8", True), f("wd", wq.shape)
    f("b", (O,), "f32", True), f("resq", resq.shape, "s8"), f("resd", resq.shape)
    for n in ("y", "r", "s"):
        f(n, (1, O, H, H))
    f("out", (1, O, H, H), "s8")
    g.add("quantize", ["x"], "xq", scale=sx)
    g.add("dequantize", ["xq"], "xd", scale=sx)
    g.add("dequantize", ["wq"], "wd", scales=sw, axis=0)
    g.add("dequantize", ["resq"], "resd", scale=sr)
    g.add("convolution", ["xd", "wd", "b"], "y", conv=conv_params((1, 1), (1, 1), (1, 1)))
    g.add("relu", ["y"], "r")
    g.add("add", ["r", "resd"], "s")
    g.add("quantize", ["s"], "out", scale=so)
    xq = q8(x, sx)
    xd = xq.astype(np.float32) * sx
    wd = wq.astype(np.float32) * sw[:, None, None, None]
    y = F.conv2d(torch.from_numpy(xd).double(), torch.from_numpy(wd).double(), torch.from_numpy(b).double(), padding=1)
    s = F.relu(y).numpy() + resq.astype(np.float64) * sr
    return g, {"x": x, "wq": wq, "b": b, "resq": resq}, q8(s, so)


def test_int8_qdq_block_fuses_to_one_int8_primitive(lib):
    g, feeds, ref = _int8_block(lib)
    parts = g.compile_all()
    assert [p.supported for p in parts] == [True, True], parts
    q, conv = parts
    assert q.ops == ["quantize0"] and q.outputs == ["xq"]
    assert set(conv.ops) == {"dequantize1", "dequantize2", "dequantize3", "convolution4", "relu5", "add6",
                             "quantize7"}, conv.ops
    assert "(int8)" in conv.info and not conv.info.endswith("ref:convolution"), conv.info
    assert set(conv.inputs) == {"xq", "wq", "b", "resq"} and conv.outputs == ["out"]
    env = g.run(feeds)
    impl = g.compiled[conv].impl
    assert not impl.startswith("ref:"), impl     # the AIE GEMM path (its x86 reference here)
    # PyTorch in float64: within one code (aiednn's epilogue is defined in f32)
    d = np.abs(env["out"].astype(int) - ref.astype(int))
    assert d.max() <= 1 and (d > 0).mean() < 0.05, (d.max(), (d > 0).mean())
    # the same fused problem on ref:: bit-exact, the library's own gate for every AIE implementation
    lib.set_filter("ref:")
    try:
        g2, _, _ = _int8_block(lib)
        g2.compile_all()
        assert g2.compiled == {} and all(p.supported for p in g2.parts)
        out_ref = g2.run(dict(feeds))["out"]
        assert g2.compiled[g2.parts[1]].impl.startswith("ref:")
    finally:
        lib.set_filter(None)
    np.testing.assert_array_equal(env["out"], out_ref)


def test_no_fuse_matches_fused(lib):
    g1, feeds, _ = _int8_block(lib, fuse=True)
    g2, _, _ = _int8_block(lib, fuse=False)
    a = run(g1, dict(feeds))["out"]
    parts = g2.compile_all()
    assert len(parts) == 8
    assert [p.ops for p in parts if not p.supported] == [["dequantize2"]]   # per-channel DQ alone
    sw = (np.random.default_rng(1).random(32, np.float32) * 0.02 + 0.005).astype(np.float32)  # as _int8_block

    def fallback(p, env):   # the framework runs what aiednn does not take
        env["wd"][...] = env["wq"].astype(np.float32) * sw[:, None, None, None]
    b = g2.run(dict(feeds), fallback)["out"]
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_unsupported_op_splits_and_falls_back(lib):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 64), np.float32)
    w = rng.standard_normal((32, 64), np.float32)
    g = Graph(lib)
    g.tensor("x", x.shape, "f32"), g.tensor("w", w.shape, "f32", True)
    for n in ("y", "u", "z"):
        g.tensor(n, (2, 32), "f32")
    g.add("matmul", ["x", "w"], "y", weights_transposed=1)
    g.add("wildcard", ["y"], "u")
    g.add("gelu", ["u"], "z")
    parts = g.compile_all()
    assert [p.supported for p in parts] == [True, False, True]
    assert parts[1].inputs == ["y"] and parts[1].outputs == ["u"]
    env = g.run({"x": x, "w": w}, lambda p, e: e.__setitem__("u", np.cos(e["y"]).astype(np.float32)))
    ref = F.gelu(torch.from_numpy(np.cos(x @ w.T))).numpy()
    np.testing.assert_allclose(env["z"], ref, rtol=1e-4, atol=1e-4)


def test_shared_output_is_not_fused(lib):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 16), np.float32)
    w = rng.standard_normal((8, 16), np.float32)
    g = Graph(lib)
    g.tensor("x", x.shape, "f32"), g.tensor("w", w.shape, "f32", True)
    for n in ("y", "a", "b", "s"):
        g.tensor(n, (4, 8), "f32")
    g.add("matmul", ["x", "w"], "y", weights_transposed=1)
    g.add("relu", ["y"], "a")
    g.add("sigmoid", ["y"], "b")
    g.add("add", ["a", "b"], "s")
    parts = g.compile_all()
    assert parts[0].ops == ["matmul0"], parts        # y has two consumers
    env = g.run({"x": x, "w": w})
    y = x @ w.T
    np.testing.assert_allclose(env["s"], np.maximum(y, 0) + 1 / (1 + np.exp(-y)), rtol=1e-4, atol=1e-5)


def test_marked_output_is_not_fused(lib):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 16), np.float32)
    w = rng.standard_normal((8, 16), np.float32)
    g = Graph(lib)
    g.tensor("x", x.shape, "f32"), g.tensor("w", w.shape, "f32", True)
    g.tensor("y", (4, 8), "f32"), g.tensor("z", (4, 8), "f32")
    g.add("matmul", ["x", "w"], "y", weights_transposed=1)
    g.add("relu", ["y"], "z")
    g.mark_output("y")
    parts = g.compile_all()
    assert [p.ops for p in parts] == [["matmul0"], ["relu1"]]


def test_misc_ops(lib):
    """Pool, softmax, layer norm + quantize, concat, reshape, transpose, sdpa: one partition each."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((1, 4, 8, 8), np.float32)
    g = Graph(lib)
    T = g.tensor
    T("x", x.shape, "f32"), T("p", (1, 4, 4, 4), "f32"), T("cat", (1, 8, 4, 4), "f32"), T("flat", (1, 8, 16), "f32")
    T("tr", (1, 16, 8), "f32"), T("sm", (1, 16, 8), "f32"), T("g", (8,), "f32", True), T("bt", (8,), "f32", True)
    T("ln", (1, 16, 8), "f32"), T("lnq", (1, 16, 8), "s8")
    T("q", (1, 2, 5, 4), "f32"), T("k", (1, 2, 6, 4), "f32"), T("v", (1, 2, 6, 4), "f32"), T("att", (1, 2, 5, 4), "f32")
    g.add("max_pool", ["x"], "p", pool=pool_params((2, 2), (2, 2)))
    g.add("concat", ["p", "p"], "cat", axis=1)
    g.add("reshape", ["cat"], "flat")
    g.add("transpose", ["flat"], "tr", perm=(0, 2, 1))
    g.add("softmax", ["tr"], "sm", axis=-1)
    g.add("layer_norm", ["sm", "g", "bt"], "ln", eps=1e-5)
    g.add("quantize", ["ln"], "lnq", scale=0.03)
    g.add("sdpa", ["q", "k", "v"], "att", sdpa_scale=0.5)
    parts = g.compile_all()
    assert all(p.supported for p in parts), parts
    assert ["layer_norm5", "quantize6"] in [p.ops for p in parts]
    gm, bt = rng.standard_normal(8).astype(np.float32), rng.standard_normal(8).astype(np.float32)
    q, k, v = (rng.standard_normal(s, np.float32) for s in ((1, 2, 5, 4), (1, 2, 6, 4), (1, 2, 6, 4)))
    env = g.run({"x": x, "g": gm, "bt": bt, "q": q, "k": k, "v": v})
    t = F.max_pool2d(torch.from_numpy(x), 2)
    t = torch.cat([t, t], 1).reshape(1, 8, 16).transpose(1, 2)
    t = torch.softmax(t, -1)
    ln = F.layer_norm(t, (8,), torch.from_numpy(gm), torch.from_numpy(bt), 1e-5).numpy()
    np.testing.assert_allclose(env["sm"], torch.softmax(torch.from_numpy(env["tr"]), -1).numpy(), rtol=1e-5)
    assert np.abs(env["lnq"].astype(int) - q8(ln, 0.03).astype(int)).max() <= 1
    att = F.scaled_dot_product_attention(*(torch.from_numpy(a) for a in (q, k, v)), scale=0.5).numpy()
    np.testing.assert_allclose(env["att"], att, rtol=1e-4, atol=1e-5)


def test_partition_order_is_executable(lib):
    """ops added out of order (consumer first) still give an executable partition order"""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((3, 5), np.float32)
    g = Graph(lib)
    for n in ("x", "a", "b"):
        g.tensor(n, (3, 5), "f32")
    g.add("sigmoid", ["a"], "b")
    g.add("relu", ["x"], "a")
    parts = g.compile_all()
    assert [p.ops for p in parts] == [["relu1"], ["sigmoid0"]]
    env = g.run({"x": x})
    np.testing.assert_allclose(env["b"], 1 / (1 + np.exp(-np.maximum(x, 0))), rtol=1e-5)


def test_find_set_impl_stream_workspace(lib):
    """find / set_impl and stream execute with a workspace through a compiled partition."""
    g, feeds, _ = _int8_block(lib)
    g.compile_all()
    out = g.run(dict(feeds))["out"].copy()
    conv = g.parts[1]
    cp = g.compiled[conv]
    heur = cp.find(FIND_HEURISTIC)
    names = [h[0] for h in heur]
    assert len(heur) >= 2 and heur[0][1] and names[-1].startswith("ref:"), heur   # AIE first, ref: last
    meas = cp.find(FIND_MEASURE)
    assert sorted(m[0] for m in meas) == sorted(names) and all(m[2] >= 0 for m in meas if m[4] == 0), meas
    env = {k: v for k, v in feeds.items()}
    env["xq"] = q8(feeds["x"], 0.05)
    ins = {k: env[k] for k in conv.inputs}
    ws = aligned_workspace(cp.workspace_size)
    s = Stream(lib)
    for impl in (names[0], names[-1]):
        cp.set_impl(impl)
        assert cp.impl.startswith(impl)   # impl_info adds the inner GEMM a composite dispatched to
        ws = aligned_workspace(cp.workspace_size)
        o = np.zeros_like(out)
        cp.execute(ins, {"out": o}, stream=s.h, workspace=ws)
        assert s.wait() == 0
        np.testing.assert_array_equal(o, out)



def _qdq_model():
    from pathlib import Path
    for root in (Path(__file__).resolve().parents[3], Path("/scratch/staff/bkirinci/aiehlc_aiesim")):
        p = root / "worklocal" / "tvmrelay_deploy" / "resnet18-v1-7.int8.qdq.onnx"
        if p.is_file():
            return p
    return None


def test_onnx_qdq_resnet18_matches_onnxruntime(lib):
    """onnxruntime-quantized ResNet-18 end to end: all partitions supported, close to the integer run."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    path = _qdq_model()
    if path is None:
        pytest.skip("resnet18-v1-7.int8.qdq.onnx not built (deploy_flow.py)")
    from aiegraph.dnn_onnx_graph import OnnxGraph
    m = OnnxGraph.load(lib, path)
    assert all(p.supported for p in m.parts), m.report()
    gemm8 = [p for p in m.parts if "(int8)" in p.info and ("conv" in p.info or "matmul" in p.info)]
    assert len(gemm8) == 21, m.report()
    # the 8 residual adds: dequantize(s) -> add [-> relu] -> quantize, one quantized binary each
    assert sum("add" in p.info and "(int8)" in p.info for p in m.parts) == 8, m.report()
    # placement: every int8 conv / the fc on the AIE (u8 src, zero point 113: the gemm+ composites'
    # exact zero-point correction); only the two pools are left on ref:
    gemms = gemm8
    assert all(not p.info.split("-> ")[-1].startswith("ref:") for p in gemms), m.report()
    on_ref = [p.info for p in m.parts if p.info.split("-> ")[-1].startswith("ref:")]
    assert all("pool" in i for i in on_ref) and len(on_ref) <= 2, on_ref
    x = np.random.default_rng(7).uniform(-2.1, 2.6, (1, 3, 224, 224)).astype(np.float32)

    def ort_run(level):
        so = ort.SessionOptions()
        so.graph_optimization_level = level
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"]).run(None, {"data": x})[0]
    yi = ort_run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    yf = ort_run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    y = m.run({"data": x})[m.outputs[0]]
    assert y.argmax() == yi.argmax()
    # no further from onnxruntime's integer run than its float run is
    assert np.abs(y - yi).max() <= np.abs(yi - yf).max() * 1.001
    assert (np.abs(y - yi) > 1e-6).mean() <= (np.abs(yi - yf) > 1e-6).mean()


def test_linear_on_tokens_and_deconv(lib):
    """nn.Linear on tokens + bias + gelu in one partition; a transposed conv + relu in one partition."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal((2, 7, 32), np.float32)
    w = rng.standard_normal((24, 32), np.float32)
    b = rng.standard_normal(24).astype(np.float32)
    xi = rng.standard_normal((1, 8, 6, 6), np.float32)
    wd = rng.standard_normal((8, 4, 3, 3), np.float32)        # ConvTranspose2d(8, 4): [Cin, Cout, kh, kw]
    g = Graph(lib)
    T = g.tensor
    T("x", x.shape, "f32"), T("w", w.shape, "f32", True), T("b", (24,), "f32", True)
    T("y", (2, 7, 24), "f32"), T("yb", (2, 7, 24), "f32"), T("z", (2, 7, 24), "f32")
    T("xi", xi.shape, "f32"), T("wd", (4, 8, 3, 3), "f32", True), T("d", (1, 4, 11, 11), "f32"), T("dr", (1, 4, 11, 11), "f32")
    g.add("matmul", ["x", "w"], "y", weights_transposed=1)
    g.add("bias_add", ["y", "b"], "yb")
    g.add("gelu", ["yb"], "z")
    g.add("deconvolution", ["xi", "wd"], "d", conv=conv_params((2, 2), (1, 1), (1, 1)))
    g.add("relu", ["d"], "dr")
    parts = g.compile_all()
    assert sorted(tuple(p.ops) for p in parts) == [("deconvolution3", "relu4"), ("matmul0", "bias_add1", "gelu2")], parts
    assert all(p.supported for p in parts), parts
    wdo = np.ascontiguousarray(wd.transpose(1, 0, 2, 3))      # aiednn deconv weights [O, C, kh, kw]
    env = g.run({"x": x, "w": w, "b": b, "xi": xi, "wd": wdo})
    np.testing.assert_allclose(env["z"], F.gelu(torch.from_numpy(x @ w.T + b)).numpy(), rtol=1e-4, atol=1e-4)
    ref = F.relu(F.conv_transpose2d(torch.from_numpy(xi), torch.from_numpy(wd), stride=2, padding=1)).numpy()
    np.testing.assert_allclose(env["dr"], ref, rtol=1e-4, atol=1e-4)


def test_resampling_reduction_group_norm_prelu(lib):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((1, 6, 5, 5), np.float32)
    t = rng.standard_normal((1, 9, 16), np.float32)
    gm, bt = rng.standard_normal(6).astype(np.float32), rng.standard_normal(6).astype(np.float32)
    sl = rng.random(6).astype(np.float32)
    g = Graph(lib)
    T = g.tensor
    T("x", x.shape, "f32"), T("up", (1, 6, 10, 10), "f32"), T("upl", (1, 6, 10, 10), "f32")
    T("t", t.shape, "f32"), T("tm", (1, 16), "f32")
    T("gm", (6,), "f32", True), T("bt", (6,), "f32", True), T("gn", x.shape, "f32")
    T("sl", (6, 1, 1), "f32", True), T("pr", x.shape, "f32")
    g.add("resampling", ["x"], "up")
    g.add("resampling", ["x"], "upl", alg="resampling_linear")
    g.add("reduction", ["t"], "tm", alg="reduction_mean", reduce_mask=0b010)   # token mean, keepdim off
    g.add("group_norm", ["x", "gm", "bt"], "gn", groups=3, eps=1e-5)
    g.add("prelu", ["x", "sl"], "pr")
    parts = g.compile_all()
    assert all(p.supported for p in parts) and len(parts) == 5, parts
    env = g.run({"x": x, "t": t, "gm": gm, "bt": bt, "sl": sl})
    X = torch.from_numpy(x)
    np.testing.assert_allclose(env["up"], F.interpolate(X, scale_factor=2, mode="nearest").numpy(), rtol=1e-6)
    np.testing.assert_allclose(env["upl"], F.interpolate(X, scale_factor=2, mode="bilinear", align_corners=False).numpy(),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(env["tm"], t.mean(1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(env["gn"], F.group_norm(X, 3, torch.from_numpy(gm), torch.from_numpy(bt), 1e-5).numpy(),
                               rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(env["pr"], F.prelu(X, torch.from_numpy(sl.reshape(6))).numpy(), rtol=1e-6)


def test_identity_reshape_inside_a_chain(lib):
    """conv -> identity reshape -> residual add: one partition."""
    rng = np.random.default_rng(10)
    x = rng.standard_normal((1, 8, 6, 6), np.float32)
    w = rng.standard_normal((8, 8, 1, 1), np.float32)
    g = Graph(lib)
    T = g.tensor
    T("x", x.shape, "f32"), T("w", w.shape, "f32", True)
    for n in ("y", "v", "s"):
        T(n, x.shape, "f32")
    g.add("convolution", ["x", "w"], "y", conv=conv_params())
    g.add("reshape", ["y"], "v")
    g.add("add", ["v", "x"], "s")
    parts = g.compile_all()
    assert [p.ops for p in parts] == [["convolution0", "reshape1", "add2"]], parts
    env = g.run({"x": x, "w": w})
    ref = (F.conv2d(torch.from_numpy(x), torch.from_numpy(w)) + torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(env["s"], ref, rtol=1e-4, atol=1e-4)


def _residual_graph(lib, fuse):
    """PT2E's residual: conv -> quantize -> dequantize, + dequantize(skip) -> relu -> quantize"""
    rng = np.random.default_rng(11)
    C, H = 16, 8
    sx, sy, sr, so = 0.05, 0.07, 0.04, 0.09
    x = rng.standard_normal((1, C, H, H), np.float32)
    wq = rng.integers(-127, 128, (C, C, 3, 3), dtype=np.int8)
    sw = (rng.random(C, np.float32) * 0.02 + 0.005).astype(np.float32)
    skq = rng.integers(-90, 90, (1, C, H, H), dtype=np.int8)
    g = Graph(lib, fuse=fuse)
    T = g.tensor
    T("x", x.shape, "f32"), T("xq", x.shape, "s8"), T("xd", x.shape, "f32")
    T("wq", wq.shape, "s8", True), T("wd", wq.shape, "f32"), T("skq", skq.shape, "s8"), T("skd", skq.shape, "f32")
    for n in ("y", "yd", "s", "r"):
        T(n, x.shape, "f32")
    T("yq", x.shape, "s8"), T("out", x.shape, "s8")
    g.add("quantize", ["x"], "xq", scale=sx)
    g.add("dequantize", ["xq"], "xd", scale=sx)
    g.add("dequantize", ["wq"], "wd", scales=sw, axis=0)
    g.add("convolution", ["xd", "wd"], "y", conv=conv_params((1, 1), (1, 1), (1, 1)))
    g.add("quantize", ["y"], "yq", scale=sy)
    g.add("dequantize", ["yq"], "yd", scale=sy)
    g.add("dequantize", ["skq"], "skd", scale=sr)
    g.add("add", ["yd", "skd"], "s")
    g.add("relu", ["s"], "r")
    g.add("quantize", ["r"], "out", scale=so)
    return g, {"x": x, "wq": wq, "skq": skq}


def test_quantized_residual_add_one_partition_bit_exact(lib):
    """PT2E's residual folds into the conv as a requant_binary post-op at level 1, byte-identical."""
    g1, feeds = _residual_graph(lib, 1)
    parts = g1.compile_all()
    conv = [p for p in parts if "convolution3" in p.ops][0]
    assert {"quantize4", "dequantize5", "dequantize6", "add7", "relu8", "quantize9"} <= set(conv.ops), parts
    assert set(conv.inputs) == {"xq", "wq", "skq"} and conv.outputs == ["out"], conv
    gs, _ = _residual_graph(lib, 1)
    gs.mark_output("yq")                     # the conv's int8 output is read: no fold
    ps = gs.compile_all()
    add = [p for p in ps if "add7" in p.ops][0]
    assert "convolution3" not in add.ops and set(add.inputs) == {"yq", "skq"}, ps
    np.testing.assert_array_equal(g1.run(dict(feeds))["out"], gs.run(dict(feeds))["out"])


def test_fusion_level2_residual_stays_exact(lib):
    """A residual add behind a requantized conv: level 2 = level 1, bit for bit."""
    g2, feeds = _residual_graph(lib, 2)
    parts = g2.compile_all()
    conv = [p for p in parts if "convolution3" in p.ops][0]
    assert {"add7", "relu8", "quantize9"} <= set(conv.ops), parts      # level 1's exact requant_binary
    g1, _ = _residual_graph(lib, 1)
    g1.compile_all()
    np.testing.assert_array_equal(g2.run(dict(feeds))["out"], g1.run(dict(feeds))["out"])


def test_relu6_tail_folds_into_conv_exactly(lib):
    """PT2E's ReLU6 tail folds into the conv at level 1 as a requant_eltwise post-op, byte-identical."""
    rng = np.random.default_rng(14)
    C, H = 8, 6

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("x", (1, C, H, H), "f32"), T("xq", (1, C, H, H), "s8"), T("xd", (1, C, H, H), "f32")
        T("wq", (C, C, 3, 3), "s8", True), T("wd", (C, C, 3, 3), "f32")
        for n in ("y", "yd", "r"):
            T(n, (1, C, H, H), "f32")
        T("yq", (1, C, H, H), "s8"), T("out", (1, C, H, H), "s8")
        g.add("quantize", ["x"], "xq", scale=0.05)
        g.add("dequantize", ["xq"], "xd", scale=0.05)
        g.add("dequantize", ["wq"], "wd", scale=0.01)
        g.add("convolution", ["xd", "wd"], "y", conv=conv_params((1, 1), (1, 1), (1, 1)))
        g.add("quantize", ["y"], "yq", scale=0.06)
        g.add("dequantize", ["yq"], "yd", scale=0.06)
        g.add("clamp", ["yd"], "r", alpha=0.0, beta=6.0)
        g.add("quantize", ["r"], "out", scale=6.0 / 127)
        return g
    g2, g1 = build(2), build(1)
    p2, p1 = g2.compile_all(), g1.compile_all()
    for parts in (p1, p2):   # level 1 already folds the tail exactly (requant_eltwise); level 2 adds nothing
        conv = [p for p in parts if "convolution3" in p.ops][0]
        assert {"quantize4", "dequantize5", "clamp6", "quantize7"} <= set(conv.ops) and conv.outputs == ["out"], parts
    feeds = {"x": rng.standard_normal((1, C, H, H), np.float32),
             "wq": rng.integers(-127, 128, (C, C, 3, 3), dtype=np.int8)}
    a, b = g2.run(dict(feeds))["out"], g1.run(dict(feeds))["out"]
    np.testing.assert_array_equal(a, b)
    # byte-identical to the conv with its int8 dst + the quantized eltwise (fold blocked by reading yq)
    gs = build(1)
    gs.mark_output("yq")
    ps = gs.compile_all()
    assert any("clamp6" in p.ops and "convolution3" not in p.ops for p in ps), ps
    np.testing.assert_array_equal(b, gs.run(dict(feeds))["out"])


def test_fusion_level2_shared_requant_stays_exported(lib):
    """Level 2: a requantized conv output with two consumers is exported, not swallowed."""
    rng = np.random.default_rng(12)
    C, H, s = 8, 6, 0.05
    g = Graph(lib, fuse=2)
    T = g.tensor
    T("x", (1, C, H, H), "f32"), T("xq", (1, C, H, H), "s8"), T("xd", (1, C, H, H), "f32")
    for w in ("w1", "w2"):
        T(w + "q", (C, C, 1, 1), "s8", True), T(w + "d", (C, C, 1, 1), "f32")
    for n in ("y1", "y1d", "y2", "y2d", "s"):
        T(n, (1, C, H, H), "f32")
    T("y1q", (1, C, H, H), "s8"), T("y2q", (1, C, H, H), "s8"), T("out", (1, C, H, H), "s8")
    g.add("quantize", ["x"], "xq", scale=s)
    g.add("dequantize", ["xq"], "xd", scale=s)
    g.add("dequantize", ["w1q"], "w1d", scale=0.02)
    g.add("dequantize", ["w2q"], "w2d", scale=0.02)
    g.add("convolution", ["xd", "w1d"], "y1", conv=conv_params())
    g.add("quantize", ["y1"], "y1q", scale=s)
    g.add("dequantize", ["y1q"], "y1d", scale=s)                         # two consumers
    g.add("convolution", ["y1d", "w2d"], "y2", conv=conv_params())
    g.add("quantize", ["y2"], "y2q", scale=s)
    g.add("dequantize", ["y2q"], "y2d", scale=s)
    g.add("add", ["y2d", "y1d"], "s")
    g.add("quantize", ["s"], "out", scale=s)
    parts = g.compile_all()
    produced = {"x", "w1q", "w2q"}
    for p in parts:
        assert p.supported and set(p.inputs) <= produced, (p, produced)
        produced |= set(p.outputs)
    c1 = [p for p in parts if "convolution4" in p.ops][0]
    assert c1.outputs == ["y1q"], c1                                     # exported as 8-bit, not swallowed
    c2 = [p for p in parts if "convolution7" in p.ops][0]
    assert {"add10", "quantize11"} <= set(c2.ops), c2   # the residual folds in as level 1's requant_binary
    feeds = {"x": rng.standard_normal((1, C, H, H), np.float32),
             "w1q": rng.integers(-127, 128, (C, C, 1, 1), dtype=np.int8),
             "w2q": rng.integers(-127, 128, (C, C, 1, 1), dtype=np.int8)}
    g1 = Graph(lib, fuse=1)
    g1.tensors, g1.tensor_id, g1.tensor_name, g1.pending = g.tensors, g.tensor_id, g.tensor_name, g.pending
    g1.compile_all()
    a, b = g.run(dict(feeds))["out"], g1.run(dict(feeds))["out"]
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_quantized_eltwise_tail_one_partition_bit_exact(lib):
    """dequantize -> eltwise chain -> quantize is one eltwise partition, bit-identical."""
    rng = np.random.default_rng(13)
    x = rng.integers(-128, 128, (1, 8, 6, 6), dtype=np.int8)

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("q", x.shape, "s8")
        for n in ("d", "c", "h"):
            T(n, x.shape, "f32")
        T("out", x.shape, "u8")
        g.add("dequantize", ["q"], "d", scale=0.047, zero_point=-3)
        g.add("clamp", ["d"], "c", alpha=0.0, beta=6.0)
        g.add("hardswish", ["c"], "h")
        g.add("quantize", ["h"], "out", scale=0.025, zero_point=7)
        return g
    g1 = build(1)
    parts = g1.compile_all()
    assert [sorted(p.ops) for p in parts] == [sorted(["dequantize0", "clamp1", "hardswish2", "quantize3"])], parts
    assert parts[0].inputs == ["q"] and parts[0].outputs == ["out"]
    g0 = build(0)
    assert len(g0.compile_all()) == 4
    np.testing.assert_array_equal(g1.run({"q": x})["out"], g0.run({"q": x})["out"])


def test_quantized_maxpool_on_codes_bit_exact(lib):
    """dequantize -> max_pool -> quantize with equal parameters is one 8-bit max_pool partition."""
    rng = np.random.default_rng(15)
    x = rng.integers(0, 256, (1, 8, 12, 12), dtype=np.uint8)

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("q", x.shape, "u8"), T("d", x.shape, "f32"), T("m", (1, 8, 6, 6), "f32"), T("out", (1, 8, 6, 6), "u8")
        g.add("dequantize", ["q"], "d", scale=0.031, zero_point=113)
        g.add("max_pool", ["d"], "m", pool=pool_params((3, 3), (2, 2), (1, 1), (0, 0)))
        g.add("quantize", ["m"], "out", scale=0.031, zero_point=113)
        return g
    g1, g0 = build(1), build(0)
    parts = g1.compile_all()
    assert [sorted(p.ops) for p in parts] == [["dequantize0", "max_pool1", "quantize2"]], parts
    assert parts[0].inputs == ["q"] and parts[0].outputs == ["out"]
    assert len(g0.compile_all()) == 3
    np.testing.assert_array_equal(g1.run({"q": x})["out"], g0.run({"q": x})["out"])


def test_quantized_view_on_codes(lib):
    """dequantize -> view -> quantize with equal parameters is a view on the codes, elided when possible."""
    rng = np.random.default_rng(16)
    x = rng.integers(-128, 128, (1, 4, 3, 5), dtype=np.int8)

    def build(fuse, keep_fq):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("q", x.shape, "s8"), T("d", x.shape, "f32"), T("f", (1, 60), "f32"), T("fq", (1, 60), "s8")
        T("d2", (1, 60), "f32"), T("t", (60, 1), "f32"), T("tq", (60, 1), "s8")
        g.add("dequantize", ["q"], "d", scale=0.02, zero_point=3)
        g.add("reshape", ["d"], "f")
        g.add("quantize", ["f"], "fq", scale=0.02, zero_point=3)
        g.add("dequantize", ["fq"], "d2", scale=0.02, zero_point=3)
        g.add("transpose", ["d2"], "t", perm=(1, 0))
        g.add("quantize", ["t"], "tq", scale=0.02, zero_point=3)
        if keep_fq:
            g.mark_output("fq")
        return g
    g1, g0 = build(1, True), build(0, True)
    parts = g1.compile_all()
    assert len(parts) == 2 and all(len(p.ops) == 3 for p in parts), parts
    assert parts[0].inputs == ["q"] and parts[1].inputs == ["fq"] and parts[1].outputs == ["tq"]
    g0.compile_all()
    a, b = g1.run({"q": x}), g0.run({"q": x})
    np.testing.assert_array_equal(a["tq"], b["tq"])
    np.testing.assert_array_equal(a["fq"].reshape(-1), x.reshape(-1))
    ge = build(1, False)                      # fq feeds only the transpose partition: the view is elided
    pe = ge.compile_all()
    assert len(pe) == 1 and pe[0].inputs == ["q"] and len(pe[0].ops) == 6, pe
    np.testing.assert_array_equal(ge.run({"q": x})["tq"], b["tq"])


def test_layout_propagation_nhwc_bit_identical(lib):
    """Two int8 convs: the edge between them is kept nhwc, bit-identical to propagation off."""
    rng = np.random.default_rng(17)
    C = C_ = 8
    H = 7

    def build(layouts):
        g = Graph(lib, layouts=layouts)
        T = g.tensor
        T("x", (1, C, H, H), "f32"), T("xq", (1, C, H, H), "s8"), T("xd", (1, C, H, H), "f32")
        for w in ("a", "b"):
            T(w + "q", (C, C, 3, 3), "s8", True), T(w + "d", (C, C, 3, 3), "f32")
        T("y", (1, C, H, H), "f32"), T("yq", (1, C, H, H), "s8"), T("yd", (1, C, H, H), "f32")
        T("z", (1, C, H, H), "f32"), T("r", (1, C, H, H), "f32"), T("out", (1, C, H, H), "s8")
        g.add("quantize", ["x"], "xq", scale=0.05)
        g.add("dequantize", ["xq"], "xd", scale=0.05)
        g.add("dequantize", ["aq"], "ad", scale=0.01)
        g.add("dequantize", ["bq"], "bd", scale=0.01)
        g.add("convolution", ["xd", "ad"], "y", conv=conv_params((1, 1), (1, 1), (1, 1)))
        g.add("relu", ["y"], "r")
        g.add("quantize", ["r"], "yq", scale=0.04)
        g.add("dequantize", ["yq"], "yd", scale=0.04)
        g.add("convolution", ["yd", "bd"], "z", conv=conv_params((1, 1), (1, 1), (1, 1)))
        g.add("quantize", ["z"], "out", scale=0.06)
        return g
    on, off = build(True), build(False)
    p_on, p_off = on.compile_all(), off.compile_all()
    assert [p.ops for p in p_on] == [p.ops for p in p_off]
    so = on.lib.so
    def fmt_of(g, part, name):
        n = so.aiednn_partition_get_inputs(part.h, None, 0)
        from aiegraph.dnn_graph import LT
        import ctypes as CT
        arr = (LT * n)()
        so.aiednn_partition_get_inputs(part.h, arr, n)
        return {g.tensor_name[arr[i].id]: arr[i].md.format for i in range(n)}.get(name)
    conv2 = [p for p in p_on if "convolution8" in p.ops][0]
    # nhwc exactly when the library's conv takes it natively (no "+fmtwrap"): the graph never buys a
    # format conversion inside a primitive
    import ctypes as CT
    from aiegraph.dnn_ctypes import Attr, md
    sm, wm, dm = md((1, C_, H, H), "s8"), md((C_, C_, 3, 3), "s8"), md((1, C_, H, H), "s8")
    so.aiednn_memory_desc_set_format(CT.byref(sm), 1), so.aiednn_memory_desc_set_format(CT.byref(dm), 1)
    at, sc = Attr(), (CT.c_float * 1)(0.01)
    at.n_scales, at.scales, at.dst_scale = 1, sc, 0.04
    pr = CT.c_void_p()
    assert so.aiednn_convolution_forward_create_attr(CT.byref(pr), CT.byref(sm), CT.byref(wm), None, CT.byref(dm),
                                                     CT.byref(conv_params((1, 1), (1, 1), (1, 1))), CT.byref(at)) == 0
    native = "fmtwrap" not in so.aiednn_primitive_impl_info(pr).decode()
    so.aiednn_primitive_destroy(pr)
    assert fmt_of(on, conv2, "yq") == (1 if native else 0)
    conv1 = [p for p in p_on if "convolution4" in p.ops][0]
    assert fmt_of(on, conv1, "xq") == 0                     # produced by the standalone quantize (a reorder)
    feeds = {"x": rng.standard_normal((1, C, H, H), np.float32),
             "aq": rng.integers(-127, 128, (C, C, 3, 3), dtype=np.int8),
             "bq": rng.integers(-127, 128, (C, C, 3, 3), dtype=np.int8)}
    np.testing.assert_array_equal(on.run(dict(feeds))["out"], off.run(dict(feeds))["out"])


@pytest.mark.parametrize("form", ["transpose_div", "wt_mul_mask"])
def test_decomposed_attention_is_one_sdpa(lib, form):
    """HF-style attention is one sdpa partition, matching torch."""
    rng = np.random.default_rng(18)
    B, H, Sq, Skv, D = 1, 2, 5, 7, 4
    q, k, v = (rng.standard_normal(s, np.float32) for s in ((B, H, Sq, D), (B, H, Skv, D), (B, H, Skv, D)))
    mask = rng.standard_normal((1, 1, Sq, Skv)).astype(np.float32)
    g = Graph(lib)
    T = g.tensor
    T("q", q.shape, "f32"), T("k", k.shape, "f32"), T("v", v.shape, "f32"), T("c", (1,), "f32", True)
    T("s", (B, H, Sq, Skv), "f32"), T("ss", (B, H, Sq, Skv), "f32"), T("p", (B, H, Sq, Skv), "f32")
    T("o", (B, H, Sq, D), "f32")
    if form == "transpose_div":
        T("kt", (B, H, D, Skv), "f32")
        g.add("transpose", ["k"], "kt", perm=(0, 1, 3, 2))
        g.add("matmul", ["q", "kt"], "s")
        g.add("binary", ["s", "c"], "ss", alg="binary_div")
        g.add("softmax", ["ss"], "p", axis=-1)
        c = np.array([np.sqrt(D)], np.float32)
        scale = 1 / np.sqrt(D)
    else:
        T("m", mask.shape, "f32"), T("sm", (B, H, Sq, Skv), "f32")
        g.add("matmul", ["q", "k"], "s", weights_transposed=1)
        g.add("multiply", ["s", "c"], "ss")
        g.add("add", ["ss", "m"], "sm")
        g.add("softmax", ["sm"], "p", axis=-1)
        c = np.array([0.37], np.float32)
        scale = 0.37
    g.add("matmul", ["p", "v"], "o")
    parts = g.compile_all()
    assert len(parts) == 1 and parts[0].supported, parts
    assert "sdpa" in parts[0].info or "sdpa" in g.compiled.get(parts[0], parts[0].compile()).impl, parts[0].info
    feeds = {"q": q, "k": k, "v": v, "c": c}
    if form != "transpose_div":
        feeds["m"] = mask
    out = g.run(feeds)["o"]
    tm = None if form == "transpose_div" else torch.from_numpy(mask)
    ref = F.scaled_dot_product_attention(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v),
                                         attn_mask=tm, scale=float(scale)).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_slice_windows(lib):
    """slice: split / chunk outputs and x[:, 0] as strided copies of the window"""
    rng = np.random.default_rng(19)
    x = rng.standard_normal((2, 5, 12), np.float32)
    g = Graph(lib)
    T = g.tensor
    T("x", x.shape, "f32"), T("a", (2, 5, 4), "f32"), T("b", (2, 5, 8), "f32"), T("c", (2, 1, 12), "f32")
    g.add("slice", ["x"], "a", starts=(0, 0, 0))
    g.add("slice", ["x"], "b", starts=(0, 0, 4))
    g.add("slice", ["x"], "c", starts=(0, 2, 0))
    parts = g.compile_all()
    assert all(p.supported for p in parts) and len(parts) == 3, parts
    env = g.run({"x": x})
    np.testing.assert_array_equal(env["a"], x[:, :, :4])
    np.testing.assert_array_equal(env["b"], x[:, :, 4:])
    np.testing.assert_array_equal(env["c"], x[:, 2:3, :])


def test_nongemm_postop_chains_and_levels(lib):
    """Post-ops on non-GEMM ops fuse at level 1 (exact) or level 2 (gelu), bit-identical when exact."""
    rng = np.random.default_rng(20)
    s = rng.standard_normal((2, 3, 8), np.float32)
    x = rng.standard_normal((2, 6, 8), np.float32)
    res = rng.standard_normal((2, 6, 8), np.float32)
    gm, bt = rng.standard_normal(8).astype(np.float32), rng.standard_normal(8).astype(np.float32)
    y = rng.standard_normal((1, 4, 5, 5), np.float32)

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("s", s.shape, "f32"), T("p", s.shape, "f32"), T("pq", s.shape, "u8")
        T("x", x.shape, "f32"), T("res", res.shape, "f32"), T("g", (8,), "f32", True), T("b", (8,), "f32", True)
        T("ln", x.shape, "f32"), T("lr", x.shape, "f32"), T("lo", x.shape, "f32")
        T("y", y.shape, "f32"), T("yn", y.shape, "f32"), T("yg", y.shape, "f32")
        g.add("softmax", ["s"], "p", axis=-1)
        g.add("quantize", ["p"], "pq", scale=1 / 255, zero_point=0)
        g.add("layer_norm", ["x", "g", "b"], "ln", eps=1e-5)
        g.add("add", ["ln", "res"], "lr")
        g.add("relu", ["lr"], "lo")
        g.add("group_norm", ["y"], "yn", groups=2, eps=1e-5)
        g.add("gelu", ["yn"], "yg")
        return g
    feeds = {"s": s, "x": x, "res": res, "g": gm, "b": bt, "y": y}
    g0, g1, g2 = build(0), build(1), build(2)
    p0, p1, p2 = g0.compile_all(), g1.compile_all(), g2.compile_all()
    ops1 = [sorted(p.ops) for p in p1]
    assert ["quantize1", "softmax0"] in ops1, p1                       # exact: a final quantize
    assert ["add3", "layer_norm2", "relu4"] in ops1, p1              # exact: one add, then relu
    assert ["gelu6"] in ops1, p1                                       # gelu is not exact: never at level 1
    e0, e1 = g0.run(dict(feeds)), g1.run(dict(feeds))
    for t in ("pq", "lo", "yg"):
        np.testing.assert_array_equal(e1[t], e0[t])


def test_sdpa_head_split_merge_folded(lib):
    """q / k / v as [B, S, H, D] through transposes around sdpa: one partition, matching torch."""
    rng = np.random.default_rng(21)
    B, S, H, D = 1, 6, 2, 4
    qs, ks, vs = (rng.standard_normal((B, S, H, D), np.float32) for _ in range(3))
    g = Graph(lib)
    T = g.tensor
    for n in ("q", "k", "v"):
        T(n + "s", (B, S, H, D), "f32"), T(n, (B, H, S, D), "f32")
        g.add("transpose", [n + "s"], n, perm=(0, 2, 1, 3))
    T("o", (B, H, S, D), "f32"), T("os", (B, S, H, D), "f32")
    g.add("sdpa", ["q", "k", "v"], "o", sdpa_scale=0.5)
    g.add("transpose", ["o"], "os", perm=(0, 2, 1, 3))
    parts = g.compile_all()
    assert len(parts) == 1, parts
    assert set(parts[0].inputs) == {"qs", "ks", "vs"} and parts[0].outputs == ["os"], parts[0]
    out = g.run({"qs": qs, "ks": ks, "vs": vs})["os"]
    t = lambda a: torch.from_numpy(a).transpose(1, 2)
    ref = F.scaled_dot_product_attention(t(qs), t(ks), t(vs), scale=0.5).transpose(1, 2).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_rope_one_partition_bit_exact(lib):
    """HF rotate_half RoPE is one aiednn_rope_create, bit-identical to the unfused ops."""
    rng = np.random.default_rng(22)
    B, H, S, D = 1, 2, 5, 8
    x = rng.standard_normal((B, H, S, D), np.float32)
    cos, sin = (rng.standard_normal((1, 1, S, D)).astype(np.float32) for _ in range(2))

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("x", x.shape, "f32"), T("cos", cos.shape, "f32"), T("sin", sin.shape, "f32")
        T("x1", (B, H, S, D // 2), "f32"), T("x2", (B, H, S, D // 2), "f32"), T("nx2", (B, H, S, D // 2), "f32")
        for n in ("r", "xc", "rs", "o"):
            T(n, x.shape, "f32")
        g.add("slice", ["x"], "x1", starts=(0, 0, 0, 0))
        g.add("slice", ["x"], "x2", starts=(0, 0, 0, D // 2))
        g.add("eltwise", ["x2"], "nx2", alg="eltwise_neg")
        g.add("concat", ["nx2", "x1"], "r", axis=-1)
        g.add("multiply", ["x", "cos"], "xc")
        g.add("multiply", ["r", "sin"], "rs")
        g.add("add", ["xc", "rs"], "o")
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    assert len(p1) == 1 and "rope" in p1[0].info.lower() + g1.compiled.setdefault(p1[0], p1[0].compile()).impl, p1
    assert len(g0.compile_all()) == 7
    a, b = g1.run({"x": x, "cos": cos, "sin": sin})["o"], g0.run({"x": x, "cos": cos, "sin": sin})["o"]
    np.testing.assert_array_equal(a, b)


def test_int8_concat_one_partition_bit_exact(lib):
    """dequantize each input -> concat -> quantize is one aiednn_concat_create_q, bit-identical to the ops"""
    rng = np.random.default_rng(23)
    a = rng.integers(-128, 128, (1, 3, 4, 4), dtype=np.int8)
    b = rng.integers(0, 256, (1, 5, 4, 4), dtype=np.uint8)

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("a", a.shape, "s8"), T("b", b.shape, "u8"), T("ad", a.shape, "f32"), T("bd", b.shape, "f32")
        T("c", (1, 8, 4, 4), "f32"), T("cq", (1, 8, 4, 4), "u8")
        g.add("dequantize", ["a"], "ad", scale=0.03, zero_point=2)
        g.add("dequantize", ["b"], "bd", scale=0.02, zero_point=128)
        g.add("concat", ["ad", "bd"], "c", axis=1)
        g.add("quantize", ["c"], "cq", scale=0.025, zero_point=120)
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    assert len(p1) == 1 and set(p1[0].inputs) == {"a", "b"}, p1
    g0.compile_all()
    np.testing.assert_array_equal(g1.run({"a": a, "b": b})["cq"], g0.run({"a": a, "b": b})["cq"])


def test_gated_mlp_level2(lib):
    """SwiGLU is one aiednn_gated_mlp_create at level 2, within 1e-4 of torch; level 1 leaves it."""
    rng = np.random.default_rng(24)
    S, Dm, Ff = 5, 16, 24
    x = rng.standard_normal((1, S, Dm), np.float32)
    wg, wu = (rng.standard_normal((Ff, Dm), np.float32) * 0.2 for _ in range(2))
    wd = rng.standard_normal((Dm, Ff), np.float32) * 0.2

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("x", x.shape, "f32"), T("wg", wg.shape, "f32", True), T("wu", wu.shape, "f32", True)
        T("wd", wd.shape, "f32", True)
        for n in ("g", "a", "u", "m"):
            T(n, (1, S, Ff), "f32")
        T("o", (1, S, Dm), "f32")
        g.add("matmul", ["x", "wg"], "g", weights_transposed=1)
        g.add("eltwise", ["g"], "a", alg="eltwise_swish", alpha=1.0)
        g.add("matmul", ["x", "wu"], "u", weights_transposed=1)
        g.add("multiply", ["a", "u"], "m")
        g.add("matmul", ["m", "wd"], "o", weights_transposed=1)
        return g
    g2, g1 = build(2), build(1)
    p2, p1 = g2.compile_all(), g1.compile_all()
    assert len(p2) == 1, p2
    assert len(p1) > 1, p1
    X = torch.from_numpy(x)
    ref = (F.silu(X @ torch.from_numpy(wg).T) * (X @ torch.from_numpy(wu).T)) @ torch.from_numpy(wd).T
    out = g2.run({"x": x, "wg": wg, "wu": wu, "wd": wd})["o"]
    np.testing.assert_allclose(out, ref.numpy(), rtol=1e-4, atol=1e-4)


def test_reorder_with_postops_bit_exact(lib):
    """transpose -> add(residual) -> relu: one reorder partition with post-ops at level 1, bit-identical"""
    rng = np.random.default_rng(25)
    x = rng.standard_normal((2, 3, 4), np.float32)
    r = rng.standard_normal((2, 4, 3), np.float32)

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("x", x.shape, "f32"), T("r", r.shape, "f32")
        for n in ("t", "s", "o"):
            T(n, (2, 4, 3), "f32")
        g.add("transpose", ["x"], "t", perm=(0, 2, 1))
        g.add("add", ["t", "r"], "s")
        g.add("relu", ["s"], "o")
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    assert [sorted(p.ops) for p in p1] == [["add1", "relu2", "transpose0"]], p1
    g0.compile_all()
    np.testing.assert_array_equal(g1.run({"x": x, "r": r})["o"], g0.run({"x": x, "r": r})["o"])


def test_dequantize_read_through_layer_norm_and_sdpa(lib):
    """An 8-bit dequantize folds into layer norm and into sdpa's Q / K / V."""
    rng = np.random.default_rng(26)
    B, S, H, D = 1, 6, 2, 4
    xq = rng.integers(-128, 128, (B, S, H * D), dtype=np.int8)
    gm, bt = rng.standard_normal(H * D).astype(np.float32), rng.standard_normal(H * D).astype(np.float32)
    qkv = [rng.integers(-128, 128, (B, S, H * D), dtype=np.int8) for _ in range(3)]

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("xq", xq.shape, "s8"), T("xd", xq.shape, "f32"), T("g", (H * D,), "f32", True), T("b", (H * D,), "f32", True)
        T("ln", xq.shape, "f32"), T("lq", xq.shape, "s8")
        g.add("dequantize", ["xq"], "xd", scale=0.05, zero_point=-3)
        g.add("layer_norm", ["xd", "g", "b"], "ln", eps=1e-5)
        g.add("quantize", ["ln"], "lq", scale=0.03)
        for n, sc in zip("qkv", (0.02, 0.03, 0.04)):
            T(n + "8", (B, S, H * D), "s8"), T(n + "f", (B, S, H * D), "f32"), T(n + "v", (B, S, H, D), "f32")
            T(n, (B, H, S, D), "f32")
            g.add("dequantize", [n + "8"], n + "f", scale=sc, zero_point=1)
            g.add("reshape", [n + "f"], n + "v")
            g.add("transpose", [n + "v"], n, perm=(0, 2, 1, 3))
        T("o", (B, H, S, D), "f32")
        g.add("sdpa", ["q", "k", "v"], "o", sdpa_scale=0.5)
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    ln = [p for p in p1 if "layer_norm" in " ".join(p.ops)][0]
    assert ln.inputs[0] == "xq" and ln.outputs == ["lq"], ln
    at = [p for p in p1 if any(o.startswith("sdpa") for o in p.ops)][0]
    assert {"q8", "k8", "v8"} <= set(at.inputs), at
    g0.compile_all()
    feeds = {"xq": xq, "g": gm, "b": bt, "q8": qkv[0], "k8": qkv[1], "v8": qkv[2]}
    e1, e0 = g1.run(dict(feeds)), g0.run(dict(feeds))
    np.testing.assert_array_equal(e1["lq"], e0["lq"])
    np.testing.assert_allclose(e1["o"], e0["o"], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("style,int8", [("gpt2", False), ("gpt2", True), ("vit", False)])
def test_packed_qkv_views_fold_into_sdpa(lib, style, int8):
    """Attention from one packed qkv projection through view ops folds into one sdpa."""
    rng = np.random.default_rng(27)
    S, H, D = 7, 2, 4
    E = H * D

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        names = []
        if style == "gpt2":
            base = "qkv"
            if int8:
                T("qkv8", (1, S, 3 * E), "s8"), T("qkv", (1, S, 3 * E), "f32")
                g.add("dequantize", ["qkv8"], "qkv", scale=0.03, zero_point=2)
            else:
                T("qkv", (1, S, 3 * E), "f32")
            for i, n in enumerate("qkv"):
                T(n + "s", (1, S, E), "f32"), T(n + "v", (1, S, H, D), "f32"), T(n + "t", (1, H, S, D), "f32")
                g.add("slice", [base], n + "s", starts=(0, 0, i * E))
                g.add("reshape", [n + "s"], n + "v")
                g.add("transpose", [n + "v"], n + "t", perm=(0, 2, 1, 3))
                names.append(n + "t")
        else:
            T("p", (S, 1, 3 * E), "f32"), T("u", (S, 1, 3, E), "f32"), T("u1", (1, S, 1, 3, E), "f32")
            T("tp", (3, S, 1, 1, E), "f32"), T("sq", (3, S, 1, E), "f32")
            g.add("reshape", ["p"], "u")
            g.add("reshape", ["u"], "u1")
            g.add("transpose", ["u1"], "tp", perm=(3, 1, 2, 0, 4))
            g.add("reshape", ["tp"], "sq")
            for i, n in enumerate("qkv"):
                T(n + "1", (1, S, 1, E), "f32"), T(n + "s", (S, 1, E), "f32"), T(n + "w", (S, H, D), "f32")
                T(n + "x", (H, S, D), "f32"), T(n + "t", (1, H, S, D), "f32")
                g.add("slice", ["sq"], n + "1", starts=(i, 0, 0, 0))
                g.add("reshape", [n + "1"], n + "s")
                g.add("reshape", [n + "s"], n + "w")
                g.add("transpose", [n + "w"], n + "x", perm=(1, 0, 2))
                g.add("reshape", [n + "x"], n + "t")
                names.append(n + "t")
        T("o", (1, H, S, D), "f32")
        g.add("sdpa", names, "o", sdpa_scale=0.5)
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    at = [p for p in p1 if any(o.startswith("sdpa") for o in p.ops)]
    assert len(p1) == 1 and len(at) == 1, p1
    g0.compile_all()
    if style == "gpt2":
        feeds = {"qkv8": rng.integers(-128, 128, (1, S, 3 * E), dtype=np.int8)} if int8 else \
                {"qkv": rng.standard_normal((1, S, 3 * E)).astype(np.float32)}
    else:
        feeds = {"p": rng.standard_normal((S, 1, 3 * E)).astype(np.float32)}
    np.testing.assert_allclose(g1.run(dict(feeds))["o"], g0.run(dict(feeds))["o"], rtol=1e-4, atol=1e-5)


def test_seq_first_batch1_permutes_fold(lib):
    """Seq-first attention at batch 1: the size-1 transpose and head-merge permute fold."""
    rng = np.random.default_rng(31)
    S, H, D = 7, 2, 4
    E = H * D

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("x", (1, S, E), "f32"), T("g", (E,), "f32", True), T("bt", (E,), "f32", True)
        T("ln", (1, S, E), "f32"), T("lt", (S, 1, E), "f32"), T("lq", (S, 1, E), "s8")
        g.add("layer_norm", ["x", "g", "bt"], "ln", eps=1e-5)
        g.add("transpose", ["ln"], "lt", perm=(1, 0, 2))
        g.add("quantize", ["lt"], "lq", scale=0.03)
        T("q", (1, H, S, D), "f32"), T("k", (1, H, S, D), "f32"), T("v", (1, H, S, D), "f32")
        T("o", (1, H, S, D), "f32"), T("om", (S, 1, H, D), "f32")
        g.add("sdpa", ["q", "k", "v"], "o", sdpa_scale=0.5)
        g.add("transpose", ["o"], "om", perm=(2, 0, 1, 3))
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    assert len(p1) == 2 and all(p.supported for p in p1), p1
    assert any("om" in p.outputs and any(o.startswith("sdpa") for o in p.ops) for p in p1), p1
    assert any("lq" in p.outputs and any(o.startswith("layer_norm") for o in p.ops) for p in p1), p1
    g0.compile_all()
    feeds = {"x": rng.standard_normal((1, S, E)).astype(np.float32),
             "g": rng.standard_normal(E).astype(np.float32), "bt": rng.standard_normal(E).astype(np.float32)}
    feeds.update({n: rng.standard_normal((1, H, S, D)).astype(np.float32) for n in "qkv"})
    e1, e0 = g1.run(dict(feeds)), g0.run(dict(feeds))
    np.testing.assert_allclose(e1["om"], e0["om"], rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(e1["lq"], e0["lq"])


@pytest.mark.parametrize("scale2", [0.05, 0.07])
def test_same_param_requant_tail_folds(lib, scale2):
    """quantize -> dequantize -> view -> quantize with equal parameters folds; a real requant stays."""
    rng = np.random.default_rng(37)
    S, H, D = 7, 2, 4
    E = H * D

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("x8", (S, E), "s8"), T("w8", (E, E), "s8", True), T("x", (S, E), "f32"), T("w", (E, E), "f32")
        T("y", (S, E), "f32"), T("yq", (S, E), "s8")
        T("yd", (S, E), "f32"), T("yv", (S, 1, E), "f32"), T("yq2", (S, 1, E), "s8")
        g.add("dequantize", ["x8"], "x", scale=0.04, zero_point=0)
        g.add("dequantize", ["w8"], "w", scale=0.01, zero_point=0)
        g.add("matmul", ["x", "w"], "y", weights_transposed=1)
        g.add("quantize", ["y"], "yq", scale=0.05, zero_point=1)
        g.add("dequantize", ["yq"], "yd", scale=0.05, zero_point=1)
        g.add("reshape", ["yd"], "yv")
        g.add("quantize", ["yv"], "yq2", scale=scale2, zero_point=1)
        T("q", (1, H, S, D), "f32"), T("k", (1, H, S, D), "f32"), T("v", (1, H, S, D), "f32")
        T("o", (1, H, S, D), "f32"), T("om", (1, S, H, D), "f32"), T("ov", (S, E), "f32"), T("oq", (S, E), "s8")
        g.add("sdpa", ["q", "k", "v"], "o", sdpa_scale=0.5)
        g.add("transpose", ["o"], "om", perm=(0, 2, 1, 3))
        g.add("reshape", ["om"], "ov")
        g.add("quantize", ["ov"], "oq", scale=0.02, zero_point=-3)
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    assert all(p.supported for p in p1), p1
    gemm = next(p for p in p1 if "matmul2" in p.ops)
    assert gemm.outputs == (["yq2"] if scale2 == 0.05 else ["yq"]), p1
    assert any(p.outputs == ["oq"] and any(o.startswith("sdpa") for o in p.ops) for p in p1), p1
    assert len(p1) == (2 if scale2 == 0.05 else 4), p1   # a real requant: its dequantize + quantize
    g0.compile_all()
    feeds = {"x8": rng.integers(-128, 128, (S, E), dtype=np.int8), "w8": rng.integers(-127, 128, (E, E), dtype=np.int8)}
    feeds.update({n: rng.standard_normal((1, H, S, D)).astype(np.float32) for n in "qkv"})
    e1, e0 = g1.run(dict(feeds)), g0.run(dict(feeds))
    np.testing.assert_array_equal(e1["yq2"], e0["yq2"])
    assert np.abs(e1["oq"].astype(int) - e0["oq"].astype(int)).max() <= 1


def test_dequantize_slice_quantize_is_a_slice_of_the_codes(lib):
    """dequantize -> slice -> quantize with equal parameters is a slice of the int8 codes."""
    q = np.random.default_rng(41).integers(-128, 128, (1, 8, 5, 5), dtype=np.int8)

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        g.tensor("q", q.shape, "s8"), g.tensor("f", q.shape, "f32"), g.tensor("h", (1, 4, 5, 5), "f32")
        g.tensor("hq", (1, 4, 5, 5), "s8")
        g.add("dequantize", ["q"], "f", scale=0.1, zero_point=3)
        g.add("slice", ["f"], "h", starts=(0, 4, 0, 0))
        g.add("quantize", ["h"], "hq", scale=0.1, zero_point=3)
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    assert len(p1) == 1 and p1[0].supported and "slice" in p1[0].info, p1
    g0.compile_all()
    np.testing.assert_array_equal(g1.run({"q": q})["hq"], g0.run({"q": q})["hq"])
    np.testing.assert_array_equal(g1.run({"q": q})["hq"], q[:, 4:])


@pytest.mark.parametrize("dt", ["s8", "u8"])
def test_f32_slice_absorbs_its_quantize(lib, dt):
    """slice -> quantize of an f32 tensor is one aiednn_slice_create_attr partition, bit-identical."""
    s, zp = 0.1, 3 if dt == "s8" else 131
    x = ((np.random.default_rng(43).integers(-60, 60, (1, 8, 5, 5)) + 0.5) * s).astype(np.float32)

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        g.tensor("x", x.shape, "f32"), g.tensor("h", (1, 4, 5, 5), "f32"), g.tensor("hq", (1, 4, 5, 5), dt)
        g.add("slice", ["x"], "h", starts=(0, 4, 0, 0))
        g.add("quantize", ["h"], "hq", scale=s, zero_point=zp)
        return g
    g1, g0 = build(1), build(0)
    p1 = g1.compile_all()
    assert len(p1) == 1 and p1[0].supported and p1[0].ops == ["slice0", "quantize1"], p1
    g0.compile_all()
    np.testing.assert_array_equal(g1.run({"x": x})["hq"], g0.run({"x": x})["hq"])


def test_rmsnorm_pattern_folds_at_level_2(lib):
    """PT2E's hand-written RMSNorm is one rms_norm partition at level 2, within two codes."""
    rng = np.random.default_rng(47)
    S, D = 7, 32
    x = rng.standard_normal((1, S, D)).astype(np.float32)
    w8 = rng.integers(60, 127, D, dtype=np.int8)
    sw, so = 0.008, 0.03

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        T("x", x.shape, "f32"), T("xq", x.shape, "s8"), T("xd", x.shape, "f32"), T("sq", x.shape, "f32")
        T("m", (1, S, 1), "f32"), T("mq", (1, S, 1), "s8"), T("md", (1, S, 1), "f32")
        T("e8", (1,), "s8", True), T("e", (1,), "f32"), T("a", (1, S, 1), "f32"), T("aq", (1, S, 1), "s8")
        T("ad", (1, S, 1), "f32"), T("r", (1, S, 1), "f32"), T("rq", (1, S, 1), "s8"), T("rd", (1, S, 1), "f32")
        T("y1", x.shape, "f32"), T("y1q", x.shape, "s8"), T("y1d", x.shape, "f32"), T("w8", (D,), "s8", True)
        T("w", (D,), "f32"), T("y", x.shape, "f32"), T("yq", x.shape, "s8")
        g.add("quantize", ["x"], "xq", scale=0.03, zero_point=0), g.add("dequantize", ["xq"], "xd", scale=0.03, zero_point=0)
        g.add("eltwise", ["x"], "sq", alg="eltwise_pow", alpha=1.0, beta=2.0)
        g.add("reduction", ["sq"], "m", alg="reduction_mean", reduce_mask=4)
        g.add("quantize", ["m"], "mq", scale=0.01, zero_point=-128), g.add("dequantize", ["mq"], "md", scale=0.01, zero_point=-128)
        g.add("dequantize", ["e8"], "e", scale=1e-4, zero_point=0)
        g.add("add", ["md", "e"], "a")
        g.add("quantize", ["a"], "aq", scale=0.01, zero_point=-128), g.add("dequantize", ["aq"], "ad", scale=0.01, zero_point=-128)
        g.add("eltwise", ["ad"], "r", alg="eltwise_rsqrt")
        g.add("quantize", ["r"], "rq", scale=0.02, zero_point=-128), g.add("dequantize", ["rq"], "rd", scale=0.02, zero_point=-128)
        g.add("multiply", ["xd", "rd"], "y1")
        g.add("quantize", ["y1"], "y1q", scale=so, zero_point=0), g.add("dequantize", ["y1q"], "y1d", scale=so, zero_point=0)
        g.add("dequantize", ["w8"], "w", scale=sw, zero_point=0)
        g.add("multiply", ["y1d", "w"], "y")
        g.add("quantize", ["y"], "yq", scale=so, zero_point=0)
        return g
    g2, g1 = build(2), build(1)
    p2 = g2.compile_all()
    norm = [p for p in p2 if "reduction" in " ".join(p.ops) or any(o.startswith("reduction") for o in p.ops)]
    assert len(norm) == 1 and norm[0].supported and norm[0].outputs == ["yq"], p2
    assert len(g1.compile_all()) > 6
    for eps8 in (10, 100):                                        # eps 1e-3, then 1e-2: read at execute
        e = eps8 * 1e-4
        y = g2.run({"x": x, "w8": w8, "e8": np.array([eps8], np.int8)})["yq"].astype(int)
        ref = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + e) * (w8 * sw)
        assert np.abs(y - q8(ref, so).astype(int)).max() <= 2, eps8


def test_quantized_rope_folds_at_level_2(lib):
    """PT2E's RoPE is one aiednn_rope_create_attr partition at level 2, within two codes."""
    rng = np.random.default_rng(53)
    S, D = 6, 16
    h = D // 2
    x = rng.standard_normal((1, 2, S, D)).astype(np.float32)
    ang = np.arange(S)[:, None] * (1.0 / 100 ** (np.arange(h) / h))[None, :]
    cosf, sinf = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([np.sin(ang)] * 2, -1)
    c8, s8 = np.round(cosf / (1 / 127)).astype(np.int8), np.round(sinf / (1 / 127)).astype(np.int8)
    sx, so = 0.02, 0.03

    def build(fuse):
        g = Graph(lib, fuse=fuse)
        T = g.tensor
        sh, hh = (1, 2, S, D), (1, 2, S, h)
        for n, s_, dt in (("x", sh, "f32"), ("xq", sh, "s8"), ("xd", sh, "f32"), ("c8", (S, D), "s8"), ("c", (S, D), "f32"),
                          ("s8_", (S, D), "s8"), ("s", (S, D), "f32"), ("m1", sh, "f32"), ("m1q", sh, "s8"), ("m1d", sh, "f32"),
                          ("hi", hh, "f32"), ("lo", hh, "f32"), ("ng", hh, "f32"), ("ngq", hh, "s8"), ("ngd", hh, "f32"),
                          ("loq", hh, "s8"), ("lod", hh, "f32"), ("cat", sh, "f32"), ("catq", sh, "s8"), ("catd", sh, "f32"),
                          ("m2", sh, "f32"), ("m2q", sh, "s8"), ("m2d", sh, "f32"), ("y", sh, "f32"), ("yq", sh, "s8")):
            T(n, s_, dt, n in ("c8", "s8_"))
        Q = lambda a, b, s_=sx: (g.add("quantize", [a], b + "q", scale=s_, zero_point=0),
                                 g.add("dequantize", [b + "q"], b + "d", scale=s_, zero_point=0))
        g.add("quantize", ["x"], "xq", scale=sx, zero_point=0), g.add("dequantize", ["xq"], "xd", scale=sx, zero_point=0)
        g.add("dequantize", ["c8"], "c", scale=1 / 127, zero_point=0), g.add("dequantize", ["s8_"], "s", scale=1 / 127, zero_point=0)
        g.add("multiply", ["xd", "c"], "m1"), Q("m1", "m1")
        g.add("slice", ["x"], "hi", starts=(0, 0, 0, h)), g.add("slice", ["x"], "lo", starts=(0, 0, 0, 0))
        g.add("eltwise", ["hi"], "ng", alg="eltwise_neg"), Q("ng", "ng"), Q("lo", "lo")
        g.add("concat", ["ngd", "lod"], "cat", axis=-1), Q("cat", "cat")
        g.add("multiply", ["catd", "s"], "m2"), Q("m2", "m2")
        g.add("add", ["m1d", "m2d"], "y"), g.add("quantize", ["y"], "yq", scale=so, zero_point=0)
        return g
    g2, g1 = build(2), build(1)
    p2 = g2.compile_all()
    rope = [p for p in p2 if "rope" in p.info]
    assert len(rope) == 1 and rope[0].outputs == ["yq"] and len(p2) == 1, p2
    assert not any("rope" in p.info for p in g1.compile_all())
    y = g2.run({"x": x, "c8": c8, "s8_": s8})["yq"].astype(int)
    cq, sq = c8 / 127.0, s8 / 127.0
    ref = x * cq + np.concatenate([-x[..., h:], x[..., :h]], -1) * sq
    assert np.abs(y - q8(ref, so).astype(int)).max() <= 2
