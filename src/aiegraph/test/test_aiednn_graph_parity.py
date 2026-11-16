"""Every graph op kind of include/aiednn_graph.h against PyTorch (forward values, autograd gradients)."""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from aiegraph.dnn_ctypes import conv_params, pool_params
from aiegraph.dnn_graph import GOP, Graph, build_graph_lib

# oneDNN graph op (dnnl_graph_types.h) -> the aiednn graph kind that serves it. Ops served by a generic kind
# carry the attribute that selects them. PENDING: no library primitive yet.
ONEDNN = {
    "Abs": "eltwise", "AbsBackward": "eltwise_backward", "Add": "add", "AvgPool": "avg_pool",
    "AvgPoolBackward": "pool_backward", "BatchNormForwardTraining": "batch_norm_training",
    "BatchNormInference": "batch_norm_inference", "BatchNormTrainingBackward": "batch_norm_backward",
    "BiasAdd": "bias_add", "BiasAddBackward": "bias_add_backward", "Clamp": "clamp",
    "ClampBackward": "eltwise_backward", "Concat": "concat", "Convolution": "convolution",
    "ConvolutionBackwardData": "convolution_backward_data",
    "ConvolutionBackwardWeights": "convolution_backward_weights", "ConvTranspose": "deconvolution",
    "ConvTransposeBackwardData": "deconvolution_backward_data",
    "ConvTransposeBackwardWeights": "deconvolution_backward_weights", "Dequantize": "dequantize",
    "Divide": "binary", "DynamicDequantize": "dynamic_dequantize", "DynamicQuantize": "dynamic_quantize", "Elu": "eltwise",
    "EluBackward": "eltwise_backward", "End": "wildcard", "Exp": "eltwise", "GELU": "gelu",
    "GELUBackward": "eltwise_backward", "GenIndex": "gen_index", "GreaterEqual": "binary",
    "GroupNorm": "group_norm", "HardSigmoid": "eltwise", "HardSigmoidBackward": "eltwise_backward",
    "HardSwish": "hardswish", "HardSwishBackward": "eltwise_backward", "Interpolate": "resampling",
    "InterpolateBackward": "resampling_backward", "LayerNorm": "layer_norm",
    "LayerNormBackward": "layer_norm_backward", "LeakyReLU": "eltwise", "Log": "eltwise",
    "LogSoftmax": "softmax", "LogSoftmaxBackward": "softmax_backward", "MatMul": "matmul",
    "Maximum": "binary", "MaxPool": "max_pool", "MaxPoolBackward": "pool_backward", "Minimum": "binary",
    "Mish": "eltwise", "MishBackward": "eltwise_backward", "Multiply": "multiply", "Pow": "eltwise",
    "PReLU": "prelu", "PReLUBackward": "prelu_backward", "Quantize": "quantize", "Reciprocal": "eltwise",
    "ReduceL1": "reduction", "ReduceL2": "reduction", "ReduceMax": "reduction", "ReduceMean": "reduction",
    "ReduceMin": "reduction", "ReduceProd": "reduction", "ReduceSum": "reduction", "ReLU": "relu",
    "ReLUBackward": "eltwise_backward", "Reorder": "reorder", "Round": "eltwise", "Sigmoid": "sigmoid",
    "SigmoidBackward": "eltwise_backward", "SoftMax": "softmax", "SoftMaxBackward": "softmax_backward",
    "SoftPlus": "eltwise", "SoftPlusBackward": "eltwise_backward", "Sqrt": "eltwise",
    "SqrtBackward": "eltwise_backward", "Square": "eltwise", "SquaredDifference": "squared_difference",
    "StaticReshape": "reshape", "StaticTranspose": "transpose", "Subtract": "binary", "Tanh": "tanh",
    "TanhBackward": "eltwise_backward", "TypeCast": "type_cast", "Wildcard": "wildcard", "Select": "select",
    "RMSNorm": "rms_norm", "Dropout": "dropout",
    # beyond oneDNN's set: linear attention (Gated DeltaNet / KDA / GLA)
    "(aiednn) LinearAttention": "linear_attention",
}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_graph_lib(tmp_path_factory.mktemp("graphlib_parity"))


def one(lib, kind, ins: dict, outs: dict, consts=(), **kw):
    """a one-op graph: ins / outs name -> (shape, dtype); returns (partition, run(feeds) -> env)"""
    g = Graph(lib)
    for n, (s, dt) in {**ins, **outs}.items():
        g.tensor(n, s, dt, n in consts)
    g.add(kind, list(ins), list(outs) if len(outs) > 1 else next(iter(outs)), **kw)
    parts = g.compile_all()
    assert len(parts) == 1 and parts[0].supported, parts
    read = [] if kind == "gen_index" else sorted(ins)      # gen_index takes only its input's shape
    assert parts[0].outputs == list(outs) and sorted(parts[0].inputs) == read, parts[0]
    if os.environ.get("AIEDNN_PARITY_IMPL"):     # the placement census (doc/design/aiednn_framework_api.md §7)
        print(f"\nIMPL {kind} {kw.get('alg', '')} -> {parts[0].info.split('-> ')[-1]}")
    return parts[0], lambda feeds: g.run(dict(feeds))


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def test_every_onednn_graph_op_has_a_kind():
    missing = [op for op, k in ONEDNN.items() if k != "PENDING" and k not in GOP]
    assert not missing, missing
    assert len(ONEDNN) >= 88


# ── forward kinds ─────────────────────────────────────────────────────────

def test_gen_index(lib):
    x = np.zeros((2, 3, 4), np.float32)
    _, run = one(lib, "gen_index", {"x": (x.shape, "f32")}, {"y": (x.shape, "s32")}, axis=1)
    np.testing.assert_array_equal(run({"x": x})["y"], np.broadcast_to(np.arange(3)[None, :, None], x.shape))


@pytest.mark.parametrize("per_channel", [False, True])
def test_dynamic_quantize_dequantize(lib, per_channel):
    rng = np.random.default_rng(14)
    x = rng.standard_normal((2, 4, 5)).astype(np.float32)
    n = 4 if per_channel else 1
    s = (rng.random(n) * 0.05 + 0.01).astype(np.float32)
    z = rng.integers(-5, 6, n).astype(np.int32)
    ax = 1 if per_channel else -1
    sb = s.reshape((1, n, 1)) if per_channel else s[0]
    zb = z.reshape((1, n, 1)) if per_channel else z[0]
    _, run = one(lib, "dynamic_quantize", {"x": (x.shape, "f32"), "s": ((n,), "f32"), "z": ((n,), "s32")},
                 {"q": (x.shape, "s8")}, axis=ax)
    q = run({"x": x, "s": s, "z": z})["q"]
    np.testing.assert_array_equal(q, np.clip(np.round(x / sb) + zb, -128, 127).astype(np.int8))
    _, run = one(lib, "dynamic_dequantize", {"q": (x.shape, "s8"), "s": ((n,), "f32"), "z": ((n,), "s32")},
                 {"y": (x.shape, "f32")}, axis=ax)
    np.testing.assert_allclose(run({"q": q, "s": s, "z": z})["y"], (q.astype(np.float32) - zb) * sb, rtol=1e-6)


def test_type_cast_and_reorder(lib):
    x = (np.random.default_rng(0).standard_normal((2, 3, 5)) * 90).astype(np.float32)
    _, run = one(lib, "type_cast", {"x": (x.shape, "f32")}, {"y": (x.shape, "s8")})
    np.testing.assert_array_equal(run({"x": x})["y"], np.clip(np.round(x), -128, 127).astype(np.int8))
    _, run = one(lib, "reorder", {"x": (x.shape, "f32")}, {"y": (x.shape, "f32")})
    np.testing.assert_array_equal(run({"x": x})["y"], x)


def test_select(lib):
    rng = np.random.default_rng(1)
    c = (rng.random((2, 1, 6)) > 0.5).astype(np.uint8)
    a, b = rng.standard_normal((2, 4, 6)).astype(np.float32), rng.standard_normal((1, 4, 6)).astype(np.float32)
    _, run = one(lib, "select", {"c": (c.shape, "u8"), "a": (a.shape, "f32"), "b": (b.shape, "f32")},
                 {"y": (a.shape, "f32")})
    np.testing.assert_array_equal(run({"c": c, "a": a, "b": b})["y"], np.where(c != 0, a, b))


def test_squared_difference(lib):
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((1, 6)).astype(np.float32), rng.standard_normal((3, 6)).astype(np.float32)
    _, run = one(lib, "squared_difference", {"a": (a.shape, "f32"), "b": (b.shape, "f32")}, {"y": (b.shape, "f32")})
    np.testing.assert_allclose(run({"a": a, "b": b})["y"], (a - b) ** 2, rtol=1e-6)


def test_dropout(lib):
    x = np.random.default_rng(3).standard_normal((4, 256)).astype(np.float32)
    _, run = one(lib, "dropout", {"x": (x.shape, "f32")}, {"y": (x.shape, "f32")}, alpha=0.0)
    np.testing.assert_array_equal(run({"x": x})["y"], x)
    _, run = one(lib, "dropout", {"x": (x.shape, "f32")}, {"y": (x.shape, "f32"), "m": (x.shape, "u8")},
                 alpha=0.5, seed=7)
    env = run({"x": x})
    m = env["m"].astype(bool)
    assert 0.4 < m.mean() < 0.6
    np.testing.assert_allclose(env["y"], np.where(m, x / 0.5, 0), rtol=1e-6)


def test_rms_norm_and_log_softmax(lib):
    rng = np.random.default_rng(4)
    x, gm = rng.standard_normal((3, 64)).astype(np.float32), rng.standard_normal(64).astype(np.float32)
    _, run = one(lib, "rms_norm", {"x": (x.shape, "f32"), "g": ((64,), "f32")}, {"y": (x.shape, "f32")},
                 consts=("g",), eps=1e-6)
    ref = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * gm
    np.testing.assert_allclose(run({"x": x, "g": gm})["y"], ref, rtol=1e-4, atol=1e-5)
    _, run = one(lib, "softmax", {"x": (x.shape, "f32")}, {"y": (x.shape, "f32")}, axis=-1, alg="softmax_log")
    np.testing.assert_allclose(run({"x": x})["y"], F.log_softmax(t(x), -1).numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["layer_norm", "group_norm", "batch_norm_training"])
def test_norm_training_forward(lib, kind):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 8, 4, 4)).astype(np.float32)
    C = 8 if kind != "layer_norm" else 4
    gm, bt = rng.standard_normal(C).astype(np.float32), rng.standard_normal(C).astype(np.float32)
    st = {"layer_norm": (2, 8, 4), "group_norm": (2, 2), "batch_norm_training": (8,)}[kind]
    kw = {"groups": 2} if kind == "group_norm" else {}
    _, run = one(lib, kind, {"x": (x.shape, "f32"), "g": ((C,), "f32"), "b": ((C,), "f32")},
                 {"y": (x.shape, "f32"), "mu": (st, "f32"), "var": (st, "f32")}, eps=1e-5, **kw)
    env = run({"x": x, "g": gm, "b": bt})
    X = t(x).double()
    if kind == "layer_norm":
        ref, dims = F.layer_norm(X, (4,), t(gm).double(), t(bt).double(), 1e-5), (-1,)
        mu, var = X.mean(-1), X.var(-1, unbiased=False)
    elif kind == "group_norm":
        ref = F.group_norm(X, 2, t(gm).double(), t(bt).double(), 1e-5)
        Xg = X.reshape(2, 2, -1)
        mu, var = Xg.mean(-1), Xg.var(-1, unbiased=False)
    else:
        ref = F.batch_norm(X, None, None, t(gm).double(), t(bt).double(), True, 0.0, 1e-5)
        mu, var = X.mean((0, 2, 3)), X.var((0, 2, 3), unbiased=False)
    np.testing.assert_allclose(env["y"], ref.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(env["mu"], mu.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(env["var"], var.numpy(), rtol=1e-5, atol=1e-6)


# ── training (backward) kinds vs autograd ───────────────────────────────────

ELT = [("eltwise_relu", 0.0, 0.0, F.relu), ("eltwise_relu", 0.1, 0.0, lambda v: F.leaky_relu(v, 0.1)),
       ("eltwise_tanh", 0, 0, torch.tanh), ("eltwise_logistic", 0, 0, torch.sigmoid),
       ("eltwise_gelu_tanh", 0, 0, lambda v: F.gelu(v, approximate="tanh")), ("eltwise_gelu_erf", 0, 0, F.gelu),
       ("eltwise_elu", 1.0, 0, F.elu), ("eltwise_mish", 0, 0, F.mish), ("eltwise_soft_relu", 1.0, 0, F.softplus),
       ("eltwise_hardswish", 1 / 6, 0.5, F.hardswish), ("eltwise_hardsigmoid", 1 / 6, 0.5, F.hardsigmoid),
       ("eltwise_clip", -0.5, 0.7, lambda v: v.clamp(-0.5, 0.7)), ("eltwise_abs", 0, 0, torch.abs),
       ("eltwise_sqrt", 0, 0, torch.sqrt)]


@pytest.mark.parametrize("alg,a,b,fn", ELT, ids=[e[0] + str(e[1]) for e in ELT])
def test_eltwise_backward(lib, alg, a, b, fn):
    rng = np.random.default_rng(6)
    x = rng.standard_normal((3, 40)).astype(np.float32)
    if alg == "eltwise_sqrt":
        x = np.abs(x) + 0.1
    dy = rng.standard_normal(x.shape).astype(np.float32)
    _, run = one(lib, "eltwise_backward", {"x": (x.shape, "f32"), "dy": (x.shape, "f32")}, {"dx": (x.shape, "f32")},
                 alg=alg, alpha=a, beta=b)
    X = t(x).double().requires_grad_()
    fn(X).backward(t(dy).double())
    np.testing.assert_allclose(run({"x": x, "dy": dy})["dx"], X.grad.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("log", [False, True])
def test_softmax_backward(lib, log):
    rng = np.random.default_rng(7)
    x, dy = rng.standard_normal((4, 10)).astype(np.float32), rng.standard_normal((4, 10)).astype(np.float32)
    X = t(x).double().requires_grad_()
    y = F.log_softmax(X, -1) if log else F.softmax(X, -1)
    y.backward(t(dy).double())
    _, run = one(lib, "softmax_backward", {"y": (x.shape, "f32"), "dy": (x.shape, "f32")}, {"dx": (x.shape, "f32")},
                 axis=-1, alg="softmax_log" if log else "softmax_accurate")
    np.testing.assert_allclose(run({"y": y.detach().float().numpy(), "dy": dy})["dx"], X.grad.numpy(),
                               rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("alg", ["pooling_max", "pooling_avg_include_padding"])
def test_pool_backward(lib, alg):
    rng = np.random.default_rng(8)
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    dy = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)
    X = t(x).double().requires_grad_()
    y = F.max_pool2d(X, 3, 2, 1) if alg == "pooling_max" else F.avg_pool2d(X, 3, 2, 1, count_include_pad=True)
    y.backward(t(dy).double())
    ins = {"dy": (dy.shape, "f32"), "x": (x.shape, "f32")} if alg == "pooling_max" else {"dy": (dy.shape, "f32")}
    _, run = one(lib, "pool_backward", ins, {"dx": (x.shape, "f32")}, alg=alg,
                 pool=pool_params((3, 3), (2, 2), (1, 1), (1, 1)))
    np.testing.assert_allclose(run({"dy": dy, "x": x})["dx"], X.grad.numpy(), rtol=1e-4, atol=1e-5)


def test_resampling_and_bias_add_backward(lib):
    rng = np.random.default_rng(9)
    dy = rng.standard_normal((1, 2, 8, 8)).astype(np.float32)
    X = torch.zeros((1, 2, 4, 4), dtype=torch.float64, requires_grad=True)
    F.interpolate(X, scale_factor=2, mode="nearest").backward(t(dy).double())
    _, run = one(lib, "resampling_backward", {"dy": (dy.shape, "f32")}, {"dx": ((1, 2, 4, 4), "f32")},
                 alg="resampling_nearest")
    np.testing.assert_allclose(run({"dy": dy})["dx"], X.grad.numpy(), rtol=1e-5, atol=1e-5)
    _, run = one(lib, "bias_add_backward", {"dy": (dy.shape, "f32")}, {"db": ((2,), "f32")})
    np.testing.assert_allclose(run({"dy": dy})["db"], dy.sum((0, 2, 3)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("deconv", [False, True])
def test_convolution_backward(lib, deconv):
    rng = np.random.default_rng(10)
    N, C, O, H = 1, 4, 6, 7
    x = rng.standard_normal((N, C, H, H)).astype(np.float32)
    wt = rng.standard_normal((C, O, 3, 3) if deconv else (O, C, 3, 3)).astype(np.float32)   # torch layout
    X, W = t(x).double().requires_grad_(), t(wt).double().requires_grad_()
    B = torch.zeros(O, dtype=torch.float64, requires_grad=True)
    y = F.conv_transpose2d(X, W, B, 2, 1) if deconv else F.conv2d(X, W, B, 2, 1)
    dy = rng.standard_normal(y.shape).astype(np.float32)
    y.backward(t(dy).double())
    w = wt.transpose(1, 0, 2, 3).copy() if deconv else wt       # aiednn deconvolution weights [O, C, KH, KW]
    cp = conv_params((2, 2), (1, 1), (1, 1))
    pre = "deconvolution" if deconv else "convolution"
    _, run = one(lib, pre + "_backward_data", {"dy": (dy.shape, "f32"), "w": (w.shape, "f32")},
                 {"dx": (x.shape, "f32")}, conv=cp)
    np.testing.assert_allclose(run({"dy": dy, "w": w})["dx"], X.grad.numpy(), rtol=1e-4, atol=1e-4)
    _, run = one(lib, pre + "_backward_weights", {"x": (x.shape, "f32"), "dy": (dy.shape, "f32")},
                 {"dw": (w.shape, "f32"), "db": ((O,), "f32")}, conv=cp)
    env = run({"x": x, "dy": dy})
    gw = W.grad.numpy().transpose(1, 0, 2, 3) if deconv else W.grad.numpy()
    np.testing.assert_allclose(env["dw"], gw, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(env["db"], B.grad.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["batch_norm_backward", "layer_norm_backward", "group_norm_backward"])
def test_norm_backward(lib, kind):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 8, 4, 4)).astype(np.float32)
    dy = rng.standard_normal(x.shape).astype(np.float32)
    C = 4 if kind == "layer_norm_backward" else 8
    gm, bt = rng.standard_normal(C).astype(np.float32), rng.standard_normal(C).astype(np.float32)
    X, G, Bt = t(x).double().requires_grad_(), t(gm).double().requires_grad_(), t(bt).double().requires_grad_()
    if kind == "batch_norm_backward":
        y, mu, var = F.batch_norm(X, None, None, G, Bt, True, 0.0, 1e-5), X.mean((0, 2, 3)), X.var((0, 2, 3), unbiased=False)
        kw = {}
    elif kind == "layer_norm_backward":
        y, mu, var = F.layer_norm(X, (4,), G, Bt, 1e-5), X.mean(-1), X.var(-1, unbiased=False)
        kw = {}
    else:
        Xg = X.reshape(2, 2, -1)
        y, mu, var = F.group_norm(X, 2, G, Bt, 1e-5), Xg.mean(-1), Xg.var(-1, unbiased=False)
        kw = {"groups": 2}
    y.backward(t(dy).double())
    mu, var = mu.detach().float().numpy(), var.detach().float().numpy()
    _, run = one(lib, kind, {"x": (x.shape, "f32"), "dy": (x.shape, "f32"), "mu": (mu.shape, "f32"),
                             "var": (var.shape, "f32"), "g": ((C,), "f32")},
                 {"dx": (x.shape, "f32"), "dg": ((C,), "f32"), "db": ((C,), "f32")}, eps=1e-5, **kw)
    env = run({"x": x, "dy": dy, "mu": mu, "var": var, "g": gm})
    np.testing.assert_allclose(env["dx"], X.grad.numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(env["dg"], G.grad.numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(env["db"], Bt.grad.numpy(), rtol=1e-3, atol=1e-4)


def test_prelu_backward(lib):
    rng = np.random.default_rng(12)
    x, dy = rng.standard_normal((2, 3, 5)).astype(np.float32), rng.standard_normal((2, 3, 5)).astype(np.float32)
    s = rng.random((3, 1)).astype(np.float32)
    X, S = t(x).double().requires_grad_(), t(s).double().requires_grad_()
    torch.where(X > 0, X, X * S).backward(t(dy).double())
    _, run = one(lib, "prelu_backward", {"x": (x.shape, "f32"), "s": (s.shape, "f32"), "dy": (x.shape, "f32")},
                 {"dx": (x.shape, "f32"), "ds": (s.shape, "f32")})
    env = run({"x": x, "s": s, "dy": dy})
    np.testing.assert_allclose(env["dx"], X.grad.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(env["ds"], S.grad.numpy(), rtol=1e-4, atol=1e-5)


# ── ONNX import: the standard operator set onto the graph kinds (dnn_onnx_graph) ──

def _onnx_case(op, shapes, attrs=None, consts=None, out_shape=None, out_dt=1):
    from onnx import TensorProto, helper, numpy_helper
    consts = consts or {}
    names = [f"i{k}" for k in range(len(shapes))]
    inits = [numpy_helper.from_array(consts[n], n) for n in names if n in consts]
    ins = [helper.make_tensor_value_info(n, TensorProto.BOOL if s[1] == "b" else TensorProto.FLOAT, s[0])
           for n, s in zip(names, shapes) if n not in consts]
    out = helper.make_tensor_value_info("y", out_dt, out_shape)
    node = helper.make_node(op, names, ["y"], **(attrs or {}))
    return helper.make_model(helper.make_graph([node], "t", ins, [out], inits),
                             opset_imports=[helper.make_opsetid("", 18)])


_R = np.random.default_rng(13)
ONNX_CASES = [
    ("Abs", [((2, 8), "f")], {}, {}), ("Exp", [((2, 8), "f")], {}, {}), ("Neg", [((2, 8), "f")], {}, {}),
    ("Erf", [((2, 8), "f")], {}, {}), ("Floor", [((2, 8), "f")], {}, {}), ("Softplus", [((2, 8), "f")], {}, {}),
    ("Mish", [((2, 8), "f")], {}, {}), ("Elu", [((2, 8), "f")], {"alpha": 0.7}, {}),
    ("LeakyRelu", [((2, 8), "f")], {"alpha": 0.2}, {}), ("HardSigmoid", [((2, 8), "f")], {}, {}),
    ("Sub", [((2, 8), "f"), ((1, 8), "f")], {}, {}), ("Div", [((2, 8), "f"), ((2, 8), "f")], {}, {}),
    ("Max", [((2, 8), "f"), ((2, 8), "f")], {}, {}), ("Min", [((2, 8), "f"), ((1, 8), "f")], {}, {}),
    ("Where", [((2, 8), "b"), ((2, 8), "f"), ((2, 8), "f")], {}, {}),
    ("LogSoftmax", [((2, 8), "f")], {"axis": -1}, {}), ("Identity", [((2, 8), "f")], {}, {}),
    ("Dropout", [((2, 8), "f")], {}, {}),
    ("ReduceMean", [((2, 4, 6), "f"), ((1,), "i")], {"keepdims": 0}, {"i1": np.array([2], np.int64)}),
    ("ReduceSum", [((2, 4, 6), "f"), ((2,), "i")], {}, {"i1": np.array([0, 2], np.int64)}),
    ("ReduceL2", [((2, 4, 6), "f"), ((1,), "i")], {}, {"i1": np.array([-1], np.int64)}),
    ("LayerNormalization", [((2, 3, 16), "f"), ((16,), "f"), ((16,), "f")], {},
     {"i1": _R.standard_normal(16).astype(np.float32), "i2": _R.standard_normal(16).astype(np.float32)}),
    ("BatchNormalization", [((1, 4, 3, 3), "f")] + [((4,), "f")] * 4, {},
     {"i1": _R.random(4).astype(np.float32) + 0.5, "i2": _R.standard_normal(4).astype(np.float32),
      "i3": _R.standard_normal(4).astype(np.float32), "i4": _R.random(4).astype(np.float32) + 0.5}),
    ("InstanceNormalization", [((1, 4, 5, 5), "f"), ((4,), "f"), ((4,), "f")], {},
     {"i1": _R.standard_normal(4).astype(np.float32), "i2": _R.standard_normal(4).astype(np.float32)}),
    ("PRelu", [((1, 4, 3, 3), "f"), ((4, 1, 1), "f")], {}, {"i1": _R.random((4, 1, 1)).astype(np.float32)}),
    ("Slice", [((4, 10), "f"), ((1,), "i"), ((1,), "i"), ((1,), "i")], {},
     {"i1": np.array([2], np.int64), "i2": np.array([7], np.int64), "i3": np.array([1], np.int64)}),
    ("ConvTranspose", [((1, 3, 5, 5), "f"), ((3, 4, 3, 3), "f")], {"strides": [2, 2], "pads": [1, 1, 1, 1]},
     {"i1": _R.standard_normal((3, 4, 3, 3)).astype(np.float32)}),
]


@pytest.mark.parametrize("op,shapes,attrs,consts", ONNX_CASES, ids=[c[0] for c in ONNX_CASES])
def test_onnx_ops_map_and_match(lib, op, shapes, attrs, consts):
    from onnx import shape_inference
    from onnx.reference import ReferenceEvaluator
    from aiegraph.dnn_onnx_graph import OnnxGraph
    m = shape_inference.infer_shapes(_onnx_case(op, shapes, attrs, consts))
    feeds = {}
    for k, (s, dt) in enumerate(shapes):
        n = f"i{k}"
        if n in consts:
            continue
        feeds[n] = (_R.random(s) > 0.5) if dt == "b" else (np.abs(_R.standard_normal(s)) + 0.1 if op == "Div" and k
                                                          else _R.standard_normal(s)).astype(np.float32)
    ref = ReferenceEvaluator(m).run(None, feeds)[0]
    og = OnnxGraph(lib, m)
    assert not og.wildcards, og.wildcards
    assert all(p.supported for p in og.parts), og.report()
    y = og.run({k: (v.astype(np.uint8) if v.dtype == bool else v) for k, v in feeds.items()})["y"]
    np.testing.assert_allclose(y.reshape(ref.shape), ref, rtol=1e-4, atol=1e-5)


def test_onnx_dequantize_with_runtime_scale(lib):
    """DequantizeLinear whose scale is a graph input, not an initializer -> dynamic_dequantize"""
    from onnx import TensorProto, helper
    from aiegraph.dnn_onnx_graph import OnnxGraph
    m = helper.make_model(helper.make_graph(
        [helper.make_node("DequantizeLinear", ["q", "s"], ["y"])], "t",
        [helper.make_tensor_value_info("q", TensorProto.INT8, (2, 6)), helper.make_tensor_value_info("s", TensorProto.FLOAT, (1,))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, (2, 6))]), opset_imports=[helper.make_opsetid("", 18)])
    og = OnnxGraph(lib, m)
    assert not og.wildcards and all(p.supported for p in og.parts), og.report()
    q = np.arange(-6, 6, dtype=np.int8).reshape(2, 6)
    np.testing.assert_allclose(og.run({"q": q, "s": np.array([0.25], np.float32)})["y"], q * 0.25)


@pytest.mark.parametrize("fuse", [1, 2])
def test_quantized_mul_tolerance_only_at_level_2(lib, fuse):
    """Quantized mul: level 1 keeps the exact table, level 2 the AIE mul within one code."""
    rng = np.random.default_rng(15)
    a, b = rng.integers(-128, 128, (4, 64), dtype=np.int8), rng.integers(-128, 128, (4, 64), dtype=np.int8)

    def build(f):
        g = Graph(lib, fuse=f)
        for n, dt in (("a", "s8"), ("b", "s8"), ("af", "f32"), ("bf", "f32"), ("m", "f32"), ("q", "s8")):
            g.tensor(n, a.shape, dt)
        g.add("dequantize", ["a"], "af", scale=0.05, zero_point=3)
        g.add("dequantize", ["b"], "bf", scale=0.04, zero_point=-2)
        g.add("multiply", ["af", "bf"], "m")
        g.add("quantize", ["m"], "q", scale=0.03, zero_point=1)
        return g
    g, g1 = build(fuse), build(1)
    parts = g.compile_all()
    assert len(parts) == 1 and parts[0].supported, parts
    assert ("fvec:aie" in parts[0].info) == (fuse == 2), parts[0].info
    g1.compile_all()
    d = np.abs(g.run({"a": a, "b": b})["q"].astype(int) - g1.run({"a": a, "b": b})["q"].astype(int))
    assert d.max() <= (1 if fuse == 2 else 0)


def _la_ref(q, k, v, gate, beta, state, scale, per_channel, delta):
    """the gated delta rule per (b, h): h <- diag(e^g) h; u = beta (v - h^T k) (delta) or v; h += k u^T; o = s h^T q"""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    o = np.zeros((B, H, T, Dv))
    hs = np.zeros((B, H, Dk, Dv))
    for b in range(B):
        for hh in range(H):
            h = state[b, hh].astype(np.float64).copy() if state is not None else np.zeros((Dk, Dv))
            for t in range(T):
                gg = np.exp(gate[b, hh, t]) if per_channel else np.exp(gate[b, hh, t]) * np.ones(Dk)
                h = gg[:, None] * h
                u = beta[b, hh, t] * (v[b, hh, t] - h.T @ k[b, hh, t]) if delta else v[b, hh, t]
                h = h + np.outer(k[b, hh, t], u)
                o[b, hh, t] = scale * h.T @ q[b, hh, t]
            hs[b, hh] = h
    return o, hs


@pytest.mark.parametrize("per_channel,delta,with_state", [(False, True, False), (True, True, True), (False, False, True)],
                         ids=["gated_deltanet", "kda_state", "gla_state"])
def test_linear_attention(lib, per_channel, delta, with_state):
    rng = np.random.default_rng(16)
    B, H, T, Dk, Dv = 1, 2, 9, 8, 6
    q, k = (rng.standard_normal((B, H, T, Dk)).astype(np.float32) * 0.5 for _ in range(2))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((B, H, T, Dv)).astype(np.float32)
    gate = -rng.random((B, H, T, Dk) if per_channel else (B, H, T)).astype(np.float32) * 0.3
    beta = rng.random((B, H, T)).astype(np.float32)
    st = rng.standard_normal((B, H, Dk, Dv)).astype(np.float32) * 0.1
    ins = {"q": (q.shape, "f32"), "k": (k.shape, "f32"), "v": (v.shape, "f32"), "g": (gate.shape, "f32")}
    if delta:
        ins["b"] = (beta.shape, "f32")
    if with_state:
        ins["s"] = (st.shape, "f32")
    outs = {"o": (v.shape, "f32")} | ({"so": (st.shape, "f32")} if with_state else {})
    flags = (1 if per_channel else 0) | (0 if delta else 2)
    _, run = one(lib, "linear_attention", ins, outs, flags=flags, sdpa_scale=0.0)
    env = run({"q": q, "k": k, "v": v, "g": gate, "b": beta, "s": st})
    o, hs = _la_ref(q, k, v, gate, beta, st if with_state else None, 1 / np.sqrt(Dk), per_channel, delta)
    np.testing.assert_allclose(env["o"], o, rtol=1e-4, atol=1e-4)
    if with_state:
        np.testing.assert_allclose(env["so"], hs, rtol=1e-4, atol=1e-4)
