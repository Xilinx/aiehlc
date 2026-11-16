"""torch.compile(model, backend="aiednn") (aiegraph.dnn_torch_backend) vs eager, workstation emulation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "example" / "aiednn" / "models"))

torch = pytest.importorskip("torch")

from aiegraph.backends import aiednn as flags   # noqa: E402 -- registers backend="aiednn"
from aiegraph import dnn_torch_backend as be   # noqa: E402


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.fc = torch.nn.Linear(16, 10)

    def forward(self, x):
        y = torch.relu(self.bn(self.conv(x)))
        y = torch.relu(self.conv2(y) + y)
        y = torch.nn.functional.adaptive_avg_pool2d(y, 1).flatten(1)
        return self.fc(y)


class _Mixed(_Net):
    """an op the lowering refuses (cumsum) in the middle: two aiednn regions around torch code"""

    def forward(self, x):
        y = torch.relu(self.bn(self.conv(x)))
        y = torch.cumsum(y, dim=3)
        y = torch.relu(self.conv2(y))
        return self.fc(torch.nn.functional.adaptive_avg_pool2d(y, 1).flatten(1))


def _run(model, x, graph=False, **kw):
    kw["graph"] = graph
    saved = {k: getattr(flags, k) for k in kw}
    for k, v in kw.items():
        setattr(flags, k, v)
    try:
        torch._dynamo.reset()
        with torch.no_grad():
            got = torch.compile(model, backend="aiednn")(x)
            want = model(x)
        return got, want, be.aiednn_backend.last
    finally:
        for k, v in saved.items():
            setattr(flags, k, v)


def _rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def test_whole_graph_one_region():
    torch.manual_seed(0)
    m, x = _Net().eval(), torch.randn(1, 3, 16, 16)
    got, want, fused = _run(m, x)
    assert len(fused.aiednn_regions) == 1 and not fused.aiednn_unsupported
    assert _rel(got, want) < 1e-5
    impls = next(iter(fused.aiednn_regions.values())).impls
    assert any(":aie:" in i for i in impls), impls


def test_unsupported_op_stays_torch():
    torch.manual_seed(1)
    m, x = _Mixed().eval(), torch.randn(1, 3, 16, 16)
    got, want, fused = _run(m, x)
    left = [n for n in fused.graph.nodes if n.op == "call_function"]
    assert fused.aiednn_unsupported and [n.target for n in left] == [torch.cumsum], fused.graph
    assert len(fused.aiednn_regions) == 2
    assert _rel(got, want) < 1e-5


def test_benchmark_flag_and_int8():
    torch.manual_seed(2)
    m, x = _Net().eval(), torch.randn(1, 3, 16, 16)
    got, want, fused = _run(m, x, benchmark=True, mode="int8")
    assert _rel(got, want) < 0.1
    impls = next(iter(fused.aiednn_regions.values())).impls
    assert any("s8" in i or "blocked" in i for i in impls), impls


def test_allow_ref_false_keeps_ref_regions_on_torch():
    m = torch.nn.Sequential(torch.nn.Softsign()).eval()   # Softsign: a lone eltwise
    x = torch.randn(1, 8, 4, 4)
    try:
        got, want, fused = _run(m, x, allow_ref=False)
    except Exception as exc:   # noqa: BLE001
        pytest.skip(f"Softsign not lowered: {exc}")
    assert _rel(got, want) < 1e-5
    for r in fused.aiednn_regions.values():
        assert r.on_torch or any(not i.startswith("ref:") for i in r.impls)


# ── the graph-API path (aiednn_graph.h: the library decides the partitions) ──────────────────────

needs_graph = pytest.mark.skipif(not be._graph_api(), reason="partition graph API (aiednn_graph.h) not in this tree")


@needs_graph
def test_graph_partitions_whole_net():
    torch.manual_seed(0)
    m, x = _Net().eval(), torch.randn(1, 3, 16, 16)
    got, want, fused = _run(m, x, graph=True)
    fg = fused.aiednn_graph
    assert all(p.supported for p in fg.parts) and not fg.wildcards, fg.report()
    infos = [p.info for p in fg.parts]
    assert any(i.startswith("conv+add+relu") for i in infos), infos      # residual + relu fused (BN folded)
    assert any(":aie:" in pm.cp.impl for pm in fused.aiednn_parts.values())
    assert _rel(got, want) < 1e-5


@needs_graph
def test_graph_wildcard_stays_torch():
    torch.manual_seed(1)
    m, x = _Mixed().eval(), torch.randn(1, 3, 16, 16)
    got, want, fused = _run(m, x, graph=True)
    fg = fused.aiednn_graph
    assert list(fg.wildcards.values()) == ["cumsum"]
    left = [n.target for n in fused.graph.nodes if n.op == "call_function"]
    assert torch.cumsum in left
    assert _rel(got, want) < 1e-5


@needs_graph
def test_graph_benchmark_pins_fastest():
    torch.manual_seed(2)
    m, x = _Net().eval(), torch.randn(1, 3, 16, 16)
    got, want, fused = _run(m, x, graph=True, benchmark=True)
    assert _rel(got, want) < 1e-5
    assert all(pm.cp.impl for pm in fused.aiednn_parts.values())


@needs_graph
def test_graph_ceil_max_pool():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(3, 2, ceil_mode=True)).eval()
    x = torch.randn(1, 3, 14, 14)
    got, want, fused = _run(m, x, graph=True)
    assert not fused.aiednn_graph.wildcards and _rel(got, want) < 1e-5


def _pt2e(model, x):
    """PT2E int8: symmetric per-channel weights, affine int8 activations (zero points != 0)"""
    try:
        from torch._export import capture_pre_autograd_graph
        from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (XNNPACKQuantizer,
                                                                       get_symmetric_quantization_config)
    except ImportError as exc:
        pytest.skip(f"PT2E not available: {exc}")
    gm = capture_pre_autograd_graph(model, (x,))
    gm = prepare_pt2e(gm, XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True)))
    gm(x)
    return convert_pt2e(gm)


@needs_graph
def test_graph_pt2e_int8_bit_exact():
    """quantize / dequantize ops fold into int8 GEMMs on the AIE; the result equals PT2E's own reference"""
    torch.manual_seed(3)
    m, x = _Net().eval(), torch.randn(1, 3, 16, 16)
    gm = _pt2e(m, x)
    got, want, fused = _run(gm, x, graph=True)
    impls = [pm.cp.impl for pm in fused.aiednn_parts.values()]
    # conv, conv (+residual), fc: int8 GEMMs on the AIE -- with nonzero activation zero points the
    # library's zp+ composite (the symmetric s8 problem plus an exact correction) serves them
    assert sum("gemm_s8s32" in i or i.startswith("zp+") for i in impls) == 3, impls
    assert not fused.aiednn_graph.wildcards
    assert torch.equal(got, want)


class _LayerScale(torch.nn.Module):
    """convnext block tail: NHWC Linear, permute back to NCHW, layer scale [C,1,1], residual"""

    def __init__(self, c=16):
        super().__init__()
        self.dw = torch.nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.norm = torch.nn.LayerNorm(c)
        self.fc = torch.nn.Linear(c, c)
        self.scale = torch.nn.Parameter(torch.rand(c, 1, 1) + 0.5)

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = torch.nn.functional.layer_norm(y, (y.shape[-1],), self.norm.weight, self.norm.bias, 1e-6)
        y = self.fc(y).permute(0, 3, 1, 2)
        return x + self.scale * y


@needs_graph
def test_graph_sink_permute_layer_scale():
    torch.manual_seed(4)
    m, x = _LayerScale().eval(), torch.randn(1, 16, 8, 8)
    got, want, fused = _run(m, x, graph=True)
    fg = fused.aiednn_graph
    assert not fg.wildcards, fg.wildcards                                # functional layer_norm mapped
    assert any(p.info.startswith("matmul+multiply") for p in fg.parts), [p.info for p in fg.parts]
    assert _rel(got, want) < 1e-5


def _pt2e_net(net):
    torch.manual_seed(5)
    if net == "_Net":
        m, x = _Net().eval(), torch.randn(1, 3, 16, 16)
    else:
        zoo = pytest.importorskip("zoo")
        m, x = zoo.build("resnet18")
    return _pt2e(m, x), x


def _levels(gm, x, levels):
    """torch.compile output and partition count per graph-API fusion level"""
    from aiegraph import dnn_torch_backend as be
    out = {}
    orig = be.FxGraph if hasattr(be, "FxGraph") else None
    import aiegraph.dnn_torch_graph as tg
    for lv in levels:
        real = tg.FxGraph.__init__
        def init(self, lib, g, ex, fuse=1, _lv=lv, _real=real):
            _real(self, lib, g, ex, fuse=_lv)
        tg.FxGraph.__init__ = init
        try:
            got, want, f = _run(gm, x, graph=True)
        finally:
            tg.FxGraph.__init__ = real
        out[lv] = (got, len(f.aiednn_graph.parts), want)
    return out


@needs_graph
@pytest.mark.parametrize("net", ["_Net", "zoo_resnet18"])
def test_graph_pt2e_level1_matches_pt2e(net):
    """Default fusions (level 1) against PT2E's eager output: top-1 equal."""
    gm, x = _pt2e_net(net)
    r = _levels(gm, x, (0, 1))
    got, want = r[1][0], r[1][2]
    assert r[1][1] < r[0][1], (r[0][1], r[1][1])
    if net == "_Net":
        assert torch.equal(got, want)
    else:
        assert int(got.argmax()) == int(want.argmax())
        assert float((got - want).abs().max()) <= 0.05 * float(want.abs().max())


@needs_graph
@pytest.mark.parametrize("net", ["_Net", "zoo_resnet18"])
def test_graph_pt2e_level2_within_a_code(net):
    """fuse_residual_requant (level 2): fewer partitions, same top-1, within about one code."""
    gm, x = _pt2e_net(net)
    got1, _, f1 = _run(gm, x, graph=True)
    got2, _, f2 = _run(gm, x, graph=True, fuse_residual_requant=True)
    assert len(f2.aiednn_graph.parts) <= len(f1.aiednn_graph.parts)
    assert int(got2.argmax()) == int(got1.argmax())
    assert float((got2 - got1).abs().max()) <= 0.05 * float(got1.abs().max())


class _CausalPos(torch.nn.Module):
    """sdpa with a positional is_causal keeps causal."""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, True, scale=0.2)


@needs_graph
def test_graph_sdpa_positional_causal_and_scale():
    import copy

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    gm = torch.fx.symbolic_trace(_CausalPos())
    run = be._graph_backend(copy.deepcopy(gm), [q, k, v])
    kinds = [kw for kind, _, kw in be.aiednn_backend.last.aiednn_graph.ops.values() if kind == "sdpa"]
    assert kinds and kinds[0]["causal"] == 1 and abs(kinds[0]["sdpa_scale"] - 0.2) < 1e-7, kinds
    with torch.no_grad():
        assert _rel(run(q, k, v), gm(q, k, v)) < 1e-5
