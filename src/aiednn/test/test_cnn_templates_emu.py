"""aiednn CNN templates on x86 (sprint unet-m4 P5k): every template runs through the kemu kernel model +
aie_api_x86emu.h (bf16 mmul / mac into accfloat) on REAL layers of resnet18 / yolo26n -- shape, stride, pad and
weights from the ONNX file -- and matches a float64 numpy reference on the same bf16-rounded inputs (cos >= 0.99999).

Test names carry the template keys: stem7x7, conv_s2, conv1x1, fc_gemm, dw3x3, conv_silu.
"""
import zlib

import numpy as np
import pytest

from emu_cnn import (COS_MIN, bf16, bf16_bits, conv_np, cos, emulate, largest_divisor, layer, tco)


def _rng(tag: str):
    return np.random.default_rng(zlib.crc32(tag.encode()))


def _act(consumers) -> int:
    return 1 if "Relu" in consumers else 2 if "Sigmoid" in consumers else 0


def _conv_case(template, model, x_shape, w_shape, s, p, mesh, kxk=True, tag=""):
    C, H, W = x_shape
    CO, CG, KH, KW = w_shape
    w, b, attrs, cons = layer(model, "Conv", (1, C, H, W), w_shape, (s, s))
    assert list(attrs.get("pads", [0] * 4)) == [p] * 4 and list(attrs["kernel_shape"]) == [KH, KW]
    act = _act(cons)
    OH, OW = (H + 2 * p - KH) // s + 1, (W + 2 * p - KW) // s + 1
    rng = _rng(tag or template + str(x_shape))
    x = rng.uniform(-1, 1, x_shape).astype(np.float32)
    toh = largest_divisor(OH, max(1, 256 // OW))
    prm = dict(NAME="aiek_" + template.split("/")[1].split("_bf16")[0], CI=C, H=H, W=W, CO=CO, OH=OH, OW=OW, S=s,
               TCO=tco(CO), TOH=toh)
    if kxk:
        tci = C if "stem7x7" in template else largest_divisor(C, max(1, 288 // (KH * KW)))
        prm.update(KH=KH, KW=KW, P=p, K=C * KH * KW, TCI=tci)
    else:
        prm.update(TCI=largest_divisor(C, 64, 8))
    if "conv_silu" not in template:
        prm["ACT"] = act
    else:
        assert act == 2, f"{model} conv {x_shape} {w_shape} is not followed by Sigmoid (consumers {cons})"
    y = emulate(template, prm, [bf16_bits(x), bf16_bits(w).reshape(CO, -1), b], (CO, OH, OW), False, mesh)
    ref = conv_np(bf16(x), bf16(w), b, s, p, act=act)
    c = cos(y, ref)
    print(f"{template} {model} x{x_shape} w{w_shape} s{s} act{act} tiles {prm}: cos={c:.9f}")
    assert c >= COS_MIN, c
    return c


# ---------------------------------------------------------------- stem 7x7 s2 (ResNet conv1)
def test_stem7x7_resnet18_conv1():
    _conv_case("conv2d/stem7x7_bf16.cc", "resnet18", (3, 224, 224), (64, 3, 7, 7), 2, 3, (4, 2))


# ---------------------------------------------------------------- KxK stride 2
@pytest.mark.parametrize("model,x,w", [
    ("resnet18", (64, 56, 56), (128, 64, 3, 3)),
    ("resnet18", (256, 14, 14), (512, 256, 3, 3)),
    ("yolo26n", (128, 40, 40), (128, 128, 3, 3)),
], ids=["r18_layer2", "r18_layer4", "y26_p4"])
def test_conv_s2(model, x, w):
    _conv_case("conv2d/conv_s2_bf16.cc", model, x, w, 2, 1, (2, 2))


# ---------------------------------------------------------------- 1x1 (pointwise / downsample shortcut)
@pytest.mark.parametrize("model,x,w,s", [
    ("resnet18", (64, 56, 56), (128, 64, 1, 1), 2),
    ("resnet18", (256, 14, 14), (512, 256, 1, 1), 2),
    ("yolo26n", (96, 80, 80), (128, 96, 1, 1), 1),
    ("yolo26n", (80, 20, 20), (80, 80, 1, 1), 1),
    ("yolo26n", (16, 20, 20), (4, 16, 1, 1), 1),
], ids=["r18_ds2", "r18_ds4", "y26_96to128", "y26_head80", "y26_box4"])
def test_conv1x1(model, x, w, s):
    _conv_case("conv2d/conv1x1_bf16.cc", model, x, w, s, 0, (2, 2), kxk=False)


# ---------------------------------------------------------------- KxK conv + fused SiLU (YOLO Conv block)
@pytest.mark.parametrize("x,w,s", [
    ((3, 640, 640), (16, 3, 3, 3), 2),
    ((32, 40, 40), (32, 32, 3, 3), 1),
    ((16, 80, 80), (16, 16, 3, 3), 1),
    ((64, 20, 20), (128, 64, 3, 3), 1),
], ids=["y26_stem", "y26_c3k_40", "y26_c3k_80", "y26_64to128"])
def test_conv_silu(x, w, s):
    _conv_case("conv2d/conv_silu_bf16.cc", "yolo26n", x, w, s, 1, (2, 2))


# ---------------------------------------------------------------- depthwise 3x3
@pytest.mark.parametrize("c,h", [(128, 20), (80, 80), (256, 20), (64, 80)], ids=["c128_20", "c80_80", "c256_20", "c64_80"])
def test_dw3x3_yolo26n(c, h):
    w, b, attrs, cons = layer("yolo26n", "Conv", (1, c, h, h), (c, 1, 3, 3), (1, 1), group=c)
    assert list(attrs["pads"]) == [1] * 4
    act = _act(cons)
    x = _rng(f"dw{c}_{h}").uniform(-1, 1, (c, h, h)).astype(np.float32)
    prm = dict(NAME="aiek_dw3x3", C=c, H=h, W=h, TC=largest_divisor(c, 16), TOH=largest_divisor(h, 8), ACT=act)
    y = emulate("conv2d/dw3x3_bf16.cc", prm, [bf16_bits(x), bf16_bits(w).reshape(c, 9), b], (c, h, h), False, (2, 2))
    ref = conv_np(bf16(x), bf16(w), b, 1, 1, group=c, act=act)
    cc = cos(y, ref)
    print(f"dw3x3 c{c} {h}x{h} act{act}: cos={cc:.9f}")
    assert cc >= COS_MIN, cc


# ---------------------------------------------------------------- FC (Gemm 512 -> 1000)
def test_fc_gemm_resnet18_classifier():
    w, b, attrs, _ = layer("resnet18", "Gemm", (1, 512), (1000, 512))
    assert attrs.get("transB", 0) == 1 and attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
    x = np.abs(_rng("fc").standard_normal(512)).astype(np.float32)  # post-ReLU average-pooled features
    y = emulate("gemm/fc_gemm_bf16.cc", dict(NAME="aiek_fc", K=512, N=1000, NB=25), [bf16_bits(x), bf16_bits(w), b],
                (1000,), True, (2, 4))
    ref = bf16(w).astype(np.float64) @ bf16(x).astype(np.float64) + b
    c = cos(y, ref)
    print(f"fc_gemm 512->1000: cos={c:.9f} top5 {np.argsort(-y)[:5]} ref {np.argsort(-ref)[:5]}")
    assert c >= COS_MIN, c
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-4)
