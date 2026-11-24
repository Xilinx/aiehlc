###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Build-time planners moved out of the example (unet-decl P2): kacc weight pre-layout
(aiegraph.wlayout), the carrier choice (aiegraph.callsite.pick_carrier) and partition keep_layout rows
realized by layout folds (aiegraph.fold.keep_layout_plan). x86 only, synthetic shapes / graphs; every
planner has a counter-example (test_<fn>_rejects*).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from aiegraph import cnn  # noqa: E402
from aiegraph.callsite import check_conv_arg0, pick_carrier  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.fold import keep_layout_plan  # noqa: E402
from aiegraph.wlayout import kacc_dense_wlayout, kacc_wlayout, prelayout_weights  # noqa: E402


# ───────────────────────────── kacc weight layout ─────────────────────────────
def _ref_chunks(wn):
    """element-by-element reference of the kacc W chunk order (nb row j q) of a natural [N][K] matrix"""
    N, K = wn.shape
    kch = cnn.kacc_geom(N, K, _PX)["kch"]
    out = np.empty(N * K, wn.dtype)
    nj = K // kch
    for n in range(N):
        nb, mi, mm = n // 16, (n % 16) // 8, n % 8
        for k in range(K):
            j, kb, kk = k // kch, (k % kch) // 8, k % 8
            f = mi * 8 * kch + kb * 64 + mm * 8 + kk
            out[((nb * 16 + f // kch) * nj + j) * kch + f % kch] = wn[n, k]
    return out


_PX = 64


def _w6(N, C, kh, kw, ib, ob, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65535, (N // ob, C // ib, kh, kw, ib, ob)).astype(np.uint16)


def test_kacc_wlayout_matches_reference():
    N, C, kh, kw = 64, 16, 2, 2          # K = 64
    w6 = _w6(N, C, kh, kw, ib=4, ob=8)
    K = C * kh * kw
    wn = w6.transpose(0, 5, 1, 4, 2, 3).reshape(N, K)     # n = o*ob + oi, k = c*KH*KW + t
    got = kacc_wlayout(w6, N, K, _PX)
    assert got.dtype == np.uint16 and got.size == N * K
    assert np.array_equal(got, _ref_chunks(wn))


def test_kacc_wlayout_rejects_wrong_shape():
    w6 = _w6(64, 16, 2, 2, ib=4, ob=8)
    with pytest.raises(AiegraphError, match="not N=128"):
        kacc_wlayout(w6, 128, 64, _PX)
    with pytest.raises(AiegraphError, match="6-D"):
        kacc_wlayout(np.zeros((64, 64), np.uint16), 64, 64, _PX)


def test_kacc_wlayout_rejects_no_kacc_walk():
    w6 = _w6(32, 8, 2, 2, ib=4, ob=8)    # N = 32: no gemm_kacc instance (dims multiple of 64)
    with pytest.raises(AiegraphError, match="multiple of 64"):
        kacc_wlayout(w6, 32, 32, _PX)


def test_kacc_dense_wlayout_same_order_as_conv():
    N, K, nb = 64, 128, 16
    rng = np.random.default_rng(1)
    wn = rng.integers(0, 65535, (N, K)).astype(np.uint16)
    wp = wn.reshape(N // nb, nb, K).transpose(0, 2, 1)              # TVM dense_pack [N/nb][K][nb]
    got = kacc_dense_wlayout(wp, N, K, _PX)
    assert np.array_equal(got, _ref_chunks(wn))
    w6 = wn.reshape(N // 8, 8, K // 4, 4, 1, 1).transpose(0, 2, 4, 5, 3, 1)  # the same matrix as a 1x1 conv
    assert np.array_equal(got, kacc_wlayout(w6, N, K, _PX))


def test_kacc_dense_wlayout_rejects_wrong_shape():
    wp = np.zeros((4, 128, 16), np.uint16)
    with pytest.raises(AiegraphError, match="not N=64 x K=64"):
        kacc_dense_wlayout(wp, 64, 64, _PX)
    with pytest.raises(AiegraphError, match="3-D"):
        kacc_dense_wlayout(np.zeros((64, 128), np.uint16), 64, 128, _PX)


def _blob(tmp_path):
    """weights.bin: 8 B header, a conv weight, 6 B gap, a dense weight, 4 B tail"""
    rng = np.random.default_rng(2)
    w6 = _w6(64, 16, 2, 2, ib=4, ob=8, seed=3)
    wp = rng.integers(0, 65535, (4, 128, 16)).astype(np.uint16)
    head, gap, tail = b"HEADER!!", b"gapgap", b"TAIL"
    data = head + w6.tobytes() + gap + wp.tobytes() + tail
    p = tmp_path / "weights.bin"
    p.write_bytes(data)
    conv = dict(woff=len(head), wshape=list(w6.shape), N=64, K=64, M=_PX)
    lin = dict(kind="linear", woff=len(head) + w6.nbytes + len(gap), wshape=list(wp.shape), N=64, K=128, pxs=_PX)
    return p, data, w6, wp, conv, lin, (head, gap, tail)


def test_prelayout_weights_in_place(tmp_path):
    p, data, w6, wp, conv, lin, (head, gap, tail) = _blob(tmp_path)
    assert prelayout_weights(p, [conv, lin]) == w6.nbytes + wp.nbytes
    new = p.read_bytes()
    assert len(new) == len(data)
    want = (head + kacc_wlayout(w6, 64, 64, _PX).tobytes() + gap + kacc_dense_wlayout(wp, 64, 128, _PX).tobytes()
            + tail)
    assert new == want


def test_prelayout_weights_rejects_out_of_file(tmp_path):
    p, data, *_, conv, lin, _ = _blob(tmp_path)
    bad = dict(lin, woff=len(data) - 100)
    with pytest.raises(AiegraphError, match="outside"):
        prelayout_weights(p, [conv, bad])
    assert p.read_bytes() == data       # nothing written


def test_prelayout_weights_rejects_overlap(tmp_path):
    p, data, *_, conv, lin, _ = _blob(tmp_path)
    bad = dict(lin, woff=conv["woff"] + 2)
    with pytest.raises(AiegraphError, match="overlap"):
        prelayout_weights(p, [conv, bad])
    assert p.read_bytes() == data


# ───────────────────────────── carrier choice ─────────────────────────────
def _layers(tmp_path, defs):
    d = tmp_path / "layers"
    for name, syms in defs.items():
        (d / name).mkdir(parents=True)
        (d / name / "k.c").write_text("".join(f"int32_t {s}(void* args, int32_t* codes) {{ return 0; }}\n"
                                              for s in syms))
    return d


def test_pick_carrier_skips_shared_function(tmp_path):
    d = _layers(tmp_path, {"2_a": ["tvmgen_default_fa"], "5_b": ["tvmgen_default_fb"]})
    sel = [dict(node=10, symbol="tvmgen_default_fa"), dict(node=11, symbol="tvmgen_default_fb")]
    calls = sel + [dict(node=30, symbol="tvmgen_default_fa")]      # fa also runs on the CPU at node 30
    lidx, car = pick_carrier(sel, calls, d)
    assert (lidx, car["node"]) == (5, 11)
    lidx, car = pick_carrier(sel, sel, d)                            # no CPU site: the first one
    assert (lidx, car["node"]) == (2, 10)


def test_pick_carrier_rejects_all_shared(tmp_path):
    d = _layers(tmp_path, {"2_a": ["tvmgen_default_fa"]})
    sel = [dict(node=10, symbol="tvmgen_default_fa")]
    with pytest.raises(AiegraphError, match="no carrier"):
        pick_carrier(sel, sel + [dict(node=30, symbol="tvmgen_default_fa")], d)


def test_pick_carrier_rejects_ambiguous_definition(tmp_path):
    d = _layers(tmp_path, {"2_a": ["tvmgen_default_fa"], "3_a": ["tvmgen_default_fa"]})
    sel = [dict(node=10, symbol="tvmgen_default_fa")]
    with pytest.raises(AiegraphError, match="2 layer files"):
        pick_carrier(sel, sel, d)


# ── check_conv_arg0 (unet-decl P4: was the example's check_data_arg0) ──
_CONV_DIRECT = (" tvmgen_default_c0(void* args, int32_t* codes) {\n"
                "  y = (y + (((float*)p0_1)[i] * ((float*)p1_1)[j]));\n}\n")
_CONV_PADDED = (" tvmgen_default_c1(void* args, int32_t* codes) {\n"
                "  ((float*)data_pad)[i] = condval = ((float*)p0_1)[i];\n"
                "  y = (y + (((float*)data_pad)[i] * ((float*)p1_1)[j]));\n}\n")
_CONV_RESID = (" tvmgen_default_c2(void* args, int32_t* codes) {\n"
               "  y = (y + (((float*)p2_1)[i] * ((float*)p1_1)[j]));\n}\n")


def _conv_layers(tmp_path, files):
    d = tmp_path / "layers"
    for name, text in files.items():
        (d / name).mkdir(parents=True)
        (d / name / "k.c").write_text("int32_t" + text)
    return d


def test_check_conv_arg0_direct_and_padded(tmp_path):
    d = _conv_layers(tmp_path, {"0_a": _CONV_DIRECT, "1_b": _CONV_PADDED})
    assert check_conv_arg0(d, ["tvmgen_default_c0", "tvmgen_default_c1", "tvmgen_default_c0"]) == 2


def test_check_conv_arg0_rejects_activation_from_other_arg(tmp_path):
    d = _conv_layers(tmp_path, {"0_a": _CONV_DIRECT, "2_c": _CONV_RESID})
    with pytest.raises(AiegraphError, match="p2_1"):
        check_conv_arg0(d, ["tvmgen_default_c0", "tvmgen_default_c2"])


def test_check_conv_arg0_rejects_missing_or_duplicate_definition(tmp_path):
    d = _conv_layers(tmp_path, {"0_a": _CONV_DIRECT, "1_a": _CONV_DIRECT})
    with pytest.raises(AiegraphError, match="2 layer files"):
        check_conv_arg0(d, ["tvmgen_default_c0"])
    with pytest.raises(AiegraphError, match="0 layer files"):
        check_conv_arg0(d, ["tvmgen_default_c9"])


# ───────────────────────────── keep_layout rows -> layout folds ─────────────────────────────
def _fold(kind, calls):
    return SimpleNamespace(info=dict(kind=kind), calls=list(calls))


def test_keep_layout_plan_realized_by_fold():
    from test_keep_layout import _attn_like
    m, part = _attn_like()
    layers = [dict(onnx="c0", call=4), dict(onnx="to_q", call=9)]
    rep = keep_layout_plan(m, part, layers, folds=[_fold("conv+layout", [4, 5])])
    assert rep["kept"] == 1 and rep["folded"] == 1 and rep["label_ops"] == 2
    assert layers[0]["keep_layout"] == "nhwc" and layers[1]["keep_in"] == ["c0"]


def test_keep_layout_plan_no_rows_untouched():
    layers = [dict(onnx="c0", call=4)]
    rep = keep_layout_plan(None, {"layers": [{"node": "c0"}]}, layers, folds=[])
    assert rep == dict(edges=[], label_ops=0, kept=0) and layers == [dict(onnx="c0", call=4)]


def test_keep_layout_plan_rejects_unfolded_producer():
    from test_keep_layout import _attn_like
    m, part = _attn_like()
    layers = [dict(onnx="c0", call=4), dict(onnx="to_q", call=9)]
    with pytest.raises(AiegraphError, match="not realized by a layout fold.*c0"):
        keep_layout_plan(m, part, layers, folds=[_fold("conv+layout", [7, 8])])
    with pytest.raises(AiegraphError, match="not realized"):      # a non-layout fold does not count
        keep_layout_plan(m, part, layers, folds=[_fold("dense+silu", [4])])
    assert "keep_layout" not in layers[0]                           # nothing tagged on failure


def test_keep_layout_plan_rejects_host_consumer():
    from test_keep_layout import _attn_like
    m, part = _attn_like(consumer_on_aie=False)
    with pytest.raises(AiegraphError, match="host"):
        keep_layout_plan(m, part, [dict(onnx="c0", call=4)], folds=[_fold("conv+layout", [4])])
