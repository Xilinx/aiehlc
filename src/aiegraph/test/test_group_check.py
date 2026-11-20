###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""S4 (design 2026-11-16-aiegraph-group-offload): the group-level x86 check, on real data.

Real network: the converted bk-sdm-tiny UNet (fp32 TVM, 194 layers; ``UNET_AG_WORK``) and the
model-zoo golden of its conv_in (``op_golden/p2_conv_s0_cond/c000``: the ONNX Conv's real input
``sample``, weight, bias and output). The conv_in group is TVM calls [0] layout_transform +
[1] conv2d_NCHWc_add; one step (a bf16 conv, the AIE numerics) replaces both.

  * the TVM chain on x86 (the C the board ELF runs) reproduces the golden output;
  * the bf16 step passes against the chain: cos >= 0.999, nonfinite == 0 (maxabs printed);
  * a corrupted step output (a layout bug, a scale bug) FAILS, a nonfinite one FAILS, a missing
    one FAILS -- AiegraphError, no flag turns it into a pass;
  * the layer offload form (``check="group"``): the UNet carrier layer 14 with build_unet's
    carrier reference passes, a wrong reference is rejected; ``force`` is refused with it;
  * build_unet.py no longer bypasses the checks.

Missing data is a FAILURE here, not a skip. Run:
  python3 -m pytest -q src/aiegraph/test/test_group_check.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import aiegraph as ag                                     # noqa: E402
from aiegraph.errors import AiegraphError                 # noqa: E402
from aiegraph.group import CallGraph, group_step, make_group, plan_group  # noqa: E402
from aiegraph.groupcheck import COS_MIN, compare, group_check, layer_checker, tvm_chain  # noqa: E402

WORK = Path(os.environ.get("UNET_AG_WORK",
                           "/scratch/staff/huaj/aiengineers/shuhua/tasks/2026-11-17_unet-m3-s4/unet_ag_work"))
GOLDEN = Path(os.environ.get("OPG_GOLDEN", "/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/op_golden")) \
    / "p2_conv_s0_cond" / "c000_conv_1x4x32x32_320x4x3x3"
UNET_EX = REPO / "example" / "aiegraph" / "unet_bk_sdm_tiny"


def _need(p: Path):
    if not p.exists():
        pytest.fail(f"missing test data {p} (set UNET_AG_WORK / OPG_GOLDEN)")


def _bf16(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def _nchwc(y, cb=4):
    n, c, h, w = y.shape
    return np.ascontiguousarray(y.reshape(n, c // cb, cb, h, w).transpose(0, 1, 3, 4, 2))


@pytest.fixture(scope="module")
def conv_in():
    """conv_in group plan + golden data."""
    _need(WORK / "resnet18_graph.json")
    _need(GOLDEN / "y.npy")
    cg = CallGraph.load(WORK)
    g = make_group(cg, [0, 1], "conv_in")
    sample = next(t.tensor for t in g.consts if cg.names.get(t.tensor) == "sample")
    out = g.outputs[0].tensor
    plan = plan_group(cg, g, [group_step("unet_conv_in", reads=[sample], writes=[out])])
    x, w, b, y = (np.load(GOLDEN / f"{k}.npy") for k in ("a", "b", "c", "y"))
    assert x.shape == (1, 4, 32, 32) and w.shape == (320, 4, 3, 3) and y.shape == (1, 320, 32, 32)
    return NS(cg=cg, g=g, plan=plan, sample=sample, out=out, x=x, w=w, b=b, y=y)


def _bf16_conv_step(d):
    """The step on x86 at AIE precision: bf16 activation and weight, fp32 accumulation, NCHW4c out."""
    def run(inputs):
        x = _bf16(inputs[d.sample])[0]
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))   # c,h,w,3,3
        o = np.einsum("chwrq,ocrq->ohw", win, _bf16(d.w), optimize=True) + d.b[:, None, None]
        return {d.out: _nchwc(o[None].astype(np.float32))}
    return run


def test_tvm_chain_reproduces_golden(conv_in):
    d = conv_in
    ref = tvm_chain(WORK, d.g.calls, {d.sample: d.x}, [d.out], cg=d.cg)[d.out]
    c = compare("conv_in", _nchwc(d.y), ref, against="golden")
    print(c.summary())
    assert c.ok and c.cos >= 0.99999 and c.maxabs < 1e-4, c.summary()


def test_group_check_real_conv_passes(conv_in):
    d = conv_in
    rep = group_check(WORK, d.plan, _bf16_conv_step(d), inputs={d.sample: d.x},
                      golden={d.out: _nchwc(d.y)})
    print(rep)
    assert rep.ok and len(rep.checks) == 2
    for c in rep.checks:
        assert c.cos >= COS_MIN and c.nonfinite == 0 and c.ref_nonfinite == 0, c.summary()
    assert {c.against for c in rep.checks} == {"tvm", "golden"}


@pytest.mark.parametrize("bug", ["channel_order", "scale", "zero_block"])
def test_corrupted_group_output_fails(conv_in, bug):
    d = conv_in
    good = _bf16_conv_step(d)

    def run(inputs):
        o = good(inputs)[d.out].copy()
        if bug == "channel_order":      # layout bug: the 4 channels of the c-block reversed
            o = np.ascontiguousarray(o[..., ::-1])
        elif bug == "scale":            # a forgotten constant: sign of a whole half of the channels
            o[:, : o.shape[1] // 2] *= -1.0
        else:                           # an unwritten slice: the step never wrote half of the blocks
            o[:, o.shape[1] // 2:] = 0.0
        return {d.out: o}

    with pytest.raises(AiegraphError, match=r"group-check.*conv_in.*cos=.*FAIL"):
        group_check(WORK, d.plan, run, inputs={d.sample: d.x})
    rep = group_check(WORK, d.plan, run, inputs={d.sample: d.x}, raise_on_fail=False)
    assert not rep.ok and rep.failed[0].cos < COS_MIN, str(rep)


@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_nonfinite_group_output_fails(conv_in, val):
    d = conv_in
    good = _bf16_conv_step(d)

    def run(inputs):
        o = good(inputs)[d.out].copy()
        o.flat[12345] = val             # ONE element: cos over the rest would still be ~1
        return {d.out: o}

    with pytest.raises(AiegraphError, match=r"nonfinite=1 "):
        group_check(WORK, d.plan, run, inputs={d.sample: d.x})


def test_missing_group_output_fails(conv_in):
    d = conv_in
    with pytest.raises(AiegraphError, match="no step wrote it"):
        group_check(WORK, d.plan, lambda inputs: {}, inputs={d.sample: d.x})


def test_wrong_golden_fails(conv_in):
    """With real data the TVM reference itself is held to the golden: a wrong golden (a
    different class's output) does not pass."""
    d = conv_in
    y2 = _nchwc(d.y[:, ::-1].copy())
    with pytest.raises(AiegraphError, match="vs golden"):
        group_check(WORK, d.plan, _bf16_conv_step(d), inputs={d.sample: d.x}, golden={d.out: y2})


def test_missing_boundary_input_fails(conv_in):
    d = conv_in
    with pytest.raises(AiegraphError, match="has no feed"):
        group_check(WORK, d.plan, _bf16_conv_step(d), inputs={})


def test_compare_unit():
    r = np.linspace(-1, 1, 1000, dtype=np.float32)
    assert compare("t", r, r).ok
    assert compare("t", r, _bf16(r)).ok                         # bf16 rounding is within 0.999
    assert not compare("t", r, -r).ok
    bad = r.copy()
    bad[3] = np.nan
    c = compare("t", r, bad)
    assert not c.ok and c.nonfinite == 1
    c = compare("t", bad, r)
    assert not c.ok and c.ref_nonfinite == 1
    assert compare("t", r, r[:10]).error


# ── the layer offload form: flow.offload(checker=...) gates 3-4 ─────────────────────────────

def _carrier_kp(net, ref: Path, layer: int = 14):
    class KP:
        def __init__(self):
            self.spec = NS(entry="aiek_unetconv", reference=ref, name=f"carrier_{ref.stem}")
            self.selected, self.rejected, self.calib = [layer], {}, {}
            self.verdicts = {layer: NS(calls=[NS(call=c) for c in net.layers()[layer]])}

        def reject(self, lay, gate, reason):
            self.selected.remove(lay)
            self.rejected[lay] = (gate, reason)
    return KP()


@pytest.fixture(scope="module")
def unet_net():
    _need(WORK / "resnet18_params.bin")
    import aiegraph._env as env
    env.frontend("")
    from frontend.tvmrelay.aiekernel.graph import load_network
    return load_network(WORK)


def test_carrier_layer_group_check(unet_net, tmp_path):
    import aiednn
    ref = tmp_path / "aiek_unetconv_ref.c"
    ref.write_text(aiednn.template("conv2d/conv_bf16_carrier_ref.c").replace("@E@", "aiek_unetconv"))
    kp = _carrier_kp(unet_net, ref)
    layer_checker()(unet_net, [kp], True)
    assert kp.selected == [14] and not kp.rejected, kp.rejected
    c = kp.calib[14][0].report.checks[0]
    assert c.cos >= COS_MIN and c.nonfinite == 0, c.summary()


def test_carrier_layer_wrong_reference_rejected(unet_net, tmp_path):
    import aiednn
    ref = tmp_path / "aiek_unetconv_ref.c"           # the stride/pad of the model off by one
    ref.write_text(aiednn.template("conv2d/conv_bf16_carrier_ref.c").replace("@E@", "aiek_unetconv")
                   .replace("PH = (KH - 1) / 2", "PH = 0"))
    kp = _carrier_kp(unet_net, ref)
    layer_checker()(unet_net, [kp], True)
    assert kp.selected == [] and kp.rejected[14][0] == "group-check", kp.rejected
    ref.write_text("#include \"aie_user_kernel.h\"\nAIEK_REFERENCE(aiek_unetconv) "
                   "{ (void)args; (void)num_args; return AIEK_ERR_KERNEL; }\n")
    kp = _carrier_kp(unet_net, ref)                   # the old stub: refused, not skipped
    layer_checker()(unet_net, [kp], True)
    assert kp.selected == [] and "rc_ref=-4" in kp.rejected[14][1], kp.rejected


def test_force_refused_with_group_check():
    import aiegraph._env as env
    env.frontend("")
    from frontend.tvmrelay.aiekernel.flow import offload
    with pytest.raises(ValueError, match="binding"):
        offload("/nonexistent", [], force=True, checker=lambda *a: None)


def test_build_unet_has_no_check_bypass():
    # P4 (unet-m4): the offload declarations moved from build_unet.py to unet_offload.py
    src = (UNET_EX / "build_unet.py").read_text() + (UNET_EX / "unet_offload.py").read_text()
    assert src.count("force=True") == 0
    assert "_calibrate" not in src and "_emulate" not in src
    assert 'check="group"' in src
