###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.flashattn (unet-decl P3: the flash-attention group code moved from the example's fa_attn.py).

* the class table / kernel parts / step emit are byte-identical to the last example copy (git show d5e9f15);
* every public planner has a counter-example (test_<fn>_rejects*).
"""
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import aiegraph as ag  # noqa: E402
from aiegraph import boardmain as bm, flashattn as fa  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402

FA_LAST = "d5e9f15"   # last commit with the example's fa_attn.py
FA_CM = "a6c59eea"    # costmodel branch tip: the same fa_attn.py + the FA_HS host-split probes in fa_host (M2 P0)


def _row(group, role, H, N, M, D, **kw):
    if role == "qk":
        r = dict(group=group, role=role, xshape=[H, N, D], wshape=[H, M, D], yshape=[H, N, M])
    else:
        r = dict(group=group, role=role, xshape=[H, N, M], wshape=[H, D, M], yshape=[H, N, D])
    r.update(kw)
    return r


def _att(shapes=((8, 1024, 1024, 40), (8, 1024, 77, 40), (8, 256, 256, 80), (8, 64, 77, 160), (8, 1024, 1024, 40))):
    return [r for g, s in enumerate(shapes) for r in (_row(g, "qk", *s), _row(g, "pv", *s))]


@pytest.fixture(scope="module")
def old_fa():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{FA_LAST}:example/aiegraph/unet_bk_sdm_tiny/fa_attn.py"],
                         capture_output=True, text=True).stdout
    if not src:
        pytest.skip(f"git show {FA_LAST}:fa_attn.py unavailable")
    m = types.ModuleType("old_fa_attn")
    m.__file__ = str(ROOT / "example/aiegraph/unet_bk_sdm_tiny/fa_attn.py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


@pytest.fixture(scope="module")
def cm_fa():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{FA_CM}:example/aiegraph/unet_bk_sdm_tiny/fa_attn.py"],
                         capture_output=True, text=True).stdout
    if not src:
        pytest.skip(f"git show {FA_CM}:fa_attn.py unavailable")
    m = types.ModuleType("cm_fa_attn")
    m.__file__ = str(ROOT / "example/aiegraph/unet_bk_sdm_tiny/fa_attn.py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


def test_same_as_example_copy(old_fa, cm_fa):
    a, b = _att(), _att()
    new, old = fa.fa_classes(a), old_fa.fa_classes(b)
    assert new == old and [c["fa"] for c in a] == [c["fa"] for c in b] == [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]
    assert fa.fa_prologue(new) == old_fa.fa_prologue(old)
    # fa_host = the d5e9f15 text + the costmodel host-split probes, byte-identical to the costmodel copy
    assert fa.fa_host(new) == cm_fa.fa_host(cm_fa.fa_classes(_att())) == fa.fa_host_split(old_fa.fa_host(old))
    kw = fa.fa_kernel_parts(new)
    assert kw["kernels"] == "\n\n" + old_fa.FA_KERNEL and kw["prologue"] == old_fa.fa_prologue(old) + "\n"
    assert kw["host_entries"] == fa.fa_host_split(old_fa.fa_host(old)).lstrip("\n")
    assert kw["exports"] == ("aiednn_aie_attn", "aiednn_aie_attn_kl")
    assert fa.FA_STEPS == old_fa.fa_steps() and fa.FA_STEP_ROLES == old_fa.FA_STEP_ROLES
    assert fa.DECL_FA == old_fa.DECL_FA and fa.DECL_KL == old_fa.DECL_KL


def test_fa_classes_rejects_half_group_and_bad_shapes():
    with pytest.raises(AiegraphError, match="need exactly qk and pv"):
        fa.fa_classes([_row(0, "qk", 8, 256, 256, 80)])
    bad_pv = [_row(0, "qk", 8, 256, 256, 80), _row(0, "pv", 8, 256, 77, 80)]
    with pytest.raises(AiegraphError, match="does not continue QK"):
        fa.fa_classes(bad_pv)
    with pytest.raises(AiegraphError, match="multiple of 4"):
        fa.fa_classes(_att(((6, 256, 256, 80),)))
    with pytest.raises(AiegraphError, match="multiple of 128"):
        fa.fa_classes(_att(((8, 200, 256, 80),)))
    with pytest.raises(AiegraphError, match="needs N == 64"):
        fa.fa_classes(_att(((8, 128, 77, 160),)))
    with pytest.raises(AiegraphError, match="multiple of 8"):
        fa.fa_classes(_att(((8, 128, 77, 36),)))


def test_fa_prologue_rejects_no_class():
    with pytest.raises(AiegraphError, match="no flash-attention class"):
        fa.fa_prologue([])
    assert fa.fa_prologue(fa.fa_classes(_att())).endswith("// AIEHLC_KERNEL_PROLOGUE_END\n")


def test_fa_host_rejects_incomplete_class():
    c = dict(fa.fa_classes(_att())[0])
    del c["qkrow"]
    with pytest.raises(AiegraphError, match="lacks"):
        fa.fa_host([c])


def test_fa_kernel_parts_rejects_no_class():
    with pytest.raises(AiegraphError, match="no flash-attention class"):
        fa.fa_kernel_parts([])


class _G:
    """the fields of an attention.AttentionGroup the step planner reads"""
    def __init__(self, scale=0.125, scale_k=1.0):
        self.name, self.scale, self.scale_k = "attn0", scale, scale_k
        self.q, self.k, self.v, self.o = (ag.buf((1024, 40), "float32", ld=320, tensor=50 + i) for i in range(4))


def test_fa_group_steps_rejects_unscaled_group():
    steps = fa.fa_group_steps(ag, _G(), 4, 5, 0)
    assert [s.name for s in steps] == list(fa.FA_STEP_ROLES)
    with pytest.raises(AiegraphError, match="no Q/K scale"):
        fa.fa_group_steps(ag, _G(scale=None), 4, 5, 0)


def test_fa_step_emit_rejects_foreign_step_and_tensor():
    g = _G()
    emit = fa.fa_step_emit(g)
    e = emit(dict(name="fa_step_run_o", writes=[53], reads=[], consts=dict(cid=5, cls=0)))
    assert e["c"][1].startswith("aiednn_aie_attn_kl(5, 6, 0, 0, 0, (float *)tensors[53].data, 320, 1.0f);")
    assert e["tag"] == "aiednn_aie_attn 5"
    with pytest.raises(AiegraphError, match="touches"):
        emit(dict(name="fa_step_pack_q", reads=[99], writes=[], consts=dict(cid=4, cls=0, s=0.5)))
    with pytest.raises(AiegraphError, match="not a flash-attention step"):
        emit(dict(name="aiednn_aie_conv", reads=[50], writes=[], consts={}))


def test_fa_driver_glue_rejects_unplaced_rows():
    assert fa.fa_driver_glue([]) == bm.driver_glue()
    att = [dict(fa=0, fa_steps=True)]
    assert fa.fa_driver_glue(att) == bm.driver_glue(fa.DECL_FA + fa.DECL_KL, attn=True)
    with pytest.raises(AiegraphError, match="not FA group steps"):
        fa.fa_driver_glue([dict(fa=0)])
    with pytest.raises(AiegraphError, match="not FA group steps"):
        fa.fa_driver_glue([dict(fa=None, fa_steps=True, onnx="/attn/MatMul")])


def test_fa_host_split_rejects_missing_or_repeated_anchor():
    h = fa.fa_host(fa.fa_classes(_att()))
    with pytest.raises(AiegraphError, match="anchor found 0 times"):
        fa.fa_host_split("int main() { return 0; }")
    a0 = fa.FA_HS[0][0]
    raw = h.replace(fa.FA_HS[0][1], a0)   # undo probe 0, then duplicate its anchor
    with pytest.raises(AiegraphError, match="anchor found 2 times"):
        fa.fa_host_split(raw + a0)
