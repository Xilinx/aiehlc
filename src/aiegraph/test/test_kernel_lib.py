###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``ag.kernel_lib`` / ``use_kernels`` / ``match_kernel`` / ``lib.add``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import aiegraph as ag  # noqa: E402
from aiegraph.region import Region, Regions  # noqa: E402
from aiegraph.report import CheckReport, RegionCheck  # noqa: E402

CONV_PARAMS = ("param int32 [1,{o},1,1,4] zero_point_a\n"
               "param int32 [1,{o},1,1,4] bias\n")


def _spec_src(entry, weight, out, op="conv2d_requant", match="contrib_conv2d_NCHWc", extra=""):
    return (f'#include "aie_user_kernel.h"\n'
            f'AIEK_SPEC({entry},\n'
            f'    "op {op}\\n"\n'
            f'    "match {match}\\n"\n'
            f'    "in int8 [1,CIC,HI,WI,4] ifm\\n"\n'
            f'    "param int8 {weight} weight\\n"\n'
            f'    "out int8 {out} ofm\\n");\n'
            f'AIEK_REFERENCE({entry}) {{ (void)args; (void)num_args; return 0; }}\n{extra}')


@pytest.fixture
def kdir(tmp_path):
    """stem7 (fixed 7x7, O 64 I 3), conv3 (3x3, channels open), anyconv (all `*`)."""
    d = tmp_path / "aietensorop"
    for sub, entry, w in [("stem", "stem7", "[16,1,7,7,3,4]"),
                          ("conv3x3", "conv3", "[COC,CIC,3,3,4,4]"),
                          ("anyconv", "anyconv", "[COC,CIC,*,*,4,4]")]:
        (d / sub).mkdir(parents=True)
        (d / sub / f"{entry}_uk.cc").write_text(_spec_src(entry, w, "[1,COC,HO,WO,4]"))
    (d / "broken").mkdir()
    (d / "broken" / "bad_uk.cc").write_text('#include "aie_user_kernel.h"\n'
                                            'AIEK_SPEC(bad, "in int8 [1,2\\n");\n')
    (d / "notes.txt").write_text("AIEK_SPEC( in a text file is ignored")
    return d


def _region(name, oihw, layer=1, op="conv2d", dtype="int8"):
    return Region(name=name, node=name, op=op, weight_shape=tuple(oihw), weight_dtype=dtype,
                  quantization="per_channel", input_layout="NCHW" if op == "conv2d" else "NC",
                  weight_layout="OIHW" if op == "conv2d" else "OI", layer=layer)


STEM = _region("conv1", (64, 3, 7, 7), layer=1)
L1C1 = _region("layer1.0.conv1", (64, 64, 3, 3), layer=4)
L1C1B = _region("layer1.1.conv1", (64, 64, 3, 3), layer=4)
DS = _region("layer2.0.downsample.0", (128, 64, 1, 1), layer=9)
FC = _region("fc", (1000, 512), layer=40, op="dense")


def test_index_reads_every_spec_and_skips_bad_files(kdir, capsys):
    lib = ag.kernel_lib(kdir)
    assert sorted(e.name for e in lib) == ["anyconv", "conv3", "stem7"]
    assert lib["stem7"].weight == (64, 3, 7, 7)            # [16,1,7,7,3,4] unpacked
    assert lib["conv3"].weight == (None, None, 3, 3)
    assert lib["stem7"].contract.op == "conv2d"
    assert [str(p) for p, _ in lib.skipped] == ["broken/bad_uk.cc"]
    assert "skipped broken/bad_uk.cc" in capsys.readouterr().out


def test_index_imports_neither_tvm_nor_onnx(kdir):
    code = ("import sys, aiegraph as ag; lib = ag.kernel_lib(sys.argv[1], quiet=True); "
            "assert len(lib) == 3; "
            "bad = [m for m in ('tvm', 'onnx', 'frontend.tvmrelay') if m in sys.modules]; "
            "assert not bad, bad")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    res = subprocess.run([sys.executable, "-c", code, str(kdir)], env=env,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_fixed_spec_beats_wildcard(kdir):
    lib = ag.kernel_lib(kdir, quiet=True)
    assert lib.match([STEM], reference_only=True).winner.name == "stem7"
    assert lib.match([L1C1, L1C1B], reference_only=True).winner.name == "conv3"
    d = lib.match([DS], reference_only=True)          # 1x1: only the wildcard accepts it
    assert d.winner.name == "anyconv"
    assert "KH 1, spec says 3" in repr(d)


def test_mixed_kh_group_is_none(kdir):
    d = ag.kernel_lib(kdir, quiet=True).match([L1C1, DS], reference_only=True)
    assert d.winner is None and "disagree on (op, KH)" in d.reason


def test_equally_specific_tie_is_none(kdir):
    (kdir / "conv3x3" / "conv3b_uk.cc").write_text(
        _spec_src("conv3b", "[COC,CIC,3,3,4,4]", "[1,COC,HO,WO,4]"))
    d = ag.kernel_lib(kdir, quiet=True).match([L1C1], reference_only=True)
    assert d.winner is None and d.reason.startswith("ambiguous: ")


def test_fixed_channels_must_agree(kdir):
    d = ag.kernel_lib(kdir, quiet=True).match([_region("x", (32, 3, 7, 7))], reference_only=True)
    assert d.winner.name == "anyconv"                 # stem7 says O 64
    assert "O 32, spec says 64" in repr(d)


def test_dense_and_dtype(kdir):
    (kdir / "fc").mkdir()
    (kdir / "fc" / "fc_uk.cc").write_text(
        _spec_src("fc", "[125,512,8]", "[1,1000]", op="dense_requant", match="contrib_dense_pack"))
    lib = ag.kernel_lib(kdir, quiet=True)
    assert lib["fc"].contract.op == "dense" and lib["fc"].weight == (1000, 512)
    assert lib.match([FC], reference_only=True).winner.name == "fc"
    u8 = _region("conv1", (64, 3, 7, 7), dtype="uint8")
    assert "weight uint8, spec says int8" in repr(lib.match([u8], reference_only=True))


def test_reference_only_is_not_dispatched_by_default(kdir):
    lib = ag.kernel_lib(kdir, quiet=True)
    d = lib.match([STEM])
    assert d.winner is None and "reference-only" in repr(d)
    lib["stem7"].contract.lint.kernels.append(("stem7_kernel", 0))   # pretend an AIE body
    assert lib.match([STEM]).winner.name == "stem7"


def test_add_waits_for_numerics(kdir, tmp_path):
    lib = ag.kernel_lib(kdir, quiet=True)
    ported = tmp_path / "aiproto" / "conv1x1_l9.cc"
    ported.parent.mkdir()
    ported.write_text(_spec_src("conv1x1_l9", "[COC,CIC,1,1,4,4]", "[1,COC,HO,WO,4]"))
    e = lib.add("conv1x1_l9", ported, needs_numerics_test=True)
    assert e.origin == "added" and not e.verified
    d = lib.match([DS], reference_only=True)
    assert d.winner.name == "anyconv" and "awaiting numerics" in repr(d)
    lib.record_numerics("conv1x1_l9", bit_exact=False)
    assert not lib["conv1x1_l9"].verified
    lib.record_numerics("conv1x1_l9", bit_exact=True)
    assert lib.match([DS], reference_only=True).winner.name == "conv1x1_l9"


def test_add_refuses_a_failed_port(kdir, tmp_path):
    lib = ag.kernel_lib(kdir, quiet=True)
    bad = tmp_path / "bad.cc"
    bad.write_text('#include "aie_user_kernel.h"\nint x;\n')
    with pytest.raises(ag.AiegraphError, match="no AIEK_SPEC"):
        lib.add("bad", bad)
    assert "bad" not in lib


def test_duplicate_name_is_refused(kdir):
    (kdir / "dup").mkdir()
    (kdir / "dup" / "stem7_uk.cc").write_text(_spec_src("stem7", "[16,1,7,7,3,4]", "[1,16,HO,WO,4]"))
    with pytest.raises(ag.AiegraphError, match="two kernels named 'stem7'"):
        ag.kernel_lib(kdir, quiet=True)


# ── module.use_kernels / match_kernel / check_offload records numerics ───

def _module(tmp_path, regions):
    emod = ag.EmbeddedModel(tmp_path / "wd")
    emod._regions = Regions(regions)
    return emod


def test_use_kernels_and_match_kernel(kdir, tmp_path, capsys):
    emod = _module(tmp_path, [STEM, L1C1, L1C1B, DS])
    with pytest.raises(ag.AiegraphError, match="no kernel library"):
        emod.match_kernel("conv1")
    lib = ag.kernel_lib(kdir, quiet=True)
    for e in lib:
        e.contract.lint.kernels.append((e.name, 0))
    assert emod.use_kernels(lib) == list(lib.entries)
    assert set(emod._kernels) == {"stem7", "conv3", "anyconv"}
    assert emod.match_kernel("conv1").name == "stem7"
    assert emod.match_kernel(["layer1.0.conv1", "layer1.1.conv1"]).name == "conv3"
    assert emod.match_kernel(["layer1.0.conv1", "layer2.0.downsample.0"]) is None
    emod.match_kernel("conv1", verbose=True)
    assert "match conv1 -> stem7" in capsys.readouterr().out


def test_use_kernels_refuses_a_conflicting_registration(kdir, tmp_path):
    emod = _module(tmp_path, [STEM])
    other = tmp_path / "stem7_other.cc"
    other.write_text(_spec_src("stem7", "[16,1,7,7,3,4]", "[1,16,HO,WO,4]"))
    emod.register_kernel("stem7", other, "conv2d")
    with pytest.raises(ag.AiegraphError, match="already registered"):
        emod.use_kernels(ag.kernel_lib(kdir, quiet=True))


def _gates(src, layer, calib):
    raw = {"kernels": [{"source": str(src), "name": "k", "lint_ok": True, "lint": [],
                        "selected": [layer], "rejected": {},
                        "layers": {str(layer): {"calls": [0], "calibration": calib}}}],
           "report": None}
    return CheckReport("offload_check", raw, {Path(src): [layer]})


def test_bit_exact_needs_every_call(tmp_path):
    src = tmp_path / "k.cc"
    exact = RegionCheck("k", [DS], {DS.name: []}, _gates(src, 9, ["call 3: BIT-EXACT"]))
    off = RegionCheck("k", [DS], {DS.name: []}, _gates(src, 9, ["call 3: BIT-EXACT",
                                                               "call 5: MISMATCH 2"]))
    none = RegionCheck("k", [DS], {DS.name: []}, _gates(src, 9, []))
    assert exact.bit_exact and not off.bit_exact and not none.bit_exact


def test_check_offload_records_numerics_for_an_added_kernel(kdir, tmp_path, monkeypatch):
    emod = _module(tmp_path, [DS])
    lib = ag.kernel_lib(kdir, quiet=True)
    ported = tmp_path / "conv1x1_l9.cc"
    ported.write_text(_spec_src("conv1x1_l9", "[COC,CIC,1,1,4,4]", "[1,COC,HO,WO,4]"))
    lib.add("conv1x1_l9", ported)
    emod.use_kernels(lib)
    monkeypatch.setattr(emod, "offload_check",
                        lambda layers, src, verbose=False: _gates(src, 9, ["call 3: BIT-EXACT"]))
    assert emod.check_offload(region=DS.name, kernel="conv1x1_l9").ok
    assert lib["conv1x1_l9"].verified
    assert emod.match_kernel(DS.name, reference_only=True).name == "conv1x1_l9"
