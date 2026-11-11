###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Routing IR API (routing_ir.py over _aiebackend.routing_ir_*).

Fixtures are the routing IRs aiehlc dumps (ir/dfschedule/0_initial.mlir) for
example/tileprogram/ccode/simplegemm_kernel_port.cc and simpleconv2dkernel_port.cc.
Needs the built _aiebackend (make _aiebackend); skipped otherwise.

    PYTHONPATH=src/mlir/mlirfront/frontend python3 -m pytest -q \
        src/mlir/mlirfront/frontend/aiebackend/test/test_routing_ir.py
"""
from pathlib import Path

import pytest

import aiebackend

try:
    aiebackend.load()
except Exception as e:  # not built in this tree
    pytest.skip(f"_aiebackend not available: {e}", allow_module_level=True)

from aiebackend.routing_ir import RoutingIR  # noqa: E402

DATA = Path(__file__).parent / "data"
GEMM = DATA / "gemm_64x64x256_routing_ir.mlir"
CONV = DATA / "conv_stem_routing_ir.mlir"


def test_view_gemm():
    ir = RoutingIR.load(GEMM)
    assert ir.mesh()["rows"] == 4 and ir.mesh()["cols"] == 4
    t = {x["arg"]: x for x in ir.tensors()}
    assert t[0]["shape"] == [64, 256] and t[0]["direction"] == "input"
    assert t[2]["shape"] == [64, 64] and t[2]["direction"] == "output"
    a = ir.partition(0)
    assert a["partition"]["hwAxisOwner"] == "row" and a["partition"]["splitnum"] == 4
    k = a["tiling"][1]["levels"]
    assert [(l["slice"], l["rounds"]) for l in k] == [(256, 1), (64, 4)]  # K: 4 chunks of 64
    assert ir.view["module"]["routing.fullconnect_auto"] == 1


def test_view_conv_halo():
    ir = RoutingIR.load(CONV)
    ifm = ir.tensor(0)
    assert "halo" in ifm["attrs"] and "slice = 61" in ifm["attrs"]["halo"]
    assert ir.view["func"]["attrs"]["routing.kernel_tiling"].startswith("#routing.kernel_tiling<")
    ofm = ir.partition(2)
    assert [d["axis"] for d in ofm["tiling"]] == [1, 0, 2]  # mesh row, on-core, mesh col


@pytest.mark.parametrize("path", [GEMM, CONV], ids=["gemm", "conv"])
def test_known_good_irs_lint_clean(path):
    assert RoutingIR.load(path).lint() == []


def test_edit_is_one_line_and_recorded():
    base = RoutingIR.load(GEMM)
    ir = RoutingIR(base.text).set_pp_depth(2, 1)
    assert ir.tensor(2)["attrs"]["pp_depth"] == 1
    diff = [l for l in ir.diff(base).splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert len(diff) == 2  # one line out, one line in
    assert ir.history == [{"target": "tensor", "arg": 2, "attr": "pp_depth", "value": "1 : i64"}]
    assert RoutingIR(ir.text).view == ir.view  # the edited text parses back to the same view


def test_set_level_nested():
    ir = RoutingIR.load(GEMM).set_level(0, dim=1, depth=1, total=256, slice=128, step=128, rounds=2)
    assert ir.partition(0)["tiling"][1]["levels"][1] == {"base": 256, "total": 256, "slice": 128, "step": 128,
                                                         "rounds": 2}
    assert ir.partition(0)["tiling"][1]["levels"][0]["slice"] == 256  # outer level untouched


def test_lint_findings():
    base = RoutingIR.load(GEMM).text
    assert any("total 256 != slice 128 x rounds 4" in f for f in
               RoutingIR(base).set_level(0, dim=1, depth=1, slice=128, step=128).lint())
    bad = RoutingIR(base).set_attr(
        "partition", "partition",
        '#routing.partition<splitnum = 3, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", '
        'singleTileOwner = "">', tensor_arg=0)
    f = bad.lint()
    assert any("splitnum 3 != mesh rows 4" in x for x in f)
    assert any("not divisible by splitnum 3" in x for x in f)
    assert any("no effect" in x for x in RoutingIR(base).set_pp_depth(1, 1).lint())


def test_bad_edits_raise():
    ir = RoutingIR.load(GEMM)
    with pytest.raises(RuntimeError, match="unknown level field 'slicex'"):
        ir.set_level(0, dim=0, slicex=3)
    with pytest.raises(RuntimeError, match="cannot parse attribute"):
        ir.set_attr("module", "routing.fullconnect_auto", "#nope<")
    with pytest.raises(RuntimeError, match="no partitiontensor for tensor arg 7"):
        ir.set_level(7, dim=0, slice=1)
    assert ir.history == []  # failed edits leave the IR and its history alone


def test_check_runs_the_pipeline(tmp_path):
    ir = RoutingIR.load(GEMM).set_level(0, dim=1, depth=1, total=256, slice=128, step=128, rounds=2)
    r = ir.check(work_dir=str(tmp_path))
    assert r.ok and r.verified and not r.errors
    assert r.buffers["buf_in_ping_0"] == 2048  # the K chunk edit reaches the core buffers
    assert r.core_free_bytes > 0
    assert (Path(r.artifacts) / "host.cc").is_file()
