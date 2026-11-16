###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Model-flow HTML (``aiegraph.dnn_viz``) for lowered aiednn graphs, on x86."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "example" / "aiednn" / "models"))

import zoo  # noqa: E402

from aiegraph import dnn_torch as dt  # noqa: E402
from aiegraph import dnn_viz  # noqa: E402
from aiegraph.dnn_ctypes import AiednnLib  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
ITEM = {"f32": 4, "s32": 4, "s8": 1, "u8": 1}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_viz"))


def _page_data(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    assert m, "embedded data block missing"
    return json.loads(m.group(1).replace("<\\/", "</"))


@pytest.mark.parametrize("mode", ["f32", "int8"])
@pytest.mark.parametrize("name", ["tiny_unet", "mixer_head"])
def test_model_html(lib, tmp_path, name, mode):
    m, x = zoo.build(name, pretrained=False)
    g = dt.lower(m, x, mode, name)
    out = dnn_viz.write_model_html(g, lib, x.numpy(), tmp_path / f"{name}_{mode}.html")
    html = out.read_text()
    assert not re.search(r"""(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", html, re.I), "external resource"
    assert "<script src" not in html and "<link" not in html
    d = _page_data(html)
    assert [o["name"] for o in d["ops"]] == [o.name for o in g.ops]
    for o in g.ops:
        assert f"{o.name}" in html
    for t in d["tensors"]:
        assert t["bytes"] == int(np.prod(t["shape"])) * ITEM[t["dt"]], t
    for rec, op in zip(d["ops"], g.ops):
        exp_in = sum(d["tensors"][t]["bytes"] for a, t in op.exec if a != "DST" and g.tensors[t].kind != "const")
        exp_c = sum(d["tensors"][t]["bytes"] for a, t in op.exec if a != "DST" and g.tensors[t].kind == "const")
        assert rec["in_bytes"] == exp_in and rec["const_bytes"] == exp_c
        assert rec["out_bytes"] == d["tensors"][op.out]["bytes"]
        assert rec["t_ms"] is not None and rec["t_ms"] >= 0
        gm = rec["gemm"]
        if gm and gm["on_aie"]:
            assert rec["place"] == "aie" and gm["launches"] >= gm["gemms"]
            assert gm["dma_in"] == gm["launches"] * (dnn_viz.BLOCK["a_stage"] + dnn_viz.BLOCK["b_stage"])
            assert gm["dma_out"] == gm["launches"] * dnn_viz.BLOCK["c_stage"]
    s = d["summary"]
    assert s["ops"] == len(g.ops) and s["aie_ops"] + s["cpu_ops"] == len(g.ops)
    if mode == "f32":  # f32 reaches the AIE only through the float modes: bf16x3 matmuls, FVEC row / pointwise ops,
        # FPOOL windowed pooling
        assert all(o["place"] != "aie" or any(k in o["impl"] for k in ("bf16x3", "fvec", "fpool")) for o in d["ops"]), \
            [o["impl"] for o in d["ops"] if o["place"] == "aie"]
    else:
        assert s["aie_ops"] > 0
    arena = d["memory"]["arena"]
    for t in d["tensors"]:
        if t["loc"]["space"] == "arena":
            assert 0 <= t["loc"]["off"] and t["loc"]["off"] + t["loc"]["size"] <= arena


def test_board_log_and_routing(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("PROF ref:reorder                                 3 ops      1.50 ms\r\n"
                   "PROF blocked:aie:gemm_s8s32                      1 ops      2.00 ms\n"
                   "aiednn_stats: blocked:aie:gemm_s8s32                1            1\n"
                   "MODEL m mode=int8 ops=4 time_ms=3.5 relerr=1.0e-07 top1=1 ref_top1=1 float_top1=1 "
                   "gemm_checked=1 gemm_bad=0\n===== RESULT: PASS =====\n")
    b = dnn_viz.parse_board_log(log)
    assert b["prof"]["ref:reorder"] == {"n": 3, "ms": 1.5}
    assert b["stats"]["blocked:aie:gemm_s8s32"]["launches"] == 1
    assert b["model"]["ops"] == 4 and b["passed"]
    rc = tmp_path / "routing.cc"
    rc.write_text("""void routing(XAie_DevInst* v1) {
  //round is 0 hw split in : col -----------
  if (v2) {
    int32_t v3 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(2,0), 3);
    int32_t v4 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2,0), SOUTH, 3, NORTH, 0);
    int32_t v5 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(2,1), SOUTH, 0, DMA, 1);
  }
  //round is 0 hw split in : row -----------
  if (v2) {
    int32_t v6 = XAie_EnableShimDmaToAieStrmPort(v1, XAie_TileLoc(6,0), 7);
    int32_t v7 = XAie_StrmConnCctEnable(v1, XAie_TileLoc(6,0), SOUTH, 7, DMA, 0);
    int32_t v8 = XAie_StrmPktSwSlaveSlotEnable(v1, XAie_TileLoc(3,3), DMA, 0, 0, XAie_PacketInit(4, 0), 31, 0, 0);
    int32_t v9 = XAie_EnableAieToShimDmaStrmPort(v1, XAie_TileLoc(3,0), 2);
    int32_t v10 = XAie_StrmPktSwMstrPortEnable(v1, XAie_TileLoc(3,3), SOUTH, 2, XAIE_SS_PKT_DONOT_DROP_HEADER, 0, 1);
  }
}
""")
    r = dnn_viz.parse_routing(rc)
    by = {s["name"]: s for s in r["streams"]}
    assert by["B0"]["shim"] == [2, 0] and by["B0"]["dests"] == [[2, 1]] and by["B0"]["edges"] == [[2, 0, 2, 1]]
    assert by["A0"]["shim"] == [6, 1] and by["C0"]["dir"] == "S2MM" and by["C0"]["edges"] == [[3, 3, 3, 2]]
    dev = dnn_viz.device_geometry()
    assert dev["cols"] == 38 and dev["core_rows"][0] == 3


def test_fused_ops_and_megakernel_view(lib, tmp_path):
    """MLIR-optimized int8 graph: folded source ops, AIE entries and per-mode totals in the page."""
    from aiegraph import dnn_mlir as dm
    if dm.aiednn_opt_path() is None:
        pytest.skip("aiednn-opt not built")
    m, x = zoo.build("mobilenet_v2", pretrained=False)
    g, _, _ = dt.references(m, x, "int8", "mobilenet_v2", mlir=True)
    bdir = tmp_path / "vek280"
    bdir.mkdir()
    cfg = (REPO / "src" / "aietensorop" / "ops" / "aie_modes_cfg.h").read_text()
    from aiegraph.dnn import modes_header
    (bdir / "aie_modes_cfg.h").write_text(modes_header(cfg, {"GEMMQ", "DW", "DWQ"}))
    (bdir / "run.log").write_text("MODEL mobilenet_v2 mode=int8 ops=56 time_ms=330.0 relerr=0.03 top1=1 ref_top1=1\n")
    d = _page_data(dnn_viz.write_model_html(g, lib, x.numpy(), tmp_path / "p.html", board_log=bdir / "run.log").read_text())
    ops = d["ops"]
    assert sum(len(o["fused"]) for o in ops) > 40          # relu6 + quantize per conv, folded by the passes
    assert any(o["quant"] for o in ops) and all(o["opcodes"] for o in ops if o["place"] == "aie")
    mk = d["megakernel"]
    byname = {m_["mode"]: m_ for m_ in mk["modes"]}
    assert byname["GEMMQ"]["opcode"] == "0xF1" and byname["LUT"]["opcode"] == "0x7A"   # from aie_modes_lut.h
    assert byname["GEMMQ"]["compiled"] is True and byname["POOL"]["compiled"] is False
    used = {m_ for o in ops for c in o["opcodes"] for m_ in c["modes"]}
    # STEPS: the network input's f32 -> s8 quantize, FVEC / FPOOL: f32 pooling (on the AIE under the default
    # aiednn_prefer_aie)
    assert used <= {"GEMM", "GEMMQ", "DW", "DWQ", "CONV", "MMT", "STEPS", "FVEC", "FPOOL"} and "GEMMQ" in used and used & {"DW", "DWQ"}
    for m_ in mk["modes"]:                                   # per-mode launches = the ops' calls
        assert m_["launches"] == sum(c["calls"] for o in ops for c in o["opcodes"] if m_["mode"] in c["modes"])
    assert d["summary"]["counted_launches"] == sum(e["calls"] for e in mk["entries"])
