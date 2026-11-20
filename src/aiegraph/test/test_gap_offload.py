###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-aiegraph-offload-7c4415: aiegraph plans a conv as ONE launch per layer with K
accumulated in the core (``mode="kincore"``), not one launch per 64x64x64 tile.

    PYTHONPATH=src python3 -m pytest -q src/aiegraph/test/test_gap_offload.py

x86 only. The UNet tests need onnx + the bk-sdm-tiny export and skip without them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.partition import plan_partition_launches  # noqa: E402
from aiegraph.plan import kincore_geom, plan_conv, plan_conv_layers  # noqa: E402

UNET_DIR = REPO / "example" / "aiegraph" / "unet_bk_sdm_tiny"
PARTITION = UNET_DIR / "partition.json"
UNET_ONNX = Path("/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx")
GEOM_KEYS = ("ntile", "S", "tkw", "kch", "nkc", "kp", "tr", "arows", "brows", "l1")


def _shapes():
    d = json.loads(PARTITION.read_text())
    return sorted({tuple(x) for l in d["layers"] for x in l["mkn"]})


def _example_kacc_geom():
    pytest.importorskip("numpy")
    # unet-decl P4d: the example's unet_aie.py (whose kacc_geom delegated here) is gone; d5e9f15 has it
    from aiegraph.kaccbench import kacc_geom
    return kacc_geom


def test_tile_mode_counts_every_tile():
    p = plan_conv(1024, 2880, 320, mode="tile")
    assert p["mode"] == "tile" and p["launches"] == 16 * 5 * 45 == p["tile_launches"]


def test_kincore_one_launch_per_layer():
    p = plan_conv(1024, 2880, 320, mode="kincore")
    assert p["mode"] == "kincore" and p["launches"] == 1 and p["tile_launches"] == 3600
    g = p["geom"]
    assert g["nkc"] * g["tkw"] * 2 == g["kp"] == 2880   # the whole K walked in the core


def test_kincore_transposed_when_port_buffer_too_small():
    g = kincore_geom(320, 2880, 1024)       # px=1024: c block {16,256} fp32 does not fit
    assert g["legal"] and g["tr"] == 1 and g["arows"] == 1024 and g["brows"] == 320
    assert g["l1"] <= 29696


def test_kincore_pads_square_w_port():
    g = kincore_geom(1280, 2560, 64)        # ch == k/2 words -> one zero chunk
    assert g["legal"] and g["kp"] == 2560 + 2 * g["tkw"]


def test_kincore_falls_back_to_tile_when_illegal():
    p = plan_conv(100, 2880, 320, mode="kincore")
    assert p["mode"] == "tile" and p["launches"] == 2 * 5 * 45 and "px%64" in p["why"]


def test_bad_mode():
    with pytest.raises(AiegraphError):
        plan_conv(64, 64, 64, mode="fast")


@pytest.mark.skipif(not PARTITION.is_file(), reason="no unet partition.json")
def test_kincore_geom_matches_example_kacc_geom():
    """the planner IS the board-proven walk of aiegraph.kaccbench.kacc_geom (the bench uses it)"""
    kacc_geom = _example_kacc_geom()
    shapes = _shapes()
    assert len(shapes) >= 10
    for m, k, n in shapes:
        mine, ref = kincore_geom(n, k, m), kacc_geom(n, k, m)
        assert mine["legal"], (m, k, n, mine["why"])
        assert {x: mine[x] for x in GEOM_KEYS} == {x: ref[x] for x in GEOM_KEYS}, (m, k, n)


def test_plan_conv_layers_on_small_onnx(tmp_path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.zeros((128, 64, 3, 3), np.float32), "w")
    node = helper.make_node("Conv", ["x", "w"], ["y"], name="c0", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 16, 16])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], [w])
    path = tmp_path / "c.onnx"
    onnx.save(helper.make_model(g), str(path))
    (r,) = plan_conv_layers(path, mode="kincore")   # output shape via shape inference
    assert (r["m"], r["k"], r["n"], r["launches"], r["tile_launches"]) == (256, 576, 128, 1, 4 * 2 * 9)
    part = {"layers": [{"node": "c0", "mkn": [[256, 576, 128]]}]}
    assert plan_partition_launches(path, part)["launches_planned"] == 1
    with pytest.raises(AiegraphError):
        plan_partition_launches(path, {"layers": [{"node": "c0", "mkn": [[256, 576, 64]]}]})


@pytest.mark.skipif(not (UNET_ONNX.is_file() and PARTITION.is_file()), reason="no bk-sdm-tiny export")
def test_unet_partition_kincore_37_launches():
    pytest.importorskip("onnx")
    doc = json.loads(PARTITION.read_text())
    r = plan_partition_launches(UNET_ONNX, PARTITION, "kincore")
    assert len(r["layers"]) == len(doc["layers"]) == 37
    assert r["fallbacks"] == [] and r["launches_planned"] == 37
    assert r["tile_launches"] == doc["N"] == sum(l["launches"] for l in doc["layers"])
    t = plan_partition_launches(UNET_ONNX, PARTITION, "tile")
    assert t["launches_planned"] == doc["N"]
