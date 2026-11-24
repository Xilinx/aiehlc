###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.dnn_export: torch.export programs through the model compiler (flow 2's export path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "example" / "aiednn" / "models"))
sys.path.insert(0, str(REPO / "example" / "aiednn" / "flows"))

import zoo  # noqa: E402
from model import model_and_input  # noqa: E402

from aiegraph import dnn_export as de  # noqa: E402
from aiegraph import dnn_torch as dt  # noqa: E402
from aiegraph.dnn_ctypes import AiednnLib  # noqa: E402


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("lib"))


@pytest.mark.parametrize("name", ["tiny", "resnet18", "gpt2_small", "tiny_llama"])
def test_to_fx_matches_eager(name):
    """the translated graph is the exported program: identical output, parameters rebuilt as modules"""
    m, x = model_and_input(name) if name == "tiny" else zoo.build(name, pretrained=False)
    gm = de.to_fx(torch.export.export(de.prepare_for_export(m), (x,)), [x])
    with torch.no_grad():
        assert torch.equal(gm(x), m(x))
    assert any(isinstance(c, (torch.nn.Conv2d, torch.nn.Linear)) for c in gm.children())
    assert not any("aten" in str(n.target) for n in gm.graph.nodes)


def test_saved_program_translates_the_same(tmp_path):
    """torch.export.load spells max_pool2d's stride / padding (and mean's keepdim) as keywords: the arguments are
    normalized against the op schema, or the reloaded resnet18 pools with the wrong stride"""
    m, x = zoo.build("resnet18", pretrained=False)
    ep = torch.export.export(m.eval(), (x,))
    torch.export.save(ep, tmp_path / "r18.pt2")
    gm = de.to_fx(torch.export.load(tmp_path / "r18.pt2"), [x])
    with torch.no_grad():
        assert torch.equal(gm(x), m(x))


def test_exported_llm_folds_like_the_direct_path(lib):
    """export flattens every Linear on a 3-D input (view, addmm, view) and writes GQA's K / V repeat as
    unsqueeze / expand / view: both are folded back, so the exported tiny_llama compiles to the direct path's
    primitives (int8, MLIR pipeline, every GEMM exact)"""
    m, x = zoo.build("tiny_llama", pretrained=False)
    gm = de.to_fx(torch.export.export(de.prepare_for_export(m), (x,)), [x])
    r = dt.model_test(gm, x, "int8", lib, "tiny_llama", mlir=True)
    d = dt.model_test(m, x, "int8", lib, "tiny_llama", mlir=True)
    assert r.ok and not r.gemm_bad and r.ops == d.ops
    assert np.isclose(r.relerr, d.relerr)


def test_compile_exported_builds_and_runs_the_c_app(tmp_path):
    """compile_exported: the exported program -> C app -> x86 build, run, PASS against PyTorch"""
    m, x = model_and_input("tiny")
    rep, _, _ = de.compile_exported(torch.export.export(m.eval(), (x,)), x, name="tiny", out=tmp_path)
    assert rep.ok and rep.run_ok is None                 # compiled, not run
    de.run(rep)
    assert rep.run_ok, rep
    assert "PASS" in rep.run_log.read_text()


def test_compile_exported_session_api(tmp_path):
    """emit=model: session header/cc (create/run/destroy) + smoke host; x86 PASS vs PyTorch"""
    m, x = model_and_input("tiny")
    out = tmp_path / "session"
    rep, g, _ = de.compile_exported(torch.export.export(m.eval(), (x,)), x, name="tiny", out=out,
                                    emit="model", mlir=False)
    assert (out / "tiny_model.h").is_file() and (out / "tiny_model.cc").is_file()
    hdr = (out / "tiny_model.h").read_text()
    assert "tiny_create" in hdr and "tiny_run" in hdr and "tiny_destroy" in hdr
    assert "tiny_input_desc" in hdr and "tiny_output_desc" in hdr
    assert "int main(" not in (out / "tiny_model.cc").read_text()
    assert rep.ok and rep.run_ok is None
    de.run(rep)
    assert rep.run_ok, rep
    log = rep.run_log.read_text()
    assert "PASS" in log and "session=1" in log
