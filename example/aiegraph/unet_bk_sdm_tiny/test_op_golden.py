###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-SG x86 per-op golden regression for the bk-sdm-tiny UNet (sprint unet_m3-s1).

Wraps the model-zoo gate zoo/op_golden/compare.py (cos >= 0.999 per class) over two golden sets of
step 0 / cond:
  * p7_attn_fused_s0_cond      : every fused-attention class (SDPA self/cross, h=8, d=40/80/160, + the hard variant)
  * p5_groupnorm_fused_s0_cond : every GroupNorm(+SiLU) class (C=320..2560, eps 1e-5/1e-6)

compare.py --dir only counts the files present in the output dir (1/1 is a PASS), so this wrapper asserts
"PASS total N/N classes" with N = the number of golden classes (dirs with y.npy) of the set.

OPG_OUT unset -> golden self-compare smoke (y.npy of each class as the "AIE output"), plus bf16-rounded outputs
                 (must PASS: the AIE precision) and negative controls (one corrupted class must FAIL, a missing class
                 must not reach N/N), so the gate itself is proven to catch attention/GN numeric bugs before the board.
OPG_OUT=<dir>  -> real check: <dir>/<set>/<cls>.npy|.bin for every class of both sets (OPG_DTYPE f32|bf16 for .bin).
Overrides: OPG_COMPARE (compare.py), OPG_GOLDEN (golden root), OPG_COS (default 0.999).
Run: python3 -m pytest -q example/aiegraph/unet_bk_sdm_tiny/test_op_golden.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

COMPARE = Path(os.environ.get("OPG_COMPARE", "/scratch/staff/huaj/aiengineers/shuhua/zoo/op_golden/compare.py"))
GOLDEN = Path(os.environ.get("OPG_GOLDEN", "/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/op_golden"))
OUT = os.environ.get("OPG_OUT")
DTYPE = os.environ.get("OPG_DTYPE", "f32")
COS = os.environ.get("OPG_COS", "0.999")
SETS = {"p7_attn_fused_s0_cond": 7, "p5_groupnorm_fused_s0_cond": 14}  # expected golden class counts


def classes(s):
    return sorted(d.name for d in (GOLDEN / s).iterdir() if (d / "y.npy").is_file())


def run_dir(s, d, dtype="f32"):
    r = subprocess.run([sys.executable, str(COMPARE), str(GOLDEN / s), "--dir", str(d), "--dtype", dtype, "--cos", COS],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def assert_full_pass(s, d, dtype="f32"):
    n = len(classes(s))
    rc, out = run_dir(s, d, dtype)
    print(out)
    assert rc == 0, out
    assert re.search(rf"^PASS total {n}/{n} classes$", out, re.M), f"want PASS total {n}/{n}:\n{out}"
    assert len(re.findall(r"^PASS cls=", out, re.M)) == n, out


@pytest.mark.parametrize("s", sorted(SETS))
def test_golden_set_complete(s):
    assert COMPARE.is_file(), COMPARE
    cl = classes(s)
    assert len(cl) == SETS[s], f"{s}: {len(cl)} classes {cl}"


@pytest.mark.parametrize("s", sorted(SETS))
def test_op_golden(s, tmp_path):
    if OUT:
        assert_full_pass(s, Path(OUT) / s, DTYPE)
        return
    for c in classes(s):  # smoke: golden vs itself
        (tmp_path / f"{c}.npy").symlink_to(GOLDEN / s / c / "y.npy")
    assert_full_pass(s, tmp_path)


@pytest.mark.skipif(bool(OUT), reason="smoke-only controls")
@pytest.mark.parametrize("s", sorted(SETS))
def test_bf16_output_passes(s, tmp_path):
    """bf16 .bin outputs (upper 16 bits, round-to-nearest-even) = AIE output precision -> must still PASS N/N."""
    for c in classes(s):
        u = np.load(GOLDEN / s / c / "y.npy").astype(np.float32).view(np.uint32)
        b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
        b.tofile(tmp_path / f"{c}.bin")
    assert_full_pass(s, tmp_path, "bf16")


@pytest.mark.skipif(bool(OUT), reason="smoke-only controls")
@pytest.mark.parametrize("s", sorted(SETS))
def test_negative_controls(s, tmp_path):
    cl = classes(s)
    n = len(cl)
    bad = cl[0]
    for c in cl:
        y = np.load(GOLDEN / s / c / "y.npy").astype(np.float32)
        if c == bad:  # a layout bug: last axis reversed (wrong head/channel order) -> numerically wrong
            y = np.ascontiguousarray(y[..., ::-1])
        np.save(tmp_path / f"{c}.npy", y)
    rc, out = run_dir(s, tmp_path)
    assert rc != 0 and re.search(rf"^FAIL cls={re.escape(bad)} ", out, re.M), out
    assert re.search(rf"^FAIL total {n - 1}/{n} classes$", out, re.M), out
    # missing class: compare.py alone says PASS (n-1)/(n-1); the wrapper must reject it
    (tmp_path / f"{bad}.npy").unlink()
    rc, out = run_dir(s, tmp_path)
    assert rc == 0 and f"PASS total {n - 1}/{n - 1}" in out, out
    with pytest.raises(AssertionError):
        assert_full_pass(s, tmp_path)
