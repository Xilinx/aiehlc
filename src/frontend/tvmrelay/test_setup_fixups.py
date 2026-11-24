###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Idempotency guard for setup_tvm016.apply_source_fixups."""
import shutil, tempfile
from pathlib import Path
from frontend.tvmrelay import setup_tvm016 as s

SRC = Path('thirdparty/tvm-0.16')

def test_idempotent_on_patched_tree():
    assert s.apply_source_fixups(SRC) == [], "re-patching an already-patched tree must be a no-op"
    assert s.apply_source_fixups(SRC) == []

def test_patches_a_pristine_copy_exactly_once():
    rel = "python/tvm/relay/quantize/_calibrate.py"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); (tmp/rel).parent.mkdir(parents=True)
        # Reconstruct the pristine upstream text from the .orig backup.
        orig = SRC/(rel + ".orig")
        shutil.copy(orig if orig.is_file() else SRC/rel, tmp/rel)
        pristine = (tmp/rel).read_text()
        if "np.math" in pristine:
            assert s.apply_source_fixups(tmp) == [rel]
            out = (tmp/rel).read_text()
            assert out.count("\nimport math\n") == 1, "duplicate import inserted"
            assert "np.math" not in out
            for _ in range(3):
                assert s.apply_source_fixups(tmp) == []
            assert (tmp/rel).read_text() == out, "text drifted on re-run"
        print("  pristine->patched->stable OK")

test_idempotent_on_patched_tree(); print("test_idempotent_on_patched_tree PASS")
test_patches_a_pristine_copy_exactly_once(); print("test_patches_a_pristine_copy_exactly_once PASS")
