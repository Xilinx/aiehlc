###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""x86 unit tests: aiegraph driver template (unet-m3 S6) -- placement at checked anchors + refusals."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph import driver_template as dt  # noqa: E402

DRV = (
    "int aiednn_aie_conv(int cid, void *args, int nargs);   /* aiegraph group step */\n"
    "static uint8_t g_wscratch[64];\n"
    + dt.EXPAND_HEAD +
    "    uint32_t *d = (uint32_t *)dst;\n"
    + dt.EXPAND_LOOP +
    "static uint8_t kernel_ws[1024];\n"
    + dt.RUN_HEAD +
    "    aiednn_aie_conv(0, args, 3);\n"
    "    return 0;\n}\n"
)
PARTS = dict(decls="/*D*/\nint k(void);\n", expand_enter="    E0();\n", expand_exit="    E1();\n",
             run_enter="    R();\n")


def test_template_placed_at_anchors():
    t = dt.apply_text(DRV, **PARTS)
    assert t.index("int k(void);") < t.index("static uint8_t kernel_ws[")
    assert t.index("int aiednn_aie_conv(") < t.index("int k(void);")
    body = t.split(dt.EXPAND_HEAD)[1].split("}\n")[0]
    assert body.index("E0();") < body.index("for (uint64_t") < body.index("E1();")
    assert t.split(dt.RUN_HEAD)[1].startswith("    R();\n")
    assert dt.apply_text(DRV) == DRV   # empty template = identity


def test_template_applied_twice_raises():
    t = dt.apply_text(DRV, **PARTS)
    with pytest.raises(dt.DriverTemplateError, match="applied twice"):
        dt.apply_text(t, **PARTS)


@pytest.mark.parametrize("cut,part", [
    ("static uint8_t kernel_ws[1024];\n", "decls"),
    (dt.EXPAND_LOOP, "expand_enter"),
    (dt.RUN_HEAD, "run_enter"),
])
def test_template_missing_anchor_raises(cut, part):
    with pytest.raises(dt.DriverTemplateError):
        dt.apply_text(DRV.replace(cut, ""), **{part: PARTS[part]})


def test_template_duplicate_anchor_raises():
    with pytest.raises(dt.DriverTemplateError, match="2 times"):
        dt.apply_text(DRV + "static uint8_t kernel_ws[8];\n", decls="int k(void);\n")


def test_template_pass_rewrites_file(tmp_path):
    f = tmp_path / "graph_driver.c"
    f.write_text(DRV)
    p = dt.DriverTemplatePass(**PARTS)
    rep = p(f)
    assert f.read_text() == dt.apply_text(DRV, **PARTS)
    assert rep["decls"] == len(PARTS["decls"])
