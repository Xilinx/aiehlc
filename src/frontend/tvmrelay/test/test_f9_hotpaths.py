###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""F9 (-fast phase profile): uint32 f32_to_bf16_rne and the regex brace scanner
must give exactly what the old code gave."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frontend.tvmrelay.arm_build import f32_to_bf16_rne
from frontend.tvmrelay.split_layers import _match_brace, function_spans, iter_c_functions


def _ref_bf16(a):
    u = np.ascontiguousarray(a, dtype="<f4").view("<u4").astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(a, dtype="<f4"))
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r


def test_bf16_rne_bit_exact():
    sp = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800001,
          0x7FC00000, 0xFFFFFFFF, 0x7FFFFFFF, 0x3F808000, 0x3F818000, 0x3F807FFF, 0x8000, 0x18000]
    rnd = np.random.default_rng(0).integers(0, 2**32, 200_000, dtype=np.uint64).astype(np.uint32)
    u = np.concatenate([np.array(sp, np.uint32), rnd])
    keep = u.copy()
    with np.errstate(all="ignore"):
        assert np.array_equal(f32_to_bf16_rne(u.view(np.float32)), _ref_bf16(u.view(np.float32)))
    assert np.array_equal(u, keep)          # input not modified


def test_match_brace_skips_literals_and_comments():
    s = '{ a("}{"); b = \'}\'; // }\n /* } */ c("\\"}"); { } }tail'
    assert s[_match_brace(s, 0):] == "tail"


def test_function_spans_first_wins():
    src = ('#ifdef __cplusplus\nextern "C"\n#endif\nTVM_DLL int32_t f(void* a) { return "}"[0]; }\n'
           '#ifdef __cplusplus\nextern "C"\n#endif\nTVM_DLL int32_t g(void* a) { { } return 0; }\n')
    spans = function_spans(src)
    assert {n: (s, e) for n, s, e in iter_c_functions(src)} == spans
    assert src[spans["g"][0]:spans["g"][1]].rstrip().endswith("return 0; }")
