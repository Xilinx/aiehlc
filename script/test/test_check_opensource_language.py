###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""script/ci/check_opensource.py language check: published text must be English."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "check_opensource.py"
_spec = importlib.util.spec_from_file_location("check_opensource", SCRIPT)
assert _spec is not None and _spec.loader is not None
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)


@pytest.mark.parametrize("text", [
    "\u4e2d\u6587\u6ce8\u91ca",                    # CJK ideographs
    "11/17\uff1aPASS",                 # fullwidth colon (U+FF1A)
    "\u3053\u308c\u306f\u30c6\u30b9\u30c8",                # kana
    "\ud14c\uc2a4\ud2b8",                      # Hangul
    "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430",                    # Cyrillic
    "a\uff0cb",                        # fullwidth comma
    "\u300cquoted\u300d",                  # CJK brackets (U+300C/D)
])
def test_non_english_is_flagged(text):
    assert any(co._non_english(c) for c in text), text


@pytest.mark.parametrize("text", [
    "plain ASCII: x = a * b; // ok",
    "latency 372 \u00b5s, A\u1d40 B, x\u00b2, 10\u00b3",   # micro sign, modifier T, superscripts
    "\u03b1 + \u03a3 w, \u2192 \u2264 \u00d7 \u2212 \u2014 \u00b0",  # Greek, arrows, math, dash
    "na\u00efve caf\u00e9 Jos\u00e9",                 # accented Latin passes (cannot be told apart)
    "\u2139 note, \u2122",                            # letterlike symbols U+2100-214F
    "done \u2705 \U0001F680",                         # emoji are symbols, not a language
])
def test_english_technical_text_passes(text):
    assert not any(co._non_english(c) for c in text), [hex(ord(c)) for c in text if co._non_english(c)]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "REPO_ROOT", tmp_path)
    return tmp_path


def test_check_language_files_lines_and_skips(repo, monkeypatch):
    (repo / "doc").mkdir()
    (repo / "doc" / "a.md").write_text("English line\n\u7b2c\u4e8c\u884c\u662f\u4e2d\u6587\nok\n", encoding="utf-8")
    (repo / "doc" / "b.cc").write_text("// all English \u2192 fine\n", encoding="utf-8")
    (repo / "thirdparty").mkdir()
    (repo / "thirdparty" / "v.c").write_text("/* \u4f9b\u5e94\u5546\u4ee3\u7801 */\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\0\x01\xe4\xb8\xad")
    (repo / "i18n.txt").write_text("\u4f60\u597d\n", encoding="utf-8")
    (repo / "link.md").symlink_to(repo / "doc" / "a.md")
    monkeypatch.setattr(co, "LANG_ALLOW_GLOBS", ("i18n.txt",))
    errs, bad = co.check_language(["doc/a.md", "doc/b.cc", "thirdparty/v.c", "blob.bin", "i18n.txt", "link.md"])
    assert bad == ["doc/a.md"]                      # vendor, binary, allow-listed and symlink skipped
    assert len(errs) == 1 and errs[0].startswith("doc/a.md: line 2 col 1: non-English text")
    assert "CJK UNIFIED IDEOGRAPH" in errs[0]


def test_check_language_caps_lines_per_file(repo):
    (repo / "x.md").write_text("\u4e2d\n" * (co.LANG_MAX_PER_FILE + 3), encoding="utf-8")
    errs, bad = co.check_language(["x.md"])
    assert bad == ["x.md"] and len(errs) == co.LANG_MAX_PER_FILE + 1
    assert errs[-1] == "x.md: ... 3 more line(s) with non-English text"


def test_report_has_language_section():
    text = co.format_report(scope="s", errors=["a.md: line 1 col 1: non-English text \"\u4e2d\" (U+4E2D X)"],
                            missing_files=[], lang_files=["a.md"])
    assert "files with non-English text: 1" in text
    assert "Files with non-English text:\na.md" in text
    assert "must be English" in text and "Expected header" not in text
