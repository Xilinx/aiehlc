#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HEADER_BYTES = 4096
HEADER_LINES = 40

HEADER_EXTS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".inc",
    ".ll",
    ".mlir",
    ".py",
    ".s",
    ".S",
    ".sh",
    ".bash",
    ".td",
    ".cmake",
}

HEADER_NAMES = {
    "CMakeLists.txt",
    "Makefile",
    "makefile",
}

SKIP_PREFIXES = (
    "thirdparty/",
    ".cursor/",
    ".claude/",
    "ir/",
)

SKIP_EXACT = {
    "LICENSE",
    "NOTICE",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
}

ALLOWED_SPDX = {"Apache-2.0", "MIT"}

# ---- language check: published text must be English ------------------------------------------
# Every tracked text file is scanned (docs, comments, string literals, skills), not only the files
# that need a license header: .cursor/ and .claude/ are published too. Skipped: vendor code and
# generated IR dumps. Binary files (a NUL in the first 8 KB) are not text.
LANG_SKIP_PREFIXES = (
    "thirdparty/",
    "ir/",
)
# Files that must keep non-English text on purpose (e.g. an i18n test fixture): fnmatch globs.
# Keep this empty unless a file genuinely needs it -- and say why next to the entry.
LANG_ALLOW_GLOBS: tuple[str, ...] = ()
LANG_SAMPLE_BYTES = 8192
LANG_MAX_PER_FILE = 5      # offending lines reported per file (the count covers all of them)
# Letters of these scripts are allowed: English is written in Latin; Greek letters are math
# notation (alpha, Sigma). Accented Latin (a name like "Jose" with an accent) cannot be told
# apart from English here and passes.
_LANG_OK_LETTER_PREFIXES = ("LATIN", "GREEK", "MICRO SIGN", "MODIFIER LETTER")

RE_SPDX = re.compile(
    r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)", re.IGNORECASE
)
RE_COPYRIGHT = re.compile(r"Copyright", re.IGNORECASE)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _tracked_files() -> list[str]:
    return [f for f in _git("ls-files").splitlines() if f]


def _changed_files(base: str) -> list[str]:
    out = _git("diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD")
    return [f for f in out.splitlines() if f]


def _default_base() -> str | None:
    for ref in ("origin/main", "origin/master", "main", "master"):
        try:
            _git("rev-parse", "--verify", ref)
            return _git("merge-base", ref, "HEAD")
        except subprocess.CalledProcessError:
            continue
    return None


def _needs_header(path: str) -> bool:
    if path in SKIP_EXACT:
        return False
    if path.startswith(SKIP_PREFIXES):
        return False
    name = os.path.basename(path)
    if name in HEADER_NAMES:
        return True
    ext = os.path.splitext(path)[1]
    return ext in HEADER_EXTS


def _read_header(path: Path) -> str | None:
    with path.open("rb") as fh:
        raw = fh.read(HEADER_BYTES)
    if b"\0" in raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[:HEADER_LINES])


def _non_english(ch: str) -> bool:
    """True for a character of a non-English writing system: a letter outside Latin / Greek
    (CJK, kana, Hangul, Cyrillic, Arabic, ...) or CJK / fullwidth punctuation (U+3000-303F,
    U+FF00-FFEF, e.g. the fullwidth colon). Symbols English technical text uses -- arrows, math
    operators, the micro sign, superscripts, letterlike symbols (U+2100-214F) -- pass."""
    o = ord(ch)
    if o < 0x80:
        return False
    if 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF:
        return True
    if 0x2100 <= o <= 0x214F:
        return False
    if not unicodedata.category(ch).startswith("L"):
        return False
    return not unicodedata.name(ch, "").startswith(_LANG_OK_LETTER_PREFIXES)


def _lang_scanned(path: str) -> bool:
    if path.startswith(LANG_SKIP_PREFIXES):
        return False
    return not any(fnmatch.fnmatch(path, g) for g in LANG_ALLOW_GLOBS)


def _lang_file_errors(rel: str, text: str) -> list[str]:
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        bad = [(col, ch) for col, ch in enumerate(line, 1) if _non_english(ch)]
        if bad:
            hits.append((lineno, bad))
    errors = []
    for lineno, bad in hits[:LANG_MAX_PER_FILE]:
        col, ch = bad[0]
        run = "".join(ch for _, ch in bad)[:40]
        errors.append(
            f"{rel}: line {lineno} col {col}: non-English text \"{run}\" "
            f"(U+{ord(ch):04X} {unicodedata.name(ch, '?')})"
        )
    if len(hits) > LANG_MAX_PER_FILE:
        errors.append(f"{rel}: ... {len(hits) - LANG_MAX_PER_FILE} more line(s) with non-English text")
    return errors


def check_language(files: list[str]) -> tuple[list[str], list[str]]:
    """Every scanned text file must be English (see _non_english). Returns (errors, files)."""
    errors: list[str] = []
    bad_files: list[str] = []
    checked = 0
    for rel in files:
        if not _lang_scanned(rel):
            continue
        path = REPO_ROOT / rel
        if path.is_symlink() or not path.is_file():
            continue            # a symlink's target is scanned under its own name
        try:
            raw = path.read_bytes()
        except OSError as exc:
            errors.append(f"{rel}: cannot read ({exc})")
            bad_files.append(rel)
            continue
        if b"\0" in raw[:LANG_SAMPLE_BYTES]:
            continue
        checked += 1
        errs = _lang_file_errors(rel, raw.decode("utf-8", errors="replace"))
        if errs:
            errors.extend(errs)
            bad_files.append(rel)
    print(f"language: checked {checked} file(s)", flush=True)
    return errors, bad_files


def check_root_license() -> list[str]:
    license_path = REPO_ROOT / "LICENSE"
    if not license_path.is_file():
        return ["LICENSE: missing root LICENSE file"]
    text = license_path.read_text(encoding="utf-8", errors="replace")
    if "Apache License" not in text or "Version 2.0" not in text:
        return ["LICENSE: root LICENSE does not look like Apache License 2.0"]
    return []


def check_license_headers(files: list[str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    missing_files: list[str] = []
    checked = 0
    for rel in files:
        if not _needs_header(rel):
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            errors.append(f"{rel}: tracked path missing from working tree")
            missing_files.append(rel)
            continue
        try:
            header = _read_header(path)
        except OSError as exc:
            errors.append(f"{rel}: cannot read ({exc})")
            missing_files.append(rel)
            continue
        if header is None:
            continue
        checked += 1
        spdx_match = RE_SPDX.search(header)
        has_copyright = bool(RE_COPYRIGHT.search(header))
        problems: list[str] = []
        if not has_copyright:
            problems.append("missing Copyright")
        if not spdx_match:
            problems.append("missing SPDX-License-Identifier")
        elif spdx_match.group(1) not in ALLOWED_SPDX:
            problems.append(
                f"SPDX '{spdx_match.group(1)}' not in allowed set "
                f"{sorted(ALLOWED_SPDX)}"
            )
        if problems:
            errors.append(f"{rel}: {'; '.join(problems)}")
            missing_files.append(rel)
    print(f"license-header: checked {checked} file(s)", flush=True)
    return errors, missing_files


LICENSE_HINT = [
    "Expected header (comment syntax as appropriate):",
    "  Copyright (C) <YEAR> Advanced Micro Devices, Inc. All Rights Reserved.",
    "  SPDX-License-Identifier: Apache-2.0",
]
LANGUAGE_HINT = [
    "Published text must be English: translate the comments / docs / strings above. A file that",
    "genuinely needs another language (e.g. an i18n test fixture) goes into LANG_ALLOW_GLOBS in",
    "script/ci/check_opensource.py, with the reason next to the entry.",
]


def format_report(
    *,
    scope: str,
    errors: list[str],
    missing_files: list[str],
    lang_files: list[str] | None = None,
) -> str:
    lang_files = lang_files or []
    lines = [
        "AIEHLC open-source check FAILED",
        f"scope: {scope}",
        f"violations: {len(errors)}",
        f"files missing license info: {len(missing_files)}",
        f"files with non-English text: {len(lang_files)}",
        "",
        "Files missing license info:",
    ]
    lines.extend(missing_files or ["(none)"])
    lines.extend(["", "Files with non-English text:"])
    lines.extend(lang_files or ["(none)"])
    lines.extend(["", "Violation details:"])
    lines.extend(f"  - {e}" for e in errors)
    lines.append("")
    if missing_files:
        lines.extend(LICENSE_HINT + [""])
    if lang_files:
        lines.extend(LANGUAGE_HINT + [""])
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    scope: str,
    errors: list[str],
    missing_files: list[str],
    lang_files: list[str] | None = None,
) -> str:
    text = format_report(
        scope=scope, errors=errors, missing_files=missing_files, lang_files=lang_files
    )
    path.write_text(text, encoding="utf-8")
    return text


def write_step_summary(report_text: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    md = [
        "## Open-source check failed",
        "",
        "```",
        report_text.rstrip(),
        "```",
        "",
    ]
    with open(summary, "a", encoding="utf-8") as fh:
        fh.write("\n".join(md))


def emit_annotations(missing_files: list[str], errors: list[str]) -> None:
    by_file: dict[str, list[str]] = {}
    for err in errors:
        if ": " not in err:
            continue
        rel, detail = err.split(": ", 1)
        by_file.setdefault(rel, []).append(detail)
    # Cap annotations so huge trees stay readable in the Checks UI.
    limit = 50
    for rel in missing_files[:limit]:
        detail = "; ".join(by_file.get(rel, ["open-source violation"]))
        print(
            f"::error file={rel},title=Open-source check::{detail}",
            flush=True,
        )
    if len(missing_files) > limit:
        print(
            f"::warning title=Open-source check::"
            f"{len(missing_files) - limit} more file(s) omitted from annotations; "
            "see job summary / report artifact",
            flush=True,
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open-source compliance checks for tracked first-party files."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--all",
        action="store_true",
        help="scan every tracked file (default when no --base/--changed)",
    )
    g.add_argument(
        "--changed",
        action="store_true",
        help="scan files changed vs merge-base of origin/main|master",
    )
    g.add_argument(
        "--base",
        metavar="REF",
        help="scan files changed vs git REF (e.g. origin/main)",
    )
    p.add_argument(
        "--report",
        metavar="PATH",
        default="opensource-check-report.txt",
        help="write failure report to PATH (default: opensource-check-report.txt)",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    if args.base:
        files = _changed_files(args.base)
        scope = f"changed vs {args.base}"
    elif args.changed:
        base = _default_base()
        if not base:
            print(
                "ERROR: --changed needs origin/main, origin/master, main, or master",
                file=sys.stderr,
            )
            return 2
        files = _changed_files(base)
        scope = f"changed vs merge-base {base[:12]}"
    else:
        files = _tracked_files()
        scope = "all tracked files"

    print(f"open-source check: {scope} ({len(files)} path(s))", flush=True)

    errors: list[str] = []
    missing_files: list[str] = []
    root_errs = check_root_license()
    errors.extend(root_errs)
    if root_errs:
        missing_files.append("LICENSE")
    hdr_errs, hdr_missing = check_license_headers(files)
    errors.extend(hdr_errs)
    missing_files.extend(hdr_missing)
    missing_files = list(dict.fromkeys(missing_files))
    lang_errs, lang_files = check_language(files)
    errors.extend(lang_errs)

    report_path = Path(args.report)

    if errors:
        report_text = write_report(
            report_path,
            scope=scope,
            errors=errors,
            missing_files=missing_files,
            lang_files=lang_files,
        )
        write_step_summary(report_text)
        emit_annotations(list(dict.fromkeys(missing_files + lang_files)), errors)
        print(f"\nFAIL: {len(errors)} open-source violation(s):", flush=True)
        print(report_text.split("\n", 3)[3], flush=True)
        print(f"Report written to {report_path}", flush=True)
        return 1

    if report_path.exists():
        report_path.unlink()
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Open-source check passed\n\nNo license-header or language violations.\n")
    print("PASS: open-source checks clean", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
