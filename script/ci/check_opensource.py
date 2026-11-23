#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
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


def format_report(
    *,
    scope: str,
    errors: list[str],
    missing_files: list[str],
) -> str:
    lines = [
        "AIEHLC open-source check FAILED",
        f"scope: {scope}",
        f"violations: {len(errors)}",
        f"files missing license info: {len(missing_files)}",
        "",
        "Files missing license info:",
    ]
    if missing_files:
        lines.extend(missing_files)
    else:
        lines.append("(none — see violation details below)")
    lines.extend(["", "Violation details:"])
    lines.extend(f"  - {e}" for e in errors)
    lines.extend(
        [
            "",
            "Expected header (comment syntax as appropriate):",
            "  Copyright (C) <YEAR> Advanced Micro Devices, Inc. All Rights Reserved.",
            "  SPDX-License-Identifier: Apache-2.0",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    scope: str,
    errors: list[str],
    missing_files: list[str],
) -> str:
    text = format_report(
        scope=scope, errors=errors, missing_files=missing_files
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
        detail = "; ".join(by_file.get(rel, ["missing license info"]))
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

    report_path = Path(args.report)

    if errors:
        report_text = write_report(
            report_path,
            scope=scope,
            errors=errors,
            missing_files=missing_files,
        )
        write_step_summary(report_text)
        emit_annotations(missing_files, errors)
        print(f"\nFAIL: {len(errors)} open-source violation(s):", flush=True)
        print(
            f"Files missing license info ({len(missing_files)}):",
            flush=True,
        )
        for rel in missing_files:
            print(f"  {rel}", flush=True)
        print("\nViolation details:", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        print(
            "\nExpected header (comment syntax as appropriate):\n"
            "  Copyright (C) <YEAR> Advanced Micro Devices, Inc. All Rights Reserved.\n"
            "  SPDX-License-Identifier: Apache-2.0",
            flush=True,
        )
        print(f"\nReport written to {report_path}", flush=True)
        return 1

    if report_path.exists():
        report_path.unlink()
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Open-source check passed\n\nNo license-header violations.\n")
    print("PASS: open-source checks clean", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
