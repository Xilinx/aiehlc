#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Mechanical LLVM 19 -> 24 API rewrites, driven by a ninja log (skill llvmupgrade).

    python3 script/llvm24_codemod.py build_llvm24/ninja.log [--dry-run]

Only lines the compiler flagged are touched:
  x.cast<T>() / dyn_cast / isa / dyn_cast_or_null   -> llvm::cast<T>(x) ...
  <op value>.dump()   ("has no member named 'dump'")  -> <op value>->dump()
and, file-wide in files with errors, the 1:1 renames below.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASTS = ("dyn_cast_or_null", "dyn_cast", "cast", "isa")
RENAMES = [
    (r"\bbufferization::ToMemrefOp\b", "bufferization::ToBufferOp"),
    (r"\bapplyPatternsAndFoldGreedily\b", "applyPatternsGreedily"),
    (r"\b(\w+)\.maxIterations\s*=\s*([^;]+);", r"\1.setMaxIterations(\2);"),
]
_OPEN, _CLOSE = "([", ")]"


def _expr_start(s: str, i: int) -> int:
    """start of the receiver expression that ends right before s[i] (the '.')"""
    j = i - 1
    while j >= 0:
        c = s[j]
        if c in _CLOSE:
            depth, k = 0, j
            while k >= 0:
                if s[k] in _CLOSE:
                    depth += 1
                elif s[k] in _OPEN:
                    depth -= 1
                    if depth == 0:
                        break
                k -= 1
            j = k - 1
            continue
        if c.isalnum() or c in "_:.":
            j -= 1
            continue
        if c == ">" and j > 0 and s[j - 1] == "-":          # ->
            j -= 2
            continue
        if c == "*" and j > 0 and s[j - 1] == "(":           # (*it)
            j -= 1
            continue
        break
    return j + 1


def _template_end(s: str, k: int) -> int:
    """index after the '>' closing the '<' at s[k]"""
    depth = 0
    while k < len(s):
        if s[k] == "<":
            depth += 1
        elif s[k] == ">":
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return -1


def fix_casts(line: str) -> str:
    pat = re.compile(r"\.(" + "|".join(CASTS) + r")<")
    while True:
        m = pat.search(line)
        if not m:
            return line
        dot, name = m.start(), m.group(1)
        t_end = _template_end(line, m.end() - 1)
        if t_end < 0 or not line[t_end:].startswith("()"):
            return line                                     # not the member form we know
        start = _expr_start(line, dot)
        recv, targs = line[start:dot], line[m.end() - 1:t_end]
        if not recv:
            return line
        line = line[:start] + f"llvm::{name}{targs}({recv})" + line[t_end + 2:]


def main(argv) -> int:
    log, dry = Path(argv[1]), "--dry-run" in argv
    err = re.compile(r"^(/\S+?):(\d+):\d+: error: (.*)$")
    lines = defaultdict(set)
    dumps = defaultdict(set)
    for raw in log.read_text(errors="replace").splitlines():
        m = err.match(raw)
        if not m or not m.group(1).startswith(str(ROOT)) or "/inc/" in m.group(1):
            continue
        lines[m.group(1)].add(int(m.group(2)))
        if "has no member named ‘dump’" in m.group(3):
            dumps[m.group(1)].add(int(m.group(2)))
    changed = 0
    for path, nums in sorted(lines.items()):
        p = Path(path)
        src = p.read_text().split("\n")
        before = list(src)
        for n in nums:
            s = fix_casts(src[n - 1])
            if n in dumps[path]:
                s = re.sub(r"(\b[\w\]\)]+)\.dump\(\)", r"\1->dump()", s)
            src[n - 1] = s
        text = "\n".join(src)
        for a, b in RENAMES:
            text = re.sub(a, b, text)
        if text != "\n".join(before):
            changed += 1
            diff = sum(1 for x, y in zip(before, text.split("\n")) if x != y)
            print(f"{'would fix' if dry else 'fixed'} {p.relative_to(ROOT)}: {diff} lines")
            if not dry:
                p.write_text(text)
    print(f"{changed} files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
