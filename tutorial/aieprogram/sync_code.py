#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Keep the full example listings in the tutorials identical to the sources.

    python3 tutorial/aieprogram/sync_code.py           # rewrite the listings
    python3 tutorial/aieprogram/sync_code.py --check   # exit 1 if one is stale

The root README.md is synced too. A listing is everything between
    <!-- code: example/tileprogram/ccode/<file>.cc -->
and
    <!-- /code -->
(paths relative to the repo root); it is replaced by a ```cpp block holding
the whole file.
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BLOCK = re.compile(r"(<!-- code: (?P<path>\S+) -->\n)(?P<body>.*?)(<!-- /code -->)", re.S)


def render(md_text):
    def fill(m):
        src = (REPO / m["path"]).read_text()
        return f"{m.group(1)}```cpp\n{src.rstrip()}\n```\n{m.group(4)}"
    return BLOCK.sub(fill, md_text)


def main():
    ap = argparse.ArgumentParser(description="sync the tutorials' full example listings")
    ap.add_argument("--check", action="store_true", help="only report stale listings")
    args = ap.parse_args()
    stale = []
    for md in sorted(HERE.rglob("*.md")) + [REPO / "README.md"]:  # the root README has one listing too
        old = md.read_text()
        new = render(old)
        if new != old:
            stale.append(str(md.relative_to(REPO)))
            if not args.check:
                md.write_text(new)
    if args.check:
        print("stale listings: " + ", ".join(stale) if stale else "listings up to date")
        return 1 if stale else 0
    print("updated: " + ", ".join(stale) if stale else "listings already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
