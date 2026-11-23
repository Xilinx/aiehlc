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
the whole file. Options after the path shorten it: `nocomments`, `notimer`, `nocheck`,
`devicelast`; after a `;`, `after <anchor>: <text>` adds a `// <text>` line after
the line containing <anchor> (e.g.
`<!-- code: example/.../x.cc nocomments notimer; after foo(: what foo does -->`).
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BLOCK = re.compile(r"(<!-- code: (?P<path>\S+)(?P<opts>[^\n]*?) -->\n)(?P<body>.*?)(<!-- /code -->)", re.S)


def shorten(src, opts):
    """Options after the path: `nocomments` drops // comments (whole lines and
    trailing), `notimer` drops the XTime timing lines (declaration, the two
    XTime_GetTime calls and the [BENCH] printf), `nocheck` drops the golden
    check (from the scalar_conv2d( reference call through the PASS / FAIL printf),
    `devicelast` moves the device setup (aieSetDevice / aieArray / partition)
    to just before the <<<mesh>>> launch."""
    out = []
    in_check = False
    for line in src.splitlines():
        if "nocheck" in opts and line.strip().startswith("scalar_conv2d("):  # the call, not a comment
            in_check = True
        if in_check:
            in_check = "mismatches out of" not in line
            continue
        if "notimer" in opts and ("XTime" in line or "[BENCH]" in line):
            continue
        if "nocomments" in opts:
            if line.strip().startswith("//"):
                continue
            code = line.split("//", 1)[0]
            if "//" in line and code.count('"') % 2 == 0:  # not inside a string literal
                line = code.rstrip()
        out.append(line)
    if "devicelast" in opts:
        # Move the device setup next to the launch: allocation needs no device
        # (__Runtime_Alloc is an aligned host allocation).
        dev = [l for l in out if l.strip().startswith(("aieSetDevice(", "aieArray device", "aieMesh mesh = device."))]
        launch = next((k for k, l in enumerate(out) if "<<<mesh>>>" in l), None)
        if dev and launch is not None:
            rest = [l for l in out if l not in dev]
            at = next(k for k, l in enumerate(rest) if "<<<mesh>>>" in l)
            out = rest[:at] + dev + rest[at:]
    return "\n".join(out)


def annotate(src, notes):
    """`after <anchor>: <text>` inserts `// <text>` after the first line containing
    <anchor>, with that line's indentation (a comment of the listing only);
    `replace <old> => <new>` rewrites text of the listing (exactly once)."""
    for note in notes:
        if note.strip().startswith("replace "):
            old, new = note.strip()[len("replace "):].split("=>", 1)
            old, new = old.rstrip(), new.strip()  # an empty <new> deletes <old>
            assert src.count(old) == 1, f"replace: {old!r} must occur exactly once"
            src = src.replace(old, new)
    lines = src.split("\n")
    for note in [n for n in notes if n.strip().startswith("after ")]:
        anchor, text = note.split(":", 1)
        anchor = anchor.strip()[len("after "):].strip()
        at = next(k for k, l in enumerate(lines) if anchor in l)
        indent = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
        lines.insert(at + 1, f"{indent}// {text.strip()}")
    return "\n".join(lines)


FIRST_BLOCK = re.compile(r"\A/\*(?:(?!\*/).)*\*/\n", re.S)


def drop_license(src):
    """Listings omit the license: drop the Copyright / SPDX lines of the file's
    first comment block, and the whole block if nothing but decoration remains."""
    m = FIRST_BLOCK.match(src)
    if not m or "SPDX-License-Identifier" not in m.group(0):
        return src
    kept = [l for l in m.group(0).splitlines(keepends=True)
            if not re.search(r"Copyright \(C\)|SPDX-License-Identifier", l)]
    if all(not re.sub(r"[/*\s]", "", l) for l in kept):
        return src[m.end():]
    return "".join(kept) + src[m.end():]


def render(md_text):
    def fill(m):
        opts, *notes = m["opts"].split(";")
        src = shorten(drop_license((REPO / m["path"]).read_text()), opts.split())
        src = annotate(src, [n for n in notes if n.strip().startswith(("after ", "replace "))])
        return f"{m.group(1)}```cpp\n{src.rstrip()}\n```\n{m.group(5)}"
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
