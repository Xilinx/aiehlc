#!/usr/bin/env python3
"""win_lint.py <file|dir>... -- no console I/O inside a timed window.

A timed window is, per C function, the span from the first to the last timing probe
(XTime_GetTime / uc_pmc( / uc_rt_cyc( / uc_hs_wait( ). Any printf / fprintf / puts /
putchar / fflush / xil_printf inside that span is a hit: console output blocks on the
UART FIFO and lands in the measured N segment.

.py files: every string literal that contains a timing probe (the C patch templates,
e.g. fa_attn.py FA_HS) is scanned as one window (the whole template).

Not counted as in-window:
  - an error exit: the print is followed, in its own brace block, by `return` before any
    probe or the end of the block (control leaves the function; the window is abandoned);
  - a line carrying (or directly preceded by a line carrying) `winlint: ok <reason>`.
Fragment sources (a function split across two files that the build concatenates): a
leading stray `}` closes a function that starts at the top of the file; a function still
open at EOF ends at EOF.

Output:
  WINLINT <file> windows=<n>
  WINLINT_HIT <file>:<line>
  WINLINT_OK <file>:<line> suppressed|error-exit          (informational)
  WINLINT ALL files=<n> windows=<n> in_window=<n> PASS|FAIL      (rc=1 on any hit or error)
"""
import ast
import os
import re
import sys

PROBE = re.compile(r"\bXTime_GetTime\s*\(|\buc_pmc\s*\(|\buc_rt_cyc\s*\(|\buc_hs_wait\s*\(")
PRINT = re.compile(r"\b(printf|fprintf|puts|putchar|fflush|xil_printf)\s*\(")
C_EXT = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inc")
PY_EXT = (".py",)


def blank_c(src):
    """Blank comments, string/char literals and preprocessor lines (keep newlines)."""
    out = []
    i, n = 0, len(src)
    bol = True  # at beginning of line (only whitespace so far)
    while i < n:
        c = src[i]
        if bol and c == "#":
            while i < n:  # preprocessor line incl. continuations
                if src[i] == "\\" and i + 1 < n and src[i + 1] == "\n":
                    out.append(" \n")
                    i += 2
                    continue
                if src[i] == "\n":
                    break
                out.append(" ")
                i += 1
            continue
        if c == "/" and src.startswith("//", i):
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(c + re.sub(r"[^\n]", " ", src[i + 1:j - 1]) + (c if j - 1 > i else ""))
            i = j
            bol = False
            continue
        out.append(c)
        if c == "\n":
            bol = True
        elif not c.isspace():
            bol = False
        i += 1
    return "".join(out)


def c_functions(code):
    """Yield (start, end) offsets of function bodies in blanked C/C++ code."""
    stack = []  # entries: True if this brace opens a function body
    in_fn = 0
    start = None
    for m in re.finditer(r"[{}]", code):
        if m.group() == "{":
            is_fn = False
            if not in_fn:
                pre = code[:m.start()].rstrip()
                pre = re.sub(r"(\s*\b(const|noexcept|override|final|volatile)\b)+$", "", pre).rstrip()
                pre = re.sub(r"\s*->\s*[\w:<>\s\*&]+$", "", pre).rstrip()  # trailing return type
                is_fn = pre.endswith(")")
            stack.append(is_fn)
            if is_fn:
                in_fn = 1
                start = m.start()
        else:
            if not stack:
                if in_fn:  # cannot happen: a function brace is on the stack
                    raise ValueError("unbalanced '}' at offset %d" % m.start())
                yield 0, m.end()  # fragment: tail of a function begun in another file
                continue
            if stack.pop():
                in_fn = 0
                yield start, m.end()
    if in_fn:
        yield start, len(code)  # fragment: function continues in another file


def line_of(text, off, base=1):
    return base + text.count("\n", 0, off)


def error_exit(body, pos):
    """True if the print at body[pos] is followed in its own block by `return` before any probe."""
    depth = 0
    for m in re.finditer(r"[{}]|\breturn\b|" + PROBE.pattern, body[pos:]):
        t = m.group()
        if t == "{":
            depth += 1
        elif t == "}":
            depth -= 1
            if depth < 0:
                return False
        elif t == "return":
            if depth == 0:
                return True
        else:
            return False
    return False


def suppressed(lines, ln):
    return any("winlint: ok" in lines[i - 1] for i in (ln, ln - 1) if 1 <= i <= len(lines))


def scan_c(path, src):
    code = blank_c(src)
    lines = src.split("\n")
    windows, hits, oks = 0, [], []
    for a, b in c_functions(code):
        body = code[a:b]
        probes = [m.start() for m in PROBE.finditer(body)]
        if not probes:
            continue
        windows += 1
        lo, hi = probes[0], probes[-1]
        for m in PRINT.finditer(body):
            if lo <= m.start() <= hi:
                ln = line_of(code, a + m.start())
                if suppressed(lines, ln):
                    oks.append((ln, "suppressed"))
                elif error_exit(body, m.end()):
                    oks.append((ln, "error-exit"))
                else:
                    hits.append((ln, m.group(1)))
    return windows, hits, oks


def scan_py(path, src):
    tree = ast.parse(src, filename=path)
    windows, hits, oks = 0, [], []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        s = node.value
        if not PROBE.search(s):
            continue
        windows += 1  # the whole template is one window
        code = blank_c(s)
        for m in PRINT.finditer(code):
            hits.append((line_of(code, m.start(), node.lineno), m.group(1)))
    return windows, hits, oks


def collect(args):
    files = []
    for a in args:
        if os.path.isdir(a):
            for root, dirs, names in os.walk(a):
                dirs.sort()
                for nm in sorted(names):
                    if nm.endswith(C_EXT + PY_EXT):
                        files.append(os.path.join(root, nm))
        elif os.path.isfile(a):
            files.append(a)
        else:
            raise FileNotFoundError(a)
    return files


def main(argv):
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    err = 0
    try:
        files = collect(argv)
    except FileNotFoundError as e:
        print("WINLINT_ERR %s: no such file or directory" % e, file=sys.stderr)
        print("WINLINT ALL files=0 windows=0 in_window=0 FAIL")
        return 1
    tw = th = 0
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            w, hits, oks = (scan_py if f.endswith(PY_EXT) else scan_c)(f, src)
        except (ValueError, SyntaxError) as e:
            print("WINLINT_ERR %s: %s" % (f, e))
            err += 1
            continue
        print("WINLINT %s windows=%d" % (f, w))
        for ln, call in hits:
            print("WINLINT_HIT %s:%d" % (f, ln))
        for ln, why in oks:
            print("WINLINT_OK %s:%d %s" % (f, ln, why))
        tw += w
        th += len(hits)
    ok = th == 0 and err == 0
    print("WINLINT ALL files=%d windows=%d in_window=%d %s" % (len(files), tw, th, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
