#!/usr/bin/env python3
"""Keyed column compare of two TSV files (costmodel M3.1.2.2 PNL: reproduce lpred / npred bit-for-bit).

  tsv_cmp.py A B --key k1,k2.. --cols c1,c2.. [--tol X]
  tsv_cmp.py --selftest

Both files have a header row; key and value columns are looked up by header name (column order may differ).
Rows are matched on the tuple of key columns (string compare). For every matched key and every --cols column:
  both parse as float -> diff = |a - b| (nan == nan -> 0; nan vs number -> inf; +-inf equal -> 0)
  otherwise           -> diff = 0 if the strings are equal else inf
Prints one line:
  TSVCMP a=<A> b=<B> keys=<matched> only_a=<n> only_b=<n> cols=<c1,c2> maxdiff=<x> PASS|FAIL
PASS iff only_a == only_b == 0, no duplicate key in either file, every column is present and maxdiff <= --tol
(default 0 = exact). rc 0 = PASS, 1 = FAIL (also on a read error; the line is still printed).
Problems (missing column, duplicate key, first worst row) are printed as 'TSVCMP NOTE ...' lines before the result.
"""
import argparse
import math
import os
import sys
import tempfile


def read_tsv(path):
    with open(path) as f:
        lines = [ln.rstrip("\n").rstrip("\r") for ln in f]
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        raise ValueError(f"{path}: empty")
    hdr = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    return hdr, rows


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def cell_diff(a, b):
    fa, fb = _num(a), _num(b)
    if fa is None or fb is None:
        return 0.0 if a == b else math.inf
    if math.isnan(fa) and math.isnan(fb):
        return 0.0
    if math.isnan(fa) or math.isnan(fb):
        return math.inf
    if fa == fb:
        return 0.0  # covers equal infinities
    return abs(fa - fb)


def index(hdr, rows, keys, path, notes):
    miss = [k for k in keys if k not in hdr]
    if miss:
        notes.append(f"{path}: missing key column(s) {','.join(miss)}")
        return None, 0
    ki = [hdr.index(k) for k in keys]
    out, dup = {}, 0
    for r in rows:
        r = r + [""] * (len(hdr) - len(r))
        k = tuple(r[i] for i in ki)
        if k in out:
            dup += 1
            continue
        out[k] = dict(zip(hdr, r))
    if dup:
        notes.append(f"{path}: {dup} duplicate key row(s)")
    return out, dup


def compare(pa, pb, keys, cols, tol=0.0):
    """Return (ok, line, notes)."""
    notes = []
    ha, ra = read_tsv(pa)
    hb, rb = read_tsv(pb)
    ia, da = index(ha, ra, keys, pa, notes)
    ib, db = index(hb, rb, keys, pb, notes)
    colok = True
    for c in cols:
        for p, h in ((pa, ha), (pb, hb)):
            if c not in h:
                notes.append(f"{p}: missing column {c}")
                colok = False
    if ia is None or ib is None:
        line = (f"TSVCMP a={pa} b={pb} keys=0 only_a=0 only_b=0 cols={','.join(cols)} maxdiff=inf FAIL")
        return False, line, notes
    common = [k for k in ia if k in ib]
    only_a = sum(1 for k in ia if k not in ib)
    only_b = sum(1 for k in ib if k not in ia)
    maxdiff, worst = 0.0, None
    if colok:
        for k in common:
            for c in cols:
                d = cell_diff(ia[k][c], ib[k][c])
                if d > maxdiff or (worst is None and d > 0):
                    maxdiff, worst = d, (k, c, ia[k][c], ib[k][c])
    else:
        maxdiff = math.inf
    if worst is not None:
        notes.append(f"worst key={'|'.join(worst[0])} col={worst[1]} a={worst[2]} b={worst[3]}")
    ok = colok and only_a == 0 and only_b == 0 and da == 0 and db == 0 and maxdiff <= tol
    md = "inf" if math.isinf(maxdiff) else f"{maxdiff:.6g}"
    line = (f"TSVCMP a={pa} b={pb} keys={len(common)} only_a={only_a} only_b={only_b} "
            f"cols={','.join(cols)} maxdiff={md} {'PASS' if ok else 'FAIL'}")
    return ok, line, notes


# ------------------------------------------------------------------ selftest
def selftest():
    cases = []
    with tempfile.TemporaryDirectory() as td:
        def w(name, text):
            p = os.path.join(td, name)
            with open(p, "w") as f:
                f.write(text)
            return p
        base = w("a.tsv", "cfg\trun\tsite\tL_hat\tN_hat\n"
                          "off\ta\tconv0\t1.5\tnan\n"
                          "off\ta\tconv1\t2.0\t3.0\n"
                          "aot\tb\tconv0\t0.25\tx\n")
        same_reord = w("b.tsv", "run\tsite\tcfg\tN_hat\tL_hat\n"
                                "a\tconv1\toff\t3.000\t2\n"
                                "b\tconv0\taot\tx\t0.25\n"
                                "a\tconv0\toff\tnan\t1.5\n")
        small = w("c.tsv", "cfg\trun\tsite\tL_hat\tN_hat\n"
                           "off\ta\tconv0\t1.5000000001\tnan\n"
                           "off\ta\tconv1\t2.0\t3.0\n"
                           "aot\tb\tconv0\t0.25\tx\n")
        extra = w("d.tsv", "cfg\trun\tsite\tL_hat\tN_hat\n"
                           "off\ta\tconv0\t1.5\tnan\n"
                           "off\ta\tconv1\t2.0\t3.0\n"
                           "aot\tb\tconv0\t0.25\tx\n"
                           "aot\tb\tconv9\t0.25\tx\n")
        nanvs = w("e.tsv", "cfg\trun\tsite\tL_hat\tN_hat\n"
                           "off\ta\tconv0\t1.5\t1.0\n"
                           "off\ta\tconv1\t2.0\t3.0\n"
                           "aot\tb\tconv0\t0.25\tx\n")
        dup = w("f.tsv", "cfg\trun\tsite\tL_hat\tN_hat\n"
                         "off\ta\tconv0\t1.5\tnan\n"
                         "off\ta\tconv0\t1.5\tnan\n"
                         "off\ta\tconv1\t2.0\t3.0\n"
                         "aot\tb\tconv0\t0.25\tx\n")
        strdiff = w("g.tsv", "cfg\trun\tsite\tL_hat\tN_hat\n"
                             "off\ta\tconv0\t1.5\tnan\n"
                             "off\ta\tconv1\t2.0\t3.0\n"
                             "aot\tb\tconv0\t0.25\ty\n")
        K, C = ["cfg", "run", "site"], ["L_hat", "N_hat"]

        def chk(name, ok_exp, pa, pb, keys=K, cols=C, tol=0.0, sub=None):
            ok, line, _ = compare(pa, pb, keys, cols, tol)
            good = ok == ok_exp and (sub is None or sub in line) and line.endswith("PASS" if ok_exp else "FAIL")
            cases.append((name, good, line))

        chk("identical", True, base, base, sub="keys=3 only_a=0 only_b=0")
        chk("reordered cols/rows, nan==nan, 3.000==3", True, base, same_reord, sub="maxdiff=0 ")
        chk("small diff exact -> FAIL", False, base, small)
        chk("small diff within tol", True, base, small, tol=1e-9)
        chk("extra row only_b", False, base, extra, sub="only_b=1")
        chk("extra row only_a", False, extra, base, sub="only_a=1")
        chk("nan vs number", False, base, nanvs, sub="maxdiff=inf")
        chk("duplicate key", False, dup, base)
        chk("string differs", False, base, strdiff, sub="maxdiff=inf")
        chk("missing column", False, base, base, cols=["L_hat", "nope"])
        chk("missing key column", False, base, base, keys=["cfg", "part"])
        chk("cell_diff", cell_diff("nan", "NaN") == 0 and cell_diff("inf", "inf") == 0
            and cell_diff("1", "1.0") == 0 and math.isinf(cell_diff("1", "a")), base, base)
    n_ok = 0
    for name, good, line in cases:
        n_ok += good
        print(f"{'ok  ' if good else 'BAD '} {name}: {line}")
    tag = "PASS" if n_ok == len(cases) else "FAIL"
    print(f"TSVCMP SELFTEST {tag} ({n_ok}/{len(cases)})")
    return n_ok == len(cases)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--key", default="")
    ap.add_argument("--cols", default="")
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    keys = [k for k in a.key.split(",") if k]
    cols = [c for c in a.cols.split(",") if c]
    if not a.a or not a.b or not keys or not cols:
        ap.error("need A B --key ... --cols ...")
    try:
        ok, line, notes = compare(a.a, a.b, keys, cols, a.tol)
    except (OSError, ValueError) as e:
        print(f"TSVCMP NOTE {e}")
        print(f"TSVCMP a={a.a} b={a.b} keys=0 only_a=0 only_b=0 cols={','.join(cols)} maxdiff=inf FAIL")
        return 1
    for n in notes:
        print(f"TSVCMP NOTE {n}")
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
