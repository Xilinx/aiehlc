#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""kgeom_equiv -- are the M3.1.2.1 UNet build logs (built from inputs=<base>) still valid for HEAD?

Proves, offline (no build, no board):
  (1) scope = INPUT_PATHS of example/aiegraph/unet_bk_sdm_tiny/build_unet_cfg.sh (the build-input key), test_*.py
      excluded: every diff hunk <base>..HEAD in scope lies inside unet_aie.kacc_geom (line ranges by ast, base
      side and HEAD side) -> hunks_outside=0;  no non-test unet_bk_sdm_tiny/*.py imports aiegraph.plan / autopart
      (plan_used) or aiperf / costmodel (imports_cm) -- by ast, so comments that name them do not count;
  (2) every kacc_geom call the UNet build planning makes gives the same result in both versions: the base module
      is `git show <base>:unet_aie.py` written to a temp dir (work tree untouched); HEAD's kacc_geom is wrapped by
      a recorder and the real HEAD planning functions are driven with the layer shapes of the build logs
      (driver=buildlog: kacc_split for linear (:145), bmm_geom for attn (:243-250, + kacc_gemm.plan),
      kacc_wlayout (:552), kacc_dense_wlayout (:564), kacc_host -> kacc_name (:594, :654, :657, :660));
      each deduplicated (ch, k, px) is fed to both versions, an exception counts as a result: its type, the
      outcome callers branch on (a same-type, different-message pair is printed as exc_msg_diff, not a diff);
  (3) env_knobs = KACC_TR_TKW_CAP / KACC_KP_PAD set in os.environ or named in <build-root>/*.build.log, *.wrap.log.
Output: "KGEOM call ch= k= px= same=0|1" per call, last line
  KGEOM ALL base= scope=input_paths commits= hunks= hunks_outside= calls= diff= env_knobs= plan_used= imports_cm= SAME|DIFF
rc=1 on DIFF.  --selftest: KGEOM SELFTEST PASS (last line).  Importable: compare(), scope_check(), env_knobs(),
drive_build(), load_base(), load_head().
"""
import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parents[1]
EX = TOP / "example/aiegraph/unet_bk_sdm_tiny"
UA_REL = "example/aiegraph/unet_bk_sdm_tiny/unet_aie.py"
for _p in (str(EX), str(TOP / "src"), str(TOP / "src/aiegraph")):   # -I mode: no cwd / PYTHONPATH
    if _p not in sys.path:
        sys.path.insert(0, _p)

KNOBS = ("KACC_TR_TKW_CAP", "KACC_KP_PAD")
# unet-decl P4d (main) deleted the example unet_aie.py: the UNet kernel assembly is aiegraph.kincore / cnn.kacc_geom.
# Every M3.1.2.1 build is then OUTSIDE (rebuild); the selftest keeps checking the comparison machinery on the last
# unet_aie.py of the costmodel line.
UA_LAST = "a6c59eea"
FUNC = "kacc_geom"


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(TOP), *args], check=True, capture_output=True, text=True).stdout


def input_paths() -> list:
    """INPUT_PATHS=(...) of build_unet_cfg.sh (the build-input key of the M3.1.2.1 builds)"""
    txt = (EX / "build_unet_cfg.sh").read_text()
    m = re.search(r"^INPUT_PATHS=\(([^)]*)\)", txt, re.M)
    assert m, "INPUT_PATHS not found in build_unet_cfg.sh"
    return m.group(1).split()


def func_range(src: str, name: str = FUNC) -> tuple:
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise KeyError(name)


def hunks_of(diff: str) -> list:
    """unified -U0 diff -> [(file, old_start, old_len, new_start, new_len)]"""
    out, f = [], None
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            f = ln[6:] if ln.startswith("+++ b/") else None
        elif ln.startswith("--- ") and f is None:
            pass
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
        if m:
            out.append((f, int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)))
    return out


def inside(start, n, rng) -> bool:
    lo, hi = rng
    if n == 0:   # pure insertion/deletion side: the position is the line before the change
        return lo <= start <= hi
    return lo <= start and start + n - 1 <= hi


def classify(hunks: list, base_rng: tuple, head_rng: tuple, ua_rel: str = UA_REL) -> int:
    """number of hunks outside kacc_geom (any hunk not in unet_aie.py is outside)"""
    bad = 0
    for f, os_, on, ns, nn in hunks:
        if f != ua_rel or not (inside(os_, on, base_rng) and inside(ns, nn, head_rng)):
            bad += 1
    return bad


def static_imports(paths) -> tuple:
    """(plan_used, imports_cm, detail): ast imports of aiegraph.plan / autopart and aiperf / costmodel"""
    plan_used = imports_cm = 0
    detail = []
    for p in paths:
        tree = ast.parse(Path(p).read_text())
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = n.module or ""
                mods = [base] + [f"{base}.{a.name}" if base else a.name for a in n.names]
            for mname in mods:
                parts = set(mname.split("."))
                if parts & {"plan", "autopart"}:
                    plan_used += 1
                    detail.append(f"{Path(p).name}:{n.lineno} {mname}")
                if parts & {"aiperf", "costmodel"}:
                    imports_cm += 1
                    detail.append(f"{Path(p).name}:{n.lineno} {mname}")
    return plan_used, imports_cm, detail


def scope_check(base: str) -> dict:
    paths = input_paths()
    spec = paths + [":(exclude,glob)**/test_*.py"]
    commits = int(git("rev-list", "--count", f"{base}..HEAD", "--", *spec).strip() or 0)
    files = [f for f in git("diff", "--name-only", f"{base}..HEAD", "--", *spec).split() if f]
    hunks = hunks_of(git("diff", "-U0", f"{base}..HEAD", "--", *spec))
    bsrc = git("show", f"{base}:{UA_REL}")
    hsrc = (TOP / UA_REL).read_text()
    brng, hrng = func_range(bsrc), func_range(hsrc)
    outside = classify(hunks, brng, hrng)
    py = sorted(p for p in EX.glob("*.py") if not p.name.startswith("test_"))
    plan_used, imports_cm, detail = static_imports(py)
    return dict(commits=commits, files=files, hunks=len(hunks), hunks_outside=outside, base_rng=brng,
                head_rng=hrng, plan_used=plan_used, imports_cm=imports_cm, import_detail=detail,
                hunk_list=hunks)


def env_knobs(build_root) -> tuple:
    hits = [f"env:{k}" for k in KNOBS if k in os.environ]
    if build_root:
        for p in sorted(list(Path(build_root).glob("*.build.log")) + list(Path(build_root).glob("*.wrap.log"))):
            t = p.read_text(errors="replace")
            hits += [f"{p.name}:{k}" for k in KNOBS if k in t]
    return len(hits), hits


def load_src(src: str, name: str, tmpdir: str):
    p = Path(tmpdir) / f"{name}.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_geom(src: str):
    """only the kacc_geom function of a unet_aie.py source (the rest of an old copy imports aiednn templates that
    unet-decl P4d removed) -> namespace with .kacc_geom"""
    import types
    a, b = func_range(src)
    ns = {"_os": os, "os": os}
    exec(compile("\n".join(src.splitlines()[a - 1:b]), "<kacc_geom>", "exec"), ns)
    return types.SimpleNamespace(kacc_geom=ns[FUNC])


def load_base(base: str, tmpdir: str):
    """unet_aie.py at <base> as module 'unet_aie_base' (temp file, work tree untouched)"""
    return load_src(git("show", f"{base}:{UA_REL}"), "unet_aie_base", tmpdir)


def head_has_ua() -> bool:
    return (TOP / UA_REL).is_file()


def load_head():
    """HEAD's unet_aie as 'unet_aie' (same object kacc_gemm imported as ua before unet-decl P4d)"""
    if not head_has_ua():
        raise RuntimeError(f"{UA_REL} removed at HEAD (unet-decl P4d): no kacc_geom equivalence, rebuild")
    import unet_aie
    return unet_aie


def result_of(fn, ch, k, px):
    try:
        return ("ok", tuple(sorted(fn(ch, k, px).items())))
    except Exception as e:  # noqa: BLE001 -- an exception is a result too
        return ("exc", type(e).__name__, repr(e.args))


def outcome(r) -> tuple:
    """what a caller can observe: the returned dict, or the exception TYPE (kacc_split / kacc_gemm.fits branch on
    `except AssertionError`; no caller reads the assert message, which carries diagnostic locals such as tkw)"""
    return r if r[0] == "ok" else r[:2]


def compare(base_fn, head_fn, triples) -> list:
    """[(ch, k, px, same, base_result, head_result, exc_msg_diff)] for each deduplicated triple;
    same = equal outcome(); exc_msg_diff = same exception type, different assert message (reported, not a diff)"""
    out = []
    for ch, k, px in sorted(set(triples)):
        rb, rh = result_of(base_fn, ch, k, px), result_of(head_fn, ch, k, px)
        same = int(outcome(rb) == outcome(rh))
        out.append((ch, k, px, same, rb, rh, int(same and rb != rh)))
    return out


LAYER_RE = {
    "conv": re.compile(r"\baie conv\s+(\d+) .* M=(\d+) N=(\d+) K=(\d+) s="),
    "linear": re.compile(r"\baie linear\s+(\d+) .* M=(\d+) N=(\d+) K=(\d+) px_split=(\d+)x(\d+)"),
    "attn": re.compile(r"\baie attn\s+(\d+) .* H=(\d+) M=(\d+) K=(\d+) N=(\d+) swap=(\d+) ch=(\d+) k=(\d+) px=(\d+) S=(\d+)"),
}


def build_layers(build_root) -> dict:
    """layer shapes per build log: {log name: {"conv": [...], "linear": [...], "attn": [...]}}"""
    out = {}
    for p in sorted(Path(build_root).glob("*.build.log")):
        d = {k: [] for k in LAYER_RE}
        for ln in p.read_text(errors="replace").splitlines():
            for k, rx in LAYER_RE.items():
                m = rx.search(ln)
                if m:
                    d[k].append(tuple(int(x) for x in m.groups()))
        out[p.name] = d
    return out


def drive_build(ua, layers: dict, partition=None) -> tuple:
    """drive HEAD's planning functions with the build-log shapes, kacc_geom wrapped by a recorder.
    Returns (recorded triples in call order, info lines, mismatches vs the build log)."""
    import numpy as np
    import kacc_gemm  # noqa: F401  (imports unet_aie as ua: same module object, so the recorder sees its calls)
    assert kacc_gemm.ua is ua, "kacc_gemm.ua is not the HEAD unet_aie module"
    orig = ua.kacc_geom
    rec = []

    def wrapped(ch, k, px):
        rec.append((ch, k, px))
        return orig(ch, k, px)
    ua.kacc_geom = wrapped
    info, bad = [], []
    try:
        for name, d in layers.items():
            if partition is not None:
                import collections
                pc = collections.Counter(L.get("cls") for L in partition["layers"])
                for cls in ("conv", "linear", "attn"):
                    if pc.get(cls, 0) != len(d[cls]):
                        bad.append(f"{name}: {cls} log={len(d[cls])} partition={pc.get(cls, 0)}")
                byidx = {L["idx"]: L for L in partition["layers"]}
                for cid, M, N, K in d["conv"]:
                    L = byidx.get(cid)
                    if L is None or list(L["mkn"][0]) != [M, K, N]:
                        bad.append(f"{name}: conv {cid} M={M} N={N} K={K} vs partition {L and L['mkn'][0]}")
            lin = []
            for cid, M, N, K, S, pxs in d["linear"]:
                s2, p2 = ua.kacc_split(N, K, -(-M // 64) * 64)          # :218 -> :145
                if (s2, p2) != (S, pxs):
                    bad.append(f"{name}: linear {cid} split {s2}x{p2} != log {S}x{pxs}")
                lin.append((N, K, -(-M // 64) * 64, p2))
                ua.kacc_dense_wlayout(np.zeros((N // 16, K, 16), np.uint16), N, K, p2)   # :564
            for cid, H, M, K, N, swap, ch, kk, px, S in d["attn"]:
                g = ua.bmm_geom(H, M, K, N)                             # :243-250 (+ kacc_gemm.plan)
                got = (g["swap"], g["chs"], g["Kp"], g["pxs"], g["S"])
                if got != (swap, ch, kk, px, S):
                    bad.append(f"{name}: attn {cid} geo {got} != log {(swap, ch, kk, px, S)}")
            shapes = sorted({(N, K, M) for _, M, N, K in d["conv"]} | {(n, k, p) for n, k, mp, p in lin})
            for _, M, N, K in d["conv"]:
                ua.kacc_wlayout(np.zeros((N // 16, K, 1, 1, 1, 16), np.uint16), N, K, M)   # :552
            host = ua.kacc_host(shapes, lin)                             # :594 :654 :657 :660
            info.append((name, len(d["conv"]), len(d["linear"]), len(d["attn"]), len(shapes), shapes, lin, host))
    finally:
        ua.kacc_geom = orig
    return rec, info, bad


def run(a) -> int:
    if not head_has_ua():
        print(f"KGEOM ALL base={a.base} scope=input_paths OUTSIDE {UA_REL} removed at HEAD (unet-decl P4d: the UNet "
              "kernel assembly is aiegraph.kincore / cnn.kacc_geom) -- rebuild DIFF")
        return 1
    sc = scope_check(a.base)
    print(f"KGEOM scope input_paths={','.join(input_paths())} (test_*.py excluded) files={','.join(sc['files'])}")
    print(f"KGEOM kacc_geom lines base={sc['base_rng'][0]}-{sc['base_rng'][1]} head={sc['head_rng'][0]}-{sc['head_rng'][1]}")
    for h in sc["hunk_list"]:
        print(f"KGEOM hunk {h[0]} -{h[1]},{h[2]} +{h[3]},{h[4]}")
    for d in sc["import_detail"]:
        print(f"KGEOM import {d}")
    nk, kh = env_knobs(a.build_root)
    for h in kh:
        print(f"KGEOM env_knob {h}")
    srcs = sorted(Path(a.build_root).glob("*.src"))
    for p in srcs:
        print(f"KGEOM build_src {p.name}: {p.read_text().strip()}")
    import json
    part = json.loads(Path(a.partition).read_text()) if a.partition else None
    layers = build_layers(a.build_root)
    if not layers:
        print(f"KGEOM no *.build.log under {a.build_root}")
        return 2
    with tempfile.TemporaryDirectory(prefix="kgeom_base_") as td:
        ub = load_base(a.base, td)
        uh = load_head()
        rec, info, bad = drive_build(uh, layers, part)
        for name, nc, nl, na, ns, shapes, lin, host in info:
            # the generated host dispatch text, base module vs HEAD module (info only)
            try:
                hb = ub.kacc_host(shapes, lin)
            except Exception as e:  # noqa: BLE001
                hb = repr(e)
            print(f"KGEOM driver=buildlog {name} conv={nc} linear={nl} attn={na} shapes={ns} "
                  f"host_text_same={int(hb == host)}")
        for b in bad:
            print(f"KGEOM driver_mismatch {b}")
        res = compare(ub.kacc_geom, uh.kacc_geom, rec)
    for ch, k, px, same, rb, rh, md in res:
        print(f"KGEOM call ch={ch} k={k} px={px} same={same}" +
              ("" if same and not md else f" base={rb} head={rh}" + (" exc_msg_diff=1" if md else "")))
    ndiff = sum(1 for r in res if not r[3])
    print(f"KGEOM exc_results={sum(1 for r in res if r[4][0] == 'exc')} exc_msg_diff={sum(r[6] for r in res)} "
          f"(same exception type, different assert message; counted same)")
    ok = (ndiff == 0 and sc["hunks_outside"] == 0 and nk == 0 and sc["plan_used"] == 0 and sc["imports_cm"] == 0
          and not bad and len(res) > 0)
    print(f"KGEOM recorded_calls={len(rec)} unique={len(res)}")
    print(f"KGEOM ALL base={a.base} scope=input_paths commits={sc['commits']} hunks={sc['hunks']} "
          f"hunks_outside={sc['hunks_outside']} calls={len(res)} diff={ndiff} env_knobs={nk} "
          f"plan_used={sc['plan_used']} imports_cm={sc['imports_cm']} {'SAME' if ok else 'DIFF'}")
    return 0 if ok else 1


def selftest() -> int:
    fails = []

    def check(name, cond):
        print(f"KGEOM selftest {name} {'ok' if cond else 'FAIL'}")
        if not cond:
            fails.append(name)
    hsrc = (TOP / UA_REL).read_text() if head_has_ua() else git("show", f"{UA_LAST}:{UA_REL}")
    with tempfile.TemporaryDirectory(prefix="kgeom_st_") as td:
        if head_has_ua():
            uh, ub, u2 = load_head(), load_base("f531f051", td), load_src(hsrc, "unet_aie_same", td)
        else:   # unet-decl P4d: geometry function only, from the last costmodel copy
            uh, ub, u2 = load_geom(hsrc), load_geom(git("show", f"f531f051:{UA_REL}")), load_geom(hsrc)
        # 1. tr and px == k//2: HEAD pads kp by 2*tkw, f531f051 did not -> must be a diff
        g = uh.kacc_geom(320, 2048, 1024)
        check("tr_px_eq_khalf_is_tr", g["tr"] == 1 and 1024 == 2048 // 2)
        r = compare(ub.kacc_geom, uh.kacc_geom, [(320, 2048, 1024)])
        check("tr_px_eq_khalf_diff", r[0][3] == 0)
        # 2. the same source in two modules -> SAME on a spread of triples (ok and exception results)
        tri = [(320, 2880, 1024), (320, 320, 1024), (1280, 2560, 64), (640, 1280, 256), (5120, 640, 1024),
               (320, 2048, 1024), (64, 64, 64), (65, 64, 64)]
        r = compare(u2.kacc_geom, uh.kacc_geom, tri)
        check("same_source_same", all(x[3] for x in r) and len(r) == len(tri))
        check("exception_is_result", any(x[4][0] == "exc" for x in r))
        # 3. exception type is the outcome: a different type is a diff, the same type with another message is not
        def f_ok(ch, k, px):
            return {"kp": k}

        def f_zde(ch, k, px):
            raise ZeroDivisionError("integer division or modulo by zero")

        def f_a1(ch, k, px):
            assert False, (ch, k, px, 1)

        def f_a2(ch, k, px):
            assert False, (ch, k, px, 2)
        check("exc_type_diff", compare(f_zde, f_a1, [(64, 64, 64)])[0][3] == 0)
        check("ok_vs_exc_diff", compare(f_ok, f_a1, [(64, 64, 64)])[0][3] == 0)
        r = compare(f_a1, f_a2, [(64, 64, 64)])[0]
        check("exc_msg_only_same", r[3] == 1 and r[6] == 1)
    # 4. hunk classification
    rb, rh = (10, 20), (10, 25)
    check("hunk_inside", classify([(UA_REL, 12, 1, 12, 3)], rb, rh) == 0)
    check("hunk_outside_range", classify([(UA_REL, 30, 1, 35, 1)], rb, rh) == 1)
    check("hunk_other_file", classify([("src/aiednn/x.py", 12, 1, 12, 1)], rb, rh) == 1)
    check("hunks_parse", hunks_of("+++ b/a.py\n@@ -3,0 +4,2 @@\n@@ -9 +11 @@\n") ==
          [("a.py", 3, 0, 4, 2), ("a.py", 9, 1, 11, 1)])
    # 5. import scan ignores comments, catches real imports
    with tempfile.TemporaryDirectory(prefix="kgeom_imp_") as td:
        p1, p2 = Path(td) / "a.py", Path(td) / "b.py"
        p1.write_text("# costmodel and aiegraph.plan are only named here\nimport os\n")
        p2.write_text("from costmodel import estimate\nfrom aiegraph import plan\n")
        check("imports_comment_free", static_imports([p1])[:2] == (0, 0))
        pu, cm, _ = static_imports([p2])
        check("imports_found", pu >= 1 and cm >= 1)
    n = 13
    print(f"KGEOM SELFTEST {'PASS' if not fails else 'FAIL'} {n - len(fails)}/{n}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base", default="f531f051")
    ap.add_argument("--build-root", default=None)
    ap.add_argument("--partition", default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.build_root:
        ap.error("--build-root is required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
