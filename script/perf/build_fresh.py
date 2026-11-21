#!/usr/bin/env python3
"""build_fresh -- is <outroot>/unet_<cfg> (main.elf) still the build of the current UNet build inputs?

Content-based successor of `build_unet_cfg.sh --check-fresh` (same rule as build_unet_cfg.sh:55-87, which is
NOT modified: it is itself in INPUT_PATHS). Reads <outroot>/unet_<cfg>.src ("BUILDSRC inputs=<h> start=<epoch>
site_perf=<n> kincore=<n>", written by the real build); reads no stamp of its own and writes nothing.

  scope       = INPUT_PATHS of example/aiegraph/unet_bk_sdm_tiny/build_unet_cfg.sh, test_*.py excluded
                (tests do not reach the ELF)
  head_inputs = `git log -1 --format=%h -- <scope>`;  dirty = `git status --porcelain -- <scope>` lines
  elf_newer   = main.elf mtime >= start;  site_perf / kincore from the .src line
  equiv       = EXACT    inputs == head_inputs
                SAME     inputs != head_inputs, every hunk inputs..HEAD in scope lies inside
                         unet_aie.kacc_geom (ast line ranges, both sides) and every kacc_geom call of the UNet build
                         planning (driven with this cfg's build-log shapes, kgeom_equiv.drive_build) gives the same
                         outcome in both versions (kgeom_equiv.compare) -- or there is no hunk at all
                DIFF     some kacc_geom call differs
                OUTSIDE  some hunk outside kacc_geom (or inputs unknown / not a commit)
PASS iff equiv in {EXACT, SAME}, dirty=0, elf_newer=1, site_perf=1, kincore=1.  Output (one line):
  BUILD FRESH cfg=<c> inputs=<h> head_inputs=<h> dirty=<n> elf_newer=<n> site_perf=<n> kincore=<n> equiv=<E> PASS|FAIL
FAIL -> rc=1.   usage: build_fresh.py <off|aot-kco> <outroot>  |  build_fresh.py --selftest  (BFRESH SELFTEST PASS)
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:          # -I: the script dir is not on sys.path
    sys.path.insert(0, str(HERE))
import kgeom_equiv as kg  # noqa: E402

TOP = HERE.parents[1]
UDIR = "example/aiegraph/unet_bk_sdm_tiny"
UA_REL = f"{UDIR}/unet_aie.py"
CFGS = ("off", "aot-kco")


def git(top, *a, check=True) -> str:
    return subprocess.run(["git", "-C", str(top), *a], check=check, capture_output=True, text=True).stdout


def input_paths(top) -> list:
    txt = (Path(top) / UDIR / "build_unet_cfg.sh").read_text()
    m = re.search(r"^INPUT_PATHS=\(([^)]*)\)", txt, re.M)
    assert m, "INPUT_PATHS not found in build_unet_cfg.sh"
    return m.group(1).split()


def scope(top) -> list:
    return input_paths(top) + [":(exclude,glob)**/test_*.py"]


def read_src(path) -> dict:
    d = dict(inputs="none", start="", site_perf="0", kincore="0")
    p = Path(path)
    if p.is_file():
        for ln in p.read_text(errors="replace").splitlines():
            if ln.startswith("BUILDSRC "):
                for kv in ln.split()[1:]:
                    k, _, v = kv.partition("=")
                    if k in d:
                        d[k] = v
                break
    return d


def real_geom(top, base, cfg, outroot):
    """kacc_geom calls of the real UNet build planning (this cfg's build log), base vs HEAD -> compare() rows"""
    layers = kg.build_layers(outroot)
    layers = {k: v for k, v in layers.items() if k == f"unet_{cfg}.build.log"}
    if not layers:
        raise RuntimeError(f"no unet_{cfg}.build.log under {outroot}")
    with tempfile.TemporaryDirectory(prefix="bfresh_base_") as td:
        ub = kg.load_src(git(top, "show", f"{base}:{UA_REL}"), "unet_aie_bfbase", td)
        uh = kg.load_head()
        rec, _info, bad = kg.drive_build(uh, layers)
        if bad:
            raise RuntimeError("driver mismatch vs build log: " + "; ".join(bad))
        if not rec:
            raise RuntimeError("no kacc_geom call recorded")
        return kg.compare(ub.kacc_geom, uh.kacc_geom, rec)


def equiv_of(top, base, head_inputs, cfg, outroot, geom=real_geom) -> tuple:
    """(equiv, detail lines)"""
    if base == "none" or not base:
        return "OUTSIDE", ["no BUILDSRC inputs"]
    if base == head_inputs:
        return "EXACT", []
    if subprocess.run(["git", "-C", str(top), "cat-file", "-e", f"{base}^{{commit}}"],
                      capture_output=True).returncode != 0:
        return "OUTSIDE", [f"inputs {base} is not a commit"]
    sp = scope(top)
    hunks = kg.hunks_of(git(top, "diff", "-U0", f"{base}..HEAD", "--", *sp))
    if not hunks:
        return "SAME", ["no hunk in scope"]
    try:
        brng = kg.func_range(git(top, "show", f"{base}:{UA_REL}"))
        hrng = kg.func_range(git(top, "show", f"HEAD:{UA_REL}"))
    except (subprocess.CalledProcessError, KeyError, SyntaxError) as e:
        return "OUTSIDE", [f"kacc_geom range: {e!r}"]
    out = kg.classify(hunks, brng, hrng, UA_REL)
    det = [f"hunk {h[0]} -{h[1]},{h[2]} +{h[3]},{h[4]}" for h in hunks]
    if out:
        return "OUTSIDE", det + [f"hunks_outside={out}"]
    rows = geom(top, base, cfg, outroot)
    nd = sum(1 for r in rows if not r[3])
    det.append(f"kacc_geom calls={len(rows)} diff={nd}")
    det += [f"call ch={r[0]} k={r[1]} px={r[2]} base={r[4]} head={r[5]}" for r in rows if not r[3]]
    return ("DIFF" if nd else "SAME"), det


def check(cfg, outroot, top=TOP, geom=real_geom, verbose=False) -> tuple:
    """(line, ok)"""
    sp = scope(top)
    head = git(top, "log", "-1", "--format=%h", "--", *sp).strip() or "none"
    dirty = len([ln for ln in git(top, "status", "--porcelain", "--", *sp).splitlines() if ln.strip()])
    s = read_src(Path(outroot) / f"unet_{cfg}.src")
    elf = Path(outroot) / f"unet_{cfg}" / "main.elf"
    elf_newer = int(elf.is_file() and s["start"].isdigit() and int(elf.stat().st_mtime) >= int(s["start"]))
    try:
        eq, det = equiv_of(top, s["inputs"], head, cfg, outroot, geom)
    except Exception as e:  # noqa: BLE001 -- cannot prove equivalence -> DIFF (FAIL)
        eq, det = "DIFF", [f"error: {e!r}"]
    if verbose:
        for d in det:
            print(f"BFRESH {cfg} {d}")
    ok = (eq in ("EXACT", "SAME") and dirty == 0 and elf_newer == 1 and s["site_perf"] == "1"
          and s["kincore"] == "1")
    line = (f"BUILD FRESH cfg={cfg} inputs={s['inputs']} head_inputs={head} dirty={dirty} elf_newer={elf_newer} "
            f"site_perf={s['site_perf']} kincore={s['kincore']} equiv={eq} {'PASS' if ok else 'FAIL'}")
    return line, ok


# ---------------------------------------------------------------- selftest helpers (also used by the pytest)
UA_BASE = '''import os


def helper(x):
    return x + 1


def kacc_geom(ch, k, px):
    tkw = 64
    kp = -(-k // tkw) * tkw
    return {"kp": kp, "tkw": tkw, "px": px}


def other(y):
    return y * 2
'''
UA_SAME = UA_BASE.replace('    tkw = 64\n', '    # same result, written differently\n    tkw = 32 * 2\n')
UA_DIFF = UA_BASE.replace('    tkw = 64\n', '    tkw = 128\n')
UA_OUTSIDE = UA_BASE.replace('return y * 2', 'return y * 3')
TRIPLES = [(64, 100, 64), (320, 2880, 1024), (16, 64, 32)]


def toy_geom(top, base, cfg, outroot):
    """selftest/pytest stand-in for real_geom: toy unet_aie.py at base vs HEAD on fixed triples"""
    with tempfile.TemporaryDirectory(prefix="bfresh_toy_") as td:
        ub = kg.load_src(git(top, "show", f"{base}:{UA_REL}"), "toy_ua_base", td)
        uh = kg.load_src(git(top, "show", f"HEAD:{UA_REL}"), "toy_ua_head", td)
        return kg.compare(ub.kacc_geom, uh.kacc_geom, TRIPLES)


def make_case(root) -> tuple:
    """temp git repo with the UNet input layout + an outroot whose unet_off.src names the base commit.
    Returns (repo, outroot, base_hash)."""
    r, o = Path(root) / "repo", Path(root) / "out"
    u = r / UDIR
    u.mkdir(parents=True)
    (u / "build_unet_cfg.sh").write_text(
        "#!/bin/bash\nINPUT_PATHS=(src/mlir/runtime src/aiegraph/runtime src/aiednn "
        "example/aiegraph/unet_bk_sdm_tiny)\n")
    (u / "unet_aie.py").write_text(UA_BASE)
    (u / "test_unet.py").write_text("def test_x():\n    assert True\n")
    for d in ("src/mlir/runtime", "src/aiegraph/runtime", "src/aiednn", "script/perf"):
        (r / d).mkdir(parents=True)
        (r / d / "f.txt").write_text("x\n")
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    base = git(r, "log", "-1", "--format=%h").strip()
    (o / "unet_off").mkdir(parents=True)
    (o / "unet_off" / "main.elf").write_text("elf\n")
    (o / "unet_off.src").write_text(f"BUILDSRC inputs={base} start=1 site_perf=1 kincore=1\n")
    return r, o, base


def commit_file(r, rel, text, msg="change"):
    (Path(r) / rel).write_text(text)
    git(r, "commit", "-qam", msg)


def selftest() -> int:
    fails = []

    def ck(name, line, ok, want_ok, want_eq):
        good = ok == want_ok and f" equiv={want_eq} " in line and line.endswith("PASS" if want_ok else "FAIL")
        print(f"BFRESH selftest {name} {'ok' if good else 'FAIL'}: {line}")
        if not good:
            fails.append(name)

    with tempfile.TemporaryDirectory(prefix="bfresh_st_") as td:
        n = 0

        def case():
            nonlocal n
            n += 1
            return make_case(Path(td) / f"c{n}")
        r, o, b = case()
        ck("unchanged_exact", *check("off", o, r, toy_geom), True, "EXACT")
        commit_file(r, f"{UDIR}/test_unet.py", "def test_x():\n    assert 1\n", "test only")
        ck("test_only_exact", *check("off", o, r, toy_geom), True, "EXACT")
        commit_file(r, "script/perf/f.txt", "perf\n", "not an input")
        ck("non_input_exact", *check("off", o, r, toy_geom), True, "EXACT")
        # .src recorded a test-only commit (build_unet_cfg.sh counts tests) -> no hunk in scope -> SAME
        r, o, b = case()
        commit_file(r, f"{UDIR}/test_unet.py", "def test_x():\n    assert 2\n", "test only")
        t = git(r, "log", "-1", "--format=%h").strip()
        (o / "unet_off.src").write_text(f"BUILDSRC inputs={t} start=1 site_perf=1 kincore=1\n")
        ck("src_on_test_commit_same", *check("off", o, r, toy_geom), True, "SAME")
        r, o, b = case()
        commit_file(r, UA_REL, UA_SAME)
        ck("geom_same", *check("off", o, r, toy_geom), True, "SAME")
        r, o, b = case()
        commit_file(r, UA_REL, UA_DIFF)
        ck("geom_diff", *check("off", o, r, toy_geom), False, "DIFF")
        r, o, b = case()
        commit_file(r, UA_REL, UA_OUTSIDE)
        ck("outside_ua", *check("off", o, r, toy_geom), False, "OUTSIDE")
        r, o, b = case()
        commit_file(r, "src/aiednn/f.txt", "y\n")
        ck("outside_other_file", *check("off", o, r, toy_geom), False, "OUTSIDE")
        r, o, b = case()
        (r / "src/aiegraph/runtime/f.txt").write_text("dirty\n")
        line, ok = check("off", o, r, toy_geom)
        ck("dirty", line, ok, False, "EXACT")
        if " dirty=1 " not in line:
            fails.append("dirty_count")
        r, o, b = case()
        (o / "unet_off.src").write_text(f"BUILDSRC inputs={b} start=4000000000 site_perf=1 kincore=1\n")
        ck("elf_older", *check("off", o, r, toy_geom), False, "EXACT")
        (o / "unet_off.src").write_text(f"BUILDSRC inputs={b} start=1 site_perf=0 kincore=1\n")
        ck("no_site_perf", *check("off", o, r, toy_geom), False, "EXACT")
        (o / "unet_off.src").unlink()
        ck("no_src", *check("off", o, r, toy_geom), False, "OUTSIDE")
    total = 12
    print(f"BFRESH SELFTEST {'PASS' if not fails else 'FAIL'} {total - len(set(fails) - {'dirty_count'})}/{total}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        return selftest()
    verbose = "-v" in argv
    args = [a for a in argv if a != "-v"]
    if len(args) != 2 or args[0] not in CFGS:
        print("usage: build_fresh.py <off|aot-kco> <outroot> [-v] | --selftest", file=sys.stderr)
        return 2
    line, ok = check(args[0], args[1], verbose=verbose)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
