###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Tests for build_unet_cfg.sh build-input stamping (M3.1.2 PF): BUILDCMD inputs=/dirty=,
dirty refusal (rc=3), <outroot>/unet_<cfg>.src and --check-fresh.

Each test copies the script into a throw-away git repo with the same layout, so the real
worktree's state (other people's uncommitted work) does not matter.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "example/aiegraph/unet_bk_sdm_tiny/build_unet_cfg.sh"
UDIR = "example/aiegraph/unet_bk_sdm_tiny"

STUB_PY = """#!/bin/bash
# stand-in for python3 build_unet.py: create <--out>/main.elf
out=
while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
mkdir -p "$out" && echo elf > "$out/main.elf"
"""


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    u = r / UDIR
    u.mkdir(parents=True)
    shutil.copy(SCRIPT, u / "build_unet_cfg.sh")
    (u / "partition_p4a.json").write_text('{"N": 216200}\n')
    (u / "a78_layer_cflags.txt").write_text("-O2\n")
    for d in ("src/mlir/runtime", "src/aiegraph/runtime", "src/aiednn", "script/perf"):
        (r / d).mkdir(parents=True)
        (r / d / "f.txt").write_text("x\n")
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    stub = tmp_path / "stubpy"
    stub.write_text(STUB_PY)
    stub.chmod(0o755)
    return r, stub


def run(repo, *a, env_py=None):
    r, stub = repo
    env = dict(os.environ, PY=str(env_py or stub))
    return subprocess.run(["bash", str(r / UDIR / "build_unet_cfg.sh"), *a], capture_output=True, text=True, env=env)


def inputs_head(r):
    return git(r, "log", "-1", "--format=%h", "--", "src/mlir/runtime", "src/aiegraph/runtime", "src/aiednn",
               UDIR).strip()


def test_build_fresh_dryrun_stamp(repo, tmp_path):
    r, _ = repo
    p = run(repo, "off", str(tmp_path / "o"), "--site-perf", "--dry-run")
    assert p.returncode == 0, p.stderr
    last = p.stdout.strip().splitlines()[-1]
    assert last == ("BUILDCMD cfg=off kincore=1 attn_fa=1 partition=partition_p4a.json N=216200 args_ok=1 "
                    f"site_perf=1 out={tmp_path}/o/unet_off inputs={inputs_head(r)} dirty=0")
    (r / "src/aiednn/f.txt").write_text("y\n")
    (r / "src/aiednn/new.txt").write_text("n\n")
    p = run(repo, "off", str(tmp_path / "o"), "--dry-run")
    assert p.returncode == 0
    assert p.stdout.strip().endswith(f"site_perf=0 out={tmp_path}/o/unet_off inputs={inputs_head(r)} dirty=2")
    assert not (tmp_path / "o").exists()


def test_build_fresh_dirty_refuses(repo, tmp_path):
    r, _ = repo
    (r / "src/mlir/runtime/f.txt").write_text("y\n")
    p = run(repo, "off", str(tmp_path / "o"), "--site-perf")
    assert p.returncode == 3
    assert not (tmp_path / "o").exists()


def test_build_fresh_real_build_then_check(repo, tmp_path):
    r, _ = repo
    o = tmp_path / "o"
    p = run(repo, "off", str(o), "--site-perf")
    assert p.returncode == 0, p.stderr
    h = inputs_head(r)
    assert p.stdout.strip().endswith(f"inputs={h} dirty=0 rc=0")
    src = (o / "unet_off.src").read_text()
    assert re.fullmatch(rf"BUILDSRC inputs={h} start=\d+ site_perf=1 kincore=1\n", src)
    c = run(repo, "off", str(o), "--check-fresh")
    assert c.returncode == 0, c.stdout
    assert c.stdout.strip() == (f"BUILD FRESH cfg=off inputs={h} head_inputs={h} dirty=0 elf_newer=1 "
                                "site_perf=1 kincore=1 PASS")
    # script/perf is not a build input: committing there keeps it FRESH
    (r / "script/perf/f.txt").write_text("k_diag\n")
    git(r, "commit", "-qam", "perf only")
    assert run(repo, "off", str(o), "--check-fresh").returncode == 0
    # uncommitted change to an input -> FAIL dirty=1
    (r / "src/aiegraph/runtime/f.txt").write_text("y\n")
    c = run(repo, "off", str(o), "--check-fresh")
    assert c.returncode == 1 and " dirty=1 " in c.stdout and c.stdout.strip().endswith(" FAIL")
    # committed input change -> head_inputs moves -> FAIL
    git(r, "commit", "-qam", "rt change")
    h2 = inputs_head(r)
    c = run(repo, "off", str(o), "--check-fresh")
    assert c.returncode == 1
    assert f"inputs={h} head_inputs={h2} dirty=0 " in c.stdout and c.stdout.strip().endswith(" FAIL")


def test_build_fresh_fail_modes(repo, tmp_path):
    r, _ = repo
    h = inputs_head(r)
    o = tmp_path / "o"
    # no .src at all (pre-PF build) -> FAIL rc=1
    c = run(repo, "off", str(o), "--check-fresh")
    assert c.returncode == 1
    assert c.stdout.strip() == (f"BUILD FRESH cfg=off inputs=none head_inputs={h} dirty=0 elf_newer=0 "
                                "site_perf=0 kincore=0 FAIL")
    # built without --site-perf -> FAIL
    assert run(repo, "off", str(o)).returncode == 0
    c = run(repo, "off", str(o), "--check-fresh")
    assert c.returncode == 1 and " site_perf=0 kincore=1 FAIL" in c.stdout
    # main.elf older than the recorded start -> elf_newer=0
    (o / "unet_off.src").write_text(f"BUILDSRC inputs={h} start=4000000000 site_perf=1 kincore=1\n")
    c = run(repo, "off", str(o), "--check-fresh")
    assert c.returncode == 1 and " elf_newer=0 " in c.stdout
    # missing main.elf -> elf_newer=0
    (o / "unet_off.src").write_text(f"BUILDSRC inputs={h} start=1 site_perf=1 kincore=1\n")
    assert run(repo, "off", str(o), "--check-fresh").returncode == 0
    (o / "unet_off/main.elf").unlink()
    c = run(repo, "off", str(o), "--check-fresh")
    assert c.returncode == 1 and " elf_newer=0 " in c.stdout
    # --dry-run and --check-fresh together is a usage error
    assert run(repo, "off", str(o), "--dry-run", "--check-fresh").returncode == 2


# ---- script/perf/build_fresh.py (M3.1.2.2 PF): content-based freshness, test_*.py out of scope,
# inputs != head_inputs -> kacc_geom-only hunks + same kacc_geom results -> SAME (toy unet_aie.py in a temp repo)
import importlib.util  # noqa: E402

_BF = REPO / "script/perf/build_fresh.py"
_spec = importlib.util.spec_from_file_location("build_fresh", _BF)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)


def _check(r, o):
    return bf.check("off", o, r, bf.toy_geom)


def test_build_fresh_exact_test_only(tmp_path):
    r, o, b = bf.make_case(tmp_path)
    bf.commit_file(r, f"{UDIR}/test_unet.py", "def test_x():\n    assert 1\n", "test only")
    line, ok = _check(r, o)
    assert ok and line == (f"BUILD FRESH cfg=off inputs={b} head_inputs={b} dirty=0 elf_newer=1 "
                           "site_perf=1 kincore=1 equiv=EXACT PASS")


def test_build_fresh_equiv_same(tmp_path):
    r, o, b = bf.make_case(tmp_path)
    bf.commit_file(r, bf.UA_REL, bf.UA_SAME)
    line, ok = _check(r, o)
    h = git(r, "log", "-1", "--format=%h").strip()
    assert ok and line.endswith(f"inputs={b} head_inputs={h} dirty=0 elf_newer=1 site_perf=1 kincore=1 "
                                "equiv=SAME PASS")


def test_build_fresh_equiv_diff(tmp_path):
    r, o, b = bf.make_case(tmp_path)
    bf.commit_file(r, bf.UA_REL, bf.UA_DIFF)
    line, ok = _check(r, o)
    assert not ok and line.endswith(" equiv=DIFF FAIL")


def test_build_fresh_outside(tmp_path):
    r, o, b = bf.make_case(tmp_path)
    bf.commit_file(r, bf.UA_REL, bf.UA_OUTSIDE)
    line, ok = _check(r, o)
    assert not ok and line.endswith(" equiv=OUTSIDE FAIL")
    r, o, b = bf.make_case(tmp_path / "2")
    bf.commit_file(r, "src/aiednn/f.txt", "y\n")
    line, ok = _check(r, o)
    assert not ok and line.endswith(" equiv=OUTSIDE FAIL")


def test_build_fresh_dirty(tmp_path):
    r, o, b = bf.make_case(tmp_path)
    (r / "src/aiegraph/runtime/f.txt").write_text("dirty\n")
    line, ok = _check(r, o)
    assert not ok and " dirty=1 " in line and line.endswith(" equiv=EXACT FAIL")


def test_build_fresh_cli_rc(tmp_path):
    r, o, b = bf.make_case(tmp_path)
    (o / "unet_off.src").unlink()
    assert bf.main(["off", str(o)]) == 1      # real TOP, no .src -> FAIL rc=1
    assert bf.main(["bogus", str(o)]) == 2
