"""M3.1.2.2.3.2 PFZ: script/perf/kln_frozen.sh -- rc rules and BUILD_ROOT/kln is never written.

The six steps are stubbed through KLN_PY (a fake interpreter that logs its args and exits with
STUB_RC_<script stem>), so this test needs no build dir or board logs.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SH = ROOT / "script" / "perf" / "kln_frozen.sh"
STEPS = ("m3_site", "n_fit", "k_dmafeat", "kln_site")  # k_diag runs twice (core, khat) -> STUB_RC_k_diag

STUB = """#!/usr/bin/env bash
s=$(basename "$2" .py); echo "$s $*" >> "$STUB_LOG"
v=STUB_RC_$s; exit ${!v:-0}
"""


@pytest.fixture
def env(tmp_path):
    b, d = tmp_path / "build", tmp_path / "logs"
    (b / "kln").mkdir(parents=True)
    d.mkdir()
    for c in ("off", "aot-kco"):
        for r in "abc":
            (d / ("unet_%s.%s.log" % (c, r))).write_text("x\n")
    py = tmp_path / "fakepy"
    py.write_text(STUB)
    py.chmod(0o755)
    e = dict(os.environ, KLN_PY=str(py), STUB_LOG=str(tmp_path / "stub.log"))
    return b, d, tmp_path / "out", e


def _run(b, d, o, e, **rcs):
    e = dict(e, **{"STUB_RC_" + k: str(v) for k, v in rcs.items()})
    p = subprocess.run(["bash", str(SH), str(b), str(d), str(o)], env=e, capture_output=True, text=True)
    return p.returncode, p.stdout


@pytest.mark.parametrize("rcs,want", [
    ({}, 0),
    ({"kln_site": 1}, 1),
    ({"m3_site": 1, "n_fit": 1}, 0),      # only kln_site's rc=1 counts as a metric miss
    ({"k_diag": 2}, 2),
    ({"m3_site": 3}, 2),
    ({"kln_site": 1, "k_dmafeat": 2}, 2),  # a step >1 wins over the metric miss
    ({"kln_site": 2}, 2),
])
def test_rc_rules(env, rcs, want):
    b, d, o, e = env
    rc, out = _run(b, d, o, e, **rcs)
    assert rc == want, out
    assert out.count("KLNFROZEN PARAMS file=") == 5
    last = [l for l in out.splitlines() if l.startswith("KLNFROZEN rc_l=")]
    assert len(last) == 1 and "rc_kln=%d" % rcs.get("kln_site", 0) in last[0]


def test_never_writes_build_root_kln(env):
    b, d, o, e = env
    before = sorted(os.listdir(b / "kln"))
    rc, _ = _run(b, d, o, e)
    assert rc == 0 and sorted(os.listdir(b / "kln")) == before
    calls = Path(e["STUB_LOG"]).read_text().splitlines()
    assert [c.split()[0] for c in calls] == ["m3_site", "n_fit", "k_diag", "k_dmafeat", "k_diag", "kln_site"]
    for c in calls:
        assert str(b / "kln") not in c
    assert "--refit-a no" in calls[1] and "--dwpredict" in calls[3] and "--params" in calls[4]
    assert "--loo" not in calls[5] and "--k-label frozen" in calls[5]
    # all 6 logs are --check (no --fit): ALL covers 6 logs / 828 sites; per-site TSV in OUTDIR
    assert "--fit" not in calls[5].split() and "--site-tsv %s" % (o / "site_pairs.tsv") in calls[5]
    chk = calls[5].split("--check", 1)[1]
    assert all("unet_%s.%s.log" % (c, r) in chk for c in ("off", "aot-kco") for r in "abc")


def test_k_json_env_default_and_override(env, tmp_path):
    b, d, o, e = env
    rc, out = _run(b, d, o, e)
    assert rc == 0 and "KLNFROZEN PARAMS file=src/costmodel/params/k_palmyra213.m3112c.json md5=" in out
    alt = tmp_path / "k_alt.json"
    alt.write_text((ROOT / "src/costmodel/params/k_palmyra213.m3112c.json").read_text())
    rc, out = _run(b, d, o, dict(e, KLN_K_JSON=str(alt)))
    assert rc == 0 and "KLNFROZEN PARAMS file=%s md5=" % alt in out
    calls = Path(e["STUB_LOG"]).read_text().splitlines()
    assert "--params %s" % alt in calls[-2]


def test_k_json_missing_rc2(env):
    b, d, o, e = env
    rc, out = _run(b, d, o, dict(e, KLN_K_JSON="/nonexistent.json"))
    assert rc == 2 and "missing params /nonexistent.json" in out and not Path(e["STUB_LOG"]).exists()


def test_outdir_under_build_root_kln_refused(env):
    b, d, _, e = env
    rc, out = _run(b, d, b / "kln" / "x", e)
    assert rc == 2 and not (b / "kln" / "x").exists() and not Path(e["STUB_LOG"]).exists()
    rc, _ = _run(b, d, b / "kln", e)
    assert rc == 2


def test_missing_log_rc2(env):
    b, d, o, e = env
    (d / "unet_aot-kco.c.log").unlink()
    rc, out = _run(b, d, o, e)
    assert rc == 2 and "missing log" in out
