"""costmodel M3.1.2.2.5 PO: n_fit --fit-ovl cls -- one overlap term per (cfg, kind), fitted on the M3121 a log only.
ovl_hat = cls[kind].k * (glue_hat + corr_ms) + cls[kind].c; N kinds/coefficients are copied, so N_hat is unchanged."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PERF = ROOT / "src" / "aiegraph" / "costmodel" / "tools"
PR = PERF.parent / "params"
J0 = PR / "n_unet_palmyra213.m3121.json"
JN = PR / "n_unet_palmyra213.m3125.json"
O0 = Path("/scratch/staff/huaj/tmp/costmodel_m3121")
D0 = Path("/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks/2026-11-20_M3121")
sys.path.insert(0, str(PERF))
import n_fit  # noqa: E402


def _synth(seed):
    rng = np.random.default_rng(seed)
    fr, perf = [], {}
    for i in range(30):
        kind = n_fit.KINDS[i % 3]
        g = float(rng.uniform(0.2, 5.0))
        o = (0.3, 0.8, 0.2)[i % 3] * g + 0.05 + float(rng.normal(0, 0.01))
        fr.append({"kind": kind, "part": i, "launches": 1, "MK": g})
        perf[i] = {"ovl_cyc": o * n_fit_aie_mhz() * 1e3, "perf_n": 1}
    ovl = {"kinds": {k: {"coef": {"MK": 1.0}, "d": 0.0} for k in n_fit.KINDS}}
    return fr, perf, ovl


def n_fit_aie_mhz():
    from k_diag import AIE_MHZ
    return AIE_MHZ


def test_cls_fit_deterministic_and_per_kind():
    fr, perf, ovl = _synth(1)
    a = n_fit.fit_ovl_cls(fr, perf, [0.0] * len(fr), ovl)
    b = n_fit.fit_ovl_cls(fr, perf, [0.0] * len(fr), ovl)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    o, loo = a
    assert o["form"] == "cls" and o["fit_runs"] == ["a"] and sorted(o["cls"]) == sorted(n_fit.KINDS)
    assert abs(o["cls"]["linear"]["k"] - 0.8) < 0.02 and abs(o["cls"]["fa"]["k"] - 0.2) < 0.02
    assert 0.0 <= loo < 0.2
    s = {"kind": "linear", "MK": 2.0}  # linear site: ovl_hat = k * (glue_hat + corr) + c
    pc = {"kinds": ovl["kinds"], "ovl": o}
    nh, oh = n_fit.npred_site(pc, s, corr_ms=0.5)
    c = o["cls"]["linear"]
    assert nh == 2.0 and oh == c["k"] * 2.5 + c["c"]


def test_committed_json_fit_runs_and_no_holdout_paths():
    t = JN.read_text()
    d = json.loads(t)
    assert sorted(d) == ["aot-kco", "off"]  # no 'cfg' wrapper
    for c in ("off", "aot-kco"):
        assert d[c]["ovl"]["fit_runs"] == ["a"] and d[c]["ovl"]["form"] == "cls"
        assert sorted(d[c]["ovl"]["cls"]) == sorted(n_fit.KINDS)
    assert "m31223" not in t.lower() and "m31225" not in t.lower()


def test_n_part_unchanged():
    """everything but 'ovl' is identical to the M3121 params -> N_hat bit-identical; glue kinds also frozen"""
    d0, d1 = json.loads(J0.read_text()), json.loads(JN.read_text())
    for c in ("off", "aot-kco"):
        a = {k: v for k, v in d0[c].items() if k != "ovl"}
        b = {k: v for k, v in d1[c].items() if k != "ovl"}
        assert a == b
        assert d0[c]["ovl"]["kinds"] == d1[c]["ovl"]["kinds"]
        s = {f: 1000.0 * (i + 1) for i, f in enumerate(n_fit.FEATS)}
        for kind in n_fit.KINDS:
            s["kind"] = kind
            assert n_fit.npred_site(d0[c], s)[0] == n_fit.npred_site(d1[c], s)[0]


@pytest.mark.skipif(not (O0 / "unet_off").is_dir() or not (D0 / "unet_off.a.log").is_file(), reason="no M3121 data")
def test_refit_reproduces_committed_json(tmp_path):
    out = tmp_path / "n.json"
    p = subprocess.run([sys.executable, "-I", str(PERF / "n_fit.py"), "--fit-ovl", "cls", "--params", str(J0),
                        "--out-params", str(out), "--build-root", str(O0), "off=%s" % (D0 / "unet_off.a.log"),
                        "aot-kco=%s" % (D0 / "unet_aot-kco.a.log")], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out.read_bytes() == JN.read_bytes()
    lo = [ln for ln in p.stdout.splitlines() if ln.startswith("OVLLOO ")]
    assert [ln.split()[1] for ln in lo] == ["cfg=off", "cfg=aot-kco"]
