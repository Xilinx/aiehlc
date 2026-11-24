###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""n_fit.py --predict (costmodel M3.1.2.1 PN): N_hat / ovl_hat read only static site features (no log field)."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

PERF = Path(__file__).resolve().parents[1] / "tools"


def _nf():
    spec = importlib.util.spec_from_file_location("n_fit_npred", PERF / "n_fit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# two launch sites + one host-only attn site; every FEATS key + derive() inputs, nothing from a log
_SITES = [
    dict(kind="conv", cid=0, launches=1, up="arm", down="arm", kernels=["k0"]),
    dict(kind="linear", cid=1, launches=2, up="aie", down="arm", kernels=["k1", "k1"]),
    dict(kind="attn", cid=2, launches=0, up="arm", down="arm", kernels=[]),
]
for _i, _s in enumerate(_SITES):
    for _f in ("MK", "MN", "wbytes", "cp_w", "bdcfg_w", "rr", "rr_pair", "kfirst", "in_pack_bytes",
               "out_scatter_bytes", "relayout", "pack_slow_elems"):
        _s[_f] = (_i + 1) * 1000 if _f in ("MK", "MN", "wbytes", "in_pack_bytes", "out_scatter_bytes") else _i
    _s["kswitch"] = _s["launches"]

_PRM = {"aot-kco": {"attn_host_ms": 0.5, "kinds": {
    "conv": {"coef": {"MK": 1e-3, "in_pack_bytes": 2e-4}, "d": 0.1},
    "linear": {"coef": {"launches": 0.3, "out_scatter_bytes": 1e-4}, "d": 0.2},
    "fa": {"coef": {}, "d": 1.0}},
    "ovl": {"ovl_k": 0.5, "kinds": {"conv": {"coef": {"MK": 5e-4}, "d": 0.0},
                                    "linear": {"coef": {"launches": 0.1}, "d": 0.0}}}}}


def _log(scale):
    """synthetic board log of _SITES; scale multiplies every measured time (static predictions must not move)"""
    g = lambda x: "%.3f" % (x * scale)  # noqa: E731
    return "\n".join([
        "PROF part=3 op=aie_host_conv0 ms=%s bf16_expand_ms=0.001 site_ms=%s aie_ms=%s" % (g(1.0), g(3.0), g(1.0)),
        "PROF aie_host_split part=3 kind=conv pack_ms=0.1 scatter_ms=0.1 dma_ms=1.0 wait_ms=1.0 glue_ms=%s "
        "flush_ms=%s inval_ms=%s launches=1" % (g(0.5), g(0.1), g(0.05)),
        "PROF aie_site_perf part=3 kind=conv launches=1 perf_n=1 io_act_cyc=10 lock_stall_cyc=1 stream_stall_cyc=0 "
        "pre_io_act_cyc=0 snap_gap_cyc=0 compute_cyc=100 ovl_cyc=%d dact_cyc=5" % int(400000 * scale),
        "PROF part=5 op=aie_host_linear1 ms=%s bf16_expand_ms=0.001 site_ms=%s aie_ms=%s" % (g(1.2), g(4.0), g(1.5)),
        "PROF aie_host_split part=5 kind=linear pack_ms=0.1 scatter_ms=0.1 dma_ms=1.0 wait_ms=1.0 glue_ms=%s "
        "flush_ms=%s inval_ms=%s launches=2" % (g(0.6), g(0.1), g(0.05)),
        "PROF aie_site_perf part=5 kind=linear launches=2 perf_n=2 io_act_cyc=10 lock_stall_cyc=1 stream_stall_cyc=0 "
        "pre_io_act_cyc=0 snap_gap_cyc=0 compute_cyc=100 ovl_cyc=%d dact_cyc=5" % int(300000 * scale),
        "PROF part=7 op=aie_host_attn2 ms=%s bf16_expand_ms=0.001 site_ms=%s aie_ms=0.000" % (g(0.5), g(0.5)),
        "LAUNCH warm ms_per_launch=1.0 launches=3 aie_wall_ms=3.0",
        "LAUNCH split_detail calls kload=3 host_glue_bdcfg=%s" % ("%.6f" % ((0.5 + 0.6) * scale / 3)),
        "CMP unet board=palmyra213 data=x aie_launches=3 coreen_calls=3 run_rc=0 PASS", ""])


@pytest.fixture
def nfp(monkeypatch, tmp_path):
    nf = _nf()
    monkeypatch.setattr(nf, "n_sites", lambda build, blog: [dict(s) for s in _SITES])
    seen = []
    real = nf.predict

    def spy(p, r):
        seen.append(set(r))
        return real(p, r)
    monkeypatch.setattr(nf, "predict", spy)
    pj = tmp_path / "n.json"
    pj.write_text(json.dumps(_PRM))
    logs = []
    for run, sc in (("a", 1.0), ("b", 2.0), ("c", 0.5)):
        lg = tmp_path / ("unet_aot-kco.%s.log" % run)
        lg.write_text(_log(sc))
        logs.append(str(lg))
    return nf, seen, pj, logs, tmp_path


def _run(nf, capsys, pj, logs, tsv, *extra):
    rc = nf.main(["--predict", "--ovl", "--refit-a", "no", "--params", str(pj), "--tsv", str(tsv),
                  "--build-root", "/nonexistent", *extra, "aot-kco=" + ",".join(logs)])
    return rc, capsys.readouterr().out


def test_npred_static_inputs_have_no_log_fields(nfp, capsys):
    """every predict() call of --predict (N_hat and the glue part of ovl_hat) gets a row without log fields"""
    nf, seen, pj, logs, tmp = nfp
    rc, out = _run(nf, capsys, pj, logs, tmp / "sub" / "npred.tsv")
    assert re.search(r"^OVLP cfg=aot-kco static=1 sites=2 r2=-?\d+\.\d{4} maxrel=\d+\.\d{4}$", out, re.M), out
    assert re.search(r"^NSRC cfg=aot-kco log=\S+ board=palmyra213 launches=3 run_rc=0 sites=2$", out, re.M), out
    assert seen and all(not (k & nf.LOG_FIELDS) for k in seen), [sorted(k & nf.LOG_FIELDS) for k in seen]
    assert nf.LOG_FIELDS >= {"N", "glue", "ms", "site_ms", "aie_ms", "part"}
    with pytest.raises(ValueError):
        nf.npred_site(_PRM["aot-kco"], dict(nf.static_rows("/x")[0], N=1.0))
    rows = (tmp / "sub" / "npred.tsv").read_text().splitlines()
    assert rows[0].split("\t") == list(nf.TSV_COLS) and len(rows) == 1 + 3 * 2


def test_npred_static_hat_independent_of_log(nfp, capsys):
    """runs a/b/c differ only in the measured times (x1, x2, x0.5): N_hat and ovl_hat identical per site,
    equal to the params applied to the static features"""
    nf, _seen, pj, logs, tmp = nfp
    _run(nf, capsys, pj, logs, tmp / "npred.tsv")
    cols = list(nf.TSV_COLS)
    by = {}
    for ln in (tmp / "npred.tsv").read_text().splitlines()[1:]:
        r = dict(zip(cols, ln.split("\t")))
        by.setdefault(r["site"], set()).add((r["N_hat"], r["ovl_hat"]))
        assert r["N_meas"] != r["N_hat"]
    assert all(len(v) == 1 for v in by.values()), by
    s0 = _SITES[0]
    n0 = 1e-3 * s0["MK"] + 2e-4 * s0["in_pack_bytes"] + 0.1
    assert by["conv0"] == {("%.4f" % n0, "%.4f" % (0.5 * 5e-4 * s0["MK"]))}


def test_npred_static_refit_only_a(nfp, capsys, monkeypatch):
    """a failing frozen cfg is refit once on its a log only (nrefit=1, <params>.m3121.json); b/c never fitted"""
    nf, _seen, pj, logs, tmp = nfp
    fitted = []
    real = nf._refit_cfg
    kinds = {k: dict(v, method="minimax", n=1, loo_a=0.0) for k, v in _PRM["aot-kco"]["kinds"].items()}
    monkeypatch.setattr(nf, "_refit_cfg", lambda rows, m="minimax": fitted.append(rows) or {
        "kinds": kinds, "attn_host_ms": 0.5})
    monkeypatch.setattr(nf, "_fit_ovl", lambda fr, perf, corr: _PRM["aot-kco"]["ovl"])
    rc = nf.main(["--predict", "--ovl", "--params", str(pj), "--tsv", str(tmp / "t.tsv"), "--build-root", "/x",
                  "aot-kco=" + ",".join(logs)])
    out = capsys.readouterr().out
    assert len(fitted) == 1 and fitted[0][0]["N"] == pytest.approx(1.0 + 0.5 + 0.1 + 0.05), out
    assert re.search(r"^NCHK ALL logs=3 sites=6 nrefit=1 N_maxrel=\d\.\d{4} Tshare_max=\d\.\d{4} (PASS|FAIL)$",
                     out, re.M), out
    assert (tmp / "n.m3121.json").is_file() and real is not None
    assert rc in (0, 1)
