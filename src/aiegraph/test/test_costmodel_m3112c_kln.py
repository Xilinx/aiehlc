"""M3.1.1.2c PG: kln_site --k-tsv / --kswap (external K, every *_ms per launch)."""
import importlib.util
from pathlib import Path

PERF = Path(__file__).resolve().parents[1] / "costmodel" / "tools"


def _ks():
    spec = importlib.util.spec_from_file_location("kln_site", PERF / "kln_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    rows = []
    for run, fit in (("a", True), ("b", False)):
        for part, (site, n, ms) in enumerate((("conv0", 1, 10.0), ("fa1", 4, 40.0), ("linear2", 2, 13.0))):
            rows.append(dict(cfg="off", run=run, part=part, site=site, launches=n, site_ms_log=ms + part,
                             L_meas=1.0, L_hat=1.1, N_meas=2.0, N_hat=1.9, fit=fit, log=None))
    return rows


def _ktsv(ks, rows, loo=True):
    kt = {(r["cfg"], r["run"], r["part"]): dict(site=r["site"], K_hat_ms=ks.k_meas(r, "log") / r["launches"],
                                                K_loo_ms=(ks.k_meas(r, "log") / r["launches"]
                                                          if loo and r["run"] == "a" else float("nan")),
                                                dw_hat_ms=float("nan")) for r in rows}
    return ks.kmap_tsv(rows, kt)


def test_kln_ktsv_kswap_units_loo(capsys):
    ks = _ks()
    rows = _rows()
    assert ks.selftest()
    capsys.readouterr()
    # per-launch K TSV x launches == the oracle K (launches=4 site included) -> T_hat == site_ms / launches
    km = _ktsv(ks, rows)
    for r in rows:
        assert abs(km[(r["cfg"], r["run"], r["part"])]["K"] - ks.k_meas(r, "log")) < 1e-9
    assert ks.khat_eval(rows, "log", km, "meastsv", 0.9, 0.2, 0.2)
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "KLNHAT ALL k=meastsv logs=1 sites=3 r2_min=1.0000 maxrel=0.0000 loo_max=0.0000 PASS"
    assert "KLNHAT cfg=off run=a k=meastsv sites=3 r2=1.0000 maxrel=0.0000 sum_rel=0.0000 PASS fit" in out
    # zero delta swap == oracle; LOO = the swapped K on run a
    sw = ks.kmap_swap(rows, "log", {}, {})
    assert ks.khat_eval(rows, "log", sw, "swap0", 0.9, 0.2, 0.2)
    assert capsys.readouterr().out.splitlines()[-1].endswith("loo_max=0.0000 PASS")
    # per-launch delta on the launches=4 site: K += 4 x delta
    d = {("off", "b", 1): [0.5]}
    sw = ks.kmap_swap(rows, "log", d, {("off", "b", 1): "fa1"})
    r = rows[4]
    assert abs(sw[("off", "b", 1)]["K"] - (ks.k_meas(r, "log") + 2.0)) < 1e-12
    # no K_loo -> loo_max literal nan, FAIL under --loo; PASS without --loo
    km = _ktsv(ks, rows, loo=False)
    assert not ks.khat_eval(rows, "log", km, "x", 0.9, 0.2, 0.2)
    out = capsys.readouterr().out.splitlines()
    assert "KLNHAT LOO cfg=off run=a k=x sites=3 loo_maxrel=nan FAIL" in out
    assert out[-1].endswith("loo_max=nan FAIL")
    assert ks.khat_eval(rows, "log", km, "x", 0.9, 0.2, None)
