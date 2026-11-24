"""M3.1.2.2 PKF: k_diag.py --khat-static --params (frozen K_hat params, no fit): same K_hat as the fit that
wrote the json, K_loo=nan, --params with --fit -> rc 2, audit covers khats_params_rows."""
import contextlib
import csv
import importlib.util
import io
import math
from pathlib import Path

from test_costmodel_m3112c_khats import _w

PERF = Path(__file__).resolve().parents[1] / "costmodel" / "tools"


def _kd():
    spec = importlib.util.spec_from_file_location("k_diag_khat_frozen", PERF / "k_diag.py")
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    return kd


def _inputs(kd, tmp_path):
    sites = [(10 + i, "linear%d" % i, "linear", 1 + i % 3) for i in range(8)]
    _w(tmp_path / "core.tsv", kd.KCORE_COLS,
       [("s", p, n, c, L, "%.6f" % (1.0 + 0.3 * i), "iss") for i, (p, n, c, L) in enumerate(sites)])
    _w(tmp_path / "dw.tsv", ("cfg", "part", "site", "cls", "launches", "ddr_t_ms", "dw_hat_ms"),
       [("s", p, n, c, L, 0.1, 0.2 + 0.05 * i) for i, (p, n, c, L) in enumerate(sites)])
    _w(tmp_path / "np.tsv", ("cfg", "run", "part", "site", "launches", "site_ms", "N_meas", "N_hat", "ovl_meas",
                             "ovl_hat"),
       [("s", r, p, n, L, 9.0, 1.0, 1.0, 0.1, 0.1 * (i % 3)) for r in "abc" for i, (p, n, c, L) in enumerate(sites)])
    return sites


def _rd(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_frozen_params_reproduce_fit(tmp_path, monkeypatch):
    kd = _kd()
    sites = _inputs(kd, tmp_path)

    class KL:
        @staticmethod
        def norm_run(x):
            return x if len(x) == 1 else "a"

        @staticmethod
        def parse_logs(specs):
            return [("s", "/x/unet_s.a.log")]

        @staticmethod
        def k_meas(r, src):
            return r["Km"]

    rows = [dict(cfg="s", run="a", part=p, site=n, launches=L, site_ms_log=10.0 * L, Km=(1.2 + 0.4 * i) * L)
            for i, (p, n, c, L) in enumerate(sites)]
    monkeypatch.setattr(kd, "khats_fit_rows", lambda a: (rows, True, KL))

    class A:
        core_tsv, dwhat, npred = str(tmp_path / "core.tsv"), str(tmp_path / "dw.tsv"), str(tmp_path / "np.tsv")
        khat_tsv, out_params, fit, params = str(tmp_path / "fit.tsv"), str(tmp_path / "k.json"), ["s=x"], None

    assert kd.khat_static(A) == 0

    class B(A):
        khat_tsv, out_params, fit, params, check, lpred = str(tmp_path / "rep.tsv"), None, None, A.out_params, None, "-"

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert kd.khat_static(B) == 0
    assert any(ln.startswith("KHATS PARAMS src=%s md5=" % A.out_params) and ln.endswith(" fit=0")
               for ln in buf.getvalue().splitlines())
    fit, rep = _rd(A.khat_tsv), _rd(B.khat_tsv)
    assert len(fit) == len(rep) == 24
    for x, y in zip(fit, rep):
        assert (x["cfg"], x["run"], x["part"], x["K_hat_ms"], x["dw_hat_ms"]) == \
               (y["cfg"], y["run"], y["part"], y["K_hat_ms"], y["dw_hat_ms"])
        assert math.isnan(float(y["K_loo_ms"]))


def test_params_with_fit_rc2_and_audit(tmp_path):
    kd = _kd()
    assert kd.main(["--khat-static", "--params", "x.json", "--fit", "s=a.log"]) == 2
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        assert kd.khats_audit()
    assert "fn=khats_params_rows" in buf.getvalue()
    assert buf.getvalue().splitlines()[-1] == "KHATS AUDIT meas_at_predict=0"
