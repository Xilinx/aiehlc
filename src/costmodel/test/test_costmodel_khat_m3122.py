"""M3.1.2.2 PK (M3122): k_diag --khat-static refit -- clamp row (body >= 0 on every fit site), per-class form with
cfg-form fallback (old jsons reproduce), --khat-obj t reads only the PREDICTED L_hat / N_hat."""
import csv
import importlib.util
import json
from pathlib import Path

from test_costmodel_m3112c_khats import _w

PERF = Path(__file__).resolve().parents[1] / "tools"


def _kd():
    spec = importlib.util.spec_from_file_location("k_diag_khat_m3122", PERF / "k_diag.py")
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    return kd


def _g(kd, pts):
    """sites (core_hat, dw_hat, ovl_hat, launches, K per launch), w = 1"""
    return [dict(part=i, cls="x", f=kd.khats_feat(dict(launches=n, core_hat_ms=c * n, dw_hat_ms=d, ovl_hat=o * n)),
                 K=k, w=1.0) for i, (c, d, o, n, k) in enumerate(pts)]


def test_clamp_row_keeps_body_nonnegative():
    """data that an unclamped LP fits with a negative body (t0 compensating) -> with the clamp row no fit site
    has max(0, body) active, and the LP still equals what khats_predict computes on the fit sites"""
    kd = _kd()
    # small-core sites have K well above alpha*core and large-ovl sites pull beta up: unclamped body < 0 there
    pts = [(0.1, 0.0, 2.0, 1, 0.30), (0.2, 0.0, 2.5, 1, 0.32), (2.0, 0.0, 0.1, 1, 2.1), (3.0, 0.0, 0.2, 1, 3.1),
           (1.0, 0.0, 1.0, 1, 0.9), (0.5, 0.0, 3.0, 1, 0.25)]
    g = _g(kd, pts)
    for form in kd.KHATS_FORMS:
        p = kd.khats_fit(g, form)
        assert all(p[k] >= 0 for k in kd.KHATS_PNAMES)
        bodies = [kd.khats_body(p, form, s["f"]) for s in g]
        assert min(bodies) >= -1e-9, (form, bodies)
        # body >= 0 -> predict == LP linear model (no hidden clamp)
        for s, b in zip(g, bodies):
            lin = max(b, 0.0) + p["t0"] + p["kfirst"] * s["f"]["inv_launches"]
            assert abs(kd.khats_predict(p, form, s["f"]) - lin) < 1e-12
    # the clamp is a row, not a bound: beta / t0 may still be 0 or > 0
    p = kd._khats_lp(g, "sum")
    assert p is not None


def test_class_form_fallback_reproduces_old_json(tmp_path):
    kd = _kd()
    base = dict(alpha=1.2, gamma=0.8, beta=0.3, t0=0.01, kfirst=0.02)
    assert kd.khats_cls_form(dict(base), "max") == "max" and kd.khats_cls_form(dict(base), "sum") == "sum"
    assert kd.khats_cls_form(dict(base, form="max"), "sum") == "max"
    stat = {"s": {1: dict(site="x1", cls="a", launches=2, core_hat_ms=2.0, dw_hat_ms=1.5, ovl_hat=0.2),
                  2: dict(site="x2", cls="b", launches=1, core_hat_ms=0.5, dw_hat_ms=0.9, ovl_hat=0.1)}}
    old = {"cfg": {"s": {"form": "max", "classes": {"a": dict(base), "b": dict(base)}}}}
    new = {"cfg": {"s": {"form": "sum", "classes": {"a": dict(base, form="max"), "b": dict(base, form="max")}}}}
    ro, rn = kd.khats_params_rows(old, stat, ["a"], ["s"]), kd.khats_params_rows(new, stat, ["a"], ["s"])
    assert [r[4] for r in ro] == [r[4] for r in rn]          # class form == old cfg form -> identical
    for (_c, _r, _p, s, kh, _l) in ro:
        assert kh == kd.khats_predict(base, "max", kd.khats_feat(s))
    mix = {"cfg": {"s": {"form": "max", "classes": {"a": dict(base, form="sum"), "b": dict(base)}}}}
    rm = {r[2]: r[4] for r in kd.khats_params_rows(mix, stat, ["a"], ["s"])}
    assert rm[1] == kd.khats_predict(base, "sum", kd.khats_feat(stat["s"][1]))
    assert rm[2] == kd.khats_predict(base, "max", kd.khats_feat(stat["s"][2]))   # fallback to cfg form


def test_obj_t_reads_only_predicted_l_n(tmp_path, monkeypatch):
    kd = _kd()
    r = dict(site_ms_log=10.0, L_hat=2.0, N_hat=1.0, launches=4, L_meas=3.0, N_meas=0.5)
    assert kd.khats_target_t(r) == (10.0 - 2.0 - 1.0) / 4
    assert kd.khats_target_t(dict(r, L_meas=1e9, N_meas=-1e9, K_meas=7.0)) == kd.khats_target_t(r)
    assert kd.khats_target_t in kd.KHATS_TARGET_FNS and kd.khats_audit()

    sites = [(10 + i, "linear%d" % i, "linear", 1 + i % 3) for i in range(8)]
    _w(tmp_path / "core.tsv", kd.KCORE_COLS,
       [("s", p, n, c, L, "%.6f" % (1.0 + 0.3 * i), "iss") for i, (p, n, c, L) in enumerate(sites)])
    _w(tmp_path / "dw.tsv", ("cfg", "part", "site", "cls", "launches", "ddr_t_ms", "dw_hat_ms"),
       [("s", p, n, c, L, 0.1, 0.2 + 0.05 * i) for i, (p, n, c, L) in enumerate(sites)])
    _w(tmp_path / "np.tsv", ("cfg", "run", "part", "site", "launches", "site_ms", "N_meas", "N_hat", "ovl_meas",
                             "ovl_hat"),
       [("s", r_, p, n, L, 9.0, 1.0, 1.0, 0.1, 0.1 * (i % 3)) for r_ in "abc" for i, (p, n, c, L) in enumerate(sites)])

    class KL:
        @staticmethod
        def norm_run(x):
            return x if len(x) == 1 else "a"

        @staticmethod
        def parse_logs(specs):
            return [("s", "/x/unet_s.a.log")]

        @staticmethod
        def k_meas(r_, src):
            raise AssertionError("obj=t must not read K_meas / L_meas / N_meas")

    def rows(lmeas):
        return [dict(cfg="s", run="a", part=p, site=n, launches=L, site_ms_log=10.0 * L, L_hat=0.5 * L,
                     N_hat=0.2 * L + 0.1 * i, L_meas=lmeas, N_meas=lmeas) for i, (p, n, c, L) in enumerate(sites)]

    outs = []
    for lm in (0.0, 1e6):
        monkeypatch.setattr(kd, "khats_fit_rows", lambda a, lm=lm: (rows(lm), True, KL))

        class A:
            core_tsv, dwhat, npred = str(tmp_path / "core.tsv"), str(tmp_path / "dw.tsv"), str(tmp_path / "np.tsv")
            khat_tsv, out_params, fit, khat_obj = (str(tmp_path / ("k%d.tsv" % lm)), str(tmp_path / "k.json"),
                                                   ["s=x"], "t")

        assert kd.khat_static(A) == 0
        outs.append(Path(A.khat_tsv).read_text())
        d = json.loads(Path(A.out_params).read_text())
        assert all(set(v) <= set(kd.KHATS_PNAMES) | {"form"} and v["form"] in kd.KHATS_FORMS
                   for v in d["cfg"]["s"]["classes"].values())
        assert all(Path(x).is_absolute() for x in d["inputs"].values())
    assert outs[0] == outs[1]   # measured L / N never change the obj=t fit
    t = list(csv.DictReader(open(tmp_path / "k0.tsv"), delimiter="\t"))
    assert len(t) == 3 * len(sites)
