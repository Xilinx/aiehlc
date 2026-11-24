"""M3.1.1.2c PW: k_dmafeat --dwfit static data-wait model (no logs needed)."""
import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))


def _kf(name):
    spec = importlib.util.spec_from_file_location(name, SRC / "costmodel" / "tools" / "k_dmafeat.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_dw_fit_nonneg_clip_and_loo():
    """w >= 0 with a signed offset, dw_hat clipped at 0, exact model recovered with LOO share 0, <= 3 params"""
    kf = _kf("k_dmafeat_dw1")
    xs = [dict(zip(kf.DW_FEATS, (v, 1.0, 2.0, 3.0, 0.5, 0.01 * v * v))) for v in range(3, 12)]
    pts = [(x, max(0.0, 0.9 * x["ddr_t_ms"] - 2.0), 1.0) for x in xs]
    p = kf.dw_select(pts)
    assert len(p["feats"]) + 1 <= 3 and all(w >= 0 for w in p["w"])
    assert p["loo_share"] < 1e-9 and kf.dw_errs(p, pts)[0] < 1e-9
    assert kf.dw_predict(p, dict(xs[0], ddr_t_ms=1.0)) == 0.0


def test_dw_audit_ignores_measured_fields():
    kf = _kf("k_dmafeat_dw2")
    s = dict(ddr_t=68.5381, maxch_t=15521.7, dom_bytes=2.94912e7, shim_ch=3.0, rw_ratio=0.98, bd_n=384.0,
             lock=63.4, stream=0.0, dwait_ms=63.4, site_ms=80.5)
    p = dict(feats=("ddr_t_ms", "bd_k"), w=[0.8, 5.0], b=0.09)
    h, bad = kf.dw_audit(p, s)
    assert bad == 0 and abs(h - (0.8 * 68.5381 + 5.0 * 0.384 + 0.09)) < 1e-9
    assert kf.run_name("/d/unet_off.c.log", 0) == "c"
