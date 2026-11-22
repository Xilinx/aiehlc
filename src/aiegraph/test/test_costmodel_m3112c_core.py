"""M3.1.1.2c PS: k_diag.py --kdelta core (static core from kcyc_unet x launches, per-run K delta)."""
import importlib.util
import math
from pathlib import Path

PERF = Path(__file__).resolve().parents[3] / "script" / "perf"


def _kd():
    spec = importlib.util.spec_from_file_location("k_diag_m3112c", PERF / "k_diag.py")
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    return kd


def _site(kind, cid, part, n, cc, cls):
    return dict(site="%s%d" % (kind, cid), kind=kind, part=part, launches=n, cls=cls,
                perf=dict(perf_n=n, has_ovl=True, compute_cyc=cc))


def test_kdelta_core_cfg_static_and_delta():
    """core_hat only from kcyc (x launches); KCORE meas = median of runs; delta per run per launch"""
    kd = _kd()
    pred = {("conv", 0): dict(core_cyc=2.0e6, src="iss"), ("fa", 1): dict(core_cyc=1.0e6, src="iss")}
    runs = [[_site("conv", 0, 14, 1, c, "conv_big"), _site("fa", 1, 20, 2, f, "fa16")]
            for c, f in ((2.2e6, 2.0e6), (2.0e6, 2.5e6), (2.4e6, 1.8e6))]
    o = kd.kdelta_core_cfg("off", pred, runs)
    assert [r["core_hat_ms"] for r in o["static"]] == [2.0, 2.0]
    assert len(o["delta"]) == 6 and o["same_sites"]
    conv = [x for x in o["kcore"] if x[5] == "conv"][0]
    assert conv[1] == 2.2e6 and abs(conv[2] - 0.2 / 2.2) < 1e-12
    fb = [d for d in o["delta"] if d["run"] == "b" and d["site"] == "fa1"][0]
    assert abs(fb["delta_ms"] - (2.0 - 2.5) / 2) < 1e-12 and fb["compute_meas_ms"] == 2.5
    # a run without the ovl tail -> nan meas, still a delta row
    runs[2][0]["perf"]["has_ovl"] = False
    o2 = kd.kdelta_core_cfg("off", pred, runs)
    assert math.isnan([d for d in o2["delta"] if d["run"] == "c"][0]["delta_ms"])


def test_kdelta_selftest():
    kd = _kd()
    assert kd.selftest()
