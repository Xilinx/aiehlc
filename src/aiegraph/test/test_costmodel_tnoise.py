"""M3.1.2.2 PN: script/perf/t_noise.py -- per-site T (site_ms) a/b/c jitter; runs under python -I."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / "script" / "perf" / "t_noise.py"
HDR = "cfg\trun\tpart\tsite\tkind\tlaunches\tsite_ms\tL_meas\tL_hat\n"


def run(*a):
    p = subprocess.run([sys.executable, "-I", str(PY), *a], capture_output=True, text=True)
    return p.returncode, p.stdout.splitlines()


def tsv(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(HDR + "".join("%s\t%s\t%s\t%s\tconv\t1\t%s\t0\t0\n" % r for r in rows))
    return str(p)


def rows_for(t_by_run):
    out = []
    for c in ("off", "aot-kco"):
        for r, t in t_by_run.items():
            out.append((c, "unet_%s.%s.log" % (c, r), "24", "conv2", "%.4f" % t))
            out.append((c, "unet_%s.%s.log" % (c, r), "25", "conv3", "1.0000"))
    return out


def test_selftest():
    rc, o = run("--selftest")
    assert rc == 0 and o[-1] == "TNOISE SELFTEST PASS"


def test_two_sets_site_order_and_margin(tmp_path):
    a = tsv(tmp_path, "a.tsv", rows_for({"a": 2.851, "b": 2.833, "c": 2.901}))
    b = tsv(tmp_path, "b.tsv", rows_for({"a": 2.915, "b": 2.914, "c": 2.892}))
    rc, o = run("m31223=" + a, "m3121=" + b, "--site", "aot-kco:conv2", "--margin", "0.05")
    assert rc == 0
    assert "TNOISE cfg=aot-kco set=m31223 sites=2 runs=3 spread_max=0.0240 spread_med=0.0120 worst=conv2 n_ge3pct=0" in o
    assert "TNOISE SITE cfg=aot-kco set=m31223 site=conv2 T=2.8510,2.8330,2.9010 spread=0.0240" in o
    assert "TNOISE SITE cfg=aot-kco set=m3121 site=conv2 T=2.9150,2.9140,2.8920 spread=0.0080" in o
    assert o[-1] == "TNOISE MARGIN spread_max=0.0240 margin=0.05 margin_ok=1"
    rc, o = run("m31223=" + a, "--margin", "0.02")
    assert rc == 0 and o[-1] == "TNOISE MARGIN spread_max=0.0240 margin=0.02 margin_ok=0"


def test_errors(tmp_path):
    good = rows_for({"a": 2.0, "b": 2.1})
    assert run("x=" + tsv(tmp_path, "d.tsv", good + [good[0]]))[0] == 1
    assert run("x=" + tsv(tmp_path, "m.tsv", good[:-1]))[0] == 1
    assert run("x=" + tsv(tmp_path, "g.tsv", good), "--site", "off:nope")[0] == 1
    assert run()[0] == 2
