"""costmodel M3.1.1.2c PV: k_ovl_check.py --npred / --delta-tsv (static ovl_hat vs counter ovl, kdelta_ovl)"""
import subprocess
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1] / "costmodel" / "tools"


def test_k_ovl_check_selftest_npred():
    """selftest PASS (n/n) and carries the npred cases (OVLSRC/OVLHAT/KDELTA math, rc=2 on a missing site)"""
    r = subprocess.run([sys.executable, "-I", str(PERF / "k_ovl_check.py"), "--selftest"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert any(ln.startswith("KOVL SELFTEST npred_ovlhat ok") for ln in lines)
    assert any(ln.startswith("KOVL SELFTEST npred_missing_rc2 ok") for ln in lines)
    last = lines[-1].split("(")[1].rstrip(")").split("/")
    assert lines[-1].startswith("KOVL SELFTEST PASS (") and last[0] == last[1]


def test_k_ovl_check_npred_rc2_without_tsv(tmp_path):
    """a npred TSV without the n_fit --predict header is rejected (ValueError)"""
    sys.path.insert(0, str(PERF))
    import k_ovl_check as k
    p = tmp_path / "n.tsv"
    p.write_text("cfg\trun\n")
    try:
        k.load_npred(str(p))
    except ValueError:
        return
    raise AssertionError("bad npred header accepted")
