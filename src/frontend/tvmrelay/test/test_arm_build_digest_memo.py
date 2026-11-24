"""F23: per-process memo of _digest_file (params.bin + layer weight files).

The memo may only answer while the file is provably unchanged: same
(dev, ino, size, mtime_ns, ctime_ns) AND the file was already >2 s old when it
was hashed ("racy clean" rule), so a same-tick same-size rewrite cannot hide.

Run:  python -m pytest -q src/frontend/tvmrelay/test/test_arm_build_digest_memo.py
"""
import hashlib
import os
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frontend.tvmrelay import arm_build as ab  # noqa: E402


def _sha1(b):
    return hashlib.sha1(b).hexdigest()


def _count_opens(monkeypatch):
    n = {"open": 0}
    real = open

    def counting(p, *a, **k):
        n["open"] += 1
        return real(p, *a, **k)
    monkeypatch.setattr(ab, "open", counting, raising=False)
    return n


def test_fresh_file_not_memoised(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "_DIGEST_MEMO", {})
    f = tmp_path / "w.bin"
    f.write_bytes(b"a" * 1000)
    n = _count_opens(monkeypatch)
    assert ab._digest_file(f) == _sha1(b"a" * 1000)
    assert ab._digest_file(f) == _sha1(b"a" * 1000)
    assert n["open"] == 2                           # just written -> racy -> read twice
    assert str(f) not in ab._DIGEST_MEMO


def test_old_file_memoised_and_invalidated(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "_DIGEST_MEMO", {})
    monkeypatch.setattr(ab, "_DIGEST_RACY_NS", -10**18)   # treat every file as old
    f = tmp_path / "w.bin"
    f.write_bytes(b"a" * 1000)
    n = _count_opens(monkeypatch)
    d = ab._digest_file(f)
    assert ab._digest_file(f) == d and n["open"] == 1     # second call from memo
    st = f.stat()
    time.sleep(0.05)   # past one coarse-clock tick: within a tick only the 2 s rule protects
    with open(f, "r+b") as fh:                            # same size, in place
        fh.write(b"b")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))      # even with mtime put back
    assert ab._digest_file(f) == _sha1(b"b" + b"a" * 999)  # ctime moved -> re-read
    tmp = tmp_path / "w.tmp"
    tmp.write_bytes(b"c" * 1000)
    os.replace(tmp, f)                                    # new inode
    assert ab._digest_file(f) == _sha1(b"c" * 1000)


def test_memo_false_always_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "_DIGEST_MEMO", {})
    monkeypatch.setattr(ab, "_DIGEST_RACY_NS", -10**18)
    f = tmp_path / "w.bin"
    f.write_bytes(b"x" * 10)
    n = _count_opens(monkeypatch)
    ab._digest_file(f, memo=False)
    ab._digest_file(f, memo=False)
    assert n["open"] == 2 and not ab._DIGEST_MEMO
