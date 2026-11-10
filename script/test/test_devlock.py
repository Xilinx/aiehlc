"""BDD-style tests for devlock: one run per device, crash releases the lock."""
import os, signal, subprocess, sys, time, textwrap
import pytest
sys.path.insert(0, os.path.dirname(__file__))
import devlock

HOLDER = textwrap.dedent("""
    import sys, time; sys.path.insert(0, {here!r})
    import devlock
    with devlock.DeviceLock({dev!r}, lock_dir={d!r}, wait_secs=1):
        print("HELD", flush=True); time.sleep(60)
""")

def spawn_holder(tmp_path, dev="huaj@10.23.224.213"):
    code = HOLDER.format(here=os.path.dirname(os.path.abspath(__file__)), dev=dev, d=str(tmp_path))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.strip() == "HELD":
            break
    else:
        raise AssertionError("holder never acquired the lock")
    return p

def test_same_file_as_board_py(tmp_path):
    assert devlock.lock_path("huaj@10.23.224.213", str(tmp_path)).endswith("10.23.224.213_3121.lock")

def test_second_run_is_refused_while_first_holds(tmp_path):
    p = spawn_holder(tmp_path)
    try:
        with pytest.raises(SystemExit) as e:
            with devlock.DeviceLock("huaj@10.23.224.213", lock_dir=str(tmp_path), wait_secs=0):
                pass
        assert "BOARD BUSY" in str(e.value) and str(p.pid) in str(e.value)
    finally:
        p.kill(); p.wait()

def test_sigkilled_holder_releases_lock_immediately(tmp_path):
    p = spawn_holder(tmp_path)
    p.send_signal(signal.SIGKILL); p.wait()
    t = time.time()
    with devlock.DeviceLock("huaj@10.23.224.213", lock_dir=str(tmp_path), wait_secs=0):
        assert time.time() - t < 1

def test_waiter_gets_lock_when_holder_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(devlock, "POLL_SECS", 0.2)
    p = spawn_holder(tmp_path)
    subprocess.Popen(["sh", "-c", f"sleep 1; kill {p.pid}"])
    with devlock.DeviceLock("huaj@10.23.224.213", lock_dir=str(tmp_path), wait_secs=20):
        pass
    p.wait()

def test_different_devices_do_not_block(tmp_path):
    p = spawn_holder(tmp_path, dev="huaj@10.23.224.213")
    try:
        with devlock.DeviceLock("huaj@10.23.224.214", lock_dir=str(tmp_path), wait_secs=0):
            pass
    finally:
        p.kill(); p.wait()

def test_lock_break_kills_hung_local_holder(tmp_path):
    p = spawn_holder(tmp_path)
    msg = devlock.break_lock("huaj@10.23.224.213", lock_dir=str(tmp_path))
    assert str(p.pid) in msg
    p.wait(timeout=10)
    with devlock.DeviceLock("huaj@10.23.224.213", lock_dir=str(tmp_path), wait_secs=0):
        pass
