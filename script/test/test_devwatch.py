"""BDD-style tests for the apppaltest self-watchdog and the frozen-holder heartbeat."""
import io, json, os, signal, subprocess, sys, textwrap, time
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import devlock, devwatch

DEV = "huaj@10.23.224.213"

# A holder that takes the lock, starts the watchdog, spawns a child (like xsdb),
# then either goes silent or keeps reporting progress.
HOLDER = textwrap.dedent("""
    import subprocess, sys, time; sys.path.insert(0, {here!r})
    import devlock, devwatch
    with devlock.DeviceLock({dev!r}, lock_dir={d!r}, wait_secs=1) as lk:
        child = subprocess.Popen(["sleep", "300"], stdout=subprocess.DEVNULL)
        print("CHILD", child.pid, flush=True)
        wd = devwatch.Watchdog(stall_secs={stall}, max_run={maxrun}, heartbeat=lk.heartbeat,
                               heartbeat_secs=0.2, tick=0.1)
        with wd:
            print("HELD", flush=True)
            for _ in range(int({run} / 0.2)):
                if {chatty}:
                    wd.touch("console output")
                time.sleep(0.2)
    print("CLEAN EXIT", flush=True)
""")


def spawn(tmp_path, stall=1, maxrun=0, run=30, chatty=False):
    code = HOLDER.format(here=HERE, dev=DEV, d=str(tmp_path), stall=stall, maxrun=maxrun,
                         run=run, chatty=chatty)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    child = None
    for line in p.stdout:
        if line.startswith("CHILD"):
            child = int(line.split()[1])
        if line.strip() == "HELD":
            break
    return p, child


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split(")")[1].split()[0] != "Z"


# --- pure decision logic ---------------------------------------------------

def test_no_progress_past_stall_is_a_stall():
    wd = devwatch.Watchdog(stall_secs=10, max_run=0)
    assert wd.check(wd.start + 9) is None
    assert "no progress for 10s" in wd.check(wd.start + 10)


def test_progress_resets_the_stall_clock():
    wd = devwatch.Watchdog(stall_secs=10, max_run=0)
    wd.touch("xsdb `dow` returned")
    assert wd.check(wd.last + 9) is None
    assert "xsdb `dow` returned" in wd.check(wd.last + 11)


def test_quiet_step_extends_the_allowance():
    wd = devwatch.Watchdog(stall_secs=10, max_run=0)
    wd.quiet(100, "building netboot.elf")
    assert wd.check(wd.last + 50) is None
    assert "building netboot.elf" in wd.check(wd.last + 101)


def test_max_run_fires_even_with_progress():
    wd = devwatch.Watchdog(stall_secs=0, max_run=60)
    wd.touch("console output")
    assert "max-run 60s" in wd.check(wd.start + 61)


def test_zero_disables_both():
    wd = devwatch.Watchdog(stall_secs=0, max_run=0)
    assert wd.check(wd.start + 10 ** 6) is None


# --- real processes ----------------------------------------------------------

def test_silent_holder_stalls_kills_children_and_frees_the_lock(tmp_path):
    p, child = spawn(tmp_path, stall=1)
    out = p.stdout.read()
    assert p.wait(timeout=20) == devwatch.STALL_EXIT
    assert "STALLED: no progress" in out and "CLEAN EXIT" not in out
    time.sleep(0.2)
    assert not alive(child), "watchdog left its child (xsdb stand-in) running"
    with devlock.DeviceLock(DEV, lock_dir=str(tmp_path), wait_secs=0):
        pass


def test_chatty_holder_is_not_misjudged(tmp_path):
    p, child = spawn(tmp_path, stall=1, run=3, chatty=True)
    out = p.stdout.read()
    assert p.wait(timeout=20) == 0
    assert "STALLED" not in out and "CLEAN EXIT" in out
    os.kill(child, signal.SIGKILL)


def test_max_run_stops_a_chatty_holder(tmp_path):
    p, _ = spawn(tmp_path, stall=0, maxrun=1, run=30, chatty=True)
    out = p.stdout.read()
    assert p.wait(timeout=20) == devwatch.STALL_EXIT
    assert "max-run 1s" in out


def test_heartbeat_is_written_to_owner_file(tmp_path):
    p, child = spawn(tmp_path, stall=0, run=30, chatty=True)
    try:
        path = devlock.lock_path(DEV, str(tmp_path)) + ".owner"
        hb1 = json.load(open(path))["heartbeat"]
        time.sleep(0.8)
        assert json.load(open(path))["heartbeat"] > hb1
    finally:
        p.kill(); p.wait(); os.kill(child, signal.SIGKILL)


def test_frozen_holder_on_same_host_is_broken_by_waiter(tmp_path, monkeypatch):
    monkeypatch.setattr(devlock, "POLL_SECS", 0.2)
    monkeypatch.setattr(devlock, "FROZEN_SECS", 1)
    p, child = spawn(tmp_path, stall=0, run=30, chatty=True)
    os.kill(p.pid, signal.SIGSTOP)  # whole process frozen, watchdog thread too
    buf = io.StringIO()
    t = time.time()
    with devlock.DeviceLock(DEV, lock_dir=str(tmp_path), wait_secs=15, out=buf):
        assert time.time() - t < 10
    assert "frozen" in buf.getvalue()
    p.wait(timeout=5)
    os.kill(child, signal.SIGKILL)


def test_frozen_holder_on_other_host_only_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(devlock, "POLL_SECS", 0.2)
    monkeypatch.setattr(devlock, "FROZEN_SECS", 1)
    p, child = spawn(tmp_path, stall=0, run=30, chatty=True)
    os.kill(p.pid, signal.SIGSTOP)
    path = devlock.lock_path(DEV, str(tmp_path)) + ".owner"
    o = json.load(open(path)); o.update(host="otherhost", heartbeat=time.time() - 100)
    json.dump(o, open(path, "w"))
    buf = io.StringIO()
    try:
        with pytest.raises(SystemExit) as e:
            with devlock.DeviceLock(DEV, lock_dir=str(tmp_path), wait_secs=1, out=buf):
                pass
        assert "BOARD BUSY" in str(e.value)
        assert "ALERT" in buf.getvalue() and "otherhost" in buf.getvalue()
        assert alive(p.pid), "must not kill a holder on another host"
    finally:
        os.kill(p.pid, signal.SIGKILL); p.wait(); os.kill(child, signal.SIGKILL)


def test_board_py_owner_without_heartbeat_is_never_broken(tmp_path, monkeypatch):
    monkeypatch.setattr(devlock, "FROZEN_SECS", 0)
    lk = devlock.DeviceLock(DEV, lock_dir=str(tmp_path), out=io.StringIO())
    lk._handle_frozen({"raw": "board.py huaj 123"})  # plain-text owner from board.py
    lk._handle_frozen({"pid": 1, "host": "x"})        # pre-heartbeat apppaltest
