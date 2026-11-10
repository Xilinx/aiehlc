"""apppaltest hws preflight (plan-1109-b1597a s5) against a fake hws, both board types.

No board, ssh or xsdb: the hws library is replaced by FakeHws, whose status/up/
down/clean/restart results are scripted per test.
"""
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hws  # noqa: E402
import hwspreflight as hp  # noqa: E402

PAL, VEK = ("10.23.224.213", "palmyra"), ("crimini1", "vek385")
BOTH = pytest.mark.parametrize("board,btype", [PAL, VEK], ids=["palmyra", "vek385"])


class HeldLock:
    fd = object()  # DeviceLock.fd is not None while held


class FakeHws:
    """Scripted hws: ``script[op]`` is a list of states (or HwsError) consumed per call."""

    def __init__(self, board, btype, **script):
        self.board, self.btype = board, btype
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []
        for name in ("UP", "DOWN", "DEGRADED", "FOREIGN", "UNREACHABLE", "HwsError", "Health", "LAB_POINTER"):
            setattr(self, name, getattr(hws, name))
        for name in dir(hws):
            if name.startswith("RC_"):
                setattr(self, name, getattr(hws, name))

    def _op(self, op, board, btype, **kw):
        assert (board, btype) == (self.board, self.btype)
        self.calls.append((op, kw.get("locked")))
        seq = self.script.get(op)
        if not seq:
            raise AssertionError(f"unexpected hws.{op}()")
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        rc = {hws.UP: 0, hws.DOWN: 0, hws.DEGRADED: hws.RC_DEGRADED, hws.FOREIGN: hws.RC_FOREIGN,
              hws.UNREACHABLE: hws.RC_UNREACHABLE}[r]
        return hws.Health(board, btype, r, {"port": r == hws.UP}, f"fake {op} {r}", rc)

    def __getattr__(self, op):
        if op in ("status", "up", "down", "clean", "restart"):
            return lambda board, btype, **kw: self._op(op, board, btype, **kw)
        raise AttributeError(op)

    def ops(self):
        return [c[0] for c in self.calls]


def make(board, btype, smoke=None, lock=HeldLock(), max_restarts=1, **script):
    fake = FakeHws(board, btype, **script)
    lines = []
    pf = hp.Preflight(board, btype, lock=lock, smoke=smoke, max_restarts=max_restarts,
                      hws_mod=fake, out=lines.append)
    return pf, fake, lines


def smoke_ok():
    return True, "nbspin5.elf ran and exited in 6.8s"


# -- ensure: down -> up, degraded -> restart, retry limit ------------------
@BOTH
def test_up_service_needs_nothing(board, btype):
    pf, fake, _ = make(board, btype, smoke=smoke_ok, status=[hws.UP])
    pf.before_run(hp.NETLOAD)
    assert fake.ops() == ["status"]


@BOTH
def test_down_service_is_started_with_up(board, btype):
    pf, fake, lines = make(board, btype, smoke=smoke_ok, status=[hws.DOWN], up=[hws.UP])
    pf.before_run(hp.NETLOAD)
    assert fake.ops() == ["status", "up"]
    assert ("up", True) in fake.calls  # locked=True: apppaltest already holds the board lock
    assert any("smoke test PASS" in s for s in lines)


@BOTH
def test_degraded_service_gets_clean_restart(board, btype):
    pf, fake, lines = make(board, btype, smoke=smoke_ok, status=[hws.DEGRADED], restart=[hws.UP])
    pf.before_run(hp.NETLOAD)
    assert fake.ops() == ["status", "restart"]  # hws restart = down -> clean -> up
    assert any("clean + restart (1/1)" in s for s in lines)


@BOTH
def test_down_then_failed_up_falls_back_to_restart(board, btype):
    err = hws.HwsError(hws.RC_DEGRADED, "H3 targets has no Cortex-R52 after 2 attempts")
    pf, fake, _ = make(board, btype, smoke=smoke_ok, status=[hws.DOWN], up=[err], restart=[hws.UP])
    pf.before_run(hp.NETLOAD)
    assert fake.ops() == ["status", "up", "restart"]


@BOTH
def test_retry_limit_gives_clear_error(board, btype):
    pf, fake, _ = make(board, btype, smoke=smoke_ok, status=[hws.DEGRADED], restart=[hws.DEGRADED])
    with pytest.raises(hp.PreflightError) as ei:
        pf.before_run(hp.NETLOAD)
    e = ei.value
    assert e.rc == hws.RC_DEGRADED
    assert "retry limit reached" in e.msg and "1 restart(s)" in e.msg
    assert board in e.msg and "ask Amit" in e.msg  # lab pointer from the contract
    assert fake.ops() == ["status", "restart"]  # exactly one restart, no loop


@BOTH
def test_retry_limit_is_configurable(board, btype):
    pf, fake, _ = make(board, btype, smoke=smoke_ok, max_restarts=2, status=[hws.DEGRADED],
                       restart=[hws.DEGRADED, hws.UP])
    pf.before_run(hp.NETLOAD)
    assert fake.ops() == ["status", "restart", "restart"]


@BOTH
def test_zero_restarts_fails_at_once_when_degraded(board, btype):
    pf, fake, _ = make(board, btype, max_restarts=0, status=[hws.DEGRADED])
    with pytest.raises(hp.PreflightError, match="after 0 restart"):
        pf.before_run(hp.NETLOAD)
    assert fake.ops() == ["status"]


@BOTH
@pytest.mark.parametrize("state,rc", [(hws.FOREIGN, hws.RC_FOREIGN), (hws.UNREACHABLE, hws.RC_UNREACHABLE)])
def test_foreign_or_unreachable_touches_nothing(board, btype, state, rc):
    pf, fake, _ = make(board, btype, smoke=smoke_ok, status=[state])
    with pytest.raises(hp.PreflightError) as ei:
        pf.before_run(hp.NETLOAD)
    assert ei.value.rc == rc and "nothing touched" in ei.value.msg
    assert fake.ops() == ["status"]


@BOTH
def test_lock_busy_during_up_fails_fast(board, btype):
    busy = hws.HwsError(hws.RC_BUSY, "HWS LOCK BUSY")
    pf, fake, _ = make(board, btype, lock=None, status=[hws.DOWN], up=[busy])
    with pytest.raises(hp.PreflightError) as ei:
        pf.before_run(hp.NETLOAD)
    assert ei.value.rc == hws.RC_BUSY
    assert fake.calls == [("status", None), ("up", False)]  # --no-lock: hws takes the lock itself


# -- smoke test --------------------------------------------------------------
@BOTH
def test_smoke_failure_fails_the_run(board, btype):
    pf, fake, _ = make(board, btype, smoke=lambda: (False, "load nbspin5.elf rc=1: refused"), status=[hws.UP])
    with pytest.raises(hp.PreflightError) as ei:
        pf.before_run(hp.NETLOAD)
    assert ei.value.rc == hws.RC_DEGRADED
    assert "netload smoke test failed: load nbspin5.elf rc=1" in ei.value.msg


@BOTH
def test_smoke_skipped_is_not_a_failure(board, btype):
    pf, _, lines = make(board, btype, smoke=lambda: (None, "supervisor not answering"), status=[hws.UP])
    pf.before_run(hp.NETLOAD)
    assert any("smoke test skipped: supervisor not answering" in s for s in lines)


@BOTH
def test_smoke_not_run_when_service_cannot_be_fixed(board, btype):
    ran = []
    pf, _, _ = make(board, btype, smoke=lambda: ran.append(1) or (True, ""), status=[hws.DEGRADED],
                    restart=[hws.DEGRADED])
    with pytest.raises(hp.PreflightError):
        pf.before_run(hp.NETLOAD)
    assert ran == []


# -- full mode per board type (contract §8) -----------------------------------
def test_palmyra_full_mode_takes_hws_down_then_up():
    ran = []
    pf, fake, _ = make(*PAL, smoke=lambda: ran.append(1) or (True, ""), down=[hws.DOWN], up=[hws.UP])
    pf.before_run(hp.FULL)
    assert fake.ops() == ["down"] and ran == []  # no status/ensure/smoke: the run opens its own session
    pf.after_power_cycle()  # palmyra: no-op
    assert pf.after_run() == 0
    assert fake.calls == [("down", True), ("up", True)]
    assert pf.after_run() == 0 and len(fake.calls) == 2  # idempotent


def test_palmyra_full_mode_failed_up_after_run_is_a_warning_only():
    err = hws.HwsError(hws.RC_DEGRADED, "H2 port 3121 closed after 2 attempts")
    pf, _, lines = make(*PAL, down=[hws.DOWN], up=[err])
    pf.before_run(hp.FULL)
    assert pf.after_run() == hws.RC_DEGRADED  # returned, never raised
    assert any(s.startswith("HWS WARN 10.23.224.213") and "rc=3" in s for s in lines)


def test_palmyra_full_mode_down_failure_stops_the_run():
    pf, fake, _ = make(*PAL, down=[hws.HwsError(hws.RC_BUSY, "busy")])
    with pytest.raises(hp.PreflightError) as ei:
        pf.before_run(hp.FULL)
    assert ei.value.rc == hws.RC_BUSY
    assert pf.after_run() == 0 and fake.ops() == ["down"]  # nothing to bring back


def test_palmyra_attach_mode_ensures_without_smoke_or_down():
    ran = []
    pf, fake, _ = make(*PAL, smoke=lambda: ran.append(1) or (True, ""), status=[hws.DOWN], up=[hws.UP])
    pf.before_run(hp.ATTACH)
    assert fake.ops() == ["status", "up"] and ran == []
    assert pf.after_run() == 0 and fake.ops() == ["status", "up"]


def test_vek385_full_mode_reuses_service_and_rechecks_after_power_cycle():
    pf, fake, _ = make(*VEK, status=[hws.UP, hws.DEGRADED], restart=[hws.UP])
    pf.before_run(hp.FULL)
    assert fake.ops() == ["status"]  # no down/up on VEK385
    pf.after_power_cycle()  # hw_server did not survive power 0/1 -> restarted within budget
    assert fake.ops() == ["status", "status", "restart"]
    assert pf.after_run() == 0 and len(fake.calls) == 3


def test_vek385_power_cycle_recheck_failure_raises():
    pf, _, _ = make(*VEK, status=[hws.UP, hws.DEGRADED], restart=[hws.DEGRADED])
    pf.before_run(hp.FULL)
    with pytest.raises(hp.PreflightError, match="retry limit reached"):
        pf.after_power_cycle()


# -- guards ------------------------------------------------------------------
def test_lock_object_must_be_held():
    with pytest.raises(AssertionError, match="not held"):
        hp.Preflight(*PAL, lock=types.SimpleNamespace(fd=None))


def test_unknown_board_type_or_mode_is_rejected():
    with pytest.raises(ValueError):
        hp.Preflight("x", "zynq")
    pf, _, _ = make(*PAL)
    with pytest.raises(ValueError):
        pf.before_run("bogus")


# -- apppaltest wiring ---------------------------------------------------------
def _apppaltest():
    os.environ.setdefault("USERNAME", "u")
    os.environ.setdefault("PALIP", "10.0.0.9")
    os.environ.setdefault("BOARDNAME", "palmyra")
    import apppaltest
    return apppaltest


def _args(app, *argv):
    return app.build_arg_parser().parse_args(list(argv))


def test_cli_has_hws_passthrough_and_skip_preflight():
    app = _apppaltest()
    a = _args(app, "--hws", "status")
    assert a.hws == "status" and not a.skip_preflight and a.hws_restarts == 1
    assert _args(app, "--skip-preflight", "x.elf").skip_preflight
    with pytest.raises(SystemExit):
        _args(app, "--hws", "clean")


def test_preflight_mode_mapping():
    app = _apppaltest()
    assert app.preflight_mode(_args(app, "--netload")) == hp.NETLOAD
    assert app.preflight_mode(_args(app)) == hp.FULL
    assert app.preflight_mode(_args(app, "-nonreboot")) == (hp.FULL if app.IS_VEK385 else hp.ATTACH)


def test_hws_passthrough_status_down_exits_1(monkeypatch, capsys):
    app = _apppaltest()
    monkeypatch.setattr(app.hws, "status", lambda b, t: hws.Health(b, t, hws.DOWN, {}, "no session", 0))
    assert app.hws_passthrough(_args(app, "--hws", "status")) == hws.RC_DOWN
    assert "HWS DOWN" in capsys.readouterr().out


def test_hws_passthrough_error_rc(monkeypatch):
    app = _apppaltest()

    def boom(b, t, **kw):
        raise hws.HwsError(hws.RC_BUSY, "busy")
    monkeypatch.setattr(app.hws, "up", boom)
    assert app.hws_passthrough(_args(app, "--hws", "up")) == hws.RC_BUSY


def test_skip_preflight_runs_without_hws(monkeypatch):
    app = _apppaltest()
    monkeypatch.setattr(app, "_main", lambda a: "ran")
    monkeypatch.setattr(app.hwspreflight, "Preflight", lambda *a, **k: pytest.fail("preflight ran"))
    assert app._run_with_preflight(_args(app, "--skip-preflight"), None) == "ran"


def test_preflight_failure_exits_before_the_run(monkeypatch, capsys):
    app = _apppaltest()
    monkeypatch.setattr(app, "_main", lambda a: pytest.fail("run started"))
    monkeypatch.setattr(app.hws, "status", lambda b, t, **k: hws.Health(b, t, hws.FOREIGN, {}, "held by x", 5))
    with pytest.raises(SystemExit) as ei:
        app._run_with_preflight(_args(app, "--netload"), HeldLock())
    assert ei.value.code == hws.RC_FOREIGN
    assert "Test FAILED (hws preflight)" in capsys.readouterr().out


def _fake_netload(load, statuses):
    calls = []

    def nl(argv, **kw):
        calls.append(argv[0])
        return load if argv[0] == "load" else statuses.pop(0)
    return nl, calls


def test_hws_smoke_passes_when_app_exits(monkeypatch, tmp_path):
    app = _apppaltest()
    elf = tmp_path / "nbspin5.elf"
    elf.write_bytes(b"\x7fELF")
    nl, calls = _fake_netload((0, "[netload] load rc=0 state=running runs=9"),
                              [(0, "[netload] status rc=0 state=running runs=9"),
                               (0, "[netload] status rc=0 state=exited runs=9")])
    monkeypatch.setattr(app, "netload_remote", nl)
    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    ok, msg = app.hws_smoke(_args(app, "--netload", "--smoke-elf", str(elf)))
    assert ok is True and "nbspin5.elf ran and exited" in msg
    assert calls == ["load", "status", "status"]


def test_hws_smoke_fails_on_refused_load_and_skips_without_supervisor(monkeypatch, tmp_path):
    app = _apppaltest()
    elf = tmp_path / "nbspin5.elf"
    elf.write_bytes(b"\x7fELF")
    a = _args(app, "--netload", "--smoke-elf", str(elf))
    monkeypatch.setattr(app, "netload_remote", lambda argv, **kw: (1, "[netload] load refused"))
    ok, msg = app.hws_smoke(a)
    assert ok is False and "rc=1" in msg
    monkeypatch.setattr(app, "netload_remote", lambda argv, **kw: (2, "[netload] no answer"))
    ok, msg = app.hws_smoke(a)
    assert ok is None and "bootstraps" in msg


def test_hws_smoke_fails_when_app_is_killed(monkeypatch, tmp_path):
    app = _apppaltest()
    elf = tmp_path / "nbspin5.elf"
    elf.write_bytes(b"\x7fELF")
    nl, _ = _fake_netload((0, "load rc=0 state=running runs=3"), [(0, "status rc=0 state=watchdog-timeout runs=3")])
    monkeypatch.setattr(app, "netload_remote", nl)
    ok, msg = app.hws_smoke(_args(app, "--netload", "--smoke-elf", str(elf)))
    assert ok is False and "did not exit by itself" in msg


def test_hws_smoke_missing_elf(tmp_path):
    app = _apppaltest()
    ok, msg = app.hws_smoke(_args(app, "--netload", "--smoke-elf", str(tmp_path / "nope.elf")))
    assert ok is False and "not found" in msg
