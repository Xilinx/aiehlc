"""apppaltest preflight: make sure the board's hw_server service (hws) is healthy
before a netload or full-mode run, under apppaltest's own board lock.

Contract: aiengineers/xiaojiang/tasks/2026-11-10_hws/contract.md (plan-1109-b1597a s2),
sections 5 (lock), 6 (retry budget) and 8 (full mode per board type).

    status UP        -> ok
    status DOWN      -> up                       (hws.up has its own 2 start attempts)
    status DEGRADED  -> clean + restart, at most ``max_restarts`` times (contract: 1)
    FOREIGN / host unreachable / lock busy -> fail at once, nothing touched
    then (netload only) a short netload smoke test, e.g. nbspin5.

ATTACH (palmyra full mode with -nonreboot: systest-client + ``connect -url`` to the
resident hw_server): ensure only, no smoke test, no down/up.

Full mode (contract §8):
    palmyra: hws down before the run (two palmyra sessions conflict), hws up after it
             (a failed up afterwards is a ``HWS WARN``; it never changes the test rc).
    vek385:  reuse the service: ensure before the run and again after the power
             cycle, before ``connect -url`` (``after_power_cycle``).

Every hws call passes ``locked=True`` only while the caller really holds the lock
(``lock.fd is not None``); with ``--no-lock`` hws takes the lock itself per call.
The module has no apppaltest import-time side effects, so it is unit-testable.
"""
import contextlib

import hws

NETLOAD, FULL, ATTACH = "netload", "full", "attach"  # ATTACH: full JTAG via the resident hw_server
DEFAULT_MAX_RESTARTS = 1  # contract §6: at most 1 restart per call
_FAIL_FAST = (hws.RC_BUSY, hws.RC_FOREIGN, hws.RC_UNREACHABLE, hws.RC_USAGE)


class PreflightError(Exception):
    """The run must not start. ``rc`` uses the hws exit codes (3 degraded, 5 foreign, ...)."""

    def __init__(self, rc, msg):
        super().__init__(msg)
        self.rc = rc
        self.msg = msg


class Preflight:
    """One run's preflight. ``smoke()`` -> (ok: bool|None, msg); None = skipped."""

    def __init__(self, board, btype, lock=None, smoke=None, max_restarts=DEFAULT_MAX_RESTARTS,
                 hws_mod=hws, out=print, quiet=None, hws_kw=None):
        if btype not in ("palmyra", "vek385"):
            raise ValueError(f"unknown board type {btype!r}")
        if lock is not None and getattr(lock, "fd", None) is None:
            raise AssertionError("Preflight: lock given but not held (lock.fd is None)")
        self.board = board
        self.btype = btype
        self.locked = lock is not None
        self.smoke = smoke
        self.max_restarts = max(0, int(max_restarts))
        self.hws = hws_mod
        self.out = out
        self.quiet = quiet or (lambda secs, what: contextlib.nullcontext())
        self.kw = dict(hws_kw or {})
        self.took_down = False

    # -- hws calls --------------------------------------------------------
    def _call(self, op, secs=300):
        fn = getattr(self.hws, op)
        args = {} if op == "status" else {"locked": self.locked}
        with self.quiet(secs, f"hws {op}"):
            return fn(self.board, self.btype, **args, **self.kw)

    def _fail(self, rc, why):
        msg = f"HWS PREFLIGHT FAIL {self.board} ({self.btype}): {why}"
        if rc == hws.RC_DEGRADED and hws.LAB_POINTER not in msg:
            msg += f" {hws.LAB_POINTER}"
        return PreflightError(rc, msg)

    def ensure_service(self):
        """status -> up / clean+restart within the budget. Returns the final Health."""
        try:
            h = self._call("status", 120)
        except hws.HwsError as e:
            raise self._fail(e.rc, f"status failed: {e.msg}") from None
        self.out(f"[hws] status {h.state}: {h.msg}".rstrip(": "))
        if h.state == hws.UP:
            return h
        if h.state in (hws.FOREIGN, hws.UNREACHABLE):
            raise self._fail(h.rc or (hws.RC_FOREIGN if h.state == hws.FOREIGN else hws.RC_UNREACHABLE),
                             f"state {h.state}, nothing touched: {h.msg}")
        if h.state == hws.DOWN:
            h = self._step("up", h)
            if h.state == hws.UP:
                return h
        for i in range(1, self.max_restarts + 1):
            # hws restart = down (our session) -> clean (own stale helpers) -> up
            self.out(f"[hws] {h.state}: clean + restart ({i}/{self.max_restarts})")
            h = self._step("restart", h)
            if h.state == hws.UP:
                return h
        raise self._fail(hws.RC_DEGRADED, f"still {h.state} after {self.max_restarts} restart(s) "
                                          f"(retry limit reached): {h.msg}")

    def _step(self, op, last):
        """One mutating hws op; fail-fast rcs raise, other HwsErrors leave the state DEGRADED."""
        try:
            h = self._call(op)
        except hws.HwsError as e:
            if e.rc in _FAIL_FAST:
                raise self._fail(e.rc, f"hws {op}: {e.msg}") from None
            self.out(f"[hws] {op} failed rc={e.rc}: {e.msg}")
            return e.health or hws.Health(self.board, self.btype, hws.DEGRADED, dict(last.checks), e.msg, e.rc)
        self.out(f"[hws] {op} -> {h.state}")
        return h

    def smoke_test(self):
        if self.smoke is None:
            return
        with self.quiet(180, "hws smoke test"):
            ok, msg = self.smoke()
        if ok is None:
            self.out(f"[hws] smoke test skipped: {msg}")
        elif ok:
            self.out(f"[hws] smoke test PASS: {msg}")
        else:
            raise self._fail(hws.RC_DEGRADED, f"netload smoke test failed: {msg}")

    # -- run hooks --------------------------------------------------------
    def before_run(self, mode):
        if mode not in (NETLOAD, FULL, ATTACH):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == FULL and self.btype == "palmyra":
            self.out("[hws] palmyra full mode: taking hws down for this run (back up afterwards)")
            try:
                h = self._call("down")
            except hws.HwsError as e:
                raise self._fail(e.rc, f"hws down before full mode: {e.msg}") from None
            if h.state != hws.DOWN:
                raise self._fail(h.rc or hws.RC_DEGRADED, f"hws down left state {h.state}: {h.msg}")
            self.took_down = True
            return
        self.ensure_service()
        if mode == NETLOAD:
            self.smoke_test()

    def after_power_cycle(self):
        """VEK385: hw_server must still answer after `power 0/1`, before `connect -url`."""
        if self.btype == "vek385":
            self.out("[hws] vek385: re-checking hw_server after the power cycle")
            self.ensure_service()

    def after_run(self):
        """palmyra full mode: bring hws back. Returns the rc of `up` (0 ok); never raises."""
        if not self.took_down:
            return 0
        self.took_down = False
        try:
            h = self._call("up")
            rc = 0 if h.state == hws.UP else (h.rc or hws.RC_DEGRADED)
            msg = h.msg
        except hws.HwsError as e:
            rc, msg = e.rc, e.msg
        except Exception as e:  # never mask the test's own result
            rc, msg = hws.RC_DEGRADED, repr(e)
        if rc:
            self.out(f"HWS WARN {self.board}: hws up after full mode failed rc={rc}: {msg}")
        else:
            self.out("[hws] hws back UP after full mode")
        return rc
