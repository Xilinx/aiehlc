"""Self-watchdog for a board run (apppaltest.py).

The device lock (devlock.py) is a kernel flock, so a CRASHED holder frees the
board at once. A holder that is alive but HUNG would keep it forever. Waiters
cannot tell "hung" from "a long board run", so the holder judges itself:

  * no-progress timeout (--stall-secs): progress = new console output, an xsdb
    command returning, a netload line. A step that is legitimately quiet for a
    long time (netboot build, ELF download) runs inside ``with quiet(secs, what):``;
    the allowance ends with the step.
  * hard cap (--max-run) on the whole run.

On either, the watchdog kills this process's own children (xsdb / hw_server /
ssh systest-client sessions, so no stale session is left on the lab host like
the Oct-14 one on palmyra), prints ``STALLED: <reason>`` and exits nonzero.
The exit releases the flock, so the next waiter gets the board.

It also refreshes a heartbeat in the lock's .owner file; a waiter that sees the
heartbeat stop (the whole process frozen, watchdog thread included) can break
the lock -- see devlock.FROZEN_SECS.
"""
import contextlib
import os
import signal
import sys
import threading
import time

STALL_EXIT = 124  # same code `timeout` uses


def descendants(pid):
    """All descendant pids of ``pid`` (walks /proc; no psutil needed)."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(ppid, []).append(int(d))
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def kill_children(pid=None, grace=2.0):
    """SIGTERM then SIGKILL every descendant of ``pid`` (default: this process)."""
    pids = descendants(pid or os.getpid())
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            try:
                os.kill(p, sig)
            except (ProcessLookupError, PermissionError):
                pass
        if sig == signal.SIGTERM and pids:
            time.sleep(grace)
    return pids


class Watchdog:
    def __init__(self, stall_secs=600, max_run=3600, heartbeat=None, heartbeat_secs=30,
                 out=None, exit_fn=None, tick=1.0):
        self.stall_secs = stall_secs
        self.max_run = max_run
        self.heartbeat = heartbeat          # callable, e.g. DeviceLock.heartbeat
        self.heartbeat_secs = heartbeat_secs
        self.out = out or sys.stdout
        self.exit_fn = exit_fn or os._exit  # os._exit: works from a thread, no cleanup races
        self.tick = tick
        self.start = time.time()
        self.last = self.start
        self.what = "start"
        self.quiet_until = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = None
        self.fired = None

    def touch(self, what=None):
        with self._lock:
            self.last = time.time()
            if what:
                self.what = what

    def quiet(self, secs, what):
        """Allow up to ``secs`` without progress for a known-slow step.

        Returns the previous allowance; pass it to ``end_quiet`` when the step
        ends, or the window outlives the step and masks a real stall after it
        (seen on 213: a netload `load` left 600 s of allowance behind, so
        --stall-secs 60 could not fire while the app hung silently)."""
        with self._lock:
            prev = self.quiet_until
            self.quiet_until = max(prev, time.time() + secs)
            self.what = what
            self.last = time.time()
            return prev

    def end_quiet(self, prev=0.0, what=None):
        """The slow step finished: restore the allowance that was in force before it."""
        with self._lock:
            self.quiet_until = prev
            self.last = time.time()
            if what:
                self.what = what

    def check(self, now=None):
        """Return the stall reason, or None. Pure: easy to unit-test."""
        now = now or time.time()
        with self._lock:
            if self.max_run and now - self.start >= self.max_run:
                return f"run exceeded --max-run {self.max_run}s (last progress: {self.what})"
            idle = now - self.last
            if self.stall_secs and idle >= self.stall_secs and now >= self.quiet_until:
                return f"no progress for {int(idle)}s; last progress was: {self.what}"
        return None

    def _fire(self, reason):
        self.fired = reason
        print(f"\n[devwatch] STALLED: {reason}", file=self.out, flush=True)
        killed = kill_children()
        print(f"[devwatch] killed {len(killed)} child process(es); exiting {STALL_EXIT} "
              "(the device lock is released on exit)", file=self.out, flush=True)
        self.exit_fn(STALL_EXIT)

    def _loop(self):
        next_hb = 0.0
        while not self._stop.wait(self.tick):
            now = time.time()
            if self.heartbeat and now >= next_hb:
                try:
                    self.heartbeat()
                except Exception:
                    pass
                next_hb = now + self.heartbeat_secs
            reason = self.check(now)
            if reason:
                self._fire(reason)
                return

    def __enter__(self):
        self._th = threading.Thread(target=self._loop, name="devwatch", daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)
        return False


# Module-level handle so deep call sites can report progress without plumbing.
current = None


def progress(what=None):
    if current:
        current.touch(what)


@contextlib.contextmanager
def quiet(secs, what):
    """``with devwatch.quiet(secs, what):`` -- a known-slow step; the allowance ends with the block."""
    wd = current
    prev = wd.quiet(secs, what) if wd else None
    try:
        yield
    finally:
        if wd:
            wd.end_quiet(prev, f"{what}: finished")
