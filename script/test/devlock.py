"""Per-device lock for board runs (apppaltest.py and friends).

One run per device at a time. The lock is a kernel flock on a file under
BOARD_LOCK_DIR, the same file skills/board-mgmt/board.py uses
(``<host>_3121.lock``), so the two tools also exclude each other.

Why flock and not a "ping the holder over a port" protocol: the kernel drops
an flock as soon as the holding process exits for any reason (normal exit,
exception, Ctrl-C, SIGKILL, OOM). There is no stale lock to detect or steal,
so there is no race where two waiters both decide the holder is dead and
both take the board. On NFS (/scratch) the lock lives on the server; if the
holder's whole host dies the server reclaims it when the client lease expires.

The one case flock cannot solve is a holder that is alive but hung. For that
the ``.owner`` sidecar records pid/host/user/start/argv so a waiter can print
who holds it, and ``--lock-break`` lets a human kill a hung local holder.
"""
import fcntl
import json
import os
import signal
import socket
import sys
import time

LOCK_DIR = os.environ.get("BOARD_LOCK_DIR", "/scratch/staff/huaj/tmp/board_locks")
HWS_PORT = 3121
POLL_SECS = 5
HEARTBEAT_SECS = 30
# A holder whose heartbeat is this stale is frozen (its watchdog thread is stuck too).
FROZEN_SECS = int(os.environ.get("DEVLOCK_FROZEN_SECS", 3 * HEARTBEAT_SECS))


def lock_path(device, lock_dir=None):
    """Lock file for a device. ``device`` may be 'user@ip' or a bare host."""
    dev = device.split("@")[-1]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in dev)
    return os.path.join(lock_dir or LOCK_DIR, f"{safe}_{HWS_PORT}.lock")


def read_owner(path):
    try:
        with open(path + ".owner") as f:
            raw = f.read().strip()
    except OSError:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw}  # board.py writes a plain text line


def describe(owner):
    if not owner:
        return "unknown holder"
    if "raw" in owner:
        return owner["raw"]
    age = int(time.time() - owner.get("start", time.time()))
    return (f"pid {owner.get('pid')} on {owner.get('host')} by {owner.get('user')}, "
            f"held {age // 60}m{age % 60:02d}s: {owner.get('cmd', '')[:120]}")


class DeviceLock:
    """Context manager: blocks up to ``wait_secs`` for the device, then exits."""

    def __init__(self, device, wait_secs=1800, lock_dir=None, out=None):
        self.device = device
        self.wait_secs = wait_secs
        self.path = lock_path(device, lock_dir)
        self.out = out or sys.stdout
        self.fd = None

    def _try(self):
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = open(self.path, "a+")
        deadline = time.time() + self.wait_secs
        announced = False
        while not self._try():
            owner = read_owner(self.path)
            if time.time() >= deadline:
                self.fd.close()
                sys.exit(f"[devlock] BOARD BUSY after {self.wait_secs}s: {self.device} "
                         f"held by {describe(owner)} ({self.path})")
            if not announced:
                print(f"[devlock] {self.device} busy, waiting (up to {self.wait_secs}s): "
                      f"{describe(owner)}", file=self.out, flush=True)
                announced = True
            self._handle_frozen(owner)
            time.sleep(POLL_SECS)
        self.owner = {"pid": os.getpid(), "host": socket.gethostname(),
                      "user": os.environ.get("USER", "?"), "start": time.time(),
                      "heartbeat": time.time(), "cmd": " ".join(sys.argv)}
        self._write_owner()
        print(f"[devlock] acquired {self.device} ({self.path})", file=self.out, flush=True)
        return self

    def _write_owner(self):
        tmp = f"{self.path}.owner.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.owner, f)
        os.replace(tmp, self.path + ".owner")

    def heartbeat(self):
        """Called by devwatch every HEARTBEAT_SECS while the holder is alive."""
        if self.fd is not None:
            self.owner["heartbeat"] = time.time()
            self._write_owner()

    def _handle_frozen(self, owner):
        """Holder alive but its heartbeat stopped: break it if it is on this host."""
        if not owner or "heartbeat" not in owner or "pid" not in owner:
            return  # board.py / pre-heartbeat apppaltest: no heartbeat to judge by
        stale = time.time() - owner["heartbeat"]
        if stale < FROZEN_SECS:
            return
        key = (owner["pid"], owner.get("host"))
        if owner.get("host") == socket.gethostname():
            print(f"[devlock] holder pid {owner['pid']} frozen (no heartbeat for {int(stale)}s): "
                  f"breaking the lock", file=self.out, flush=True)
            print(f"[devlock] {break_lock(self.device, os.path.dirname(self.path), signal.SIGKILL)}",
                  file=self.out, flush=True)
        elif getattr(self, "_alerted", None) != key:
            self._alerted = key
            print(f"[devlock] ALERT: holder pid {owner['pid']} on {owner.get('host')} looks frozen "
                  f"(no heartbeat for {int(stale)}s). Not killing across hosts: run "
                  f"`apppaltest.py --lock-break` on {owner.get('host')}", file=self.out, flush=True)

    def __exit__(self, *exc):
        if self.fd is None:
            return False
        try:
            os.unlink(self.path + ".owner")
        except OSError:
            pass
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()
        self.fd = None
        return False


def break_lock(device, lock_dir=None, sig=signal.SIGTERM):
    """Kill a hung holder on THIS host. A crashed holder never needs this."""
    path = lock_path(device, lock_dir)
    owner = read_owner(path)
    if not owner or "pid" not in owner:
        return f"no breakable holder recorded for {device}"
    if owner.get("host") != socket.gethostname():
        return f"holder is on {owner.get('host')}, not here: run --lock-break there"
    try:
        os.kill(int(owner["pid"]), sig)
    except ProcessLookupError:
        return f"holder pid {owner['pid']} is already gone (lock is free)"
    return f"sent {sig.name} to holder pid {owner['pid']}"
