#!/usr/bin/env python3
"""hws: check, clean, recover and keep hw_server running for a board.

Contract v1: aiengineers/xiaojiang/tasks/2026-11-10_hws/contract.md (plan-1109-b1597a s2).

    hws.py --board <ip|farmhost> [--type palmyra|vek385] [--json] [--wait-lock SECS] \
           status|up|down|clean|restart

The library is the source of truth; the CLI is a thin wrapper. apppaltest calls
``hws.ensure(board, locked=True)`` while it already holds ``<ip>_3121.lock`` --
never the CLI, which would block on its parent's flock.

Backends: ``hws_palmyra.Backend`` (palmyra 213, s3) and ``hws_vek385.Backend``
(VEK385 farm hosts, s4). A backend is ``Backend(board, io, log, lock_owner)`` with
methods ``status() up() down() clean()`` returning :class:`Health` or raising
:class:`HwsError`.
"""
import argparse
import dataclasses
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devlock  # noqa: E402

PORT = devlock.HWS_PORT

# States (contract §3)
UP, DOWN, DEGRADED, FOREIGN, UNREACHABLE = "UP", "DOWN", "DEGRADED", "FOREIGN", "UNREACHABLE"

# Exit codes (contract §7)
RC_OK, RC_DOWN, RC_USAGE, RC_DEGRADED, RC_BUSY, RC_FOREIGN, RC_UNREACHABLE, RC_STALLED = 0, 1, 2, 3, 4, 5, 6, 124

LOG_ROOT = os.environ.get("HWS_LOG_DIR", "/scratch/staff/huaj/tmp/hws")
XSDB = os.environ.get("HWS_XSDB", "/proj/custbuilds/2026.2_daily_latest/installs/lin64/2026.2/Vitis/bin/xsdb")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10"]
LAB_POINTER = "Likely a board/JTAG/cable issue -- lab check needed: ask Amit (JTAG cable, board power, UART)."
COMMANDS = ("status", "up", "down", "clean", "restart")


@dataclasses.dataclass(frozen=True)
class Health:
    board: str
    type: str
    state: str
    checks: dict
    msg: str = ""
    rc: int = RC_OK

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


class HwsError(Exception):
    def __init__(self, rc, msg, health=None):
        super().__init__(msg)
        self.rc = rc
        self.msg = msg
        self.health = health


class Io:
    """All side effects (ssh, port probe, xsdb, clock). Unit tests swap this out."""

    def ssh(self, target, cmd, timeout=30):
        """Run ``cmd`` on ``target`` (user@host). Returns (rc, output); rc 255 = ssh failed."""
        try:
            p = subprocess.run(["ssh", *SSH_OPTS, target, cmd], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, timeout=timeout)
            return p.returncode, p.stdout
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return RC_STALLED, out

    def port_open(self, host, port=PORT, timeout=3):
        return subprocess.run(["nc", "-z", "-w", str(timeout), host, str(port)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

    def xsdb_targets(self, host, port=PORT, timeout=60):
        """Remote `targets` from this machine. Returns (rc, output)."""
        tcl = (f"if {{[catch {{connect -url TCP:{host}:{port}}} e]}} {{puts \"HWS_CONNECT_ERR $e\"; exit 1}}\n"
               "after 2000\nputs [targets]\ncatch {disconnect}\nexit 0\n")
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as f:
            f.write(tcl)
        try:
            p = subprocess.run([XSDB, f.name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=timeout)
            return p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            return RC_STALLED, "HWS xsdb targets timed out"
        except OSError as e:
            return 127, f"HWS cannot run xsdb {XSDB}: {e}"
        finally:
            os.unlink(f.name)

    def sleep(self, secs):
        time.sleep(secs)

    def now(self):
        return time.time()


class Log:
    """Human log line to stdout (unless quiet) and to the per-command log file."""

    def __init__(self, board, cmd, root=None, quiet=False):
        d = os.path.join(root or LOG_ROOT, host_of(board))
        self.path = None
        self.quiet = quiet
        try:
            os.makedirs(d, exist_ok=True)
            self.path = os.path.join(d, f"hws_{cmd}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        except OSError:
            pass

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')} hws] {msg}"
        if not self.quiet:
            print(line, flush=True)
        if self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass


def host_of(board):
    return board.split("@")[-1]


def board_type(board, explicit=None):
    """palmyra for 10.23.224.*, vek385 for farm hosts / vek385 names, else HwsError(2)."""
    if explicit:
        if explicit not in ("palmyra", "vek385"):
            raise HwsError(RC_USAGE, f"unknown board type {explicit!r}")
        return explicit
    host = host_of(board)
    if host.startswith("10.23.224."):
        return "palmyra"
    if re.match(r"^(vek385|crimini|gomphus)", host):
        return "vek385"
    raise HwsError(RC_USAGE, f"cannot tell the board type of {board!r}: pass --type palmyra|vek385")


def _backend(board, btype, io, log, lock_owner):
    mod = importlib.import_module(f"hws_{btype}")
    return mod.Backend(board, io, log, lock_owner=lock_owner)


def _run(cmd, board, type=None, locked=False, io=None, wait_lock=1800, log=None, lock_dir=None):
    btype = board_type(board, type)
    io = io or Io()
    log = log or Log(board, cmd)
    if cmd == "status" or locked:
        owner = devlock.read_owner(devlock.lock_path(board, lock_dir))
        return _dispatch(cmd, _backend(board, btype, io, log, owner))
    try:
        lock = devlock.DeviceLock(board, wait_secs=wait_lock, lock_dir=lock_dir)
        with lock:
            return _dispatch(cmd, _backend(board, btype, io, log, devlock.read_owner(lock.path)))
    except SystemExit as e:  # DeviceLock reports BOARD BUSY this way
        raise HwsError(RC_BUSY, f"HWS LOCK BUSY {board}: {e}") from None


def _dispatch(cmd, be):
    if cmd == "restart":
        be.down()
        be.clean()
        return be.up()
    return getattr(be, cmd)()


def status(board, type=None, **kw):
    return _run("status", board, type, **kw)


def up(board, type=None, locked=False, **kw):
    return _run("up", board, type, locked, **kw)


def down(board, type=None, locked=False, **kw):
    return _run("down", board, type, locked, **kw)


def clean(board, type=None, locked=False, **kw):
    return _run("clean", board, type, locked, **kw)


def restart(board, type=None, locked=False, **kw):
    return _run("restart", board, type, locked, **kw)


def ensure(board, type=None, locked=False, **kw):
    """status, then at most one restart if not UP. Raises HwsError unless UP."""
    h = status(board, type, **kw)
    if h.state == UP:
        return h
    if h.state in (FOREIGN, UNREACHABLE):
        raise HwsError(h.rc, h.msg, h)
    h = restart(board, type, locked, **kw)
    if h.state != UP:
        raise HwsError(h.rc or RC_DEGRADED, h.msg, h)
    return h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--board", required=True)
    ap.add_argument("--type", choices=("palmyra", "vek385"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--wait-lock", type=int, default=1800)
    ap.add_argument("cmd", choices=COMMANDS)
    a = ap.parse_args(argv)
    try:
        h = _run(a.cmd, a.board, a.type, wait_lock=a.wait_lock)
        rc = RC_DOWN if (a.cmd == "status" and h.state == DOWN) else h.rc
        h = dataclasses.replace(h, rc=rc)
    except HwsError as e:
        h = e.health or Health(a.board, a.type or "?", "ERROR", {}, e.msg, e.rc)
        h = dataclasses.replace(h, rc=e.rc, msg=e.msg)
    print(h.to_json() if a.json else f"HWS {h.state} {h.board} rc={h.rc} {h.msg} checks={h.checks}")
    return h.rc


if __name__ == "__main__":
    # Run through the importable module so backends' `import hws` shares HwsError/Health.
    import hws as _hws
    sys.exit(_hws.main())
