###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``emod.run(board, host)``: run ``main.elf`` on a lab board and read the result.

    emod.run()                          # palmyra, host from $PALIP / envlocal.sh
    emod.run(ag.VEK385, "crimini2")     # a VEK385 by host name

A thin driver over the repo's board scripts -- ``script/test/apppaltest.py``
(palmyra) and ``script/test/appvek385.py`` (VEK385) -- which own the SSH, xsdb,
console and power handling. This module only picks the script and its flags,
fills the environment those scripts read, streams the console to a log, and
parses what the ELF printed (top-1 / top-5, inference time, AIE ENTER/EXIT).

Three rules come from runs that went wrong:
  * one board run at a time -- an overlapping run looks exactly like an engine
    hang, so a second run of the same script by this user is refused;
  * palmyra defaults to ``--netload``: the 12-16 MB TVM ELF fails JTAG ``dow``
    (``Invalid DAP IDCODE``) while the netboot supervisor streams it in ~0.2 s;
  * VEK385 defaults to powering off after the run: its ``-nonreboot`` mode keeps
    the process alive holding the board until it is killed.
"""

from __future__ import annotations

import getpass
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import _env
from .errors import AiegraphError
from .report import RunReport

__all__ = ["PALMYRA", "VEK385", "BOARDS", "run_on_board", "parse_console"]

PALMYRA = "palmyra"
VEK385 = "vek385"


@dataclass(frozen=True)
class Board:
    name: str
    script: str             # under script/test/
    host_env: str           # env var the script reads the lab host / board IP from
    required: tuple         # env vars the script refuses to run without
    netload: bool           # default transport: netboot supervisor (palmyra only)
    reboot: bool            # default: power off when done (False = -nonreboot)


BOARDS = {
    PALMYRA: Board(PALMYRA, "apppaltest.py", "PALIP", ("USERNAME", "PALIP", "BOARDNAME"),
                   netload=True, reboot=False),
    VEK385: Board(VEK385, "appvek385.py", "VEK385IP", ("USERNAME", "VEK385IP"),
                  netload=False, reboot=True),
}
_ALIASES = {"pal": PALMYRA, "palboard": PALMYRA, "vek": VEK385, "vek385": VEK385}


def board_spec(board) -> Board:
    key = str(board or PALMYRA).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in BOARDS:
        raise AiegraphError("run", f"unknown board {board!r}",
                            hint=f"one of: ag.PALMYRA ({PALMYRA!r}), ag.VEK385 ({VEK385!r})")
    return BOARDS[key]


def _envlocal(script_dir: Path) -> dict:
    """The variables ``script/test/envlocal.sh`` exports (what the scripts source)."""
    f = script_dir / "envlocal.sh"
    if not f.is_file():
        return {}
    r = subprocess.run(["bash", "-c", f'source "{f}" >/dev/null 2>&1 && env -0'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    return dict(kv.split("=", 1) for kv in r.stdout.split("\0") if "=" in kv)


def board_env(b: Board, host: str = None) -> dict:
    """Environment for the board script. Every variable it checks is set here --
    otherwise it re-sources envlocal.sh and that would overwrite a *host* override."""
    env = dict(os.environ)
    local = _envlocal(_env.repo_root() / "script" / "test")
    for k in b.required + ("BOARDNAME",):
        if not env.get(k) and local.get(k):
            env[k] = local[k]
    env.setdefault("USERNAME", getpass.getuser())
    if host:
        env[b.host_env] = host
    missing = [k for k in b.required if not env.get(k)]
    if missing:
        raise AiegraphError("run", f"{b.name}: {', '.join(missing)} not set",
                            hint=f"pass the host (emod.run({b.name!r}, 'hostname')), export "
                                 f"{' '.join(missing)}, or put them in script/test/envlocal.sh")
    return env


def _busy(script: str) -> list:
    """PIDs of this user's running board sessions of *script*."""
    r = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", f"script/test/{script}"],
                       capture_output=True, text=True)
    return [int(p) for p in r.stdout.split() if p.strip().isdigit() and int(p) != os.getpid()]


def board_command(b: Board, elf: Path, netload: bool, reboot: bool, args=()) -> list:
    if netload and not b.netload:
        raise AiegraphError("run", f"{b.name} has no netboot supervisor (netload=True)",
                            hint="netload is palmyra only; run without it")
    cmd = [sys.executable, str(_env.repo_root() / "script" / "test" / b.script), "-y"]
    if netload:
        cmd.append("--netload")
    elif not reboot:
        cmd.append("-nonreboot")
    return cmd + [str(a) for a in args] + [str(elf)]


def _stream(proc, log: Path, echo: bool) -> list:
    lines = []
    with log.open("w") as fh:
        for line in proc.stdout:
            lines.append(line)
            fh.write(line)
            fh.flush()
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
    return lines


_TOP1 = re.compile(r"top1 class=(\d+) logit=(-?[\d.]+)")
_TOPK = re.compile(r"^\s*(\d+)\.\s+class=(\d+)\s+(.*?)\s+logit=(-?[\d.]+)")
_INFER = re.compile(r"^\s*inference\s+([\d.]+) ms")
# make PROFILE=1 (graph_driver.c graph_prof_print): "PROF  12      3.456 ms  cpu <op>"
_PROF = re.compile(r"^PROF\s+(\d+)\s+(-?[\d.]+) ms\s+(cpu|aie)?\s*(.*)$")
_FAIL = ("AIE ERROR", "TIMEOUT", "Test FAILED", "PLM stall")


def parse_console(text: str) -> dict:
    """What the TVM ``main.elf`` printed: top-1/top-5, inference ms, AIE layers."""
    out = {"top1": None, "top5": [], "inference_ms": None,
           "aie_enter": len(re.findall(r"\[aie-offload\].*ENTER", text)),
           "aie_exit": len(re.findall(r"\[aie-offload\].*EXIT", text)),
           "failures": sorted({f for f in _FAIL if f in text}), "profile": []}
    for line in text.splitlines():
        m = _TOP1.search(line)
        if m:
            out["top1"] = (int(m.group(1)), float(m.group(2)))
        m = _TOPK.match(line)
        if m and len(out["top5"]) < 5:
            out["top5"].append((int(m.group(2)), m.group(3).strip(), float(m.group(4))))
        m = _INFER.match(line)
        if m:
            out["inference_ms"] = float(m.group(1))
        m = _PROF.match(line.strip())
        if m:
            out["profile"].append((int(m.group(1)), float(m.group(2)), m.group(3) or "?",
                                   m.group(4).strip()))
    return out


def run_on_board(elf, board=PALMYRA, host: str = None, *, netload: bool = None,
                 reboot: bool = None, timeout: float = 1800, log=None, args=(),
                 verbose: bool = True) -> RunReport:
    """Run *elf* on *board* (``PALMYRA`` / ``VEK385``) at *host* and parse the console."""
    b = board_spec(board)
    elf = Path(elf).resolve()
    if not elf.is_file():
        raise AiegraphError("run", f"{elf} does not exist", hint="emod.build(target='a78') first")
    netload = b.netload if netload is None else netload
    reboot = b.reboot if reboot is None else reboot
    env = board_env(b, host)
    pids = _busy(b.script)
    if pids:
        raise AiegraphError("run", f"another {b.script} run is active (pid {', '.join(map(str, pids))})",
                            hint="one board session at a time -- an overlapping run looks like a hang; "
                                 "wait for it or stop it")
    cmd = board_command(b, elf, netload, reboot, args)
    log = Path(log) if log else elf.parent / f"run_{b.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    where = env[b.host_env]
    print(f"[aiegraph] run: {elf.name} on {b.name} @ {where}"
          f" ({'netload' if netload else 'JTAG'}, log {log})")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(_env.repo_root()), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                            errors="replace", bufsize=1)
    box = {}
    reader = threading.Thread(target=lambda: box.setdefault("lines", _stream(proc, log, verbose)))
    reader.start()
    reason = ""
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait()
        reason = f"no result after {timeout:.0f} s; board script killed"
    reader.join(timeout=10)
    text = "".join(box.get("lines", []))
    got = parse_console(text)
    if not reason:
        if rc != 0:
            reason = f"{b.script} exited {rc}"
        elif got["failures"]:
            reason = "console reports " + ", ".join(got["failures"])
        elif got["top1"] is None:
            reason = "the ELF printed no top1 line (did it finish?)"
        elif got["aie_enter"] != got["aie_exit"]:
            reason = f"AIE layer ENTER x{got['aie_enter']} but EXIT x{got['aie_exit']}"
    return RunReport(ok=not reason, board=b.name, host=where, elf=str(elf), rc=rc,
                     log=str(log), reason=reason, top1=got["top1"], top5=got["top5"],
                     inference_ms=got["inference_ms"], aie_enter=got["aie_enter"],
                     aie_exit=got["aie_exit"], seconds=time.time() - t0, command=cmd,
                     profile=got["profile"])
