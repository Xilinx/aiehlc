###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``emod.offload_AI_proto(layer, kernel)``: let ``claude -p`` port a kernel that
``offload_check`` refuses into one that passes it.

The kernel's directory is copied (flat, as aiehlc wants it) to
``<workdir>/aiproto/<layer>_<kernel>/``; Claude edits only that copy, with the
TVM C of the layer, the refusal and ``doc/tutorial_userkernel.md`` in hand, and
may run one command: ``aiegraph check``. Whatever Claude says, the verdict is
our own ``offload_check`` after each round; a failure goes back to the same
session (``--resume``) until it is READY or ``rounds`` run out. The original
kernel and the model are never touched. Nothing here runs on a board.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import _env
from .errors import AiegraphError
from .report import AIProtoResult

__all__ = ["offload_ai_proto"]

_COPY = {".cc", ".cpp", ".c", ".h", ".hpp", ".inc", ".txt", ".md"}
_DOC = "doc/tutorial_userkernel.md"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(src: Path, dest: Path, new_name: str) -> Path:
    """Fresh flat copy of the kernel's directory; the entry renamed to *new_name*.

    Never deletes: the source is checked BEFORE anything is touched, a *dest*
    that contains the source is refused, and an existing *dest* (an earlier
    run's copy, possibly the kernel a layer is offloaded from) is moved aside
    to ``<dest>.prev-<time>``. A stale source path once made this rmtree the
    very kernel layer 1 ran, then fail."""
    src, dest = Path(src).resolve(), Path(dest).resolve()
    if not src.is_file():
        raise AiegraphError("stage", f"kernel source {src} does not exist (nothing was touched)",
                            hint="the layer's current source: emod.source_path('layerN')")
    if dest == src.parent or dest in src.parents:
        raise AiegraphError("stage", f"{dest} contains the source {src}; refusing to restage over it",
                            hint="pass out_dir= a different directory")
    if dest.exists():
        keep = dest.with_name(f"{dest.name}.prev-{time.strftime('%Y%m%d-%H%M%S')}")
        dest.rename(keep)
        print(f"[aiegraph] kept the previous copy: {keep}")
    dest.mkdir(parents=True)
    for f in src.parent.iterdir():
        if f.is_file() and f.suffix in _COPY and f != src:
            shutil.copy2(f, dest / f.name)
    shutil.copy2(src, dest / new_name)
    return dest / new_name


def _failures(rep) -> str:
    return "\n".join(f"  layer{l if l is not None else '?'}: {t}" for _s, l, ok, t in rep.rows
                     if not ok) or f"  {rep.emulation}"


def _prompt(emod, layers, orig: Path, entry: Path, rep, check_cmd: str) -> str:
    wd = emod.workdir
    lay = "\n".join(f"  {emod.layer(i)!r}\n  TVM C (the exact computation): "
                    f"{wd / 'layers' / emod.layer(i).dir}/" for i in layers)
    repo = _env.repo_root()
    return f"""You are porting an AIE user kernel so that one layer of an int8 network can run on it.

Target layer(s) of the model in {wd}:
{lay}

Starting point: {orig} (the user's kernel; read it, do not edit it).
A copy of its directory is your working directory {entry.parent}; the entry file is
{entry.name}. Edit ONLY files in that directory (add new ones there if needed).

The current kernel is refused for the layer(s):
{_failures(rep)}

Make {entry.name} pass this check (exit status 0, verdict READY):
  {check_cmd}
It lints the source, matches its AIEK_SPEC against the layer's TVM call (shapes,
dtypes, value requirements), runs it bit-exact against TVM (calibrate) and the
whole network with it swapped in (emulate), all on x86.

Rules:
- Read {repo / _DOC} first (sections 2, 6, 7, 8): the AIEK_ENTRY/AIEK_SPEC contract,
  the reference file, and the source rules lint enforces.
- Rename the AIEK_ENTRY / AIEK_SPEC name and the spec's "kernel" line to {entry.stem}
  so it can coexist with the original kernel.
- The result must be a real AIE kernel the aiehlc flow can compile for AIE2PS, not a
  scalar CPU copy of the TVM C: keep the original's structure (vector code, tiling,
  data layout) and change what the layer needs (shapes, kernel size, stride,
  padding, epilogue). Other kernels under {repo / 'src/aietensorop'} show patterns.
- Never weaken the check: no `require=` removed to hide a real value range, no
  spec that matches a different layer, no edits outside your directory.
- When it passes, stop and reply with a short summary of what you changed.
  If you cannot make it pass, stop and explain what blocks it.
"""


def _follow_up(rep, check_cmd: str) -> str:
    return (f"Our own check of your kernel still says NOT READY:\n{_failures(rep)}\n"
            f"  emulate: {rep.emulation}\nFix it; verify with:\n  {check_cmd}")


def _run_claude(cmd: list, prompt: str, cwd: Path, log, timeout_s: float, quiet: bool):
    """One ``claude -p`` turn, streamed: progress lines on stdout, raw events in *log*."""
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)          # allow running from inside a Claude Code session
    log.write(f"\n===== prompt {time.strftime('%F %T')}\n{prompt}\n===== events\n")
    # The prompt goes on stdin: --allowedTools/--add-dir are variadic and would eat it.
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.stdin.write(prompt)
    proc.stdin.close()
    result, t0 = {}, time.time()
    try:
        for line in proc.stdout:
            log.write(line)
            if time.time() - t0 > timeout_s:
                proc.kill()
                raise AiegraphError("ai_proto", f"claude ran past {timeout_s:.0f} s",
                                    hint="raise timeout_s=, or see the log")
            try:
                ev = json.loads(line)
            except ValueError:
                if not quiet:
                    print(f"  [claude] {line.rstrip()[:160]}")
                continue
            if ev.get("type") == "result":
                result = ev
            elif ev.get("type") == "assistant" and not quiet:
                for c in ev.get("message", {}).get("content", []):
                    if c.get("type") == "tool_use":
                        arg = c.get("input", {})
                        what = arg.get("command") or arg.get("file_path") or arg.get("pattern") or ""
                        print(f"  [claude] {c.get('name')}: {str(what)[:140]}")
                    elif c.get("type") == "text" and c.get("text", "").strip():
                        print(f"  [claude] {c['text'].strip().splitlines()[0][:160]}")
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    return proc.returncode, result


def offload_ai_proto(emod, spec, source, *, rounds: int = 3, model: str = None,
                     claude: str = "claude", timeout_s: float = 1800, out_dir=None,
                     quiet: bool = False, verbose: bool = False) -> AIProtoResult:
    if source is None:
        raise AiegraphError("ai_proto", "needs a starting kernel",
                            hint='emod.offload_AI_proto("layer4", "path/to/kernel_uk.cc")')
    orig = Path(source).resolve()
    if not orig.is_file():
        raise AiegraphError("ai_proto", f"{source} does not exist")
    layers = emod.resolve(spec)
    if not layers:
        raise AiegraphError("ai_proto", "name the layer(s) to port to, e.g. \"layer4\"")
    exe = shutil.which(claude)
    if exe is None:
        raise AiegraphError("ai_proto", f"no `{claude}` on PATH",
                            hint="install Claude Code (https://claude.com/claude-code) or pass claude=<path>")
    tag = "layer" + "_".join(map(str, layers))
    rep = emod.offload_check(layers, orig, verbose=verbose)
    if rep.ok:
        print(f"[aiegraph] {orig.name} is already READY for {tag}: nothing to port")
        return AIProtoResult(orig, layers, rep, 0, None, None, orig)

    dest = Path(out_dir).resolve() if out_dir else emod.workdir / "aiproto" / f"{tag}_{orig.stem}"
    entry = _stage(orig, dest, f"{orig.stem}_{tag.replace('layer', 'l')}{orig.suffix}")
    before = _digest(orig)
    sel = ",".join(map(str, layers))
    check_cmd = f'{sys.executable} -m aiegraph check --workdir {emod.workdir} "{entry}@{sel}"'
    tools = ["Read", "Grep", "Glob", f"Edit(/{dest}/**)", f"Write(/{dest}/**)",
             f"Bash({sys.executable} -m aiegraph check:*)"]
    base = [exe, "-p", "--output-format", "stream-json", "--verbose",
            "--allowedTools", ",".join(tools), "--add-dir", str(dest)]
    if model:
        base += ["--model", model]
    log_path = dest / "ai_proto.log"
    session, n = None, 0
    print(f"[aiegraph] offload_AI_proto: {orig.name} is NOT READY for {tag}; asking claude to port "
          f"it in {dest} (up to {rounds} round(s), each verified by offload_check)")
    with open(log_path, "a") as log:
        for n in range(1, rounds + 1):
            prompt = _prompt(emod, layers, orig, entry, rep, check_cmd) if session is None \
                else _follow_up(rep, check_cmd)
            cmd = base + (["--resume", session] if session else [])
            print(f"[aiegraph] round {n}/{rounds}: claude -p ...")
            rc, res = _run_claude(cmd, prompt, dest, log, timeout_s, quiet)
            session = res.get("session_id") or session
            if res.get("result"):
                print(f"  [claude] {res['result'].strip()[:600]}")
            if rc != 0 and not res:
                raise AiegraphError("ai_proto", f"claude exited {rc} without a result",
                                    hint=f"see {log_path} (logged in? `claude -p hi`)")
            if not entry.is_file():
                raise AiegraphError("ai_proto", f"claude removed {entry.name}", hint=str(log_path))
            rep = emod.offload_check(layers, entry, verbose=verbose)
            if rep.ok:
                break
    if _digest(orig) != before:
        print(f"[aiegraph] WARNING: {orig} changed during the port")
    out = AIProtoResult(entry, layers, rep, n, session, log_path, orig)
    print(repr(out))
    return out
